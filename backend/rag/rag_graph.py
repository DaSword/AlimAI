"""
LangGraph StateGraph workflow for RAG pipeline.

This module defines the complete RAG workflow as a state machine with:
- State management
- Node connections
- Conditional routing
- Error handling
- Persistence/checkpointing
"""

from typing import TypedDict, Annotated, Sequence
from operator import add

from langgraph.graph import StateGraph, END

from backend.rag.rag_nodes import (
    analyze_query_complexity_node,
    classify_query_node,
    expand_query_node,
    retrieve_node,
    light_retrieve_node,
    rank_context_node,
    generate_response_node,  # Now handles both streaming and non-streaming
    generate_conversational_response_node,  # Now handles both streaming and non-streaming
    format_citations_node,
    update_messages_node,
    error_handler_node,
    route_by_complexity,
    route_by_question_type,
    route_simple_query,
    should_retrieve,
    has_documents,
)


# ============================================================================
# State Definition
# ============================================================================

class RAGWorkflowState(TypedDict, total=False):
    """
    State schema for the RAG workflow.
    
    This defines what data flows through the graph nodes.
    """
    # User input
    user_query: str
    
    # Query complexity analysis
    query_complexity: str  # simple_conversational, follow_up, simple_factual, complex
    needs_new_retrieval: bool  # Whether new retrieval is needed for follow-ups
    
    # Query classification
    question_type: str  # fiqh, aqidah, tafsir, hadith, general
    
    # Expanded queries
    expanded_queries: list[str]
    
    # Retrieved documents (as dicts for JSON serialization)
    retrieved_docs: list[dict]
    
    # Ranked and filtered documents
    ranked_docs: list[dict]
    
    # Generated response
    response: str
    
    # Formatted citations
    citations: list[dict]
    
    # Conversation history
    messages: Annotated[Sequence[dict], add]
    
    # User preferences
    madhab_preference: str  # hanafi, maliki, shafi, hanbali
    max_sources: int
    score_threshold: float
    
    # Error handling
    error: str


# ============================================================================
# Graph Construction
# ============================================================================

def create_rag_graph(enable_streaming: bool = None) -> StateGraph:
    """
    Create the RAG workflow graph with all nodes and edges.
    
    New workflow with query complexity routing:
    - Simple conversational → Direct response (no retrieval)
    - Follow-up (use existing) → Response with previous sources
    - Follow-up (new retrieval) → Full pipeline
    - Simple factual → Light retrieval → Response
    - Complex → Full pipeline with expansion
    
    Args:
        enable_streaming: Whether to use streaming nodes (defaults to config.ENABLE_STREAMING)
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Import config
    from backend.core.config import Config
    config = Config()
    
    # Determine if streaming is enabled
    if enable_streaming is None:
        enable_streaming = config.ENABLE_STREAMING
    
    # Initialize graph with state schema
    workflow = StateGraph(RAGWorkflowState)
    
    mode_str = "STREAMING" if enable_streaming else "NON-STREAMING"
    print(f"🔧 Creating RAG graph in {mode_str} mode")
    
    # Add nodes (unified nodes now handle both streaming and non-streaming internally)
    workflow.add_node("analyze_complexity", analyze_query_complexity_node)
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("classify_query_simple", classify_query_node)  # For simple queries
    workflow.add_node("expand_query", expand_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("light_retrieve", light_retrieve_node)
    workflow.add_node("rank_context", rank_context_node)
    workflow.add_node("generate_response", generate_response_node)  # Unified: handles both modes
    workflow.add_node("generate_conversational", generate_conversational_response_node)  # Unified: handles both modes
    workflow.add_node("format_citations", format_citations_node)
    workflow.add_node("update_messages", update_messages_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Set entry point - always analyze complexity first
    workflow.set_entry_point("analyze_complexity")
    
    # Add edges based on new routing logic
    
    # 1. Analyze Complexity → Route by complexity
    workflow.add_conditional_edges(
        "analyze_complexity",
        route_by_complexity,
        {
            "generate_conversational": "generate_conversational",  # Simple conversational
            "generate_response": "generate_response",  # Follow-up using existing sources
            "classify_query": "classify_query",  # Complex or follow-up needing retrieval
            "classify_query_simple": "classify_query_simple",  # Simple factual
        }
    )
    
    # 2. Classify (Complex/Follow-up) → Expand
    workflow.add_edge("classify_query", "expand_query")
    
    # 3. Classify (Simple) → Light Retrieve
    workflow.add_edge("classify_query_simple", "light_retrieve")
    
    # 4. Expand → Retrieve
    workflow.add_edge("expand_query", "retrieve")
    
    # 5. Retrieve → Rank (if documents) or Generate (if no documents)
    workflow.add_conditional_edges(
        "retrieve",
        has_documents,
        {
            "rank_context": "rank_context",
            "generate_response": "generate_response",
        }
    )
    
    # 6. Light Retrieve → Rank (if documents) or Generate (if no documents)
    workflow.add_conditional_edges(
        "light_retrieve",
        has_documents,
        {
            "rank_context": "rank_context",
            "generate_response": "generate_response",
        }
    )
    
    # 7. Rank → Generate
    workflow.add_edge("rank_context", "generate_response")
    
    # 8. Generate → Format Citations
    workflow.add_edge("generate_response", "format_citations")
    
    # 9. Generate Conversational → Format Citations (empty)
    workflow.add_edge("generate_conversational", "format_citations")
    
    # 10. Format Citations → Update Messages
    workflow.add_edge("format_citations", "update_messages")
    
    # 11. Update Messages → END
    workflow.add_edge("update_messages", END)
    
    # 12. Error Handler → END
    workflow.add_edge("error_handler", END)
    
    # Compile graph without checkpointer
    # Note: LangGraph Server provides automatic persistence, so no custom checkpointer needed
    # For local testing, checkpointing can be added in the invoke functions if needed
    compiled_graph = workflow.compile()
    
    return compiled_graph


# ============================================================================
# Graph Execution Helpers
# ============================================================================

def invoke_rag_workflow(
    user_query: str,
    madhab_preference: str = None,
    max_sources: int = 10,
    score_threshold: float = 0.7,
    thread_id: str = "default",
) -> dict:
    """
    Execute the RAG workflow for a single query.
    
    Args:
        user_query: User's question
        madhab_preference: Preferred madhab for fiqh questions
        max_sources: Maximum number of sources to retrieve
        score_threshold: Minimum similarity score threshold
        thread_id: Thread ID for conversation persistence
        
    Returns:
        Final state dict with response and citations
    """
    # Create initial state
    # Note: messages field is managed by LangGraph state persistence, not set here
    initial_state = {
        "user_query": user_query,
        "madhab_preference": madhab_preference,
        "max_sources": max_sources,
        "score_threshold": score_threshold,
    }
    
    # Get compiled graph
    graph = create_rag_graph()
    
    # Execute workflow
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        final_state = graph.invoke(initial_state, config=config)
        return final_state
    except Exception as e:
        print(f"❌ Workflow execution error: {e}")
        return {
            "user_query": user_query,
            "response": f"I apologize, but I encountered an error: {str(e)}",
            "citations": [],
            "error": str(e),
        }


async def astream_rag_workflow(
    user_query: str,
    madhab_preference: str = None,
    max_sources: int = 10,
    score_threshold: float = 0.7,
    thread_id: str = "default",
):
    """
    Stream the RAG workflow execution for a single query.
    
    Yields state updates as the workflow progresses through nodes.
    
    Args:
        user_query: User's question
        madhab_preference: Preferred madhab for fiqh questions
        max_sources: Maximum number of sources to retrieve
        score_threshold: Minimum similarity score threshold
        thread_id: Thread ID for conversation persistence
        
    Yields:
        State updates from each node
    """
    # Create initial state
    initial_state = {
        "user_query": user_query,
        "madhab_preference": madhab_preference,
        "max_sources": max_sources,
        "score_threshold": score_threshold,
    }
    
    # Get compiled graph
    graph = create_rag_graph()
    
    # Execute workflow with streaming
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        async for event in graph.astream(initial_state, config=config):
            yield event
    except Exception as e:
        print(f"❌ Workflow streaming error: {e}")
        yield {
            "error_handler": {
                "user_query": user_query,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "citations": [],
                "error": str(e),
            }
        }


def get_graph_visualization(enable_streaming: bool = None) -> str:
    """
    Get a text-based visualization of the graph structure.
    
    Args:
        enable_streaming: Whether streaming is enabled (defaults to config)
    
    Returns:
        String representation of the graph
    """
    from backend.core.config import Config
    config = Config()
    
    if enable_streaming is None:
        enable_streaming = config.ENABLE_STREAMING
    
    streaming_note = "(STREAMING MODE - tokens streamed in real-time)" if enable_streaming else "(NON-STREAMING MODE)"
    
    visualization = f"""
RAG Workflow Graph Structure (Multi-turn Aware):
=================================================
{streaming_note}

START
  ↓
[analyze_complexity] - Analyze query complexity and conversation context
  ↓
  ├─→ simple_conversational (hi, thanks) → [generate_conversational] → [format_citations] → [update_messages] → END
  │
  ├─→ follow_up (no new retrieval) → [generate_response] (reuses ranked_docs) → [format_citations] → [update_messages] → END
  │
  ├─→ simple_factual (basic questions) → [classify_query_simple] → [light_retrieve] → [rank_context] → [generate_response] → [format_citations] → [update_messages] → END
  │
  └─→ complex/follow_up (new retrieval) → [classify_query] → [expand_query] → [retrieve] → [rank_context] → [generate_response] → [format_citations] → [update_messages] → END

Key Features:
  - Conversation history included in all generation nodes
  - Simple queries skip expensive retrieval
  - Follow-ups can reuse previous sources
  - Message persistence managed by LangGraph state
  - Lightweight retrieval for factual queries
  - Token-level streaming (when enabled): Responses stream as they're generated

State Flow:
  - user_query (input)
  - query_complexity (simple_conversational/follow_up/simple_factual/complex)
  - needs_new_retrieval (bool)
  - question_type (fiqh/aqidah/tafsir/hadith/general)
  - expanded_queries (generated for complex)
  - retrieved_docs (fetched)
  - ranked_docs (ranked)
  - response (generated with conversation context, streamed if enabled)
  - citations (formatted)
  - messages (accumulated history)
"""
    return visualization


# ============================================================================
# Utility Functions
# ============================================================================

def get_thread_history(thread_id: str = "default") -> list:
    """
    Get conversation history for a thread.
    
    Args:
        thread_id: Thread ID
        
    Returns:
        List of messages in the thread
    """
    # This would typically query the checkpointer
    # For now, return empty list (to be implemented with persistent storage)
    return []


def clear_thread_history(thread_id: str = "default") -> bool:
    """
    Clear conversation history for a thread.
    
    Args:
        thread_id: Thread ID
        
    Returns:
        Success status
    """
    # This would typically clear the checkpointer
    # For now, return True (to be implemented with persistent storage)
    return True


def list_active_threads() -> list[str]:
    """
    List all active conversation threads.
    
    Returns:
        List of thread IDs
    """
    # This would typically query the checkpointer
    # For now, return empty list (to be implemented with persistent storage)
    return []


# ============================================================================
# Export
# ============================================================================

# Import config for default graph instance
from backend.core.config import Config
_config = Config()

# Create a singleton graph instance for import (uses config default)
graph = create_rag_graph(enable_streaming=_config.ENABLE_STREAMING)

# Export main functions
__all__ = [
    "RAGWorkflowState",
    "create_rag_graph",
    "invoke_rag_workflow",
    "astream_rag_workflow",
    "get_graph_visualization",
    "get_thread_history",
    "clear_thread_history",
    "list_active_threads",
    "graph",
]


# ============================================================================
# CLI Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test the RAG workflow from command line.
    """
    import sys
    
    # Print graph visualization
    print(get_graph_visualization())
    print("\n")
    
    # Test query
    test_query = sys.argv[1] if len(sys.argv) > 1 else "What is charity in Islam?"
    
    print(f"Testing RAG workflow with query: {test_query}")
    print("="*60)
    
    # Execute workflow
    result = invoke_rag_workflow(
        user_query=test_query,
        max_sources=5,
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESPONSE:")
    print("="*60)
    print(result.get("response", "No response generated"))
    
    print("\n" + "="*60)
    print("CITATIONS:")
    print("="*60)
    citations = result.get("citations", [])
    for i, citation in enumerate(citations, 1):
        print(f"\n{i}. [{citation.get('source_type', 'unknown').upper()}] {citation.get('book_title', 'Unknown')}")
        if citation.get('verse_key'):
            print(f"   Verse: {citation['verse_key']}")
        if citation.get('hadith_number'):
            print(f"   Hadith: {citation['hadith_number']} ({citation.get('authenticity', 'unknown')})")
        print(f"   Text: {citation.get('text', '')[:100]}...")


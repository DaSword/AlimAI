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
    classify_query_node,
    expand_query_node,
    retrieve_node,
    rank_context_node,
    generate_response_node,
    format_citations_node,
    update_messages_node,
    error_handler_node,
    route_by_question_type,
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

def create_rag_graph() -> StateGraph:
    """
    Create the RAG workflow graph with all nodes and edges.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph with state schema
    workflow = StateGraph(RAGWorkflowState)
    
    # Add nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("expand_query", expand_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rank_context", rank_context_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("format_citations", format_citations_node)
    workflow.add_node("update_messages", update_messages_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    # Add edges
    # 1. Classify → Expand
    workflow.add_edge("classify_query", "expand_query")
    
    # 2. Expand → Retrieve
    workflow.add_edge("expand_query", "retrieve")
    
    # 3. Retrieve → Rank (if documents found) or Generate (if no documents)
    workflow.add_conditional_edges(
        "retrieve",
        has_documents,
        {
            "rank_context": "rank_context",
            "generate_response": "generate_response",
        }
    )
    
    # 4. Rank → Generate
    workflow.add_edge("rank_context", "generate_response")
    
    # 5. Generate → Format Citations
    workflow.add_edge("generate_response", "format_citations")
    
    # 6. Format Citations → Update Messages
    workflow.add_edge("format_citations", "update_messages")
    
    # 7. Update Messages → END
    workflow.add_edge("update_messages", END)
    
    # 8. Error Handler → END
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
    initial_state = {
        "user_query": user_query,
        "madhab_preference": madhab_preference,
        "max_sources": max_sources,
        "score_threshold": score_threshold,
        "messages": [],
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
        "messages": [],
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


def get_graph_visualization() -> str:
    """
    Get a text-based visualization of the graph structure.
    
    Returns:
        String representation of the graph
    """
    visualization = """
RAG Workflow Graph Structure:
==============================

START
  ↓
[classify_query] - Classify question type (fiqh/aqidah/tafsir/hadith/general)
  ↓
[expand_query] - Generate 2-3 reformulated queries
  ↓
[retrieve] - Retrieve documents from vector store
  ↓
  ├─→ (if documents found) → [rank_context] - Rank by authenticity & relevance
  │                            ↓
  │                          [generate_response] - Generate answer with LLM
  │                            ↓
  └─→ (if no documents) ────→ [generate_response]
                               ↓
                             [format_citations] - Structure source citations
                               ↓
                             [update_messages] - Update conversation history
                               ↓
                             END

Error Handling:
  Any node error → [error_handler] → END

State Flow:
  - user_query (input)
  - question_type (classified)
  - expanded_queries (generated)
  - retrieved_docs (fetched)
  - ranked_docs (ranked)
  - response (generated)
  - citations (formatted)
  - messages (updated history)
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

# Create a singleton graph instance for import
graph = create_rag_graph()

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


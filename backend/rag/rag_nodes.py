"""
Individual workflow nodes for LangGraph RAG pipeline.

This module contains node functions that process the RAG state:
- Query classification
- Query expansion
- Document retrieval
- Context ranking
- Response generation
- Citation formatting
"""

from typing import Dict, Any, List
import threading

from llama_index.core.llms import ChatMessage, MessageRole
from langgraph.config import get_stream_writer

# Global cancellation flag for streaming
_streaming_cancelled = threading.Event()
_streaming_lock = threading.Lock()

def cancel_streaming():
    """Cancel any ongoing streaming generation."""
    global _streaming_cancelled
    _streaming_cancelled.set()

def reset_streaming():
    """Reset the streaming cancellation flag."""
    global _streaming_cancelled
    _streaming_cancelled.clear()

from backend.core.models import (
    RAGState,
    QuestionType,
    Message,
    DocumentChunk,
)
from backend.rag.prompts import (
    QUERY_CLASSIFIER_PROMPT,
    QUERY_COMPLEXITY_PROMPT,
    CONVERSATIONAL_RESPONSE_PROMPT,
    FOLLOW_UP_EXPANSION_PROMPT,
    SYSTEM_IDENTITY,
    get_expansion_prompt,
    get_generation_prompt,
    format_prompt,
    create_system_message,
)
from backend.rag.retrieval import IslamicRetriever, expand_query_with_llm
from backend.rag.context_formatter import (
    format_context_for_llm,
    rank_documents,
    create_citation_list,
)
from backend.llama.llama_config import get_llm
from backend.core.config import Config


# Initialize config
config = Config()


# ============================================================================
# Helper Functions
# ============================================================================

def format_conversation_history(messages: List[Dict[str, Any]], max_turns: int = 10) -> str:
    """
    Format conversation history for inclusion in prompts.
    Excludes the last user message (current query being processed).
    
    Args:
        messages: List of message dicts with role and content
        max_turns: Maximum number of message pairs to include
        
    Returns:
        Formatted conversation history string
    """
    if not messages:
        return "No previous conversation."
    
    # Exclude the last user message (the current query)
    msgs_to_format = messages[:]
    for i in range(len(msgs_to_format) - 1, -1, -1):
        if msgs_to_format[i].get("role") == "user":
            msgs_to_format = msgs_to_format[:i]
            break
    
    if not msgs_to_format:
        return "No previous conversation."
    
    # Take last N messages (max_turns * 2 for user/assistant pairs)
    recent_messages = msgs_to_format[-(max_turns * 2):]
    
    formatted = []
    for msg in recent_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            formatted.append(f"User: {content}")
        elif role == "assistant":
            # Truncate long responses for context
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"Assistant: {content}")
    
    return "\n\n".join(formatted) if formatted else "No previous conversation."


# ============================================================================
# Node 0: Query Complexity Analysis
# ============================================================================

def analyze_query_complexity_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze query complexity and determine routing strategy.
    
    Classifies queries as:
    - simple_conversational: Greetings, thanks, acknowledgments
    - follow_up: References previous conversation (vague/contextual queries)
    - simple_factual: Basic factual questions
    - complex: Requires comprehensive retrieval
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with query_complexity and needs_new_retrieval
    """
    messages = state.get("messages", [])
    
    # Extract the last user message from the conversation
    user_query = ""
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
    
    if not user_query:
        user_query = state.get("user_query", "")  # Fallback to direct input
    
    print(f"🔍 Analyzing query complexity: {user_query}")
    
    # Format conversation history
    conversation_history = format_conversation_history(messages, max_turns=3)
    
    # Format complexity analysis prompt
    complexity_prompt = format_prompt(
        QUERY_COMPLEXITY_PROMPT,
        user_query=user_query,
        conversation_history=conversation_history,
    )
    
    # Get LLM
    llm = get_llm(backend=config.LLM_BACKEND)
    
    try:
        # Analyze complexity
        response = llm.complete(complexity_prompt)
        response_text = str(response).strip().lower()
        
        # Parse response
        query_complexity = "complex"  # Default
        needs_new_retrieval = True  # Default
        
        # Extract complexity
        if "simple_conversational" in response_text:
            query_complexity = "simple_conversational"
            needs_new_retrieval = False
        elif "follow_up" in response_text:
            query_complexity = "follow_up"
            # Check if new retrieval is needed
            if "needs_new_retrieval: no" in response_text or "use_existing" in response_text:
                needs_new_retrieval = False
        elif "simple_factual" in response_text:
            query_complexity = "simple_factual"
            needs_new_retrieval = True  # But will use light retrieval
        else:
            query_complexity = "complex"
            needs_new_retrieval = True
        
        print(f"✅ Complexity: {query_complexity}, New retrieval: {needs_new_retrieval}")
        
        return {
            "user_query": user_query,  # Pass along for subsequent nodes
            "query_complexity": query_complexity,
            "needs_new_retrieval": needs_new_retrieval,
        }
    
    except Exception as e:
        print(f"❌ Complexity analysis error: {e}")
        # Default to complex query with retrieval
        return {
            "user_query": user_query,  # Pass along even on error
            "query_complexity": "complex",
            "needs_new_retrieval": True,
        }


# ============================================================================
# Node 1: Query Classification
# ============================================================================

def classify_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the user query into a question type.
    
    This determines the retrieval strategy and response format.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with question_type and user_query
    """
    # Extract user query from either direct input or messages array
    user_query = state.get("user_query", "")
    
    if not user_query:
        # Extract from messages array (when coming from LangGraph Server)
        messages = state.get("messages", [])
        if messages:
            # Get the last user message
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break
    
    print(f"📝 Processing query: {user_query}")
    
    # Format classification prompt
    classification_prompt = format_prompt(
        QUERY_CLASSIFIER_PROMPT,
        user_query=user_query,
    )
    
    # Get LLM
    llm = get_llm(backend=config.LLM_BACKEND)
    
    try:
        # Generate classification
        response = llm.complete(classification_prompt)
        response_text = str(response).strip().lower()
        
        # Extract question type
        question_type = QuestionType.GENERAL  # Default
        
        for qtype in QuestionType:
            if qtype.value in response_text:
                question_type = qtype
                break
        
        print(f"🔍 Query classified as: {question_type.value}")
        
        return {
            "user_query": user_query,  # Ensure user_query is in state for subsequent nodes
            "question_type": question_type.value,
        }
    
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return {
            "user_query": user_query,
            "question_type": QuestionType.GENERAL.value,
        }


# ============================================================================
# Node 2: Query Expansion
# ============================================================================

def expand_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand the query for better retrieval coverage.
    
    Generates 2-3 reformulated queries that are more Islam-centric.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with expanded_queries
    """
    user_query = state.get("user_query", "")
    question_type = QuestionType(state.get("question_type", "general"))
    
    print(f"🔄 Expanding query: {user_query}")
    
    try:
        # Use LLM-based query expansion
        expanded_queries = expand_query_with_llm(
            query=user_query,
            question_type=question_type,
            llm_backend=config.LLM_BACKEND,
            num_expansions=2,
        )
        
        print(f"✅ Generated {len(expanded_queries)} query variations")
        for i, q in enumerate(expanded_queries, 1):
            print(f"   {i}. {q}")
        
        return {
            "expanded_queries": expanded_queries,
        }
    
    except Exception as e:
        print(f"❌ Query expansion error: {e}")
        # Fallback to original query
        return {
            "expanded_queries": [user_query],
        }


# ============================================================================
# Node 3: Document Retrieval
# ============================================================================

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant documents from vector store.
    
    Uses question type to apply intelligent filtering and retrieval strategy.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with retrieved_docs
    """
    user_query = state.get("user_query", "")
    question_type = QuestionType(state.get("question_type", "general"))
    expanded_queries = state.get("expanded_queries", [user_query])
    max_sources = state.get("max_sources", 10)
    madhab_preference = state.get("madhab_preference")
    
    print(f"📚 Retrieving documents for {question_type.value} question...")
    
    try:
        # Initialize retriever
        retriever = IslamicRetriever(
            collection_name=config.QDRANT_COLLECTION,
            embedding_backend=config.EMBEDDING_BACKEND,
            llm_backend=config.LLM_BACKEND,
        )
        
        
        # Retrieve for each expanded query
        all_docs = []
        for query in expanded_queries:
            docs = retriever.retrieve_for_question_type(
                query=query,
                question_type=question_type,
                top_k=max_sources,
                madhab_preference=madhab_preference,
            )
            all_docs.extend(docs)
        
        # Deduplicate by ID
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            if doc.id not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc.id)
        
        print(f"✅ Retrieved {len(unique_docs)} unique documents")
        
        # Convert to dict for JSON serialization
        retrieved_docs_dict = [
            {
                "id": doc.id,
                "text_content": doc.text_content,
                "metadata": doc.metadata.model_dump(),
                "score": doc.score,
            }
            for doc in unique_docs
        ]
        
        return {
            "retrieved_docs": retrieved_docs_dict,
        }
    
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return {
            "retrieved_docs": [],
        }


# ============================================================================
# Node 4: Context Ranking
# ============================================================================

def rank_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rank retrieved documents by relevance and authenticity.
    
    Applies authenticity weighting (Quran > Sahih Hadith > other sources).
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with ranked_docs
    """
    retrieved_docs_dict = state.get("retrieved_docs", [])
    question_type = QuestionType(state.get("question_type", "general"))
    max_sources = state.get("max_sources", 10)
    score_threshold = state.get("score_threshold", 0.7)
    
    print(f"⚖️ Ranking {len(retrieved_docs_dict)} documents...")
    
    try:
        # Convert dicts back to DocumentChunk objects
        from backend.core.models import QdrantPayload
        
        retrieved_docs = []
        for doc_dict in retrieved_docs_dict:
            doc = DocumentChunk(
                id=doc_dict["id"],
                text_content=doc_dict["text_content"],
                metadata=QdrantPayload(**doc_dict["metadata"]),
                score=doc_dict.get("score"),
            )
            retrieved_docs.append(doc)
        
        # Rank documents
        ranked_docs = rank_documents(
            documents=retrieved_docs,
            question_type=question_type,
            prioritize_authenticity=True,
        )
        
        # Filter by score threshold, but keep at least top 3 documents
        filtered_docs = [
            doc for doc in ranked_docs 
            if doc.score is None or doc.score >= score_threshold
        ]
        
        # If filtering removed all docs, keep top 3 from ranked results
        if not filtered_docs and ranked_docs:
            print(f"⚠️ All docs below threshold {score_threshold}, keeping top 3 anyway")
            filtered_docs = ranked_docs[:3]
        
        # Limit to max sources
        final_docs = filtered_docs[:max_sources]
        
        print(f"✅ Ranked and filtered to {len(final_docs)} documents")
        
        # Convert back to dict
        ranked_docs_dict = [
            {
                "id": doc.id,
                "text_content": doc.text_content,
                "metadata": doc.metadata.model_dump(),
                "score": doc.score,
            }
            for doc in final_docs
        ]
        
        return {
            "ranked_docs": ranked_docs_dict,
        }
    
    except Exception as e:
        print(f"❌ Ranking error: {e}")
        # Fallback to using retrieved docs as-is
        return {
            "ranked_docs": retrieved_docs_dict[:max_sources],
        }


# ============================================================================
# Node 5: Response Generation
# ============================================================================

def generate_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a response using the LLM and ranked context.
    
    Supports both streaming and non-streaming modes based on configuration.
    When streaming is enabled, uses get_stream_writer() to emit tokens in real-time.
    
    Synthesizes information from sources with proper citations.
    Includes conversation history for context-aware responses.
    
    Args:
        state: Current RAG state
        config: Optional LangGraph configuration dict
        
    Returns:
        Updated state with response
    """
    user_query = state.get("user_query", "")
    question_type = QuestionType(state.get("question_type", "general"))
    ranked_docs_dict = state.get("ranked_docs", [])
    messages_history = state.get("messages", [])
    
    # Check if streaming is enabled (defaults to True for streaming)
    # Can be overridden by setting "enable_streaming" in state
    enable_streaming = state.get("enable_streaming", True)
    
    mode_str = "STREAMING" if enable_streaming else "NON-STREAMING"
    print(f"🤖 Generating response ({mode_str})...")
    
    try:
        # Get stream writer if streaming is enabled
        writer = None
        if enable_streaming:
            try:
                writer = get_stream_writer()
            except Exception:
                # Stream writer not available (e.g., not called with stream mode)
                writer = None
        
        # Convert dicts back to DocumentChunk objects
        from backend.core.models import QdrantPayload
        
        ranked_docs = []
        for doc_dict in ranked_docs_dict:
            doc = DocumentChunk(
                id=doc_dict["id"],
                text_content=doc_dict["text_content"],
                metadata=QdrantPayload(**doc_dict["metadata"]),
                score=doc_dict.get("score"),
            )
            ranked_docs.append(doc)
        
        # Format conversation history
        conversation_history = format_conversation_history(messages_history, max_turns=10)
        
        # Check if we have ranked docs to use as context
        if not ranked_docs:
            print(f"⚠️ No ranked docs available, generating without specific sources")
            # Generate without context, with a disclaimer
            no_context_prompt = f"""The user asked: "{user_query}"

**Conversation History:**
{conversation_history}

We couldn't find specific sources in our Islamic knowledge base for this query. 
However, you should provide a general Islamic perspective based on well-known Islamic principles.

Please provide a helpful response based on general Islamic knowledge, but include a disclaimer that specific sources couldn't be retrieved."""
            
            generation_prompt = no_context_prompt
        else:
            # Format context for LLM
            context = format_context_for_llm(
                documents=ranked_docs,
                question_type=question_type,
                max_sources=len(ranked_docs),
            )
            
            # Get appropriate generation prompt
            generation_prompt_template = get_generation_prompt(question_type)
            
            # Format generation prompt with conversation history
            generation_prompt = format_prompt(
                generation_prompt_template,
                user_query=user_query,
                context=context,
                conversation_history=conversation_history,
            )
        
        # Get LLM (use module-level config, not function parameter)
        from backend.core.config import Config
        app_config = Config()
        llm = get_llm(backend=app_config.LLM_BACKEND)
        
        # Create messages using ChatMessage objects
        system_msg_dict = create_system_message(include_identity=True)
        chat_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_msg_dict["content"]),
            ChatMessage(role=MessageRole.USER, content=generation_prompt),
        ]
        
        # Generate response (streaming or non-streaming)
        if enable_streaming and writer:
            # Streaming mode: emit tokens as they're generated
            accumulated_response = ""
            response_stream = None
            
            # Reset cancellation flag for this request
            reset_streaming()
            
            try:
                response_stream = llm.stream_chat(chat_messages)
                
                for chunk in response_stream:
                    # Check cancellation FIRST before processing chunk
                    if _streaming_cancelled.is_set():
                        raise InterruptedError("Streaming cancelled by user")
                    
                    if chunk.delta:
                        accumulated_response += chunk.delta
                        try:
                            # Stream token via writer
                            writer({"token": chunk.delta, "response": accumulated_response})
                        except Exception as write_error:
                            # Client disconnected or stream writer failed
                            print(f"⚠️ Stream writer failed (likely client disconnect): {write_error}")
                            print("⚠️ Setting cancellation flag...")
                            cancel_streaming()
                            raise InterruptedError("Client disconnected")
            except Exception as e:
                print(f"⚠️ Stream error: {e}")
            finally:
                # Aggressively close the stream to stop LLM generation
                if response_stream is not None:
                    try:
                        # Close the generator
                        if hasattr(response_stream, 'close'):
                            response_stream.close()
                        # Force cleanup
                        del response_stream
                    except Exception as cleanup_error:
                        print(f"⚠️ Error during stream cleanup: {cleanup_error}")
                response_text = accumulated_response if accumulated_response else "Generation cancelled."
        else:
            # Non-streaming mode: wait for complete response
            response = llm.chat(chat_messages)
            response_text = str(response.message.content)
        
        print(f"✅ Generated response ({len(response_text)} chars)")
        
        return {
            "response": response_text,
        }
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return {
            "response": f"I encountered an error generating the response: {str(e)}",
        }


# ============================================================================
# Node 7: Generate Conversational Response (Unified - Streaming & Non-Streaming)
# ============================================================================

def generate_conversational_response_node(state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a simple conversational response without retrieval.
    
    Supports both streaming and non-streaming modes based on configuration.
    When streaming is enabled, uses get_stream_writer() to emit tokens in real-time.
    
    For greetings, acknowledgments, and simple conversational queries.
    
    Args:
        state: Current RAG state
        config: Optional LangGraph configuration dict
        
    Returns:
        Updated state with response
    """
    user_query = state.get("user_query", "")
    messages = state.get("messages", [])
    
    # Check if streaming is enabled (defaults to True for streaming)
    # Can be overridden by setting "enable_streaming" in state
    enable_streaming = state.get("enable_streaming", True)
    
    mode_str = "STREAMING" if enable_streaming else "NON-STREAMING"
    print(f"💬 Generating conversational response ({mode_str})...")
    
    try:
        # Get stream writer if streaming is enabled
        writer = None
        if enable_streaming:
            try:
                writer = get_stream_writer()
            except Exception:
                # Stream writer not available
                writer = None
        
        # Format conversation history
        conversation_history = format_conversation_history(messages, max_turns=5)
        
        # Format conversational prompt
        conversational_prompt = format_prompt(
            CONVERSATIONAL_RESPONSE_PROMPT,
            user_query=user_query,
            conversation_history=conversation_history,
        )
        
        # Get LLM (use module-level config, not function parameter)
        from backend.core.config import Config
        app_config = Config()
        llm = get_llm(backend=app_config.LLM_BACKEND)
        
        # Create messages
        system_msg_dict = create_system_message(include_identity=True)
        chat_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_msg_dict["content"]),
            ChatMessage(role=MessageRole.USER, content=conversational_prompt),
        ]
        
        # Generate response (streaming or non-streaming)
        if enable_streaming and writer:
            # Streaming mode: emit tokens as they're generated
            accumulated_response = ""
            response_stream = None
            
            # Reset cancellation flag for this request
            reset_streaming()
            
            try:
                response_stream = llm.stream_chat(chat_messages)
                
                for chunk in response_stream:
                    # Check cancellation FIRST before processing chunk
                    if _streaming_cancelled.is_set():
                        raise InterruptedError("Streaming cancelled by user")
                    
                    if chunk.delta:
                        accumulated_response += chunk.delta
                        try:
                            # Stream token via writer
                            writer({"token": chunk.delta, "response": accumulated_response})
                        except Exception as write_error:
                            # Client disconnected or stream writer failed
                            print(f"⚠️ Conversational stream writer failed (likely client disconnect): {write_error}")
                            print("⚠️ Setting cancellation flag...")
                            cancel_streaming()
                            raise InterruptedError("Client disconnected")
            except Exception as e:
                print(f"⚠️ Conversational stream error: {e}")
            finally:
                # Aggressively close the stream to stop LLM generation
                if response_stream is not None:
                    try:
                        # Close the generator
                        if hasattr(response_stream, 'close'):
                            response_stream.close()
                            print("⚠️ Conversational stream generator closed")
                        # Force cleanup
                        del response_stream
                    except Exception as cleanup_error:
                        print(f"⚠️ Error during conversational stream cleanup: {cleanup_error}")
                response_text = accumulated_response if accumulated_response else "Generation cancelled."
        else:
            # Non-streaming mode: wait for complete response
            response = llm.chat(chat_messages)
            response_text = str(response.message.content)
        
        print(f"✅ Generated conversational response ({len(response_text)} chars)")
        
        return {
            "response": response_text,
        }
    
    except Exception as e:
        print(f"❌ Conversational generation error: {e}")
        return {
            "response": "I'm here to help with questions about Islam. How can I assist you today?",
        }


# ============================================================================
# Node 6: Citation Formatting
# ============================================================================

def format_citations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format citations from the ranked documents.
    
    Creates a structured list of sources used in the response.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with citations
    """
    ranked_docs_dict = state.get("ranked_docs", [])
    
    print(f"📝 Formatting citations...")
    
    try:
        # Convert dicts back to DocumentChunk objects
        from backend.core.models import QdrantPayload
        
        ranked_docs = []
        for doc_dict in ranked_docs_dict:
            doc = DocumentChunk(
                id=doc_dict["id"],
                text_content=doc_dict["text_content"],
                metadata=QdrantPayload(**doc_dict["metadata"]),
                score=doc_dict.get("score"),
            )
            ranked_docs.append(doc)
        
        # Create citation list
        citations = create_citation_list(ranked_docs)
        
        print(f"✅ Formatted {len(citations)} citations")
        
        return {
            "citations": citations,
        }
    
    except Exception as e:
        print(f"❌ Citation formatting error: {e}")
        return {
            "citations": [],
        }


# ============================================================================
# Helper Node: Message History Update
# ============================================================================

def update_messages_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update message history with user query and assistant response.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with messages
    """
    user_query = state.get("user_query", "")
    response = state.get("response", "")
    messages = state.get("messages", [])
    
    # Add user message
    user_message = {
        "role": "user",
        "content": user_query,
        "timestamp": None,  # Will be set by Message model
    }
    
    # Add assistant message
    assistant_message = {
        "role": "assistant",
        "content": response,
        "timestamp": None,
    }
    
    # Append to messages
    updated_messages = messages + [user_message, assistant_message]
    
    return {
        "messages": updated_messages,
    }





# ============================================================================
# Node 8: Light Retrieval (Simple Factual Queries)
# ============================================================================

def light_retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight retrieval for simple factual queries.
    
    Uses fewer sources and skips query expansion for efficiency.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with retrieved_docs
    """
    user_query = state.get("user_query", "")
    question_type = QuestionType(state.get("question_type", "general"))
    madhab_preference = state.get("madhab_preference")
    
    print(f"📚 Light retrieval for simple query...")
    
    try:
        # Initialize retriever
        retriever = IslamicRetriever(
            collection_name=config.QDRANT_COLLECTION,
            embedding_backend=config.EMBEDDING_BACKEND,
            llm_backend=config.LLM_BACKEND,
        )
        
        # Retrieve with fewer sources (top 3-5)
        docs = retriever.retrieve_for_question_type(
            query=user_query,
            question_type=question_type,
            top_k=5,  # Fewer sources for simple queries
            madhab_preference=madhab_preference,
        )
        
        print(f"✅ Retrieved {len(docs)} documents (light retrieval)")
        
        # Convert to dict for JSON serialization
        retrieved_docs_dict = [
            {
                "id": doc.id,
                "text_content": doc.text_content,
                "metadata": doc.metadata.model_dump(),
                "score": doc.score,
            }
            for doc in docs
        ]
        
        return {
            "retrieved_docs": retrieved_docs_dict,
        }
    
    except Exception as e:
        print(f"❌ Light retrieval error: {e}")
        return {
            "retrieved_docs": [],
        }


# ============================================================================
# Conditional Edge Functions
# ============================================================================

def route_by_question_type(state: Dict[str, Any]) -> str:
    """
    Route to appropriate query expansion based on question type.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    question_type = state.get("question_type", "general")
    
    # All question types use the same expansion node for now
    # In the future, could route to specialized expansion nodes
    return "expand_query"


def should_retrieve(state: Dict[str, Any]) -> str:
    """
    Determine if retrieval should be performed.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    expanded_queries = state.get("expanded_queries", [])
    
    if expanded_queries:
        return "retrieve"
    else:
        # No queries to retrieve for - skip to generation
        return "generate_response"


def has_documents(state: Dict[str, Any]) -> str:
    """
    Check if documents were retrieved.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    retrieved_docs = state.get("retrieved_docs", [])
    
    if retrieved_docs:
        return "rank_context"
    else:
        # No documents found - generate empty response
        return "generate_response"


def route_by_complexity(state: Dict[str, Any]) -> str:
    """
    Route workflow based on query complexity.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    query_complexity = state.get("query_complexity", "complex")
    needs_new_retrieval = state.get("needs_new_retrieval", True)
    
    if query_complexity == "simple_conversational":
        return "generate_conversational"
    elif query_complexity == "follow_up":
        if needs_new_retrieval:
            # Need new sources, go through full pipeline
            return "classify_query"
        else:
            # Can use existing sources, skip to generation
            return "generate_response"
    elif query_complexity == "simple_factual":
        # Use light retrieval
        return "classify_query_simple"  # Will route to light retrieval
    else:
        # Complex query, full pipeline
        return "classify_query"


def route_simple_query(state: Dict[str, Any]) -> str:
    """
    Route simple factual queries to light retrieval.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    return "light_retrieve"


# ============================================================================
# Error Handling Node
# ============================================================================

def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle errors in the workflow.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with error message
    """
    error = state.get("error", "Unknown error occurred")
    
    print(f"❌ Error in workflow: {error}")
    
    return {
        "response": f"I apologize, but I encountered an error: {error}",
        "citations": [],
    }


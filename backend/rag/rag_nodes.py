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
import json
import re

from backend.core.models import (
    RAGState,
    QuestionType,
    Message,
    DocumentChunk,
)
from backend.rag.prompts import (
    QUERY_CLASSIFIER_PROMPT,
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
# Node 1: Query Classification
# ============================================================================

def classify_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the user query into a question type.
    
    This determines the retrieval strategy and response format.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with question_type
    """
    user_query = state.get("user_query", "")
    
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
            "question_type": question_type.value,
        }
    
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return {
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
        
        # Filter by score threshold
        filtered_docs = [
            doc for doc in ranked_docs 
            if doc.score is None or doc.score >= score_threshold
        ]
        
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
    
    Synthesizes information from sources with proper citations.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with response
    """
    user_query = state.get("user_query", "")
    question_type = QuestionType(state.get("question_type", "general"))
    ranked_docs_dict = state.get("ranked_docs", [])
    
    print(f"🤖 Generating response...")
    
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
        
        if not ranked_docs:
            response = (
                "I apologize, but I couldn't find relevant sources in the Islamic knowledge base "
                "to answer your question. Please try rephrasing your question or ask about a different topic."
            )
            return {"response": response}
        
        # Format context for LLM
        context = format_context_for_llm(
            documents=ranked_docs,
            question_type=question_type,
            max_sources=len(ranked_docs),
        )
        
        # Get appropriate generation prompt
        generation_prompt_template = get_generation_prompt(question_type)
        
        # Format generation prompt
        generation_prompt = format_prompt(
            generation_prompt_template,
            user_query=user_query,
            context=context,
        )
        
        # Get LLM
        llm = get_llm(backend=config.LLM_BACKEND)
        
        # Create messages
        messages = [
            create_system_message(include_identity=True),
            {"role": "user", "content": generation_prompt},
        ]
        
        # Generate response
        response = llm.chat(messages)
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


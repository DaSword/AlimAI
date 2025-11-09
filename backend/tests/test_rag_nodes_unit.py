"""
Unit tests for RAG nodes with mocked dependencies.

These tests are fast and isolated, testing node logic without external dependencies.
Run with: pytest backend/tests/test_rag_nodes_unit.py -m unit
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from backend.rag.rag_nodes import (
    format_conversation_history,
    analyze_query_complexity_node,
    classify_query_node,
    expand_query_node,
    retrieve_node,
    light_retrieve_node,
    rank_context_node,
    generate_response_node,
    generate_conversational_response_node,
    format_citations_node,
    update_messages_node,
    route_by_complexity,
    route_by_question_type,
    should_retrieve,
    has_documents,
)
from backend.core.models import QuestionType
from backend.tests.fixtures import (
    create_simple_conversation,
    create_multi_turn_conversation,
    get_sample_documents,
    create_test_state,
)


# ============================================================================
# Helper Function Tests
# ============================================================================

@pytest.mark.unit
class TestFormatConversationHistory:
    """Test the format_conversation_history helper function."""
    
    def test_empty_conversation(self):
        """Test formatting an empty conversation."""
        result = format_conversation_history([])
        assert result == "No previous conversation."
    
    def test_simple_conversation(self):
        """Test formatting a simple conversation."""
        messages = create_simple_conversation()
        # Add a current query at the end
        messages.append({"role": "user", "content": "Current query"})
        result = format_conversation_history(messages)
        
        assert "User:" in result
        assert "Assistant:" in result
        assert "What is fasting in Islam?" in result
        assert "Current query" not in result  # Current query should be excluded
    
    def test_excludes_last_user_message(self):
        """Test that the last user message is excluded."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Current query"},
        ]
        result = format_conversation_history(messages)
        
        assert "First question" in result
        assert "First answer" in result
        assert "Current query" not in result
    
    def test_max_turns_limit(self):
        """Test that max_turns limits the history."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})
        
        # Add current query
        messages.append({"role": "user", "content": "Current query"})
        
        result = format_conversation_history(messages, max_turns=2)
        
        # Should only include last 2 turns (4 messages) before current query
        assert "Question 8" in result or "Question 7" in result
        assert "Question 0" not in result
    
    def test_truncates_long_responses(self):
        """Test that long assistant responses are truncated."""
        long_response = "x" * 600  # More than 500 chars
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": long_response},
            {"role": "user", "content": "Current query"},  # Add current query
        ]
        result = format_conversation_history(messages)
        
        assert "..." in result
        assert len(result) < len(long_response) + 100  # Should be truncated


# ============================================================================
# Node 0: Complexity Analysis Tests
# ============================================================================

@pytest.mark.unit
class TestAnalyzeQueryComplexityNode:
    """Test the analyze_query_complexity_node."""
    
    def test_simple_conversational_greeting(self, patch_llm):
        """Test classification of simple greetings."""
        state = {
            "messages": [{"role": "user", "content": "Assalamu alaikum"}],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        assert "user_query" in result
        assert result["user_query"] == "Assalamu alaikum"
    
    def test_complex_query(self, patch_llm):
        """Test classification of complex queries."""
        state = {
            "messages": [{"role": "user", "content": "What is the Islamic ruling on mortgage interest?"}],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        assert "needs_new_retrieval" in result
    
    def test_follow_up_query(self, patch_llm):
        """Test classification of follow-up queries."""
        messages = create_simple_conversation()
        messages.append({"role": "user", "content": "Tell me more about that"})
        
        state = {"messages": messages}
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        assert "needs_new_retrieval" in result
    
    def test_extracts_query_from_messages(self, patch_llm):
        """Test that query is extracted from messages array."""
        state = {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Second question"},
            ],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert result["user_query"] == "Second question"
    
    def test_fallback_to_user_query_field(self, patch_llm):
        """Test fallback when messages array is empty."""
        state = {
            "user_query": "Direct query",
            "messages": [],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert result["user_query"] == "Direct query"
    
    def test_error_handling(self, patch_llm):
        """Test error handling returns default complexity."""
        patch_llm.complete.side_effect = Exception("LLM Error")
        
        state = {
            "user_query": "Test query",
            "messages": [],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert result["query_complexity"] == "complex"
        assert result["needs_new_retrieval"] == True


# ============================================================================
# Node 1: Query Classification Tests
# ============================================================================

@pytest.mark.unit
class TestClassifyQueryNode:
    """Test the classify_query_node."""
    
    def test_classify_fiqh_question(self, patch_llm):
        """Test classification of fiqh questions."""
        state = {"user_query": "Is it permissible to pray with shoes?"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
        assert "user_query" in result
    
    def test_classify_aqidah_question(self, patch_llm):
        """Test classification of aqidah questions."""
        state = {"user_query": "What are the attributes of Allah?"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
    
    def test_classify_tafsir_question(self, patch_llm):
        """Test classification of tafsir questions."""
        state = {"user_query": "Explain Surah Al-Fatiha verse 1"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
    
    def test_classify_hadith_question(self, patch_llm):
        """Test classification of hadith questions."""
        state = {"user_query": "What is the hadith about intentions?"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
    
    def test_extract_query_from_messages(self, patch_llm):
        """Test extracting query from messages when user_query is empty."""
        messages = [
            {"role": "user", "content": "What is charity?"},
            {"role": "assistant", "content": "Charity is..."},
            {"role": "user", "content": "Tell me about Zakat"},
        ]
        state = {"messages": messages}
        
        result = classify_query_node(state)
        
        assert result["user_query"] == "Tell me about Zakat"
    
    def test_error_handling(self, patch_llm):
        """Test error handling returns default classification."""
        patch_llm.complete.side_effect = Exception("LLM Error")
        
        state = {"user_query": "Test query"}
        
        result = classify_query_node(state)
        
        assert result["question_type"] == QuestionType.GENERAL.value


# ============================================================================
# Node 2: Query Expansion Tests
# ============================================================================

@pytest.mark.unit
class TestExpandQueryNode:
    """Test the expand_query_node."""
    
    def test_expand_query_success(self, patch_expand_query):
        """Test successful query expansion."""
        state = {
            "user_query": "What is charity?",
            "question_type": QuestionType.GENERAL.value,
        }
        
        result = expand_query_node(state)
        
        assert "expanded_queries" in result
        assert isinstance(result["expanded_queries"], list)
        assert len(result["expanded_queries"]) > 0
    
    def test_expand_fiqh_query(self, patch_expand_query):
        """Test expansion of fiqh query."""
        state = {
            "user_query": "Is mortgage permissible?",
            "question_type": QuestionType.FIQH.value,
        }
        
        result = expand_query_node(state)
        
        assert "expanded_queries" in result
        assert len(result["expanded_queries"]) > 0
    
    def test_error_handling(self, patch_expand_query):
        """Test error handling returns original query."""
        with patch('backend.rag.rag_nodes.expand_query_with_llm', side_effect=Exception("Expansion error")):
            state = {
                "user_query": "Test query",
                "question_type": QuestionType.GENERAL.value,
            }
            
            result = expand_query_node(state)
            
            assert result["expanded_queries"] == ["Test query"]


# ============================================================================
# Node 3: Retrieval Tests
# ============================================================================

@pytest.mark.unit
class TestRetrieveNode:
    """Test the retrieve_node."""
    
    def test_retrieve_documents(self, patch_retriever):
        """Test successful document retrieval."""
        state = {
            "user_query": "What is fasting?",
            "question_type": QuestionType.GENERAL.value,
            "expanded_queries": ["What is fasting in Islam?", "What is Sawm?"],
            "max_sources": 10,
        }
        
        result = retrieve_node(state)
        
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
    
    def test_retrieve_with_madhab_preference(self, patch_retriever):
        """Test retrieval with madhab filtering."""
        state = {
            "user_query": "What is the ruling on X?",
            "question_type": QuestionType.FIQH.value,
            "expanded_queries": ["Ruling on X"],
            "max_sources": 10,
            "madhab_preference": "hanafi",
        }
        
        result = retrieve_node(state)
        
        assert "retrieved_docs" in result
    
    def test_deduplicate_documents(self, patch_retriever):
        """Test that duplicate documents are removed."""
        state = {
            "user_query": "Test query",
            "question_type": QuestionType.GENERAL.value,
            "expanded_queries": ["Query 1", "Query 2", "Query 3"],
            "max_sources": 10,
        }
        
        result = retrieve_node(state)
        
        # Check that IDs are unique
        doc_ids = [doc["id"] for doc in result["retrieved_docs"]]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate documents found"
    
    def test_error_handling(self, patch_retriever):
        """Test error handling returns empty list."""
        with patch('backend.rag.rag_nodes.IslamicRetriever', side_effect=Exception("Retrieval error")):
            state = {
                "user_query": "Test query",
                "question_type": QuestionType.GENERAL.value,
                "expanded_queries": ["Test"],
                "max_sources": 10,
            }
            
            result = retrieve_node(state)
            
            assert result["retrieved_docs"] == []


@pytest.mark.unit
class TestLightRetrieveNode:
    """Test the light_retrieve_node."""
    
    def test_light_retrieval(self, patch_retriever):
        """Test light retrieval with fewer sources."""
        state = {
            "user_query": "Who was Prophet Muhammad?",
            "question_type": QuestionType.GENERAL.value,
        }
        
        result = light_retrieve_node(state)
        
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
    
    def test_error_handling(self, patch_retriever):
        """Test error handling returns empty list."""
        with patch('backend.rag.rag_nodes.IslamicRetriever', side_effect=Exception("Retrieval error")):
            state = {
                "user_query": "Test query",
                "question_type": QuestionType.GENERAL.value,
            }
            
            result = light_retrieve_node(state)
            
            assert result["retrieved_docs"] == []


# ============================================================================
# Node 4: Context Ranking Tests
# ============================================================================

@pytest.mark.unit
class TestRankContextNode:
    """Test the rank_context_node."""
    
    def test_rank_documents(self, sample_documents):
        """Test document ranking."""
        state = create_test_state(
            user_query="What is fasting?",
            question_type=QuestionType.GENERAL.value,
            retrieved_docs=sample_documents,
            max_sources=10,
            score_threshold=0.7,
        )
        
        result = rank_context_node(state)
        
        assert "ranked_docs" in result
        assert isinstance(result["ranked_docs"], list)
        assert len(result["ranked_docs"]) > 0
    
    def test_score_threshold_filtering(self, sample_documents):
        """Test that documents below threshold are filtered."""
        # Set some docs to low scores, some to acceptable scores
        docs_for_test = sample_documents[:5]
        docs_for_test[0].score = 0.95  # This one is good
        docs_for_test[1].score = 0.80  # This one is good
        docs_for_test[2].score = 0.75  # This one is good
        docs_for_test[3].score = 0.60  # Below threshold
        docs_for_test[4].score = 0.50  # Below threshold
        
        state = create_test_state(
            user_query="Test query",
            question_type=QuestionType.GENERAL.value,
            retrieved_docs=docs_for_test,
            max_sources=10,
            score_threshold=0.7,
        )
        
        result = rank_context_node(state)
        
        # Should keep documents above threshold
        assert len(result["ranked_docs"]) >= 3, "Should keep at least 3 documents above threshold"
    
    def test_max_sources_limit(self, sample_documents):
        """Test that max_sources limits the output."""
        state = create_test_state(
            user_query="Test query",
            question_type=QuestionType.GENERAL.value,
            retrieved_docs=sample_documents,
            max_sources=3,
            score_threshold=0.5,
        )
        
        result = rank_context_node(state)
        
        assert len(result["ranked_docs"]) <= 3
    
    def test_error_handling(self, sample_documents):
        """Test error handling falls back to retrieved docs."""
        state = create_test_state(
            user_query="Test query",
            question_type=QuestionType.GENERAL.value,
            retrieved_docs=sample_documents,
            max_sources=5,
        )
        
        with patch('backend.rag.rag_nodes.rank_documents', side_effect=Exception("Ranking error")):
            result = rank_context_node(state)
            
            assert "ranked_docs" in result
            assert len(result["ranked_docs"]) <= 5


# ============================================================================
# Node 5: Response Generation Tests
# ============================================================================

@pytest.mark.unit
class TestGenerateResponseNode:
    """Test the generate_response_node."""
    
    def test_generate_with_context(self, patch_llm, sample_documents):
        """Test response generation with context."""
        state = create_test_state(
            user_query="What is fasting?",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=sample_documents[:3],
            messages=create_simple_conversation(),
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
    
    def test_generate_without_context(self, patch_llm):
        """Test response generation without context."""
        state = create_test_state(
            user_query="What is fasting?",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=[],
            messages=create_simple_conversation(),
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
        assert isinstance(result["response"], str)
    
    def test_includes_conversation_history(self, patch_llm, sample_documents):
        """Test that conversation history is included."""
        messages = create_multi_turn_conversation()
        state = create_test_state(
            user_query="Who can receive Zakat?",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=sample_documents[:3],
            messages=messages,
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
    
    def test_error_handling(self, patch_llm):
        """Test error handling returns error message."""
        patch_llm.chat.side_effect = Exception("Generation error")
        
        state = create_test_state(
            user_query="Test query",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=[],
            messages=[],
        )
        
        result = generate_response_node(state)
        
        assert "error" in result["response"].lower()


@pytest.mark.unit
class TestGenerateConversationalResponseNode:
    """Test the generate_conversational_response_node."""
    
    def test_generate_greeting_response(self, patch_llm):
        """Test generating response to greeting."""
        state = {
            "user_query": "Assalamu alaikum",
            "messages": [],
        }
        
        result = generate_conversational_response_node(state)
        
        assert "response" in result
        assert isinstance(result["response"], str)
    
    def test_generate_thanks_response(self, patch_llm):
        """Test generating response to thanks."""
        state = {
            "user_query": "Thank you",
            "messages": create_simple_conversation(),
        }
        
        result = generate_conversational_response_node(state)
        
        assert "response" in result
    
    def test_error_handling(self, patch_llm):
        """Test error handling returns default message."""
        patch_llm.chat.side_effect = Exception("Generation error")
        
        state = {
            "user_query": "Hello",
            "messages": [],
        }
        
        result = generate_conversational_response_node(state)
        
        assert "response" in result
        assert "help" in result["response"].lower()


# ============================================================================
# Node 6: Citation Formatting Tests
# ============================================================================

@pytest.mark.unit
class TestFormatCitationsNode:
    """Test the format_citations_node."""
    
    def test_format_citations(self, sample_documents):
        """Test citation formatting."""
        state = create_test_state(
            ranked_docs=sample_documents[:3],
        )
        
        result = format_citations_node(state)
        
        assert "citations" in result
        assert isinstance(result["citations"], list)
    
    def test_empty_documents(self):
        """Test formatting with no documents."""
        state = create_test_state(ranked_docs=[])
        
        result = format_citations_node(state)
        
        assert result["citations"] == []
    
    def test_error_handling(self, sample_documents):
        """Test error handling returns empty list."""
        state = create_test_state(ranked_docs=sample_documents)
        
        with patch('backend.rag.rag_nodes.create_citation_list', side_effect=Exception("Citation error")):
            result = format_citations_node(state)
            
            assert result["citations"] == []


# ============================================================================
# Node 7: Message Update Tests
# ============================================================================

@pytest.mark.unit
class TestUpdateMessagesNode:
    """Test the update_messages_node."""
    
    def test_update_messages(self):
        """Test updating message history."""
        state = {
            "user_query": "What is fasting?",
            "response": "Fasting is...",
            "messages": create_simple_conversation(),
        }
        
        result = update_messages_node(state)
        
        assert "messages" in result
        messages = result["messages"]
        
        # Should have original messages + 2 new ones
        assert len(messages) == len(create_simple_conversation()) + 2
        
        # Last two should be user and assistant
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"] == "What is fasting?"
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Fasting is..."
    
    def test_empty_initial_messages(self):
        """Test updating with no initial messages."""
        state = {
            "user_query": "First question",
            "response": "First answer",
            "messages": [],
        }
        
        result = update_messages_node(state)
        
        assert len(result["messages"]) == 2


# ============================================================================
# Routing Function Tests
# ============================================================================

@pytest.mark.unit
class TestRoutingFunctions:
    """Test the routing/conditional edge functions."""
    
    def test_route_by_complexity_conversational(self):
        """Test routing for simple conversational queries."""
        state = {"query_complexity": "simple_conversational"}
        
        result = route_by_complexity(state)
        
        assert result == "generate_conversational"
    
    def test_route_by_complexity_follow_up_no_retrieval(self):
        """Test routing for follow-up without new retrieval."""
        state = {
            "query_complexity": "follow_up",
            "needs_new_retrieval": False,
        }
        
        result = route_by_complexity(state)
        
        assert result == "generate_response"
    
    def test_route_by_complexity_follow_up_with_retrieval(self):
        """Test routing for follow-up with new retrieval."""
        state = {
            "query_complexity": "follow_up",
            "needs_new_retrieval": True,
        }
        
        result = route_by_complexity(state)
        
        assert result == "classify_query"
    
    def test_route_by_complexity_simple_factual(self):
        """Test routing for simple factual queries."""
        state = {"query_complexity": "simple_factual"}
        
        result = route_by_complexity(state)
        
        assert result == "classify_query_simple"
    
    def test_route_by_complexity_complex(self):
        """Test routing for complex queries."""
        state = {"query_complexity": "complex"}
        
        result = route_by_complexity(state)
        
        assert result == "classify_query"
    
    def test_route_by_question_type(self):
        """Test routing by question type."""
        state = {"question_type": QuestionType.FIQH.value}
        
        result = route_by_question_type(state)
        
        assert result == "expand_query"
    
    def test_should_retrieve_with_queries(self):
        """Test should_retrieve with expanded queries."""
        state = {"expanded_queries": ["Query 1", "Query 2"]}
        
        result = should_retrieve(state)
        
        assert result == "retrieve"
    
    def test_should_retrieve_without_queries(self):
        """Test should_retrieve without queries."""
        state = {"expanded_queries": []}
        
        result = should_retrieve(state)
        
        assert result == "generate_response"
    
    def test_has_documents_true(self, sample_documents):
        """Test has_documents when documents exist."""
        state = create_test_state(retrieved_docs=sample_documents)
        
        result = has_documents(state)
        
        assert result == "rank_context"
    
    def test_has_documents_false(self):
        """Test has_documents when no documents."""
        state = create_test_state(retrieved_docs=[])
        
        result = has_documents(state)
        
        assert result == "generate_response"


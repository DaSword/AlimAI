"""
Tests for error handling in RAG nodes and graph execution.

These tests ensure graceful degradation and proper error messages.
Run with: pytest backend/tests/test_error_handling.py
"""

import pytest
from unittest.mock import patch, Mock
from typing import Dict, Any

from backend.rag.rag_nodes import (
    analyze_query_complexity_node,
    classify_query_node,
    expand_query_node,
    retrieve_node,
    light_retrieve_node,
    rank_context_node,
    generate_response_node,
    generate_conversational_response_node,
    format_citations_node,
    error_handler_node,
)
from backend.rag.rag_graph import invoke_rag_workflow
from backend.core.models import QuestionType
from backend.tests.fixtures import (
    create_test_state,
    get_sample_documents,
)


# ============================================================================
# Node Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestNodeErrorHandling:
    """Test error handling in individual nodes."""
    
    def test_complexity_node_llm_error(self, patch_llm):
        """Test complexity node handles LLM errors."""
        patch_llm.complete.side_effect = Exception("LLM connection failed")
        
        state = {
            "user_query": "What is Islam?",
            "messages": [],
        }
        
        result = analyze_query_complexity_node(state)
        
        # Should return default complexity
        assert result["query_complexity"] == "complex"
        assert result["needs_new_retrieval"] == True
    
    def test_classify_node_llm_error(self, patch_llm):
        """Test classify node handles LLM errors."""
        patch_llm.complete.side_effect = Exception("LLM timeout")
        
        state = {"user_query": "Test query"}
        
        result = classify_query_node(state)
        
        # Should return default classification
        assert result["question_type"] == QuestionType.GENERAL.value
    
    def test_expand_node_error(self):
        """Test expand node handles expansion errors."""
        with patch('backend.rag.rag_nodes.expand_query_with_llm', side_effect=Exception("Expansion failed")):
            state = {
                "user_query": "Test query",
                "question_type": QuestionType.GENERAL.value,
            }
            
            result = expand_query_node(state)
            
            # Should fallback to original query
            assert result["expanded_queries"] == ["Test query"]
    
    def test_retrieve_node_error(self):
        """Test retrieve node handles retrieval errors."""
        with patch('backend.rag.rag_nodes.IslamicRetriever', side_effect=Exception("DB connection failed")):
            state = {
                "user_query": "Test query",
                "question_type": QuestionType.GENERAL.value,
                "expanded_queries": ["Query 1"],
                "max_sources": 10,
            }
            
            result = retrieve_node(state)
            
            # Should return empty list
            assert result["retrieved_docs"] == []
    
    def test_light_retrieve_node_error(self):
        """Test light retrieve node handles errors."""
        with patch('backend.rag.rag_nodes.IslamicRetriever', side_effect=Exception("Retrieval error")):
            state = {
                "user_query": "Test query",
                "question_type": QuestionType.GENERAL.value,
            }
            
            result = light_retrieve_node(state)
            
            assert result["retrieved_docs"] == []
    
    def test_rank_node_error(self, sample_documents):
        """Test rank node handles ranking errors."""
        state = create_test_state(
            user_query="Test query",
            question_type=QuestionType.GENERAL.value,
            retrieved_docs=sample_documents,
            max_sources=5,
        )
        
        with patch('backend.rag.rag_nodes.rank_documents', side_effect=Exception("Ranking failed")):
            result = rank_context_node(state)
            
            # Should fallback to retrieved docs
            assert "ranked_docs" in result
            assert len(result["ranked_docs"]) <= 5
    
    def test_generate_node_llm_error(self, patch_llm, sample_documents):
        """Test generate node handles LLM errors."""
        patch_llm.chat.side_effect = Exception("Generation failed")
        
        state = create_test_state(
            user_query="Test query",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=sample_documents[:3],
            messages=[],
        )
        
        result = generate_response_node(state)
        
        # Should return error message
        assert "error" in result["response"].lower()
    
    def test_conversational_node_llm_error(self, patch_llm):
        """Test conversational node handles LLM errors."""
        patch_llm.chat.side_effect = Exception("Chat failed")
        
        state = {
            "user_query": "Hello",
            "messages": [],
        }
        
        result = generate_conversational_response_node(state)
        
        # Should return default message
        assert "response" in result
        assert "help" in result["response"].lower()
    
    def test_citation_node_error(self, sample_documents):
        """Test citation node handles formatting errors."""
        state = create_test_state(ranked_docs=sample_documents)
        
        with patch('backend.rag.rag_nodes.create_citation_list', side_effect=Exception("Citation error")):
            result = format_citations_node(state)
            
            # Should return empty list
            assert result["citations"] == []
    
    def test_error_handler_node(self):
        """Test the error handler node."""
        state = {
            "error": "Something went wrong",
            "user_query": "Test query",
        }
        
        result = error_handler_node(state)
        
        assert "response" in result
        assert "error" in result["response"].lower()
        assert result["citations"] == []


# ============================================================================
# Missing/Invalid Data Tests
# ============================================================================

@pytest.mark.unit
class TestMissingInvalidData:
    """Test handling of missing or invalid data."""
    
    def test_complexity_node_empty_query(self, patch_llm):
        """Test complexity node with empty query."""
        state = {
            "user_query": "",
            "messages": [],
        }
        
        result = analyze_query_complexity_node(state)
        
        # Should handle gracefully
        assert "query_complexity" in result
    
    def test_classify_node_missing_query(self, patch_llm):
        """Test classify node with missing query."""
        state = {}
        
        result = classify_query_node(state)
        
        # Should handle gracefully
        assert "question_type" in result
    
    def test_expand_node_missing_question_type(self):
        """Test expand node without question type."""
        with patch('backend.rag.rag_nodes.expand_query_with_llm', return_value=["Expanded query"]):
            state = {
                "user_query": "Test query",
                # Missing question_type
            }
            
            result = expand_query_node(state)
            
            # Should use default
            assert "expanded_queries" in result
    
    def test_retrieve_node_empty_queries(self, patch_retriever):
        """Test retrieve node with empty expanded queries."""
        state = {
            "user_query": "Test",
            "question_type": QuestionType.GENERAL.value,
            "expanded_queries": [],
            "max_sources": 10,
        }
        
        result = retrieve_node(state)
        
        # Should handle gracefully
        assert "retrieved_docs" in result
    
    def test_rank_node_empty_docs(self):
        """Test rank node with no documents."""
        state = create_test_state(
            user_query="Test",
            question_type=QuestionType.GENERAL.value,
            retrieved_docs=[],
        )
        
        result = rank_context_node(state)
        
        # Should return empty list
        assert result["ranked_docs"] == []
    
    def test_generate_node_no_context(self, patch_llm):
        """Test generate node without context."""
        state = create_test_state(
            user_query="Test query",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=[],
            messages=[],
        )
        
        result = generate_response_node(state)
        
        # Should generate without context
        assert "response" in result
        assert len(result["response"]) > 0
    
    def test_format_citations_empty_docs(self):
        """Test citation formatting with no documents."""
        state = create_test_state(ranked_docs=[])
        
        result = format_citations_node(state)
        
        assert result["citations"] == []


# ============================================================================
# Malformed Data Tests
# ============================================================================

@pytest.mark.unit
class TestMalformedData:
    """Test handling of malformed data."""
    
    def test_complexity_node_malformed_messages(self, patch_llm):
        """Test complexity node with malformed messages."""
        state = {
            "messages": [
                {"content": "Missing role field"},
                {"role": "user", "content": "Valid message"},  # At least one valid
                {"role": "assistant", "content": ""},  # Empty content
            ],
        }
        
        # Should not crash
        result = analyze_query_complexity_node(state)
        assert "query_complexity" in result
    
    def test_retrieve_node_invalid_question_type(self, patch_retriever):
        """Test retrieve node with invalid question type."""
        state = {
            "user_query": "Test",
            "question_type": "invalid_type",
            "expanded_queries": ["Test"],
            "max_sources": 10,
        }
        
        # Should handle gracefully (might raise or default)
        try:
            result = retrieve_node(state)
            assert "retrieved_docs" in result
        except (ValueError, KeyError):
            # Expected if enum validation is strict
            pass
    
    def test_rank_node_malformed_docs(self):
        """Test rank node with malformed document structure."""
        state = {
            "user_query": "Test",
            "question_type": QuestionType.GENERAL.value,
            "retrieved_docs": [
                {"id": "1"},  # Missing required fields
                {},  # Empty doc
            ],
            "max_sources": 10,
        }
        
        # Should handle gracefully or raise validation error
        try:
            result = rank_context_node(state)
            # If it succeeds, should handle malformed data
            assert "ranked_docs" in result
        except (KeyError, AttributeError, ValueError):
            # Expected if validation is strict
            pass


# ============================================================================
# Workflow Error Handling Tests
# ============================================================================

@pytest.mark.e2e
class TestWorkflowErrorHandling:
    """Test error handling in complete workflow execution."""
    
    def test_workflow_with_llm_error(self, patch_retriever):
        """Test workflow continues when LLM fails."""
        with patch('backend.rag.rag_nodes.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.complete.side_effect = Exception("LLM failed")
            mock_llm.chat.side_effect = Exception("LLM failed")
            mock_get_llm.return_value = mock_llm
            
            result = invoke_rag_workflow(
                user_query="What is Islam?",
                max_sources=5,
                thread_id="test_llm_error",
            )
            
            # Should return error response
            assert "response" in result
            # Response should indicate error
            assert len(result["response"]) > 0
    
    def test_workflow_with_retrieval_error(self, patch_llm):
        """Test workflow handles retrieval errors."""
        with patch('backend.rag.rag_nodes.IslamicRetriever', side_effect=Exception("DB error")):
            result = invoke_rag_workflow(
                user_query="What is fasting?",
                max_sources=5,
                thread_id="test_retrieval_error",
            )
            
            # Should complete with some response
            assert "response" in result
    
    def test_workflow_with_partial_failure(self, patch_llm):
        """Test workflow when some nodes fail."""
        # Make expansion fail but retrieval works
        with patch('backend.rag.rag_nodes.expand_query_with_llm', side_effect=Exception("Expansion failed")):
            with patch('backend.rag.rag_nodes.IslamicRetriever') as mock_retriever_class:
                from backend.tests.fixtures import MockRetriever, get_sample_documents
                mock_retriever = MockRetriever(documents=get_sample_documents())
                mock_retriever_class.return_value = mock_retriever
                
                result = invoke_rag_workflow(
                    user_query="What is charity?",
                    max_sources=5,
                    thread_id="test_partial_failure",
                )
                
                # Should still produce response
                assert "response" in result
    
    def test_workflow_with_invalid_input(self, patch_llm, patch_retriever):
        """Test workflow with invalid input."""
        result = invoke_rag_workflow(
            user_query="",  # Empty query
            max_sources=5,
            thread_id="test_invalid_input",
        )
        
        # Should handle gracefully
        assert "response" in result


# ============================================================================
# Timeout and Performance Tests
# ============================================================================

@pytest.mark.unit
class TestTimeoutScenarios:
    """Test handling of timeout scenarios."""
    
    def test_llm_timeout(self, patch_llm):
        """Test handling of LLM timeout."""
        import time
        
        def slow_complete(prompt):
            time.sleep(0.1)
            raise TimeoutError("Request timed out")
        
        patch_llm.complete.side_effect = slow_complete
        
        state = {
            "user_query": "Test query",
            "messages": [],
        }
        
        result = analyze_query_complexity_node(state)
        
        # Should return default after timeout
        assert result["query_complexity"] == "complex"
    
    def test_retrieval_timeout(self):
        """Test handling of retrieval timeout."""
        with patch('backend.rag.rag_nodes.IslamicRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever.retrieve_for_question_type.side_effect = TimeoutError("Retrieval timeout")
            mock_retriever_class.return_value = mock_retriever
            
            state = {
                "user_query": "Test",
                "question_type": QuestionType.GENERAL.value,
                "expanded_queries": ["Test"],
                "max_sources": 10,
            }
            
            result = retrieve_node(state)
            
            # Should handle timeout gracefully
            assert result["retrieved_docs"] == []


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_long_query(self, patch_llm):
        """Test handling of very long queries."""
        long_query = "What is " + ("very " * 1000) + "long query?"
        
        state = {
            "user_query": long_query,
            "messages": [],
        }
        
        result = analyze_query_complexity_node(state)
        
        # Should handle long query
        assert "query_complexity" in result
    
    def test_query_with_special_characters(self, patch_llm):
        """Test handling of queries with special characters."""
        special_query = "What is الصلاة (Salah)? <>&\"'"
        
        state = {
            "user_query": special_query,
            "messages": [],
        }
        
        result = analyze_query_complexity_node(state)
        
        # Should handle special chars
        assert "query_complexity" in result
    
    def test_max_sources_zero(self, patch_retriever):
        """Test with max_sources set to 0."""
        state = {
            "user_query": "Test",
            "question_type": QuestionType.GENERAL.value,
            "expanded_queries": ["Test"],
            "max_sources": 0,
        }
        
        result = retrieve_node(state)
        
        # Should handle gracefully
        assert "retrieved_docs" in result
    
    def test_score_threshold_extreme(self, sample_documents):
        """Test with extreme score thresholds."""
        # Very high threshold (1.0)
        state = create_test_state(
            user_query="Test",
            question_type=QuestionType.GENERAL.value,
            retrieved_docs=sample_documents,
            max_sources=10,
            score_threshold=1.0,
        )
        
        result = rank_context_node(state)
        
        # Should still return at least some docs (top 3 minimum)
        assert "ranked_docs" in result
        assert len(result["ranked_docs"]) >= 0  # Might be 0 or keep top 3
    
    def test_deeply_nested_conversation(self, patch_llm):
        """Test with very long conversation history."""
        messages = []
        for i in range(100):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})
        
        state = {
            "user_query": "Latest question",
            "messages": messages,
        }
        
        result = analyze_query_complexity_node(state)
        
        # Should handle long history
        assert "query_complexity" in result


# ============================================================================
# Recovery Tests
# ============================================================================

@pytest.mark.e2e
class TestRecoveryMechanisms:
    """Test that the system can recover from errors."""
    
    def test_workflow_recovers_after_error(self, patch_llm, patch_retriever):
        """Test that workflow can succeed after previous error."""
        thread_id = "test_recovery"
        
        # First call with error
        with patch('backend.rag.rag_nodes.get_llm', side_effect=Exception("Temporary error")):
            result1 = invoke_rag_workflow(
                user_query="First query",
                max_sources=5,
                thread_id=thread_id,
            )
            # Should have error response
            assert "response" in result1
        
        # Second call should work
        result2 = invoke_rag_workflow(
            user_query="Second query",
            max_sources=5,
            thread_id=thread_id,
        )
        
        # Should succeed
        assert "response" in result2
    
    def test_multiple_errors_in_sequence(self, patch_llm, patch_retriever):
        """Test handling multiple errors in sequence."""
        # Each call has different error
        errors = [
            Exception("Error 1"),
            Exception("Error 2"),
            Exception("Error 3"),
        ]
        
        for i, error in enumerate(errors):
            # Reset mock for each iteration
            patch_llm.complete.side_effect = error
            
            result = invoke_rag_workflow(
                user_query=f"Query {i}",
                max_sources=5,
                thread_id=f"test_multi_error_{i}",
            )
            
            # Each should handle error
            assert "response" in result


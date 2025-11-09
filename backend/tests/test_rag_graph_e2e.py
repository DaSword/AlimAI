"""
End-to-end tests for the complete RAG workflow graph.

These tests validate the entire workflow from query to response.
Run with: pytest backend/tests/test_rag_graph_e2e.py -m e2e
"""

import pytest
from unittest.mock import patch, Mock
from typing import Dict, Any

from backend.rag.rag_graph import (
    create_rag_graph,
    invoke_rag_workflow,
    get_graph_visualization,
)
from backend.tests.fixtures import (
    get_sample_documents,
    create_simple_conversation,
    create_multi_turn_conversation,
    MockRetriever,
    validate_response_quality,
)


# ============================================================================
# Graph Structure Tests
# ============================================================================

@pytest.mark.unit
class TestGraphStructure:
    """Test the graph structure and visualization."""
    
    def test_create_graph(self):
        """Test that graph can be created."""
        graph = create_rag_graph()
        
        assert graph is not None
        # Graph should be compiled
        assert hasattr(graph, 'invoke')
    
    def test_graph_visualization(self):
        """Test graph visualization output."""
        viz = get_graph_visualization()
        
        assert "analyze_complexity" in viz
        assert "classify_query" in viz
        assert "retrieve" in viz
        assert "generate_response" in viz
        assert "simple_conversational" in viz
        assert "follow_up" in viz


# ============================================================================
# E2E Tests: Simple Conversational Path
# ============================================================================

@pytest.mark.e2e
class TestSimpleConversationalPath:
    """Test the simple conversational path (no retrieval)."""
    
    def test_greeting_flow(self, patch_llm, patch_retriever):
        """Test greeting goes through conversational path."""
        result = invoke_rag_workflow(
            user_query="Assalamu alaikum",
            max_sources=5,
            thread_id="test_greeting",
        )
        
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        
        print(f"\nGreeting flow response: {result['response']}")
    
    def test_thanks_flow(self, patch_llm, patch_retriever):
        """Test thanks goes through conversational path."""
        result = invoke_rag_workflow(
            user_query="Thank you for the explanation",
            max_sources=5,
            thread_id="test_thanks",
        )
        
        assert "response" in result
        print(f"\nThanks flow response: {result['response']}")
    
    def test_acknowledgment_flow(self, patch_llm, patch_retriever):
        """Test acknowledgment goes through conversational path."""
        result = invoke_rag_workflow(
            user_query="Ok, I understand",
            max_sources=5,
            thread_id="test_ack",
        )
        
        assert "response" in result
        print(f"\nAcknowledgment flow response: {result['response']}")


# ============================================================================
# E2E Tests: Simple Factual Path
# ============================================================================

@pytest.mark.e2e
class TestSimpleFactualPath:
    """Test the simple factual path (light retrieval)."""
    
    def test_simple_who_question(self, patch_llm, patch_retriever):
        """Test simple 'who' question."""
        result = invoke_rag_workflow(
            user_query="Who was Prophet Muhammad?",
            max_sources=5,
            thread_id="test_who",
        )
        
        assert "response" in result
        assert "citations" in result
        
        print(f"\nSimple 'who' question response length: {len(result['response'])}")
        print(f"Citations: {len(result.get('citations', []))}")
    
    def test_simple_what_question(self, patch_llm, patch_retriever):
        """Test simple 'what' question."""
        result = invoke_rag_workflow(
            user_query="What is Islam?",
            max_sources=5,
            thread_id="test_what",
        )
        
        assert "response" in result
        print(f"\nSimple 'what' question response length: {len(result['response'])}")


# ============================================================================
# E2E Tests: Complex Query Path
# ============================================================================

@pytest.mark.e2e
class TestComplexQueryPath:
    """Test the complex query path (full pipeline)."""
    
    def test_complex_fiqh_question(self, patch_llm, patch_retriever, patch_expand_query):
        """Test complex fiqh question goes through full pipeline."""
        result = invoke_rag_workflow(
            user_query="What is the Islamic ruling on mortgage interest?",
            max_sources=10,
            thread_id="test_complex_fiqh",
        )
        
        assert "response" in result
        assert "citations" in result
        
        print(f"\nComplex fiqh response length: {len(result['response'])}")
        print(f"Citations: {len(result.get('citations', []))}")
    
    def test_complex_aqidah_question(self, patch_llm, patch_retriever, patch_expand_query):
        """Test complex aqidah question."""
        result = invoke_rag_workflow(
            user_query="What are the attributes of Allah according to Islamic theology?",
            max_sources=10,
            thread_id="test_complex_aqidah",
        )
        
        assert "response" in result
        print(f"\nComplex aqidah response length: {len(result['response'])}")
    
    def test_with_madhab_preference(self, patch_llm, patch_retriever, patch_expand_query):
        """Test query with madhab preference."""
        result = invoke_rag_workflow(
            user_query="What is the ruling on wiping over socks?",
            madhab_preference="hanafi",
            max_sources=10,
            thread_id="test_madhab",
        )
        
        assert "response" in result
        print(f"\nMadhab-specific response length: {len(result['response'])}")


# ============================================================================
# E2E Tests: Follow-up Query Path
# ============================================================================

@pytest.mark.e2e
class TestFollowUpPath:
    """Test the follow-up query paths."""
    
    def test_follow_up_with_existing_sources(self, patch_llm, patch_retriever):
        """Test follow-up that can use existing sources."""
        # This is harder to test in isolation, but we can verify the workflow completes
        result = invoke_rag_workflow(
            user_query="Tell me more about that",
            max_sources=5,
            thread_id="test_followup_existing",
        )
        
        assert "response" in result
        print(f"\nFollow-up (existing sources) response: {result['response'][:100]}...")
    
    def test_follow_up_needs_new_retrieval(self, patch_llm, patch_retriever, patch_expand_query):
        """Test follow-up that needs new retrieval."""
        result = invoke_rag_workflow(
            user_query="What about the Shafi'i view on this?",
            max_sources=10,
            thread_id="test_followup_new",
        )
        
        assert "response" in result
        print(f"\nFollow-up (new retrieval) response: {result['response'][:100]}...")


# ============================================================================
# E2E Tests: Multi-turn Conversations
# ============================================================================

@pytest.mark.e2e
class TestMultiTurnConversations:
    """Test multi-turn conversation flows."""
    
    def test_two_turn_conversation(self, patch_llm, patch_retriever, patch_expand_query):
        """Test a two-turn conversation."""
        # First query
        result1 = invoke_rag_workflow(
            user_query="What is fasting in Islam?",
            max_sources=5,
            thread_id="test_multiturn_1",
        )
        
        assert "response" in result1
        
        # Second query (follow-up)
        result2 = invoke_rag_workflow(
            user_query="When is it obligatory?",
            max_sources=5,
            thread_id="test_multiturn_1",  # Same thread
        )
        
        assert "response" in result2
        
        print(f"\nTurn 1: {result1['response'][:100]}...")
        print(f"Turn 2: {result2['response'][:100]}...")
    
    def test_three_turn_with_different_types(self, patch_llm, patch_retriever, patch_expand_query):
        """Test conversation with different query types."""
        thread_id = "test_multiturn_2"
        
        # Turn 1: Complex query
        result1 = invoke_rag_workflow(
            user_query="What is charity in Islam?",
            max_sources=5,
            thread_id=thread_id,
        )
        assert "response" in result1
        
        # Turn 2: Follow-up
        result2 = invoke_rag_workflow(
            user_query="Tell me more about Zakat",
            max_sources=5,
            thread_id=thread_id,
        )
        assert "response" in result2
        
        # Turn 3: Conversational
        result3 = invoke_rag_workflow(
            user_query="Thank you",
            max_sources=5,
            thread_id=thread_id,
        )
        assert "response" in result3


# ============================================================================
# E2E Tests: Response Quality
# ============================================================================

@pytest.mark.e2e
class TestResponseQuality:
    """Test that end-to-end responses meet quality standards."""
    
    def test_response_has_all_fields(self, patch_llm, patch_retriever):
        """Test that response has all required fields."""
        result = invoke_rag_workflow(
            user_query="What is prayer in Islam?",
            max_sources=5,
            thread_id="test_fields",
        )
        
        assert "user_query" in result
        assert "response" in result
        assert "citations" in result
        assert isinstance(result["citations"], list)
    
    def test_response_is_substantive(self, patch_llm, patch_retriever, patch_expand_query):
        """Test that responses are substantive."""
        result = invoke_rag_workflow(
            user_query="Explain the concept of Tawhid",
            max_sources=10,
            thread_id="test_substantive",
        )
        
        response = result["response"]
        
        # Should be a reasonable length
        assert len(response) > 50, "Response should be substantive"
        
        print(f"\nResponse length: {len(response)} characters")
    
    def test_citations_formatted_correctly(self, patch_llm, patch_retriever, patch_expand_query):
        """Test that citations are properly formatted."""
        result = invoke_rag_workflow(
            user_query="What does the Quran say about patience?",
            max_sources=5,
            thread_id="test_citations",
        )
        
        citations = result.get("citations", [])
        
        print(f"\nNumber of citations: {len(citations)}")
        
        # Each citation should be a dict
        for citation in citations:
            assert isinstance(citation, dict)
            # Should have some standard fields
            # (Exact fields depend on citation formatting implementation)


# ============================================================================
# E2E Tests: No Documents Scenario
# ============================================================================

@pytest.mark.e2e
class TestNoDocumentsScenario:
    """Test behavior when no documents are retrieved."""
    
    def test_no_documents_found(self, patch_llm, empty_retriever):
        """Test graceful handling when no documents found."""
        with patch('backend.rag.rag_nodes.IslamicRetriever', return_value=empty_retriever):
            result = invoke_rag_workflow(
                user_query="What is the Islamic view on quantum mechanics?",
                max_sources=5,
                thread_id="test_no_docs",
            )
            
            assert "response" in result
            assert len(result["response"]) > 0
            
            # Should have no citations
            assert len(result.get("citations", [])) == 0
            
            print(f"\nNo docs response: {result['response'][:200]}...")


# ============================================================================
# E2E Tests: Different Settings
# ============================================================================

@pytest.mark.e2e
class TestDifferentSettings:
    """Test workflow with different configuration settings."""
    
    def test_low_max_sources(self, patch_llm, patch_retriever):
        """Test with low max_sources."""
        result = invoke_rag_workflow(
            user_query="What is Salah?",
            max_sources=2,
            thread_id="test_low_sources",
        )
        
        assert "response" in result
        citations = result.get("citations", [])
        assert len(citations) <= 2
    
    def test_high_max_sources(self, patch_llm, patch_retriever):
        """Test with high max_sources."""
        result = invoke_rag_workflow(
            user_query="What is Salah?",
            max_sources=20,
            thread_id="test_high_sources",
        )
        
        assert "response" in result
    
    def test_high_score_threshold(self, patch_llm, patch_retriever):
        """Test with high score threshold."""
        result = invoke_rag_workflow(
            user_query="What is fasting?",
            max_sources=10,
            score_threshold=0.9,
            thread_id="test_high_threshold",
        )
        
        assert "response" in result
        # With high threshold, might have fewer results


# ============================================================================
# E2E Integration Tests: With Real LLM
# ============================================================================

@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
class TestRealLLMEndToEnd:
    """End-to-end tests with real LLM (slower, more realistic)."""
    
    def test_simple_query_real_llm(self, real_llm):
        """Test simple query with real LLM."""
        with patch('backend.rag.rag_nodes.IslamicRetriever') as mock_retriever_class:
            mock_retriever = MockRetriever(documents=get_sample_documents())
            mock_retriever_class.return_value = mock_retriever
            
            result = invoke_rag_workflow(
                user_query="What is fasting in Islam?",
                max_sources=5,
                thread_id="test_real_simple",
            )
            
            assert "response" in result
            response = result["response"]
            
            print(f"\n{'='*60}")
            print(f"REAL LLM E2E TEST")
            print(f"{'='*60}")
            print(f"Query: What is fasting in Islam?")
            print(f"Response:\n{response}")
            print(f"Citations: {len(result.get('citations', []))}")
            print(f"{'='*60}\n")
            
            # Validate response quality
            quality = validate_response_quality(response)
            print(f"Quality checks: {quality}")
            
            assert quality["islamic_style"], "Response should use Islamic style"
    
    def test_greeting_real_llm(self, real_llm):
        """Test greeting with real LLM."""
        result = invoke_rag_workflow(
            user_query="Assalamu alaikum",
            max_sources=5,
            thread_id="test_real_greeting",
        )
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"GREETING WITH REAL LLM")
        print(f"{'='*60}")
        print(f"Query: Assalamu alaikum")
        print(f"Response: {response}")
        print(f"{'='*60}\n")
        
        # Should respond appropriately
        response_lower = response.lower()
        assert any(term in response_lower for term in ["alaikum", "wa alaikum", "assalam", "help", "assist"])
    
    def test_complex_query_real_llm(self, real_llm):
        """Test complex query with real LLM."""
        with patch('backend.rag.rag_nodes.IslamicRetriever') as mock_retriever_class:
            mock_retriever = MockRetriever(documents=get_sample_documents())
            mock_retriever_class.return_value = mock_retriever
            
            result = invoke_rag_workflow(
                user_query="What are the conditions for fasting to be valid?",
                max_sources=10,
                thread_id="test_real_complex",
            )
            
            assert "response" in result
            response = result["response"]
            
            print(f"\n{'='*60}")
            print(f"COMPLEX QUERY WITH REAL LLM")
            print(f"{'='*60}")
            print(f"Query: What are the conditions for fasting to be valid?")
            print(f"Response:\n{response}")
            print(f"Citations: {len(result.get('citations', []))}")
            print(f"{'='*60}\n")
            
            quality = validate_response_quality(response)
            print(f"Quality checks: {quality}")
            
            assert len(response) > 100, "Response should be substantive"


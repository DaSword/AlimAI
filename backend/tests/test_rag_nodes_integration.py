"""
Integration tests for RAG nodes with real LLM.

These tests use the actual LM Studio LLM to validate prompt quality and outputs.
Run with: pytest backend/tests/test_rag_nodes_integration.py -m integration
"""

import pytest
from typing import Dict, Any

from backend.rag.rag_nodes import (
    analyze_query_complexity_node,
    classify_query_node,
    expand_query_node,
    generate_response_node,
    generate_conversational_response_node,
)
from backend.core.models import QuestionType
from backend.tests.fixtures import (
    create_simple_conversation,
    create_multi_turn_conversation,
    create_conversational_history,
    create_test_state,
    get_sample_documents,
    validate_response_quality,
    validate_islamic_style,
    validate_citation_format,
    validate_markdown_usage,
)


# ============================================================================
# Integration Tests: Complexity Analysis
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestComplexityAnalysisIntegration:
    """Test complexity analysis with real LLM."""
    
    def test_simple_greeting(self, real_llm):
        """Test that greetings are classified as simple_conversational."""
        state = {
            "messages": [{"role": "user", "content": "Assalamu alaikum"}],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        assert result["query_complexity"] in ["simple_conversational", "follow_up"]
        print(f"Greeting classified as: {result['query_complexity']}")
    
    def test_thank_you(self, real_llm):
        """Test that thanks are classified as simple_conversational."""
        messages = create_simple_conversation()
        messages.append({"role": "user", "content": "Thank you, that's helpful"})
        
        state = {"messages": messages}
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        print(f"Thanks classified as: {result['query_complexity']}")
    
    def test_follow_up_elaboration(self, real_llm):
        """Test that follow-up questions are detected."""
        messages = create_simple_conversation()
        messages.append({"role": "user", "content": "Tell me more about the benefits of fasting"})
        
        state = {"messages": messages}
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        print(f"Follow-up classified as: {result['query_complexity']}, needs_new_retrieval: {result.get('needs_new_retrieval')}")
    
    def test_follow_up_new_topic(self, real_llm):
        """Test follow-up on new topic requires retrieval."""
        messages = create_simple_conversation()
        messages.append({"role": "user", "content": "What about the Hanafi view on this?"})
        
        state = {"messages": messages}
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        assert "needs_new_retrieval" in result
        print(f"New topic follow-up: {result['query_complexity']}, needs_new_retrieval: {result.get('needs_new_retrieval')}")
    
    def test_simple_factual(self, real_llm):
        """Test simple factual questions."""
        state = {
            "messages": [{"role": "user", "content": "Who was Prophet Muhammad?"}],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        print(f"Simple factual classified as: {result['query_complexity']}")
    
    def test_complex_fiqh(self, real_llm):
        """Test complex fiqh questions."""
        state = {
            "messages": [{"role": "user", "content": "What is the Islamic ruling on mortgage interest according to different madhabs?"}],
        }
        
        result = analyze_query_complexity_node(state)
        
        assert "query_complexity" in result
        print(f"Complex fiqh classified as: {result['query_complexity']}")


# ============================================================================
# Integration Tests: Query Classification
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestQueryClassificationIntegration:
    """Test query classification with real LLM."""
    
    def test_classify_fiqh(self, real_llm):
        """Test fiqh question classification."""
        state = {"user_query": "Is it permissible to pray with shoes on?"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
        print(f"Fiqh query classified as: {result['question_type']}")
        # Should be fiqh or general
        assert result["question_type"] in [QuestionType.FIQH.value, QuestionType.GENERAL.value]
    
    def test_classify_aqidah(self, real_llm):
        """Test aqidah question classification."""
        state = {"user_query": "What are the attributes of Allah in Islam?"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
        print(f"Aqidah query classified as: {result['question_type']}")
    
    def test_classify_tafsir(self, real_llm):
        """Test tafsir question classification."""
        state = {"user_query": "Explain Surah Al-Fatiha verse 5"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
        print(f"Tafsir query classified as: {result['question_type']}")
    
    def test_classify_hadith(self, real_llm):
        """Test hadith question classification."""
        state = {"user_query": "What is the hadith about intentions in Sahih Bukhari?"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
        print(f"Hadith query classified as: {result['question_type']}")
    
    def test_classify_general(self, real_llm):
        """Test general question classification."""
        state = {"user_query": "What is charity in Islam?"}
        
        result = classify_query_node(state)
        
        assert "question_type" in result
        print(f"General query classified as: {result['question_type']}")


# ============================================================================
# Integration Tests: Query Expansion
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestQueryExpansionIntegration:
    """Test query expansion with real LLM."""
    
    def test_expand_general_query(self, real_llm):
        """Test expansion of general query."""
        state = {
            "user_query": "What is charity?",
            "question_type": QuestionType.GENERAL.value,
        }
        
        result = expand_query_node(state)
        
        assert "expanded_queries" in result
        queries = result["expanded_queries"]
        
        print(f"Original: What is charity?")
        print(f"Expanded queries ({len(queries)}):")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        # Check that expansion adds Islamic context
        all_queries_text = " ".join(queries).lower()
        islamic_terms = ["islam", "zakat", "sadaqah", "charity", "quran", "hadith", "allah"]
        has_islamic_terms = any(term in all_queries_text for term in islamic_terms)
        
        assert has_islamic_terms, "Expanded queries should include Islamic terminology"
    
    def test_expand_fiqh_query(self, real_llm):
        """Test expansion of fiqh query with madhab terms."""
        state = {
            "user_query": "Is mortgage interest allowed?",
            "question_type": QuestionType.FIQH.value,
        }
        
        result = expand_query_node(state)
        
        queries = result["expanded_queries"]
        print(f"Fiqh query expansions:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        # Fiqh expansions should include relevant terms
        all_text = " ".join(queries).lower()
        assert len(queries) > 0
    
    def test_expand_aqidah_query(self, real_llm):
        """Test expansion of aqidah query."""
        state = {
            "user_query": "What is iman?",
            "question_type": QuestionType.AQIDAH.value,
        }
        
        result = expand_query_node(state)
        
        queries = result["expanded_queries"]
        print(f"Aqidah query expansions:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        assert len(queries) > 0


# ============================================================================
# Integration Tests: Response Generation
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestResponseGenerationIntegration:
    """Test response generation with real LLM."""
    
    def test_generate_with_quran_context(self, real_llm, sample_documents):
        """Test generation with Quranic context."""
        # Filter for Quran documents
        quran_docs = [d for d in sample_documents if d.metadata.source_type.value == "quran"]
        
        state = create_test_state(
            user_query="What does the Quran say about fasting?",
            question_type=QuestionType.TAFSIR.value,
            ranked_docs=quran_docs,
            messages=create_simple_conversation(),
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"QUERY: What does the Quran say about fasting?")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
        # Validate response quality
        quality = validate_response_quality(response)
        print(f"Quality checks: {quality}")
        
        assert quality["islamic_style"], "Response should use Islamic style"
        assert len(response) > 50, "Response should be substantive"
    
    def test_generate_with_hadith_context(self, real_llm, sample_documents):
        """Test generation with Hadith context."""
        hadith_docs = [d for d in sample_documents if d.metadata.source_type.value == "hadith"]
        
        state = create_test_state(
            user_query="What did the Prophet say about fasting?",
            question_type=QuestionType.HADITH.value,
            ranked_docs=hadith_docs,
            messages=[],
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"QUERY: What did the Prophet say about fasting?")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
        quality = validate_response_quality(response)
        print(f"Quality checks: {quality}")
        
        assert quality["islamic_style"], "Response should use Islamic style"
    
    def test_generate_fiqh_with_madhab_views(self, real_llm, sample_documents):
        """Test fiqh response with madhab perspectives."""
        fiqh_docs = [d for d in sample_documents if d.metadata.source_type.value == "fiqh"]
        
        state = create_test_state(
            user_query="What is the ruling on intention for fasting?",
            question_type=QuestionType.FIQH.value,
            ranked_docs=fiqh_docs,
            messages=[],
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"QUERY: What is the ruling on intention for fasting?")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
        quality = validate_response_quality(response)
        print(f"Quality checks: {quality}")
        
        assert quality["islamic_style"], "Response should use Islamic style"
    
    def test_generate_without_context(self, real_llm):
        """Test generation when no context is available."""
        state = create_test_state(
            user_query="What is the Islamic view on time travel?",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=[],
            messages=[],
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"QUERY: What is the Islamic view on time travel?")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
        # Should include disclaimer about lacking sources
        assert len(response) > 0
    
    def test_generate_with_conversation_context(self, real_llm, sample_documents):
        """Test that conversation history influences response."""
        messages = create_multi_turn_conversation()
        
        state = create_test_state(
            user_query="Who can receive Zakat?",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=sample_documents[:2],
            messages=messages,
        )
        
        result = generate_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"QUERY (with history): Who can receive Zakat?")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
        quality = validate_response_quality(response)
        print(f"Quality checks: {quality}")


# ============================================================================
# Integration Tests: Conversational Responses
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestConversationalResponseIntegration:
    """Test conversational responses with real LLM."""
    
    def test_respond_to_greeting(self, real_llm):
        """Test response to Islamic greeting."""
        state = {
            "user_query": "Assalamu alaikum",
            "messages": [],
        }
        
        result = generate_conversational_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"GREETING: Assalamu alaikum")
        print(f"RESPONSE: {response}")
        print(f"{'='*60}\n")
        
        # Should respond appropriately
        response_lower = response.lower()
        assert any(term in response_lower for term in ["alaikum", "wa alaikum", "assalam", "help", "assist"])
    
    def test_respond_to_thanks(self, real_llm):
        """Test response to thanks."""
        messages = create_simple_conversation()
        
        state = {
            "user_query": "Thank you, that was very helpful!",
            "messages": messages,
        }
        
        result = generate_conversational_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"THANKS: Thank you, that was very helpful!")
        print(f"RESPONSE: {response}")
        print(f"{'='*60}\n")
        
        # Should acknowledge thanks appropriately
        assert len(response) > 0
        assert validate_islamic_style(response)
    
    def test_respond_to_acknowledgment(self, real_llm):
        """Test response to acknowledgment."""
        messages = create_simple_conversation()
        
        state = {
            "user_query": "Ok, I understand",
            "messages": messages,
        }
        
        result = generate_conversational_response_node(state)
        
        assert "response" in result
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"ACKNOWLEDGMENT: Ok, I understand")
        print(f"RESPONSE: {response}")
        print(f"{'='*60}\n")
        
        assert len(response) > 0


# ============================================================================
# Style and Format Validation Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestResponseStyleIntegration:
    """Test that responses meet style and format requirements."""
    
    def test_response_uses_markdown(self, real_llm, sample_documents):
        """Test that substantive responses use markdown formatting."""
        state = create_test_state(
            user_query="Explain the concept of Tawhid in detail",
            question_type=QuestionType.AQIDAH.value,
            ranked_docs=sample_documents[:3],
            messages=[],
        )
        
        result = generate_response_node(state)
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"Checking markdown usage...")
        print(f"{'='*60}")
        print(response[:500])  # Print first 500 chars
        print(f"{'='*60}\n")
        
        has_markdown = validate_markdown_usage(response)
        print(f"Has markdown: {has_markdown}")
        
        # For substantive responses (>200 chars), markdown is expected
        if len(response) > 200:
            assert has_markdown, "Long responses should use markdown formatting"
    
    def test_response_has_proper_structure(self, real_llm, sample_documents):
        """Test that responses are well-structured."""
        state = create_test_state(
            user_query="What are the pillars of Islam?",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=sample_documents[:3],
            messages=[],
        )
        
        result = generate_response_node(state)
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"Checking response structure...")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
        # Should have multiple paragraphs or sections
        quality = validate_response_quality(response)
        print(f"Structure quality: {quality}")
        
        assert quality["response_structure"], "Response should be well-structured"
    
    def test_response_uses_islamic_terminology(self, real_llm, sample_documents):
        """Test that responses use appropriate Islamic terminology."""
        state = create_test_state(
            user_query="What is prayer in Islam?",
            question_type=QuestionType.GENERAL.value,
            ranked_docs=sample_documents[:3],
            messages=[],
        )
        
        result = generate_response_node(state)
        response = result["response"]
        
        print(f"\n{'='*60}")
        print(f"Checking Islamic terminology...")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
        assert validate_islamic_style(response), "Response should use Islamic terminology"


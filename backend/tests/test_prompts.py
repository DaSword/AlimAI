"""
Tests for prompt templates and formatting.

These tests validate that prompts are correctly formatted and produce consistent outputs.
Run with: pytest backend/tests/test_prompts.py
"""

import pytest
from typing import Dict, Any

from backend.rag.prompts import (
    SYSTEM_IDENTITY,
    QUERY_CLASSIFIER_PROMPT,
    QUERY_COMPLEXITY_PROMPT,
    QUERY_EXPANSION_PROMPT,
    FIQH_QUERY_EXPANSION,
    AQIDAH_QUERY_EXPANSION,
    TAFSIR_QUERY_EXPANSION,
    FOLLOW_UP_EXPANSION_PROMPT,
    CONVERSATIONAL_RESPONSE_PROMPT,
    RESPONSE_GENERATION_PROMPT,
    FIQH_GENERATION_PROMPT,
    AQIDAH_GENERATION_PROMPT,
    format_prompt,
    create_system_message,
    get_context_template,
    get_generation_prompt,
    get_expansion_prompt,
)
from backend.core.models import QuestionType


# ============================================================================
# Prompt Formatting Tests
# ============================================================================

@pytest.mark.unit
class TestPromptFormatting:
    """Test prompt template formatting."""
    
    def test_format_prompt_simple(self):
        """Test formatting a simple prompt."""
        template = "User asked: {query}"
        result = format_prompt(template, query="What is Islam?")
        
        assert result == "User asked: What is Islam?"
    
    def test_format_prompt_multiple_placeholders(self):
        """Test formatting with multiple placeholders."""
        template = "Query: {query}\nType: {type}"
        result = format_prompt(template, query="Test", type="fiqh")
        
        assert "Query: Test" in result
        assert "Type: fiqh" in result
    
    def test_format_complexity_prompt(self):
        """Test formatting the complexity analysis prompt."""
        result = format_prompt(
            QUERY_COMPLEXITY_PROMPT,
            user_query="What is fasting?",
            conversation_history="User: Hello\nAssistant: Hi there!",
        )
        
        assert "What is fasting?" in result
        assert "User: Hello" in result
        assert "complexity:" in result.lower()
    
    def test_format_classification_prompt(self):
        """Test formatting the classification prompt."""
        result = format_prompt(
            QUERY_CLASSIFIER_PROMPT,
            user_query="Is prayer obligatory?",
        )
        
        assert "Is prayer obligatory?" in result
        assert "fiqh" in result.lower()
        assert "aqidah" in result.lower()
    
    def test_format_expansion_prompt(self):
        """Test formatting the expansion prompt."""
        result = format_prompt(
            QUERY_EXPANSION_PROMPT,
            user_query="What is charity?",
            question_type="general",
        )
        
        assert "What is charity?" in result
        assert "reformulate" in result.lower()
    
    def test_format_generation_prompt(self):
        """Test formatting the generation prompt."""
        result = format_prompt(
            RESPONSE_GENERATION_PROMPT,
            user_query="What is fasting?",
            context="[Quran 2:183] O you who believe...",
            conversation_history="No previous conversation.",
        )
        
        assert "What is fasting?" in result
        assert "[Quran 2:183]" in result
        assert "conversation history" in result.lower()


# ============================================================================
# System Message Tests
# ============================================================================

@pytest.mark.unit
class TestSystemMessage:
    """Test system message creation."""
    
    def test_create_system_message_with_identity(self):
        """Test creating system message with full identity."""
        result = create_system_message(include_identity=True)
        
        assert result["role"] == "system"
        assert "Islamic knowledge assistant" in result["content"]
        assert "Quran" in result["content"]
        assert "Hadith" in result["content"]
    
    def test_create_system_message_without_identity(self):
        """Test creating minimal system message."""
        result = create_system_message(include_identity=False)
        
        assert result["role"] == "system"
        assert "helpful" in result["content"].lower()


# ============================================================================
# Template Selection Tests
# ============================================================================

@pytest.mark.unit
class TestTemplateSelection:
    """Test that correct templates are selected for question types."""
    
    def test_get_expansion_prompt_fiqh(self):
        """Test getting fiqh expansion prompt."""
        result = get_expansion_prompt(QuestionType.FIQH)
        
        assert result == FIQH_QUERY_EXPANSION
        assert "madhab" in result.lower()
    
    def test_get_expansion_prompt_aqidah(self):
        """Test getting aqidah expansion prompt."""
        result = get_expansion_prompt(QuestionType.AQIDAH)
        
        assert result == AQIDAH_QUERY_EXPANSION
        assert "belief" in result.lower()
    
    def test_get_expansion_prompt_tafsir(self):
        """Test getting tafsir expansion prompt."""
        result = get_expansion_prompt(QuestionType.TAFSIR)
        
        assert result == TAFSIR_QUERY_EXPANSION
        assert "tafsir" in result.lower()
    
    def test_get_expansion_prompt_general(self):
        """Test getting general expansion prompt."""
        result = get_expansion_prompt(QuestionType.GENERAL)
        
        assert result == QUERY_EXPANSION_PROMPT
    
    def test_get_generation_prompt_fiqh(self):
        """Test getting fiqh generation prompt."""
        result = get_generation_prompt(QuestionType.FIQH)
        
        assert result == FIQH_GENERATION_PROMPT
        assert "madhab" in result.lower()
    
    def test_get_generation_prompt_aqidah(self):
        """Test getting aqidah generation prompt."""
        result = get_generation_prompt(QuestionType.AQIDAH)
        
        assert result == AQIDAH_GENERATION_PROMPT
        assert "belief" in result.lower()
    
    def test_get_generation_prompt_general(self):
        """Test getting general generation prompt."""
        result = get_generation_prompt(QuestionType.GENERAL)
        
        assert result == RESPONSE_GENERATION_PROMPT
    
    def test_get_context_template_fiqh(self):
        """Test getting fiqh context template."""
        result = get_context_template(QuestionType.FIQH)
        
        assert "{quran_verses}" in result
        assert "{fiqh_rulings}" in result
    
    def test_get_context_template_aqidah(self):
        """Test getting aqidah context template."""
        result = get_context_template(QuestionType.AQIDAH)
        
        assert "{quran_verses}" in result
        assert "{aqidah_sources}" in result


# ============================================================================
# Prompt Content Tests
# ============================================================================

@pytest.mark.unit
class TestPromptContent:
    """Test prompt content and structure."""
    
    def test_system_identity_has_core_principles(self):
        """Test that system identity includes core principles."""
        assert "Quran" in SYSTEM_IDENTITY
        assert "Hadith" in SYSTEM_IDENTITY
        assert "authentic" in SYSTEM_IDENTITY.lower()
        assert "cite" in SYSTEM_IDENTITY.lower() or "source" in SYSTEM_IDENTITY.lower()
    
    def test_complexity_prompt_has_categories(self):
        """Test that complexity prompt defines all categories."""
        assert "simple_conversational" in QUERY_COMPLEXITY_PROMPT
        assert "follow_up" in QUERY_COMPLEXITY_PROMPT
        assert "simple_factual" in QUERY_COMPLEXITY_PROMPT
        assert "complex" in QUERY_COMPLEXITY_PROMPT
    
    def test_classifier_prompt_has_all_types(self):
        """Test that classifier prompt includes all question types."""
        assert "fiqh" in QUERY_CLASSIFIER_PROMPT.lower()
        assert "aqidah" in QUERY_CLASSIFIER_PROMPT.lower()
        assert "tafsir" in QUERY_CLASSIFIER_PROMPT.lower()
        assert "hadith" in QUERY_CLASSIFIER_PROMPT.lower()
        assert "general" in QUERY_CLASSIFIER_PROMPT.lower()
    
    def test_generation_prompts_request_citations(self):
        """Test that generation prompts request citations."""
        prompts = [
            RESPONSE_GENERATION_PROMPT,
            FIQH_GENERATION_PROMPT,
            AQIDAH_GENERATION_PROMPT,
        ]
        
        for prompt in prompts:
            assert "cite" in prompt.lower() or "source" in prompt.lower() or "citation" in prompt.lower()
    
    def test_generation_prompts_request_structure(self):
        """Test that generation prompts request structured output."""
        prompts = [
            RESPONSE_GENERATION_PROMPT,
            FIQH_GENERATION_PROMPT,
            AQIDAH_GENERATION_PROMPT,
        ]
        
        for prompt in prompts:
            # Should mention markdown or structure
            has_structure = (
                "markdown" in prompt.lower() or
                "heading" in prompt.lower() or
                "format" in prompt.lower() or
                "clear" in prompt.lower()
            )
            assert has_structure, f"Prompt should request structured output"
    
    def test_fiqh_prompt_mentions_madhabs(self):
        """Test that fiqh prompts mention madhabs."""
        assert "madhab" in FIQH_GENERATION_PROMPT.lower()
        assert "hanafi" in FIQH_GENERATION_PROMPT.lower()
    
    def test_conversational_prompt_is_friendly(self):
        """Test that conversational prompt is friendly."""
        assert "help" in CONVERSATIONAL_RESPONSE_PROMPT.lower() or "assist" in CONVERSATIONAL_RESPONSE_PROMPT.lower()


# ============================================================================
# Prompt Consistency Tests
# ============================================================================

@pytest.mark.unit
class TestPromptConsistency:
    """Test consistency across prompts."""
    
    def test_all_expansion_prompts_request_reformulation(self):
        """Test that all expansion prompts request reformulation."""
        expansion_prompts = [
            QUERY_EXPANSION_PROMPT,
            FIQH_QUERY_EXPANSION,
            AQIDAH_QUERY_EXPANSION,
            TAFSIR_QUERY_EXPANSION,
        ]
        
        for prompt in expansion_prompts:
            assert "reformulate" in prompt.lower() or "rephrase" in prompt.lower()
    
    def test_all_generation_prompts_include_conversation_history(self):
        """Test that generation prompts include conversation history."""
        generation_prompts = [
            RESPONSE_GENERATION_PROMPT,
            FIQH_GENERATION_PROMPT,
            AQIDAH_GENERATION_PROMPT,
        ]
        
        for prompt in generation_prompts:
            assert "{conversation_history}" in prompt
    
    def test_all_generation_prompts_include_context(self):
        """Test that generation prompts include context placeholder."""
        generation_prompts = [
            RESPONSE_GENERATION_PROMPT,
            FIQH_GENERATION_PROMPT,
            AQIDAH_GENERATION_PROMPT,
        ]
        
        for prompt in generation_prompts:
            assert "{context}" in prompt
    
    def test_all_prompts_include_user_query(self):
        """Test that all prompts that need it include user_query."""
        prompts_needing_query = [
            QUERY_CLASSIFIER_PROMPT,
            QUERY_COMPLEXITY_PROMPT,
            QUERY_EXPANSION_PROMPT,
            RESPONSE_GENERATION_PROMPT,
            CONVERSATIONAL_RESPONSE_PROMPT,
        ]
        
        for prompt in prompts_needing_query:
            assert "{user_query}" in prompt


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.unit
class TestPromptEdgeCases:
    """Test prompt handling of edge cases."""
    
    def test_format_with_empty_values(self):
        """Test formatting with empty values."""
        result = format_prompt(
            QUERY_COMPLEXITY_PROMPT,
            user_query="",
            conversation_history="",
        )
        
        # Should still be valid prompt
        assert "complexity:" in result.lower()
    
    def test_format_with_long_values(self):
        """Test formatting with very long values."""
        long_query = "x" * 1000
        long_history = "y" * 5000
        
        result = format_prompt(
            RESPONSE_GENERATION_PROMPT,
            user_query=long_query,
            context="Some context",
            conversation_history=long_history,
        )
        
        # Should handle long inputs
        assert long_query in result
        assert len(result) > 5000
    
    def test_format_with_special_characters(self):
        """Test formatting with special characters."""
        result = format_prompt(
            QUERY_CLASSIFIER_PROMPT,
            user_query="What is 'Aqidah'? And \"Tawhid\"?",
        )
        
        # Should preserve special characters
        assert "'Aqidah'" in result
        assert '"Tawhid"' in result
    
    def test_format_with_unicode(self):
        """Test formatting with Arabic text."""
        result = format_prompt(
            QUERY_EXPANSION_PROMPT,
            user_query="What is الصلاة (Salah)?",
            question_type="fiqh",
        )
        
        # Should preserve Arabic text
        assert "الصلاة" in result


# ============================================================================
# Integration: Prompt Quality with Real LLM
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPromptQualityWithLLM:
    """Test that prompts produce good outputs with real LLM."""
    
    def test_complexity_prompt_produces_valid_output(self, real_llm):
        """Test that complexity prompt produces valid classification."""
        from backend.rag.prompts import format_prompt
        
        test_cases = [
            ("Assalamu alaikum", "simple_conversational"),
            ("What is fasting?", "simple_factual"),
            ("What is the Islamic ruling on mortgage interest across madhabs?", "complex"),
        ]
        
        for query, expected_category in test_cases:
            prompt = format_prompt(
                QUERY_COMPLEXITY_PROMPT,
                user_query=query,
                conversation_history="No previous conversation.",
            )
            
            response = real_llm.complete(prompt)
            response_text = str(response).lower()
            
            print(f"\nQuery: {query}")
            print(f"Expected: {expected_category}")
            print(f"Response: {response_text[:200]}")
            
            # Should contain a valid complexity category
            valid_categories = ["simple_conversational", "follow_up", "simple_factual", "complex"]
            has_valid_category = any(cat in response_text for cat in valid_categories)
            
            assert has_valid_category, f"Response should contain a valid complexity category"
    
    def test_classifier_prompt_produces_valid_output(self, real_llm):
        """Test that classifier prompt produces valid question type."""
        from backend.rag.prompts import format_prompt
        
        test_cases = [
            "Is prayer obligatory?",
            "What are the attributes of Allah?",
            "Explain Surah Al-Fatiha",
        ]
        
        for query in test_cases:
            prompt = format_prompt(
                QUERY_CLASSIFIER_PROMPT,
                user_query=query,
            )
            
            response = real_llm.complete(prompt)
            response_text = str(response).lower()
            
            print(f"\nQuery: {query}")
            print(f"Classification: {response_text[:100]}")
            
            # Should contain a valid question type
            valid_types = ["fiqh", "aqidah", "tafsir", "hadith", "general"]
            has_valid_type = any(qt in response_text for qt in valid_types)
            
            assert has_valid_type, f"Response should contain a valid question type"
    
    def test_expansion_prompt_produces_multiple_queries(self, real_llm):
        """Test that expansion prompt produces multiple queries."""
        from backend.rag.prompts import format_prompt
        
        prompt = format_prompt(
            QUERY_EXPANSION_PROMPT,
            user_query="What is charity?",
            question_type="general",
        )
        
        response = real_llm.complete(prompt)
        response_text = str(response)
        
        print(f"\nExpansion response:\n{response_text}")
        
        # Should contain multiple lines or numbered queries
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        # Should have at least 2 expanded queries
        assert len(lines) >= 2, "Should produce at least 2 expanded queries"
    
    def test_generation_prompt_produces_structured_response(self, real_llm):
        """Test that generation prompt produces structured response."""
        from backend.rag.prompts import format_prompt
        
        context = """
**QURANIC FOUNDATION:**
Surah 2:183 - O you who have believed, decreed upon you is fasting...

**PROPHETIC GUIDANCE:**
Sahih Bukhari 1903 - Whoever observes fasts during Ramadan...
"""
        
        prompt = format_prompt(
            RESPONSE_GENERATION_PROMPT,
            user_query="What is fasting in Islam?",
            context=context,
            conversation_history="No previous conversation.",
        )
        
        from llama_index.core.llms import ChatMessage, MessageRole
        
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_IDENTITY),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        
        response = real_llm.chat(messages)
        response_text = str(response.message.content)
        
        print(f"\nGenerated response:\n{response_text}")
        
        # Should be substantive
        assert len(response_text) > 100, "Response should be substantive"
        
        # Should use Islamic terminology
        response_lower = response_text.lower()
        islamic_terms = ["islam", "fasting", "ramadan", "allah", "quran", "hadith"]
        has_islamic_terms = any(term in response_lower for term in islamic_terms)
        
        assert has_islamic_terms, "Response should use Islamic terminology"


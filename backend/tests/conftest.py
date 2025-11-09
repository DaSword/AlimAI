"""
Pytest configuration and shared fixtures for RAG node testing.

This module provides:
- Pytest configuration
- Shared fixtures for LLM, retriever, and state setup
- Test utilities
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from backend.tests.fixtures import (
    get_sample_documents,
    create_empty_conversation,
    create_simple_conversation,
    create_multi_turn_conversation,
    create_test_state,
    MockRetriever,
    MockLLMResponse,
    get_mock_complexity_response,
    get_mock_classification_response,
    get_mock_expansion_response,
    get_mock_generation_response,
    get_mock_conversational_response,
)
from backend.core.models import QuestionType
from backend.llama.llama_config import get_llm


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, mocked)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (slower, uses real LLM)"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test (slowest, full workflow)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return get_sample_documents()


@pytest.fixture
def empty_conversation():
    """Provide an empty conversation history."""
    return create_empty_conversation()


@pytest.fixture
def simple_conversation():
    """Provide a simple 2-turn conversation."""
    return create_simple_conversation()


@pytest.fixture
def multi_turn_conversation():
    """Provide a multi-turn conversation with follow-ups."""
    return create_multi_turn_conversation()


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_simple():
    """Provide a simple mock LLM that returns fixed responses."""
    mock = Mock()
    
    def complete_side_effect(prompt):
        """Return responses based on prompt content."""
        prompt_lower = str(prompt).lower()
        
        # Complexity analysis
        if "complexity" in prompt_lower:
            if "hello" in prompt_lower or "thank" in prompt_lower or "assalam" in prompt_lower:
                return MockLLMResponse(get_mock_complexity_response("simple_conversational"))
            elif "tell me more" in prompt_lower or "elaborate" in prompt_lower:
                return MockLLMResponse(get_mock_complexity_response("follow_up", needs_new_retrieval=False))
            elif "what about" in prompt_lower and "view" in prompt_lower:
                return MockLLMResponse(get_mock_complexity_response("follow_up", needs_new_retrieval=True))
            elif "when was" in prompt_lower or "who was" in prompt_lower or "what is" in prompt_lower:
                return MockLLMResponse(get_mock_complexity_response("simple_factual"))
            else:
                return MockLLMResponse(get_mock_complexity_response("complex"))
        
        # Query classification
        elif "classify" in prompt_lower:
            if "prayer" in prompt_lower or "halal" in prompt_lower or "permissible" in prompt_lower:
                return MockLLMResponse(get_mock_classification_response(QuestionType.FIQH))
            elif "belief" in prompt_lower or "attributes" in prompt_lower or "faith" in prompt_lower:
                return MockLLMResponse(get_mock_classification_response(QuestionType.AQIDAH))
            elif "explain" in prompt_lower and ("verse" in prompt_lower or "surah" in prompt_lower):
                return MockLLMResponse(get_mock_classification_response(QuestionType.TAFSIR))
            elif "hadith" in prompt_lower or "narration" in prompt_lower:
                return MockLLMResponse(get_mock_classification_response(QuestionType.HADITH))
            else:
                return MockLLMResponse(get_mock_classification_response(QuestionType.GENERAL))
        
        # Query expansion
        elif "reformulate" in prompt_lower:
            # Extract query from prompt
            query = "the query"
            return MockLLMResponse(get_mock_expansion_response(query))
        
        # Default response
        return MockLLMResponse("Mock LLM response")
    
    def chat_side_effect(messages):
        """Return chat responses based on messages."""
        last_message = messages[-1].content if messages else ""
        last_lower = last_message.lower()
        
        # Conversational responses
        if "hello" in last_lower or "assalam" in last_lower:
            return MockLLMResponse(get_mock_conversational_response(last_message))
        elif "thank" in last_lower:
            return MockLLMResponse(get_mock_conversational_response(last_message))
        
        # Generation responses
        elif "answer this question" in last_lower or "user question" in last_lower:
            has_context = "retrieved sources" in last_lower
            return MockLLMResponse(get_mock_generation_response(last_message, has_context))
        
        # Default
        return MockLLMResponse("This is a mock response from the assistant.")
    
    mock.complete.side_effect = complete_side_effect
    mock.chat.side_effect = chat_side_effect
    
    return mock


@pytest.fixture
def mock_llm_configurable():
    """Provide a configurable mock LLM for specific test scenarios."""
    mock = Mock()
    mock.responses = {}
    
    def set_response(prompt_keyword: str, response: str):
        """Set a specific response for a prompt containing a keyword."""
        mock.responses[prompt_keyword] = response
    
    def complete_side_effect(prompt):
        prompt_lower = str(prompt).lower()
        for keyword, response in mock.responses.items():
            if keyword.lower() in prompt_lower:
                return MockLLMResponse(response)
        return MockLLMResponse("Default mock response")
    
    def chat_side_effect(messages):
        if not messages:
            return MockLLMResponse("Default chat response")
        last_content = messages[-1].content.lower()
        for keyword, response in mock.responses.items():
            if keyword.lower() in last_content:
                return MockLLMResponse(response)
        return MockLLMResponse("Default chat response")
    
    mock.complete.side_effect = complete_side_effect
    mock.chat.side_effect = chat_side_effect
    mock.set_response = set_response
    
    return mock


# ============================================================================
# Mock Retriever Fixtures
# ============================================================================

@pytest.fixture
def mock_retriever(sample_documents):
    """Provide a mock retriever with sample documents."""
    return MockRetriever(documents=sample_documents)


@pytest.fixture
def empty_retriever():
    """Provide a mock retriever that returns no documents."""
    return MockRetriever(documents=[])


# ============================================================================
# Real LLM Fixtures (for integration tests)
# ============================================================================

@pytest.fixture
def real_llm():
    """
    Provide a real LLM instance for integration testing.
    
    Uses the configured LLM backend (lmstudio by default).
    Mark tests using this fixture as @pytest.mark.integration
    """
    try:
        from backend.core.config import config
        llm = get_llm(backend=config.LLM_BACKEND)
        return llm
    except Exception as e:
        pytest.skip(f"Real LLM not available: {e}")


# ============================================================================
# State Fixtures
# ============================================================================

@pytest.fixture
def basic_state():
    """Provide a basic test state."""
    return create_test_state(
        user_query="What is fasting in Islam?",
        messages=create_simple_conversation(),
    )


@pytest.fixture
def state_with_documents(sample_documents):
    """Provide a test state with retrieved documents."""
    return create_test_state(
        user_query="What is fasting in Islam?",
        messages=create_simple_conversation(),
        question_type=QuestionType.GENERAL.value,
        retrieved_docs=sample_documents,
    )


@pytest.fixture
def state_with_ranked_docs(sample_documents):
    """Provide a test state with ranked documents."""
    return create_test_state(
        user_query="What is fasting in Islam?",
        messages=create_simple_conversation(),
        question_type=QuestionType.GENERAL.value,
        retrieved_docs=sample_documents,
        ranked_docs=sample_documents[:3],
    )


@pytest.fixture
def state_with_response(sample_documents):
    """Provide a test state with a generated response."""
    response = """# Fasting in Islam

Fasting (Sawm) is one of the Five Pillars of Islam and involves abstaining from food, drink, and intimate relations from dawn until sunset.

**Quranic Evidence:**
As stated in Surah 2:183, "O you who have believed, decreed upon you is fasting..."

**Prophetic Guidance:**
The Prophet (ﷺ) said in Sahih Bukhari 1903: "Whoever observes fasts during Ramadan out of sincere faith..."
"""
    return create_test_state(
        user_query="What is fasting in Islam?",
        messages=create_simple_conversation(),
        question_type=QuestionType.GENERAL.value,
        retrieved_docs=sample_documents,
        ranked_docs=sample_documents[:3],
        response=response,
    )


# ============================================================================
# Patching Fixtures
# ============================================================================

@pytest.fixture
def patch_llm(mock_llm_simple):
    """Patch the LLM retrieval in nodes."""
    with patch('backend.rag.rag_nodes.get_llm', return_value=mock_llm_simple):
        yield mock_llm_simple


@pytest.fixture
def patch_retriever(mock_retriever):
    """Patch the retriever in nodes."""
    with patch('backend.rag.rag_nodes.IslamicRetriever', return_value=mock_retriever):
        yield mock_retriever


@pytest.fixture
def patch_expand_query(mock_llm_simple):
    """Patch query expansion function."""
    def mock_expand(query, question_type, llm_backend, num_expansions=2):
        return [
            f"Expanded: {query} from Islamic perspective",
            f"What do Islamic sources say about {query}",
        ][:num_expansions]
    
    with patch('backend.rag.rag_nodes.expand_query_with_llm', side_effect=mock_expand):
        yield


# ============================================================================
# Helper Functions for Tests
# ============================================================================

@pytest.fixture
def assert_state_has_keys():
    """Provide a helper to assert state has expected keys."""
    def _assert(state: Dict[str, Any], expected_keys: List[str]):
        for key in expected_keys:
            assert key in state, f"Expected key '{key}' not found in state"
    return _assert


@pytest.fixture
def assert_valid_complexity():
    """Provide a helper to validate complexity values."""
    def _assert(complexity: str):
        valid = ["simple_conversational", "follow_up", "simple_factual", "complex"]
        assert complexity in valid, f"Invalid complexity: {complexity}"
    return _assert


@pytest.fixture
def assert_valid_question_type():
    """Provide a helper to validate question type values."""
    def _assert(question_type: str):
        valid = [qt.value for qt in QuestionType]
        assert question_type in valid, f"Invalid question type: {question_type}"
    return _assert


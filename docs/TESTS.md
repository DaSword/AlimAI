# RAG Node Testing Suite

Comprehensive test suite for the Islamic Chatbot RAG pipeline with unit tests, integration tests, and end-to-end tests.

## Test Files Overview

### Core Test Files

1. **`fixtures.py`** - Test fixtures and sample data
   - Sample document chunks (Quran, Hadith, Tafsir, Fiqh)
   - Sample conversation histories
   - Mock LLM responses
   - Response validators
   - Helper functions

2. **`conftest.py`** - Pytest configuration
   - Shared fixtures for LLM, retriever, and state
   - Pytest markers configuration
   - Test utilities

### Unit Tests (Fast, Mocked)

3. **`test_rag_nodes_unit.py`** - Unit tests for all nodes
   - Tests helper functions
   - Tests all 8 node types
   - Tests all routing functions
   - Uses mocked LLM and retriever
   - **Run time: ~5 seconds**

### Integration Tests (Real LLM)

4. **`test_rag_nodes_integration.py`** - Integration tests with real LLM
   - Tests complexity analysis with real LLM
   - Tests query classification accuracy
   - Tests query expansion quality
   - Tests response generation style and citations
   - Tests conversational responses
   - Validates Islamic terminology and formatting
   - **Run time: ~30-60 seconds**

### Prompt Quality Tests

5. **`test_prompts.py`** - Prompt template validation
   - Tests prompt formatting
   - Tests template selection logic
   - Tests prompt content and structure
   - Tests prompt consistency
   - Tests edge cases (long queries, special chars)
   - Integration tests for prompt quality with real LLM
   - **Run time: ~20 seconds**

### End-to-End Tests

6. **`test_rag_graph_e2e.py`** - Complete workflow tests
   - Tests all routing paths (conversational, factual, complex, follow-up)
   - Tests multi-turn conversations
   - Tests response quality end-to-end
   - Tests different settings (max_sources, thresholds)
   - Tests with real LLM
   - **Run time: ~30-60 seconds**

### Error Handling Tests

7. **`test_error_handling.py`** - Error handling and edge cases
   - Tests LLM errors
   - Tests retrieval errors
   - Tests missing/invalid data
   - Tests malformed data
   - Tests timeout scenarios
   - Tests edge cases (long queries, special chars)
   - Tests recovery mechanisms
   - **Run time: ~10 seconds**

## Running Tests

### Quick Start

```bash
# Activate virtual environment
source alimenv/bin/activate

# Run all unit tests (fast)
pytest backend/tests -m unit -v

# Run all integration tests (slower, requires LM Studio)
pytest backend/tests -m integration -v

# Run all end-to-end tests
pytest backend/tests -m e2e -v

# Run all tests
pytest backend/tests -v
```

### Run Specific Test Files

```bash
# Unit tests only
pytest backend/tests/test_rag_nodes_unit.py -v

# Integration tests only
pytest backend/tests/test_rag_nodes_integration.py -v

# Prompt quality tests
pytest backend/tests/test_prompts.py -v

# E2E tests
pytest backend/tests/test_rag_graph_e2e.py -v

# Error handling tests
pytest backend/tests/test_error_handling.py -v
```

### Run Specific Test Classes

```bash
# Test only helper functions
pytest backend/tests/test_rag_nodes_unit.py::TestFormatConversationHistory -v

# Test only complexity analysis
pytest backend/tests/test_rag_nodes_unit.py::TestAnalyzeQueryComplexityNode -v

# Test response generation with real LLM
pytest backend/tests/test_rag_nodes_integration.py::TestResponseGenerationIntegration -v
```

### Run by Marker

```bash
# Run only fast unit tests
pytest backend/tests -m "unit and not slow" -v

# Run only slow tests
pytest backend/tests -m slow -v

# Skip integration tests
pytest backend/tests -m "not integration" -v
```

## Test Markers

The following pytest markers are available:

- `@pytest.mark.unit` - Fast unit tests with mocked dependencies
- `@pytest.mark.integration` - Integration tests with real LLM
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.slow` - Tests that take longer to run

## Prerequisites

### For Unit Tests
- No external dependencies required
- All services are mocked
- Fast execution

### For Integration/E2E Tests
- **LM Studio** must be running at `http://localhost:1234`
- Model must be loaded in LM Studio
- Set environment variables in `.env`:
  ```bash
  LLM_BACKEND=lmstudio
  LMSTUDIO_URL=http://localhost:1234/v1
  LMSTUDIO_CHAT_MODEL=your-model-name
  ```

## Test Coverage

The test suite provides comprehensive coverage of:

### Nodes Tested
- ✅ `analyze_query_complexity_node` - 15+ test cases
- ✅ `classify_query_node` - 10+ test cases
- ✅ `expand_query_node` - 8+ test cases
- ✅ `retrieve_node` - 10+ test cases
- ✅ `light_retrieve_node` - 5+ test cases
- ✅ `rank_context_node` - 8+ test cases
- ✅ `generate_response_node` - 12+ test cases
- ✅ `generate_conversational_response_node` - 8+ test cases
- ✅ `format_citations_node` - 5+ test cases
- ✅ `update_messages_node` - 4+ test cases

### Routing Functions Tested
- ✅ `route_by_complexity` - All 4 paths
- ✅ `route_by_question_type`
- ✅ `should_retrieve`
- ✅ `has_documents`

### Workflow Paths Tested
- ✅ Simple conversational (greetings, thanks)
- ✅ Simple factual (basic questions)
- ✅ Complex queries (full pipeline)
- ✅ Follow-up queries (with/without retrieval)
- ✅ Multi-turn conversations

### Error Scenarios Tested
- ✅ LLM failures
- ✅ Retrieval failures
- ✅ Missing data
- ✅ Invalid data
- ✅ Timeouts
- ✅ Edge cases

## Viewing Test Results

### Verbose Output
```bash
pytest backend/tests -v
```

### With Coverage
```bash
pip install pytest-cov
pytest backend/tests --cov=backend/rag --cov-report=html
open htmlcov/index.html
```

### With Output
```bash
# Show print statements
pytest backend/tests -v -s

# Show only failed tests output
pytest backend/tests -v --tb=short
```

## Continuous Integration

### Fast CI (Every Commit)
```bash
# Run only unit tests (< 5 seconds)
pytest backend/tests -m unit
```

### Full CI (Before Deployment)
```bash
# Run all tests including integration
pytest backend/tests
```

## Writing New Tests

### Adding a New Unit Test

```python
@pytest.mark.unit
class TestNewFeature:
    """Test new feature."""
    
    def test_feature_basic(self, patch_llm):
        """Test basic functionality."""
        # Arrange
        state = {"user_query": "Test"}
        
        # Act
        result = new_feature_node(state)
        
        # Assert
        assert "expected_field" in result
```

### Adding an Integration Test

```python
@pytest.mark.integration
@pytest.mark.slow
class TestNewFeatureIntegration:
    """Test new feature with real LLM."""
    
    def test_feature_with_real_llm(self, real_llm):
        """Test with real LLM."""
        # Test implementation
        result = new_feature_node(state)
        
        # Validate quality
        quality = validate_response_quality(result["response"])
        assert quality["islamic_style"]
```

## Troubleshooting

### Integration Tests Fail
- Ensure LM Studio is running
- Check model is loaded
- Verify environment variables
- Check network connectivity

### Tests Are Slow
- Run only unit tests: `pytest -m unit`
- Skip slow tests: `pytest -m "not slow"`
- Run specific test file instead of all tests

### Import Errors
- Ensure virtual environment is activated
- Install test dependencies: `pip install pytest pytest-asyncio`

### Fixture Not Found
- Check `conftest.py` is in the tests directory
- Ensure fixture is properly decorated with `@pytest.fixture`

## Best Practices

1. **Run unit tests frequently** - They're fast and catch most issues
2. **Run integration tests before commits** - Validate prompt quality
3. **Run E2E tests before deployment** - Ensure complete workflow works
4. **Use markers** - Tag tests appropriately for filtering
5. **Keep tests independent** - Each test should be able to run alone
6. **Use descriptive names** - Test names should explain what they test
7. **Document complex tests** - Add comments for non-obvious test logic

## Test Statistics

- **Total Test Files**: 7
- **Total Test Classes**: 40+
- **Total Test Cases**: 150+
- **Unit Test Coverage**: All nodes and routing functions
- **Integration Test Coverage**: All critical prompts
- **E2E Test Coverage**: All workflow paths
- **Error Test Coverage**: Comprehensive error scenarios

## Next Steps

1. Run unit tests to ensure everything works
2. Start LM Studio and run integration tests
3. Review test output and quality checks
4. Add new tests as you modify prompts or nodes
5. Maintain test coverage as codebase evolves


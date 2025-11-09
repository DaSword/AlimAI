#!/bin/bash
# Convenient test runner script for RAG node tests

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}   RAG Node Testing Suite${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo -e "${YELLOW}Attempting to activate alimenv...${NC}"
    if [ -f "../../alimenv/bin/activate" ]; then
        source ../../alimenv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    else
        echo -e "${RED}✗ Could not find virtual environment${NC}"
        echo -e "${RED}  Please activate manually: source alimenv/bin/activate${NC}"
        exit 1
    fi
fi

# Function to run tests with specific marker
run_test_category() {
    local category=$1
    local marker=$2
    local description=$3
    
    echo -e "${BLUE}-------------------------------------------${NC}"
    echo -e "${BLUE}Running ${category}...${NC}"
    echo -e "${BLUE}${description}${NC}"
    echo -e "${BLUE}-------------------------------------------${NC}"
    
    if pytest backend/tests -m "${marker}" -v; then
        echo -e "${GREEN}✓ ${category} passed${NC}"
        return 0
    else
        echo -e "${RED}✗ ${category} failed${NC}"
        return 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    unit)
        echo -e "${BLUE}Running Unit Tests Only (Fast)${NC}"
        run_test_category "Unit Tests" "unit" "Tests with mocked dependencies"
        ;;
    
    integration)
        echo -e "${BLUE}Running Integration Tests (Requires LM Studio)${NC}"
        run_test_category "Integration Tests" "integration" "Tests with real LLM"
        ;;
    
    e2e)
        echo -e "${BLUE}Running End-to-End Tests${NC}"
        run_test_category "E2E Tests" "e2e" "Complete workflow tests"
        ;;
    
    fast)
        echo -e "${BLUE}Running Fast Tests Only${NC}"
        run_test_category "Fast Tests" "unit and not slow" "Quick unit tests"
        ;;
    
    slow)
        echo -e "${BLUE}Running Slow Tests${NC}"
        run_test_category "Slow Tests" "slow" "Long-running tests"
        ;;
    
    coverage)
        echo -e "${BLUE}Running Tests with Coverage${NC}"
        pytest backend/tests --cov=backend/rag --cov-report=term-missing --cov-report=html
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    file)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please specify a test file${NC}"
            echo -e "Usage: ./run_tests.sh file test_rag_nodes_unit.py"
            exit 1
        fi
        echo -e "${BLUE}Running Test File: $2${NC}"
        pytest "backend/tests/$2" -v
        ;;
    
    class)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Please specify file and class${NC}"
            echo -e "Usage: ./run_tests.sh class test_rag_nodes_unit.py TestFormatConversationHistory"
            exit 1
        fi
        echo -e "${BLUE}Running Test Class: $3 from $2${NC}"
        pytest "backend/tests/$2::$3" -v
        ;;
    
    all)
        echo -e "${BLUE}Running All Tests${NC}"
        echo ""
        
        # Run unit tests
        if run_test_category "Unit Tests" "unit" "Fast tests with mocks"; then
            UNIT_PASSED=1
        else
            UNIT_PASSED=0
        fi
        
        echo ""
        
        # Run integration tests
        if run_test_category "Integration Tests" "integration" "Tests with real LLM"; then
            INTEGRATION_PASSED=1
        else
            INTEGRATION_PASSED=0
        fi
        
        echo ""
        
        # Run E2E tests
        if run_test_category "E2E Tests" "e2e" "Complete workflows"; then
            E2E_PASSED=1
        else
            E2E_PASSED=0
        fi
        
        echo ""
        echo -e "${BLUE}===============================================${NC}"
        echo -e "${BLUE}Test Summary${NC}"
        echo -e "${BLUE}===============================================${NC}"
        
        if [ $UNIT_PASSED -eq 1 ]; then
            echo -e "${GREEN}✓ Unit Tests: PASSED${NC}"
        else
            echo -e "${RED}✗ Unit Tests: FAILED${NC}"
        fi
        
        if [ $INTEGRATION_PASSED -eq 1 ]; then
            echo -e "${GREEN}✓ Integration Tests: PASSED${NC}"
        else
            echo -e "${RED}✗ Integration Tests: FAILED${NC}"
        fi
        
        if [ $E2E_PASSED -eq 1 ]; then
            echo -e "${GREEN}✓ E2E Tests: PASSED${NC}"
        else
            echo -e "${RED}✗ E2E Tests: FAILED${NC}"
        fi
        
        if [ $UNIT_PASSED -eq 1 ] && [ $INTEGRATION_PASSED -eq 1 ] && [ $E2E_PASSED -eq 1 ]; then
            echo ""
            echo -e "${GREEN}🎉 All tests passed!${NC}"
            exit 0
        else
            echo ""
            echo -e "${RED}Some tests failed. Please review the output above.${NC}"
            exit 1
        fi
        ;;
    
    help|--help|-h)
        echo "Usage: ./run_tests.sh [category]"
        echo ""
        echo "Categories:"
        echo "  all          - Run all tests (default)"
        echo "  unit         - Run only unit tests (fast)"
        echo "  integration  - Run only integration tests (requires LM Studio)"
        echo "  e2e          - Run only end-to-end tests"
        echo "  fast         - Run only fast tests"
        echo "  slow         - Run only slow tests"
        echo "  coverage     - Run tests with coverage report"
        echo "  file <name>  - Run specific test file"
        echo "  class <file> <class> - Run specific test class"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh unit"
        echo "  ./run_tests.sh integration"
        echo "  ./run_tests.sh file test_rag_nodes_unit.py"
        echo "  ./run_tests.sh class test_prompts.py TestPromptFormatting"
        ;;
    
    *)
        echo -e "${RED}Unknown category: $1${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac


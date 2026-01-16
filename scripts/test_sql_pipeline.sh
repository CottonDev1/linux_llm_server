#!/bin/bash
#==============================================================================
# SQL Chat Pipeline Test Script
#==============================================================================
# This script tests the SQL text-to-SQL pipeline:
#   1. Send a natural language query
#   2. Receive generated SQL
#   3. Optionally execute the SQL
#   4. Report LLM metrics and timing
#
# Requirements:
#   - Node.js server running on localhost:3000
#   - Python service running on localhost:8001
#   - llama.cpp SQL model running on localhost:8080
#   - MongoDB accessible for rules lookup
#   - jq installed for JSON parsing
#
# Usage: ./test_sql_pipeline.sh [--execute]
#==============================================================================

set -e

# Configuration
NODE_SERVER="http://localhost:3000"
PYTHON_SERVER="http://localhost:8001"
LLM_SQL_SERVER="http://localhost:8080"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test queries
declare -a TEST_QUERIES=(
    "How many tickets were created today?"
    "Show me the top 10 customers by ticket count"
    "List all open tickets from last week"
    "What is the average resolution time for tickets?"
    "Count tickets by status"
)

# Timing
declare -A QUERY_TIMES
TOTAL_START=0

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

check_services() {
    log_step "CHECKING SERVICES"

    # Check LLM SQL server
    log_info "Checking LLM SQL server at ${LLM_SQL_SERVER}..."
    if curl -s "${LLM_SQL_SERVER}/health" > /dev/null 2>&1; then
        log_success "LLM SQL server is running"
    else
        log_error "LLM SQL server not responding"
        exit 1
    fi

    # Check Python service
    log_info "Checking Python service at ${PYTHON_SERVER}..."
    if curl -s "${PYTHON_SERVER}/health" > /dev/null 2>&1; then
        log_success "Python service is running"
    else
        log_error "Python service not responding"
        exit 1
    fi

    # Check Node.js server (optional)
    log_info "Checking Node.js server at ${NODE_SERVER}..."
    if curl -s "${NODE_SERVER}/health" > /dev/null 2>&1; then
        log_success "Node.js server is running"
    else
        log_info "Node.js server not running (will use Python service directly)"
    fi
}

test_sql_query() {
    local query="$1"
    local database="${2:-EWRCentral}"
    local execute="${3:-false}"

    log_info "Query: \"$query\""
    log_info "Database: $database"

    local start=$(date +%s%3N)

    # Create request
    local request=$(jq -n \
        --arg q "$query" \
        --arg db "$database" \
        '{
            question: $q,
            database: $db,
            execute: false,
            include_schema: true
        }')

    # Send to Python service SQL endpoint
    local response=$(curl -s -X POST "${PYTHON_SERVER}/sql/generate" \
        -H "Content-Type: application/json" \
        -d "$request" 2>&1)

    local end=$(date +%s%3N)
    local elapsed=$((end - start))

    if echo "$response" | jq -e '.sql' > /dev/null 2>&1; then
        local sql=$(echo "$response" | jq -r '.sql')
        local model=$(echo "$response" | jq -r '.model // "unknown"')
        local tokens=$(echo "$response" | jq -r '.tokens_used // "N/A"')
        local rule_match=$(echo "$response" | jq -r '.rule_matched // "none"')

        log_success "SQL Generated (${elapsed}ms)"
        echo -e "${YELLOW}Generated SQL:${NC}"
        echo "$sql" | head -20
        echo ""
        log_info "Model: $model"
        log_info "Tokens: $tokens"
        log_info "Rule Match: $rule_match"

        QUERY_TIMES["$query"]=$elapsed
        return 0
    else
        log_error "Failed to generate SQL"
        log_info "Response: $response"
        QUERY_TIMES["$query"]=-1
        return 1
    fi
}

test_llm_direct() {
    log_step "TESTING LLM DIRECT (localhost:8080)"

    local prompt="Generate a SQL query to count all records in a table called Users."

    log_info "Testing direct LLM completion..."
    local start=$(date +%s%3N)

    local response=$(curl -s -X POST "${LLM_SQL_SERVER}/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"n_predict\": 100, \"temperature\": 0.1}" 2>&1)

    local end=$(date +%s%3N)
    local elapsed=$((end - start))

    if echo "$response" | jq -e '.content' > /dev/null 2>&1; then
        local content=$(echo "$response" | jq -r '.content')
        local tokens=$(echo "$response" | jq -r '.tokens_predicted // "N/A"')
        local speed=$(echo "$response" | jq -r '.timings.predicted_per_second // "N/A"')

        log_success "LLM Response (${elapsed}ms)"
        echo -e "${YELLOW}Response:${NC}"
        echo "$content"
        echo ""
        log_info "Tokens predicted: $tokens"
        log_info "Speed: ${speed} tokens/sec"

        QUERY_TIMES["llm_direct"]=$elapsed
    else
        log_error "LLM request failed: $response"
        QUERY_TIMES["llm_direct"]=-1
    fi
}

run_sql_tests() {
    log_step "RUNNING SQL PIPELINE TESTS"

    local success=0
    local failed=0

    for query in "${TEST_QUERIES[@]}"; do
        echo ""
        echo -e "${CYAN}--- Test Query ---${NC}"
        if test_sql_query "$query" "EWRCentral"; then
            ((success++))
        else
            ((failed++))
        fi
        echo ""
    done

    log_step "TEST SUMMARY"
    log_info "Successful: $success"
    log_info "Failed: $failed"
}

print_metrics() {
    log_step "PERFORMANCE METRICS"

    local total_end=$(date +%s%3N)
    local total_time=$((total_end - TOTAL_START))

    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              SQL PIPELINE TEST METRICS                      ║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"

    for query in "${!QUERY_TIMES[@]}"; do
        local time=${QUERY_TIMES[$query]}
        if [ $time -gt 0 ]; then
            printf "${CYAN}║${NC} %-40s %15s ${CYAN}║${NC}\n" "${query:0:40}" "${time}ms"
        else
            printf "${CYAN}║${NC} %-40s %15s ${CYAN}║${NC}\n" "${query:0:40}" "FAILED"
        fi
    done

    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    printf "${CYAN}║${NC} ${GREEN}%-40s %15s${NC} ${CYAN}║${NC}\n" "TOTAL TIME" "${total_time}ms"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
}

main() {
    TOTAL_START=$(date +%s%3N)

    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           SQL CHAT PIPELINE TEST SCRIPT                     ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Tests the text-to-SQL generation pipeline including:       ║${NC}"
    echo -e "${CYAN}║  - LLM direct completion                                    ║${NC}"
    echo -e "${CYAN}║  - SQL generation from natural language                     ║${NC}"
    echo -e "${CYAN}║  - Rule matching and auto-fix                               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    check_services
    test_llm_direct
    run_sql_tests
    print_metrics

    echo ""
    log_success "SQL Pipeline tests completed!"
    echo ""
}

main "$@"

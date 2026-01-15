#!/bin/bash
#==============================================================================
# SQL Chat Pipeline Test Script
#==============================================================================
# This script tests the SQL text-to-SQL pipeline:
#   1. Send a natural language query
#   2. Receive generated SQL
#   3. Execute the SQL and show results
#   4. Report LLM metrics and timing
#
# Requirements:
#   - Python service running on localhost:8001
#   - llama.cpp SQL model running on localhost:8080
#   - jq installed for JSON parsing
#
# Usage: ./test_sql_pipeline.sh "your natural language query"
#        ./test_sql_pipeline.sh  (runs default test queries)
#==============================================================================

set -e

# Configuration
PYTHON_SERVER="http://localhost:8001"
LLM_SQL_SERVER="http://localhost:8080"
DEFAULT_DATABASE="EWRCentral"
DEFAULT_SERVER="EWRSQLPROD"
USE_CACHE="false"  # Bypass cache to ensure fresh LLM generation

# SQL Server credentials (can be overridden via environment variables)
SQL_USER="${SQL_USER:-EWR\\chad.walker}"
SQL_PASS="${SQL_PASS:-6454@@Christina}"
SQL_DOMAIN="${SQL_DOMAIN:-EWR}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Timing
declare -A QUERY_TIMES
declare -A LLM_METRICS
TOTAL_START=0

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
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
}

test_sql_query() {
    local query="$1"
    local database="${2:-$DEFAULT_DATABASE}"
    local execute="${3:-true}"

    log_step "SQL QUERY TEST"

    echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│                    USER QUERY                               │${NC}"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
    echo "$query"
    echo ""
    log_info "Database: $database"
    log_info "Execute: $execute"

    local start=$(date +%s%3N)

    # Create request for SQL query endpoint (bypass cache for fresh generation)
    local request=$(jq -n \
        --arg q "$query" \
        --arg db "$database" \
        --argjson cache "$USE_CACHE" \
        '{
            naturalLanguage: $q,
            database: $db,
            options: {
                use_cache: $cache
            }
        }')

    # Send to Python service SQL endpoint
    local response=$(curl -s -X POST "${PYTHON_SERVER}/api/sql/query" \
        -H "Content-Type: application/json" \
        -d "$request" 2>&1)

    local end=$(date +%s%3N)
    local elapsed=$((end - start))
    QUERY_TIMES["sql_generation"]=$elapsed

    # Check if successful
    if echo "$response" | jq -e '.sql' > /dev/null 2>&1; then
        local sql=$(echo "$response" | jq -r '.sql')
        local explanation=$(echo "$response" | jq -r '.explanation // "N/A"')
        local confidence=$(echo "$response" | jq -r '.confidence // "N/A"')
        local processing_time=$(echo "$response" | jq -r '.processing_time // "N/A"')
        local tokens_prompt=$(echo "$response" | jq -r '.token_usage.prompt_tokens // "N/A"')
        local tokens_response=$(echo "$response" | jq -r '.token_usage.response_tokens // "N/A"')
        local tokens_total=$(echo "$response" | jq -r '.token_usage.total_tokens // "N/A"')
        local matched_rules=$(echo "$response" | jq -r '.matched_rules | length // 0')
        local rule_id=$(echo "$response" | jq -r '.rule_id // "none"')
        local is_exact=$(echo "$response" | jq -r '.is_exact_match // false')
        local exec_result=$(echo "$response" | jq '.execution_result')

        # Store metrics
        LLM_METRICS["model"]="sqlcoder-7b"
        LLM_METRICS["tokens_prompt"]="$tokens_prompt"
        LLM_METRICS["tokens_completion"]="$tokens_response"
        LLM_METRICS["tokens_total"]="$tokens_total"
        LLM_METRICS["confidence"]="$confidence"
        LLM_METRICS["processing_time"]="$processing_time"
        LLM_METRICS["rule_match"]="$rule_id"
        LLM_METRICS["matched_rules_count"]="$matched_rules"
        LLM_METRICS["is_exact_match"]="$is_exact"
        LLM_METRICS["generation_time_ms"]="$elapsed"

        log_success "SQL Generated (${elapsed}ms)"

        # Display generated SQL
        echo ""
        echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
        echo -e "${YELLOW}│                    GENERATED SQL                            │${NC}"
        echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
        echo -e "${GREEN}$sql${NC}"
        echo ""

        # Display LLM explanation if available
        if [ "$explanation" != "N/A" ] && [ "$explanation" != "null" ] && [ -n "$explanation" ]; then
            echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
            echo -e "${YELLOW}│                    LLM EXPLANATION                          │${NC}"
            echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
            echo "$explanation" | fold -w 80 -s
            echo ""
        fi

        # Display LLM metrics
        echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
        echo -e "${YELLOW}│                    LLM GENERATION METRICS                   │${NC}"
        echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
        echo "Model: sqlcoder-7b (SQLCoder)"
        echo "Prompt Tokens: $tokens_prompt"
        echo "Response Tokens: $tokens_response"
        echo "Total Tokens: $tokens_total"
        echo "Confidence: $confidence"
        echo "Processing Time: ${processing_time}s"
        echo "Matched Rules: $matched_rules"
        echo "Rule ID: $rule_id"
        echo "Exact Match: $is_exact"
        echo "Total Request Time: ${elapsed}ms"
        echo ""

        # Check for execution results
        if [ "$exec_result" != "null" ] && [ -n "$exec_result" ]; then
            local results=$(echo "$exec_result" | jq '.data // .results // empty')
            local row_count=$(echo "$exec_result" | jq '.row_count // (.data | length) // 0')
            local exec_time=$(echo "$exec_result" | jq -r '.execution_time // "N/A"')
            local columns=$(echo "$exec_result" | jq -r '.columns // empty')

            if [ -n "$results" ] && [ "$results" != "null" ] && [ "$results" != "[]" ]; then
                LLM_METRICS["execution_time_ms"]="$exec_time"
                LLM_METRICS["row_count"]="$row_count"

                echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
                echo -e "${YELLOW}│                    QUERY RESULTS                            │${NC}"
                echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
                echo -e "${BLUE}Rows returned: $row_count${NC}"
                echo -e "${BLUE}Execution time: ${exec_time}${NC}"
                echo ""

                # Display results as a formatted table
                echo "$results" | jq -r '
                    if type == "array" and length > 0 then
                        (.[0] | keys_unsorted) as $keys |
                        ($keys | join("\t")),
                        "------------------------------------------------------------",
                        (.[] | [.[$keys[]]] | map(tostring) | join("\t"))
                    else
                        "No results"
                    end
                ' 2>/dev/null | column -t -s $'\t' | head -50 || echo "$results" | jq '.' | head -50
                echo ""
            else
                log_warning "No execution results returned (query may need to be executed separately)"
                log_info "Use the generated SQL to execute manually if needed"
            fi
        else
            # Execute the SQL separately if execution flag is true
            if [ "$execute" = "true" ]; then
                log_info "Executing generated SQL..."

                # Build execution request with credentials using Python for proper JSON escaping
                local exec_request=$(python3 -c "
import json
import sys
data = {
    'sql': '''$sql''',
    'database': '$database',
    'credentials': {
        'server': '$DEFAULT_SERVER',
        'database': '$database',
        'username': '$SQL_USER',
        'password': '$SQL_PASS',
        'use_windows_auth': True,
        'domain': '$SQL_DOMAIN'
    }
}
print(json.dumps(data))
" 2>/dev/null)

                local exec_response=$(curl -s -X POST "${PYTHON_SERVER}/api/sql/execute" \
                    -H "Content-Type: application/json" \
                    -d "$exec_request" 2>&1)

                if echo "$exec_response" | jq -e '.success == true' > /dev/null 2>&1; then
                    local results=$(echo "$exec_response" | jq '.data // .results // empty')
                    local row_count=$(echo "$exec_response" | jq '.row_count // (.data | length) // 0')
                    local exec_time=$(echo "$exec_response" | jq -r '.execution_time_ms // .execution_time // "N/A"')

                    LLM_METRICS["execution_time_ms"]="$exec_time"
                    LLM_METRICS["row_count"]="$row_count"

                    echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
                    echo -e "${YELLOW}│                    QUERY RESULTS                            │${NC}"
                    echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
                    echo -e "${BLUE}Rows returned: $row_count${NC}"
                    echo -e "${BLUE}Execution time: ${exec_time}${NC}"
                    echo ""

                    # Display results as a formatted table
                    if [ -n "$results" ] && [ "$results" != "null" ] && [ "$results" != "[]" ]; then
                        echo "$results" | jq -r '
                            if type == "array" and length > 0 then
                                (.[0] | keys_unsorted) as $keys |
                                ($keys | join("\t")),
                                "------------------------------------------------------------",
                                (.[] | [.[$keys[]]] | map(tostring) | join("\t"))
                            else
                                "No results"
                            end
                        ' 2>/dev/null | column -t -s $'\t' | head -50 || echo "$results" | jq '.' | head -50
                    else
                        echo "No results returned"
                    fi
                    echo ""
                else
                    log_warning "Query execution failed"
                    echo "$exec_response" | jq '.' 2>/dev/null || echo "$exec_response"
                fi
            fi
        fi

        return 0
    else
        log_error "Failed to generate SQL"
        echo -e "${RED}Response:${NC}"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
        return 1
    fi
}

get_llm_server_stats() {
    log_step "LLM SERVER STATISTICS"

    echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│                    LLM SERVER STATUS                        │${NC}"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"

    # Get detailed stats from SQL LLM server
    local health=$(curl -s "${LLM_SQL_SERVER}/health" 2>/dev/null)
    local props=$(curl -s "${LLM_SQL_SERVER}/props" 2>/dev/null)

    echo -e "${BLUE}SQL LLM Server (8080):${NC}"
    if [ -n "$health" ]; then
        echo "$health" | jq -r '
            "  Status: \(.status // "unknown")",
            "  Slots Idle: \(.slots_idle // "N/A")",
            "  Slots Processing: \(.slots_processing // "N/A")"
        ' 2>/dev/null || echo "  Status: available"
    else
        echo "  Status: unavailable"
    fi

    if [ -n "$props" ]; then
        echo ""
        echo -e "${BLUE}Model Properties:${NC}"
        echo "$props" | jq -r '
            "  Model: \(.model_alias // "N/A")",
            "  Context Size: \(.default_generation_settings.n_ctx // "N/A")",
            "  Total Slots: \(.total_slots // "N/A")",
            "  Temperature: \(.default_generation_settings.params.temperature // "N/A")"
        ' 2>/dev/null || true
    fi

    # Get metrics from /metrics endpoint if available
    local metrics=$(curl -s "${LLM_SQL_SERVER}/metrics" 2>/dev/null)
    if [ -n "$metrics" ] && [ "$metrics" != "null" ]; then
        echo ""
        echo -e "${BLUE}Server Metrics:${NC}"
        echo "$metrics" | grep -E 'requests_processing|tokens_predicted|prompt_tokens' | head -10 || true
    fi

    # Check other LLM servers
    echo ""
    echo -e "${BLUE}Other LLM Servers:${NC}"

    for port in 8081 8082 8083; do
        local name=""
        case $port in
            8081) name="General" ;;
            8082) name="Code" ;;
            8083) name="Embedding" ;;
        esac

        local status=$(curl -s "http://localhost:${port}/health" 2>/dev/null | jq -r '.status // "unavailable"' 2>/dev/null || echo "unavailable")
        echo "  ${name} (${port}): $status"
    done
    echo ""
}

print_final_metrics() {
    log_step "FINAL METRICS REPORT"

    local total_end=$(date +%s%3N)
    local total_time=$((total_end - TOTAL_START))

    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              SQL PIPELINE TEST METRICS                      ║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} ${YELLOW}LLM Generation:${NC}                                            ${CYAN}║${NC}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Model:" "${LLM_METRICS["model"]:-N/A}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Prompt Tokens:" "${LLM_METRICS["tokens_prompt"]:-N/A}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Response Tokens:" "${LLM_METRICS["tokens_completion"]:-N/A}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Total Tokens:" "${LLM_METRICS["tokens_total"]:-N/A}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Confidence:" "${LLM_METRICS["confidence"]:-N/A}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Processing Time:" "${LLM_METRICS["processing_time"]:-N/A}s"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} ${YELLOW}Query Execution:${NC}                                           ${CYAN}║${NC}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Rows Returned:" "${LLM_METRICS["row_count"]:-N/A}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Execution Time:" "${LLM_METRICS["execution_time_ms"]:-N/A}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Matched Rules:" "${LLM_METRICS["matched_rules_count"]:-0}"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Exact Match:" "${LLM_METRICS["is_exact_match"]:-false}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    printf "${CYAN}║${NC}   ${GREEN}%-25s %30s${NC} ${CYAN}║${NC}\n" "TOTAL TIME:" "${total_time}ms"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
}

show_usage() {
    echo "Usage: $0 [OPTIONS] [QUERY]"
    echo ""
    echo "Options:"
    echo "  -d, --database DB    Specify database (default: EWRCentral)"
    echo "  -n, --no-execute     Don't execute the generated SQL"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 \"How many tickets were entered by each user?\""
    echo "  $0 -d EWR \"Show all customers\""
    echo "  $0 -n \"Count all records\""
    echo ""
}

main() {
    TOTAL_START=$(date +%s%3N)

    # Parse arguments
    local database="$DEFAULT_DATABASE"
    local execute="true"
    local query=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--database)
                database="$2"
                shift 2
                ;;
            -n|--no-execute)
                execute="false"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                query="$1"
                shift
                ;;
        esac
    done

    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           SQL CHAT PIPELINE TEST SCRIPT                     ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Tests the text-to-SQL generation pipeline including:       ║${NC}"
    echo -e "${CYAN}║  - Natural language to SQL conversion                       ║${NC}"
    echo -e "${CYAN}║  - SQL execution and results                                ║${NC}"
    echo -e "${CYAN}║  - LLM metrics and performance                              ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    check_services

    if [ -z "$query" ]; then
        log_error "No query provided"
        show_usage
        exit 1
    fi

    test_sql_query "$query" "$database" "$execute"
    get_llm_server_stats
    print_final_metrics

    echo ""
    log_success "SQL Pipeline test completed!"
    echo ""
}

main "$@"

#!/bin/bash

# Test All LLMs - Comprehensive test for each model with performance metrics

echo "========================================"
echo "LLM Comprehensive Test Suite"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

# Arrays to store metrics for summary
declare -a model_names
declare -a response_times
declare -a tokens_generated
declare -a tokens_per_sec

test_embedding() {
    echo "----------------------------------------"
    echo -e "${YELLOW}[1/4] Testing Embedding Model (Port 8083)${NC}"
    echo "Model: nomic-embed-text-v1.5"
    echo "Input: \"The quick brown fox jumps over the lazy dog.\""
    echo "----------------------------------------"

    start_time=$(date +%s.%N)

    response=$(curl -s --max-time 30 http://localhost:8083/v1/embeddings \
        -H "Content-Type: application/json" \
        -d '{
            "model": "nomic-embed-text-v1.5",
            "input": "The quick brown fox jumps over the lazy dog."
        }' 2>/dev/null)

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo -e "${RED}[FAILED] No response from embedding service${NC}"
        echo "Check if service is running: nc -z localhost 8083"
        ((test_failed++))
        return 1
    fi

    # Check for error in response
    error=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('error',''))" 2>/dev/null)
    if [ -n "$error" ]; then
        echo -e "${RED}[FAILED] Error: $error${NC}"
        ((test_failed++))
        return 1
    fi

    # Extract dimensions
    dims=$(echo "$response" | python3 -c "
import sys, json
r = json.load(sys.stdin)
data = r.get('data', [{}])
if data:
    emb = data[0].get('embedding', [])
    print(len(emb))
else:
    print(0)
" 2>/dev/null)

    if [ "$dims" -gt 0 ]; then
        echo -e "${GREEN}[PASSED] Embedding generated successfully${NC}"
        echo ""
        echo -e "${CYAN}Performance Metrics:${NC}"
        printf "  Dimensions:    %s\n" "$dims"
        printf "  Response Time: %.2fs\n" "$elapsed"

        # Store metrics
        model_names+=("Embedding")
        response_times+=("$elapsed")
        tokens_generated+=("-")
        tokens_per_sec+=("-")

        ((test_passed++))
    else
        echo -e "${RED}[FAILED] Invalid embedding response${NC}"
        echo "Response: $response"
        ((test_failed++))
    fi
    echo ""
}

test_mistral() {
    echo "----------------------------------------"
    echo -e "${YELLOW}[2/4] Testing General Model (Port 8081)${NC}"
    echo "Model: mistral-7b-instruct-v0.2"
    echo "Question: What are the three laws of robotics?"
    echo "----------------------------------------"

    start_time=$(date +%s.%N)

    response=$(curl -s --max-time 60 http://localhost:8081/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mistral-7b-instruct",
            "messages": [
                {"role": "user", "content": "What are the three laws of robotics? Answer briefly."}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }' 2>/dev/null)

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo -e "${RED}[FAILED] No response from general model${NC}"
        ((test_failed++))
        return 1
    fi

    # Extract content and metrics
    metrics=$(echo "$response" | python3 -c "
import sys, json
r = json.load(sys.stdin)
choices = r.get('choices', [{}])
usage = r.get('usage', {})
content = ''
if choices:
    msg = choices[0].get('message', {})
    content = msg.get('content', '')
prompt_tokens = usage.get('prompt_tokens', 0)
completion_tokens = usage.get('completion_tokens', 0)
total_tokens = usage.get('total_tokens', 0)
print(f'{completion_tokens}|{total_tokens}|{content}')
" 2>/dev/null)

    completion_tokens=$(echo "$metrics" | cut -d'|' -f1)
    total_tokens=$(echo "$metrics" | cut -d'|' -f2)
    content=$(echo "$metrics" | cut -d'|' -f3-)

    if [ -n "$content" ]; then
        tps=$(echo "scale=2; $completion_tokens / $elapsed" | bc 2>/dev/null || echo "0")

        echo -e "${GREEN}[PASSED] General model responded${NC}"
        echo ""
        echo -e "${CYAN}Performance Metrics:${NC}"
        printf "  Response Time:    %.2fs\n" "$elapsed"
        printf "  Tokens Generated: %s\n" "$completion_tokens"
        printf "  Tokens/Second:    %s\n" "$tps"
        echo ""
        echo "Response:"
        echo "$content"

        # Store metrics
        model_names+=("Mistral-7B")
        response_times+=("$elapsed")
        tokens_generated+=("$completion_tokens")
        tokens_per_sec+=("$tps")

        ((test_passed++))
    else
        echo -e "${RED}[FAILED] No content in response${NC}"
        echo "Raw: $response"
        ((test_failed++))
    fi
    echo ""
}

test_sqlcoder() {
    echo "----------------------------------------"
    echo -e "${YELLOW}[3/4] Testing SQL Model (Port 8080)${NC}"
    echo "Model: sqlcoder-8b"
    echo "Question: Write a SQL query to count orders by customer using GROUP BY"
    echo "----------------------------------------"

    start_time=$(date +%s.%N)

    response=$(curl -s --max-time 60 http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "sqlcoder-8b",
            "messages": [
                {"role": "system", "content": "You are a SQL expert. Write only the SQL query, no explanations."},
                {"role": "user", "content": "Write a SQL query that counts the number of orders per customer from an Orders table that has CustomerID and OrderID columns. Use GROUP BY to group the results."}
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }' 2>/dev/null)

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo -e "${RED}[FAILED] No response from SQL model${NC}"
        ((test_failed++))
        return 1
    fi

    # Extract content and metrics
    metrics=$(echo "$response" | python3 -c "
import sys, json
r = json.load(sys.stdin)
choices = r.get('choices', [{}])
usage = r.get('usage', {})
content = ''
if choices:
    msg = choices[0].get('message', {})
    content = msg.get('content', '')
completion_tokens = usage.get('completion_tokens', 0)
print(f'{completion_tokens}|{content}')
" 2>/dev/null)

    completion_tokens=$(echo "$metrics" | cut -d'|' -f1)
    content=$(echo "$metrics" | cut -d'|' -f2-)

    if [ -n "$content" ]; then
        tps=$(echo "scale=2; $completion_tokens / $elapsed" | bc 2>/dev/null || echo "0")

        # Check if response contains SQL keywords
        if echo "$content" | grep -iq "SELECT\|GROUP BY"; then
            echo -e "${GREEN}[PASSED] SQL model generated valid query${NC}"
        else
            echo -e "${YELLOW}[WARNING] Response may not be valid SQL${NC}"
        fi
        echo ""
        echo -e "${CYAN}Performance Metrics:${NC}"
        printf "  Response Time:    %.2fs\n" "$elapsed"
        printf "  Tokens Generated: %s\n" "$completion_tokens"
        printf "  Tokens/Second:    %s\n" "$tps"
        echo ""
        echo "Generated SQL:"
        echo "$content"

        # Store metrics
        model_names+=("SQLCoder-8B")
        response_times+=("$elapsed")
        tokens_generated+=("$completion_tokens")
        tokens_per_sec+=("$tps")

        ((test_passed++))
    else
        echo -e "${RED}[FAILED] No content in response${NC}"
        echo "Raw: $response"
        ((test_failed++))
    fi
    echo ""
}

test_qwen_coder() {
    echo "----------------------------------------"
    echo -e "${YELLOW}[4/4] Testing Code Model (Port 8082)${NC}"
    echo "Model: qwen2.5-coder-7b-instruct"
    echo "Question: Write a C# method to check if a string is a palindrome"
    echo "----------------------------------------"

    start_time=$(date +%s.%N)

    response=$(curl -s --max-time 60 http://localhost:8082/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen2.5-coder",
            "messages": [
                {"role": "system", "content": "You are a C# expert. Write clean, well-commented code."},
                {"role": "user", "content": "Write a C# method called IsPalindrome that takes a string parameter and returns a boolean indicating whether the string is a palindrome. Include a brief example of usage."}
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }' 2>/dev/null)

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo -e "${RED}[FAILED] No response from code model${NC}"
        ((test_failed++))
        return 1
    fi

    # Extract content and metrics
    metrics=$(echo "$response" | python3 -c "
import sys, json
r = json.load(sys.stdin)
choices = r.get('choices', [{}])
usage = r.get('usage', {})
content = ''
if choices:
    msg = choices[0].get('message', {})
    content = msg.get('content', '')
completion_tokens = usage.get('completion_tokens', 0)
print(f'{completion_tokens}|{content}')
" 2>/dev/null)

    completion_tokens=$(echo "$metrics" | cut -d'|' -f1)
    content=$(echo "$metrics" | cut -d'|' -f2-)

    if [ -n "$content" ]; then
        tps=$(echo "scale=2; $completion_tokens / $elapsed" | bc 2>/dev/null || echo "0")

        # Check if response contains C# keywords
        if echo "$content" | grep -iq "bool\|string\|return\|public"; then
            echo -e "${GREEN}[PASSED] Code model generated C# code${NC}"
        else
            echo -e "${YELLOW}[WARNING] Response may not be valid C#${NC}"
        fi
        echo ""
        echo -e "${CYAN}Performance Metrics:${NC}"
        printf "  Response Time:    %.2fs\n" "$elapsed"
        printf "  Tokens Generated: %s\n" "$completion_tokens"
        printf "  Tokens/Second:    %s\n" "$tps"
        echo ""
        echo "Generated C#:"
        echo "$content"

        # Store metrics
        model_names+=("Qwen-Coder-7B")
        response_times+=("$elapsed")
        tokens_generated+=("$completion_tokens")
        tokens_per_sec+=("$tps")

        ((test_passed++))
    else
        echo -e "${RED}[FAILED] No content in response${NC}"
        echo "Raw: $response"
        ((test_failed++))
    fi
    echo ""
}

# Check which services are running first
echo "Checking service availability..."
echo ""

services_ok=true

for port in 8080 8081 8082 8083; do
    if nc -z localhost $port 2>/dev/null; then
        echo "  Port $port: ✓ Available"
    else
        echo "  Port $port: ✗ Not available"
        services_ok=false
    fi
done

echo ""

if [ "$services_ok" = false ]; then
    echo -e "${YELLOW}Warning: Some services are not running.${NC}"
    echo "Start them with: ./llm_start.sh"
    echo ""
    read -p "Continue anyway? (y/n): " yn
    if [[ $yn != "y" ]]; then
        exit 1
    fi
fi

# Run tests
echo ""
test_embedding

# Test General model if port 8081 is up
if nc -z localhost 8081 2>/dev/null; then
    test_mistral
else
    echo "----------------------------------------"
    echo -e "${YELLOW}[SKIP] General Model (Port 8081) - Not running${NC}"
    echo "Note: SQL and General models time-share GPU 0"
    echo "To start: sudo systemctl stop llama-sql && sudo systemctl start llama-general"
    echo "----------------------------------------"
    echo ""
fi

# Test SQL model if port 8080 is up
if nc -z localhost 8080 2>/dev/null; then
    test_sqlcoder
else
    echo "----------------------------------------"
    echo -e "${YELLOW}[SKIP] SQL Model (Port 8080) - Not running${NC}"
    echo "Note: SQL and General models time-share GPU 0"
    echo "To start: sudo systemctl stop llama-general && sudo systemctl start llama-sql"
    echo "----------------------------------------"
    echo ""
fi

test_qwen_coder

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$test_passed${NC}"
echo -e "Failed: ${RED}$test_failed${NC}"
echo "========================================"
echo ""

# Performance comparison table
if [ ${#model_names[@]} -gt 0 ]; then
    echo -e "${CYAN}Performance Comparison:${NC}"
    echo "------------------------------------------------------------"
    printf "%-16s %12s %12s %12s\n" "MODEL" "TIME (s)" "TOKENS" "TOKENS/SEC"
    echo "------------------------------------------------------------"

    for i in "${!model_names[@]}"; do
        printf "%-16s %12s %12s %12s\n" \
            "${model_names[$i]}" \
            "${response_times[$i]}" \
            "${tokens_generated[$i]}" \
            "${tokens_per_sec[$i]}"
    done
    echo "------------------------------------------------------------"
    echo ""
fi

# GPU status after tests
echo "GPU Status After Tests:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while read line; do
    gpu_id=$(echo $line | cut -d',' -f1)
    gpu_name=$(echo $line | cut -d',' -f2 | xargs)
    mem_used=$(echo $line | cut -d',' -f3 | xargs)
    mem_total=$(echo $line | cut -d',' -f4 | xargs)
    gpu_util=$(echo $line | cut -d',' -f5 | xargs)
    printf "  GPU %s: %s - %sMB / %sMB (Util: %s%%)\n" "$gpu_id" "$gpu_name" "$mem_used" "$mem_total" "$gpu_util"
done

echo ""
echo "========================================"

if [ $test_failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed! LLMs are working correctly.${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check the output above.${NC}"
    exit 1
fi

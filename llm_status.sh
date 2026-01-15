#!/bin/bash

# LLM Status Check - Quick view of all LLM services

echo "========================================"
echo "LLM Service Status"
echo "========================================"

check_service() {
    local name="$1"
    local port="$2"
    local model="$3"

    # Check if port is listening
    if nc -z localhost $port 2>/dev/null; then
        # Try to get model info from health endpoint
        health=$(curl -s --max-time 2 http://localhost:$port/health 2>/dev/null)
        if [ $? -eq 0 ]; then
            printf "%-12s %-6s %-10s %s\n" "$name" "$port" "✓ RUNNING" "$model"
        else
            printf "%-12s %-6s %-10s %s\n" "$name" "$port" "✓ RUNNING" "$model"
        fi
    else
        printf "%-12s %-6s %-10s %s\n" "$name" "$port" "✗ STOPPED" "$model"
    fi
}

printf "%-12s %-6s %-10s %s\n" "SERVICE" "PORT" "STATUS" "MODEL"
echo "----------------------------------------"
check_service "SQL" 8080 "sqlcoder-8b"
check_service "General" 8081 "mistral-7b"
check_service "Code" 8082 "qwen2.5-coder-7b"
check_service "Embedding" 8083 "nomic-embed-v1.5"
echo "========================================"

# GPU Memory Usage
echo ""
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while read line; do
    gpu_id=$(echo $line | cut -d',' -f1)
    gpu_name=$(echo $line | cut -d',' -f2 | xargs)
    mem_used=$(echo $line | cut -d',' -f3 | xargs)
    mem_total=$(echo $line | cut -d',' -f4 | xargs)
    printf "  GPU %s: %s - %sMB / %sMB\n" "$gpu_id" "$gpu_name" "$mem_used" "$mem_total"
done

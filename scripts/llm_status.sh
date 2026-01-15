#!/bin/bash
# LLM Services Status Script

echo "=========================================="
echo "         LLM Server Status"
echo "=========================================="
echo ""

# Service status
echo "SERVICES:"
echo "------------------------------------------"
printf "%-20s %-10s %-6s %s\n" "SERVICE" "STATUS" "PORT" "MODEL"
echo "------------------------------------------"

services=(
    "llama-sql:8080:SQLCoder-7B"
    "llama-embedding:8081:Nomic-Embed"
    "llama-code:8082:Qwen2.5-Coder-1.5B"
    "llama-general:8083:Qwen2.5-3B"
)

for svc in "${services[@]}"; do
    name=$(echo "$svc" | cut -d: -f1)
    port=$(echo "$svc" | cut -d: -f2)
    model=$(echo "$svc" | cut -d: -f3)

    if systemctl is-active --quiet "$name"; then
        status="RUNNING"
    else
        status="STOPPED"
    fi

    printf "%-20s %-10s %-6s %s\n" "$name" "$status" "$port" "$model"
done

echo ""

# Health checks
echo "HEALTH CHECKS:"
echo "------------------------------------------"
for svc in "${services[@]}"; do
    name=$(echo "$svc" | cut -d: -f1)
    port=$(echo "$svc" | cut -d: -f2)

    health=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://localhost:$port/health" 2>/dev/null)
    if [ "$health" = "200" ]; then
        printf "%-20s OK\n" "$name"
    else
        printf "%-20s FAIL (HTTP $health)\n" "$name"
    fi
done

echo ""

# GPU status
echo "GPU USAGE:"
echo "------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while read line; do
    idx=$(echo "$line" | cut -d, -f1 | xargs)
    name=$(echo "$line" | cut -d, -f2 | xargs)
    mem_used=$(echo "$line" | cut -d, -f3 | xargs)
    mem_total=$(echo "$line" | cut -d, -f4 | xargs)
    util=$(echo "$line" | cut -d, -f5 | xargs)
    printf "GPU %s: %s - %sMB/%sMB (%s%% util)\n" "$idx" "$name" "$mem_used" "$mem_total" "$util"
done

echo ""
echo "=========================================="

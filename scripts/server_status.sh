#!/bin/bash
# Server Status Script

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

get_status() {
    if systemctl is-active --quiet "$1" 2>/dev/null; then
        echo -e "${GREEN}Active${NC}"
    else
        echo -e "${RED}Inactive${NC}"
    fi
}

get_mongo_status() {
    # Check Docker container first
    if docker ps --filter "name=mongodb" --format "{{.Status}}" 2>/dev/null | grep -q "Up"; then
        echo -e "${GREEN}Active${NC} (Docker)"
    # Fall back to systemd service check
    elif systemctl is-active --quiet mongod 2>/dev/null; then
        echo -e "${GREEN}Active${NC}"
    # Check if port is responding as last resort
    elif nc -z localhost 27017 2>/dev/null; then
        echo -e "${GREEN}Active${NC} (port)"
    else
        echo -e "${RED}Inactive${NC}"
    fi
}

get_health() {
    local code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://localhost:$1/health" 2>/dev/null)
    if [ "$code" = "200" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAIL${NC}"
    fi
}

echo ""
echo "=========================================="
echo "  SERVER STATUS - $(hostname)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

echo ""
echo "SYSTEM"
echo "------------------------------------------"
printf "  %-16s %s\n" "Uptime:" "$(uptime -p | sed 's/up //')"
printf "  %-16s %s\n" "Load:" "$(cat /proc/loadavg | awk '{print $1, $2, $3}')"
printf "  %-16s %s\n" "Memory:" "$(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
printf "  %-16s %s\n" "Disk (/):" "$(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"
printf "  %-16s %s\n" "Disk (/data):" "$(df -h /data 2>/dev/null | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}' || echo 'N/A')"

echo ""
echo "CORE SERVICES"
echo "------------------------------------------"
printf "  %-24s %-12b\n" "ssh" "$(get_status ssh)"
printf "  %-24s %-12b\n" "cron" "$(get_status cron)"
printf "  %-24s %-12b\n" "nginx" "$(get_status nginx)"
printf "  %-24s %-12b\n" "docker" "$(get_status docker)"
printf "  %-24s %-12b\n" "sssd (AD Auth)" "$(get_status sssd)"

echo ""
echo "DATABASE & MONITORING"
echo "------------------------------------------"
printf "  %-24s %-12b %s\n" "mongodb" "$(get_mongo_status)" "27017"
printf "  %-24s %-12b %s\n" "grafana-server" "$(get_status grafana-server)" "3000"
printf "  %-24s %-12b %s\n" "cockpit" "$(get_status cockpit)" "9090"
printf "  %-24s %-12b %s\n" "prefect-server" "$(get_status prefect-server)" "4200"

echo ""
echo "LLM SERVICES"
echo "------------------------------------------"
printf "  %-20s %-12s %-6s %-6s %s\n" "SERVICE" "STATUS" "PORT" "HEALTH" "MODEL"
printf "  %-20s %-12b %-6s %-14b %s\n" "llama-sql" "$(get_status llama-sql)" "8080" "$(get_health 8080)" "SQLCoder-7B"
printf "  %-20s %-12b %-6s %-14b %s\n" "llama-embedding" "$(get_status llama-embedding)" "8081" "$(get_health 8081)" "Nomic-Embed"
printf "  %-20s %-12b %-6s %-14b %s\n" "llama-code" "$(get_status llama-code)" "8082" "$(get_health 8082)" "Qwen2.5-Coder-1.5B"
printf "  %-20s %-12b %-6s %-14b %s\n" "llama-general" "$(get_status llama-general)" "8083" "$(get_health 8083)" "Qwen2.5-3B"

echo ""
echo "GPU STATUS"
echo "------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while read line; do
    idx=$(echo "$line" | cut -d, -f1 | xargs)
    name=$(echo "$line" | cut -d, -f2 | xargs)
    mem_used=$(echo "$line" | cut -d, -f3 | xargs)
    mem_total=$(echo "$line" | cut -d, -f4 | xargs)
    util=$(echo "$line" | cut -d, -f5 | xargs)
    printf "  GPU %s: %-28s %5sMB / %sMB  (%s%% util)\n" "$idx" "$name" "$mem_used" "$mem_total" "$util"
done

echo ""
echo "NETWORK"
echo "------------------------------------------"
ip -4 addr show | grep -E "inet " | grep -v "127.0.0.1" | awk '{print "  " $NF ": " $2}'

echo ""
echo "=========================================="
echo ""

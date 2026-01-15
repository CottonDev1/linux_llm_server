#!/bin/bash
# Stop Management Services Script
# Location on server: /data/projects/llm_website/stop-management-services.sh
#
# This script stops all management and monitoring services:
# - Cockpit (port 9090)
# - Grafana (port 3001)
# - Prometheus (port 9091)
# - Prefect (port 4200)

echo "=========================================="
echo "  Stopping Management Services"
echo "=========================================="

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

stop_service() {
    local service=$1
    echo -n "Stopping $service... "
    if sudo systemctl stop "$service" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC} (may not be running)"
    fi
}

# Stop Cockpit
stop_service "cockpit.socket"
stop_service "cockpit"

# Stop Grafana
stop_service "grafana-server"

# Stop Prometheus
stop_service "prometheus"

# Stop Prefect
echo -n "Stopping Prefect... "
if pkill -f "prefect server" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}NOT RUNNING${NC}"
fi

echo ""
echo "All management services stopped."
echo ""

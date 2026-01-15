#!/bin/bash
# Start Management Services Script
# Location on server: /data/projects/llm_website/start-management-services.sh
#
# This script starts all management and monitoring services:
# - Cockpit (port 9090) - System administration
# - Grafana (port 3001) - Metrics visualization
# - Prometheus (port 9091) - Metrics collection
# - Prefect (port 4200) - Workflow orchestration

set -e

echo "=========================================="
echo "  Starting Management Services"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

start_service() {
    local service=$1
    local port=$2
    local url=$3

    echo -n "Starting $service... "
    if sudo systemctl start "$service" 2>/dev/null; then
        sleep 2
        if sudo systemctl is-active --quiet "$service"; then
            echo -e "${GREEN}OK${NC} ($url)"
        else
            echo -e "${RED}FAILED${NC}"
        fi
    else
        echo -e "${RED}FAILED${NC} (service not found)"
    fi
}

# Start Cockpit
start_service "cockpit.socket" "9090" "https://$(hostname -I | awk '{print $1}'):9090"

# Start Prometheus
start_service "prometheus" "9091" "http://$(hostname -I | awk '{print $1}'):9091"

# Start Grafana
start_service "grafana-server" "3001" "http://$(hostname -I | awk '{print $1}'):3001"

# Start Prefect (if installed as a service)
if systemctl list-unit-files | grep -q prefect; then
    start_service "prefect" "4200" "http://$(hostname -I | awk '{print $1}'):4200"
else
    echo -n "Starting Prefect... "
    # Check if prefect is already running
    if pgrep -f "prefect server" > /dev/null; then
        echo -e "${GREEN}Already running${NC} (http://$(hostname -I | awk '{print $1}'):4200)"
    else
        # Start Prefect in background
        if command -v prefect &> /dev/null; then
            nohup prefect server start --host 0.0.0.0 > /var/log/prefect-server.log 2>&1 &
            sleep 3
            if pgrep -f "prefect server" > /dev/null; then
                echo -e "${GREEN}OK${NC} (http://$(hostname -I | awk '{print $1}'):4200)"
            else
                echo -e "${RED}FAILED${NC}"
            fi
        else
            echo -e "${RED}NOT INSTALLED${NC}"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "  Service Status"
echo "=========================================="
echo ""
echo "Checking ports..."
echo ""

check_port() {
    local port=$1
    local name=$2
    if ss -tlnp | grep -q ":$port "; then
        echo -e "  $name (port $port): ${GREEN}LISTENING${NC}"
    else
        echo -e "  $name (port $port): ${RED}NOT LISTENING${NC}"
    fi
}

check_port 9090 "Cockpit"
check_port 9091 "Prometheus"
check_port 3001 "Grafana"
check_port 4200 "Prefect"

echo ""
echo "=========================================="
echo "  Access URLs"
echo "=========================================="
IP=$(hostname -I | awk '{print $1}')
echo ""
echo "  Cockpit:    https://$IP:9090  (chad/admin)"
echo "  Grafana:    http://$IP:3001   (admin/admin)"
echo "  Prometheus: http://$IP:9091"
echo "  Prefect:    http://$IP:4200"
echo ""

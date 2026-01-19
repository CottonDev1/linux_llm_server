#!/bin/bash
#==============================================================================
# Script Menu - Interactive launcher for website scripts
#==============================================================================

SCRIPT_DIR="/home/chad/website"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Script definitions: script_name|description
SCRIPTS=(
    "llm_start.sh|LLM Service Manager - Start/stop individual or all LLM models"
    "llm_status.sh|LLM Status Check - Quick view of all LLM service states and GPU usage"
    "start-management-services.sh|Start Management Services - Cockpit, Grafana, Prometheus, Prefect"
    "stop-management-services.sh|Stop Management Services - Stop all monitoring services"
    "test_all_llms.sh|Test All LLMs - Run comprehensive tests on all models with metrics"
    "web_start.sh|Web Application Layer - Start/stop Docker, MongoDB, Python, Node.js services"
    "web_connections.sh|Web Connections - View active connections to web and LLM ports"
)

show_menu() {
    clear
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}       Website Scripts Menu${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""

    local i=1
    for item in "${SCRIPTS[@]}"; do
        IFS='|' read -r script desc <<< "$item"
        echo -e "  ${CYAN}$i)${NC} ${BOLD}$script${NC}"
        echo -e "     $desc"
        echo ""
        ((i++))
    done

    echo -e "  ${YELLOW}8)${NC} ${BOLD}Utility Scripts Menu${NC}"
    echo -e "     Open submenu with backup, restore, testing, and utility scripts"
    echo ""
    echo -e "  ${RED}q)${NC} Quit"
    echo ""
    echo -e "${BOLD}========================================${NC}"
}

run_script() {
    local script="$1"
    local script_path="$SCRIPT_DIR/$script"

    if [ ! -f "$script_path" ]; then
        echo -e "${RED}Error: Script not found: $script_path${NC}"
        return 1
    fi

    if [ ! -x "$script_path" ]; then
        echo -e "${YELLOW}Making script executable...${NC}"
        chmod +x "$script_path"
    fi

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running: $script${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Run the script
    cd "$SCRIPT_DIR"
    bash "$script_path"

    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Script completed: $script${NC}"
    echo -e "${YELLOW}========================================${NC}"
}

# Main loop
while true; do
    show_menu
    echo -n "Enter choice [1-8 or q]: "
    read -r choice

    case "$choice" in
        [1-7])
            idx=$((choice - 1))
            IFS='|' read -r script desc <<< "${SCRIPTS[$idx]}"
            run_script "$script"
            echo ""
            read -p "Press Enter to return to menu..."
            ;;
        8)
            # Launch scripts submenu
            bash "$SCRIPT_DIR/scripts/menu.sh"
            ;;
        q|Q)
            echo ""
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            sleep 1
            ;;
    esac
done

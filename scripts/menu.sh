#!/bin/bash
#==============================================================================
# Scripts Menu - Interactive launcher for utility scripts
#==============================================================================

SCRIPT_DIR="/home/chad/website/scripts"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Script definitions: script_name|description
SCRIPTS=(
    "llm_status.sh|LLM Status - Quick view of LLM service states"
    "install_claude_code.sh|Install Claude Code - Install Claude Code CLI tool"
    "start_prefect_flows.sh|Start Prefect Flows - Launch Prefect workflow orchestration"
    "test_audio_pipeline.sh|Test Audio Pipeline - End-to-end audio processing test"
    "test_document_pipeline.sh|Test Document Pipeline - Document upload, embed, and search test"
    "test_sql_pipeline.sh|Test SQL Pipeline - Natural language to SQL generation test"
    "test_service_management.sh|Test Service Management - Python service API tests"
)

show_menu() {
    clear
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}       Utility Scripts Menu${NC}"
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

    echo -e "  ${YELLOW}8)${NC} ${BOLD}Website Maintenance Menu${NC}"
    echo -e "     Backup, restore, cleanup, and server status scripts"
    echo ""
    echo -e "  ${RED}q)${NC} Return to main menu"
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
            # Launch website maintenance submenu
            bash "$SCRIPT_DIR/website_maintenance/menu.sh"
            ;;
        q|Q)
            echo ""
            echo "Returning to main menu..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            sleep 1
            ;;
    esac
done

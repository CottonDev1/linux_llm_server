#!/bin/bash
#==============================================================================
# Web Connections Monitor
# Shows active connections, logged-in users, and session information
#==============================================================================

# Configuration
DB_PATH="/home/chad/website/data/EWR_AI.db"
NODE_LOG="/data/logs/llama/node_service.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Web ports to monitor
WEB_PORTS="3000|8001"
LLM_PORTS="8080|8081|8082|8083"
ALL_PORTS="3000|8001|8080|8081|8082|8083"

show_connections() {
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}  Web Connections & User Sessions${NC}"
    echo -e "${BOLD}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BOLD}========================================${NC}"

    # Active Sessions (users with valid refresh tokens)
    echo ""
    echo -e "${CYAN}Logged-In Users (Active Sessions):${NC}"
    echo "----------------------------------------"
    if [ -f "$DB_PATH" ]; then
        local now=$(date -u +"%Y-%m-%dT%H:%M:%S")
        local sessions=$(sqlite3 -separator '|' "$DB_PATH" "
            SELECT u.username, u.role, u.last_login_at, COUNT(r.token_id) as session_count
            FROM users u
            JOIN refresh_tokens r ON u.user_id = r.user_id
            WHERE r.expires_at > '$now'
            GROUP BY u.user_id
            ORDER BY u.last_login_at DESC;
        " 2>/dev/null)

        if [ -n "$sessions" ]; then
            printf "  ${BOLD}%-20s %-10s %-20s %s${NC}\n" "USERNAME" "ROLE" "LAST LOGIN" "SESSIONS"
            echo "$sessions" | while IFS='|' read -r username role last_login count; do
                printf "  %-20s %-10s %-20s %s\n" "$username" "$role" "$last_login" "$count"
            done
        else
            echo -e "  ${YELLOW}No active sessions${NC}"
        fi
    else
        echo -e "  ${RED}Database not found${NC}"
    fi

    # Recent Logins
    echo ""
    echo -e "${CYAN}Recent Login Activity:${NC}"
    echo "----------------------------------------"
    if [ -f "$NODE_LOG" ]; then
        local logins=$(grep -i "authenticated successfully" "$NODE_LOG" 2>/dev/null | tail -5)
        if [ -n "$logins" ]; then
            echo "$logins" | while read -r line; do
                echo "  $line"
            done
        else
            echo -e "  ${YELLOW}No recent logins in log${NC}"
        fi
    else
        echo -e "  ${YELLOW}Log file not found${NC}"
    fi

    # Active TCP Connections
    echo ""
    echo -e "${CYAN}Active TCP Connections:${NC}"
    echo "----------------------------------------"

    # Web application ports
    echo -e "  ${BOLD}Web Ports (3000, 8001):${NC}"
    local web_conns=$(ss -tn state established 2>/dev/null | grep -E ":($WEB_PORTS)" | grep -v "127.0.0.1")
    if [ -n "$web_conns" ]; then
        echo "$web_conns" | awk '{print "    " $3 " <- " $4}'
    else
        echo -e "    ${YELLOW}No external connections${NC}"
    fi

    # LLM service ports
    echo -e "  ${BOLD}LLM Ports (8080-8083):${NC}"
    local llm_conns=$(ss -tn state established 2>/dev/null | grep -E ":($LLM_PORTS)" | grep -v "127.0.0.1")
    if [ -n "$llm_conns" ]; then
        echo "$llm_conns" | awk '{print "    " $3 " <- " $4}'
    else
        echo -e "    ${YELLOW}No external connections${NC}"
    fi

    # Connection counts by IP
    echo ""
    echo -e "${CYAN}Connections by Remote IP:${NC}"
    echo "----------------------------------------"
    local by_ip=$(ss -tn state established 2>/dev/null | grep -E ":($ALL_PORTS)" | grep -v "127.0.0.1" | awk '{print $4}' | rev | cut -d: -f2- | rev | sort | uniq -c | sort -rn)
    if [ -n "$by_ip" ]; then
        echo "$by_ip" | while read count ip; do
            printf "  %-20s %s connections\n" "$ip" "$count"
        done
    else
        echo -e "  ${YELLOW}No external connections${NC}"
    fi

    # All Users Summary
    echo ""
    echo -e "${CYAN}All Users:${NC}"
    echo "----------------------------------------"
    if [ -f "$DB_PATH" ]; then
        printf "  ${BOLD}%-20s %-10s %-20s %s${NC}\n" "USERNAME" "ROLE" "LAST LOGIN" "STATUS"
        sqlite3 -separator '|' "$DB_PATH" "
            SELECT username, role, COALESCE(last_login_at, 'Never'),
                   CASE WHEN is_active = 1 THEN 'Active' ELSE 'Disabled' END
            FROM users ORDER BY last_login_at DESC;
        " 2>/dev/null | while IFS='|' read -r username role last_login status; do
            printf "  %-20s %-10s %-20s %s\n" "$username" "$role" "$last_login" "$status"
        done
    fi

    # Listening ports
    echo ""
    echo -e "${CYAN}Listening Ports:${NC}"
    echo "----------------------------------------"
    ss -tlnp 2>/dev/null | grep -E ":($ALL_PORTS)" | awk '{print "  " $4}' | sort -t: -k2 -n
    echo ""
}

# Main
case "${1:-}" in
    watch)
        while true; do
            clear
            show_connections
            echo -e "${YELLOW}Refreshing every 5 seconds... (Ctrl+C to stop)${NC}"
            sleep 5
        done
        ;;
    *)
        show_connections
        echo "Tip: Run with 'watch' argument for live updates:"
        echo "  ./web_connections.sh watch"
        echo ""
        ;;
esac

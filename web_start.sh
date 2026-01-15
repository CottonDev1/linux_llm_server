#!/bin/bash
#==============================================================================
# Web Application Layer Start Script
#==============================================================================
# Starts all web services required for the LLM application:
#   - Docker daemon (if not running)
#   - MongoDB container (starts or restarts if unhealthy)
#   - Python FastAPI service (port 8001)
#   - Node.js RAG server (port 3000)
#
# Usage: ./web_start.sh [start|stop|status|restart]
#==============================================================================

set -e

# Configuration
PROJECT_DIR="/data/projects/llm_website"
LOG_DIR="/data/logs/llama"
PID_DIR="/data/run"
PYTHON_VENV="$PROJECT_DIR/python_services/venv"
MONGODB_CONTAINER="mongodb"

# Ports
PYTHON_PORT=8001
NODE_PORT=3000
MONGO_PORT=27017

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p "$LOG_DIR" "$PID_DIR"

#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

is_port_open() {
    local port="$1"
    nc -z localhost "$port" 2>/dev/null
    return $?
}

wait_for_port() {
    local port="$1"
    local timeout="${2:-30}"
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if is_port_open "$port"; then
            return 0
        fi
        sleep 1
        ((elapsed++))
    done
    return 1
}

#------------------------------------------------------------------------------
# Docker Functions
#------------------------------------------------------------------------------

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        return 1
    fi

    if ! docker info &> /dev/null; then
        log_warning "Docker daemon is not running"
        return 1
    fi

    return 0
}

start_docker() {
    log_info "Checking Docker daemon..."

    if check_docker; then
        log_success "Docker daemon is running"
        return 0
    fi

    log_info "Starting Docker daemon..."
    sudo systemctl start docker
    sleep 3

    if check_docker; then
        log_success "Docker daemon started"
        return 0
    else
        log_error "Failed to start Docker daemon"
        return 1
    fi
}

#------------------------------------------------------------------------------
# MongoDB Functions
#------------------------------------------------------------------------------

get_mongo_status() {
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${MONGODB_CONTAINER}$"; then
        echo "not_found"
        return
    fi

    local status=$(docker inspect --format='{{.State.Status}}' "$MONGODB_CONTAINER" 2>/dev/null)
    local health=$(docker inspect --format='{{.State.Health.Status}}' "$MONGODB_CONTAINER" 2>/dev/null || echo "none")

    if [ "$status" != "running" ]; then
        echo "stopped"
    elif [ "$health" = "unhealthy" ]; then
        echo "unhealthy"
    elif [ "$health" = "starting" ]; then
        echo "starting"
    else
        echo "healthy"
    fi
}

start_mongodb() {
    log_info "Checking MongoDB container..."

    local status=$(get_mongo_status)

    case "$status" in
        "not_found")
            log_error "MongoDB container '$MONGODB_CONTAINER' not found"
            log_info "Please create it first with: docker run -d --name mongodb -p 27017:27017 mongodb/mongodb-atlas-local:8"
            return 1
            ;;
        "stopped")
            log_info "Starting MongoDB container..."
            docker start "$MONGODB_CONTAINER"
            ;;
        "unhealthy")
            log_warning "MongoDB container is unhealthy, restarting..."
            docker restart "$MONGODB_CONTAINER"
            ;;
        "starting")
            log_info "MongoDB container is starting..."
            ;;
        "healthy")
            log_success "MongoDB is already running and healthy"
            return 0
            ;;
    esac

    # Wait for MongoDB to be ready
    log_info "Waiting for MongoDB to be ready..."
    local timeout=60
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        status=$(get_mongo_status)
        if [ "$status" = "healthy" ]; then
            log_success "MongoDB is ready"
            return 0
        elif [ "$status" = "unhealthy" ]; then
            log_error "MongoDB became unhealthy"
            return 1
        fi
        sleep 2
        ((elapsed+=2))
    done

    # Check if port is at least open even if health check isn't complete
    if is_port_open $MONGO_PORT; then
        log_success "MongoDB port is open (health check may still be in progress)"
        return 0
    fi

    log_error "MongoDB failed to start within $timeout seconds"
    return 1
}

stop_mongodb() {
    log_info "Stopping MongoDB container..."
    docker stop "$MONGODB_CONTAINER" 2>/dev/null || true
    log_success "MongoDB stopped"
}

#------------------------------------------------------------------------------
# Python Service Functions
#------------------------------------------------------------------------------

start_python_service() {
    log_info "Starting Python FastAPI service..."

    if is_port_open $PYTHON_PORT; then
        log_success "Python service is already running on port $PYTHON_PORT"
        return 0
    fi

    cd "$PROJECT_DIR/python_services"
    source "$PYTHON_VENV/bin/activate"

    nohup python main.py > "$LOG_DIR/python_service.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_DIR/python_service.pid"

    log_info "Waiting for Python service to start..."
    if wait_for_port $PYTHON_PORT 30; then
        log_success "Python service started (PID: $pid)"
        return 0
    else
        log_error "Python service failed to start - check $LOG_DIR/python_service.log"
        tail -20 "$LOG_DIR/python_service.log"
        return 1
    fi
}

stop_python_service() {
    log_info "Stopping Python service..."

    # Try PID file first
    if [ -f "$PID_DIR/python_service.pid" ]; then
        local pid=$(cat "$PID_DIR/python_service.pid")
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
            sleep 2
        fi
        rm -f "$PID_DIR/python_service.pid"
    fi

    # Also check for any .pid files created by the Python service itself
    rm -f "$PROJECT_DIR/python_services/.python_service_${PYTHON_PORT}"*.pid 2>/dev/null

    # Kill by port if still running
    if is_port_open $PYTHON_PORT; then
        fuser -k $PYTHON_PORT/tcp 2>/dev/null || true
        sleep 1
    fi

    log_success "Python service stopped"
}

#------------------------------------------------------------------------------
# Node.js Service Functions
#------------------------------------------------------------------------------

start_node_service() {
    log_info "Starting Node.js RAG server..."

    if is_port_open $NODE_PORT; then
        log_success "Node.js service is already running on port $NODE_PORT"
        return 0
    fi

    cd "$PROJECT_DIR"

    nohup node rag-server.js > "$LOG_DIR/node_service.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_DIR/node_service.pid"

    log_info "Waiting for Node.js service to start..."
    if wait_for_port $NODE_PORT 30; then
        log_success "Node.js service started (PID: $pid)"
        return 0
    else
        log_error "Node.js service failed to start - check $LOG_DIR/node_service.log"
        tail -20 "$LOG_DIR/node_service.log"
        return 1
    fi
}

stop_node_service() {
    log_info "Stopping Node.js service..."

    if [ -f "$PID_DIR/node_service.pid" ]; then
        local pid=$(cat "$PID_DIR/node_service.pid")
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
            sleep 2
        fi
        rm -f "$PID_DIR/node_service.pid"
    fi

    # Kill by port if still running
    if is_port_open $NODE_PORT; then
        fuser -k $NODE_PORT/tcp 2>/dev/null || true
        sleep 1
    fi

    log_success "Node.js service stopped"
}

#------------------------------------------------------------------------------
# Status Function
#------------------------------------------------------------------------------

show_status() {
    echo ""
    echo "========================================"
    echo "Web Application Layer Status"
    echo "========================================"
    echo ""

    # Docker
    if check_docker 2>/dev/null; then
        echo -e "Docker:         ${GREEN}✓ RUNNING${NC}"
    else
        echo -e "Docker:         ${RED}✗ STOPPED${NC}"
    fi

    # MongoDB
    local mongo_status=$(get_mongo_status)
    case "$mongo_status" in
        "healthy")
            echo -e "MongoDB:        ${GREEN}✓ RUNNING${NC} (healthy)"
            ;;
        "starting")
            echo -e "MongoDB:        ${YELLOW}⟳ STARTING${NC}"
            ;;
        "unhealthy")
            echo -e "MongoDB:        ${RED}✗ UNHEALTHY${NC}"
            ;;
        "stopped")
            echo -e "MongoDB:        ${RED}✗ STOPPED${NC}"
            ;;
        "not_found")
            echo -e "MongoDB:        ${RED}✗ NOT FOUND${NC}"
            ;;
    esac

    # Python service
    if is_port_open $PYTHON_PORT; then
        echo -e "Python (8001):  ${GREEN}✓ RUNNING${NC}"
    else
        echo -e "Python (8001):  ${RED}✗ STOPPED${NC}"
    fi

    # Node.js service
    if is_port_open $NODE_PORT; then
        echo -e "Node.js (3000): ${GREEN}✓ RUNNING${NC}"
    else
        echo -e "Node.js (3000): ${RED}✗ STOPPED${NC}"
    fi

    echo ""
}

#------------------------------------------------------------------------------
# Main Commands
#------------------------------------------------------------------------------

start_all() {
    echo ""
    echo "========================================"
    echo "Starting Web Application Layer"
    echo "========================================"
    echo ""

    start_docker || exit 1
    start_mongodb || exit 1
    start_python_service || exit 1
    start_node_service || exit 1

    echo ""
    log_success "All web services started successfully!"
    show_status
}

stop_all() {
    echo ""
    echo "========================================"
    echo "Stopping Web Application Layer"
    echo "========================================"
    echo ""

    stop_node_service
    stop_python_service
    # Don't stop MongoDB by default as other services might need it

    echo ""
    log_success "Web services stopped (MongoDB left running)"
}

restart_all() {
    stop_all
    sleep 2
    start_all
}

#------------------------------------------------------------------------------
# Main Entry Point
#------------------------------------------------------------------------------

case "${1:-start}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    restart)
        restart_all
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        echo ""
        echo "  start   - Start all web services (default)"
        echo "  stop    - Stop Python and Node.js services"
        echo "  status  - Show status of all services"
        echo "  restart - Restart all web services"
        exit 1
        ;;
esac

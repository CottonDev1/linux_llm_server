#!/bin/bash

# LLM Start Menu - Interactive service launcher

LLAMA_SERVER="/data/projects/llama.cpp/build/bin/llama-server"
MODEL_DIR="/data/projects/llm_website/models/llamacpp"
EMBED_DIR="/data/models"
LOG_DIR="/data/logs/llama"
PID_DIR="/data/run"

mkdir -p "$LOG_DIR" "$PID_DIR"

# Service definitions: name|port|model|gpu|extra_args|model_dir_override
# GPU 0 = RTX 3050 (8GB), GPU 1 = GTX 1660 (6GB)
SERVICES=(
    "SQL|8080|sqlcoder-7b-2.Q4_K_M.gguf|0||"
    "General|8081|Qwen2.5-7B-Instruct-Q4_K_M.gguf|0||"
    "Code|8082|qwen2.5-coder-1.5b-instruct-q4_k_m.gguf|1||"
    "Embedding|8083|nomic-embed-text-v1.5-f16.gguf|1|--embedding|$EMBED_DIR"
)

is_running() {
    local port="$1"
    nc -z localhost $port 2>/dev/null
    return $?
}

start_service() {
    local name="$1"
    local port="$2"
    local model="$3"
    local gpu="$4"
    local extra="$5"
    local model_dir_override="$6"

    # Use override dir if provided, otherwise default MODEL_DIR
    local use_model_dir="${model_dir_override:-$MODEL_DIR}"

    if is_running $port; then
        echo "[$name] Already running on port $port"
        return 0
    fi

    if [ ! -f "$use_model_dir/$model" ]; then
        echo "[$name] ERROR: Model not found: $use_model_dir/$model"
        return 1
    fi

    echo "[$name] Starting on port $port (GPU $gpu with CUDA)..."
    echo "[$name] Model: $use_model_dir/$model"

    # Set CUDA device and start server with full GPU offload
    CUDA_VISIBLE_DEVICES=$gpu \
    $LLAMA_SERVER \
        -m "$use_model_dir/$model" \
        --host 0.0.0.0 \
        --port $port \
        --n-gpu-layers 99 \
        --ctx-size 4096 \
        --flash-attn \
        $extra \
        > "$LOG_DIR/${name,,}.log" 2>&1 &

    local pid=$!
    echo $pid > "$PID_DIR/${name,,}.pid"

    sleep 3
    if is_running $port; then
        echo "[$name] Started successfully (PID: $pid)"
    else
        echo "[$name] FAILED - check $LOG_DIR/${name,,}.log"
        tail -20 "$LOG_DIR/${name,,}.log"
    fi
}

stop_service() {
    local name="$1"
    local port="$2"

    local pid_file="$PID_DIR/${name,,}.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            echo "[$name] Stopping (PID: $pid)..."
            kill $pid 2>/dev/null
            sleep 2
            kill -9 $pid 2>/dev/null
        fi
        rm -f "$pid_file"
    fi

    # Also kill by port if still running
    if is_running $port; then
        fuser -k $port/tcp 2>/dev/null
    fi
    echo "[$name] Stopped"
}

show_menu() {
    echo ""
    echo "========================================"
    echo "LLM Service Manager"
    echo "========================================"
    echo ""
    printf "%-4s %-12s %-6s %-10s %-8s %s\n" "#" "SERVICE" "PORT" "STATUS" "GPU" "MODEL"
    echo "------------------------------------------------------------"

    local i=1
    for svc in "${SERVICES[@]}"; do
        IFS='|' read -r name port model gpu extra model_dir_override <<< "$svc"
        if is_running $port; then
            status="✓ RUNNING"
        else
            status="✗ STOPPED"
        fi
        printf "%-4s %-12s %-6s %-10s %-8s %s\n" "$i)" "$name" "$port" "$status" "GPU $gpu" "${model%.gguf}"
        ((i++))
    done

    echo ""
    echo "5)  Start ALL"
    echo "6)  Stop ALL"
    echo "7)  Restart ALL"
    echo "q)  Quit"
    echo ""
}

while true; do
    show_menu
    read -p "Choice: " choice

    case $choice in
        1|2|3|4)
            idx=$((choice - 1))
            IFS='|' read -r name port model gpu extra model_dir_override <<< "${SERVICES[$idx]}"
            if is_running $port; then
                read -p "[$name] is running. Stop it? (y/n): " yn
                if [[ $yn == "y" ]]; then
                    stop_service "$name" "$port"
                fi
            else
                start_service "$name" "$port" "$model" "$gpu" "$extra" "$model_dir_override"
            fi
            ;;
        5)
            echo "Starting all services with CUDA..."
            for svc in "${SERVICES[@]}"; do
                IFS='|' read -r name port model gpu extra model_dir_override <<< "$svc"
                start_service "$name" "$port" "$model" "$gpu" "$extra" "$model_dir_override"
            done
            ;;
        6)
            echo "Stopping all services..."
            for svc in "${SERVICES[@]}"; do
                IFS='|' read -r name port model gpu extra model_dir_override <<< "$svc"
                stop_service "$name" "$port"
            done
            ;;
        7)
            echo "Restarting all services..."
            for svc in "${SERVICES[@]}"; do
                IFS='|' read -r name port model gpu extra model_dir_override <<< "$svc"
                stop_service "$name" "$port"
            done
            sleep 2
            for svc in "${SERVICES[@]}"; do
                IFS='|' read -r name port model gpu extra model_dir_override <<< "$svc"
                start_service "$name" "$port" "$model" "$gpu" "$extra" "$model_dir_override"
            done
            ;;
        q|Q)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
done

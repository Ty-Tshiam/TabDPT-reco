#!/usr/bin/env bash
# ==============================================================================
# TabDPT Triton gRPC & Services Remote Management Script (Native Mode)
# ==============================================================================

set -e

ACTION="${1:-status}"
SESSION_NAME="triton"

# Dynamically resolve repository directory and log location
if [ -d "/workspace/TabDPT-reco" ]; then
    REPO_DIR="/workspace/TabDPT-reco"
    LOG_DIR="/workspace"
elif [ -d "$HOME/TabDPT-reco" ]; then
    REPO_DIR="$HOME/TabDPT-reco"
    LOG_DIR="$HOME"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    LOG_DIR="$REPO_DIR"
fi

LOG_FILE="$LOG_DIR/triton.log"
WORKER_LOG="$LOG_DIR/worker.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# Dynamically resolve Python interpreter
if [ -f "/venv/main/bin/python" ]; then
    PYTHON_BIN="/venv/main/bin/python"
elif [ -f "/opt/conda/bin/python" ]; then
    PYTHON_BIN="/opt/conda/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="python"
fi

case "$ACTION" in
    status)
        echo "=== [1/4] Service Process Status ==="
        for svc in "triton_grpc_server.py:Triton gRPC (:8001)" "event_processor.py:Streaming Worker" "streamlit.*app.py:Streamlit Frontend (:8501)"; do
            pattern="${svc%%:*}"
            label="${svc##*:}"
            if ps aux | grep -v grep | grep -E "$pattern" > /dev/null; then
                pid=$(ps aux | grep -v grep | grep -E "$pattern" | awk '{print $2}' | head -n1)
                echo "  [OK] $label is RUNNING (PID: $pid)"
            else
                echo "  [STOPPED] $label is NOT running"
            fi
        done

        if pgrep -x redis-server >/dev/null 2>&1; then
            echo "  [OK] Redis Server (:6379) is RUNNING"
        else
            echo "  [STOPPED] Redis Server is NOT running"
        fi

        echo -e "\n=== [2/4] GPU Status & Memory ==="
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "No GPU detected via nvidia-smi"

        echo -e "\n=== [3/4] Active tmux Sessions ==="
        tmux list-sessions 2>/dev/null || echo "No active tmux sessions"

        echo -e "\n=== [4/4] Recent Triton Logs ==="
        if [ -f "$LOG_FILE" ]; then
            tail -n 12 "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;

    restart|start)
        echo "Starting/Restarting Triton gRPC server..."
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        pkill -f "triton_grpc_server.py" 2>/dev/null || true
        rm -f "$LOG_FILE"
        sleep 1
        tmux new-session -d -s "$SESSION_NAME" \
            "cd $REPO_DIR && $PYTHON_BIN src/triton_grpc_server.py 2>&1 | tee $LOG_FILE"
        echo "Waiting for Triton to bind port 8001 (loading 5.6GB KV-cache)..."
        READY=0
        for i in $(seq 1 45); do
            if grep -q "Triton gRPC Inference Server listening on port 8001" "$LOG_FILE" 2>/dev/null; then
                READY=1
                break
            fi
            sleep 1
        done
        if [ "$READY" -eq 1 ]; then
            echo "[OK] Triton server is ready and listening on port 8001."
        else
            echo "[WARN] Server is initializing in background. Current log:"
            tail -n 10 "$LOG_FILE" 2>/dev/null || true
        fi
        ;;

    start_all)
        echo "Starting complete native stack (Redis + Triton + Worker + Frontend)..."
        
        # 1. Redis
        if ! pgrep -x redis-server >/dev/null 2>&1; then
            which redis-server >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq redis-server >/dev/null 2>&1 || true)
            service redis-server start 2>/dev/null || redis-server --daemonize yes 2>/dev/null || true
            echo "[OK] Redis server started"
        fi

        # 2. Triton gRPC
        tmux kill-session -t triton 2>/dev/null || true
        pkill -f "triton_grpc_server.py" 2>/dev/null || true
        rm -f "$LOG_FILE"
        tmux new-session -d -s triton \
            "cd $REPO_DIR && $PYTHON_BIN src/triton_grpc_server.py 2>&1 | tee $LOG_FILE"
        echo "[OK] Triton server launched in tmux session 'triton'"

        # 3. Streaming Worker
        tmux kill-session -t worker 2>/dev/null || true
        pkill -f "event_processor.py" 2>/dev/null || true
        tmux new-session -d -s worker \
            "cd $REPO_DIR && $PYTHON_BIN src/streaming/event_processor.py 2>&1 | tee $WORKER_LOG"
        echo "[OK] Worker launched in tmux session 'worker'"

        # 4. Streamlit Frontend
        tmux kill-session -t frontend 2>/dev/null || true
        pkill -f "streamlit.*app.py" 2>/dev/null || true
        tmux new-session -d -s frontend \
            "cd $REPO_DIR && $PYTHON_BIN -m streamlit run src/frontend/app.py --server.port=8501 --server.address=0.0.0.0 2>&1 | tee $FRONTEND_LOG"
        echo "[OK] Frontend launched in tmux session 'frontend'"

        echo -e "\nWaiting for Triton port 8001 binding..."
        READY=0
        for i in $(seq 1 45); do
            if grep -q "Triton gRPC Inference Server listening on port 8001" "$LOG_FILE" 2>/dev/null; then
                READY=1
                break
            fi
            sleep 1
        done
        if [ "$READY" -eq 1 ]; then
            echo "[OK] Triton is active on port 8001!"
        else
            echo "[WARN] Triton server is initializing in background. Current log:"
            tail -n 10 "$LOG_FILE" 2>/dev/null || true
        fi
        ;;

    stop|stop_all)
        echo "Stopping all services..."
        tmux kill-session -t triton 2>/dev/null || true
        tmux kill-session -t worker 2>/dev/null || true
        tmux kill-session -t frontend 2>/dev/null || true
        pkill -f "triton_grpc_server.py" 2>/dev/null || true
        pkill -f "event_processor.py" 2>/dev/null || true
        pkill -f "streamlit.*app.py" 2>/dev/null || true
        echo "All services stopped."
        ;;

    bootstrap)
        echo "=== [Bootstrap] Checking & Installing Python Dependencies ==="
        which tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1 || true)
        echo "[Bootstrap] Using Python: $PYTHON_BIN"
        $PYTHON_BIN -m pip install --quiet huggingface_hub safetensors scipy scikit-learn omegaconf 'tritonclient[grpc]' streamlit redis polars pyarrow 2>/dev/null || \
        $PYTHON_BIN -m pip install --quiet --break-system-packages huggingface_hub safetensors scipy scikit-learn omegaconf 'tritonclient[grpc]' streamlit redis polars pyarrow
        echo "[OK] Dependencies successfully installed."
        ;;

    build_model)
        echo "=== [Model] Building KV-cache & model artifact ==="
        cd "$REPO_DIR" && "$PYTHON_BIN" src/save_model.py
        ;;

    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;

    *)
        echo "Usage: $0 {status|restart|start|start_all|stop|stop_all|logs|bootstrap|build_model}"
        exit 1
        ;;
esac

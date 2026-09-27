#!/usr/bin/env bash
# ==============================================================================
# TabDPT Triton gRPC Server Remote Management Script
# ==============================================================================

set -e

ACTION="${1:-status}"
SESSION_NAME="triton"
REPO_DIR="/workspace/TabDPT-reco"
LOG_FILE="/workspace/triton.log"
PYTHON_BIN="/venv/main/bin/python"

case "$ACTION" in
    status)
        echo "=== [1/3] Triton Process Status ==="
        if ps aux | grep -v grep | grep "src/triton_grpc_server.py" > /dev/null; then
            echo "[OK] Triton gRPC server is RUNNING (Port 8001)"
            ps aux | grep -v grep | grep "src/triton_grpc_server.py" | awk '{print "  PID:", $2, "CPU%:", $3, "MEM%:", $4}'
        else
            echo "[STOPPED] Triton gRPC server is NOT running."
        fi

        echo -e "\n=== [2/3] GPU Status & Memory ==="
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

        echo -e "\n=== [3/3] Recent Server Logs ==="
        if [ -f "$LOG_FILE" ]; then
            tail -n 15 "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;

    restart)
        echo "Restarting Triton gRPC server..."
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        pkill -f "triton_grpc_server.py" 2>/dev/null || true
        sleep 2
        tmux new-session -d -s "$SESSION_NAME" \
            "cd $REPO_DIR && $PYTHON_BIN src/triton_grpc_server.py 2>&1 | tee $LOG_FILE"
        sleep 4
        echo "Server restarted. Current log:"
        tail -n 12 "$LOG_FILE"
        ;;

    stop)
        echo "Stopping Triton gRPC server..."
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        pkill -f "triton_grpc_server.py" 2>/dev/null || true
        echo "Server stopped."
        ;;

    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;

    *)
        echo "Usage: $0 {status|restart|stop|logs}"
        exit 1
        ;;
esac

#!/bin/bash
# GPU-buffer NIXL test: bypass CPU bounce on PD side
# Patches:
#   - embedding_transfer.py: dynamic descriptor allocation now uses GPU device
#   - start script: mem_fraction_static 0.85->0.78 (free ~30 GB headroom)
#   - start script: max-running-requests 64->32 (cap descriptor memory at ~5 GB)

set -u
LOG=/tmp/pdtp2_gpubuf_orchestrator.log
exec >> "$LOG" 2>&1

echo "=========================================="
echo "PD-TP=2 GPU-buf Orchestrator started: $(date)"
echo "=========================================="

cleanup_servers() {
    echo "[cleanup] killing servers..."
    PIDS=$(pgrep -f "dynamo.sglang|dynamo.frontend|nats-server -js -p 14222|etcd.*12379" 2>/dev/null | tr '\n' ' ')
    [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null
    sleep 8
    echo "[cleanup] done. GPU state:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 4,5,7
}

wait_for_server() {
    local max_wait=$1
    for i in $(seq 1 $max_wait); do
        out=$(curl -s --max-time 5 http://172.26.46.75:7001/v1/models 2>&1)
        if echo "$out" | grep -q "Qwen"; then
            echo "[wait] server ready after ${i}*15s = $((i*15))s"
            sleep 5
            return 0
        fi
        sleep 15
    done
    echo "[wait] TIMEOUT waiting for server"
    return 1
}

NAME="pdtp2_gpubuf"
START_SCRIPT="/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2_gpubuf.sh"
RESULT_BASE="/hongming/res11_pd_tp2_gpubuf/h200_disagg_pdtp2_gpubuf_v2_32b_image8_1080p_np64"

cleanup_servers

echo ""
echo "[$NAME] starting server at $(date)"
nohup bash "$START_SCRIPT" > "/tmp/${NAME}_start.log" 2>&1 &
echo "[$NAME] start script PID=$!"
disown

if ! wait_for_server 48; then
    echo "[$NAME] FAILED to start server"
    tail -50 "/tmp/${NAME}_start.log"
    tail -50 /hongming/dynamo/logs/pd_worker.log 2>/dev/null
    tail -50 /hongming/dynamo/logs/encoder_worker.log 2>/dev/null
    cleanup_servers
    exit 1
fi

echo "[$NAME] running rate=1.0 only (np=32) at $(date)"
cd /hongming/dynamo
RESULT_BASE="$RESULT_BASE" \
    bash test_sglang_32b_pd_tp2_gpubuf_1080p_np64_over_rates.sh 1.0 \
    > "/tmp/${NAME}_bench.log" 2>&1
echo "[$NAME] bench completed at $(date)"

LATEST=$(ls -dt "$RESULT_BASE"/test_sglang_multi_rates_1080p_* 2>/dev/null | head -1)
if [ -n "$LATEST" ] && [ -f "$LATEST/results_summary.csv" ]; then
    echo "[$NAME] results_summary.csv:"
    cat "$LATEST/results_summary.csv"
fi

cleanup_servers

echo ""
echo "=========================================="
echo "PD-TP=2 GPU-buf Orchestrator finished: $(date)"
echo "=========================================="

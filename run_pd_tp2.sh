#!/bin/bash
# Orchestrator: run 32B disagg PD-TP=2 sweep
# Starts server, runs 5-rate sweep, cleanly stops.
# Logs to /tmp/pdtp2_orchestrator.log

set -u
LOG=/tmp/pdtp2_orchestrator.log
exec >> "$LOG" 2>&1

echo "=========================================="
echo "PD-TP=2 Orchestrator started: $(date)"
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

NAME="pdtp2"
START_SCRIPT="/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2.sh"
RESULT_BASE="/hongming/res8_pd_tp2/h200_disagg_pdtp2_32b_image8_1080p_np64"

cleanup_servers

echo ""
echo "[$NAME] starting server at $(date)"
nohup bash "$START_SCRIPT" > "/tmp/${NAME}_start.log" 2>&1 &
echo "[$NAME] start script PID=$!"
disown

# 12 min for TP=2 PD + encoder to come up (TP=2 has NCCL init overhead)
if ! wait_for_server 48; then
    echo "[$NAME] FAILED to start server"
    tail -40 "/tmp/${NAME}_start.log"
    echo "--- pd_worker.log ---"
    tail -50 /hongming/dynamo/logs/pd_worker.log 2>/dev/null
    echo "--- encoder_worker.log ---"
    tail -50 /hongming/dynamo/logs/encoder_worker.log 2>/dev/null
    cleanup_servers
    exit 1
fi

echo "[$NAME] running 5-rate sweep at $(date)"
cd /hongming/dynamo
RESULT_BASE="$RESULT_BASE" \
    bash test_sglang_32b_pd_tp2_1080p_np64_over_rates.sh 0.1 0.25 0.5 1.0 1.25 \
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
echo "PD-TP=2 Orchestrator finished: $(date)"
echo "=========================================="

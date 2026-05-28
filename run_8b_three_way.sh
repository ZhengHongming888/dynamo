#!/bin/bash
# Orchestrator: run 8B 8img@1080p bench across TP=1 agg, TP=2 agg, disagg
# Logs to /tmp/8b_orchestrator.log

set -u
LOG=/tmp/8b_orchestrator.log
exec >> "$LOG" 2>&1

echo "=========================================="
echo "8B Orchestrator started: $(date)"
echo "=========================================="

cleanup_servers() {
    echo "[cleanup] killing servers..."
    PIDS=$(pgrep -f "dynamo.sglang|dynamo.frontend|nats-server -js -p 14222|etcd.*12379" 2>/dev/null | tr '\n' ' ')
    [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null
    sleep 8
    echo "[cleanup] done. GPU state:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 4,5
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

run_config() {
    local NAME=$1
    local START_SCRIPT=$2
    local RESULT_BASE=$3

    echo ""
    echo "=========================================="
    echo "[$NAME] starting at $(date)"
    echo "=========================================="

    cleanup_servers

    nohup bash "$START_SCRIPT" > "/tmp/8b_${NAME}_start.log" 2>&1 &
    echo "[$NAME] start script PID=$!"
    disown

    # 8B is faster to load; allow up to 12 minutes still (matching old disagg need)
    if ! wait_for_server 48; then
        echo "[$NAME] FAILED to start server, skipping"
        tail -40 "/tmp/8b_${NAME}_start.log"
        echo "--- worker logs ---"
        tail -30 /hongming/dynamo/logs/epd_worker_server.log 2>/dev/null
        tail -30 /hongming/dynamo/logs/pd_worker.log 2>/dev/null
        tail -30 /hongming/dynamo/logs/encoder_worker.log 2>/dev/null
        return 1
    fi

    echo "[$NAME] running 8B bench (5 rates, np=64) at $(date)"
    cd /hongming/dynamo
    RESULT_BASE="$RESULT_BASE" \
        bash test_sglang_8b_1080p_np64_over_rates.sh 0.1 0.25 0.5 1.0 1.25 \
        > "/tmp/8b_${NAME}_bench.log" 2>&1
    echo "[$NAME] bench completed at $(date)"

    LATEST=$(ls -dt "$RESULT_BASE"/test_sglang_multi_rates_1080p_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/results_summary.csv" ]; then
        echo "[$NAME] results_summary.csv:"
        cat "$LATEST/results_summary.csv"
    fi

    cleanup_servers
}

run_config "tp1" \
    "/hongming/dynamo/01_cuda_sh/agg_h200_8b/start_h200_aggregate_epd_server_8b_tp1.sh" \
    "/hongming/res7_8B/h200_agg_tp1_8b_image8_1080p_np64"

run_config "tp2" \
    "/hongming/dynamo/01_cuda_sh/agg_h200_8b/start_h200_aggregate_epd_server_8b_tp2.sh" \
    "/hongming/res7_8B/h200_agg_tp2_8b_image8_1080p_np64"

run_config "disagg" \
    "/hongming/dynamo/01_cuda_sh/disagg_h200_8b/start_disagg_h200_8b_combined.sh" \
    "/hongming/res7_8B/h200_disagg_8b_image8_1080p_np64"

echo ""
echo "=========================================="
echo "8B Orchestrator finished: $(date)"
echo "=========================================="

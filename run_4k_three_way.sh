#!/bin/bash
# Orchestrator: run 8img_4k bench across TP=1 agg, TP=2 agg, disagg
# Each config: start server -> wait -> bench all rates -> kill -> drain
# Logs to /tmp/4k_orchestrator.log

set -u
LOG=/tmp/4k_orchestrator.log
exec >> "$LOG" 2>&1

echo "=========================================="
echo "Orchestrator started: $(date)"
echo "=========================================="

# Common cleanup
cleanup_servers() {
    echo "[cleanup] killing servers..."
    PIDS=$(pgrep -f "dynamo.sglang|dynamo.frontend|nats-server -js -p 14222|etcd.*12379" 2>/dev/null | tr '\n' ' ')
    [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null
    sleep 8
    echo "[cleanup] done. GPU state:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 4,5
}

# Wait for /v1/models endpoint to return Qwen
wait_for_server() {
    local max_wait=$1
    for i in $(seq 1 $max_wait); do
        out=$(curl -s --max-time 5 http://172.26.46.75:7001/v1/models 2>&1)
        if echo "$out" | grep -q "Qwen"; then
            echo "[wait] server ready after ${i}*15s"
            sleep 5  # extra settle
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

    nohup bash "$START_SCRIPT" > "/tmp/4k_${NAME}_start.log" 2>&1 &
    echo "[$NAME] start script PID=$!"
    disown

    if ! wait_for_server 20; then
        echo "[$NAME] FAILED to start server, skipping"
        tail -30 "/tmp/4k_${NAME}_start.log"
        tail -30 /hongming/dynamo/logs/epd_worker_server.log 2>/dev/null
        return 1
    fi

    # Run bench
    echo "[$NAME] running 4K bench (5 rates, np=64) at $(date)"
    cd /hongming/dynamo
    RESULT_BASE="$RESULT_BASE" \
        bash test_sglang_8img_4k.sh 0.1 0.25 0.5 1.0 1.25 \
        > "/tmp/4k_${NAME}_bench.log" 2>&1
    echo "[$NAME] bench completed at $(date)"

    # Show results CSV
    LATEST=$(ls -dt "$RESULT_BASE"/test_sglang_multi_rates_1080p_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/results_summary.csv" ]; then
        echo "[$NAME] results_summary.csv:"
        cat "$LATEST/results_summary.csv"
    fi

    cleanup_servers
}

# Run all 3 configs
run_config "tp1" \
    "/hongming/dynamo/01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh" \
    "/hongming/res6_img8_4k/h200_agg_tp1_32b_image8_4k_np64"

run_config "tp2" \
    "/hongming/dynamo/01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh" \
    "/hongming/res6_img8_4k/h200_agg_tp2_32b_image8_4k_np64"

run_config "disagg" \
    "/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh" \
    "/hongming/res6_img8_4k/h200_disagg_32b_image8_4k_np64"

echo ""
echo "=========================================="
echo "Orchestrator finished: $(date)"
echo "=========================================="

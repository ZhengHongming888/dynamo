#!/bin/bash
# Start script for Dynamo SGLang Aggregated EPD Server - 32B FP8 model
# TWO TP=1 workers in data-parallel behind a single frontend KV router
# Each worker gets its own GPU (4 and 5) and its own internal ports.

set -e

# Configuration
export IP_LOCAL=172.26.46.75
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001

# Worker A
export CUDA_DEVICE_A=4
export KV_EVENT_PORT_A=22080
export SIDE_CHANNEL_PORT_A=20098

# Worker B
export CUDA_DEVICE_B=5
export KV_EVENT_PORT_B=22081
export SIDE_CHANNEL_PORT_B=20099

# Where the bench script will drop results (read by test_sglang_mult_rates_*_over_rates.sh)
export RESULT_BASE=/hongming/res4/h200_agg_2xtp1_32b_image8_1080p_np64_rates

# Model configuration
MODEL_PATH="/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8"
MAX_RUNNING_REQUESTS=25
MEM_FRACTION=0.88
export SGLANG_MM_FEATURE_CACHE_MB=2048

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Aggregated EPD Server"
echo "Model: Qwen3-VL-32B-Instruct-FP8"
echo "MODE: TWO TP=1 workers (data-parallel, single frontend)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - Worker A: GPU $CUDA_DEVICE_A  KV-evt:$KV_EVENT_PORT_A  side:$SIDE_CHANNEL_PORT_A"
echo "  - Worker B: GPU $CUDA_DEVICE_B  KV-evt:$KV_EVENT_PORT_B  side:$SIDE_CHANNEL_PORT_B"
echo "  - Each worker: TP=1, max_running=$MAX_RUNNING_REQUESTS, mem_frac=$MEM_FRACTION"
echo "  - NATS Port: $PORT_NATS"
echo "  - etcd Port: $PORT_ETCD"
echo "  - HTTP Port: $PORT_HTTP"
echo ""

# Step 1: Start NATS
echo "Starting NATS server on port $PORT_NATS..."
nats-server -js -p $PORT_NATS -m 18222 > "$LOG_DIR/nats_epd_server.log" 2>&1 &
NATS_PID=$!
sleep 2

if ! kill -0 $NATS_PID 2>/dev/null; then
    echo "❌ Failed to start NATS. Check $LOG_DIR/nats_epd_server.log"
    exit 1
fi
echo "✓ NATS started (PID: $NATS_PID)"

# Step 2: Start etcd
echo "Starting etcd on port $PORT_ETCD..."
rm -rf /tmp/etcd-epd-server-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://0.0.0.0:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-epd-server-$$ > "$LOG_DIR/etcd_epd_server.log" 2>&1 &
ETCD_PID=$!
sleep 3

if ! kill -0 $ETCD_PID 2>/dev/null; then
    echo "❌ Failed to start etcd. Check $LOG_DIR/etcd_epd_server.log"
    exit 1
fi
echo "✓ etcd started (PID: $ETCD_PID)"

# Step 3: Start Frontend
echo "Starting Dynamo frontend on port $PORT_HTTP..."
export ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD
export ETCD_LEASE_TTL=600
DYN_REQUEST_PLANE=tcp \
DYN_EVENT_PLANE=zmq \
SGLANG_LOG_LEVEL=info python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states > "$LOG_DIR/frontend_epd_server.log" 2>&1 &
FRONTEND_PID=$!
sleep 5

if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Failed to start frontend. Check $LOG_DIR/frontend_epd_server.log"
    exit 1
fi
echo "✓ Frontend started (PID: $FRONTEND_PID)"

# Step 4: Start Worker A on GPU $CUDA_DEVICE_A
echo "Starting SGLang Worker A on GPU $CUDA_DEVICE_A..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_A \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
DYN_VLLM_KV_EVENT_PORT=$KV_EVENT_PORT_A \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT_A \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=mlx5_0:1 \
UCX_MEMTYPE_CACHE=0 \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
PYTHONHASHSEED=0 \
python3 -m dynamo.sglang \
    --model $MODEL_PATH \
    --enable-multimodal \
    --enable-mm-global-cache \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --tensor-parallel-size 1 \
    --mem-fraction-static $MEM_FRACTION \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_A'","enable_kv_cache_events":true}' \
    >> "$LOG_DIR/epd_worker_A.log" 2>&1 &
WORKER_A_PID=$!
echo "Worker A starting (PID: $WORKER_A_PID, GPU $CUDA_DEVICE_A)..."

# Step 5: Start Worker B on GPU $CUDA_DEVICE_B (in parallel)
echo "Starting SGLang Worker B on GPU $CUDA_DEVICE_B..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_B \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
DYN_VLLM_KV_EVENT_PORT=$KV_EVENT_PORT_B \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT_B \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=mlx5_0:1 \
UCX_MEMTYPE_CACHE=0 \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
PYTHONHASHSEED=0 \
python3 -m dynamo.sglang \
    --model $MODEL_PATH \
    --enable-multimodal \
    --enable-mm-global-cache \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --tensor-parallel-size 1 \
    --mem-fraction-static $MEM_FRACTION \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_B'","enable_kv_cache_events":true}' \
    >> "$LOG_DIR/epd_worker_B.log" 2>&1 &
WORKER_B_PID=$!
echo "Worker B starting (PID: $WORKER_B_PID, GPU $CUDA_DEVICE_B)..."

echo "Both workers starting in parallel; this may take 3-5 minutes for model loading..."

# Monitor BOTH workers' startup
STARTUP_TIMEOUT=600
STARTUP_COUNT=0
A_READY=0
B_READY=0
while [ $STARTUP_COUNT -lt $STARTUP_TIMEOUT ]; do
    if ! kill -0 $WORKER_A_PID 2>/dev/null; then
        echo "❌ Worker A died during startup. Check $LOG_DIR/epd_worker_A.log"
        tail -30 "$LOG_DIR/epd_worker_A.log"
        exit 1
    fi
    if ! kill -0 $WORKER_B_PID 2>/dev/null; then
        echo "❌ Worker B died during startup. Check $LOG_DIR/epd_worker_B.log"
        tail -30 "$LOG_DIR/epd_worker_B.log"
        exit 1
    fi

    if [ $A_READY -eq 0 ] && grep -q "Registered endpoint" "$LOG_DIR/epd_worker_A.log" 2>/dev/null; then
        echo "✓ Worker A registered"
        A_READY=1
    fi
    if [ $B_READY -eq 0 ] && grep -q "Registered endpoint" "$LOG_DIR/epd_worker_B.log" 2>/dev/null; then
        echo "✓ Worker B registered"
        B_READY=1
    fi

    if [ $A_READY -eq 1 ] && [ $B_READY -eq 1 ]; then
        echo "✓ Both workers registered"
        break
    fi

    if [ $((STARTUP_COUNT % 15)) -eq 0 ]; then
        echo -n "."
    fi

    sleep 1
    STARTUP_COUNT=$((STARTUP_COUNT + 1))
done

if [ $STARTUP_COUNT -ge $STARTUP_TIMEOUT ]; then
    echo "❌ Worker startup timed out after $STARTUP_TIMEOUT seconds"
    echo "Worker A tail:"; tail -20 "$LOG_DIR/epd_worker_A.log"
    echo "Worker B tail:"; tail -20 "$LOG_DIR/epd_worker_B.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All Services Started Successfully"
echo "=========================================="
echo ""
echo "Service Endpoints:"
echo "  - Frontend HTTP: http://localhost:$PORT_HTTP"
echo "  - NATS: nats://$IP_LOCAL:$PORT_NATS"
echo "  - etcd: http://$IP_LOCAL:$PORT_ETCD"
echo ""
echo "Workers:"
echo "  - Worker A (GPU $CUDA_DEVICE_A) PID $WORKER_A_PID"
echo "  - Worker B (GPU $CUDA_DEVICE_B) PID $WORKER_B_PID"
echo "  - Frontend PID $FRONTEND_PID"
echo "  - NATS PID $NATS_PID"
echo "  - etcd PID $ETCD_PID"
echo ""
echo "Logs:"
echo "  - $LOG_DIR/epd_worker_A.log"
echo "  - $LOG_DIR/epd_worker_B.log"
echo "  - $LOG_DIR/frontend_epd_server.log"
echo ""
echo "=========================================="
echo "Two TP=1 workers ready for benchmarking!"
echo "=========================================="
echo ""
echo "To run benchmarks in another terminal:"
echo "  RESULT_BASE=$RESULT_BASE \\"
echo "    bash /hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh"
echo ""
echo "To stop:"
echo "  kill $NATS_PID $ETCD_PID $FRONTEND_PID $WORKER_A_PID $WORKER_B_PID"
echo "  # or: pkill -f 'nats-server|etcd.*12379|dynamo.frontend|dynamo.sglang'"
echo ""

#!/bin/bash
# Start script for Dynamo SGLang Aggregated EPD Server - 8B model (Qwen3-VL-8B-Instruct, bf16)
# This starts: NATS, etcd, frontend, and aggregated EPD worker (all-in-one)
# TP=1 single GPU on device 4

set -e

# Configuration
export IP_LOCAL=172.26.46.75
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001
export CUDA_DEVICE=4
export KV_EVENT_PORT=22080
export SIDE_CHANNEL_PORT=20098

# Increase TCP request plane message size limit and HTTP body limit for large multimodal payloads
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB
export DYN_HTTP_BODY_LIMIT_MB=256          # frontend HTTP body limit (default 45 MB)

# Where the bench script will drop results
export RESULT_BASE=/hongming/res7_8B/h200_agg_tp1_8b_image8_1080p_np64_rates

# Model configuration
MODEL_PATH="/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct"
MAX_RUNNING_REQUESTS=64
MEM_FRACTION=0.85

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Aggregated EPD Server"
echo "Model: Qwen3-VL-8B-Instruct (bf16)"
echo "TP=1, GPU $CUDA_DEVICE"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - CUDA Device: $CUDA_DEVICE"
echo "  - Tensor Parallel: 1"
echo "  - Max Running Requests: $MAX_RUNNING_REQUESTS"
echo "  - Memory Fraction: $MEM_FRACTION"
echo "  - NATS Port: $PORT_NATS"
echo "  - etcd Port: $PORT_ETCD"
echo "  - HTTP Port: $PORT_HTTP"
echo "  - KV Event Port: $KV_EVENT_PORT"
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
sleep 8

# Wait for etcd to be reachable
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s -o /dev/null "http://localhost:$PORT_ETCD/version"; then
        echo "  etcd reachable on attempt $i"
        break
    fi
    sleep 2
done

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

# Step 4: Start SGLang Aggregated EPD
echo "Starting SGLang Aggregated EPD worker..."
echo "This may take 1-3 minutes for model loading (8B bf16)..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
DYN_VLLM_KV_EVENT_PORT=$KV_EVENT_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=mlx5_0:1 \
UCX_MEMTYPE_CACHE=0 \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
PYTHONHASHSEED=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,P2P \
python3 -m dynamo.sglang \
    --model $MODEL_PATH \
    --enable-multimodal \
    --enable-mm-global-cache \
    --dtype auto \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --tensor-parallel-size 1 \
    --mem-fraction-static $MEM_FRACTION \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    >> "$LOG_DIR/epd_worker_server.log" 2>&1 &
EPD_PID=$!

echo "EPD worker starting (PID: $EPD_PID)..."

# Monitor EPD startup
STARTUP_TIMEOUT=300
STARTUP_COUNT=0
while [ $STARTUP_COUNT -lt $STARTUP_TIMEOUT ]; do
    if ! kill -0 $EPD_PID 2>/dev/null; then
        echo "❌ EPD process died during startup. Check $LOG_DIR/epd_worker_server.log"
        tail -50 "$LOG_DIR/epd_worker_server.log"
        exit 1
    fi

    if grep -q "Registered endpoint" "$LOG_DIR/epd_worker_server.log" 2>/dev/null; then
        echo "✓ EPD worker initialized successfully"
        break
    fi

    if [ $((STARTUP_COUNT % 10)) -eq 0 ]; then
        echo -n "."
    fi

    sleep 1
    STARTUP_COUNT=$((STARTUP_COUNT + 1))
done

if [ $STARTUP_COUNT -ge $STARTUP_TIMEOUT ]; then
    echo "❌ EPD startup timed out after $STARTUP_TIMEOUT seconds"
    tail -50 "$LOG_DIR/epd_worker_server.log"
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
echo "Worker Configuration:"
echo "  - Model: Qwen3-VL-8B-Instruct"
echo "  - Mode: Aggregated EPD (all-in-one)"
echo "  - CUDA Device: $CUDA_DEVICE"
echo "  - Tensor Parallel: 1"
echo "  - Max Requests: $MAX_RUNNING_REQUESTS"
echo ""
echo "Process PIDs:"
echo "  - NATS: $NATS_PID"
echo "  - etcd: $ETCD_PID"
echo "  - Frontend: $FRONTEND_PID"
echo "  - EPD Worker: $EPD_PID"
echo ""
echo "Logs:"
echo "  - EPD Worker: $LOG_DIR/epd_worker_server.log"
echo ""
echo "To run benchmarks in another terminal:"
echo "  RESULT_BASE=$RESULT_BASE \\"
echo "    bash /hongming/dynamo/test_sglang_8b_1080p_np64_over_rates.sh"
echo ""
echo "To stop all services:"
echo "  pkill -f 'nats-server|etcd.*12379|dynamo.frontend|dynamo.sglang'"
echo ""

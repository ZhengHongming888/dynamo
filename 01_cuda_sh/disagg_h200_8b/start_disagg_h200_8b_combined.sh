#!/bin/bash
# Combined start script for Dynamo SGLang Disaggregated Prefill/Decode - 8B model
# Starts: NATS, etcd, frontend, PD worker (GPU 5), Encoder worker (GPU 4)
# Model: Qwen3-VL-8B-Instruct (bf16)

set -e

# ========================================
# CONFIGURATION
# ========================================

export IP_LOCAL=172.26.46.75
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001

# GPU assignments
export CUDA_DEVICE_PD=5      # PD worker on GPU 5
export CUDA_DEVICE_ENCODER=4 # Encoder on GPU 4

# Network settings
export KV_EVENT_PORT_PD=22091
export KV_EVENT_PORT_ENCODER=22090
export SIDE_CHANNEL_PORT=20098

# Increase TCP and HTTP body limits
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB
export DYN_HTTP_BODY_LIMIT_MB=256          # 256 MB
export UCX_NIC=mlx5_4:1

export RESULT_BASE=/hongming/res7_8B/h200_disagg_8b_image8_1080p_np64_rates

# Model path
MODEL_PATH="/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-8B-Instruct"
MODEL_NAME_GREP="Qwen3-VL-8B"

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Disaggregated H200 Setup"
echo "Model: Qwen3-VL-8B-Instruct (bf16)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - PD Worker: GPU $CUDA_DEVICE_PD"
echo "  - Encoder: GPU $CUDA_DEVICE_ENCODER"
echo "  - Server IP: $IP_LOCAL"
echo ""

# ========================================
# Start Infrastructure
# ========================================

echo "Starting NATS..."
nats-server -js -p $PORT_NATS -m 18222 > "$LOG_DIR/nats.log" 2>&1 &
NATS_PID=$!
sleep 2

echo "Starting etcd..."
etcd --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
     --advertise-client-urls=http://0.0.0.0:$PORT_ETCD \
     --listen-peer-urls=http://0.0.0.0:12380 \
     --initial-advertise-peer-urls=http://0.0.0.0:12380 \
     --initial-cluster=default=http://0.0.0.0:12380 \
     --data-dir=/tmp/etcd-disagg-h200-8b-$$ \
     > "$LOG_DIR/etcd.log" 2>&1 &
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

echo "Starting Frontend..."
ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
DYN_EVENT_PLANE=zmq \
DYN_LOG=debug \
SGLANG_LOG_LEVEL=debug \
python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states \
    > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
sleep 5

# ========================================
# Start PD Worker (GPU 5)
# ========================================

echo "Starting PD Worker on GPU $CUDA_DEVICE_PD..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_PD \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
DYN_LOG=debug \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT_PD} \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
UCX_MEMTYPE_CACHE=0 \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,P2P \
python3 -m dynamo.sglang \
    --model $MODEL_PATH \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --max-running-requests 64 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --page-size 16 \
    --chunked-prefill-size 16384 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_PD'","enable_kv_cache_events":true}' \
    > "$LOG_DIR/pd_worker.log" 2>&1 &
PD_PID=$!

echo "  PD Worker starting (PID: $PD_PID)..."

# Wait a bit before starting encoder
sleep 10

# ========================================
# Start Encoder Worker (GPU 4)
# ========================================

echo "Starting Encoder Worker on GPU $CUDA_DEVICE_ENCODER..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ENCODER \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
DYN_LOG=debug \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT_ENCODER} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
NIXL_MAX_BUFFER_SIZE=805306368 \
NIXL_BUFFER_COUNT=256 \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
ZMQ_SNDHWM=0 \
ZMQ_RCVHWM=0 \
python3 -m dynamo.sglang \
    --model $MODEL_PATH \
    --enable-multimodal \
    --multimodal-encode-worker \
    --multimodal-embedding-cache-capacity-gb 16 \
    --chat-template qwen2-vl \
    --dtype auto \
    --mem-fraction-static 0.85 \
    --page-size 16 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_ENCODER'","enable_kv_cache_events":true}' \
    > "$LOG_DIR/encoder_worker.log" 2>&1 &
ENCODER_PID=$!

echo "  Encoder Worker starting (PID: $ENCODER_PID)..."

# ========================================
# Display Status
# ========================================

echo ""
echo "=========================================="
echo "All Services Started"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  - NATS:       $NATS_PID"
echo "  - etcd:       $ETCD_PID"
echo "  - Frontend:   $FRONTEND_PID"
echo "  - PD Worker:  $PD_PID"
echo "  - Encoder:    $ENCODER_PID"
echo ""
echo "Services:"
echo "  - NATS:     nats://$IP_LOCAL:$PORT_NATS"
echo "  - etcd:     http://$IP_LOCAL:$PORT_ETCD"
echo "  - Frontend: http://localhost:$PORT_HTTP"
echo ""
echo "GPUs:"
echo "  - PD Worker: GPU $CUDA_DEVICE_PD"
echo "  - Encoder:   GPU $CUDA_DEVICE_ENCODER"
echo ""

# Wait for model to be ready (8B is faster, but we still allow 10 min for safety)
echo "Checking model registration..."
for i in {1..120}; do
    sleep 5
    if curl -s http://localhost:$PORT_HTTP/v1/models 2>/dev/null | grep -q "$MODEL_NAME_GREP"; then
        echo ""
        echo "✓ Model registered successfully after $((i*5))s"
        echo ""
        echo "Ready for inference."
        exit 0
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "  Waiting... ($((i*5))s elapsed)"
    fi
done

echo ""
echo "⚠ Timeout waiting for model registration after 600s."
echo "Check logs:"
echo "  tail -50 $LOG_DIR/pd_worker.log"
echo "  tail -50 $LOG_DIR/encoder_worker.log"
echo ""
exit 1

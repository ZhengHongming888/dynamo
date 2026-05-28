#!/bin/bash
# Combined start script for Dynamo SGLang Disaggregated Prefill/Decode
# This starts: NATS, etcd, frontend, PD worker (GPU 1), and Encoder worker (GPU 0)
# Model: Qwen3-VL-32B-Instruct-FP8

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

# Increase TCP request plane message size limit for large multimodal payloads
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB
export DYN_HTTP_BODY_LIMIT_MB=256          # frontend HTTP body limit (default 45 MB)
# NIC selection for UCX (NUMA-affined to GPUs 4 & 5 on NUMA 2):
#   mlx5_0  PIX→GPU0 (NUMA 0)  - WRONG for GPUs 4,5 (cross-socket SYS)
#   mlx5_4  PIX→GPU4 (NUMA 2)  - active, correct for both GPU 4 and GPU 5 (NODE to GPU 5)
#   mlx5_5  PIX→GPU5 (NUMA 2)  - currently PORT_DOWN, can't use
# Note: UCX_NIC is not a real UCX env var; UCX uses UCX_NET_DEVICES. This is just a script alias.
# For same-host disagg this is irrelevant anyway (NIXL uses cuda_ipc over NVLink).
export UCX_NIC=mlx5_4:1

# Where the bench script will drop results (read by test_sglang_mult_rates_*_over_rates.sh)
# nixlread_AB: nixlread + max_running=64 + chunked_prefill=16384
export RESULT_BASE=/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates_nixlread_AB

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Disaggregated H200 Setup"
echo "Model: Qwen3-VL-32B-Instruct-FP8"
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
     --data-dir=/tmp/etcd-disagg-h200-32b-$$ \
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
# Start PD Worker (GPU 1)
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
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
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
echo "  This will take 2-3 minutes for model loading..."

# Wait a bit for PD worker to start initializing
sleep 10

# ========================================
# Start Encoder Worker (GPU 0)
# ========================================

echo "Starting Encoder Worker on GPU $CUDA_DEVICE_ENCODER..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ENCODER \
NATS_SERVER=nats://${IP_REMOTE:-${IP_LOCAL}}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE:-${IP_LOCAL}}:${PORT_ETCD} \
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
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --multimodal-encode-worker \
    --multimodal-embedding-cache-capacity-gb 16 \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --page-size 16 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_ENCODER'","enable_kv_cache_events":true}' \
    > "$LOG_DIR/encoder_worker.log" 2>&1 &
ENCODER_PID=$!

echo "  Encoder Worker starting (PID: $ENCODER_PID)..."
echo "  This will take 1-2 minutes for model loading..."

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
echo "  - PD Worker: GPU $CUDA_DEVICE_PD (decode/generation)"
echo "  - Encoder:   GPU $CUDA_DEVICE_ENCODER (vision encoding)"
echo ""
echo "Logs (with DYN_LOG=debug):"
echo "  - NATS:       $LOG_DIR/nats.log"
echo "  - etcd:       $LOG_DIR/etcd.log"
echo "  - Frontend:   $LOG_DIR/frontend.log"
echo "  - PD Worker:  $LOG_DIR/pd_worker.log"
echo "  - Encoder:    $LOG_DIR/encoder_worker.log"
echo ""
echo "=========================================="
echo "Monitoring Commands:"
echo "=========================================="
echo ""
echo "Watch PD Worker loading:"
echo "  tail -f $LOG_DIR/pd_worker.log | grep -E 'warmup|Registered|Ready'"
echo ""
echo "Watch Encoder loading:"
echo "  tail -f $LOG_DIR/encoder_worker.log | grep -E 'registered|Ready|init finish'"
echo ""
echo "Check when both workers are ready:"
echo "  curl -s http://localhost:$PORT_HTTP/v1/models | python3 -m json.tool"
echo ""
echo "Test with image:"
echo "  curl -s http://localhost:$PORT_HTTP/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8\", \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Describe this image.\"}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"http://images.cocodataset.org/val2017/000000039769.jpg\"}}]}], \"max_tokens\": 50}'"
echo ""
echo "=========================================="
echo "To stop all services:"
echo "=========================================="
echo ""
echo "  kill $NATS_PID $ETCD_PID $FRONTEND_PID $PD_PID $ENCODER_PID"
echo "  # or: pkill -f 'nats-server|etcd|dynamo.frontend|dynamo.sglang'"
echo ""
echo "=========================================="
echo "To run benchmarks in another terminal:"
echo "=========================================="
echo ""
echo "  RESULT_BASE=$RESULT_BASE \\"
echo "    bash /hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh"
echo ""
echo "=========================================="
echo "Waiting for workers to load (this takes 3-5 minutes)..."
echo "=========================================="
echo ""

# Wait for model to be ready
echo "Checking model registration..."
for i in {1..60}; do
    sleep 5
    if curl -s http://localhost:$PORT_HTTP/v1/models 2>/dev/null | grep -q "Qwen3-VL-32B"; then
        echo ""
        echo "✓ Model registered successfully!"
        echo ""
        echo "Ready for inference. Services are running in background."
        echo "Check logs in: $LOG_DIR/"
        exit 0
    fi
    echo "  Waiting... ($((i*5))s elapsed)"
done

echo ""
echo "⚠ Timeout waiting for model registration."
echo "Check logs for errors:"
echo "  tail -50 $LOG_DIR/pd_worker.log"
echo "  tail -50 $LOG_DIR/encoder_worker.log"
echo ""

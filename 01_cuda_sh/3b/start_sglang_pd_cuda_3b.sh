#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (CUDA side)
# Model: Qwen2.5-VL-3B-Instruct (smaller, faster)
# This starts: NATS, etcd, frontend, and decode worker with multimodal support
# Encode worker should run separately on XPU machine

set -e

export IP_LOCAL=172.26.46.162
export PORT_NATS=14222  # Same as 32B model
export PORT_ETCD=12379  # Same as 32B model
export PORT_HTTP=7001   # Same as 32B model
export CUDA_DEVICE=1
export KV_EVENT_PORT=22081  # Same as 32B PD model
export SIDE_CHANNEL_PORT=20098  # Same as 32B model

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Prefill/Decode (CUDA)"
echo "Model: Qwen2.5-VL-3B-Instruct"
echo "=========================================="
echo ""

# Start NATS
echo "Starting NATS..."
nats-server -js -p $PORT_NATS -m 18222 > "$LOG_DIR/nats_pd_3b.log" 2>&1 &
sleep 2

# Start etcd
echo "Starting etcd..."
rm -rf /tmp/etcd-sglang-pd-3b-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://0.0.0.0:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-sglang-pd-3b-$$ > "$LOG_DIR/etcd_pd_3b.log" 2>&1 &
sleep 3

# Start Frontend
echo "Starting Frontend..."
export ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD
DYN_REQUEST_PLANE=tcp \
SGLANG_LOG_LEVEL=debug python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states > "$LOG_DIR/frontend_pd_3b.log" 2>&1 &
sleep 5

# Start Decode Worker (Prefill/Decode with multimodal worker)
echo "Starting SGLang Decode Worker (Prefill/Decode with multimodal)..."
echo "Model: Qwen2.5-VL-3B-Instruct (faster than 32B)..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
DYN_VLLM_KV_EVENT_PORT=$KV_EVENT_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=mlx5_0:1 \
UCX_MEMTYPE_CACHE=0 \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
PYTHONHASHSEED=0 \
python3 -m dynamo.sglang \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --kv-cache-dtype auto \
    --max-running-requests 80 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    >> "$LOG_DIR/decode_worker_3b.log" 2>&1 &

echo ""
echo "Services starting in background..."
echo "=========================================="
echo "CUDA Side (Decode Worker + Infrastructure):"
echo "  - Model: Qwen2.5-VL-3B-Instruct"
echo "  - NATS: nats://$IP_LOCAL:$PORT_NATS"
echo "  - etcd: http://$IP_LOCAL:$PORT_ETCD"
echo "  - Frontend: http://localhost:$PORT_HTTP"
echo "  - Decode Worker: Loading 3B model (multimodal-worker mode)"
echo "  - KV Events Port: $KV_EVENT_PORT"
echo ""
echo "Optimizations for 3B model:"
echo "  • Max requests: 80 (vs 40 for 32B)"
echo "  • Memory fraction: 0.85 (vs 0.95 for 32B)"
echo "  • Faster load time: ~2 min (vs ~3 min)"
echo ""
echo "Note: Uses same ports as 32B model"
echo "  • Cannot run 32B and 3B simultaneously"
echo ""
echo "Logs:"
echo "  - NATS: $LOG_DIR/nats_pd_3b.log"
echo "  - etcd: $LOG_DIR/etcd_pd_3b.log"
echo "  - Frontend: $LOG_DIR/frontend_pd_3b.log"
echo "  - Decode Worker: $LOG_DIR/decode_worker_3b.log"
echo ""
echo "=========================================="
echo "IMPORTANT: Start encode worker on XPU machine separately"
echo "Use: ./start_sglang_pd_xpu_intel_3b.sh"
echo "=========================================="
echo ""
echo "To monitor decode worker startup:"
echo "  tail -f $LOG_DIR/decode_worker_3b.log"
echo ""
echo "To check when ready:"
echo "  grep 'Model registration succeeded' $LOG_DIR/decode_worker_3b.log"
echo ""
echo "To stop all:"
echo "  pkill -f 'nats-server|etcd.*12379|dynamo.frontend|dynamo.sglang'"

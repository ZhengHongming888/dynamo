#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (CUDA side)
# This starts: NATS, etcd, frontend, and decode worker with multimodal support
# Encode worker should run separately on XPU machine

set -e

export IP_LOCAL=172.26.46.162
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001
export CUDA_DEVICE=2
export KV_EVENT_PORT=22081  # Different port from aggregated mode
export SIDE_CHANNEL_PORT=20098

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Prefill/Decode (CUDA)"
echo "=========================================="
echo ""

# Start NATS
echo "Starting NATS..."
nats-server -js -p $PORT_NATS -m 18222 > "$LOG_DIR/nats_pd.log" 2>&1 &
sleep 2

# Start etcd
echo "Starting etcd..."
rm -rf /tmp/etcd-sglang-pd-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://0.0.0.0:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-sglang-pd-$$ > "$LOG_DIR/etcd_pd.log" 2>&1 &
sleep 3

# Start Frontend
echo "Starting Frontend..."
export ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD
DYN_REQUEST_PLANE=tcp \
SGLANG_LOG_LEVEL=debug python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states > "$LOG_DIR/frontend_pd.log" 2>&1 &
sleep 5

# Start Decode Worker (Prefill/Decode with multimodal worker)
echo "Starting SGLang Decode Worker (Prefill/Decode with multimodal)..."
echo "This takes a few minutes for model loading..."
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
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 40 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.95 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    >> "$LOG_DIR/decode_worker.log" 2>&1 &

echo ""
echo "Services starting in background..."
echo "=========================================="
echo "CUDA Side (Decode Worker + Infrastructure):"
echo "  - NATS: nats://$IP_LOCAL:$PORT_NATS"
echo "  - etcd: http://$IP_LOCAL:$PORT_ETCD"
echo "  - Frontend: http://localhost:$PORT_HTTP"
echo "  - Decode Worker: Loading model (multimodal-worker mode)"
echo "  - KV Events Port: $KV_EVENT_PORT"
echo ""
echo "Logs:"
echo "  - NATS: $LOG_DIR/nats_pd.log"
echo "  - etcd: $LOG_DIR/etcd_pd.log"
echo "  - Frontend: $LOG_DIR/frontend_pd.log"
echo "  - Decode Worker: $LOG_DIR/decode_worker.log"
echo ""
echo "=========================================="
echo "IMPORTANT: Start encode worker on XPU machine separately"
echo "=========================================="
echo ""
echo "To monitor decode worker startup:"
echo "  tail -f $LOG_DIR/decode_worker.log"
echo ""
echo "To check when ready:"
echo "  grep 'Model registration succeeded' $LOG_DIR/decode_worker.log"
echo ""
echo "To stop all:"
echo "  pkill -f 'nats-server|etcd.*$PORT_ETCD|dynamo.frontend|dynamo.sglang'"

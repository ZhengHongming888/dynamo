#!/bin/bash
# Quick start script for Dynamo SGLang services (keeps running)

set -e

export IP_LOCAL=172.26.46.162
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001
export CUDA_DEVICE=2
export KV_EVENT_PORT=22080
export SIDE_CHANNEL_PORT=20098

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

# Start NATS
echo "Starting NATS..."
nats-server -js -p $PORT_NATS -m 18222 > "$LOG_DIR/nats.log" 2>&1 &
sleep 2

# Start etcd
echo "Starting etcd..."
rm -rf /tmp/etcd-sglang-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://0.0.0.0:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-sglang-$$ > "$LOG_DIR/etcd.log" 2>&1 &
sleep 3

# Start Frontend
echo "Starting Frontend..."
export ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD
DYN_REQUEST_PLANE=tcp \
SGLANG_LOG_LEVEL=debug python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states > "$LOG_DIR/frontend.log" 2>&1 &
sleep 5

# Start EPD
echo "Starting SGLang EPD (this takes a few minutes)..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
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
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 40 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.95 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    >> "$LOG_DIR/epd.log" 2>&1 &

echo ""
echo "Services starting in background..."
echo "  - NATS: nats://$IP_LOCAL:$PORT_NATS"
echo "  - etcd: http://$IP_LOCAL:$PORT_ETCD"
echo "  - Frontend: http://localhost:$PORT_HTTP"
echo "  - EPD: Loading model (check logs/epd.log)"
echo ""
echo "To monitor EPD startup: tail -f $LOG_DIR/epd.log"
echo "To stop all: pkill -f 'nats-server|etcd.*12379|dynamo.frontend|dynamo.sglang'"

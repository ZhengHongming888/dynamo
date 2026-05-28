#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (H200 CUDA Encode Worker)
# This starts: Encode worker on H200 CUDA for multimodal processing
# Run this AFTER starting the decode side (start_sglang_pd_cuda_3b.sh)

set -e

# ========================================
# CONFIGURATION - UPDATE THESE VALUES
# ========================================

# Remote server (where NATS/etcd/frontend/decode are running)
export IP_REMOTE=172.26.46.162  # Decode server IP (can be same server)
export PORT_NATS=14222
export PORT_ETCD=12379

# Local H200 server
export IP_LOCAL=$(hostname -I | awk '{print $1}')  # Auto-detect or set manually
# export IP_LOCAL=172.26.46.162  # Uncomment to set manually if same server

# CUDA Device
export CUDA_DEVICE=0  # H200 CUDA device ID (different from decode worker)

# Network and Transfer Settings
export SIDE_CHANNEL_PORT=20098  # NIXL side channel port
export KV_EVENT_PORT=22080      # KV events port
export UCX_NIC=mlx5_0:1         # Network device (InfiniBand)

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang H200 CUDA Encode Worker"
echo "=========================================="
echo ""
echo "Remote Decode server: $IP_REMOTE"
echo "Local H200 server:    $IP_LOCAL"
echo "CUDA Device:          $CUDA_DEVICE"
echo ""

# Verify remote server is reachable
echo "Checking connectivity to decode server..."
if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/$IP_REMOTE/$PORT_NATS" 2>/dev/null; then
    echo "WARNING: Cannot reach NATS at $IP_REMOTE:$PORT_NATS"
    echo "Make sure decode side is running: ./start_sglang_pd_cuda_3b.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start H200 CUDA Encode Worker
echo "Starting H200 CUDA Encode Worker..."
echo "This takes a few minutes for model loading..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
python3 -m dynamo.sglang \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --enable-multimodal \
    --multimodal-encode-worker \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.7 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_DIR/encode_h200.log" &

ENCODE_PID=$!

echo ""
echo "=========================================="
echo "H200 CUDA Encode Worker Started"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Remote Decode Server: $IP_REMOTE"
echo "  - NATS: nats://$IP_REMOTE:$PORT_NATS"
echo "  - etcd: http://$IP_REMOTE:$PORT_ETCD"
echo "  - Local H200 IP: $IP_LOCAL"
echo "  - CUDA Device: $CUDA_DEVICE"
echo "  - Side Channel: $IP_LOCAL:$SIDE_CHANNEL_PORT"
echo "  - KV Events Port: $KV_EVENT_PORT"
echo "  - UCX Network: $UCX_NIC"
echo "  - Process PID: $ENCODE_PID"
echo ""
echo "H200 CUDA Optimizations:"
echo "  ✓ CUDA_VISIBLE_DEVICES (GPU selection)"
echo "  ✓ UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy (InfiniBand + CUDA)"
echo "  ✓ DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read"
echo "  ✓ dtype=auto, kv-cache-dtype=auto"
echo ""
echo "Log:"
echo "  - $LOG_DIR/encode_h200.log"
echo ""
echo "=========================================="
echo ""
echo "To monitor encode worker:"
echo "  tail -f $LOG_DIR/encode_h200.log"
echo ""
echo "To check when ready:"
echo "  grep -i 'registered\\|ready' $LOG_DIR/encode_h200.log"
echo ""
echo "To stop:"
echo "  kill $ENCODE_PID"
echo "  # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""

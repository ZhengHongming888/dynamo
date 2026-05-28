#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (Intel XPU Encode Worker)
# This starts: Encode worker on Intel XPU for multimodal processing
# Run this AFTER starting the CUDA side (start_sglang_pd_cuda.sh)

set -e

# ========================================
# CONFIGURATION - UPDATE THESE VALUES
# ========================================

# Remote CUDA server (where NATS/etcd/frontend/decode are running)
export IP_REMOTE=172.26.46.162  # CUDA server IP
export PORT_NATS=14222
export PORT_ETCD=12379

# Local XPU server
export IP_LOCAL=$(hostname -I | awk '{print $1}')  # Auto-detect or set manually
# export IP_LOCAL=<your-xpu-server-ip>  # Uncomment to set manually

# XPU Device
export XPU_DEVICE=0  # Intel XPU device ID (for ZE_AFFINITY_MASK)

# Network and Transfer Settings
export SIDE_CHANNEL_PORT=22098  # NIXL side channel port
export KV_EVENT_PORT=22080      # KV events port
export UCX_NIC=mlx5_1:1         # Network device (may differ per server)

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Intel XPU Encode Worker"
echo "=========================================="
echo ""
echo "Remote CUDA server: $IP_REMOTE"
echo "Local XPU server:   $IP_LOCAL"
echo "XPU Device:         $XPU_DEVICE"
echo ""

# Verify remote server is reachable
echo "Checking connectivity to CUDA server..."
if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/$IP_REMOTE/$PORT_NATS" 2>/dev/null; then
    echo "WARNING: Cannot reach NATS at $IP_REMOTE:$PORT_NATS"
    echo "Make sure CUDA side is running: ./start_sglang_pd_cuda.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Intel XPU Encode Worker
echo "Starting Intel XPU Encode Worker..."
echo "This takes a few minutes for model loading..."

ZE_AFFINITY_MASK=$XPU_DEVICE \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=ze_copy,rc,tcp \
UCX_NET_DEVICES=${UCX_NIC} \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
VISION_ENCODE_SERIALIZE=1 \
NIXL_USE_CPU_HOST_MEMORY=1 \
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --multimodal-encode-worker \
    --chat-template qwen2-vl \
    --dtype bfloat16 \
    --mem-fraction-static 0.95 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_DIR/encode_xpu.log" &

ENCODE_PID=$!

echo ""
echo "=========================================="
echo "Intel XPU Encode Worker Started"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Remote CUDA Server: $IP_REMOTE"
echo "  - NATS: nats://$IP_REMOTE:$PORT_NATS"
echo "  - etcd: http://$IP_REMOTE:$PORT_ETCD"
echo "  - Local XPU IP: $IP_LOCAL"
echo "  - XPU Device (ZE_AFFINITY_MASK): $XPU_DEVICE"
echo "  - Side Channel: $IP_LOCAL:$SIDE_CHANNEL_PORT"
echo "  - KV Events Port: $KV_EVENT_PORT"
echo "  - UCX Network: $UCX_NIC"
echo "  - Process PID: $ENCODE_PID"
echo ""
echo "Intel XPU Optimizations:"
echo "  ✓ ZE_AFFINITY_MASK (Level Zero device selection)"
echo "  ✓ UCX_TLS=ze_copy (XPU memory copy)"
echo "  ✓ VISION_ENCODE_SERIALIZE=1"
echo "  ✓ NIXL_USE_CPU_HOST_MEMORY=1"
echo "  ✓ dtype=bfloat16 (XPU optimized)"
echo ""
echo "Log:"
echo "  - $LOG_DIR/encode_xpu.log"
echo ""
echo "=========================================="
echo ""
echo "To monitor encode worker:"
echo "  tail -f $LOG_DIR/encode_xpu.log"
echo ""
echo "To check when ready:"
echo "  grep -i 'registered\\|ready' $LOG_DIR/encode_xpu.log"
echo ""
echo "To stop:"
echo "  kill $ENCODE_PID"
echo "  # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""

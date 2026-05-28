#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (2x Intel XPU Encode Workers)
# This starts: 2 Encode workers on Intel XPU devices 2 and 6 for multimodal processing
# Run this AFTER starting the CUDA side (start_sglang_pd_cuda.sh)
# FIXED: Each encoder uses its own dedicated NIXL side channel port

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

# XPU Devices
export XPU_DEVICE_1=2  # First Intel XPU device ID (for ZE_AFFINITY_MASK)
export XPU_DEVICE_2=6  # Second Intel XPU device ID (for ZE_AFFINITY_MASK)

# Network and Transfer Settings - FIXED: Separate side channel ports
export SIDE_CHANNEL_PORT_1=22098  # NIXL side channel port for encoder 1
export SIDE_CHANNEL_PORT_2=22099  # NIXL side channel port for encoder 2 (FIXED: uncommented)
export KV_EVENT_PORT_1=22080      # KV events port for encoder 1
export KV_EVENT_PORT_2=22083      # KV events port for encoder 2
export UCX_NIC_1=mlx5_1:1         # InfiniBand NIC for encoder 1 (verified active)
export UCX_NIC_2=mlx5_2:1         # InfiniBand NIC for encoder 2 (verified active)

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang 2x Intel XPU Encode Workers"
echo "=========================================="
echo ""
echo "Remote CUDA server: $IP_REMOTE"
echo "Local XPU server:   $IP_LOCAL"
echo "XPU Devices:        $XPU_DEVICE_1, $XPU_DEVICE_2"
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

# Start Intel XPU Encode Worker 1 on Device 2
echo "Starting Intel XPU Encode Worker 1 (Device $XPU_DEVICE_1)..."
echo "This takes a few minutes for model loading..."

ZE_AFFINITY_MASK=$XPU_DEVICE_1 \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT_1} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT_1} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=ze_copy,rc,tcp \
UCX_NET_DEVICES=${UCX_NIC_1} \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
VISION_ENCODE_SERIALIZE=1 \
NIXL_USE_CPU_HOST_MEMORY=1 \
python3 -m dynamo.sglang \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --enable-multimodal \
    --multimodal-encode-worker \
    --chat-template qwen2-vl \
    --dtype bfloat16 \
    --mem-fraction-static 0.7 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_1'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_DIR/encode_xpu_1.log" &

ENCODE_PID_1=$!
echo "Encode Worker 1 started with PID: $ENCODE_PID_1"

# Wait a bit before starting second worker
sleep 5

# Start Intel XPU Encode Worker 2 on Device 6 - FIXED: Uses SIDE_CHANNEL_PORT_2
echo ""
echo "Starting Intel XPU Encode Worker 2 (Device $XPU_DEVICE_2)..."

ZE_AFFINITY_MASK=$XPU_DEVICE_2 \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT_2} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT_2} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=ze_copy,rc,tcp \
UCX_NET_DEVICES=${UCX_NIC_2} \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
VISION_ENCODE_SERIALIZE=1 \
NIXL_USE_CPU_HOST_MEMORY=1 \
python3 -m dynamo.sglang \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --enable-multimodal \
    --multimodal-encode-worker \
    --chat-template qwen2-vl \
    --dtype bfloat16 \
    --mem-fraction-static 0.7 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_2'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_DIR/encode_xpu_2.log" &

ENCODE_PID_2=$!
echo "Encode Worker 2 started with PID: $ENCODE_PID_2"

echo ""
echo "=========================================="
echo "2x Intel XPU Encode Workers Started"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Remote CUDA Server: $IP_REMOTE"
echo "  - NATS: nats://$IP_REMOTE:$PORT_NATS"
echo "  - etcd: http://$IP_REMOTE:$PORT_ETCD"
echo "  - Local XPU IP: $IP_LOCAL"
echo ""
echo "Encoder 1:"
echo "  - XPU Device (ZE_AFFINITY_MASK): $XPU_DEVICE_1"
echo "  - InfiniBand NIC: $UCX_NIC_1"
echo "  - Side Channel: $IP_LOCAL:$SIDE_CHANNEL_PORT_1"
echo "  - KV Events Port: $KV_EVENT_PORT_1"
echo "  - Process PID: $ENCODE_PID_1"
echo "  - Log: $LOG_DIR/encode_xpu_1.log"
echo ""
echo "Encoder 2:"
echo "  - XPU Device (ZE_AFFINITY_MASK): $XPU_DEVICE_2"
echo "  - InfiniBand NIC: $UCX_NIC_2"
echo "  - Side Channel: $IP_LOCAL:$SIDE_CHANNEL_PORT_2"
echo "  - KV Events Port: $KV_EVENT_PORT_2"
echo "  - Process PID: $ENCODE_PID_2"
echo "  - Log: $LOG_DIR/encode_xpu_2.log"
echo ""
echo "Intel XPU Optimizations:"
echo "  ✓ ZE_AFFINITY_MASK (Level Zero device selection)"
echo "  ✓ UCX_TLS=ze_copy (XPU memory copy)"
echo "  ✓ Dedicated InfiniBand NICs per encoder (no bandwidth contention)"
echo "  ✓ Dedicated NIXL side channel ports per encoder (FIXED)"
echo "  ✓ VISION_ENCODE_SERIALIZE=1"
echo "  ✓ NIXL_USE_CPU_HOST_MEMORY=1"
echo "  ✓ dtype=bfloat16 (XPU optimized)"
echo ""
echo "=========================================="
echo ""
echo "To monitor encode workers:"
echo "  tail -f $LOG_DIR/encode_xpu_1.log"
echo "  tail -f $LOG_DIR/encode_xpu_2.log"
echo ""
echo "To check when ready:"
echo "  grep -i 'registered\\|ready' $LOG_DIR/encode_xpu_1.log"
echo "  grep -i 'registered\\|ready' $LOG_DIR/encode_xpu_2.log"
echo ""
echo "To stop all encode workers:"
echo "  kill $ENCODE_PID_1 $ENCODE_PID_2"
echo "  # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""

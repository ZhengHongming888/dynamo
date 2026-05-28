#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (2x Intel XPU Encode Workers)
# Model: Qwen3-VL-32B-Instruct-FP8
# This starts: 2 Encode workers on Intel XPU devices for multimodal processing
# Run this AFTER starting the CUDA side (start_sglang_pd_cuda_32b_fp8.sh)
# Each encoder uses its own dedicated NIXL side channel port

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

# XPU Devices - Choose 2 different devices from: 0, 1, 2, 3, 4, 5, 6, 7
export XPU_DEVICE_1=2  # First Intel XPU device ID (for ZE_AFFINITY_MASK)
export XPU_DEVICE_2=4  # Second Intel XPU device ID (for ZE_AFFINITY_MASK)

# Network and Transfer Settings - Separate ports for each encoder
export SIDE_CHANNEL_PORT_1=22098  # NIXL side channel port for encoder 1
export SIDE_CHANNEL_PORT_2=22099  # NIXL side channel port for encoder 2
export KV_EVENT_PORT_1=22080      # KV events port for encoder 1
export KV_EVENT_PORT_2=22083      # KV events port for encoder 2
export UCX_NIC_1=mlx5_1:1         # InfiniBand NIC for encoder 1
export UCX_NIC_2=mlx5_2:1         # InfiniBand NIC for encoder 2

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang 2x Intel XPU Encode Workers"
echo "Model: Qwen3-VL-32B-Instruct-FP8"
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
    echo "Make sure CUDA side is running: ./start_sglang_pd_cuda_32b_fp8.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Intel XPU Encode Worker 1
echo "Starting Intel XPU Encode Worker 1 (Device $XPU_DEVICE_1)..."
echo "This takes several minutes for 32B model loading..."

ZE_AFFINITY_MASK=$XPU_DEVICE_1 \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
ETCD_REQUEST_TIMEOUT=600 \
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
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --multimodal-encode-worker \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.7 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_1'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_DIR/encode_xpu_32b_1.log" &

ENCODE_PID_1=$!
echo "Encode Worker 1 started with PID: $ENCODE_PID_1"

# Wait before starting second worker to avoid resource contention
echo ""
echo "Waiting 10 seconds before starting second encoder..."
sleep 10

# Start Intel XPU Encode Worker 2
echo ""
echo "Starting Intel XPU Encode Worker 2 (Device $XPU_DEVICE_2)..."

ZE_AFFINITY_MASK=$XPU_DEVICE_2 \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
ETCD_REQUEST_TIMEOUT=600 \
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
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --multimodal-encode-worker \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.7 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_2'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_DIR/encode_xpu_32b_2.log" &

ENCODE_PID_2=$!
echo "Encode Worker 2 started with PID: $ENCODE_PID_2"

echo ""
echo "=========================================="
echo "2x Intel XPU Encode Workers Started"
echo "Model: Qwen3-VL-32B-Instruct-FP8"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Model: Qwen3-VL-32B-Instruct-FP8 (same as CUDA decode worker)"
echo "  - Remote CUDA Server: $IP_REMOTE"
echo "  - NATS: nats://$IP_REMOTE:$PORT_NATS"
echo "  - etcd: http://$IP_REMOTE:$PORT_ETCD"
echo "  - Local XPU IP: $IP_LOCAL"
echo ""
echo "Encoder 1 (32B FP8):"
echo "  - XPU Device (ZE_AFFINITY_MASK): $XPU_DEVICE_1"
echo "  - InfiniBand NIC: $UCX_NIC_1"
echo "  - Side Channel: $IP_LOCAL:$SIDE_CHANNEL_PORT_1"
echo "  - KV Events Port: $KV_EVENT_PORT_1"
echo "  - Process PID: $ENCODE_PID_1"
echo "  - Log: $LOG_DIR/encode_xpu_32b_1.log"
echo ""
echo "Encoder 2 (32B FP8):"
echo "  - XPU Device (ZE_AFFINITY_MASK): $XPU_DEVICE_2"
echo "  - InfiniBand NIC: $UCX_NIC_2"
echo "  - Side Channel: $IP_LOCAL:$SIDE_CHANNEL_PORT_2"
echo "  - KV Events Port: $KV_EVENT_PORT_2"
echo "  - Process PID: $ENCODE_PID_2"
echo "  - Log: $LOG_DIR/encode_xpu_32b_2.log"
echo ""
echo "Intel XPU Optimizations:"
echo "  ✓ ZE_AFFINITY_MASK (Level Zero device selection)"
echo "  ✓ UCX_TLS=ze_copy (XPU memory copy)"
echo "  ✓ Dedicated InfiniBand NICs per encoder (no bandwidth contention)"
echo "  ✓ Dedicated NIXL side channel ports per encoder"
echo "  ✓ VISION_ENCODE_SERIALIZE=1"
echo "  ✓ NIXL_USE_CPU_HOST_MEMORY=1"
echo "  ✓ dtype=auto, kv-cache-dtype=auto (FP8 support)"
echo "  ✓ encoder_only mode enabled (loads vision encoder ONLY, not full LLM)"
echo ""
echo "32B Model Notes:"
echo "  • With encoder_only fix: only vision encoder loaded (~2-4GB per device)"
echo "  • Total XPU memory: ~4-8GB for both encoders"
echo "  • Load time: ~5-10 minutes per encoder"
echo "  • Same model as CUDA decode worker for consistency"
echo ""
echo "=========================================="
echo ""
echo "To monitor encode workers:"
echo "  tail -f $LOG_DIR/encode_xpu_32b_1.log"
echo "  tail -f $LOG_DIR/encode_xpu_32b_2.log"
echo ""
echo "To check when ready:"
echo "  grep -i 'registered\\|ready\\|succeeded' $LOG_DIR/encode_xpu_32b_1.log"
echo "  grep -i 'registered\\|ready\\|succeeded' $LOG_DIR/encode_xpu_32b_2.log"
echo ""
echo "To stop all encode workers:"
echo "  kill $ENCODE_PID_1 $ENCODE_PID_2"
echo "  # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker.*32B'"
echo ""
echo "Performance tip:"
echo "  Monitor both encoders to ensure balanced load distribution"
echo "  Each encoder should process ~50% of incoming multimodal requests"
echo ""

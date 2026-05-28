#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (XPU side)
# This starts: Encode worker on XPU
# Run this AFTER starting the CUDA side (start_sglang_pd_cuda.sh)

set -e

# IMPORTANT: Update these to match your CUDA server
export IP_CUDA=172.26.46.162  # IP of CUDA server running NATS/etcd/frontend
export PORT_NATS=14222
export PORT_ETCD=12379
export XPU_DEVICE=0  # XPU device ID
export SIDE_CHANNEL_PORT=20098

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Prefill/Decode (XPU Encode)"
echo "=========================================="
echo ""
echo "Connecting to CUDA server at: $IP_CUDA"
echo ""

# Start Encode Worker on XPU
echo "Starting SGLang Encode Worker on XPU..."
echo "This takes a few minutes for model loading..."

# Note: Adjust XPU environment variables as needed for your XPU setup
# These are example variables - modify based on your XPU requirements
CUDA_VISIBLE_DEVICES="" \
XPU_VISIBLE_DEVICES=$XPU_DEVICE \
NATS_SERVER=nats://${IP_CUDA}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_CUDA}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_CUDA} \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
PYTHONHASHSEED=0 \
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --encoder-only \
    --dtype auto \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.95 \
    >> "$LOG_DIR/encode_worker.log" 2>&1 &

ENCODE_PID=$!

echo ""
echo "=========================================="
echo "XPU Side (Encode Worker):"
echo "  - Connected to NATS: nats://$IP_CUDA:$PORT_NATS"
echo "  - Connected to etcd: http://$IP_CUDA:$PORT_ETCD"
echo "  - XPU Device: $XPU_DEVICE"
echo "  - Process PID: $ENCODE_PID"
echo ""
echo "Log:"
echo "  - Encode Worker: $LOG_DIR/encode_worker.log"
echo ""
echo "=========================================="
echo ""
echo "To monitor encode worker startup:"
echo "  tail -f $LOG_DIR/encode_worker.log"
echo ""
echo "To check when ready:"
echo "  grep 'registered' $LOG_DIR/encode_worker.log"
echo ""
echo "To stop:"
echo "  kill $ENCODE_PID"

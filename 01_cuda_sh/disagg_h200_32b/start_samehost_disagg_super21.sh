#!/bin/bash
# Same-host disagg launcher: Encoder on GPU 0 + PD on GPU 1, both on super21.
# GPU 0 ↔ GPU 1 share NVLink (NV18 = 18 lanes ≈ 900 GB/s P2P), both on NUMA 0.
# NIXL transfers go via cuda_ipc over NVLink (vs cross-host RoCE NIXL ~25-50 GB/s).
#
# Reuses existing NATS/etcd/frontend. Frontend should be running with kv router mode.
#
# Differences from cross-host launcher:
#  - Encoder + PD on same physical host
#  - UCX_NET_DEVICES=mlx5_0:1 (paired with GPU 0/1 NUMA 0; NVLink takes the embedding)
#  - VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 (loopback; NVLink ipc doesn't use ROCE)
#  - mem-fraction=0.85 for both (no contention since separate GPUs)

set -e

export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.133,192.165.123.52,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

export IP_LOCAL=172.26.46.133
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001

# GPU assignments — same NUMA 0, NVLink-connected
export CUDA_DEVICE_PD=1       # PD on GPU 1
export CUDA_DEVICE_ENCODER=0  # Encoder on GPU 0

# Network ports
export KV_EVENT_PORT_PD=22091
export KV_EVENT_PORT_ENCODER=22090
export SIDE_CHANNEL_PORT_PD=20098
export SIDE_CHANNEL_PORT_ENCODER=20099

export DYN_TCP_MAX_MESSAGE_SIZE=268435456
export DYN_HTTP_BODY_LIMIT_MB=256

# NIC paired with GPU 0/1 (PIX/NODE on NUMA 0). For same-host disagg, NIXL uses
# cuda_ipc over NVLink so this NIC is mostly irrelevant, but UCX still wants a device.
export UCX_NIC=mlx5_0:1

# Mem fraction — both encoder and PD have a full 143 GB H200 each
# (Patches reverted: NIXL receive on CPU, no cuda_ipc memory contention)
export MEM_FRAC_PD=0.85
export MEM_FRAC_ENC=0.85

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Same-host disagg setup (super21 NVLink)"
echo "  Encoder GPU: $CUDA_DEVICE_ENCODER (NUMA 0)"
echo "  PD GPU:      $CUDA_DEVICE_PD       (NUMA 0)"
echo "  Mem-frac:    enc=$MEM_FRAC_ENC pd=$MEM_FRAC_PD"
echo "=========================================="

# Pre-flight
if ! pgrep -f "nats-server.*$PORT_NATS" > /dev/null; then
    echo "ERROR: NATS not running."; exit 1
fi
if ! pgrep -f "etcd.*listen-client-urls.*$PORT_ETCD" > /dev/null; then
    echo "ERROR: etcd not running."; exit 1
fi
if ! pgrep -f "dynamo.frontend.*--http-port $PORT_HTTP" > /dev/null; then
    echo "ERROR: frontend not running."; exit 1
fi
echo "  control plane OK"

for G in $CUDA_DEVICE_PD $CUDA_DEVICE_ENCODER; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')
    if [ "$USED" -gt 1000 ]; then
        echo "ERROR: GPU $G has $USED MiB used."
        exit 1
    fi
done
echo "  GPUs $CUDA_DEVICE_ENCODER and $CUDA_DEVICE_PD free"

pkill -9 -f "dynamo.sglang.*Qwen3-VL-32B" 2>/dev/null || true
sleep 2

PD_LOG="$LOG_DIR/samehost_pd_${TS}.log"
ENC_LOG="$LOG_DIR/samehost_encoder_${TS}.log"
ln -sf "$PD_LOG" "$LOG_DIR/samehost_pd.log"
ln -sf "$ENC_LOG" "$LOG_DIR/samehost_encoder.log"

echo "  PD log:      $PD_LOG"
echo "  Encoder log: $ENC_LOG"
echo ""

# ========================================
# Start PD Worker
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
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT_PD} \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
UCX_MEMTYPE_CACHE=0 \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write \
ENABLE_ENCODER_CACHE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,P2P \
setsid nohup python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 64 \
    --tensor-parallel-size 1 \
    --mem-fraction-static $MEM_FRAC_PD \
    --page-size 16 \
    --chunked-prefill-size 16384 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_PD'","enable_kv_cache_events":true}' \
    > "$PD_LOG" 2>&1 < /dev/null &
PD_PID=$!
disown $PD_PID 2>/dev/null || true
echo "  PD PID: $PD_PID"

# Stagger encoder start so model file is in OS cache
sleep 15

# ========================================
# Start Encoder Worker
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
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT_ENCODER} \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT_ENCODER} \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
UCX_MEMTYPE_CACHE=0 \
NIXL_MAX_BUFFER_SIZE=805306368 \
NIXL_BUFFER_COUNT=256 \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write \
ENABLE_ENCODER_CACHE=0 \
ZMQ_SNDHWM=0 \
ZMQ_RCVHWM=0 \
setsid nohup python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --multimodal-encode-worker \
    --multimodal-embedding-cache-capacity-gb 16 \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static $MEM_FRAC_ENC \
    --page-size 16 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT_ENCODER'","enable_kv_cache_events":true}' \
    > "$ENC_LOG" 2>&1 < /dev/null &
ENC_PID=$!
disown $ENC_PID 2>/dev/null || true
echo "  Encoder PID: $ENC_PID"
echo ""

# Wait for both to register
echo "Waiting for both workers to register (~5-10 min for 32B FP8)..."
for i in {1..180}; do
    sleep 5
    h=$(curl -s --max-time 3 http://localhost:$PORT_HTTP/health 2>/dev/null)
    has_backend=$(echo "$h" | grep -c "backend/generate" || true)
    has_encoder=$(echo "$h" | grep -c "encoder/generate" || true)
    if [ "$has_backend" -gt 0 ] && [ "$has_encoder" -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "Both workers ready at $(date -u)"
        echo "=========================================="
        echo "  PD log:      $PD_LOG"
        echo "  Encoder log: $ENC_LOG"
        exit 0
    fi
    echo "  $((i*5))s... pd=$has_backend enc=$has_encoder"
done

echo "Timeout waiting for both workers."
echo "  tail -100 $PD_LOG"
echo "  tail -100 $ENC_LOG"
exit 1

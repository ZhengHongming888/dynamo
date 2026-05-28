#!/bin/bash
# Start agg EPD worker on super21 GPU 5.
# Uses existing NATS/etcd/frontend (already alive — bring them up first if not).
#
# Differences from disagg PD launcher:
#  - NO --multimodal-worker flag (agg = vision tower in same process)
#  - mem-fraction-static=0.85 (no NIXL buffer headroom needed)
#  - max-running-requests=40 (matches original agg launcher; fits in 0.85 frac)
#
# Encoder on dell06 must be stopped (and its MDC cleaned from etcd) before running this.

set -e

export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.133,192.165.123.52,172.26.46.13,172.26.46.162,172.26.46.172,192.165.123.40,192.165.123.38,192.165.123.37,192.165.123.39,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

export IP_LOCAL_MGMT=172.26.46.133
export IP_LOCAL_ROCE=192.165.123.52
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001
export CUDA_DEVICE_AGG=5
export KV_EVENT_PORT=22080
export SIDE_CHANNEL_PORT=20098
export DYN_TCP_MAX_MESSAGE_SIZE=268435456
export DYN_HTTP_BODY_LIMIT_MB=256
export UCX_NIC=mlx5_4:1

# Agg config
export MEM_FRAC=0.85
export MAX_RUNNING=40

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting agg EPD on GPU $CUDA_DEVICE_AGG"
echo "  mem-frac:  $MEM_FRAC"
echo "  max-run:   $MAX_RUNNING"
echo "=========================================="

# Pre-flight
if ! pgrep -f "nats-server.*$PORT_NATS" > /dev/null; then
    echo "ERROR: NATS not running."
    exit 1
fi
if ! pgrep -f "etcd.*listen-client-urls.*$PORT_ETCD" > /dev/null; then
    echo "ERROR: etcd not running."
    exit 1
fi
if ! pgrep -f "dynamo.frontend.*--http-port $PORT_HTTP" > /dev/null; then
    echo "ERROR: frontend not running."
    exit 1
fi
echo "  control plane OK"

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_DEVICE_AGG | tr -d ' ')
if [ "$GPU_USED" -gt 1000 ]; then
    echo "ERROR: GPU $CUDA_DEVICE_AGG has $GPU_USED MiB used."
    exit 1
fi
echo "  GPU $CUDA_DEVICE_AGG free"

pkill -9 -f "dynamo.sglang.*Qwen3-VL-32B" 2>/dev/null || true
sleep 2

LOGFILE="$LOG_DIR/agg_epd_worker_$(date -u +%Y%m%d_%H%M%S).log"
ln -sf "$LOGFILE" "$LOG_DIR/agg_epd_worker.log"

echo "  log: $LOGFILE"
echo ""
echo "Starting agg EPD worker..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_AGG \
NATS_SERVER=nats://${IP_LOCAL_MGMT}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL_MGMT}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
DYN_LOG=debug \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL_ROCE} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
UCX_MEMTYPE_CACHE=0 \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,P2P \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests $MAX_RUNNING \
    --tensor-parallel-size 1 \
    --mem-fraction-static $MEM_FRAC \
    --page-size 16 \
    --chunked-prefill-size 16384 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    > "$LOGFILE" 2>&1 < /dev/null &
PID=$!
disown $PID 2>/dev/null || true
echo "  PID: $PID"
echo ""

echo "Waiting for agg EPD model registration (~5-8 min)..."
for i in {1..180}; do
    sleep 5
    out=$(curl -s --max-time 3 http://localhost:$PORT_HTTP/v1/models 2>/dev/null)
    if echo "$out" | grep -q "Qwen3-VL-32B" && curl -s --max-time 3 http://localhost:$PORT_HTTP/health 2>/dev/null | grep -q "backend"; then
        echo ""
        echo "=========================================="
        echo "Agg EPD ready at $(date -u)"
        echo "=========================================="
        exit 0
    fi
    echo "  $((i*5))s..."
done

echo "Timeout waiting for model registration."
echo "  tail -100 $LOGFILE"
exit 1

#!/bin/bash
# Restart ONLY the PD worker (NATS/etcd/frontend already alive from earlier startup).
# Uses lower mem-fraction-static (0.50 instead of 0.65) to leave ~30 GB headroom
# in case an external process intrudes on the same physical GPU.
#
# Encoders on dell06: their etcd leases (600s TTL) have expired since the prior
# PD crash, so they need to be RESTARTED on dell06 too (separate action there).

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
export CUDA_DEVICE_PD=5
export KV_EVENT_PORT=22081
export SIDE_CHANNEL_PORT=20098
export DYN_TCP_MAX_MESSAGE_SIZE=268435456
export DYN_HTTP_BODY_LIMIT_MB=256
export UCX_NIC=mlx5_4:1

# Mem fraction: 0.50 (lower than original 0.65) to leave ~30 GB headroom
# for unexpected GPU intrusions. Trade-off: smaller KV cache (max_total_num_tokens
# drops from 467k -> ~280k), but np=32 8img/1080p only needs ~50k tokens active so fine.
export MEM_FRAC=0.50

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

# ----- pre-flight -----
echo "=========================================="
echo "Restarting PD only (low mem variant)"
echo "  GPU:       $CUDA_DEVICE_PD"
echo "  mem-frac:  $MEM_FRAC"
echo "=========================================="

# Verify control plane still alive
if ! pgrep -f "nats-server.*$PORT_NATS" > /dev/null; then
    echo "ERROR: NATS not running. Use start_sglang_pd_cuda_32b_fp8_giga01.sh for full bring-up."
    exit 1
fi
if ! pgrep -f "etcd.*listen-client-urls.*$PORT_ETCD" > /dev/null; then
    echo "ERROR: etcd not running. Use start_sglang_pd_cuda_32b_fp8_giga01.sh for full bring-up."
    exit 1
fi
if ! pgrep -f "dynamo.frontend.*--http-port $PORT_HTTP" > /dev/null; then
    echo "ERROR: frontend not running. Use start_sglang_pd_cuda_32b_fp8_giga01.sh for full bring-up."
    exit 1
fi
echo "  control plane OK (NATS, etcd, frontend alive)"

# Verify GPU 5 is free
GPU5_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_DEVICE_PD | tr -d ' ')
if [ "$GPU5_USED" -gt 1000 ]; then
    echo "ERROR: GPU $CUDA_DEVICE_PD has $GPU5_USED MiB used. Pick a different GPU or wait."
    exit 1
fi
echo "  GPU $CUDA_DEVICE_PD free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $CUDA_DEVICE_PD | tr -d ' ') MiB"

# Kill any stale PD worker
pkill -9 -f "dynamo.sglang.*Qwen3-VL-32B" 2>/dev/null || true
sleep 2

LOGFILE="$LOG_DIR/pd_worker_giga01_lowmem_$(date -u +%Y%m%d_%H%M%S).log"
ln -sf "$LOGFILE" "$LOG_DIR/pd_worker_giga01.log"  # keep canonical symlink current

echo "  log: $LOGFILE"
echo ""
echo "Starting PD Worker on GPU $CUDA_DEVICE_PD (mem-fraction=$MEM_FRAC)..."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_PD \
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
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
UCX_MEMTYPE_CACHE=0 \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,P2P \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 64 \
    --tensor-parallel-size 1 \
    --mem-fraction-static $MEM_FRAC \
    --page-size 16 \
    --chunked-prefill-size 16384 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    > "$LOGFILE" 2>&1 < /dev/null &
PD_PID=$!
disown $PD_PID 2>/dev/null || true
echo "  PD PID: $PD_PID"
echo ""

# Wait for model registration (allow up to 15 min for DeepGEMM warmup at 32B FP8)
echo "Waiting for PD model registration (model load + warmup ~5-8 min)..."
for i in {1..180}; do
    sleep 5
    if curl -s http://localhost:$PORT_HTTP/v1/models 2>/dev/null | grep -q "Qwen3-VL-32B"; then
        echo ""
        echo "=========================================="
        echo "PD ready at $(date -u)"
        echo "=========================================="
        echo "  PD log:  $LOGFILE"
        echo "  health:  curl http://$IP_LOCAL_MGMT:$PORT_HTTP/health"
        echo ""
        echo "Next step: restart the 2 encoders on dell06 so they re-register."
        echo "Their previous leases expired during the PD downtime."
        exit 0
    fi
    echo "  $((i*5))s..."
done

echo "Timeout waiting for model registration."
echo "  tail -100 $LOGFILE"
exit 1

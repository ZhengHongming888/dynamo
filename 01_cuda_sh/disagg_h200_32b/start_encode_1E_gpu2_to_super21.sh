#!/bin/bash
# 1E encoder-only worker on dell06 H200 GPU 2, paired with PD on super21-h200.
# Cross-host disagg, Qwen3-VL-32B-Instruct-FP8 multimodal.
#
# Topology:
#   - dell06 (this host, 172.26.46.162) runs 1 encoder worker on CUDA GPU 2
#   - super21-h200 (172.26.46.133) runs PD + frontend + etcd + NATS
#   - Control plane: NATS (14222) / etcd (12379) on super21-h200
#   - Data plane:    NIXL READ over RoCE 192.165.123.0/24 via mlx5_0
#
# NUMA notes (this host, dell06 sc09dell06-nvd):
#   - GPU 2 is on NUMA 1; mlx5_0 is on NUMA 0 (cross-NUMA over UPI to the NIC).
#     Same setup that ran fine in the 2E run on this host. Penalty is ~1-2 ms
#     per 64 MB embedding, negligible vs ViT forward.
#   - mlx5_0 is the only RoCE NIC with a routable IP on this box.
#
# Pre-flight:
#   - super21-h200:14222 / 12379 / 7001 must be reachable
#   - GPU 2 must be free
#   - mlx5_0 must be ACTIVE on 192.165.123.25
#
# Usage:
#   ./start_encode_1E_gpu2_to_super21.sh
#   tail -F logs/encode_1E_gpu2_to_super21.log
#   pkill -f 'multimodal-encode-worker'
#
# Logs:
#   logs/encode_1E_gpu2_to_super21.log

set -e

# ---- Proxy bypass for control / data plane endpoints -----------------------
export no_proxy="localhost,127.0.0.1,172.26.46.133,172.26.46.0/24,192.165.123.0/24,.intel.com"
export NO_PROXY="$no_proxy"

# ---- Remote (PD) host -------------------------------------------------------
export IP_REMOTE=172.26.46.133              # super21-h200 PD host
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_FRONTEND=7001                   # for sanity check only

# ---- Local host (dell06) ----------------------------------------------------
export IP_LOCAL_ROCE=192.165.123.25         # dell06 mlx5_0 RoCE IP (only routable one)
export IP_LOCAL_MGMT=172.26.46.162          # dell06 mgmt IP

# ---- Encoder config ---------------------------------------------------------
export UCX_NIC=mlx5_0:1                     # ACTIVE RoCE NIC, NUMA 0
export MODEL_PATH="/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8"

GPU=2
SIDE_CHANNEL_PORT=20098
KV_EVENT_PORT=22080

# NUMA pinning: GPU 2 is on NUMA 1
NUMA_NODE=1

# ---- Logs -------------------------------------------------------------------
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/encode_1E_gpu${GPU}_to_super21.log"

cat <<EOF
==========================================
H200 1E Encode Worker (GPU $GPU)
            -> super21-h200 PD
==========================================
Model:           Qwen3-VL-32B-Instruct-FP8
Local mgmt IP:   $IP_LOCAL_MGMT
Local RoCE IP:   $IP_LOCAL_ROCE  (NIXL side channel)
Remote PD host:  $IP_REMOTE
NATS:            nats://$IP_REMOTE:$PORT_NATS
etcd:            http://$IP_REMOTE:$PORT_ETCD
UCX_NIC:         $UCX_NIC (NUMA 0)
GPU:             $GPU (NUMA $NUMA_NODE, cross-NUMA to NIC)
SIDE_CHANNEL:    $IP_LOCAL_ROCE:$SIDE_CHANNEL_PORT
KV_EVENT_PORT:   $KV_EVENT_PORT
Log file:        $LOG_FILE
==========================================
EOF

# ---- Pre-flight ------------------------------------------------------------
echo "[pre-flight] Checking control plane on $IP_REMOTE ..."
for port in "$PORT_NATS" "$PORT_ETCD" "$PORT_FRONTEND"; do
  if ! timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$port" 2>/dev/null; then
    echo "ERROR: $IP_REMOTE:$port not reachable. Start PD/NATS/etcd on super21-h200 first."
    exit 1
  fi
  echo "  $IP_REMOTE:$port reachable"
done

echo "[pre-flight] Checking GPU $GPU is free ..."
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$used" -gt 500 ]; then
  echo "ERROR: GPU $GPU already has ${used} MiB in use. Free it first."
  exit 1
fi
echo "  GPU $GPU free (${used} MiB used)"

echo "[pre-flight] Checking mlx5_0 is ACTIVE ..."
state=$(ibv_devinfo -d mlx5_0 2>/dev/null | awk '/state:/ {print $2; exit}')
if [ "$state" != "PORT_ACTIVE" ]; then
  echo "ERROR: mlx5_0 state is '$state', expected PORT_ACTIVE"
  exit 1
fi
echo "  mlx5_0 PORT_ACTIVE"

if ! command -v numactl >/dev/null 2>&1; then
  echo "WARNING: numactl not found; encoder will not be NUMA-pinned."
  NUMACTL_CMD=""
else
  NUMACTL_CMD="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
fi

echo
echo "[launch] Starting 1 encoder worker on GPU $GPU ..."
: > "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU \
NVIDIA_VISIBLE_DEVICES=$GPU \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
ETCD_LEASE_TTL=600 \
DYN_LOG=debug \
DYN_REQUEST_PLANE=tcp \
DYN_TCP_MAX_MESSAGE_SIZE=268435456 \
DYN_HTTP_BODY_LIMIT_MB=256 \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL_ROCE} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
UCX_IB_ROCE_REACHABILITY_MODE=all \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
nohup setsid $NUMACTL_CMD python3 -m dynamo.sglang \
    --model "$MODEL_PATH" \
    --enable-multimodal \
    --multimodal-encode-worker \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --page-size 16 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    >>"$LOG_FILE" 2>&1 &

PID=$!
disown $PID 2>/dev/null || true
echo "$PID" > "$LOG_DIR/encode_1E_gpu${GPU}.pid"

echo
echo "[done] Encoder launched on GPU $GPU, PID $PID"
echo "  Watch:   tail -F $LOG_FILE"
echo "  Stop:    pkill -f 'multimodal-encode-worker'   (or kill $PID)"
echo "  Verify:  pgrep -af multimodal-encode-worker"
echo
echo "Wait ~60-90s for sglang model load, then verify etcd registration on super21-h200:"
echo "  ssh ${IP_REMOTE} 'etcdctl --endpoints=http://localhost:${PORT_ETCD} get --prefix v1/instances/dynamo/encoder'"

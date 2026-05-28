#!/bin/bash
# 2E encoder-only worker pool on dell06 H200 GPUs 0 & 1, paired with PD on super21-h200.
# Cross-host disagg, Qwen3-VL-32B-Instruct-FP8 multimodal.
#
# Topology:
#   - dell06 (this host, 172.26.46.162) runs 2 encoder workers on CUDA GPU 0 + GPU 1
#   - super21-h200 (172.26.46.133) runs PD + frontend + etcd + NATS
#   - Control plane: NATS (14222) / etcd (12379) on super21-h200
#   - Data plane:    NIXL READ over RoCE 192.165.123.0/24 via mlx5_0
#
# NUMA notes (this host, dell06 sc09dell06-nvd):
#   - GPUs 0,1 are on NUMA 0 (CPU affinity 0,2,4,6,8,10,...)
#   - Only mlx5_0 has a routable RoCE IP (192.165.123.25), and it is on NUMA 0
#     -> GPU 1 is PIX with mlx5_0 (best affinity), GPU 0 is NODE (same NUMA, fine)
#   - mlx5_1 is mgmt (172.26.46.162), mlx5_2 is PORT_DOWN, mlx5_3 has no netdev/IP.
#   - `numactl --cpunodebind=0 --membind=0` keeps Python/CUDA-host allocations on
#     NUMA 0, same as the NIC and GPUs.
#
# Pre-flight:
#   - super21-h200:14222 / 12379 / 7001 must be reachable
#   - GPUs 0 and 1 must be free (no other CUDA processes)
#   - mlx5_0 must be ACTIVE on 192.165.123.25
#
# Usage:
#   ./start_encode_2E_gpu23_to_super21.sh           # launch both encoders, return PIDs
#   tail -F logs/encode_2E_gpu{0,1}_to_super21.log  # watch
#   pkill -f 'multimodal-encode-worker'             # stop both
#
# Logs:
#   logs/encode_2E_gpu0_to_super21.log
#   logs/encode_2E_gpu1_to_super21.log

set -e

# ---- Proxy bypass for control / data plane endpoints -----------------------
# Without this the corporate http_proxy intercepts traffic to super21 / RoCE.
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

# ---- Shared encoder config --------------------------------------------------
export UCX_NIC=mlx5_0:1                     # ACTIVE RoCE NIC, NUMA 0, PIX with GPU 1
export MODEL_PATH="/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8"

# Per-encoder ports (20098+i / 22080+i convention)
GPUS=(0 2)
SIDE_CHANNEL_PORTS=(20098 20099)
KV_EVENT_PORTS=(22080 22081)

# NUMA pinning per-GPU. GPU 0 is on NUMA 0 (same NUMA as mlx5_0); GPU 2 is on
# NUMA 1 (cross-NUMA over UPI to the NIC). We pin each encoder to its GPU's
# NUMA node so Python/CUDA-host allocations stay close to the GPU; the NIC-side
# memcpy traverses UPI in the cross-NUMA case (~1-2 ms penalty per 64 MB
# embedding, negligible vs the ~1.5 s ViT forward).
declare -A GPU_NUMA=( [0]=0 [2]=1 [3]=1 [4]=2 [5]=2 )

# ---- Logs -------------------------------------------------------------------
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"

cat <<EOF
==========================================
H200 2E Encode Workers (GPU ${GPUS[0]} + GPU ${GPUS[1]})
            -> super21-h200 PD
==========================================
Model:           Qwen3-VL-32B-Instruct-FP8
Local mgmt IP:   $IP_LOCAL_MGMT
Local RoCE IP:   $IP_LOCAL_ROCE  (NIXL side channel)
Remote PD host:  $IP_REMOTE
NATS:            nats://$IP_REMOTE:$PORT_NATS
etcd:            http://$IP_REMOTE:$PORT_ETCD
UCX_NIC:         $UCX_NIC (NUMA 0)
NUMA pin:        per-GPU (GPU${GPUS[0]}=NUMA${GPU_NUMA[${GPUS[0]}]}, GPU${GPUS[1]}=NUMA${GPU_NUMA[${GPUS[1]}]})

  Encoder #1: GPU ${GPUS[0]}, side-channel ${SIDE_CHANNEL_PORTS[0]}, kv-event ${KV_EVENT_PORTS[0]}
  Encoder #2: GPU ${GPUS[1]}, side-channel ${SIDE_CHANNEL_PORTS[1]}, kv-event ${KV_EVENT_PORTS[1]}

Logs:
  $LOG_DIR/encode_2E_gpu${GPUS[0]}_to_super21.log
  $LOG_DIR/encode_2E_gpu${GPUS[1]}_to_super21.log
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

echo "[pre-flight] Checking GPUs ${GPUS[*]} are free ..."
for g in "${GPUS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 500 ]; then
    echo "ERROR: GPU $g already has ${used} MiB in use. Free it first."
    exit 1
  fi
  echo "  GPU $g free (${used} MiB used)"
done

echo "[pre-flight] Checking mlx5_0 is ACTIVE ..."
state=$(ibv_devinfo -d mlx5_0 2>/dev/null | awk '/state:/ {print $2; exit}')
if [ "$state" != "PORT_ACTIVE" ]; then
  echo "ERROR: mlx5_0 state is '$state', expected PORT_ACTIVE"
  exit 1
fi
echo "  mlx5_0 PORT_ACTIVE"

if ! command -v numactl >/dev/null 2>&1; then
  echo "WARNING: numactl not found; encoders will not be NUMA-pinned."
  HAVE_NUMACTL=0
else
  HAVE_NUMACTL=1
fi

echo
echo "[launch] Starting 2 encoder workers ..."

# ---- Launch one encoder ----------------------------------------------------
launch_encoder() {
  local gpu=$1
  local sc_port=$2
  local kv_port=$3
  local log_file="$LOG_DIR/encode_2E_gpu${gpu}_to_super21.log"

  echo "  [GPU $gpu] -> $log_file"

  # Truncate previous log so the run is easy to find
  : > "$log_file"

  local numa_node="${GPU_NUMA[$gpu]}"
  local numactl_cmd=""
  if [ "$HAVE_NUMACTL" = "1" ] && [ -n "$numa_node" ]; then
    numactl_cmd="numactl --cpunodebind=$numa_node --membind=$numa_node"
    echo "  [GPU $gpu] NUMA pin: $numactl_cmd"
  fi

  CUDA_VISIBLE_DEVICES=$gpu \
  NVIDIA_VISIBLE_DEVICES=$gpu \
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
  VLLM_NIXL_SIDE_CHANNEL_PORT=${sc_port} \
  DYN_VLLM_KV_EVENT_PORT=${kv_port} \
  UCX_MEMTYPE_CACHE=0 \
  UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
  UCX_NET_DEVICES=${UCX_NIC} \
  UCX_IB_ROCE_REACHABILITY_MODE=all \
  DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
  ENABLE_ENCODER_CACHE=0 \
  nohup setsid $numactl_cmd python3 -m dynamo.sglang \
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
      --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$kv_port'","enable_kv_cache_events":true}' \
      >>"$log_file" 2>&1 &

  local pid=$!
  disown $pid 2>/dev/null || true
  echo "$pid" > "$LOG_DIR/encode_2E_gpu${gpu}.pid"
  echo "  [GPU $gpu] PID $pid"
}

launch_encoder "${GPUS[0]}" "${SIDE_CHANNEL_PORTS[0]}" "${KV_EVENT_PORTS[0]}"
# Short stagger so etcd registrations don't race
sleep 5
launch_encoder "${GPUS[1]}" "${SIDE_CHANNEL_PORTS[1]}" "${KV_EVENT_PORTS[1]}"

echo
echo "[done] Both encoders launched."
echo "  Watch:   tail -F $LOG_DIR/encode_2E_gpu${GPUS[0]}_to_super21.log $LOG_DIR/encode_2E_gpu${GPUS[1]}_to_super21.log"
echo "  Stop:    pkill -f 'multimodal-encode-worker'"
echo "  Verify:  pgrep -af multimodal-encode-worker"
echo
echo "Wait ~60-90s for sglang model load, then verify etcd registrations on super21-h200:"
echo "  ssh ${IP_REMOTE} 'etcdctl --endpoints=http://localhost:${PORT_ETCD} get --prefix v1/instances/dynamo/encoder'"

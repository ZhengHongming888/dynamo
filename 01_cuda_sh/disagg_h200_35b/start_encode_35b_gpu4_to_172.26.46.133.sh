#!/bin/bash
# 35B encoder-only worker on dell06 GPU 4, pairing with PD on 172.26.46.133.
# Cross-host disagg, Qwen3.5-35B-A3B (BF16 MoE multimodal).
#
# Topology mirrors the 32B run we just did:
#   - This host (dell06, 172.26.46.162) runs the encoder on CUDA GPU 4
#   - 172.26.46.133 runs PD + frontend + etcd + NATS
#   - Control plane: NATS/etcd on 172.26.46.133
#   - Data plane:    NIXL READ over RoCE 192.165.123.0/24 via mlx5_0 (192.165.123.25)
#
# Derived from start_sglang_encode_cuda_35b_dell06.sh, with these changes:
#   - IP_REMOTE = 172.26.46.133 (was 172.26.46.75)
#   - CUDA_DEVICE_ENCODER = 4 (was 5; GPU 4 is also NUMA 2)
#   - UCX_NIC = mlx5_0:1 (only mlx5_* with routable RoCE IP on this host)
#   - IP_LOCAL_ROCE = 192.165.123.25 (mlx5_0)
#   - Logs to encode_35b_gpu4_to_172.26.46.133.log

set -e

# ---- Remote (PD) host -------------------------------------------------------
export IP_REMOTE_MGMT=172.26.46.133
export PORT_NATS=14222
export PORT_ETCD=12379

# ---- Local host -------------------------------------------------------------
export IP_LOCAL_MGMT=172.26.46.162
export IP_LOCAL_ROCE=192.165.123.25     # dell06 mlx5_0
export UCX_NIC=mlx5_0:1
export LOCAL_NDEV=eno17095np0

# ---- GPU --------------------------------------------------------------------
export CUDA_DEVICE_ENCODER=${CUDA_DEVICE_ENCODER:-4}

# ---- Ports for THIS encoder instance ---------------------------------------
# Distinct from the 32B encoder's 20098/22080 in case both run concurrently.
export SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT:-20198}
export KV_EVENT_PORT=${KV_EVENT_PORT:-22181}

# ---- Misc tuning -----------------------------------------------------------
export DYN_TCP_MAX_MESSAGE_SIZE=268435456       # 256 MB
export DYN_HTTP_BODY_LIMIT_MB=256

# ---- Model -----------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}
ENCODER_MEM_FRACTION=${ENCODER_MEM_FRACTION:-0.5}

# ---- Logs -------------------------------------------------------------------
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/encode_35b_gpu${CUDA_DEVICE_ENCODER}_to_${IP_REMOTE_MGMT}.log"

cat <<EOF
==========================================
H200 35B Encode Worker (GPU $CUDA_DEVICE_ENCODER) -> PD @ $IP_REMOTE_MGMT
==========================================
Model:           Qwen3.5-35B-A3B (BF16 MoE multimodal)
This host:       $(hostname) (mgmt $IP_LOCAL_MGMT, RoCE $IP_LOCAL_ROCE)
Remote PD host:  $IP_REMOTE_MGMT
NATS:            nats://$IP_REMOTE_MGMT:$PORT_NATS
etcd:            http://$IP_REMOTE_MGMT:$PORT_ETCD
CUDA_DEVICE:     $CUDA_DEVICE_ENCODER (NUMA 2)
UCX_NIC:         $UCX_NIC ($LOCAL_NDEV)
SIDE_CHANNEL:    $IP_LOCAL_ROCE:$SIDE_CHANNEL_PORT
KV_EVENT_PORT:   $KV_EVENT_PORT
mem-fraction:    $ENCODER_MEM_FRACTION
Log file:        $LOG_FILE
==========================================
EOF

# ---- Pre-flight: GPU free
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_DEVICE_ENCODER 2>/dev/null | tr -d ' ')
if [ -z "$GPU_USED" ]; then
  echo "ERROR: cannot query GPU $CUDA_DEVICE_ENCODER"; exit 1
fi
if [ "$GPU_USED" -gt 1024 ]; then
  echo "WARN: GPU $CUDA_DEVICE_ENCODER already has ${GPU_USED} MiB in use"
fi
echo "[ok] GPU $CUDA_DEVICE_ENCODER (${GPU_USED} MiB used)"

# ---- Pre-flight: control plane reachable
for p in $PORT_NATS $PORT_ETCD; do
  if timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE_MGMT/$p" 2>/dev/null; then
    echo "[ok]  $IP_REMOTE_MGMT:$p reachable"
  else
    echo "ERROR: $IP_REMOTE_MGMT:$p not reachable. Start PD/NATS/etcd first."
    exit 1
  fi
done

# ---- Pre-flight: model path
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: model path not found: $MODEL_PATH"; exit 1
fi
echo "[ok] model: $MODEL_PATH"
echo

# ---- Launch
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ENCODER \
NVIDIA_VISIBLE_DEVICES=$CUDA_DEVICE_ENCODER \
NATS_SERVER=nats://${IP_REMOTE_MGMT}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE_MGMT}:${PORT_ETCD} \
ETCD_REQUEST_TIMEOUT=600 \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
DYN_LOG=info \
DYN_TCP_MAX_MESSAGE_SIZE=$DYN_TCP_MAX_MESSAGE_SIZE \
DYN_HTTP_BODY_LIMIT_MB=$DYN_HTTP_BODY_LIMIT_MB \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
DYN_VLLM_KV_EVENT_PORT=$KV_EVENT_PORT \
VLLM_NIXL_SIDE_CHANNEL_HOST=$IP_LOCAL_ROCE \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=$UCX_NIC \
UCX_MEMTYPE_CACHE=0 \
UCX_IB_ROCE_REACHABILITY_MODE=all \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
VISION_ENCODE_SERIALIZE=1 \
NIXL_USE_CPU_HOST_MEMORY=0 \
python3 -m dynamo.sglang \
    --model "$MODEL_PATH" \
    --enable-multimodal \
    --multimodal-encode-worker \
    --encoder-only \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype auto \
    --mem-fraction-static $ENCODER_MEM_FRACTION \
    --page-size 16 \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_FILE"

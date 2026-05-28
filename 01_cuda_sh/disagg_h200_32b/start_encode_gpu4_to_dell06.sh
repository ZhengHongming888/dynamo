#!/bin/bash
# Encoder-only worker on giga01 GPU 4, pairing with PD on dell06 (172.26.46.133).
# Cross-host disagg, 32B FP8 multimodal.
#
# Topology:
#   - This host (giga01, 172.26.46.162) runs the encoder on CUDA GPU 4
#   - dell06 (172.26.46.133) runs PD + frontend + etcd + NATS
#   - Control plane: NATS/etcd on dell06
#   - Data plane:    NIXL READ over RoCE 192.165.123.0/24 via mlx5_0
#
# Verified before launch:
#   - dell06:14222 (NATS), 12379 (etcd), 7001 (frontend) all reachable
#   - GPU 4 free (143 GB)
#   - mlx5_0 ACTIVE on 192.165.123.25/24

set -e

# ---- Remote (PD) host -------------------------------------------------------
export IP_REMOTE=172.26.46.133          # dell06 PD host
export PORT_NATS=14222
export PORT_ETCD=12379

# ---- Local host -------------------------------------------------------------
# RoCE IP for NIXL side-channel; falls back to mgmt IP if RoCE unset.
export IP_LOCAL_ROCE=192.165.123.25     # giga01 mlx5_0
export IP_LOCAL_MGMT=172.26.46.162

# ---- GPU + NIC --------------------------------------------------------------
export CUDA_DEVICE=5                    # this run uses GPU 5
export UCX_NIC=mlx5_0:1                 # only mlx5_* with a routable RoCE IP

# ---- Ports for this encoder instance ---------------------------------------
export SIDE_CHANNEL_PORT=20098          # NIXL side channel (embeddings metadata)
export KV_EVENT_PORT=22080              # KV events ZMQ port

# ---- Logs -------------------------------------------------------------------
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/encode_gpu4_to_dell06.log"

cat <<EOF
==========================================
H200 Encode Worker (GPU 4) -> dell06 PD
==========================================
Model:           Qwen3-VL-32B-Instruct-FP8
Local mgmt IP:   $IP_LOCAL_MGMT
Local RoCE IP:   $IP_LOCAL_ROCE  (NIXL side channel)
Remote PD host:  $IP_REMOTE
NATS:            nats://$IP_REMOTE:$PORT_NATS
etcd:            http://$IP_REMOTE:$PORT_ETCD
CUDA_DEVICE:     $CUDA_DEVICE
UCX_NIC:         $UCX_NIC
SIDE_CHANNEL:    $IP_LOCAL_ROCE:$SIDE_CHANNEL_PORT
KV_EVENT_PORT:   $KV_EVENT_PORT
Log file:        $LOG_FILE
==========================================
EOF

# Pre-flight: control plane must be reachable
for port in "$PORT_NATS" "$PORT_ETCD"; do
  if ! timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$port" 2>/dev/null; then
    echo "ERROR: $IP_REMOTE:$port not reachable. Start PD/NATS/etcd on dell06 first."
    exit 1
  fi
done
echo "Control plane reachable on $IP_REMOTE."
echo

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
NVIDIA_VISIBLE_DEVICES=$CUDA_DEVICE \
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
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
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
    2>&1 | tee -a "$LOG_FILE"

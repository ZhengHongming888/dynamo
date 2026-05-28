#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (Intel B70 XPU Encode Worker)
# Model: Qwen3.5-35B-A3B (MoE, ~67 GB BF16, ~3B activated, multimodal)
# This starts: Encode worker on Intel B70 (Battlemage) XPU for multimodal processing.
# Pairs with H200 (giga01 / sc09super21-h200) running prefill + decode workers.
#
# Run this AFTER the H200 side has started NATS / etcd / frontend / P / D workers.
#
# Host    : sc09giga01-b70
# Hardware: 8x Intel(R) Graphics [0xe223] (Battlemage), 8x ConnectX-7 (RoCE)
# Pair    : sc09super21-h200.sc.intel.com (172.26.46.75)
#           giga01 NIC = mlx5_4 (192.165.123.52, NUMA 2)
#
# B70 NUMA topology (XPU -> NUMA -> best NIC):
#   XPU 0..3  (NUMA 0)  -> mlx5_0 (.40) or mlx5_1 (.38)
#   XPU 4..7  (NUMA 2)  -> mlx5_2 (.37) or mlx5_3 (.39)
# The script auto-picks a NUMA-local NIC based on $XPU_DEVICE.
#
# Encoder-only mode: only the vision tower is loaded onto the B70 XPU,
# NOT the 67 GiB language model -- that fits in the 32 GiB B70 VRAM.

set -e

# ========================================
# CONFIGURATION - matches H200 side
# ========================================

# Remote H200 server (NATS / etcd / frontend / P / D)
export IP_REMOTE=172.26.46.75   # giga01 / sc09super21-h200 (mgmt IP)
export PORT_NATS=14222
export PORT_ETCD=12379

# Local model path (Qwen3.5-35B-A3B, multimodal MoE, ~67 GB BF16)
export MODEL_PATH=${MODEL_PATH:-/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}

# XPU Device (Battlemage Pro 70) - 0..7. Default 3 (NUMA 0).
export XPU_DEVICE=${XPU_DEVICE:-3}

# NUMA-local NIC selection. Override UCX_NIC / IP_LOCAL on the command line
# to force a specific NIC (e.g. UCX_NIC=mlx5_2:1 IP_LOCAL=192.165.123.37).
if [ -z "${UCX_NIC:-}" ] || [ -z "${IP_LOCAL:-}" ]; then
    case "$XPU_DEVICE" in
        0|1|2|3)  AUTO_NIC=mlx5_0; AUTO_NDEV=ens12np0; AUTO_IP=192.165.123.40 ;;  # NUMA 0
        4|5|6|7)  AUTO_NIC=mlx5_2; AUTO_NDEV=ens9np0;  AUTO_IP=192.165.123.37 ;;  # NUMA 2
        *)        AUTO_NIC=mlx5_0; AUTO_NDEV=ens12np0; AUTO_IP=192.165.123.40 ;;
    esac
    : "${UCX_NIC:=${AUTO_NIC}:1}"
    : "${IP_LOCAL:=${AUTO_IP}}"
    : "${LOCAL_NDEV:=${AUTO_NDEV}}"
fi
export UCX_NIC IP_LOCAL LOCAL_NDEV

# Network and Transfer Settings - must match H200 expectations
export SIDE_CHANNEL_PORT=20098   # NIXL side channel (H200 firewall opens 20098)
export KV_EVENT_PORT=22081       # ZMQ KV events    (H200 firewall opens 22081)

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Intel B70 XPU Encode Worker"
echo "Model: Qwen3.5-35B-A3B (MoE, multimodal)"
echo "Pair : H200 @ $IP_REMOTE"
echo "=========================================="
echo ""
echo "Model path        : $MODEL_PATH"
echo "Remote H200 server: $IP_REMOTE  (giga01 NIC mlx5_4 / 192.165.123.52, NUMA 2)"
echo "Local  B70 fabric : $UCX_NIC -> $LOCAL_NDEV -> $IP_LOCAL"
echo "XPU Device        : $XPU_DEVICE  (NUMA $([ $XPU_DEVICE -lt 4 ] && echo 0 || echo 2))"
echo ""

# Pre-flight: control-plane reachability (NATS / etcd)
echo "[pre-flight] control plane on $IP_REMOTE ..."
for p in $PORT_NATS $PORT_ETCD; do
    if timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$p" 2>/dev/null; then
        echo "  OK   port $p"
    else
        echo "  FAIL port $p (firewall? service not started on H200?)"
    fi
done

# Pre-flight: model path exists
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: model path not found or config.json missing: $MODEL_PATH"
    exit 1
fi

# Pre-flight: data-plane fabric IP must exist on the chosen netdev
if ! python3 -c "
import socket, fcntl, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', b'$LOCAL_NDEV'[:15]))[20:24])
assert ip == '$IP_LOCAL', f'$LOCAL_NDEV has {ip}, expected $IP_LOCAL'
" 2>/dev/null; then
    echo "WARNING: $LOCAL_NDEV ($UCX_NIC) does not have $IP_LOCAL. Verify NIC<->IP mapping."
fi

if ! timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$PORT_NATS" 2>/dev/null; then
    read -p "Control plane unreachable. Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Start Intel B70 XPU Encode Worker
echo ""
echo "Starting Intel B70 XPU Encode Worker (Qwen3.5-35B-A3B)..."
echo "Model load can take 5-10 minutes."

ZE_AFFINITY_MASK=$XPU_DEVICE \
NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
ETCD_REQUEST_TIMEOUT=600 \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
UCX_MEMTYPE_CACHE=0 \
UCX_TLS=ze_copy,rc,tcp \
UCX_NET_DEVICES=${UCX_NIC} \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
VISION_ENCODE_SERIALIZE=1 \
BENCH_DISABLE_XPU_PATCH=${BENCH_DISABLE_XPU_PATCH:-0} \
python3 -m dynamo.sglang \
    --model "${MODEL_PATH}" \
    --enable-multimodal \
    --multimodal-encode-worker \
    --encoder-only \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype auto \
    --mem-fraction-static 0.5 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    2>&1 | tee -a "$LOG_DIR/encode_xpu_35b_b70.log" &

ENCODE_PID=$!

echo ""
echo "=========================================="
echo "Intel B70 XPU Encode Worker Started"
echo "=========================================="
echo "  Model:                Qwen3.5-35B-A3B (MoE, multimodal)"
echo "  Remote H200:          $IP_REMOTE  (NATS:$PORT_NATS  etcd:$PORT_ETCD)"
echo "  Local B70 fabric IP:  $IP_LOCAL  (NIC $UCX_NIC)"
echo "  Side-channel:         $IP_LOCAL:$SIDE_CHANNEL_PORT"
echo "  KV events port:       $KV_EVENT_PORT"
echo "  XPU (ZE_AFFINITY):    $XPU_DEVICE"
echo "  PID:                  $ENCODE_PID"
echo ""
echo "  UCX_TLS=ze_copy,rc,tcp   (no cuda_ipc, as required)"
echo "  DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read"
echo "  DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read"
echo ""
echo "Log:    $LOG_DIR/encode_xpu_35b_b70.log"
echo "Tail:   tail -f $LOG_DIR/encode_xpu_35b_b70.log"
echo "Ready:  grep -i 'registered\\|ready' $LOG_DIR/encode_xpu_35b_b70.log"
echo "Stop:   kill $ENCODE_PID   # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""

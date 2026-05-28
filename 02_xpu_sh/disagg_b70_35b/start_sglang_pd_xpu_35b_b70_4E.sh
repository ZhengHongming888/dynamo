#!/bin/bash
# Start script for Dynamo SGLang Disaggregated Prefill/Decode (4x Intel B70 XPU Encode Workers)
# Model: Qwen3.5-35B-A3B (MoE, ~67 GB BF16, ~3B activated, multimodal)
# Pair : H200 @ 172.26.46.75 (sc09super21-h200) - matches start_sglang_pd_xpu_35b_b70.sh
#
# This starts: 4 multimodal Encode workers on Intel B70 (Battlemage) XPUs.
# Run this AFTER the H200 side has started NATS / etcd / frontend / P / D workers.
#
# Host    : sc09giga01-b70
# Hardware: 8x Intel(R) Graphics [0xe223] (Battlemage), 4x ConnectX-7 (RoCE)
#
# B70 NUMA topology (XPU -> NUMA -> best NICs):
#   XPU 0..3  (NUMA 0)  -> mlx5_0 (.40) / mlx5_1 (.38)
#   XPU 4..7  (NUMA 2)  -> mlx5_2 (.37) / mlx5_3 (.39)
#
# Encoder-only mode: each worker loads ONLY the vision tower onto the XPU,
# NOT the 67 GiB language model -- that fits in the 32 GiB B70 VRAM.

set -e

# ========================================
# CONFIGURATION
# ========================================

# Remote H200 server (NATS / etcd / frontend / P / D)
export IP_REMOTE=172.26.46.75   # giga01 / sc09super21-h200 (mgmt IP)
export PORT_NATS=14222
export PORT_ETCD=12379

# Local model path (Qwen3.5-35B-A3B, multimodal MoE, ~67 GB BF16)
export MODEL_PATH=${MODEL_PATH:-/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}

# XPU Devices (Battlemage Pro 70). Pick 4 free devices.
# NOTE: On this host XPUs 4-7 are typically busy (~31 GiB in use), so we
# default to NUMA-0 XPUs 0..3. Override via env to use other devices.
export XPU_DEVICE_1=${XPU_DEVICE_1:-0}
export XPU_DEVICE_2=${XPU_DEVICE_2:-1}
export XPU_DEVICE_3=${XPU_DEVICE_3:-2}
export XPU_DEVICE_4=${XPU_DEVICE_4:-3}

# Per-worker NUMA-local NIC + IP. Helper picks NUMA-local NIC from XPU id.
# Mapping:
#   XPU 0,2  -> mlx5_0:1 (192.165.123.40, ens12np0)  NUMA 0
#   XPU 1,3  -> mlx5_1:1 (192.165.123.38, ens10np0)  NUMA 0
#   XPU 4,6  -> mlx5_2:1 (192.165.123.37, ens9np0)   NUMA 2
#   XPU 5,7  -> mlx5_3:1 (192.165.123.39, ens11np0)  NUMA 2
pick_nic() {
    local xpu="$1"
    case "$xpu" in
        0|2)  echo "mlx5_0:1 192.165.123.40 ens12np0" ;;
        1|3)  echo "mlx5_1:1 192.165.123.38 ens10np0" ;;
        4|6)  echo "mlx5_2:1 192.165.123.37 ens9np0"  ;;
        5|7)  echo "mlx5_3:1 192.165.123.39 ens11np0" ;;
        *)    echo "mlx5_0:1 192.165.123.40 ens12np0" ;;
    esac
}

read UCX_NIC_1 IP_LOCAL_1 NDEV_1 <<<"$(pick_nic "$XPU_DEVICE_1")"
read UCX_NIC_2 IP_LOCAL_2 NDEV_2 <<<"$(pick_nic "$XPU_DEVICE_2")"
read UCX_NIC_3 IP_LOCAL_3 NDEV_3 <<<"$(pick_nic "$XPU_DEVICE_3")"
read UCX_NIC_4 IP_LOCAL_4 NDEV_4 <<<"$(pick_nic "$XPU_DEVICE_4")"

# NIXL side-channel ports + KV-event ports (each worker needs unique ports).
# Same scheme as start_sglang_pd_xpu_32b_b70_4E.sh.
export SIDE_CHANNEL_PORT_1=${SIDE_CHANNEL_PORT_1:-22098}
export SIDE_CHANNEL_PORT_2=${SIDE_CHANNEL_PORT_2:-22099}
export SIDE_CHANNEL_PORT_3=${SIDE_CHANNEL_PORT_3:-22100}
export SIDE_CHANNEL_PORT_4=${SIDE_CHANNEL_PORT_4:-22101}
export KV_EVENT_PORT_1=${KV_EVENT_PORT_1:-22080}
export KV_EVENT_PORT_2=${KV_EVENT_PORT_2:-22083}
export KV_EVENT_PORT_3=${KV_EVENT_PORT_3:-22086}
export KV_EVENT_PORT_4=${KV_EVENT_PORT_4:-22089}

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang 4x Intel B70 XPU Encode Workers"
echo "Model: Qwen3.5-35B-A3B (MoE, multimodal)"
echo "Pair : H200 @ $IP_REMOTE"
echo "=========================================="
echo ""
echo "Model path: $MODEL_PATH"
echo ""
printf "  Worker 1: XPU=%s NIC=%s IP=%s side=%s kv=%s\n" \
    "$XPU_DEVICE_1" "$UCX_NIC_1" "$IP_LOCAL_1" "$SIDE_CHANNEL_PORT_1" "$KV_EVENT_PORT_1"
printf "  Worker 2: XPU=%s NIC=%s IP=%s side=%s kv=%s\n" \
    "$XPU_DEVICE_2" "$UCX_NIC_2" "$IP_LOCAL_2" "$SIDE_CHANNEL_PORT_2" "$KV_EVENT_PORT_2"
printf "  Worker 3: XPU=%s NIC=%s IP=%s side=%s kv=%s\n" \
    "$XPU_DEVICE_3" "$UCX_NIC_3" "$IP_LOCAL_3" "$SIDE_CHANNEL_PORT_3" "$KV_EVENT_PORT_3"
printf "  Worker 4: XPU=%s NIC=%s IP=%s side=%s kv=%s\n" \
    "$XPU_DEVICE_4" "$UCX_NIC_4" "$IP_LOCAL_4" "$SIDE_CHANNEL_PORT_4" "$KV_EVENT_PORT_4"
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

if ! timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE/$PORT_NATS" 2>/dev/null; then
    read -p "Control plane unreachable. Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Helper: launch a single encode worker.
# Args: WORKER_NUM XPU_DEVICE UCX_NIC IP_LOCAL SIDE_CHANNEL_PORT KV_EVENT_PORT
launch_encoder() {
    local N="$1" XPU="$2" NIC="$3" IPL="$4" SCP="$5" KVP="$6"
    local LOG="$LOG_DIR/encode_xpu_35b_b70_${N}.log"

    echo ""
    echo "Starting Intel B70 XPU Encode Worker $N (XPU=$XPU, NIC=$NIC, IP=$IPL)..."

    ZE_AFFINITY_MASK=$XPU \
    NATS_SERVER=nats://${IP_REMOTE}:${PORT_NATS} \
    ETCD_ENDPOINTS=http://${IP_REMOTE}:${PORT_ETCD} \
    ETCD_REQUEST_TIMEOUT=600 \
    ETCD_LEASE_TTL=600 \
    DYN_REQUEST_PLANE=tcp \
    TRANSFER_LOCAL=0 \
    PYTHONHASHSEED=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${IPL} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=${SCP} \
    DYN_VLLM_KV_EVENT_PORT=${KVP} \
    UCX_MEMTYPE_CACHE=0 \
    UCX_TLS=ze_copy,rc,tcp \
    UCX_NET_DEVICES=${NIC} \
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
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KVP"'","enable_kv_cache_events":true}' \
        2>&1 | tee -a "$LOG" &
    local PID=$!
    echo "Encode Worker $N started with PID: $PID"
    eval "ENCODE_PID_${N}=$PID"
}

echo ""
echo "Launching 4 encode workers (encoder-only mode)..."

launch_encoder 1 "$XPU_DEVICE_1" "$UCX_NIC_1" "$IP_LOCAL_1" "$SIDE_CHANNEL_PORT_1" "$KV_EVENT_PORT_1"
echo "Waiting 10 seconds before starting next encoder..."
sleep 10

launch_encoder 2 "$XPU_DEVICE_2" "$UCX_NIC_2" "$IP_LOCAL_2" "$SIDE_CHANNEL_PORT_2" "$KV_EVENT_PORT_2"
echo "Waiting 10 seconds before starting next encoder..."
sleep 10

launch_encoder 3 "$XPU_DEVICE_3" "$UCX_NIC_3" "$IP_LOCAL_3" "$SIDE_CHANNEL_PORT_3" "$KV_EVENT_PORT_3"
echo "Waiting 10 seconds before starting next encoder..."
sleep 10

launch_encoder 4 "$XPU_DEVICE_4" "$UCX_NIC_4" "$IP_LOCAL_4" "$SIDE_CHANNEL_PORT_4" "$KV_EVENT_PORT_4"

echo ""
echo "=========================================="
echo "4x Intel B70 XPU Encode Workers Started"
echo "Model: Qwen3.5-35B-A3B (MoE, multimodal)"
echo "=========================================="
echo "  Remote H200: $IP_REMOTE  (NATS:$PORT_NATS  etcd:$PORT_ETCD)"
echo ""
echo "Encoder 1: XPU=$XPU_DEVICE_1 NIC=$UCX_NIC_1 IP=$IP_LOCAL_1 side=$SIDE_CHANNEL_PORT_1 kv=$KV_EVENT_PORT_1 PID=$ENCODE_PID_1"
echo "Encoder 2: XPU=$XPU_DEVICE_2 NIC=$UCX_NIC_2 IP=$IP_LOCAL_2 side=$SIDE_CHANNEL_PORT_2 kv=$KV_EVENT_PORT_2 PID=$ENCODE_PID_2"
echo "Encoder 3: XPU=$XPU_DEVICE_3 NIC=$UCX_NIC_3 IP=$IP_LOCAL_3 side=$SIDE_CHANNEL_PORT_3 kv=$KV_EVENT_PORT_3 PID=$ENCODE_PID_3"
echo "Encoder 4: XPU=$XPU_DEVICE_4 NIC=$UCX_NIC_4 IP=$IP_LOCAL_4 side=$SIDE_CHANNEL_PORT_4 kv=$KV_EVENT_PORT_4 PID=$ENCODE_PID_4"
echo ""
echo "Logs:    $LOG_DIR/encode_xpu_35b_b70_{1,2,3,4}.log"
echo "Tail:    tail -f $LOG_DIR/encode_xpu_35b_b70_{1,2,3,4}.log"
echo "Ready:   grep -i 'registered\\|ready\\|succeeded' $LOG_DIR/encode_xpu_35b_b70_{1,2,3,4}.log"
echo "Stop:    kill $ENCODE_PID_1 $ENCODE_PID_2 $ENCODE_PID_3 $ENCODE_PID_4"
echo "         # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""
echo "Notes:"
echo "  - --encoder-only loads ONLY vision tower, NOT 67 GiB LLM weights."
echo "  - mem-fraction-static=0.5 keeps headroom on 32 GiB B70."
echo "  - UCX_TLS=ze_copy,rc,tcp (no cuda_ipc, as required for XPU)."
echo ""

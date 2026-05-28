#!/bin/bash
# Cross-host disagg encoder-only script for dell06 H200 host
# Pairs with: PD running on giga01 (172.26.46.75)
#
# This host (dell06):  hostname=sc09dell06-nvd, mgmt IP=172.26.46.162
# Remote host (giga01): hostname=sc09super21-h200, mgmt IP=172.26.46.75
#
# This script starts:  Encode worker on local CUDA GPU 5
# Pairs with:          giga01 PD on GPU 4 (started by start_sglang_pd_cuda_35b_giga01.sh)
#                      Optionally also B70 XPU encoders (they coexist via
#                      dynamo round-robin under namespace `dynamo.encoder.generate`)
#
# Model: Qwen3.5-35B-A3B (BF16 MoE multimodal)
# Derived from:
#   ../../02_xpu_sh/disagg_b70_35b/start_sglang_pd_xpu_35b_b70.sh  (encoder pattern)
#   ./start_sglang_pd_cuda_35b_giga01.sh                          (PD-side env reference)
#
# Cross-host setup notes:
#   1. NATS/etcd/frontend run on giga01 (172.26.46.75). DO NOT start them here.
#   2. NIXL data plane goes over RoCE 192.165.123.0/24 fabric.
#      - dell06 GPU 5 is NUMA 2, paired with mlx5_3 (NODE topology, 192.165.123.26).
#      - giga01 PD reaches us at 192.165.123.26:$SIDE_CHANNEL_PORT.
#   3. NIXL transport: forced RoCE-only (no cuda_ipc — encoder is on a different
#      host than PD so cuda_ipc is unusable anyway). Mirrors B70 transport for
#      apples-to-apples comparison vs the B70 1E baseline.
#   4. Distinct ports from PD's:
#      - SIDE_CHANNEL_PORT=20198 (PD uses 20098)
#      - KV_EVENT_PORT     =22181 (PD uses 22081)
#      so that if anything ever runs on the same host they don't collide.
#   5. ENABLE_ENCODER_CACHE=0 to match the B70 setup.
#
# Pre-flight requirements on this host (dell06):
#   - GPU 5 free (nvidia-smi -i 5 should show 0 MiB used)
#   - mlx5_3 ACTIVE 400 Gb/s on 192.165.123.26
#   - giga01 PD already started (i.e. NATS/etcd/PD model is registered)
#   - Firewall on giga01 already opens 14222/12379/20198/22181 to dell06.
#     (Sibling ports 20098/22081 should already be open for the giga01 PD itself.)
#
# To run:
#   cd /hongming/dynamo/01_cuda_sh/disagg_h200_35b
#   bash start_sglang_encode_cuda_35b_dell06.sh
#
# To stop:
#   pkill -f 'dynamo.sglang.*multimodal-encode-worker'

set -e

# ========================================
# CONFIGURATION
# ========================================

# Remote giga01 host (NATS / etcd / frontend / PD)
export IP_REMOTE_MGMT=172.26.46.75      # giga01 mgmt
export IP_REMOTE_ROCE=192.165.123.52    # giga01 RoCE (mlx5_4)

export PORT_NATS=14222
export PORT_ETCD=12379

# Local dell06 fabric IPs / NICs:
#   mlx5_0 -> eno17095np0 -> 192.165.123.25 (NIC0, NUMA 0)
#   mlx5_3 -> eno16695np0 -> 192.165.123.26 (NIC3, NUMA 2)  <-- best for GPU 5
# GPU 5 is NUMA 2, mlx5_3 has NODE topology to GPU 5.
export IP_LOCAL_MGMT=172.26.46.162
export IP_LOCAL_ROCE=192.165.123.26
export UCX_NIC=mlx5_3:1
export LOCAL_NDEV=eno16695np0

# CUDA encoder GPU. GPU 5 is NUMA 2, NODE-paired with mlx5_3.
export CUDA_DEVICE_ENCODER=${CUDA_DEVICE_ENCODER:-5}

# Network/coordination ports — DISTINCT from giga01 PD's 20098/22081.
export SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT:-20198}
export KV_EVENT_PORT=${KV_EVENT_PORT:-22181}

# Increase TCP request plane message size limit for large multimodal payloads.
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB

# Model path (35B-A3B MoE, BF16 multimodal). Same as PD's.
MODEL_PATH=${MODEL_PATH:-/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754}

# Encoder-side mem-fraction. 0.5 mirrors the B70 setting; on H200 with 144 GB
# this leaves >70 GB headroom which is plenty for the vision tower in
# --encoder-only mode. Lower if you need to share GPU 5 with anything else.
ENCODER_MEM_FRACTION=${ENCODER_MEM_FRACTION:-0.5}

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Cross-Host Disagg — Encoder side (dell06)"
echo "Model: Qwen3.5-35B-A3B (BF16 MoE multimodal)"
echo "=========================================="
echo ""
echo "This host (dell06):"
echo "  hostname:    $(hostname)"
echo "  mgmt IP:     $IP_LOCAL_MGMT  (control plane)"
echo "  RoCE IP:     $IP_LOCAL_ROCE  (NIXL data plane, $UCX_NIC -> $LOCAL_NDEV)"
echo "  Encoder GPU: $CUDA_DEVICE_ENCODER  (NUMA 2)"
echo ""
echo "Remote PD (giga01):"
echo "  mgmt IP:     $IP_REMOTE_MGMT      (NATS:$PORT_NATS  etcd:$PORT_ETCD)"
echo "  RoCE IP:     $IP_REMOTE_ROCE      (giga01 mlx5_4)"
echo ""
echo "NIXL data plane:"
echo "  side-ch host:$IP_LOCAL_ROCE:$SIDE_CHANNEL_PORT"
echo "  KV event:    *:$KV_EVENT_PORT"
echo "  transport:   ib,rc,ud,rc_verbs,ud_verbs,cuda_copy  (RoCE-only, no cuda_ipc)"
echo "  mode:        nixl-read"
echo ""

# ========================================
# Pre-flight checks
# ========================================

echo "[pre-flight] checking GPU $CUDA_DEVICE_ENCODER is free..."
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CUDA_DEVICE_ENCODER 2>/dev/null | tr -d ' ')
if [ -z "$GPU_USED" ]; then
    echo "  ERROR: cannot query GPU $CUDA_DEVICE_ENCODER"
    exit 1
fi
if [ "$GPU_USED" -gt 1024 ]; then
    echo "  WARNING: GPU $CUDA_DEVICE_ENCODER already has ${GPU_USED} MiB in use"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
else
    echo "  OK   GPU $CUDA_DEVICE_ENCODER free (${GPU_USED} MiB used)"
fi

echo "[pre-flight] checking RoCE NIC $UCX_NIC is ACTIVE..."
NIC_STATE=$(cat /sys/class/infiniband/${UCX_NIC%:*}/ports/${UCX_NIC#*:}/state 2>/dev/null | awk '{print $2}')
if [ "$NIC_STATE" != "ACTIVE" ]; then
    echo "  WARNING: $UCX_NIC state is '$NIC_STATE', expected ACTIVE"
else
    echo "  OK   $UCX_NIC ACTIVE"
fi

echo "[pre-flight] verifying $LOCAL_NDEV has $IP_LOCAL_ROCE..."
ACTUAL_IP=$(ip -4 -br addr show "$LOCAL_NDEV" 2>/dev/null | awk '{print $3}' | cut -d/ -f1)
if [ "$ACTUAL_IP" != "$IP_LOCAL_ROCE" ]; then
    echo "  WARNING: $LOCAL_NDEV has '$ACTUAL_IP', expected $IP_LOCAL_ROCE"
else
    echo "  OK   $LOCAL_NDEV -> $IP_LOCAL_ROCE"
fi

echo "[pre-flight] checking giga01 control plane..."
for p in $PORT_NATS $PORT_ETCD; do
    if timeout 2 bash -c "cat </dev/null >/dev/tcp/$IP_REMOTE_MGMT/$p" 2>/dev/null; then
        echo "  OK   $IP_REMOTE_MGMT:$p reachable"
    else
        echo "  FAIL $IP_REMOTE_MGMT:$p (PD not started? firewall?)"
    fi
done

echo "[pre-flight] checking giga01 RoCE reachable..."
if ping -c 1 -W 2 $IP_REMOTE_ROCE > /dev/null 2>&1; then
    echo "  OK   $IP_REMOTE_ROCE pingable from $IP_LOCAL_ROCE"
else
    echo "  WARN $IP_REMOTE_ROCE not pingable (RoCE fabric issue?)"
fi

echo "[pre-flight] verifying etcd has the PD model registered..."
if curl -s "http://$IP_REMOTE_MGMT:$PORT_ETCD/v3/kv/range" \
        -X POST -d '{"key":"di8vaW5zdGFuY2Vz","range_end":"di8vaW5zdGFuY2Vz0A=="}' 2>/dev/null \
        | grep -q "Qwen3.5-35B"; then
    echo "  OK   PD model registered in etcd"
else
    echo "  WARN PD model not yet registered in etcd (or curl probe failed)"
    echo "       Make sure giga01 PD has finished loading. The encoder will"
    echo "       still register and start; it just won't have a paired PD until then."
fi

echo "[pre-flight] verifying model path..."
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "  ERROR: model path not found: $MODEL_PATH"
    exit 1
fi
echo "  OK   $MODEL_PATH"

# ========================================
# Start CUDA H200 Encode Worker
# ========================================

echo ""
echo "Starting H200 CUDA Encode Worker on GPU $CUDA_DEVICE_ENCODER..."
echo "Model load can take 5-10 minutes."

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ENCODER \
NATS_SERVER=nats://${IP_REMOTE_MGMT}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_REMOTE_MGMT}:${PORT_ETCD} \
ETCD_REQUEST_TIMEOUT=600 \
ETCD_LEASE_TTL=600 \
DYN_REQUEST_PLANE=tcp \
DYN_LOG=info \
TRANSFER_LOCAL=0 \
PYTHONHASHSEED=0 \
DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT} \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL_ROCE} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=${UCX_NIC} \
UCX_MEMTYPE_CACHE=0 \
DYN_TCP_MAX_MESSAGE_SIZE=${DYN_TCP_MAX_MESSAGE_SIZE} \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
VISION_ENCODE_SERIALIZE=1 \
NIXL_USE_CPU_HOST_MEMORY=0 \
python3 -m dynamo.sglang \
    --model "${MODEL_PATH}" \
    --enable-multimodal \
    --multimodal-encode-worker \
    --encoder-only \
    --chat-template qwen2-vl \
    --dtype auto \
    --kv-cache-dtype auto \
    --mem-fraction-static $ENCODER_MEM_FRACTION \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    > "$LOG_DIR/encode_cuda_35b_dell06.log" 2>&1 &

ENCODE_PID=$!

echo ""
echo "=========================================="
echo "dell06 H200 CUDA Encode Worker Started"
echo "=========================================="
echo "  Model:                Qwen3.5-35B-A3B (BF16 MoE multimodal)"
echo "  Local GPU:            $CUDA_DEVICE_ENCODER  (NUMA 2)"
echo "  Local fabric IP:      $IP_LOCAL_ROCE  (NIC $UCX_NIC, $LOCAL_NDEV)"
echo "  Side-channel:         $IP_LOCAL_ROCE:$SIDE_CHANNEL_PORT"
echo "  KV events port:       $KV_EVENT_PORT"
echo "  PID:                  $ENCODE_PID"
echo ""
echo "  Pair (giga01):"
echo "    NATS:               $IP_REMOTE_MGMT:$PORT_NATS"
echo "    etcd:               $IP_REMOTE_MGMT:$PORT_ETCD"
echo ""
echo "  UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy   (RoCE only)"
echo "  DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read"
echo ""
echo "Log:    $LOG_DIR/encode_cuda_35b_dell06.log"
echo "Tail:   tail -f $LOG_DIR/encode_cuda_35b_dell06.log"
echo "Ready:  grep -i 'registered\\|ready\\|model registered' $LOG_DIR/encode_cuda_35b_dell06.log"
echo "Stop:   kill $ENCODE_PID   # or: pkill -f 'dynamo.sglang.*multimodal-encode-worker'"
echo ""
echo "After ready, send a smoke-test request to giga01 frontend:"
echo "  curl --noproxy '*' http://$IP_REMOTE_MGMT:7001/v1/models"
echo ""

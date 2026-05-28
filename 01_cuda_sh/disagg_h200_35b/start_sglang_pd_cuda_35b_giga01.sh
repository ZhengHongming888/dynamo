#!/bin/bash
# Cross-host disagg PD-only start script for giga01 (this H200 host)
# Pairs with: encoder running on a separate B70 host
#
# This host (giga01):  hostname=sc09super21-h200, IP=172.26.46.75
# This script starts:  NATS, etcd, frontend, PD worker (1 GPU)
# Encoder runs on:     remote B70 host (separate script there)
#
# Model: Qwen3.5-35B-A3B (BF16 MoE multimodal), TP=1
# Derived from: ../disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh
#               ../agg_h200_35b/start_h200_aggregate_epd_server_35b_tp1.sh
#
# 35B-specific deltas vs the 32B-FP8 PD script (key things to know):
#   1. BF16 model (~70 GB weights), no FP8 KV cache (no --kv-cache-dtype fp8_e4m3)
#   2. mem-fraction-static=0.75 — same as agg_h200_35b/_tp1; 0.85 OOM-killed the
#      scheduler during cuda graph capture (35B-specific empirical finding)
#   3. Frontend MUST use --router-mode round-robin, not kv. The kv-router
#      panics with "block_size must be greater than 1" because the hybrid
#      linear-attention layers force sglang page_size=1.
#   4. Hybrid attention backends required:
#        --linear-attn-backend triton  (for the 30 linear-attention layers)
#        --attention-backend fa3       (for the 10 full-attention layers)
#   5. No --chunked-prefill-size flag — sglang chooses default for the hybrid
#      attention model. Forcing 16384 is unnecessary at this model's per-token
#      compute (~3 B activated parameters from MoE).
#
# Pairing with B70 35B encoder:
#   The 35B model is multimodal (image_token_id=248056, has vision_config).
#   The B70 host needs a 35B encoder-only launcher; the 32B B70 encoder
#   script (start_sglang_pd_xpu_32b_b70_4E.sh) won't work for this model
#   because the 35B has different vision tower dimensions.
#
# IMPORTANT: Before running, ensure firewall on giga01 allows from B70:
#   - $PORT_NATS  (NATS, default 14222)
#   - $PORT_ETCD  (etcd client,  12379)
#   - 12380       (etcd peer)
#   - $SIDE_CHANNEL_PORT (NIXL side-channel, 20098)
#   - $KV_EVENT_PORT     (ZMQ KV events, 22081)
#   - RDMA traffic on the chosen NIC (mlx5_4)

set -e

# ========================================
# PROXY (corporate proxy intercepts localhost; bypass for local services)
# ========================================
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
# Bypass proxy for: localhost, this host's mgmt IP, RoCE IP, dell06 + B70 IPs
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.133,192.165.123.52,172.26.46.13,172.26.46.162,172.26.46.172,192.165.123.40,192.165.123.38,192.165.123.37,192.165.123.39,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

# ========================================
# CONFIGURATION
# ========================================

# Two IPs on this host (same as 32B cross-host PD script):
#  - IP_LOCAL_MGMT: management network (172.26.46.0/22 on enp86s0f0).
#    Used for NATS/etcd advertise URLs and frontend HTTP.
#    B70 (172.26.46.13) reaches this for control-plane registration.
#  - IP_LOCAL_ROCE: RoCE fabric (192.165.123.0/24 on mlx5_4 / enp155s0np0).
#    Used as VLLM_NIXL_SIDE_CHANNEL_HOST so the B70 encoder dials this for
#    NIXL data-plane reads. Must match UCX_NIC selection below.
export IP_LOCAL_MGMT=172.26.46.133
export IP_LOCAL_ROCE=192.165.123.52
export IP_LOCAL=$IP_LOCAL_MGMT  # control plane (NATS/etcd/frontend)

export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001

# PD worker GPU. GPU 4 is on NUMA 2, paired with mlx5_4 (also NUMA 2).
# Keeps the PCIe path PIX/NODE instead of cross-NUMA SYS for cross-host RDMA.
export CUDA_DEVICE_PD=4

# Network/coordination ports
export KV_EVENT_PORT=22081
export SIDE_CHANNEL_PORT=20098

# Increase TCP request plane message size limit for large multimodal payloads
# (encoder sends embedding metadata over TCP control plane)
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB
export DYN_HTTP_BODY_LIMIT_MB=256          # frontend HTTP body limit (default 45 MB)

# RDMA NIC. mlx5_4 = enp155s0np0 = 192.165.123.52, NUMA 2, 400 Gb/s NDR (RoCEv2).
# B70 confirmed reachable on this fabric (192.165.123.0/24).
# B70-side recommended pairing: mlx5_0 (192.165.123.40) — same fabric.
export UCX_NIC=mlx5_4:1

# Model path (35B-A3B MoE, BF16 multimodal)
MODEL_PATH="/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"

# Tuning knobs (mirrors agg_h200_35b/_tp1 conventions)
MAX_RUNNING_REQUESTS=40
MEM_FRACTION=0.75

# Result base for downstream bench scripts
export RESULT_BASE=/hongming/res21_crosshost_giga01_35b/h200_pd_b70_encoder_tp1_35b

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Cross-Host Disagg — PD side (giga01)"
echo "Model: Qwen3.5-35B-A3B (BF16 MoE multimodal)"
echo "=========================================="
echo ""
echo "This host (giga01):"
echo "  mgmt IP:     $IP_LOCAL_MGMT  (control plane)"
echo "  RoCE IP:     $IP_LOCAL_ROCE  (NIXL data plane)"
echo "Encoder runs on:    remote B70 host (start it separately)"
echo ""
echo "Configuration:"
echo "  - PD Worker GPU:     $CUDA_DEVICE_PD"
echo "  - Frontend HTTP:     http://$IP_LOCAL_MGMT:$PORT_HTTP"
echo "  - NATS:              nats://$IP_LOCAL_MGMT:$PORT_NATS"
echo "  - etcd:              http://$IP_LOCAL_MGMT:$PORT_ETCD"
echo "  - NIXL side ch:      $IP_LOCAL_ROCE:$SIDE_CHANNEL_PORT  (RoCE)"
echo "  - RDMA NIC:          $UCX_NIC"
echo "  - Max Running:       $MAX_RUNNING_REQUESTS"
echo "  - Mem Fraction:      $MEM_FRACTION  (35B needs <=0.75)"
echo "  - Router mode:       round-robin (kv panics on linear-attn page_size=1)"
echo ""

# ========================================
# Start NATS (bind 0.0.0.0 so B70 encoder can connect)
# ========================================
echo "Starting NATS..."
nats-server -js -a 0.0.0.0 -p $PORT_NATS -m 18222 \
    > "$LOG_DIR/nats_giga01.log" 2>&1 &
NATS_PID=$!
sleep 2

# ========================================
# Start etcd (bind 0.0.0.0 so B70 encoder can register)
# ========================================
echo "Starting etcd..."
rm -rf /tmp/etcd-sglang-pd-35b-giga01-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://$IP_LOCAL:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-sglang-pd-35b-giga01-$$ \
  > "$LOG_DIR/etcd_giga01.log" 2>&1 &
ETCD_PID=$!
sleep 5

# Verify etcd is reachable
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s -o /dev/null "http://localhost:$PORT_ETCD/version"; then
        echo "  etcd reachable on attempt $i"
        break
    fi
    sleep 2
done

# ========================================
# Start Frontend
# ========================================
# Note: --router-mode round-robin (NOT kv).
# The kv-router asserts block_size > 1 and crashes for the 35B hybrid
# linear-attention model (sglang reports page_size=1). Single-worker setup
# anyway, kv-aware routing is meaningless.
echo "Starting Frontend..."
ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD \
ETCD_LEASE_TTL=600 \
ETCD_REQUEST_TIMEOUT=600 \
DYN_REQUEST_PLANE=tcp \
DYN_EVENT_PLANE=zmq \
DYN_LOG=info \
SGLANG_LOG_LEVEL=info \
python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode round-robin \
    > "$LOG_DIR/frontend_giga01.log" 2>&1 &
FRONTEND_PID=$!
sleep 5

# ========================================
# Start PD Worker
# ========================================
# Notes on env vars (same as 32B cross-host script unless flagged 35B-specific):
#  - UCX_TLS: drop cuda_ipc (no NVLink across hosts), keep IB/RC verbs +
#    cuda_copy so RDMA path is selected for cross-host.
#  - DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read: correct var for SGLang backend.
#  - VLLM_NIXL_SIDE_CHANNEL_HOST=$IP_LOCAL_ROCE: B70 encoder dials this RoCE
#    IP for NIXL data-plane (embedding) reads. Must NOT be the mgmt IP.
#  - mem-fraction-static=0.75: 35B BF16 weights ~70 GB; with 0.85 the
#    scheduler OOM-kills during cuda graph capture (verified empirically
#    in the agg sweep session). Drop further if OOM.
#  - max-running-requests=40 keeps parity with agg_h200_35b/_tp1.
#  - --linear-attn-backend triton: required for the 30 linear-attention
#    layers (GDN/KDA path). 35B-specific.
#  - --attention-backend fa3: for the 10 full-attention layers.
#  - No --kv-cache-dtype fp8_e4m3: model is BF16, native KV cache.
#  - No --chunked-prefill-size override: hybrid linear-attn picks its own.

echo "Starting PD Worker on GPU $CUDA_DEVICE_PD..."
echo "Model loading takes ~5-7 min for 35B BF16 (14 shards, ~70 GB)..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_PD \
NATS_SERVER=nats://${IP_LOCAL_MGMT}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL_MGMT}:${PORT_ETCD} \
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
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,P2P \
python3 -m dynamo.sglang \
    --model $MODEL_PATH \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --tensor-parallel-size 1 \
    --mem-fraction-static $MEM_FRACTION \
    --page-size 16 \
    --attention-backend fa3 \
    --linear-attn-backend triton \
    --enable-request-time-stats-logging \
    --show-time-cost \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    > "$LOG_DIR/pd_worker_giga01.log" 2>&1 &
PD_PID=$!

echo "  PD Worker starting (PID: $PD_PID)..."

# ========================================
# Status
# ========================================

echo ""
echo "=========================================="
echo "PD-side services started (waiting for model load)"
echo "=========================================="
echo ""
echo "Process IDs:"
echo "  - NATS:      $NATS_PID"
echo "  - etcd:      $ETCD_PID"
echo "  - Frontend:  $FRONTEND_PID"
echo "  - PD Worker: $PD_PID"
echo ""
echo "Logs:"
echo "  - NATS:      $LOG_DIR/nats_giga01.log"
echo "  - etcd:      $LOG_DIR/etcd_giga01.log"
echo "  - Frontend:  $LOG_DIR/frontend_giga01.log"
echo "  - PD Worker: $LOG_DIR/pd_worker_giga01.log"
echo ""
echo "=========================================="
echo "Next steps on the B70 host (172.26.46.13):"
echo "=========================================="
echo ""
echo "  1. On the B70 encoder script (35B-A3B encoder), set:"
echo "       NATS_SERVER=nats://${IP_LOCAL_MGMT}:${PORT_NATS}"
echo "       ETCD_ENDPOINTS=http://${IP_LOCAL_MGMT}:${PORT_ETCD}"
echo "       VLLM_NIXL_SIDE_CHANNEL_HOST=192.165.123.40   # B70 mlx5_0 RoCE IP"
echo "       UCX_TLS=ze_copy,rc,tcp                        # XPU side (Intel L0)"
echo "       UCX_NET_DEVICES=mlx5_0:1                       # B70-side NIC"
echo "       UCX_MEMTYPE_CACHE=0"
echo "       DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read"
echo "       DYN_TCP_MAX_MESSAGE_SIZE=268435456"
echo ""
echo "     Note: 35B has different vision-tower dimensions than 32B-FP8;"
echo "     B70's 32B encoder script will not work for this model. A 35B-"
echo "     specific encoder script must be created on B70 first."
echo ""
echo "  2. From B70, sanity-check connectivity:"
echo "       nc -zv ${IP_LOCAL_MGMT} ${PORT_NATS}              # NATS over mgmt"
echo "       curl -s http://${IP_LOCAL_MGMT}:${PORT_ETCD}/version"
echo "       nc -zv ${IP_LOCAL_ROCE} ${SIDE_CHANNEL_PORT}      # NIXL side-channel over RoCE"
echo ""
echo "  3. Start B70 encoder (separate script on that host)."
echo ""
echo "=========================================="
echo "Wait for PD model registration..."
echo "=========================================="
for i in {1..120}; do
    sleep 5
    if curl -s http://localhost:$PORT_HTTP/v1/models 2>/dev/null | grep -q "Qwen3.5-35B"; then
        echo ""
        echo "PD model registered. Now start the B70 encoder."
        echo "  tail -f $LOG_DIR/pd_worker_giga01.log"
        exit 0
    fi
    echo "  Waiting... ($((i*5))s elapsed)"
done

echo ""
echo "Timeout waiting for PD model registration."
echo "  tail -100 $LOG_DIR/pd_worker_giga01.log"
exit 1

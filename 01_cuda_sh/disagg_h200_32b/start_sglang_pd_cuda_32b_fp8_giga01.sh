#!/bin/bash
# Cross-host disagg PD-only start script for giga01 (this H200 host)
# Pairs with: encoder running on a separate B70 host
#
# This host (giga01):  hostname=sc09super21-h200, IP=172.26.46.75
# This script starts:  NATS, etcd, frontend, PD worker (1 GPU)
# Encoder runs on:     remote B70 host (separate script there)
#
# Model: Qwen3-VL-32B-Instruct-FP8, TP=1
# Derived from: start_sglang_pd_cuda_32b_fp8.sh and start_disagg_h200_32b_combined.sh
#
# Key differences from the same-host disagg script:
#   1. NATS / etcd bind to 0.0.0.0 (already did) but must be reachable from B70
#   2. UCX_TLS drops cuda_ipc (no shared GPU bus across hosts) — RDMA only
#   3. mem_fraction_static can go higher (no encoder competing for GPU memory)
#   4. Uses DYN_SGL_EMBEDDING_TRANSFER_MODE (correct for SGLang backend)
#
# IMPORTANT: Before running, ensure firewall on giga01 allows from B70:
#   - $PORT_NATS  (NATS, default 14222)
#   - $PORT_ETCD  (etcd client,  12379)
#   - 12380       (etcd peer)
#   - $SIDE_CHANNEL_PORT (NIXL side-channel, 20098)
#   - $KV_EVENT_PORT     (ZMQ KV events, 22081)
#   - RDMA traffic on the chosen NIC (mlx5_0)

set -e

# ========================================
# PROXY (corporate proxy intercepts localhost; bypass for local services)
# ========================================
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export ftp_proxy=http://proxy.ims.intel.com:911
# Bypass proxy for: localhost, this host's mgmt IP, RoCE IP, B70 IPs, dell06 IPs
export no_proxy=0.0.0.0,127.0.0.1,localhost,172.26.46.133,192.165.123.52,172.26.46.13,172.26.46.162,172.26.46.172,192.165.123.40,192.165.123.38,192.165.123.37,192.165.123.39,.intel.com
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

# ========================================
# CONFIGURATION
# ========================================

# Two IPs on this host:
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
# This keeps the PCIe path PIX/NODE instead of cross-NUMA SYS, important for
# cross-host RDMA throughput.
# 2026-05-27: GPU 4 has stale memory from a zombie process (another container).
# Falling back to GPU 5 — same NUMA 2, NODE proximity to mlx5_4 (vs GPU 4's PIX).
export CUDA_DEVICE_PD=5

# Network/coordination ports
export KV_EVENT_PORT=22081
export SIDE_CHANNEL_PORT=20098

# Increase TCP request plane message size limit for large multimodal payloads
# (encoder sends embedding metadata over TCP control plane)
export DYN_TCP_MAX_MESSAGE_SIZE=268435456  # 256 MB
export DYN_HTTP_BODY_LIMIT_MB=256          # frontend HTTP body limit (default 45 MB)

# RDMA NIC. mlx5_4 = enp155s0np0 = 192.165.123.52, NUMA 2, 400 Gb/s NDR (RoCEv2).
# B70 confirmed reachable on this fabric (192.165.123.0/24).
# B70-side recommended pairing: mlx5_0 (192.165.123.40) — same fabric, comparable NIC.
export UCX_NIC=mlx5_4:1

# Result base for downstream bench scripts
export RESULT_BASE=/hongming/res12_crosshost_giga01/h200_pd_b70_encoder_tp1_32b_image8_1080p_np64

# ========================================

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Dynamo SGLang Cross-Host Disagg — PD side (giga01)"
echo "Model: Qwen3-VL-32B-Instruct-FP8"
echo "=========================================="
echo ""
echo "This host (giga01):"
echo "  mgmt IP:     $IP_LOCAL_MGMT  (control plane)"
echo "  RoCE IP:     $IP_LOCAL_ROCE  (NIXL data plane)"
echo "Encoder runs on:    remote B70 host (start it separately)"
echo ""
echo "Configuration:"
echo "  - PD Worker GPU:  $CUDA_DEVICE_PD"
echo "  - Frontend HTTP:  http://$IP_LOCAL_MGMT:$PORT_HTTP"
echo "  - NATS:           nats://$IP_LOCAL_MGMT:$PORT_NATS"
echo "  - etcd:           http://$IP_LOCAL_MGMT:$PORT_ETCD"
echo "  - NIXL side ch:   $IP_LOCAL_ROCE:$SIDE_CHANNEL_PORT  (RoCE)"
echo "  - RDMA NIC:       $UCX_NIC"
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
rm -rf /tmp/etcd-sglang-pd-32b-giga01-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://$IP_LOCAL:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-sglang-pd-32b-giga01-$$ \
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
echo "Starting Frontend..."
ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD \
ETCD_LEASE_TTL=600 \
ETCD_REQUEST_TIMEOUT=600 \
DYN_REQUEST_PLANE=tcp \
DYN_EVENT_PLANE=zmq \
DYN_LOG=debug \
SGLANG_LOG_LEVEL=debug \
python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states \
    > "$LOG_DIR/frontend_giga01.log" 2>&1 &
FRONTEND_PID=$!
sleep 5

# ========================================
# Start PD Worker
# ========================================
# Notes on env vars:
#  - UCX_TLS: drop cuda_ipc (no NVLink across hosts), keep IB/RC verbs + cuda_copy
#    so RDMA path is selected for cross-host.
#  - DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read: correct var for SGLang backend
#    (per SESSION_MEMORY.md — original PD script used DYN_VLLM_* by mistake).
#  - VLLM_NIXL_SIDE_CHANNEL_HOST=$IP_LOCAL_ROCE: B70 encoder dials this RoCE
#    IP for NIXL data-plane (embedding) reads. Must NOT be the mgmt IP.
#  - mem-fraction-static=0.92: cross-host means no encoder on this GPU, so
#    we can push higher than the same-host 0.85. Drop if OOM.
#  - max-running-requests=64 keeps parity with the same-host disagg run we
#    already have results for, easier comparison.

echo "Starting PD Worker on GPU $CUDA_DEVICE_PD..."
echo "Model loading takes ~3-5 min for 32B FP8..."
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
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --multimodal-worker \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 64 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.65 \
    --page-size 16 \
    --chunked-prefill-size 16384 \
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
echo "  1. On the B70 encoder script set:"
echo "       NATS_SERVER=nats://${IP_LOCAL_MGMT}:${PORT_NATS}"
echo "       ETCD_ENDPOINTS=http://${IP_LOCAL_MGMT}:${PORT_ETCD}"
echo "       VLLM_NIXL_SIDE_CHANNEL_HOST=192.165.123.40   # B70 mlx5_0 RoCE IP"
echo "       UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy   # no cuda_ipc"
echo "       UCX_NET_DEVICES=mlx5_0:1                       # B70-side NIC (NUMA-pair to encoder GPU)"
echo "       UCX_MEMTYPE_CACHE=0"
echo "       DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read"
echo "       DYN_TCP_MAX_MESSAGE_SIZE=268435456"
echo ""
echo "  2. From B70, sanity-check connectivity:"
echo "       nc -zv ${IP_LOCAL_MGMT} ${PORT_NATS}              # NATS over mgmt"
echo "       curl -s http://${IP_LOCAL_MGMT}:${PORT_ETCD}/version"
echo "       nc -zv ${IP_LOCAL_ROCE} ${SIDE_CHANNEL_PORT}      # NIXL side-channel over RoCE"
echo "       ucp_perftest ${IP_LOCAL_ROCE} -t tag_lat          # run with no args here first"
echo ""
echo "  3. Start B70 encoder (separate script on that host)."
echo ""
echo "=========================================="
echo "Wait for PD model registration..."
echo "=========================================="
for i in {1..60}; do
    sleep 5
    if curl -s http://localhost:$PORT_HTTP/v1/models 2>/dev/null | grep -q "Qwen3-VL-32B"; then
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

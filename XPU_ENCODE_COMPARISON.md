# XPU Encode Worker Comparison

This document compares the generic XPU script vs Intel XPU-optimized script.

## Scripts Overview

| Script | Purpose | Hardware | Status |
|--------|---------|----------|--------|
| `start_sglang_pd_xpu.sh` | Generic XPU encode worker | Generic XPU | ⚠️ Template only |
| `start_sglang_pd_xpu_intel.sh` | Intel XPU encode worker | Intel Data Center GPU Max | ✅ Production ready |

## Key Differences

### 1. Device Selection

**Generic:**
```bash
XPU_VISIBLE_DEVICES=$XPU_DEVICE  # Generic approach
```

**Intel XPU:**
```bash
ZE_AFFINITY_MASK=$XPU_DEVICE     # Intel Level Zero API
```
- Intel XPUs use Level Zero runtime
- `ZE_AFFINITY_MASK` is the correct way to select Intel GPU devices

### 2. UCX Transport Layer

**Generic:**
```bash
# Not set - assumes defaults
```

**Intel XPU:**
```bash
UCX_TLS=ze_copy,rc,tcp           # XPU memory copy support
UCX_NET_DEVICES=mlx5_1:1         # Specific InfiniBand NIC
UCX_MEMTYPE_CACHE=0              # Disable memory type cache
```
- `ze_copy`: Enables direct Intel XPU memory operations
- Critical for performance with XPU devices

### 3. Vision Encoding Optimizations

**Generic:**
```bash
# Not set
```

**Intel XPU:**
```bash
VISION_ENCODE_SERIALIZE=1        # Serialize vision encoding operations
NIXL_USE_CPU_HOST_MEMORY=1       # Use CPU memory for NIXL transfers
```
- `VISION_ENCODE_SERIALIZE=1`: Prevents race conditions in vision processing
- `NIXL_USE_CPU_HOST_MEMORY=1`: Optimizes memory transfers between XPU and CUDA

### 4. Worker Type Flag

**Generic:**
```bash
--encoder-only                    # Wrong flag!
```

**Intel XPU:**
```bash
--multimodal-encode-worker        # Correct flag for multimodal encoding
```
- `--encoder-only` is for text-only encoding
- `--multimodal-encode-worker` handles images/video

### 5. Model Configuration

**Generic:**
```bash
--dtype auto                      # Auto-detect
# Missing chat template
```

**Intel XPU:**
```bash
--dtype bfloat16                  # XPU-optimized dtype
--chat-template qwen2-vl          # Qwen2-VL specific template
--page-size 16                    # Explicit page size
```
- `bfloat16` is optimal for Intel XPU
- Qwen2-VL requires specific chat template

### 6. Network Configuration

**Generic:**
```bash
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_CUDA}  # Wrong: uses CUDA IP
VLLM_NIXL_SIDE_CHANNEL_PORT=20098
```

**Intel XPU:**
```bash
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL}  # Correct: uses XPU's own IP
VLLM_NIXL_SIDE_CHANNEL_PORT=22098        # Different port
```
- Side channel host must be the XPU server's IP
- Different port to avoid conflicts

### 7. KV Events Configuration

**Generic:**
```bash
# Not configured
```

**Intel XPU:**
```bash
DYN_VLLM_KV_EVENT_PORT=22080
--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:22080","enable_kv_cache_events":true}'
```
- Full ZMQ publisher configuration
- Matches decode worker's KV event port

## Environment Variables Comparison

| Variable | Generic | Intel XPU | Notes |
|----------|---------|-----------|-------|
| `ZE_AFFINITY_MASK` | ❌ | ✅ | **Required** for Intel XPU |
| `UCX_TLS` | ❌ | `ze_copy,rc,tcp` | **Required** for XPU memory ops |
| `UCX_NET_DEVICES` | ❌ | `mlx5_1:1` | Specific to your network |
| `VISION_ENCODE_SERIALIZE` | ❌ | `1` | **Required** for stability |
| `NIXL_USE_CPU_HOST_MEMORY` | ❌ | `1` | **Required** for performance |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `IP_CUDA` ❌ | `IP_LOCAL` ✅ | Must be XPU's IP |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `20098` | `22098` | Different port |
| `DYN_VLLM_KV_EVENT_PORT` | ❌ | `22080` | Needed for coordination |

## Usage Comparison

### Generic Script (Don't Use for Intel XPU)
```bash
# Edit IP_CUDA
nano start_sglang_pd_xpu.sh

# Run (will have issues on Intel XPU)
./start_sglang_pd_xpu.sh
```

### Intel XPU Script (Recommended)
```bash
# Edit configuration at top of script
nano start_sglang_pd_xpu_intel.sh
# Set: IP_REMOTE, IP_LOCAL, XPU_DEVICE, UCX_NIC

# Run
./start_sglang_pd_xpu_intel.sh
```

## Performance Impact

| Missing Feature | Impact | Severity |
|----------------|---------|----------|
| `ZE_AFFINITY_MASK` | Won't use correct GPU | 🔴 Critical |
| `ze_copy` in UCX_TLS | 10-100x slower transfers | 🔴 Critical |
| `VISION_ENCODE_SERIALIZE` | Race conditions, crashes | 🔴 Critical |
| `NIXL_USE_CPU_HOST_MEMORY` | 2-5x slower transfers | 🟡 High |
| `--multimodal-encode-worker` | Wrong worker type | 🔴 Critical |
| `--chat-template qwen2-vl` | Incorrect tokenization | 🟡 High |
| `bfloat16` vs `auto` | Suboptimal precision | 🟢 Medium |

## Recommendation

**Always use `start_sglang_pd_xpu_intel.sh` for Intel XPU devices.**

The generic script (`start_sglang_pd_xpu.sh`) was a template and will not work correctly on Intel XPUs.

## Complete Setup Example

### 1. On CUDA Server
```bash
./start_sglang_pd_cuda.sh
tail -f logs/decode_worker.log  # Wait for "Model registration succeeded"
```

### 2. On Intel XPU Server
```bash
# Edit configuration
nano start_sglang_pd_xpu_intel.sh
# Set:
#   IP_REMOTE=172.26.46.162  (your CUDA server)
#   IP_LOCAL=<your-xpu-ip>
#   XPU_DEVICE=0
#   UCX_NIC=mlx5_1:1  (check your NIC)

# Run
./start_sglang_pd_xpu_intel.sh
tail -f logs/encode_xpu.log  # Monitor startup
```

### 3. Test
```bash
curl -X POST http://172.26.46.162:7001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [...]}'
```

## Network Device Detection

To find your InfiniBand device:
```bash
# List available devices
ibv_devices

# Check device details
ibv_devinfo

# Common values:
# - mlx5_0:1 (first port of first device)
# - mlx5_1:1 (first port of second device)
```

## Troubleshooting

### Wrong Device Selected
```bash
# Check Intel GPU devices
sycl-ls

# Verify ZE_AFFINITY_MASK
ZE_AFFINITY_MASK=0 python3 -c "import intel_extension_for_pytorch as ipex; print(ipex.xpu.device_count())"
```

### UCX Transport Issues
```bash
# Check UCX transports
ucx_info -d

# Verify ze_copy is available
ucx_info -d | grep -i "ze_copy"
```

### NIXL Connection Issues
```bash
# Check if side channel port is open
netstat -tulpn | grep 22098

# Verify XPU can reach CUDA server
ping 172.26.46.162
```

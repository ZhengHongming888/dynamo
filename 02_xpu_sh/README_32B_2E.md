# 32B Model - 2 Encoder (2E) Configuration

## Overview

`start_sglang_pd_xpu_32b_2E.sh` runs **2 multimodal encode workers** with the Qwen3-VL-32B-Instruct-FP8 model on Intel XPU devices.

## Key Features

✅ **Dual Encoder Setup**: 2 independent encode workers for increased throughput  
✅ **encoder_only Mode**: Only loads vision encoder (~2-4GB per device)  
✅ **Separate Resources**: Each encoder has dedicated ports and InfiniBand NICs  
✅ **Load Balancing**: Requests distributed across both encoders  

## Configuration

### Default XPU Devices
- **Encoder 1**: Device 2
- **Encoder 2**: Device 4

### Network Ports
| Encoder | Side Channel | KV Events | InfiniBand NIC |
|---------|--------------|-----------|----------------|
| 1       | 22098        | 22080     | mlx5_1:1       |
| 2       | 22099        | 22083     | mlx5_2:1       |

## Resource Usage

### Memory per Encoder (with encoder_only fix)
- Vision encoder only: **~2-4GB**
- Total for 2 encoders: **~4-8GB**

### Without encoder_only fix (old behavior)
- Vision encoder + full 32B LLM: **~20-24GB**
- Would cause OOM with 2 encoders

## Comparison: 1E vs 2E

| Metric | 1 Encoder (1E) | 2 Encoders (2E) |
|--------|----------------|-----------------|
| **Throughput** | Baseline | ~2x (parallel processing) |
| **Memory** | ~2-4GB | ~4-8GB |
| **XPU Devices** | 1 device | 2 devices |
| **Ports** | 2 ports | 4 ports (2 per encoder) |
| **Startup Time** | ~5 min | ~10-15 min (sequential) |
| **Fault Tolerance** | Single point of failure | Degraded operation if one fails |

## Usage

### Start 2 Encoders
```bash
./start_sglang_pd_xpu_32b_2E.sh
```

### Monitor Both Encoders
```bash
# Terminal 1
tail -f logs/encode_xpu_32b_1.log

# Terminal 2  
tail -f logs/encode_xpu_32b_2.log
```

### Check Readiness
```bash
# Encoder 1
grep -i 'registered\|ready\|succeeded' logs/encode_xpu_32b_1.log

# Encoder 2
grep -i 'registered\|ready\|succeeded' logs/encode_xpu_32b_2.log
```

### Stop Both Encoders
```bash
pkill -f 'dynamo.sglang.*multimodal-encode-worker.*32B'
```

Or use PIDs shown at startup:
```bash
kill <PID_1> <PID_2>
```

## Prerequisites

1. **CUDA side must be running** with the 32B decode worker:
   ```bash
   ./start_sglang_pd_cuda_32b_fp8.sh  # On CUDA server
   ```

2. **encoder_only fix must be applied** (already done):
   - Without this fix, each encoder would try to load ~20GB
   - With the fix, each encoder only loads ~2-4GB

3. **Remote CUDA server reachable**:
   - NATS: `172.26.46.162:14222`
   - etcd: `172.26.46.162:12379`

## Customization

### Change XPU Devices

Edit the script to use different devices:
```bash
export XPU_DEVICE_1=0  # Change from 2 to 0
export XPU_DEVICE_2=6  # Change from 4 to 6
```

Available devices: 0, 1, 2, 3, 4, 5, 6, 7 (8 total)

### Change Network Ports

If ports are already in use:
```bash
export SIDE_CHANNEL_PORT_1=23098  # Different port
export SIDE_CHANNEL_PORT_2=23099  # Different port
export KV_EVENT_PORT_1=23080      # Different port
export KV_EVENT_PORT_2=23083      # Different port
```

### Adjust Memory Fraction

If you need more memory for other processes:
```bash
# In the script, change:
--mem-fraction-static 0.7 \  # to 0.5 or 0.6
```

## Troubleshooting

### Encoder 1 starts but Encoder 2 fails

**Symptom**: First encoder succeeds, second encoder crashes with OOM

**Possible causes**:
1. Not enough available XPU devices
2. encoder_only fix not applied
3. Port conflicts

**Solution**:
```bash
# Check available XPU devices
python3 -c "import torch; print(f'Available XPUs: {torch.xpu.device_count()}')"

# Verify encoder_only fix
grep "encoder_only" /usr/local/lib/python3.12/dist-packages/dynamo/sglang/args.py

# Check port availability
netstat -tuln | grep -E "22098|22099|22080|22083"
```

### Both encoders stuck at "Detected fp8 checkpoint"

**This is normal!** The message appears during model loading but is just informational. Wait 5-10 minutes for loading to complete.

Look for these success messages:
```
[INFO] encode_server.__init__: rank 0 init finish
[INFO] Successfully registered LLM with runtime config
[INFO] Model registration succeeded
```

### High CPU usage during startup

**Normal behavior**: Model loading from disk to XPU memory is CPU-intensive. CPU usage should drop after both encoders finish loading.

### One encoder performing all work

**Check load balancing** via request logs. If imbalanced:
1. Verify both encoders registered successfully
2. Check NATS connection for both
3. Restart both encoders

## Performance Tips

1. **Sequential startup**: The script waits 10 seconds between starting encoders to avoid resource contention

2. **Monitor memory**: Check XPU memory usage:
   ```bash
   watch -n 1 'python3 -c "import torch; [print(f\"Device {i}: {torch.xpu.memory_allocated(i)/1024**3:.2f}GB\") for i in range(torch.xpu.device_count())]"'
   ```

3. **InfiniBand NICs**: Each encoder uses a separate NIC (mlx5_1:1 and mlx5_2:1) to avoid network bandwidth contention

4. **Throughput scaling**: With 2 encoders, expect ~1.8-2x throughput compared to 1 encoder (not exactly 2x due to coordination overhead)

## Related Files

- `start_sglang_pd_xpu_32b.sh` - Single encoder (1E) for 32B model
- `start_sglang_pd_xpu_3b_2E_fixed.sh` - Dual encoder for 3B model  
- `start_sglang_pd_cuda_32b_fp8.sh` - CUDA side for 32B model (runs on remote server)

## Architecture

```
┌─────────────────────────────────────────────┐
│         CUDA Server (172.26.46.162)         │
│  ┌─────────────────────────────────────┐   │
│  │  NATS (14222) + etcd (12379)        │   │
│  │  Frontend + Decode Worker (32B FP8) │   │
│  └─────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │ Network
                   │
┌──────────────────┴──────────────────────────┐
│         XPU Server (Local)                  │
│  ┌─────────────────────────────────────┐   │
│  │  Encode Worker 1 (XPU Device 2)     │   │
│  │  - Vision Encoder Only (~2-4GB)     │   │
│  │  - Side Channel: 22098              │   │
│  │  - InfiniBand: mlx5_1:1             │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Encode Worker 2 (XPU Device 4)     │   │
│  │  - Vision Encoder Only (~2-4GB)     │   │
│  │  - Side Channel: 22099              │   │
│  │  - InfiniBand: mlx5_2:1             │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Success Criteria

Both encoders are ready when you see:

```
[INFO] encode_server.__init__: rank 0 init finish
[INFO] Successfully registered LLM with runtime config  
[INFO] Model registration succeeded; processing queued requests
```

Check both log files:
- `logs/encode_xpu_32b_1.log`
- `logs/encode_xpu_32b_2.log`

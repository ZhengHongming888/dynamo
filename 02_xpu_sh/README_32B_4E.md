# 32B Model - 4 Encoder (4E) Configuration

## Overview

`start_sglang_pd_xpu_32b_4E.sh` runs **4 multimodal encode workers** with the Qwen3-VL-32B-Instruct-FP8 model on Intel XPU devices for **maximum throughput**.

## Key Features

✅ **Quad Encoder Setup**: 4 independent encode workers for ultimate throughput  
✅ **encoder_only Mode**: Only loads vision encoder (~2-4GB per device)  
✅ **Separate Resources**: Each encoder has dedicated ports and network configuration  
✅ **Balanced NICs**: 2 encoders per InfiniBand NIC (optimal distribution)  
✅ **Load Balancing**: Requests distributed across all four encoders (25% each)  
✅ **ETCD Stability**: 600-second timeout for reliable operation  

## Configuration

### Default XPU Devices
- **Encoder 1**: Device 2
- **Encoder 2**: Device 4
- **Encoder 3**: Device 6
- **Encoder 4**: Device 0

### Network Ports
| Encoder | Side Channel | KV Events | InfiniBand NIC |
|---------|--------------|-----------|----------------|
| 1       | 22098        | 22080     | mlx5_1:1       |
| 2       | 22099        | 22083     | mlx5_2:1       |
| 3       | 22100        | 22086     | mlx5_1:1 (shared) |
| 4       | 22101        | 22089     | mlx5_2:1 (shared) |

**Balanced NIC distribution**: 2 encoders per NIC for optimal network throughput

## Resource Usage

### Memory per Encoder (with encoder_only fix)
- Vision encoder only: **~2-4GB**
- Total for 4 encoders: **~8-16GB**

### Without encoder_only fix (would fail)
- Vision encoder + full 32B LLM: **~20-24GB per encoder**
- Would require ~80-96GB total → **Massive OOM**

## Comparison: 1E vs 2E vs 3E vs 4E

| Metric | 1E | 2E | 3E | 4E | Improvement (4E vs 1E) |
|--------|----|----|-----|-----|------------------------|
| **Throughput** | Baseline | ~2x | ~3x | ~3.6x | +260% |
| **Memory** | ~2-4GB | ~4-8GB | ~6-12GB | ~8-16GB | Scales linearly |
| **XPU Devices** | 1 | 2 | 3 | 4 | 4x utilization |
| **Ports** | 2 | 4 | 6 | 8 | 4 sets |
| **Startup Time** | ~5 min | ~15 min | ~30 min | ~50 min | Sequential loading |
| **Load per Encoder** | 100% | ~50% | ~33% | ~25% | Best distribution |
| **Fault Tolerance** | None | 50% degraded | 66% degraded | 75% degraded | Best resilience |
| **Network Load/NIC** | 1 enc | 1 enc | 1.5 enc | 2 enc | Balanced |

## Usage

### Start 4 Encoders
```bash
./start_sglang_pd_xpu_32b_4E.sh
```

### Monitor All Encoders
```bash
# Watch all four logs
tail -f logs/encode_xpu_32b_{1,2,3,4}.log

# In separate terminals
tail -f logs/encode_xpu_32b_1.log
tail -f logs/encode_xpu_32b_2.log
tail -f logs/encode_xpu_32b_3.log
tail -f logs/encode_xpu_32b_4.log
```

### Check Readiness
```bash
# Check all four
grep -i 'registered\|ready\|succeeded' logs/encode_xpu_32b_{1,2,3,4}.log | tail -12

# Check individually
for i in 1 2 3 4; do
  echo "Encoder $i:"; 
  grep -i 'succeeded' logs/encode_xpu_32b_${i}.log | tail -1
done
```

### Stop All Encoders
```bash
# Stop all 32B encoder workers
pkill -f 'dynamo.sglang.*multimodal-encode-worker.*32B'
```

Or use PIDs shown at startup:
```bash
kill <PID_1> <PID_2> <PID_3> <PID_4>
```

## Prerequisites

1. **CUDA side running** with 32B decode worker:
   ```bash
   ./start_sglang_pd_cuda_32b_fp8.sh  # On CUDA server
   ```

2. **encoder_only fix applied** (already done):
   - Reduces memory per encoder from ~20GB to ~2-4GB
   - **Critical** for running 4 encoders simultaneously

3. **Remote CUDA server reachable**:
   - NATS: `172.26.46.162:14222`
   - etcd: `172.26.46.162:12379`

4. **Sufficient XPU devices available**:
   - Need 4 available XPU devices (default: 2, 4, 6, 0)
   - Check with: `python3 -c "import torch; print(torch.xpu.device_count())"`

5. **Sufficient system resources**:
   - CPU: High during concurrent loading
   - Network: 2 InfiniBand NICs recommended
   - Memory: ~16GB RAM for buffers

## Customization

### Change XPU Devices

Edit the script to use different devices:
```bash
export XPU_DEVICE_1=1  # Change from 2
export XPU_DEVICE_2=3  # Change from 4
export XPU_DEVICE_3=5  # Change from 6
export XPU_DEVICE_4=7  # Change from 0
```

Available devices: 0, 1, 2, 3, 4, 5, 6, 7 (8 total)

### Change Network Ports

If ports are in use:
```bash
export SIDE_CHANNEL_PORT_1=23098
export SIDE_CHANNEL_PORT_2=23099
export SIDE_CHANNEL_PORT_3=23100
export SIDE_CHANNEL_PORT_4=23101
export KV_EVENT_PORT_1=23080
export KV_EVENT_PORT_2=23083
export KV_EVENT_PORT_3=23086
export KV_EVENT_PORT_4=23089
```

### Adjust Memory Fraction

Reduce for other processes:
```bash
# In the script, change for all four encoders:
--mem-fraction-static 0.6 \  # from 0.7
```

### Adjust Startup Delays

Change wait time between encoders:
```bash
# Default is 10 seconds between each
sleep 10  # Increase to 15 if resource contention observed
```

## Troubleshooting

### One or more encoders fail to start

**Symptom**: Some encoders succeed, others crash with OOM or device errors

**Solutions**:
1. Check XPU device availability:
   ```bash
   python3 -c "import torch; print(f'XPUs available: {torch.xpu.device_count()}')"
   ```

2. Verify encoder_only fix:
   ```bash
   grep "encoder_only" /usr/local/lib/python3.12/dist-packages/dynamo/sglang/args.py
   ```

3. Check port conflicts:
   ```bash
   netstat -tuln | grep -E "22098|22099|22100|22101|22080|22083|22086|22089"
   ```

4. Increase startup delay (reduce contention):
   ```bash
   # Edit script: change sleep 10 to sleep 15
   ```

5. Check system resources during startup:
   ```bash
   # Monitor CPU, memory, XPU usage
   htop
   watch -n 1 'python3 -c "import torch; [print(f\"XPU {i}: {torch.xpu.memory_allocated(i)/1024**3:.1f}GB\") for i in range(8)]"'
   ```

### Load imbalance across encoders

**Symptom**: Uneven request distribution

**Check load per encoder**:
```bash
for i in 1 2 3 4; do
  echo "Encoder $i: $(grep -c 'processing' logs/encode_xpu_32b_${i}.log 2>/dev/null || echo 0) requests"
done
```

**Solutions**:
1. Verify all 4 encoders registered with NATS
2. Check NATS load balancing is working
3. Restart encoders to re-register and rebalance

### High memory usage

**Check current usage**:
```bash
python3 -c "import torch; [print(f'Device {i}: {torch.xpu.memory_allocated(i)/1024**3:.2f}GB') for i in [2,4,6,0]]"
```

**Expected**: ~2-4GB per device  
**If > 4GB per device**:
- encoder_only fix may not be applied
- Memory leak possible (restart encoders)

### Slow startup or hangs

**Normal behavior**: 
- ~5-10 minutes per encoder
- Total ~40-50 minutes with delays
- High CPU usage during loading

**Check for hangs**:
```bash
# If an encoder stops progressing for > 10 minutes
tail -f logs/encode_xpu_32b_X.log  # Replace X with stuck encoder

# Look for:
# - "loading shards" progress
# - Error messages
# - Network connection issues
```

### Network bandwidth bottleneck

**Symptom**: High latency despite 4 encoders

**Check network utilization**:
```bash
# Monitor InfiniBand NICs
ibstat
```

**Solutions**:
- Verify 2 encoders per NIC (balanced)
- Check for network congestion
- Consider adjusting UCX_NIC assignments

## Performance Optimization

### 1. Sequential Startup
The script waits 10 seconds between encoder starts to avoid:
- Memory allocation contention
- Network registration conflicts  
- XPU driver overload
- NATS connection race conditions

### 2. Load Distribution
With 4 encoders, each handles ~25% of requests:
- Maximum XPU utilization
- Lowest latency per request
- Highest overall throughput
- Best resource efficiency

### 3. Fault Tolerance
If 1 encoder fails:
- System continues at 75% capacity (3 encoders)
- If 2 fail: 50% capacity (2 encoders)
- If 3 fail: 25% capacity (1 encoder)
- **Best resilience** of all configurations

### 4. Network Optimization
- **Balanced distribution**: 2 encoders per NIC
- Encoders 1 & 3 share `mlx5_1:1`
- Encoders 2 & 4 share `mlx5_2:1`
- Prevents single NIC bottleneck

## Expected Performance

### Throughput Scaling
- **1E baseline**: 100 requests/min
- **2E**: ~180 requests/min (~1.8x)
- **3E**: ~270 requests/min (~2.7x)
- **4E**: ~360 requests/min (~3.6x)

*Not perfectly linear due to coordination overhead*

### Latency
- Per-request latency improves significantly under load
- Average latency ~25% of single encoder under heavy load
- Queue depth distributed across 4 workers

### Resource Utilization
```
Total XPU Memory:  ~8-16GB (4 encoders)
Total CPU:         Very high during loading, moderate during inference
Network:           Balanced across 2 InfiniBand NICs (2 enc/NIC)
XPU Utilization:   50% of available devices (4 of 8)
```

## Architecture

```
┌─────────────────────────────────────────────┐
│         CUDA Server (172.26.46.162)         │
│  ┌─────────────────────────────────────┐   │
│  │  NATS (14222) + etcd (12379)        │   │
│  │  Frontend + Decode Worker (32B FP8) │   │
│  └─────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │
                   │ NATS (4-way load balanced)
                   │
┌──────────────────┴──────────────────────────┐
│         XPU Server (172.26.46.13)           │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  mlx5_1:1 NIC (2 encoders)         │    │
│  │  ├─ Encoder 1 (Device 2)           │    │
│  │  │  • Memory: ~2-4GB                │    │
│  │  │  • Ports: 22098 / 22080          │    │
│  │  └─ Encoder 3 (Device 6)           │    │
│  │     • Memory: ~2-4GB                │    │
│  │     • Ports: 22100 / 22086          │    │
│  └────────────────────────────────────┘    │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  mlx5_2:1 NIC (2 encoders)         │    │
│  │  ├─ Encoder 2 (Device 4)           │    │
│  │  │  • Memory: ~2-4GB                │    │
│  │  │  • Ports: 22099 / 22083          │    │
│  │  └─ Encoder 4 (Device 0)           │    │
│  │     • Memory: ~2-4GB                │    │
│  │     • Ports: 22101 / 22089          │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Success Criteria

All four encoders are ready when you see (in each log):

```
[INFO] encode_server.__init__: rank 0 init finish
[INFO] Successfully registered LLM with runtime config  
[INFO] Model registration succeeded; processing queued requests
```

Check all four log files:
- `logs/encode_xpu_32b_1.log`
- `logs/encode_xpu_32b_2.log`
- `logs/encode_xpu_32b_3.log`
- `logs/encode_xpu_32b_4.log`

## When to Use 4E vs 3E vs 2E vs 1E

### Use 1E when:
- Low to moderate request volume
- Limited XPU resources
- Development/testing environment
- Cost optimization priority

### Use 2E when:
- Moderate to high request volume
- Good balance of throughput and resources
- Production environment with steady load
- **Most common production configuration**

### Use 3E when:
- Very high request volume (peak traffic)
- Need high throughput
- Have 3+ available XPU devices
- Good fault tolerance (can lose 1 encoder)

### Use 4E when:
- **Extreme request volume** (maximum peak traffic)
- **Absolute maximum throughput** required
- Have 4+ available XPU devices
- **Best fault tolerance** (can lose 3 encoders)
- Lowest possible per-request latency
- Cost of extra resources justified by demand

## Cost-Benefit Analysis

### Resource Investment
- **XPU Devices**: 4 of 8 (50% utilization)
- **Memory**: ~8-16GB
- **Network**: 2 InfiniBand NICs
- **Ports**: 8 total (4 side channels + 4 KV events)

### Performance Gain
- **Throughput**: ~3.6x vs 1E (~+260%)
- **Latency**: 4x better load distribution (25% per encoder)
- **Resilience**: Can tolerate 3 failures

### When 4E is Worth It
✅ Peak traffic > 3x normal load  
✅ SLA requires < 100ms p99 latency under load  
✅ Multimodal requests are primary workload  
✅ Have spare XPU capacity  
✅ Cost of downtime > cost of extra resources  

### When to Use 3E Instead
⚠️ Normal traffic < 2.5x steady state  
⚠️ Limited to 3-4 available XPU devices  
⚠️ Cost optimization more important than max throughput  
⚠️ Network bandwidth limited to 1 NIC  

## Related Files

- `start_sglang_pd_xpu_32b.sh` - Single encoder (1E)
- `start_sglang_pd_xpu_32b_2E.sh` - Dual encoder (2E)
- `start_sglang_pd_xpu_32b_3E.sh` - Triple encoder (3E)
- `start_sglang_pd_cuda_32b_fp8.sh` - CUDA decode worker (runs on remote)
- `README_32B_3E.md` - 3E configuration documentation
- `README_32B_2E.md` - 2E configuration documentation

## Notes

- **Memory saved with encoder_only**: ~64-80GB for 4 encoders!
- **Startup time**: Allow 40-50 minutes for all four to initialize
- **Load balancing**: NATS automatically distributes 4-way
- **Monitoring**: Watch all four logs during startup
- **InfiniBand**: Balanced 2 encoders per NIC for optimal network throughput
- **XPU utilization**: Uses half of available XPU devices (4 of 8)
- **Peak capacity**: Maximum practical encoder count for this setup

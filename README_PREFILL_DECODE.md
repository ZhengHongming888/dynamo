# Dynamo SGLang Disaggregated Prefill/Decode Setup

This setup separates encode (prefill) and decode operations across CUDA and XPU devices.

## Architecture

```
┌─────────────────────────────────────┐
│  CUDA Server (172.26.46.162)       │
│  ┌─────────────────────────────┐   │
│  │ NATS (14222)                │   │
│  │ etcd (12379)                │   │
│  │ Frontend (7001)             │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │ Decode Worker (CUDA)        │   │
│  │ - Prefill/Decode            │   │
│  │ - Multimodal Worker         │   │
│  │ - Port: 22081               │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
                ↕ (NATS/TCP)
┌─────────────────────────────────────┐
│  XPU Server                         │
│  ┌─────────────────────────────┐   │
│  │ Encode Worker (XPU)         │   │
│  │ - Encoder Only              │   │
│  │ - Multimodal Processing     │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Quick Start

### Step 1: Start CUDA Side (Infrastructure + Decode)

On CUDA server:
```bash
./start_sglang_pd_cuda.sh
```

This starts:
- NATS server (port 14222)
- etcd (port 12379) 
- Frontend (port 7001)
- Decode worker with `--multimodal-worker` flag

Wait for decode worker to be ready (~3 minutes):
```bash
tail -f logs/decode_worker.log
# Wait for: "Model registration succeeded"
```

### Step 2: Start XPU Side (Encode)

On XPU server (or same server with different device):

1. **Edit the script** to set your CUDA server IP:
   ```bash
   nano start_sglang_pd_xpu.sh
   # Update: export IP_CUDA=<your-cuda-server-ip>
   ```

2. **Run the encode worker**:
   ```bash
   ./start_sglang_pd_xpu.sh
   ```

### Step 3: Test

```bash
curl -sS -X POST http://localhost:7001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            }
          },
          {
            "type": "text",
            "text": "Describe what you see in this image."
          }
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq .
```

## Key Differences from Aggregated Mode

| Feature | Aggregated | Prefill/Decode (P/D) |
|---------|------------|----------------------|
| Architecture | Single unified worker | Separate encode (XPU) + decode (CUDA) |
| Flag | No special flag | `--multimodal-worker` on decode |
| Encode worker | N/A | `--encoder-only` on XPU |
| KV Event Port | 22080 | 22081 |
| ETCD Lease | Default | 600s (10 min) |
| Python Hash | Random | Deterministic (PYTHONHASHSEED=0) |

## Configuration Details

### CUDA Side (Decode Worker)
- Uses `--multimodal-worker` flag
- Handles both prefill and decode operations
- Multimodal cache enabled
- Port 22081 for KV events
- TCP request plane
- NIXL embedding transfer mode

### XPU Side (Encode Worker)  
- Uses `--encoder-only` flag
- Processes multimodal inputs (images/video)
- Connects to CUDA server's NATS/etcd
- Transfers embeddings via NIXL

## Monitoring

```bash
# Check decode worker
tail -f logs/decode_worker.log

# Check encode worker (on XPU machine)
tail -f logs/encode_worker.log

# Check frontend
tail -f logs/frontend_pd.log

# Check health
curl http://localhost:7001/health | jq .

# List models
curl http://localhost:7001/v1/models | jq .
```

## Stop Services

### CUDA side:
```bash
pkill -f 'nats-server|etcd.*12379|dynamo.frontend|dynamo.sglang'
```

### XPU side:
```bash
pkill -f 'dynamo.sglang.*encoder-only'
```

## Troubleshooting

1. **Encode worker can't connect to CUDA server**
   - Verify IP_CUDA is correct
   - Check firewall allows ports 14222 (NATS) and 12379 (etcd)
   - Ensure CUDA side services are running

2. **No multimodal processing**
   - Verify encode worker started successfully
   - Check `logs/encode_worker.log` for errors
   - Ensure XPU device is available

3. **Slow inference**
   - Check NIXL side channel connection
   - Verify UCX/InfiniBand settings if using RDMA
   - Monitor network latency between CUDA and XPU servers

## Files

- `start_sglang_pd_cuda.sh` - Start CUDA side (NATS, etcd, frontend, decode worker)
- `start_sglang_pd_xpu.sh` - Start XPU side (encode worker)
- `start_sglang_services.sh` - Original aggregated mode (for comparison)
- `test_sglang_aggregated_epd.sh` - Automated test script

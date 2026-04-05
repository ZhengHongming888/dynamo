# Testing PD Disaggregation on Pure CPU

This guide describes how to test prefill/decode (PD) disaggregation using pure CPU workers after the CPU docker build completes.

## Prerequisites

- CPU docker image built successfully: `dynamo:cpu-vllm-runtime-new`
- NIXL working correctly (verified with `TEST_CPU_BUILD.sh`)
- Model files available (e.g., on shared storage or HuggingFace)

## Test Environment Setup

### 1. Verify Image
```bash
./TEST_CPU_BUILD.sh dynamo:cpu-vllm-runtime-new
```

Expected: All 7 tests pass ✓

### 2. Start Container with Workspace Mount
```bash
./container/run.sh \
  --image dynamo:cpu-vllm-runtime-new \
  --mount-workspace \
  --name dynamo-cpu-test
```

This launches a container with:
- Your code mounted at `/workspace`
- User `dynamo` (non-root)
- Access to NIXL, vLLM, and all dependencies

## Basic PD Disaggregation Test

### Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Prefill Worker │         │  Decode Worker  │
│    (CPU only)   │◄───────►│   (CPU only)    │
│                 │  NIXL   │                 │
│  Port: 8000     │  UCX    │  Port: 8001     │
└─────────────────┘         └─────────────────┘
         │                           │
         └───────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │   Router    │
              │  (if used)  │
              └─────────────┘
```

### Step 1: Start ETCD (State Management)

In the container:
```bash
# Terminal 1: Start etcd
etcd --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://localhost:2379
```

### Step 2: Start NATS (Message Bus)

```bash
# Terminal 2: Start NATS
nats-server -p 4222
```

### Step 3: Start Prefill Worker

```bash
# Terminal 3: Prefill worker
python -m dynamo.vllm \
  --model facebook/opt-125m \
  --role prefill \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --device cpu \
  --kv-cache-dtype auto \
  --port 8000 \
  --etcd-endpoints http://localhost:2379 \
  --nats-url nats://localhost:4222
```

Expected output:
```
INFO: Starting prefill worker on CPU...
INFO: Model loaded: facebook/opt-125m
INFO: NIXL initialized for inter-worker communication
INFO: Waiting for decode worker...
```

### Step 4: Start Decode Worker

```bash
# Terminal 4: Decode worker
python -m dynamo.vllm \
  --model facebook/opt-125m \
  --role decode \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --device cpu \
  --kv-cache-dtype auto \
  --port 8001 \
  --etcd-endpoints http://localhost:2379 \
  --nats-url nats://localhost:4222
```

Expected output:
```
INFO: Starting decode worker on CPU...
INFO: Model loaded: facebook/opt-125m
INFO: NIXL initialized for inter-worker communication
INFO: Connected to prefill worker
INFO: Ready to serve requests
```

### Step 5: Send Test Request

```bash
# Terminal 5: Send inference request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Expected response:
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "facebook/opt-125m",
  "choices": [{
    "text": " there was a ...",
    "index": 0,
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
```

## Verification Checklist

### NIXL Communication
- [ ] Prefill worker logs show: `NIXL agent initialized`
- [ ] Decode worker logs show: `NIXL connected to prefill worker`
- [ ] No CUDA-related errors (e.g., `nixl_cu12` import errors)
- [ ] UCX transport working (check logs for `UCX initialized`)

### vLLM CPU Operation
- [ ] No errors about `at::cpu::L2_cache_size()` 
- [ ] oneDNN backend loaded correctly
- [ ] Model loaded on CPU (no GPU memory allocation)
- [ ] Inference completes successfully

### Performance Metrics
- [ ] Prefill latency: ~100-500ms (depends on model size)
- [ ] Decode latency: ~50-200ms per token
- [ ] KV cache transfer working via NIXL

## Troubleshooting

### Issue: `nixl_cu12` import error
**Solution**: Rebuild image with `--no-cache` and verify only `nixl_cpu` is installed:
```bash
docker run --rm dynamo:cpu-vllm-runtime-new \
  ls /opt/dynamo/venv/lib/python3.12/site-packages/ | grep nixl
```

### Issue: `at::cpu::L2_cache_size()` undefined symbol
**Solution**: Verify L2 cache patch was applied during build:
```bash
grep "Applying CPU L2 cache" /tmp/docker-build-clean.log
```

### Issue: Workers can't communicate via NIXL
**Check**:
1. ETCD is running and accessible
2. NATS is running and accessible
3. Network connectivity between workers
4. UCX libraries are loaded: `ldd /opt/dynamo/venv/lib/python3.12/site-packages/nixl_cpu/_bindings.*.so`

### Issue: Slow inference on CPU
**Optimizations**:
- Set `OMP_NUM_THREADS` to number of physical cores
- Use `LD_PRELOAD` with tcmalloc (already in image)
- Enable AVX512 if supported: `VLLM_CPU_AVX512=true`
- Reduce `max-model-len` to fit in L2 cache

## Advanced: Multi-Instance Test

Run multiple prefill/decode pairs to test scalability:

```bash
# Prefill 1 + Decode 1
python -m dynamo.vllm --role prefill --port 8000 --instance-id worker-1a &
python -m dynamo.vllm --role decode --port 8001 --instance-id worker-1b &

# Prefill 2 + Decode 2
python -m dynamo.vllm --role prefill --port 8002 --instance-id worker-2a &
python -m dynamo.vllm --role decode --port 8003 --instance-id worker-2b &
```

## Success Criteria

✅ **Build Verification**
- Image builds without errors
- Only CPU packages installed (no CUDA)
- NIXL CPU wheel present and working

✅ **Runtime Verification**
- Workers start without errors
- NIXL communication established
- vLLM loads model on CPU

✅ **Functional Verification**
- Inference requests complete successfully
- Prefill/decode disaggregation working
- KV cache transfer via NIXL successful
- No CUDA-related errors in logs

✅ **Performance Verification**
- Inference latency acceptable for CPU
- No memory leaks during extended runs
- Multiple requests handled correctly

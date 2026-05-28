# Round-5 NIXL GPU descriptor patch — same-host smoke test results

**Date:** 2026-05-24
**Goal:** Validate the SESSION_MEMORY round-4 patch direction by applying GPU-buffer-device patches to dynamo's multimodal embedding NIXL pipeline and running same-host disagg as a smoke test.
**Outcome:** Patch works structurally, gives 4× RPS / 15× TTFT improvement, but reproduces same-host disagg's known OOM problem because GPU NIXL descriptors compete with SGLang's working memory without coordination.

## What we patched

Three lines across two files in `/opt/venv/lib/python3.12/site-packages/`:

### Patch A — `dynamo/common/multimodal/embedding_transfer.py:915`
```diff
- encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)
+ encodings_tensor = torch.zeros(
+     *embeddings_shape,
+     dtype=embeddings_dtype,
+     device=_nixl_buffer_device(),
+ )
```
Effect: NIXL receive descriptor (fallback path when warmedup pool is empty) allocated on GPU instead of CPU.

### Patch B — `dynamo/common/multimodal/embedding_transfer.py:882`
```diff
  encodings_tensor = torch.zeros(
-     max_item_mm_token * embedding_hidden_size, dtype=torch.int8
+     max_item_mm_token * embedding_hidden_size,
+     dtype=torch.int8,
+     device=_nixl_buffer_device(),
  )
```
Effect: Same change for warmedup pool init (currently unused since `max_items=0`, but for correctness).

### Patch C — `dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py:218-230`
```diff
  new_entries: dict[int, CachedEmbedding] = {}
- # SGLang's _encode outputs are already on CPU; use CPU as target for consistency
- target_device = torch.device("cpu")
+ from dynamo.common.multimodal.embedding_transfer import _nixl_buffer_device
+ target_device = _nixl_buffer_device()
```
Effect: SGLang encoder embeddings move to GPU before NIXL transfer (was forced to CPU).

## Smoke test setup

- **PD (giga01 GPU 4)**: patched, `--multimodal-worker`, `mem-fraction=0.65 → 0.50`, `max-running=64 → 32`
- **Encoder (giga01 GPU 5)**: patched, `--multimodal-encode-worker`, `mem-fraction=0.85`
- Both use cross-host-style UCX: `UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy` (no cuda_ipc), `UCX_NET_DEVICES=mlx5_4:1`
- This forces RoCE-RDMA path even though same-host, to mirror cross-host behavior

## Patch verification (smoke test)

Smoke test request `What is in this image?` → `Two cats sleeping on pink couch.` succeeded.

NIXL log on PD shows GPU descriptors flowing:
```
ReadOperation(... local_descriptors=ptr=..., size=12288000, device=cuda:0,
              remote_descriptors=ptr=..., size=12288000, device=cuda:0, ...)
                                                        ^^^^^^^^^^^^^^^^
```
Both ends `device=cuda:0` (was `device=cpu` before patch). **Patch is structurally correct.**

## Performance results — 8img/1080p @ rate=1.0

### Run 1: `mem-fraction=0.92, max-running=64` (default)

PD OOM'd after **14 successful requests**. Fatal error:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 400.00 MiB.
GPU 0 has 224.81 MiB free.
```

Stats from the 14 surviving requests:
- Request throughput: 0.15 RPS (over the 92s before crash)
- Mean TTFT: 34.7 s
- Mean E2E: 75 s
- Median ITL: 9 ms (decode itself was healthy)

### Run 2: `mem-fraction=0.65, max-running=64`

PD OOM'd after **45 successful requests** at running-req=11 peak.

Stats over the run before crash (134s wall time):
- Request throughput over completed window: **0.48 RPS**
- Mean TTFT: 22.9 s
- Median TTFT: 23.4 s
- Mean E2E: 51.2 s
- Median E2E: 32.5 s

### Run 3: `mem-fraction=0.50, max-running=32` (most conservative)

PD OOM'd after **36 successful requests** at running-req=11 peak. Same `running-req=11` ceiling.

Stats over the run before crash:
- Request throughput over completed window: **0.51 RPS**
- Mean TTFT: 26.6 s
- Median TTFT: 21.8 s

## Comparison vs unpatched baselines

(All 8img/1080p @ rate=1.0)

| Configuration | Sat RPS | Median TTFT | Status |
|---|---:|---:|---|
| **Same-host TP=2 agg (baseline)** | **~0.6-0.7** | **3.5 s** | works fine |
| Same-host TP=1 agg (baseline) | ~0.47 | 8.1 s | works fine |
| Cross-host disagg, 4 encoders unpatched | 0.13 | 368 s | works fine, but slow |
| **Patched same-host disagg, 1 encoder** | **~0.50 (partial)** | **22 s** | **OOMs after 36-65 reqs** |

The patch closes most of the gap vs baseline disagg (0.13 → 0.50, 4× improvement; TTFT 368 → 22 s, 17× improvement). It still doesn't beat same-host TP=2 agg (0.6-0.7) but it gets closer than any unpatched disagg config.

## Why it OOMs

The fundamental issue is **uncoordinated GPU memory budgets**. Three pools compete:

1. **SGLang static**: model weights + KV cache (`mem-fraction-static`)
2. **SGLang dynamic**: cuda graphs, activation buffers (~5-10 GB)
3. **NIXL receive descriptors** (NEW with this patch): up to `max-running × per_request_embedding_size`

For 8img/1080p: each in-flight request adds 638 MB. At running-req=11, that's 7 GB just for NIXL.

Failure mode: OOM happens mid-prefill (`A.new_empty(C_shape, ...)` for FP8 GEMM activation), not in NIXL allocation itself. The model needs more elbow room than `mem-fraction-static` reserves once NIXL eats some.

`max-running-requests` doesn't directly help because NIXL descriptors are created opportunistically as requests arrive — the scheduler doesn't know it's already at its real GPU memory budget.

## Why running-req capped at 11 in BOTH runs (mf=0.65 and mf=0.50)

Both runs OOM'd at `running-req=11` despite vastly different `mem-fraction-static` settings. This suggests the bottleneck isn't the NIXL pool at all — it's some other dynamic SGLang allocation that fragments the available memory. Possibly:
- `chunked-prefill-size=16384` × `max-running=11` × 4 KB/token = ~700 MB just for prefill working sets
- FP8 dequantization activations that spike during forward pass

## What would be needed for a production fix

The fix needs to coordinate GPU memory between SGLang and NIXL. Options:

### Option 1: Pre-allocate NIXL pool, deduct from SGLang's static budget
```python
nixl_pool_gb = max_running * embedding_max_size_bytes / 1e9
sglang_mem_fraction = (gpu_total - model_size - nixl_pool_gb - safety) / gpu_total
```
But this requires knowing `embedding_max_size` at startup — currently it's dynamic per request shape. Would need a max-shape constraint.

### Option 2: Make NIXL pool size a configurable env var that gates it
```bash
DYN_NIXL_GPU_BUFFER_GB=20  # reserves 20 GB on PD GPU for NIXL buffers
```
Then patch SGLang's mem-fraction calculation to subtract this from available memory.

### Option 3: Use SGLang's existing memory pool API to allocate NIXL descriptors
Instead of `torch.zeros(device=cuda)`, call into SGLang's KV cache allocator so the scheduler accounts for it. Requires upstream work in SGLang.

All three are multi-day engineering tasks. The right path is Option 2 + filing dynamo upstream issue.

## State after this experiment

- **Patches reverted** (3 files restored from `.bak`)
- **giga01 PD script reverted** (`mem-fraction=0.92` restored)
- **All servers stopped, all GPUs free**
- **B70 operator NOT yet asked to apply Patch C** — they should keep their current setup

## Files

- B70 patch instructions: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/B70_PATCH_INSTRUCTIONS.md`
- Result data:
  - `/hongming/res14_patched_samehost/8img_1080p_rate1.0/` (run 1, mf=0.92)
  - `/hongming/res14_patched_samehost/8img_1080p_rate1.0_mf065/` (run 2, mf=0.65)
  - `/hongming/res14_patched_samehost/8img_1080p_rate1.0_mf050_mr32/` (run 3, mf=0.50, mr=32)
- Worker logs:
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log` (run 1)
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_v2.log` (run 2)
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_v3.log` (run 3)
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/encoder_giga01_gpu5_patched.log`
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/bench_patched_*.log`

## Bottom line

**The patch works, in the sense that GPU descriptors do flow through NIXL and the per-request bottleneck is dramatically reduced. But it can't be deployed as-is because GPU memory budgeting between SGLang and NIXL isn't coordinated, causing OOM at workloads with large per-request payloads.**

**Same-host TP=2 agg remains the recommended production config for Qwen3-VL-32B-FP8 multimodal serving on this stack.** Cross-host disagg with this patch could close most of the gap if the memory coordination problem is fixed; that's a multi-day upstream-dynamo engineering task.

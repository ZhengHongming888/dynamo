# Does the vLLM Backend Solve the Per-Request Hand-off Problem?

**Question:** We saw an ~11.5 s per-request hand-off floor in the SGLang+dynamo disagg path. Does dynamo's vLLM backend (`/opt/venv/lib/python3.12/site-packages/dynamo/vllm/`) have the same problem, or is it solved?

**Answer (short):** **Mostly solved, in three meaningful ways.** The vLLM backend has architectural improvements specifically targeting the hand-off bottleneck. We did not benchmark it, but reading the code shows clear differences in approach.

---

## What's the same (still has hand-off)

The fundamental architecture is the same:
- Encoder is a separate process from PD
- Frontend → Encoder via TCP request plane → PD via TCP request plane
- Embeddings transferred via NIXL (cuda_ipc on same host)
- PD calls into vLLM engine to start prefill

So the **basic hand-off skeleton exists in both backends.** But the per-step costs are different.

---

## What's better in the vLLM backend

### Improvement 1: `stage_embeddings=True` is default

In `dynamo/vllm/multimodal_handlers/encode_worker_handler.py` line 357:

```python
self.embedding_sender.send_embeddings(
    embedding_item.embeddings, stage_embeddings=True   # ← already enabled
)
```

In `dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py` line 415:

```python
self.embedding_sender.send_embeddings(precomputed_embeddings)  # stage=False default
```

**Impact:** SGLang's path does an extra `embeddings.clone().detach()` (~160 MB GPU memcopy per request for 8 × 1080p). vLLM avoids this clone by using the original tensor directly.

### Improvement 2: Tensor lifecycle tracking via `send_complete_queue`

The SGLang backend doesn't track when the NIXL transfer completes — it relies on Python GC timing. That's why my earlier attempt to patch SGLang with `stage_embeddings=True` made things worse: the source tensor was freed mid-NIXL-read.

vLLM solves this properly. From `encode_worker_handler.py`:

```python
self.send_complete_queue: asyncio.Queue[tuple[Any, Any]] = asyncio.Queue()
self.send_complete_checker_task = asyncio.create_task(
    self.check_complete(self.send_complete_queue)
)

# ...later, after queueing the transfer:
self.send_complete_queue.put_nowait(
    (transfer_request[1], embedding_item.embeddings)   # keep ref alive
)
```

```python
async def check_complete(self, queue):
    while True:
        transfer_future, embedding = await queue.get()
        if transfer_future is None:
            break
        await transfer_future       # block until NIXL transfer completes
        queue.task_done()           # only NOW the embedding ref is dropped
```

The `embedding_item.embeddings` reference is held in the queue until NIXL confirms transfer is done. **This is the missing safety wrapper that made our SGLang patch unsafe.**

### Improvement 3: Scheduler-authoritative CPU embedding cache (the BIG one)

This is the architectural difference that matters most. vLLM has a `DynamoMultimodalEmbeddingCacheConnector` (in `multimodal_utils/multimodal_embedding_cache_connector.py`) that integrates with vLLM's native `ECConnectorBase` pattern.

**How it works:**

1. **Scheduler-side (PD process):** the scheduler maintains a logical LRU `OrderedDict[hash, size_bytes]` mirroring what's actually cached on the worker side.

2. **For each request, the scheduler decides one of three paths:**
   - **GPU encoder cache hit** → skip everything, use the GPU-resident embedding
   - **CPU embedding cache hit** (`has_cache_item(hash)` returns True) → emit a `load` command; worker copies CPU tensor to GPU on next step (`encoder_cache[mm_hash] = self._cpu_store[mm_hash].to("cuda", non_blocking=True)`)
   - **Cache miss** → schedule encoder compute, then save result to CPU cache (`save_caches`)

3. **Worker-side:** dumb dict of CPU tensors. Receives `loads`/`saves`/`evicts` commands as part of normal `SchedulerOutput`. No independent caching decisions.

**Why this matters for hand-off:**

For repeated images (which is real-world scenario — same image attached to multiple turns of a chat, or duplicated in multi-image requests), **the entire ViT encode + NIXL transfer is skipped.** The PD scheduler simply tells the worker "load tensor X from CPU cache to GPU encoder cache" — a `cudaMemcpyAsync` of a few MB on the same GPU, no inter-process round-trip.

The SGLang backend has a similar embedding cache (`MultimodalEmbeddingCacheManager`) but it's **encoder-side only and not integrated with the SGLang scheduler.** That means:
- PD doesn't know about it
- Every request still goes through full TCP request plane → encoder process → ViT (with cache check inside encoder) → NIXL → PD pipeline
- Cache hit only saves ViT compute (~3 s); doesn't avoid TCP/NIXL hand-off (~11 s)

vLLM's design saves **both** the ViT compute AND the hand-off when the embedding is cached.

### Improvement 4: Detailed per-stage timing (tooling, not perf)

vLLM logs this on every request:
```
[EPD_TIMING] req=... E-CACHE-CHECK=... E-IMAGE-LOAD=... E-PREPROCESS=...
             E-SEM-WAIT=... E-VISION-ENCODE=... E-TO-CPU=... E-NIXL-STAGE=...
             E-TRANSFER=... E-TOTAL=... n_images=...
```

Plus state transitions for the `llm-pd-log-viz` tool:
```
req_id=... Enter State 'E-RUNNING' at <ts>
req_id=... Enter State 'E-CACHE-CHECK' at <ts>
req_id=... Enter State 'E-IMAGE-LOAD' at <ts>
... etc.
```

This means if you actually run the vLLM backend, you can attribute every microsecond of the hand-off to its specific cause. SGLang has nothing equivalent — that's why we had to manually correlate timestamps across two log files.

### Improvement 5: SPLIT_ENCODE for parallel multi-image requests

In `prefill_worker_utils.py` line 157:

```python
encode_batch_size = (
    max(1, len(image_urls) // encode_worker_count)
    if SPLIT_ENCODE
    else len(image_urls)
)
```

When you have multiple encoder workers (DP encoders), a request with 8 images can be split across them in parallel. Per-request hand-off cost amortizes across the parallelism. SGLang's encode handler doesn't have this fan-out logic.

### Improvement 6: `NIXL_USE_CPU_HOST_MEMORY=0` default (direct GPU RDMA)

In `encode_worker_handler.py` line 54-58:
```python
# When set to "1", copy embeddings to CPU host memory before NIXL transfer.
# By default (0), CUDA and XPU tensors stay on device for direct
# GPU memory RDMA. Set to "1" only as a workaround when NIXL
# fails for device tensors.
NIXL_USE_CPU_HOST_MEMORY = int(os.getenv("NIXL_USE_CPU_HOST_MEMORY", 0))
```

vLLM defaults to direct GPU→GPU RDMA via NIXL, with an opt-in CPU staging fallback. The SGLang backend doesn't expose this knob; the descriptor allocation in `embedding_transfer.py` line 882 defaults to CPU tensors:
```python
encodings_tensor = torch.zeros(
    max_item_mm_token * embedding_hidden_size, dtype=torch.int8
)  # default CPU
```

Hard to tell from code review alone what gets transferred where, but the vLLM path is explicit about avoiding CPU staging.

---

## What's NOT solved in vLLM backend either

The base hand-off architecture remains:
- Frontend → Encoder TCP request plane round-trip
- Encoder → PD TCP request plane round-trip (separate from data transfer)
- NIXL setup overhead (`create_readable`, `begin_read`)
- Python async overhead between request boundaries
- PD scheduler still needs to allocate KV slots after embeddings arrive

For requests that **miss the embedding cache** (random images, novel content), vLLM still pays a meaningful per-request hand-off — probably similar magnitude to SGLang's, just without the unnecessary clone.

The cache only helps when there's **request-to-request locality** in the image content. If your workload is "every request has 8 different random images" (our benchmark with `--image-content random`), vLLM's cache hit rate is 0% and the architectural advantage disappears.

---

## Estimated speedup if we ran vLLM instead

For our benchmark (8 random 1080p images, np=64, 5-rate sweep, no cache locality):

- `stage_embeddings=True` clone elimination: **maybe 50-100 ms saved per request** (160 MB GPU memcopy at ~500 GB/s in-GPU bandwidth = ~30 ms, plus async overhead)
- Tensor lifecycle done correctly: avoids the corruption that hurt our patched SGLang attempt
- Scheduler embedding cache: **0% hit rate** in our random-image benchmark, no benefit
- SPLIT_ENCODE: only useful with multiple encoder workers; we use 1
- Detailed timing: dev-tool, not perf
- Direct GPU RDMA: maybe 10-50 ms, hard to estimate from code review

**Best case for vLLM with random images: ~100-200 ms shaved off the ~11.5 s hand-off.** Not a game-changer.

For a workload with **image cache locality** (same images repeated across requests, like a chatbot with conversation history): vLLM's scheduler-cache could **eliminate the entire hand-off** for cached items. That could be a **10×+ speedup** for that workload.

---

## Bottom line

**vLLM backend has cleaner code with proper lifecycle handling and a real architectural improvement (scheduler-integrated CPU embedding cache).** It is meaningfully better than the SGLang backend for workloads with image reuse.

But for our benchmark scenario (random new images every request), vLLM probably wouldn't win by much. **The fundamental per-request hand-off (TCP plane + NIXL + Python async) is still there in both backends** — just slightly faster in vLLM.

To **truly** solve per-request hand-off would require:
1. Pipelined hand-off (overlap encoder N+1 with NIXL N with prefill N-1 across requests) — neither backend does this
2. Co-located encoder + scheduler in one process — neither does this
3. Skip NIXL entirely on same-host (use cuda_ipc shared pool) — neither does this

These are real engineering work, not config knobs.

---

## Files referenced

- vLLM encoder handler: `/opt/venv/lib/python3.12/site-packages/dynamo/vllm/multimodal_handlers/encode_worker_handler.py`
- vLLM PD utilities: `/opt/venv/lib/python3.12/site-packages/dynamo/vllm/multimodal_utils/prefill_worker_utils.py`
- vLLM embedding cache connector: `/opt/venv/lib/python3.12/site-packages/dynamo/vllm/multimodal_utils/multimodal_embedding_cache_connector.py`
- SGLang encoder handler (for comparison): `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py`
- Companion analysis: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/per_request_handoff.md`
- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/vllm_vs_sglang_handoff.md`

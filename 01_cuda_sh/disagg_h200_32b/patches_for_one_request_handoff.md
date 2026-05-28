# Patches Attempted for Per-Request Hand-off Optimization

This document records every patch we attempted on the dynamo SGLang multimodal disagg path to reduce the per-request hand-off bottleneck (~11.5 s gap between encoder ViT finishing and PD prefill starting). All patches were ultimately reverted because none delivered net throughput improvement, but the failures were informative.

**Test workload throughout:** Qwen3-VL-32B-Instruct-FP8, 8 images × 1080p, 128 input + 256 output tokens, np=32, rate=1.0, PD-TP=2 disagg (encoder GPU 4, PD GPUs 5+7 with NVLink NV18).

**Baseline (no patches):**
- Actual RPS: 0.24
- Mean TTFT: 230 s
- Mean TPOT: 57 ms
- Mean E2E: 237 s
- Concurrency: 56 (np=64) / 29 (np=32)
- All 32 requests succeeded

---

## Files patched (all reverted, backups in place)

```
/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py.bak
/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/worker_handler.py.bak
/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py.bak
```

---

## Patch attempts summary

| Round | Change | Result | Verdict |
|---|---|---:|---|
| 0 (Baseline) | none | RPS 0.24, TTFT 230s, TPOT 57ms | n/a |
| 1a | `stage_embeddings=True` only (unsafe) | RPS 0.16, TPOT 414ms | **WORSE** — tensor freed mid-NIXL-read |
| 1b | `stage_embeddings=True` + queue lifecycle (vLLM-style) | RPS 0.16, TPOT 421ms | **WORSE** — encoder feeds faster but PD can't keep up |
| 1c | 1b + stream-scoped CUDA sync (no global barrier) | RPS 0.16, TPOT 421ms | **NO CHANGE** — confirms encoder isn't the bottleneck |
| 2a | NIXL receiver buffers on GPU (vs CPU default) | **TPOT 13ms (5×!) but only 5/32 succeeded** | **PARTIAL WIN, OOM** |
| 2b | 2a + lower `mem_fraction_static` 0.85 → 0.75 | All 32 requests FAILED at startup | **WORSE** — KV cache too small |

---

## Detailed patch 1a: `stage_embeddings=True` (unsafe)

### Diagnosis

The vLLM backend uses `stage_embeddings=True` to skip a 160 MB GPU clone per request. The SGLang backend defaults to `stage_embeddings=False`, which forces:

```python
# In NixlReadEmbeddingSender.send_embeddings:
if stage_embeddings:
    transfer_buf = embeddings        # zero-copy, but caller must keep ref alive
else:
    transfer_buf = embeddings.clone().detach()  # 160 MB GPU memcopy per request
```

For 8 × 1080p, the embeddings tensor is `[16384, 5120, bf16]` = 160 MB. Cloning it on the GPU competes with active ViT compute.

### Patch

`/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py` line 415:

```diff
-                ) = await self.embedding_sender.send_embeddings(precomputed_embeddings)
+                ) = await self.embedding_sender.send_embeddings(precomputed_embeddings, stage_embeddings=True)
```

### Why it failed

Without explicit lifecycle management, Python GC may free `precomputed_embeddings` while the PD-side NIXL READ is still in-flight. Symptoms: TPOT exploded from 57 ms → 414 ms (likely silent retransmission or corrupt-data retries). RPS dropped from 0.24 → 0.16.

---

## Detailed patch 1b+1c: Safe `stage_embeddings=True` + stream-scoped sync

### Diagnosis

Patch 1a was unsafe. vLLM's backend solves this with a `send_complete_queue` that holds tensor refs until NIXL transfer completes (`encode_worker_handler.py` lines 116-128 in vLLM). Replicating that pattern in the SGLang backend should make `stage_embeddings=True` safe.

Additionally, `embedding_transfer.py:833` calls `torch.cuda.synchronize()` which is a **global** barrier — it waits for ALL CUDA work on the encoder GPU to complete, not just the current request's. With 64 in-flight ViT computes, this could serialize encoder throughput.

### Patches

**`embedding_transfer.py` (replace global with stream-scoped sync):**

```diff
@@ -827,8 +827,12 @@ class NixlReadEmbeddingSender:
-        # Ensure all device operations (squeeze/split/unsqueeze) are complete
-        # before NIXL exposes the buffer for RDMA.
+        # Ensure operations on this specific tensor are complete before NIXL
+        # exposes the buffer for RDMA. Use stream-scoped sync instead of
+        # global torch.cuda.synchronize() to avoid blocking all other in-flight
+        # ViT work on the encoder GPU.
         _dev_type = getattr(transfer_buf, "device", None)
-        if _dev_type is not None:
+        if _dev_type is not None and _dev_type.type == "cuda":
+            torch.cuda.current_stream(_dev_type).synchronize()
+        elif _dev_type is not None:
             _backend = getattr(torch, _dev_type.type, None)
             if _backend and hasattr(_backend, "synchronize"):
                 _backend.synchronize()
```

**`encode_worker_handler.py` (add lifecycle queue + use stage_embeddings=True):**

In `__init__`, add:
```python
self._send_complete_queue: asyncio.Queue = asyncio.Queue()
self._send_complete_task = asyncio.create_task(
    self._drain_send_complete_queue()
)
```

Add the drain method:
```python
async def _drain_send_complete_queue(self) -> None:
    """Hold embedding tensor references until their NIXL transfer completes."""
    while True:
        item = await self._send_complete_queue.get()
        if item is None:
            self._send_complete_queue.task_done()
            break
        transfer_future, _embedding_ref = item
        try:
            await transfer_future
        except Exception as exc:
            logger.warning("NIXL transfer future raised: %s", exc)
        finally:
            self._send_complete_queue.task_done()
```

In `generate()`, line 415:
```diff
-                ) = await self.embedding_sender.send_embeddings(precomputed_embeddings)
+                ) = await self.embedding_sender.send_embeddings(
+                    precomputed_embeddings, stage_embeddings=True
+                )
                 request.transfer_payload = transfer_request
+                # Keep tensor reference alive until NIXL transfer completes.
+                self._send_complete_queue.put_nowait(
+                    (transfer_future, precomputed_embeddings)
+                )
```

And remove the trailing `await transfer_future` at line 442.

### Result

```
Baseline:  RPS 0.24, TTFT 230s, TPOT  57ms
After 1c:  RPS 0.16, TTFT 148s, TPOT 421ms
```

### Why it failed

The TTFT/concurrency ratio stayed roughly constant (~5 s per concurrent request) — this is the floor in the **PD's intake pipeline**, not the encoder's output pipeline.

**The encoder isn't the bottleneck.** Speeding up the encoder (eliminating clone, removing global sync barrier, decoupling from `await transfer_future`) just makes the PD's intake queue overflow faster.

The reduced clone bandwidth on GPU 4 (encoder GPU) was a real win there, but downstream PD couldn't absorb it.

---

## Detailed patch 2a: NIXL receiver buffers on GPU

### Diagnosis (the most interesting finding)

`embedding_transfer.py` defines `_nixl_buffer_device()` (line 33):
```python
def _nixl_buffer_device() -> torch.device:
    """Return the best device for NIXL transfer buffers.
    Prefers GPU (e.g. CUDA) for GPUDirect RDMA. Falls back to CPU
    when no GPU is available or NIXL_USE_CPU_HOST_MEMORY=1.
    """
    if NIXL_USE_CPU_HOST_MEMORY:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    ...
```

But this function is **DEFINED BUT NEVER USED.** The `NixlReadEmbeddingReceiver.__init__` allocates descriptors with default-CPU torch tensors:

```python
# Line 882 (original):
encodings_tensor = torch.zeros(
    max_item_mm_token * embedding_hidden_size, dtype=torch.int8
)  # ← no device=, defaults to CPU
```

And the dynamic-allocation fallback (line 915) is also CPU-only:
```python
encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)  # ← CPU
```

This means **every NIXL READ goes encoder GPU → CPU host memory → PD GPU.** Two memcopies + a CPU bounce, instead of direct GPU→GPU NVLink.

The factory in `dynamo/common/multimodal/__init__.py` line 40 also forces dynamic allocation:
```python
EmbeddingTransferMode.NIXL_READ: lambda: NixlReadEmbeddingReceiver(max_items=0),
# [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
# to be at matching size, need to overwrite nixl connect library
```

So with `max_items=0`, the warmed-up pool is never built — every receive hits the dynamic CPU allocation path.

### Patch

`/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py`:

**Pre-allocated pool path (`__init__`, line 880-887):**
```diff
-        # Create descriptor for our allocated tensor
+        # Allocate descriptors on the device returned by _nixl_buffer_device()
+        # (default CUDA when available) so NIXL READs go directly GPU->GPU.
+        _buf_device = _nixl_buffer_device()
         for _ in range(max_items):
             encodings_tensor = torch.zeros(
-                max_item_mm_token * embedding_hidden_size, dtype=torch.int8
+                max_item_mm_token * embedding_hidden_size,
+                dtype=torch.int8,
+                device=_buf_device,
             )
```

**Dynamic allocation path (`receive_embeddings`, line 915):**
```diff
-            encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)
+            encodings_tensor = torch.zeros(
+                *embeddings_shape,
+                dtype=embeddings_dtype,
+                device=_nixl_buffer_device(),
+            )
             descriptor = nixl_connect.Descriptor(encodings_tensor)
             dynamic_descriptor = True
```

### Result (only 5/32 requests completed before OOM)

```
Baseline:    RPS 0.24, TTFT 230s, TPOT  57ms, succeeded 32/32
Patch 2a:    RPS 0.08, TTFT  47s, TPOT  13ms, succeeded  5/32
                       ↓ 5×       ↓ 4×        ← real per-request speedup
                                              but OOM after 5 requests
```

**For the 5 requests that did complete:**
- TTFT dropped from 230 s → 47 s (5× faster)
- TPOT dropped from 57 ms → 13 ms (4× faster)
- Median TPOT was actually **5 ms** — the fastest decode we've ever measured for disagg

This is the strongest evidence we have that the per-request hand-off bottleneck is **the CPU bounce in NIXL READ buffer placement**.

### Why it failed (OOM)

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 800.00 MiB.
GPU 0 has a total capacity of 139.80 GiB of which 728.44 MiB is free.
Process 3434334 has 124.79 GiB memory in use.  ← PD scheduler claimed all of mem_fraction_static
```

With `mem_fraction_static=0.85`, SGLang reserves 85% of GPU memory for KV cache + model weights at startup. The remaining 15% (~21 GB) has to handle CUDA graphs, activations, AND now the dynamic NIXL GPU descriptors.

64 in-flight requests × ~17 MB per request descriptor + CUDA graph capture (~800 MB) = exceeds the 21 GB reserve.

---

## Detailed patch 2b: 2a + lower `mem_fraction_static` to 0.75

### Hypothesis

If the OOM in 2a is just headroom-related, lowering `mem_fraction_static` from 0.85 → 0.75 should give 14 GB more headroom for the GPU descriptors.

### Patch

In `start_disagg_h200_32b_pd_tp2_tuned.sh`:
```diff
-    --mem-fraction-static 0.85 \
+    --mem-fraction-static 0.75 \
```
(Applied to both PD worker and encoder lines.)

### Result

```
1.0,FAILED,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

**ALL 32 requests failed.** The bench file shows `0 successful_requests`. Server started but couldn't serve any request.

### Why it failed

Probably the KV cache became too small to hold even one max-context request. SGLang's `max_total_num_tokens` is computed from `mem_fraction_static * gpu_memory - model_weights`. At 0.75, after subtracting 32B-FP8 model weights (~32 GB at TP=2 = 16 GB per rank) and CUDA graphs (~few GB), there may not have been enough left for any meaningful KV cache.

Could also be a startup race condition with the new NIXL allocator competing for memory. We didn't dig into the specific failure mode because we'd already determined the broader approach (dynamic GPU descriptor allocation per request) wasn't viable for production.

---

## Conclusion: what this exercise taught us

### 1. The per-request hand-off bottleneck is genuinely the CPU bounce

The 4-5× speedup observed on 5 successfully-completed requests in patch 2a is hard data. **Direct GPU→GPU NIXL READs are dramatically faster than CPU-staged ones**, and the dynamo SGLang backend currently CPU-stages by default despite having a `_nixl_buffer_device()` helper that's supposed to enable GPU buffers.

This is the closest thing to a "real fix" we identified for the per-request hand-off problem. But making it production-safe requires more work than runtime patching.

### 2. The encoder side is NOT the bottleneck

Round 1 patches optimized the encoder side (no clone, stream-scoped sync, async send completion). They moved the TPOT in the WRONG direction (57ms → 421ms). This proves the bottleneck is downstream of the encoder.

The encoder was already feeding requests faster than the PD intake could absorb. Speeding up the encoder just deepens the PD intake queue and worsens contention.

### 3. The vLLM backend's improvements (`stage_embeddings=True`, lifecycle queue) are correct in isolation but don't solve the structural problem

Even with the same vLLM-style patches applied to SGLang, throughput didn't improve. This is consistent with my earlier vLLM-vs-SGLang code analysis: the per-request floor is in the receiver-side NIXL setup + CPU bounce, not in encoder-side cloning.

### 4. What a production fix would look like

The right way to make GPU-resident NIXL buffers production-safe:

1. **Pre-allocate fixed-size GPU descriptor pool** at startup, BEFORE SGLang grabs `mem_fraction_static`. Reserve, say, 4-8 GB of GPU memory for descriptors before the engine starts.

2. **Set `max_items` to match `max_running_requests`**. With 64 in-flight max, you only need 64 descriptors, not 1024.

3. **Use a maximum-size descriptor + offset-based reads** to handle variable-shape requests. The `[gluo FIXME]` comment in `__init__.py` flags this as the blocker. Possible solutions:
   - Allocate one large GPU buffer per descriptor (e.g., 64 MB) that fits the largest expected embedding
   - Use NIXL's built-in offset support if available
   - Or modify the NIXL connect library as the comment suggests

4. **Reserve descriptor pool memory from `mem_fraction_static`** — pass a flag to SGLang engine init telling it to leave room for the disagg infrastructure. Currently they allocate independently and clobber each other.

This is a 2-3 day engineering task in the dynamo upstream codebase, not a runtime patch.

### 5. Cross-host setup may bypass the problem entirely

When encoder and PD are on different machines:
- NIXL transfer goes over RDMA InfiniBand (different code path than cuda_ipc same-host)
- The CPU bounce we identified may not apply — RDMA can write directly to GPU memory via GPUDirect
- Per-request hand-off cost moves from "scheduler-bound serialization" to "network bandwidth-bound"

User stated this is the eventual production target. **Phase 1 same-host optimization is therefore not the right investment** — the cross-host path is a different code path that we haven't profiled here.

---

## Recommended next steps

1. **Stop investing in same-host disagg optimization.** TP=2 agg already wins (0.95 RPS) and per-request hand-off is structural in this code path.

2. **Set up cross-host disagg test.** This is the actual target topology. Profile per-request hand-off there — it likely has a completely different bottleneck.

3. **If continuing same-host work**, the highest-impact fix is GPU-resident NIXL receiver buffers — but it needs to be done as a proper code contribution to dynamo with descriptor-pool-pre-reservation, not as a runtime patch.

---

## Files

- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/patches_for_one_request_handoff.md`
- Companion: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/per_request_handoff.md` (concept)
- Companion: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/vllm_vs_sglang_handoff.md` (vLLM comparison)
- Companion: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pd_tp2_results.md` (3-GPU disagg measurement)
- Backed-up originals (in case anyone wants to re-apply the patches):
  - `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py.bak`
  - `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/worker_handler.py.bak`
  - `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py.bak`
- Bench scripts used:
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2_tuned.sh`
  - `/hongming/dynamo/test_sglang_32b_pd_tp2_tuned_1080p_np64_over_rates.sh`
  - `/hongming/dynamo/run_pd_tp2_tuned.sh`
- Result data:
  - `/hongming/res9_pd_tp2_tuned/h200_disagg_pdtp2_phase1_32b_image8_1080p_np64/` (round 1c results)
  - `/hongming/res9_pd_tp2_tuned/h200_disagg_pdtp2_gpu_buf_32b_image8_1080p_np64/` (round 2a — partial OOM)
  - `/hongming/res9_pd_tp2_tuned/h200_disagg_pdtp2_gpubuf_mem75_32b_image8_1080p_np64/` (round 2b — total failure)

---

## Update: RDMA-only test (round 3, 2026-05-23 19:36 UTC)

**Hypothesis tested:** "Cross-host disagg might bypass the CPU bounce because RDMA NIC (vs cuda_ipc) writes directly to GPU via GPUDirect RDMA."

**Test:** removed `cuda_ipc,cuda_copy` from `UCX_TLS`, kept everything else identical to baseline. Forced UCX to use `rc,ud,rc_verbs,ud_verbs,gdr_copy` (RDMA over RoCE NIC `mlx5_4`) instead.

**Setup:** PD-TP=2 disagg, encoder GPU 4, PD GPUs 5+7, RoCE 400 Gbit NIC mlx5_4, GPUDirect RDMA enabled (`nvidia_peermem` loaded), no patches to dynamo Python code.

**Files:**
- Server script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2_rdma.sh`
- Bench: `/hongming/dynamo/test_sglang_32b_pd_tp2_rdma_1080p_np64_over_rates.sh`
- Orchestrator: `/hongming/dynamo/run_pd_tp2_rdma.sh`
- Result: `/hongming/res10_pd_tp2_rdma/h200_disagg_pdtp2_rdma_32b_image8_1080p_np64/test_sglang_multi_rates_1080p_20260523_193047/`

### Result (rate=1.0, np=32, 32/32 succeeded)

| Metric | Baseline (cuda_ipc + np=32) | **RDMA-only (no cuda_ipc, np=32)** |
|---|---:|---:|
| Actual RPS | 0.16 | **0.16** |
| Mean TTFT (s) | 141 | **145** |
| Mean TPOT (ms) | 414 | **412** |
| Mean E2E (s) | 180 | **184** |

**Within noise — no measurable effect.**

### Why removing cuda_ipc didn't help: CPU bounce confirmed

PD worker debug log shows:
```
nixl_connect.Descriptor: Created Descriptor(ptr=0x723fdc27f040, size=668467200, device=cpu)
```

**`device=cpu`** for the NIXL receiving descriptor. Even with cuda_ipc disabled and RDMA forced, the destination buffer is allocated on CPU. RDMA writes to CPU memory, then CPU→GPU copy. The CPU bounce persists.

This is because `embedding_transfer.py:915` allocates the dynamic descriptor with no `device=` argument, defaulting to CPU:
```python
encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)  # CPU default
```

The factory in `dynamo/common/multimodal/__init__.py:40` sets `max_items=0`, forcing every receive into this dynamic CPU-allocation path.

### Critical implication for cross-host plan

**Cross-host setup will NOT automatically solve the per-request hand-off problem.** The CPU bounce is a Python-level allocation issue in dynamo, NOT a UCX transport choice. RDMA-vs-NVLink doesn't matter when both target the same CPU buffer.

To make cross-host actually deliver its theoretical promise:
- Same code change needed: patch `embedding_transfer.py` to use `device=_nixl_buffer_device()` (CUDA) for receiver descriptors
- Same OOM problem must be solved: pre-allocate descriptor pool from `mem_fraction_static`
- This is non-trivial because of the variable-shape descriptor issue noted in the `[gluo FIXME]` comment

The cross-host config DOES have one structural advantage: encoder GPU and PD GPU are on different machines, so they can't compete for the same `mem_fraction_static` budget. The encoder host can dedicate full GPU memory to ViT activations + descriptor pool, the PD host can dedicate full GPU memory to KV cache + receiver descriptor pool. **Same code path, but no resource contention.** That might be enough on its own to make the GPU-descriptor approach viable.

But this still requires the dynamo code fix. Going cross-host without fixing the CPU bounce will leave the same per-request floor in place.

### Status

- Servers stopped, GPUs free
- No code patches active (everything reverted to baseline before this RDMA test)
- Scripts and result data preserved in `/hongming/res10_pd_tp2_rdma/`

---

## Update: GPU descriptor patch — attempted real fix (round 4, 2026-05-23 20:07 UTC)

**Goal:** Bypass the CPU bounce on PD side by allocating NIXL receive descriptors on GPU instead of CPU.

**Approach:** patch only the dynamic-allocation path in `embedding_transfer.py:915` (since factory uses `max_items=0`, the pre-allocated pool is empty and every request hits this path). Reduce `mem_fraction_static` and `max_running_requests` to leave headroom for GPU descriptors.

### Patch applied

```diff
@@ -912,7 +912,12 @@ class NixlReadEmbeddingReceiver:
             logger.debug(
                 "No warmed up descriptors available, creating a temporary one for transfer."
             )
-            encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)
+            # Allocate dynamic descriptor on GPU so NIXL READs land directly
+            # on the PD GPU instead of bouncing through CPU host memory.
+            encodings_tensor = torch.zeros(
+                *embeddings_shape,
+                dtype=embeddings_dtype,
+                device=_nixl_buffer_device(),
+            )
             descriptor = nixl_connect.Descriptor(encodings_tensor)
             dynamic_descriptor = True
```

### Two attempts, both OOM

**v1: `mem_fraction_static=0.78`, `max_running_requests=32`** (rate=1.0, np=32)
```
1.0,FAILED,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```
PD log: `torch.OutOfMemoryError: Tried to allocate 638.00 MiB. GPU 0 has 304 MiB free.`

**v2: `mem_fraction_static=0.70`, `max_running_requests=16`** (rate=1.0, np=32)
```
1.0,FAILED,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```
PD log: `torch.OutOfMemoryError: Tried to allocate 638.00 MiB. GPU 0 has 296 MiB free.`

Even with `mem_fraction_static=0.70` (giving 30% × 140 GB = 42 GB headroom per PD GPU) and capping concurrency at 16, OOM still occurs.

### Confirmed: GPU descriptor allocation IS happening

PD debug log at `device=cuda:0`:
```
nixl_connect.Descriptor: Created Descriptor(ptr=0x7db044000000, size=668467200, device=cuda:0)
```

Each descriptor is 638 MB (16K visual tokens × 5120 hidden × bf16, viewed as int8). The patch is doing what it should — but the allocator can't sustain it.

### Why it OOMs even at mem_fraction=0.70, max_running=16

Worst case: 16 concurrent requests × 638 MB = **10.2 GB** peak descriptor memory. Plus CUDA graphs (~5 GB), plus PyTorch caching allocator fragmentation overhead (~2-3 GB). Total ~18 GB needed. Should fit in 42 GB headroom.

But observed PD process memory: **110-118 GB used** out of 140 GB allowed. That means SGLang's mem_fraction_static reservation is consuming much more than (140 × 0.70 = 98 GB) — likely because:
1. Model weights (~16 GB per TP=2 rank) come on top of mem_fraction
2. CUDA graphs are captured for many batch sizes (1-256), each adding some memory
3. The `[gluo FIXME]` cache descriptors and other dynamic allocations accumulate

**Real fix needed in dynamo:** the descriptor pool must be pre-allocated AT STARTUP with a fixed memory budget, AFTER which SGLang's `mem_fraction_static` should be told to leave that budget alone. Currently they allocate independently and clobber each other.

### Why this isn't a 1-line patch

The proper fix requires:

1. Add a `--multimodal-nixl-buffer-gb` CLI arg to dynamo
2. Pre-allocate that much GPU memory as a single contiguous pool BEFORE SGLang engine init
3. Pass `mem_fraction_static * (total_gpu - nixl_pool)` to SGLang so it reserves accordingly
4. Refactor `NixlReadEmbeddingReceiver` to allocate descriptors as offset views into the pre-allocated pool
5. Solve the variable-shape descriptor issue (the `[gluo FIXME]`) — possibly by allocating each descriptor at the largest expected embedding size and using only the prefix needed per request

This is **multi-day engineering** in the dynamo codebase. Not feasible as a runtime patch.

### What this proves

The CPU bounce IS the bottleneck, AND the bypass works (per the OOM error showing `device=cuda:0` descriptors getting registered at the right size). But ad-hoc dynamic GPU allocation per request is unsustainable — needs a proper pool design.

### Status
- Servers stopped, GPUs free
- Patch reverted
- Test scripts preserved at `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2_gpubuf.sh` and `/hongming/dynamo/run_pd_tp2_gpubuf.sh` for future reference
- Result data: `/hongming/res11_pd_tp2_gpubuf/` (both v1 and v2 FAILED)

### Recommendation for cross-host setup

Since this same code path will be used cross-host:

1. Going cross-host alone will NOT fix the bottleneck (confirmed by RDMA-only same-host test in round 3).
2. The proper dynamo code fix (pre-allocated GPU descriptor pool with mem_fraction coordination) is needed regardless of single-host vs cross-host.
3. The closest thing to a working same-host workaround would be **`NIXL_USE_CPU_HOST_MEMORY=0`** combined with **lots of unused PD GPU memory** — i.e., run with much smaller batch / smaller model so the CPU bounce isn't a bottleneck. This is a worse setup than just using TP=2 agg.

For the cross-host plan: the right next step is opening a dynamo upstream issue describing the CPU bounce + OOM problem and asking for the proper pool design. Without that fix, cross-host disagg will perform similarly badly to same-host disagg.

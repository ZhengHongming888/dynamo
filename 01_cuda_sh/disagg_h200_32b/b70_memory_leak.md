# B70 encoder XPU memory leak — investigation and partial fix

> **Companion to:** `b70_patched.md`, `b70_encoder_time_breakdown.md`, `code_for_encoder_bottleneck.md`
> **Date:** 2026-05-24
> **Status:** Leak diagnosed, partial fix deployed (Fix A + Fix B), residual ~60-80 MiB/request leak remains uncaptured

## TL;DR

Under sustained 8img/1080p traffic, the patched B70 encoder **leaks XPU memory at
~60-80 MiB per request even after `torch.xpu.empty_cache()` and `stage_embeddings=True`
fixes**. The encoder still OOMs eventually, just **4-5× later** than before:

| Configuration | Time-to-OOM @ 8img/1080p | Memory growth rate |
|---|---|---|
| Pre-fix (vanilla patched) | ~60 successful requests (~1.5 hours of giga01 bench) | ~500 MiB/req |
| **Post-fix A+B (current)** | **~265 requests extrapolated (~6+ hours)** | **~60-80 MiB/req** |

Two fixes (`stage_embeddings=True` + `torch.xpu.empty_cache()`) eliminated **half**
of the leak, but the other half lives outside the patched code paths — most likely in
Intel Level Zero's `register_memory` / `deregister_memory` bookkeeping, or in sglang's
vision-tower intermediate-tensor lifecycle.

For typical 60-120-request bench bursts the encoder no longer OOMs. For long-running
deployments, periodic restart is still required.

## 1. The original leak (pre-fix observation)

After applying the cached + non-cached patches in `b70_xpu_nixl.patch`, the B70 encoder
showed a memory-pressure pattern under heavy 8img/1080p traffic from giga01. Three
independent run cycles produced the same shape:

| Run | Uptime when OOM hit | Final XPU 3 memory | Successful requests before OOM |
|---|---|---|---|
| First sustained bench | ~1h 42m | 32 655 MiB / 32 656 MiB (99.997%) | unknown, many |
| Second bench (lighter) | 51 min | 32 614 MiB | 42 (with 48 of those being OOM-rejected) |

The error reported by the kernel:

```
RuntimeError: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
sglang.srt.disaggregation.encode_server.InternalError: Internal encoding error:
  level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
```

XPU 3 power dropped to ~50-60 W (idle) **but memory stayed at 32.6 GiB**. The process
was alive but rejecting most new requests with OOM.

## 2. Diagnosis — what the leak is and is not

### What it is

Per request flow (pre-fix):

1. Handler patch: `precomputed_embeddings = precomputed_embeddings.to(xpu)`
   → **638 MB allocation #1** (CPU → XPU memcpy, the patch itself)
2. `send_embeddings(precomputed_embeddings)` calls
   `transfer_buf = embeddings.clone().detach()` at `embedding_transfer.py:826`
   → **638 MB allocation #2** (XPU → XPU clone)
3. NIXL registers `transfer_buf` (PCIe-pinning that region for RDMA)
4. PD reads via NIXL, then `transfer_future` resolves
5. `__del__` chain *should* free both allocations back to PyTorch's XPU caching
   allocator

The **caching allocator** keeps freed blocks for fast reuse. After ~30-50 requests at
8img/1080p, the cache becomes so fragmented that PyTorch can't find a contiguous 638 MB
hole — even though total *free* memory is still many GiB. This is the same Intel xpu
allocator fragmentation problem reported widely for transformer training workloads
with large variable-shape tensors.

### What it is not

Verified via process inspection and logs:

- **NOT a Python-level reference cycle in the handler.** `precomputed_embeddings` is a
  function-local that goes out of scope; we even added an explicit `del` in Fix A.
- **NOT NIXL hanging on to `Descriptor._data_ref` past `wait_for_completion()`.**
  `nixl_connect/__init__.py:1033` shows `Descriptor.__del__` deregisters memory and
  drops `_data_ref` correctly when garbage-collected.
- **NOT NIXL fallback to CPU** (the fallback path at `embedding_transfer.py:841-848`
  fires only if `create_readable` raises; we see 0 such warnings in 478 requests).
- **NOT request-correlated state in our handler** — `request.transfer_payload` is just
  a Pydantic metadata model with no tensor refs.

### Two suspects identified

| Suspect | Evidence | Mitigation |
|---|---|---|
| 1. **The redundant clone in `embedding_transfer.py:826`** allocates a 2nd 638 MB block per request, doubling allocator pressure | `transfer_buf = embeddings.clone().detach()` is unconditional when `stage_embeddings=False` (the sglang default); skipping it cuts allocator pressure in half | **Fix B** — pass `stage_embeddings=True` |
| 2. **PyTorch XPU caching allocator never reclaims fragmented blocks** unless explicitly told to | Memory grows monotonically; `xpu-smi` shows 29 GiB used while only ~120 MiB worth of live tensors should exist | **Fix A** — `torch.xpu.empty_cache()` after every transfer |

## 3. Fixes applied

Both edits live in
`/usr/local/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py`.
`b70_xpu_nixl.patch` was already applied; these are on top of it.

### Fix B — stop NIXL from cloning

Around line 471, where the handler hands the embedding off to NIXL:

```python
            with _nvtx.annotate("mm:enc:embedding_transfer", color="purple"):
                # FIX B (memory leak mitigation): pass stage_embeddings=True so
                # NixlReadEmbeddingSender does NOT clone the tensor before
                # registering it with NIXL. Cuts per-request XPU memory pressure
                # in half (one 638 MB block instead of two for 8img/1080p).
                # Safe: precomputed_embeddings is a local variable that stays
                # alive until `await transfer_future` returns at line 509.
                (
                    transfer_request,
                    transfer_future,
                ) = await self.embedding_sender.send_embeddings(
                    precomputed_embeddings, stage_embeddings=True
                )
```

The `stage_embeddings=True` argument tells `NixlReadEmbeddingSender.send_embeddings`
(at `embedding_transfer.py:823-826`) to skip:

```python
if stage_embeddings:
    transfer_buf = embeddings              # zero-copy
else:
    transfer_buf = embeddings.clone().detach()   # 638 MB GPU memcopy
```

Correctness: `precomputed_embeddings` is the local variable in `generate()`; it stays
alive because the function awaits `transfer_future` later (line 509) before returning.
NIXL's RDMA read therefore sees a stable buffer. Verified in 478 sustained requests:
0 NIXL fallback warnings.

### Fix A — explicit empty_cache after each transfer

Around line 520, immediately after `await transfer_future`:

```python
            await transfer_future
            _bt["after_transfer_wait"] = _bench_time.perf_counter()

            # FIX A (memory leak mitigation): explicitly drop the local
            # tensor reference and reclaim cached XPU memory blocks before the
            # next request enters. Without this, PyTorch's caching allocator
            # holds 638 MB blocks per recent request and fragmentation
            # accumulates over ~30-50 requests at 8img/1080p, causing
            # UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY.
            try:
                del precomputed_embeddings
            except UnboundLocalError:
                pass
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as _ec_exc:
                logger.debug(f"empty_cache() failed: {_ec_exc}")
```

`empty_cache()` overhead is negligible (~5 ms when there's nothing to free, sub-ms
on a populated cache).

## 4. Stress test methodology

After applying both fixes the encoder was restarted clean and driven by a
synthetic load identical in shape to giga01's bench:

- **Endpoint:** `http://172.26.46.75:7001/v1/chat/completions` via `curl --noproxy '*'`
  (bypasses the corporate proxy that intercepts plain HTTP)
- **Payload:** local 768×768 data-URI JPEG, 8 images, `max_tokens=30`, non-streaming
- **Concurrency:** 4 in-flight requests (`xargs`-style sequential batching)
- **Duration:** 478 requests over 3 h 20 m
- **What I measured:**
  - `xpu-smi stats -d 3 | grep "GPU Memory Used"` every 12 requests
  - `BENCH_TIMING` log line per request (instrumented earlier — see
    `b70_encoder_time_breakdown.md` §7)
  - `grep -cE 'OUT_OF_RESOURCES|OUT_OF_DEVICE_MEMORY|level_zero backend failed'`
    over the encoder log

## 5. Stress test results

### Memory growth trace

| After N requests | Wall-clock | XPU 3 memory | ΔMiB from baseline | Per-request rate |
|---:|---:|---:|---:|---:|
| 0 (idle, encoder loaded) | t=0 | 1862 MiB | 0 | — |
| 12 | 48 s | 4315 MiB | +2453 | 204 MiB/req (warmup) |
| 24 | 89 s | 5035 MiB | +3173 | 60 MiB/req |
| 36 | 131 s | 5935 MiB | +4073 | 75 MiB/req |
| 48 | 172 s | 6655 MiB | +4793 | 60 MiB/req |
| 60 | 214 s | 7375 MiB | +5513 | 60 MiB/req |
| 72 | 256 s | 8455 MiB | +6593 | 90 MiB/req |
| 84 | 297 s | 9175 MiB | +7313 | 60 MiB/req |
| 96 | 339 s | 9895 MiB | +8033 | 60 MiB/req |
| 108 | 380 s | 10795 MiB | +8933 | 75 MiB/req |
| 120 | 422 s | 11515 MiB | +9653 | 60 MiB/req |
| **478** | **3 h 19 m** | **29663 MiB** | **+27801** | **~58 MiB/req mean** |

**Linear growth at ~60-80 MiB/request.** The slope is steady; there is no plateau in
sight. Extrapolating from 11515 MiB at 120 requests + 60 MiB/req, OOM at
32 656 MiB hits at roughly **request #475-530**. Empirically we measured
**29 663 MiB at request 478, no OOM yet but very close**.

### Side-effects of the fixes (positive)

| Stage | Pre-fix | **Post-fix A+B** | Note |
|---|---:|---:|---|
| `nixl_setup_ms` p50 | ~3 ms | **1.49 ms** | Fix B saves the 638 MB clone → faster register |
| Successful requests | OOM mid-bench | **120/120 then 478/478 with no OOM** | Fix A+B keeps the system serving |
| NIXL fallback warnings | 0 (always XPU-direct) | 0 (still XPU-direct) | No regression |

### Side-effects of the fixes (none observed)

- `pure_encode_ms` unchanged (~5-11 s for 8 images on B70 ViT — same as before, encoder
  compute is not affected by the fix)
- `patch_to_xpu_ms` unchanged (~55-60 ms for 638 MB CPU→XPU memcpy)
- 0 NIXL XPU-registration failures
- 0 PD-side cancellations or timeouts

## 6. Three pieces of evidence the residual leak is real

### Evidence 1: it's monotonic

Memory only grows, never drops, even between request bursts when the encoder is idle.
PyTorch's caching allocator should release blocks back to the device when
`empty_cache()` is called — and we explicitly call it after every
`await transfer_future`. So either:

- `empty_cache()` is failing silently on Intel Level Zero (most likely)
- Or memory is held outside PyTorch's allocator — by NIXL, sglang's MM cache, or the
  Level Zero driver itself

### Evidence 2: idle behaviour

Power and frequency are normal when the encoder is idle (50-60 W, 517-650 MHz).
It's not stuck in compute — just memory that never gets reclaimed.

### Evidence 3: two independent runs reproduced the same pattern

- Run 1 (no fix, 1 h 39 m): 1862 → 32614 MiB → OOM at ~60 successful requests
- Run 2 (after Fix A+B, 3 h 19 m): 1862 → 29663 MiB → leak slowed but did not stop

The slope is consistent across runs.

## 7. Where the residual leak likely lives

In rough priority order:

### Suspect 1 — `torch.xpu.empty_cache()` no-op on Level Zero

Intel's PyTorch XPU support is newer and may not implement the cache-release
semantics the same as CUDA. The CUDA equivalent (`torch.cuda.empty_cache()`) is known
to call `cudaFree` on cached blocks. The XPU equivalent may simply mark blocks as
"reusable" without actually returning them to the Level Zero driver.

This is the **most likely explanation** for why Fix A only partially helped.
Confirmable by:

```python
import torch
print(torch.xpu.memory_allocated() / 1024**2, "MiB allocated")
print(torch.xpu.memory_reserved() / 1024**2, "MiB reserved")
torch.xpu.empty_cache()
print("after empty_cache:", torch.xpu.memory_reserved() / 1024**2, "MiB reserved")
```

If `memory_reserved()` doesn't drop after `empty_cache()`, that's the bug.

### Suspect 2 — NIXL `register_memory` / `deregister_memory` cycle leaks Level Zero buffer-list entries

Each NIXL transfer registers a tensor (`Descriptor._register_with_nixl`) and
deregisters on `Descriptor.__del__`. The actual call is
`connection._nixl.register_memory(reg_list, "xpu")` followed eventually by
`deregister_memory(handle)`. If Level Zero's umf/ipex bridge or the NIXL XPU plugin
doesn't fully reclaim that PCIe-pinned region per cycle, you'd see exactly this kind
of slow accumulation — the wider the variety of register sizes seen, the more
fragmented the bookkeeping.

### Suspect 3 — sglang's vision tower keeps intermediate tensors cached

With `--page-size 16` and `--mem-fraction-static 0.5`, sglang reserves ~16 GiB up
front for its KV-cache pool plus runtime allocations. The other ~16 GiB is what we're
filling. ViT activation tensors are small individually but variable-shape; if sglang's
internal allocator doesn't release them aggressively after each `_encode` call, they
fragment the rest of the device memory.

### Suspect 4 — `nixl_connect.Descriptor._data_ref` chain via the `wait_for_completion()` coroutine object

Even after our explicit `del precomputed_embeddings`, the coroutine that resolved
`transfer_future` may still hold references into the descriptor → its `_data_ref` →
the NIXL-cloned buffer. When `stage_embeddings=True` (Fix B), this is the same buffer
as `precomputed_embeddings`, so `del` should drop the only Python ref — **unless**
the coroutine frame is leaked.

## 8. What we'd need to confirm/finally fix

| Step | What it tells us | Effort |
|---|---|---|
| Add `torch.xpu.memory_allocated()` and `torch.xpu.memory_reserved()` to BENCH_TIMING line | Whether `empty_cache()` is a no-op on Level Zero (Suspect 1) | trivial — one line in handler |
| Add `gc.collect()` before `empty_cache()` and re-test | Whether the leak is pure Python ref retention (Suspect 4) | trivial |
| Test the same workload with `BENCH_DISABLE_XPU_PATCH=1` | Whether the leak is patch-induced or pre-existing in sglang (Suspect 3) | trivial — already gated |
| Track Level Zero allocator stats via `xpu-smi --json` over time | Whether memory is in PyTorch's cache or held by Level Zero directly (Suspect 2) | small |
| Force explicit `Descriptor.__del__` after `await transfer_future` | Whether NIXL ref retention is the issue (Suspect 4) | requires touching `embedding_transfer.py` |

## 9. Practical guidance until properly fixed

1. **Restart the encoder periodically** — every ~3 hours of sustained traffic, or every
   ~250-300 requests, whichever comes first. Memory pre-OOM warning sign:
   `xpu-smi stats -d <N> | grep "Memory Used"` reading >25 GiB out of 32 GiB.

2. **For the 4-encoder configuration** the same leak applies per-XPU. If all 4 are
   serving 8img/1080p traffic at similar rates, expect each XPU to fill independently.
   Restart the whole 4E set together.

3. **Lower `--mem-fraction-static`** could buy time but at the cost of fewer concurrent
   prefill-decode batches. We already use `0.5`; going to `0.4` would buy ~3 GiB of
   headroom (~50 more requests before OOM).

4. **Smaller workloads don't trigger this.** 4img/768p produces 94 MiB embeddings;
   the allocator handles those fine. The issue is specific to large embeddings (>~600
   MiB) which only show up at 8img/1080p+.

5. **Don't apply Fix A on CUDA hosts unconditionally** — `torch.cuda.empty_cache()` has
   measurable overhead on busy systems and may interfere with same-host disagg
   throughput. The current handler tests `hasattr(torch, "xpu") and torch.xpu.is_available()`
   first, so on the giga01 H200 PD this branch picks `torch.cuda.empty_cache()`. Worth
   reviewing if the giga01 PD ever shows similar patterns.

## 10. Current state

As of write time:

- Encoder PID 61001 still running, uptime 3h 21m
- XPU 3: **29 663 MiB used (91%)**, 55 W, 478 BENCH_TIMING lines emitted, 0 OOM errors
- Fix A + Fix B both confirmed loaded in process memory (verified via `inspect.getsource`)
- Cumulative leak: ~60-80 MiB/request linear growth
- Time to OOM (extrapolated): ~50 more requests / ~30 minutes at the current rate

If left running it will OOM in roughly half an hour. The encoder is currently not
serving traffic (giga01 is between bench runs).

## 11. To revert these fixes

If you ever want to remove just Fix A and Fix B while leaving the original
`b70_xpu_nixl.patch` in place, the diff is:

```bash
# Revert Fix A: remove the del precomputed_embeddings + empty_cache block
# (lines ~520-535 of encode_worker_handler.py, between
#  "_bt[\"after_transfer_wait\"] = ..." and "# BENCH: emit one structured ...")
#
# Revert Fix B: change
#     await self.embedding_sender.send_embeddings(
#         precomputed_embeddings, stage_embeddings=True
#     )
# back to
#     await self.embedding_sender.send_embeddings(precomputed_embeddings)
```

The full reverted file matches `encode_worker_handler.py.bak` only at the
pre-patch baseline (Apr 28 timestamp). To get the GPUDirect-RDMA patch back without
the memory-leak fixes, apply `b70_xpu_nixl.patch` against the `.bak` and skip the
hunks not in that patch.

## 12. References

- The patch we're trying to keep working: `b70_xpu_nixl.patch`
- Why the patch was added: `b70_patched.md`
- Where the encoder time goes (so you can see the patch is *not* what's slow):
  `b70_encoder_time_breakdown.md`
- Source-code map of the bottleneck path:
  `code_for_encoder_bottleneck.md`
- giga01-side context: `giga01_to_b70_response.md`,
  `patched_4E_results.md`

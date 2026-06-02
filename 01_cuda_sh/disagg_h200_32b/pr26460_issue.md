# PR sgl-project/sglang#26460 — Issues found applying it locally

**PR:** https://github.com/sgl-project/sglang/pull/26460
**Title:** [Intel GPU][Encoder] Add xpu_attn backend for encoder vision attention
**Upstream commit:** `186c058d2906a6cc5db320af208ebdb1cc123f87`
**Local apply date:** 2026-05-28 (see `pr26460_apply.md`)
**Tested live:** 2026-05-30 with 4 encoders on B70 XPU 4–7, against H200 PD on
super21 (172.26.46.133), Qwen3-VL-32B-Instruct-FP8 model.

This document captures every functional and structural issue we hit while
testing the PR end-to-end. The PR was applied successfully (compiles, imports,
class registers, CLI flag accepted) but **fails at first request** when used
with Qwen3-VL.

## TL;DR

PR #26460 adds an `xpu_attn` vision-attention backend that calls
`sgl_kernel.flash_attn.flash_attn_varlen_func`. **It does not work for
Qwen3-VL (head_size=72)** because the underlying SYCL kernel in
`sgl-kernel-xpu` only has compiled paths for `{64, 96, 128, 192, 256, 512}`.
Most other vision-tower models will hit the same wall (Qwen2-VL, Qwen2.5-VL,
many InternVL variants).

For our cross-host disagg workload, encoders **fall back to `triton_attn`**
(the prior default) and the PR is functionally inert.

## Issue 1 (BLOCKER) — Unsupported head_size at first request

### Symptom

Every encoder fails on the first real Qwen3-VL request:

```
ERROR encode_worker_handler.generate: Error processing request:
  Internal encoding error: Unsupported head size 72 for chunk-prefill MHA
  File "/opt/sglang/python/sglang/srt/layers/attention/vision.py", line 1177, in forward
    output = self.qkv_backend.forward(...)
  File "/opt/sglang/python/sglang/srt/layers/attention/vision.py", line 772, in forward
RuntimeError: Unsupported head size 72 for chunk-prefill MHA
sglang.srt.disaggregation.encode_server.InternalError:
    Internal encoding error: Unsupported head size 72 for chunk-prefill MHA
```

Line 772 is inside the new `VisionIntelXPUAttention.forward()`, on the
`flash_attn_varlen_func(q, k, v, **fa_kwargs)` call.

### Root cause

`sgl_kernel.flash_attn.flash_attn_varlen_func` calls `torch.ops.sgl_kernel.fwd.default`,
which dispatches in C++:

```cpp
// /opt/sgl-kernel-xpu/src/sycl/flash_attention.cpp:693-718
TORCH_CHECK(
    params.d == 64 || params.d == 96 || params.d == 128
    || params.d == 192 || params.d == 256 || params.d == 512,
    "Unsupported head size for prefill attention: ", params.d);

switch (params.d) {
  case 64:  DISPATCH_PREFILL_KERNEL(64);  break;
  case 96:  DISPATCH_PREFILL_KERNEL(96);  break;
  case 128: DISPATCH_PREFILL_KERNEL(128); break;
  case 192: DISPATCH_PREFILL_KERNEL(192); break;
  case 256: DISPATCH_PREFILL_KERNEL(256); break;
  case 512: DISPATCH_PREFILL_KERNEL(512); break;
  default:
    TORCH_CHECK(false, "Unsupported head size for prefill attention: ", params.d);
}
```

The kernel was designed for **LLM attention**, where head_size is typically in
{64, 128} on real production models. **Vision towers commonly use head_size=72
or 80**, which aren't in the compiled switch.

The C++ helper `round_up_headdim()` is defined in the same file and computes
`params.d_rounded` (72 → 96), but the dispatch switch uses `params.d` (raw) —
so the rounding infrastructure exists but isn't wired through to make odd
head sizes work via padding.

### Models hit by this issue

| Model family | Vision head size | Will it work with PR #26460? |
|---|---:|---|
| **Qwen3-VL / Qwen2-VL** | **72** | ❌ FAIL (this report) |
| Qwen2.5-VL | 80 | ❌ FAIL (also missing) |
| Pixtral | 64 | ✅ likely OK |
| Llava-OneVision (CLIP-L) | 64 | ✅ likely OK |
| InternVL2 (varies) | varies | ❌ many odd values |

### Why nobody caught this in CI

Looking at the PR description, it was tested with:

> CI Run #26497585947 (Base) — failed
> CI Run #26497585772 (Extra) — failed

The CI was already failing, so head-size compatibility on real vision models
likely was not exercised. The test models in CI may have head_size ∈ {64, 128}
which wouldn't trigger this.

## Issue 2 (LATENT) — Possible kwargs mismatch

The PR's review by `gemini-code-assist` flagged that `flash_attn_varlen_func`
might not accept `sinks` or `window_size` kwargs. **We verified this is a
non-issue at the Python signature level** — both kwargs are accepted by
`/opt/sgl-kernel-xpu/python/sgl_kernel/flash_attn.py`'s
`flash_attn_varlen_func`:

```python
def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    ...
    sinks=None,
    ...
    window_size=(-1, -1),
    ...
):
```

So the kwargs are forwarded to the C++ op as `sinks` and as `window_size[0]/[1]`.
However, **whether the SYCL kernel actually supports non-default values for these
parameters at every (head_size, dtype) combination is not documented**. We
didn't reach this codepath because Issue 1 fires first.

If a model passes `window_size != (-1, -1)` (e.g. Qwen3-VL has windowed-attention
ViT layers), this could be a second-stage failure once Issue 1 is fixed.

## Issue 3 (COSMETIC) — Double `.to()` chain

The PR does:

```python
cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
```

This is two CUDA-stream operations (one dtype cast, one device move) where one
would suffice:

```python
cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32)
```

Microscopic perf loss; harmless. Also flagged by gemini-code-assist.

## Issue 4 (DESIGN) — No automatic fallback

When `--mm-attention-backend xpu_attn` is selected (or auto-picked via
`SGLANG_USE_SGL_XPU=1`) and the kernel rejects a head_size, **the request fails
hard** rather than dropping back to `triton_attn` (which works fine for
head_size=72).

A defensive `_determine_attention_backend` could:

```python
elif _is_xpu and use_intel_xpu_backend():
    backend = (
        "xpu_attn" if model.vision_head_size in {64, 96, 128, 192, 256, 512}
        else "triton_attn"
    )
```

…but this requires the auto-detect path to know the vision head size, which it
doesn't easily today. Alternative: catch `RuntimeError` inside
`VisionIntelXPUAttention.forward()` and re-dispatch through `triton_attn`.

## What it would take to actually fix the kernel

Three options, in order of effort:

### Option A — Python-side padding (cheapest, ~5 min)

Pad q/k/v from head_size=72 to 96 inside `VisionIntelXPUAttention.forward()`
before calling the kernel; slice the output back to 72 afterwards.

**Pros:** No C++ recompile.
**Cons:** Wastes 33% of attention compute (96/72 = 1.33×). May not beat
`triton_attn`'s native d=72 path.

### Option B — Patch C++ dispatch to use `params.d_rounded` (~30 min + SYCL build)

Edit `flash_attention.cpp` to:
1. Pad input tensors from `params.d` to `params.d_rounded` in the kernel
   parameter prep
2. Switch on `params.d_rounded` instead of `params.d`
3. Slice output back to `params.dv` (the V head size)

**Pros:** Works for all odd sizes (72, 80, …) without per-kernel changes.
**Cons:** Same 33% over-compute as Option A. Requires Intel oneAPI / SYCL build
toolchain (`cd /opt/sgl-kernel-xpu/build && cmake .. && make -j`). Build time
~5–15 min on this host.

### Option C — Add native d=72 kernel instantiation (hours/days, recompile)

Add `case 72: DISPATCH_PREFILL_KERNEL(72); break;` and a corresponding template
instantiation in the kernel codegen.

**Pros:** Optimal — no padding overhead.
**Cons:** Hard. SYCL/CUTLASS template instantiation depends on tile sizes,
shared-memory layout, register pressure tuning that may not hold at
non-power-of-2 head sizes. Could either fail to compile or compile to slow code.
Properly upstreaming this to `sgl-kernel-xpu` is the right path.

## Why we did not pursue any fix

For the cross-host disagg workload this investigation cares about, the PR's
upside is small even if it worked perfectly:

1. Per `b70_encoder_time_breakdown.md`, the encoder bottleneck is the entire
   ViT forward (~25 s for 8img/1080p on B70 XPU), not specifically the
   attention layer. Faster attention saves single-digit % of ViT time.
2. Per `comparison_5way_35b.md`, switching from B70 XPU encoders to H200
   encoders gives **22× throughput** on 8img/1080p — this is the structural
   win that matters.
3. The patches that **did** matter for cross-host disagg are
   `b70_xpu_nixl.patch` (encoder embeddings on XPU before NIXL) and
   `h200_cuda_nixl.patch` (PD receive descriptors on cuda:0). Both are applied
   and verified working. PR #26460 is orthogonal to those.

## Status of PR #26460 in our local install

- **Code applied** to `/opt/sglang` (see `pr26460_apply.md`)
- **Functionally inert** — every encoder we ran picked `triton_attn` (the
  default) because we did not pass `--mm-attention-backend xpu_attn` /
  `SGLANG_USE_SGL_XPU=1` after confirming the failure on Issue 1
- **Backups in place** (`vision.py.bak.pre-pr26460`,
  `server_args.py.bak.pre-pr26460`) for clean revert if desired

## Verification trail

Live test on 2026-05-30 19:19 UTC, 4 encoders, XPU 4–7:

```
[2026-05-30T19:20:17Z INFO] Using xpu_attn as multimodal attention backend.
… encoders register OK in etcd, all 4 alive on XPUs 4–7 …
[2026-05-30T19:31:26Z ERROR] encode_worker_handler.generate:
   Error processing request: Internal encoding error:
   Unsupported head size 72 for chunk-prefill MHA
```

Re-launched at 19:43 UTC without `MM_ATTN_BACKEND` env, encoders auto-picked
`triton_attn`, proceeded normally. Stopped at 19:53 UTC.

## Files

- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_issue.md`
- Application notes: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_apply.md`
- Local diff: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_local.patch`
- Source of the head-size restriction:
  - Python wrapper: `/opt/sgl-kernel-xpu/python/sgl_kernel/flash_attn.py`
  - C++ dispatch: `/opt/sgl-kernel-xpu/src/sycl/flash_attention.cpp` (lines
    412–425 for `round_up_headdim`, 693–718 for the prefill switch)
- Companion docs:
  - `b70_encoder_time_breakdown.md` (attention is one of many ViT sub-ops; not
    the dominant cost)
  - `comparison_5way_35b.md` (the structural win is encoder hardware, not
    encoder kernels)
  - `b70_patched.md`, `b70_xpu_nixl.patch` (the encoder-side patches that
    actually mattered for cross-host disagg)

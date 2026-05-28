# 8 Images @ 4K — Test Blocked by Dynamo Frontend HTTP Body Limit

**Date:** 2026-05-22
**Goal:** Run 8 images at 4K resolution (3840×2160) to push the workload further toward ViT-bound, where disagg should structurally win.

**Result:** **Could not run.** Each request payload (~67 MB of base64-encoded JPEG) exceeds the dynamo frontend's hardcoded HTTP body size limit. Test attempted at TP=1 agg, **failed with `Payload Too Large: Failed to buffer the request body: length limit exceeded`** before any inference happened.

---

## Workload that was attempted

- Bench script: `/hongming/dynamo/test_sglang_8img_4k.sh`
- 8 × **3840×2160 (4K)** random images per request
- 128 input + 256 output text tokens
- np=64, rate=1.0
- Avg request body: **66.75 MB** (warmup log: `Created 5 random jpeg images with average 66753836 bytes per request`)

## What blocked the test

Three layers of size limits encountered, in order:

### 1. Dynamo TCP request plane limit (32 MB) — fixable

Found env var: **`DYN_TCP_MAX_MESSAGE_SIZE`** (default 32 MB, `33554432` bytes).

This is the limit on TCP messages between dynamo frontend and worker. The error showed up in the worker log:
```
Failed to read TCP request: message too large: 66899815 bytes (max: 33554432 bytes)
TCP connection error: message too large: 66899815 bytes (max: 33554432 bytes)
```

**Fix applied:** added `export DYN_TCP_MAX_MESSAGE_SIZE=268435456` (256 MB) to all 3 server start scripts (`start_h200_aggregate_epd_server_32b_tp1.sh`, `..._tp2.sh`, `start_disagg_h200_32b_combined.sh`).

Verified working — no more "message too large" errors after this fix.

### 2. SGLang worker GPU OOM during ViT activations — fixable

After fixing the TCP limit, ran into:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.00 MiB.
GPU 0 has a total capacity of 139.80 GiB of which 14.81 MiB is free.
```

Root cause: `mem_fraction_static=0.95` left only 5% (~7 GB) of GPU memory for ViT activations + working set. With 16+ images at 1080p, ViT activations exceeded that headroom.

**Fix applied:** reduced `mem_fraction_static` from 0.95 → **0.85** in all 3 server scripts.

Verified working — 16 images at 1080p ran successfully after this fix (see `16img_1080p_three_way.md`).

### 3. Dynamo frontend HTTP body limit (~2 MB axum default) — NOT fixable without source patch

The 8 imgs @ 4K test still fails at the **frontend HTTP layer** before reaching the worker:

```
ValueError: Warmup failed - Please make sure benchmark arguments are correctly specified.
Error: Payload Too Large: Failed to buffer the request body: length limit exceeded
```

This is the **`axum::DefaultBodyLimit`** (default 2 MB) hardcoded in dynamo's compiled frontend (`/opt/venv/lib/python3.12/site-packages/dynamo/_core.abi3.so`). The error string `Failed to buffer the request body: length limit exceeded` and the bundled `axum-core-0.5.6` library code paths confirm this is the framework's default body limit applied to the chat/completions HTTP endpoint.

Searched for env vars and CLI flags — **no exposed knob** in this dynamo build. The body size limit is set at compile time when the Rust frontend builds its axum Router. To change it would require:
1. Getting the dynamo Rust source code
2. Finding the line that builds the axum `Router` (likely calling `.layer(DefaultBodyLimit::disable())` or `.layer(DefaultBodyLimit::max(N))`)
3. Patching it to allow larger bodies (e.g., 256 MB)
4. Recompiling dynamo with `cargo build --release`
5. Replacing `_core.abi3.so` in the venv

This is a substantial change (tens of minutes of build time, risk of breaking other things) and the current dynamo binary doesn't expose this configuration.

---

## Summary of what we learned

| Workload | Attempted? | Result |
|---|---|---|
| 8 imgs @ 1080p | ✓ tested previously | TP=2 agg wins (0.95 RPS) |
| 16 imgs @ 1080p (~32 MB body) | ✓ tested (after TCP+OOM fixes) | TP=2 agg wins (0.28 RPS), still LLM-bound (~25% ViT/total) |
| **8 imgs @ 4K (~67 MB body)** | **❌ blocked at HTTP frontend** | **Cannot test on this dynamo build** |
| 32 imgs @ 1080p (~67 MB body) | ❌ blocked at HTTP frontend | Cannot test on this dynamo build |

**Practical body-size ceiling for this dynamo build: roughly 32-33 MB per request.** This corresponds to:
- ~16 imgs @ 1080p
- ~4 imgs @ 4K
- ~32 imgs @ 720p

To meaningfully test ViT-bound regimes (where disagg should win), we need:
- A model with bigger ViT relative to LLM (Pixtral, InternVL2-large-ViT)
- OR a smaller LLM (Qwen3-VL-7B / 3B class)
- OR a custom dynamo build with raised body limit

---

## Updated GPU memory profile after fixes

After applying both fixes (`DYN_TCP_MAX_MESSAGE_SIZE=256 MB`, `mem_fraction_static=0.85`):
- TP=1 agg startup: GPU 4 = ~122 GB (previously ~137 GB at mem_fraction=0.95)
- TP=2 agg startup: GPU 4,5 = ~122 GB each (previously 126 GB each at mem_fraction=0.88)
- Disagg startup: encoder ~1.8 GB, PD ~123 GB

**Effect on prior baselines:** the 1080p × 8 images sweep (`1080p_sweep_three_way.md`) was done at the old mem_fraction. These new memory settings could marginally affect KV cache size but shouldn't significantly change the relative ordering or saturation behavior.

---

## Recommended path forward for ViT-bound testing

1. **Try Qwen3-VL-7B** (if model weights are available locally). With smaller LLM + same ViT, the ViT/LLM ratio flips toward ViT. Same workload (8 imgs @ 1080p) should clearly demonstrate disagg-winning regime. Easiest test that doesn't fight infrastructure.

2. **Build dynamo from source with raised HTTP body limit** if 4K testing on Qwen3-VL-32B is mandatory. Several hours of work; risk to dev environment.

3. **Test 4 imgs @ 4K** instead of 8 (33 MB ÷ 2 = ~17 MB body, fits the 32 MB limit). Less ViT compute (2.4 s vs 4.8 s) but native 4K resolution. May still be enlightening.

---

## Files

- Bench script (4K, blocked): `/hongming/dynamo/test_sglang_8img_4k.sh`
- Bench script (16 imgs 1080p, worked): `/hongming/dynamo/test_sglang_16img_1080p.sh`
- Result dirs:
  - 4K attempt (failed): `/hongming/res6_img8_4k/h200_agg_tp1_32b_image8_4k_np64/test_sglang_multi_rates_1080p_20260522_194806/results_summary.csv` (FAILED row)
  - 16 imgs @ 1080p (succeeded): `/hongming/res5/h200_agg_tp1_32b_image16_1080p_np64/...`
- Server start scripts (with mem_fraction=0.85 + DYN_TCP_MAX_MESSAGE_SIZE=256 MB):
  - TP=1 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
  - TP=2 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
  - Disagg: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh`
- Worker error logs: `/hongming/dynamo/01_cuda_sh/agg_h200_32b/logs/logs/epd_worker_server.log` (look for `Failed to read TCP request` and `OutOfMemoryError`)
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/16img_1080p_three_way.md` — what worked
  - `01_cuda_sh/disagg_h200_32b/when_vit_is_bottleneck.md` — predicted regimes
- This document: `01_cuda_sh/disagg_h200_32b/8img_4k_blocked.md`

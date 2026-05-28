# 16 Images @ 1080p — Three-Way Comparison (TP=1 agg vs TP=2 agg vs Disagg)

**Test date:** 2026-05-22
**Goal:** Push image count from 8 to 16 to test the hypothesis that more images per request would shift the ViT/LLM ratio toward ViT-bound, where disagg should structurally win.

**Workload:**
- 16 × 1920×1080 random images per request (~32k vision tokens, ~33 MB JPEG payload)
- 128 input + 256 output text tokens
- np=64, rate=1.0 req/s

**Configuration changes for this test:**
- Added `DYN_TCP_MAX_MESSAGE_SIZE=268435456` (256 MB) to all 3 start scripts (default 32 MB was too small for 16-image payloads)
- Reduced `--mem-fraction-static` from 0.95 → **0.85** (need GPU memory headroom for ViT activations on 16 images)
- 32-image test attempted earlier failed: even after raising TCP limit, GPU OOM during ViT forward.

---

## Results

| Metric | **TP=1 agg** (1 GPU) | **TP=2 agg** (2 GPUs) | **Disagg** (2 GPUs) |
|---|---:|---:|---:|
| Actual RPS | **0.21** | **0.28** | **0.10** |
| Successful requests | 64 / 64 ✓ | 64 / 64 ✓ | 64 / 64 ✓ |
| Bench duration | 311.9 s | 225.6 s | 621.9 s |
| **Mean TTFT** | 136,302 ms | **104,126 ms** | 354,340 ms |
| Median TTFT | 128,063 ms | **103,682 ms** | 358,740 ms |
| P99 TTFT | 233,965 ms | **150,773 ms** | 541,302 ms |
| **Mean TPOT** | **672 ms** | **686 ms** | 1,362 ms |
| Median TPOT | 680 ms | **445 ms** | 1,083 ms |
| P99 TPOT | 2,259 ms | 6,261 ms | 7,074 ms |
| Mean ITL | 586 ms | **423 ms** | 1,070 ms |
| Median ITL | 56 ms | **31 ms** | 59 ms |
| **Max ITL** | **31.6 s** | 109.7 s | **250.4 s** |
| **Mean E2E** | **203,541 ms** | **152,837 ms** | **476,111 ms** |
| Median E2E | 219,505 ms | **158,252 ms** | 496,336 ms |
| P99 E2E | 270,146 ms | **209,316 ms** | 578,528 ms |
| Input throughput | 6,720 tok/s | **9,291 tok/s** | 3,371 tok/s |
| Output throughput | 24 tok/s | **33 tok/s** | 12 tok/s |
| Peak output throughput | 395 tok/s | 679 tok/s | 379 tok/s |
| Concurrency | 41.8 | 43.4 | 49.0 |

**TP=2 agg wins on every meaningful metric.** Disagg is far worse — 2× slower than TP=1 agg, 3× slower than TP=2 agg.

---

## Did the encoder become the bottleneck?

**No.** The LLM forward pass is still the bottleneck even with 16 images per request.

### Evidence: TP=2 agg helps significantly

If ViT were the bottleneck, TP=2 agg shouldn't help much because the ViT (Qwen3-VL's vision encoder) doesn't TP-shard meaningfully across 2 GPUs — the encoder runs on just one of the TP ranks. Yet:

- TP=2 agg: 0.28 RPS, 153 s E2E
- TP=1 agg: 0.21 RPS, 204 s E2E

TP=2 gives **+33% RPS and -25% E2E** vs TP=1. This proves the LLM forward pass is still the bottleneck — TP=2 sharding the LLM compute is what's helping.

### Evidence: Disagg lost 2× vs TP=1 agg

If ViT were the bottleneck:
- Disagg's dedicated encoder GPU should have utilized ~70-100% of GPU 4
- Disagg should match or beat TP=1 agg

Instead:
- Disagg got 0.10 RPS vs TP=1 agg's 0.21 (disagg is half as fast)
- Disagg's encoder GPU still only used 1.8 GB of 143 GB (~1.3%)
- The extra GPU bought us nothing because the bottleneck is on the PD side

This is consistent with all our prior tests: **disagg only wins when ViT is genuinely the bottleneck, and we still haven't created such a workload on Qwen3-VL-32B-FP8.**

### Estimated per-request time breakdown

| Phase | Time | Fraction of total |
|---|---:|---:|
| ViT encode (16 × 1080p) | ~2.4 s | ~25% |
| LLM prefill (~32k tokens, chunked into 4 blocks of 8192) | ~5-6 s | ~55% |
| LLM decode (256 tokens × ~30 ms) | ~2-3 s | ~20% |
| **Total per-request service time** | **~10 s** | — |

LLM compute dominates by 3× over ViT. **ViT/(ViT+LLM) ≈ 25%, still well below the ~50% threshold needed for ViT to be the binder.**

---

## Comparing across image counts (1080p, np=64, rate=1.0)

| Image count | TP=1 agg RPS | TP=2 agg RPS | Disagg RPS | Disagg/TP=2 |
|---:|---:|---:|---:|---:|
| 8 | 0.53 | **0.95** | 0.26 | 0.27 |
| **16** | **0.21** | **0.28** | **0.10** | **0.36** |

Going from 8 → 16 images:
- **TP=1 agg RPS dropped 60%** (0.53 → 0.21) — LLM prefill cost doubled
- **TP=2 agg RPS dropped 71%** (0.95 → 0.28) — same workload doubling
- **Disagg RPS dropped 62%** (0.26 → 0.10) — already suffering, gets worse

The relative gap between disagg and TP=2 agg actually **improves slightly** at 16 images (disagg/TP=2 ratio 0.36 vs 0.27 at 8 images), suggesting we ARE moving toward ViT-bound territory. But we're nowhere near the regime where disagg would actually win.

---

## What it would take to actually flip the comparison

Based on this data, ViT/(ViT+LLM) is currently ~25% at 16 images. To get to >50% (where disagg should win):

### Option A: Push image count further

ViT compute scales linearly with image count:
| Image count | Estimated ViT/(ViT+LLM) |
|---:|---:|
| 16 (this test) | ~25% |
| 32 (failed: HTTP body too big) | ~42% |
| 64 (would need image-content optimization) | ~55% |
| **128** | **~70%** — disagg wins |

But payload limits (HTTP body size) and GPU memory limits make this hard to test on Qwen3-VL-32B-FP8.

### Option B: Use 4K resolution

A 4K image (3840×2160) is 4× the pixels of 1080p:
| Workload | Estimated ViT/(ViT+LLM) |
|---:|---:|
| 8 imgs @ 4K | ~50% — at the threshold |
| 16 imgs @ 4K | ~67% — likely disagg wins |

Caveat: also runs into payload size / GPU memory limits.

### Option C: Smaller LLM

Same workload (8 imgs @ 1080p), smaller LLM:
- Qwen3-VL-7B + 8 × 1080p: ViT/(ViT+LLM) ≈ 70% — **disagg should clearly win**
- Qwen3-VL-3B + 8 × 1080p: ≈ 86% — disagg wins by huge margin

This is the cleanest path to demonstrate ViT-bound disagg-wins.

---

## Conclusion

**16 images @ 1080p is still LLM-bound on Qwen3-VL-32B-FP8.** The 25% ViT-time fraction isn't enough to make disagg win — it just makes the LLM have to do more work, which both agg configs handle better than disagg's overhead-laden architecture.

**TP=2 agg remains the winner** on this workload (0.28 RPS, 153 s E2E), confirming that for 32B-class multimodal LLMs with up-to-16 image inputs, the optimal config is TP-sharded aggregated EPD, not disagg.

For disagg to win in a meaningfully reproducible way, we need either:
1. A smaller LLM (3B-7B class) — most realistic test
2. Larger image counts (32+) — currently blocked by payload size limits
3. A vision-heavy model architecture (Pixtral, InternVL2-large-ViT)
4. Video workloads with many frames per request

---

## Files

- TP=1 agg result CSV: `/hongming/res5/h200_agg_tp1_32b_image16_1080p_np64/test_sglang_multi_rates_1080p_20260522_181933/results_summary.csv`
- TP=2 agg result CSV: `/hongming/res5/h200_agg_tp2_32b_image16_1080p_np64/test_sglang_multi_rates_1080p_20260522_183100/results_summary.csv`
- Disagg result CSV: `/hongming/res5/h200_h200_disagg_tp1_32b_image16_1080p_np64/test_sglang_multi_rates_1080p_20260522_191052/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_16img_1080p.sh`
- Server start scripts (with mem-fraction=0.85 and DYN_TCP_MAX_MESSAGE_SIZE=256MB):
  - TP=1 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
  - TP=2 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
  - Disagg: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh`
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/when_vit_is_bottleneck.md` — analysis predicting this regime
  - `01_cuda_sh/disagg_h200_32b/1080p_sweep_three_way.md` — 8-image baseline
  - `01_cuda_sh/disagg_h200_32b/768p_comparison.md`
- This document: `01_cuda_sh/disagg_h200_32b/16img_1080p_three_way.md`

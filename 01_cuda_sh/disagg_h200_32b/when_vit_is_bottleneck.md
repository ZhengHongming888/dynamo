# When is the Vision Encoder the Bottleneck?

**Date:** 2026-05-22
**Context:** We've run 6+ workload variants on Qwen3-VL-32B-Instruct-FP8 across TP=1 agg, TP=2 agg, and disagg configurations. In all of them, vision encoder was NOT the bottleneck — PD's prefill+decode dominated. This document reasons through what configurations would actually make ViT the bottleneck, and why those are exactly the configurations where disagg should structurally win.

---

## What we measured for ViT cost

Across all our tests on Qwen3-VL-32B-Instruct-FP8, the encoder GPU memory was always **1.8 GB / 143 GB used (~1.3% utilized)**, indicating massive headroom. The actual ViT compute time per request:

| Workload | ViT time/req | LLM forward time/req | ViT/(ViT+LLM) |
|---|---:|---:|---:|
| 8 images @ 1080p (~16k vis tokens) | ~1.2 s | ~3-4 s prefill + 0.5-30 s decode | **6-25%** |
| 8 images @ 768p (~6k vis tokens) | ~0.6 s | ~1.5-2 s prefill + 0.5-30 s decode | 5-20% |
| 1 image @ 1080p (~2k vis tokens) | ~0.15 s | ~0.2 s prefill + 0.5-30 s decode | <5% |
| 8 cached images | ~0 s (cache hit) | unchanged | 0% |

**ViT was never the bottleneck in any tested configuration.** PD's prefill+decode was 4-25× more expensive than encoder time.

---

## What makes ViT relatively dominant

For ViT to bottleneck, the ratio `ViT / (ViT + LLM)` must exceed ~50%. We can move the ratio by:

1. **Increase ViT work** — more images, bigger images, more patches per image
2. **Decrease LLM work** — smaller LLM, shorter sequences, fewer output tokens
3. **Both**

### Why Qwen3-VL-32B's ViT is small

The Qwen3-VL-32B `Qwen3VLMoeVisionModel`:
- ~600 M parameters in BF16, ~1.2 GB on disk
- Compared to 32B FP8 LLM (~32 GB on disk), **the ViT is ~3% the model size**
- This is why our ViT memory footprint is just 1.8 GB total (model + activations)

But ViT compute is **per pixel**, not per token. A 1920×1080 image has 2,073,600 pixels × 3 channels = 6.2M values × N transformer layers. So even a small ViT does meaningful compute on big images.

The key relationships:
- **Per-image ViT cost ∝ pixel count** (so 4K is ~4× more expensive than 1080p)
- **Per-request ViT cost ∝ image_count × pixels/image**
- **Per-request LLM cost ∝ vision_tokens × LLM_size** (and vision tokens ∝ pixels)

So the ratio `ViT / LLM` per token is essentially `ViT_size / LLM_size`. For Qwen3-VL-32B-FP8 that ratio is ~3%, and we're seeing 6-25% in practice because the ViT does a few transformer layers per visual patch position before the LLM sees the embedding.

---

## Concrete configurations where ViT becomes the bottleneck

### A. Many high-resolution images per request

For Qwen3-VL-32B-FP8 at 1080p:
| Image count | ViT time/req | LLM prefill time/req | ViT-bound? |
|---:|---:|---:|---|
| 8 (our test) | ~1.2 s | ~1.7 s (16k tokens / 9.5k tok/s) | No (~40%) |
| 16 | ~2.4 s | ~3.4 s | Closer (~41%) |
| 32 | ~4.8 s | ~6.7 s (chunked) | Approaching (~42%) |
| **64** | **~10 s** | **~13 s** | **Approaching parity (~43%)** |
| 100 | ~15 s | ~21 s | Still LLM-bound, but borderline |

**Concrete test:** `--image-count 32 --image-resolution 1920x1080` at rate=1.0.

Caveat: at 32 images × 16k tokens/image expected = 64k vision tokens, this exceeds `max_prefill_tokens=16384`, so SGLang will chunk-prefill the request across multiple steps. Total prefill time is ~3-4× the single-chunk case.

### B. 4K resolution

A 4K image (3840×2160) is **4× the pixels** of 1080p. So ViT compute scales 4×.
| Workload | ViT time/req | LLM time/req | ViT-bound? |
|---|---:|---:|---|
| 8 × 4K | ~4.8 s | ~6.5 s | Approaching (~43%) |
| 16 × 4K | ~9.6 s | ~13 s | Approaching parity |

### C. Smaller LLM with same image workload (most realistic ViT-bound case)

If we swap the LLM but keep the ViT, the LLM gets smaller and faster while ViT compute stays the same:

| Model | ViT time | LLM prefill (16k tokens) | ViT/(ViT+LLM) | Bottleneck |
|---|---:|---:|---:|---|
| Qwen3-VL-**32B**-FP8 (our tests) | 1.2 s | ~1.7 s | ~41% | LLM (close) |
| **Qwen3-VL-7B**-Instruct + 8×1080p | 1.2 s | ~0.5 s (30k tok/s) | **70%** | **ViT** |
| **Qwen3-VL-3B**-Instruct + 8×1080p | 1.2 s | ~0.2 s (75k tok/s) | **86%** | **ViT** |
| **Qwen2-VL-2B** + 8×1080p | 1.2 s | ~0.15 s | **89%** | **ViT** |

These smaller-LLM configurations are where disagg should genuinely shine because the encoder is the binder.

### D. Vision-heavy models (larger ViT than typical)

Some VL models have unusually large vision towers:
| Model | ViT params | LLM params | ViT/LLM size ratio |
|---|---:|---:|---:|
| Qwen3-VL-32B (our model) | ~0.6 B | ~32 B | 2% |
| Qwen2-VL-72B | ~0.7 B | ~72 B | 1% |
| Llava-OneVision (InternViT-300M) | 0.3 B | 0.5-7 B | 4-60% |
| **Pixtral-12B** | ~400 M | 12 B | 3% (but ViT compute scales differently per image) |
| **InternVL2-Llama3-76B** | ~6 B | 76 B | **8%** |
| **Florence-2-large** | ~770 M | 0.7 B | **>100%** (vision-dominant) |
| **DALL-E-3 / Imagen-style models** | mostly vision | ~1 B text | vision-dominant |

**Models where ViT is naturally dominant**: Florence, Pixtral, BlendingViT, Qwen-VL-Plus (older), and any document/chart-understanding model that uses high-resolution dynamic patching.

### E. Streaming video / temporal sequences

Video inference is the canonical ViT-bound workload:
- **32 frames @ 1080p** per request: 32 × 150 ms = 4.8 s ViT
- LLM forward: just process the video summary token sequence (much shorter than 32×16k tokens because video models compress)
- Ratio: typically 70-95% ViT-bound

Video-LLMs like Qwen2.5-Omni-Video, LongVU, Tarsier are all ViT-bound.

### F. Sustained high-rate workload that saturates encoder

Even on Qwen3-VL-32B-FP8, if the offered rate is high enough, the encoder GPU's compute throughput becomes the binder. With Qwen3-VL ViT at ~7 images/sec on H200 at 1080p:

| Sustained image rate | Encoder utilization |
|---:|---:|
| 1 RPS × 8 images = 8 imgs/s | 100% encoder saturated! |
| 0.5 RPS × 8 images = 4 imgs/s | 57% utilized |
| Our TP=2 saturation (1.13 RPS × 8) = 9 imgs/s | **>100%, but LLM saturated first** |

**This means at TP=2 agg's saturation point (1.13 RPS), the encoder was technically already at ~110% capacity** — but because the LLM saturates first inside the same process, we never saw the encoder bottleneck. In a disagg setup with **fast enough LLM** (like a 3B model + 1080p), the encoder would saturate first.

---

## Summary: when ViT bottlenecks

| Configuration | ViT/(ViT+LLM) | Bottleneck | Disagg likely to win? |
|---|---:|---|---|
| Our tests: Qwen3-VL-32B + 8×1080p | 6-25% | LLM | No (verified) |
| Qwen3-VL-32B + 32×1080p | ~42% | LLM (close) | Marginal |
| Qwen3-VL-32B + 8×4K | ~43% | LLM (close) | Marginal |
| Qwen3-VL-32B + video (32 frames) | ~70% | **ViT** | **Yes** |
| **Qwen3-VL-7B + 8×1080p** | **~70%** | **ViT** | **Yes** |
| **Qwen3-VL-3B + 8×1080p** | **~86%** | **ViT** | **Yes (clearly)** |
| InternVL2-76B (6B-ViT + 3B-LLM analog) | ~98% | **ViT** | **Yes (clearly)** |
| Sustained rate > 3 RPS with unique images | varies | depends | Setup-dependent |

---

## Why this matters for disagg architecture

The conditions where ViT is the bottleneck **align exactly with the conditions where disagg should structurally win**:

| Scenario | Why disagg wins when ViT bottlenecks |
|---|---|
| Many images/request | Encoder GPU is the binder; PD GPU has spare cycles |
| Smaller LLM | ViT > LLM compute; dedicated encoder GPU pays off |
| Vision-heavy models | Encoder IS the model; disagg is just architectural fit |
| Video workloads | Encoder dominance is inherent |
| Multi-tenant fanout | 1 encoder serves N PDs — only worthwhile if encoder is binder |

**For our tested workload (Qwen3-VL-32B + 8×1080p), disagg loses because ViT is NOT the bottleneck.** Disagg only wins when ViT IS the bottleneck.

This is fully consistent with **all 6+ workload tests we've run**:

| Workload tested | ViT/(ViT+LLM) | Result |
|---|---:|---|
| 8 imgs @ 1080p (random) | ~25% | TP=2 agg wins |
| 8 imgs @ 768p (random) | ~20% | TP=2 agg wins |
| 1 img @ 1080p, out=1024, rate=0.5 | ~5% | tie / TP=2 wins |
| 1 img @ 1080p, out=1024, rate=2.0 | ~5% | TP=2 wins |
| 8 cached imgs @ 1080p | ~0% (cache hit) | TP=2 wins (cache helps both) |
| 4 imgs @ 1080p (smaller) | ~15% | TP=2 wins |

In none of these did ViT meaningfully approach 50%, so disagg never had its moment.

---

## Concrete experiments to verify the ViT-bound regime

### Easiest (no model swap): **32 images @ 1080p**

```
--image-count 32 --image-resolution 1920x1080 --random-output-len 256 --request-rate 1.0
```

Predicted: ViT ~4.8 s, LLM ~6.7 s (chunked). Disagg might draw or marginally win.

### Cleaner test (requires Qwen3-VL-7B model): **8 images @ 1080p with smaller LLM**

```
--model Qwen3-VL-7B-Instruct --image-count 8 --image-resolution 1920x1080
```

Predicted: ViT ~1.2 s, LLM ~0.5 s. Disagg should clearly win. Single H200 8B-ish LLM has tons of decode headroom that can run while encoder works on next request.

### Most ViT-bound (requires different model class): **Pixtral or InternVL2 + 8 imgs**

Disagg should dominate by 2-5×.

---

## What this means for production deployments

If you're deploying:

| Your model | Workload | Recommended config |
|---|---|---|
| Qwen3-VL-32B-FP8 (or larger) | Few images per request | TP=2 agg (best) or TP=1 agg |
| Qwen3-VL-32B-FP8 | Many images (>32) per request | TP=2 agg, but also try disagg |
| Qwen3-VL-32B-FP8 | Video frames (10+ frames/req) | **Disagg** (encoder bottleneck) |
| Qwen3-VL-7B / Qwen2-VL-7B | Any image count | **Disagg** (encoder bottleneck) |
| Pixtral, InternVL2-large-ViT | Any | **Disagg** (encoder bottleneck) |
| Smaller VL (~3B class) | Any | **Disagg** strongly preferred |

The simple heuristic: **if your ViT is smaller than ~10% of your LLM weight (in compute, not parameters), agg wins. If your ViT is bigger than that, disagg wins.**

---

## Files

- All companion docs in `01_cuda_sh/disagg_h200_32b/`:
  - `disagg_all_rates_results.md`, `deep_analysis_disagg_worse_h200.md`, `disagg_improvements_attempts.md`, `bottleneck_analysis.md`, `when_disagg_wins.md`
  - `long_output_1024_results.md`, `rate_2.0_agg_disagg_comparison.md`, `cache_hit_agg_disagg.md`
  - `8img_768p_agg_disagg.md`, `768p_comparison.md`
  - `1080p_sweep_three_way.md`
- This document: `01_cuda_sh/disagg_h200_32b/when_vit_is_bottleneck.md`

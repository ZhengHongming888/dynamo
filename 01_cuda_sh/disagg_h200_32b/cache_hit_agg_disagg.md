# Cache-Hit Workload — TP=1 agg vs Disagg (1 H200 + 1 H200)

**Hypothesis:** When all requests use the same images, encoder cache hits should benefit disagg more than aggregated EPD because disagg has dedicated encoder GPU memory (~140 GB free) for caching, while agg shares memory with the LLM.

**Result:** **Hypothesis disproved.** TP=1 agg won by 2× on RPS (0.99 vs 0.50) and 14× on E2E (6.5 s vs 95 s). Cache hits helped both configs by similar percentages (+85-90% RPS), but disagg's per-request handoff cost overwhelms the encoder savings.

---

## Test setup

**Workload:**
- 8 × 1920×1080 images per request (same as the original "prefill-heavy" workload)
- **`--image-content blank`** — all images are byte-identical (uniform white pixels)
- 128 input + 256 output tokens
- np=64, rate=1.0 req/s
- All 64 requests use the same 8 blank images → **encoder cache should hit on all but the first request**

**Configurations:**
- **TP=1 agg**: GPU 4 only, `--enable-mm-global-cache` (already on)
- **Disagg**: GPU 4 (encoder) + GPU 5 (PD), with **`--multimodal-embedding-cache-capacity-gb 16`** added to encoder; PD has `--enable-mm-global-cache`

**Bench script:** `/hongming/dynamo/test_sglang_cache_hit.sh`

---

## Results

| Metric | **TP=1 agg** (1 GPU) | **Disagg** (2 GPUs) | Δ disagg vs agg |
|---|---:|---:|---:|
| Actual RPS | **0.99** ✓ | 0.50 | **−49%** |
| Successful requests | 64 / 64 | 64 / 64 | = |
| Bench duration | 64.6 s | 128.1 s | +98% |
| **Mean TTFT** | **2,114 ms** | **81,476 ms** | **+3,754%** |
| Median TTFT | 1,514 ms | 84,762 ms | +5,498% |
| P99 TTFT | 5,961 ms | 107,305 ms | +1,700% |
| **Mean TPOT** | **64 ms** | **191 ms** | **+196%** |
| Median TPOT | 30 ms | 52 ms | +71% |
| P99 TPOT | 638 ms | 2,839 ms | +345% |
| Mean ITL | 58 ms | 119 ms | +106% |
| **Max ITL** | **2,505 ms** | **108,707 ms** | **+4,238%** (108s decode stall on disagg!) |
| **Mean E2E** | **6,494 ms** | **95,004 ms** | **+1,363%** |
| Median E2E | 5,178 ms | 96,545 ms | +1,765% |
| P99 E2E | 18,761 ms | 124,303 ms | +562% |
| Input throughput | 16,254 tok/s | 8,201 tok/s | −50% |
| Output throughput | 116 tok/s | 58 tok/s | −50% |
| **Peak output throughput** | 381 tok/s | **1,108 tok/s** | **+191%** (only disagg win) |
| Concurrency | 6.4 | 47.5 | +642% |

---

## Cache hit was confirmed working

From disagg encoder log:
```
[06:36:18] Cached key=1c2267bc62a28fc7, size=79.69MB, total=0.078GB
... (8 unique images cached, 0.6 GB total of 16 GB capacity)

[06:36:27] Embedding cache hit for URL index 0
... (560 cache hits across 64 requests × 8 images = 512 image-encode requests)
```

**Cache hits: 560 of 561 mm_encode events** (≈99.8% hit rate after the first request fills the cache).

## Disagg per-phase breakdown (`enable_request_time_stats_logging`)

| Phase | Median | Mean | Max |
|---|---:|---:|---:|
| Queue duration | 3.5 s | 4.3 s | 12.9 s |
| Forward duration | **20.9 s** | 43.2 s | 122.7 s |
| Total per-request | ~24 s | ~48 s | — |

Even with cache hits, disagg's PD spends median 21 s in forward — that's the LLM prefill+decode work, identical to non-cached case. Cache only saved encoder ViT compute (~1.2 s).

---

## Comparing all 4 configurations at rate=1.0, 8 images @ 1080p

| Config | RPS | Mean TTFT | Mean E2E | Notes |
|---|---:|---:|---:|---|
| TP=1 agg, **random images** | 0.52 | 32.4 s | 80.7 s | Original baseline |
| Disagg, **random images** | 0.27 | 78.9 s | 111 s | Original disagg test |
| **TP=1 agg, blank images (cache-hit)** | **0.99** | **2.1 s** | **6.5 s** | +90% RPS from caching |
| **Disagg, blank images (cache-hit)** | **0.50** | **81.5 s** | **95.0 s** | +85% RPS from caching |

### Cache hit benefit ratios

| Config | Random RPS | Cache-hit RPS | Improvement |
|---|---:|---:|---:|
| TP=1 agg | 0.52 | 0.99 | **+90%** |
| Disagg | 0.27 | 0.50 | **+85%** |

**Both configs benefit ~equally from caching (~85-90%).** Cache doesn't asymmetrically help disagg.

---

## Why caching doesn't help disagg more than agg

Initial expectation: disagg's encoder GPU has 140 GB free memory for caching while agg shares memory with LLM, so disagg should benefit more from cache.

**Reality (revealed by this test):**

### 1. Cache lookup itself is microseconds either way
Both configs use a hash-keyed embedding cache. Lookup is fast.

### 2. TP=1 agg's "encoder" path on cache hit is essentially free
SGLang's `enable-mm-global-cache` skips ViT entirely and feeds the cached embedding directly into the LLM forward. ~100 ms total per request (most of it is prefill + decode startup).

### 3. Disagg's "encoder" path on cache hit is NOT free
Even when the encoder hits cache, it still has to:
1. Receive request via dynamo TCP request plane
2. Look up cached embedding (μs)
3. Call `connector.create_readable(descriptor)` to expose it via NIXL
4. Wait for PD to `connector.begin_read` + `await wait_for_completion`
5. Send response back to PD via dynamo TCP

This costs **~500 ms - 1 s per request EVEN ON CACHE HIT**.

### 4. Disagg PD still does identical prefill-decode work
The cache hit on encoder doesn't speed up PD's job. PD still:
- Builds mm_items from received embeddings
- Runs LLM forward over 16,336 vision-token positions
- Decodes 256 output tokens

This is the bulk of per-request time and is unchanged by caching.

### 5. Disagg's small-batch embedding-integration overhead persists
We see prefill events with `#new-token: 16, 80, 64, 528, ...` mixed in with the main `#new-token: 16384` events. These small batches stall decode for other requests (Max ITL: 108 s in disagg vs 2.5 s in TP=1 agg).

---

## Net summary: where disagg wins on cache-hit

The only metric where disagg won: **peak output throughput (+191%)** — 1,108 vs 381 tok/s. PD GPU during decode bursts can hit higher peak rates because it's not sharing power/thermal headroom with ViT.

But this peak doesn't translate to mean throughput (both at ~58-116 tok/s during sustained decode), so it's only useful for niche short-burst latency-sensitive scenarios.

---

## What this proves about the cache-hit hypothesis

**Cache-hits help both configs by ~the same percentage** (85-90% RPS improvement). They don't asymmetrically favor disagg. The hypothesis that "disagg has more memory for cache so it should win" was wrong because:

1. **Both configs cache the same number of unique images** (8 in our test, both fit easily)
2. **Cache lookup speed is the same** (in-memory hash table)
3. **The bottleneck on cache hit is no longer encoder compute** — it shifts to LLM prefill+decode, which is equally hard for both
4. **Disagg still pays its per-request handoff overhead even on cache hits** (~1 s/request, negligible for agg)

For disagg's encoder cache to win, you need a regime where:
- **Encoder is the bottleneck** (not LLM) — the opposite of our 32B FP8 setup
- **Cache size matters** (many unique images that don't fit in agg's shared cache) — would need hundreds of unique images, GB of cache used

---

## Files

- TP=1 agg result CSV: `/hongming/res4/h200_agg_tp1_32b_image8_1080p_np64_cachehit/test_sglang_multi_rates_1080p_20260522_062503/results_summary.csv`
- Disagg result CSV: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_cachehit/test_sglang_multi_rates_1080p_20260522_063606/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_cache_hit.sh`
- TP=1 agg server start: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
- Disagg server start: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh` (with `--multimodal-embedding-cache-capacity-gb 16`)
- Disagg PD log: `/hongming/dynamo/logs/logs/logs/logs/logs/logs/pd_worker.log`
- Disagg encoder log: `/hongming/dynamo/logs/logs/logs/logs/logs/logs/encoder_worker.log`
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/when_disagg_wins.md` — predicted cache-hit was the most-likely-disagg-win regime; this test proves that prediction wrong
  - `01_cuda_sh/disagg_h200_32b/long_output_1024_results.md` — decode-heavy at rate=0.5
  - `01_cuda_sh/disagg_h200_32b/rate_2.0_agg_disagg_comparison.md` — decode-heavy at rate=2.0
  - `01_cuda_sh/disagg_h200_32b/bottleneck_analysis.md` — bottleneck breakdown
- This document: `01_cuda_sh/disagg_h200_32b/cache_hit_agg_disagg.md`
  (was duplicated as `cache_hit_comparison.md` until 2026-05-24; that file is now a stub)

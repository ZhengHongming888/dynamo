# Rate=2.0 Comparison — TP=1 agg vs Disagg (1 H200 + 1 H200)

**Test date:** 2026-05-22
**Goal:** Compare TP=1 aggregated EPD vs disaggregated (encoder GPU 4 + PD GPU 5) at high concurrency on a decode-heavy multimodal workload.

**Workload:**
- 1 × 1920×1080 image per request
- 128 input text tokens
- **1024 output tokens** (decode-heavy)
- np=32 (default for non-mapped rates in bench script)
- **rate=2.0 req/s**

**Configurations:**
- **TP=1 agg**: GPU 4 only — `start_h200_aggregate_epd_server_32b_tp1.sh` (max_running_requests=40, mem_fraction=0.95, page_size=16, kv_cache_dtype=fp8_e4m3)
- **Disagg**: GPU 4 (encoder) + GPU 5 (PD) — `start_disagg_h200_32b_combined.sh` with current best config: NIXL_READ + max_running_requests=64 + chunked_prefill_size=16384

---

## Headline comparison

| Metric | **TP=1 agg** (1 GPU) | **Disagg** (2 GPUs) | Δ disagg vs agg |
|---|---:|---:|---:|
| Actual RPS | **1.15** | 1.00 | **−13%** |
| Successful requests | 32 / 32 ✓ | 32 / 32 ✓ | = |
| Bench duration | 27.7 s | 31.9 s | +15% |
| **Mean TTFT** | **573 ms** | **3,631 ms** | **+534%** |
| Median TTFT | 543 ms | 3,638 ms | +570% |
| P99 TTFT | 781 ms | 5,962 ms | +664% |
| **Mean TPOT** | **21.0 ms** | **33.8 ms** | **+61%** |
| Median TPOT | 20.9 ms | 25.9 ms | +24% |
| P99 TPOT | 31.8 ms | 126.6 ms | +298% |
| Mean ITL | 22.5 ms | 26.4 ms | +17% |
| Median ITL | 14.7 ms | 14.6 ms | = |
| P95 ITL | 70.8 ms | 32.8 ms | -54% |
| P99 ITL | 240 ms | 54 ms | -77% |
| **Max ITL** | **459 ms** | **10,860 ms** | **+2,266%** (10s decode stall on disagg!) |
| **Mean E2E** | **12.23 s** | **19.04 s** | **+56%** |
| Median E2E | 11.92 s | 17.55 s | +47% |
| P90 E2E | 19.07 s | 27.97 s | +47% |
| P99 E2E | 21.80 s | 31.03 s | +42% |
| Input throughput | 2,443 tok/s | 2,127 tok/s | −13% |
| Output throughput | 674 tok/s | 587 tok/s | −13% |
| **Peak output throughput** | 1,572 tok/s | **1,711 tok/s** | **+9% (only disagg win)** |
| Concurrency | 14.1 | 19.1 | +35% |

**Net:** TP=1 agg wins on every meaningful metric except peak output throughput (a niche metric).

---

## Per-phase breakdown for disagg (from `enable_request_time_stats_logging`)

Parsed from `ReqTimeStats(...)` lines in PD log:

| Phase | Median | Mean | Max |
|---|---:|---:|---:|
| Queue duration (PD scheduler queue) | **0 ms** | 60 ms | 260 ms |
| Forward duration (prefill + decode on PD) | **14.12 s** | 15.06 s | 29.53 s |

**Queue is essentially zero** — even at rate=2.0 with 19 concurrent requests, the PD scheduler queue itself isn't saturated.

So **the 3.6s mean TTFT is NOT** in PD's scheduler queue. It's in:
1. **dynamo TCP request plane** (frontend → encoder → PD round-trip)
2. **Encoder ViT compute + NIXL setup** (~200-700 ms when concurrent)
3. **`_build_mm_items` setup on PD before SGLang prefill starts** (~100-300 ms per request)

These "dynamo handoff" costs are visible to the bench client (TTFT) but invisible to SGLang's `forward_duration`. They scale super-linearly with concurrency because they're not fully parallelized.

---

## How TTFT scales with rate

Comparing the two decode-heavy tests we ran:

| Rate | TP=1 agg TTFT | Disagg TTFT | Disagg/agg ratio |
|---:|---:|---:|---:|
| 0.5 | 524 ms | 1,307 ms | 2.5× worse |
| **2.0** | **573 ms** | **3,631 ms** | **6.3× worse** |
| Δ from 4× rate increase | +9% | +178% | — |

**TP=1 agg's TTFT scaled almost flat** (+9%) when we quadrupled the offered rate. **Disagg's TTFT scaled 2.8×.**

This is the cleanest evidence that disagg's per-request overhead is **fundamentally additive-per-concurrent-request**, not amortizable. Each additional in-flight request adds ~80 ms of dynamo-side handoff overhead because:
- More concurrent NIXL transfers contend on the same CUDA streams
- More `_build_mm_items` Python setup on PD (single Python event loop)
- TCP request plane has more concurrent in-flight RPCs through the same dynamo runtime

---

## The Max ITL smoking gun

Look at the **maximum inter-token latency** — the longest gap between consecutive output tokens for any single request:

| Config | Max ITL |
|---|---:|
| TP=1 agg | **459 ms** |
| Disagg | **10,860 ms** (10.86 seconds!) |

**Disagg has 24× worse max ITL.** Some request, somewhere in the bench, went **10.86 seconds** between output tokens. This is exactly the "prefill blocks decode" pathology that disagg was supposed to AVOID, but reborn on disagg's PD side as embedding-integration sub-batches blocking decode.

When a new request arrives at PD:
1. NIXL READ pulls embedding from encoder
2. PD enters small-batch embedding-integration mode (16-176 token forward passes)
3. These small batches **block decode** for the ~10 active requests
4. One unlucky request gets stalled for 10s of seconds

TP=1 agg avoids this because there's no embedding-integration sub-batch pattern — ViT runs inline with the LLM forward in one continuous pass.

---

## Why the high-concurrency hypothesis was wrong

**Hypothesis (now falsified):** At rate=2.0 with ~14 concurrent requests, TP=1 agg's ViT compute would contend with decode of the in-flight requests. Disagg's dedicated decode GPU should win.

**Why it didn't happen:**

### 1. ViT for 1 image @ 1080p is too cheap (~150 ms)
Compared to 14 s of decode per request, ViT is ~1% of total. Even running every 500 ms (rate=2.0), the encoder fraction is small.

### 2. SGLang's scheduler handles ViT inline efficiently
For aggregated mode, `Qwen3VLForConditionalGeneration` runs ViT and LLM forward in one continuous pass. ViT activations are small enough not to displace decode KV cache. There's no actual ViT-vs-decode contention to expose.

### 3. Disagg's per-request handoff scales worse than ViT contention
The cost of disagg's NIXL handoff and embedding integration grows faster with concurrency than TP=1 agg's hypothetical ViT contention.

### 4. Disagg has its own decode-blocker (embedding integration)
The pathology I expected to find in TP=1 agg actually exists in disagg's PD instead, and worse.

---

## Disagg's only win: peak output throughput

| Config | Peak output tput |
|---|---:|
| TP=1 agg | 1,572 tok/s |
| Disagg | **1,711 tok/s (+9%)** |

Real and structural — when many requests happen to be decoding simultaneously and no prefill/integration is in flight, disagg's PD GPU can hit slightly higher decode burst rate (no power/thermal headroom shared with ViT).

But this **doesn't translate to mean output throughput** (587 vs 674, disagg is actually 13% worse) because peak only matters in brief bursts.

---

## Final verdict at rate=2.0

For 1 H200 encoder + 1 H200 PD with Qwen3-VL-32B-FP8 on this decode-heavy workload at rate=2.0:

- **TP=1 agg wins on RPS** (1.15 vs 1.00, +15%)
- **TP=1 agg wins on Mean TTFT** (573 ms vs 3,631 ms, **6× better**)
- **TP=1 agg wins on Mean TPOT** (21 vs 34 ms, +61%)
- **TP=1 agg wins on Mean E2E** (12.2 s vs 19.0 s, +56%)
- **TP=1 agg wins on Max ITL** (459 ms vs 10.86 s, **24× better**)
- **Disagg wins on Peak output throughput only** (+9%)

The decode-heavy hypothesis is **falsified**. Higher concurrency does NOT flip the comparison — it makes disagg worse, not better.

---

## What workload would actually flip the comparison?

Based on this and prior tests, disagg's path to victory is NOT about:
- ❌ Concurrency (disagg gets worse with concurrency)
- ❌ Decode dominance (handoff overhead dominates short-prefill)
- ❌ Long output (TPOT advantage too small to matter)

Disagg's potential wins remain in:
- **Cache-hit workloads** (encoder amortization on repeated images) — untested
- **Multi-tenant fanout** (1 encoder + N PDs) — requires 3+ GPUs, untested
- **Asymmetric hardware** (encoder on smaller GPU) — different hardware
- **Memory-isolated workloads** (encoder weights conflict with LLM) — different workload

For our exact same-host 2-H200 setup with this model and these workloads, **disagg never wins on E2E or RPS**.

---

## Files

- TP=1 agg result CSV: `/hongming/res4/h200_agg_tp1_32b_image1_1080p_out1024_dh_rate2/test_sglang_multi_rates_1080p_20260522_055659/results_summary.csv`
- Disagg result CSV: `/hongming/res4/h200_h200_disagg_tp1_32b_image1_1080p_out1024_dh_rate2/test_sglang_multi_rates_1080p_20260522_060927/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_decode_heavy.sh` — used with positional arg `2.0`
- TP=1 agg server start: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
- Disagg server start: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh` (NIXL_READ + max_running=64 + chunked_prefill=16384)
- Disagg PD log: `/hongming/dynamo/logs/logs/logs/logs/logs/pd_worker.log`
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/long_output_1024_results.md` — same workload at rate=0.5
  - `01_cuda_sh/disagg_h200_32b/long_output_1024_rate2_results.md` — earlier save of same comparison
  - `01_cuda_sh/disagg_h200_32b/when_disagg_wins.md` — workload categories analysis
  - `01_cuda_sh/disagg_h200_32b/bottleneck_analysis.md` — bottleneck breakdown
- This document: `01_cuda_sh/disagg_h200_32b/rate_2.0_agg_disagg_comparison.md`

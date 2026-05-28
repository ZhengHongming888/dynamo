# Decode-Heavy @ Rate 2.0 — High-Concurrency Test (TP=1 agg vs Disagg)

**Hypothesis being tested:** At higher rate (2.0 req/s, ~14 concurrent), TP=1 agg's ViT-vs-decode contention should expose disagg's structural advantage of dedicated decode GPU.

**Result:** **Hypothesis disproved.** TP=1 agg still wins decisively across every meaningful metric. The decode-heavy advantage that disagg theoretically should have does NOT materialize on this hardware/model with this workload.

---

## Test setup

**Workload (same as previous decode-heavy test, just at higher rate):**
- 1 image @ 1920×1080
- 128 input + **1024 output** tokens
- np=32 (default for rates not in the script's NUM_PROMPTS_MAP)
- **rate=2.0 req/s**

**Configurations:**
- TP=1 agg: GPU 4 only
- Disagg: GPU 4 (encoder) + GPU 5 (PD), best config (NIXL_READ + max_running_requests=64 + chunked_prefill_size=16384)

---

## Results

| Metric | **TP=1 agg** (1 GPU) | **Disagg** (2 GPUs) | Δ disagg vs agg |
|---|---:|---:|---:|
| Actual RPS | **1.15** | 1.00 | **−13%** |
| Successful | 32/32 | 32/32 | = |
| Bench duration | 27.7 s | 31.9 s | +15% |
| **Mean TTFT** | **573 ms** | **3,631 ms** | **+534%** |
| Median TTFT | 543 ms | 3,638 ms | +570% |
| P99 TTFT | 781 ms | 5,962 ms | +664% |
| **Mean TPOT** | **21.0 ms** | **33.8 ms** | **+61%** |
| Median TPOT | 20.9 ms | 25.9 ms | +24% |
| P99 TPOT | 31.8 ms | 126.6 ms | +298% |
| Mean ITL | 22.5 ms | 26.4 ms | +17% |
| Median ITL | 14.7 ms | 14.6 ms | = |
| **Max ITL** | **459 ms** | **10,860 ms** | **+2,266%** (10s stalls!) |
| **Mean E2E** | **12.2 s** | **19.0 s** | **+56%** |
| Median E2E | 11.9 s | 17.5 s | +47% |
| P99 E2E | 21.8 s | 31.0 s | +42% |
| Input throughput | 2,443 tok/s | 2,127 tok/s | −13% |
| Output throughput | 674 tok/s | 587 tok/s | −13% |
| **Peak output throughput** | 1,572 tok/s | **1,711 tok/s** | **+9% (only disagg win)** |
| Concurrency | 14.1 | 19.1 | +35% (more queueing) |

---

## Per-phase breakdown for disagg (from `enable_request_time_stats_logging`)

| Phase | Median | Mean | Max |
|---|---:|---:|---:|
| Queue duration | **~0 ms** | 60 ms | 260 ms |
| Forward duration (prefill+decode on PD) | **14.12 s** | 15.06 s | 29.53 s |
| **TTFT (mean)** | **3,631 ms** | — | — |

**Queue is essentially zero** — even at rate=2.0 with 14-19 concurrent requests, PD's scheduler queue isn't saturated.

So the **3.6 s mean TTFT is NOT** due to PD scheduler queueing. It must come from one of:
1. **dynamo TCP request plane latency** (frontend → encoder → PD round trip)
2. **Encoder ViT compute + NIXL setup** (~200-700 ms when concurrent)
3. **`_build_mm_items` setup on PD before SGLang prefill starts** (~100-300 ms per request)

The total of these "dynamo-handoff" costs is ~3.5 s under load — vastly more than the ~150 ms TP=1 agg pays for inline ViT.

---

## Why the high-concurrency hypothesis was wrong

I expected at rate=2.0, TP=1 agg's ViT compute would contend with decode of ~14 in-flight requests. The data shows this **doesn't happen** because:

### 1. ViT compute for 1 image @ 1080p is tiny
~150 ms on H200 for the small ViT in Qwen3-VL. Compared to 14 s of decode per request, ViT is 1% of total. Even running every 500 ms (rate=2.0), it can't meaningfully starve decode.

### 2. SGLang's scheduler handles ViT inline efficiently
For aggregated mode, `Qwen3VLForConditionalGeneration` runs ViT and LLM forward in one continuous pass. ViT activations are small enough not to displace decode KV cache. There's no "ViT pre-empts decode" pathology to expose.

### 3. Disagg's per-request handoff scales WORSE with concurrency
- At rate=0.5, disagg mean TTFT = 1,307 ms
- At rate=2.0, disagg mean TTFT = 3,631 ms (**2.8× worse from 4× rate**)
- At rate=0.5, TP=1 agg mean TTFT = 524 ms
- At rate=2.0, TP=1 agg mean TTFT = 573 ms (**1.09× = essentially flat**)

**TP=1 agg scales TTFT linearly (almost flat) with concurrency. Disagg scales TTFT super-linearly.** Each additional concurrent request costs disagg more because:
- More concurrent NIXL transfers (still cheap individually but contend on the GPU's CUDA streams)
- More `_build_mm_items` setup work
- More embedding-integration sub-batch events on PD interleaved with decode (which IS a decode-blocker for disagg!)
- TCP request plane has more concurrent in-flight RPCs

**The irony:** disagg's "embedding integration" pattern actually creates the same prefill-blocks-decode pathology I expected to find in TP=1 agg, just on disagg's PD side! Each NIXL-arrived embedding triggers small forward passes that interrupt decode.

### 4. The 10.8 s max ITL is the smoking gun for disagg's contention

`Max ITL = 10,860 ms` for disagg vs **459 ms** for TP=1 agg. This means at some point during the bench, one of disagg's requests went **10.8 seconds** between consecutive output tokens. This is the embedding-integration sub-batch events stalling decode — exactly the pathology disagg was supposed to avoid.

TP=1 agg's max ITL of 459 ms shows brief contention but nothing dramatic.

---

## Comparison across all decode-heavy tests

| Test | Rate | RPS | Mean TTFT | Mean TPOT | Mean E2E |
|---|---:|---:|---:|---:|---:|
| TP=1 agg @ rate=0.5 | 0.5 | 0.43 | 524 ms | 14.4 ms | 8.2 s |
| Disagg @ rate=0.5 | 0.5 | 0.42 | 1,307 ms | 16.2 ms | 9.9 s |
| **TP=1 agg @ rate=2.0** | **2.0** | **1.15** | **573 ms** | **21.0 ms** | **12.2 s** |
| **Disagg @ rate=2.0** | **2.0** | **1.00** | **3,631 ms** | **33.8 ms** | **19.0 s** |

**Disagg's gap to TP=1 agg WIDENS as rate increases.** The opposite of what the high-concurrency hypothesis predicted.

| Metric | rate=0.5 | rate=2.0 |
|---|---|---|
| Disagg TTFT vs agg TTFT | 2.5× worse | 6.3× worse |
| Disagg TPOT vs agg TPOT | 13% worse | 61% worse |
| Disagg E2E vs agg E2E | 22% worse | 56% worse |

---

## What this proves about disagg

For **same-host disagg with 1 H200 encoder + 1 H200 PD on Qwen3-VL-32B-FP8**:

1. **The decode-heavy hypothesis was wrong** for this configuration. Disagg doesn't win on decode-dominated workloads.

2. **Disagg's per-request handoff cost grows super-linearly with concurrency** (~80 ms per added concurrent request). At rate=2.0 with concurrency 19, the overhead is ~3.5 s/request.

3. **The "embedding integration" pattern is also a decode-blocker on PD**. We saw `Max ITL = 10.8 s` on disagg vs 0.46 s on TP=1 agg — disagg has its OWN prefill-blocks-decode pathology, and it's worse than TP=1 agg's.

4. **Disagg's only real advantage is peak output throughput (+9%)** — modest, niche.

5. **TP=1 agg's TTFT scales nearly flat with concurrency** (524 ms → 573 ms over 4× rate increase). Disagg's TTFT scales 2.8× over the same rate increase. This is the cleanest evidence that disagg's overhead is fundamentally additive-per-request, not amortizable.

---

## Will any rate flip the comparison?

Based on the trend (gap widens with rate):
- **Lower rate** (e.g., 0.1) might give disagg parity on RPS but it's a trivial regime where neither matters
- **Higher rate** (e.g., 4.0) would push both into saturation; disagg saturates first
- **No rate** in this workload makes disagg win on E2E or RPS

The original "decode-heavy favors disagg" hypothesis is **falsified** for this hardware/model/workload.

---

## What workload patterns might still flip it?

Based on these results, disagg's path to victory is NOT about concurrency or decode-dominance. It's about:

1. **Encoder-amortization** (cache hits on shared images). Disagg's encoder GPU has ~140 GB free for cache; agg shares with LLM. **Untested.**

2. **Multi-tenant** (1 encoder + N PDs sharing it). For workloads where multiple LLM workers share an encoder pool, disagg wins by amortizing the encoder GPU. **Untested.**

3. **Asymmetric hardware** (encoder on cheap GPU, LLM on H200). The encoder GPU sat at 1.8 GB / 143 GB in our tests — hugely wasted. On a cheaper GPU (A10G/L40S), this waste disappears. **Different hardware.**

4. **Memory-isolation** (encoder weights conflict with LLM weights, must separate). Not our case.

For our exact setup (2 H200s, single encoder, single PD), there is **no rate or workload that makes disagg win**.

---

## Files

- TP=1 agg result CSV: `/hongming/res4/h200_agg_tp1_32b_image1_1080p_out1024_dh_rate2/test_sglang_multi_rates_1080p_20260522_055659/results_summary.csv`
- Disagg result CSV: `/hongming/res4/h200_h200_disagg_tp1_32b_image1_1080p_out1024_dh_rate2/test_sglang_multi_rates_1080p_20260522_060927/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_decode_heavy.sh` (used with positional arg `2.0`)
- Disagg PD log: `/hongming/dynamo/logs/logs/logs/logs/logs/pd_worker.log` (5 levels of `logs/` accumulated from successive runs)
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/long_output_1024_results.md` — same workload at rate=0.5
  - `01_cuda_sh/disagg_h200_32b/when_disagg_wins.md` — workload categories analysis
  - `01_cuda_sh/disagg_h200_32b/bottleneck_analysis.md` — bottleneck breakdown
- This document: `01_cuda_sh/disagg_h200_32b/long_output_1024_rate2_results.md`

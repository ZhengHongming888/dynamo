# Decode-Heavy Workload — TP=1 agg vs Disagg (1 H200 encoder + 1 H200 PD)

**Goal:** Test the hypothesis that **decode-heavy workload** (long output, short input) would let disagg structurally win because PD's dedicated GPU is freed from ViT contention.

**Result:** **Disagg did NOT win.** TP=1 agg was slightly better on every metric except peak output throughput.

---

## Test setup

**Workload:**
- 1 image @ 1920×1080 (~2,100 vision tokens)
- 128 input text tokens
- **1024 output tokens** (vs the 256 of our prefill-heavy test)
- 64 prompts, rate=0.5 req/s

**Decode/prefill ratio:** ~1024 / 2228 = **46% decode** by token count, but by *time* decode dominates because each output token takes ~14 ms × 1024 = ~14 s vs prefill ~200 ms.

**Configurations tested:**
- TP=1 agg: GPU 4 only, `start_h200_aggregate_epd_server_32b_tp1.sh`
- Disagg: GPU 4 (encoder) + GPU 5 (PD), best config (NIXL_READ + max_running_requests=64 + chunked_prefill_size=16384)

**Bench script:** `/hongming/dynamo/test_sglang_decode_heavy.sh` (variant of the standard bench with INPUT_LEN=128, OUTPUT_LEN=1024, NUM_IMAGES=1, DEFAULT_RATES=(0.5))

---

## Results

| Metric | **TP=1 agg** (1 GPU) | **Disagg** (2 GPUs) | Δ |
|---|---:|---:|---:|
| Actual RPS | **0.43** | 0.42 | −2% |
| Successful requests | 64/64 | 64/64 | = |
| Bench duration | 149.4 s | 151.5 s | +1% |
| **Mean TTFT** | **524 ms** | **1,307 ms** | **+150% (worse)** |
| Median TTFT | 514 ms | 1,236 ms | +140% |
| P99 TTFT | 639 ms | 2,095 ms | +228% |
| **Mean TPOT** | **14.4 ms** | **16.2 ms** | **+13%** |
| Median TPOT | 14.2 ms | 15.7 ms | +10% |
| P99 TPOT | 17.5 ms | 23.7 ms | +35% |
| Mean ITL | 15.6 ms | 17.5 ms | +12% |
| Median ITL | 13.0 ms | 12.9 ms | = |
| Max ITL | 322 ms | 1,177 ms | +266% |
| **Mean E2E** | **8.15 s** | **9.93 s** | **+22% (worse)** |
| Median E2E | 8.29 s | 10.15 s | +22% |
| P99 E2E | 15.6 s | 19.2 s | +23% |
| Input throughput | 908 tok/s | 896 tok/s | = |
| Output throughput | 226 tok/s | 223 tok/s | = |
| **Peak output throughput** | 649 tok/s | **792 tok/s** | **+22% (disagg wins)** |
| Concurrency | 3.5 | 4.2 | +20% |

---

## Per-phase breakdown for disagg (from `enable_request_time_stats_logging`)

| Phase | Median | Mean | Max |
|---|---:|---:|---:|
| Queue duration | **0 ms** | 0 ms | 60 ms |
| Forward duration | **9.07 s** | 8.55 s | 19.86 s |

**Queue is essentially zero** — at rate=0.5 with `max_running_requests=64`, the system is NOT saturated. This means we're testing the per-request cost in isolation, not the saturation knee.

## Prefill stats for disagg

| Metric | Value |
|---|---:|
| Most common chunk sizes | 2,064-2,192 (1 image + 128 text fits in 1 chunk) |
| Embedding-integration small batches | only 2 events (vs 53 in prefill-heavy test) |
| Median input throughput | 1,325 tok/s (chunks are small, GPU underutilized per chunk) |
| Peak input throughput | 22,980 tok/s |
| Decode events | 284 |
| Median decode gen throughput | 157 tok/s |
| Max decode gen throughput | 790 tok/s |

---

## Why disagg lost

### 1. TTFT penalty is huge for short prefill

| Cost component | TP=1 agg | Disagg |
|---|---:|---:|
| ViT encode (1 image @ 1080p) | inline ~150 ms | parallel ~150 ms (hidden) |
| **NIXL handoff round-trip** | 0 | **~700-1000 ms** |
| `_build_mm_items` setup on PD | 0 | ~100-300 ms |
| Prefill (~2k tokens) | ~200 ms | ~200 ms |
| **TTFT total** | **~524 ms** | **~1,307 ms** (+783 ms penalty) |

The ~700 ms NIXL+handoff penalty is **fixed** regardless of workload. For prefill-heavy workload (8 images, ~16k tokens), this is hidden inside the ~3-4 s prefill time. For decode-heavy with 1 image (only 2k tokens prefill), it's exposed.

### 2. Decode TPOT is virtually identical (+13%)

The hypothesis was that disagg's PD-only GPU would have cleaner decode (no ViT contention). At rate=0.5 we don't see this:
- TP=1 agg TPOT: 14.4 ms
- Disagg TPOT: 16.2 ms

Difference is small. Both are close to **native single-H200 decode speed** for 32B FP8. ViT contention during decode wasn't a meaningful problem at this concurrency level.

### 3. Concurrency too low to expose decode contention

The test ran at concurrency ~3-4, far below `max_running_requests=64`. At low concurrency:
- TP=1 agg's "prefill blocks decode" pathology rarely triggers (few prefills happening)
- Disagg's "dedicated decode GPU" advantage doesn't materialize because there's no contention to avoid

The decode-heavy hypothesis would need **much higher concurrency** to expose disagg's structural advantage. At rate=2.0 or higher with this workload, things might change.

### 4. Disagg DID win on peak output throughput (+22%)

The one metric where disagg won: **peak output throughput 792 vs 649 tok/s**. This is the per-window decode burst rate when many requests happen to be decoding simultaneously. Disagg's PD GPU can hit higher decode bursts because:
- Encoder GPU is doing nothing → no power/thermal contention
- KV cache is fully dedicated to LLM (no ViT activations sharing memory pool)

But this peak-throughput win **doesn't translate to mean output throughput** (both at 223-226 tok/s) because peak only matters in brief bursts.

---

## Comparison table — prefill-heavy vs decode-heavy

| Metric | Prefill-heavy (8 imgs, 256 out) | Decode-heavy (1 img, 1024 out) |
|---|---|---|
| TP=1 agg RPS @ rate=1.0 | 0.52 | n/a (rate=0.5 here) |
| Disagg RPS @ rate=1.0 | 0.23 | n/a |
| TP=1 agg E2E | 81 s | 8.15 s |
| Disagg E2E | 245 s | 9.93 s |
| Disagg/TP=1 RPS ratio | 0.44 | 0.98 |
| Disagg/TP=1 E2E ratio | 3.0× worse | 1.22× worse |
| Disagg "wins" on | nothing | peak output tput only |

**The decode-heavy workload makes disagg less of a loser** (3× worse → 1.2× worse on E2E), but it's still not a win.

---

## What we learned

1. **The decode-heavy hypothesis was directionally right but quantitatively wrong.** Disagg's gap to TP=1 agg shrinks dramatically (from 3× worse to 1.2× worse) when you stop loading it with prefill, but it doesn't actually flip to a win.

2. **The fixed ~700-1000 ms NIXL handoff overhead is the killer for short-prefill workloads.** Per-request handoff cost is constant; it stops mattering when prefill is large (16k tokens), starts hurting again when prefill is small (2k tokens).

3. **Peak output throughput is genuinely better in disagg** (+22%). For workloads that care about output burst rate (e.g., interactive streaming where you want fastest possible token-by-token feel during decode), disagg has real value.

4. **At rate=0.5, neither config is saturated.** Both run at ~3-4 concurrent requests. To test the "decode contention" hypothesis we'd need much higher concurrency to actually expose the prefill-blocks-decode pathology in TP=1 agg.

---

## What would actually make disagg win for decode-heavy?

Two possible follow-ups that could flip this:

### Option A: Higher rate (e.g., rate=2.0) with same workload

At rate=2.0 with each request taking ~14 s, equilibrium concurrency would be ~28. At that load:
- TP=1 agg: ViT for 1 new image every 0.5 s contends with decode of 28 in-flight requests → prefill-blocks-decode pathology starts to bite
- Disagg: encoder produces embeddings continuously, PD's decode is uninterrupted → potential structural win

Predicted outcome: disagg RPS may match or modestly exceed TP=1 agg, TPOT advantage becomes real (~30-50% lower).

### Option B: Even longer outputs (4096 or 8192)

Push output to 4096 tokens. Now per-request decode is ~57 s. The fixed 700 ms NIXL handoff becomes 1.2% of total time instead of 9%. Disagg's overhead becomes negligible.

Predicted outcome: at rate=0.5 with 4096 output, disagg E2E ≈ TP=1 agg E2E. At higher rate, disagg might marginally win on TPOT.

---

## Files

- TP=1 agg result CSV: `/hongming/res4/h200_agg_tp1_32b_image1_1080p_out1024_decodeheavy/test_sglang_multi_rates_1080p_20260522_052648/results_summary.csv`
- Disagg result CSV: `/hongming/res4/h200_h200_disagg_tp1_32b_image1_1080p_out1024_decodeheavy/test_sglang_multi_rates_1080p_20260522_054230/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_decode_heavy.sh` (INPUT_LEN=128, OUTPUT_LEN=1024, NUM_IMAGES=1, DEFAULT_RATES=(0.5))
- Disagg PD log: `/hongming/dynamo/logs/logs/logs/logs/pd_worker.log` (4 levels of `logs/`)
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md`
  - `01_cuda_sh/disagg_h200_32b/deep_analysis_disagg_worse_h200.md`
  - `01_cuda_sh/disagg_h200_32b/disagg_improvements_attempts.md`
  - `01_cuda_sh/disagg_h200_32b/bottleneck_analysis.md`
  - `01_cuda_sh/disagg_h200_32b/when_disagg_wins.md`
- This document: `01_cuda_sh/disagg_h200_32b/long_output_1024_results.md`

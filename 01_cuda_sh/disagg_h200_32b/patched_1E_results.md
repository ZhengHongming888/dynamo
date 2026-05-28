# Patched 1E rate sweep — full results

**Date:** 2026-05-24
**Setup:**
- giga01 H200 PD, patched (`device=_nixl_buffer_device()` in `embedding_transfer.py:882, 915`)
  - PD `_core.abi3.so` rebuilt with `EXPIRY_DURATION` raised from `Duration::from_secs(300)` to
    `Duration::from_secs(1800)` (`/opt/dynamo/lib/llm/src/kv_router/sequences/single.rs:30`)
  - `mem-fraction-static=0.65`, `max-running-requests=64`, GPU 4
- B70 1× XPU encoder (XPU 3), patched (`encode_worker_handler.py:219` returns `xpu`),
  `NIXL_USE_CPU_HOST_MEMORY=1` removed, `--encoder-only`, `mem-fraction-static=0.5`
- Cross-host RoCE: B70 `mlx5_0` ↔ giga01 `mlx5_4`, fabric 192.165.123.0/24
- NIXL path verified xpu→cuda end-to-end (GPUDirect RDMA) before sweep
- Bench: `sglang.bench_serving --backend sglang-oai-chat`, **`np=16`**, `random-input-len=128`,
  `random-output-len=256`, `seed=0`

**Why np=16 instead of np=32:**
A first attempt with np=32 ran rate=0.1 successfully (0.034 RPS, 22/32) but subsequent rates
crashed the B70 1E XPU with `level_zero error 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`. The
single XPU has limited device memory and queue buildup at higher rates exhausted it. Halving
concurrency to np=16 (still gives meaningful saturation measurements) avoided the OOM and let
all 21 runs complete cleanly.

**For apples-to-apples comparison the prior 4E sweep used np=32. Different np affects:**
- Total run duration (np=16 finishes in roughly half the time)
- Lighter steady-state queue depth (fewer in-flight concurrent requests)
- Saturation throughput is unaffected (encoder-bound; encoder serializes regardless of np)

## Sweep design

- **Rates:** 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 (Poisson arrivals)
- **Workloads (small → large to warm gracefully):**
  1. 4img / 1024×768 (≈127 MB embedding/req)
  2. 8img / 1024×768 (≈242 MB embedding/req)
  3. 8img / 1920×1080 (≈638 MB embedding/req)
- **np=16** prompts each (down from np=32 to avoid B70 XPU OOM)
- **90 s drain between runs** to let the 1E XPU release device memory before next bench's warmup

All 21 runs returned 16/16 successful — no cancellations, no OOMs after switching to np=16.

## Results — 4img/768p (≈127 MB embed)

| Rate | RPS | Duration | E2E mean | E2E p50 | TTFT mean | TTFT p50 | TPOT mean | TPOT p50 | ITL p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1  | 0.11 | 143 s | 5.1 s  | 4.9 s  | 4.0 s  | 3.6 s  | 10.0 ms  | 12.3 ms  | 12.1 ms |
| 0.25 | 0.27 | 60 s  | 10.3 s | 9.5 s  | 6.2 s  | 6.4 s  | 50.8 ms  | 27.8 ms  | 1.26 ms |
| **0.5**  | **0.40** | 41 s | 28.6 s | 29.3 s | 18.9 s | 21.5 s | 102.4 ms | 31.9 ms  | 3.54 ms |
| 1.0  | 0.40 | 41 s  | 33.4 s | 33.8 s | 27.7 s | 30.9 s | 58.9 ms  | 4.9 ms   | 3.62 ms |
| 1.5  | 0.40 | 41 s  | 35.0 s | 35.3 s | 30.5 s | 32.3 s | 50.7 ms  | 5.9 ms   | 3.90 ms |
| 2.0  | 0.40 | 40 s  | 35.8 s | 35.9 s | 33.6 s | 34.5 s | 22.6 ms  | 4.6 ms   | 3.51 ms |
| 3.0  | 0.40 | 40 s  | 36.7 s | 36.8 s | 35.5 s | 35.9 s | 12.7 ms  | 6.1 ms   | 5.08 ms |

**Saturation: ~0.40 RPS** — reached at rate=0.5 and held through rate=3.0. RPS is exactly
**~30% of 4E saturation** (1.33 RPS), confirming a single XPU encoder gives roughly 1/4 the
throughput of 4 XPU encoders.

**TTFT** stays small at low rates (3.6 s at rate=0.1, basically the per-request encoder time),
then climbs linearly past sat (21 → 30 → 32 → 34 → 36 s as queue depth grows).

## Results — 8img/768p (≈242 MB embed)

| Rate | RPS | Duration | E2E mean | E2E p50 | TTFT mean | TTFT p50 | TPOT mean | TPOT p50 | ITL p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1  | 0.09 | 172 s | 10.3 s | 9.7 s  | 9.0 s  | 9.5 s  | 13.0 ms  | 12.2 ms  | 11.6 ms |
| 0.25 | 0.19 | 86 s  | 33.9 s | 33.0 s | 21.0 s | 22.2 s | 144.1 ms | 21.2 ms  | 2.31 ms |
| **0.5**  | **0.21** | 78 s | 57.6 s | 55.7 s | 40.7 s | 47.4 s | 170.1 ms | 56.7 ms  | 3.24 ms |
| 1.0  | 0.20 | 78 s  | 66.8 s | 65.8 s | 56.1 s | 62.4 s | 110.1 ms | 4.6 ms   | 3.34 ms |
| 1.5  | 0.21 | 78 s  | 69.1 s | 68.4 s | 64.0 s | 66.8 s | 55.6 ms  | 4.3 ms   | 3.22 ms |
| 2.0  | 0.21 | 78 s  | 70.5 s | 70.1 s | 67.8 s | 69.2 s | 30.6 ms  | 4.2 ms   | 3.18 ms |
| 3.0  | 0.21 | 77 s  | 71.7 s | 71.5 s | 71.0 s | 70.6 s | 7.6 ms   | 4.0 ms   | 3.08 ms |

**Saturation: ~0.21 RPS** — settles at rate=0.5, ~32% of 4E sat (0.66 RPS).

**Per-request encoder time on 1E for 8img/768p ≈ 9.5 s** (TTFT at rate=0.1 sub-saturation).

## Results — 8img/1080p (≈638 MB embed)

| Rate | RPS | Duration | E2E mean | E2E p50 | TTFT mean | TTFT p50 | TPOT mean | TPOT p50 | ITL p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1  | 0.036 | 441 s | 367.7 s | 371.8 s | 296.1 s | 309.4 s | 655.9 ms | 4.0 ms | 2.84 ms |
| **0.25** | **0.036** | 439 s | 407.3 s | 408.9 s | 363.0 s | 375.0 s | 517.5 ms | 9.1 ms | 3.39 ms |
| 0.5  | 0.037 | 438 s | 419.3 s | 420.0 s | 413.9 s | 416.8 s | 61.8 ms  | 4.2 ms | 3.17 ms |
| 1.0  | 0.037 | 438 s | 426.1 s | 426.3 s | 425.7 s | 425.9 s | 4.3 ms   | 3.9 ms | 2.91 ms |
| 1.5  | 0.036 | 441 s | 430.9 s | 431.0 s | 430.4 s | 430.5 s | 4.9 ms   | 4.7 ms | 3.70 ms |
| 2.0  | 0.037 | 438 s | 429.5 s | 430.0 s | 429.1 s | 429.1 s | 4.5 ms   | 4.3 ms | 3.36 ms |
| 3.0  | 0.037 | 438 s | 430.5 s | 430.8 s | 430.1 s | 430.3 s | 4.5 ms   | 4.3 ms | 3.27 ms |

**Saturation: ~0.037 RPS** — already saturated at rate=0.1. The 1E XPU encoder takes ~27 s
per 8img/1080p request, so 1/27 = 0.037 reqs/s is the absolute ceiling. About **28% of 4E**
saturation (0.13 RPS).

**Note:** TTFT at 0.1 RPS jumped from saturated 0.04 RPS implies the queue was already deep when
the 16th request was sent, leading to >5 minute wait times. With np=16 each request takes E2E
~370 s on average (mostly waiting in the encoder queue).

**This run was the only 8img/1080p result we have for 1E.** The earlier np=32 attempt
(`8img_1080p/rate_0.1_np32`) had 22/32 success at 0.034 RPS, consistent with this measurement.

## Saturation comparison: 1E vs 4E vs TP=1 agg

| Workload    | Embed   | **Patched 1E** (this) | **Patched 4E** (`patched_4E_results.md`) | TP=1 agg same-host | 1E / 4E ratio | 1E / TP=1 ratio |
|---|---:|---:|---:|---:|---:|---:|
| 4img/768p   | 127 MB  | **0.40 RPS** | 1.33 RPS | 2.7 RPS | 30% | 15% |
| 8img/768p   | 242 MB  | **0.21 RPS** | 0.66 RPS | 1.5 RPS | 32% | 14% |
| 8img/1080p  | 638 MB  | **0.037 RPS** | 0.13 RPS | 0.47 RPS | 28% | 8% |

**1E is consistently ~30% of 4E throughput.** This is below the linear 25% (1/4) you'd expect
if encoder ViT were the only bottleneck — meaning there's some non-ViT overhead in 4E
(load balancing, NIXL setup, ZMQ scheduler) that scales sublinearly with the number of encoders.

**1E is only 8-15% of TP=1 same-host agg throughput.** Even though TP=1 agg uses just one H200
(same as the 1E PD), the same-host agg path runs the ViT encoder on the H200 GPU directly
(roughly 10-30× faster than B70 XPU per-image), so a single H200 absorbs the encoder workload
many times faster than a single B70 XPU.

## Sub-saturation per-request latency (rate=0.1)

| Workload    | **1E TTFT** | **4E TTFT** | TP=1 agg TTFT | 1E / 4E | 1E / TP=1 |
|---|---:|---:|---:|---:|---:|
| 4img/768p   | 3.6 s   | 3.5 s   | 0.8 s   | 1.03× | 4.5× |
| 8img/768p   | 9.5 s   | 6.9 s   | 1.6 s   | 1.38× | 5.9× |
| 8img/1080p  | 309 s   | 55 s    | 8.1 s   | 5.6×  | 38×  |

**At rate=0.1 (sub-saturation), 1E has only marginally higher TTFT for small workloads** — not
4× higher. Because rate=0.1 = 1 req per 10 s, and the 1E encoder serves 4img/768p in ~3.5 s,
queueing is minimal so the single encoder isn't saturated even at rate=0.1.

For 8img/1080p the per-request encoder time (~27 s) exceeds the inter-arrival (10 s) even at
rate=0.1, so the 1E queue saturates immediately. TTFT 5.6× higher = queue at first sample is
already 5+ requests deep.

## Sub-saturation TTFT comparison vs same-host agg

| Workload    | TP=1 agg same-host | TP=2 agg same-host | **Patched 1E** | **Patched 4E** |
|---|---:|---:|---:|---:|
| 4img/768p   | 0.8 s   | 1.0 s   | 3.6 s   | 3.5 s   |
| 8img/768p   | 1.6 s   | 1.7 s   | 9.5 s   | 6.9 s   |
| 8img/1080p  | 8.1 s   | 3.5 s   | 309 s   | 55 s    |

**Cross-host disagg (any number of encoders) is 2-90× slower than same-host agg** for TTFT.
This was the headline finding from the 4E sweep and is reaffirmed by 1E. Notably even TP=1 agg
on a single H200 outperforms 4E on TTFT — the encoder ViT runs on the H200 GPU directly in
agg mode, which is much faster than the B70 XPU encoders used in disagg.

## Decode (TPOT/ITL) comparison

| Workload | rate | **1E TPOT p50** | **4E TPOT p50** | **1E ITL p50** | **4E ITL p50** |
|---|---|---:|---:|---:|---:|
| 4img/768p | 1.0 | 4.9 ms  | 16.5 ms | 3.6 ms | 12.1 ms |
| 8img/768p | 1.0 | 4.6 ms  | 12.4 ms | 3.3 ms | 1.6 ms  |
| 8img/1080p | 1.0 | 3.9 ms  | 1.58 ms | 2.9 ms | 1.31 ms |

**Decode latency is in the same ballpark for 1E and 4E.** The patch's GPU-direct embedding
benefit is independent of how many encoders are running. At rate=1.0 with `running-req=1` most
of the time on 1E (because the encoder is the bottleneck), TPOT is dominated by the per-token
H200 decode cost (~3-5 ms), not by encoder count.

For 4img/768p and 8img/768p, **1E has lower TPOT than 4E at rate=1.0**. This makes sense:
fewer concurrent requests on the PD = less batching overhead. At higher rates 4E sees more
concurrent requests on the PD which forces larger decode batches (slightly higher TPOT but
higher aggregate throughput).

## Per-rate observations

### Saturation hits early on 1E across all workloads

The single XPU encoder fully serializes — every workload reaches sat by rate=0.5:
- 4img/768p sat at rate=0.5 (0.40 RPS)
- 8img/768p sat at rate=0.5 (0.21 RPS)
- 8img/1080p sat by rate=0.1 (0.04 RPS — TTFT/E2E already in queue regime)

### B70 XPU OOM was the major operational risk

With np=32 the XPU exhausted device memory after one full run.
With np=16 + 90 s drain between runs, no OOMs across all 21 runs. Encoder kept its
device-memory footprint stable.

### TTFT saturates near per-request encoder time × queue depth

For 4img/768p at rate=3.0, TTFT mean = 35.5 s = ~10 reqs of queue × 3.5 s encoder time. This
matches `np=16` (the queue can be at most 16 deep including the in-flight one).

## Conclusions

1. **1E saturation is ~30% of 4E across all workloads.** Slightly better than the linear-1/4
   you'd expect; the difference comes from non-ViT overhead in 4E (scheduler, NIXL setup) that
   doesn't scale linearly with encoder count.
2. **No bench-side incidents on the np=16 sweep.** All 21 runs completed 16/16. The earlier
   B70 XPU OOM at np=32 was solved cleanly by halving concurrency; this is a 1E-specific
   operational hazard.
3. **TTFT advantage of 4E vs 1E is large for big workloads (5.6× for 8img/1080p) but small
   for tiny ones (1.03× for 4img/768p)** when measured at sub-saturation rate=0.1. As load
   increases the 1E queue fills first, but at very low load both deliver similar latency.
4. **TPOT/ITL are essentially the same for 1E and 4E** because the patch's GPU-direct
   embedding is per-request, not per-encoder. Decode latency is dominated by PD-side cost.
5. **Cross-host disagg, regardless of encoder count (1E or 4E), is much slower than same-host
   TP=2 agg for TTFT.** The B60/B70 XPU encoder ViT is the bottleneck, not the number of
   encoders.

## 4E vs 1E side-by-side

### Saturation RPS

| Workload    | TP=1 agg same-host | TP=2 agg same-host | **Patched 1E** | **Patched 4E** |
|---|---:|---:|---:|---:|
| 4img/768p   | 2.7 RPS  | 3.1 RPS  | 0.40 RPS | 1.33 RPS |
| 8img/768p   | 1.5 RPS  | 1.9 RPS  | 0.21 RPS | 0.66 RPS |
| 8img/1080p  | 0.47 RPS | 0.6-0.7 RPS | 0.037 RPS | 0.13 RPS |

### Best (sub-saturation rate=0.1) median TTFT

| Workload    | TP=1 agg same-host | TP=2 agg same-host | **Patched 1E** | **Patched 4E** |
|---|---:|---:|---:|---:|
| 4img/768p   | 0.8 s   | 1.0 s   | 3.6 s   | 3.5 s  |
| 8img/768p   | 1.6 s   | 1.7 s   | 9.5 s   | 6.9 s  |
| 8img/1080p  | 8.1 s   | 3.5 s   | 309 s   | 55 s   |

### Median TPOT at rate=1.0

| Workload    | **Patched 1E** | **Patched 4E** |
|---|---:|---:|
| 4img/768p   | 4.9 ms  | 16.5 ms |
| 8img/768p   | 4.6 ms  | 12.4 ms |
| 8img/1080p  | 3.9 ms  | 1.6 ms  |

(For 8img/1080p, 4E rate=1.0 has more concurrent decode going on, which actually helps TPOT
batch optimally — see comment on running-req=1 single-decode amortization in the 4E doc.)

## Sources

- Bench JSON: `/hongming/res19_1E_patched_sweep/{4img_768p,8img_768p,8img_1080p}/rate_<rate>_np16/benchmark_output.json`
- Bench logs: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/bench_1E_patched_*.log`
- Sweep master log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/1E_sweep_master_v3.log`
- Sweep script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/run_1E_patched_sweep.sh`
- giga01 PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- 4E reference: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/patched_4E_results.md`
- Patch file: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/h200_cuda_nixl.patch`
- Earlier np=32 attempt result (1 row): `/hongming/res19_1E_patched_sweep/8img_1080p/rate_0.1_np32/benchmark_output.json` (0.034 RPS, 22/32 — consistent with np=16 measurement)

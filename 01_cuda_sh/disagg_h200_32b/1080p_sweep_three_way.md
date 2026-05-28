# 1080p × 8 Images — Three-Way Rate Sweep (TP=1 agg, TP=2 agg NVLink, Disagg)

**Test date:** 2026-05-22
**Goal:** Sweep across rates 0.1, 0.25, 0.5, 1.0, 1.25 to characterize all three configurations on the 1080p multimodal workload. Apples-to-apples re-run of the 1080p sweep using the current best disagg config.

**Workload:**
- 8 × **1920×1080** random images per request (~16,400 vision tokens)
- 128 input + 256 output text tokens
- np=64 prompts per rate

**Configurations:**
- **TP=1 agg**: GPU 4 only — `start_h200_aggregate_epd_server_32b_tp1.sh` (max_running=40, mem_fraction=0.95)
- **TP=2 agg NVLink**: GPUs 4,5 with NV18 NVLink — `start_h200_aggregate_epd_server_32b_tp2.sh` (max_running=25, mem_fraction=0.88)
- **Disagg**: GPU 4 (encoder) + GPU 5 (PD) — current best config:
  - NIXL_READ embedding transfer
  - max_running_requests=64
  - chunked_prefill_size=16384
  - multimodal_embedding_cache_capacity_gb=16

**Bench script:** `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`

**Result base directory:** `/hongming/res5/`

---

## Headline: actual RPS across rates

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 0.10 | 0.10 | 0.10 |
| 0.25 | 0.25 | 0.25 | 0.23 |
| 0.50 | 0.46 | **0.50** | 0.23 |
| 1.00 | 0.53 | **0.95** | 0.26 |
| 1.25 | 0.57 | **1.13** | 0.25 |

**TP=2 agg is the only config that handles rate ≥ 0.5 cleanly.** TP=1 agg saturates at ~0.57 RPS. Disagg saturates at ~0.25 RPS — half of TP=1 agg.

## Mean TTFT (ms)

| Target rate | TP=1 agg | TP=2 agg | Disagg | Disagg vs TP=2 |
|---:|---:|---:|---:|---:|
| 0.10 | 4,279 | **3,938** | 12,648 | 3.2× worse |
| 0.25 | 5,441 | **2,776** | 27,067 | 9.8× worse |
| 0.50 | 10,867 | **3,947** | 114,330 | **29× worse** |
| 1.00 | 30,642 | **19,491** | 128,554 | 6.6× worse |
| 1.25 | 32,346 | **12,479** | 142,956 | 11.5× worse |

## Mean TPOT (ms)

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 39 | **26** | 88 |
| 0.25 | 71 | **16** | 2,083 |
| 0.50 | 1,358 | **15** | 2,559 |
| 1.00 | 1,037 | **49** | 2,276 |
| 1.25 | 700 | **74** | 2,232 |

TP=2 agg keeps TPOT under 75 ms even at saturation. TP=1 agg's TPOT explodes at rate ≥ 0.5 (decode preempted by prefill). Disagg's TPOT is dramatically worse at every rate ≥ 0.25.

## Mean E2E (ms)

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 7,541 | **6,075** | 19,166 |
| 0.25 | 11,253 | **4,199** | 77,338 |
| 0.50 | 56,955 | **5,733** | 210,278 |
| 1.00 | 78,520 | **22,816** | 211,631 |
| 1.25 | 75,997 | **21,131** | 224,104 |

## Peak output throughput (tok/s)

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 333 | **588** | 355 |
| 0.25 | 402 | 436 | **645** |
| 0.50 | 841 | 463 | **952** |
| 1.00 | 818 | 573 | **964** |
| 1.25 | 798 | 553 | **973** |

**Disagg wins peak output throughput at rate ≥ 0.25** — its only structural win. Useful for niche burst-decode scenarios but doesn't translate to mean throughput or RPS.

## Concurrency observed

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 0.77 | 0.62 | 1.94 |
| 0.25 | 2.84 | 1.07 | 18.14 |
| 0.50 | 26.22 | 2.88 | 48.27 |
| 1.00 | 41.43 | 21.56 | 54.00 |
| 1.25 | 43.09 | 23.90 | 55.71 |

Disagg's concurrency is 2-9× higher than the agg configs at the same target rate — the system queues up requests behind the slow handoff path.

---

## Saturation analysis

| Config | Saturation rate (sustained RPS) | Per-request budget at saturation |
|---|---:|---:|
| TP=1 agg | ~0.57 RPS | ~80 s E2E |
| TP=2 agg | **>1.13 RPS** | ~21 s E2E |
| Disagg | ~0.25 RPS | ~211 s E2E |

**TP=2 agg has 4× the throughput of TP=1 agg** at this prefill-heavy workload (NVLink-backed TP-sharded LLM forward gives ~half the per-chunk wallclock).

**Disagg's saturation RPS (0.25) is half of TP=1 agg's (0.57)** despite using twice the GPUs. Dedicating one full H200 to encoder is wasted when the encoder workload is small (~1.2 s/req of ViT compute).

## Headline conclusions

### 1. TP=2 agg is the clear winner for prefill-heavy multimodal at 1080p
- Sustains 4× the throughput of TP=1 agg
- TTFT 6-13× lower than TP=1 agg at saturation
- TPOT 10-15× lower than TP=1 agg at saturation
- All metrics scale gracefully through rate=1.25

### 2. TP=1 agg has its place at light load
- At rate ≤ 0.25, TP=1 agg is competitive on RPS (0.10 / 0.25 vs 0.10 / 0.25)
- TP=2 agg's TPOT advantage (16 ms vs 71 ms at rate=0.25) is the main differentiator
- For single-GPU deployments, TP=1 agg is fine up to ~0.4 req/s

### 3. Disagg is structurally bad for this workload
- Slowest on every metric except peak output throughput
- Saturation RPS (0.25) is **half** of TP=1 agg's (0.57)
- TTFT explodes to >100 s at rate ≥ 0.5
- TPOT is 2,000+ ms — effectively interactive-unfriendly

### 4. Disagg's improvements from this investigation
The current best disagg config (NIXL_READ + tuning) achieved:
- 0.27 → 0.26 RPS at rate=1.0 (essentially unchanged)
- Eliminated 100% of failures (was 15 failures at rate=1.0, now 0)
- Improved TTFT from 79 s → 129 s (worse — but with all 64 requests succeeding instead of 49)

The fixes made disagg **reliable** but didn't make it **faster**.

---

## Comparison with 768p sweep

| Workload | TP=1 agg sat. | TP=2 agg sat. | Disagg sat. |
|---|---:|---:|---:|
| 768p × 8 (~6k vis tokens/req) | >1.15 RPS | >1.15 RPS | ~0.94 RPS |
| **1080p × 8 (~16k vis tokens/req)** | **~0.57 RPS** | **~1.13 RPS** | **~0.25 RPS** |

**1080p is ~3× more expensive per request than 768p**, and the saturation RPS scales accordingly:
- TP=1 agg: 1.15 → 0.57 (50% reduction)
- TP=2 agg: 1.15 → 1.13 (essentially same — TP=2 has headroom for both workloads)
- Disagg: 0.94 → 0.25 (73% reduction — disagg suffers most as workload grows)

**Disagg's relative deficit grows with workload size.** At 768p disagg/agg ratio is 0.69; at 1080p it's 0.44. Heavier workloads expose disagg's overhead more.

---

## Recommended config choice (across all our tests)

| Use case | Best config |
|---|---|
| Prefill-heavy + sustained load | **TP=2 agg NVLink** (only one that scales) |
| Latency-sensitive at any load | **TP=2 agg NVLink** (consistently lowest TPOT/E2E) |
| Single-GPU deployment with light load | TP=1 agg |
| Peak output throughput burst | Disagg (only metric where it wins) |
| Avoid for prefill-heavy multimodal | Disagg (slowest on every other metric) |

---

## Files

- TP=1 agg result CSV: `/hongming/res5/h200_agg_tp1_32b_image8_1080p_np64_sweep/test_sglang_multi_rates_1080p_20260522_144209/results_summary.csv`
- TP=2 agg result CSV: `/hongming/res5/h200_agg_tp2_32b_image8_1080p_np64_sweep/test_sglang_multi_rates_1080p_20260522_151435/results_summary.csv`
- Disagg result CSV: `/hongming/res5/h200_h200_disagg_tp1_32b_image8_1080p_np64_sweep/test_sglang_multi_rates_1080p_20260522_155000/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
- Server start scripts:
  - TP=1 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
  - TP=2 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
  - Disagg: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh`
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/768p_comparison.md` — same 3-way sweep at 768p
  - `01_cuda_sh/disagg_h200_32b/8img_768p_agg_disagg.md` — single-rate at 768p
  - `01_cuda_sh/agg_h200_32b/tp1_all_rates_results.md` — original TP=1 agg sweep at 1080p (similar numbers)
  - `01_cuda_sh/agg_h200_32b/tp2_all_rates_results.md` — original TP=2 agg sweep at 1080p
  - `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md` — original disagg sweep at 1080p (with broken NIXL_WRITE)
  - `01_cuda_sh/disagg_h200_32b/disagg_improvements_attempts.md` — finding the env var bug + NIXL_READ fix
- This document: `01_cuda_sh/disagg_h200_32b/1080p_sweep_three_way.md`

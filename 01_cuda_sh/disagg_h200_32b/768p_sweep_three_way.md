# 768p × 8 Images — Three-Way Rate Sweep (TP=1 agg, TP=2 agg NVLink, Disagg)

**Test date:** 2026-05-22
**Goal:** Sweep across rates 0.1, 0.25, 0.5, 1.0, 1.25 to characterize how all three configurations scale on the 768p multimodal workload.

**Workload:**
- 8 × **1024×768** images per request (~6,200 vision tokens, vs ~16,400 for 1080p)
- 128 input + 256 output text tokens
- np=64 prompts per rate

**Configurations:**
- **TP=1 agg**: GPU 4 only, `start_h200_aggregate_epd_server_32b_tp1.sh`
- **TP=2 agg NVLink**: GPUs 4,5 with NV18 NVLink, `start_h200_aggregate_epd_server_32b_tp2.sh`
- **Disagg**: GPU 4 (encoder) + GPU 5 (PD), `start_disagg_h200_32b_combined.sh` with current best config:
  - NIXL_READ embedding transfer
  - max_running_requests=64
  - chunked_prefill_size=16384
  - multimodal_embedding_cache_capacity_gb=16

**Bench script:** `/hongming/dynamo/test_sglang_8img_768p_sweep.sh`

---

## Headline: actual RPS across rates

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 0.09 | 0.09 | 0.09 |
| 0.25 | 0.23 | 0.23 | 0.23 |
| 0.50 | 0.47 | 0.47 | 0.46 |
| 1.00 | **0.92** | **0.93** | 0.85 |
| 1.25 | **1.15** | **1.15** | 0.94 |

**Both agg configs sustain rate=1.25 (1.15 RPS observed). Disagg falls behind at rate=1.00 and saturates at rate=1.25.**

## Mean TTFT (ms)

| Target rate | TP=1 agg | TP=2 agg | Disagg | Disagg/agg ratio |
|---:|---:|---:|---:|---:|
| 0.10 | **1,388** | **1,248** | 3,449 | 2.5–2.8× |
| 0.25 | **929** | **1,020** | 3,114 | 3.1–3.4× |
| 0.50 | **1,097** | **1,128** | 3,662 | 3.2–3.3× |
| 1.00 | **1,451** | **1,527** | 5,303 | 3.5–3.7× |
| 1.25 | **1,704** | **1,720** | 12,740 | **7.4–7.5×** |

Disagg's TTFT is 2.5-3.5× worse at low/medium rates, then explodes at saturation.

## Mean TPOT (ms)

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 14.3 | **9.0** | 19.3 |
| 0.25 | 13.5 | **9.9** | 19.5 |
| 0.50 | 14.1 | **11.1** | 21.8 |
| 1.00 | 13.8 | **10.7** | **96.1** ⚠ |
| 1.25 | 12.2 | **10.4** | **278.0** ⚠ |

TP=2 agg consistently wins TPOT. Disagg's TPOT explodes 10–25× at saturation (rate≥1.0) due to the embedding-integration small-batch pattern stalling decode.

## Mean E2E (ms)

| Target rate | TP=1 agg | TP=2 agg | Disagg | Disagg/TP=2 ratio |
|---:|---:|---:|---:|---:|
| 0.10 | 3,005 | **2,325** | 5,514 | 2.4× |
| 0.25 | 2,488 | **2,161** | 5,203 | 2.4× |
| 0.50 | 2,717 | **2,414** | 5,968 | 2.5× |
| 1.00 | 3,060 | **2,837** | 11,562 | **4.1×** |
| 1.25 | 3,273 | **3,011** | 25,600 | **8.5×** |

## Peak output throughput (tok/s)

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 154 | 228 | 219 |
| 0.25 | 158 | 225 | **272** |
| 0.50 | 265 | 353 | **370** |
| 1.00 | 280 | 488 | **876** |
| 1.25 | 443 | 627 | **1,370** |

**Disagg wins peak output throughput at every rate ≥ 0.25**, increasingly so as rate climbs. This is the only metric where disagg is structurally better.

## Concurrency observed

| Target rate | TP=1 agg | TP=2 agg | Disagg |
|---:|---:|---:|---:|
| 0.10 | 0.28 | 0.22 | 0.52 |
| 0.25 | 0.58 | 0.51 | 1.22 |
| 0.50 | 1.27 | 1.13 | 2.76 |
| 1.00 | 2.83 | 2.63 | **9.87** |
| 1.25 | 3.76 | 3.46 | **23.96** |

Disagg's concurrency is 2× higher than agg at low rates (because of handoff queueing), and explodes 6× higher at saturation.

---

## Saturation analysis

**Where each config hits its knee:**

| Config | Knee location | Saturation RPS |
|---|---|---:|
| TP=1 agg | Above rate=1.25 (still keeping up) | >1.15 |
| TP=2 agg | Above rate=1.25 (still keeping up) | >1.15 |
| Disagg | Between rate=1.0 and rate=1.25 | ~0.94 |

**The 768p workload is light enough that both agg configs handle rate=1.25 cleanly.** Disagg's per-request overhead saturates it earlier, around 0.9-1.0 RPS.

## Headline conclusions

### 1. At rates ≤ 0.5, all three configs serve the same RPS

The actual RPS values are essentially identical (0.09, 0.23, 0.46-0.47). The system handles target throughput regardless of architecture. **Differentiation is in latency, not throughput, at low load.**

### 2. TP=2 agg is the latency winner at all rates

TP=2 agg consistently has the lowest TPOT (~9-11 ms) and lowest E2E (2.2-3.0 s). The NVLink-backed TP shards the LLM forward, halving per-token compute.

### 3. Disagg's TTFT penalty is constant and visible

Even at rate=0.10 (essentially idle), disagg's TTFT is 2.5× worse than agg. The ~1-2 s NIXL handoff overhead is fixed per-request, not amortizable.

### 4. Disagg has its own knee that agg doesn't hit

At rate=1.0, disagg starts choking (TPOT 96 ms vs 11 ms TP=2). At rate=1.25, disagg falls dramatically behind (TPOT 278 ms, E2E 25.6 s). The agg configs barely notice the rate increase.

### 5. Disagg's "win" is only at peak output throughput

For decode bursts (peak tok/s during simultaneous decoding), disagg's PD GPU can hit ~2× higher than TP=2 agg. But this is a niche metric — mean output throughput stays the same as RPS suggests.

---

## Comparison with the 1080p sweep

For prefill-heavy workloads at 8 images:

| Workload | TP=1 agg saturation | TP=2 agg saturation | Disagg saturation |
|---|---:|---:|---:|
| 1080p (~16k vis tokens/req) | ~0.57 RPS | ~1.15 RPS | ~0.27 RPS |
| **768p (~6k vis tokens/req)** | **>1.15 RPS** | **>1.15 RPS** | **~0.94 RPS** |

**Smaller workload uniformly accelerates all three, but the relative ordering is unchanged.** Disagg lags TP=1/TP=2 agg by similar ratios at both resolutions.

The 768p workload reveals that **at light per-request load, TP=1 agg and TP=2 agg perform almost identically on RPS** — TP=2's value is only in TPOT (10 ms vs 14 ms) and absolute E2E (2.4 s vs 2.7 s).

---

## Recommended config choice

Based on this 768p sweep:

| Use case | Best config |
|---|---|
| Latency-sensitive (low TTFT/TPOT/E2E) | **TP=2 agg NVLink** |
| Single-GPU deployment | **TP=1 agg** (essentially same RPS as TP=2 at light load) |
| High peak output burst rate | Disagg (only metric where it wins) |
| High concurrency / sustained load | TP=2 agg, then TP=1 agg |
| Avoid | Disagg (always worst on TTFT/TPOT/E2E for this workload) |

---

## Files

- TP=1 agg result CSV: `/hongming/res4/h200_agg_tp1_32b_image8_768p_np64_sweep/test_sglang_multi_rates_1080p_20260522_071430/results_summary.csv`
- TP=2 agg result CSV: `/hongming/res4/h200_agg_tp2_32b_image8_768p_np64_sweep/test_sglang_multi_rates_1080p_20260522_074355/results_summary.csv`
- Disagg result CSV: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_768p_np64_sweep/test_sglang_multi_rates_1080p_20260522_081849/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_8img_768p_sweep.sh`
- Server start scripts:
  - TP=1 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
  - TP=2 agg: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
  - Disagg: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh`
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/8img_768p_agg_disagg.md` — single-rate (1.0) result of TP=1 agg vs disagg at 768p
  - `01_cuda_sh/agg_h200_32b/tp1_all_rates_results.md` — TP=1 agg at 1080p
  - `01_cuda_sh/agg_h200_32b/tp2_all_rates_results.md` — TP=2 agg at 1080p
  - `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md` — disagg at 1080p
- This document: `01_cuda_sh/disagg_h200_32b/768p_sweep_three_way.md`

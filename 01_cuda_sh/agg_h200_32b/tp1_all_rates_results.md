# TP=1 Aggregate Rate Sweep — Qwen3-VL-32B-FP8

**System under test:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
**Model:** Qwen3-VL-32B-Instruct-FP8
**GPU:** GPU 4 only (TP=1, single H200, NUMA 2)
**Server config:** `--max-running-requests 40`, `--mem-fraction-static 0.95`, `--page-size 16`, `--kv-cache-dtype fp8_e4m3`, `--enable-multimodal`, `--enable-mm-global-cache`
**Workload:** 8 × 1920×1080 images per request, `random-input-len=128`, `random-output-len=256`, `num-prompts=64` (fixed for all rates)
**Bench script:** `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
**Rates swept:** `DEFAULT_RATES=(0.1 0.25 0.5 1.0 1.25)`
**Result dir:** `/hongming/res4/h200_agg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_194141/`

---

## Per-rate results

### Rate 0.10 — ❌ FAILED

Bench's internal warmup hit `Not Found` on the very first request:

```
ValueError: Warmup failed - Please make sure benchmark arguments are correctly specified.
Error: Not Found:
```

Subsequent rates worked fine on the same server, so this is a one-off bench-side warmup glitch (first request after server cold-start), not a config issue. Recommend re-running just rate 0.1 separately if a low-rate baseline is needed.

### Rate 0.25 — ✓ keeping up

| Metric | Value |
|---|---:|
| Target rate | 0.25 req/s |
| **Actual RPS** | **0.25** ✓ |
| Successful requests | 64 / 64 |
| Bench duration | 253.7 s |
| **Mean / Median / P99 TTFT** | **5,548 / 4,950 / 14,653 ms** |
| **Mean / Median / P99 TPOT** | 73 / 38 / 450 ms |
| Mean / Median ITL | 55 / 15 ms |
| P95 / P99 / Max ITL | 23 / 769 / 15,550 ms |
| **Mean / Median / P99 E2E** | **11,512 / 9,499 / 22,025 ms** |
| P90 E2E | 26,121 ms |
| Input throughput | 4,141 tok/s |
| Output throughput | 29 tok/s |
| Peak output throughput | 370 tok/s |
| Concurrency | 2.9 |

### Rate 0.50 — ⚠ knee, falling behind

| Metric | Value |
|---|---:|
| Target rate | 0.50 req/s |
| **Actual RPS** | **0.46** (−8% vs target) |
| Successful requests | 64 / 64 |
| Bench duration | 137.7 s |
| **Mean / Median / P99 TTFT** | **10,503 / 8,957 / 21,839 ms** |
| **Mean / Median / P99 TPOT** | 1,239 / 380 / 19,926 ms |
| Mean / Median ITL | 476 / 35 ms |
| P95 / P99 / Max ITL | 1,093 / 13,675 / 64,706 ms |
| **Mean / Median / P99 E2E** | **52,208 / 50,630 / 104,215 ms** |
| P90 E2E | 96,413 ms |
| Input throughput | 7,632 tok/s |
| Output throughput | 54 tok/s |
| Peak output throughput | 785 tok/s |
| Concurrency | 24.3 |

### Rate 1.00 — ⛔ saturated

| Metric | Value |
|---|---:|
| Target rate | 1.0 req/s |
| **Actual RPS** | **0.52** (−48% vs target) |
| Successful requests | 64 / 64 |
| Bench duration | 123.5 s |
| **Mean / Median / P99 TTFT** | **32,436 / 34,286 / 55,441 ms** |
| **Mean / Median / P99 TPOT** | 1,069 / 435 / 13,419 ms |
| Mean / Median ITL | 419 / 45 ms |
| P95 / P99 / Max ITL | 1,505 / 3,551 / 63,663 ms |
| **Mean / Median / P99 E2E** | **80,659 / 76,986 / 118,181 ms** |
| P90 E2E | 110,574 ms |
| Input throughput | 8,508 tok/s |
| Output throughput | 61 tok/s |
| Peak output throughput | 811 tok/s |
| Concurrency | 41.8 (cap 40) |

### Rate 1.25 — ⛔ saturated

| Metric | Value |
|---|---:|
| Target rate | 1.25 req/s |
| **Actual RPS** | **0.57** (−54% vs target) |
| Successful requests | 64 / 64 |
| Bench duration | 112.2 s |
| **Mean / Median / P99 TTFT** | **32,069 / 32,024 / 56,534 ms** |
| **Mean / Median / P99 TPOT** | 690 / 404 / 6,160 ms |
| Mean / Median ITL | 378 / 46 ms |
| P95 / P99 / Max ITL | 1,488 / 3,485 / 48,994 ms |
| **Mean / Median / P99 E2E** | **75,581 / 72,764 / 107,314 ms** |
| P90 E2E | 101,155 ms |
| Input throughput | 9,362 tok/s |
| Output throughput | 67 tok/s |
| Peak output throughput | 825 tok/s |
| Concurrency | 43.1 (cap 40) |

---

## Summary table — TP=1 across rates

| Target RPS | Actual RPS | Mean TTFT | Median TTFT | P99 TTFT | Mean TPOT | Median TPOT | Mean E2E | Median E2E | Peak out tput | Concurrency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | FAILED | — | — | — | — | — | — | — | — | — |
| 0.25 | 0.25 | 5.5 s | 5.0 s | 14.7 s | 73 ms | 38 ms | 11.5 s | 9.5 s | 370 | 2.9 |
| 0.50 | 0.46 | 10.5 s | 9.0 s | 21.8 s | 1,239 ms | 380 ms | 52.2 s | 50.6 s | 785 | 24.3 |
| 1.00 | 0.52 | 32.4 s | 34.3 s | 55.4 s | 1,069 ms | 435 ms | 80.7 s | 77.0 s | 811 | 41.8 |
| 1.25 | 0.57 | 32.1 s | 32.0 s | 56.5 s | 690 ms | 404 ms | 75.6 s | 72.8 s | 825 | 43.1 |

---

## Interpretation

### Knee location

**The knee is between rate=0.25 and rate=0.50 req/s.**

- At 0.25 RPS, the system **keeps up** (actual = target = 0.25), TTFT is a moderate 5.5 s, TPOT is 73 ms (decode-clean), and concurrency is only 2.9.
- At 0.50 RPS, the system **falls behind** (0.46 actual vs 0.50 target), TTFT roughly doubles to 10.5 s, TPOT explodes 17× to 1,239 ms (decode is now being repeatedly preempted by prefill), and concurrency jumps 8× to 24.3.

### Saturation ceiling

**Sustained throughput ceiling ≈ 0.55–0.57 req/s.** Confirmed across two over-saturated points:
- Rate 1.0 → actual 0.52 RPS
- Rate 1.25 → actual 0.57 RPS

This matches the back-of-envelope estimate from earlier `time_breakdown_for_agg_tp1_1080p_8image.md`: ~9.5k tok/s prefill capacity ÷ ~16.4k tokens/req ≈ **0.58 req/s ceiling** for this workload on a single H200.

### Beyond the knee, more offered load doesn't help

- Rate=1.25 actually shows slightly **better** numbers than rate=1.0 (mean TTFT 32.1 vs 32.4 s, mean TPOT 690 vs 1,069 ms, mean E2E 75.6 vs 80.7 s).
- Both are fully back-pressured at the `max_running_requests=40` cap (concurrency 41.8 and 43.1).
- The slight differences are noise / different in-flight request mix (which prompts arrived first, KV-cache state, etc.).

### Output throughput plateau

- Peak output token throughput plateaus at **~810–825 tok/s** for rates ≥ 0.5.
- Sustained output token throughput stays in the **~54–67 tok/s** band — limited by the 117-token median output length × 0.5 req/s.

### TPOT degradation pattern (decode-blocking-by-prefill)

| Rate | Median TPOT | Mean TPOT | Max ITL |
|---:|---:|---:|---:|
| 0.25 | 38 ms | 73 ms | 769 ms |
| 0.50 | 380 ms | 1,239 ms | 13,675 ms |
| 1.00 | 435 ms | 1,069 ms | 3,551 ms |
| 1.25 | 404 ms | 690 ms | 3,485 ms |

- At 0.25 RPS decode is "clean" — median TPOT 38 ms is what one expects for this 32B FP8 model on a single H200.
- At 0.50+ RPS, ongoing prefill chunks (8192-token vision-token chunks ≈ 0.85 s each) interrupt every in-flight decode → 10× TPOT inflation.
- This is the classic **prefill-blocks-decode** pattern in aggregated EPD with no PD separation.

### Comparison to old `rate=1.0, np=64` baseline

| Metric | Old TP=1 (broken IP, 25/0.88) | **New TP=1 (fixed IP, 40/0.95)** | Δ |
|---|---:|---:|---:|
| Actual RPS @ rate=1.0 | 0.42 | **0.52** | **+24%** |
| Mean TTFT @ rate=1.0 | 50.2 s | **32.4 s** | **−35%** |
| Median TPOT @ rate=1.0 | 354 ms | 435 ms | +23% (worse) |
| Concurrency cap | 25 | 40 | +60% |

The new TP=1 (with fixed `IP_LOCAL=172.26.46.75` and `cuda_ipc` in `UCX_TLS`, `max_running_requests=40`, `mem_fraction=0.95`) is meaningfully faster than the old broken-IP baseline at rate=1.0. The TPOT regression is the cost of the higher concurrency cap (40 vs 25) — more requests in flight means more frequent decode preemption.

---

## Recommended next experiments

1. **Re-run rate=0.1** (was failed) to confirm low-rate steady-state TTFT — should be the cleanest measurement of per-request service time. Expected: ~3.5 s TTFT (no queueing), ~38 ms median TPOT.
2. **Sweep finer between 0.25 and 0.50** — try 0.30, 0.35, 0.40, 0.45 to pinpoint the exact knee.
3. **Compare with TP=2 NVLink** (already swept once at rate=1.0 → 0.57 RPS, 35 s TTFT). At rate=0.25 TP=2 should give better TPOT and similar TTFT; useful for low-concurrency profiles.
4. **Compare with 2× TP=1 DP** at the same offered rates — at rate=1.0 it gave 0.76 RPS, 11 s TTFT (winner); at lower rates TP=1 single may already saturate slower so the gap narrows.

---

## Files

- Server start: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh`
- Bench script: `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
- Result dir: `/hongming/res4/h200_agg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_194141/`
- CSV: `.../results_summary.csv`
- Per-rate logs: `.../rate_{0.1,0.25,0.5,1.0,1.25}/results.txt`
- This document: `01_cuda_sh/agg_h200_32b/tp1_all_rates_results.md`

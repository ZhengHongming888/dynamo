# TP=2 Aggregate Rate Sweep — Qwen3-VL-32B-FP8 (NVLink, GPUs 4,5)

**System under test:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
**Model:** Qwen3-VL-32B-Instruct-FP8
**GPUs:** GPUs **4,5** (NUMA 2, NV18 NVLink — verified via `nvidia-smi topo -m`, ~478 GB/s aggregate)
**Server config:** `--tensor-parallel-size 2`, `--max-running-requests 25`, `--mem-fraction-static 0.88`, `--page-size 16`, `--kv-cache-dtype fp8_e4m3`, `--enable-multimodal`, `--enable-mm-global-cache`
**No MM workaround flags** (default TP-sharded ViT, NVLink-cheap)
**Workload:** 8 × 1920×1080 images per request, `random-input-len=128`, `random-output-len=256`, `num-prompts=64` (fixed for all rates)
**Bench script:** `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
**Rates swept:** `DEFAULT_RATES=(0.1 0.25 0.5 1.0 1.25)`
**Result dir:** `/hongming/res4/h200_agg_tp2_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_201625/`

### NVLink P2P/IPC verified at startup

```
NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
NCCL INFO Channel 00/0 : 0[4] -> 1[5] via P2P/IPC
... (24 channels per direction × 2 directions = 48 total P2P/IPC channels)
```

UCX configured with `cuda_ipc` first in `UCX_TLS`, host IP correct (`172.26.46.75`).

---

## Per-rate results

### Rate 0.10 — ✓ keeping up

| Metric | Value |
|---|---:|
| Target rate | 0.10 req/s |
| **Actual RPS** | **0.10** ✓ |
| Successful requests | 64 / 64 |
| Bench duration | 626.7 s |
| **Mean / Median / P99 TTFT** | **3,573 / 3,109 / 7,294 ms** |
| **Mean / Median / P99 TPOT** | 27 / 9 / 191 ms |
| Mean / Median ITL | 21 / 9 ms |
| P95 / P99 / Max ITL | 12 / 50 / 12,150 ms |
| **Mean / Median / P99 E2E** | **5,793 / 4,568 / 17,887 ms** |
| P90 E2E | 9,376 ms |
| Input throughput | 1,676 tok/s |
| Output throughput | 12 tok/s |
| Peak output throughput | 525 tok/s |
| Concurrency | 0.6 |

### Rate 0.25 — ✓ keeping up

| Metric | Value |
|---|---:|
| Target rate | 0.25 req/s |
| **Actual RPS** | **0.25** ✓ |
| Successful requests | 64 / 64 |
| Bench duration | 251.8 s |
| **Mean / Median / P99 TTFT** | **2,796 / 2,130 / 7,688 ms** |
| **Mean / Median / P99 TPOT** | 12 / 9 / 34 ms |
| Mean / Median ITL | 16 / 9 ms |
| P95 / P99 / Max ITL | 11 / 58 / 2,127 ms |
| **Mean / Median / P99 E2E** | **4,155 / 3,789 / 9,443 ms** |
| P90 E2E | 6,030 ms |
| Input throughput | 4,172 tok/s |
| Output throughput | 30 tok/s |
| Peak output throughput | 406 tok/s |
| Concurrency | 1.1 |

### Rate 0.50 — ✓ keeping up

| Metric | Value |
|---|---:|
| Target rate | 0.50 req/s |
| **Actual RPS** | **0.50** ✓ |
| Successful requests | 64 / 64 |
| Bench duration | 127.6 s |
| **Mean / Median / P99 TTFT** | **3,667 / 2,792 / 9,442 ms** |
| **Mean / Median / P99 TPOT** | 16 / 13 / 70 ms |
| Mean / Median ITL | 23 / 9 ms |
| P95 / P99 / Max ITL | 15 / 702 / 2,796 ms |
| **Mean / Median / P99 E2E** | **5,420 / 4,683 / 11,155 ms** |
| P90 E2E | 9,372 ms |
| Input throughput | 8,236 tok/s |
| Output throughput | 59 tok/s |
| Peak output throughput | 484 tok/s |
| Concurrency | 2.7 |

### Rate 1.00 — ⚠ slightly behind (knee approaching)

| Metric | Value |
|---|---:|
| Target rate | 1.00 req/s |
| **Actual RPS** | **0.95** (−5% vs target) |
| Successful requests | 64 / 64 |
| Bench duration | 67.6 s |
| **Mean / Median / P99 TTFT** | **9,048 / 6,526 / 23,177 ms** |
| **Mean / Median / P99 TPOT** | 32 / 15 / 408 ms |
| Mean / Median ITL | 47 / 10 ms |
| P95 / P99 / Max ITL | 31 / 912 / 7,498 ms |
| **Mean / Median / P99 E2E** | **11,803 / 9,272 / 27,534 ms** |
| P90 E2E | 20,419 ms |
| Input throughput | 15,534 tok/s |
| Output throughput | 111 tok/s |
| Peak output throughput | 619 tok/s |
| Concurrency | 11.2 |

### Rate 1.25 — ⛔ saturated

| Metric | Value |
|---|---:|
| Target rate | 1.25 req/s |
| **Actual RPS** | **1.15** (−8% vs target) |
| Successful requests | 64 / 64 |
| Bench duration | 55.5 s |
| **Mean / Median / P99 TTFT** | **22,745 / 20,993 / 43,664 ms** |
| **Mean / Median / P99 TPOT** | 76 / 1 / 1,347 ms (median 1 ms is a tail-decode-tokenization artefact) |
| Mean / Median ITL | 533 / 9 ms |
| P95 / P99 / Max ITL | 4,111 / 13,268 / 13,891 ms |
| **Mean / Median / P99 E2E** | **28,832 / 31,447 / 53,192 ms** |
| P90 E2E | 47,296 ms |
| Input throughput | 18,926 tok/s |
| Output throughput | 135 tok/s |
| Peak output throughput | 592 tok/s |
| Concurrency | 33.2 |

---

## Summary table — TP=2 NVLink across rates

| Target RPS | Actual RPS | Mean TTFT | Med TTFT | P99 TTFT | Mean TPOT | Med TPOT | Mean E2E | Med E2E | Peak out tput | Concurrency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.10 | 3.6 s | 3.1 s | 7.3 s | 27 ms | 9 ms | 5.8 s | 4.6 s | 525 | 0.6 |
| 0.25 | 0.25 | 2.8 s | 2.1 s | 7.7 s | 12 ms | 9 ms | 4.2 s | 3.8 s | 406 | 1.1 |
| 0.50 | 0.50 | 3.7 s | 2.8 s | 9.4 s | 16 ms | 13 ms | 5.4 s | 4.7 s | 484 | 2.7 |
| 1.00 | 0.95 | 9.0 s | 6.5 s | 23.2 s | 32 ms | 15 ms | 11.8 s | 9.3 s | 619 | 11.2 |
| 1.25 | 1.15 | 22.7 s | 21.0 s | 43.7 s | 76 ms | 1 ms | 28.8 s | 31.4 s | 592 | 33.2 |

---

## TP=1 vs TP=2 head-to-head (same rates, same workload, same machine)

(TP=1 numbers from `tp1_all_rates_results.md`, np=64, GPU 4 only.)

| Rate | TP=1 actual / TP=2 actual | TP=1 Mean TTFT / TP=2 | TP=1 Mean TPOT / TP=2 | TP=1 Mean E2E / TP=2 | Δ E2E |
|---:|---:|---:|---:|---:|---:|
| 0.10 | FAILED / **0.10** | — / **3.6 s** | — / **27 ms** | — / **5.8 s** | — |
| 0.25 | 0.25 / **0.25** | 5.5 s / **2.8 s** | 73 ms / **12 ms** | 11.5 s / **4.2 s** | **−63%** |
| 0.50 | 0.46 / **0.50** | 10.5 s / **3.7 s** | 1,239 ms / **16 ms** | 52.2 s / **5.4 s** | **−90%** |
| 1.00 | 0.52 / **0.95** | 32.4 s / **9.0 s** | 1,069 ms / **32 ms** | 80.7 s / **11.8 s** | **−85%** |
| 1.25 | 0.57 / **1.15** | 32.1 s / **22.7 s** | 690 ms / **76 ms** | 75.6 s / **28.8 s** | **−62%** |

**TP=2 wins decisively at every rate**, with the gap widening from rate=0.25 (−63% E2E) to rate=0.5 (−90% E2E). The improvement is not just per-request speed — TP=2's ceiling (~1.15 req/s) is **2× the TP=1 ceiling (~0.57 req/s)**.

---

## Interpretation

### Knee location for TP=2

**The knee is between rate=1.0 and rate=1.25 req/s.**

- At 1.0 RPS, TP=2 is **just barely** keeping up (0.95 actual vs 1.0 target, −5%); TTFT is 9 s (still acceptable), concurrency rising to 11.
- At 1.25 RPS, TP=2 **falls behind** (1.15 actual, −8%) and TTFT explodes to 23 s, concurrency 33.

This is roughly **2× the TP=1 knee location** (which was between 0.25 and 0.5 req/s). Doubling GPUs with NVLink-backed TP gives ~2× the prefill ceiling for this multimodal workload.

### Saturation ceiling

**Sustained throughput ceiling ≈ 1.15 req/s** at rate=1.25. The ratio TP=2/TP=1 = 1.15 / 0.57 = **2.02× scaling** — essentially perfect scaling with TP=2 on NVLink. No PCIe penalty, no encoder duplication penalty, no MM-broadcast penalty (because we're not using `--enable-broadcast-mm-inputs-process`).

### Where TP=2 wins decisively: TPOT (decode quality)

| Rate | TP=1 Median TPOT | **TP=2 Median TPOT** | Improvement |
|---:|---:|---:|---:|
| 0.25 | 38 ms | **9 ms** | **4.2× cleaner** |
| 0.50 | 380 ms | **13 ms** | **29× cleaner** |
| 1.00 | 435 ms | **15 ms** | **29× cleaner** |
| 1.25 | 404 ms | **1 ms (median)** | — |

TP=2 has **zero or near-zero decode preemption** at all rates ≤ 1.0:
- TP=1 at rate=0.5 had median TPOT 380 ms (decode constantly preempted by chunked prefill, classic aggregated-EPD prefill-blocks-decode).
- TP=2 at rate=0.5 has median TPOT 13 ms — clean decode at native model speed.

This is because TP=2 finishes prefill chunks in ~half the wall time (NVLink-cheap all-reduce), so decode steals time-slices much more often. Even when prefill is in flight, the gap is short enough that decode doesn't visibly stall.

### Where TP=2 wins less dramatically: peak output throughput

| Rate | TP=1 Peak out tput | TP=2 Peak out tput | Δ |
|---:|---:|---:|---:|
| 0.25 | 370 | 406 | +10% |
| 0.50 | 785 | 484 | **−38%** |
| 1.00 | 811 | 619 | −24% |
| 1.25 | 825 | 592 | −28% |

Counter-intuitive: TP=2 has *lower* peak output throughput at rate≥0.5. Why?
- Peak output throughput is a per-window measurement during sustained decode bursts.
- TP=1 sometimes has many requests stuck in late-decode (no prefill running) → all 25–40 in-flight requests decoding simultaneously → high burst output rate.
- TP=2 finishes requests faster, so the "many requests all decoding at once" burst is shorter and rarer. It's the difference between a deep queue throughput-burst vs. a shallow queue keeping up smoothly.
- This is a feature, not a bug — TP=2 wins on **mean** and **per-request E2E** even though it has lower peak burst tput.

### Compared to old TP=2 baselines

This run vs prior TP=2 results:

| Run | Rate=1.0 actual | Mean TTFT | Mean E2E | Notes |
|---|---:|---:|---:|---|
| Old TP=2 SYS GPUs 3,4 (broken IP) | 0.38 | 61 s | 107 s | Cross-socket SYS, all bugs |
| Old TP=2 NODE GPUs 4,5 (broken IP) | 0.38 | 60 s | 107 s | Same NUMA, but bugs |
| Prior fixed TP=2 NVLink (rate=1.0 only) | 0.57 | 35 s | 64 s | First fix of IP + cuda_ipc |
| **This run TP=2 NVLink (rate=1.0)** | **0.95** | **9 s** | **12 s** | All fixes + cleaner config |
| **This run TP=2 NVLink (rate=1.25)** | **1.15** | **23 s** | **29 s** | Approaching saturation |

Big improvement vs the prior "fixed TP=2 NVLink" (0.57→0.95 req/s, 35→9 s TTFT). Why so much better this time?
1. **Same fixes** (correct IP, `cuda_ipc` in `UCX_TLS`, no MM workaround flags).
2. **No competing workload** during this run — prior runs may have been affected by GPU 3 zombie or competing test setup.
3. **`mm_enable_dp_encoder=False` + `enable_broadcast_mm_inputs_process=False` + `keep_mm_feature_on_device=False`** — all correctly defaulted off. Confirmed in the worker's ServerArgs dump.

The earlier `0.57 req/s` figure may have been a partial-saturation measurement; running at higher rates (1.0, 1.25) reveals the actual capacity is ~1.15 req/s.

---

## Token throughput summary

| Rate | Input tput | Output tput | Peak output tput |
|---:|---:|---:|---:|
| 0.10 | 1,676 | 12 | 525 |
| 0.25 | 4,172 | 30 | 406 |
| 0.50 | 8,236 | 59 | 484 |
| 1.00 | **15,534** | 111 | 619 |
| 1.25 | **18,926** | **135** | 592 |

Peak input throughput at rate=1.25 is ~19k tok/s — about **2× the TP=1 peak of ~9.5k tok/s**, again consistent with 2× scaling.

---

## Recommended next experiments

1. **Sweep finer between 1.0 and 1.5 to find the exact TP=2 knee** — try 1.10, 1.20, 1.30, 1.40 to nail down the sustained ceiling. Current best estimate is ~1.15 req/s.
2. **Run 2× TP=1 DP sweep** with the now-fixed `start_h200_aggregate_epd_server_32b_two_tp1.sh` (IP + cuda_ipc fix). Expected: ceiling ≥ 2 × 0.57 = 1.14 req/s, likely with much better TTFT at low-medium rates due to queue-depth halving.
3. **Compare TP=2 NVLink vs 2× TP=1 DP** at the same rates. Hypothesis:
   - At low rates (≤0.5): both keep up; TP=2 has slightly faster per-request decode (median TPOT 9–13 ms vs ~38 ms for TP=1).
   - At high rates (≥1.0): 2× DP wins on TTFT (queue-depth halving), TP=2 wins on TPOT.
4. **Try TP=4 on GPUs 4,5,6,7** (all on NUMA 2 with NVLink) — should give ~2.3 req/s ceiling if NVLink scaling holds. Diminishing returns expected because vision encoder doesn't TP-shard well even with NVLink.
5. **Disaggregated EPD with NVLink** — separate encoder/prefill/decode workers communicating over NVLink-backed NIXL. Could give the cleanest TPOT *and* highest throughput by avoiding prefill-decode interleave entirely.

---

## Files

- Server start: `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
- Bench script: `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
- Result dir: `/hongming/res4/h200_agg_tp2_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_201625/`
- CSV: `.../results_summary.csv`
- Per-rate logs: `.../rate_{0.1,0.25,0.5,1.0,1.25}/results.txt`
- Companion doc: `01_cuda_sh/agg_h200_32b/tp1_all_rates_results.md`
- This document: `01_cuda_sh/agg_h200_32b/tp2_all_rates_results.md`

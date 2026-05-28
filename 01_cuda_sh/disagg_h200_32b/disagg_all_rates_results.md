# Disaggregated EPD Rate Sweep — Qwen3-VL-32B-FP8 (encoder GPU 4 + PD GPU 5)

**System under test:** `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh` (after fixes — see "Fixes applied" below)
**Model:** Qwen3-VL-32B-Instruct-FP8
**Architecture:** Disaggregated multimodal — encoder runs on GPU 4, PD (prefill+decode) on GPU 5, NIXL transfers MM features encoder→PD via NVLink
**Topology:** GPUs 4↔5 are NUMA 2, NV18 NVLink (~478 GB/s aggregate)
**Workload:** 8 × 1920×1080 images per request, `random-input-len=128`, `random-output-len=256`, `num-prompts=64` (fixed for all rates)
**Bench script:** `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
**Rates swept:** `DEFAULT_RATES=(0.1 0.25 0.5 1.0 1.25)`
**Result dir:** `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_211224/`

---

## TL;DR — disagg lost decisively to both TP=1 agg and TP=2 agg

| Config | Sustained ceiling | Mean TTFT @ rate=0.25 | Mean TPOT @ rate=0.25 | Failed requests at rate ≥ 0.5 |
|---|---:|---:|---:|---:|
| **TP=1 agg** (1 GPU) | ~0.57 req/s | 5.5 s | 73 ms | 0 |
| **TP=2 agg NVLink** (2 GPUs) | ~1.15 req/s | **2.8 s** | **12 ms** | 0 |
| **Disagg** (encoder + PD, 2 GPUs) | **~0.27 req/s** | **53.4 s** | **2,026 ms** | **48 / 192 (25%)** |

**Disagg is 4× slower than TP=2 agg on the same hardware**, with high failure rates due to a NIXL buffer pool exhaustion bug in the encoder→PD coordination. Recommend NOT using disagg for this workload until the NIXL flow control is fixed.

---

## Per-rate results

### Rate 0.10 — ✓ keeping up, but already much slower than agg

| Metric | Value |
|---|---:|
| Target rate | 0.10 req/s |
| **Actual RPS** | **0.10** ✓ |
| Successful requests | 64 / 64 |
| Bench duration | 632.6 s |
| **Mean / Median / P99 TTFT** | **15,397 / 13,965 / 38,029 ms** |
| **Mean / Median / P99 TPOT** | 104 / 42 / 674 ms |
| Mean / Median ITL | 76 / 14 ms |
| P95 / P99 / Max ITL | 26 / 1,341 / 37,545 ms |
| **Mean / Median / P99 E2E** | **23,091 / 18,762 / 57,388 ms** |
| Input throughput | 1,661 tok/s |
| Output throughput | 12 tok/s |
| Peak output throughput | 396 tok/s |
| Concurrency | 2.3 |

Comparison to agg @ same rate:
- TP=1 (rate 0.10 failed in earlier run, no comparable number)
- **TP=2 agg @ rate=0.10**: Mean TTFT 3.6 s, Mean TPOT 27 ms — disagg is 4.3× worse on TTFT, 3.9× worse on TPOT.

### Rate 0.25 — ⚠ falling behind significantly

| Metric | Value |
|---|---:|
| Target rate | 0.25 req/s |
| **Actual RPS** | **0.21** (−16% vs target) |
| Successful requests | 64 / 64 |
| Bench duration | 309.1 s |
| **Mean / Median / P99 TTFT** | **53,351 / 56,321 / 106,034 ms** |
| **Mean / Median / P99 TPOT** | 2,026 / 872 / 26,438 ms |
| Mean / Median ITL | 965 / 35 ms |
| P95 / P99 / Max ITL | 3,272 / 8,445 / 153,267 ms |
| **Mean / Median / P99 E2E** | **150,318 / 146,159 / 278,972 ms** |
| Input throughput | 3,398 tok/s |
| Output throughput | 24 tok/s |
| Peak output throughput | 811 tok/s |
| Concurrency | 31.1 |

Disagg already saturated by rate=0.25. Compared to TP=2 agg (which was at 0.25 actual = target, mean TTFT 2.8 s, mean TPOT 12 ms), disagg is **20× worse on TTFT, 169× worse on TPOT**.

### Rate 0.50 — ⛔ saturated + failures begin

| Metric | Value |
|---|---:|
| Target rate | 0.50 req/s |
| **Actual RPS** | **0.24** (−52% vs target) |
| **Successful requests** | **52 / 64 (12 failed)** ⚠ |
| Bench duration | 214.9 s |
| **Mean / Median / P99 TTFT** | **61,232 / 58,340 / 97,848 ms** |
| **Mean / Median / P99 TPOT** | 2,022 / 640 / 28,948 ms |
| Mean / Median / P99 / Max ITL | 735 / 37 / 4,954 / 131,391 ms |
| **Mean / Median / P99 E2E** | **140,328 / 143,608 / 209,363 ms** |
| Input throughput | 3,972 tok/s |
| Output throughput | 28 tok/s |
| Peak output throughput | 751 tok/s |
| Concurrency | 34.0 |

### Rate 1.00 — ⛔ heavy failures, throughput regressing

| Metric | Value |
|---|---:|
| Target rate | 1.00 req/s |
| **Actual RPS** | **0.27** (−73% vs target) |
| **Successful requests** | **49 / 64 (15 failed)** ⚠ |
| Bench duration | 184.5 s |
| **Mean / Median / P99 TTFT** | **78,892 / 75,944 / 121,534 ms** |
| **Mean / Median / P99 TPOT** | 752 / 256 / 9,111 ms |
| **Mean / Median / P99 E2E** | **111,316 / 99,582 / 164,330 ms** |
| Concurrency | 29.6 |

### Rate 1.25 — ⛔ worst failure rate, lowest throughput

| Metric | Value |
|---|---:|
| Target rate | 1.25 req/s |
| **Actual RPS** | **0.24** (−81% vs target) |
| **Successful requests** | **43 / 64 (21 failed)** ⚠ |
| Bench duration | 180.5 s |
| **Mean / Median / P99 TTFT** | **90,716 / 95,582 / 129,707 ms** |
| **Mean / Median / P99 TPOT** | 488 / 263 / 4,921 ms |
| **Mean / Median / P99 E2E** | **117,351 / 127,885 / 165,289 ms** |
| Concurrency | 28.0 |

Note: rate=1.25 has **worse throughput (0.24 RPS)** than rate=1.0 (0.27 RPS) — a clear sign the system has fallen into a backpressure death-spiral where buffer-timeout failures actually slow the recovery further.

---

## Summary table — Disagg across rates

| Target RPS | Actual RPS | Success | Mean TTFT | Mean TPOT | Mean E2E | Concurrency |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.10 | 64/64 | 15.4 s | 104 ms | 23.1 s | 2.3 |
| 0.25 | 0.21 | 64/64 | 53.4 s | 2,026 ms | 150.3 s | 31.1 |
| 0.50 | 0.24 | **52/64** | 61.2 s | 2,022 ms | 140.3 s | 34.0 |
| 1.00 | 0.27 | **49/64** | 78.9 s | 752 ms | 111.3 s | 29.6 |
| 1.25 | 0.24 | **43/64** | 90.7 s | 488 ms | 117.4 s | 28.0 |

---

## Three-way comparison: TP=1 agg vs TP=2 agg vs Disagg (same hardware, same workload)

All rate=0.25 (representative low-saturation point):

| Metric | TP=1 agg (GPU 4) | TP=2 agg NVLink (4,5) | **Disagg (encoder GPU 4 + PD GPU 5)** |
|---|---:|---:|---:|
| Actual RPS | 0.25 | 0.25 | **0.21** ⚠ |
| Mean TTFT | 5.5 s | **2.8 s** | **53.4 s** ❌ |
| Median TPOT | 38 ms | **9 ms** | **872 ms** ❌ |
| Mean E2E | 11.5 s | **4.2 s** | **150.3 s** ❌ |
| Failed requests | 0 | 0 | 0 |

All rate=1.0 (high-saturation):

| Metric | TP=1 agg | TP=2 agg NVLink | **Disagg** |
|---|---:|---:|---:|
| Actual RPS | 0.52 | **0.95** | **0.27** ❌ |
| Mean TTFT | 32 s | **9 s** | 79 s |
| Median TPOT | 435 ms | **15 ms** | 256 ms |
| Mean E2E | 81 s | **12 s** | 111 s |
| Failed requests | 0 | 0 | **15/64** ❌ |

---

## Root cause analysis

### Symptom

**48 requests timed out with "Timeout while waiting for available buffer."** across rates 0.5/1.0/1.25 (12+15+21 = 48).

### Diagnosis

**NIXL buffer pool exhaustion on the PD side.** The error appears in the PD worker log:

```
ERROR worker_handler.generate: Error in multimodal generation: Timeout while waiting for available buffer.
```

And the corresponding error in encoder/frontend logs:

```
WARN ... Failed deserializing JSON to response
  err=invalid type: unit variant, expected newtype variant at line 1 column 55
  json_str={"data":{"data":{"token_ids":[],"finish_reason":"error","error":"Timeout while waiting for available buffer."}},"complete_final":false}
```

### Mechanism

1. **Encoder produces MM features fast** — Qwen3-VL ViT processes 8×1080p images in ~1 second per request.
2. **NIXL_WRITE allocates a buffer slot** in the pool (`NIXL_BUFFER_COUNT=256`, `NIXL_MAX_BUFFER_SIZE=805306368` = 768 MB each).
3. **PD must consume each MM feature** by running prefill on it, which takes ~2-4 seconds per request (16k vision tokens prefill + small text prefill).
4. **Encoder produces faster than PD consumes** at any rate ≥ 0.5 req/s for this workload size.
5. **Buffer pool fills up.** New requests from encoder block waiting for a free buffer. After timeout, the PD-side request fails.
6. **Cascading failures**: PD also throws **307 "Mismatch: More 'IMAGE' tokens found than corresponding data provided"** warnings, indicating PD-side prefill scheduler races ahead of completed NIXL_WRITE — the disagg coordinator has no proper backpressure between encoder output rate and PD prefill rate.

### Why this didn't happen with TP=2 agg

In TP=2 agg, the encoder, prefill, and decode all run on the same GPUs in lockstep — there's no inter-process buffer pool. The encoder simply doesn't run if there's no GPU memory for its output. SGLang's scheduler natively handles the back-pressure.

In disagg, the inter-process buffer pool **decouples** encoder and PD, but **only one direction** of decoupling is implemented (PD pulling from encoder via NIXL_READ). There's no way to throttle the encoder when PD is busy.

---

## Fixes applied to script before this run

| # | Bug | Before | After |
|---|---|---|---|
| 1 | Wrong host IP | `IP_LOCAL=172.26.46.162` | `IP_LOCAL=172.26.46.75` |
| 2 | Encoder UCX missing `cuda_ipc` | `UCX_TLS=ib,rc,...` | `UCX_TLS=cuda_ipc,ib,rc,...` |
| 3 | PD worker missing UCX/NIXL env entirely | (none) | Full NIXL+UCX env including `cuda_ipc` first |
| 4 | Etcd wait too short | `sleep 2` | `sleep 8` + 10× `curl /version` poll |
| 5 | Missing `RESULT_BASE` | (none) | `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates` |
| 6 | Missing bench-run hint | (none) | Final echo with full command line |
| 7 | Missing NCCL diagnostics | (none) | Added `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,P2P` to PD |

UCX and IP fixes were **necessary but not sufficient** — the deeper bottleneck is the NIXL buffer pool exhaustion, which is unaffected by transport choice.

---

## Recommended next experiments

1. **Throttle the encoder.** SGLang's encoder doesn't currently use `--max-running-requests`. Try setting it to a small value (e.g., 4-8) to back-pressure the encoder when PD is busy. This is the most likely fix.
2. **Reduce `NIXL_BUFFER_COUNT`** from 256 → 32 or 16. Smaller pool means earlier back-pressure, fewer in-flight requests holding buffers, and faster failure detection.
3. **Confirm `DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read` is actually being honored.** The encoder's CONFIG_DUMP shows `embedding_transfer_mode=NIXL_WRITE` despite the env var saying `nixl-read` — possibly a bug in the dynamo arg parser. NIXL_READ would have PD pulling from encoder, which has natural back-pressure (PD only pulls when ready); NIXL_WRITE has encoder pushing, which can overwhelm PD.
4. **Use `keep-mm-feature-on-device`** on the encoder so MM features stay on the encoder's GPU and PD reads them directly via cuda_ipc P2P (NVLink). This avoids the NIXL host-buffer round-trip entirely.
5. **Run with smaller image counts** (e.g. 4 instead of 8 images) to verify the issue is buffer-pool-related and not architectural.
6. **Compare to a "true" disagg with PD on a different host** (where NIXL would be IB-based and tuning would be different).

---

## Files

- Server start: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh`
- Bench script: `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
- Result dir: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_211224/`
- CSV: `.../results_summary.csv`
- Per-rate logs: `.../rate_{0.1,0.25,0.5,1.0,1.25}/results.txt`
- PD worker log: `/hongming/dynamo/logs/pd_worker.log`
- Encoder worker log: `/hongming/dynamo/logs/encoder_worker.log`
- Frontend log: `/hongming/dynamo/logs/frontend.log`
- Companion docs:
  - `01_cuda_sh/agg_h200_32b/tp1_all_rates_results.md`
  - `01_cuda_sh/agg_h200_32b/tp2_all_rates_results.md`
- This document: `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md`

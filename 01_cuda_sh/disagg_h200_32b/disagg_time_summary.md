# Disaggregated EPD time breakdown analysis

**System:** Encoder on **GPU 4** (NIXL_WRITE), PD worker on **GPU 5** (TP=1), single frontend KV router.
**Setup script:** `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh` (with `CUDA_DEVICE_PD=5`, `CUDA_DEVICE_ENCODER=4`).
**Workload:** rate=1 req/s, 64 prompts, 8×1920×1080 images each.
**Bench window:** 2026-05-21 01:22:11 – 01:28:14.
**Result file:** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_012229/rate_1.0/`

---

## 1. Bench-side outcome

| Metric | Value |
|---|---:|
| **Successful requests** | **39 / 64** ⚠️ (25 client-cancelled at ~01:26:32 due to bench-client timeout) |
| Bench duration | 208.1 s |
| Successful-only request throughput | 0.19 req/s (would be ~0.31 if all 64 had completed) |
| Input tput (skewed by missing reqs) | 3,076 tok/s |
| Mean TTFT | **98.0 s** |
| Median TTFT | 101.0 s |
| P99 TTFT | 141.1 s |
| Mean E2E | 172.1 s |
| Mean TPOT | 3,374 ms |
| Median TPOT | 769 ms |
| P99 TPOT | **50,782 ms** (50 s per output token in the tail!) |
| Mean ITL | 727 ms |
| Median ITL | 38 ms (best-case decode is fine) |
| Concurrency | 32.3 (peaked then queue overran) |
| Peak output tput | 708 tok/s |

---

## 2. PD worker (GPU 5) wall-clock breakdown

Bench window (01:22:11 – 01:28:14, ~330 s of activity):

| Phase | Time | % of window |
|---|---:|---:|
| **Prefill (LLM forward)** | 146.2 s | 44.3 % |
| **Decode** | 14.6 s | 4.4 % |
| **Unaccounted (waiting on encoder/transfer)** | **169.3 s** | **51.3 %** |

This is the smoking gun: **over half of PD worker time was spent waiting for embeddings to arrive from the encoder.** Compare to aggregated configurations where unaccounted time is 0–29 % depending on whether the encoder is co-located.

---

## 3. Per-chunk timing on PD worker

Computed from successive `report_prefill_stats` log timestamps:

| Phase | n | median | mean | min | max |
|---|---:|---:|---:|---:|---:|
| Consecutive 8192-chunk forward (model.forward only) | 39 | **1.095 s** | 1.237 s | 0.692 s | 2.750 s |
| Post-flush → next 8192 (encoder transfer + scheduling) | 27 | **1.538 s** | 1.686 s | 1.515 s | 2.716 s |

Compared to baselines:
- TP=1 single agg: 0.737 s forward / 1.185 s gap
- TP=2 NODE: 0.684 s forward / 1.474 s gap
- 2× TP=1 DP: similar to single agg per-worker
- **Disagg PD: 1.095 s forward (+48 %!) / 1.538 s gap (+30 %)**

The PD-only forward time is ~50 % slower than aggregated TP=1. That's surprising for a dedicated GPU and suggests the embedding gather + RoPE/text-token interleaving with externally-supplied features is more expensive than expected.

---

## 4. Embedding transfer timing (encoder → PD via NIXL)

47 transfer events observed during the bench window, with a strongly bimodal distribution:

| Bucket | Count | Notes |
|---|---:|---|
| **fast (<0.1 s)** | **35** | NIXL local transfer working — happy path |
| medium (0.1–1 s) | 0 | none in this range |
| slow (1–5 s) | 4 | first hint of stalls |
| very slow (5–10 s) | 2 | clearly degraded |
| **abusive (>10 s)** | **6** | up to **15.730 s** wait for one tensor |
| Median | 0.020 s | |
| Mean | 2.117 s | dragged up by tail |
| Max | 15.730 s | |

**Most transfers are fast (~20 ms), but a tail of 6 transfers each took 10–16 seconds.** Those tail transfers blocked the PD worker from making forward progress, which is what caused the activity gap and the timeouts.

---

## 5. The 75-second activity gap (the smoking gun)

Activity in 15-second windows from the start of the bench window:

```
window         PD prefill   PD decode   encoder send
   0– 15 s          4            2           3
  15– 30 s          5            3           3
  30– 45 s          7            6           0
  45– 60 s          0            0           0   ← silence begins
  60– 75 s          0            0           0
  75– 90 s          0            0           0
  90–105 s          0            0           0
 105–120 s          0            0           1   ← 75 s of near-total silence!
 120–135 s          3            0           1   ← system recovering
 135–150 s          2            5           7
 150–165 s          4            0           5
 165–180 s         12            0           0
 180–195 s          9            0           8
 195–210 s          4            0           0
 210–225 s         11            0           6
 225–240 s          2            0           2
 240–255 s         11            0           2
 255–270 s         10            0           2
 270–285 s          5            0           5
 285–300 s         10            0           1
 300–315 s         11            0           0
 315–330 s         10            1           0
 330–345 s          2            5           0
```

Reading the timeline:
1. **First 45 s:** All three components active. ~9 requests pipeline through cleanly.
2. **45–120 s (75 s gap):** Near-total silence on all components. Queue grows at 1 req/s arrival rate during this period.
3. **120 s onwards:** System resumes prefill processing, but **decode has collapsed to near-zero** — only 1 decode-step report from t=195 s through end. ITL/TPOT stretch out massively.
4. **t=255 s (wall-clock 01:26:32):** Bench client cancels 25 in-flight requests due to per-request timeout. Frontend log shows `Stream closed unexpectedly; issuing cancellation` x25 around this point.

---

## 6. Why the 75-second gap?

Most likely cause: **NIXL buffer back-pressure or deadlock when the queue fills up.**

Evidence:
- The 6 "abusive" transfer waits (10–16 s) happen exactly during the resumption-and-aftermath phase.
- Encoder configured with `NIXL_BUFFER_COUNT=256`, `NIXL_MAX_BUFFER_SIZE=805306368` (805 MB), `DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read` on PD side and `EmbeddingTransferMode.NIXL_WRITE` on encoder side.
- The pattern matches a known issue documented in `01_cuda_sh/disagg_h200_32b/DEBUG_PROCESS.md`:
  > "NIXL buffer deadlock in disaggregated multimodal E/PD mode"

Mechanism (hypothesized):
- The encoder produces MM feature tensors faster than PD can consume them.
- The NIXL ring buffer fills up (256 slots × ~16 KB MB per 8-image set ≈ a few GB of pending features).
- Producers block waiting for free slots, encoder stalls.
- PD has nothing to consume, also stalls.
- Eventually something drains (timeout? buffer release?) and the system restarts, but now ~75 s worth of arrivals have piled up.

---

## 7. Decode collapse (mean TPOT 3.4 s, p99 50.8 s!)

After the gap, decode batches stay tiny:
- Median `#running-req` during decode: **1**
- Decode gen throughput: median 68.8 tok/s (looks fine per-token), but median TPOT is 769 ms (terrible).

Why decode is so slow:
- PD is dominated by giant 8192-token prefill chunks (each ~1.1 s of forward).
- Each prefill chunk preempts any in-flight decode.
- With 32+ requests queued and only one PD worker, decode rarely gets back-to-back forward passes.
- Inter-token latency grows to several seconds, which is why p99 TPOT is 50.8 s.

This is the classic **prefill-blocking-decode** pathology — and it's actually *worse* in disagg than agg because the encoder transfer gaps create additional preemption points.

---

## 8. Summary table — six configurations

| Config | Successful | req/s | Mean TTFT | Mean E2E | Notes |
|---|---:|---:|---:|---:|---|
| TP=1 single (1 GPU) | 64 | 0.42 | 50.2 s | 93.2 s | baseline |
| TP=2 SYS (3,4) | 64 | 0.38 | 61.0 s | 107.4 s | cross-socket |
| TP=2 NODE (4,5) | 64 | 0.38 | 60.4 s | 107.3 s | same socket |
| TP=2 NODE + MM flags | 64 | 0.29 | 119.9 s | 160.9 s | broadcast hurt |
| **2× TP=1 DP (4,5)** | **64** | **0.76** | **11.2 s** | **45.2 s** | **winner** |
| **Disagg (E=4, PD=5)** | **39** ⚠️ | 0.19 | 98.0 s | 172.1 s | NIXL stall + 25 cancellations |

---

## 9. Root cause in one sentence

**The dynamo-sglang disagg encoder→PD pipeline has a NIXL buffer back-pressure deadlock that locks up the system for ~75 seconds at moderate load (1 req/s), causing the queue to overflow and ~40 % of requests to time out at the client.**

---

## 10. The fundamental architectural cost

Even ignoring the deadlock, disagg can't beat 2× TP=1 DP on this workload because:

1. **Per-chunk model.forward on PD is 50 % slower than aggregated TP=1** (1.10 s vs 0.74 s). The MM-feature-aware prefill path adds significant overhead.
2. **Encoder transfer adds 20 ms – 16 s per request** (median 20 ms NIXL on the happy path, 10+ s in the tail).
3. **One PD GPU + one encoder GPU = 2 GPUs total**, same as 2× TP=1 DP — but DP gives 2× independent parallel processing while disagg only gives ~1× pipelined processing for this workload (since both encode and prefill are bottlenecks at this scale).

---

## When disagg makes sense

Disagg helps when:
- **Encoder is much smaller than the LLM** (so dedicating a small GPU is efficient).
- **KV-cache reuse is high** (prefix caching wins, encoder can be a different "shape" of compute).
- **TTFT/TPOT trade-off favors decoupling** (e.g., very long generations where decode needs to be uninterrupted).

None of these conditions hold for this benchmark:
- Qwen3-VL-32B encoder is ~3 GB, but the per-image preprocessing+ViT pass for 8×1080p images is comparable to the LLM prefill for those tokens.
- Random images = no KV reuse possible.
- Output is short (256 tokens) so decode quality matters less than prefill throughput.

---

## Bottom line

**Disagg as configured here is broken for this load profile** (NIXL deadlock around 1 req/s). Even if the NIXL deadlock were fixed, the architecture wouldn't beat 2× TP=1 DP for high-image-token workloads on this hardware. The pre-existing `DEBUG_PROCESS.md` in the same directory acknowledges this NIXL issue.

**Recommendation:** Stay with **2× TP=1 DP** (best-known: 0.76 req/s, 11.2 s TTFT) for this workload. Don't pursue disagg unless the NIXL deadlock can be reproduced and fixed upstream.

---

## Files in this experiment series

- `time_breakdown_for_agg_tp1_1080p_8image.md` — TP=1 baseline analysis
- `time_breakdown_for_agg_tp2_1080p_8image.md` — TP=2 SYS (GPUs 3,4)
- `time_breakdown_for_agg_tp2_gpu45_1080p_8image.md` — TP=2 NODE (GPUs 4,5)
- `why_tp2_worse_than_tp1_reason.md` — root cause of TP=2 regression
- `attemp_on_different_solutions_for_tp2.md` — failed SGLang flag fixes
- `two_tp1_summary.md` — 2× TP=1 DP winning configuration
- **`disagg_time_summary.md`** — this document (disagg analysis)

## Bench result directory

- Disagg: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_012229/`

## Worker log files

- PD worker: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker.log`
- Encoder worker: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/encoder_worker.log`
- Frontend: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/frontend.log`

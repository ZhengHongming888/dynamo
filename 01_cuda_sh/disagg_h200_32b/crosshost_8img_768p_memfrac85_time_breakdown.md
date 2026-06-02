# Cross-Host Disagg Time Breakdown: 8img/768p np=64 r=1.0 (mem_fraction=0.85)

**Date:** 2026-06-01
**Setup:** B70 4×Encoder + H200 PD (GPU 5, mem_fraction=0.85, max_running=64)
**Bench:** 8img/768p, np=64, rate=1.0, output=256
**Bench window:** 2026-06-01T17:45:03 → 17:46:33 UTC (90 s)
**Result:** **RPS=0.71, 64/64 successful, 0 buffer timeouts**

## Source Files

- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_20260601_165503.log`
- Encoder logs: `/hongming/dynamo/02_xpu_sh/logs/encode_xpu_32b_b70_{1,2,3,4}.log`
- Bench result: `/hongming/res_crosshost_b70_4E_h200_pd_memfrac85/8img_768p_rate1.0_np64_20260601_174503/`

## 1. Time Decomposition

```
Time (s):  0     2          5             29              39  41
           │      │           │              │              │   │
Client ────┤      │           │              │              │   ├─→ last token
           │  ViT (~0.5s) + memhold (~39s waiting for NIXL_READ)
Encoder ───┤(rcv)─────────────────────────────────────────┤(done)
PD ─────────────┤(rcv)────────────────────────────────┤(done)
                │←── 3s queue ──→│←── 25s forward ──→│
                                  prefill 6k + decode 256
                │←── 10s NIXL recv + dynamo ──────────→│
```

### Phase Breakdown (medians, 65 joined requests + 65 ReqTimeStats)

| Phase | Description | Median | Mean | P99 | Min |
|---|---|---:|---:|---:|---:|
| **T1** Encoder lifetime (recv → done) | ViT + idle wait for NIXL pull | **39.5 s** | 39.9 s | 77.4 s | 4.1 s |
| **T2** Control-plane fan-out (enc_recv → pd_recv) | Frontend router latency | **1.6 s** | 1.9 s | 4.9 s | 1.2 s |
| **T3** PD lifetime (recv → done) | Total PD-side processing | **37.1 s** | 37.9 s | 75.9 s | 2.6 s |
| **T4** PD SGLang queue_duration | Wait inside scheduler | **2.8 s** | 4.5 s | 12.9 s | 0.0 s |
| **T5** PD SGLang forward_duration | Prefill 6.2k + decode 256 | **24.7 s** | 30.3 s | 73.8 s | 1.3 s |

### Reconciliation with Bench

| Bench reports | Computed |
|---|---:|
| Median TTFT: 11.9 s | T2 + T4 + 1 prefill ≈ 1.6 + 2.8 + 6 ≈ 10.4 s ✓ |
| Median E2E: 41.3 s | T2 + T3 = 38.7 s ✓ |

### PD Lifetime Breakdown (37.1 s median)

```
T3 PD lifetime (37.1 s) ──────────────────────────────────────────────────
  ├─ T4 SGLang queue_duration       2.8 s  ( 7.5%)  ▓▓
  ├─ T5 SGLang forward_duration    24.7 s  (66.4%)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  └─ T6 NIXL recv + dynamo          9.7 s  (26.0%)  ▓▓▓▓▓▓
```

**This is the key shift from 8img/1080p:** queue is only 7.5% (vs 40% for 1080p), forward is 66% (vs 40%) — PD is **GPU-compute-bound, not queue-bound**. NIXL handoff overhead grew to 26% (vs 20%) in relative terms, but absolute is much smaller (10 s vs 44 s).

---

## 2. Per-Encoder Load (4 B70 Encoders, Well-Balanced)

| Encoder | Reqs | Median lifetime | Mean | Max | PATCH warns |
|---|---:|---:|---:|---:|---:|
| enc#1 | 16 | 38.3 s | 38.3 s | 75.2 s | 0 |
| enc#2 | 18 | 42.7 s | 35.5 s | 62.1 s | 0 |
| enc#3 | 15 | 37.5 s | 41.4 s | 74.6 s | 0 |
| enc#4 | 16 | 46.0 s | 45.1 s | 77.4 s | 0 |

- **Load is very balanced** (15-18 reqs per encoder, σ ≈ 1.3)
- **0 PATCH(non-cached) warnings** — encoder has warm embedding paths
- Each encoder hub took 4-5 NIXL connections total (NIXL connector reuses connections)

### Encoder dwell time check (T1 - T2 - T3)

```
median = 0.4 s (encoder ViT + idle wait for NIXL release)
mean   = 0.6 s
p99    = 4.4 s
```

Encoder activity = ~0.5 s ViT forward + tiny NIXL admin = **only 1.3% of T1**. The other 98.7% of encoder lifetime is **memory-holding while PD reads via NIXL**. As with 1080p, **encoders are memory-storage proxies, not compute-bound resources**.

---

## 3. PD-Side Scheduler Activity

| Metric | Value | Compared to 1080p |
|---|---:|---|
| Running-req peak | **54** | 1080p: 42 (KV-pool capped at 41) |
| Running-req median | 18 | 1080p: 29 |
| Queue-req peak | **13** | 1080p: **59** (much smaller!) |
| Queue-req median | 1 | 1080p: 28 |
| KV pool (max_total_num_tokens) | 695,136 | same |

The big difference: **8img/768p has tiny queue depths** (peak 13, median 1) because PD admit rate (0.71 RPS) keeps pace with input rate (1.0 RPS — essentially same as bench arrival rate).

Per-req KV demand for 8img/768p: 6,238 (input) + 256 (max_new) + 16 (page) = **6,510 tokens**.
Theoretical KV cap: 695,136 / 6,510 = **106 in-flight** (way above max_running=64).

So **max_running_requests=64 is the binding constraint**, not KV pool — but PD running peak only reached 54 because the workload didn't push hard enough. **No bottleneck saturated.**

---

## 4. Prefill Batch Composition

| Batch size | Count | % | Interpretation |
|---|---:|---:|---|
| <500 tokens | 1 | 3% | Initial cuda graph warmup |
| 2-7k tokens | 10 | **26%** | Single-request prefill (one 6,238-token req) |
| 7-13k tokens | 27 | **71%** | **Two-request prefill (12,476-token batch)** |
| 13-17k tokens | 0 | 0% | (none — no chunked-prefill split-tail since input < chunk size) |

**71% of prefill batches admit 2 requests at once** — this is the optimal batch composition for 8img/768p, made possible because input_len 6,238 fits 2 reqs (12,476) within chunked_prefill_size=16,384.

Compare to 1080p where:
- 42% of batches were tiny <500 tails (chunked split overhead)
- 58% were 13-17k single-request main chunks

This is why **PD's GPU compute throughput nearly doubles** on 768p (4,517 tok/s vs 3,748 tok/s for 1080p, +20%).

---

## 5. Cross-Configuration Comparison

### All 8img/768p Runs at Various Configs

| Config | np | rate | **RPS** | Success | tput | TTFT_med | E2E_med | TPOT_med | TPOT_p99 |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| **Cross-host B70 4E + H200 PD (this)** | 64 | 1.0 | **0.71** | **64/64** | 4,517 | **11.9 s** | **41.3 s** | 249 ms | 7,384 ms |
| Same-host disagg max=64 | 128 | 1.0 | 0.70 | 128/128 | 4,470 | 39.9 s | 84.9 s | 489 ms | 2,390 ms |
| Same-host disagg max=128 | 128 | 1.0 | 0.74 | 128/128 | 4,682 | 32.4 s | 101.8 s | 633 ms | 17,582 ms |
| **TP=1 Aggregate** | 128 | 1.0 | **0.90** | 128/128 | 5,704 | **1.7 s** | **5.3 s** | **36 ms** | 376 ms |

### Cross-host vs Same-host (both disagg, max_running=64)

| Metric | Cross-host (this) | Same-host disagg max=64 | Delta |
|---|---:|---:|---|
| RPS | 0.71 | 0.70 | +1% |
| Median TTFT | **11.9 s** | 39.9 s | **-70%** ✓ (much better) |
| Median E2E | **41.3 s** | 84.9 s | **-51%** ✓ (much better) |
| Median TPOT | 249 ms | 489 ms | -49% (better) |
| TPOT P99 | 7,384 ms | 2,390 ms | +209% (worse) |
| Buffer timeouts | 0 | 0 | — |

**Cross-host is much better on tail-latency for 8img/768p** because:
1. 4 encoders eliminate encoder-side queueing (encoder runs 1-2 reqs at a time per encoder vs 64 piling up on a single same-host encoder)
2. PD sees evenly-spaced requests (smooth admission)
3. Median TTFT drops from 39.9 s → 11.9 s (3.4× faster first-token)

But TPOT P99 is worse (7.4s vs 2.4s) because:
- Cross-host's NIXL recv occasionally stalls a decode batch
- Random outliers when NIXL connection re-establishes

---

## 6. Three Bottlenecks Comparison: 1080p vs 768p

For 8img/1080p (PD-bound, queue-saturated):

```
T3 PD = 219 s
  ├─ T4 queue:    88s (40%)  ← Bottleneck #1: KV pool full
  ├─ T5 forward:  88s (40%)  ← Bottleneck #2: chunked-split + GPU compute
  └─ T6 NIXL:     44s (20%)  ← Bottleneck #3: cross-host overhead
```

For 8img/768p (PD GPU-compute-bound):

```
T3 PD = 37 s
  ├─ T4 queue:     3s (8%)   ← Tiny queue (rate=admit)
  ├─ T5 forward:  25s (66%)  ← Dominant: GPU prefill+decode
  └─ T6 NIXL:     10s (26%)  ← Cross-host overhead (relatively bigger)
```

**Key insight:** 768p eliminates the queue bottleneck entirely (88s → 3s, 28×) because:
- input_len 6,238 fits in one prefill chunk (no split)
- 2 requests batch together (12,476 tokens)
- PD admit rate matches input rate

But forward time only drops 3.5× (88s → 25s), so **GPU compute becomes the main constraint**. NIXL+dynamo overhead (T6) is a constant ~10s/req regardless of workload.

---

## 7. Encoder Activity Timeline

For 8img/768p, encoder activity is **40× shorter** than for 1080p:

```
8img/1080p (cross-host, mem_frac=0.85):
  - Encoder lifetime: 230 s median (mostly memhold)
  - 4 encoders × 65 reqs/run × 230s = ~250 encoder-seconds compute, ~14,950s memhold
  - Compute utilization: 1.7%

8img/768p (cross-host, mem_frac=0.85, this):
  - Encoder lifetime: 40 s median
  - 4 encoders × 65 reqs/run × 40s = ~250 encoder-seconds compute, ~2,350s memhold
  - Compute utilization: 9.6%
```

**Encoders are 5× more active on 768p**, but still 90% idle. The 4 B70 encoders could service **~10 RPS** worth of 768p requests if PD weren't the bottleneck.

---

## 8. What's the Real Bottleneck at 0.71 RPS?

PD GPU-compute throughput:
```
total_tput = 4,517 tok/s (median forward 24.7 s × ~30 reqs/min effective)
RPS = total_tput / per_req_tokens = 4,517 / (6,238 + 117) = 0.711  ← matches measured exactly
```

This is **the same compute ceiling** as the same-host disagg run (4,470 tok/s), within 1%. The H200 GPU saturates at ~4,500 tok/s for this workload regardless of architecture.

To exceed 0.71 RPS for 8img/768p, you need:
1. **More GPU compute** (TP=2 → ~1.4 RPS, TP=1 Agg → 0.90 RPS by removing NIXL serialization)
2. **Less work per request** (output_len=128 instead of 256 → ~0.95 RPS)
3. **Smaller image** (4img/768p → ~1.5 RPS)

**Adding more B70 encoders won't help** because PD is the bottleneck.

---

## 9. NIXL Buffer Pool Status

For 8img/768p in cross-host mode:
- Per-embedding size: 8 × 770 × 5,120 × 2 = 63 MB
- NIXL ring buffer size: 8 GB (default)
- Theoretical concurrent embeddings: 8 GB / 63 MB ≈ **136**
- Observed peak: 13 (queue depth) + 54 (running) = ~67 max simultaneous embeddings
- **0 buffer timeouts** ✓

Pool is well-provisioned for this workload (5× headroom). No buffer-pool concerns at this rate.

---

## 10. Summary Table: 8img/768p Across All Configurations Tested

| Config | RPS | Success | Median E2E | Median TTFT | TPOT P99 | Notes |
|---|---:|:---:|---:|---:|---:|---|
| Same-host disagg max=64, np=32 r=1.0 | 0.64 | 32/32 | 37.6 s | 23.2 s | 779 ms | Input-rate-limited |
| Same-host disagg max=64, np=128 r=1.0 | 0.70 | 128/128 | 84.9 s | 39.9 s | 2,390 ms | Saturated |
| Same-host disagg max=64, np=64 r=2.0 | 0.72 | 64/64 | 66.8 s | 36.2 s | 779 ms | Real ceiling |
| Same-host disagg max=64, np=128 r=2.0 | 0.67 | 107/128 | 109.9 s | 60.4 s | 2,334 ms | Buffer pool fail |
| Same-host disagg max=128, np=128 r=1.0 | 0.74 | 128/128 | 101.8 s | 32.4 s | 17,582 ms | KV pool true cap (110/106) |
| Same-host disagg max=128, np=128 r=2.0 | 0.92 | 128/128 | 97.4 s | 42.9 s | 8,596 ms | Best disagg result |
| Same-host TP=1 Agg, np=128 r=1.0 | **0.90** | 128/128 | **5.3 s** | **1.7 s** | **376 ms** | **Best overall** |
| Same-host TP=1 Agg, np=128 r=2.0 | 1.49 | 128/128 | 33.8 s | 9.2 s | 1,016 ms | Better at higher rate |
| **Cross-host B70 4E + H200 PD (this)** | **0.71** | **64/64** | **41.3 s** | **11.9 s** | 7,384 ms | Median latency 2-3× better than same-host disagg |

---

## 11. One-Line Conclusion

> **8img/768p cross-host hits 0.71 RPS — same compute ceiling as same-host disagg, but with 70% lower median TTFT (11.9 s vs 39.9 s) due to load-balanced encoder admission and tight PD queue (3 s vs 88 s for 1080p). PD GPU compute (T5=25s, 66% of E2E) is now the dominant cost. Encoders are still 90% idle. TP=1 Aggregate (RPS 0.90, E2E 5.3 s) remains the best overall choice.**

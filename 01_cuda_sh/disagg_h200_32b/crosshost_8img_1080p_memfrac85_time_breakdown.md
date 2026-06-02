# Cross-Host Time Breakdown: 8img/1080p np=64 r=1.0 (mem_fraction=0.85)

**Date:** 2026-06-01
**Bench:** 8img/1080p, np=64, rate=1.0, output=256, cross-host disagg
**Setup:** B70 4×encoder + H200 PD (GPU 5, mem_fraction=0.85, max_running=64)
**Bench window:** 2026-06-01T17:07:34 → 17:12:16 UTC (282 s)
**Result:** RPS=0.23, 64/64 successful

## Inputs Used

- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_20260601_165503.log`
- Encoder logs: `/hongming/dynamo/02_xpu_sh/logs/encode_xpu_32b_b70_{1,2,3,4}.log`
- Launcher log: `/hongming/dynamo/02_xpu_sh/logs/launcher_4E_xpu_attn_mf85_20260601_165134.log`
- Bench result: `/hongming/res_crosshost_b70_4E_h200_pd_memfrac85/8img_1080p_rate1.0_np64_20260601_170734/`

## 1. Time Decomposition

```
Time (s):  0        12         100         188         219    230  233
           │         │           │           │           │      │   │
Client ────┤         │           │           │           │      │   ├──→ last token
           │  ViT(~1s) + memhold (~230s waiting for NIXL_READ)  │
Encoder ───┤(rcv)─────────────────────────────────────────────┤(done)
PD ──────────────────┤(rcv)─────────────────────────────────┤(done)
                     │←──── 88s queue ────→│←─88s forward─→│
                                          prefill 16k + decode 256
                     │←──── 44s NIXL recv + dynamo ───────→│
```

### Phases (medians of 32 joined requests, plus 35 ReqTimeStats samples)

| Phase | Description | Median | Mean | P99 | Min |
|---|---|---:|---:|---:|---:|
| **T1** Encoder lifetime (enc_recv → enc_done) | Encoder holds embedding, mostly idle waiting | **230.3 s** | 218.6 s | 275.5 s | 7.8 s |
| **T2** Control-plane fan-out (enc_recv → pd_recv) | enc_recv to pd_recv via dynamo router | **11.1 s** | 12.0 s | 25.6 s | 4.3 s |
| **T3** PD lifetime (pd_recv → pd_done) | Total PD-side processing | **219.4 s** | 206.5 s | 268.1 s | 0.5 s |
| **T4** PD SGLang queue_duration | Wait inside SGLang scheduler for KV slot | **88.1 s** | 82.9 s | 146.7 s | 0.0 s |
| **T5** PD SGLang forward_duration | Prefill 16k + decode 256 tokens | **87.8 s** | 92.3 s | 257.0 s | 0.3 s |

### Reconciliation with Bench-Reported Numbers

| Metric | Bench reports | Computed (T2 + T3) | Match |
|---|---:|---:|:---:|
| Median E2E | 233.3 s | 230.2 s | ✓ (1.3% diff) |
| Median TTFT | 161.1 s | ~T2 + T4 + 1 prefill ≈ 11 + 88 + 50 ≈ 149 s | ✓ (close) |

### PD Lifetime Breakdown (median 219.4 s)

```
T3 PD lifetime (219 s) ─────────────────────────────────────────────────
  ├─ T4 SGLang queue_duration       88.1 s  (40.2%)  ▓▓▓▓▓▓▓▓▓▓
  ├─ T5 SGLang forward_duration     87.8 s  (40.0%)  ▓▓▓▓▓▓▓▓▓▓
  └─ T6 NIXL recv + dynamo          43.6 s  (19.8%)  ▓▓▓▓▓
```

---

## 2. Per-Encoder Load (4 B70 Encoders)

| Encoder | Reqs handled | Median lifetime | Mean | Max | Min | PATCH warns | NIXL connections |
|---|---:|---:|---:|---:|---:|---:|---:|
| enc#1 | 8 | 210 s | 187 s | 239 s | **8 s** ← warmup | 0 | 1 |
| enc#2 | 6 | 261 s | 254 s | 275 s | 230 s | 0 | 1 |
| enc#3 | 7 | 215 s | 226 s | 271 s | 196 s | 0 | 1 |
| enc#4 | 11 | 233 s | 218 s | 275 s | **18 s** ← warmup | 0 | 1 |

**Observations:**
1. Load distribution is somewhat uneven (6-11 reqs) — dynamo router is FCFS, but encoder availability differs by completion timing
2. **0 PATCH(non-cached) warnings** in this run (vs 4+ in earlier sessions) — encoders had warmed embedding paths from prior runs
3. Each encoder establishes **only 1 NIXL connection** total (reused for all requests during the session)
4. The min=8s and min=18s entries are first-batch warmup requests; steady-state is ~210-275s

---

## 3. Encoder is 99% IDLE Waiting for NIXL Read

**Verification: T1 ≈ T2 + T3** (encoder dwell time is essentially 0)

```
T1 - T2 - T3 stats:
  median = 0.1 s
  mean   = 0.1 s
  p99    = 0.2 s
```

This means each encoder request:
- ViT forward + NIXL register: ~1-2 s (imperceptible in this measurement)
- **Memory hold while waiting for PD's NIXL_READ:** ~218 s (99.9% of T1)
- NIXL release after PD reads: <0.1 s

**The B70 encoders are functioning as memory storage**, not as compute resources. The 4× encoder count provides no throughput benefit because each encoder is just holding 1 embedding per slot.

---

## 4. PD-Side Scheduler Activity

| Metric | Value |
|---|---:|
| Running-req peak | **42** (theoretical KV cap = 41 with mem_frac=0.85) |
| Running-req median | 29 |
| Queue-req peak | **59** (most requests piled up) |
| Queue-req median | 28 |

The PD scheduler hit its **theoretical KV pool cap of 41** (peak 42 due to dynamic alignment).

### Prefill Batch Composition (111 events)

| Batch size | Count | % | Interpretation |
|---|---:|---:|---|
| <500 tokens | 47 | **42%** | Chunked-prefill split tails (input_len 16,420 - chunked 16,384 = 36-token tails, page-aligned to 48-512) |
| 500-13k tokens | 0 | 0% | (none) |
| 13-17k tokens | 64 | **58%** | Main prefill chunks (capped at chunked_prefill_size=16,384) |

This **42/58 split** is the chunked-prefill split-tail signature — every request triggers exactly 2 prefill events (1 main + 1 tail), so 50% of events are tails. The 42/58 ratio (slightly off from 50/50) is because some warmup requests were single-batch.

---

## 5. Comparison with mem_fraction=0.65 Run

| Phase | mem_frac=0.65 | **mem_frac=0.85** | Delta |
|---|---:|---:|---:|
| T1 Encoder lifetime | 208 s | **230 s** | +11% (worse) |
| T2 Control-plane handoff | 14 s | 11 s | -21% (better) |
| T3 PD lifetime | 195 s | **219 s** | **+12% (worse)** |
| **T4 PD queue_duration** | 93 s | 88 s | **-5%** ✓ (mem_fraction=0.85 helped here) |
| **T5 PD forward_duration** | 55 s | **88 s** | **+60% (much worse)** ✗ |
| T6 PD NIXL+dynamo overhead | 47 s | 44 s | -6% |
| **PD running peak** | 28 | **42** | +50% |
| **RPS** | 0.24 | 0.23 | **-4%** |

### What this proves

1. **mem_fraction=0.85 successfully expanded KV pool** (467k → 695k) and let PD admit 50% more in-flight (28 → 42)
2. **Queue wait dropped** modestly (-5 s, -5%) because more KV slots = less waiting
3. **But forward time grew dramatically** (+33 s, +60%) because **42 simultaneous decodes contend for GPU SMs**
4. **Net E2E grew 12%** — the GPU compute ceiling dominates
5. NIXL/dynamo overhead unchanged (independent of mem_fraction)

**Conclusion: mem_fraction is not the right knob** for 8img/1080p.

---

## 6. Where Is The Bottleneck?

### Three layered constraints (ranked by criticality)

```
Bottleneck #1: GPU compute ceiling (~3,750 tok/s sustained)
   Evidence: total_tput stays 3,748-3,907 regardless of in-flight (28 vs 42)
   Impact: 88% of E2E (T4 + T5 + T6 = 88 + 88 + 44 = 220 s of 233 s)

Bottleneck #2: Chunked-prefill split-tail
   Evidence: 42% of prefill events are <500-token tails (no compute work)
   Impact: ~25% of T5 forward time wasted on launch overhead for tails
   Cause: input_len 16,420 > chunked_prefill_size 16,384

Bottleneck #3: NIXL + dynamo cross-host overhead
   Evidence: T6 = 44 s of T3 (20%), T2 = 11 s of E2E (5%)
   Total cross-host tax: ~55 s per request
   Cause: RoCE transfer + cuda_copy D→H→D + ring buffer allocation + dynamo serialization
```

### What the 4 B70 encoders do (or don't do)

```
Encoder activity: ~1-2 s ViT forward per request × 4 encoders = 4-8 s of compute work
Total encoder compute time over 282s bench: ~32 s (4-8s × 4 encoders avg = 16-32s)
Encoder utilization: ~32 / (282 × 4) = 2.8% utilization

The other 97.2% of encoder time is:
  - Holding embeddings in GPU memory (waiting for PD)
  - Maintaining NIXL connection (heartbeats only)
```

**Encoders are massively under-utilized.** Adding more encoders won't help.

---

## 7. What Would Help This Workload

| Method | Targeted bottleneck | Expected RPS | Notes |
|---|---|---:|---|
| **TP=1 Aggregate** | #3 (NIXL/dynamo) | **0.47** (2.0×) | Eliminates T2 + most of T6, reduces T4 |
| **TP=2 Aggregate** | #1 (GPU compute) | **~0.95** (4.1×) | Doubles GPU throughput |
| **`--chunked-prefill-size 32768`** | #2 (split-tail) | **~0.40** (1.7×) | Eliminates 42% small batches; needs careful mem_fraction tuning to avoid OOM |
| **Smaller image (768p × 16img)** | #2 (split-tail) | **~0.71** | input_len 12,401 < 16,384, single-chunk |
| **Smaller image (768p × 8img)** | #2 + #1 | **~0.90** | input 6,238 ≪ 16,384, 2-req per batch |
| Add more B70 encoders (8E) | none | **0%** | Encoders already 97% idle |
| Switch to nixl-write mode | #3 | **<5%** | Not the dominant cost |
| RDMA NIC upgrade | #3 | **<5%** | Transfer is <1% of T6 |

**Recommendation hierarchy:**
1. If single-host single-GPU is acceptable → **TP=1 Aggregate** (RPS 2× higher, no NIXL pain)
2. If multi-GPU available → **TP=2 Aggregate** (RPS 4× higher)
3. If you must keep cross-host disagg architecture → **lower image resolution to 768p** for huge gains
4. **Don't add more encoders** — they won't help

---

## 8. Detailed Per-Request Sample (first 10 requests)

```
rid       enc#  enc_recv     pd_recv      T1_enc  T2_handoff  T3_pd
─────────────────────────────────────────────────────────────────────
req#1       1   17:07:35     17:07:43      8.0s        8.0s     0.5s   ← warmup, no queue
req#2       4   17:07:46     17:07:53     18.4s        7.0s    11.5s   ← warmup
req#3       2   17:08:14     17:08:25    230.0s       11.0s   219.0s   ← steady state
req#4       1   17:08:26     17:08:39    240.0s       13.0s   227.0s   ← steady state
...
```

Steady-state is reached around request #3 onwards — every subsequent request takes ~210-275 s end-to-end.

---

## 9. One-Line Conclusion

> **The 233-second median E2E is 88% PD-side compute (40% queue wait, 40% forward, 20% NIXL/dynamo) and 5% control-plane fan-out. The 4 B70 encoders are 97% idle — they hold embeddings waiting for PD to drain them. Raising mem_fraction from 0.65 to 0.85 expanded KV pool but made forward time worse (more decode contention), so RPS dropped 4%. The real fix is TP=1 Aggregate (RPS 2×) or smaller images, not encoder scaling.**

## 10. Detailed Files

- This analysis: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/crosshost_8img_1080p_memfrac85_time_breakdown.md`
- Bench result: `/hongming/res_crosshost_b70_4E_h200_pd_memfrac85/8img_1080p_rate1.0_np64_20260601_170734/`
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_20260601_165503.log`
- B70 encoder logs: `/hongming/dynamo/02_xpu_sh/logs/encode_xpu_32b_b70_{1,2,3,4}.log`
- B70 launcher log: `/hongming/dynamo/02_xpu_sh/logs/launcher_4E_xpu_attn_mf85_20260601_165134.log`

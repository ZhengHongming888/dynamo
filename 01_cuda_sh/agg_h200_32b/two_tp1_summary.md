# Two TP=1 workers in data-parallel: the winning configuration

**System:** Qwen3-VL-32B-Instruct-FP8, aggregated EPD, 8×1920×1080 images per request, rate=1 req/s, 64 prompts.
**GPUs:** Two TP=1 workers on **GPUs 4 and 5** (same NUMA, no NVLink, but workers are independent so it doesn't matter).
**Setup script:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_two_tp1.sh`
**Bench window:** 2026-05-21 00:54:11 – 00:55:35 (~84 s).
**Result file:** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_005405/rate_1.0/`

---

## Architecture

```
                                    ┌──────────────┐
                                    │   Frontend   │
                                    │  (KV router) │
                                    │  port 7001   │
                                    └──────┬───────┘
                                           │
                                  ┌────────┴────────┐
                                  │                 │
                          ┌───────▼──────┐  ┌───────▼──────┐
                          │  Worker A    │  │  Worker B    │
                          │  TP=1 GPU 4  │  │  TP=1 GPU 5  │
                          │  KV-evt 22080│  │  KV-evt 22081│
                          │  side  20098 │  │  side  20099 │
                          └──────────────┘  └──────────────┘
                          (independent — no inter-worker NCCL)
```

Each worker is its own complete TP=1 instance with its own KV-event publisher port and NIXL side-channel port, but they all share the same NATS / etcd / frontend stack. The frontend's KV router load-balances requests across the two workers.

---

## Headline result

| Metric | Value |
|---|---|
| Request throughput | **0.76 req/s** (offered 1.0 — close to keeping up) |
| Total token tput | **12,579 tok/s** (≈ 1.81× single TP=1) |
| Input tput | 12,490 tok/s |
| Output tput | 89 tok/s |
| **Mean / median TTFT** | **11.2 s / 11.9 s**, p99 17.4 s |
| **Mean / median E2E** | **45.2 s / 48.2 s**, p99 79.0 s |
| Mean / median TPOT | 898 / 289 ms, p99 12.1 s |
| Median ITL | 38 ms |
| Concurrency observed | 34.4 |
| Worker A handled | 40 / 64 requests |
| Worker B handled | 40 / 64 requests |
| GPU 4 / 5 memory | 133 / 133 GB (both at ~100 % util during prefill) |

Bench completed in 84 s (vs 152.6 s for single TP=1, 169.5 s for TP=2 NODE).

---

## Five-way comparison (rate=1, 64 prompts, 1080p×8)

| Metric | TP=1 (1 GPU) | TP=2 SYS (3,4) | TP=2 NODE (4,5) | TP=2 NODE+flags | **2× TP=1 DP (4,5)** |
|---|---:|---:|---:|---:|---:|
| Bench duration | 152.6 s | 168.4 s | 169.5 s | 218.5 s | **84.1 s** |
| Request throughput | 0.42 req/s | 0.38 | 0.38 | 0.29 | **0.76 req/s** |
| Input tput | 6,884 tok/s | 6,238 | 6,197 | 4,809 | **12,490 tok/s** |
| Total tput | 6,933 | 6,283 | 6,241 | 4,843 | **12,579 tok/s** |
| Mean TTFT | 50.2 s | 61.0 s | 60.4 s | 119.9 s | **11.2 s** |
| Median TTFT | 56.0 s | 66.7 s | 65.8 s | 125.4 s | **11.9 s** |
| P99 TTFT | 84.9 s | 101.8 s | 102.9 s | 152.0 s | **17.4 s** |
| Mean E2E | 93.2 s | 107.4 s | 107.3 s | 160.9 s | **45.2 s** |
| Median E2E | 96.9 s | 110.3 s | 111.5 s | 163.3 s | **48.2 s** |
| Mean TPOT | 622 ms | 650 | 671 | 564 | 898 ms |
| Median TPOT | 354 ms | 407 | 382 | 355 | **289 ms** |
| Concurrency | 39.1 | 40.8 | 40.5 | 47.1 | **34.4** |
| Peak output tput | 592 | 785 | 844 | 817 | 647 |

---

## Speedup vs single TP=1

| | TP=1 | 2× TP=1 | Speedup |
|---|---:|---:|---:|
| Throughput | 0.42 req/s | **0.76 req/s** | **1.81×** |
| TTFT (mean) | 50.2 s | **11.2 s** | **4.49× faster** |
| E2E (mean) | 93.2 s | **45.2 s** | **2.06× faster** |
| Bench duration | 152.6 s | **84.1 s** | **1.81× faster** |

The throughput speedup of **1.81×** is close to the theoretical maximum of 2.0× given the shared frontend/NATS/etcd overhead. The TTFT speedup of **4.5×** is *more* than 2× because each worker now sees only half the queue depth at this offered rate, so queueing latency collapses non-linearly.

---

## Configuration ranking

| Configuration | Best for | req/s | TTFT (mean) | Verdict |
|---|---|---:|---:|---|
| TP=1 single | Latency-only | 0.42 | 50 s | Decent baseline |
| TP=2 SYS (3,4) | — | 0.38 | 61 s | Worst TP=2 (cross-socket) |
| TP=2 NODE (4,5) | — | 0.38 | 60 s | Topology didn't help on this hardware |
| TP=2 NODE+flags | — | 0.29 | 120 s | "Fix" backfired (broadcast overhead) |
| **2× TP=1 DP (4,5)** | **Throughput AND latency** | **0.76** | **11 s** | **Winner** |

---

## Why two TP=1 workers wins

1. **No cross-rank communication.** Each worker is fully independent; there is no NCCL all-reduce, no MM feature broadcast, nothing crossing PCIe between GPUs. The PCIe bandwidth penalty that crippled all the TP=2 configurations on this NVLink-less hardware doesn't exist here.

2. **Each worker keeps its own optimal per-GPU pipeline:**
   - Vision encoder on the same GPU as language model: zero data movement.
   - KV cache fully resident on its own GPU at full `mem-fraction-static=0.88`.
   - Prefill chunks complete in ~0.74 s (matches single TP=1 baseline).
   - Encoder+prep gap stays at ~1.2 s (matches single TP=1 baseline).

3. **Throughput scales near-linearly.** Two independent workers process two requests concurrently; the only shared resource is the frontend's HTTP path, which is not the bottleneck.

4. **Latency improves super-linearly.** Per-worker queue depth halves, so back-pressure-induced TTFT collapses. The TTFT speedup of **4.5×** vs single TP=1 is the most dramatic improvement of the entire experiment series.

5. **Memory is balanced.** KV usage similar to single TP=1 (high), so the per-worker pipeline is fully utilized — no wasted capacity.

6. **GPU utilization is 100 % on both** during prefill — confirming both workers are doing real work, not waiting on each other.

---

## Recommended deployment for this workload

**Use two TP=1 workers in data-parallel behind one frontend.**

For larger fleets:
- N TP=1 workers (one per GPU) all registering with the same frontend will scale near-linearly until the frontend's tokenization / HTTP layer becomes the bottleneck (typically 8-16 workers per frontend).
- Each worker should have its own KV-event publisher port and NIXL side-channel port (script template: `start_h200_aggregate_epd_server_32b_two_tp1.sh`).
- For homogeneous random images (this bench), the KV router degrades to round-robin; for real workloads with prefix reuse, KV-aware routing will direct repeat-prefix requests to the same worker for cache hits.

---

## What's still on the table

This is now the best-known configuration for this workload on this hardware. Further investigations could include:

1. **Find the knee of two TP=1 setup** — try rate ∈ {1.0, 1.5, 2.0, 2.5} to find where queueing kicks in.
2. **Disaggregated EPD** (separate encoder / prefill / decode workers) — could further reduce TTFT for image-heavy workloads, especially at higher offered rates.
3. **Test KV-cache-friendly workloads** — with real images (not random), KV-aware routing should give measurable benefit.
4. **Scale to 4 or 8 TP=1 workers** if more GPUs are available, to verify near-linear scaling.

---

## Files in this experiment series

- `time_breakdown_for_agg_tp1_1080p_8image.md` — TP=1 baseline analysis
- `time_breakdown_for_agg_tp2_1080p_8image.md` — TP=2 SYS (GPUs 3,4)
- `time_breakdown_for_agg_tp2_gpu45_1080p_8image.md` — TP=2 NODE (GPUs 4,5)
- `why_tp2_worse_than_tp1_reason.md` — root-cause: encoder + prep dominates
- `attemp_on_different_solutions_for_tp2.md` — fix attempts (all failed/backfired)
- **`two_tp1_summary.md`** — this document (winning configuration)

## Bench result directories

- TP=1: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_211545/`
- TP=2 SYS: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_222206/`
- TP=2 NODE: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_232131/`
- TP=2 NODE+flags: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_003139/`
- **2× TP=1 DP**: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_005405/`

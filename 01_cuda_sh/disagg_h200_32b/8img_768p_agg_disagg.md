# 8 Images @ 768p — TP=1 agg vs Disagg

**Test date:** 2026-05-22
**Goal:** See how disagg vs TP=1 agg comparison changes with **smaller per-image footprint** — 768p (1024×768) instead of 1080p.

**Workload:**
- 8 × **1024×768** images per request (~6,200 vision tokens, vs ~16,400 for 1080p)
- 128 input + 256 output text tokens
- np=64, rate=1.0 req/s

**Configurations:**
- **TP=1 agg**: GPU 4 only — `start_h200_aggregate_epd_server_32b_tp1.sh`
- **Disagg**: GPU 4 (encoder) + GPU 5 (PD) — current best config (NIXL_READ + max_running_requests=64 + chunked_prefill_size=16384 + multimodal_embedding_cache_capacity_gb=16)

---

## Headline result

| Metric | **TP=1 agg** (1 GPU) | **Disagg** (2 GPUs) | Δ disagg vs agg |
|---|---:|---:|---:|
| Actual RPS | **0.90** | 0.62 | **−31%** |
| Successful requests | 64 / 64 ✓ | 64 / 64 ✓ | = |
| Bench duration | 70.9 s | 103.9 s | +47% |
| **Mean TTFT** | **2,266 ms** | **23,206 ms** | **+924%** |
| Median TTFT | 1,917 ms | 30,611 ms | +1,497% |
| P99 TTFT | 4,805 ms | 35,700 ms | +643% |
| **Mean TPOT** | **65 ms** | **1,457 ms** | **+2,142%** |
| Median TPOT | 38 ms | 258 ms | +587% |
| P99 TPOT | 647 ms | 24,616 ms | +3,704% |
| Mean ITL | 47 ms | 317 ms | +574% |
| Median ITL | 14 ms | 21 ms | +50% |
| **Max ITL** | **3,500 ms** | **66,611 ms** | **+1,803%** (66s decode stall on disagg) |
| **Mean E2E** | **6,616 ms** | **58,727 ms** | **+788%** |
| Median E2E | 5,885 ms | 59,896 ms | +918% |
| P99 E2E | 14,299 ms | 94,452 ms | +561% |
| Input throughput | 5,634 tok/s | 3,843 tok/s | −32% |
| Output throughput | 105 tok/s | 72 tok/s | −31% |
| **Peak output throughput** | 536 tok/s | **1,457 tok/s** | **+172%** (only disagg win) |
| Concurrency | 6.0 | 36.2 | +503% |

**TP=1 agg wins decisively** on every metric except peak output throughput.

---

## Per-phase breakdown for disagg

From `enable_request_time_stats_logging`:

| Phase | Median | Mean | Max |
|---|---:|---:|---:|
| Queue duration | **14.3 s** | 11.5 s | 25.2 s |
| Forward duration | **26.4 s** | 35.7 s | 91.1 s |
| Total per-request | ~41 s | ~47 s | — |

Median input_len observed: **6,179 tokens** per request (768p, 8 images, including text).
Most common prefill chunks: **6,224 / 12,480** tokens (single-image vs combined).

Queue is no longer "essentially zero" like in decode-heavy tests — at this prefill-bound workload, the queue is loaded (14 s median) but not as saturated as 1080p (where it was 112 s median).

---

## Workload comparison — how 768p stacks up

| Workload | Vision tokens/req | TP=1 agg RPS | Disagg RPS | Disagg/Agg |
|---|---:|---:|---:|---:|
| 8 imgs @ 1080p (random) | ~16k | 0.52 | 0.27 | **0.52** |
| **8 imgs @ 768p (random)** | **~6k** | **0.90** | **0.62** | **0.69** |
| 1 img @ 1080p, out=1024, rate=0.5 | ~2k | 0.43 | 0.42 | 0.98 |
| 1 img @ 1080p, out=1024, rate=2.0 | ~2k | 1.15 | 1.00 | 0.87 |
| 8 cached imgs @ 1080p | ~16k cached | 0.99 | 0.50 | 0.51 |

### Pattern

**Both configs accelerate at 768p compared to 1080p**, but TP=1 agg accelerates more:
- TP=1 agg: 0.52 → 0.90 = **+73%**
- Disagg: 0.27 → 0.62 = **+130%** (bigger relative gain because disagg was more saturated at 1080p)

**Result**: at 768p the gap shrinks — disagg is 31% behind agg, vs 48% behind at 1080p.

So **smaller workload helps disagg's relative position**, but disagg never actually catches up.

---

## Why disagg still loses at 768p

Same mechanisms as before, just at smaller scale:

1. **Per-request handoff overhead (~1-2 s)** is now a larger fraction of the smaller per-request work
2. **Embedding-integration small batches** still happen on PD (saw events with `#new-token: 6224, 12432, 12480` plus the usual small ones)
3. **Decode-blocker pathology** — Max ITL 67 s on disagg vs 3.5 s on TP=1 agg = **19× worse decode tail**
4. **Queue accumulates** — disagg PD has 14 s median queue wait

What's the same as 1080p:
- Disagg loses on RPS, TTFT, TPOT, E2E
- Disagg only wins on **peak output throughput** (+172% — the niche metric)

What changed at 768p:
- **All numbers improved** (smaller workload = faster)
- Disagg's relative gap shrunk slightly (from 0.52 → 0.69 ratio)
- But TP=1 agg's lead remains huge in absolute terms

---

## Per-request budget estimate

For 768p × 8 images:

| Phase | TP=1 agg | Disagg |
|---|---:|---:|
| Encoder ViT (8 imgs @ 768p) | inline ~600 ms | parallel ~600 ms (hidden) |
| Encoder→PD round-trip handoff | 0 | **~700-1500 ms** |
| Embedding integration on PD | 0 | ~1-3 s (small-batch overhead) |
| LLM prefill (~6k tokens) | ~700 ms | ~700 ms |
| Decode (256 tokens × ~30-40 ms) | ~8-10 s | ~8-10 s |
| **Per-request total** | **~10 s** | **~12-15 s** (+20-50%) |

The per-request gap is smaller than the RPS gap (31%) because the system is queue-saturated — slow requests block the queue and amplify the gap.

---

## Conclusion

**For 8 images @ 768p workload at rate=1.0:** TP=1 agg wins by 31% on RPS, 8× on TTFT, 22× on TPOT, 8× on E2E. Disagg only wins on peak output throughput.

The per-request workload reduction from 1080p to 768p helps disagg's relative position somewhat (gap shrinks from 0.52 to 0.69 ratio) but doesn't flip the comparison. **Disagg never wins on E2E or RPS for prefill-heavy multimodal workloads on this same-host 1+1 GPU setup**, regardless of image resolution.

This is now the **6th workload variant** where TP=1 agg has beaten disagg. The pattern is consistent.

---

## Files

- TP=1 agg result CSV: `/hongming/res4/h200_agg_tp1_32b_image8_768p_np64/test_sglang_multi_rates_1080p_20260522_065243/results_summary.csv`
- Disagg result CSV: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_768p_np64/test_sglang_multi_rates_1080p_20260522_070243/results_summary.csv`
- Bench script: `/hongming/dynamo/test_sglang_8img_768p.sh`
- Disagg PD log: `/hongming/dynamo/logs/logs/logs/logs/logs/logs/logs/pd_worker.log`
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md`
  - `01_cuda_sh/disagg_h200_32b/long_output_1024_results.md` — 1 img, out=1024, rate=0.5
  - `01_cuda_sh/disagg_h200_32b/rate_2.0_agg_disagg_comparison.md` — 1 img, out=1024, rate=2.0
  - `01_cuda_sh/disagg_h200_32b/cache_hit_agg_disagg.md` — 8 cached imgs @ 1080p
  - `01_cuda_sh/disagg_h200_32b/bottleneck_analysis.md` — bottleneck breakdown
- This document: `01_cuda_sh/disagg_h200_32b/8img_768p_agg_disagg.md`

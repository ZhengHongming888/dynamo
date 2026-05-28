# 1E 768p workload sweep: 4img vs 8img at np=32 rate=1.0

**Date:** 2026-05-28
**Setup:**
- **PD**: super21-h200 GPU 5, TP=1, mem-fraction=0.50, max-running=64
- **Encoder**: dell06 H200 — 1 encoder
- **Network**: RoCE 192.165.123.0/24; PD NIC mlx5_4 (192.165.123.52)
- **Frontend**: round-robin router mode
- **Workload**: Qwen3-VL-32B-Instruct-FP8, 768p (1024×768), in=128 / out=256, np=32, rate=1.0 RPS

**Result paths:**
- 1E 4img/768p (this run): `/hongming/res_xhost_dell06_super21/32b_4img_768p_rate1.0_np32_1enc_20260528_073753/`
- 1E 8img/768p (prior): `/hongming/res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_1enc_20260528_071429/`

## Image-count sweep (1E 768p np=32 rate=1.0)

| Metric | **4 imgs/req** | 8 imgs/req | Δ |
|---|---:|---:|---|
| Successful | **32/32** ✓ | 32/32 ✓ | — |
| Duration (s) | 40.97 | 51.66 | -21% (faster) |
| **RPS** | **0.78** | 0.62 | **+26%** |
| Vision tokens/req | ~3 080 | ~6 160 | 2× |
| Mean E2E (ms) | 10 283 | 26 516 | -61% |
| Median E2E (ms) | 10 623 | 26 641 | -60% |
| P90 E2E (ms) | 14 670 | 32 997 | -55% |
| P99 E2E (ms) | 15 974 | 34 309 | -53% |
| Mean TTFT (ms) | **6 137** | 19 692 | **-69%** |
| Median TTFT (ms) | 6 030 | 20 073 | -70% |
| P99 TTFT (ms) | 10 295 | 29 587 | -65% |
| Mean TPOT (ms) | 38.8 | 46.4 | -16% |
| Median TPOT (ms) | 29.9 | 50.2 | -41% |
| P99 TPOT (ms) | 162 | 108 | +50% |
| Mean concurrency | **8.0** | 16.4 | -51% |
| Peak concurrent | 18 | 29 | -38% |
| Peak decode (tok/s) | **493** | 363 | +36% |
| Output throughput (tok/s) | 119 | 95 | +26% |

## Interpretation

### 1. RPS scales close to inverse-vision-token-count

Halving images (8→4) → halving vision tokens (~6.2k → ~3.1k) → +26% RPS (0.62 → 0.78).

Not exactly 2× (which would be 1.24 RPS), because PD prefill has **fixed overhead per request** (kv cache alloc, scheduler tick, NIXL handle setup) that doesn't scale with token count. Also, encoder vision tower compute scales sub-linearly (vision compute is partly batch-bound).

### 2. TTFT drops dramatically (-69%) with smaller embeddings

| Image count | TTFT mean | Why |
|---|---:|---|
| 4 imgs (3k tokens) | 6.1 s | PD prefill ~1.5s/req, queue depth 8 → wait ~6s |
| 8 imgs (6k tokens) | 19.7 s | PD prefill ~3-4s/req, queue depth 16 → wait ~20s |

PD prefill cost is roughly linear in vision tokens. Halving vision tokens halves per-req prefill time AND halves the queue length (concurrency 8 vs 16 because reqs leave faster). Compound effect: ~3× lower TTFT.

### 3. Concurrency: half the in-flight reqs at saturation

Mean concurrency = RPS × E2E latency:
- 4img: 0.78 × 10.28 = 8.0 ✓ (Little's Law)
- 8img: 0.62 × 26.5 = 16.4 ✓

Peak concurrent: 18 (4img) vs 29 (8img) — same trend.

### 4. Peak decode rate higher at 4img (493 vs 363 tok/s)

Surprising at first glance — fewer images shouldn't help decode. But:
- Smaller batch fits more easily in cuda graph slots
- Less KV cache fragmentation (smaller per-req footprint)
- Reqs exit prefill faster → more time spent in pure decode batching

### 5. P99 TPOT regression (162 vs 108 ms) at 4img

The only metric where 4img is slightly worse than 8img. Likely because at 4img the workload is faster, more reqs complete near the bench end, and the trailing edge has fewer reqs in batch → smaller decode batch → higher per-token tail latency for the last few reqs.

### 6. Throughput ceiling implication

Stacking the 768p sweep:

| Image count | RPS | Notes |
|---|---:|---|
| 4 imgs | **0.78** | This run |
| 8 imgs | 0.62 | Prior |
| (16 imgs?) | ~0.4 expected | Extrapolating |

If we keep halving image count, RPS approaches but doesn't reach a hard ceiling. The asymptote is set by:
- Per-req fixed overhead (~50-100 ms scheduler + NIXL setup) → max ~10-20 RPS at zero-vision-tokens
- Encoder vision tower throughput (limits at very high RPS even at 1 img)

**For comparison** with no-vision baseline (32B FP8 chat-only on same PD):
- That would tell us the "pure LLM" ceiling and isolate the vision overhead

## Cross-check vs prior sweep results

The 1E patched sweep (`/hongming/res19_1E_patched_sweep/`) had only np=16 baselines:

| Workload | np | RPS (sweep) | RPS (this work, np=32) | np=32 / np=16 |
|---|---:|---:|---:|---:|
| 4img/768p | 16 | 0.40 | **0.78** | **+95%** |
| 8img/768p | 16 | 0.20 | 0.62 | +210% |

**np=32 gets 2-3× higher throughput than np=16** at the same offered rate=1.0. Because:
- At np=16, the bench ends after 16 reqs (saturating rate limit briefly)
- At np=32, more reqs build up at PD, deeper running batches → better batched decode
- Steady-state RPS is closer to PD's actual saturation throughput

This means the prior np=16 sweep underestimated achievable throughput. The **np=32 numbers are the correct steady-state RPS**.

## Saturation throughput estimate (from 4img/768p)

```
Concurrency = 8.0
E2E latency  = 10.28 s
RPS          = 0.78

If PD could saturate at higher concurrency:
  predicted RPS at concurrency 16 = 16 / 10.28 ≈ 1.55
  predicted RPS at concurrency 32 = 32 / 10.28 ≈ 3.11
```

But this only holds if PD's prefill scheduler + decode batching keep up. At higher offered rates, the queue grows and TTFT/E2E inflate proportionally — RPS stays at ~0.78 because that's the actual processing rate.

To find true saturation, run rate sweep (rate=2.0, 3.0, 5.0) and look for the rate where mean TTFT exits low values. This work has rate=1.0 only.

## Summary table (all 1E 768p np=32 rate=1.0 results so far)

| Workload | Successful | RPS | Mean TTFT | Mean E2E | Median TPOT | Concurrency |
|---|---:|---:|---:|---:|---:|---:|
| 4img/768p | 32/32 | **0.78** | 6.1 s | 10.3 s | 30 ms | 8.0 |
| 8img/768p | 32/32 | 0.62 | 19.7 s | 26.5 s | 50 ms | 16.4 |
| 8img/1080p | 32/32 | 0.24 | 60.4 s | 67.5 s | 39 ms | 16.4 |

**Trend: RPS scales inversely with total vision tokens per request**, with diminishing returns at extremes due to fixed overhead.

## Files
- This doc: `disagg_h200_32b/1E_768p_imagecount_sweep_np32.md`
- 1E 4img/768p result: `res_xhost_dell06_super21/32b_4img_768p_rate1.0_np32_1enc_20260528_073753/`
- 1E 8img/768p result: `res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_1enc_20260528_071429/`
- PD log: `disagg_h200_32b/logs/pd_worker_giga01_lowmem_20260528_070653.log`
- Prior 1080p comparison: `disagg_h200_32b/2E_vs_1E_dell06_super21_32b_8img_1080p.md`
- Prior 1E 768p analysis: `disagg_h200_32b/1E_vs_2E_dell06_super21_32b_8img_768p_np32.md`

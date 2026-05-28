# Qwen3-VL-8B-Instruct — Three-Way Comparison: TP=1 agg vs TP=2 agg vs Disagg

**Date:** 2026-05-23 (run completed ~03:38 UTC)
**Model:** `Qwen/Qwen3-VL-8B-Instruct` (bf16, 8B activated params, 36 layers, hidden 4096)
**Workload:** 8 × 1080p (1920×1080) random images, 128 input + 256 output tokens, np=64 per rate
**Rates swept:** 0.1, 0.25, 0.5, 1.0, 1.25 RPS
**Hardware:** H200, GPUs 4 (NUMA 2) and 5 (NUMA 2), NVLink NV18 (~478 GB/s)

This is the first run on the smaller 8B model. The hypothesis from `where_is_bottleneck.md`: with a smaller LLM, the ViT/(ViT+LLM) ratio shifts toward ViT, so disagg should perform better relatively (or even win). **Result: disagg still loses, but by a smaller absolute margin than at 32B.**

---

## Saturation summary

| Config | Rate=0.1 | Rate=0.25 | Rate=0.5 | Rate=1.0 | Rate=1.25 | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| **TP=1 agg** | 0.10 | 0.25 | 0.50 | 0.74 | **0.84** | **~0.84 RPS** |
| **TP=2 agg** | 0.10 | 0.25 | 0.50 | 0.96 | **1.11** | **~1.11 RPS** |
| **Disagg** (encoder GPU 4 + PD GPU 5) | 0.10 | 0.25 | 0.36 | 0.38 | **0.37** | **~0.38 RPS** |

**Ranking: TP=2 (1.11) > TP=1 (0.84) > Disagg (0.38).**

TP=2 wins by 1.32× over TP=1 (sub-linear scaling — 8B is small enough that comm overhead matters at TP=2). Disagg loses to TP=1 by 2.2×.

---

## Direct comparison: 8B vs 32B-FP8

| Workload (8 imgs @ 1080p, np=64, 5-rate sweep) | 32B-FP8 sat. RPS | 8B sat. RPS | 8B / 32B speedup |
|---|---:|---:|---:|
| **TP=1 agg** | 0.52 | **0.84** | 1.6× |
| **TP=2 agg** | 0.95 | **1.11** | 1.2× |
| **Disagg** | 0.23 | **0.38** | 1.65× |

**Observations:**
- TP=1 sees the biggest absolute speedup — 8B is ~3-4× smaller than 32B in compute, but H200 is partially memory-bandwidth limited even on 8B, and ViT+overhead don't scale with LLM, capping speedup at ~1.6×
- TP=2 gains less because at 8B size, the comm/sync overhead at TP=2 is a larger relative fraction
- Disagg gains the most relatively (1.65×), exactly as predicted in `where_is_bottleneck.md` — but the ratio of disagg to TP=1 stayed nearly identical (0.45 for 8B vs 0.44 for 32B)

**The hypothesis was wrong**: smaller LLM does NOT make disagg win on this stack. It just makes everything proportionally faster.

---

## Detailed CSV (all 5 rates)

### TP=1 agg
```
target_rate, actual_rps, ttft_mean_ms, tpot_mean_ms, e2e_mean_ms, peak_out_tok/s, concurrency
0.10,        0.10,       2886,         14.0,          4177,        319,             0.43
0.25,        0.25,       3614,         22.3,          5505,        315,             1.40
0.50,        0.50,       4518,         50.3,          8480,        525,             4.23
1.00,        0.74,      19510,        815.1,         51848,       1252,            38.31
1.25,        0.84,      25159,        246.1,         46117,       1190,            38.63
```

### TP=2 agg
```
target_rate, actual_rps, ttft_mean_ms, tpot_mean_ms, e2e_mean_ms, peak_out_tok/s, concurrency
0.10,        0.10,       3191,          9.9,          4142,        342,             0.42
0.25,        0.25,       3184,          5.8,          3784,        369,             0.96
0.50,        0.50,       3991,          5.1,          4580,        386,             2.30
1.00,        0.96,       9550,         21.5,         10372,        592,             9.97
1.25,        1.11,      32404,          5.5,         33249,        393,            36.98
```

### Disagg
```
target_rate, actual_rps, ttft_mean_ms, tpot_mean_ms, e2e_mean_ms, peak_out_tok/s, concurrency
0.10,        0.10,       9685,         36.1,         12311,        268,             1.25
0.25,        0.25,      12943,        216.2,         21463,        652,             5.32
0.50,        0.36,      45445,       1912.7,        109467,        975,            39.24
1.00,        0.38,      78668,       1273.8,        132062,       1042,            49.79
1.25,        0.37,      88798,       1284.3,        143219,       1042,            52.38
```

---

## Why disagg still loses (analysis)

### Disagg capacity wall around rate=0.5

Disagg handles up to rate=0.25 cleanly (0.25 RPS achieved, TTFT 12.9 s). But at rate=0.5 it can only sustain 0.36 RPS. The cliff between 0.25 and 0.5 RPS shows saturation right at ~0.4 RPS — half the TP=1 saturation point.

Comparing rate=0.5 between configs:

| Metric | TP=1 agg | TP=2 agg | Disagg |
|---|---:|---:|---:|
| Actual RPS | **0.50** | **0.50** | 0.36 |
| Mean TTFT (ms) | 4,518 | 3,991 | **45,445** |
| Mean TPOT (ms) | 50.3 | **5.1** | 1,912.7 |
| Mean E2E (ms) | 8,480 | **4,580** | 109,467 |
| Concurrency | 4.2 | 2.3 | **39.2** |

**Disagg's disasters at rate=0.5:**
- TTFT 45.4 s vs 4.5 s for TP=1 (10× worse)
- TPOT 1.9 s vs 50 ms for TP=1 (38× worse — decode is starved)
- Concurrency 39 vs 4 — requests piling up massively
- Despite having 2 GPUs, throughput is 28% lower than 1-GPU TP=1

### Why disagg has a single-GPU bottleneck

Same reason as 32B: the disagg PD worker is **TP=1**. The encoder takes care of vision (which is fast on a single GPU at 8B+ViT), then hands off embeddings to the PD which must do all LLM work on one GPU. At 8B, the LLM is fast enough that the overhead from:
- ZMQ encoder→scheduler request hand-off
- NIXL embedding transfer (cuda_ipc, ~28 MB/req)
- Scheduling/queuing in PD worker

...adds up to ~7 s per request even at low load. At higher rates, that overhead amplifies catastrophically because each request is queued behind the slow path.

### What ViT vs LLM looks like at 8B

For 8B at 8 × 1080p:
- ViT compute: ~1 s (16K visual tokens × 0.1 ms/token vision encoder ≈ 1.5 s)
- LLM prefill: ~1-2 s (16K × 7 GFLOPs/token × 1500 TFLOPs effective = 0.075 s in pure compute, but bandwidth-bound = ~1-2 s realistic)
- **ViT/(ViT+LLM) ≈ 40-60%**

So ViT IS a meaningful fraction here — but not enough to overcome disagg's hop overhead at this small LLM size.

---

## Conclusion

**On Qwen3-VL-8B-Instruct, same-host single-PD disagg still loses to TP=1 aggregate.** The smaller LLM did make ViT a meaningful fraction (~40-60% vs ~17% on 32B), and disagg did improve by 1.65× (0.23 → 0.38 RPS) — but TP=1 also improved by 1.6× and remains ahead.

**The fundamental problem**: with a single TP=1 PD worker, disagg adds hop overhead without adding LLM compute. The encoder→PD hand-off costs ~7-10 s of latency per request even at low load.

**To make disagg actually win, you'd need:**
1. **Multiple PD workers in DP** behind one encoder — encoder isn't bottlenecked, can serve 2-4 PD workers in parallel
2. **TP=2 PD** (3-GPU disagg) — gives PD same compute as TP=2 agg, but encoder offloads vision work
3. **Even smaller LLM** (Qwen3-VL-2B or 4B) — at some point ViT does dominate enough to win

For our 2-GPU same-host setup, **TP=2 agg is always the right answer** for any Qwen3-VL model size from 8B to 32B.

---

## Time spent
- TP=1 8B sweep: ~29 min (5 rates)
- TP=2 8B sweep: ~28 min (5 rates)
- Disagg 8B sweep: ~34 min (5 rates)
- Total: ~95 min unattended (compared to ~3 hr for 32B at 4K)

---

## Files

- TP=1 results: `/hongming/res7_8B/h200_agg_tp1_8b_image8_1080p_np64/test_sglang_multi_rates_1080p_20260523_020542/`
- TP=2 results: `/hongming/res7_8B/h200_agg_tp2_8b_image8_1080p_np64/test_sglang_multi_rates_1080p_20260523_023533/`
- Disagg results: `/hongming/res7_8B/h200_disagg_8b_image8_1080p_np64/test_sglang_multi_rates_1080p_20260523_030443/`
- Server scripts:
  - `/hongming/dynamo/01_cuda_sh/agg_h200_8b/start_h200_aggregate_epd_server_8b_tp1.sh`
  - `/hongming/dynamo/01_cuda_sh/agg_h200_8b/start_h200_aggregate_epd_server_8b_tp2.sh`
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_8b/start_disagg_h200_8b_combined.sh`
- Bench script: `/hongming/dynamo/test_sglang_8b_1080p_np64_over_rates.sh`
- Orchestrator: `/hongming/dynamo/run_8b_three_way.sh`
- Companion analysis: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/where_is_bottleneck.md`
- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_8b/8b_three_way.md`

# Comprehensive 5-way comparison written to `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/comparison_5way_35b.md`

## Summary of findings

**Saturation RPS table:**

| Workload | agg_TP1 | agg_TP2 | B70_1E | B70_4E | dell06_1E | best |
|---|---:|---:|---:|---:|---:|---|
| 4img/768p | **3.01** | 2.73 | 0.44 | 1.55 | >2.49 | agg_TP1 |
| 8img/768p | 2.09 | 2.03 | 0.23 | 0.83 | **2.52** | dell06_1E |
| 8img/1080p | **1.14** | 1.04 | 0.038 | 0.147 | 0.85 | agg_TP1 |

## Key findings

1. **Workload-dependent winner.** No single topology dominates. agg_TP1 wins for the smallest (4img/768p) and largest (8img/1080p) workloads. dell06_1E wins for the middle (8img/768p) — a result that surprised me.

2. **agg_TP2 is consistently slower than agg_TP1 for 35B.** 10-15% gap across all workloads. Caused by MoE all-to-all overhead (256 experts × top-8) + hybrid attention asymmetry. This **contradicts the 32B-FP8 finding** where TP=2 always won. The 35B docs that previously projected "TP=2 will win" turned out to be wrong.

3. **dell06_1E TTFT beats agg_TP1 TTFT for 768p workloads** at low-mid rate (24% faster at 8img/768p rate=1.0). Cross-host disagg with H200 encoder is **better than same-host agg for serving latency** at certain workload sizes — first time this has been measured for 35B.

4. **B70 1E is unusable at 1080p** (TTFT 581-844 s). B70 4E is "usable but slow" (~200 s TTFT at sat). Switching to dell06 H200 encoder gives a 22× throughput improvement at 8img/1080p.

5. **Cost efficiency**: agg_TP1 has 2-3× better RPS per GPU than dell06_1E. The case for cross-host disagg is operational (independent scaling, hardware diversity), not raw efficiency.

6. **TPOT in B70 setups is misleading.** Lowest measured TPOTs (1-2 ms) come from B70 4E at saturation — but only because requests queue for 200 s and decode runs in massive batches. Real serving TPOT is more like 5-15 ms across all configs at sub-saturation.

## Recommendation matrix

| Goal | Choice |
|---|---|
| Maximum RPS / GPU | agg_TP1 |
| Lowest TTFT (4img/768p, 8img/768p) | dell06_1E |
| Lowest TTFT (8img/1080p) | agg_TP1 |
| Operational flexibility | dell06_1E |
| Avoid same-device contention | dell06_1E |
| Avoid B70 entirely | dell06_1E or agg_TP1 |
| Limit to 1 H200 | agg_TP1 |
| Have B70 only | B70_4E (never 1E) |

Stack still up. Anything else, or shall we tear down?

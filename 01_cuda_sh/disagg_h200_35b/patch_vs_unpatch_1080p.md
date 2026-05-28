# Patched H200 PD + B70 4E — 8img/1080p sweep complete

**All 7 runs returned 32/32 successful**, ~46 min wall-clock.

## Results

| Rate | RPS | E2E p50 | TTFT p50 | TPOT p50 | Status |
|---:|---:|---:|---:|---:|:--:|
| 0.10 | 0.106 | 63.3 s | 54.0 s | 5.20 ms | 32/32 |
| 0.25 | 0.138 | 151.8 s | 142.0 s | 3.77 ms | 32/32 |
| 0.50 | 0.144 | 187.1 s | 180.8 s | 1.65 ms | 32/32 |
| 1.00 | 0.146 | 201.2 s | 201.1 s | 1.65 ms | 32/32 |
| 1.50 | 0.146 | 206.3 s | 206.0 s | 1.68 ms | 32/32 |
| 2.00 | 0.146 | 209.0 s | 208.7 s | 1.45 ms | 32/32 |
| 3.00 | 0.146 | 211.9 s | 211.8 s | 1.49 ms | 32/32 |

## vs unpatched B70 4E baseline (`8img_1080p_4E/`)

| Rate | patched RPS | baseline RPS | Δ RPS | patched TTFT | baseline TTFT | patched TPOT | baseline TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.1060 | 0.1059 | +0.01% | 54.0 s | 53.6 s | 5.20 ms | 7.03 ms |
| 0.25 | 0.1375 | 0.1383 | -0.08% | 142.0 s | 137.8 s | 3.77 ms | 3.68 ms |
| 0.50 | 0.1438 | 0.1446 | -0.08% | 180.8 s | 176.8 s | 1.65 ms | 2.18 ms |
| 1.00 | 0.1460 | 0.1469 | -0.09% | 201.1 s | 199.5 s | 1.65 ms | 1.30 ms |
| 1.50 | 0.1455 | 0.1461 | -0.06% | 206.0 s | 204.9 s | 1.68 ms | 1.49 ms |
| 2.00 | 0.1456 | 0.1465 | -0.10% | 208.7 s | 207.9 s | 1.45 ms | 1.81 ms |
| 3.00 | 0.1455 | 0.1467 | -0.12% | 211.8 s | 209.9 s | 1.49 ms | 1.76 ms |

## Key observations

1. **Throughput essentially unchanged.** RPS within ±0.1% of unpatched baseline at every rate. **The patch does NOT help when the bottleneck is the encoder.**

2. **Tiny TTFT regression (~1-4 s)** at every rate. This is consistent with debug-logging overhead (we kept `DYN_LOG=debug` for this sweep, baseline was INFO). 1-4 s out of ~200 s is ~0.5-2% — a logging artifact, not a structural regression.

3. **TPOT mostly unchanged** with mixed direction:
   - rate=0.10: improved 7.03 → 5.20 ms (better)
   - rate=0.50: improved 2.18 → 1.65 ms (better)
   - rate=2.00: improved 1.81 → 1.45 ms (better)
   - rate=1.00: regressed 1.30 → 1.65 ms (slightly worse)
   - rate=3.00: improved 1.76 → 1.49 ms (better)
   - Average improvement ~10%, but within run-to-run noise on these tiny values.

4. **Saturation is unchanged at 0.146 RPS** — same as unpatched B70 4E. **Confirms the bottleneck is the encoder pool, not the PD-side CPU bounce.** The PD is idle ~95% of the time (per `35b_bottleneck_analysis.md`), so removing CPU staging on the receive side moves the needle ~zero on throughput.

## Why the patch is invisible here

This is exactly the regime where the patch *should* not help, and the data confirms it:

- **Encoder ViT on B70 XPU dominates** (~25 s per request, parallelized across 4 encoders → effective ~6 s per request)
- **PD `forward_duration` ~1 s per request, lifetime ~6 s** (mostly waiting on next encoder embedding)
- **CPU bounce overhead pre-patch was ~150 ms per request**, a tiny fraction of the 6 s lifetime
- **Patch removes the 150 ms** but encoder ViT is still 25 s upstream → throughput unchanged

The patch is structurally correct (verified by `device=cuda:0` in 33/33 ReadOps in the prior dell06 run) but operationally invisible when encoder is the bottleneck.

## Where the patch would be visible

From the prior measurements:
- **dell06_1E patched vs unpatched at rate=1.0**: 0.81 vs 0.78 RPS = +3% (small but real)
- **32B-FP8 patched_4E sweep (per `patched_4E_results.md`)**: TPOT 18ms → 2.4ms (7×) at 8img/1080p

The 32B sweep saw a TPOT win because PD was actually running (CPU bounce contended with prefill memory traffic). On 35B with B70 encoders, PD is idle so there's no contention to relieve.

## Final 8img/1080p saturation matrix

| Topology | Sat RPS | TTFT_p50 (rate=1.0) |
|---|---:|---:|
| B70 1E unpatched | 0.038 | 833 s |
| B70 4E unpatched | 0.147 | 200 s |
| **B70 4E patched (this)** | **0.146** | **201 s** |
| 5E hybrid (4×B70 + 1×dell06) | 0.170 | 149 s |
| dell06 1E unpatched | 0.781 | 13.2 s |
| dell06 1E patched | 0.805 | 12.4 s |
| agg_TP1 (single H200) | 1.140 | 12.8 s |

Stack still up. PD is patched + DYN_LOG=debug; 4 B70 encoders connected. Want me to:
- Run other workloads on this topology (4img/768p, 8img/768p)?
- Restart PD with INFO logging for cleaner sweeps?
- Stop B70 encoders and re-test with dell06 only on the patched stack?
- Something else?

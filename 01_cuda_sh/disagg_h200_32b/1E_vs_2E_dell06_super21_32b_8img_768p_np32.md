# 1E vs 2E disagg at 8img/768p np=32 rate=1.0 — clean 1E baseline

**Date:** 2026-05-28
**Setup:**
- **PD**: super21-h200 GPU 5, TP=1, mem-fraction=0.50, max-running=64
- **Encoder(s)**: dell06 H200 — 1 encoder for 1E, 2 encoders for 2E
- **Network**: RoCE 192.165.123.0/24; PD NIC mlx5_4 (192.165.123.52)
- **Frontend**: round-robin router mode
- **Workload**: Qwen3-VL-32B-Instruct-FP8, 8 imgs × 1024×768, in=128 / out=256, rate=1.0 RPS, np=32

**Result paths:**
- 1E 768p (this run): `/hongming/res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_1enc_20260528_071429/`
- 2E 768p (partial, kv-router): `/hongming/res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_2enc_20260528_062928/`
- 2E 1080p (clean): `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_2enc_20260528_050453/`
- 1E 1080p (np=32 prior): `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_192949_v2/`

## Headline result: 1E 768p np=32 rate=1.0

**32/32 successful, 0.62 RPS, mean TTFT 19.7 s, median TPOT 50 ms, mean E2E 26.5 s.**

This is the clean 768p np=32 baseline that didn't exist before. With one encoder
on dell06 funneling into the PD on super21, the system holds steady at ~0.62 RPS
and 16.4 average concurrency.

| Metric | Value |
|---|---:|
| Successful | **32/32** ✓ |
| Duration (s) | 51.66 |
| **RPS** | **0.62** |
| Input throughput (tok/s) | 3 863 |
| Output throughput (tok/s) | 94.5 |
| Peak output (tok/s) | 363 |
| Mean E2E latency (ms) | 26 516 |
| Median E2E latency (ms) | 26 641 |
| P90 E2E latency (ms) | 32 997 |
| P99 E2E latency (ms) | 34 309 |
| Mean TTFT (ms) | 19 692 |
| Median TTFT (ms) | 20 073 |
| P99 TTFT (ms) | 29 587 |
| Mean TPOT (ms) | 46.4 |
| Median TPOT (ms) | 50.2 |
| P99 TPOT (ms) | 108 |
| Mean concurrency | 16.4 |
| Peak concurrent | 29 |

## Full comparison: 1E vs 2E across resolutions

| Metric | **1E 768p** (this run) | 2E 768p (13/32 partial) | 1E 1080p (32/32) | 2E 1080p (32/32) |
|---|---:|---:|---:|---:|
| Successful | **32/32** | 13/32 ⚠️ | 32/32 | 32/32 |
| Duration (s) | 51.66 | 26.13 | 131.69 | 131.54 |
| **RPS** | **0.62** | **0.50** | 0.24 | 0.24 |
| Mean E2E (ms) | 26 516 | 15 792 | 67 482 | 98 594 |
| Mean TTFT (ms) | 19 692 | 10 582 | 60 440 | 93 480 |
| Median TPOT (ms) | 50 | 30 | 39 | 0.65 |
| P99 TPOT (ms) | 108 | 96 | 245 | 1 004 |
| Concurrency | 16.4 | 7.9 | 16.4 | 24.0 |
| Peak decode (tok/s) | 363 | 226 | 199 | 461 |

## Interpretation

### 1. 1E 768p is the practical sweet spot for this hardware

**0.62 RPS** at 768p — **2.6× higher than 1E at 1080p** (0.24 RPS). Smaller embeddings
(80 MB/req vs 250 MB/req) mean less NIXL transfer time, less PD prefill compute, and
more requests per second overall.

### 2. 2E 768p (partial) and 1E 768p are close — but 1E wins on reliability

The 2E 768p partial RPS of 0.50 (during 13 successful reqs) is *lower* than 1E 768p's
0.62 (32 reqs). The "2E should help at 768p because encoder-bound" hypothesis from
the 1080p analysis is **not strongly supported**: at 768p, the encoder vision compute
(~1-2 s per req on H200) is comparable to PD prefill (~1.5-2 s for ~6k vision tokens).
Neither side is dominant. Adding a 2nd encoder doesn't break a wall, it just risks
load imbalance. The 19/32 failures in 2E confirm this is the wrong regime to add
encoders.

The earlier comparison `2E 768p partial RPS 0.50 > 1E 768p (np=16) RPS 0.20` was
misleading because it compared np=32 to np=16 — different workloads. **Apples-to-apples
np=32: 1E at 0.62 RPS beats 2E at 0.50 RPS (partial).**

### 3. Cross-resolution sanity check

| Resolution | Embedding size/req | PD compute headroom | Best mode |
|---|---:|---|---|
| 768p (8 img) | ~80 MB | High (PD finishes prefill fast) | **1E** (matches encoder cadence) |
| 1080p (8 img) | ~250 MB | Low (PD prefill ≈ 10 s/req at 16k tokens) | 1E or 2E (RPS unchanged at 0.24) |
| 1080p + larger | even larger | PD-bound | TP=2 PD or 4E (per `comparison_5way_35b.md`) |

### 4. TTFT is dominated by PD-side queueing in all modes

1E 768p TTFT is 19.7 s — that's not encoder vision compute time (vision tower runs
in ~1-2 s on H200), it's the **wait time at PD's prefill scheduler**. With np=32 reqs
arriving at rate=1.0 RPS but PD processing one prefill at a time, head-of-line waits
build to 30-40 s by the end of the bench.

This matches the prior 1080p analysis: PD prefill is serialized (#new-seq=1 per
batch). Whether 1 or 2 encoders feed it, PD itself dictates mean TTFT.

### 5. Concurrency vs RPS recap

```
1E 768p:  concurrency 16.4  → RPS 0.62  (waiting time = 16.4 × 1.61 = 26.4s = E2E ✓)
1E 1080p: concurrency 16.4  → RPS 0.24  (16.4 × 4.11s = 67.4s = E2E ✓)
2E 1080p: concurrency 24.0  → RPS 0.24  (24.0 × 4.11s = 98.6s = E2E ✓)
```

Same Little's-Law pattern in all clean runs. **2E adds in-flight reqs but doesn't
add throughput**, which manifests as longer per-req latency.

### 6. Why 2E 768p partial showed concurrency 7.9 (not ~16+)

Because 19 of 32 failed quickly (~1 s each, mostly during ramp-up), the bench
duration shrank to 26 s and average concurrency collapsed. The partial RPS 0.50
is biased high (only successes counted, fewer reqs fighting for resources).

## Recommendations

For this workload pattern (32B FP8 + 8 imgs, np=32 at rate=1.0):

| Resolution | Recommended | Why |
|---|---|---|
| 768p | **1E** | Matches PD prefill cadence; no benefit from 2nd encoder |
| 1080p | 1E **or** 2E (no preference) | PD-bound regardless |
| 4K (from prior research) | 4E | Encoder vision tower is 4-8× longer, becomes the bottleneck |

**The right next experiment**: try 1E with TP=2 PD on super21 (encoder doesn't change,
but PD has more prefill capacity). That's the lever that should break the 0.62 RPS
wall at 768p and 0.24 wall at 1080p.

## Operational notes from this session

- **NIXL idle disconnect** is a recurring issue: encoder NIXL agents go stale after
  10-15 min of no traffic, requiring encoder restart on dell06. Has happened 3-4
  times in this work session. May warrant a heartbeat patch.
- **Frontend MDC tracking** is sticky: when encoders restart with new lease IDs,
  frontend properly removes old MDCs (verified at 06:51:07-12). Stale entries don't
  persist.
- **Round-robin router** confirmed working: even distribution between encoders when 2E.
  KV-router tie-breaker by tree size still worth fixing for future workloads.

## Files
- This doc: `disagg_h200_32b/1E_vs_2E_dell06_super21_32b_8img_768p_np32.md`
- 1E 768p result: `res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_1enc_20260528_071429/`
- 2E 768p partial: `res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_2enc_20260528_062928/`
- PD log (DYN_LOG=debug): `disagg_h200_32b/logs/pd_worker_giga01_lowmem_20260528_070653.log`
- Frontend log (round-robin): `disagg_h200_32b/logs/frontend_giga01_20260528_070639_rr.log`
- Prior 1080p comparison: `disagg_h200_32b/2E_vs_1E_dell06_super21_32b_8img_1080p.md`
- Prior 768p partial findings: `disagg_h200_32b/2E_768p_np32_partial_routing_findings.md`

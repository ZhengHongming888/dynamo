# 2E vs 1E disagg comparison: 32B FP8 / 8img×1080p / np=32 rate=1.0

**Date:** 2026-05-28
**Setup:**
- **PD**: super21-h200 GPU 5, TP=1, **mem-fraction=0.50** (lowered from 0.65 to leave headroom for shared-host GPU intrusions), max-running=64
- **Encoders**: dell06 H200 GPUs 0+1 (172.26.46.162) — **2 instances**
- **Network**: RoCE 192.165.123.0/24; PD NIC mlx5_4 (192.165.123.52); encoder NICs 192.165.123.25:20098 / :20099
- **Patches**: `h200_cuda_nixl.patch` applied on PD side
- **Workload**: Qwen3-VL-32B-Instruct-FP8, 8 imgs × 1920×1080, in=128 / out=256, rate=1.0 RPS, np=32

**Bench result paths:**
- 2E: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_2enc_20260528_050453/`
- 1E baseline: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_192949_v2/`

## Headline result

**More encoders did NOT improve throughput.** Same 0.24 RPS, same 131.5 s duration.
What 2E did change: it shifted the bottleneck from **encoder-paced** to **PD prefill-bound**,
making per-request latency **worse**, but raising **peak parallelism** on the PD.

```
                          1E         2E       Δ
RPS                       0.24       0.24      0%       ← bottleneck unchanged: PD
mean E2E (ms)            67 482     98 594    +46% slower per request
mean TTFT (ms)           60 440     93 480    +55% slower first token
median TPOT (ms)            39          1     -98%       ← decode much faster (cuda graphs hit)
peak decode (tok/s)         199        461    +132%      ← 2× peak parallelism
mean concurrency           16.4       24.0    +46%       ← more in flight at PD
```

## Per-metric breakdown

| Metric | 1E | 2E | Δ |
|---|---:|---:|---|
| Successful | 32/32 | 32/32 | — |
| Duration (s) | 131.69 | 131.54 | same |
| Request throughput (RPS) | 0.24 | 0.24 | **same** |
| Input throughput (tok/s) | 3 988 | 3 993 | same |
| Output throughput (tok/s) | 37.1 | 37.1 | same |
| **Peak output (tok/s)** | 199 | **461** | **+132%** |
| Mean E2E latency (ms) | 67 482 | 98 594 | +46% slower |
| Median E2E latency (ms) | 64 893 | 102 799 | +58% slower |
| P90 E2E latency (ms) | 104 100 | 110 117 | +6% |
| P99 E2E latency (ms) | 105 029 | 112 595 | +7% |
| Mean TTFT (ms) | 60 440 | 93 480 | +55% slower |
| Median TTFT (ms) | 58 237 | 98 252 | +69% slower |
| P99 TTFT (ms) | 101 780 | 109 286 | +7% |
| Mean TPOT (ms) | 42.6 | 85.6 | +101% |
| **Median TPOT (ms)** | **39.2** | **0.65** | **-98%** |
| P99 TPOT (ms) | 244.8 | 1 004.3 | +310% |
| Mean concurrency | 16.4 | **24.0** | +46% |
| Max concurrent reqs | 32 | 31 | — |

## What changed and why

### 1. RPS unchanged → encoders were never the bottleneck

In 1E, 1 encoder produced embeddings at one rate; in 2E, two encoders produce at 2× that rate.
Yet end-to-end RPS stayed at 0.24 — meaning the encoders weren't gating throughput in 1E.
The bottleneck was always downstream on PD: prefill compute + decode + KV space.

### 2. Peak decode tok/s doubled (199 → 461)

Two encoders feed embeddings to PD ~2× faster, so PD's queue depth goes up — at peak we now
have ~12 reqs simultaneously in decode (vs ~6 in 1E), giving 2× higher peak output rate.
**Median TPOT collapsed from 39 ms → 0.65 ms** because the bigger running batch hits cuda
graphs aggressively (tokens stream out together). This is purely a batching effect, not
NIXL/handoff improvement.

### 3. TTFT got *worse* (+55%)

With 2 encoders shoveling embeddings concurrently, more requests queue up at PD's prefill
stage. Each request now waits behind ~12 others (vs ~6) before its prefill executes:

```
PD log running-req histogram during 2E bench (180 s window, 57 prefill events):
  0:  ###########  (11)   ← idle gaps between waves
  1:  ########   (8)
  2:  #######    (7)
  3:  #######    (7)
  4:  ########   (8)
  5:  ####       (4)
  6:  ###        (3)
  7:  ##         (2)
  8:  #          (1)
  9:  #          (1)
 10:  ##         (2)
 11:  ##         (2)
 12:  #          (1)   ← peak running concurrency
```

Mean #running-req = 3.5; max = 12. Compare to 1E's mean #running-req of ~1.5, max 6
(`time_breakdown_dell06_super21_32b_8img_1080p.md`). Under a serialized prefill scheduler
(`#new-seq=1` per batch — confirmed in both modes), more running reqs = longer queue ahead
of each new arrival = larger TTFT.

### 4. Why mean TPOT got worse but median improved

The mean TPOT is dragged by long tail when a chunked prefill of one big request stalls
the decode batch (we see P99 TPOT 1004 ms in 2E vs 245 ms in 1E). But the steady-state
median is dominated by cuda-graph-batched decode, which is faster with bigger running batches:
median TPOT 0.65 ms in 2E vs 39 ms in 1E. **2E has more outliers but a faster steady state.**

## Encoder feeding cadence (the heart of the matter)

In 1E we observed PD got a fresh embedding once every ~1.4 s (1 encoder × 1.4 s/req
end-to-end). With 2E, two encoders independently feed PD, so the inter-arrival time at PD
halves to ~0.7 s. PD prefill compute for an 8img/1080p request takes ~10-12 s on this
hardware — so prefills queue up regardless of feeding rate.

The encoder is fundamentally fast enough that 1E already saturates PD's prefill capacity
for this workload. Adding a second encoder just raises the queue-depth water level without
changing the through-rate.

## Implication: where would 2E actually help?

2E is a useful lever **only when the encoder is the bottleneck**. Conditions that move
the bottleneck onto the encoder side:

1. **Lower-resolution / fewer images** → PD prefill is short, encoder vision tower dominates
2. **Larger PD model with batched-prefill capacity headroom** (e.g., bigger TP) → PD can
   eat reqs faster than 1 encoder can feed
3. **Higher input rates with smaller per-request prefill cost** (text-heavy or tiny images)
4. **Encoder running on slower GPU** → already encoder-bound at 1E

For 32B FP8 + 8img/1080p on H200 TP=1, the PD is the bottleneck in 1E and stays the
bottleneck in 2E. The data confirm: same RPS, deeper queue, worse latency.

## Mem-fraction headroom check

PD launched with `mem-fraction-static=0.50` (vs 0.65 in 1E baseline):

```
max_total_num_tokens =  296,864 (vs 467,072 at 0.65) — still way more than np=32 needs
available_gpu_mem    =  69.23 GB
GPU 5 used after bench = 125 GB / 143 GB total (18 GB free)
```

The lower fraction did NOT cause throughput regression for this workload (np=32 is well
under the KV-cache budget). The trade-off was free: same throughput, ~30 GB more headroom
to survive any external GPU intrusion (the same kind that crashed the prior 2E attempt).

## Conclusion

| | 1E | 2E |
|---|---|---|
| RPS | 0.24 | 0.24 |
| Per-request latency | 67.5 s mean | 98.6 s mean (+46%) |
| Peak decode tok/s | 199 | 461 (+132%) |
| Decision | Better for **single-shot latency** | Better for **bursty peak parallelism** |

**For sustained throughput on this workload, 2E adds no value.** PD prefill is the wall.
The right next step to break that wall is **PD-side parallelism** (TP=2 or PP), not more
encoders.

## Files
- This doc: `disagg_h200_32b/2E_vs_1E_dell06_super21_32b_8img_1080p.md`
- 2E result JSON: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_2enc_20260528_050453/benchmark_output.json`
- 2E bench stdout: same dir, `results.txt`
- 2E PD log (DYN_LOG=debug): `disagg_h200_32b/logs/pd_worker_giga01_lowmem_20260528_045112.log`
- 1E baseline result JSON: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_192949_v2/benchmark_output.json`
- 1E PD-side breakdown: `disagg_h200_32b/time_breakdown_dell06_super21_32b_8img_1080p.md`
- 1E + encoder breakdown: `disagg_h200_32b/time_breakdown_dell06_super21_32b_8img_1080p_v2_with_encoder.md`

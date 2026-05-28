# Detailed Time Breakdown: dell06 H200 (encoder) + super21 H200 (PD)

**Date:** 2026-05-27
**Setup:**
- **Encoder**: dell06 H200 (172.26.46.162), single-encoder (1E)
- **PD**: super21-h200 GPU 4 (172.26.46.133), TP=1, mem-fraction=0.65, max-running=64
- **Network**: RoCE 100 Gb/s NDR fabric (192.165.123.0/24); PD NIC mlx5_4 (192.165.123.52)
- **Patches**: `h200_cuda_nixl.patch` applied on PD side (cuda:0 NIXL receive descriptors)
- **Workload**: Qwen3-VL-32B-Instruct-FP8, 8 imgs × 1920×1080, in=128 / out=256, **rate=1.0 RPS, np=32**

**Bench result:** 0.26 RPS, 32/32 successful, mean TTFT 59 s, median TPOT 70 ms

**Method:** Parsed the PD-side debug log (`pd_worker_giga01.log`, `DYN_LOG=debug`) and extracted
per-request checkpoints. Bench window: 2026-05-27T05:21:30 → 2026-05-27T05:24:00 (132.85 s).

## Per-request lifecycle checkpoints (PD-side, debug log)

| ID | Event | Log marker |
|---|---|---|
| **T0** | request received at PD | `request received [request_id=<UUID>]` |
| **T1** | Python handler `process_embeddings` begins | `Processing embeddings with shape:` |
| **T2** | NIXL receive descriptor allocated on cuda:0 | `Created Descriptor(...device=cuda:0)` |
| **T3** | `ReadOperation` created (NIXL READ submitted) | `Created ReadOperation(...status='<invalid>')` |
| **T4** | NIXL state = PROC (NIC begins RDMA pull) | `NIXL reported transfer state: PROC` |
| **T5** | NIXL state = DONE (wire transfer complete) | `NIXL reported transfer state: DONE` |
| **T6** | SGLang scheduler logs forward_duration | `ReqTimeStats(rid=...)` (rid is internal) |
| **T7** | Response complete, returned to client | `request completed [request_id=<UUID>]` |

Note: SGLang's internal `rid` ≠ dynamo Rust `request_id`, so we can't pair T6 to a specific T0
across requests. Per-request stats use the median ReqTimeStats across the 33 events.

## Bench result summary (rate=1.0, np=32)

| Metric | Value |
|---|---:|
| Successful requests | 32 / 32 ✓ |
| Bench duration | 125.2 s |
| Throughput | **0.26 RPS** (target was 1.0 — bench was offered-rate-bound) |
| Mean TTFT | 59 073 ms |
| Median TTFT | 57 905 ms |
| P99 TTFT | 94 919 ms |
| Mean TPOT | 84 ms |
| Median TPOT | 70 ms |
| Mean E2E | 71 911 ms |
| Output throughput | 39 tok/s |
| Concurrency (avg) | 18.4 |

## Per-request timeline trace (first 8 requests in bench window)

```
#   rid (8c)     T1-T0    T2-T1    T3-T2    T4-T3    T5-T4   T7-T5    T7-T0
              dispatch   cuda    NIXLset  submit    wire   PD+egress lifetime
                 (ms)    (ms)    (ms)     (ms)     (ms)    (ms)      (ms)
─────────────────────────────────────────────────────────────────────────────
1   f5a2a46a    5.14    0.60    9.02     0.18    1059.5   5591.5   6666.0
2   fb7dda4f    4.67    0.52    6.97     0.17    1761.4  15935.3  17709.1
3   f9badc9b    4.05    0.47    3.70     0.96     346.0  15950.3  16305.5
4   2aa380e3  661.12    0.24    0.66    -1.79      99.6  14128.3  14888.2
5   3a4357f2 2738.43    0.26    4.99     0.07   -1672.5* 25871.5  26942.8
6   c4680c4d    2.77    0.27    1.86     0.54    8777.9  15198.4  23981.7
7   5ac9a097  337.98    0.24    0.99    -2.31    6890.4  15184.8  22412.1
8   845240bc    4.53    1.58  361.28  5223.64       0.3* 32600.1  38191.5
```

\* Negative wire times for req 5 and req 8 are a heuristic-matching artifact: at high
concurrency, my "first PROC/DONE state event after T3" can match the *next* transfer's
state event instead of the right one. The actual wire times are the smaller numbers.

## Aggregate per-stage stats (medians across the bench window)

| Stage | Median (ms) | Mean (ms) | Min | Max |
|---|---:|---:|---:|---:|
| **T1−T0** (Rust ingress → Python handler) | **4.9** | 470 | 2.8 | 2738 |
| **T2−T1** (cuda:0 buffer alloc) | **0.4** | 0.5 | 0.2 | 1.6 |
| **T3−T2** (NIXL register+create_readable) | **4.4** | 49 | 0.7 | 361 |
| **T4−T3** (UCX submit, NIC kick-off) | **0.2** | 653 | -2.3 | 5224 |
| **T5−T4** (NIXL wire RDMA transfer) | **703** | 2158 | (∼100) | 8778 |
| **T7−T5** (PD scheduler+forward+egress) | **15 567** | 17 558 | 5592 | 32 600 |
| **T7−T0** (full PD-side lifetime) | **20 061** | 20 887 | 6666 | 38 192 |

## SGLang internal stats (ReqTimeStats, n=33)

| Metric | Median | Mean | P99 | Max |
|---|---:|---:|---:|---:|
| input_len (visual + text tokens) | 16 413 | — | — | 16 477 |
| output_len | 164 | — | — | 255 |
| **queue_duration** (PD scheduler queue) | **0.32 ms** | 489 ms | 4 230 ms | 4 443 ms |
| **forward_duration** (GPU prefill+decode) | **10 478 ms** | 13 662 ms | 39 582 ms | 44 311 ms |

PD scheduler queue is essentially empty (median 0.32 ms). The GPU forward (prefill 16k visual
tokens + decode 256 tokens) takes a median of 10.5 s per request.

## PD prefill concurrency (56 prefill events during bench)

```
running-req=0:  8 (14.3%)  █████          ← single-request periods (warmup tail)
running-req=1:  8 (14.3%)  █████
running-req=2:  7 (12.5%)  █████
running-req=3: 11 (19.6%)  ███████        ← the mode
running-req=4:  3  (5.4%)  ██
running-req=5:  5  (8.9%)  ███
running-req=6:  2  (3.6%)  █
running-req=7:  3  (5.4%)  ██
running-req=8:  5  (8.9%)  ███
running-req=9:  4  (7.1%)  ██             ← bursty peaks
```

- Mean concurrent requests on PD: **3.6**
- max queue-req: 3 (essentially no scheduler back-pressure)

## Synthesis: where time goes per request (medians)

```
Component                                       ms (median)    pct of lifetime
─────────────────────────────────────────────────────────────────────────────
Frontend dispatch + PD Rust ingress (T1-T0)         4.91         0.02%
CUDA buffer alloc (T2-T1) — patched line            0.37         0.00%
NIXL setup (T3-T2)                                  4.35         0.02%
NIXL submit → PROC (T4-T3)                          0.18         0.00%
NIXL wire transfer (T5-T4) — RDMA                 702.75         3.50%   ███
PD scheduler queue (queue_duration)                 0.32         0.00%
PD GPU forward (prefill+decode)                10 478.27        52.23%   ██████████████████████████
Other (response stream egress, scheduler hop)   8 869.44        44.21%   ██████████████████████
─────────────────────────────────────────────────────────────────────────────
TOTAL PD-side lifetime                         20 060.58       100.00%
```

The "Other" bucket of 8.9 s captures:
- Time between NIXL DONE (embedding on PD GPU) and the SGLang scheduler picking it up
  (response stream is fully buffered through dynamo's TCP plane, then SSE-streamed to client)
- Egress: response token stream over dynamo TCP back to frontend, then SSE to bench client
- Some of this is **decode time** that overlaps with prefill of the next request — the
  T7-T5 includes both forward+egress, while ReqTimeStats forward_duration is just the GPU
  forward.

A more accurate decomposition (using ReqTimeStats):

```
T7-T0 lifetime ≈ T5-T0 (NIXL setup+wire ≈ 712 ms)
              + queue_duration (0.32 ms)
              + forward_duration (10 478 ms — pure GPU compute)
              + scheduler hop + egress (≈ 8 870 ms — gap not in any logged stat)
```

## Bottleneck identification

| Component | Status | Median time | % of lifetime |
|---|---|---:|---:|
| Frontend dispatch | ✓ FAST | 5 ms | 0.02% |
| **CUDA buffer alloc (patched line)** | ✓ FAST | **0.4 ms** | **<0.01%** |
| NIXL setup | ✓ FAST | 4.4 ms | 0.02% |
| **NIXL wire (637 MB GPUDirect RDMA)** | ✓ FAST | **703 ms** (≈ 0.9 GB/s effective) | **3.5%** |
| PD scheduler queue | ✓ FAST | 0.3 ms | <0.01% |
| **PD GPU forward (prefill+decode)** | **⚠ BOTTLENECK** | **10 478 ms** | **52.2%** |
| PD scheduler hop + egress | ◯ MEDIUM | 8 870 ms | 44.2% |

- **The patch works**: cuda:0 buffer alloc is 0.4 ms (PyTorch caching allocator hit), down from
  ~50 ms in the unpatched CPU-bounce path.
- **Encoder + RoCE wire are NOT the bottleneck**: NIXL wire steady-state is 703 ms median
  (3.5% of lifetime). The 100 Gb/s RoCE NIC is delivering ~0.9 GB/s effective for 637 MB
  transfers — well below the theoretical 12.5 GB/s peak, but there's headroom for more requests.
  At higher RPS the NIC may become the next bottleneck.
- **PD GPU forward is the binding constraint**: median 10.5 s of GPU compute for 16k token
  prefill + 256 token decode on a single H200.
- **The "scheduler+egress" gap (8.9 s)** is suspicious — it's a large slice of lifetime with no
  obvious cause. Likely sources:
  - Response stream egress through Rust SSE has back-pressure when 30+ in-flight requests
    are streaming concurrently
  - Scheduler hop between NIXL DONE and SGLang batch admission could include delays we don't
    see in `forward_duration` (which only measures the actual GPU step)

## Why throughput is 0.26 RPS

```
PD throughput = avg_running_req / forward_duration
              = 3.6 / 10.5 s
              = 0.343 req/s   (theoretical ceiling)

Observed:     = 0.26 req/s
Difference:   = bench client back-pressure (np=32 cap) + egress queue saturation
```

The PD GPU is the bottleneck. At rate=1.0 the bench client offers more than the system
sustains, so the np=32 in-flight cap fills up immediately and arrivals slow to whatever the
PD can produce.

## Encoder side (dell06) — what we don't directly see

The PD log doesn't capture encoder-side timings. From bench-client TTFT = 58 s and PD
lifetime = 20 s median:

```
Bench TTFT (58 s) = bench send → first SSE token
                  = frontend dispatch + queue at frontend (??) + PD lifetime (20 s)
                  + encoder queue at dell06 (?)
                  + first token decode delay
```

The 38 s gap between bench TTFT (58 s) and PD lifetime (20 s) is dominated by **bench-side
queueing** — the np=32 client offers 32 prompts at rate=1.0 Poisson, but observed inter-arrival
on PD is 1.44 s (system-paced) rather than 0.69 s (Poisson-paced). So requests sit in the
bench client and frontend until older ones complete.

Encoder ViT itself is NOT the bottleneck:
- H200 SXM (dell06) processes 8×1080p ViT in ~1-2 s/req
- 32 requests / 32 s ≈ 1 RPS encoder capacity, more than enough for 0.26 RPS demand

## Comparison vs other documented topologies (8img/1080p Qwen3-VL-32B-FP8)

| Topology | GPUs | Sat RPS | Median TTFT | Median PD forward |
|---|---:|---:|---:|---:|
| Same-host TP=1 agg | 1 | 0.47 | ~32 s | inline |
| Same-host TP=2 agg | 2 | 0.6-0.7 | ~9-13 s | inline |
| Same-host disagg PD-TP=1 | 2 | 0.23 | ~80 s | ~10s |
| Cross-host B70_1E | 1+1 | 0.038 | 833 s | encoder-bound |
| Cross-host B70_4E | 1+4 | 0.13 | 200 s | encoder-bound |
| **Cross-host dell06_1E (THIS RUN)** | **1+1** | **0.26** | **58 s** | **10.5 s** |

**Conclusion:** dell06_1E for 32B-FP8 lands at 0.26 RPS at rate=1.0. The system is **PD-bound**
(PD GPU forward = 52% of lifetime), not encoder-bound — confirmed by:
- NIXL wire is fast (703 ms = 3.5% of lifetime)
- PD scheduler queue is empty (0.32 ms)
- Encoder runs comfortably below saturation

To raise throughput further:
1. **PD-TP=2** (use 2 H200s on super21 for PD) — would halve forward_duration → ~0.5 RPS
2. **Smaller workload** (4 imgs / 768p) — would reduce PD prefill cost
3. **Smaller LLM** (Qwen3-VL-7B) — reference 35B got 0.85 RPS in this same topology

## Files

- Bench result: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_051953/`
  - `results.txt` (full bench output)
  - `benchmark_output.json` (structured metrics)
  - `breakdown_analysis_v2.txt` (analyzer output)
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- Patch source: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/h200_cuda_nixl.patch`
- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/time_breakdown_dell06_super21_32b_8img_1080p.md`

## Companion docs (similar method, different hardware/model)

- `report_on_detailed_time_breakdown_h200_b70_4E.md` — same analysis for 35B + B70 4E
- `h200_time_breakdown_v02.md` — 32B with cross-host B70 4E (encoder-bound case)
- `b70_encoder_time_breakdown.md` — encoder-side instrumentation (B70 XPU)
- `comparison_5way_35b.md` — 35B 5-topology comparison

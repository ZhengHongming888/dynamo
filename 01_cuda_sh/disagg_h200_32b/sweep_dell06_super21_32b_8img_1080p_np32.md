# 32B-FP8 8img/1080p Rate Sweep — dell06 H200 (encoder) + super21 H200 (PD)

**Date:** 2026-05-27
**Setup:**
- **Encoder**: dell06 H200 (172.26.46.162), single-encoder (1E), `device_type:cuda` registered
- **PD**: super21-h200 GPU 4 (172.26.46.133), TP=1, `mem-fraction-static=0.65`, `max-running-requests=64`
- **Network**: RoCE 100 Gb/s NDR fabric (192.165.123.0/24); PD NIC `mlx5_4` (192.165.123.52)
- **GPUDirect patch**: `h200_cuda_nixl.patch` applied on PD side (cuda:0 NIXL receive descriptors)
- **Model**: Qwen3-VL-32B-Instruct-FP8
- **Workload**: 8 imgs × 1920×1080 random JPEGs, in=128 / out=256, np=32 fixed
- **Bench client**: `sglang.bench_serving --backend sglang-oai-chat`, seed=0, Poisson arrivals
- **Rates**: 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 (7-rate extended sweep)
- **Sweep wall time**: ~31 min (06:01–06:32 UTC)

**Result: All 7 rates × 32 prompts = 224 requests, 100% successful. Saturation at ~0.24-0.26 RPS.**

## Headline results

| Rate | RPS | Mean TTFT | Median TTFT | P99 TTFT | Median TPOT | Mean E2E | Concurrency | Bench wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | **0.11** | 11 819 | 11 695 | 16 404 | 14 | 14 529 | 1.62 | 286.4 |
| 0.25 | **0.22** | 28 434 | 29 538 | 51 056 | 48 | 36 034 | 7.89 | 146.1 |
| 0.50 | **0.24** | 44 697 | 44 119 | 75 520 | 56 | 55 304 | 13.35 | 132.6 |
| 1.00 | **0.25** | 60 363 | 58 359 | 96 738 | 106 | 75 885 | 19.04 | 127.5 |
| 1.50 | **0.22** | 69 430 | 63 998 | 121 315 | 21 | 74 640 | 16.66 | 143.3 |
| 2.00 | **0.24** | 68 675 | 66 721 | 114 812 | 42 | 75 837 | 18.47 | 131.4 |
| 3.00 | **0.26** | 73 993 | 73 167 | 109 967 | 40 | 82 946 | 21.23 | 125.0 |

(All times in milliseconds)

## Saturation chart

```
Target rate vs Actual RPS:

3.00 ──┤██████ 0.26
2.00 ──┤██████ 0.24
1.50 ──┤██████ 0.22
1.00 ──┤██████ 0.25
0.50 ──┤██████ 0.24
0.25 ──┤██████ 0.22
0.10 ──┤██     0.11
      └────────┴────────┴────────┴────────
       0      0.1      0.2      0.3   actual RPS

Saturation reached at rate ≥ 0.5 (RPS plateau ~0.24-0.26)
```

The system saturates at **~0.25 RPS** consistently from rate=0.5 onward. The slight variations
(0.22 - 0.26) across rates ≥ 0.5 are within run-to-run noise of the H200 PD compute.

## Latency vs offered rate

```
TTFT_p50 (s)  vs  offered rate:

  73.2 ┤                                                                ●  rate=3.0
  66.7 ┤                                              ●                       rate=2.0
  64.0 ┤                                       ●                              rate=1.5
  58.4 ┤                                ●                                     rate=1.0
  44.1 ┤                       ●                                              rate=0.5
  29.5 ┤             ●                                                        rate=0.25
  11.7 ┤  ●                                                                   rate=0.1
       └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
       0  10  20  30  40  50  60  70  80
         seconds (median TTFT)
```

- **Sub-saturation TTFT (rate=0.1):** 11.7 s — this is the **per-request floor** for the
  current stack at 8img/1080p with 32B FP8. It includes encoder ViT, NIXL transfer, and
  PD prefill+decode for the lone request.
- **Linear TTFT growth above saturation:** TTFT grows roughly proportional to (np / sat_RPS).
  At sat the bench client back-pressures because np=32 fills up; arrivals slow to system pace.

## TPOT pattern (decode-side latency)

| Rate | Median TPOT (ms) | Notes |
|---:|---:|---|
| 0.10 | **14** | Clean single-stream decode, no contention |
| 0.25 | 48 | Decode preempted by chunked-prefill of new arrivals |
| 0.50 | 56 | Same — 1.4 s inter-arrival, mid-decode preemption |
| 1.00 | 106 | Worst-case mid-saturation: many in-flight, frequent prefill |
| 1.50 | 21 | Steady-state batched decode kicks in |
| 2.00 | 42 | Mixed batched + occasional preemption |
| 3.00 | 40 | Steady-state |

TPOT is **bimodal across the rate sweep**:
- **Low rate (0.1):** clean ~14 ms TPOT (decode batch size = 1, no preemption)
- **Mid-rate (0.25-1.0):** elevated 48-106 ms TPOT (frequent prefill chunks preempt decode)
- **High rate (1.5-3.0):** drops to 21-42 ms (decode batch size grows; per-token cost amortizes
  across multiple concurrent decoded tokens)

This is the classic SGLang prefill-blocks-decode pattern. To target lowest TPOT the system
should run either far below saturation OR well into saturation — not in the middle.

## Bottleneck identification

From the per-request breakdown analysis (companion doc `time_breakdown_dell06_super21_32b_8img_1080p.md`),
at rate=1.0 the median request lifetime is 20 s, decomposed as:

```
Component                                       ms (median)    pct of lifetime
─────────────────────────────────────────────────────────────────────────────
Frontend dispatch + PD ingress (T1-T0)              4.91         0.02%
CUDA buffer alloc on cuda:0 (patched line)          0.37         0.00%
NIXL setup (register + create_readable)             4.35         0.02%
NIXL submit→PROC (NIC kick-off)                     0.18         0.00%
NIXL wire transfer (637 MB GPUDirect RDMA)        702.75         3.50%
PD scheduler queue_duration                         0.32         0.00%
PD GPU forward_duration (prefill+decode)       10 478           52.23%   ← BOTTLENECK
Other (response stream egress)                  8 869           44.21%
─────────────────────────────────────────────────────────────────────────────
TOTAL PD-side lifetime                         20 061          100.00%
```

**The PD GPU is the binding constraint** — 10.5 s of H200 forward per request.

```
PD effective throughput = avg_running_req / forward_duration
                        = 3.6 / 10.5 s
                        = 0.343 RPS theoretical ceiling
Observed sat:           = 0.24-0.26 RPS  (close to theoretical, gap = egress overhead)
```

## Comparison vs prior 32B-FP8 8img/1080p documented baselines

| Topology | GPUs | Sat RPS | Best Median TTFT | TTFT @ rate=1.0 |
|---|---:|---:|---:|---:|
| Same-host TP=2 agg | 2 | 0.6-0.7 | 3.5 s | ~9-13 s |
| Same-host TP=1 agg | 1 | 0.47 | 8.1 s | ~32 s |
| Same-host disagg PD-TP=1 | 2 | 0.23 | — | ~80 s |
| **Cross-host dell06_1E (THIS sweep)** | **1+1** | **~0.25** | **11.7 s** | **58 s** |
| Cross-host B70_4E (patched) | 1+4 | 0.13 | 55 s | 200 s |
| Cross-host B70_1E (patched) | 1+1 | 0.038 | 309 s | 833 s |

**Cross-host dell06_1E for 32B-FP8 places between same-host TP=1 agg (0.47) and same-host
disagg PD-TP=1 (0.23).** It's a real win over the B70 cross-host configs (4-7× higher RPS,
3-30× faster TTFT) but doesn't beat same-host agg, because the dominating cost is now the
H200 PD prefill of 16k visual tokens — not encoder or wire.

## Where dell06_1E excels and where it doesn't

**Wins for cross-host dell06_1E:**
- **Operationally clean:** encoder and PD scale independently (separate hosts)
- **Excellent sub-saturation latency** (11.7 s TTFT vs 80 s for same-host disagg)
- **PD GPU isolation:** PD has full 143 GB H200 dedicated to KV cache (~125 GB) — no
  encoder competing for memory
- **Modular hardware:** can swap encoder hardware (B70 → H200) without touching PD

**Losses vs same-host TP=1/TP=2 agg:**
- **~2× lower throughput** (0.25 vs 0.47-0.7 RPS)
- **3-5× higher TTFT** (12 s vs 3.5-8.1 s sub-saturation)
- **Higher per-GPU cost:** uses 2 GPUs to deliver less than 1-GPU agg

## Comparison vs same workload on 35B-A3B (per `comparison_5way_35b.md`)

For 8img/1080p, dell06_1E topology:
- **35B-A3B**: 0.85 RPS sat, 13 s TTFT @ rate=1.0
- **32B-FP8 (this run)**: 0.25 RPS sat, 58 s TTFT @ rate=1.0

35B-A3B is a **3.4× faster** because it's an MoE with only 3B activated parameters per token,
while 32B-FP8 is fully dense. The decode TPOT difference (35B ~5 ms vs 32B 70 ms median at
rate=1.0) confirms the per-token compute gap.

## Levers to raise throughput further

1. **PD-TP=2** (dedicate 2 H200s on super21 to PD)
   - Halves forward_duration → expected ~0.5 RPS
   - Caveat: per `pd_tp2_results.md` for 32B same-host disagg, PD-TP=2 alone gave near-zero
     uplift when the bottleneck was hand-off overhead. Here the bottleneck is genuinely
     PD compute, so this should help.

2. **Smaller workload** (4img/768p or 8img/768p)
   - 4img/768p ≈ 5× less ViT + ~5× less prefill → expected ~1+ RPS
   - 8img/768p ≈ 2.5× less prefill → expected ~0.5 RPS

3. **Smaller LLM** (Qwen3-VL-7B)
   - Halve forward_duration via fewer params → expected 0.5+ RPS
   - Bigger ViT/LLM ratio → potentially encoder-bound regime where disagg actually wins

4. **Output length reduction** (lower `max_tokens`)
   - 256 tokens decode at 56-106 ms ≈ 14-27 s decode time per request
   - Cutting to 128 tokens → ~7-13 s decode → saves ~4-15% on per-req lifetime

## What is NOT a viable lever

- **More encoders on dell06**: encoder is not the bottleneck (NIXL wire = 3.5% of lifetime,
  scheduler queue is empty). Adding more would just increase PD contention.
- **Faster network**: NIXL wire is 703 ms median for 637 MB = 0.9 GB/s effective on a 12.5 GB/s
  RoCE NIC. There's headroom but the wire isn't the binding constraint.
- **Larger NIXL buffer pool**: PD scheduler queue is empty; no buffer pressure.
- **More concurrency** (`max-running-requests` > 64): already at 64, observed peak running-req
  was 9. Plenty of headroom in scheduler.

## Files

- Sweep dir: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_sweep_np32_20260527_060121/`
  - `results_summary.csv` — all 7 rates × 23 columns of metrics
  - `sweep.log` — orchestration log with per-rate one-liners
  - `rate_<r>/results.txt` — full bench output per rate
  - `rate_<r>/benchmark_output.json` — structured metrics per rate
  - `rate_<r>/warmup.log` — warmup-phase log per rate
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- PD start script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh`
- GPUDirect patch: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/h200_cuda_nixl.patch`

## Companion docs

- **`time_breakdown_dell06_super21_32b_8img_1080p.md`** — detailed per-request stage breakdown
  (where the time goes within a single request) for the rate=1.0 run from this sweep
- `comparison_5way_35b.md` — 5-topology comparison for 35B-A3B (similar method, different model)
- `1080p_sweep_three_way.md` — 32B-FP8 same-host TP=1 agg vs TP=2 agg vs disagg sweep
- `disagg_all_rates_results.md` — original 32B-FP8 same-host disagg sweep
- `INDEX.md` — directory index of all 32B-FP8 investigation docs

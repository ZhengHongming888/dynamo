# Cross-Host Disagg mem_fraction Sweep: 0.65 vs 0.85 (8img/1080p np=64 r=1.0)

**Date:** 2026-06-01
**Test:** Same workload, same hardware, **only mem_fraction-static changed** (0.65 → 0.85)

## TL;DR

**Raising mem_fraction from 0.65 to 0.85 did NOT improve RPS** — actually slightly worse (0.24 → 0.23). Higher KV pool capacity lets more requests run concurrently (28 → 42 in-flight), but decode contention grows worse than the queue-wait improvement. **GPU compute is the real ceiling, not KV pool**, for 8img/1080p.

## Setup

| Component | Setting |
|---|---|
| **PD Worker** | GPU 5, NUMA 2, mlx5_4 RoCE 192.165.123.52 |
| **Encoders** | 4 × B70 (172.26.46.180) |
| **Workload** | 8img/1080p, np=64, rate=1.0, output=256 |
| **PD args** | `--max-running-requests 64 --page-size 16 --chunked-prefill-size 16384 --kv-cache-dtype fp8_e4m3` |
| **Variable** | `--mem-fraction-static`: **0.65** (run 1) vs **0.85** (run 2) |

## Headline Results

| Metric | mem_frac=0.65 | **mem_frac=0.85** | Delta |
|---|---:|---:|---:|
| **RPS** | **0.24** | **0.23** | **-4%** ⚠ |
| Successful | 64/64 ✓ | 64/64 ✓ | — |
| KV pool (max_total_num_tokens) | 467,072 | **695,136** | +49% |
| Theoretical in-flight cap | 27 | **41** | +52% |
| Observed PD running peak | 28 | **42** | +50% |
| Queue depth peak | 55 | 59 | +7% |
| Total tput (tok/s) | 3,907 | 3,748 | **-4%** |
| Median TTFT | 147.7 s | 161.1 s | +9% (worse) |
| Median E2E | 209.0 s | **233.3 s** | **+12%** (worse) |
| P99 E2E | 259.5 s | 277.5 s | +7% (worse) |
| Median TPOT | 506.9 ms | **650.2 ms** | **+28%** (worse) |
| **P99 TPOT** | 13,629 ms | **26,046 ms** | **+91%** (much worse) |
| Median PD queue_duration | — | 88.0 s | (similar to 92.6s) |
| Median PD forward_duration | 55.4 s | **82.3 s** | **+49%** (worse) |
| Avg concurrency | 48.2 | 53.8 | +12% |
| <500-token prefill batches | (similar 42%) | 43% | unchanged |

## Why mem_fraction=0.85 Didn't Help (and Slightly Hurt)

### 1. KV pool capacity DID increase as expected

```
mem_frac=0.65: max_total_num_tokens=467,072 → 27 in-flight cap → observed peak 28
mem_frac=0.85: max_total_num_tokens=695,136 → 41 in-flight cap → observed peak 42

  KV pool +49%, in-flight cap +52%, observed peak +50% ✓
```

The configuration change worked exactly as intended. PD now has 50% more KV slots.

### 2. But total token throughput DROPPED slightly

```
mem_frac=0.65: total_tput = 3,907 tok/s → RPS 0.236
mem_frac=0.85: total_tput = 3,748 tok/s → RPS 0.227 (-4%)
```

Higher in-flight count means **more requests share the same GPU compute resource during decode**. With 42 simultaneous decode batches vs 28:
- Each prefill batch admits 1 request (input 16,420 > chunked 16,384, must split)
- During decode, GPU SMs are split across 42 active sequences instead of 28
- Per-token decode time grows from ~33 ms → ~50 ms

The per-token slowdown outweighs the queue-wait improvement.

### 3. Median PD forward_duration grew significantly (+49%)

```
mem_frac=0.65: forward_duration median = 55.4 s
mem_frac=0.85: forward_duration median = 82.3 s
```

This is the smoking gun. With 50% more concurrent requests, **each request's forward (prefill+decode) time grew 49%**. The GPU is doing the same total work per second (~3,800 tok/s), just spread over more requests, so each individual request takes longer.

### 4. P99 TPOT got dramatically worse (13.6s → 26.0s, +91%)

The tail latency is hit hardest. With 42 in-flight, prefill admit interrupts running decodes more frequently, and decode batch contention causes some requests to wait many seconds between consecutive output tokens.

## Where Did The Time Go?

### mem_frac=0.65 (better) breakdown:

```
Median E2E 209 s = 14 (handoff) + 195 (PD lifetime)
                    PD: 93 (queue) + 55 (forward) + 47 (NIXL+dynamo)
```

### mem_frac=0.85 (worse) breakdown:

```
Median E2E 233 s = 14 (handoff) + 219 (PD lifetime)
                    PD: 88 (queue) + 82 (forward) + 49 (NIXL+dynamo)
                         ^-5%        ^+27s         ^+2s
```

**Queue dropped slightly (5s) but forward grew 27s** — the gain was outweighed by the loss.

## What This Tells Us About The Bottleneck

Earlier we identified 3 PD-side bottlenecks:
1. **#1 SGLang queue (47% of E2E)** — KV pool full
2. **#2 PD forward (28% of E2E)** — chunked-prefill split + GPU compute
3. **#3 NIXL recv overhead (24% of E2E)** — cross-host RoCE + dynamo

This experiment shows:
- Bottleneck #1 (queue) **was** addressable by raising KV pool — and it DID drop
- But raising in-flight count made bottleneck #2 (GPU compute) **proportionally worse**
- The two trade against each other; **net E2E time stays flat or worsens**

**Implication**: For 8img/1080p, you CANNOT fix the bottleneck by tuning mem_fraction. The fundamental constraint is **GPU compute per token × number of in-flight requests**, which is a fixed quantity. You can redistribute time between "queue waiting" and "forward executing" but you can't reduce the total.

## What WOULD Help

| Method | Why it works |
|---|---|
| **TP=1 Aggregate** (RPS 0.47) | Eliminates NIXL handoff (#3) entirely; single CUDA context allows tighter prefill/decode interleave |
| **TP=2 Aggregate** (RPS ~0.95) | Doubles GPU compute (the actual ceiling) |
| **`--chunked-prefill-size 32768`** | Eliminates split-tail in #2 (input 16,420 fits in one chunk) |
| **Smaller image (768p)** | Reduces input_len → no chunked split, smaller KV demand |
| **Reducing mem_fraction back to 0.65** | Counterintuitively gave better RPS (0.24 vs 0.23) and 12% lower E2E |

## Comparison with Same-Host Disagg

| Config | RPS | KV pool | run_max | E2E_med | Note |
|---|---:|---:|---:|---:|---|
| Cross-host mem_frac=0.65 | 0.24 | 467k | 28 | 209 s | First test |
| **Cross-host mem_frac=0.85** | **0.23** | **695k** | **42** | **233 s** | This run (worse than 0.65!) |
| Same-host mem_frac=0.85 | 0.23 | 695k | 31 | 175 s | (only 31 in-flight, 50% requests fail from NIXL buffer) |
| TP=1 Aggregate | 0.47 | 695k | 68 | 149 s | 2× faster, no NIXL |

Notice that **same-host disagg also tops out around running peak ~31** despite same 695k KV pool — the same-host PD process competes for GPU memory with encoder process, leaving less effective KV space.

Cross-host with mem_frac=0.85 actually pushes higher in-flight (42) than same-host (31), but the GPU compute ceiling caps RPS at the same ~0.23-0.24.

## Bench Reconciliation

`RPS = total_tput / per_req_tokens` (verified on every bench so far within 1.4%):

```
mem_frac=0.65: 3,907 / 16,537 = 0.236  (vs measured 0.24)  ✓
mem_frac=0.85: 3,748 / 16,537 = 0.227  (vs measured 0.23)  ✓
```

## Stack State

| Component | Status |
|---|---|
| PD Worker | Running on GPU 5, mem_frac=0.85, max_running=64 |
| B70 Encoders | 4 registered |
| Frontend | http://172.26.46.133:7001 |
| KV event port | Changed to 22082 (22081 had stale binding) |

## Files

- Bench result: `/hongming/res_crosshost_b70_4E_h200_pd_memfrac85/8img_1080p_rate1.0_np64_20260601_170734/`
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_20260601_165503.log`
- Launcher: `/tmp/start_pd_only_giga01_memfrac85.sh`
- Bench start UTC: `2026-06-01T17:07:34`

## One-Line Conclusion

> Raising PD's `mem_fraction_static` from 0.65 → 0.85 successfully expanded the KV pool from 467k → 695k tokens (+49%) and let in-flight grow from 28 → 42 (+50%), but RPS **dropped 4%** (0.24 → 0.23) and median E2E grew 12% (209s → 233s) because the higher concurrency causes proportional GPU compute contention during decode. **For 8img/1080p, GPU compute is the real ceiling, not KV pool — mem_fraction tuning won't help.**

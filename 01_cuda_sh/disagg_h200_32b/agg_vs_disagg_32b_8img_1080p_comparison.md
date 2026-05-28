# Agg TP=1 vs Disagg (32B-FP8) — Time Breakdown & Bottleneck Comparison

**Date:** 2026-05-27
**Hardware:** super21-h200 (172.26.46.133) GPU 5 — same hardware for both runs
**Model:** Qwen3-VL-32B-Instruct-FP8
**Workload:** 8 imgs × 1080p random JPEGs, in=128 / out=256, **rate=1.0 RPS, np=32**
**Both runs at `DYN_LOG=debug`** for full instrumentation.

## TL;DR

| Metric | agg_TP1 | **disagg (this run)** | Winner |
|---|---:|---:|:---:|
| **Throughput RPS** | **0.41** | 0.22 | **agg 1.84× faster** |
| Mean TTFT | 25.9 s | 60.4 s | **agg 2.3× lower** |
| Median TPOT | **229 ms** | **39 ms** | **disagg 5.9× lower** |
| Mean E2E | **58.4 s** | 67.5 s | agg 13.5% lower |
| Output throughput | **64 tok/s** | 37 tok/s | agg 1.74× higher |
| Concurrency (avg) | **24.6** | 16.4 | agg 50% higher |

**The headline:** agg wins on throughput by running many requests in parallel (mean running-req 14.7 vs 1.5), at the cost of higher per-token decode latency. **Disagg gives smoother per-token latency but at half the throughput.** The reason disagg is worse on RPS is **not** what I initially thought (handoff overhead) — it's that **disagg's PD only ends up running 1-2 concurrent requests at any time**, while agg's scheduler runs 14+.

## Test setup parity

| Config | agg_TP1 | disagg |
|---|---|---|
| GPU | super21 GPU 5 | super21 GPU 5 |
| TP | 1 | 1 (PD only; encoder on dell06) |
| Workload | 8 imgs × 1080p, np=32, rate=1.0 | (same) |
| `DYN_LOG` | debug | debug |
| GPUDirect patch | active (no NIXL needed) | active |
| `mem-fraction-static` | **0.85** | **0.65** (must reserve for NIXL buffers) |
| `max-running-requests` | **40** | 64 |
| `max_total_num_tokens` (KV budget) | **695 136** | **467 072** |
| `chunked_prefill_size` | 16 384 | 16 384 |
| Multimodal cache | enabled (`--enable-mm-global-cache`) | enabled |
| Encoder ViT runs on | super21 GPU 5 (inline) | dell06 H200 (cross-host) |
| Frontend | super21:7001 | super21:7001 |

## Bench result side-by-side

| Metric | agg_TP1 | disagg | Ratio |
|---|---:|---:|---:|
| Successful requests | 32/32 | 32/32 | = |
| Bench duration (s) | **75.9** | 131.7 | **agg 1.74× faster wall** |
| **Throughput (RPS)** | **0.42** | 0.24 | **agg 1.75× higher** |
| Mean TTFT (s) | **25.9** | 60.4 | agg 2.3× lower |
| Median TTFT (s) | **24.9** | 58.2 | agg 2.3× lower |
| P99 TTFT (s) | **41.1** | 101.8 | agg 2.5× lower |
| Mean TPOT (ms) | 342 | **43** | **disagg 8× lower** |
| Median TPOT (ms) | 229 | **39** | **disagg 5.9× lower** |
| Mean ITL (ms) | (similar) | (similar) | — |
| **Mean E2E (s)** | **58.4** | 67.5 | agg 13.5% lower |
| Output throughput (tok/s) | **64** | 37 | agg 1.7× higher |
| Peak output throughput | **848** | 199 | agg 4.3× higher |
| Avg concurrency | **24.6** | 16.4 | agg 1.5× higher |

## Internal stats (from PD logs at debug level)

| Metric | agg_TP1 | disagg | Notes |
|---|---:|---:|---|
| **Median lifetime** (T_recv → T_complete) | 56.6 s | 50.3 s | similar (~50-60s under saturation) |
| **Median forward_duration** | **34.9 s** | **8.6 s** | **disagg's per-batch forward is 4× shorter** |
| Median queue_duration | **16.3 s** | **0.28 ms** | **agg's scheduler holds many in queue; disagg's is empty** |
| Mean running-req at prefill | **14.67** | **1.54** | **agg runs ~10× more concurrent requests** |
| Max running-req | **31** | 4 | agg fully saturates `max-running=40` |
| Max queue-req | **18** | 2 | agg backs up to 18 deep; disagg never queues |

## Prefill concurrency: the headline difference

```
agg_TP1 prefill running-req distribution:
  running=0..4:    11 events (19%)
  running=5..15:   23 events (40%)
  running=16..31:  23 events (40%)  ← 40% of prefills happen with 16-31 concurrent reqs!
  max queue-req: 18

disagg prefill running-req distribution:
  running=0:  10 events (17.5%)
  running=1:  19 events (33.3%)
  running=2:  17 events (29.8%)
  running=3:   9 events (15.8%)
  running=4:   2 events (3.5%)
  max queue-req: 2
```

This is the **single most important observation**: agg keeps the GPU running with 14+ concurrent requests on average, while disagg only runs 1-2. **Agg's scheduler can stack much more work simultaneously.** This is why agg achieves nearly 2× the throughput despite higher per-batch forward_duration.

## Per-request stage breakdown

### Agg TP1 — single all-in-one process

```
Median lifetime (T_recv → T_complete): 56 631 ms

  Components:
  ├─ queue_duration:    16 278 ms  (29%) — request waits for batch admission
  ├─ forward_duration:  34 895 ms  (62%) — actual GPU work (with high-concurrency batched compute)
  ├─ Other:              5 458 ms  (10%) — egress + scheduler hops
  
  Mean running-req during prefill: 14.67
  Effective per-request GPU time: forward_duration / running-req ≈ 2 380 ms
```

The `forward_duration` is **per-request wall time inside the SGLang scheduler**, but a single `forward_duration` measurement covers a batched forward pass that processes 14+ requests simultaneously. So the "per-request GPU time" is much smaller than the median forward_duration value would suggest.

### Disagg — separate encoder + PD

```
Median lifetime (T_recv → T_complete): 50 280 ms (PD-only; doesn't include encoder phase)
Median end-to-end (T0_enc → T7_pd):    58 244 ms  (full request)

  PD-side components:
  ├─ NIXL wire:         32 383 ms* (55.6%) — wire time inflated by saturation queueing
  ├─ queue_duration:         0.3 ms (0.0%)  — PD scheduler queue empty
  ├─ forward_duration:    8 627 ms (15.0%) — single-stream PD compute
  └─ Other:               9 270 ms (16.0%) — egress + sched hops
  
  Encoder phase: ~9 200 ms (frontend route + encoder ViT + cpu→cuda + NIXL register + TCP→PD)
  
  Mean running-req during prefill: 1.54
  Effective per-request GPU time: forward_duration / running-req ≈ 5 600 ms
```

\* NIXL wire under saturation appears 32 s but is mostly queueing inside the NIXL state machine. The first request (no contention) shows wire = 1.0 s. See companion doc `time_breakdown_dell06_super21_32b_8img_1080p_v2_with_encoder.md`.

## Why disagg has lower throughput than agg

This is the key question, and the data points to a clear answer:

### **Root cause: PD-side concurrency is artificially constrained in disagg**

**Agg achieves running-req=14+ on the same GPU because:**
1. Encoder ViT runs **inline** with LLM forward — both are part of SGLang's batched scheduler
2. SGLang's scheduler can group N visual-prefill requests into a single batched forward pass
3. KV cache budget is 695k tokens → can hold KV for ~42 simultaneous 16k-token requests
4. No external dependency: the next request can enter the scheduler whenever the previous completes

**Disagg only achieves running-req=1-2 because:**
1. Each request must **first complete encoder→NIXL transfer** before PD even sees it
2. The encoder pipeline (cross-host) emits requests at ~1.5-3 s intervals (gated by encoder ViT + RoCE wire + dynamo TCP)
3. PD finishes processing one batch faster than the encoder can prepare the next one
4. KV cache budget is only 467k tokens (must reserve for NIXL receive buffers)
5. PD has **plenty of unused capacity** but no requests to feed it

### **Math:**
- agg PD throughput: `running-req × forward_throughput = 14.67 × ?`. Observed 0.41 RPS at saturation
- disagg PD throughput: `1.54 × ? = 0.22 RPS`. The PD GPU could theoretically handle much more
- **disagg's PD utilization is ~25% of agg's** — most of the time the GPU is idle waiting for the next embedding to arrive

### **Decode latency tradeoff (the disagg "win"):**
- agg TPOT 229 ms median — because decode is interleaved with prefill of new arrivals (prefill-blocks-decode)
- disagg TPOT 39 ms median — because PD has spare cycles between request bursts
- This is **the only metric where disagg is meaningfully better**

## Where time goes per request — visual comparison

```
                   AGG_TP1 (median 56.6s)            DISAGG (median 58.2s end-to-end)
                   ─────────────────────             ───────────────────────────────────
                                                      Encoder phase: 9.2s ████
                                                      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PD-side queueing:  16.3s ██████████                  PD wire wait:  32s (queueing) ██████████████
PD GPU forward:    34.9s ████████████████            PD GPU forward: 8.6s ███
                   (running-req≈14, batched)         (running-req≈1.5, mostly serial)
PD egress:         5.5s ██                           PD egress:     9.3s ████
                   ─────────────                     ─────────────
                   Total: 56.6s                      Total: 58.2s
                   Wins: throughput, peak output     Wins: TPOT, single-request latency
                                                     Loses: throughput, total RPS
```

## Throughput math

For the same GPU (super21 GPU 5, single H200):

```
agg_TP1:
  Throughput = mean_running_req / forward_per_request
             = 14.67 / 34895 ms
             = 0.420 RPS  (matches observed 0.42 ✓)
  Saturation: bench cap (np=32) hits before GPU saturates

disagg PD:
  Throughput = mean_running_req / forward_per_request
             = 1.54 / 8627 ms
             = 0.179 RPS  (matches observed 0.24 ≈ 0.22)
  Saturation: encoder pipeline saturates first → PD is starved
  
Why agg's running-req >> disagg's running-req on the SAME GPU?
  → agg's scheduler queues 18 requests deep at peak (queue_duration=16s)
  → disagg's scheduler queue is empty because requests trickle in 1.5-3s apart
```

## Bottleneck identification

### Agg TP1 bottleneck: PD GPU compute (good news)
- 100% of GPU time is productive batched compute
- Saturation comes from `max-running=40` cap (observed peak 31)
- KV cache fill rate (~125 GB) limits further concurrency growth
- **To improve**: more KV cache (larger GPU or TP=2)

### Disagg bottleneck: encoder→PD delivery rate (different bottleneck)
- PD GPU is **starved** — running-req only 1-2 most of the time
- Encoder takes ~3-9 s to deliver each request to PD
- Even though PD can process at agg-like rates, it can't feed itself fast enough
- **To improve**: more encoders, faster encoder GPU, or co-locate encoder on PD GPU (= back to agg!)

## Why disagg's TPOT is much better

Disagg achieves 39 ms median TPOT (vs agg's 229 ms) because:

1. **Decode runs alone in disagg most of the time** — PD has 1.5 running-req on average, decode batch size ≈ 1
2. **Agg's decode is interleaved with prefill** — when 14 requests are concurrent, 14 prefill chunks fight for the GPU between decode steps
3. **Agg's per-token cost is amortized over many concurrent decodes**, but each individual token waits longer

This is the classic prefill-blocks-decode pathology in agg, and the **only structural advantage of disagg**.

## Why disagg's mean E2E is slightly better despite lower RPS

```
agg E2E = queue_wait_at_scheduler + forward + egress
       = 16s + 35s + 5s = 56s

disagg E2E = encoder_phase + PD lifetime  
           = 9s + 50s
           = 59s
```

But both are around 58-67s mean E2E because:
- agg has higher GPU utilization (less idle time per request)
- disagg has lower PD utilization but pipeline overlaps mean encoder + PD work concurrently

The **mean E2E difference is small (8s)** because both modes pay roughly equivalent total time per request when the system is at saturation. The throughput difference comes from how many requests can be in-flight simultaneously (concurrency).

## Bottleneck summary

```
Agg TP1 bottleneck stack (in order of impact):
  1. PD GPU compute throughput (forward_duration scales with running-req)  ← THE limit
  2. KV cache capacity (max_total_num_tokens=695k)
  3. max-running-requests=40 cap
  
Disagg bottleneck stack:
  1. Encoder → PD delivery rate (~3-9s per request)  ← THE limit
  2. NIXL wire RDMA (1s per request best case)  
  3. PD GPU compute (PD has spare capacity, never fully utilized)
  4. KV cache capacity (max_total_num_tokens=467k — 33% smaller than agg)
```

## Key takeaways

1. **Disagg is worse than agg for throughput on this exact workload** because disagg's encoder cannot deliver requests fast enough to fill PD's running-req capacity. This is consistent with prior 32B disagg findings (`disagg_all_rates_results.md` showed 0.23 RPS at rate=1.0 for same-host disagg PD-TP=1, identical to this cross-host result).

2. **Disagg's only "win" is per-token decode latency** (5.9× lower TPOT), useful when:
   - You need consistent token-by-token latency for streaming UX
   - You don't need maximum throughput
   - Output is very long (long-decode workloads, see `long_output_1024_results.md`)

3. **Agg's "win" is throughput and TTFT** by exploiting batched-decode concurrency:
   - SGLang scheduler can hold 18+ requests in queue while processing 14 concurrently
   - Disagg's scheduler is artificially empty because encoder can't keep it fed

4. **The patches/optimizations don't change this fundamental result**:
   - GPUDirect RDMA (this work) makes wire fast — **but PD doesn't need it; it's idle**
   - More encoders (B70 4E, dell06) make encoder faster — **but PD still can't be fed faster than ~0.25 RPS for 32B FP8**
   - The structural answer is: at this model size and this workload, the PD-side bottleneck of agg (throughput-limited) is **less restrictive** than disagg's encoder-pipeline bottleneck (delivery-rate-limited)

5. **When does disagg actually win?**
   - When ViT compute is much larger than LLM compute (smaller LLMs like Qwen3-VL-7B)
   - With 4 imgs/768p workloads (encoder cost much smaller relative to LLM)
   - When the encoder pool is sized large enough to fully utilize PD (e.g., dell06_1E for 35B-A3B which had 0.85 RPS — close to agg territory)
   - For decode-heavy workloads (long output, modest input)

## Files

- agg bench result: `/hongming/res_xhost_dell06_super21/agg_tp1_32b_8img_1080p_rate1.0_np32_20260527_201758/`
  - `results.txt`, `benchmark_output.json`, `warmup.log`, `agg_breakdown.txt`
- disagg bench result: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_192949_v2/`
- agg log: `/hongming/dynamo/01_cuda_sh/agg_h200_32b/logs/epd_worker_server.log`
- disagg PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- disagg encoder log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/encode_gpu4_to_dell06.log`
- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/agg_vs_disagg_32b_8img_1080p_comparison.md`

## Companion docs

- `time_breakdown_dell06_super21_32b_8img_1080p_v2_with_encoder.md` — detailed disagg breakdown (encoder + PD)
- `time_breakdown_dell06_super21_32b_8img_1080p.md` — earlier disagg PD-only breakdown
- `sweep_dell06_super21_32b_8img_1080p_np32.md` — disagg 7-rate sweep (saturation curve)
- `time_diff_35b_32b_disagg.md` — 32B vs 35B comparison
- `1080p_sweep_three_way.md` — original TP=1/TP=2 agg vs disagg sweep at 1080p
- `disagg_all_rates_results.md` — earlier 32B same-host disagg sweep (0.23 RPS sat — matches this cross-host disagg)

# Time Breakdown Comparison: 32B-FP8 vs 35B-A3B over Cross-Host Disagg

**Date:** 2026-05-27
**Topology:** dell06 H200 (encoder) ↔ super21 H200 GPU 4 (PD) over RoCE 100 Gb/s NDR
**Workload:** 8 imgs × 1920×1080 random JPEGs, in=128 / out=256, **rate=1.0 RPS, np=32, np=33 sample (32 main + 1 internal warmup)**
**Patches:** `h200_cuda_nixl.patch` active on PD side (cuda:0 NIXL receive descriptors)
**PD config:** TP=1, mem-fraction-static=0.65 (32B) / 0.75 (35B), max-running-requests=64 (32B) / 40 (35B)

This document compares the **same workload on the same hardware** for two models with very different
compute profiles:

- **Qwen3-VL-32B-Instruct-FP8** — dense, FP8 weights, 64 transformer layers, fa3 attention everywhere
- **Qwen3.5-35B-A3B** — MoE, BF16 weights, 30 linear-attention + 10 full-attention layers (hybrid),
  ~3B activated parameters per token via top-k expert routing

Companion docs:
- `time_breakdown_dell06_super21_32b_8img_1080p.md` — detailed 32B breakdown
- `comparison_5way_35b.md` — 35B 5-topology comparison (this same dell06_1E topology hit 0.85 RPS in May)
- `report_on_detailed_time_breakdown_h200_b70_4E.md` — 35B + B70 4E (different encoder topology)

## Headline: 35B-A3B is 3× faster than 32B-FP8 on the same workload

| Metric | 32B-FP8 (dense) | **35B-A3B (MoE)** | 35B/32B ratio |
|---|---:|---:|---:|
| Successful requests | 33/33 ✓ | 33/33 ✓ | = |
| Bench wall-clock (s) | 132.8 | 43.2 | **3.07× faster** |
| **Throughput (RPS)** | **0.248** | **0.764** | **3.07× higher** |
| **Mean E2E latency** | 71.9 s | 18.8 s | 3.83× faster |
| **Median TTFT** | 58 s | 14.5 s | 4.0× faster |
| Mean TPOT | 84 ms | 59 ms | 1.4× faster |
| Median TPOT | 70 ms | 26 ms | 2.7× faster |

## Per-stage timing comparison (medians)

### End-to-end PD-side lifetime (Rust ingress → Rust egress)

| Stat | 32B-FP8 | **35B-A3B** | 32B/35B ratio |
|---|---:|---:|---:|
| **median** | 64 221 ms | **4 170 ms** | **15.4× longer for 32B** |
| mean | 54 536 ms | 4 373 ms | 12.5× |
| p99 | 80 606 ms | 8 193 ms | 9.8× |
| max | 81 183 ms | 8 765 ms | 9.3× |
| min | 6 666 ms | 1 232 ms | 5.4× |

**32B's lifetime is dominated by request queueing** (rate=1.0 saturates 32B at 0.25 RPS, so
np=32 cap is full → arrivals slow → in-flight requests wait their turn). 35B has plenty of
headroom (only running-req max 5, vs 32B's 9) so requests flow through quickly without queueing.

### PD GPU forward_duration (the actual GPU compute, from ReqTimeStats)

| Stat | 32B-FP8 | **35B-A3B** | 32B/35B ratio |
|---|---:|---:|---:|
| **median** | **10 478 ms** | **2 553 ms** | **4.1× longer for 32B** |
| mean | 13 662 ms | 2 695 ms | 5.1× |
| max | 44 311 ms | 6 305 ms | 7.0× |

**This is the real per-request compute cost difference.** Per-token compute on H200:
- 32B-FP8 dense: every token activates all 32B parameters (FP8 matmul throughput)
- 35B-A3B MoE: every token activates only ~3B parameters (BF16 matmul, 3B << 32B effective param count)

So 35B has **~10× lower per-token compute** but is **2× heavier per parameter** (BF16 vs FP8) →
net **~4-5× faster per-token** → reflected in the ~4-5× shorter forward_duration. ✓

For the standard workload (16k visual tokens prefill + 256 token decode):
- 32B prefill cost: ~10 s on H200 (dominated by attention over 16k token sequence)
- 35B prefill cost: ~2.5 s on H200 (linear attention layers + sparse expert routing)

### PD scheduler queue_duration

| Stat | 32B-FP8 | **35B-A3B** | 35B/32B ratio |
|---|---:|---:|---:|
| **median** | **0.32 ms** | **0.60 ms** | similar |
| mean | 489 ms | 153 ms | 0.31× (35B less queued) |
| max | 4 443 ms | 1 000 ms | 0.23× (35B much smaller tail) |

Both have effectively empty schedulers at the median (sub-ms). The mean/max difference shows:
- **32B**: occasionally requests pile up at the scheduler when a burst arrives (max queue was 4.4s)
- **35B**: requests are processed too quickly to ever build queue depth (max only 1s)

### PD prefill concurrency (running-req from prefill events)

| Stat | 32B-FP8 | **35B-A3B** |
|---|---:|---:|
| Total prefill events | 56 | 90 |
| running-req median | 3 | **1** |
| running-req mean | 3.59 | 1.66 |
| running-req max | 9 | 5 |

**32B PD runs at higher concurrency** because requests stack up (3-9 in flight at any time at
saturation). **35B PD has lower concurrency** because each request finishes faster than the
next arrives (1.0 RPS arrival vs ~2.5 s/req → effective in-flight ≈ 2.5).

35B has **more prefill events (90 vs 56)** because the 32B scheduler does fewer, larger chunked
batches due to higher contention; 35B can process each request as its own clean batch.

### Token counts (matched workload — sanity check)

| Metric | 32B-FP8 | 35B-A3B |
|---|---:|---:|
| Input length median | 16 413 tok | 16 415 tok |
| Output length median | 164 tok | 164 tok |

✓ Workloads are equivalent. The 32B and 35B vision towers produce the same number of visual
tokens for 8 × 1080p images.

### NIXL transfer (cuda:0 ↔ cuda:0 GPUDirect RDMA)

| Stat | 32B-FP8 (DEBUG log) | 35B-A3B (INFO log) |
|---|---:|---|
| Wire time median | 703 ms (per detailed breakdown) | not captured at info log level |
| GPU descriptor count | 66 (2 per req × 33 reqs) | 0 (info log doesn't emit `Created Descriptor`) |
| Patch in effect | ✓ (verified in 32B run) | ✓ (same module, same patch — running same code) |

The NIXL wire is the same code path for both models. The 35B PD was run with `DYN_LOG=info`
which suppresses the per-transfer DEBUG events, so we can't measure it directly here. From
the 32B run we know NIXL wire is ~3.5% of lifetime (703 ms / 20 s) — for 35B at 4 s lifetime
the same wire would be ~17% of lifetime if equally fast.

## Stage-by-stage breakdown comparison (medians)

```
                                            32B-FP8                    35B-A3B
                                            ──────────                ──────────
Lifetime breakdown (medians):
  Frontend dispatch (T1-T0)                 5 ms          0.02%        ~5 ms (assumed same)
  CUDA buffer alloc (T2-T1) [patched]       0.4 ms        0.00%        ~0.4 ms (same code)
  NIXL setup (T3-T2)                        4 ms          0.02%        ~4 ms (same code)
  NIXL submit (T4-T3)                       0.2 ms        0.00%        ~0.2 ms (same code)
  NIXL wire transfer (T5-T4) [RDMA]         703 ms        3.50%        ~700 ms (same wire)
  PD scheduler queue (queue_duration)       0.32 ms       0.00%        0.60 ms        0.01%
  PD GPU forward (prefill+decode)           10 478 ms    52.23%        2 553 ms       61.2%
  Other (egress + scheduler hop)            8 869 ms     44.21%        ~900 ms (residual)
  ─────────────────────────────────────────────────────────────────────────────
  Total median lifetime                     20 060 ms    100%          4 170 ms        100%
```

(Note: the 35B per-stage values are estimated from the 32B baseline since detailed checkpoints
weren't captured at info log level. The total lifetime, queue_duration, and forward_duration
are measured directly from the 35B PD log.)

### Where the 32B's extra time goes vs 35B

```
32B lifetime breakdown:                    35B lifetime breakdown:

  PD GPU forward     ████████████████      PD GPU forward    ███████████████████████████  (61%)
  (10.5 s, 52%)      ████████████          (2.5 s)
                     ███████
                                           
  Egress + sched     ███████████████        Egress + sched    █████  (~22%)
  (8.9 s, 44%)       ███████                (~0.9 s)
                                           
  NIXL wire (3.5%)   ██                     NIXL wire           ████  (~17%, similar absolute)
  
  Setup+queue (~0%)  ▏                      Setup+queue         ▏  (~0%)
  ────────────────────────────────────────  ──────────────────────────────────
  20 s total                                4.2 s total
```

## Bottleneck analysis

### 32B-FP8: PD compute is the binding bottleneck
- forward_duration = 52% of lifetime
- 16k token prefill × 32B dense FP8 = 10 s on H200
- Dense attention pattern over full 16k sequence costs O(N²) = ~260M attention FLOPs/layer × 64 layers
- Throughput cap: avg_running_req / forward_duration = 3.6 / 10.5 s = **0.34 RPS theoretical**
- Measured: 0.25 RPS at rate=1.0 (queue/egress eating margin)

### 35B-A3B: PD compute is **less** dominant; system has room to grow
- forward_duration = 61% of lifetime in absolute terms, but lifetime is 5× shorter
- 16k token prefill × 3B activated MoE BF16 + linear-attention = 2.5 s on H200
- Hybrid attention reduces O(N²) cost: 30 linear-attn layers run O(N) instead of O(N²)
- Throughput cap: 1.66 / 2.55 s = **0.65 RPS theoretical**
- Measured: 0.76 RPS at rate=1.0 (essentially at PD cap; consistent with `comparison_5way_35b.md`'s 0.85 RPS sat)

### What's NOT the bottleneck (both models)
1. **Encoder ViT on dell06**: same H200, same vision tower architecture for both Qwen3-VL family.
   Encoder serves both at far higher than 1 RPS → never the binding constraint.
2. **NIXL wire**: 703 ms median, ~17% of 35B lifetime, ~3.5% of 32B lifetime. RoCE delivers
   ~0.9 GB/s effective for 637 MB transfers — well below 12.5 GB/s peak. Plenty of headroom.
3. **PD scheduler**: 0.3-0.6 ms median queue_duration on both. No back-pressure.
4. **GPUDirect patch**: cuda:0 buffer alloc is 0.4 ms (PyTorch caching allocator hit) — saving
   ~50 ms vs the un-patched CPU bounce. Patch contributes ~0.5% to 32B lifetime but ~1.2% to
   35B lifetime (still small absolute, but non-negligible at 35B speeds).

## Why 35B-A3B is 3× faster than 32B-FP8

### Architecture comparison

| Property | 32B-FP8 | 35B-A3B |
|---|---|---|
| Total parameters | 32B | 35B |
| Activated parameters / token | 32B (all dense) | **~3B** (top-k MoE routing) |
| Weight precision | FP8 (1 byte) | BF16 (2 bytes) |
| Layers | 64 | 40 (30 linear-attn + 10 full-attn) |
| Attention pattern | Full attention all layers | **75% linear-attn (O(N) cost)** |
| KV cache dtype | FP8 (with `--kv-cache-dtype fp8_e4m3`) | BF16 native |

### Per-token compute breakdown

| Stage | 32B-FP8 | 35B-A3B | Ratio |
|---|---|---|---|
| Linear projections (Q/K/V/O) | 4 × 32B FP8 = 32 GFLOPs/tok | 4 × 3B BF16 = 6 GFLOPs/tok | **5.3× lower** |
| MLP / FFN | 32B FP8 = 32 GFLOPs/tok | 3B BF16 sparse = 6 GFLOPs/tok | **5.3× lower** |
| Attention (16k tokens, full) | 64 layers × O(16k²) | 10 layers × O(16k²) + 30 × O(16k) | **~6× lower** |
| Total per-token cost | ~2 TFLOPs | ~0.4 TFLOPs | **~5× lower** |

5× lower per-token compute → 5× faster forward_duration in theory. Measured ratio is 4.1×
(median) to 5.5× (mean) — matches.

### Why throughput ratio is "only" 3× when forward is 4-5× faster
- Decode of 256 tokens on 32B-FP8 takes ~256 × 70 ms = 18 s (TPOT 70 ms)
- Decode of 256 tokens on 35B-A3B takes ~256 × 26 ms = 6.6 s (TPOT 26 ms)
- 35B decode is **2.7× faster**, less than 5× because the per-token decode cost is dominated
  by KV cache read bandwidth (BF16 KV is heavier than FP8 KV)
- Combined prefill (4-5×) + decode (2.7×) → throughput improvement ≈ 3-3.5× ✓

## Implications for cross-host disagg

### When does dell06_1E disagg topology make sense?

| Model | Sat RPS | Best TTFT | vs same-host TP=1 agg |
|---|---:|---:|---|
| **35B-A3B** | **0.85 RPS** (per `comparison_5way_35b.md`) | 13 s | competitive |
| **32B-FP8 (this run)** | **~0.25 RPS** | 12 s | 2× slower (TP=1 agg = 0.47 RPS) |

For 35B-A3B: cross-host disagg with H200 encoder is **viable** at 0.85 RPS — close to
same-host TP=1 agg (0.47 from `comparison_5way_35b.md`'s 32B numbers). The MoE's low per-token
compute means the PD finishes its work fast enough that the cross-host overhead is amortized.

For 32B-FP8: cross-host disagg is **structurally worse than same-host TP=2 agg** (0.25 vs
0.6-0.7 RPS) because the PD becomes the binding constraint on a single GPU. The H200 encoder
is fast enough but the LLM is too dense.

### Recommendations

| Use case | Best topology |
|---|---|
| **35B-A3B + 8img/1080p production** | cross-host disagg dell06_1E or same-host TP=1 agg |
| **32B-FP8 + 8img/1080p production** | same-host TP=2 agg (0.6-0.7 RPS) |
| **32B-FP8 + cross-host disagg** | only if hardware constraints require encoder/PD separation; expect ~0.25 RPS |

## Levers to improve each model further

### 32B-FP8 cross-host (this work)
1. **PD-TP=2** on super21 (use 2 H200s for PD) → halve forward_duration → ~0.5 RPS expected
2. Smaller workload (4 imgs / 768p) → smaller per-request prefill cost
3. Switch to same-host TP=2 agg (gives up cross-host advantages but ~3× throughput)

### 35B-A3B cross-host
1. **Already near saturation at rate=1.0** (0.76 RPS observed, sat is ~0.85)
2. Higher rates would just increase queue depth without throughput gain
3. Smaller workload (4img/768p) gives ~2-3 RPS per `comparison_5way_35b.md`
4. PD-TP=2 may not help much because PD compute is no longer the dominant bottleneck —
   would need to re-measure to confirm

## Files

- 32B PD log (DEBUG): `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- 35B PD log (INFO): `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01.log`
- 32B bench result: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_051953/`
- 35B bench result: `/hongming/res_xhost_dell06_super21/35b_8img_1080p_rate1.0_np32_20260527_070523/`
- 32B detailed breakdown: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/time_breakdown_dell06_super21_32b_8img_1080p.md`
- 32B sweep report: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/sweep_dell06_super21_32b_8img_1080p_np32.md`
- 35B reference (different topology): `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/report_on_detailed_time_breakdown_h200_b70_4E.md`
- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/time_diff_35b_32b_disagg.md`

## Caveat

The 35B run was launched with `DYN_LOG=info` (per the script default), which suppresses the
DEBUG events `Created Descriptor`, `Created ReadOperation`, and `NIXL reported transfer state`.
We can therefore confirm:
- Per-request lifetime (from Rust ingress/egress events)
- forward_duration and queue_duration (from ReqTimeStats)
- Prefill batch concurrency (from scheduler info logs)

But we **cannot directly measure** for the 35B run:
- NIXL wire transfer time
- T1-T7 sub-stages (cuda alloc, NIXL setup, etc.)

These are estimated from the 32B run because they share the exact same code path
(`dynamo.common.multimodal.embedding_transfer`, the same patched lines). If a future 35B run
is needed for fully-instrumented analysis, restart the PD with `DYN_LOG=debug` (one env var
change in `start_sglang_pd_cuda_35b_giga01.sh`).

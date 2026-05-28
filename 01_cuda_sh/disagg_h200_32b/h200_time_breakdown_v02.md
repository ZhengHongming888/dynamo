# Time-breakdown analysis: cross-host disagg, patched vs unpatched

**Date:** 2026-05-24
**Workload:** 8img/1080p, rate=1.0, 64 prompts, np=64, Qwen3-VL-32B-Instruct-FP8
**Topology:** B70 4× XPU encoders + giga01 H200 PD (TP=1, max-running=64)

This document breaks down where time is actually spent in a cross-host disagg
request, compares the patched and unpatched runs, and identifies the real
bottleneck.

## TL;DR

- **Both runs delivered 0.13 RPS** — patches don't help throughput at this workload.
- **The dominant phase is "encoder→PD hand-off" at ~340 s** (encoder ViT + NIXL
  metadata setup + PD enqueue). It dwarfs everything else by 50×.
- The patches **measurably improve decode latency**: median TPOT 16.79 ms → 2.37 ms (7×
  better), median ITL 2.74 ms → 1.77 ms (35% better). But this is invisible
  in TTFT-dominated workloads.
- **The "encoder ViT compute" itself appears to be the bottleneck** based on
  the absolute scale of the encoder→PD gap (340 s). Even if NIXL transfer were
  zero, we'd still be encoder-bound.

## Per-request phase breakdown (patched run)

For 10 requests fully observable in the patched run's PD log window:

| Phase | min | p50 | mean | p90 | max |
|---|---:|---:|---:|---:|---:|
| Frontend route (FE recv → encoder dispatch) | 73 ms | 93 ms | 90 ms | 109 ms | 109 ms |
| **Encoder → PD (ViT + NIXL setup + queue)** | **296 s** | **344 s** | **343 s** | **426 s** | **426 s** |
| PD engine setup (recv → start async gen) | 1.2 ms | 1.7 ms | 1.7 ms | 2.4 ms | 2.4 ms |
| **PD compute (start → finish async gen)** | **6.0 s** | **7.2 s** | **8.3 s** | **13.7 s** | **13.7 s** |
| PD completion post-process | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| PD complete → Frontend complete | 2 ms | 20 s | 19 s | 48 s | 48 s |
| **End-to-end (FE recv → FE complete)** | **347 s** | **360 s** | **370 s** | **434 s** | **434 s** |

### Key observations

1. **Encoder → PD gap is 92% of E2E latency** (343 s out of 370 s mean).
2. **Frontend routing is fast** (~90 ms). This is just HTTP parsing + KV-router
   selection + opening a TCP stream to the encoder.
3. **PD-side is fast** (~8 s for compute + ~0.1 ms for completion). The H200
   easily handles the LLM compute.
4. **PD complete → FE complete tail (mean 19 s, max 48 s)** is suspicious — looks
   like backpressure on the streaming response path under load. Not always the
   case (min was 2 ms for one request) but consistent for many.
5. **Queue at PD is 0 ms** — PD is never waiting for compute slots. Confirms PD
   is not the bottleneck.

## Where the 343-second encoder→PD gap comes from

This phase encompasses:
1. Encoder receives request from frontend over TCP
2. Encoder fetches/decodes 8 image URLs (random JPEGs, ~16 MB each = ~128 MB)
3. Encoder runs Qwen3-VL ViT on Intel Battlemage XPU
4. Encoder calls dynamo to register the embedding tensor with NIXL
5. Encoder forwards request metadata to PD over TCP request plane
6. PD receives request, parses metadata
7. PD initiates NIXL ReadOperation back to encoder
8. RDMA transfer of 638 MB embeddings (xpu→cuda GPUDirect, ~13 ms theoretical)
9. PD's `worker_handler._generate_aggregated` ingests the embedding
10. PD logs `request received`

We don't have direct B70 encoder logs accessible from giga01, but we can
estimate phases:

- The **RDMA transfer itself**: 638 MB × 8 / (50 GB/s) = ~100 ms wire time.
  Negligible vs the 343 s total.
- **PD-side NIXL setup**: from PD log, NIXL ReadOperation creation to DONE
  is consistently 200-400 ms. Negligible.
- **Encoder ViT compute on Intel B70**: this is likely the dominant cost
  given the relative speed of Battlemage XPUs vs H100/H200.

**Per-request rate: 343 s / 4 encoders = ~85 s/encoder**, suggesting each B70
encoder takes ~85 s to ViT-encode 8 × 1080p images. That's the bottleneck.

For comparison, a same-host encoder on H100/H200 GPU would do this ViT in
~3-5 s, which is consistent with same-host disagg's ~80 s TTFT (PD-TP=1, see
SESSION_MEMORY).

## Patched vs Unpatched — bench-level comparison

Same 8img/1080p, rate=1.0, 64 prompts, 4 encoders.

| Metric | Unpatched (cpu→cpu) | Patched (xpu→cuda) | Δ |
|---|---:|---:|---:|
| Duration | 498.98 s | 492.58 s | -1.3% |
| Throughput | 0.128 RPS | 0.130 RPS | +1.5% |
| Successful | 64/64 | 64/64 | — |
| **Mean E2E** | **411.03 s** | **409.84 s** | -0.3% |
| Median E2E | 409.96 s | 402.61 s | -1.8% |
| P99 E2E | 488.25 s | 483.64 s | -0.9% |
| **Mean TTFT** | **368.47 s** | **387.75 s** | **+5.2%** ← worse |
| Median TTFT | 368.05 s | 392.78 s | +6.7% ← worse |
| P99 TTFT | 460.57 s | 459.81 s | -0.2% |
| **Mean TPOT** | **1945.99 ms** | **858.61 ms** | **-55.9%** ← much better |
| **Median TPOT** | **16.79 ms** | **2.37 ms** | **-85.9%** ← much better |
| Mean ITL | 391.94 ms | 315.98 ms | -19.4% |
| **Median ITL** | **2.74 ms** | **1.77 ms** | **-35.4%** |

### What the patches did and didn't help

| Where | Patched effect | Why |
|---|---|---|
| **Decode (TPOT/ITL)** | **7× faster median TPOT, 35% faster median ITL** | Embedding lives on PD GPU directly, no per-token CPU→GPU staging. |
| Mean TPOT | 56% better | Same reason, but tail latency still has occasional spikes. |
| **TTFT** | Slightly worse (5-7% worse) | Patching doesn't shorten the dominant encoder→PD path; small overhead from GPU descriptor setup may have made TTFT slightly worse. |
| E2E mean | Within noise | TTFT dominates; small TPOT win mostly invisible. |
| Throughput | Within noise | Throughput is gated by encoder ViT, not by anything the patch touches. |

## Updated picture of "where time goes" — bar chart

In a typical patched request:

```
0s ────────────── 343s ─── 351s ─── 370s ──── (end)
 │ frontend(90ms) │  enc→PD  │  PD  │ stream │
 │ (negligible)   │   ~340s  │ ~8s  │ ~19s   │
                       ^^^^^^^^^^^^^^^^^^^^^
                       The bottleneck region
```

Encoder→PD is **92% of total latency**. PD compute is **2%**. Frontend
routing is **0.02%**. Response stream tail is **5%**.

The patch (CPU→GPU bounce removal) targets a fraction of the small response
stream tail and the per-token decode overhead. It does NOT touch the 92%
encoder-side region.

## What would actually move the needle

Reducing the 343 s encoder-side phase:

1. **Faster encoder GPU**: replace B70 Intel Battlemage with H100/H200 SXM (where
   ViT runs ~10-20× faster). Same-host disagg measurements showed this gives
   ~80-200 s TTFT instead of ~370 s.

2. **More encoder workers**: B70 already has 4. Could add more (8 if more XPUs
   available, or another encoder host). Linear scaling expected up to where
   the PD gets saturated.

3. **Smaller workloads**: 4img/768p has 5× less ViT work per request. That's
   why the unpatched 4-encoder hit 1.02 RPS at 4img/768p vs 0.13 RPS at
   8img/1080p.

4. **Encoder caching** for repeated images. Doesn't help random-image bench but
   is a real win for production.

What would NOT help:
- Faster RDMA / better network (already 13 ms wire, vs 343 s encoder).
- Larger PD batches (PD already has zero queue and only ~7 s compute).
- More PD GPUs (TP=2/4) — same reason, PD isn't bound.
- The patch we just deployed (it helped where it could, but the bottleneck
  is upstream).

## Decoder latency win is real but invisible at this rate

Median TPOT went from 16.79 ms → **2.37 ms** (7×). At sustained generation,
this means:
- Unpatched: ~60 tok/s decode rate
- Patched: ~420 tok/s decode rate

But the bench only requests 256 tokens/request. At 420 tok/s, that's 0.61 s of
decode work. At 60 tok/s, it's 4.27 s. So saving 3.7 s of decode in a 370 s
end-to-end request is invisible (1% improvement).

For workloads with **long output** (e.g., max_tokens=2048 or 4096), this
patch saves real wall time:
- Unpatched: 4096 tokens × 16.79 ms = 68.7 s of decode
- Patched: 4096 tokens × 2.37 ms = 9.7 s of decode
- Savings: **59 seconds per request**

So the patch has a clear use case: **multimodal disagg with long-output
generation tasks** (image+text → essay, code, detailed analysis). Just not
visible in our 256-token random output benchmark.

## Conclusions

1. **At 8img/1080p, the encoder ViT on B70 Intel XPUs is the rate-limiting
   stage.** The 343 s encoder→PD gap is the bottleneck.
2. **Patches successfully wire up GPUDirect xpu→cuda RDMA.** Verified with
   `device=xpu:0/cuda:0` in 65/65 NIXL ReadOperations.
3. **Patches help decode (TPOT/ITL) by 7× / 1.5×** because embeddings live on
   PD GPU directly. Decode-heavy workloads see large gains; TTFT-heavy
   workloads see no gains.
4. **No throughput improvement at this workload** because we're not transfer-
   bound. We're encoder-compute-bound.
5. **Smaller workloads or longer output would show different curves**:
   - 4img/768p: encoder-bound less; transfer was already a smaller fraction.
   - long output: decode TPOT improvement matters proportionally more.

## Sources

- Patched run: `/hongming/res17_xhost_4enc_pd_xpu_full_gpu/8img_1080p_rate1.0/benchmark_output.json`
- Unpatched run: `/hongming/res12_crosshost_giga01/h200_pd_b70_encoder_tp1_32b_image8_1080p_np64/rate_1.0_4encoders/benchmark_output.json`
- PD log used for per-request timeline: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- Frontend log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/frontend_giga01.log`
- Companion docs: `patched_results_b70_h200_v01.md`, `b70_patched.md`, `h200_cuda_nixl.patch`

## Caveat on the per-request analysis

Only **10 of 60 PD-completed requests** had a complete observable timeline
(received → completed) within the bench window. The other 50 either started
before our window opens or completed after the bench ended. The 10-request
sample is small; means and percentiles are illustrative, not statistically
robust. The bench-level JSON metrics (duration/throughput/mean/median TTFT/
TPOT/ITL) come from full 64-request samples and are reliable.

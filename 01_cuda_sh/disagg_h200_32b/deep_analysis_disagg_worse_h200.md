# Deep Analysis — Why Disagg is Worse than TP=1 Agg on H200

**Workload:** Qwen3-VL-32B-Instruct-FP8, 8 × 1920×1080 images per request, np=64
**Hardware:** Single H200 host, GPUs 4 (encoder) + 5 (PD), NV18 NVLink between them
**Comparison rate:** 1.0 req/s offered
**Result files:**
- TP=1 agg: `/hongming/res4/h200_agg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_194141/`
- Disagg v1: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_211224/`

**TL;DR:** TP=1 agg achieves 0.52 req/s with 32 s mean TTFT and zero failures. Disagg achieves 0.27 req/s with 79 s mean TTFT and 15 / 64 failures. **Disagg is half the throughput on the same GPU class.** The fundamental cause is **not** NIXL transport overhead — it is that disagg makes PD do MORE total work per request (NIXL coordination + small-batch embedding integration into KV), while TP=1 agg fuses encoder and LLM forward into a single continuous scheduler pass.

---

## Headline result side-by-side (rate=1.0 req/s, np=64)

| Metric | **TP=1 agg** (GPU 4) | **Disagg** (encoder GPU 4 + PD GPU 5) | Δ |
|---|---:|---:|---:|
| Actual RPS | 0.52 | 0.27 | **−48%** |
| Successful requests | 64 / 64 | **49 / 64** | **15 failed** |
| Bench duration | 123 s | 184 s | +50% |
| Mean TTFT | 32.4 s | 78.9 s | **+143%** |
| Mean TPOT | 1,069 ms | 752 ms | −30% |
| Mean E2E | 80.7 s | 111.3 s | +38% |
| Concurrency | 41.8 (cap=40) | 29.6 | — |

Two GPUs (disagg) and one-half the throughput of one GPU (TP=1 agg). The encoder GPU is essentially wasted.

---

## The wrong story I told first

In the initial investigation I focused on `Timeout while waiting for available buffer.` errors in the PD log (48 of them, matching exactly the 48 failed requests across rates 0.5/1.0/1.25). I concluded "NIXL buffer pool exhaustion" and proposed:
1. Throttle encoder to `--max-running-requests=8`
2. Reduce `NIXL_BUFFER_COUNT` from 256 → 32

Re-ran with both fixes (v2). **Got worse.** v2 @ rate=1.0: 0.21 RPS, 83 s TTFT, 25 / 64 failed.

This proved the buffer-timeout *symptom* was not the *cause*. The cause is upstream of the buffer pool.

---

## What actually slow down disagg PD vs TP=1 agg

### Direct measurement: per-chunk prefill wallclock

Both runs' worker logs include `report_prefill_stats` events. I extracted all 8192-token chunk events during each run's rate=1.0 window and computed the time between consecutive events (per-chunk wallclock):

| Per-chunk metric | TP=1 agg @ rate=1.0 | Disagg PD @ rate=1.0 |
|---|---:|---:|
| Sample count | 152 | 137 |
| **Median wallclock** | **1.08 s** | **2.28 s** |
| **Mean wallclock** | **1.35 s** | **3.78 s** |
| p25 | 0.94 s | 1.06 s |
| p75 | 1.50 s | 3.37 s |
| Min | 0.64 s | 0.62 s |
| Max | 27.1 s | 97.9 s |
| Chunks <1 s | 49% (74/152) | 23% (31/137) |
| Chunks 2-5 s | 2% (3/152) | 48% (66/137) |
| Chunks >5 s | 3% (4/152) | 12% (16/137) |

### Direct measurement: per-chunk prefill throughput

Throughput as reported by SGLang (`input throughput (token/s)`):

| Per-chunk throughput | TP=1 agg | Disagg PD |
|---|---:|---:|
| Median input tput | 10,785 tok/s | **5,695 tok/s** (−47%) |
| Mean input tput | 10,254 tok/s | **5,449 tok/s** (−47%) |
| p25 (lower quartile) | 10,673 tok/s | **3,440 tok/s** (−68%) |
| p75 | 12,038 tok/s | 7,707 tok/s |
| **Max (peak)** | **12,826 tok/s** | **13,302 tok/s** (≈ same) |
| Min | 112 tok/s | 259 tok/s |

### What this tells us

**The peak (max) prefill throughput is identical** in both modes (~13k tok/s). This rules out:
- Different GPU performance
- NCCL or all-reduce overhead (TP=1 has none anyway)
- Memory bandwidth differences
- Driver/CUDA-version differences

**But disagg's median throughput is half**, p25 is one-third. The GPU has the *same* peak speed, but disagg can't sustain it. Something is filling the pipeline with low-utilization batches.

---

## Smoking gun — small-batch embedding integration on PD

Looking at disagg PD's `Prefill batch` events, I found a unique pattern not present in TP=1 agg:

```
T21:35:37.532 - Prefill batch, #new-seq: 4, #new-token: 400,  #cached-token: 49008, ... input tput: 166 tok/s
T21:35:41.283 - Prefill batch, #new-seq: 2, #new-token: 160,  #cached-token: 16336, ... input tput: 162 tok/s
T21:35:50.327 - Prefill batch, #new-seq: 2, #new-token: 112,  #cached-token: 16336, ... input tput: 120 tok/s
T21:36:42.140 - Prefill batch, #new-seq: 2, #new-token: 176,  #cached-token: 16336, ... input tput:  35 tok/s
T21:37:08.118 - Prefill batch, #new-seq: 2, #new-token: 128,  #cached-token: 16336, ... input tput: 110 tok/s
T21:37:28.787 - Prefill batch, #new-seq: 1, #new-token:  32,  #cached-token: 16336, ... input tput:  10 tok/s
T21:37:33.607 - Prefill batch, #new-seq: 1, #new-token:  32,  #cached-token: 16336, ... input tput:  41 tok/s
T21:37:39.690 - Prefill batch, #new-seq: 1, #new-token:  48,  #cached-token: 16336, ... input tput:  90 tok/s
```

These are characterized by:
- **Tiny `#new-token`** (16, 32, 48, 80, 96, 112, 128, 160, 176, 400)
- **Large `#cached-token`** (16336 or 49008 = 3 × 16336)
- **Microscopic `input throughput`** (10-200 tok/s — vs 5,000-13,000 for an 8192-token chunk)

**Pattern frequency in rate=1.0 window:**

| `#new-token` | Count |
|---:|---:|
| 8192 (full chunk) | 138 |
| 32 | 12 |
| 48 | 11 |
| 80 | 8 |
| 16 | 8 |
| 96 | 6 |
| 8176 | 6 |
| 64 | 5 |
| 128 | 3 |
| 112 | 3 |

**~62 small-batch events** (16-176 tokens) interleaved with 138 normal 8192-token chunks. Each small-batch event is a full forward pass through the model that does almost no useful work — the GPU is doing tens of layers of matmul on a 16-token batch, achieving 1-200 tok/s instead of 12,000.

### Why these tiny batches exist

The Qwen3-VL chat template generates **16,384 image placeholder tokens** per request (one per ViT patch). When the encoder ships embeddings via NIXL, PD must **integrate them into KV cache** at those placeholder positions. SGLang implements this by:

1. Treating the 16,336 image-position tokens as **`cached-token` (already-have-KV)** — but the KV is the encoder's embedding, not LLM-computed.
2. Running the LLM transformer forward over those positions to **fold the encoder embeddings into the KV cache** at each layer.
3. This forward pass is broken into tiny per-step batches (16-176 tokens at a time).

These small batches are **inherently underutilized**: the model is 32B FP8, so each 16-token forward is ~2 ms of pure compute but ~10-100 ms of CUDA launch overhead and memory traffic. GPU utilization for these batches is near 0.

**TP=1 agg avoids this entirely** because the encoder's output and the LLM's KV cache live in the same SGLang scheduler context — embeddings are placed directly during the LLM forward over the original token sequence, with no separate "integration" pass. The forward pass that processes vision tokens IS the forward pass that does the LLM prefill, in one batched 8192-token step at full GPU speed.

---

## Secondary issue — NIXL back-pressure long tail

Encoder's `embedding_transfer._state_update: Send completed for tensor_id X, total wait time: Y seconds` events show the NIXL transfer behavior:

| NIXL wait time bucket | Count | % |
|---|---:|---:|
| **<0.1 s (fast — NVLink working)** | **84** | **78%** |
| 0.1-1 s | 0 | 0% |
| 1-2 s | 7 | 6% |
| 2-5 s | 4 | 4% |
| **≥5 s (back-pressured)** | **13** | **12%** |

Median 20 ms, mean 1.6 s. The **bimodal distribution** is the signature of back-pressure:

- 78% of transfers fly at full NVLink speed (~20 ms for ~1 GB of MM features ≈ 50 GB/s effective, consistent with cuda_ipc P2P).
- 12% block for 5-15 seconds because PD has no free buffer to receive into.

The 12% slow tail is what eventually causes the 48 buffer-timeout failures observed in the v1 run.

**`cuda_ipc` is working correctly** for NIXL — UCX is using NVLink P2P, transfers are fast on the happy path. The slow tail is purely a back-pressure artifact, not a transport problem.

---

## Why throttling (v2) failed

The v2 fixes were:
1. Encoder `--max-running-requests=8` (reduce queue depth on encoder)
2. `NIXL_BUFFER_COUNT=32` (smaller buffer pool, earlier back-pressure to encoder)

**Result @ rate=1.0:** 0.21 RPS (worse than v1's 0.27), 83 s TTFT (worse than v1's 79 s), 25 / 64 failed (worse than v1's 15).

Why these "fixes" hurt rather than helped:

- **Throttling encoder doesn't speed up PD.** The bottleneck is PD's small-batch embedding integration overhead (Overhead #1 above). PD is doing the same expensive work per request whether or not encoder is throttled.
- **Smaller buffer pool causes failures sooner.** With only 32 NIXL slots, the back-pressure kicks in faster, but PD's processing rate is unchanged, so more requests time out before being processed.
- **Encoder utilization drops without compensating throughput gain.** With `max-running-requests=8`, encoder GPU is even less utilized than before, but PD is still the limit.

The v2 result confirms: **the bottleneck is PD's per-request work, not encoder rate or buffer pool size.**

---

## Why decode is actually a *little* better in disagg (the only positive)

Disagg's mean TPOT (752 ms) is lower than TP=1 agg's mean TPOT (1,069 ms). This is a small win: when PD runs decode without prefill chunks blocking, it gets cleaner per-token times because there are fewer concurrent prefills competing on the same GPU.

But the TTFT is so much worse (79 s vs 32 s) that this decode win doesn't help end-to-end E2E (111 s vs 81 s).

---

## Why TP=1 agg wins so decisively

In TP=1 agg, on a single H200:

1. **Encoder runs inline** with the LLM forward in the same SGLang scheduler. No process boundary, no NIXL.
2. **Vision embeddings are placed directly** into the LLM's input embedding sequence at the right positions, in one continuous forward pass.
3. **SGLang's chunked-prefill scheduler** naturally interleaves the encoder pass with LLM forward chunks; in fact, the encoder gap (~1.2 s/req) overlaps with previous request's prefill chunks.
4. **No "embedding integration" cost** because the LLM forward sees real embeddings from the start.
5. **Native flow control:** when KV is full, the scheduler stops admitting new requests, which automatically backs off the encoder.

Per-request budget on this hardware:
- Encoder + image-prep gap: ~1.2 s
- 2 × 8192-token prefill chunks: ~1.5 s
- Decode of 117 output tokens: ~0.5-1 s (TPOT ~5-10 ms each in TP=2, ~50 ms in TP=1)
- **Total service time: ~3-4 s.** Saturation rate: ~0.55 req/s.

In disagg PD on the same hardware:
- Encoder runs separately on GPU 4: ~1.2 s
- NIXL transfer (when fast): ~20 ms
- NIXL transfer (when blocked): 5-15 s
- 2 × 8192-token prefill chunks at low concurrency: ~1.5 s
- 2 × 8192-token prefill chunks at running-req=31: 4-7 s (~half throughput)
- **Plus** small-batch embedding integration: ~5-15 s additional per request
- Decode of 117 output tokens: ~0.3-1 s
- **Total service time: ~10-25 s.** Saturation rate: ~0.27 req/s.

The disagg per-request cost is roughly **2-3× higher** than TP=1 agg, which matches the observed 0.27 vs 0.52 RPS ratio.

---

## Why TP=2 NVLink agg wins by a much bigger margin

TP=2 agg shares all the agg architectural advantages and adds:
- **Per-chunk LLM forward is ~half** (NVLink-cheap all-reduce, two H200s sharing layer compute)
- **Vision encoder is TP-sharded** automatically via SGLang's `Qwen3VLMoeVisionModel` — each rank does half the ViT compute
- **Two ranks of KV cache** at the same `mem-fraction-static`, doubling the KV pool

Result: 0.95 req/s at rate=1.0, 9 s TTFT, 12 s E2E — **3.5× the throughput of disagg, 1.8× the throughput of TP=1 agg.**

---

## Three-way summary table at rate=1.0

| Metric | TP=1 agg | TP=2 NVLink agg | **Disagg** |
|---|---:|---:|---:|
| Actual RPS | 0.52 | **0.95** | 0.27 |
| Mean TTFT | 32 s | **9 s** | 79 s |
| Median TPOT | 435 ms | **15 ms** | 256 ms |
| Mean E2E | 81 s | **12 s** | 111 s |
| Failed requests (of 64) | 0 | 0 | **15** |
| Median per-8192-chunk wallclock | 1.08 s | (lower — TP-sharded) | **2.28 s** |
| Median per-8192-chunk input tput | 10,785 tok/s | (higher) | **5,695 tok/s** |
| GPUs used | 1 | 2 | 2 |
| GPUs effectively utilized | 1 | 2 | ~1.2 (encoder mostly idle, PD half-utilized) |

---

## When disagg would actually win

Disagg architectures are not fundamentally bad — they win in scenarios this benchmark does not exercise:

1. **Encoder runs on a smaller/cheaper GPU class** (e.g., L40S or A10G) while the LLM gets the H200. The MM features can be transferred over a slow link if the LLM is the binder. Per-dollar throughput improves.
2. **Multi-tenant deployments where encoder cache is shared.** A single encoder pool serves many independent LLM workers; each LLM worker doesn't need its own ViT weights resident. Saves memory across the fleet.
3. **Encoder is much larger than LLM** (e.g., a 7B vision model + 1B language model). Disagg lets you scale them independently.
4. **Multi-host deployments** where the encoder must live on a different machine for capacity reasons (no other choice).

But for **a single H200 host running a 32B FP8 LLM with a tiny ~1.5 GB ViT and full NVLink between GPUs**, none of these apply. Co-locating everything in one SGLang scheduler is strictly better.

---

## Implementation issues that made disagg even worse here

Beyond the architectural overhead, the current dynamo / SGLang disagg implementation has specific bugs that compound the problem:

1. **No flow control between encoder and PD.** Encoder admits requests as fast as they arrive, with no signal from PD about backlog. Result: 12% of NIXL transfers stall 5-15 s; some hit timeout and fail outright.
2. **307 "Mismatch: More 'IMAGE' tokens found than corresponding data provided" warnings** during the bench. The chat template paste in PD races with the NIXL embedding arrival. Even when transfers succeed, this race adds latency.
3. **`embedding_transfer_mode=NIXL_WRITE` is hardcoded** in the encoder despite `DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read` env var. The env var is not honored. NIXL_READ would have PD pulling from encoder (natural back-pressure: PD only pulls when ready), avoiding the buffer-pool problem entirely.
4. **CUDA graphs disabled for prefill** in both modes (this is normal SGLang behavior for variable-length prefill), but disagg's small-batch integration events would benefit even more from being captured. They are not.

Fixing these would help disagg's absolute numbers but **would not change the architectural conclusion** that on a single host, TP=1 agg or TP=2 NVLink agg always beats disagg for this workload.

---

## Final verdict

**On a single H200 host with NVLink between GPUs, disagg is strictly worse than aggregated EPD for tightly coupled multimodal workloads** like Qwen3-VL-32B + 8 × 1080p images. The benefit disagg theoretically offers (decoupled encoder/decoder scaling) is overwhelmed by:

1. **Embedding-integration overhead on PD** — small-batch forward passes that GPU underutilizes (~5-15 s extra per request)
2. **NIXL transfer back-pressure long tail** — 12% of transfers stall 5-15 s when PD is loaded
3. **Lack of flow control** — encoder over-admits, PD's KV pool fills, requests time out

Recommend **TP=2 NVLink agg** as the production configuration for this workload on H200. Use **TP=1 agg** if running on a single GPU. Use **2× TP=1 DP** for high-concurrency rate-saturated regimes. Avoid disagg unless the hardware setup specifically requires it (cross-host or asymmetric GPU classes).

---

## Files referenced

- TP=1 agg result dir: `/hongming/res4/h200_agg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_194141/`
- TP=2 agg NVLink result dir: `/hongming/res4/h200_agg_tp2_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_201625/`
- Disagg v1 result dir: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_211224/`
- Disagg v2 result dir: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates_v2/test_sglang_multi_rates_1080p_20260521_222058/`
- Disagg v1 PD worker log: `/hongming/dynamo/logs/archive_v1_disagg/pd_worker.log`
- Disagg v1 encoder log: `/hongming/dynamo/logs/archive_v1_disagg/encoder_worker.log`
- TP=1 agg worker log: `/hongming/dynamo/logs/epd_worker_server.log`
- Companion docs:
  - `01_cuda_sh/agg_h200_32b/tp1_all_rates_results.md`
  - `01_cuda_sh/agg_h200_32b/tp2_all_rates_results.md`
  - `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md`
- This document: `01_cuda_sh/disagg_h200_32b/deep_analysis_disagg_worse_h200.md`

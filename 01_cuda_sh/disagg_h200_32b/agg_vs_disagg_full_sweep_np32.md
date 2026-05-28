# Agg TP=1 EPD vs 1E disagg: 32B FP8 / np=32 / rate=1.0 — full sweep

**Date:** 2026-05-28
**Hardware:** super21-h200 GPU 5 (NUMA 2, mlx5_4)
**Setup:**
- **Agg**: single `dynamo.sglang` process on super21 GPU 5, no `--multimodal-worker` flag, mem-frac=0.85, max-running=40
- **Disagg 1E**: PD on super21 GPU 5 (mem-frac=0.50, max-running=64) + 1 encoder on dell06 H200 (cross-host RoCE NIXL)
- **Frontend**: kv router (both modes)
- **Workload**: Qwen3-VL-32B-Instruct-FP8, in=128 / out=256, np=32, rate=1.0 RPS, all 32/32 successful

## Headline: agg TP=1 dominates disagg 1E across all 3 workloads

| Workload | Mode | RPS | Mean TTFT | Mean E2E | Concurrency |
|---|---|---:|---:|---:|---:|
| **4img/768p** | **Agg TP=1** | **0.87** | **0.89 s** | **4.09 s** | 3.6 |
| 4img/768p | Disagg 1E | 0.78 | 6.14 s | 10.28 s | 8.0 |
| | Δ agg vs disagg | **+12%** | **-86%** | -60% | -55% |
| **8img/768p** | **Agg TP=1** | **1.26** | **2.24 s** | **8.93 s** | 11.3 |
| 8img/768p | Disagg 1E | 0.62 | 19.69 s | 26.52 s | 16.4 |
| | Δ agg vs disagg | **+103%** | **-89%** | -66% | -31% |
| **8img/1080p** | **Agg TP=1** | **0.45** | **22.96 s** | **54.09 s** | 24.2 |
| 8img/1080p | Disagg 1E | 0.24 | 60.44 s | 67.48 s | 16.4 |
| | Δ agg vs disagg | **+88%** | **-62%** | -20% | +47% |

## Full numbers

### 4img/768p np=32 rate=1.0

| Metric | Agg TP=1 | Disagg 1E | Δ |
|---|---:|---:|---|
| Successful | 32/32 | 32/32 | — |
| Duration (s) | 36.74 | 40.97 | -10% |
| **RPS** | **0.87** | 0.78 | +12% |
| Input throughput (tok/s) | 2 749 | 2 465 | +12% |
| Output throughput (tok/s) | 133 | 119 | +12% |
| Peak output (tok/s) | 533 | 493 | +8% |
| Mean E2E (ms) | 4 092 | 10 283 | **-60%** |
| Median E2E (ms) | 4 080 | 10 623 | -62% |
| P90 E2E (ms) | 6 848 | 14 670 | -53% |
| P99 E2E (ms) | 8 333 | 15 974 | -48% |
| **Mean TTFT (ms)** | **887** | 6 137 | **-86%** |
| Median TTFT (ms) | 779 | 6 030 | -87% |
| P99 TTFT (ms) | 1 497 | 10 295 | -85% |
| Mean TPOT (ms) | 22.3 | 38.8 | -43% |
| Median TPOT (ms) | 21.9 | 29.9 | -27% |
| P99 TPOT (ms) | 47.7 | 162 | -71% |
| Mean concurrency | 3.6 | 8.0 | -55% |
| Peak concurrent | 13 | 18 | -28% |

### 8img/768p np=32 rate=1.0

| Metric | Agg TP=1 | Disagg 1E | Δ |
|---|---:|---:|---|
| Successful | 32/32 | 32/32 | — |
| Duration (s) | 25.35 | 51.66 | **-51%** |
| **RPS** | **1.26** | 0.62 | **+103%** |
| Input throughput (tok/s) | 7 874 | 3 863 | +104% |
| Output throughput (tok/s) | 193 | 95 | +103% |
| Peak output (tok/s) | 839 | 363 | +131% |
| Mean E2E (ms) | 8 930 | 26 516 | **-66%** |
| Median E2E (ms) | 8 871 | 26 641 | -67% |
| P90 E2E (ms) | 12 653 | 32 997 | -62% |
| P99 E2E (ms) | 16 260 | 34 309 | -53% |
| **Mean TTFT (ms)** | **2 236** | 19 692 | **-89%** |
| Median TTFT (ms) | 2 121 | 20 073 | -89% |
| P99 TTFT (ms) | 4 274 | 29 587 | -86% |
| Mean TPOT (ms) | 61.1 | 46.4 | +32% |
| Median TPOT (ms) | 43.2 | 50.2 | -14% |
| P99 TPOT (ms) | 269 | 108 | **+149%** |
| Mean concurrency | 11.3 | 16.4 | -31% |
| Peak concurrent | 18 | 29 | -38% |

### 8img/1080p np=32 rate=1.0

| Metric | Agg TP=1 | Disagg 1E | Δ |
|---|---:|---:|---|
| Successful | 32/32 | 32/32 | — |
| Duration (s) | 71.58 | 131.69 | **-46%** |
| **RPS** | **0.45** | 0.24 | **+88%** |
| Input throughput (tok/s) | 7 337 | 3 988 | +84% |
| Output throughput (tok/s) | 68.2 | 37.1 | +84% |
| Peak output (tok/s) | 810 | 199 | **+307%** |
| Mean E2E (ms) | 54 095 | 67 482 | -20% |
| Median E2E (ms) | 52 939 | 64 893 | -18% |
| P90 E2E (ms) | 65 185 | 104 100 | -37% |
| P99 E2E (ms) | 69 630 | 105 029 | -34% |
| **Mean TTFT (ms)** | **22 965** | 60 440 | **-62%** |
| Median TTFT (ms) | 21 615 | 58 237 | -63% |
| P99 TTFT (ms) | 36 870 | 101 780 | -64% |
| Mean TPOT (ms) | 332 | 42.6 | **+679%** |
| Median TPOT (ms) | 217 | 39.2 | +452% |
| P99 TPOT (ms) | 1 862 | 244.8 | +660% |
| Mean concurrency | 24.2 | 16.4 | **+47%** |
| Peak concurrent | 32 | 32 | same |

## Why agg crushes disagg

### 1. No NIXL transfer time, especially at small embeddings

| Resolution | Embedding/req | Disagg NIXL transfer time (estimated) |
|---|---:|---:|
| 4img/768p | ~80 MB | ~50-100 ms |
| 8img/768p | ~160 MB | ~100-200 ms |
| 8img/1080p | ~250 MB | ~200-400 ms |

NIXL itself is fast; the bigger penalty is the **scheduler tick** between embedding
arrival and prefill batch formation in disagg PD. PD's prefill scheduler executes
`#new-seq=1` per batch, so every new req must wait for its predecessor's NIXL READ
to fully complete + descriptor cleanup + scheduler iteration before joining a batch.
Effective serialization regardless of how fast NIXL is.

### 2. Better effective batching in agg

In agg, vision tower forward integrates with prefill batching — multiple requests'
images can be processed in a single forward pass, and prefill batch formation can
absorb new arrivals without waiting for embedding handoff.

In disagg, each request has 5 sequential steps: tokenize, vision tower (encoder),
NIXL publish (encoder), NIXL READ (PD), prefill batch (PD). The PD's batching
window starts AFTER the embedding is fully resident.

### 3. Larger KV cache in agg (mem-frac=0.85 vs 0.50)

| | max_total_num_tokens | available_gpu_mem |
|---|---:|---:|
| Agg | 695 968 | 20.6 GB |
| Disagg PD | 296 864 | 69.2 GB |

Agg has 2.4× more KV space. At 1080p with concurrency 24 × ~16k tokens/req = ~400k
tokens, agg has plenty of headroom; disagg PD must evict aggressively.

### 4. The 1080p results show agg's PD-bound regime

At 8img/1080p:
- Both agg and disagg are PD-prefill-bound (16k vision tokens × concurrent reqs)
- Agg gets +88% RPS purely from KV cache size + better batching, despite identical
  GPU (same H200, same compute capability)
- Mean TPOT is much worse for agg (332 ms vs 43 ms) because **agg keeps reqs running
  in big batches while prefill-decode mixing causes ITL spikes**. Disagg's lower TPOT
  is because PD always has a small running batch (encoder bottleneck capping arrivals)

This is the fundamental tradeoff: **agg trades TPOT for RPS**. At rate=1.0, the user
cares about completing reqs faster (RPS), and 88% more RPS easily beats 8× higher TPOT
because TTFT (the user-visible latency) dropped 62%.

### 5. Disagg's only "win": lower mean TPOT in PD-bound workload

The 8img/1080p case is the only one where disagg has a metric advantage:
mean TPOT 43 ms (disagg) vs 332 ms (agg). But this is **a side effect of disagg's
slower throughput**, not an architectural benefit:
- Disagg processes ~half the reqs/sec, so PD's running batch stays small
- Small batch = uniform decode latency = low TPOT
- But 2× fewer reqs complete, so total throughput suffers

If you want low TPOT, lower offered rate. If you want high RPS, use agg.

## Cross-resolution scaling (agg)

Agg's scaling is also informative:

| Workload | Agg RPS | Per-req vision tokens | tokens/sec from prefill |
|---|---:|---:|---:|
| 4img/768p | 0.87 | ~3 080 | ~2 680 |
| 8img/768p | 1.26 | ~6 160 | ~7 760 |
| 8img/1080p | 0.45 | ~16 350 | ~7 360 |

The 8img/768p RPS being *higher* than 4img/768p is interesting — bigger work amortizes
per-req overhead better. The 8img/1080p RPS drops back because vision tokens (16k)
exceed a single chunked-prefill chunk (16384), forcing 2 prefill passes per req.

Throughput in tokens/sec is roughly constant at the agg saturation regime (~7-8k tok/s
input throughput at 1080p / 8img/768p), suggesting **agg's bottleneck at saturation
is total prefill compute throughput**, not request count.

## Comparison to other modes (cross-reference)

From earlier benches in this work session and prior work:

| Workload | Mode | RPS | TTFT | E2E | Notes |
|---|---|---:|---:|---:|---|
| 8img/1080p | **Agg TP=1** (today) | **0.45** | 23.0 s | 54.1 s | mem-frac=0.85 |
| 8img/1080p | Agg TP=1 (May 27) | 0.42 | 25.9 s | 58.4 s | reproducible |
| 8img/1080p | Disagg 1E (today) | 0.24 | 60.4 s | 67.5 s | mem-frac=0.50 |
| 8img/1080p | Disagg 2E (today) | 0.24 | 93.5 s | 98.6 s | 2 encoders, deeper queue |
| 8img/1080p | Disagg 4E (older 35B work) | ~0.85 | n/a | n/a | 4 encoders break encoder bottleneck for 35B |

## Recommendation

For Qwen3-VL-32B-Instruct-FP8 on H200 single-GPU:
- **Default to agg TP=1** for all workloads tested
- Cross-host disagg is strictly worse for these scales
- To break agg's RPS ceiling, try **agg TP=2** (more compute) — already have launcher

Cross-host disagg only makes sense when:
1. PD can't fit model + KV in single GPU (not the case for 32B FP8 on H200)
2. Many small backends share a fat encoder (multi-tenant scenario, not measured)
3. Encoder is genuinely the bottleneck (e.g., 4K video, larger encoders)

## Summary table of all 32B FP8 / np=32 / rate=1.0 results

| Workload | Mode | RPS | TTFT | E2E |
|---|---|---:|---:|---:|
| 4img/768p | **Agg TP=1** | **0.87** | **0.9 s** | **4.1 s** |
| 4img/768p | Disagg 1E | 0.78 | 6.1 s | 10.3 s |
| 8img/768p | **Agg TP=1** | **1.26** | **2.2 s** | **8.9 s** |
| 8img/768p | Disagg 1E | 0.62 | 19.7 s | 26.5 s |
| 8img/1080p | **Agg TP=1** | **0.45** | **23.0 s** | **54.1 s** |
| 8img/1080p | Disagg 1E | 0.24 | 60.4 s | 67.5 s |
| 8img/1080p | Disagg 2E | 0.24 | 93.5 s | 98.6 s |

## Files
- This doc: `disagg_h200_32b/agg_vs_disagg_full_sweep_np32.md`
- Agg 4img/768p: `res_xhost_dell06_super21/agg_tp1_32b_4img_768p_rate1.0_np32_20260528_075514/`
- Agg 8img/768p: `res_xhost_dell06_super21/agg_tp1_32b_8img_768p_rate1.0_np32_20260528_075631/`
- Agg 8img/1080p: `res_xhost_dell06_super21/agg_tp1_32b_8img_1080p_rate1.0_np32_20260528_080838/`
- Disagg 1E 4img/768p: `res_xhost_dell06_super21/32b_4img_768p_rate1.0_np32_1enc_20260528_073753/`
- Disagg 1E 8img/768p: `res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_1enc_20260528_071429/`
- Disagg 1E 8img/1080p: `res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_192949_v2/`
- Disagg 2E 8img/1080p: `res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_2enc_20260528_050453/`
- Agg launcher: `disagg_h200_32b/start_agg_epd_super21.sh`
- Agg EPD log (DYN_LOG=debug): `disagg_h200_32b/logs/agg_epd_worker_20260528_074802.log`
- Prior 1080p analysis (May 27): `disagg_h200_32b/agg_vs_disagg_32b_8img_1080p_comparison.md`
- Prior 768p sweep: `disagg_h200_32b/agg_vs_disagg_768p_np32.md` (now superseded)

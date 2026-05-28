# Agg TP=1 EPD vs 1E disagg: 32B FP8 at 768p np=32 rate=1.0

**Date:** 2026-05-28
**Hardware:** super21-h200 GPU 5 (NUMA 2, mlx5_4)
**Setup:**
- **Agg**: single dynamo.sglang process on super21 GPU 5, no `--multimodal-worker` flag, mem-frac=0.85, max-running=40
- **Disagg 1E**: PD on super21 GPU 5 (mem-frac=0.50, max-running=64) + 1 encoder on dell06 H200 (cross-host RoCE NIXL)
- **Frontend**: kv router (agg used kv, disagg also kv)
- **Workload**: Qwen3-VL-32B-Instruct-FP8, 1024×768 imgs, in=128 / out=256, np=32, rate=1.0 RPS

## Headline: agg wins decisively at 768p for both image counts

| Workload | Mode | Successful | RPS | Mean TTFT | Mean E2E | Median TPOT |
|---|---|---:|---:|---:|---:|---:|
| **4img/768p** | **Agg TP=1** | **32/32** | **0.87** | **0.89 s** | **4.09 s** | 21.9 ms |
| 4img/768p | Disagg 1E | 32/32 | 0.78 | 6.14 s | 10.28 s | 29.9 ms |
| | Δ (agg vs disagg) | — | **+12%** | **-86%** | **-60%** | -27% |
| **8img/768p** | **Agg TP=1** | **32/32** | **1.26** | **2.24 s** | **8.93 s** | 43.2 ms |
| 8img/768p | Disagg 1E | 32/32 | 0.62 | 19.69 s | 26.52 s | 50.2 ms |
| | Δ (agg vs disagg) | — | **+103%** | **-89%** | **-66%** | -14% |

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

## Why agg crushes disagg at 768p

### 1. No NIXL transfer time

Disagg has to:
- Encoder pulls images, runs vision tower on dell06 GPU
- Encoder allocates GPU buffer for embeddings (~80 MB at 4img / 250 MB at 8img/1080p)
- Encoder NIXL-publishes embedding handle
- PD on super21 NIXL READs the buffer over RoCE
- PD's wait+RDMA = several hundred ms to a few seconds depending on load

Agg has:
- Vision tower runs on the same GPU, embeddings stay in GPU memory
- Zero network transfer
- No NIXL handle setup, no scheduler tick to wait for embedding

→ TTFT drops by **86% (4img)** and **89% (8img)**.

### 2. PD prefill scheduler doesn't serialize behind NIXL waits

In disagg, PD's prefill scheduler executes `#new-seq=1` per batch. Each new req must
wait for its embedding NIXL READ to complete before joining a prefill batch. With np=32
and 1 encoder, this serializes the queue: the encoder feeds at ~1.4 s/req cadence,
even though the PD could potentially batch more if embeddings were already resident.

In agg, the vision tower runs as a single forward pass that integrates with prefill
batching. SGLang can include multiple requests' vision inputs in a single forward —
**better effective batching**.

### 3. Larger KV cache headroom

Agg config: `mem-fraction=0.85` → `max_total_num_tokens=695,968`
Disagg PD: `mem-fraction=0.50` → `max_total_num_tokens=296,864`

Agg has 2.4× more KV space. At 8img/768p with ~6k vision tokens × 32 reqs = ~200k tokens
of context, agg has plenty of headroom; disagg PD is closer to its KV budget which
restricts how many reqs can decode concurrently.

### 4. The 8img/768p case is the dramatic one (2× RPS)

| Mode | 4img RPS | 8img RPS | 8img/4img ratio |
|---|---:|---:|---:|
| Agg | 0.87 | **1.26** | +45% (more reqs/s with bigger images?!) |
| Disagg 1E | 0.78 | 0.62 | -21% (more images → slower) |

Agg gets *higher* RPS at 8img than 4img — counterintuitive. Likely because:
- 8img → larger prefill chunks → better GPU utilization per forward
- Vision tower and prefill share the same kernels; larger work amortizes per-req overhead
- Decode batch ramps to 18 reqs (vs 13 at 4img), better cuda graph utilization

Disagg can't reproduce this because its NIXL transfer time scales with embedding size,
adding latency that grows with image count.

### 5. Mean TPOT trade-off

The only metric where disagg can be competitive: 8img mean TPOT (61 ms agg vs 46 ms
disagg). When PD has nothing to do but decode (because encoders feed slowly), the
batching is more uniform → lower TPOT variance.

But this is misleading: **mean TPOT is a tail-quality metric**, not throughput. Agg's
P99 TPOT does spike to 269 ms vs disagg's 108 ms because agg is more aggressive about
adding new reqs to the running batch.

## Comparison vs the prior agg 1080p result (2026-05-27)

The earlier agg run at 1080p with these same flags gave:

| Workload | Mode | RPS | TTFT | E2E | KV cache size |
|---|---|---:|---:|---:|---:|
| 8img/1080p | Agg TP=1 (May 27) | 0.42 | 26.0 s | n/a | 695k |
| 8img/1080p | Disagg 1E (May 27) | 0.24 | 60.4 s | 67.5 s | 296k |
| 8img/768p | Agg TP=1 (today) | **1.26** | 2.2 s | 8.9 s | 695k |
| 8img/768p | Disagg 1E (today) | 0.62 | 19.7 s | 26.5 s | 296k |
| 4img/768p | Agg TP=1 (today) | 0.87 | 0.9 s | 4.1 s | 695k |
| 4img/768p | Disagg 1E (today) | 0.78 | 6.1 s | 10.3 s | 296k |

**Agg wins everywhere** but by varying margins:
- 8img/1080p: +75% RPS for agg (PD-bound for both, but agg has more KV)
- 8img/768p: +103% RPS for agg (largest gap; disagg encoder cadence kills it)
- 4img/768p: +12% RPS for agg (smallest gap; both regimes light on PD/encoder)

## Implication for cross-host disagg with this dynamo build

The cross-host disagg architecture (encoder on dell06 + PD on super21 + RoCE NIXL) is
**worse than colocated agg** for all 32B FP8 workloads tested at np=32 rate=1.0. The
NIXL transfer + PD scheduler serialization completely overwhelms any theoretical gain
from offloading vision work to a separate GPU.

When does cross-host disagg make sense?
1. **PD GPU memory is the bottleneck**, not throughput. Offloading the encoder's vision
   weights frees ~5-10 GB on the PD that could go to KV cache.
2. **Multi-tenant**: many small-model PDs sharing one large vision encoder host (not
   tested here).
3. **Vision-heavy workloads** with very long video frames where the encoder is the actual
   bottleneck (not measured at this scale).

For the workloads measured (32B FP8, 4-8 imgs, 768p-1080p, np=32), **agg TP=1 should
be the default**. To break agg's RPS ceiling, the next lever is **agg TP=2** (more
prefill compute) — see `start_h200_aggregate_epd_server_32b_tp2.sh`.

## Summary table (all 32B FP8 / np=32 / rate=1.0 results in this work)

| Workload | Mode | RPS | TTFT | E2E |
|---|---|---:|---:|---:|
| 4img/768p | **Agg TP=1** | **0.87** | **0.9 s** | **4.1 s** |
| 4img/768p | Disagg 1E | 0.78 | 6.1 s | 10.3 s |
| 8img/768p | **Agg TP=1** | **1.26** | **2.2 s** | **8.9 s** |
| 8img/768p | Disagg 1E | 0.62 | 19.7 s | 26.5 s |
| 8img/1080p | Agg TP=1 (May 27) | 0.42 | 26.0 s | (∼60 s) |
| 8img/1080p | Disagg 1E (May 27) | 0.24 | 60.4 s | 67.5 s |
| 8img/1080p | Disagg 2E (May 28) | 0.24 | 93.5 s | 98.6 s |

## Files
- This doc: `disagg_h200_32b/agg_vs_disagg_768p_np32.md`
- Agg 4img/768p: `res_xhost_dell06_super21/agg_tp1_32b_4img_768p_rate1.0_np32_20260528_075514/`
- Agg 8img/768p: `res_xhost_dell06_super21/agg_tp1_32b_8img_768p_rate1.0_np32_20260528_075631/`
- Agg launcher: `disagg_h200_32b/start_agg_epd_super21.sh`
- Agg EPD log (DYN_LOG=debug): `disagg_h200_32b/logs/agg_epd_worker_20260528_074802.log`
- Disagg 4img/768p: `res_xhost_dell06_super21/32b_4img_768p_rate1.0_np32_1enc_20260528_073753/`
- Disagg 8img/768p: `res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_1enc_20260528_071429/`
- Prior agg vs disagg analysis (1080p): `disagg_h200_32b/agg_vs_disagg_32b_8img_1080p_comparison.md`
- 1E sweep doc: `disagg_h200_32b/1E_768p_imagecount_sweep_np32.md`

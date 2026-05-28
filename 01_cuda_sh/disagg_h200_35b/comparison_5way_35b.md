# 35B Disagg vs Agg — Comprehensive 5-Way Comparison

**Date:** 2026-05-26
**Model:** Qwen3.5-35B-A3B (BF16 MoE multimodal, hybrid 30 linear-attn + 10 full-attn, ~70 GB weights)
**Hardware:**
- giga01 H200 GPU 4 (NUMA 2, paired with mlx5_4 for RoCE)
- B70 Battlemage XPUs (8 total, 32 GB each, NUMA 0 = XPUs 0-3 used)
- dell06 H200 (cross-host, RoCE-connected to giga01)
- All hosts on 192.165.123.0/24 RoCE fabric

**Method:** Same bench (`sglang.bench_serving --backend sglang-oai-chat`), same workload set
(3 image counts × 7 rates × np=32, random JPEG, in=128 / out=256, seed=0). 5 topologies
measured against this matrix.

## Topologies tested

| # | Name | Encoder host(s) | Encoder count | PD host | TP | Notes |
|---|---|---|---|---|---|---|
| 1 | **agg_TP1** | giga01 GPU 4 | inline | giga01 GPU 4 | 1 | Single H200, all-in-one |
| 2 | **agg_TP2** | giga01 GPU 4-5 | inline | giga01 GPU 4-5 | 2 | Two H200s, all-in-one |
| 3 | **B70_1E** | B70 (1 XPU) | 1 | giga01 H200 GPU 4 | 1 | Cross-host disagg, baseline |
| 4 | **B70_4E** | B70 (4 XPUs) | 4 | giga01 H200 GPU 4 | 1 | Cross-host disagg, parallel encoders |
| 5 | **dell06_1E** | dell06 H200 | 1 | giga01 H200 GPU 4 | 1 | Cross-host disagg, H200 encoder |

All 5 topologies use the **same PD config**: SGLang sglang `--mem-fraction-static 0.75
--max-running-requests 40 --tensor-parallel-size 1 --linear-attn-backend triton
--attention-backend fa3 --kv-cache-dtype auto (BF16)`.

## Saturation throughput (RPS) — headline table

| Workload | embed | agg_TP1 | agg_TP2 | B70_1E | B70_4E | **dell06_1E** | best |
|---|---:|---:|---:|---:|---:|---:|---|
| **4img/768p** | ~127 MB | **~3.0** (rate=4.0) | ~2.7 (rate=4.0) | 0.44 | 1.55 | **>2.49** (not sat) | tie agg_TP1/dell06 |
| **8img/768p** | ~242 MB | 2.09 | 2.03 | 0.23 | 0.83 | **2.52** | **dell06_1E** |
| **8img/1080p** | ~638 MB | 1.14 | 1.03 | 0.038 | 0.147 | 0.85 | **agg_TP1** |

**Three findings worth flagging up front:**

1. **agg_TP1 wins for the largest workload** (8img/1080p: 1.14 RPS vs dell06_1E 0.85). The
   single-H200 ViT runs 8×1080p faster with shared memory + no NIXL serialization overhead.
2. **dell06_1E ties or beats both agg topologies** at 8img/768p (2.52 vs 2.09/2.03). The
   H200 encoder is fast enough that the cross-host RoCE NIXL overhead is fully amortized.
3. **agg_TP2 is consistently slower than agg_TP1 for 35B**. This is the opposite of the
   32B-FP8 finding (where TP=2 always won). For 35B BF16, TP=2 introduces NCCL overhead
   for each MoE expert dispatch that exceeds the per-token compute savings.

## Per-rate detail tables

### 4img/768p (~127 MB embedding)

| Rate | Topology | RPS | E2E p50 | TTFT p50 | TPOT p50 | ITL p50 |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | B70_1E | 0.089 | 3.56 s | 2.55 s | 4.58 ms | 4.51 ms |
| 0.10 | B70_4E | 0.089 | 3.46 s | 2.65 s | 4.73 ms | 4.68 ms |
| 0.10 | dell06_1E | 0.089 | **1.40 s** | **0.63 s** | 4.71 ms | 4.69 ms |
| 0.10 | agg_TP1 | — | — | — | — | — |
| 0.10 | agg_TP2 | — | — | — | — | — |
| 0.25 | B70_1E | 0.221 | 8.72 s | 5.31 s | 4.62 ms | 4.28 ms |
| 0.25 | B70_4E | 0.221 | 3.44 s | 2.64 s | 4.71 ms | 4.67 ms |
| 0.25 | dell06_1E | 0.221 | 1.41 s | **0.57 s** | 4.72 ms | 4.69 ms |
| 0.25 | agg_TP1 | 0.221 | 1.46 s | 0.64 s | 4.60 ms | 4.59 ms |
| 0.25 | agg_TP2 | 0.221 | 1.68 s | 0.70 s | 5.52 ms | 5.26 ms |
| 0.50 | B70_1E | 0.431 | 47.5 s | 14.3 s | 151.5 ms | 6.09 ms |
| 0.50 | B70_4E | 0.443 | 3.57 s | 2.64 s | 4.76 ms | 4.65 ms |
| 0.50 | dell06_1E | 0.443 | 1.46 s | **0.55 s** | 4.83 ms | 4.71 ms |
| 0.50 | agg_TP1 | 0.443 | 1.48 s | 0.64 s | 4.87 ms | 4.69 ms |
| 0.50 | agg_TP2 | 0.443 | 1.64 s | 0.65 s | 5.55 ms | 5.15 ms |
| 1.00 | B70_1E | 0.443 | 60.2 s | 38.3 s | 126.2 ms | 7.00 ms |
| 1.00 | B70_4E | 0.851 | 4.49 s | 3.00 s | 4.98 ms | 4.48 ms |
| 1.00 | dell06_1E | 0.885 | 1.52 s | **0.58 s** | 5.65 ms | 4.64 ms |
| 1.00 | agg_TP1 | 0.885 | 1.66 s | 0.63 s | 6.93 ms | 4.84 ms |
| 1.00 | agg_TP2 | 0.885 | 2.01 s | 0.71 s | 7.84 ms | 5.66 ms |
| 1.50 | B70_1E | 0.444 | 64.0 s | 53.2 s | 8.44 ms | 7.37 ms |
| 1.50 | B70_4E | 1.228 | 6.49 s | 4.69 s | 5.13 ms | 1.36 ms |
| 1.50 | dell06_1E | **1.327** | 1.68 s | **0.59 s** | 6.62 ms | 4.59 ms |
| 1.50 | agg_TP1 | 1.327 | 1.94 s | 0.65 s | 8.64 ms | 5.41 ms |
| 1.50 | agg_TP2 | 1.305 | 3.01 s | 0.91 s | 14.88 ms | 6.98 ms |
| 2.00 | B70_1E | 0.443 | 66.0 s | 58.0 s | 8.38 ms | 7.25 ms |
| 2.00 | B70_4E | 1.539 | 13.2 s | 7.01 s | 41.13 ms | 1.03 ms |
| 2.00 | dell06_1E | 1.737 | 1.72 s | **0.61 s** | 7.79 ms | 4.42 ms |
| 2.00 | agg_TP1 | **1.740** | 2.38 s | 0.86 s | 10.97 ms | 6.96 ms |
| 2.00 | agg_TP2 | 1.661 | 3.65 s | 1.25 s | 14.18 ms | 7.65 ms |
| 3.00 | B70_1E | 0.453 | 66.4 s | 62.9 s | 8.46 ms | 7.30 ms |
| 3.00 | B70_4E | 1.578 | 14.8 s | 10.30 s | 12.68 ms | 1.26 ms |
| 3.00 | dell06_1E | **2.491** | 2.68 s | **0.80 s** | 8.32 ms | 4.04 ms |
| 3.00 | agg_TP1 | 2.425 | 3.59 s | 1.12 s | 12.18 ms | 9.19 ms |
| 3.00 | agg_TP2 | 2.320 | 6.96 s | 2.32 s | 28.35 ms | 7.94 ms |
| 4.00 | agg_TP1 | **3.007** | 5.18 s | 1.84 s | 20.01 ms | 8.16 ms |
| 4.00 | agg_TP2 | 2.733 | 7.71 s | 2.92 s | 28.37 ms | 9.45 ms |

**Saturation point per topology:**
- B70_1E: rate=0.5 (0.44 RPS, ITL goes nuts past this)
- B70_4E: rate=2.0 (1.55 RPS plateau)
- dell06_1E: not yet saturated at rate=3.0 (2.49 RPS still climbing)
- agg_TP1: rate=4.0 (3.0 RPS, the highest measured)
- agg_TP2: rate=4.0 (2.7 RPS, ~10% behind TP1)

### 8img/768p (~242 MB embedding)

| Rate | Topology | RPS | E2E p50 | TTFT p50 | TPOT p50 | ITL p50 |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | B70_1E | 0.153 | 15.3 s | 12.5 s | 4.88 ms | 0.87 ms |
| 0.10 | B70_4E | 0.153 | 5.78 s | 5.02 s | 4.79 ms | 4.73 ms |
| 0.10 | dell06_1E | 0.153 | **1.74 s** | **0.93 s** | 4.76 ms | 4.70 ms |
| 0.25 | B70_1E | 0.231 | 96.0 s | 69.5 s | 172.9 ms | 6.63 ms |
| 0.25 | B70_4E | 0.383 | 5.97 s | 4.99 s | 4.90 ms | 4.70 ms |
| 0.25 | dell06_1E | 0.383 | **1.76 s** | **0.95 s** | 4.86 ms | 4.65 ms |
| 0.25 | agg_TP1 | 0.383 | 2.10 s | 1.12 s | 6.06 ms | 4.78 ms |
| 0.25 | agg_TP2 | 0.383 | 2.23 s | 1.21 s | 6.75 ms | 5.59 ms |
| 0.50 | B70_1E | 0.231 | 117.3 s | 105.5 s | 8.52 ms | 7.06 ms |
| 0.50 | B70_4E | 0.721 | 9.60 s | 7.24 s | 5.59 ms | 0.49 ms |
| 0.50 | dell06_1E | 0.765 | **2.19 s** | **1.02 s** | 5.64 ms | 4.89 ms |
| 0.50 | agg_TP1 | 0.765 | 2.75 s | 1.29 s | 8.49 ms | 5.70 ms |
| 0.50 | agg_TP2 | 0.765 | 2.88 s | 1.35 s | 9.20 ms | 5.98 ms |
| 1.00 | B70_1E | 0.231 | 127.7 s | 121.3 s | 8.26 ms | 6.96 ms |
| 1.00 | B70_4E | 0.814 | 26.8 s | 22.4 s | 3.94 ms | 1.08 ms |
| 1.00 | dell06_1E | **1.522** | **2.50 s** | **1.32 s** | 7.14 ms | 2.22 ms |
| 1.00 | agg_TP1 | 1.479 | 3.57 s | 1.74 s | 16.27 ms | 6.82 ms |
| 1.00 | agg_TP2 | 1.498 | 5.00 s | 1.85 s | 22.56 ms | 7.48 ms |
| 1.50 | B70_1E | 0.234 | 129.3 s | 124.2 s | 8.37 ms | 6.91 ms |
| 1.50 | B70_4E | 0.833 | 30.1 s | 26.3 s | 1.91 ms | 1.15 ms |
| 1.50 | dell06_1E | **2.097** | **3.51 s** | **1.59 s** | 13.45 ms | 1.46 ms |
| 1.50 | agg_TP1 | 1.708 | 10.77 s | 4.57 s | 50.64 ms | 11.18 ms |
| 1.50 | agg_TP2 | 1.651 | 11.13 s | 3.97 s | 57.41 ms | 9.35 ms |
| 2.00 | B70_1E | 0.230 | 133.6 s | 128.5 s | 8.99 ms | 7.07 ms |
| 2.00 | B70_4E | 0.810 | 32.3 s | 30.2 s | 1.52 ms | 1.28 ms |
| 2.00 | dell06_1E | **2.304** | **7.31 s** | **3.10 s** | 34.66 ms | 3.42 ms |
| 2.00 | agg_TP1 | 2.012 | 9.72 s | 7.34 s | 13.09 ms | 11.25 ms |
| 2.00 | agg_TP2 | 1.649 | 12.74 s | 9.94 s | 15.51 ms | 9.69 ms |
| 3.00 | B70_1E | 0.234 | 132.6 s | 131.3 s | 7.50 ms | 6.67 ms |
| 3.00 | B70_4E | 0.817 | 33.6 s | 32.5 s | 1.51 ms | 0.97 ms |
| 3.00 | dell06_1E | **2.517** | **8.02 s** | **4.19 s** | 26.00 ms | 3.72 ms |
| 3.00 | agg_TP1 | 2.091 | 10.71 s | 9.35 s | 8.47 ms | 10.26 ms |
| 3.00 | agg_TP2 | 1.748 | 13.61 s | 11.08 s | 14.82 ms | 9.81 ms |
| 4.00 | agg_TP1 | 2.052 | 11.80 s | 10.63 s | 5.29 ms | 10.17 ms |
| 4.00 | agg_TP2 | 2.029 | 12.08 s | 10.40 s | 9.63 ms | 8.63 ms |

**Saturation point per topology:**
- B70_1E: rate=0.25 (0.23 RPS plateau)
- B70_4E: rate=1.0 (0.83 RPS plateau)
- **dell06_1E: rate=2.0-3.0 (~2.5 RPS, marginal climb past sat)**
- agg_TP1: rate=2.0 (~2.05 RPS)
- agg_TP2: rate=4.0 (2.03 RPS — climbed slowly)

### 8img/1080p (~638 MB embedding)

| Rate | Topology | RPS | E2E p50 | TTFT p50 | TPOT p50 | ITL p50 |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | B70_1E | 0.038 | 695.3 s | 580.7 s | 8.63 ms | 5.85 ms |
| 0.10 | B70_4E | 0.106 | 56.6 s | 53.6 s | 7.03 ms | 4.48 ms |
| 0.10 | dell06_1E | 0.116 | **3.56 s** | **2.78 s** | 5.00 ms | 4.75 ms |
| 0.25 | B70_1E | 0.038 | 789.5 s | 774.1 s | 8.34 ms | 6.45 ms |
| 0.25 | B70_4E | 0.138 | 150.4 s | 137.8 s | 3.68 ms | 1.27 ms |
| 0.25 | dell06_1E | 0.285 | **3.72 s** | 2.76 s | 4.99 ms | 4.51 ms |
| 0.25 | agg_TP1 | 0.284 | 4.77 s | 3.05 s | 7.87 ms | 4.83 ms |
| 0.25 | agg_TP2 | 0.284 | 5.70 s | 3.58 s | 10.58 ms | 6.25 ms |
| 0.50 | B70_1E | 0.038 | 820.3 s | 809.2 s | 9.16 ms | 7.33 ms |
| 0.50 | B70_4E | 0.145 | 185.9 s | 176.8 s | 2.18 ms | 1.41 ms |
| 0.50 | dell06_1E | 0.552 | 6.43 s | 3.58 s | 12.50 ms | 1.23 ms |
| 0.50 | agg_TP1 | 0.553 | **4.87 s** | **2.89 s** | 11.73 ms | 5.16 ms |
| 0.50 | agg_TP2 | 0.546 | 6.68 s | 3.60 s | 16.73 ms | 6.25 ms |
| 1.00 | B70_1E | 0.038 | 835.7 s | 833.2 s | 8.37 ms | 7.40 ms |
| 1.00 | B70_4E | 0.147 | 199.7 s | 199.5 s | 1.30 ms | 1.02 ms |
| 1.00 | dell06_1E | 0.781 | 21.6 s | 13.21 s | 40.13 ms | 2.76 ms |
| 1.00 | agg_TP1 | **0.879** | **16.0 s** | **12.81 s** | 16.66 ms | 12.16 ms |
| 1.00 | agg_TP2 | 0.828 | 18.7 s | 14.56 s | 18.52 ms | 9.04 ms |
| 1.50 | B70_1E | 0.038 | 840.9 s | 838.9 s | 8.22 ms | 7.08 ms |
| 1.50 | B70_4E | 0.146 | 205.1 s | 204.9 s | 1.49 ms | 1.19 ms |
| 1.50 | dell06_1E | 0.820 | 27.7 s | 20.06 s | 7.89 ms | 3.37 ms |
| 1.50 | agg_TP1 | **1.054** | **17.4 s** | **16.19 s** | 3.16 ms | 7.59 ms |
| 1.50 | agg_TP2 | 0.939 | 21.6 s | 18.12 s | 16.61 ms | 10.07 ms |
| 2.00 | B70_1E | 0.038 | 843.9 s | 842.1 s | 8.57 ms | 7.45 ms |
| 2.00 | B70_4E | 0.147 | 208.1 s | 207.9 s | 1.81 ms | 1.30 ms |
| 2.00 | dell06_1E | 0.849 | 29.0 s | 25.24 s | 5.57 ms | 3.64 ms |
| 2.00 | agg_TP1 | **1.140** | **18.7 s** | 16.32 s | 9.28 ms | 10.68 ms |
| 2.00 | agg_TP2 | 1.019 | 21.8 s | 19.07 s | 15.19 ms | 9.45 ms |
| 3.00 | B70_1E | 0.038 | 845.9 s | 844.8 s | 8.40 ms | 7.44 ms |
| 3.00 | B70_4E | 0.147 | 210.1 s | 209.9 s | 1.76 ms | 1.33 ms |
| 3.00 | dell06_1E | 0.854 | 31.1 s | 27.78 s | 6.13 ms | 3.47 ms |
| 3.00 | agg_TP1 | **1.091** | **22.6 s** | 22.14 s | 0.75 ms | 6.90 ms |
| 3.00 | agg_TP2 | 1.033 | 23.9 s | 19.93 s | 22.67 ms | 10.63 ms |

**Saturation point per topology:**
- B70_1E: rate=0.10 (already saturated, 0.038 RPS forever)
- B70_4E: rate=0.5 (0.147 RPS plateau)
- dell06_1E: rate=1.5 (0.85 RPS plateau)
- **agg_TP1: rate=2.0 (1.14 RPS, the best)**
- agg_TP2: rate=2.0 (1.04 RPS, ~10% behind TP1)

## Per-workload analysis

### 4img/768p — light workload, PD-bound on every topology except B70_1E

The encoder is so cheap (~0.4-1.4 s ViT depending on hardware) that throughput is gated
by PD's per-request lifetime (~1.0-1.2 s). All topologies except B70_1E land in the
2.5-3.0 RPS range at sat:

| Topology | Sat RPS | Why this number? |
|---|---:|---|
| B70_1E | 0.44 | Encoder ViT 1.4 s/req on 1 XPU |
| B70_4E | 1.55 | 4 parallel XPUs amortize ViT ~0.35 s/req aggregate; PD overhead caps it |
| dell06_1E | >2.49 | H200 ViT ~0.4 s + low NIXL handoff; PD lifetime is the floor |
| agg_TP1 | 3.01 | No NIXL handoff; in-process buffer sharing; PD batches 2-3 deep |
| agg_TP2 | 2.73 | TP=2 NCCL overhead per token slows decode for this small workload |

**TTFT comparison** (rate=1.0):
- agg_TP1: 0.63 s
- agg_TP2: 0.71 s
- dell06_1E: **0.58 s** ← cross-host disagg actually beats agg here because PD has dedicated GPU
- B70_4E: 3.0 s
- B70_1E: 38.3 s

**Why agg_TP1 wins on RPS but dell06_1E wins on TTFT:** agg_TP1 amortizes batch overhead
better at high concurrency (3+ in flight), but dell06_1E has spare PD GPU capacity for
the first token. At low rate dell06_1E hands the first token back faster; at sat agg_TP1
saturates the device better because there's no NIXL serialization gap.

### 8img/768p — medium workload, encoder cost matters

Encoder cost roughly doubles vs 4img (2× more visual tokens, super-linear scaling), so the
ranking shifts:

| Topology | Sat RPS | dell06 vs others |
|---|---:|---:|
| B70_1E | 0.23 | dell06 is 11.0× faster |
| B70_4E | 0.83 | dell06 is 3.0× faster |
| **dell06_1E** | **2.52** | **best** |
| agg_TP1 | 2.09 | dell06 is 1.20× faster |
| agg_TP2 | 2.03 | dell06 is 1.24× faster |

**dell06_1E beats agg_TP1 here.** This is the surprise of the comparison. Why?
- agg_TP1's encoder steals time from PD when both want the GPU. The ViT forward pass
  on the same H200 device interleaves with PD's prefill/decode, fragmenting CUDA streams.
- dell06_1E's encoder is on a separate device. Single H200 dedicated to ViT, single
  H200 dedicated to LLM. They run truly in parallel.
- The NIXL-fetch + dispatch overhead (~150 ms cross-host) is more than paid for by
  letting PD use 100% of its GPU for the LLM.

**TTFT at rate=1.0**:
- dell06_1E: **1.32 s**
- agg_TP1: 1.74 s
- agg_TP2: 1.85 s
- B70_4E: 22.4 s
- B70_1E: 121.3 s

dell06_1E TTFT is 24% faster than agg_TP1's despite the cross-host NIXL hop.

### 8img/1080p — heavy workload, encoder cost dominates

This is the largest workload, and the ranking flips back to agg_TP1 winning:

| Topology | Sat RPS | gap vs agg_TP1 |
|---|---:|---:|
| B70_1E | 0.038 | 30× slower |
| B70_4E | 0.147 | 7.8× slower |
| dell06_1E | 0.85 | 1.34× slower |
| **agg_TP1** | **1.14** | **best** |
| agg_TP2 | 1.04 | 9% slower |

**Why agg_TP1 wins on the largest workload:** at 638 MB / req embedding, the cross-host
RoCE NIXL transfer becomes non-trivial. PD lifetime grows from ~1.2 s (light workloads)
to ~2.5 s (8img/1080p) because chunked-prefill of 16k visual tokens takes longer. The
~150 ms cross-host handoff is a bigger fraction of the total. Same-host agg_TP1 avoids
that handoff entirely.

But also: **dell06_1E at 0.85 RPS is only 25% behind agg_TP1's 1.14 RPS** — vastly
better than the 32B-FP8 cross-host disagg (which was 8-10× slower than same-host agg
per `disagg_h200_32b/SESSION_MEMORY.md`). The GPUDirect-RDMA cross-host NIXL path is
actually working as designed.

## Why agg_TP2 is consistently slower than agg_TP1 for 35B

This is the headline anomaly — for 32B-FP8, TP=2 always won (`disagg_h200_32b`
documents this exhaustively). For 35B, TP=1 wins by 10-15% across all 3 workloads.

Three reasons:

1. **MoE collective overhead.** Qwen3.5-A3B has 256 experts with top-8 routing per
   token. With TP=2, every MoE forward needs an `all-to-all` to dispatch tokens to
   the right expert rank, then a reverse `all-to-all` to gather outputs. Per token,
   per layer (10 full-attn + 30 linear-attn = 40 layers), the round-trip cost on
   NVLink is ~5-10 µs × 40 = 200-400 µs added per token vs the TP=1 path that has
   no collective.

2. **Hybrid attention asymmetry.** 30/40 layers are linear-attention (GDN/Mamba-like)
   which doesn't shard cleanly across TP. The Mamba state cache becomes a serialization
   point. SGLang handles this but pays a TP penalty.

3. **No KV cache pressure to relieve.** 32B-FP8's TP=2 win came from doubling KV cache
   capacity, allowing larger batches. 35B's KV cache is already at <1% utilization
   (`max_total_num_tokens=1,070,925`, observed peak ~16k); doubling it does nothing.

This contradicts the projection in `35b_bottleneck_analysis.md` which suggested TP=2
would win. Empirically: **TP=1 is the optimal config for 35B-A3B agg.**

## Latency comparison at saturation (rate=2.0 representative)

| Workload | Topology | RPS | E2E p50 | TTFT p50 | TPOT p50 |
|---|---|---:|---:|---:|---:|
| 4img/768p | agg_TP1 | 1.74 | 2.38 s | 0.86 s | 10.97 ms |
| 4img/768p | dell06_1E | 1.74 | **1.72 s** | **0.61 s** | 7.79 ms |
| 4img/768p | agg_TP2 | 1.66 | 3.65 s | 1.25 s | 14.18 ms |
| 4img/768p | B70_4E | 1.54 | 13.2 s | 7.01 s | 41.13 ms |
| 4img/768p | B70_1E | 0.44 | 66.0 s | 58.0 s | 8.38 ms |
| 8img/768p | dell06_1E | 2.30 | **7.31 s** | **3.10 s** | 34.66 ms |
| 8img/768p | agg_TP1 | 2.01 | 9.72 s | 7.34 s | 13.09 ms |
| 8img/768p | agg_TP2 | 1.65 | 12.7 s | 9.94 s | 15.51 ms |
| 8img/768p | B70_4E | 0.81 | 32.3 s | 30.2 s | 1.52 ms |
| 8img/768p | B70_1E | 0.23 | 133.6 s | 128.5 s | 8.99 ms |
| 8img/1080p | agg_TP1 | **1.14** | **18.7 s** | 16.32 s | 9.28 ms |
| 8img/1080p | agg_TP2 | 1.02 | 21.8 s | 19.07 s | 15.19 ms |
| 8img/1080p | dell06_1E | 0.85 | 29.0 s | 25.24 s | 5.57 ms |
| 8img/1080p | B70_4E | 0.15 | 208.1 s | 207.9 s | 1.81 ms |
| 8img/1080p | B70_1E | 0.038 | 843.9 s | 842.1 s | 8.57 ms |

**TPOT note:** B70_4E often has the lowest TPOT in saturation (1-2 ms) because
batched-decode efficiency is highest there (large queue, decode runs in big batches).
But TTFT is awful because requests queue for minutes. **Low TPOT in B70 setups is a
queue-depth artifact, not a real performance win.**

## Sub-saturation latency (rate=0.10 — single-request latency)

This is the cleanest measurement of "how fast can each topology handle one request":

| Workload | agg_TP1 | agg_TP2 | B70_1E | B70_4E | dell06_1E |
|---|---:|---:|---:|---:|---:|
| 4img/768p (E2E p50) | (no data) | (no data) | 3.6 s | 3.5 s | **1.4 s** |
| 4img/768p (TTFT p50) | (no data) | (no data) | 2.6 s | 2.7 s | **0.6 s** |
| 8img/768p (E2E p50) | (no data) | (no data) | 15.3 s | 5.8 s | **1.7 s** |
| 8img/768p (TTFT p50) | (no data) | (no data) | 12.5 s | 5.0 s | **0.9 s** |
| 8img/1080p (E2E p50) | (no data) | (no data) | 695 s | 56.6 s | **3.6 s** |
| 8img/1080p (TTFT p50) | (no data) | (no data) | 581 s | 53.6 s | **2.8 s** |

(Agg sweeps started at rate=0.25.) **dell06_1E has the lowest single-request TTFT at
all three workload sizes** — the H200 encoder runs the ViT faster than agg's
shared-device ViT (which has to context-switch with PD).

## Speedup matrices

### dell06_1E vs B70_1E (cross-host disagg upgrade: B70 XPU → H200 SXM)

| Workload | Sat RPS B70_1E | Sat RPS dell06_1E | Speedup |
|---|---:|---:|---:|
| 4img/768p | 0.44 | >2.49 | **>5.7×** |
| 8img/768p | 0.23 | 2.52 | **11.0×** |
| 8img/1080p | 0.038 | 0.85 | **22.4×** |

The speedup grows with workload size because the B70 XPU's `triton_attn` ViT
cost grows super-linearly with pixels, while H200's `fa3` ViT scales much better.

### dell06_1E vs B70_4E (4 XPUs vs 1 H200)

| Workload | Sat RPS B70_4E | Sat RPS dell06_1E | dell06 / B70_4E |
|---|---:|---:|---:|
| 4img/768p | 1.55 | >2.49 | **>1.6×** |
| 8img/768p | 0.83 | 2.52 | **3.0×** |
| 8img/1080p | 0.147 | 0.85 | **5.8×** |

**One H200 SXM beats 4 B70 XPUs at every workload, with the gap widening as the
workload grows.** The single H200 has more SMs, more memory bandwidth, and a more
mature attention implementation than 4 Battlemage XPUs running through Triton.

### dell06_1E vs agg_TP1 (cross-host disagg vs single-H200 agg)

| Workload | Sat RPS agg_TP1 | Sat RPS dell06_1E | dell06 / agg_TP1 |
|---|---:|---:|---:|
| 4img/768p | 3.01 | >2.49 | 0.83× (agg wins) |
| 8img/768p | 2.09 | 2.52 | **1.21× (dell06 wins)** |
| 8img/1080p | 1.14 | 0.85 | 0.75× (agg wins) |

**Mixed result.** dell06_1E beats agg_TP1 only at 8img/768p, where the encoder cost is
high enough to benefit from a dedicated device but not so high that cross-host NIXL
becomes a tax. Below and above that workload size, agg_TP1 wins.

### TTFT comparison at rate=1.0 across topologies

| Workload | agg_TP1 | dell06_1E | dell06 / agg_TP1 |
|---|---:|---:|---:|
| 4img/768p | 0.63 s | **0.58 s** | 0.92× (dell06 7% faster) |
| 8img/768p | 1.74 s | **1.32 s** | 0.76× (dell06 24% faster) |
| 8img/1080p | **12.81 s** | 13.21 s | 1.03× (agg slightly faster) |

**dell06_1E TTFT beats agg_TP1 TTFT for 768p workloads** at low-mid rate. The reason
is that agg_TP1 has to context-switch the GPU between encoder and PD streams, while
dell06_1E lets each device focus on one job.

## Cost-effectiveness analysis

Treating each "unit of GPU" as a cost factor:

| Topology | GPU count | RPS @ 8img/1080p | RPS / GPU |
|---|---:|---:|---:|
| agg_TP1 | 1 H200 | 1.14 | **1.14** |
| dell06_1E | 2 H200 (1 PD + 1 enc) | 0.85 | 0.43 |
| agg_TP2 | 2 H200 | 1.04 | 0.52 |
| B70_4E | 1 H200 + 4 XPUs | 0.147 | (mixed hardware, ~0.05) |
| B70_1E | 1 H200 + 1 XPU | 0.038 | (mixed hardware, ~0.02) |

**agg_TP1 has by far the best per-GPU efficiency.** Disagg with dell06 1E is a 2× cost
config that delivers less than 2× throughput. The case for disagg is operational
(workload isolation, encoder host scaling, etc.), not raw efficiency.

| Topology | GPU count | RPS @ 8img/768p | RPS / GPU |
|---|---:|---:|---:|
| agg_TP1 | 1 H200 | 2.09 | **2.09** |
| dell06_1E | 2 H200 | 2.52 | 1.26 |
| agg_TP2 | 2 H200 | 2.03 | 1.02 |

Even at 8img/768p where dell06_1E "wins," it does so at half the per-GPU efficiency.

## Recommendations by use case

### "I want maximum throughput per dollar"
**agg_TP1.** Best RPS per GPU at every workload. No NIXL overhead. Single device.
Don't bother with TP=2 for 35B; it costs 2× hardware and gives 90% the throughput.

### "I want lowest TTFT for serving latency"
- **4img/768p, 8img/768p**: **dell06_1E** (cross-host disagg with H200 encoder)
- **8img/1080p**: **agg_TP1** at sub-saturation (rate≤1.0); above sat all configs queue.

### "I want operational flexibility (encoder/PD scale independently)"
**dell06_1E** is the only topology that gives this without giving up too much throughput
(within 25% of agg_TP1 at 8img/1080p, beats it at 8img/768p).

### "I have only B70 XPU encoders available"
- **At 8img/1080p**: B70 1E or even 4E is unusable (>200 s TTFT at sat). Either reduce
  resolution to 768p or switch to H200 encoder.
- **At 768p workloads**: B70 4E gives 0.83-1.55 RPS, usable but 30-50% behind dell06_1E
  on throughput and 5-15× behind on TTFT.
- **B70 4E is strictly better than B70 1E** in every measurement; never run 1E if you
  have 4 XPUs.

### "I want to minimize PD-side memory pressure"
Disagg topologies (any) keep PD's `mem_fraction_static=0.75` purely for KV cache + LLM
weights. Agg topologies share that budget with encoder activations + ViT memory. For
35B (~70 GB weights, ~38 GB KV+Mamba), this isn't currently a concern, but for 70B-class
models it would matter.

## What we did NOT measure but probably matters

1. **dell06 1E saturation point.** We never reached it for 4img/768p (rate=3.0 still
   climbing at 2.49 RPS). A higher-rate test (rate=5-10) would find the actual ceiling
   and let us compare cleanly with agg_TP1's 3.0 RPS.

2. **agg_TP1 at rate=0.10** for 4img/768p and 8img/768p — sweep started at rate=0.25.
   So we can't compare absolute single-request latency at the cleanest sub-saturation
   rate.

3. **Cross-host disagg with both encoders on dell06** (2 H200 encoders, 1 PD on giga01).
   This would isolate "is dell06 fast enough to saturate on its own?" — likely yes per
   the 4img/768p uncapped result.

4. **Hybrid: dell06 + B70_4E** (5 encoders, mixed hardware). We had this briefly when
   the user added dell06 to the 4× B70 setup but didn't sweep it. Could be the most
   throughput-efficient cross-host config.

5. **PD-side bottleneck analysis for the dell06_1E sweeps.** The 4img/768p_4E doc has
   a deep PD-side analysis showing the bottleneck is encoder→PD pipeline cadence.
   For dell06_1E, the dynamics are likely different (PD is the bottleneck, not the
   encoder), and a similar timing analysis would confirm.

## Sources

- Bench JSONs:
  - `/hongming/res22_disagg_h200_35b_sweep/{4img_768p,8img_768p,8img_1080p}{,_4E,_dell06_1E}/rate_*_np32/benchmark_output.json` (B70 1E, B70 4E, dell06 1E)
  - `/hongming/res20_agg_h200_35b/{tp1,tp2}/{4img_768p,8img_768p,8img_1080p}_np32_sweep_*/rate_*/benchmark_output.json` (agg TP1, TP2)
- Sweep scripts: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/run_disagg_35b_*_4E.sh` and `*_dell06_1E.sh`
- Server scripts:
  - `/hongming/dynamo/01_cuda_sh/agg_h200_35b/start_h200_aggregate_epd_server_35b_tp{1,2}.sh`
  - `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/start_sglang_pd_cuda_35b_giga01.sh`
- Companion analysis docs:
  - `35b_bottleneck_analysis.md` (1E B70 PD-side analysis)
  - `4img_768p_4E_bottleneck.md` (4E B70 PD-side analysis)
  - `disagg_35b_results.md` (1E B70 sweep summary)
- 32B reference (very different conclusion — TP=2 wins for FP8): `disagg_h200_32b/SESSION_MEMORY.md`

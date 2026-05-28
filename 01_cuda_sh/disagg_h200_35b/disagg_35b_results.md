# Disagg 35B cross-host sweep — full results

**Date:** 2026-05-25
**Model:** Qwen3.5-35B-A3B (BF16 MoE multimodal, 256 experts × 8/token, hybrid 30 linear + 10 full attention)
**Setup:**
- giga01 H200 PD, GPU 4, `mem-fraction-static=0.75`, `max-running-requests=40`
- B70 1× XPU encoder (PID 4537, XPU 3), `--encoder-only`, registered as `dynamo.encoder.generate` at 172.26.46.172:43957
- Cross-host RoCE: B70 `mlx5_0` ↔ giga01 `mlx5_4`, fabric 192.165.123.0/24
- Frontend `--router-mode round-robin` (kv-router crashes on linear-attn page_size=1)
- Bench: `sglang.bench_serving --backend sglang-oai-chat`, **`np=32`**, `random-input-len=128`, `random-output-len=256`, `seed=0`
- Sweep wall time: 2h 56m (08:01-10:57 UTC, 21 runs)

**All 21 runs returned 32/32 successful — no cancellations, no OOMs.**

## Sweep design

- **Rates:** 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 (Poisson arrivals)
- **Workloads (executed in this order):**
  1. 8img / 1920×1080 (~638 MB embedding/req)
  2. 8img / 1024×768 (~242 MB embedding/req)
  3. 4img / 1024×768 (~127 MB embedding/req)
- **np=32** prompts each
- **90 s drain between runs** (lets B70 XPU release device memory before next bench warmup)

## Results — 8img/1080p (~638 MB embed)

| Rate | RPS | Duration | E2E mean | E2E p50 | TTFT mean | TTFT p50 | TPOT mean | TPOT p50 | ITL p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.1** | **0.038** | 851 s | 701.3 s | 695.3 s | 494.9 s | 580.7 s | 1580.32 ms | 8.63 ms | 5.85 ms |
| 0.25 | 0.038 | 852 s | 791.9 s | 789.5 s | 730.3 s | 774.1 s | 421.22 ms | 8.34 ms | 6.45 ms |
| 0.5  | 0.038 | 852 s | 821.6 s | 820.3 s | 795.4 s | 809.2 s | 203.05 ms | 9.16 ms | 7.33 ms |
| 1.0  | 0.038 | 852 s | 836.3 s | 835.7 s | 834.2 s | 833.2 s |  15.60 ms | 8.37 ms | 7.40 ms |
| 1.5  | 0.038 | 852 s | 841.3 s | 840.9 s | 834.3 s | 838.9 s |  58.73 ms | 8.22 ms | 7.08 ms |
| 2.0  | 0.038 | 852 s | 844.3 s | 843.9 s | 833.1 s | 842.1 s |  95.11 ms | 8.57 ms | 7.45 ms |
| 3.0  | 0.038 | 852 s | 846.3 s | 845.9 s | 845.1 s | 844.8 s |   8.53 ms | 8.40 ms | 7.44 ms |

**Saturation: 0.038 RPS** — already at sat from rate=0.1. The 1E XPU encoder takes ~26 s per
8img/1080p request, so 1/26 = 0.038 reqs/s is the absolute ceiling.

**E2E grows linearly with queue depth** as rate increases (701 s → 846 s as queue piles up).
TTFT p50 climbs from 581 s to 845 s — the entire bench window is one big traffic jam.

**TPOT mean drops with load** (1580 ms at rate=0.1 → 8.5 ms at rate=3.0) — at low load decode
runs solo with no batching amortization; under load decode batches multiple requests.

## Results — 8img/768p (~242 MB embed)

| Rate | RPS | Duration | E2E mean | E2E p50 | TTFT mean | TTFT p50 | TPOT mean | TPOT p50 | ITL p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1  | 0.15 | 209 s |  15.5 s |  15.3 s |  12.5 s |  12.5 s |  32.14 ms |   4.88 ms | 0.87 ms |
| 0.25 | 0.23 | 138 s | 101.1 s |  96.0 s |  60.7 s |  69.5 s | 367.65 ms | 172.87 ms | 6.63 ms |
| **0.5**  | **0.23** | 139 s | 119.7 s | 117.3 s |  99.6 s | 105.5 s | 128.48 ms |   8.52 ms | 7.06 ms |
| 1.0  | 0.23 | 139 s | 128.9 s | 127.7 s | 106.8 s | 121.3 s | 154.69 ms |   8.26 ms | 6.96 ms |
| 1.5  | 0.23 | 137 s | 130.1 s | 129.3 s | 113.0 s | 124.2 s | 122.03 ms |   8.37 ms | 6.91 ms |
| 2.0  | 0.23 | 139 s | 134.2 s | 133.6 s | 123.8 s | 128.5 s |  88.31 ms |   8.99 ms | 7.07 ms |
| 3.0  | 0.23 | 137 s | 133.0 s | 132.6 s | 130.6 s | 131.2 s |  13.88 ms |   7.50 ms | 6.67 ms |

**Saturation: 0.23 RPS** — reached at rate=0.25, plateaus from rate=0.5 through rate=3.0.

Per-request encoder time on 1E for 8img/768p ≈ 12.5 s (TTFT at rate=0.1 sub-saturation).

## Results — 4img/768p (~127 MB embed)

| Rate | RPS | Duration | E2E mean | E2E p50 | TTFT mean | TTFT p50 | TPOT mean | TPOT p50 | ITL p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1  | 0.09 | 361 s |   4.0 s |   3.6 s |   3.2 s |   2.5 s |   4.84 ms |   4.58 ms | 4.51 ms |
| 0.25 | 0.22 | 145 s |  10.7 s |   8.7 s |   6.8 s |   5.3 s |  30.41 ms |   4.62 ms | 4.28 ms |
| **0.5**  | **0.43** | 74 s |  38.2 s |  47.5 s |  16.4 s |  14.3 s | 251.01 ms | 151.47 ms | 6.09 ms |
| 1.0  | 0.44 | 72 s |  57.0 s |  60.2 s |  35.1 s |  38.3 s | 248.05 ms | 126.20 ms | 7.00 ms |
| 1.5  | 0.44 | 72 s |  61.8 s |  64.0 s |  48.7 s |  53.2 s |  91.19 ms |   8.44 ms | 7.37 ms |
| 2.0  | 0.44 | 72 s |  64.4 s |  66.0 s |  54.2 s |  58.0 s |  67.38 ms |   8.38 ms | 7.25 ms |
| 3.0  | 0.45 | 71 s |  65.3 s |  66.4 s |  58.1 s |  62.9 s |  50.04 ms |   8.46 ms | 7.30 ms |

**Saturation: ~0.44 RPS** — reached at rate=0.5, climbs slightly to 0.45 at rate=3.0.

Per-request encoder time on 1E for 4img/768p ≈ 2.5 s (TTFT at rate=0.1 sub-saturation).

## Saturation comparison: 35B disagg vs 32B-FP8 disagg (1E)

35B is the bigger model with hybrid attention. Reference 32B-FP8 1E numbers from
`../disagg_h200_32b/patched_1E_results.md` (np=16):

| Workload    | **35B disagg np=32** | **32B-FP8 1E patched np=16** | 35B / 32B ratio |
|---|---:|---:|---:|
| 4img/768p   | 0.44 RPS  | 0.40 RPS  | 1.10× |
| 8img/768p   | 0.23 RPS  | 0.21 RPS  | 1.10× |
| 8img/1080p  | 0.038 RPS | 0.037 RPS | 1.03× |

**35B is ~3-10% higher RPS than 32B-FP8.** Surprisingly close — this is because both setups
share the same 1E B70 XPU encoder, which is the throughput bottleneck for cross-host disagg.
The 35B PD (BF16) is heavier per-token than 32B-FP8 PD, but PD-side decode is still much faster
than encoder ViT, so total throughput tracks the encoder.

The slight 35B advantage suggests the 35B encoder pipeline is marginally faster — possibly
because the 35B model (Qwen3.5) was trained with smaller-resolution image tokens or the vision
tower has fewer ViT layers than Qwen3-VL-32B. Worth investigating in a follow-up.

## TTFT vs 32B-FP8 disagg (1E) at sub-saturation rate=0.1

| Workload    | **35B disagg TTFT p50 @ rate=0.1** | **32B-FP8 1E TTFT p50** | 35B / 32B |
|---|---:|---:|---:|
| 4img/768p   | 2.5 s   | 3.6 s   | **0.69×** (35% faster) |
| 8img/768p   | 12.5 s  | 9.5 s   | 1.32× |
| 8img/1080p  | 580 s   | 309 s   | 1.88× |

**Mixed**: 35B is faster on 4img/768p (small workload), slower on bigger workloads. The 8img/1080p
TTFT is dominated by queue buildup since np=32 in the 35B sweep vs np=16 in the 32B sweep,
not directly comparable.

## TPOT comparison at rate=1.0

| Workload    | **35B TPOT p50** | **32B-FP8 1E TPOT p50** |
|---|---:|---:|
| 4img/768p   | 126.2 ms  | 4.91 ms  |
| 8img/768p   |   8.3 ms  | 4.57 ms  |
| 8img/1080p  |   8.4 ms  | 3.87 ms  |

**35B TPOT is ~2× higher than 32B at the same rate** — this is the real model-cost difference.
The 35B BF16 weights (~70 GB) and Mamba state cache (~18 GB) make per-token decode pricier.
The 4img/768p rate=1.0 TPOT (126 ms) is an outlier suggesting 1 request was alone in decode
when other requests were stuck in encoder queue.

## Per-rate observations

### Saturation hits early on 1E across all workloads

Single XPU encoder fully serializes — every workload reaches sat by rate=0.5:
- 4img/768p sat at rate=0.5 (0.43 RPS)
- 8img/768p sat at rate=0.25 (0.23 RPS)
- 8img/1080p sat by rate=0.1 (0.038 RPS — TTFT/E2E already in queue regime)

### TPOT is high at low rates, drops as rate rises (consistent across workloads)

Pattern is the SGLang single-request decode amortization issue:
- 8img/1080p rate=0.1: TPOT mean 1580 ms (decode runs solo while encoder backs up)
- 8img/1080p rate=1.0: TPOT mean 16 ms (multiple requests batch on PD)
- 8img/1080p rate=3.0: TPOT mean 8.5 ms (batched decode efficient)

### B70 1E encoder XPU was rock-solid this run

No `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` events, no warmup retries needed, all 21 runs OK
on first attempt. Contrast with the 32B-FP8 1E sweep where np=32 caused OOMs and we had to
drop to np=16. Suggests 35B encoder ViT is lighter on XPU memory than 32B encoder.

## Conclusions

1. **35B disagg with 1E saturates at the same encoder-bound throughput as 32B-FP8 1E** across
   all three workloads (within 10%). The encoder is the bottleneck, not the PD.
2. **All 21 runs completed cleanly at np=32**, unlike the 32B-FP8 1E sweep which needed np=16
   to avoid B70 XPU OOM. The 35B encoder appears more memory-frugal.
3. **35B PD-side decode (TPOT) is ~2× pricier than 32B-FP8** at the same rate, reflecting BF16
   vs FP8 weights and the Mamba/linear-attention state cache overhead.
4. **Cross-host disagg with 1E remains far below same-host agg** for any usable TTFT regime —
   typical agg TTFT for similar workloads on a 32B-class model is 1-5 s vs 12-580 s here.

## Sources

- Bench JSON: `/hongming/res22_disagg_h200_35b_sweep/{8img_1080p,8img_768p,4img_768p}/rate_<rate>_np32/benchmark_output.json`
- Bench logs: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/bench_disagg_35b_*.log`
- Sweep master log: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/35b_sweep_master.log`
- Sweep script: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/run_disagg_35b_sweep.sh`
- giga01 PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01_restart.log`
- B70 encoder log: `/hongming/dynamo/02_xpu_sh/disagg_b70_35b/logs/encode_xpu_35b_b70.log`
- 32B-FP8 1E reference: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/patched_1E_results.md`

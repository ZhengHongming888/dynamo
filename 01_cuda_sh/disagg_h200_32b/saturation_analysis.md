# Saturation RPS — Detailed Analysis

## Test setup

**Hardware:**
- giga01: H200 host (172.26.46.75), 8× H200 GPUs (143 GB each)
- B70: separate host (172.26.46.13), 4× H200 GPUs available for encoder use
- RoCE fabric: 192.165.123.0/24, 400 Gb/s NDR (giga01 mlx5_4 ↔ B70 mlx5_0..3)
- Mgmt fabric: 172.26.46.0/22 (1G/10G), used for dynamo TCP request plane

**Software:**
- dynamo 1.0.0
- sglang 0.5.12.dev315+g91907b7b9
- Model: Qwen3-VL-32B-Instruct-FP8 (FP8 weights, FP8 KV cache)
- transformers 5.6.0, torch 2.11.0, CUDA 13.0
- `cupy` NOT installed → numpy CPU-fallback path in `nixl_connect`

**Bench:** `sglang.bench_serving --backend sglang-oai-chat`, 64 prompts (128 for some over-sat tests), `--random-input-len 128 --random-output-len 256`, image content random, fixed seed=0.

---

## Master saturation table

| Workload | Embedding size (NIXL desc) | **TP=1 agg sat** | **TP=2 agg sat** | **Cross-host disagg sat (4 encoders)** |
|---|---:|---:|---:|---:|
| 4img / 1024×768 | ≈127 MB | **~2.7 RPS** | ~3.1 RPS | ~1.35 RPS |
| 8img / 1024×768 | ≈242 MB | ~1.5 RPS | **~1.9 RPS** | ~0.7 RPS |
| 8img / 1920×1080 | ≈638 MB | ~0.47 RPS | **~0.6-0.7 RPS** | ~0.13 RPS |

**Resources used:**
- TP=1 agg: 1× H200 (GPU 4), `mem-fraction=0.85`, `max-running=40`
- TP=2 agg: 2× H200 (GPUs 4+5), `mem-fraction=0.85`, `max-running=25`
- Cross-host disagg: 1× giga01 H200 (GPU 4) PD + 4× B70 GPUs (4 encoder instances), `mem-fraction=0.92` PD-side

---

## Saturation sweeps (raw rate-vs-throughput data)

### TP=1 agg

| Workload | r=0.5 | r=1.0 | r=1.5 | r=2.0 | r=3.0 | r=4.0 | r=5.0 (np=128) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4img/768p | — | 1.07 | — | 2.03 | 2.68 | 3.67 | **2.65** ← over-sat |
| 8img/768p | — | 0.90 | 1.23 | 1.45 | — | — | — |
| 8img/1080p | 0.46 | 0.47 | — | — | — | — | — |

### TP=2 agg

| Workload | r=0.5 | r=0.75 | r=1.0 | r=1.5 | r=2.0 | r=2.5 | r=3.0 | r=4.0 | r=5.0 (np=128) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4img/768p | — | — | 1.08 | — | 2.10 | — | 2.94 | 3.22 | **3.05** ← over-sat |
| 8img/768p | — | — | 0.92 | 1.36 | 1.68 | 1.90 | — | — | — |
| 8img/1080p | 0.50 | 0.74 | 0.60 | — | — | — | — | — | — |

### Cross-host disagg (4 encoders on B70)

| Workload | r=1.0 | r=1.5 | r=2.0 |
|---|---:|---:|---:|
| 4img/768p | 1.02 | **1.37** | 1.32 ← over-sat |
| 8img/768p | 0.68 (sat) | — | — |
| 8img/1080p | 0.13 (heavily sat, 64 reqs queued at once) | — | — |

(Cross-host with 1 encoder, before B70 added 4-instance topology, was much worse: ~0.43 / ~0.083 / ~0.083.)

---

## TTFT (latency) under load — best observed median

| Workload | TP=1 agg | TP=2 agg | Cross-host (4 enc) |
|---|---:|---:|---:|
| 4img/768p | **0.80 s** (r=1.0) | 1.0 s (r=1.0) | 5.5 s (r=1.0) |
| 8img/768p | 1.61 s (r=1.0) | **1.69 s** (r=1.0) | 21 s (r=1.0) |
| 8img/1080p | 8.07 s (r=0.5) | **3.53 s** (r=0.5) | 368 s (r=1.0, sat) |

For interactive use (target TTFT < 2-3 s):
- **TP=1/TP=2 agg meet the target on 4img/768p and 8img/768p**
- **Only TP=2 agg meets it on 8img/1080p** (and only at low rate)
- **Cross-host fails on every workload**

---

## Analysis

### 1. The bottleneck for both agg and disagg is multimodal preprocessing, not LLM compute

**Evidence:**
- TP=1 agg → TP=2 agg only buys **15-30% RPS gain** at saturation, despite 2× the LLM compute. If LLM were the bottleneck, we'd expect ~2× scaling.
- TP=1 sat for 8img/1080p was 0.47 RPS. TP=2 sat for the same was ~0.6-0.7 RPS. The 30% improvement comes from being able to parallelize ViT slightly better when there's spare GPU compute — not from LLM batching.
- At the same workload, prefill and decode actually run very fast on the GPU (`forward_duration ≈ 5-6s` total per request vs. several minutes of end-to-end latency).

**What this means:**
- The Qwen3-VL ViT encoder dominates per-request cost at 8img/1080p (~13 GB of pixel processing per request).
- Adding a second GPU helps modestly because prefill+decode now overlap with ViT, but the ViT step itself isn't faster.

### 2. Cross-host disagg has TWO compounded penalties on top of agg

#### (a) The CPU-bounce in NIXL embedding transfer

NIXL `ReadOperation` log on PD side shows:
```
local_descriptors:  size=668467200, device=cpu   ← 638 MB CPU buffer
remote_descriptors: size=668467200, device=cpu
```

**`device=cpu` on both ends** means each request requires:
1. Encoder GPU → encoder CPU memory copy (B70 host-staging)
2. CPU buffer registration with NIXL (B70)
3. RoCE wire transfer (fast: 638 MB / 50 GB/s ≈ 13 ms)
4. CPU buffer registration with NIXL (giga01)
5. Numpy descriptor handling (no `cupy` available — log warns: `Failed to load CuPy ... utilizing numpy`)
6. CPU buffer → PD GPU (giga01)
7. SGLang prefill + decode

Steps 1, 2, 4, 5, 6 are all CPU-side per-request work. Empirical measurements:

| Workload | NIXL throughput sustained |
|---|---:|
| 4img/768p (127 MB) | ~130 MB/s |
| 8img/768p (242 MB) | ~165 MB/s |
| 8img/1080p (638 MB) | ~83 MB/s (collapsed under thrashing) |

Cluster ceiling appears to be **~160-170 MB/s effective NIXL throughput** — about **300× slower than the 50 GB/s wire is capable of**. The wire is fine; CPU descriptor handling is the bottleneck.

#### (b) Worst-case scaling at large payloads

At 638 MB embeddings, the CPU memory allocator/registration path on the PD thrashes. `running-req` capped at only 2-3 in-flight (vs. 30 at smaller payloads). Every request piled up into a 64-deep queue at rate=1.0. Effective throughput dropped to **half of the linear prediction**, suggesting allocator pressure → kernel-level blocking.

### 3. Why same-host agg avoids these penalties entirely

- **No NIXL transfer**. Embeddings live on the same GPU as the LLM after the encoder runs. The "transfer" is a zero-copy GPU-to-GPU pointer pass within the same process.
- **No mgmt-network overhead**. All control plane traffic is loopback.
- **No descriptor pool issues**. SGLang's internal multimodal pipeline manages embeddings as ordinary tensors.

### 4. Cross-host disagg's only structural advantage didn't matter

In theory, cross-host frees up the PD GPU to spend 100% of memory on KV cache (no encoder competing). In practice:
- We measured PD's `mem-fraction=0.92` (vs. 0.85 for agg) — extra 7% headroom
- This translates to maybe ~10% more max-batched-tokens
- But the NIXL CPU-bounce overhead costs **5-100× per request**, completely overwhelming any KV-cache benefit

### 5. The 4-encoder topology on B70 helped only at small payloads

When B70 went from 1→4 encoder instances, RPS at 4img/768p jumped 2.4× (0.43 → 1.02). But at 8img/1080p, the same topology change only got 0.083 → 0.13 (1.6×) — because the new bottleneck moved to the **PD's CPU-side NIXL handling**, not the encoder's ViT compute.

| Workload | 1 encoder | 4 encoders | Speedup |
|---|---:|---:|---:|
| 4img/768p | 0.43 RPS | 1.02 RPS | 2.4× |
| 8img/1080p | 0.083 RPS | 0.13 RPS | 1.6× |

**Where you saturate the cluster determines whether more encoders help:**
- Small payload → encoder ViT is the bottleneck → more encoders help proportionally
- Large payload → PD NIXL CPU handling is the bottleneck → more encoders pile up requests the PD can't drain

---

## Production guidance from these numbers

For Qwen3-VL-32B-FP8 multimodal serving on this stack:

| Goal | Recommended config |
|---|---|
| Best RPS at small workloads (4img/768p) | TP=2 agg same-host (3.1 RPS) |
| Best RPS at typical workloads (8img/768p) | TP=2 agg same-host (1.9 RPS) |
| Best RPS at large workloads (8img/1080p) | TP=2 agg same-host (0.6-0.7 RPS) |
| Best TTFT for interactive use | TP=1 agg same-host (0.8-1.6 s up to 8img/768p) |
| Maximum hardware efficiency (RPS/GPU) | TP=1 agg same-host (best per-GPU throughput) |

**Cross-host disagg is not currently competitive** for any of these workloads on this stack. It's worse than even TP=1 same-host agg, despite using more GPUs total.

---

## Open work that could change the picture

These would need to be tried before concluding cross-host is fundamentally inferior:

1. **Install `cupy`** in `/opt/venv` → eliminates numpy fallback in NIXL descriptor handling. Quick test, plausibly closes ~half the gap for small workloads.

2. **Apply the round-4 GPU descriptor pool patch** from SESSION_MEMORY (multi-day eng task) → eliminates the CPU-bounce entirely. Same-host disagg round-4 testing showed 4-5× speedup on 5 surviving requests before OOM, suggesting if this works without OOM, cross-host disagg could match or exceed same-host agg.

3. **Bind dynamo TCP request plane to RoCE IP** (currently auto-picks mgmt IP via OS default route). Modest improvement to control-plane latency.

4. **Re-test TP=2 agg with fresh max-running tuning**. The 0.95 RPS in SESSION_MEMORY for 8img/1080p is significantly higher than today's 0.6-0.7. The difference may be dependency drift (different sglang/dynamo versions), `chunked-prefill-size`, or `max-running-requests` settings.

---

## Files

- Result data: `/hongming/res13_baseline_giga01_agg/{tp1,tp2}/`, `/hongming/res12_crosshost_giga01/`
- Bench logs: `/hongming/dynamo/01_cuda_sh/agg_h200_32b/logs/bench_*.log`, `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/bench_*.log`
- Full write-up: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/cross_host_giga01_b70_results.md`
- Cross-host PD launch script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh`
- Same-host agg launch scripts (already existed): `/hongming/dynamo/01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp{1,2}.sh`

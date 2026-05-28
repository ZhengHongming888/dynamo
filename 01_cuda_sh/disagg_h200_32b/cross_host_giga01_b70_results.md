# Cross-host disagg results (giga01 H200 PD ↔ B70 encoder)

**Date:** 2026-05-24
**Setup:**
- PD on giga01 (sc09super21-h200, 172.26.46.75), GPU 4, TP=1, Qwen3-VL-32B-Instruct-FP8
- Encoder on B70 (172.26.46.13), separate host
- Mgmt fabric: 172.26.46.0/22 (used for NATS/etcd/dynamo TCP request plane)
- RoCE fabric: 192.165.123.0/24 NDR 400 Gb/s (used for NIXL embedding reads)
- giga01 NIC: mlx5_4 (192.165.123.52), NUMA 2 — paired with GPU 4 (NUMA 2)
- Bench client running on giga01 against `localhost:7001`
- See `start_sglang_pd_cuda_32b_fp8_giga01.sh` for the giga01 PD launch script

## TL;DR

Cross-host multimodal disagg's bottleneck is the **per-request NIXL CPU-bounce** in
`embedding_transfer.py`, exactly as documented in same-host SESSION_MEMORY rounds.
Going cross-host did **not** fix this (RDMA wire is fast; CPU descriptor handling
on each side is slow). Throughput is workload-size-dependent because the per-request
CPU overhead is roughly fixed.

**Headline numbers (rate=1.0, 64 prompts):**

| Workload | 1 encoder | 4 encoders |
|---|---:|---:|
| 4img / 768p (≈127 MB embedding) | 0.43 RPS | **1.02 RPS** ← unsaturated |
| 8img / 1080p (≈638 MB embedding) | ≈0.083 RPS | 0.13 RPS |

**Compare to same-host baselines (from SESSION_MEMORY, 8img/1080p):**
- TP=2 agg same-host: **0.95 RPS** (best)
- TP=1 agg same-host: 0.52 RPS
- Same-host disagg PD-TP=1: 0.23 RPS
- **Cross-host disagg, 4 encoders, 8img/1080p: 0.13 RPS** (worse than same-host disagg)
- **Cross-host disagg, 4 encoders, 4img/768p: ≥1.02 RPS** (matches/beats TP=2 agg)

**Take-away:** cross-host disagg can match same-host TP=2 agg, but only on
workloads small enough (≤~127 MB embedding/request) that the per-request CPU
bounce on the PD doesn't saturate.

## Full result table (rate=1.0, 64 prompts, np=64)

### 4 images / 768p (≈127 MB NIXL descriptor)

| Encoders | Throughput | Mean E2E | Median TTFT | P99 TTFT | Median TPOT | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.43 RPS | 121.4 s | 93.2 s | 128.3 s | 18.2 ms | encoder-side serialised; running-req peaked at 2 |
| **4** | **1.02 RPS** | **12.0 s** | **5.5 s** | 16.5 s | 20.9 ms | **NOT saturated**; running-req peaked at 15; PD scheduler healthy |

### 8 images / 1080p (≈638 MB NIXL descriptor)

| Encoders | Throughput | Mean E2E | Median TTFT | P99 TTFT | Median TPOT | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 1 | ≈0.083 RPS | very large | very large | — | — | killed early; estimate from log drainage |
| **4** | **0.13 RPS** | 411.0 s | **368.1 s** | 460.6 s | 16.8 ms | running-req only 2-3; PD-side NIXL CPU pipeline saturated |

### 8 images / 768p — middle-ground workload (≈242 MB NIXL descriptor estimated)

| Encoders | Throughput | Mean E2E | Median TTFT | P99 TTFT | Median TPOT | Notes |
|---:|---:|---:|---:|---:|---:|---|
| **4** | **0.68 RPS** | 45.9 s | **21.0 s** | 35.5 s | 197 ms | near-saturation at rate=1.0; running-req peaked at 30 |

### Saturation curve at 4img/768p (rate sweep, 4 encoders)

| Rate | Throughput | Mean E2E | Median TTFT | Mean TPOT | Concurrency | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1.0 | 1.02 RPS | 12.0 s | 5.5 s | 58 ms | 12.2 | below sat |
| **1.5** | **1.37 RPS** | 13.9 s | 7.4 s | 183 ms | 19.1 | near sat |
| 2.0 | 1.32 RPS | 30.5 s | 16.8 s | 395 ms | 40.3 | over sat |

**Saturation ≈ 1.35 RPS for 4img/768p with 4 encoders.**

## Why 4 encoders helps 4img/768p but not 8img/1080p

The NIXL `ReadOperation` log on the PD side shows:

```
ReadOperation(operation_kind=READ,
  local_descriptors=ptr=..., size=668467200, device=cpu,    ← 638 MB CPU buffer
  remote_descriptors=ptr=..., size=668467200, device=cpu,
  ...)
```

`device=cpu` on **both** sides means each request requires:
1. Encoder GPU → encoder CPU memory copy on B70 (host-staging)
2. CPU buffer registration with NIXL agent on B70
3. RoCE wire transfer (fast: 638 MB / 50 GB/s ≈ 13 ms)
4. CPU buffer registration with NIXL agent on giga01
5. Numpy descriptor handling on giga01 (no cupy installed: log shows
   `WARN dynamo.nixl_connect: Failed to load CuPy ... utilizing numpy`)
6. CPU buffer → GPU 4 staging
7. SGLang prefill + decode

Steps 1-2 and 4-6 are CPU-side per-request overhead. Empirically:

- **8img/1080p:** 64 reqs × 638 MB / 499 s = **81.7 MB/s** effective transfer rate
- **4img/768p:** 64 reqs × ~127 MB / 63 s = **128 MB/s** effective transfer rate

Both are running at roughly **~100 MB/s NIXL throughput**, regardless of payload
size. This means the bottleneck is **fixed CPU-side per-request work** (~5-8 s),
not the wire. The wire is doing 100 MB/s out of 50 GB/s — **~500× slower than
hardware capable of**.

### Adding more encoders only helps if the PD's CPU pipeline isn't yet saturated

- 4img/768p: PD's CPU pipeline can handle ≈8 in-flight requests before backing up.
  With 4 encoders generating embeddings in parallel, the PD reaches `running-req=15`
  and decode runs healthily.
- 8img/1080p: PD's CPU pipeline saturates at only 2-3 in-flight requests because
  each request occupies ~5× more CPU bandwidth/memory. More encoders just pile up
  more pending NIXL reads that the PD can't drain. `running-req` capped at 2-3.

## Topology details discovered

- B70 has **4 encoder instances** registered in etcd (after the operator restarted
  the encoder side; was 1 instance during initial 8img/1080p run). All 4 use
  mgmt-IP TCP transport for the dynamo request plane:
  ```
  v1/instances/dynamo/encoder/generate/<id>
  {"transport":{"tcp":"172.26.46.13:<port>/.../generate"}}
  ```
- giga01 PD also binds dynamo TCP request plane to the mgmt IP:
  `172.26.46.75:37049`. Confirmed via `Started TCP server bound successfully
  actual=172.26.46.75:37049` in pd_worker_giga01.log.
- The control-plane traffic (request routing, chat completion deltas) thus
  flows over the slow mgmt fabric. Per-request RTT is fine (0.35 ms ping) but
  bandwidth caps somewhere below the 400 Gb/s RoCE.
- Only the **NIXL data plane** uses RoCE (configured via
  `VLLM_NIXL_SIDE_CHANNEL_HOST=192.165.123.52` on giga01, equivalent on B70).

## Issues / open questions

1. **The `cupy` warning at startup**:
   `WARN dynamo.nixl_connect: Failed to load CuPy for GPU acceleration, utilizing numpy`
   This means descriptor handling falls back to CPU/numpy paths. Installing cupy
   in `/opt/venv` would skip half of the per-request CPU copies. Worth trying
   before the structural patch.

2. **The structural fix**: pre-allocated GPU NIXL descriptor pool (round 4 patch
   from SESSION_MEMORY). Would eliminate the CPU-bounce entirely. Same fix needed
   for both same-host and cross-host disagg.

3. **Dynamo TCP request plane should be configurable** to bind to RoCE IP. Right
   now it picks the OS default-route IP automatically (mgmt). For correctness it
   works (B70 reaches giga01 over mgmt), but for low-latency control plane it's
   suboptimal.

## Files

- giga01 PD launch script:
  `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh`
- Logs:
  `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
  `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/frontend_giga01.log`
  `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/bench_giga01_*.log`
- Bench results:
  `/hongming/res12_crosshost_giga01/h200_pd_b70_encoder_tp1_32b_image{4,8}_*np64/`

## Same-host agg baselines (giga01) — measured 2026-05-24

To compare cross-host disagg fairly, ran TP=1 and TP=2 agg saturation sweeps
on the same giga01 host with the same workloads.

### TP=1 agg saturation sweep (GPU 4, mem-fraction=0.85, max-running=40)

| Workload | rate=0.5 | rate=1.0 | rate=1.5 | rate=2.0 | rate=3.0 | rate=4.0 | rate=5.0(np=128) | **Sat RPS** | Best median TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4img/768p | — | 1.07 | — | 2.03 | 2.68 | 3.67 | **2.65** | **~2.7** | 0.8 s |
| 8img/768p | — | 0.90 | 1.23 | 1.45 | — | — | — | **~1.5** | 1.6 s |
| 8img/1080p | 0.46 | 0.47 | — | — | — | — | — | **~0.47** | 8.1 s |

(rate=5.0 with 128 prompts triggered queue-overflow at 4img/768p — TTFT 14.5 s,
throughput dropped to 2.65 RPS = saturation confirmed.)

### TP=2 agg saturation sweep (GPUs 4+5, mem-fraction=0.85, max-running=25)

| Workload | rate=0.5 | rate=0.75 | rate=1.0 | rate=1.5 | rate=2.0 | rate=2.5 | rate=3.0 | rate=4.0 | rate=5.0(np=128) | **Sat RPS** | Best median TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4img/768p | — | — | 1.08 | — | 2.10 | — | 2.94 | 3.22 | **3.05** | **~3.1** | 1.0 s |
| 8img/768p | — | — | 0.92 | 1.36 | 1.68 | 1.90 | — | — | — | **~1.9** | 1.7 s |
| 8img/1080p | 0.50 | 0.74 | 0.60 | — | — | — | — | — | — | **~0.6-0.7** | 3.5 s |

(8img/1080p TP=2 at rate=1.0 saw median TTFT 39 s — over-saturation. Same as
TP=1 — at large workloads, increasing TP doesn't help much because encoder ViT
is the bottleneck.)

### Summary: agg saturation RPS by workload

| Workload | TP=1 sat | TP=2 sat | TP=2 / TP=1 |
|---|---:|---:|---:|
| 4img/768p (≈127 MB embedding) | ~2.7 | ~3.1 | 1.15× |
| 8img/768p (≈242 MB embedding) | ~1.5 | ~1.9 | 1.27× |
| 8img/1080p (≈638 MB embedding) | ~0.47 | ~0.6 | 1.28× |

**Observations:**
- TP=2 only buys ~15-30% over TP=1 at saturation. Most LLM compute is
  not the bottleneck — the encoder's ViT is.
- Smaller workloads scale better with TP=2 (more GPU compute relative to
  ViT cost). At 8img/1080p both TPs are heavily ViT-bound.
- All TP=1/TP=2 numbers here are **with same-host EPD aggregated**, where
  each request runs encoder+prefill+decode on the same GPU(s). The encoder
  ViT runs on the same GPU as the LLM — there's no separate encoder process.

## Cross-host disagg vs same-host agg — final comparison

(rate=1.0 column = throughput at fixed rate=1.0 target; sat = measured saturation)

### 4img/768p workload

| Config | RPS @ rate=1.0 | Sat RPS | Median TTFT (best) | Compute used |
|---|---:|---:|---:|---|
| TP=1 agg same-host | 1.07 | **~2.7** | **0.8 s** | 1× H200 |
| TP=2 agg same-host | 1.08 | ~3.1 | 1.0 s | 2× H200 |
| Cross-host disagg, 4 encoders | 1.02 | ~1.35 | 5.5 s | 1× H200 (PD) + 4× B70 GPUs (encoders) |

**Verdict:** Same-host TP=1 agg matches cross-host disagg at rate=1.0 but is
2× faster TTFT, and has 2× the saturation ceiling — using only 1 GPU and no
remote encoder. Cross-host disagg loses on every dimension here.

### 8img/768p workload

| Config | RPS @ rate=1.0 | Sat RPS | Median TTFT (best) |
|---|---:|---:|---:|
| TP=1 agg same-host | 0.90 | **~1.5** | **1.6 s** |
| TP=2 agg same-host | 0.92 | ~1.9 | 1.7 s |
| Cross-host disagg, 4 encoders | 0.68 | ~0.7 | 21 s |

**Verdict:** Same-host TP=1 agg ~2× faster sat than cross-host, ~13× better
TTFT.

### 8img/1080p workload

| Config | RPS @ rate=1.0 | Sat RPS | Median TTFT (best) |
|---|---:|---:|---:|
| TP=1 agg same-host | 0.47 (sat) | **~0.47** | **8.1 s** |
| TP=2 agg same-host | 0.60 | ~0.6-0.7 | 3.5 s |
| Cross-host disagg, 4 encoders | 0.13 | ~0.13 | 368 s |

**Verdict:** Same-host TP=2 agg dominates: 4-5× faster sat than cross-host,
100× better TTFT.

## Conclusion

**Same-host aggregated EPD beats cross-host disagg on every workload tested**,
even though cross-host has the advantage of more total compute (1+4 GPUs vs
1-2 GPUs). The bottleneck for cross-host is the per-request CPU bounce in
the NIXL embedding transfer, which prevents the PD's GPU from being kept busy.

The structural fix (GPU NIXL descriptor pool patch from SESSION_MEMORY round 4)
would close most of this gap, but it's a multi-day engineering task in dynamo
upstream.

For production today on this stack: **same-host EPD aggregated is the right
choice** for multimodal serving. Cross-host disagg is only attractive when:
1. The encoder MUST run on a different host (e.g., security/isolation reqs)
2. The workload is small enough that the CPU-bounce overhead is tolerable
3. The encoder side has so much more compute that the sum still wins despite
   the cross-host overhead

None of these conditions appear true for the 32B-FP8 workload tested here.

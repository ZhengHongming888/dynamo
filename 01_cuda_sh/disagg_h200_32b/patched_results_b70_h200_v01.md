# Cross-host disagg with full GPUDirect RDMA (B70 XPU → giga01 H200)

**Date:** 2026-05-24
**Configuration:** First successful end-to-end GPU→GPU NIXL embedding transfer in this investigation.

## TL;DR

- **Full GPUDirect RDMA path confirmed working**: B70 Intel XPU encoder → giga01 NVIDIA CUDA PD, no CPU bounce.
- 65/65 NIXL transfers showed `local=cuda:0, remote=xpu:0`. Zero CPU descriptors in this run.
- **But throughput at 8img/1080p didn't improve vs the unpatched run**: 0.13 RPS in both cases.
- The bottleneck for this workload is the encoder ViT compute on B70, not the embedding transfer.
- One genuine improvement: **median TPOT 2.37 ms** (vs 18 ms unpatched) — decode itself is 8× faster on the PD because the embedding is already on GPU.

## Cluster setup

### giga01 (H200 PD side, this host)

- **Patches applied** to `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py`:
  - Line 882 (warmedup pool init): `device=_nixl_buffer_device()` added
  - Line 915 (fallback path): `device=_nixl_buffer_device()` added
- `mem-fraction-static=0.65`, `max-running-requests=64`
- GPU 4, NIC `mlx5_4:1` (RoCE 192.165.123.52, NUMA 2)
- Launch script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh`

### B70 (encoder side, sc09giga01-b70 — Intel Battlemage XPUs)

Per `b70_patched.md` from the B70 operator:

- **Patch applied** to `/usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py`:
  - Line 219: `target_device = _nixl_buffer_device()` (was `torch.device("cpu")`)
  - Same patch as documented in `B70_PATCH_INSTRUCTIONS.md`
- **Critical fix:** Removed `NIXL_USE_CPU_HOST_MEMORY=1` env var from `start_sglang_pd_xpu_32b_b70_4E.sh`, which had been silently forcing `_nixl_buffer_device()` back to CPU. Without removing this env var, the patch was a no-op.
- **Encoder loaded `--encoder-only`** to fit the 32 GiB B70 (avoids loading the 33 GiB FP8 LM weights).
- **`mem-fraction-static=0.5`** for headroom.
- 4 encode workers on XPUs 0, 1, 2, 3 (all NUMA 0).
- Per-XPU NIC selection: XPU 0,2 → mlx5_0:1; XPU 1,3 → mlx5_1:1.
- UCX_TLS=ze_copy,rc,tcp (Intel Level Zero, not cuda_copy).

### Verification of full-GPU path

NIXL log on giga01 PD (sample):

```
ReadOperation(operation_kind=READ,
  local_descriptors=ptr=0x7ea210000000,  size=668467200, device=cuda:0,
  remote_descriptors=ptr=0xffffd557dde00000, size=668467200, device=xpu:0,
  notification_key='27a9f936-...', remote='e74db3b519cc41898f7031d58c501f53-1',
  status='DONE')
```

Both ends are GPU memory. `_resolve_remote_mem_type()` in dynamo's `nixl_connect/__init__.py:373-380`
maps the remote XPU descriptor to the local CUDA-equivalent memory module so NIXL can do the
RDMA read directly without CPU staging.

Counts for this entire bench run (65 total transfers):

| local | remote | count |
|---|---|---:|
| cuda:0 | xpu:0 | 65 (100%) |
| cuda:0 | cpu | 0 |
| cpu | * | 0 |

## Bench: 8img/1080p, rate=1.0, 64 prompts

```
Successful requests:                64/64    ← clean run, no cancellations
Benchmark duration (s):             492.58
Total input tokens:                 1,050,538
Total output tokens:                7,474
Request throughput (req/s):         0.13
Input token throughput (tok/s):     2,132.74
Output token throughput (tok/s):    15.17
Peak concurrent requests:           64
Concurrency:                        53.25
```

### End-to-End Latency

```
Mean E2E Latency (ms):              409,842   (6.83 min)
Median E2E Latency (ms):            402,612   (6.71 min)
P90 E2E Latency (ms):               468,835   (7.81 min)
P99 E2E Latency (ms):               483,637   (8.06 min)
```

### Time to First Token

```
Mean TTFT (ms):                     387,750   (6.46 min)
Median TTFT (ms):                   392,785   (6.55 min)
P99 TTFT (ms):                      459,811   (7.66 min)
```

### Time per Output Token (decode latency)

```
Mean TPOT (ms):                     858.61
Median TPOT (ms):                   2.37     ← extremely fast decode (was 18ms unpatched)
P99 TPOT (ms):                      14,840   (occasional bursts)
```

### Inter-Token Latency (decode-only)

```
Mean ITL (ms):                      315.98
Median ITL (ms):                    1.77     ← extremely fast (was ~9ms unpatched)
P95 ITL (ms):                       12.66
P99 ITL (ms):                       14.41
Max ITL (ms):                       215,466
```

## Comparison with prior 8img/1080p runs

| Configuration | RPS @ rate=1.0 | Mean E2E | Median TTFT | Median TPOT | NIXL Path |
|---|---:|---:|---:|---:|---|
| Cross-host disagg, unpatched 1-encoder | 0.083 | very large | very large | — | cpu→cpu |
| Cross-host disagg, unpatched 4-encoder | 0.13 | 411.0 s | 368.1 s | 18 ms | cpu→cpu |
| **Cross-host disagg, full-GPU 4-encoder (this run)** | **0.13** | **409.8 s** | **392.8 s** | **2.37 ms** | **xpu→cuda** |
| Same-host disagg PD-TP=1 (memory) | 0.23 | ~80 s | — | — | (cuda_ipc) |
| Same-host TP=1 agg | 0.47 | (small) | 8.1 s | — | (no NIXL) |
| Same-host TP=2 agg | 0.6-0.7 | (small) | 3.5 s | — | (no NIXL) |

**The full-GPU patch path is statistically indistinguishable from unpatched at 8img/1080p:**
0.13 RPS in both, ~410s mean E2E in both. The only real difference is TPOT: **2.37 ms patched vs 18 ms unpatched** = 7.6× faster decode. But because TTFT dominates total latency at this workload, that gain is invisible at the bench level.

## Why GPUDirect doesn't help here

The cross-host pipeline for one request is:

```
[B70: encoder ViT compute on XPU]
         |
         v
[B70: NIXL register XPU buffer, expose as readable]
         |
         v
[B70 ↔ giga01: RoCE wire transfer ~13 ms for 638 MB at 50 GB/s]
         |
         v
[giga01 PD: NIXL receive buffer on CUDA GPU, ready for prefill]
         |
         v
[giga01 PD: SGLang prefill+decode]
```

Earlier hypothesis: the CPU bounce on each end made the per-request cost dominant. Removing
the CPU bounce should be a big win.

What we measured:
- Per-request PD `forward_duration`: ~4.35 s (LLM compute)
- Per-request inter-arrival at PD: ~9-15 s
- Encoder→PD gap: ~9 seconds (encoder ViT + NIXL setup + PD enqueue)

The 9-second encoder→PD gap is **not** the wire transfer (which would have been a CPU-bounce
~5-8 s in unpatched mode). After patching, the wire is now GPUDirect and ~13 ms in theory, but
the gap stayed at ~9 s. So the gap is NOT the embedding transfer — it's something else:

1. **Encoder ViT compute on B70 Intel XPUs** for 8 images at 1920x1080 (~13 GB pixels processed).
2. **NIXL setup overhead** on each request (creating Descriptor, registering with NIXL agent,
   serializing metadata, etc).
3. **Cross-host TCP control plane** request routing (frontend → encoder → PD).
4. **Possibly:** dynamo's per-request bookkeeping costs that don't change with descriptor device.

Without instrumentation to break down the 9 s further, we can't pinpoint which dominates.
But the conclusion is clear: **at 8img/1080p, the encoder pipeline is the bottleneck, not
the data transfer**.

## What the patch DID help

1. **TPOT/ITL collapsed to ~2 ms** (from ~18 ms unpatched) — the decode side has zero overhead
   because the embedding is already on the right GPU. This matters for output-heavy workloads
   (long generation) but doesn't dominate when output is just 256 tokens like in this bench.

2. **No CPU pressure on the PD host** — the previous CPU bounce of 638 MB/request × ~10 in-flight
   requests = ~6 GB/s of CPU memory bandwidth used for staging. Now zero. This frees CPU for
   other work but isn't visible in throughput numbers.

3. **No SSE timeouts / stale expirations** — possibly because the system-level pipeline is
   smoother with GPUDirect (less CPU contention, fewer scheduler hiccups). Bench completed
   64/64 cleanly, vs prior patched-asymmetric runs that had stale-expirations and broken
   response paths.

## Conclusions

1. **Patch is structurally correct and works as designed**: GPUDirect RDMA xpu→cuda is fully
   functional. Both ends handle GPU descriptors. NIXL transports route around device-type
   mismatches via `_resolve_remote_mem_type`.

2. **No throughput improvement at 8img/1080p** because the encoder is the bottleneck. Same
   0.13 RPS as unpatched.

3. **Latency improvements are decode-only** (TPOT 8× better) and don't affect TTFT-dominant
   workloads.

4. **For workloads where the encoder is fast enough that data transfer matters**, the patch
   should help. Likely candidates: smaller workloads (4img/768p, 8img/768p) where encoder
   ViT is shorter relative to the transfer step.

5. **Remaining bottleneck investigation** would require instrumenting the 9-second encoder→PD
   gap to determine its components.

## Files

- giga01 PD launch script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh`
- giga01 patches: `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py` lines 882 & 915
- giga01 PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01.log`
- Frontend log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/frontend_giga01.log`
- Bench result: `/hongming/res17_xhost_4enc_pd_xpu_full_gpu/8img_1080p_rate1.0/benchmark_output.json`
- Bench log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/bench_xpu_full_gpu_8img_1080p_rate1.0.log`
- B70 patch report: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/b70_patched.md`
- giga01 → B70 response: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/giga01_to_b70_response.md`

## Pending tests (good candidates if you want to continue)

1. **4img/768p @ rate=1.0** with full-GPU path: previously 1.02 RPS unpatched. If transfer
   actually matters at smaller sizes, expect higher RPS or lower TTFT.
2. **8img/768p @ rate=1.0**: previously 0.68 RPS unpatched. Mid-size test.
3. **rate=1.5 or 2.0 at 4img/768p**: find new saturation point with full GPU.
4. **Instrument the encoder→PD gap** to break down what consumes the ~9 s.

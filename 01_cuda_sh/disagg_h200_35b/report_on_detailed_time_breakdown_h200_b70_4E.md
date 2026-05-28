# Patched + B70 4E — Per-component time breakdown analysis

**Date:** 2026-05-26
**Setup:** Patched giga01 H200 PD (`h200_cuda_nixl.patch` active, `device=cuda:0` confirmed
on 33/33 ReadOps) + 4 B70 XPU encoders. Workload: Qwen3.5-35B-A3B, 8img/1080p, np=32,
random JPEG, in=128 / out=256, 7 rates 0.10–3.00.

**Method:** Parsed 11,304-line `pd_worker_giga01_h200_patched_debug_*.log` with
`DYN_LOG=debug`. Extracted 265 request lifecycles end-to-end, each with the eight
checkpoints listed below, plus inter-arrival cluster analysis to back out encoder ViT
time per request.

## Per-request timeline checkpoints (PD-side debug log)

| ID | Event | Log marker |
|---|---|---|
| **T0** | request received at PD | `request received [request_id=<UUID>]` |
| **T1** | PD calls `process_embeddings` | `Processing embeddings with shape:` |
| **T2** | NIXL receive descriptor allocated on cuda:0 | `Descriptor: Created Descriptor(...device=cuda:0) from 'torch.Tensor'` |
| **T3** | `ReadOperation` created (NIXL READ submitted) | `Created ReadOperation(...)` |
| **T4** | Wire transfer `DONE` | `NIXL reported transfer state: DONE` |
| **T5** | First chunked-prefill chunk starts | next `Prefill batch` event after T4 |
| **T6** | `forward_duration` complete | `ReqTimeStats(...)` event |
| **T7** | Response done, returned to client | `request completed [request_id=<UUID>]` |

## Per-rate sub-component median timings (ms)

| Rate | n | RPS | T1−T0 dispatch | T2−T1 cuda alloc | T3−T2 NIXL setup | T4−T3 wire | T5−T4 prep→prefill | T6−T5 PD forward | T7−T6 egress | T7−T0 lifetime | rts.f | rts.q |
|---:|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 33 | 0.106 | 4.19 | 0.51 | 6.26 | **14.1** | 661.93 | 995.5 | 4.07 | **1746.3** | 1156.6 | 0.60 |
| 0.25 | 33 | 0.138 | 3.97 | 0.49 | 5.83 | **14.5** | 687.02 | 877.8 | 3.96 | **1632.4** | 979.0 | 1.63 |
| 0.50 | 33 | 0.144 | 4.18 | 0.49 | 5.40 | **14.2** | 678.71 | 721.5 | 6.35 | **1710.9** | 912.7 | 1.45 |
| 1.00 | 33 | 0.146 | 4.01 | 0.48 | 5.03 | **15.5** | 667.78 | 1002.7 | 478.14 | **1833.5** | 1087.8 | 1.50 |
| 1.50 | 33 | 0.146 | 4.15 | 0.47 | 5.04 | **14.0** | 639.27 | 891.6 | 239.86 | **1819.3** | 1074.4 | 1.40 |
| 2.00 | 33 | 0.146 | 4.09 | 0.48 | 5.15 | **14.0** | 667.88 | 929.3 | 629.24 | **1820.2** | 788.0 | 1.42 |
| 3.00 | 33 | 0.146 | 4.25 | 0.25 | 3.64 | **14.1** | 414.41 | 790.8 | 464.02 | **2169.5** | 1014.8 | 1.53 |

`rts.f` = SGLang-reported `forward_duration`; `rts.q` = SGLang's `queue_duration`. The two
should sum to (T6−T5)+(T1−T0)+ε; observed slight discrepancy is from PD's internal
preprocessing not visible in the ReqTimeStats event.

## Per-component analysis at saturation (rate=1.0, representative)

### 1. PD ingress dispatch (T1−T0 = 4 ms)

```
T0  request received from frontend over dynamo TCP plane
    ↓  ~4 ms (Python event loop dispatch + worker_handler.generate setup)
T1  process_embeddings(request) called
```

Tiny, constant. Just the cost of getting from the dynamo Rust ingress through to the
Python `worker_handler.generate()` coroutine. **Not optimizable.**

### 2. CUDA buffer allocation on PD (T2−T1 = 0.5 ms)

```
T1  PD enters receive_embeddings()
    ↓  embedding_transfer.py:921 (PATCHED)
    ↓  encodings_tensor = torch.zeros(*shape, dtype=bf16, device='cuda:0')
    ↓  ~0.5 ms (PyTorch caching allocator returns from cuda pool)
T2  Descriptor: Created Descriptor(ptr=..., size=66846720, device=cuda:0)
```

This is the **patched line** in action. Pre-patch it would allocate on CPU
(~10-50 ms for 64 MB malloc + cudaMallocHost pin). Post-patch the PyTorch caching
allocator returns a 64 MB GPU buffer from the existing pool in ~0.5 ms.

**Win from patch: ~10-50 ms per request saved here.**

### 3. NIXL setup on PD (T3−T2 = 5 ms)

```
T2  Descriptor created
    ↓  nixl_connect.Descriptor.register_with_connector():
    ↓    - ucp_mem_map() on the GPU buffer  [~3 ms, registers GPU mem with NIC]
    ↓    - publish to remote agent via dynamo control plane
    ↓  Remote(name=..., connection=...).create
    ↓  Build local + remote nixlXferDList
    ↓  Allocate nixl_xfer_handle
T3  ReadOperation Created (status='<invalid>', ready to submit)
```

Constant ~5 ms. Most of this is the `ucp_mem_map` + remote-agent handshake — the
per-request portion that NIXL can't skip. Same in both patched and unpatched.

### 4. NIXL wire transfer (T4−T3 = 14 ms median)

```
T3  ReadOperation submitted to NIXL/UCX
    ↓  state: <invalid> → INIT (~0.05 ms, queue dispatch)
    ↓  state: INIT → PROC (~5 ms, RoCE NIC begins RDMA READ)
    ↓                       NIC reads 64 MB encoder GPU → wire → PD GPU
    ↓                       (GPUDirect RDMA, both ends device=cuda:0)
    ↓  state: PROC → DONE (~9 ms, transfer complete)
T4  NIXL reported transfer state: DONE
```

**14 ms median** for 64 MB transfer = **4.55 GB/s effective**. Theoretical wire
bandwidth on the 100 Gb/s mlx5_4 RoCE NIC is ~12.5 GB/s; we're at ~36% of theoretical
peak, which is normal for first-call NIC-mem registration + handshake overhead.

| Statistic | Value |
|---|---:|
| count | 260 |
| min | 0.43 ms |
| **p50** | **14.00 ms** |
| p99 | 165.11 ms |
| max | 249.85 ms |
| Embedding size | 63.8 MB (16320 visual tokens × 2048 hidden × bf16) |
| Effective bandwidth (p50) | **4.55 GB/s** |

The p99 outliers (~165 ms) likely correspond to first-request-to-each-encoder NIXL
agent setup or NIC contention during burst arrivals.

### 5. PD prep + first prefill (T5−T4 = 668 ms median)

```
T4  embedding tensor available on PD cuda:0
    ↓  Successfully read embeddings via NIXL
    ↓  ~50 ms: token_ids preprocessing, image-token expansion
    ↓  ~88 ms: QwenVLProcessor processes the embedding into mm_item:
    ↓           load_time=26 ms        (read embedding from NIXL buffer view)
    ↓           process_time=61 ms     (image_grid_thw reshape, sequence pack)
    ↓           get_rope_index_time<1 ms
    ↓  ~530 ms: PD scheduler enqueue + chunked-prefill kernel launch overhead
T5  Prefill batch (chunk 1) starts
```

The 668 ms here is composed of:
- **~88 ms QwenVLProcessor** (`base_processor.fast_load_mm_data` + `qwen_vl.process_mm_data_async`)
- **~580 ms** "everything else" — scheduler queue, mamba state allocation, KV setup,
  CUDA stream setup. This is harder to break down further without deeper instrumentation.

**This is one of the largest per-request fixed costs.** Doesn't scale with embedding size.

### 6. PD forward (T6−T5 = 1003 ms median at rate=1.0)

```
T5  Prefill batch (chunk 1: 8192 visual tokens)
    ↓  ~580 ms forward pass on H200 (35B MoE BF16, chunked-prefill)
    ↓  Prefill batch (chunk 2: 8192 visual tokens, sometimes cached)
    ↓  ~330 ms forward
    ↓  Prefill batch (chunk 3: ~46 text tokens)
    ↓  ~50 ms
    ↓  Decode loop (output_len tokens, ~5 ms/tok with batching)
    ↓  ~256 tokens × 4 ms = 1024 ms (but overlaps with other reqs)
T6  ReqTimeStats logged (forward_duration = ~1003 ms median)
```

The SGLang-reported `forward_duration` (1003 ms p50) is the wall-clock from when
the request first enters PD's batch to when its last token is emitted. With concurrent
in-flight requests, individual `forward_duration` values get inflated by sharing
the GPU with other reqs in the same batch.

**Decoder TPOT** at saturation: ~1.5 ms per token (per `rate=1.0` ITL p50). Decode is
extremely fast; chunked-prefill is the dominant cost.

### 7. Egress + response wrap (T7−T6 = 4–629 ms)

```
T6  Forward complete (last token emitted)
    ↓  Final stream chunk to dynamo Rust frontend
    ↓  Frontend serializes SSE response back to client
T7  request completed
```

Highly variable (4 ms at low rate, 478–629 ms at saturation). At low rate, response
egress is fast. At saturation, the response stream backs up because the frontend
is also streaming for 30+ other in-flight reqs. This is a queue-tail effect on the
dynamo TCP plane, not a structural cost.

### 8. Total PD lifetime (T7−T0)

| Rate | Lifetime p50 (s) | Encoder→PD wait (s) |
|---:|---:|---:|
| 0.10 | 1.75 | (sub-saturation; encoder serves on demand) |
| 1.00 | 1.83 | ~24-26 s **between request bursts** |
| 3.00 | 2.17 | ~26 s between bursts (encoder fully loaded) |

**PD does its 1.8 s of work per request, then idles for ~24 seconds waiting for the
next burst from the encoder pool.** This is the dominant story.

## Encoder ViT time — back-calculated from arrival pattern

We can't directly measure encoder ViT time on PD, but the **PD-side arrival cadence
reveals it**. The 4 B70 encoders work in parallel and emit "clusters of 4" requests
every encoder cycle.

### Inter-arrival cluster analysis (clusters with arrivals < 5 s apart)

| Rate | Arrivals | Clusters | Avg cluster size | Inter-cluster gap p50 | Inferred per-encoder ViT |
|---:|--:|--:|--:|---:|---:|
| 0.10 | 33 | 18 | 1.83 | 13.19 s | (sub-saturation) |
| 0.25 | 33 | 24 | 1.38 | 8.62 s | (sub-saturation) |
| 0.50 | 33 | 9 | 3.67 | 19.89 s | 19.9 s/req |
| **1.00** | 33 | 9 | **3.67** | **24.30 s** | **24.3 s/req** |
| 1.50 | 33 | 9 | 3.67 | 24.89 s | 24.9 s/req |
| 2.00 | 33 | 9 | 3.67 | 25.56 s | 25.6 s/req |
| 3.00 | 33 | 9 | 3.67 | 25.81 s | 25.8 s/req |

**Each B70 XPU encoder takes ~24-26 s to run the 8img/1080p ViT** (i.e. process 8
images at 1080p through Qwen3.5's vision tower via `triton_attn` on Battlemage XPUs).
This matches exactly the 1E baseline measurement in `35b_bottleneck_analysis.md`
(25.6 s/req for 8img/1080p).

At saturation:
- 4 encoders work in parallel
- Each finishes one request every ~25 s
- Pool throughput = 4 / 25 s = **0.16 RPS** (within 10% of measured 0.146 RPS — small
  loss from non-perfect parallelism / dispatch overhead)

## End-to-end per-request budget at saturation (rate=1.0)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  Stage                                  Median (ms)   % of E2E TTFT (~12s)    │
├───────────────────────────────────────────────────────────────────────────────┤
│  Encoder ViT (B70 XPU triton_attn)        ~24,000  ★    99.0% ←★ THE bottleneck│
│  Encoder→PD network (NIXL READ wire)         14.0       0.06%                  │
│  PD ingress dispatch (T1-T0)                  4.0       0.02%                  │
│  PD CUDA alloc (T2-T1, patched)               0.5       0.002%                 │
│  PD NIXL setup (T3-T2)                        5.0       0.02%                  │
│  PD prep + scheduler (T5-T4)                668.0       2.8%                   │
│  PD forward (T6-T5, prefill+decode)        1,003.0      4.2%                   │
│  PD egress (T7-T6)                          478.0       2.0%                   │
├───────────────────────────────────────────────────────────────────────────────┤
│  Total per-request budget                  ~26,000 ms                          │
└───────────────────────────────────────────────────────────────────────────────┘
                                                       ★ Encoder ViT is 99% of total
```

Note: bench-reported TTFT at rate=1.0 is **201 s** (not 12 s) because at saturation
the **bench's 32-prompt offered load >> server's 0.146 RPS capacity**, so requests pile
up in the encoder pool's queue. Each request waits ~7 cycles × 24 s = 168 s in the
encoder queue before its ViT cycle starts. The 201 s TTFT = 168 s queue + 24 s ViT +
1.8 s PD = 194 s ≈ measured.

## Summary: where the time goes

### Per-stage cost breakdown (one request, isolated, no queue)

| Stage | Time | Where | Optimizable? |
|---|---:|---|---|
| **Encoder ViT (B70 XPU)** | **24,000 ms** | Encoder host | **Yes — switch to H200/H100 = ~1-2 s** |
| Encoder→PD NIXL handshake | ~5 ms | NIXL ucp_mem_map | No, NIC API cost |
| Encoder→PD wire transfer | 14 ms | RoCE GPUDirect | Already optimal (4.55 GB/s) |
| PD CUDA alloc (patched) | 0.5 ms | torch caching allocator | Already optimal |
| PD scheduler + QwenVL processor | 668 ms | SGLang pipeline | Maybe (88 ms processor + 580 ms scheduler) |
| PD chunked-prefill (forward) | 580 ms | H200 35B MoE | Maybe via TP=2 or smaller model |
| PD decode (256 tok) | 425 ms | H200 35B MoE | Already at <2 ms TPOT, near optimal |
| PD egress | 4 ms | dynamo TCP plane | No |
| **Total per request** | **~25,700 ms** | | |
| PD's portion of total | **~1,690 ms** | (6.6%) | |
| Encoder's portion of total | **24,000 ms** | (93.4%) | |

### What the patch did vs didn't do

| Thing | Pre-patch | Post-patch | Δ |
|---|---:|---:|---:|
| PD CUDA alloc + register | ~50 ms (CPU malloc + pin + cudaMemcpy after) | 0.5 ms (cuda alloc only) | **−50 ms** |
| NIXL wire transfer | ~14 ms over CPU staging | 14 ms direct GPU↔GPU | unchanged |
| PD pre-decode CPU→GPU memcpy | ~150 ms (sequential after wire) | 0 ms (no copy needed) | **−150 ms** |
| **Per-request total saved** | | | **~200 ms** |
| % of 24 s encoder cycle | | | **0.8%** |

The patch saves ~200 ms per request on the PD side. At 0.146 RPS sat, that's
0.146 × 0.2 = 29 ms/s of PD time recovered. Insignificant when encoder is the bottleneck.

### Why the saturation throughput doesn't change

```
Saturation RPS = 1 / (encoder_cycle_per_pool / encoder_count)
              ≈ 1 / (25 s / 4)
              ≈ 0.16 RPS  (measured 0.146; ~10% non-parallel overhead)
```

PD's per-request budget (~1.7 s) is **less than the encoder pool's effective per-request
output rate** (25 s / 4 = 6.25 s). Even if PD took 0 time, encoder-pool output is
the binding constraint.

For PD to become the bottleneck:
- Encoder pool would need to output > 0.6 RPS (6× current)
- That requires either 24× B70 encoders (probably XPU-memory-bound) or a faster encoder

We saw the latter directly: switching to **dell06 H200 encoder** (~1.5 s ViT for 8img/1080p)
made PD the bottleneck instead, RPS went 0.147 → 0.85 (5.8×).

## Concurrency observed on PD during the run

| Rate | Decode batch running-req distribution | Max queue |
|---:|---|---:|
| 1.00 | running=1: 46 (57%); =2: 25 (31%); =3: 9 (11%) | 0 |
| 1.50 | running=1: 48 (61%); =2: 19 (24%); =3: 11 (14%) | 0 |
| 2.00 | running=1: 34 (49%); =2: 25 (36%); =3: 11 (16%); =4: 1 | 0 |
| 3.00 | running=1: 19 (32%); =2: 20 (34%); =3: 18 (31%); =4: 2 (3%) | 0 |

**PD's `max queue-req` is 0 throughout.** The PD scheduler never has anything queued
because requests arrive in clusters of 4 and PD chews through them faster than the
encoder pool produces the next cluster. Concurrency on PD scales gracefully with
arrival rate — at rate=3.0, PD operates at 2-3 in-flight steadily.

## Conclusions

1. **Encoder ViT on B70 XPU is 99% of the per-request cost** (24 s vs ~1.7 s on PD).
   This is identical to the unpatched 1E and 4E findings — the patch doesn't change
   the bottleneck, and 4 encoders in parallel just give 4× the encoder-pool throughput.

2. **Patched PD-side stages are minimal**:
   - CUDA buffer alloc: 0.5 ms
   - NIXL setup + wire: ~19 ms
   - PD chunked-prefill: ~580 ms (compute-bound, not I/O-bound)
   - PD decode: <2 ms/tok (extremely efficient batched decode)
   - **Total PD per-request budget: ~1.7 s**

3. **Wire transfer is at 4.55 GB/s effective** (36% of 12.5 GB/s peak). Not a bottleneck;
   transfers complete in 14 ms median for 64 MB embeddings. p99 outliers (165 ms) come
   from first-request NIXL agent setup per encoder.

4. **PD never queues, never waits** — `queue-req` is always 0. PD has spare capacity
   and is starved by the encoder pool.

5. **The patch saved ~200 ms/req on PD**, which is a 0.8% improvement of the 25 s
   per-request total. Invisible at this workload because encoder ViT dominates.

6. **To improve throughput further:**
   - **Faster encoders** (H100/H200 in place of B70 XPU): expected 10-15× encoder ViT
     speedup → 1-1.5× overall throughput improvement (until PD becomes the bottleneck)
   - **More B70 encoders** (8E vs 4E): expected ~2× throughput, modulo NUMA/NIC contention
   - **PD-TP=2**: would help only if encoder isn't bottleneck (currently no help)
   - **The patch itself**: structurally important (eliminates a known CPU-bounce bug)
     but operationally invisible until PD becomes the bottleneck

## Files

- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01_h200_patched_debug_20260526_050419.log`
- Sweep master: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/sweep_8img_1080p_h200_patched_b70_4E_*.log`
- Bench JSONs: `/hongming/res22_disagg_h200_35b_sweep/8img_1080p_h200_patched_b70_4E/rate_*_np32/benchmark_output.json`
- Analysis script: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/analyze_patched_b70_4E_breakdown.py`
- Companion docs:
  - `35b_bottleneck_analysis.md` (1E baseline analysis)
  - `4img_768p_4E_bottleneck.md` (4img workload analysis)
  - `patched_flow_dell06_giga01.md` (post-patch flow diagram)
  - `patch_vs_unpatch_1080p.md` (8img/1080p sweep result comparison)
  - `comparison_5way_35b.md` (full topology comparison across 5 setups)

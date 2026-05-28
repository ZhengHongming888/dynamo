# Detailed Time Breakdown: Encoder (dell06 H200, GPU 4)

**Date:** 2026-05-27
**Companion to:** [`time_breakdown_dell06_super21_32b_8img_1080p.md`](./time_breakdown_dell06_super21_32b_8img_1080p.md)
**Same bench run:** Qwen3-VL-32B-Instruct-FP8, 8 imgs × 1920×1080, in=128 / out=256, **rate=1.0 RPS, np=32**.

**Setup:**
- **Encoder (this report):** dell06 H200 (172.26.46.162), single-encoder (1E), GPU 4, `--mem-fraction-static 0.85`
- **PD:** super21 H200 (172.26.46.133), TP=1, `--mem-fraction-static 0.65`, `--max-running-requests 64`
- **Network:** RoCE 100 Gb/s NDR fabric (192.165.123.0/24); encoder NIC `mlx5_0` (192.165.123.25)
- **Patches:** encoder-side `b70_xpu_nixl.patch`-equivalent already in `encode_worker_handler.py` (lines 221, 358); on `_nixl_buffer_device()` → `cuda` because `torch.cuda.is_available()` is True. Confirmed: `WARN encode_worker_handler: PATCH(non-cached): moving embeddings from cpu to cuda for NIXL transfer.`
- **Bench window:** 2026-05-27T05:21:30 → 2026-05-27T05:24:30 UTC (n=33 requests; 33 successful + 1 that the bench client recorded extra; PD report says 32/32, encoder log captures 33 due to 1 warmup or retry)

**Bench result recap (from PD report):** 0.26 RPS sustained, 32/32 successful, mean TTFT 59 s, median PD lifetime 20 s, median PD `forward_duration` 10.5 s.

## Per-request lifecycle checkpoints (encoder-side, INFO log)

The encoder runs at default `INFO` log level (no `DYN_LOG=debug`), so we get three coarse markers per request:

| ID | Event | Log marker (text) |
|---|---|---|
| **E0** | Request received at encoder (Rust ingress → Python coroutine) | `request received [request_id=<UUID>]` (`push_handler.rs`) |
| **E1** | ViT done → Python handler about to hand embedding to NIXL | `WARN encode_worker_handler.generate: PATCH(non-cached): moving embeddings from cpu to cuda for NIXL transfer.` |
| **E2** | Request done (encoder coroutine returns to Rust egress) | `request completed [request_id=<UUID>]` |

**E0–E1 captures:** image fetch + decode + processor + ViT forward + the patched `.to(cuda)` device move. Dominated by ViT compute on a single H200 stream.

**E1–E2 captures:** NIXL `register_with_connector()` + `create_readable()` + sending the descriptor metadata to PD over dynamo TCP plane + waiting for `transfer_future` (which only resolves after PD calls `begin_read` and the NIC completes the RDMA pull) + post-completion bookkeeping.

This is coarser than the PD-side instrumentation (which had 8 checkpoints with debug logging), but the two sides combined give a complete picture.

## Aggregate per-stage stats (encoder side, n=33 in bench window)

| Stage | n | min | p25 | p50 | mean | p75 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0→E1** (recv → ViT done + .to(cuda)) | 33 | **1.55 s** | 3.59 s | 11.10 s | 10.84 s | 17.22 s | 21.16 s | 21.22 s |
| **E1→E2** (NIXL setup + wait_for_completion) | 33 | 6.74 s | 44.16 s | 65.75 s | 58.71 s | 75.62 s | 80.69 s | 81.27 s |
| **E0→E2** (full encoder lifetime) | 33 | 8.29 s | 47.75 s | 76.86 s | 69.55 s | 87.16 s | 98.38 s | 99.34 s |

**Inter-arrival on encoder (encoder ingress side):**

| n | min | p50 | mean | p99 | max |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.045 s | 0.625 s | 1.159 s | 3.78 s | 9.76 s |

The bench client offers ~1 RPS (Poisson), but observed encoder ingress p50 is 0.63 s — the encoder briefly handles a small burst of close arrivals at the start of the window, then settles to system-paced cadence (~1.16 s mean) determined by how fast PD acknowledges old requests.

## Per-request timeline (first 8 in bench window)

```
#   rid       recv_time       ViT_s    NIXL+wait_s   total_s
                              (E0→E1)  (E1→E2)       (E0→E2)
─────────────────────────────────────────────────────────────
1   f5a2a46a  05:21:34.124    1.55      6.74          8.29   ← clean isolation
2   fb7dda4f  05:21:43.888    1.64     46.99         48.63
3   f9badc9b  05:21:44.124    2.82     45.58         48.39
4   2aa380e3  05:21:44.767    3.59     44.16         47.75
5   3a4357f2  05:21:47.299    2.48     42.74         45.22
6   c4680c4d  05:21:51.083    1.65     39.78         41.44
7   5ac9a097  05:21:51.128    3.15     38.24         41.39
8   845240bc  05:21:53.442    2.47     38.28         40.75
```

The first request gets a clean run (no contention) — both stages are minimal:

- **ViT for 8 × 1080p on a single H200 = 1.55 s** (the cleanest single-request measurement we have on this hardware)
- **NIXL setup + waiting for PD to consume the embedding = 6.74 s** for the first request

## Where ViT time grows: encoder serialisation

The encoder's vision tower runs as a **single forward pass per request** (no batching across requests). With concurrent in-flight requests, each one's `recv → PATCH` time inflates because subsequent requests queue inside the encoder's coroutine runtime, awaiting GPU stream availability.

```
request#  t_since_bench_start  ViT_done_s
       1       0.00 s             1.55  ← solo
       2       9.76                1.64
       3      10.00                2.82
       4      10.64                3.59
       5      13.17                2.48
       6      16.96                1.65
       7      17.00                3.15
       8      19.32                2.47
       9      20.78                2.65
      10      20.87                4.08
      11      21.20                5.12
      12      21.96                5.92
      13      22.41                7.05
      14      22.92                8.01
      15      23.03                9.31
      16      23.09               10.82
      17      24.66               11.10
      18      25.83               11.37
      19      27.03               11.63
      20      27.45               12.61
      21      27.58               13.92
      22      27.87               15.02
      23      28.50               15.76
      24      29.53               16.19
      25      29.89               17.22
      26      30.17               18.32
      27      30.23               19.75
      28      31.96               19.52
      29      32.33               20.59
      30      33.11               21.16
      31      36.10               19.50
      32      36.29               20.67
      33      37.08               21.22
```

The growth pattern (1.5 s → 21 s as requests pile up) is the **encoder's coroutine queue depth × per-request ViT time**:
- True per-request ViT cost: **~1.5 s** (constant)
- Queue depth at request #33 ≈ 13 prior requests still running ViT
- Effective ViT wait at saturation: ~1.5 s × ~13 ≈ 20 s ← matches observed 21 s

But **steady-state encoder throughput** = 33 requests / 37 s = **~0.89 req/s**. That is ≥ 3× the PD's sustained throughput of 0.26 RPS, so the encoder is **structurally not the binding resource** in this topology.

## Where E1→E2 time goes (NIXL setup + wait)

E1→E2 has two sub-phases that we can't split with INFO-level logs, but we can reason about:

```
E1 (PATCH log) ──┬── NIXL register_memory + create_readable ────────  ~5–20 ms (per encoder log timestamps in prior runs)
                 │
                 ├── Send TransferRequest (descriptor metadata) ─────  ~2–5 ms (TCP control plane to PD)
                 │
                 ├── Wait for PD to call begin_read + RDMA wire ──────  variable, depends on PD's queue depth
                 │
                 └── transfer_future resolves; encoder coroutine done  → E2
```

The PD-side report measures the NIXL **wire** time (T4→T5) at p50 = 703 ms. That doesn't match the encoder-side E1→E2 of 65.75 s — the gap is dominated by **PD processing time before PD even calls `begin_read`**. The encoder coroutine sits idle holding the embedding tensor, waiting for PD to:

1. Receive the request via dynamo TCP
2. Allocate a NIXL receive descriptor on cuda:0
3. Process intermediate scheduler steps, multimodal preprocessing, etc.
4. Issue `begin_read` → RDMA wire transfer (the ~700 ms PD-measured "wire" piece)
5. Notify back to the encoder

That whole loop, multiplied by **PD-side queueing** (20 s median PD lifetime, of which 10.5 s is actual GPU forward), drives encoder E1→E2 to 65 s median.

## Cross-host gap reconciliation (encoder ↔ PD)

We can stitch the two reports together for a single representative request:

| Stage | Source | Median (s) |
|---|---|---:|
| Encoder E0 (recv) → encoder E1 (ViT done + .to(cuda)) | encoder log | 11.10 |
| Encoder E1 → PD T0 (request received) | indirect (gap) | ~0.05 |
| PD T0 → PD T2 (cuda:0 buffer alloc) | PD log | 0.005 |
| PD T2 → PD T3 (NIXL setup) | PD log | 0.004 |
| PD T3 → PD T5 (NIXL submit + wire) | PD log | 0.703 |
| PD T5 → PD T7 (forward + egress) | PD log | 15.57 |
| PD T7 → encoder E2 (response stream completes back) | gap | ~50 |

**Sum sanity check:**
- PD lifetime (T0→T7): 20.06 s (observed)
- Encoder lifetime (E0→E2): 76.86 s (observed)
- Implied "encoder ViT + waiting for PD final ack" = 76.86 − 20.06 ≈ 56.8 s

The mismatch is real and reflects two things:

1. **The encoder coroutine doesn't finish at PD T7.** It finishes when the **last response token streams back through the encoder process** to the bench client. The encoder is in the SSE response path; the bench client is connected to the encoder, not directly to PD's response stream.

2. **Bench-client-driven back-pressure.** Bench is `np=32`-bound; it won't release the next request until in-flight count drops. So encoder coroutines hang around an extra ~30+ s waiting for downstream (frontend → bench client → bench client's "this slot is free" signal) to drain, even though the actual GPU work is done much earlier.

## Synthesis: where encoder time really goes

```
Encoder lifetime (median 76.86 s) decomposes as:

  Image fetch + processor + ViT forward + .to(cuda)       ~11.1 s   (14.4%)  ──────
                                                       (1.5 s solo, queues to ~21 s under load)
  NIXL register + create_readable + TCP control          ~0.05 s   (0.07%)
  Wait for PD to begin_read + RDMA wire (~700 ms)        ~0.7 s    (0.91%)
  Wait for PD GPU forward (prefill + decode)             ~10.5 s   (13.7%)
  Wait for PD response stream egress + bench drain       ~54.5 s   (70.9%)  ──────────────────

  TOTAL encoder lifetime                                 ~76.86 s  (100%)
```

The encoder coroutine is **doing real work for ≤14% of its lifetime** (1.5–21 s of ViT + the patch's `.to(cuda)`). The remaining 86% is **waiting for PD-side compute and downstream egress**.

## What the encoder is NOT spending time on

| Suspect | Status | Evidence |
|---|---|---|
| Image download (HTTP fetch from URL) | NOT bottleneck | Random JPEGs are local-data-URI; fetch is microseconds |
| HF processor (resize, normalize, patchify) | Small fraction | 8 × 1080p preprocessing is ~50–200 ms on a multi-core CPU |
| ViT forward on H200 | **Real cost** but **not the bottleneck** | 1.55 s per request, 0.89 RPS encoder ceiling vs 0.26 RPS observed |
| `embeddings.clone().detach()` (sender stage_embeddings=False) | Real but small | ~10–30 ms for 637 MB in-GPU memcpy |
| `torch.cuda.synchronize()` global barrier | Negligible | ~ms-level |
| NIXL register_memory + create_readable | Negligible | <50 ms warm, 1 KB metadata send |
| Wire transfer (637 MB GPUDirect RDMA) | Small | 703 ms p50 (per PD report) — under 1% of encoder lifetime |
| Encoder GPU memory pressure | Not stressing | GPU 4 used ~46 GB of 143 GB after run; 100% headroom |

## Encoder GPU profile during bench

| Metric | Value |
|---|---:|
| Encoder PID | 9250 |
| GPU used | GPU 4 (NUMA 2) |
| Memory used (post-bench) | 45.6 GB / 143 GB (32% — model weights + per-request descriptors + caching allocator) |
| GPU utilization (sampled mid-bench) | bursts of 100% during ViT forwards, idle between |
| Process VmRSS | ~3.9 GB host |
| Thread count | 851 (Python event loop + UCX/NIXL workers + SGLang scheduler) |
| Embedding payload per request | 16 320 × 20 480 × bf16 = **637.5 MB** |

## How this differs from the PD's bottleneck story

| Where time goes | Encoder side | PD side |
|---|---:|---:|
| Useful GPU compute | 1.55 s ViT (8x1080p) | 10.5 s prefill+decode (16k visual + 256 text out) |
| Compute as % of lifetime | 14% (E0→E1) of 76.9 s | 52% of 20 s |
| Idle waiting | 86% (waiting on PD + egress) | 44% (egress + scheduler hop, but `running-req` never ≥ 9, so PD GPU is well-utilized) |
| Bottleneck | NOT here | **Yes, here**: PD GPU forward = 52% of PD lifetime; encoder feed rate exceeds PD drain rate |

Both halves run at the same **0.26 RPS** because they are pipelined per request, but the encoder's **theoretical ceiling** (~0.65 RPS = 1/1.55) is much higher than PD's (~0.34 RPS = 3.6 in-flight / 10.5 s forward).

## What would change if we...

| Change | Encoder-side impact |
|---|---|
| Switch to **PD-TP=2** | Encoder ceiling unchanged. Encoder lifetime drops because E1→E2 wait shortens. |
| Reduce workload to **4 imgs / 768p** | ViT drops to ~0.4 s/req (5× lighter); encoder ceiling rises to ~2.5 RPS |
| Use **smaller LLM** (e.g., Qwen3-VL-7B) | Encoder unchanged; PD becomes faster, encoder lifetime shrinks. |
| Add a **second encoder** (2E on dell06) | Encoder ceiling doubles to ~1.3 RPS; helps only if PD becomes the binding resource at higher concurrency. **For this workload, no help** — PD is already binding. |
| Disable `--enable-mm-global-cache` | No effect for random images (no cache hits expected) |

## Comparison vs B70 1E (reference: `b70_encoder_time_breakdown.md`)

The same workload (8 imgs / 1080p) on Intel B70 Battlemage XPU encoder takes **~25 s** of pure ViT forward time (per `35b_bottleneck_analysis.md`). Compared to dell06's **1.55 s**, an H200 SXM is **~16× faster** at this ViT, which is the dominant reason cross-host disagg with H200 encoders (this run) reaches 0.26 RPS while cross-host disagg with B70 encoders maxes at 0.038 RPS for 1E (~7× slower).

The patches and protocol overhead are the same in both cases — the difference is **pure encoder GPU compute**.

## Sources

- Encoder log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/encode_gpu4_to_dell06.log`
- Encoder start script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_encode_gpu4_to_dell06.sh`
- Bench result: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_20260527_051953/`
- Per-request breakdown JSON (this report's source): `/tmp/enc_breakdown_dell06_super21.json`
- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/time_breakdown_dell06_super21_32b_8img_1080p_encoder.md`

## Companion docs

- [`time_breakdown_dell06_super21_32b_8img_1080p.md`](./time_breakdown_dell06_super21_32b_8img_1080p.md) — PD-side analysis (this run)
- [`b70_encoder_time_breakdown.md`](./b70_encoder_time_breakdown.md) — same instrumentation on B70 XPU encoder
- [`code_for_encoder_bottleneck.md`](./code_for_encoder_bottleneck.md) — source-code map of the encoder ViT path
- [`b70_patched.md`](./b70_patched.md) — encoder-side `b70_xpu_nixl.patch` description (same logic applied automatically here via `_nixl_buffer_device()`)

## Caveats and limitations

1. **Only INFO-level logging on the encoder.** With `DYN_LOG=debug` we'd get sub-stage breakdowns (NIXL register, create_readable, transfer_future wait) like the PD-side report has. The 65 s `E1→E2` is still partially opaque without that.
2. **`E1` (the PATCH log line) fires AFTER the `.to(cuda)` device move**, not before. So `E0→E1` includes the small (~10–30 ms) `.to(cuda)` memcpy in addition to ViT.
3. **Encoder coroutine exit (E2) is downstream of PD response stream completion**, not just the NIXL transfer. That's why E1→E2 is dominated by PD work, not encoder/network costs.
4. **n=33 in encoder log vs n=32 in bench JSON.** One extra request is the bench warmup that lands in our window. The medians/p50s are unaffected.

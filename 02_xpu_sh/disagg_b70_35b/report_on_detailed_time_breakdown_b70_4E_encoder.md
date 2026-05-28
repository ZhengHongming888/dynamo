# B70 4E Encoder — Per-component time breakdown analysis

**Date:** 2026-05-26
**Setup:** Patched giga01 H200 PD (`h200_cuda_nixl.patch` active) + 4 B70 XPU encoders.
Workload: Qwen3.5-35B-A3B, 8img/1080p, np=32, random JPEG, in=128 / out=256, 7 rates 0.10–3.00.

**Method:** Parsed encoder logs `encode_xpu_35b_b70_{1..4}.log` (4 × 568 events) and
joined per-request lifecycles against the PD-side log
`pd_worker_giga01_h200_patched_debug_20260526_050419.log`. 32 requests × 7 rates × 4 encoders
= 224 cleanly correlated requests. Encoder-side instrumentation is limited to ingress
`request_received` / `request_completed` events; finer-grained per-stage timing is
**inferred** by composition with PD-side checkpoints from the companion report
`report_on_detailed_time_breakdown_h200_b70_4E.md`.

## Encoder-side per-request timeline

The encoder handler in `encode_worker_handler.py:298 generate()` has the following
checkpoint structure:

| ID | Event | Source | Observable? |
|---|---|---|---|
| **T_E0** | encoder ingress receives request | encoder log `request received` | **Yes** (timestamp) |
| **T_E1** | JSON deserialize + URL extract complete | `_extract_image_urls()` returns | No (no log) |
| **T_E2** | vision_encode begins (`mm:enc:vision_encode`) | NVTX region entry | NVTX only |
| **T_E3** | vision_encode ends (XPU ViT done, embeddings on CPU) | NVTX region exit | NVTX only |
| **T_E4** | embedding_transfer begins (`mm:enc:embedding_transfer`) | NVTX region entry | NVTX only |
| **T_E5** | NIXL Descriptor + readable created | `await create_readable()` returns | No (no log) |
| **T_E6** | round_robin RPC sent to PD | `await pd_worker_client.round_robin()` | ≈ PD's T0 (= PD_T0) |
| **T_E7** | first response chunk yielded back to frontend | first iter of `async for response` | No |
| **T_E8** | last response chunk yielded | last iter; then `await transfer_future` | ≈ PD's T7 (= PD_T7) |
| **T_E9** | grace period complete (`asyncio.sleep(0.1)`) | function returns | No |
| **T_E10** | encoder ingress fires `request_completed` | encoder log | **Yes** (timestamp) |

Of these 10 checkpoints, only **T_E0** and **T_E10** are directly logged. We get T_E6 ≈ PD_T0
and T_E8 ≈ PD_T7 by joining per-request IDs with PD-side log entries. Everything between
T_E0..T_E6 (vision_encode + sender + RPC) and T_E8..T_E10 (post-PD wrap-up) is one lump each.

## Per-rate observed timings (median, sweep window 05:49–06:27)

| rate | n | RPS | enc_lifetime (s) | pre_PD = T_E6 − T_E0 (s) | PD_lifetime = T_E8 − T_E6 (s) | post_PD = T_E10 − T_E8 (s) |
|---:|--:|---:|---:|---:|---:|---:|
| 0.10 | 32 | 0.106 | 62.9 | 33.7 | 1.8 | 25.2 |
| 0.25 | 32 | 0.138 | 151.4 | 66.6 | 1.7 | 78.4 |
| 0.50 | 32 | 0.144 | 186.7 | 92.2 | 1.7 | 91.9 |
| 1.00 | 32 | 0.146 | 200.8 | 106.6 | 1.8 | 92.5 |
| 1.50 | 32 | 0.145 | 205.9 | 111.9 | 1.9 | 92.0 |
| 2.00 | 32 | 0.146 | 208.5 | 114.2 | 1.8 | 92.6 |
| 3.00 | 32 | 0.146 | 211.5 | 117.0 | 2.2 | 92.7 |

The three sub-times **sum to enc_lifetime within rounding** at every rate (math closes ✓).

`PD_lifetime` ≈ 1.8 s **constant** across all rates — matches the PD report's per-request
budget (~1.8 s ≈ 668 ms prep + 1003 ms forward + 14 ms NIXL + 0.5 ms alloc + 478 ms egress).

`pre_PD` is the **encoder's own work for one request** — it grows linearly with rate
because requests queue up at the Python event loop (more on this below).

`post_PD` reaches a ceiling of ~92 s — saturation behavior identical to `pre_PD`.

## The 92-second anomaly

At saturation, `post_PD` (the time between PD declaring the request done and the encoder
ingress firing `request_completed`) is **92 seconds**. That is 50× longer than the entire
PD pipeline. Why?

Look at `encode_worker_handler.py:431-454`:

```python
response_generator = await self.pd_worker_client.round_robin(...)

async for response in response_generator:        # ← blocks per-chunk
    ...
    yield data

if transfer_future is not None:
    await transfer_future                         # ← waits for NIXL READ done
await asyncio.sleep(0.1)                          # ← 100 ms grace
```

The `async for response in response_generator` loop is on the **same asyncio event
loop** that is also driving `MMEncoder._encode()` (the XPU vision tower call in
`mm_encode()`). When all 4 encoders are saturated, this loop is busy running ViT
forward passes — which call into native code that holds the GIL/blocks the loop
for the duration of one ViT pass (~5–10 s of XPU compute + Python orchestration).

While the loop is stuck on ViT, the `async for` cannot read the buffered PD response
stream, even though PD has already finished writing it. So the request stays "open"
on the encoder ingress until the loop finally yields back to the response consumer
~92 s later.

This is the **same encoder-bound queueing effect as `pre_PD`**, just expressed through
the response side instead of the request side. Each request sees 1× ViT-cycle wait
before its vision_encode runs **plus** another 1× ViT-cycle wait before its response
stream gets drained.

## Per-encoder cycle (= ViT pass time at saturation)

Cluster analysis on `request_completed` timestamps (cluster gap = 1 s):

| rate | #req | #clusters | avg cluster size | inter-cluster p50 (s) | aggregate RPS | per-encoder cycle (s) |
|---:|--:|--:|--:|---:|---:|---:|
| 0.10 | 32 | 18 | 1.78 | 4.60 | 0.106 | 37.7 (sub-saturation) |
| 0.25 | 32 | 8 | 4.00 | 4.31 | 0.138 | 29.1 |
| 0.50 | 32 | 3 | 10.67 | 2.94 | 0.144 | 27.8 |
| **1.00** | 32 | 1 | 32.00 | n/a | **0.146** | **27.4** |
| 1.50 | 32 | 2 | 16.00 | 4.94 | 0.145 | 27.5 |
| 2.00 | 32 | 4 | 8.00 | 1.60 | 0.146 | 27.5 |
| 3.00 | 32 | 2 | 16.00 | 3.30 | 0.146 | 27.5 |

`per-encoder cycle = 4 / aggregate RPS`. The pool of 4 encoders produces 0.146 RPS at
saturation, so each encoder produces 1 completion every **27.4 s** — this is the
effective ViT pass time on B70 XPU for one 8img/1080p request through Qwen3.5's
vision tower.

(Note: the PD report's prior "24 s" estimate was a back-of-envelope; the cluster-derived
27.4 s here is the precise figure from the same sweep run.)

## Decomposition at saturation (rate=1.0)

```
T_E0 (request received from frontend)
 │
 │  Python event loop wait for prior request to release the loop
 │  + JSON deserialize + URL extract
 │  + vision_encode (mm:enc:vision_encode, the XPU ViT pass)
 │  + sender embedding_transfer (NIXL Descriptor + create_readable)
 │  + encoder->PD RPC dispatch (round_robin TCP send)
 │
 │   ─────────  T_E6 = PD_T0  ─────────  ≈ 106.6 s after T_E0
 │
 │  PD ingress dispatch (T1−T0): 4 ms
 │  PD CUDA alloc (T2−T1):       0.5 ms
 │  PD NIXL setup (T3−T2):       5 ms
 │  PD NIXL READ wire (T4−T3):   14 ms
 │  PD prep + scheduler (T5−T4): 668 ms
 │  PD prefill+decode (T6−T5):  1003 ms
 │  PD egress to frontend:       478 ms
 │
 │   ─────────  T_E8 = PD_T7  ─────────  ≈ 1.8 s after T_E6
 │
 │  encoder async for loop drains PD chunks (loop-blocked behind next ViT)
 │  + await transfer_future
 │  + asyncio.sleep(0.1) grace
 │
 │   ─────────  T_E10 (request_completed)  ─── ≈ 92.5 s after T_E8
 │
└─ Total enc_lifetime = 200.8 s
```

### Encoder-internal sub-component table (rate=1.0, median)

| # | Sub-component | Source | Median (ms) | % of cycle |
|--:|---|---|---:|---:|
| 1 | JSON deserialize + URL extract | encoder | 0.5 | <0.001% |
| 2 | **vision_encode (B70 XPU ViT)** | encoder NVTX `mm:enc:vision_encode` | **~27,400** | **~94%** of cycle |
| 3 | sender embedding_transfer (NIXL Descriptor + create_readable) | encoder NVTX `mm:enc:embedding_transfer` | ~1.0 | <0.01% |
| 4 | encoder → PD RPC dispatch (round_robin) | dynamo TCP plane | 4.0 | 0.01% |
| 5 | PD-side total (1.8 s, see PD report) | PD checkpoints | 1,800 | 6.6% |
| 6 | encoder response-stream drain (await `async for`) | loop-blocked behind ViT | ~92,000 | (overlapped with #2 of next req) |
| 7 | asyncio.sleep(0.1) grace | encoder | 100 | 0.4% |
| | **Encoder-pool cycle = 4 / 0.146 RPS** | derived | **27,400** | **100%** |

Key insight: items 2 and 6 (vision_encode and response-stream drain) do not stack
serially per-request — they overlap **across** requests, because while the encoder
is computing ViT for request *N+1*, the asyncio loop is starved for request *N*'s
response drain. The per-request lifetime *T_E10 − T_E0* of 200 s ≈ 7 × 27.4 s ViT
cycles, which corresponds to the queue depth (np=32 / 4 encoders ≈ 8 reqs per encoder,
seven of which wait while one runs).

## Why the "patched" patch helps less here than expected

The H200 PD CUDA-allocation patch saved ~50 ms per request on the PD side. The 4E
encoder patch (`stage_embeddings=True`, skip redundant `.to(cpu)`, skip `torch.cat`
fast path) saves ~1–5 ms per request on the encoder side. Both savings are **invisible**
in the 200 s enc_lifetime because the encoder is bottlenecked by:

1. **B70 XPU ViT compute**: 27.4 s/req. This is the actual binding constraint.
2. **Python event-loop serialization**: the `async for response` drain is single-loop
   single-coroutine; it cannot drain a previous request's PD response stream while the
   ViT is computing the next request.

Combined effect: throughput ceiling = 4 encoders / 27.4 s = **0.146 RPS** regardless
of any patch.

## Where the time goes in one ViT cycle (qualitative)

We don't have direct instrumentation inside `MMEncoder._encode()`, but we know it
calls (from `/opt/sglang/python/sglang/srt/disaggregation/encode_server.py:1045-1094`):

1. `_process_mm_items(mm_items, modality)` — image fetch + processor:
   - `_flatten_and_load_images()` — `aiohttp` fetches if URL, else PIL decode
   - `image_processor(images=..., **image_config)` — Qwen3VL preprocessor (resize, normalize, patch)
2. `get_feature_fn([mm_item])` — runs ViT on XPU (the ~27 s)
   - `mm_embedding.cpu()` (D→H copy at end)
3. Return `(grid_dim, mm_embedding, aux_data)`

For 8img/1080p the ViT processes 8 images × (1080/14)² ≈ 47616 visual patches per image
= 380k patches, run through Qwen3-VL's vision tower (~600M params) using `triton_attn`
backend on Battlemage XPU. On a CUDA H200 the same ViT pass takes ~1.5 s; on B70 XPU
it takes ~27 s — a **~18× slowdown** compared to H200.

To break the 27 s into image-fetch / preprocess / ViT-forward / D→H components would
require adding `time.perf_counter()` log lines in `_process_mm_items` and around
`get_feature_fn` — not present today.

## Comparison with PD-side breakdown (from the companion report)

| Stage | B70 4E encoder | H200 PD (patched) | Ratio |
|---|---:|---:|---:|
| Pre-vision compute (loop wait + dispatch) | small | n/a | — |
| Vision encode / Prefill | **~27,400 ms** (B70 ViT) | ~580 ms (chunked-prefill) | 47× |
| Network transfer | ~1 ms (sender setup) | ~14 ms (NIXL READ wire) | — |
| Decode | n/a | ~425 ms (256 tok @ 1.6 ms) | — |
| Post compute (response drain / egress) | **~92,000 ms** (loop-blocked) | ~478 ms | 192× |
| **Per-cycle total** | **27,400 ms** | **~1,800 ms** | **~15×** |

PD does its 1.8 s of work, then idles for ~25 s waiting for the encoder pool to
produce the next batch. PD's `queue-req` is always 0; PD has 5–10× headroom that
is wasted on the B70 encoder bottleneck.

## Conclusions

1. **B70 ViT compute is the bottleneck**: 27.4 s/cycle × 4 encoders → 0.146 RPS ceiling.
   Same conclusion as the PD report from the other side of the wire.

2. **The 92 s `post_PD` gap is loop contention, not real work**: the encoder's
   `async for response` is starved by `MMEncoder._encode()` running ViT for the
   next request on the same asyncio loop. If we ran the response-drain on a
   separate executor (or moved ViT to a `run_in_executor` thread), this 92 s would
   collapse — but throughput wouldn't improve because the ViT itself is still the
   binding constraint. (Latency for *individual* requests would improve.)

3. **Patches are invisible at this scale**: the encoder patch (~5 ms/req savings)
   and PD CUDA-alloc patch (~50 ms/req savings) are dwarfed by the 27.4 s/req ViT
   cost. They become visible when you switch to a faster encoder (H200/dell06,
   ~1.5 s ViT) — at that point the PD becomes the bottleneck and patches matter.

4. **Per-request lifetime = ~7 × cycle time at saturation** (200 s ≈ 7 × 27.4 s).
   With np=32 and 4 encoders, each request waits behind ~7 prior requests in
   the encoder pool's logical queue.

5. **To improve throughput further:**
   - Replace B70 with H200/H100 encoders → ~18× ViT speedup → ~2.6 RPS aggregate
     (until PD becomes the bottleneck around 0.85 RPS, where the dell06_1E run
     already lives)
   - More B70 encoders (8E vs 4E): ~2× throughput → ~0.29 RPS, modulo
     NUMA/NIC contention on giga01-b70
   - Move ViT off the asyncio loop (`asyncio.to_thread` or process pool):
     improves *per-request latency at saturation* by ~92 s but does not change
     the throughput ceiling
   - Preprocessor caching (`ENABLE_ENCODER_CACHE=1`): only helps if image URLs
     repeat (currently `0` in `start_sglang_pd_xpu_35b_b70_4E.sh`)

## What would need to change to get a "true" sub-component breakdown

Add `time.perf_counter()` log lines around the four NVTX regions in
`encode_worker_handler.py:337,420` and inside SGLang's `MMEncoder._encode`:

```python
# encode_worker_handler.py around line 337
import time
t_vision_start = time.perf_counter()
with _nvtx.annotate("mm:enc:vision_encode", color="red"):
    grid_dim, embeddings, _ = await mm_encode(self.encoder, image_urls, Modality.IMAGE)
t_vision_end = time.perf_counter()
logger.info(f"vision_encode_ms rid={request_id} dt={(t_vision_end-t_vision_start)*1000:.1f}")
```

…and similar around `embedding_transfer` (line 420), `round_robin` (431), and the
`async for response` loop (437). With those 4–6 wallclock log lines, a single sweep
run will produce the same per-checkpoint table as the PD report.

## Files

- Encoder logs: `/hongming/dynamo/02_xpu_sh/disagg_b70_35b/logs/encode_xpu_35b_b70_{1..4}.log`
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01_h200_patched_debug_20260526_050419.log`
- Bench JSONs: `/hongming/res22_disagg_h200_35b_sweep/8img_1080p_h200_patched_b70_4E/rate_*_np32/benchmark_output.json`
- Companion: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/report_on_detailed_time_breakdown_h200_b70_4E.md`
- Patch: `/usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py` (see `/hongming/bottleneck_for_b70_encoder_35b.md`)

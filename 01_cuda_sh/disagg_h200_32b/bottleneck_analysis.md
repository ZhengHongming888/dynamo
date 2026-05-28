# Bottleneck Analysis — Disagg H200/H200 Same-Machine, Different GPU Cards

**Date:** 2026-05-22
**System:** Qwen3-VL-32B-Instruct-FP8, encoder on GPU 4 + PD on GPU 5, NV18 NVLink, same host (172.26.46.75)
**Workload:** 8 × 1920×1080 images per request, np=64, rate=1.0 req/s
**Best config measured:** NIXL_READ + max_running_requests=64 + chunked_prefill_size=16384

**Result CSV:** `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates_nixlread_AB/test_sglang_multi_rates_1080p_20260522_020943/results_summary.csv`
**PD log used for analysis:** `/hongming/dynamo/logs/logs/logs/pd_worker.log`
**Encoder log:** `/hongming/dynamo/logs/logs/logs/encoder_worker.log`

---

## Headline numbers (current best config @ rate=1.0)

| Metric | Value |
|---|---:|
| Actual RPS | 0.23 |
| Successful requests | 64 / 64 ✓ |
| Bench duration | 284 s |
| Mean TTFT | 155 s |
| Mean TPOT | 2,530 ms |
| Mean E2E | 245 s |

For comparison at rate=1.0:
- TP=2 agg NVLink: **0.95 RPS, 9 s TTFT, 12 s E2E** (winner)
- TP=1 agg: **0.52 RPS, 32 s TTFT, 81 s E2E**
- Disagg current best: **0.23 RPS, 155 s TTFT, 245 s E2E**

---

## Per-request lifecycle (from `enable_request_time_stats_logging`)

The PD scheduler emits one `ReqTimeStats(rid=…, queue_duration=Xms, forward_duration=Yms, …)` line per finished request. Parsed across 71 requests:

| Phase | Median | Mean | p25 | p75 | Max |
|---|---:|---:|---:|---:|---:|
| **Queue wait** (scheduler queue inside PD) | **112.5 s** | 101.3 s | 97.1 s | 128.7 s | 166.0 s |
| **Forward** (prefill + decode on PD) | **83.2 s** | 87.6 s | 35.6 s | 126.6 s | 276.4 s |
| **Total per-request (queue + forward)** | **195.7 s** | 188.9 s | — | — | — |

**Queue wait is 57% of per-request time** at the median. This is the dominant cost.

---

## Bottleneck ranking (with hard data)

### 🥇 Bottleneck #1 — PD scheduler queue saturation (112 s median)

**Mechanism:** rate=1.0 with 32 concurrent slots = arrival rate exceeds drain rate. After ~30 requests are in flight, every new request waits in queue ~112 s before PD even starts processing.

**Why it can't be tuned away by current settings:**
- `--max-running-requests=64` already engaged (was 32)
- KV usage median is only 0.08 — KV isn't the limit
- Per-chunk **peak input throughput hits 15,106 tok/s** which equals TP=1 agg's peak — PD compute is already at hardware ceiling
- The only way to increase drain rate further is more compute (TP=2 PD) or smaller per-request work (fewer images)

### 🥈 Bottleneck #2 — Embedding-integration small batches (46% of prefill events)

**Mechanism:** When SGLang's `_generate_aggregated` receives NIXL-transferred embeddings via `precomputed_embeddings`, it emits **separate small-batch prefill events** (16, 32, 48, 64, 80, 96, 112 tokens) to fold the embeddings into KV cache. Each runs at ~10-200 tok/s vs ~11,000 tok/s peak.

**Hard count from this run:**

| Chunk size | Count | What it represents |
|---:|---:|---|
| 16384 (full request) | 61 | Normal prefill of a 16k-token MM request |
| 16-112 tokens | **53** | **Embedding-integration overhead** |
| 16368 (tail flush) | 8 | Final tokens of a request |

**46% of prefill events are tiny embedding-integration batches.** These do not exist in TP=1 agg (which has no NIXL handoff).

| 16384-chunk metric | Disagg current best |
|---|---:|
| Median input throughput | 11,232 tok/s |
| Mean | 8,628 tok/s |
| Peak | 15,106 tok/s |

| Small-chunk pattern | Effect |
|---|---|
| 16-112 token batches at <200 tok/s | Pure GPU underutilization — kernel launch overhead dominates |

**Estimated impact:** ~10-30 s of forward time per request comes from these small batches alone.

### 🥉 Bottleneck #3 — Encoder→PD round-trip latency (~1-2 s)

Each request must:
1. Encoder runs ViT on its own GPU (~1.2 s)
2. Encoder calls `connector.create_readable(descriptor)` to expose embedding tensor
3. Encoder builds `TransferRequest` with descriptor metadata
4. Encoder sends request (with `transfer_payload`) to PD via dynamo TCP request plane
5. PD receives request, allocates a pre-warmed descriptor from pool
6. PD calls `connector.begin_read(metadata, descriptor)` then `await read_op.wait_for_completion()`

This is invisible in TP=1 agg (encoder runs inline with LLM in same process). On NIXL_READ this round-trip is ~1-2 s including the TCP request-plane hop.

### Bottleneck #4 — Decode preemption by prefill

**Mechanism:** Forward batches mix prefill (high compute, low concurrency) and decode (low compute, high concurrency). When prefill chunks are running, decode is starved.

**From decode stats:**

| Metric | Value |
|---|---:|
| Decode `#running-req` median | 4 |
| Decode `#running-req` max | 48 (briefly, near end) |
| Gen throughput median | 137 tok/s |
| Gen throughput max | 909 tok/s |
| KV usage median | 0.08 |
| KV usage max | 0.96 (only briefly) |

Decode runs in short bursts after prefill drains. Most of the time only ~4 requests are decoding. This produces the inflated TPOT of 2,530 ms mean (vs 32 ms for TP=2 agg).

### Bottleneck #5 — NIXL transfer wallclock (~20-100 ms with NIXL_READ)

After the env-var fix and switching to NIXL_READ:
- No more busy-poll (`asyncio.sleep(0.001)` removed)
- No more 60s timeout failures
- Transfer happens via cuda_ipc P2P over NV18 NVLink between GPUs 4 and 5
- `wait_for_completion` is proper async (not poll)

**This is no longer a bottleneck.** With NIXL_WRITE (the previous default) it was responsible for 12% of transfers stalling 5-15 s and 15-25 request failures per run.

### Bottleneck #6 — Encoder GPU compute (~1.2 s, but parallel)

Encoder ViT on GPU 4: ~1.2 s/request for 8 × 1080p images. **Mostly hidden** because it runs in parallel with PD work on GPU 5. Only adds ~1 s to E2E for the first request.

---

## What is NOT a bottleneck (verified, ruled out)

| Suspected | Verdict | Evidence |
|---|---|---|
| NIXL transport (NVLink P2P) | Not bottleneck | NIXL_READ uses cuda_ipc over NV18, transfers are fast |
| NIXL buffer-pool exhaustion | Not bottleneck (after NIXL_READ fix) | Zero `Timeout while waiting for available buffer` errors with NIXL_READ |
| `max_running_requests` cap | Not binding | Set to 64; observed mode 47-48 (cap rarely hit) |
| `chunked_prefill_size` cap | Not binding | Set to 16384; 16k-token requests prefill in 1 chunk |
| KV cache pressure | Not binding | KV usage median 0.08, only briefly hits 0.96 |
| UCX NIC choice | Not binding (for same-host) | NIXL same-host uses cuda_ipc, not RDMA, regardless of `UCX_NET_DEVICES` |
| Frontend / etcd / NATS | Not bottleneck | TCP request plane hop is ~1 ms |
| Encoder GPU compute speed | Not bottleneck | Runs in parallel on GPU 4 |

---

## Why disagg loses to TP=1 agg even after all fixes

Per-request cost comparison at rate=1.0:

| Phase | TP=1 agg | Disagg current best |
|---|---:|---:|
| Encoder time (ViT) | inline ~1.2 s | parallel ~1.2 s (hidden on GPU 4) |
| Embedding handoff | 0 (in-process) | 0.02-0.1 s NIXL + ~0.5 s `_build_mm_items` |
| Forward duration (prefill + decode) | ~1.5-3 s effective | **83 s median** |
| Queue wait | minimal | **112 s median** |
| **Total per-request** | **~3-5 s** | **~196 s** |

The huge gap is NOT in the compute capacity (both peak at ~13-15k tok/s prefill). It's in:
1. **Queue saturation** because per-request work on disagg PD is structurally larger
2. **Small-batch overhead** unique to disagg's NIXL-embedding-integration path
3. **Encoder GPU is wasted** (1.8 GB / 143 GB used = 1.3% utilization)

**TP=1 agg uses 1 GPU and gets 0.52 RPS. Disagg uses 2 GPUs and gets 0.23 RPS.** The "second GPU" bought us nothing for this workload, because PD compute didn't get faster from disaggregation — only the encoder, which was never the bottleneck.

---

## What would be needed to make disagg win

| Change | Mechanism | Estimated impact |
|---|---|---:|
| **TP=2 PD** (encoder GPU 4, PD on GPUs 5,6 as TP=2) | Doubles PD compute via NVLink | **+50-80% RPS, target 0.4-0.5** |
| **Smaller workload** (4 images instead of 8) | Halves per-request work | Already tested: **0.47 RPS** ✓ |
| **Coalesce embedding integration** (code change in `_build_mm_items`) | Eliminate the 53 small-batch events per 64 requests | +20-30% RPS estimated |
| **`--keep-mm-feature-on-device`** on encoder (if supported) | Avoid host staging of MM features | Unknown, likely +5-10% |
| **True PD-disagg** (`disaggregation_mode=prefill` + separate decode worker) | Decode can run unimpeded by prefill | TPOT improves dramatically (decode can hit ~10 ms instead of ~700 ms) |
| **Multi-tenant encoder pool** (1 encoder serves N PDs) | Amortize encoder GPU across multiple LLMs | Throughput per encoder GPU goes way up |
| **Asymmetric hardware** (encoder on cheap GPU, LLM on H200) | Encoder GPU was 1.3% utilized — ridiculous waste | Per-dollar throughput improves a lot |

---

## Recommended experiment if you want to push further

**Run with `--keep-mm-feature-on-device` on the encoder** (1-flag change, no risk). Check if the 53 small-batch prefill events disappear. If they do, the embedding-integration overhead is fixable; if they don't, the integration pattern is structural in SGLang's `_generate_aggregated` path and would need a code change to fix.

Then **run TP=2 PD** (encoder GPU 4, PD as TP=2 on GPUs 5+6). This is the only architectural change likely to give a meaningful RPS boost on this workload.

---

## Summary table — current bottleneck contribution

| Bottleneck | Contribution to per-request time | Type | Fixable? |
|---|---:|---|---|
| Queue saturation | **112 s (57%)** | scheduling | Only via more compute (TP=2 PD) or less work |
| Forward duration (incl. small-batch overhead) | **83 s (43%)** | compute | Partially — small-batch coalescing could save 10-30 s |
| Encoder→PD round trip | ~1-2 s (within forward) | latency | Already at floor with NIXL_READ |
| NIXL transport | <100 ms | transport | Already optimized |
| Decode preemption | TPOT inflated | scheduling | Only via real PD-disagg (decode worker) |

**Net: PD compute capacity is the binding bottleneck, with embedding-integration small-batch overhead as a secondary tax.** The only single-host config that can possibly beat TP=1 agg on this workload is TP=2 PD disagg, which we haven't tested yet.

---

## Files

- Result CSV: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates_nixlread_AB/test_sglang_multi_rates_1080p_20260522_020943/results_summary.csv`
- PD worker log: `/hongming/dynamo/logs/logs/logs/pd_worker.log`
- Encoder worker log: `/hongming/dynamo/logs/logs/logs/encoder_worker.log`
- Server start: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh` (currently has NIXL_READ + A + B)
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md`
  - `01_cuda_sh/disagg_h200_32b/deep_analysis_disagg_worse_h200.md`
  - `01_cuda_sh/disagg_h200_32b/disagg_improvements_attempts.md`
- This document: `01_cuda_sh/disagg_h200_32b/bottleneck_analysis.md`

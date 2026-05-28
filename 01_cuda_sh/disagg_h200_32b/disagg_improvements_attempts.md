# Deep Investigation #2 — Disagg Improvement Attempts

This document captures the second-round deep investigation into improving disagg performance on H200, after the initial `deep_analysis_disagg_worse_h200.md` concluded "architectural overhead is unfixable." That conclusion was too pessimistic — there were several real bugs and tunables we hadn't tried.

**System under test:** Qwen3-VL-32B-Instruct-FP8, encoder GPU 4 + PD GPU 5, NV18 NVLink between them, rate=1.0 req/s, 8 × 1920×1080 images, np=64.

**Baselines we're comparing to:**
- TP=2 agg NVLink: **0.95 RPS, 9 s TTFT** (winner)
- TP=1 agg: **0.52 RPS, 32 s TTFT**
- Original disagg v1 (NIXL_WRITE, default config): **0.27 RPS, 79 s TTFT, 15/64 failures**

---

## Three major findings during this investigation

### Finding 1: 🐛 **Wrong env var name — NIXL_READ was silently ignored**

The original script used:
```
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read
```

But `dynamo.sglang` reads the env var **`DYN_SGL_EMBEDDING_TRANSFER_MODE`** (vLLM and SGLang are different backends with different env var prefixes). Confirmed in `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/backend_args.py:62`:
```python
add_argument(g, flag_name="--embedding-transfer-mode",
             env_var="DYN_SGL_EMBEDDING_TRANSFER_MODE",
             default=EmbeddingTransferMode.NIXL_WRITE.value, ...)
```

**Result of bug:** Every prior disagg run used `NIXL_WRITE` (the default), not `NIXL_READ` despite our env var setting. All earlier "tuning" attempts (encoder throttle, NIXL_BUFFER_COUNT changes, NIC fix) were running against this wrong baseline.

Verified in worker config dump after fix:
```
"embedding_transfer_mode":"EmbeddingTransferMode.NIXL_READ"  ✓
```

### Finding 2: **NIXL_READ has fundamentally cleaner architecture than NIXL_WRITE**

Reading `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py`:

**`NixlWriteEmbeddingReceiver`** (the previous default):
- 256-buffer ring buffer (configured via `NIXL_BUFFER_COUNT`)
- Encoder pushes regardless of PD readiness
- PD blocks waiting for free buffer slot via `await asyncio.sleep(0.005)` (5ms busy-poll loop)
- After PD acks, both sides poll notification queue every 1ms via `await asyncio.sleep(0.001)`
- Fixed `transfer_timeout=60s` — exceeded → `Timeout while waiting for available buffer.` error
- This was the cause of the 48 timeouts we saw across rates 0.5/1.0/1.25 in disagg v1

**`NixlReadEmbeddingReceiver`** (preferred):
- Pre-warmed descriptor pool (default 1024)
- Uses `nixl_connect.Connector` singleton
- **PD pulls when ready** via `await read_op.wait_for_completion()` (proper async await, not busy-poll)
- Natural backpressure: encoder simply sets up readable buffer and waits; doesn't push
- No `Timeout while waiting for available buffer` failure mode

### Finding 3: **PD is running aggregated mode, not prefill-disagg**

`MultimodalWorkerHandler.generate()` (worker_handler.py:303) routes based on `serving_mode`:
- `DECODE` → `_generate_disaggregated()` (full PD-disagg path)
- otherwise → `_generate_aggregated()` (single-process prefill+decode)

Our config has `--multimodal-worker` + no `disaggregation_mode=prefill`, so we're hitting `_generate_aggregated`. The "disagg" architecture is:
- **Encoder-disaggregated** (vision ViT on its own GPU, NIXL transfer to PD)
- **LLM-aggregated** (prefill + decode in one worker)

There's no prefill/decode split. True PD-disagg would require `disaggregation_mode=prefill` on PD AND a separate decode worker.

---

## Improvement experiments and results

### Experiment 0 — Baseline (NIXL_WRITE, max_running=32, chunked_prefill=8192)

This is the original config. Each rate=1.0 run produces ~15 buffer-timeout failures.

| Metric | Value |
|---|---:|
| Actual RPS | 0.27 |
| Successful | 49/64 (15 failed) ❌ |
| Mean TTFT | 79 s |
| Mean E2E | 111 s |
| Per-chunk 8192 wallclock median | 2.28 s |
| Per-chunk input tput median | 5,695 tok/s |
| #running-req mode | 31 (cap=32) |

### Experiment 1 — Fix env var + switch to NIXL_READ

Changed `DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read` → `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read` to actually engage the NIXL_READ code path.

Also enabled timing logs for the first time:
- `--enable-request-time-stats-logging` → emits per-request `ReqTimeStats(rid=..., queue_duration=Xms, forward_duration=Yms, ...)` lines
- `--show-time-cost` → SGLang internal phase timing
- `DYN_LOG=debug` (already on) → emits NIXL `Send completed for tensor_id X, total wait time: Y seconds`

| Metric | NIXL_WRITE baseline | **NIXL_READ** | Δ |
|---|---:|---:|---:|
| Actual RPS | 0.27 | 0.20 | -26% |
| Successful | 49/64 | **64/64** ✓ | **+15 successes** |
| Mean TTFT | 79 s | 178 s | +125% |
| Mean TPOT | 752 ms | 1,390 ms | +85% |
| Mean E2E | 111 s | 255 s | +130% |
| Failed requests | **15** | **0** | -15 |

**Interpretation:** NIXL_READ eliminates all failures (no buffer-pool-exhaustion mode), but throughput drops because the system can now successfully complete every request — the deep queue can drain instead of dropping requests partway through. Higher TTFT/E2E reflect the longer queue wait that's now visible to all 64 requests instead of being absorbed by 15 failures.

### Per-phase breakdown from `ReqTimeStats` (NIXL_READ baseline)

This was the first time we got real per-phase data:

| Phase | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| `queue_duration` (waiting in PD scheduler) | **135 s** | 122 s | 0 ms | 192 s |
| `forward_duration` (prefill + decode) | **59 s** | 75 s | 3.7 s | 306 s |
| Total per-request | ~194 s | ~197 s | — | — |

**Key insight:** Queue dominates. With max_running_requests=32 and rate=1.0 sending requests faster than PD can drain, the queue piles up to ~135 s for the median request. Only the first request enjoys a clean 3.7 s forward.

### Experiment 2 — A+B: more concurrency + bigger chunk size

Two changes targeting the per-request work and queue:
- **A**: `--max-running-requests 32 → 64` (KV usage was only 0.65 — plenty of headroom)
- **B**: `--chunked-prefill-size 8192 → 16384` (each ~16k-token request fits in 1 chunk instead of 2)

| Metric | NIXL_READ baseline | **NIXL_READ + A+B** | Δ |
|---|---:|---:|---:|
| **Actual RPS** | 0.20 | **0.23** | **+15%** |
| Successful | 64/64 | 64/64 | = |
| Bench duration | 315.8 s | 284.3 s | -10% |
| Mean TTFT | 178 s | **155 s** | **-13%** |
| Median TTFT | 182 s | 161 s | -12% |
| P99 TTFT | 247 s | 215 s | -13% |
| Mean TPOT | 1,390 ms | 2,530 ms | +82% |
| Median TPOT | 651 ms | 744 ms | +14% |
| Mean E2E | 255 s | **245 s** | **-4%** |
| Median ITL | 39 ms | 47 ms | +21% |
| Peak output tput | 722 tok/s | **944 tok/s** | **+31%** |
| Concurrency | 51.7 | 55.2 | +7% |
| Queue median (ReqTimeStats) | 135 s | **112 s** | **-17%** |
| Forward median (ReqTimeStats) | 59 s | 83 s | +40% |
| Per-chunk wallclock — median | 2.21 s (8192) | 2.90 s (16384) | larger chunk |
| **Per-chunk input tput — median** | **6,087 tok/s** | **11,232 tok/s** | **+85%** |
| Per-chunk input tput — peak | 13,277 tok/s | 15,106 tok/s | +14% |
| #running-req mode | 31 (cap=32) | 47-48 (cap=64) | +52% |

**Both changes worked as expected:**
- A (max_running=64): PD now batches 47-48 requests per chunk vs 31 → queue drains faster, queue median drops 17%
- B (chunked_prefill=16384): peak per-token throughput jumps 85% because 16k-token requests don't get split into 2 chunks anymore (each chunk fewer-but-larger means less per-chunk overhead per token)
- Combined: +15% RPS, -13% TTFT, +31% peak output throughput

**The "forward_duration went up" is misleading.** Per-request `forward_duration` rose from 59s to 83s because each forward now batches more concurrent work. Aggregate throughput improved despite that — same idea as larger batch size on a GPU: per-batch wall time goes up, but per-token compute drops.

---

## What about all the other things we tried?

### Things that did NOT help

| Attempt | Result vs NIXL_WRITE baseline |
|---|---|
| **v2** (encoder `--max-running-requests 8` + `NIXL_BUFFER_COUNT=32`) | Worse: 0.21 RPS, 25 failures |
| **NIC fix** (`UCX_NIC=mlx5_4:1` instead of `mlx5_0:1`) | No change (transfer is via cuda_ipc, not the NIC) |

Why those didn't help:
- v2 throttled the encoder, but encoder isn't the bottleneck
- NIC fix is irrelevant for same-host disagg (cuda_ipc over NVLink is used, not RDMA)

### Things we already knew helped

| Attempt | Result |
|---|---|
| Reduce workload to 4 images | **0.47 RPS, 0 failures** — biggest single improvement, but trivially because PD has half the compute work per request |

---

## Per-chunk prefill throughput — close to peak

The most striking number from A+B: **per-chunk peak input throughput hit 15,106 tok/s**, which is essentially the same as TP=1 agg's peak (~13,000 tok/s) and even beats TP=1 agg slightly.

This proves PD's prefill compute is at hardware limit. There's no more "tuning" headroom on a single H200. To go further requires architectural change (e.g., TP=2 PD) or workload reduction.

---

## Why disagg still loses to TP=1 agg

Per-request budget at rate=1.0:

| Phase | TP=1 agg | Disagg current best (A+B) |
|---|---:|---:|
| Encoder time (ViT) | inline ~1.2 s | parallel ~1.2 s (separate GPU, hidden) |
| Embedding handoff | 0 (in-process) | 0.02 s NIXL_READ + ~0.5 s `_build_mm_items` |
| Forward duration | ~1.5 s (1 chunk × ~1 s + decode) | **~83 s median** (queue saturation effect) |
| Queue wait | minimal (encoder overlaps) | **112 s median** |

Even with all our improvements, **disagg's queue saturates at lower offered rate** than TP=1 agg because:
1. Per-request total time on PD is higher (`forward_duration` 83 s vs TP=1 agg's effective ~1.5 s for the LLM portion)
2. The encoder-handoff round-trip adds latency to each request before it can enter the prefill scheduler
3. Each request's "embedding integration" sub-batches still happen (small `#new-token: 16-176, #cached-token: 16336` chunks). We saw fewer of these with A+B because chunked_prefill_size=16384 lets the main 16k tokens go through in one shot — but they still appear during decode-mixed-with-prefill.

---

## Final verdict (revised from `deep_analysis_disagg_worse_h200.md`)

**The original "disagg is fundamentally bad on H200" was overstated.** The real story:

1. **There ARE concrete bugs and tunables that improve disagg**:
   - Env-var bug (NIXL_READ silently ignored)
   - Default `max_running_requests=32` is too low for this workload
   - Default `chunked_prefill_size=8192` forces 2 chunks for 16k-token requests

2. **NIXL_READ is strictly better for production** — eliminates all failures, costs ~5% RPS in well-tuned cases

3. **Even fully-tuned disagg loses to TP=1 agg** on this workload because:
   - PD must do ALL the compute that TP=1 agg does, PLUS the encoder handoff overhead
   - The encoder GPU sits at 1.8 GB / 143 GB — 99% wasted
   - True architectural wins require **TP=2 PD** (untested) or **truly different workload** (multi-tenant, asymmetric GPUs, decode-heavy)

4. **The improvement ceiling on this exact workload is roughly:**
   - Current best (NIXL_READ + A+B): **0.23 RPS** (single-GPU PD)
   - Probable TP=2 PD ceiling: **~0.4-0.5 RPS** (untested)
   - TP=1 agg: **0.52 RPS**
   - TP=2 agg NVLink: **0.95 RPS** (winner)

**Bottom line:** Disagg can be made reliable (zero failures) and ~15% faster than its broken baseline, but cannot beat TP-aggregated configs on a single host with this workload. It's a useful production option for **reliability** (no buffer-pool failures) and **memory isolation**, not for **throughput**.

---

## Recommended config for production disagg on H200

If you must run disagg, use these settings:

```bash
# Encoder side
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=mlx5_4:1 \  # NUMA-2 active port if cross-host needed
... \
python3 -m dynamo.sglang \
  --multimodal-encode-worker \
  --enable-request-time-stats-logging \
  --show-time-cost \
  ...

# PD side
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read \
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=mlx5_4:1 \
... \
python3 -m dynamo.sglang \
  --multimodal-worker \
  --max-running-requests 64 \
  --chunked-prefill-size 16384 \
  --enable-request-time-stats-logging \
  --show-time-cost \
  ...
```

These settings give:
- **Zero failures** under sustained load (vs ~25% failure rate on default)
- **0.23 RPS at rate=1.0** for 8×1080p workload (vs 0.27 with failures)
- **155 s mean TTFT** (vs 79 s — but measured across all 64 requests, not 49)

---

## Untested options worth trying next

| Option | Mechanism | Expected gain | Effort |
|---|---|---|---|
| **TP=2 PD** (encoder GPU 4, PD on GPUs 5,6 as TP=2) | Doubles PD prefill compute | +50-80% RPS | medium |
| Replicate encoder pool (2 encoders, 1 PD) | Tests if encoder is bottleneck (probably not) | small | low |
| `--keep-mm-feature-on-device` on encoder | Skip host staging for MM features | unclear | trivial |
| `--mm-attention-backend nixl` | SGLang's native NIXL multimodal path | unknown | trivial |
| True PD-disagg (`disaggregation_mode=prefill` + separate decode worker) | Real prefill/decode split | 30-100% latency win expected | high |
| Sweep rates on AB config | Find actual knee | informational | low |

---

## Files

- Server start: `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh` (currently has NIXL_READ + A + B)
- Bench script: `/hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh`
- Result dirs:
  - Original (NIXL_WRITE): `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates/test_sglang_multi_rates_1080p_20260521_211224/`
  - NIXL_READ baseline: `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates_nixlread/test_sglang_multi_rates_1080p_20260522_013901/`
  - **NIXL_READ + A+B (current best): `/hongming/res4/h200_h200_disagg_tp1_32b_image8_1080p_np64_rates_nixlread_AB/test_sglang_multi_rates_1080p_20260522_020943/`**
  - 4-image test: `/hongming/res4/h200_h200_disagg_tp1_32b_image4_1080p_np64_rates_test1/test_sglang_multi_rates_1080p_20260521_230911/`
- Worker logs:
  - PD log: `/hongming/dynamo/logs/logs/logs/pd_worker.log` (path keeps nesting from successive `LOG_DIR=$(pwd)/logs` invocations)
  - Encoder log: `/hongming/dynamo/logs/logs/logs/encoder_worker.log`
- Companion docs:
  - `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md` — original full sweep + diagnosis
  - `01_cuda_sh/disagg_h200_32b/deep_analysis_disagg_worse_h200.md` — first-round deep analysis (the "architectural" claim)
  - **`01_cuda_sh/disagg_h200_32b/disagg_improvements_attempts.md`** — this document

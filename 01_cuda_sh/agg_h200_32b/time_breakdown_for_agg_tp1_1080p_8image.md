# Time-breakdown analysis: Aggregated EPD, "TP=2" → actually TP=1, 32B FP8, 1080p×8 images, rate=1 req/s

**System under test:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
**Model:** Qwen3-VL-32B-Instruct-FP8
**Benchmark:** `sglang.bench_serving --dataset-name image --image-count 8 --image-resolution 1920x1080 --random-input-len 128 --random-output-len 256 --num-prompts 64 --request-rate 1.0`
**Bench window:** 2026-05-20 21:16:02 – 21:20:08 (~245 s analysis window)
**Result file:** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_211545/rate_1.0/`

---

## Critical configuration finding (must fix first)

Despite the script being called `..._tp2.sh`, the worker is actually running **`tp_size=1`**. From the worker log:
```
device='cuda', tp_size=1, ... base_gpu_id=0, gpu_id_step=1, ...
mem_fraction_static=0.88, max_running_requests=25
```
And nvidia-smi confirms only one GPU is doing work (GPU 0 at 100%, the other "visible" GPU is idle). The script passes `--tensor-parallel-size 1` on line 120 of `start_h200_aggregate_epd_server_32b_tp2.sh:120`. So this "TP=2" run is really TP=1, on a single H200 NVL.

That matters a lot for the bottleneck story below.

---

## Workload (rate=1 req/s)

| | |
|---|---|
| Requests | 64 (1 req/s offered) |
| Images / req | 8 × 1920×1080 |
| Input text tokens | 5,029 (≈79/req) |
| Input vision tokens | **1,045,504 (≈16,336/req)** |
| Output tokens | 7,474 (≈117/req, far short of 256 cap) |
| Bench duration | 152.6 s |

So per request: ~16k vision tokens + ~80 text tokens → ~16.4k prefill tokens, ~117 output tokens. 99.5 % of input is vision.

---

## Headline performance

| Metric | Value |
|---|---|
| Request throughput | 0.42 req/s (offered 1.0 — **back-pressured**) |
| Total token tput | 6,933 tok/s |
| Input tput | 6,884 tok/s |
| Output tput | 49 tok/s |
| Mean / median TTFT | **50.2 s / 56.0 s**, p99 **84.9 s** |
| Mean / median TPOT | 622 ms / 354 ms, p99 **5.3 s** |
| Max ITL | **60.4 s** (a request stalled an entire minute mid-stream) |
| Concurrency observed | 39 (cap is `max_running_requests=25`) |

The system saturated almost instantly: queue grew to 39 reqs and stayed there.

---

## Wall-clock attribution from the worker log (4-min bench window)

Computed by attributing every gap between consecutive scheduler `report_*` events to the phase that just ran:

| Phase | Time | % of window |
|---|---:|---:|
| **Prefill** (vision-token chunked prefill) | 147.9 s | **60 %** |
| **Decode** | 26.2 s | 11 % |
| Idle / MM-preprocess / unattributed | 71.7 s | 29 % |

The "unattributed" 29 % is mostly the **vision encoder + image preprocessing** path that runs outside the scheduler step (and partly true idle near the end as the queue drains). With aggregated EPD the encoder shares the same GPU and is largely serialized w.r.t. the LLM forward pass.

---

## Inside prefill (where the bulk of time is spent)

Chunked prefill with `chunked_prefill_size=8192`. From the bench window:

| Chunk size (#new-token) | Count |
|---|---:|
| 8,192 | 127 |
| 8,176 | 9 |
| ≤ 96 (tail flushes / text) | 39 |

- Input throughput on full 8,192-token chunks: **median 9,518 tok/s, mean 8,707, max 11,829** — roughly **1 chunk every 0.85 s**.
- Each request is ~16,336 input tokens → 2 chunks → ~1.7 s of pure LLM-forward prefill **plus** the upstream vision-encoder time.
- **Cached tokens: 0** for ~99 % of prefills (`#cached-token: 0` in 186/188 stats; one 16,384 hit). With random-image content this is expected.

So per-request cost on a single H200, ignoring queueing:

```
vision encode (8 imgs @1080p)     ~ ?     (lives in the 29% unattributed)
LLM prefill of ~16k tokens         ~ 1.7 s @ 9.5k tok/s
LLM decode 117 tokens              ~ 0.6-1.0 s (TPOT 354 ms median, 622 ms mean)
```

That's ~3-5 s "service time", but **observed E2E mean is 93 s and TTFT 50 s** — i.e. ~95 % of latency is **queue-wait**.

---

## Inside decode

| | Value |
|---|---|
| Decode running-req | mostly 1; 25 (cap) only briefly at peak |
| Decode batch #tokens | median 33k, max 412k (KV cache pressure) |
| Gen throughput | median 71 tok/s, mean 127, **max 599** |
| Token-usage (KV) | sustained 0.71–0.74, peaking at 0.74 |

Decode batches are large (median 33k tokens) but get repeatedly preempted/interleaved by chunked prefill — that's why median TPOT is 354 ms (very high for decode) and ITL p99 is 3.8 s with one request hitting **60 s ITL**. That's classic **prefill-blocking-decode** in aggregated mode.

---

## Bottleneck ranking

1. **TP=1 on a 32B FP8 model serving 1080p×8 images.** Each request is ~16k prefill tokens. At ~9.5 k tok/s prefill capacity per H200, the steady-state prefill capacity is **~0.58 req/s** — so an offered rate of 1 req/s is above the knee. Result: queue saturates, TTFT explodes to 50 s. This is the dominant bottleneck. **0.42 req/s achieved ≈ 0.58 req/s ceiling minus encoder/decode overhead.**

2. **Aggregated EPD: vision-encoder + LLM compete for the same GPU.** ~29 % of wall-clock is unattributed in the scheduler trace; a sizable fraction of that is the ViT encoder pass for 8×1080p images. Because it's not pipelined with prefill, it inflates TTFT directly.

3. **Decode head-of-line blocking by prefill chunks.** With `disable_overlap_schedule=False` but no PD separation, every 8192-token chunk steals ~0.85 s from any in-flight decode → TPOT median 354 ms, max ITL 60 s.

4. **No prefix/MM cache reuse.** `#cached-token=0` for 186/188 prefill stats. `enable_mm_global_cache=True` is on, but each request uses random images → no hit. Not really actionable for this synthetic workload, but worth knowing the result is cache-cold.

5. **`max_running_requests=25` and `mem-fraction=0.88`** are not the binder right now — KV usage stayed at 0.71–0.74. So the cap is compute, not memory.

---

## Recommended next experiments (in priority order)

1. **Actually run TP=2.** Edit `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh:120` `--tensor-parallel-size 1` → `2`, and ensure `CUDA_DEVICE=3,4` is honoured (currently `set -e` plus the inline `#SGLANG_USE_CUDA_IPC_TRANSPORT=1 \` comment on line 109 may be silently breaking the env block — there's a `\` continuation right before a `#` comment, which actually works in bash but is fragile). At TP=2 prefill should ~1.7-1.9× to ~17 k tok/s and shrink TTFT roughly in half.
2. **Disaggregate E/PD** (dedicated encoder worker + prefill worker + decode worker) so the ViT pass and 8k-chunk prefills don't preempt decode. Should mainly cut TPOT/ITL tails (currently TPOT p99 = 5.3 s, ITL max = 60 s).
3. **Lower request rate or set `max_concurrency`** to find the true knee — try rate ∈ {0.25, 0.5, 0.6, 0.75} to confirm the ~0.58 req/s ceiling at TP=1.
4. **Reduce per-request vision tokens** (image_count, resolution) to verify the model is vision-bound, not language-bound.
5. **Consider increasing `chunked_prefill_size`** (e.g. 16384 = `max_prefill_tokens`) — with KV at 0.74 there's headroom, and it would let 16k-token requests prefill in a single chunk and unblock decode faster.

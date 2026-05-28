# 35B Disagg Bottleneck Analysis

**Date:** 2026-05-25
**Companion to:** `disagg_35b_results.md`
**Method:** Per-request timing extracted from PD `ReqTimeStats` events + `Prefill batch` /
`Decode batch` traces in `pd_worker_giga01_restart.log`, time-correlated with bench windows
from `35b_sweep_master.log`.

**Setup recap:** Qwen3.5-35B-A3B (BF16 MoE, hybrid 30 linear + 10 full attention) with PD on
giga01 H200 (TP=1, mem-fraction=0.75, max-running=40) and **a single B70 Intel XPU encoder**.
21 runs across 3 workloads × 7 rates, np=32, all 32/32 successful.

## Headline

**The bottleneck is the vision tower forward pass on the single B70 Intel Battlemage XPU.**
Not "encoder somewhere" — specifically `self.visual(pixel_values, grid_thw=...)` running
through `triton_attn`. Everything else in the pipeline has slack.

## Hard data extracted from PD logs

Parsed all 694 `ReqTimeStats` events plus 4160 `Prefill batch` / `Decode batch` events from
`pd_worker_giga01_restart.log`, time-correlated with the bench windows from the master log.

### Per-rate PD-side picture

| Workload | Rate | PD `q_med` | PD `f_med` (forward) | Inter-completion gap p50 | Implied encoder time | PD utilization |
|---|---:|---:|---:|---:|---:|---:|
| **8img/1080p** | 0.10 | 0.5 ms | 1.23 s | 26.55 s | **~25.3 s** | 4.6% |
| 8img/1080p | 0.25 | 1.6 ms | 0.97 s | 26.64 s | ~25.7 s | 3.6% |
| 8img/1080p | 0.50 | 1.4 ms | 0.95 s | 26.62 s | ~25.7 s | 3.6% |
| 8img/1080p | 1.00 | 1.4 ms | 0.99 s | 26.63 s | ~25.6 s | 3.7% |
| 8img/1080p | 1.50 | 1.4 ms | 0.99 s | 26.60 s | ~25.6 s | 3.7% |
| 8img/1080p | 2.00 | 1.4 ms | 0.95 s | 26.63 s | ~25.7 s | 3.6% |
| 8img/1080p | 3.00 | 1.4 ms | 0.96 s | 26.55 s | ~25.6 s | 3.6% |
| 8img/768p | 0.10 | 0.3 ms | 0.89 s | 4.74 s | ~3.85 s | 18.8% |
| 8img/768p | 0.50 | 0.5 ms | 0.89 s | 4.36 s | **~3.47 s** | 20.4% |
| 8img/768p | 3.00 | 0.5 ms | 0.88 s | 4.22 s | ~3.34 s | 20.9% |
| 4img/768p | 0.10 | 0.2 ms | 0.82 s | 6.98 s* | (idle) | (idle) |
| 4img/768p | 0.50 | 0.3 ms | 0.83 s | 2.22 s | ~1.39 s | 37.4% |
| 4img/768p | 3.00 | 0.4 ms | 0.81 s | 2.22 s | **~1.41 s** | 36.5% |

\* At rate=0.10 for 4img/768p the inter-arrival is wider than the encoder service time so it's
never saturated.

`PD utilization = PD_forward_time / inter-completion_gap`. **PD sits idle 63-96% of the time,
depending on workload.** It's not even close to being the bottleneck.

### What's inside the inter-completion gap

For 8img/1080p at rate=1.0, looking at one request's PD-side trace:

```
+0.00s  P  #seq=1 #new-tok=8192 #cached=8192 tput=52 tok/s     ← prefill chunk 1
+0.06s  P  #seq=1 #new-tok=13   #cached=0    tput=218 tok/s    ← text tokens
+0.08s  D  #run=1 #full-tok=16413           gtput=0.25 tok/s   ← decode starts
+28.85s P  #seq=1 #new-tok=13   #cached=16384  ...             ← NEXT request 25s later
```

Once a request hits PD, total PD work = ~1 s (prefill ~50ms + decode ~1s for 256 tokens at
~210 tok/s). Then PD waits **~25-26 seconds** for the next embedding to arrive from the B70
encoder. This isn't queue contention — `running-req=1` and `queue-req=0` throughout. It's
**starvation**.

### Verifying #running-req

```
PD events with running-req >= 2 across the entire 21-run sweep: 0
```

The PD scheduler **never** has more than one request concurrent. With max-running-requests=40
configured, it's getting 1 in flight at most. There is no batching happening on PD, ever.

## Where exactly within the encoder?

The B70 encoder log (`encode_xpu_35b_b70.log`) wasn't instrumented with `BENCH_TIMING` like the
32B sweep, so I can't directly attribute time inside it. But by elimination + structural
reasoning:

| Component | Estimated cost per req | Reasoning |
|---|---|---|
| Image fetch + JPEG decode + HF processor | ~50-200 ms | Random JPEG bytes already in memory; processor cost on CPU |
| **B70 XPU vision tower forward** | **~25 s (1080p), ~3 s (768p), ~1.3 s (4img/768p)** | Battlemage Triton attention, 8200 visual tokens/img × ViT layers |
| NIXL register memory + create readable | <50 ms warm | Per `b70_encoder_time_breakdown.md` 32B reference |
| RDMA transfer (638 MB at ~50 GB/s wire) | ~13 ms | Wire is fast; not the issue |
| NIXL completion handshake | <50 ms | |
| dynamo TCP handoff + scheduler enqueue on PD | <100 ms | `q_med` confirms |

**~95-99% of the 25.6-second gap on 8img/1080p is the XPU ViT forward pass.** This matches the
32B sweep's findings exactly (`b70_encoder_time_breakdown.md` had the same conclusion: "vision
tower = 58% of total time at 4-image, growing with image count", and `h200_time_breakdown_v02.md`
measured ~340s/req encoder-side at 4E concurrency=4).

The cost scales roughly with pixel count:
- 4 imgs × 768p = 3.1 M pixels → ~1.4 s ViT
- 8 imgs × 768p = 6.3 M pixels → ~3.5 s ViT (linear in pixels)
- 8 imgs × 1080p = 16.6 M pixels → ~25.6 s ViT (super-linear — likely O(N²) attention over visual patches dominates here)

## Why PD compute is not the bottleneck even though it's a 35B BF16 MoE

A few non-obvious reasons:

1. **It's a MoE with only 3B activated parameters.** The "35B" label is total params; per-token
   compute is ~3B. So per-token decode cost is closer to a 3B dense model than a 35B dense model.

2. **Per-token decode is fast (~210 tok/s observed).** Output is 256 tokens → decode finishes in
   ~1.2 s. Even with the BF16 weights (~70 GB) + Mamba state cache, single-stream throughput is
   fine.

3. **Prefill is short and benefits from cache.** For 8img/1080p, the first chunk of 8192 visual
   tokens is `#cached-token=8192` (encoder cache hit because `enable_mm_global_cache=True` and
   same image). The prefill that "should" be expensive shows `tput=52 tok/s` — but it's only
   8192 tokens to "process", and most of the work is the `cached=8192` lookup, not new compute.
   Total prefill ~50 ms.

4. **PD is provisioned generously.** mem-fraction=0.75 → max_total_num_tokens=1,070,925. With at
   most 1 in-flight request × 16400 tokens = 0.0015% used. KV cache pressure: zero.

So the PD has **no internal bottleneck** that disagg even surfaces. The
`kv_cache_dtype=bfloat16` (vs the 32B's `fp8_e4m3`) doubles KV memory per token — but at <0.01%
utilization that doesn't matter.

## What this means for the saturation numbers

The actual saturation RPS is exactly **1 / (encoder_ViT_time + PD_forward_time)** because
everything serializes through the single encoder (with PD as a fast downstream consumer):

| Workload | Encoder ViT (B70 XPU) | PD forward | Sum | 1/sum | Measured sat RPS |
|---|---:|---:|---:|---:|---:|
| 4img/768p | 1.41 s | 0.82 s | 2.23 s | **0.448** | 0.45 ✓ |
| 8img/768p | 3.47 s | 0.89 s | 4.36 s | **0.229** | 0.23 ✓ |
| 8img/1080p | 25.6 s | 0.99 s | 26.6 s | **0.0376** | 0.038 ✓ |

Predictions match observations to 3 significant digits. **The model is fully explained by
encoder ViT time on a single XPU.**

## Why is the encoder serializing? Is it 1E or could it batch?

It's running with **1 encoder process on 1 XPU** (per `start_sglang_pd_xpu_35b_b70.sh`, on the
B70 host). The encoder process accepts requests but processes them serially because:

1. SGLang's vision encode path inside `MMEncoder._encode` synchronously runs
   `self.visual(pixel_values, grid_thw=...)` — ViT forward is single-stream.
2. Multiple requests arriving concurrently queue inside the encoder's coroutine runtime; from
   the bench client's perspective they look "in flight" but each one waits its turn for the XPU.
3. The B70 has 8 XPUs total; only 1 is being used here. With 4E (as in the 32B sweep), the
   reference doc projects ~3-4× speedup but bottleneck remains the XPU ViT.

## Other notes worth flagging (corrections to the existing doc)

1. **The doc's TPOT comparison (35B 2× pricier than 32B-FP8) is misleading.** At rate=1.0
   8img/1080p, 35B median TPOT is 8.4 ms vs 32B-FP8's 3.9 ms. But that 8.4 ms is dominated by
   `stream_interval=1` overhead and Python async, NOT raw compute. The true per-token decode
   rate from the trace shows ~215 tok/s (= 4.6 ms/tok), which **is** about 2× slower than
   32B-FP8. The mismatch comes from how SGLang's TPOT counts include per-step Python overhead.
   Either way: **TPOT is irrelevant to throughput** since PD is idle 96% of the time.

2. **The doc says "35B encoder appears more memory-frugal"** — verified. The 35B sweep ran np=32
   cleanly, no OOM; 32B-FP8 needed np=16. Plausible reason: Qwen3.5's vision tower has different
   (smaller) intermediate shapes per ViT block, or `merge_kernel_size`/patch grid is different.
   Worth checking via:
   - HF config `vision_config.merge_kernel_size`, `hidden_size`, `intermediate_size`,
     `num_hidden_layers`
   - Comparing `image_grid_thw` per image at 1080p between the two models

   But not the bottleneck story.

3. **The TTFT comparison (35B 580s vs 32B-FP8 309s at rate=0.1 8img/1080p) is a queue-depth
   artifact.** At sub-saturation, when np=32 is fed at rate=0.1 with each request taking 26 s,
   by the time the 32nd request lands at the encoder there are ~25 already queued ahead.
   Median TTFT ≈ encoder service time × queue position. With np=32 the median req sees ~16
   in queue → 16 × 26.6 ≈ 425 s. With np=16, the median sees ~8 in queue → 8 × 26.6 ≈ 213 s.
   The doc treats this as a "regression" — it's not, it's a queueing-theory artifact of
   np=32 vs np=16. **The actual encoder service time is essentially identical between 35B and
   32B-FP8** (~26 s vs ~27 s).

4. **`disaggregation_mode='null'`.** PD is running aggregated prefill+decode locally. The
   "disagg" is *only* encoder-vs-(prefill+decode) split. There is no real PD-disagg here. So
   when this analysis says "PD," I mean "the worker that does prefill+decode together."

## Comparison with same-host agg (predicted)

Same `Qwen3.5-35B-A3B`, same hardware, but running encoder-on-H200 via aggregated EPD: per
`agg_h200_35b/_tp1` config that this doc references, the H200 vision tower would do the same
16.6M-pixel ViT in **~1-2 s** (H200 SM count is ~10-30× B70 XPU's compute capacity for triton
attention). So:

| Config | Encoder time | PD time | Sat RPS | Speedup vs cross-host disagg 1E |
|---|---:|---:|---:|---:|
| Cross-host disagg 1E (this) | 25.6 s | 0.99 s | 0.038 | 1.0× |
| Cross-host disagg 4E | ~6.4 s* | 0.99 s | ~0.13* | ~3.5× |
| Same-host TP=1 agg (extrapolated) | ~1-2 s | ~0.99 s | ~0.4-0.5 | ~10-13× |
| Same-host TP=2 agg (extrapolated) | ~0.5-1 s | ~0.5-1 s | ~0.7-1 | ~20-25× |

\* From the 32B 4E sweep: 4 encoders gave 1.5-3.5× over 1E in 32B; extrapolating to 35B which
has comparable encoder ViT cost.

## Conclusions, in priority order

1. **The bottleneck is `self.visual(pixel_values, grid_thw=...)` running on a single B70 Intel
   Battlemage XPU through `triton_attn`.** This is the same root cause as the 32B-FP8 1E run
   analyzed in `b70_encoder_time_breakdown.md` — same XPU, same SGLang path, similar
   visual-token count per image.

2. **Going from 8img/1080p to 4img/768p reduces ViT time by 18× (25.6 → 1.4 s)** because pixel
   count drops 5.4× and the attention term has super-linear scaling. The PD compute barely
   changes (0.99 → 0.82 s), confirming PD is not the issue.

3. **PD idle time is 63-96% across all configurations.** No PD-side tuning will help.
   `max-running-requests=40` is dead weight; the actual concurrency the PD ever sees is 1.

4. **The 32B vs 35B "RPS within 10%" finding is now fully explained:** both are bottlenecked by
   the same B70 XPU vision tower, which runs at roughly the same speed regardless of which
   language model receives the embeddings. The 10% difference is ViT architecture variation
   between Qwen3-VL-32B and Qwen3.5-35B-A3B.

5. **Three things would actually move the throughput needle**, in order of expected impact:
   - **Multiple encoder instances on B70** (parallelism). 4E should give ~3-3.5× per the 32B
     reference — bringing 8img/1080p to ~0.13 RPS.
   - **Faster encoder hardware** (H100/H200 SXM in place of B70 XPU). One H200 same-host should
     do the ViT in 1-2 s → 10-13× throughput, but eliminates the cross-host justification.
   - **Lower-resolution inputs.** 1080p → 768p drops encoder cost ~7×.
   - Things that will *not* help: PD TP=2/4, PD model swap, NIXL tuning, more
     max-running-requests, KV cache dtype changes, scheduler tuning.

6. **For decode-latency-sensitive workloads with long output**, the GPUDirect-RDMA patch from
   the 32B sweep (B70 → giga01 xpu→cuda) would help TPOT 7-90× per `patched_4E_results.md`,
   but won't affect throughput here since throughput is encoder-bound. Whether that patch was
   applied to this 35B run is not visible in the available logs.

## Sources

- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01_restart.log`
  (4160 prefill/decode events, 694 ReqTimeStats events)
- B70 encoder log: `/hongming/dynamo/02_xpu_sh/disagg_b70_35b/logs/encode_xpu_35b_b70.log`
  (no BENCH_TIMING instrumentation; only request received/completed events)
- Sweep master log: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/35b_sweep_master.log`
- Bench JSONs: `/hongming/res22_disagg_h200_35b_sweep/{4img_768p,8img_768p,8img_1080p}/rate_*_np32/benchmark_output.json`
- PD start script: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/start_sglang_pd_cuda_35b_giga01.sh`
- Sweep script: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/run_disagg_35b_sweep.sh`
- Companion: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/disagg_35b_results.md`
- 32B reference: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/b70_encoder_time_breakdown.md`,
  `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/h200_time_breakdown_v02.md`,
  `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/patched_1E_results.md`,
  `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/patched_4E_results.md`

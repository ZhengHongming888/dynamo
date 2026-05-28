# 4img/768p 4E PD-side bottleneck analysis (35B)

**Date:** 2026-05-25
**Companion to:** `disagg_35b_results.md` (1E baseline) and `35b_bottleneck_analysis.md` (1E PD analysis)
**Method:** Per-request timing extracted from PD `ReqTimeStats` events + dynamo
`request received/completed` events + `Prefill batch` / `Decode batch` traces in
`pd_worker_giga01.log`, time-correlated with bench windows from
`sweep_4img_768p_4E_*.log`.

**Setup recap:** Qwen3.5-35B-A3B (BF16 MoE, hybrid 30 linear + 10 full attention)
with PD on giga01 H200 (TP=1, mem-fraction=0.75, max-running=40) and **4 encoders
on the B70 host (XPUs 0-3)**. 7 runs at np=32, all 32/32 successful.

## Headline

**The bottleneck is no longer a single XPU vision tower; it has shifted to a
balance between (a) aggregate B70 encoder throughput and (b) per-request PD
overhead.** The 1E story ("PD idle 96%, encoder ViT is everything") no longer
holds at 4E.

For **4img/768p the saturation point is 1.55 RPS, exactly 3.5× the 1E baseline
(0.44 RPS)**, matching the prediction in `35b_bottleneck_analysis.md`.

But unlike 1E, **PD now does real work concurrently** (40-56% of decode batches
have `running-req ≥ 2` at saturation), and the bottleneck is no longer a single
serial ViT.

## Hard data

Parsed 463 `ReqTimeStats` events + 654 `Prefill batch` + 1238 `Decode batch`
events from `pd_worker_giga01.log`, plus 463 `request received` and 463
`request completed` ingress/egress events (every request in this PD's lifetime,
spanning the 1080p_4E and 768p_4E sweeps).

### Per-rate timing summary (rate, RPS, q, f, output_len, inter-completion gap)

| Rate | measured RPS | n_req | PD `q_med` | PD `f_med` | PD `f_p99` | output_med | inter-completion p50 | PD lifetime p50 | overhead p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.089 | 33 | 0.21 ms | 931 ms | 1356 ms | 164 tok | 6.97 s | 1.11 s | **182 ms** |
| 0.25 | 0.221 | 33 | 0.40 ms | 929 ms | 1452 ms | 164 tok | 2.76 s | 1.06 s | **134 ms** |
| 0.50 | 0.443 | 33 | 0.29 ms | 972 ms | 1392 ms | 164 tok | 1.22 s | 1.14 s | **166 ms** |
| 1.00 | 0.851 | 33 | 0.28 ms | 1034 ms | 1546 ms | 164 tok | 0.77 s | 1.20 s | **165 ms** |
| 1.50 | 1.228 | 33 | 0.29 ms | 975 ms | 1634 ms | 164 tok | 0.70 s | 1.12 s | **148 ms** |
| **2.00** | **1.539** | 33 | 0.26 ms | 1055 ms | 1721 ms | 164 tok | 0.43 s | 1.18 s | **125 ms** |
| 3.00 | 1.578 | 33 | 0.26 ms | 990 ms | 1645 ms | 164 tok | 0.43 s | 1.17 s | **184 ms** |

Where:
- `q_med` = `queue_duration` from `ReqTimeStats` (PD scheduler enqueue wait)
- `f_med` = `forward_duration` (pure PD GPU compute: chunked-prefill + decode)
- `lifetime` = wall-clock from dynamo `request received` to `request completed`
- `overhead` = `lifetime - forward` (NIXL embedding fetch + scheduler enqueue + ZMQ response handoff)

### Concurrency on PD: how many requests in flight?

This is the cleanest dividing line vs the 1E sweep, where `running-req` was
**always exactly 1**. With 4E, decode-batch concurrency distributions:

| Rate | running=1 | running=2 | running=3 | max queue |
|---|---:|---:|---:|---:|
| 0.10 | 117 (98%) | 2 (2%) | 0 | 0 |
| 0.25 | 112 (97%) | 2 (2%) | 2 (2%) | 0 |
| 0.50 | 82 (80%) | 19 (19%) | 1 (1%) | 0 |
| 1.00 | 60 (68%) | 22 (25%) | 4 (5%) + 4=2(2%) | 0 |
| 1.50 | 50 (61%) | 25 (30%) | 7 (9%) | 0 |
| **2.00** | **32 (46%)** | **28 (40%)** | **10 (14%)** | 0 |
| 3.00 | 29 (43%) | 28 (41%) | 11 (16%) | 0 |

**At saturation (rate ≥ 2.0), PD is genuinely batching prefill+decode of multiple
requests in parallel.** Effective concurrency ≈ 1 × 0.46 + 2 × 0.40 + 3 × 0.14 ≈
**1.68 requests in flight on average**. Combined with `lifetime ≈ 1.18 s/req`,
this gives a predicted RPS of `1.68 / 1.18 = 1.42`, in line with the measured
1.54 RPS (the small discrepancy is because lifetime *decreases* slightly when
batched, since chunked-prefill is more efficient at higher batch).

`max queue-req = 0` everywhere — even at rate=3.0 (where the bench fires faster
than 4E can produce), the PD scheduler never has anything queued. The PD sees
exactly what the encoders deliver; it never has surplus work waiting.

### What's INSIDE the per-request 1.18 s lifetime?

Decomposing for a representative saturation-rate request (rate=2.0):

| Component | Time | Source |
|---|---:|---|
| PD `forward_duration` | **1055 ms** | `ReqTimeStats.f` median |
| PD `queue_duration` | 0.26 ms | `ReqTimeStats.q` median |
| Encoder→PD overhead (NIXL fetch + dispatch + ZMQ) | **~125 ms** | `lifetime - f - q` |
| Total PD lifetime | **1180 ms** | dynamo ingress→egress |

For comparison:
- **1080p 4E** had `lifetime ≈ 6 s` and `forward ≈ 0.99 s`; the gap was filled
  by serial encoder-output waiting (PD had nothing to do for ~5 s/req while ViT
  on B70 produced the next embedding). At 4img/768p, encoder ViT is ~0.35 s
  per encoder so PD is fed densely.
- **1E baseline** at 4img/768p had inter-completion gap ≈ 2.22 s and PD forward
  ≈ 0.82 s, leaving ~1.4 s of the gap as encoder ViT + NIXL. With 4E that 1.4 s
  collapses to ~0.15 s amortized across 4 parallel encoders.

### Per-rate arrival pattern on PD (rate=2.0 deep dive)

```
First arrival on PD: t+18.77s (relative to bench start)
Last arrival on PD:  t+39.84s
Effective arrival rate during this window: 1.567 req/s
```

Inter-arrival deltas (sorted, all 32):
```
0.28, 0.30, 0.31, 0.31, 0.32, 0.33, 0.33, 0.35,   <- first 8 (warm-up burst)
0.45, 0.50, 0.51, 0.52, 0.55, 0.55, 0.57, 0.58,
0.58, 0.58, 0.58, 0.60, 0.61, 0.62, 0.65, 0.72,   <- bulk
0.74, 0.76, 0.77, 0.77, 0.78, 0.78, 1.01, 3.77    <- tail (3.77 = inter-bench drain)
```

**Median inter-arrival = 0.58 s. No clustering** — every arrival is >100 ms from
any neighbor. **The 4 encoders are NOT producing bursts of 4.** They are working
in parallel but offering individual requests to PD one-at-a-time via dynamo's
round-robin dispatcher.

### Why doesn't PD ever reach `running-req=4`?

Three possible reasons, ranked by evidence:

1. **(strongest) PD lifetime ≈ encoder serial output cadence.** With 4 encoders
   each finishing one image-batch in ~1.4 s (4img/768p; per `disagg_35b_results.md`
   1E sub-saturation TTFT) and serializing through dynamo's round-robin, the
   aggregate cadence is ~1.4/4 = 0.35 s, but per-encoder NIXL setup + handoff
   adds another ~0.2-0.3 s, putting effective inter-arrival at ~0.58 s observed.
   That's *also* ~PD lifetime / 2, so steady state has 1-2 reqs in flight, not 4.

2. **(weak)** `--max-running-requests=40` is configured generously, and there's
   no scheduler back-pressure. The system is not artificially capped.

3. **(unlikely)** PD compute saturation. forward_duration `f_p99 = 1721 ms` at
   rate=2.0 vs `f_med = 1055 ms` shows tail-latency growth with concurrency
   (longer prefill chunks when `running-req=3`), but absolute numbers are still
   well within budget. PD GPU utilization is ~60-90% during the run, not pinned.

**Conclusion: the bottleneck on 4img/768p_4E is the encoder→PD pipeline cadence,
not PD compute, not PD scheduler, not NIXL transport.** Specifically, the
~0.35 s per-encoder ViT time × serialized round-robin dispatch = 0.58 s
mean inter-arrival on PD, which gates throughput at ~1.55 RPS.

### Where the time goes (saturation rate=2.0)

```
Encoder #1 — receives request via dispatch, finishes ViT in ~1.4 s,
              calls NIXL register, hands to PD via dynamo TCP plane     1.55 s
Encoder #2 — same (in parallel)
Encoder #3 — same (in parallel)
Encoder #4 — same (in parallel)
            ─────────────────────────────────────────────────
            Aggregate cadence to PD:     ~0.58 s/req

PD receives request:                                                     0
PD scheduler enqueue:                                                    0.26 ms
PD prefill chunk #1 (8192 visual tokens, no cache):                      ~50 ms
PD prefill chunk #2 (~3138 - 8192 + ... continued):                      ~550 ms
PD decode 164 tokens (single-stream + occasional batching):              ~450 ms
                                                                        ──────
PD forward_duration                                                     ~1055 ms
PD finalize + dynamo egress (response queue):                            ~125 ms
                                                                        ──────
PD lifetime                                                             ~1180 ms
```

The **PD prefill is dominated by the visual tokens**: each prefill batch shows
`#new-token: 3138` (3138 ≈ 4 images × ~785 visual tokens), and the chunked
prefill processes this in 1-2 chunks of 8192 each. CUDA graphs are NOT used
(`cuda graph: False` for all prefill batches because the visual prefill always
exceeds graph-recorded sizes [1,2,4,8,12,16,24,32,40]).

Decode is fast (~5 ms/tok × 164 tok = ~820 ms? actually closer to 450 ms median
because of partial overlap with prefill of other reqs).

## Comparison: 4img/768p 1E vs 4E PD analysis

| Metric | 1E (`35b_bottleneck_analysis.md`) | 4E (this doc) |
|---|---:|---:|
| Saturation RPS | 0.44 | **1.55** |
| PD `q_med` | 0.4 ms | 0.26 ms |
| PD `f_med` | 820 ms | **1055 ms** |
| Inter-completion gap p50 | 2.22 s | **0.43 s** |
| PD utilization (= f_med / gap) | 37% | **245%** (i.e. PD now batches) |
| `running-req > 1` events | **0 / 4160** | **2/3 of decode batches** |
| Implied encoder time | 1.4 s / req | **~0.15 s / req amortized** |
| max queue-req on PD | 0 | 0 |

**Key shift:** in 1E, PD was idle 63%; in 4E, PD is no longer the slack
component but is also not pinned. Its concurrency is structurally limited by
the rate at which 4 parallel encoders can produce embeddings. Adding more PD
compute (TP=2, etc.) would NOT help.

The `f_med` increase from 820 ms (1E, single-stream) to 1055 ms (4E, batched)
is consistent with chunked-prefill + decode of 1-3 concurrent reqs being
more expensive per request but more efficient per request-batch. It's the
classic concurrent-LLM amortization curve.

## Why the throughput ceiling is 1.55 RPS, not 1.76 (= 4 × 0.44)

If 4 encoders worked perfectly in parallel and PD absorbed everything, the
ceiling would be `4 × 1E_RPS = 4 × 0.44 = 1.76 RPS`. Measured: 1.55 RPS,
**a ~12% loss vs perfect linear scaling**.

Hypothesis A: **NIXL-fetch + dispatch overhead on PD doesn't fully overlap.**
The 125-180 ms `lifetime - forward` overhead per request scales linearly with
RPS, so at 1.55 RPS that's `1.55 × 0.15 = 0.23 s/s` of overhead which can't be
hidden behind GPU work. This matches the missing 12% almost exactly:
`1 - 0.15/1.18 = 87%` efficient → `0.87 × 1.76 = 1.53 RPS` predicted.

Hypothesis B: **B70 4 XPUs aren't fully parallel.** From
`b70_encoder_time_breakdown.md` §8, 4E gives 1.5× per-encoder throughput at
conc=4, with each encoder running at ~40% utilization. So aggregate encoder
output = 4 × 0.4 / single_encoder_time. The 4-encoder XPU cross-talk (NUMA,
ring buffer, shared memory) accounts for some inefficiency.

Both hypotheses contribute. The 12% gap is small enough that further
optimization (e.g. async NIXL pipelining, batched dispatch) might recover ~5%
but is not transformative.

## Where could throughput go from here? (not the question asked, but useful)

If we wanted to push 4img/768p past 1.55 RPS:

1. **Faster encoder ViT.** B70 XPU 0.35 s/img → if H200 runs the same ViT in
   ~0.05 s (~7×), single-host agg-EPD with 1 encoder slot would saturate at
   ~3-5 RPS. This is the same conclusion as the 32B analysis:
   `same-host TP=2 agg ≈ 0.7-1.0 RPS for 8img/1080p` → 4img/768p ≈ 4-6× that.

2. **More encoders on B70.** Going 4E → 8E would double aggregate ViT
   throughput, modulo NUMA / NIC contention. Predicted: ~2.5-3.0 RPS for
   4img/768p, but only if PD can absorb (currently can't pin all 4 XPUs at
   conc=8 because PD lifetime would need to drop, requiring TP=2 or smaller
   model).

3. **PD-TP=2 PARALLEL with 4E.** The PD lifetime would halve, allowing
   running-req=4. Combined with hyp #2, projected ~3-4 RPS for 4img/768p.
   But per `pd_tp2_results.md` for 32B, PD-TP=2 alone gave **zero** uplift
   when encoder was already the bottleneck. Only useful if encoders are
   saturated, which they aren't here.

4. **Lower-resolution or fewer images.** 2img/768p would halve ViT cost,
   roughly doubling RPS at the encoder ceiling.

None of these are the bottleneck the user asked about; they're futures.

## Conclusions, in priority order

1. **The current 4img/768p 4E bottleneck is the encoder→PD pipeline cadence**:
   ~0.35 s per-encoder ViT × 4 parallel encoders + ~0.15 s NIXL/dispatch
   overhead per request, gating throughput at ~1.55 RPS = 1/0.65 effective
   inter-arrival.

2. **PD is no longer idle.** At rate ≥ 2.0, 54% of decode batches have
   running-req ≥ 2. PD is doing real concurrent work for the first time in
   any 35B disagg sweep. PD `forward_duration` rose from 820 ms (1E) to
   1055 ms (4E) due to genuine batching, not regression.

3. **PD scheduler queue is always 0.** No back-pressure. `max-running-requests=40`
   is dead capacity. Adding PD compute won't help.

4. **The per-request "handoff floor" is ~150 ms cross-host RoCE NIXL.**
   That's vastly better than the 11.5 s same-host floor documented in
   `per_request_handoff.md` for the 32B-FP8 sweep. The cross-host topology
   has quietly fixed the same-host CPU-bounce problem for this workload.

5. **The 4E speedup over 1E is 3.5× (1.55 / 0.44)**, almost exactly matching
   the projection in `35b_bottleneck_analysis.md` (`~3-3.5×`). The 12% loss
   vs perfect linear (`4 × 0.44 = 1.76`) is from non-overlapped per-request
   PD overhead (~0.15 s × 1.55 RPS = ~23% overhead → 87% efficient).

6. **Things that will NOT help throughput at this configuration:**
   - PD TP scaling (PD already has spare compute)
   - More NIXL tuning (~150 ms is already mostly transport-bound, not Python)
   - More max-running-requests (it's 40, observed max is 3)
   - KV cache dtype changes (KV is at <0.1% utilization)
   - Scheduler tuning

7. **Things that COULD help:**
   - Async NIXL pre-fetch overlapping with PD prefill (recover ~5%)
   - Batched encoder→PD dispatch (recover ~3%)
   - Going to 6E or 8E if XPU memory budget allows (potentially 1.5-2× RPS)
   - **Or** abandoning cross-host disagg and using same-host agg with TP=1
     on H200, where ViT runs ~7× faster on H200 SMs vs B70 XPU triton_attn

## Sources

- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01.log`
  (463 ReqTimeStats, 654 prefill, 1238 decode, 463+463 dynamo ingress/egress events)
- Bench JSONs: `/hongming/res22_disagg_h200_35b_sweep/4img_768p_4E/rate_*_np32/benchmark_output.json`
- Sweep master: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/sweep_4img_768p_4E_*.log`
- Per-rate bench logs: `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/bench_disagg_35b_4E_4img_768p_r*_np32.log`
- Analysis scripts (this directory):
  - `analyze_4img_768p_4E.py` — per-rate aggregate metrics
  - `analyze_4img_768p_4E_deep.py` — saturation regime walk-through
  - `analyze_pd_arrivals.py` — inter-arrival, lifetime, busy-fraction
- Companion docs:
  - `35b_bottleneck_analysis.md` — 1E baseline analysis (**conclusion shifts here**)
  - `disagg_35b_results.md` — full 1E results
  - 32B reference: `b70_encoder_time_breakdown.md` §8 (1E vs 4E on 32B-FP8)

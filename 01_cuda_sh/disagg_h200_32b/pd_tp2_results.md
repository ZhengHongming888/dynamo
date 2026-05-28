# 32B-FP8 Disagg with PD-TP=2 (3-GPU disagg) — Results

**Date:** 2026-05-23 (run completed ~07:21 UTC)
**Model:** Qwen3-VL-32B-Instruct-FP8
**Workload:** 8 × 1080p (1920×1080) random images, 128 input + 256 output tokens, np=64 per rate
**Rates swept:** 0.1, 0.25, 0.5, 1.0, 1.25 RPS
**Hardware:** H200 ×3
- **Encoder**: GPU 4 (NUMA 2)
- **PD worker (TP=2)**: GPUs 5,7 (5=NUMA 2, 7=NUMA 3) — NVLink NV18 between them

This run tests the hypothesis: "**disagg loses because PD is TP=1; if we give it TP=2, it should match or beat TP=2 agg.**"

**Result: NO. PD-TP=2 disagg performs the same as PD-TP=1 disagg (~0.24 RPS) — adding a 3rd GPU gave essentially zero throughput improvement.**

---

## Saturation results

| Rate | Actual RPS | Mean TTFT (s) | Mean TPOT (ms) | Mean E2E (s) | Concurrency |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.10 | 26.3 | 113 | 34.6 | 3.5 |
| 0.25 | 0.23 | 65.6 | 3,058 | 161.9 | 36.7 |
| 0.50 | 0.23 | 196.3 | 184 | 211.2 | 49.5 |
| 1.00 | **0.24** | 229.6 | 57 | 236.6 | 56.3 |
| 1.25 | **0.24** | 235.6 | 58 | 242.6 | 57.7 |

**Saturation: ~0.24 RPS** (essentially flat from rate=0.25 onward).

---

## Comparison across all 32B configs at 8 imgs × 1080p (np=64)

| Config | # GPUs | Saturation RPS | RPS / GPU | Verdict |
|---|---:|---:|---:|---|
| TP=1 agg | 1 | 0.52 | **0.52** | Best per-GPU |
| **TP=2 agg** | **2** | **0.95** | **0.48** | **Best absolute** |
| Disagg (TP=1 PD) | 2 | 0.23 | 0.12 | Worst per-GPU |
| **Disagg (TP=2 PD)** | **3** | **0.24** | **0.08** | **Adding 3rd GPU = 0 gain** |

**The 3rd GPU added zero throughput.** 3-GPU disagg = 0.24 RPS vs 2-GPU disagg = 0.23 RPS.

---

## Why didn't TP=2 PD help?

Comparing the same workload across configs:

| Metric @ rate=1.0 | TP=2 agg | Disagg (TP=1 PD) | **Disagg (TP=2 PD)** |
|---|---:|---:|---:|
| Actual RPS | 0.95 | 0.23 | **0.24** |
| Mean TTFT (s) | ~few s | ~80 | **230** |
| Mean TPOT (ms) | ~50 | ~600 | **57** |
| Mean E2E (s) | ~10 | ~150 | **237** |
| Concurrency | ~10 | ~50 | **56** |

**Three observations:**

### 1. TPOT actually GOT BETTER with TP=2 PD (57 ms vs 600 ms)
Decode is faster on 2 GPUs than on 1 — as expected. So the LLM itself is using the second GPU productively for decode.

### 2. But TTFT got WORSE (230 s vs 80 s)
This is the key signal. The prefill side became 3× SLOWER despite having 2× the compute. Why?

Looking at the data: at rate=0.5, **mean TTFT = 196 s, concurrency = 49.5**. That means ~50 requests are queued waiting for prefill — the prefill pipeline is saturated even harder than the TP=1 PD case.

### 3. Throughput stayed flat at ~0.24 RPS

This is the smoking gun. **Per-request prefill latency increased proportionally with concurrency** — meaning prefill is bottlenecked NOT on compute (TP=2 doubled compute) but on something upstream.

---

## What's actually the bottleneck

The encoder→PD hand-off pipeline. Each request has to:

1. Frontend receives HTTP request → parses 17 MB body
2. Routes to encoder worker via TCP request plane
3. Encoder runs ViT, produces ~28 MB embeddings
4. Encoder publishes embedding metadata via ZMQ to scheduler
5. PD scheduler picks up the request, allocates KV slot
6. PD worker reads embeddings via NIXL (cuda_ipc / NVLink)
7. PD does chunked prefill of ~16K visual tokens
8. Decode begins, streams back through frontend

**Steps 1-6 add per-request latency that's independent of how big the PD worker is.** Doubling PD compute (TP=1 → TP=2) makes step 7 faster, but steps 1-6 stay the same. At higher rates, those steps queue up.

Specifically, the TTFT explosion suggests **the bottleneck is at step 5-6** (scheduler ↔ NIXL transfer ↔ KV allocation). With 56 concurrent requests in flight, 56 NIXL reads and 56 scheduler events stack up — and PD can only pre-process them at some fixed rate, regardless of its compute capacity.

---

## Observation: TPOT is actually GREAT with TP=2 PD

Median TPOT = **41 ms** across all rates (vs 41 ms for TP=2 agg too).

So **once a request finally starts decoding, the decode runs at full TP=2 speed.** This confirms compute is fine. The problem is the prefill / hand-off pipeline can't keep up.

---

## What this means for disagg

This was the strongest disagg config we could test on this hardware (3 GPUs, dedicated encoder, TP=2 PD). It delivered:
- **0.24 RPS saturation** (worse than 1-GPU TP=1 agg's 0.52)
- **0.08 RPS per GPU** (worst per-GPU efficiency of any config tested)

**Conclusion: the dynamo encoder-decoder hand-off has a per-request overhead floor that dominates throughput on this stack.** No amount of GPU adding fixes it.

The configurations that would actually let disagg win on Qwen3-VL-32B:

1. **Multiple smaller PD workers in DP** — e.g., 1 encoder + 4 TP=1 PD workers, where each PD only needs to handle ~25% of requests so the per-request hand-off overhead can be absorbed by parallelism. **Not yet tested.**

2. **Pipeline the hand-off** — overlap encoder of req(N+1) with NIXL transfer of req(N) with prefill of req(N-1). May or may not be supported in current dynamo.

3. **Cross-host disagg** — encoder on machine A, PD on machine B. Then the per-request hand-off cost becomes meaningful (vs. compute) and parallelism actually helps. Same-host setup makes hand-off look expensive vs. trivially-fast same-GPU compute.

4. **Smaller LLM**: tested separately on 8B, same conclusion (disagg loses to TP=1 agg at every rate).

---

## Time spent

- Server startup (TP=2 PD + encoder): ~4 min
- Bench sweep: 5 rates × ~7-13 min each = ~40 min
- Total: ~44 min

---

## Files

- Server script: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2.sh`
- Bench script: `/hongming/dynamo/test_sglang_32b_pd_tp2_1080p_np64_over_rates.sh`
- Orchestrator: `/hongming/dynamo/run_pd_tp2.sh`
- Results: `/hongming/res8_pd_tp2/h200_disagg_pdtp2_32b_image8_1080p_np64/test_sglang_multi_rates_1080p_20260523_064107/`
- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pd_tp2_results.md`

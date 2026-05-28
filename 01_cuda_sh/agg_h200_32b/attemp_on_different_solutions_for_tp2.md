# Attempts to fix TP=2 multimodal regression: what worked, what didn't

**Context:** Earlier diagnosis (`why_tp2_worse_than_tp1_reason.md`) identified the encoder + image preprocessing path as the dominant TP=2 regression source on this hardware (no NVLink, PCIe-only). This document records the experiments to validate and address that diagnosis.

**Workload (all runs):** Qwen3-VL-32B-Instruct-FP8, aggregated EPD, rate=1 req/s, 64 prompts, 8×1920×1080 images per request.
**GPUs:** 4,5 (same NUMA, `NODE` connection, no NVLink).

---

## Outcomes of three fix attempts

| Run | Flags added | Result | Status |
|---|---|---|---|
| Attempt 1 | `--tokenizer-worker-num 2` (+ all 3 MM flags) | dynamo.sglang integration bug: `MultiTokenizerRouter has no generate_request` — every request errors out at the routing layer | ❌ Crashed every request |
| Attempt 2 | `--mm-enable-dp-encoder` + `--keep-mm-feature-on-device` + `--enable-broadcast-mm-inputs-process` (`mem-fraction-static=0.80`) | OOM mid-warmup on rank 0 (memory grows asymmetrically: 141 GB on GPU 4, 118 GB on GPU 5) | ❌ Crashed during warmup |
| Attempt 3 | `--mm-enable-dp-encoder` + `--enable-broadcast-mm-inputs-process` (`mem-fraction-static=0.88`) | **0.29 req/s, 120 s TTFT** vs baseline TP=2 NODE 0.38 req/s, 60 s TTFT | ⚠️ Completed but **worse** |

---

## Four-way comparison (rate=1, 64 prompts, 1080p×8)

| Metric | TP=1 | TP=2 SYS | TP=2 NODE (baseline) | **TP=2 NODE + dp-encoder + broadcast** |
|---|---:|---:|---:|---:|
| Bench duration | 152.6 s | 168.4 s | 169.5 s | **218.5 s** |
| Request throughput | 0.42 req/s | 0.38 | 0.38 | **0.29 req/s** |
| Mean TTFT | 50.2 s | 61.0 s | 60.4 s | **119.9 s** (≈2× worse!) |
| Median TTFT | 56.0 s | 66.7 s | 65.8 s | **125.4 s** |
| P99 TTFT | 84.9 s | 101.8 s | 102.9 s | **152.0 s** |
| Mean E2E | 93.2 s | 107.4 s | 107.3 s | **160.9 s** |
| Mean TPOT | 622 ms | 650 ms | 671 ms | 564 ms |
| Concurrency | 39.1 | 40.8 | 40.5 | **47.1** (queue piled up further) |
| Peak output tput | 592 | 785 | 844 | 817 tok/s |
| **Per-chunk model.forward (median)** | 0.737 s | 0.665 s | 0.684 s | **0.680 s** (essentially same) |
| **Encoder+prep gap (median)** | 1.185 s | 1.516 s | 1.474 s | **1.234 s** ✅ (improved!) |
| **Wall-clock prefill share** | 60 % | 91 % | 91 % | **39 %** |
| **Wall-clock decode share** | 11 % | 9 % | 9 % | **7 %** |
| **Wall-clock unaccounted** | 29 % | 0 % | 0 % | **54 %** ⚠️ |

---

## What actually happened

The flags **DID fix the encoder bottleneck** I diagnosed — encoder+prep gap dropped from **1.474 s → 1.234 s** (−16 %), exactly as predicted. **The diagnosis was correct.**

But the flags introduced a **new, larger bottleneck**: a 54 % "unaccounted" wall-clock window — versus 0 % in the working TP=2 NODE baseline.

The most likely culprit: **`--enable-broadcast-mm-inputs-process`**. It does what its name says — broadcasts MM input tensors across TP ranks. With multi-MB per-image features traveling over PCIe (no NVLink on this box), this becomes a serial bottleneck that blocks the scheduler before each request enters prefill. The scheduler ends up waiting on NCCL broadcasts off the critical path of `report_*_stats`, so the time falls into the "unaccounted" bucket.

We've replaced one type of stall (encoder duplicated on each rank: ~1.5 s/req) with a worse stall (broadcasting input tensors over PCIe: effectively ~3+ s/req of serialized waiting).

---

## Why the OOM in Attempt 2

`--keep-mm-feature-on-device` keeps multimodal feature tensors on GPU between encode and prefill, avoiding host↔device round-trips. But with `--mm-enable-dp-encoder` also on:

- The encoder DP-splits work across the 2 ranks (rank 0 processes images 1-4, rank 1 processes images 5-8)
- Kept-on-device features then need to be **gathered** to whichever rank does prefill
- During this gather, **rank 0 accumulates significantly more in-flight memory than rank 1**
- Result: rank 0 hits 141 GB out of 143 GB, OOMs

Even reducing `mem-fraction-static` from 0.88 → 0.80 wasn't enough headroom — the per-request MM features for 8×1080p images are large (>5 GB) and pile up under load.

---

## Why Attempt 1 failed (config-only bug)

`--tokenizer-worker-num 2` enables SGLang's `MultiTokenizerRouter` for parallel image preprocessing. But the dynamo.sglang glue code calls `router.generate_request(...)` directly, and `MultiTokenizerRouter` doesn't expose this method — only `TokenizerManager` does. This is a dynamo↔sglang integration gap, not a fundamental design issue. Every request hit `AttributeError`.

Reproduce: `--tokenizer-worker-num 2` + any inference request → `AttributeError: 'MultiTokenizerRouter' object has no attribute 'generate_request'`.

---

## Real lesson

**On hardware without NVLink, any TP-related cross-rank data movement is expensive.**

- The dp-encoder split saves encoder compute (~1.5 s saved)
- But the broadcast required to feed it (and to gather results) costs more (~3+ s lost)
- Same for `keep-mm-feature-on-device`: it skipped a host↔device round-trip but caused asymmetric memory growth and OOM

The original diagnosis stands, but **the fix doesn't help on this box** because every "TP optimization" still uses the same slow PCIe link. The PCIe is the binder, regardless of whether you're moving:
- LLM forward all-reduce traffic (TP=2 baseline)
- ViT encoder duplicated work (TP=2 baseline)
- Broadcast MM input tensors (`--enable-broadcast-mm-inputs-process`)
- Gathered MM features kept on device (`--keep-mm-feature-on-device`)

---

## Honest verdict

Despite confirming the bottleneck location, **none of the SGLang MM flags improve TP=2 throughput on this hardware** for this multimodal workload. The diagnosis was right, the fix was wrong because the fix uses the same slow PCIe link.

Best result so far for this workload remains **TP=1** at **0.42 req/s, 50 s TTFT**.

---

## What would actually work

1. **Stay with TP=1** for this workload (0.42 req/s, 50 s TTFT, 9.5 k tok/s prefill).
2. **Run two TP=1 workers in data-parallel** behind the KV router — should give ~0.84 req/s aggregate with no cross-rank PCIe traffic at all. (Untested as of this writing.)
3. **Disaggregated EPD** (separate encoder/prefill/decode workers on different GPUs, communicating only via NIXL/RDMA between phases) — bypasses the per-step NCCL barrier entirely.
4. The only way TP=2 wins on this workload would be **adding NVLink/NVSwitch** — hardware change, not config.

---

## Files in this experiment series

- `time_breakdown_for_agg_tp1_1080p_8image.md` — TP=1 baseline analysis
- `time_breakdown_for_agg_tp2_1080p_8image.md` — TP=2 SYS (GPUs 3,4) analysis
- `time_breakdown_for_agg_tp2_gpu45_1080p_8image.md` — TP=2 NODE (GPUs 4,5) analysis
- `why_tp2_worse_than_tp1_reason.md` — root-cause analysis (encoder + prep)
- **`attemp_on_different_solutions_for_tp2.md`** — this document (attempted fixes and outcomes)

## Bench result directories

- TP=1: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_211545/`
- TP=2 SYS: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_222206/`
- TP=2 NODE baseline: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_232131/`
- TP=2 NODE + flags (Attempt 3): `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_003139/`

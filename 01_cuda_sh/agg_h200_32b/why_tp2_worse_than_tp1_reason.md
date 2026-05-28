# Why TP=2 is worse than TP=1 on this aggregated EPD setup

**System:** Qwen3-VL-32B-Instruct-FP8, aggregated EPD, 8×1080p images per request, rate=1 req/s, 64 prompts.
**Bench script:** `test_sglang_mult_rates_32b_1080p_np_over_rates.sh`
**Three runs analyzed:** TP=1 (1 GPU), TP=2 across SYS (GPUs 3,4), TP=2 across NODE (GPUs 4,5).

---

## Correction to earlier analyses

My earlier reports (`time_breakdown_for_agg_tp2_1080p_8image.md` and `time_breakdown_for_agg_tp2_gpu45_1080p_8image.md`) said:
> "TP=2 prefill chunk throughput dropped 15-18 %, comm-bound on PCIe all-reduce."

**That was wrong.** I was reading the misleading `input throughput (token/s)` field from `report_prefill_stats`, which is amortized over mixed batches (full chunks + small flushes) and includes encoder/prep time on the side.

Re-deriving the time per phase **directly from log timestamps** tells a different story:

---

## Per-chunk model.forward time (consecutive 8192-token prefill steps)

| Run | n | median | mean |
|---|---:|---:|---:|
| TP=1 (1 GPU) | 68 | **0.737 s** | 0.875 s |
| TP=2 SYS (GPUs 3,4) | 61 | **0.665 s** | 0.867 s |
| TP=2 NODE (GPUs 4,5) | 63 | **0.684 s** | 0.905 s |

→ TP=2 model.forward on an 8192-token chunk is **~7-10 % faster** than TP=1 — exactly what you'd expect from tensor parallelism on a 32B model.

## Per-request encoder + image-prep gap (gap from end-of-flush → start of next request's first 8192 chunk)

| Run | n | median | mean |
|---|---:|---:|---:|
| TP=1 (1 GPU) | 44 | **1.185 s** | 1.123 s |
| TP=2 SYS (GPUs 3,4) | 40 | **1.516 s** | 1.402 s |
| TP=2 NODE (GPUs 4,5) | 38 | **1.474 s** | 1.351 s |

→ Encoder + image-prep time is **+24-28 % SLOWER** on TP=2.

## Per-request total budget (single-stream, no pipelining)

```
TP=1:      2 × 0.737 + 1.185 = 2.659 s   →  predicted 0.376 req/s
TP=2 SYS:  2 × 0.665 + 1.516 = 2.846 s   →  predicted 0.351 req/s
TP=2 NODE: 2 × 0.684 + 1.474 = 2.842 s   →  predicted 0.352 req/s
```

Observed: 0.42 / 0.38 / 0.38 req/s — same ranking, magnitudes match within ~10 %. The bench-measured regression of ~10 % is fully explained by the encoder+prep gap getting slower under TP=2.

---

## So the actual story is:

**TP=2 makes the LLM forward pass ~10 % faster, but slows the multimodal preprocessing path by ~25-30 %, and the latter dominates because per-request prefill is only 2 chunks (~1.4 s) but encoder+prep is also ~1.4-1.5 s.**

The math:
- LLM saved: 2 × (0.737 − 0.685) = 0.10 s/req
- Encoder lost: 1.47 − 1.19 = 0.28 s/req
- Net: −0.18 s/req → ~7 % regression. Bench measures 10 %; rest is queueing/jitter.

---

## Why does encoder+prep get slower under TP=2?

Worker config (from `report_prefill_stats` ServerArgs dump):
```
enable_broadcast_mm_inputs_process = False   ← key
mm_enable_dp_encoder              = False    ← key
keep_mm_feature_on_device         = False    ← key
tokenizer_worker_num              = 1
disable_overlap_schedule          = False
mm_attention_backend              = None
```

In the SGLang multimodal path with these settings:

1. **The vision encoder runs duplicated on every TP rank** (`mm_enable_dp_encoder=False`). Each rank does the full ViT pass on all 8 images. Per-GPU work is the same as TP=1, but TP=2 adds NCCL barriers around it without accelerating it.

2. **MM features are NOT kept on device** (`keep_mm_feature_on_device=False`). After the encoder, multimodal embeddings get moved off-GPU and re-uploaded for the language model. With TP=2 this happens on each rank, doubling host↔device traffic and adding a sync point.

3. **MM input broadcasting is disabled** (`enable_broadcast_mm_inputs_process=False`). Pixel tensors and image preprocessing happen separately per rank — image preprocessing is single-threaded (`tokenizer_worker_num=1`), so two ranks waiting on one preprocessing thread serialize on the CPU side.

4. **NCCL synchronization at the start of every step**. Even small operations (e.g., the 48-token "small flush" events) require a barrier. In TP=2 the small flushes themselves are basically free (~0.004 s between events), but the *gap before the next request begins* is dominated by encoder+prep — and that pipeline is now waiting on duplicated/serialized work.

5. **PCIe contention**. Without NVLink, MM feature transfers and TP all-reduces share the same PCIe lanes.

---

## The smoking gun in the timeline

TP=2 NODE log around 23:25 (one full request cycle):

```
23:25:01.271  P new=8192 run=24 q=31              ← request A's first chunk
23:25:01.950  P new=8192 run=24 q=31    Δ=0.680s  ← request A's second chunk
23:25:01.954  P new=  80 run=24 q=31    Δ=0.004s  ← flush
23:25:03.661  P new=8192 run=24 q=30    Δ=1.707s  ← request B's first chunk (1.7 s gap!)
23:25:04.343  P new=8192 run=24 q=30    Δ=0.682s  ← request B's second chunk
```

Per-request wall-clock: 0.68 + 0.68 + 0.004 + **1.71** = 3.07 s. The 1.71 s gap is encoder + image preprocessing + cross-rank sync for the *next* request entering prefill.

Equivalent TP=1 timeline at 21:18 shows the inter-request gap at ~1.2 s — about 70 % of TP=2's value.

---

## So WHY TP=2 is worse on this workload:

**The multimodal preprocessing pipeline does not scale with TP, and gets penalized by NCCL synchronization and host↔device round-trips on every cross-rank step.**

Contributing factors, in priority of contribution to the regression:

1. **ViT encoder is duplicated, not parallelized.** ~1 s/req of work, no speedup from TP, plus added NCCL barrier overhead.
2. **MM features moved off-device** require host↔device copies per rank, then a sync. Not pipelined with prefill chunks.
3. **Image preprocessing is CPU-bound and single-threaded** (`tokenizer_worker_num=1`). Two ranks waiting on one CPU worker create back-pressure.
4. **PCIe is shared** between MM feature transfers and TP all-reduces. They contend.

This is a different bottleneck from "PCIe all-reduce on the LLM forward pass". The LLM forward all-reduce is fine — TP=2 actually speeds the LLM up modestly. The 10-20 % E2E regression comes from the **encoder/prep stall**, not the LLM forward pass.

---

## Things that would actually help TP=2 here

In SGLang there are explicit knobs for this:

1. **`--mm-enable-dp-encoder`** — split the ViT encoder across ranks (data-parallel by image). With 8 images and TP=2, should ~halve encoder time.
2. **`--keep-mm-feature-on-device`** — keep MM features on GPU between encode and prefill, avoid host↔device round-trip.
3. **`--enable-broadcast-mm-inputs-process`** — do MM preprocess once on rank 0, broadcast features to other ranks. Avoids duplicated work.
4. **Increase `--tokenizer-worker-num`** to 2-4 so image preprocessing isn't single-threaded.

Or — much simpler and likely better — run **two TP=1 workers behind the KV router** (data-parallel serving). That gives ~2× request throughput with no NCCL/encoder coordination at all, and is well-suited to this workload where each request is large and independent.

---

## Suggested validation experiment

The cheapest single change to test this hypothesis is to flip **`--mm-enable-dp-encoder`** (if available on this SGLang version) plus **`--keep-mm-feature-on-device`** and re-run the same bench at TP=2 on GPUs 4,5.

Predicted outcome if the diagnosis is correct:
- Encoder+prep gap drops from ~1.47 s → ~0.7-0.9 s
- Per-request budget drops from ~2.84 s → ~2.0-2.2 s
- Bench req/s rises from 0.38 → 0.46-0.50 (now beating TP=1)
- Mean TTFT drops from 60 s → ~35-40 s

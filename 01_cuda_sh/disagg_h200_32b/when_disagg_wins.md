# When Does Disagg (1 H200 encoder + 1 H200 PD) Beat Aggregated (1 H200 EPD)?

This document explores **what workload configurations can structurally favor 2-GPU same-host disagg over 1-GPU aggregated EPD** for Qwen3-VL-32B-FP8.

**Context:** Through extensive testing (`disagg_all_rates_results.md`, `deep_analysis_disagg_worse_h200.md`, `disagg_improvements_attempts.md`, `bottleneck_analysis.md`), our 8×1080p / np=64 / `output_len=256` workload found:
- TP=1 agg: 0.52 RPS, 32 s TTFT
- Disagg (best, NIXL_READ + tuning): 0.23 RPS, 155 s TTFT

So for **this** workload, disagg loses. But that's one workload. What workloads would actually favor disagg?

---

## Structural advantages of disagg

When you spend 2 GPUs instead of 1, what does disagg structurally give you?

| Resource | TP=1 agg (1 GPU) | Disagg (2 GPUs) |
|---|---|---|
| Total GPU compute | 1 H200 | 2 H200 |
| Total GPU memory | 143 GB | 286 GB combined |
| KV cache budget on PD | ~136 GB shared with model+activations+ViT | ~136 GB on PD only |
| ViT compute capacity | shared with LLM forward | dedicated GPU 4 |
| LLM compute capacity | shared with ViT | dedicated GPU 5 |
| ViT model weights | resident on the 1 GPU | resident on encoder GPU |
| LLM model weights | resident on the 1 GPU | resident on PD GPU |
| Encoder cache (`enable-mm-global-cache`) | size-limited by LLM coexistence | full encoder GPU (~140 GB) |
| Compute schedules | 1 unified scheduler | 2 independent schedulers |

Concrete advantages disagg gives you:

1. **2× total compute** when both GPUs are busy simultaneously
2. **2× total GPU memory** (286 GB combined)
3. **Dedicated encoder cache** (encoder GPU is mostly empty in our tests — ~140 GB free for caching MM features)
4. **No prefill-blocks-decode contention from ViT side** — when LLM is decoding, ViT can encode the next batch
5. **Decoupled schedulers** — encoder backlog doesn't block LLM; LLM queue doesn't block encoder

---

## Structural costs of disagg

| Cost | Our measurement | Notes |
|---|---:|---|
| Per-request NIXL handoff round-trip | ~1-2 s | NIXL_READ over NV18 cuda_ipc, fast |
| Embedding-integration small batches on PD | ~10-30 s within forward (~46% of prefill events) | SGLang `_generate_aggregated` path with `precomputed_embeddings`; 53/122 prefill events were 16-112 token batches at <200 tok/s |
| Encoder GPU sitting idle | up to ~99% idle (1.8 GB / 143 GB) | When ViT is not the bottleneck |
| Two-domain backpressure complexity | Caused 15-25 failures with NIXL_WRITE | NIXL_READ fixes this |
| Frontend/scheduler overhead | ~ms | Negligible |

**Net for our workload:** disagg adds ~30-50% per-request overhead while underutilizing the encoder GPU. Hence it loses.

---

## When does disagg structurally win? — Candidate workloads

### Candidate A — Heavy ViT compute (more/larger images)

**Idea:** If ViT compute is comparable to LLM forward time, then dedicating one GPU per phase makes sense.

- **Need:** Many more images per request (e.g., 32 × 1080p) or much larger images (8 × 4K)
- **Risk:** ViT also gets slower per request — encoder GPU becomes the new bottleneck
- **Expected:** Marginal win. Our 8×1080p already has heavy ViT (16k vision tokens). Pushing harder mostly slows the encoder, not necessarily helps disagg.
- **Verdict:** Not the answer.

### Candidate B — Long-output decode-heavy workload

**Idea:** When decode dominates total time, disagg's PD-only-LLM GPU can hit higher decode throughput because there's no ViT contention.

- **Need:** Larger `random-output-len` (e.g., 2048 or 4096 instead of 256), keep input modest
- **Hint from existing data:** at rate=1.0 with 4 images, disagg already beat TP=1 agg on TPOT (703 ms vs 1,069 ms). With longer outputs, this advantage grows.
- **Expected:** Small-to-moderate disagg win on E2E (~10-30%) due to cleaner decode
- **Effort:** Trivial (change one bench arg)
- **Verdict:** Plausible win, easy to test

### Candidate C — Cache-hit-friendly workload (image reuse)

**Idea:** With `--enable-mm-global-cache=True` and repeated images, the encoder can hit cache and skip ViT compute entirely.

- **Need:** Bench where ~50-90% of images are reused across requests (real-world RAG/agent traffic often is)
- **Mechanism:** Disagg's encoder GPU has ~140 GB free for caching, vs agg sharing memory with LLM
- **Expected:** Small RPS win for disagg, larger TTFT win (encoder time drops to ~0 on cache hits)
- **Effort:** Easy (use fixed image set instead of random)
- **Verdict:** Realistic and testable. Could easily flip the comparison.

### Candidate D — Multi-question-per-image workload

**Idea:** Same image, multiple questions = encoder runs once, PD answers N questions.

- **Need:** Custom bench: 8 images, ask 8 different questions of each = 64 requests sharing 8 image groups
- **Mechanism:** Encoder runs ViT 8 times across 64 requests → effective encoder time per request ≈ 0.15 s instead of 1.2 s
- **Expected:** Large win for disagg if implemented. **But** also helps agg (just less) because agg also has encoder cache.
- **Verdict:** Win likely but not unique to disagg.

### Candidate E — Mixed text+image traffic

**Idea:** When some requests are text-only, disagg can route them straight to PD without going through encoder.

- **Need:** Mixed bench (e.g., 30% text-only, 70% image)
- **Mechanism:** Disagg PD path for text-only is faster (no NIXL handoff). Agg always processes through the same path.
- **Caveat:** SGLang's agg also skips ViT for text-only requests, so the gap might be small
- **Expected:** Small win (~10-20%) on mixed traffic
- **Verdict:** Maybe testable, but small expected gain

### Candidate F — Bursty arrival pattern

**Idea:** Disagg's two queues smooth out bursts — encoder can absorb image bursts even while PD is saturated.

- **Need:** Bursty load (e.g., 8 requests in 100 ms then idle 5 s)
- **Mechanism:** During recovery from burst, encoder is already done, so TTFT recovers faster
- **Expected:** Small TTFT-tail win, no sustained throughput win
- **Verdict:** Niche

### Candidate G — 1 encoder + N PDs

**Idea:** Production setup with 1 encoder shared across multiple LLM workers.

- **Need:** ≥3 GPUs (1 encoder + N PDs), frontend with multi-PD KV router
- **Mechanism:** Encoder is amortized across N downstreams. If encoder utilization was 1.3% with 1 PD, it can serve 30+ PDs before saturating
- **For our PD-bound workload:** doesn't help because LLM PD is the bottleneck. Total RPS = N × per-PD-RPS. Each additional PD adds 1 GPU → total gain is just additive (no super-linear advantage).
- **For ViT-bound workload:** would win significantly because 1 encoder GPU serves N LLM workers without N replicated encoders
- **Verdict:** Win conditional on ViT being the bottleneck. For our 32B FP8 workload, doesn't help.

### Candidate H — Encoder larger than LLM

**Idea:** If you have a 7B vision model + 1B language model, ViT dominates. Disagg makes total sense.

- **Need:** Different model (not testable with Qwen3-VL-32B-FP8)
- **Verdict:** Architectural win in principle but out of scope for this exact hardware/model

### Candidate I — Light per-request workload, very high concurrency

**Idea:** Smaller images, fewer per request → very fast per-request work.

- **Mechanism:** Disagg's per-request handoff overhead is FIXED (~1-2 s). If per-request work drops below ~2 s, the handoff dominates.
- **Expected:** Disagg loses by larger margin
- **Verdict:** **Avoid this regime.** Worse for disagg.

---

## Ranking — most-realistic-disagg-win-first

Based on workload characteristics realistic for Qwen3-VL-32B-FP8 production use:

| # | Candidate | Expected disagg outcome | Effort to test | Production realism |
|---:|---|---|---|---|
| 1 | **C: Cache-hit workload** | **Disagg wins ~5-15% RPS, larger TTFT win** | Easy (fixed image set) | High (real RAG/agent has reuse) |
| 2 | **B: Long-output decode-heavy** | **Disagg wins ~10-30% E2E, much better TPOT** | Trivial (change `output_len`) | High (chat/reasoning use cases) |
| 3 | **D: Multi-question per image** | **Disagg wins moderately** | Medium (custom bench) | Medium (multi-turn QA) |
| 4 | **E: Mixed text+image traffic** | **Disagg wins ~10-20%** | Medium | High (real chat workloads) |
| 5 | **F: Bursty arrival** | Tail-latency win, no throughput win | Medium | Medium |
| 6 | **A: Bigger ViT load** | Marginal | Easy | Less common |
| 7 | **G: 1 enc + N PDs (3+ GPUs)** | Doesn't help PD-bound workload; helps ViT-bound | Medium (multi-worker setup) | High but for different model/workload |
| 8 | I: Light per-request, high QPS | **Disagg loses MORE** — avoid | — | — |

---

## Honest recommendation

**For 1 H200 encoder + 1 H200 PD setup with a 32B+ LLM, the cleanest "disagg wins" config is workload C (cache-hit-friendly).** The reasoning:

- Encoder amortizes effectively to ~0 on cache hit
- Disagg PD's KV/compute is dedicated to LLM, no ViT contention
- The ~1-2 s NIXL handoff overhead is constant; everything else collapses
- Real production workloads (RAG with reference images, agent loops with screenshots) have heavy reuse

Predicted result for cache-hit workload at rate=1.0:
- Disagg: ~0.55-0.70 RPS (vs current 0.23 with random images)
- TP=1 agg: ~0.6-0.7 RPS (smaller bump because agg's encoder cache competes with LLM for memory)
- **Disagg might tie or marginally win**

For workload **B (long-output decode-heavy)**: easier to test, smaller expected win on RPS but bigger win on TPOT/E2E.

---

## When disagg structurally must win (regardless of tuning)

These setups make disagg the only correct choice:

1. **Encoder must run on different hardware** (e.g., encoder on cheaper GPU, LLM on H200) — economics
2. **Multi-tenant fleet sharing 1 encoder pool across many LLM workers** — utilization
3. **Memory-isolated workloads** where encoder ViT weights conflict with LLM weights on a single GPU
4. **Cross-host setups** where physical separation is required
5. **Workloads with ViT >> LLM compute** (vision-heavy models, e.g., Llava-OneVision with large ViT)

For our exact case (Qwen3-VL-32B-FP8 + 8×1080p + 1 encoder + 1 PD on same host), **disagg's win is conditional on workload reuse, decode-dominance, or text/image mix**. With purely random images and balanced prefill+decode, disagg loses.

---

## Specific bench parameters to test each candidate

### Test C — Cache-hit workload

Modify bench script:
```bash
# Change image-content from 'random' to 'fixed':
#   --image-count 8 --image-resolution 1920x1080 --image-content random
#   ↓
#   --image-content (some fixed local path, OR pre-download 8 jpgs and reuse)
```
Or simpler: hack the bench to use the same `seed=0` for all requests so the same random images repeat.

Run on:
- TP=1 agg (current `start_h200_aggregate_epd_server_32b_tp1.sh`)
- Disagg (current best NIXL_READ + A+B config)

Compare: RPS, TTFT, E2E.

### Test B — Long-output decode-heavy

```bash
# Current: --random-input-len 128 --random-output-len 256
# Change to: --random-input-len 128 --random-output-len 2048
```
Easy single-line change. Run rate=0.25 (lower than 1.0 because each request takes 8× longer).

### Test D — Multi-question per image

Custom bench, see `sglang.bench_serving` source. Likely needs forking or scripting:
- Generate 8 image groups
- Generate 8 questions per group = 64 prompts
- Each prompt references one of the 8 image groups
- All 64 requests sent at rate=1.0

### Test E — Mixed traffic

Custom bench: 30% of requests have `image-count=0` (text-only), 70% have `image-count=8`. Run rate=1.0.

### Test G — 1 encoder + 2 PDs (3 GPUs)

Modify start script to launch 2 PD workers with different `KV_EVENT_PORT`s and `SIDE_CHANNEL_PORT`s, plus 1 encoder. Frontend KV router fans across both PDs. Run rate=2.0 to drive both PDs.

---

## Files

- `01_cuda_sh/disagg_h200_32b/disagg_all_rates_results.md` — original sweep
- `01_cuda_sh/disagg_h200_32b/deep_analysis_disagg_worse_h200.md` — first-round bottleneck analysis (concluded "architectural")
- `01_cuda_sh/disagg_h200_32b/disagg_improvements_attempts.md` — second-round investigation; found bugs and tunables
- `01_cuda_sh/disagg_h200_32b/bottleneck_analysis.md` — current bottleneck breakdown
- This document: `01_cuda_sh/disagg_h200_32b/when_disagg_wins.md`

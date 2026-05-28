# Where is the bottleneck? Analysis of Vision-Encoder vs LLM-Prefill at 4K

**Context:** After running the 8-images @ 4K three-way comparison (TP=1 agg, TP=2 agg, disagg) on Qwen3-VL-32B-Instruct-FP8, the user asked whether vision encoding is the bottleneck. The answer is no — and exploring why uncovers fundamental scaling laws that explain why disagg cannot win on this model regardless of how we tune the workload.

This document captures the question/answer thread, with the actual measurements and math.

---

## Q1: For the image8 @ 4K case, is vision encoding the bottleneck now?

### Short answer: NO. LLM prefill is the bottleneck, not vision encoding.

### Evidence from disagg PD-worker logs (8 imgs @ 4K, rate=1.0)

```
Prefill batch, #new-token: 16384, input throughput (token/s): 2,200-5,500
Decode batch, #token: 654,512, gen throughput (token/s): 4-6
```

Each request has **~65,500 visual tokens** (8 images × ~8,200 visual tokens per 4K image). Looking at the actual workload breakdown:

- Prefill is processed in **chunked-prefill chunks of 16,384 visual tokens at a time**
- Each chunk takes ~3-5 seconds (~3,000-5,500 tok/s)
- A single 65K-token request needs ~4 chunks = **12-20 seconds of pure LLM prefill compute**
- Plus a queue (`queue-req: 8-10`) is constantly backed up
- During prefill, decode is starved (`gen throughput: 4-6 tok/s` — should be ~150)

### Where the time goes per request (~600 s E2E at saturation)

| Stage | Time | Bottleneck |
|---|---:|---|
| Encoder ViT (8 × 4K) | ~5-10 s | GPU 4 (encoder) |
| Embedding transfer (NIXL over NVLink) | <1 s | NVLink |
| LLM prefill of 65K visual tokens | ~15-20 s of pure compute | **GPU 5 (PD)** |
| Wait in queue (concurrency 24) | ~400-500 s | **GPU 5 (PD) prefill backlog** |
| LLM decode 256 tokens | ~250 s (TPOT 1 s × 256) | **GPU 5 (PD), shared with new prefills** |

**The encoder GPU (GPU 4) is mostly idle in this run.** GPU 4 utilization in our snapshots was bouncing 0%-100%, often 0% — meaning encoder finishes its work fast and waits. GPU 5 (PD) was pegged at 100% almost continuously.

### Why TP=2 wins so cleanly

TP=2 splits the **65K-token LLM prefill** across 2 GPUs:
- 2× attention compute throughput
- 2× HBM bandwidth (for KV writes during prefill)
- Faster chunked-prefill turnaround

Hence TP=2 doubles to 0.12 RPS while disagg stays at 0.04 RPS. Disagg has the same single-GPU LLM prefill bottleneck as TP=1 — but with extra hop overhead.

### What "ViT bottleneck" would actually look like

We'd see:
- GPU 4 (encoder) pegged at 100% continuously
- GPU 5 (PD) showing low utilization, idle waiting for embeddings
- Encoder request queue building up
- Decode throughput high (because prefill finishes fast once it gets embeddings)

**We see the opposite pattern.** Even at 4K with 8 images, ViT is fast and LLM prefill of 65K visual tokens is the dominant cost.

---

## Q2: What configuration can let vision encode become the bottleneck?

### The math

Per request, two compute costs race:

**ViT compute** (encoder): linear in image pixel count, independent of LLM size
- 8 × 4K images ≈ 65K visual tokens → measured ~5-10 s on H200
- Per-token ViT cost is small (~0.1 ms/visual-token), but the visual tokens are large

**LLM prefill** (PD): O(N²) attention over all visual + text tokens, with model-size-dependent constant
- 65K tokens × Qwen3-VL-**32B**-FP8 → measured ~15-20 s at full GPU
- The 32B model is extremely heavy per token

For ViT to be the bottleneck: **ViT_time > LLM_prefill_time**.

### Three levers to flip the ratio

#### 1. Make the LLM smaller (best lever, easiest to test)

LLM prefill compute scales roughly linearly with model parameters. Going from 32B → 7B → 3B drops LLM prefill by ~5× and ~10× respectively. ViT is unchanged (Qwen3-VL all use the same vision encoder).

**Estimated ViT/LLM ratio for 8 × 1080p images:**

| LLM size | ViT time | LLM prefill | ViT/(ViT+LLM) | Disagg likely wins? |
|---|---:|---:|---:|---|
| **Qwen3-VL-32B-FP8** | ~1 s | ~5 s (TP=1) | ~17% | No (we measured this) |
| **Qwen3-VL-7B** | ~1 s | ~1.5 s (TP=1) | ~40% | Plausible |
| **Qwen3-VL-3B** | ~1 s | ~0.7 s (TP=1) | **~60%** | **Yes, likely** |

**Estimated ViT/LLM ratio for 8 × 4K (~65K visual tokens):**

| LLM size | ViT time | LLM prefill | ViT/(ViT+LLM) | Disagg likely wins? |
|---|---:|---:|---:|---|
| Qwen3-VL-32B-FP8 | ~8 s | ~17 s (TP=1) | 32% | No (measured) |
| Qwen3-VL-7B | ~8 s | ~5 s (TP=1) | **~62%** | **Yes** |
| Qwen3-VL-3B | ~8 s | ~2.5 s (TP=1) | **~76%** | **Yes, clearly** |

#### 2. Make the LLM "see less" of the visual tokens

Some VL models compress visual tokens before sending them to the LLM. Qwen3-VL doesn't do this aggressively. Models that do compress (e.g., Pixtral with 2× pixel-shuffle, InternVL2 with downsampling) shift the ratio toward ViT being a bigger relative chunk because LLM sees fewer tokens.

Not a knob you can turn on Qwen3-VL — it's a model-architecture choice.

#### 3. Use a model with a heavier ViT

Qwen3-VL uses a relatively standard ViT (~600M params for the visual encoder). Some models have much bigger ViTs:
- InternVL2 with 6B-parameter ViT (~10× heavier)
- Pixtral with 400M ViT but no token compression (so feeds full token stream)

Also a model choice, not a config knob.

### Locally available models

```
Qwen3-VL-32B-Instruct-FP8           (current — too LLM-heavy)
Qwen3-VL-32B-Instruct
Qwen3-VL-32B-Thinking-FP8
Qwen3-VL-235B-A22B-Instruct-FP8     (MoE — 22B activated, even more LLM-heavy)
```

**No locally-available smaller VL model.** To make ViT the bottleneck, we'd need to download Qwen3-VL-7B or Qwen3-VL-3B (~14 GB and ~6 GB respectively).

---

## Q3: Increasing image number like 32, 64 — still not working?

### What "more images" does to the ratio

Both ViT time and LLM prefill time scale with total visual tokens, but at different rates:

- **ViT compute**: O(N_visual_tokens) — linear
- **LLM prefill**: O(N²) — quadratic in attention, linear in feedforward

So as image count grows, **LLM prefill grows FASTER than ViT** because of the N² attention term. **More images makes LLM prefill MORE dominant, not less.**

This is the opposite of what helps disagg.

### Concrete numbers for Qwen3-VL-32B-FP8

| Workload | Visual tokens | ViT time | LLM prefill (TP=1) | ViT/(ViT+LLM) |
|---|---:|---:|---:|---:|
| 8 × 1080p (measured) | ~16K | ~1 s | ~5 s | 17% |
| 8 × 4K (measured) | ~65K | ~8 s | ~17 s | 32% |
| 16 × 1080p (measured) | ~32K | ~2 s | ~12 s | 14% |
| **32 × 1080p** (extrapolated) | ~64K | ~4 s | ~30-40 s | **~10%** |
| **64 × 1080p** (extrapolated) | ~128K | ~8 s | ~80-150 s | **~5-10%** |

ViT/total **shrinks** as you add more images. The LLM prefill becomes more dominant, not less.

### The other problem: memory + body limits

Even if ratios worked in your favor, you'd hit hard walls:

| Config | Body size | KV-cache for prefill | Status |
|---|---:|---:|---|
| 32 × 1080p | ~67 MB | ~30 GB | Body fits with `DYN_HTTP_BODY_LIMIT_MB=256`, but KV-cache alone may OOM at TP=1 |
| 64 × 1080p | ~134 MB | ~60 GB | KV-cache > GPU mem at TP=1 even with mem_fraction 0.85 |
| 32 × 4K | ~270 MB | enormous | Body limit + KV-cache both fail |

We already saw 16 × 1080p OOM at mem_fraction 0.95 and only barely fit at 0.85.

### Notice: 32 × 1080p ≈ 8 × 4K in token count

```
32 imgs × 1080p ≈ 64K visual tokens  (same as 8 imgs × 4K)
```

We already know how 8 × 4K behaves: it's LLM-prefill-bound, not ViT-bound. So 32 × 1080p will be the same.

### Summary

**No — increasing image count to 32 or 64 does NOT make ViT the bottleneck on Qwen3-VL-32B-FP8.** It actually makes the LLM **more** dominant because attention is O(N²).

The fundamental issue is **the LLM is too big relative to the ViT** for this model. The only configurations that make ViT the bottleneck are:

1. **Smaller LLM**: Qwen3-VL-7B or 3B
2. **Heavier ViT**: model with a bigger vision encoder relative to the LLM
3. **Token compression in the model**: a model that downsamples visual tokens before feeding them to the LLM (Qwen3-VL doesn't)

We don't have any of those locally except 32B and 235B-MoE (which is worse).

---

## Q4: Why does smaller LLM cost drop?

### Where LLM compute time comes from

For each token in prefill, the LLM does roughly two things:

#### 1. Attention (per layer)
- Compute Q, K, V projections from input → 3 × `hidden_size²` GEMM
- Attention scores: `Q × K^T` → `seq_len² × hidden_size` ops
- Output projection: 1 × `hidden_size²` GEMM

#### 2. Feedforward / MLP (per layer)
- Up-projection: `hidden_size × intermediate_size` GEMM
- Down-projection: `intermediate_size × hidden_size` GEMM
- For a typical model, `intermediate_size ≈ 3-4 × hidden_size`

**Total per token per layer ≈ 12 × hidden_size² FLOPs** (plus the attention quadratic term).

You sum this over **all layers** to get total cost.

### Now compare model sizes

Approximate Qwen3-VL configurations:

| Model | Layers | Hidden size | Per-token FLOPs (linear part) | Activated params |
|---|---:|---:|---:|---:|
| Qwen3-VL-32B | 64 | 5,120 | 12 × 64 × 5120² ≈ **20 GFLOPs/token** | 32B |
| Qwen3-VL-7B | 28 | 3,584 | 12 × 28 × 3584² ≈ **4.3 GFLOPs/token** | 7B |
| Qwen3-VL-3B | 36 | 2,048 | 12 × 36 × 2048² ≈ **1.8 GFLOPs/token** | 3B |

**The key relationship: per-token FLOPs scale roughly with parameter count** (since each parameter shows up in a matmul once per token).

### Concrete prefill time predictions

Assume H200 delivers ~1,500 TFLOPs/s sustained on FP8 dense GEMMs (about half of the ~3 PFLOP/s spec peak — a realistic working number).

For 65K visual tokens (the 8 × 4K case):

| Model | FLOPs total (linear only) | Time on 1× H200 (linear only) |
|---|---:|---:|
| Qwen3-VL-32B | 65K × 20G = 1.3 PFLOPs | **~0.9 s** |
| Qwen3-VL-7B | 65K × 4.3G = 0.28 PFLOPs | **~0.2 s** |
| Qwen3-VL-3B | 65K × 1.8G = 0.12 PFLOPs | **~0.08 s** |

But we measured **~17 s of LLM prefill** for 32B at 65K tokens, not 0.9 s. Where does the extra 16 s come from?

### The other big term: O(N²) attention

The attention score computation is `seq_len² × hidden_size` per layer:

For 65K tokens × 32B model (64 layers × 5120 hidden):
- Attention FLOPs = 64 × 65K² × 5120 × 2 ≈ **2.8 PFLOPs** (just for attention scores)
- That's ~2× the linear-projection FLOPs

So total prefill ~ 4 PFLOPs → ~3 s on H200 at full speed. Still less than measured 17 s.

The remaining gap (3 s predicted vs. 17 s measured) comes from:
- HBM bandwidth bottleneck (reading 32B FP8 weights = 32 GB per token batch, ~10 ms per layer at 4 TB/s)
- Kernel launch overhead, chunked-prefill scheduling
- Concurrency contention (other requests' decode interleaving)

### Why smaller LLM helps disproportionately

The two big costs both scale with model size:

| Cost | Scales with |
|---|---|
| Linear projection GEMMs | `params` (linear in hidden_size² × layers) |
| Attention quadratic term | `params × N²` |
| HBM weight loading | **`params`** (you must read every weight every forward pass) |

Going from 32B → 7B (4.5× smaller):
- Compute drops 4.5× ✓
- Memory bandwidth pressure drops 4.5× ✓ (this is huge — H200 is often bandwidth-bound)
- KV cache per token drops ~1.4× (KV grows as `2 × hidden_size × layers`)

**ViT compute, in contrast, is invariant.** Qwen3-VL uses the same vision encoder (~600M params, fixed) regardless of which LLM you pair with it. So:

```
LLM_time(32B) ≈ 17 s    →  ViT/(ViT+LLM) = 8/(8+17) = 32%
LLM_time(7B)  ≈ ~5 s    →  ViT/(ViT+LLM) = 8/(8+5)  = 62%   ← ViT now dominates
LLM_time(3B)  ≈ ~2.5 s  →  ViT/(ViT+LLM) = 8/(8+2.5) = 76%  ← strongly ViT-bound
```

### The intuitive summary

**Smaller LLM** = fewer parameters → fewer FLOPs per token → less HBM bandwidth needed per token → **faster prefill**.

**Same ViT** (because we're keeping Qwen3-VL's vision encoder fixed) → **constant ViT time**.

**Ratio shifts toward ViT.** When ViT time exceeds LLM time, ViT becomes the bottleneck — the regime where running it on a separate GPU (disagg) helps, because you can pipeline ViT(req_n+1) with LLM(req_n).

### Caveat

These per-token FLOP estimates are **architectural lower bounds**. Real-world prefill is often 5-10× slower than the FLOP-only prediction because of HBM bandwidth, kernel launch overhead, attention kernel inefficiency at long sequence lengths, and queue/scheduling overhead. So the absolute numbers above are illustrative, not exact. **But the ratios between model sizes hold reliably** — going to a 4.5× smaller model gives ~3-5× faster prefill in practice.

---

## Bottom line for our experiments

1. **Qwen3-VL-32B-FP8 has no workload regime where disagg wins** on this 2-GPU same-host setup. The LLM is too dominant. We've validated this across:
   - 8 × 1080p random
   - 8 × 1080p decode-heavy (rate 0.5 and 2.0)
   - 8 × 1080p cache-hit
   - 8 × 768p
   - 16 × 1080p
   - **8 × 4K (this run)** — most ViT-heavy single-image-count test possible without OOM

2. **More images doesn't help** — it makes LLM prefill MORE dominant (O(N²) attention).

3. **The only path to a disagg-winning regime** is to switch to a smaller LLM:
   - Qwen3-VL-7B (predicted ViT/total ~60% at 8 × 4K) — disagg should win
   - Qwen3-VL-3B (predicted ViT/total ~75% at 8 × 4K) — disagg should clearly win
   - Neither is downloaded locally yet

4. **Or to a model with token compression / heavier ViT** (Pixtral, InternVL2) — also requires downloading.

---

## Files

- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/where_is_bottleneck.md`
- Companion measurements: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/8img_4k_three_way.md`
- Earlier analysis: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/when_vit_is_bottleneck.md` (predicted regimes)
- PD worker logs (4K disagg): `/hongming/dynamo/logs/pd_worker.log`
- Encoder worker logs (4K disagg): `/hongming/dynamo/logs/encoder_worker.log`

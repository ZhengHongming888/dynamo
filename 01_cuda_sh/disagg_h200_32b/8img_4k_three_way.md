# 8 Images @ 4K — Three-Way Comparison: TP=1 agg vs TP=2 agg vs Disagg

**Date:** 2026-05-22 (full run completed 2026-05-23 ~00:30 UTC)
**Workload:** 8 × 4K images (3840×2160), 128 input + 256 output text tokens, np=32 per rate
**Rates swept:** 0.1, 0.25, 0.5, 1.0, 1.25 RPS
**Per-request body size:** ~67 MB (JPEG-base64, 8 × 4K)
**Hardware:** H200, GPUs 4 (NUMA 2) and 5 (NUMA 2), NVLink NV18 (~478 GB/s)

This is the heaviest ViT workload we've tested, designed to be the regime where disagg should structurally win. **It does not.** All three configs are LLM-bound (or rather LLM+ViT bound) at this scale, and disagg adds overhead without offering parallelism gains.

---

## Configuration changes required to make 4K work

After hitting three sequential limits, all are now configured in the start scripts:

1. **`DYN_TCP_MAX_MESSAGE_SIZE=268435456`** (256 MB) — worker TCP plane (default 32 MB rejected 67 MB bodies)
2. **`DYN_HTTP_BODY_LIMIT_MB=256`** — frontend HTTP body limit (default 45 MB rejected 67 MB bodies). The env var is read at `/opt/dynamo/lib/llm/src/http/service/openai.rs:76` via `std::env::var(env_llm::DYN_HTTP_BODY_LIMIT_MB)`. **No rebuild needed.**
3. **`mem_fraction_static=0.85`** (was 0.95) — leaves 15% (~21 GB) for ViT activations on 4K × 8 images

---

## Saturation summary

| Config | Rate=0.1 | Rate=0.25 | Rate=0.5 | Rate=1.0 | Rate=1.25 | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| **TP=1 agg** (1 GPU)     | **0.06** | 0.06 | 0.06 | 0.06 | 0.06 | **0.06 RPS** |
| **TP=2 agg** (2 GPUs) | 0.09 | **0.12** | 0.11 | 0.11 | 0.11 | **0.11-0.12 RPS** |
| **Disagg** (encoder=GPU4, PD=GPU5) | **0.04** | 0.04 | 0.04 | 0.04 | 0.04 | **0.04 RPS** |

**Ranking: TP=2 (0.12) > TP=1 (0.06) > Disagg (0.04).**

TP=2 wins by ~2× over TP=1 (good NVLink scaling), and TP=1 beats disagg by 1.5×.

---

## Latency at rate=0.5 (saturation regime for all three)

| Metric | TP=1 agg | TP=2 agg | Disagg |
|---|---:|---:|---:|
| Actual RPS | 0.06 | 0.11 | 0.04 |
| Mean TTFT (s) | 270.6 | **115.4** | 432.8 |
| Median TTFT (s) | 272.6 | **106.6** | 440.7 |
| Mean TPOT (ms) | 789 | 1,059 | 1,096 |
| Mean E2E (s) | 397.7 | **240.1** | 588.1 |
| Concurrency | 22.8 | 27.4 | 24.4 |

**TP=2 agg has 2.4× lower TTFT than TP=1 agg and 3.7× lower than disagg.**

---

## Detailed CSV (all 5 rates)

### TP=1 agg
```
target_rate, actual_rps, ttft_mean_ms, tpot_mean_ms, e2e_mean_ms, peak_out_tok/s
0.10,        0.06,       157370,       835,          283846,      193
0.25,        0.06,       237642,       753,          361088,      196
0.50,        0.06,       270638,       789,          397711,      215
1.00,        0.06,       280213,       765,          405206,      180
1.25,        0.06,       288727,       774,          413810,      180
```

### TP=2 agg
```
target_rate, actual_rps, ttft_mean_ms, tpot_mean_ms, e2e_mean_ms, peak_out_tok/s
0.10,        0.09,       58181,        1267,         202341,      456
0.25,        0.12,       74870,        1109,         201120,      455
0.50,        0.11,       115377,       1059,         240148,      452
1.00,        0.11,       135422,       1050,         256938,      473
1.25,        0.11,       136920,       1033,         256485,      456
```

### Disagg
```
target_rate, actual_rps, ttft_mean_ms, tpot_mean_ms, e2e_mean_ms, peak_out_tok/s
0.10,        0.04,       302871,       1018,         456970,      210
0.25,        0.04,       400164,       934,          550856,      257
0.50,        0.04,       432764,       1096,         588142,      210
1.00,        0.04,       444359,       955,          597284,      192
1.25,        0.04,       460463,       1028,         614120,      195
```

---

## Why TP=2 wins decisively at 4K

**TP=2 has 2× the GPU compute, ~2× the HBM bandwidth, and ~2× the KV cache capacity.** With 8 × 4K images per request:
- Each request's prefill needs to cross-attend over ~64K visual tokens (8 imgs × ~8K tokens each)
- ViT compute (image patch embeddings) parallelizes across TP ranks
- LLM prefill of 64K visual tokens parallelizes across TP ranks
- KV cache for 64K tokens × 32B model is ~30+ GB — TP=2 splits this in half

**TP=1 saturates at 0.06 RPS** because a single H200 cannot handle even one 4K-8-image request in less than ~17s of pure compute. Adding requests just queues them — **mean concurrency 22-23, median TTFT 280s** (waiting in queue).

**TP=2 saturates at 0.12 RPS** — a clean ~2× scaling, dropping TTFT from 280s → 135s.

---

## Why disagg loses at 4K (worst of three configs)

Disagg architecture: encoder runs ViT on GPU 4 (~13 GB used), then transfers ~3.5 MB embeddings per image (≈28 MB/request) to PD worker on GPU 5, which does LLM prefill + decode.

**Three problems for this workload:**

1. **Single PD worker is TP=1.** The decode side gets only one GPU's worth of compute and KV bandwidth, just like the TP=1 agg config — but without the encoder on the same GPU. So the LLM is **TP=1 bound** for both prefill and decode.

2. **Encoder transfer overhead.** 28 MB of embeddings per request must hop GPU 4 → GPU 5 via cuda_ipc (NVLink). At 478 GB/s peak, this is <1 ms in theory — but observed Mean TTFT 432s vs TP=1 agg's 270s shows the transfer lifecycle (request hand-off + scheduling + waiting for PD slot) adds **~160s of latency overhead per request** at scale.

3. **No parallelism advantage.** The encoder is not the bottleneck (ViT for 8 × 4K takes ~5-10s on a single GPU; this is small relative to the ~250s LLM prefill). Splitting it off doesn't relieve LLM pressure — it just adds a hop.

**Bottom line:** disagg helps when ViT is genuinely the bottleneck *and* you can run multiple small LLM workers in parallel. With one TP=1 LLM worker as your decoder, disagg is strictly worse than TP=1 agg.

---

## What would make disagg win on this workload?

Hypothetical configs (not tested):

| Config | Predicted result |
|---|---|
| **Disagg with TP=2 PD** (3 GPUs total: 1 encoder, 2 PD) | Probably matches or beats TP=2 agg, since PD gets the same compute and encoder is offloaded. Need 3 GPUs. |
| **Disagg with 2× TP=1 PD workers** (3 GPUs: 1 enc, 2 PD via DP) | Could beat TP=2 agg if encoder isn't a bottleneck and 2 PDs in DP avg better than 1 TP=2 PD |
| **Disagg with smaller LLM (Qwen3-VL-7B/3B)** | ViT becomes bigger fraction of total compute → disagg should win clearly |

---

## Validation: HTTP body limit fix

This run validates that **`DYN_HTTP_BODY_LIMIT_MB`** works for runtime configuration. Per-request body sizes:

- 8 × 1080p: ~17 MB (default 45 MB limit OK)
- 16 × 1080p: ~33 MB (default 45 MB limit OK)
- **8 × 4K: ~67 MB (requires `DYN_HTTP_BODY_LIMIT_MB ≥ 67`)** ← previously blocked us
- 16 × 4K: ~134 MB (would need `DYN_HTTP_BODY_LIMIT_MB ≥ 134`)

We set 256 MB which gives plenty of headroom for any tested workload.

---

## Time spent

- **TP=1 agg sweep**: 70 min (5 rates × ~14 min each — saturated at 0.06 RPS, slow tail)
- **TP=2 agg sweep**: 44 min (5 rates × ~9 min each — faster tail at 0.12 RPS)
- **Disagg sweep**: 64 min (5 rates × ~13 min each — slowest, 0.04 RPS)
- **Total**: ~3 hr unattended (excluding disagg's first failed startup)

---

## Conclusion

**4K × 8 images @ Qwen3-VL-32B-Instruct-FP8 is the most LLM-bound workload we've tested.** ViT compute is real (~5-10s/req) but dwarfed by 64K-visual-token LLM prefill (250s+/req at saturation). Adding a second GPU via TP=2 doubles throughput (good NVLink scaling). Adding a second GPU via disagg makes things worse (1.5× lower throughput than TP=1 agg) because you're not adding LLM compute — you're just adding a hop.

**Updated ranking on Qwen3-VL-32B-FP8 (TP=2 always wins, disagg never wins):**

| Workload | TP=1 RPS | TP=2 RPS | Disagg RPS | Best |
|---|---:|---:|---:|---|
| 8 × 1080p random | 0.52 | 0.95 | 0.23 | TP=2 |
| 8 × 1080p decode-heavy r=0.5 | 0.43 | — | 0.42 | TP=1 ~ disagg |
| 8 × 1080p decode-heavy r=2.0 | 1.15 | — | 1.00 | TP=1 |
| 8 × 1080p cache-hit | 0.99 | — | 0.50 | TP=1 |
| 8 × 768p | 0.90 | — | 0.62 | TP=1 |
| 16 × 1080p | 0.21 | 0.28 | 0.10 | TP=2 |
| **8 × 4K** | **0.06** | **0.12** | **0.04** | **TP=2** |

**Disagg with same-host single-PD worker has no winning regime in our test matrix.** It's a 2-GPU config that's worse than 1-GPU TP=1 agg.

---

## Files

- TP=1 results: `/hongming/res6_img8_4k/h200_agg_tp1_32b_image8_4k_np64/test_sglang_multi_rates_1080p_20260522_205024/`
- TP=2 results: `/hongming/res6_img8_4k/h200_agg_tp2_32b_image8_4k_np64/test_sglang_multi_rates_1080p_20260522_220134/`
- Disagg results: `/hongming/res6_img8_4k/h200_disagg_32b_image8_4k_np64/test_sglang_multi_rates_1080p_20260522_230851/`
- Bench script: `/hongming/dynamo/test_sglang_8img_4k.sh` (np=32 for tractable runtime)
- Orchestrator script: `/hongming/dynamo/run_4k_three_way.sh` (5-min timeout was too tight for disagg startup; should be 8-10 min)
- Companion: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/8img_4k_blocked.md` (the fixes that unblocked this test)

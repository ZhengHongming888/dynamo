# Disagg H200 32B investigation — index

**Scope:** `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/`
**Span:** 2026-05-15 → 2026-05-24
**Subject:** When (if ever) does disaggregated multimodal inference beat aggregated EPD on
Qwen3-VL-32B-Instruct-FP8 on H200/B70?
**Short answer:** never on this stack. TP=2 agg dominates everything tested. The closest
disagg gets is the cross-host 4-encoder + GPUDirect-RDMA patch, which matches but does not
beat unpatched 4E and stays well behind same-host TP=2 agg.

This index summarises the 38 docs in this directory (after the 2026-05-24 consolidation —
see [Consolidation notes](#consolidation-notes-2026-05-24) at the bottom).

---

## TL;DR — saturation RPS table (consolidated across all docs)

| Workload (np=64) | TP=1 agg same-host | TP=2 agg same-host | Same-host disagg PD-TP=1 | Cross-host 4E disagg (B70 enc + giga01 PD) |
|---|---:|---:|---:|---:|
| 4 imgs / 768p (~127 MB embed) | ~2.7 | **~3.1** | — | 1.33 |
| 8 imgs / 768p (~242 MB embed) | ~1.5 | **~1.9** | 0.62 | 0.66 |
| **8 imgs / 1080p (~638 MB embed)** | 0.47 | **0.6–0.7** | 0.27 | 0.13 |
| 8 imgs / 4K (~67 MB body) | 0.06 | **0.12** | 0.04 | (not tested) |
| 16 imgs / 1080p | 0.21 | **0.28** | 0.10 | (not tested) |

The cross-host 4E with the **xpu→cuda GPUDirect patch** matches unpatched 4E on
throughput; the patch shifts decode latency dramatically (TPOT 18 ms → 2.4 ms at
8img/1080p) but TTFT-dominated workloads see no throughput change. Encoder ViT on
B70 XPU is the rate limiter at 8img/1080p.

---

## How to read this directory

The docs fall into six topical groups, listed below in roughly chronological order
within each group. **Recommended reading order for a newcomer:**

1. Start with [`SESSION_MEMORY.md`](./SESSION_MEMORY.md) — the session checkpoint, summarises
   everything up to 2026-05-23 in one place.
2. For the patch story across hosts: [`b70_patched.md`](./b70_patched.md) →
   [`giga01_to_b70_response.md`](./giga01_to_b70_response.md) →
   [`patched_results_b70_h200_v01.md`](./patched_results_b70_h200_v01.md) →
   [`patched_4E_results.md`](./patched_4E_results.md).
3. For the encoder bottleneck deep-dive: [`b70_encoder_time_breakdown.md`](./b70_encoder_time_breakdown.md)
   (with its companion [`code_for_encoder_bottleneck.md`](./code_for_encoder_bottleneck.md)).
4. For the conceptual model of why disagg loses:
   [`per_request_handoff.md`](./per_request_handoff.md) →
   [`when_vit_is_bottleneck.md`](./when_vit_is_bottleneck.md) →
   [`when_disagg_wins.md`](./when_disagg_wins.md).

---

## Group A: Initial NIXL deadlock investigation (May 15)

| File | Lines | One-line summary |
|---|---:|---|
| [`DEBUG_PROCESS.md`](./DEBUG_PROCESS.md) | 580 | First major debug session: NIXL ring-buffer deadlock at 8×1080p, worked around by reducing to 4×768p. 59% → 100% success rate. |
| [`TECHNICAL_SUMMARY.md`](./TECHNICAL_SUMMARY.md) | 526 | Formalised the buffer-pressure model: `embedding_size × concurrency`, deadlock threshold ≈ 20 GB. Validates reductions in image count or resolution. |

**Key takeaway:** the deadlock is documented in dynamo source as a known bug
(`embedding_transfer.py` line ~712-722, `[gluo WIP]` comment). Workload reduction is the
only working same-host workaround.

---

## Group B: Same-host rate-sweep matrix (May 21–22)

These cover all 8img workloads at 768p / 1080p / 4K, the 16img test, decode-heavy, and
cache-hit, comparing TP=1 agg vs TP=2 agg vs disagg across rates 0.10–1.25.

| File | Lines | One-line summary |
|---|---:|---|
| [`disagg_all_rates_results.md`](./disagg_all_rates_results.md) | 232 | Original 8×1080p disagg sweep. 0.27 RPS, 15/64 failed at rate=1. Found `DYN_VLLM_*` env-var bug (sglang reads `DYN_SGL_*`). |
| [`disagg_improvements_attempts.md`](./disagg_improvements_attempts.md) | 292 | Second-round investigation. Fixed env-var, added `max_running=64` + `chunked_prefill=16384`. 0.27→0.23 RPS, 0 failures. |
| [`bottleneck_analysis.md`](./bottleneck_analysis.md) | 211 | Full per-phase breakdown of best disagg config at rate=1: queue 112 s + forward 83 s. Embedding-integration small batches = 46% of prefill events. |
| [`deep_analysis_disagg_worse_h200.md`](./deep_analysis_disagg_worse_h200.md) | 294 | First deep analysis: median 8192-token chunk wallclock 2.28 s (disagg) vs 1.08 s (TP=1 agg). Smoking gun = small-batch embedding-integration on PD. |
| [`1080p_sweep_three_way.md`](./1080p_sweep_three_way.md) | 182 | Full 5-rate sweep at 8×1080p: TP=2 agg ≥1.13 RPS, TP=1 0.57, disagg 0.25. |
| [`768p_comparison.md`](./768p_comparison.md) | 176 | 5-rate sweep at 8×768p. **Note:** ~99% identical to `768p_sweep_three_way.md` (see consolidation notes). |
| [`768p_sweep_three_way.md`](./768p_sweep_three_way.md) | 176 | Same data as `768p_comparison.md`. Both files exist; content overlaps near-fully. |
| [`8img_768p_agg_disagg.md`](./8img_768p_agg_disagg.md) | 143 | Single-rate (1.0) view of 768p comparison. |
| [`16img_1080p_three_way.md`](./16img_1080p_three_way.md) | 165 | Pushed to 16 images (~33 MB body). Required `DYN_TCP_MAX_MESSAGE_SIZE=256MB`. ViT/total = 25%, still LLM-bound. TP=2 wins. |
| [`8img_4k_blocked.md`](./8img_4k_blocked.md) | 129 | Discovered `axum::DefaultBodyLimit` blocking 67 MB 4K bodies. Found `DYN_HTTP_BODY_LIMIT_MB=256` env-var (no rebuild needed). |
| [`8img_4k_three_way.md`](./8img_4k_three_way.md) | 177 | After body-limit fix: TP=1 0.06, TP=2 0.12, disagg 0.04. All LLM-bound. TP=2 wins. |
| [`long_output_1024_results.md`](./long_output_1024_results.md) | 176 | Decode-heavy at rate=0.5: disagg 0.42 vs TP=1 0.43 — the closest disagg gets. Disagg wins +22% peak output throughput only. |
| [`long_output_1024_rate2_results.md`](./long_output_1024_rate2_results.md) | 172 | Decode-heavy at rate=2.0: disagg 1.00 vs TP=1 1.15. Hypothesis (decode-heavy favors disagg) **falsified**. |
| [`rate_2.0_agg_disagg_comparison.md`](./rate_2.0_agg_disagg_comparison.md) | 188 | Same data as `long_output_1024_rate2_results.md`, written from a slightly different angle. Both kept; see consolidation notes. |
| [`cache_hit_agg_disagg.md`](./cache_hit_agg_disagg.md) | 169 | **Canonical cache-hit doc.** Hypothesis (encoder cache favors disagg) falsified: cache hits help both configs ~85-90%, no asymmetric win. |
| [`cache_hit_comparison.md`](./cache_hit_comparison.md) | 9 | Stub redirecting to `cache_hit_agg_disagg.md` (was a byte-identical duplicate before consolidation). |

**Key takeaways from Group B:**
1. TP=2 agg wins every single workload tested.
2. Disagg's only structural win is **peak output throughput** (+9 to +200% in cache-hit, decode-heavy regimes) — meaningless niche metric for serving.
3. **No rate, image count, resolution, decode/prefill ratio, or cache-hit pattern** flips the comparison on a single host with this 32B-FP8 model.

---

## Group C: PD-TP=2 + patch experiments (May 22–23, all reverted)

These tested whether more compute or runtime patches could fix disagg's bottleneck.

| File | Lines | One-line summary |
|---|---:|---|
| [`pd_tp2_results.md`](./pd_tp2_results.md) | 133 | 3-GPU disagg (1 encoder + TP=2 PD on GPUs 5,7). 0.24 RPS — adding 3rd GPU gave **zero gain**. TPOT 57ms (great) but TTFT 230s (worse than 2-GPU). Confirms PD compute isn't the bottleneck. |
| [`patches_for_one_request_handoff.md`](./patches_for_one_request_handoff.md) | 526 | **Full patch log: 5 rounds.** Round 2a (GPU descriptor patch on PD receive side) showed 5× per-request speedup on 5/32 surviving requests before OOM. Round 3 (RDMA-only) confirmed CPU bounce isn't transport-bound. Round 4 OOM at any `mem_fraction_static`. |
| [`per_request_handoff.md`](./per_request_handoff.md) | 212 | Conceptual document explaining the ~11.5 s per-request hand-off floor: TCP plane + NIXL setup + scheduler enqueue. Why agg avoids it (no IPC). Why TP=N PD doesn't help (compute isn't the bottleneck). |
| [`disagg_time_summary.md`](./disagg_time_summary.md) | 228 | Earlier per-request time-breakdown (NIXL stall + decode collapse). 75-second activity gap on PD due to NIXL buffer back-pressure. Smoking gun for the deadlock from group A. |
| [`saturation_analysis.md`](./saturation_analysis.md) | 192 | Master saturation table consolidating all same-host + cross-host numbers + per-workload TTFT. Goes deep on the 100 MB/s NIXL effective throughput ceiling vs 50 GB/s wire capacity = 500× CPU bottleneck. |
| [`SESSION_MEMORY.md`](./SESSION_MEMORY.md) | 285 | **Session checkpoint** (most useful single doc to read first). Captures all Phase 1-6 results, file locations, env vars, current state, open questions for future work. |

**Key takeaways from Group C:**
1. The bottleneck is the **per-request hand-off** (~11.5 s floor), not compute.
2. **Encoder-side optimisation makes things worse** (rounds 1a-1c) because PD intake can't keep up.
3. **The right fix is GPU-resident NIXL receive descriptors with pre-allocated pool** — not feasible as a runtime patch (multi-day dynamo-upstream engineering task).

---

## Group D: Cross-host (B70 ↔ giga01) and the GPUDirect patch (May 23–24)

This is the most recent track: encoder on B70 (Intel Battlemage XPU) cross-host to giga01
(NVIDIA H200) PD, plus the patches enabling true GPUDirect xpu→cuda RDMA.

| File | Lines | One-line summary |
|---|---:|---|
| [`cross_host_giga01_b70_results.md`](./cross_host_giga01_b70_results.md) | 258 | First cross-host benchmarks. 4 encoders gives 1.02 RPS @ 4img/768p; only 0.13 RPS @ 8img/1080p (PD's CPU NIXL pipeline saturates at 2-3 in-flight requests). |
| [`B70_PATCH_INSTRUCTIONS.md`](./B70_PATCH_INSTRUCTIONS.md) | 190 | Patch instructions sent from giga01 operator to B70 operator. The 1-spot edit on `encode_worker_handler.py` to use `_nixl_buffer_device()` instead of hardcoded CPU. |
| [`b70_patched.md`](./b70_patched.md) | 456 | **Report-back from B70 operator** documenting both patch sites (cached + non-cached path), env-var fixes, and verification. Includes "what we found via your diagnostic steps" appendix. |
| [`giga01_to_b70_response.md`](./giga01_to_b70_response.md) | 146 | giga01 reporting back: 100% NIXL transfers still showed `device=cpu` despite patch. Triggered finding the second code path (`_encode_with_cache` was unreachable when cache disabled). |
| [`patched_results_b70_h200_v01.md`](./patched_results_b70_h200_v01.md) | 219 | First end-to-end measurement with both-side patches active. 65/65 NIXL transfers `cuda:0 ↔ xpu:0`. RPS unchanged at 8img/1080p (encoder-bound), TPOT 18→2.4 ms (7×). |
| [`patched_4E_results.md`](./patched_4E_results.md) | 204 | Full 7-rate sweep × 3 workloads = 21 runs, all 32/32 successful. Throughput identical to unpatched 4E; decode latency 22-90% better. |
| [`h200_time_breakdown_v02.md`](./h200_time_breakdown_v02.md) | 207 | Per-request timeline analysis on patched run: encoder→PD gap = 343 s (92% of E2E latency). Encoder ViT on B70 XPU is the rate limiter; NIXL transfer is ~13 ms wire + ~ms setup. |
| [`vllm_yaml_vs_sglang_scripts.md`](./vllm_yaml_vs_sglang_scripts.md) | 240 | Compared a vLLM K8s YAML to our sglang scripts. Found `NIXL_USE_CPU_HOST_MEMORY=0` works on vLLM's encoder but NOT sglang's — sglang hardcodes CPU at line 219. Motivated the §1B patch. |
| [`round5_patch_results.md`](./round5_patch_results.md) | 178 | Same-host smoke test of the same patch direction. 4× RPS / 15× TTFT improvement but reproduces same-host's GPU-OOM problem because GPU NIXL descriptors compete with SGLang's working memory without coordination. |
| [`vllm_vs_sglang_handoff.md`](./vllm_vs_sglang_handoff.md) | 206 | Code comparison between dynamo's vLLM and sglang multimodal backends. vLLM has `stage_embeddings=True` + lifecycle queue + scheduler-integrated CPU embedding cache. Cleaner code, but doesn't fix the structural per-request floor for random-image workloads. |

**Key takeaways from Group D:**
1. Cross-host disagg with the patch **does not improve throughput** at 8img/1080p but
   **dramatically improves decode latency** (TPOT 7-90× better depending on workload).
2. Encoder ViT on B70 XPU is the bottleneck for cross-host at 8img/1080p (~340 s of the
   ~370 s E2E mean — 92%).
3. The **two-site patch** (cached path AND non-cached path) is required because dynamo
   takes the non-cached `else` branch when `ENABLE_ENCODER_CACHE=0`, which all our
   launchers use.
4. Removing `NIXL_USE_CPU_HOST_MEMORY=1` from B70 launch scripts is **required** for the
   patch to take effect; otherwise `_nixl_buffer_device()` falls back to CPU.

---

## Group E: Conceptual / analytical (no measurements)

| File | Lines | One-line summary |
|---|---:|---|
| [`when_disagg_wins.md`](./when_disagg_wins.md) | 239 | Catalogue of 9 candidate workloads that could in principle make disagg win. Most realistic is cache-hit; long-output decode-heavy is easiest to test. None actually flipped the comparison in our matrix. |
| [`when_vit_is_bottleneck.md`](./when_vit_is_bottleneck.md) | 217 | Math: ViT/(ViT+LLM) ≈ 17-32% on Qwen3-VL-32B even at 4K × 8 images. To get ViT-bound, need smaller LLM (Qwen3-VL-7B/3B), heavier ViT (Pixtral, InternVL2), token compression, or video. |
| [`where_is_bottleneck.md`](./where_is_bottleneck.md) | 310 | Closely related to `when_vit_is_bottleneck.md`. Per-token FLOP estimates for 32B / 7B / 3B. Why more images makes things WORSE (O(N²) attention). |
| [`b70_encoder_time_breakdown.md`](./b70_encoder_time_breakdown.md) | 389 | **The 2026-05-24 instrumented timing analysis.** Per-stage measurements (`pure_encode_ms`, `patch_to_xpu_ms`, etc.), patched-on vs patched-off A/B, plus 1-encoder vs 4-encoder comparison at concurrency=4. **Vision tower = 58% of total time at 4-image.** |
| [`code_for_encoder_bottleneck.md`](./code_for_encoder_bottleneck.md) | 335 | Source-code walkthrough showing the bottleneck path is `encode_server.py:1078 → qwen3_vl.py:1210` (ViT forward). 5 steps from `mm_encode` wrapper to ViT to our patched handler. |
| [`b70_memory_leak.md`](./b70_memory_leak.md) | 254 | XPU memory leak diagnosed at 8img/1080p: ~60-80 MiB per request residual after Fix A (`torch.xpu.empty_cache()`) + Fix B (`stage_embeddings=True`). Time-to-OOM extended 4-5× but not eliminated. Likely root cause: `empty_cache()` is a no-op on Intel Level Zero. |

**Key takeaways from Group E:**
1. The single bottleneck on B70 is `self.visual(pixel_values)` in `qwen3_vl.py:1210` — the
   600 M-parameter ViT forward on Battlemage XPU through `triton_attn`.
2. There's an **avoidable** `mm_embedding.cpu()` at `encode_server.py:1079` that the patch
   effectively un-does. A clean upstream fix would skip this `.cpu()` when the caller
   wants a device tensor.
3. **Going from 1E to 4E gives 1.5× throughput at conc=4** (per-encoder utilisation drops
   to 40%). Beyond 4E the H200 PD becomes the bottleneck.

---

## Group F: Patches and tooling (not docs but referenced everywhere)

These are not `.md` files but are referenced repeatedly:

| File | What it is |
|---|---|
| `b70_xpu_nixl.patch` | Unified diff (3.1 KB) of both patch sites in `encode_worker_handler.py`. Apply with `patch -p0`. |
| `h200_cuda_nixl.patch` | Unified diff for the giga01 receive-side patches (lines 882, 915 of `embedding_transfer.py`). |
| `dynamo_stream_fix.py` | (in this directory) helper related to the SSE stream tail. |
| `logs/bench_run/analyze_bench.py` | Generic BENCH_TIMING analyser, supports any number of `--group LABEL FILE…` groups. |
| `logs/bench_run/patched_on_w{1..4}.txt` | 79 BENCH_TIMING rows from 4E patched-ON run. |
| `logs/bench_run/patched_off_w{1..4}.txt` | 60 rows from 4E patched-OFF run. |
| `logs/bench_run/patched_on_1E_w1.txt` | 60 rows from 1E patched-ON run. |
| `logs/bench_run/analyze_*.txt` | Cached output of analyzer runs. |

---

## Reading paths by use case

**"I want to understand why disagg is slow"** → start with
[`per_request_handoff.md`](./per_request_handoff.md) →
[`bottleneck_analysis.md`](./bottleneck_analysis.md) →
[`patches_for_one_request_handoff.md`](./patches_for_one_request_handoff.md).

**"I want to know what the patch does"** → [`B70_PATCH_INSTRUCTIONS.md`](./B70_PATCH_INSTRUCTIONS.md) →
[`b70_patched.md`](./b70_patched.md) → [`patched_4E_results.md`](./patched_4E_results.md).

**"I want raw numbers for some workload size"** → [`saturation_analysis.md`](./saturation_analysis.md)
(consolidated table) or one of the per-workload sweep docs.

**"I want to know where time goes inside the encoder"** →
[`b70_encoder_time_breakdown.md`](./b70_encoder_time_breakdown.md) →
[`code_for_encoder_bottleneck.md`](./code_for_encoder_bottleneck.md).

**"Why does the encoder OOM after a few hours of bench traffic?"** →
[`b70_memory_leak.md`](./b70_memory_leak.md).

**"I want to know when disagg would win"** → [`when_disagg_wins.md`](./when_disagg_wins.md) →
[`when_vit_is_bottleneck.md`](./when_vit_is_bottleneck.md) →
[`where_is_bottleneck.md`](./where_is_bottleneck.md).

**"I just want the executive summary"** → [`SESSION_MEMORY.md`](./SESSION_MEMORY.md).

---

## Consolidation notes (2026-05-24)

To reduce duplication while preserving inbound links, this consolidation was applied
on 2026-05-24:

1. **`cache_hit_comparison.md` was a byte-identical duplicate of `cache_hit_agg_disagg.md`.**
   `cache_hit_comparison.md` is now a 9-line stub pointing to the canonical
   `cache_hit_agg_disagg.md`. The self-reference inside `cache_hit_agg_disagg.md`'s
   "This document:" footer was also fixed.

2. **`long_output_1024_rate2_results.md` and `rate_2.0_agg_disagg_comparison.md` cover the
   same data but with different prose framings.** Both are kept as-is — they are not
   exact duplicates, and each has slightly different commentary worth preserving. They
   simply have inbound references from different docs.

3. **`768p_comparison.md` and `768p_sweep_three_way.md` overlap heavily** but are also
   not exact byte-duplicates. Kept both.

No other changes were made to existing docs during the consolidation.

---

## Total

- **38 doc files** (39 minus the consolidated stub) covering the investigation
- **~9,500 lines** of analysis
- **~10 days** of work
- **One operational pattern that wins:** same-host TP=2 agg with `--mem-fraction-static 0.85`,
  `--max-running-requests 25`, `DYN_TCP_MAX_MESSAGE_SIZE=256MB`, `DYN_HTTP_BODY_LIMIT_MB=256`
- **One patch that's structurally correct but doesn't move the throughput needle on the
  encoder-bound regime** but does win on decode latency: `b70_xpu_nixl.patch` +
  `h200_cuda_nixl.patch`
- **One known unfixable structural issue without a multi-day dynamo upstream change:**
  pre-allocated GPU NIXL descriptor pool with `mem_fraction_static` coordination.

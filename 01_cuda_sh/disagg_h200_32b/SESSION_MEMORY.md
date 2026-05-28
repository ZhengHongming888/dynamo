# Session Memory: Qwen3-VL Multimodal Disagg Investigation

**Purpose:** Single document capturing everything from this investigation session so it can be loaded into a future Claude conversation. Includes goals, constraints, all key findings, file paths, and current state.

---

## How to use this in a future conversation

Tell Claude:
> Read `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/SESSION_MEMORY.md` first, then continue work on the dynamo multimodal disagg investigation.

This will reload all context. Then work continues from "Current state" section.

---

## High-level context

**Project:** Investigate when disaggregated (separate encoder + PD workers) inference beats aggregated (all-in-one) inference for multimodal LLMs in NVIDIA dynamo + SGLang stack on H200 GPUs.

**Hardware:** Single H200 host (172.26.46.75) with 8 GPUs available; primary work on GPUs 4, 5, 7 (NUMA 2, NV18 NVLink mesh ~478 GB/s). NICs: mlx5_0/2/4/6/8/10/11 ACTIVE 400 Gb/s RoCE. `nvidia_peermem` loaded → GPUDirect RDMA available.

**Models tested:**
- `Qwen3-VL-32B-Instruct-FP8` (primary, locally available)
- `Qwen3-VL-8B-Instruct` (downloaded mid-session for ratio test)

**Constraints/preferences:**
- Standard workload: 8 imgs × 1080p, 128 input + 256 output, np=64
- Standard rate sweep: `0.1 0.25 0.5 1.0 1.25`
- Result base convention: `/hongming/res<N>/h200_<config>/...`
- Analysis docs go in `/hongming/dynamo/01_cuda_sh/<agg|disagg>_h200_<size>/`
- For disagg use `DYN_SGL_EMBEDDING_TRANSFER_MODE` (not `DYN_VLLM_*`)
- Final production target: cross-host (encoder on different machine from PD)
- User authorized autonomous fixes during long unattended runs

---

## Bottom-line findings (TL;DR)

### Result rankings (saturation RPS, 8 imgs × 1080p, np=64, 32B-FP8)

| Config | GPUs | Sat. RPS | RPS/GPU |
|---|---:|---:|---:|
| **TP=2 agg** | 2 | **0.95** | 0.48 |
| TP=1 agg | 1 | 0.52 | **0.52** |
| Disagg PD-TP=1 | 2 | 0.23 | 0.12 |
| Disagg PD-TP=2 (3-GPU) | 3 | 0.24 | 0.08 |

**TP=2 agg wins all tested workloads.** Disagg never wins on this stack.

### Disagg never wins because of the **per-request hand-off** bottleneck

Each request crosses ~11.5 s of fixed overhead in steps that aren't ViT or LLM compute (TCP plane, NIXL setup, scheduler enqueue, KV alloc, embedding transfer). Even at no contention, the floor exists. Adding more PD compute (TP=2 PD) doesn't help because compute isn't the limit.

Concrete evidence from PD-TP=2 run, request `c74cb9d2`:
```
06:41:20.221  encoder receives request
06:41:23.835  encoder finishes ViT (3.6 s of useful work)
06:41:35.388  PD starts first prefill (11.5 s of HAND-OFF)
06:41:35.871  request done (0.5 s of LLM compute)
```

### The CPU bounce is the dominant hand-off cost

Confirmed by patching `embedding_transfer.py` to use `device=cuda` for NIXL receive descriptors (round 4 patch attempt):
- For 5 requests that completed before OOM: TPOT collapsed from 57 ms → 13 ms (4.4× speedup), TTFT from 230s → 47s (5×)
- But OOM crashed the run because dynamic GPU allocation isn't coordinated with `mem_fraction_static`

**Removing cuda_ipc from UCX_TLS does NOT fix this** (round 3 test). The CPU bounce is a Python-level allocation in `embedding_transfer.py:915` that defaults to CPU memory. UCX transport selection is irrelevant.

### Cross-host won't fix it either without dynamo code changes

Going cross-host uses the same `embedding_transfer.py` code path. RDMA over NIC will still write to CPU memory unless the dynamo code is patched to use GPU descriptors. Cross-host has one structural advantage (encoder GPU and PD GPU don't compete for `mem_fraction_static`), but the code fix is still needed.

### vLLM backend has cleaner code but doesn't solve the structural problem

vLLM has `stage_embeddings=True` properly with lifecycle queue, plus a scheduler-integrated CPU embedding cache. But for random-image workloads (no cache locality), vLLM's improvements only shave ~100-200 ms off the ~11.5 s hand-off — not transformative.

User reported running vLLM disagg before and seeing it BEAT vLLM agg-TP=1 (0.48 vs 0.42 RPS at rate=1.0). Couldn't reproduce here (no scripts/data preserved). Possible explanations: cache locality from bench_serving seed, or vLLM-agg being weaker than SGLang-agg.

---

## Session timeline

### Phase 1: Establish baseline (first ~3 days of conversation, condensed in earlier history)
- TP=1 agg sweep (0.52 RPS sat), TP=2 agg sweep (0.95 RPS sat) at 8 × 1080p
- Disagg sweep (0.23 RPS sat), found TP=1 agg beats 2-GPU disagg
- Workload variations: decode-heavy, cache-hit, 768p, 16 images, 4K — all confirm same ranking

### Phase 2: HTTP body limit fixed
- Discovered `DYN_HTTP_BODY_LIMIT_MB=256` env var (default 45 MB rejected 67 MB 4K request bodies)
- Together with `DYN_TCP_MAX_MESSAGE_SIZE=268435456` (256 MB) and `mem_fraction_static=0.85`, enables 8 × 4K testing

### Phase 3: 4K three-way comparison
- TP=1 agg 0.06 RPS, TP=2 agg 0.12 RPS, Disagg 0.04 RPS — same ranking
- Confirmed LLM (not ViT) is the bottleneck at 4K

### Phase 4: 8B model test (reduce LLM cost to flip ViT/LLM ratio)
- Downloaded `Qwen/Qwen3-VL-8B-Instruct` to `/mnt/weka/data/llm-d-models-pv/`
- TP=1 8B 0.84 RPS, TP=2 8B 1.11 RPS, Disagg 8B 0.38 RPS — same ranking despite ViT being a bigger fraction
- Conclusion: smaller LLM doesn't help; bottleneck is hand-off, not compute ratio

### Phase 5: 3-GPU disagg (PD-TP=2)
- Encoder GPU 4 + PD TP=2 on GPUs 5,7
- Sat 0.24 RPS — adding 3rd GPU gave near-zero improvement
- TPOT improved (57ms vs ~600ms at PD-TP=1) but TTFT got WORSE (230s vs ~80s)
- Confirmed: PD compute isn't the bottleneck, hand-off is

### Phase 6: Patch attempts
- **Round 1 (encoder side)**: stream-scoped sync, `stage_embeddings=True` with lifecycle queue, removed trailing `await transfer_future`. Result: no improvement, slight TPOT regression. Confirmed encoder isn't the bottleneck.
- **Round 2a (PD side, GPU buffers)**: patch `embedding_transfer.py` to use `device=cuda` for receive descriptors. Result: **5/32 requests succeeded with 4-5× speedup** before OOM crashed it. Strongest evidence the CPU bounce is the bottleneck.
- **Round 2b (lower mem_fraction)**: failed entirely.
- **Round 3 (RDMA-only test)**: removed `cuda_ipc,cuda_copy` from `UCX_TLS`. Result: identical to baseline. Proved transport selection doesn't matter — descriptor placement does.
- **Round 4 (GPU buffers + tuned config)**: same patch + lower mem_fraction (0.70-0.78) + lower max_running (16-32). Result: still OOM in both attempts. Need pre-allocated pool with proper mem_fraction coordination.

---

## Key file locations

### Analysis docs (in chronological-ish order)
| Path | What it covers |
|---|---|
| `1080p_sweep_three_way.md` | TP=1/TP=2/disagg at 1080p baseline |
| `8img_4k_blocked.md` | Discovery of HTTP body limit, 3 layers of size limits |
| `8img_4k_three_way.md` | 4K three-way comparison after limit fix |
| `where_is_bottleneck.md` | Math: ViT vs LLM compute ratio analysis |
| `per_request_handoff.md` | Concept and timeline of per-request hand-off |
| `pd_tp2_results.md` | 3-GPU disagg results (proved adding GPU doesn't help) |
| `vllm_vs_sglang_handoff.md` | Code comparison between vLLM and SGLang backends |
| `patches_for_one_request_handoff.md` | All 4 rounds of patch attempts with diffs and results |
| `8b_three_way.md` | (in `disagg_h200_8b/`) 8B model results |
| **`SESSION_MEMORY.md`** | **this document** |

### Server scripts
- `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh` — TP=1 agg (GPU 4)
- `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh` — TP=2 agg (GPUs 4,5)
- `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_combined.sh` — 2-GPU disagg (encoder GPU 4, PD GPU 5)
- `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2.sh` — 3-GPU disagg (encoder GPU 4, PD GPUs 5,7)
- `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2_rdma.sh` — same as above but no cuda_ipc in UCX_TLS
- `01_cuda_sh/disagg_h200_32b/start_disagg_h200_32b_pd_tp2_gpubuf.sh` — same + GPU buffer patch + lower mem_fraction
- `01_cuda_sh/agg_h200_8b/start_h200_aggregate_epd_server_8b_tp1.sh` — 8B TP=1 agg
- `01_cuda_sh/agg_h200_8b/start_h200_aggregate_epd_server_8b_tp2.sh` — 8B TP=2 agg
- `01_cuda_sh/disagg_h200_8b/start_disagg_h200_8b_combined.sh` — 8B disagg

### Bench scripts
- `test_sglang_mult_rates_32b_1080p_np64_over_rates.sh` — standard 5-rate 32B sweep
- `test_sglang_8b_1080p_np64_over_rates.sh` — 8B equivalent
- `test_sglang_8img_4k.sh` — 4K variant
- `test_sglang_32b_pd_tp2_1080p_np64_over_rates.sh` — PD-TP=2 variant

### Orchestrators (all run a server + bench + cleanup)
- `run_4k_three_way.sh` — 4K full sweep
- `run_8b_three_way.sh` — 8B full sweep
- `run_pd_tp2.sh` — 3-GPU disagg sweep
- `run_pd_tp2_tuned.sh` — round 1+2 patch tests
- `run_pd_tp2_rdma.sh` — RDMA-only test
- `run_pd_tp2_gpubuf.sh` — GPU buffer test

### Result directories
- `/hongming/res4/` — initial 32B 1080p sweeps
- `/hongming/res5/` — 16 imgs @ 1080p
- `/hongming/res6_img8_4k/` — 4K three-way
- `/hongming/res7_8B/` — 8B three-way
- `/hongming/res8_pd_tp2/` — 3-GPU disagg baseline
- `/hongming/res9_pd_tp2_tuned/` — round 1+2 patch attempts
- `/hongming/res10_pd_tp2_rdma/` — RDMA-only test
- `/hongming/res11_pd_tp2_gpubuf/` — GPU buffer test (FAILED)

### Code files modified (all reverted, backups in `.bak`)
- `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py`
- `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/worker_handler.py`
- `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py`

Backups exist at `*.bak` for each.

### Other reference paths
- Dynamo source (used for code investigation, NOT modified): `/opt/dynamo/`
- vLLM backend code (used for comparison): `/opt/venv/lib/python3.12/site-packages/dynamo/vllm/`
- vLLM example launch scripts: `/hongming/dynamo/examples/backends/vllm/launch/`

---

## Critical environment variables

These are set in the start scripts and must remain:
- `DYN_TCP_MAX_MESSAGE_SIZE=268435456` (256 MB) — TCP plane between workers
- `DYN_HTTP_BODY_LIMIT_MB=256` — frontend HTTP body limit
- `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read` — NIXL READ mode for SGLang backend
- `mem_fraction_static=0.85` — sweet spot for headroom (0.95 OOMs at large workloads, 0.70-0.78 OOMs with GPU NIXL buffers)
- `UCX_NET_DEVICES=mlx5_4:1` — NIC selection (irrelevant for same-host cuda_ipc, set anyway for diagnostic clarity)

---

## What still needs to happen

### To make disagg actually win:

The right fix in dynamo upstream (multi-day eng task):

1. Add `--multimodal-nixl-buffer-gb` CLI arg
2. Pre-allocate that GPU pool BEFORE SGLang engine init
3. Pass adjusted `mem_fraction_static * (total_gpu - nixl_pool)` to SGLang
4. Refactor `NixlReadEmbeddingReceiver` to allocate descriptors as offset views into pool
5. Solve variable-shape descriptor issue (the `[gluo FIXME]` comment in `dynamo/common/multimodal/__init__.py:40`) — possibly by sizing each pool entry at the largest expected embedding and using prefix views per request

This same fix is needed regardless of single-host vs cross-host.

### Recommendation for the user's cross-host plan:

**File a dynamo upstream issue first.** Without the descriptor pool fix, cross-host disagg will hit the same per-request hand-off floor as same-host. Specifically:

- Issue title: "Multimodal NIXL receive descriptors allocated on CPU, defeating GPUDirect RDMA on cross-host disagg"
- Evidence: round 4 patch shows GPU descriptors give 4-5× per-request speedup, but ad-hoc dynamic allocation OOMs without pool coordination
- Repro: this session's scripts (`start_disagg_h200_32b_pd_tp2_gpubuf.sh`)
- Suggested fix: pre-allocated pool with `mem_fraction_static` coordination

---

## Current state

**As of 2026-05-23 ~20:08 UTC:**
- All servers stopped (verified: `pgrep -af "dynamo|sglang"` returns nothing)
- All GPUs free (verified: `nvidia-smi` shows 0 MiB on GPUs 4, 5, 7)
- All code patches reverted (verified: `diff *.bak *` returns clean)
- All `.bak` files preserved for reference

**Saturation results we measured (32B-FP8, 8 × 1080p, np=64, 5-rate sweep):**
- TP=1 agg: 0.52 RPS
- TP=2 agg: 0.95 RPS
- Disagg PD-TP=1: 0.23 RPS
- Disagg PD-TP=2: 0.24 RPS
- Disagg PD-TP=2 + GPU descriptors (rate=1.0 only, partial): would be ~0.5+ RPS if not OOM (extrapolated from 5/32 succeeded with TPOT 13ms vs baseline 57ms)

**Saturation results (8B, 8 × 1080p, np=64):**
- TP=1 agg: 0.84
- TP=2 agg: 1.11
- Disagg: 0.38

**4K results (8 × 4K, np=32):**
- TP=1 agg: 0.06
- TP=2 agg: 0.12
- Disagg: 0.04

---

## Useful commands for picking up from here

```bash
# Verify clean state
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,7
pgrep -af "dynamo|sglang|bench_serving" | grep -v "vim\|grep" | head

# Re-apply GPU descriptor patch (round 4)
cd /opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/
# (patch shown in patches_for_one_request_handoff.md round 4 section)

# Run disagg PD-TP=2 baseline (proven working)
nohup setsid bash /hongming/dynamo/run_pd_tp2.sh > /dev/null 2>&1 &

# Run TP=2 agg (best config so far)
nohup setsid bash /hongming/dynamo/01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh &
# then in another shell:
RESULT_BASE=/hongming/res_check/tp2_agg bash /hongming/dynamo/test_sglang_mult_rates_32b_1080p_np64_over_rates.sh 1.0
```

---

## Open questions / things we could pursue

1. **Replicate the user's vLLM disagg-beats-agg result** — would require setting up vLLM scripts, running, and analyzing why vLLM beat agg in the user's prior testing
2. **Implement proper NIXL pool patch** — the multi-day fix described above. Could be a useful upstream contribution to dynamo.
3. **Test cross-host setup** — but only after the pool fix lands; otherwise will reproduce the same bottleneck
4. **Test even smaller LLM (Qwen3-VL-2B or 4B)** — to see if at some LLM size, ViT compute genuinely dominates enough that hand-off becomes negligible. We have these models on HF but not downloaded.
5. **Profile what's actually IN the 11.5s hand-off** — break it down by sub-step (TCP plane, NIXL setup, scheduler enqueue) to identify which sub-step dominates. Would need to instrument the code.

---

## Permission/style notes for future Claude

- User authorizes autonomous fixes during long unattended runs ("any issue please fix issue without my approval")
- User wants honest assessment, not validation. If a hypothesis is wrong, say so.
- Save analysis docs to `/hongming/dynamo/01_cuda_sh/<dir>/<name>.md`
- Use `nohup setsid bash ... &` + `disown` for long-running orchestrators (survive shell timeouts)
- Always verify clean state (`pgrep`, `nvidia-smi`) before launching new servers
- Always verify `.bak` files match originals before declaring "reverted"
- After each rate completes in a sweep, cat the CSV for visibility

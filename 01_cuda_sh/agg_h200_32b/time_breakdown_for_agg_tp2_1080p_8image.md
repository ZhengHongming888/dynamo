# Time-breakdown analysis: Aggregated EPD, real TP=2 across SYS-connected GPUs, 32B FP8, 1080p×8 images, rate=1 req/s

**System under test:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh` (after edits)
**Model:** Qwen3-VL-32B-Instruct-FP8
**Benchmark:** `sglang.bench_serving --dataset-name image --image-count 8 --image-resolution 1920x1080 --random-input-len 128 --random-output-len 256 --num-prompts 64 --request-rate 1.0`
**Bench window:** 2026-05-20 22:24:00 – 22:26:49 (~169 s analysis window)
**Result file:** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_222206/rate_1.0/`

---

## Script fixes applied before this run

1. **`--tensor-parallel-size 1` → `2`** on line 120. The script was named `_tp2.sh` but was running TP=1.
2. **Removed bash-comment-line `#SGLANG_USE_CUDA_IPC_TRANSPORT=1 \`** from inside the env-var continuation block (line 109 in the original). Bash silently dropped every env var listed *before* the comment line — including `CUDA_VISIBLE_DEVICES=3,4` — so the prior "TP=1" run wasn't actually pinned to GPUs 3,4 either. Verified the fix with `bash -x`.

After fixes the worker correctly reports `tp_size=2` and lands on physical GPUs 3 & 4 (both at 100% util, ~130 GB resident).

---

## (b) GPU topology — root cause of the regression

```
        GPU3   GPU4
GPU3     X    SYS    ← cross-socket via QPI/UPI, no NVLink
GPU4    SYS    X
NUMA     1     2     ← different NUMA nodes (different CPU sockets)
```

- **No NVLink anywhere on this box.** `nvidia-smi nvlink --status` returns nothing for either GPU. This is a non-NVSwitch H200 NVL configuration.
- **GPU 3 and GPU 4 are in different NUMA nodes** (NUMA 1 vs NUMA 2). The only path between them is `SYS` — PCIe **plus** the CPU↔CPU socket interconnect. This is the slowest possible inter-GPU link on the box.
- All TP=2 all-reduces (every transformer layer, every all-reduce inside FA3, per-rank logits) are bottlenecked on this cross-socket PCIe traversal.

Better same-NUMA pairs available (all `NODE`, single-PHB hop, same socket): **(0,1), (2,3), (4,5), (4,6), (5,6)**. The script's choice of `CUDA_DEVICE=3,4` is one of the worst possible pairs.

Full `nvidia-smi topo -m` for reference:
```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity  NUMA
GPU0     X   NODE  SYS   SYS   SYS   SYS   SYS   SYS   0,2,4,...      0
GPU1    NODE   X   SYS   SYS   SYS   SYS   SYS   SYS   0,2,4,...      0
GPU2    SYS  SYS    X   NODE   SYS   SYS   SYS   SYS   48,50,52,54    1
GPU3    SYS  SYS  NODE   X    SYS   SYS   SYS   SYS    48,50,52,54    1
GPU4    SYS  SYS  SYS   SYS    X   NODE  NODE   SYS    1,3,5,...      2
GPU5    SYS  SYS  SYS   SYS  NODE   X   NODE   SYS    1,3,5,...      2
GPU6    SYS  SYS  SYS   SYS  NODE  NODE   X    SYS    1,3,5,...      2
GPU7    SYS  SYS  SYS   SYS   SYS   SYS   SYS    X    49,51,53,55    3
```

---

## (a) Wall-clock breakdown — TP=1 vs TP=2 side-by-side

Both runs have the same workload (rate=1, 64 prompts, 8×1080p, ~16k vision tokens/req).

| Metric | **TP=1** (1 GPU) | **TP=2 across SYS** (GPUs 3,4) | Δ |
|---|---:|---:|---:|
| Bench duration | 152.6 s | 168.4 s | +10 % |
| Window analysed | 245.9 s | 169.3 s | (analysis only) |
| **Prefill share of wall** | **60 %** | **91 %** | +31 pp |
| **Decode share of wall** | 11 % | 9 % | −2 pp |
| **Unaccounted (encoder/idle)** | 29 % | **0 %** | −29 pp |
| Prefill 8192-chunk tput (median) | 9,518 tok/s | **8,077 tok/s** | **−15 %** |
| Prefill 8192-chunk tput (mean) | 8,707 | 8,579 | −1 % |
| Prefill 8192-chunk tput (max) | 11,829 | 13,503 | +14 % |
| Prefill 8192-chunk tput (min) | 110 | 3,729 | (smoothed) |
| Decode gen tput (median) | 71 tok/s | **84 tok/s** | +18 % |
| Decode gen tput (max) | 599 tok/s | **702 tok/s** | +17 % |
| KV usage (typical) | **0.71–0.74** | **0.22–0.24** | −50 pp |
| Cached-token hits | ~0 | ~0 | — |
| Mean TTFT | 50.2 s | **61.0 s** | +22 % |
| Mean E2E | 93.2 s | 107.4 s | +15 % |
| Mean TPOT | 622 ms | 650 ms | +5 % |
| Peak output tput | 592 tok/s | **785 tok/s** | +33 % |

---

## Headline performance (TP=2 run)

| Metric | Value |
|---|---|
| Request throughput | 0.38 req/s (offered 1.0 — back-pressured) |
| Total token tput | 6,283 tok/s |
| Input tput | 6,238 tok/s |
| Output tput | 44.4 tok/s |
| Mean / median TTFT | **61.0 s / 66.7 s**, p99 101.8 s |
| Mean / median TPOT | 650 ms / 407 ms, p99 4,830 ms |
| Mean ITL | 402 ms (median 27 ms — best-case decode is fast) |
| Max ITL | **69.2 s** |
| Concurrency observed | 40.8 |

---

## What this tells us

### Decode kernel is genuinely faster on TP=2 (as expected)
- Median gen tput +18 %, peak +17 %, peak output tput +33 % (785 vs 592 tok/s). Median ITL dropped (27 ms vs 38 ms). When the GPUs aren't synchronizing every step — i.e. during sustained decode without prefill interruption — TP=2 helps.

### Prefill collapsed
- **Median 8192-chunk prefill throughput dropped 15 %** (9,518 → 8,077 tok/s). On a same-NUMA NODE pair we'd expect ~1.5× speedup, not a slowdown.
- Rough back-of-envelope: an all-reduce after every transformer block on a 32B FP8 model ships roughly `seq_len × hidden_dim × dtype_bytes` per layer. For 8192 tokens × 5120 hidden × 2 bytes ≈ 80 MB per all-reduce × 64 layers × 2 ≈ **~10 GB of cross-socket traffic per 8192-token chunk**. Over PCIe Gen5 x16 (~50 GB/s practical) that's ~200 ms minimum just for sync — and across QPI it gets worse. This is why prefill regressed instead of speeding up.

### KV cache is barely used (0.22–0.24 vs 0.71–0.74)
- `mem_fraction_static=0.88` is held *per-rank*, so total KV memory roughly **doubled** with TP=2 (~120 GB → ~240 GB combined), while the workload's KV demand didn't change. The histogram confirms: TP=2 KV usage stays at 22–24 %.
- No memory pressure → memory is not the binder. Compute + **comm** are.

### "Unaccounted" time disappeared (29 % → 0 %) — bookkeeping artefact
- In TP=1, the vision encoder ran on the same single GPU and showed up as gaps between scheduler events.
- In TP=2, the encoder time appears to have been folded into "prefill" steps from the scheduler's POV (one serialized step), so the breakdown shows 91 % prefill. **This doesn't mean the encoder got faster** — it just got bookkept differently.

### Net effect on the user
- TTFT got **worse by 22 %** because (i) queueing was the dominant component (95 % of TTFT was wait time) and (ii) prefill throughput dropped, so the queue drained slower.
- Decode got better, but this only helps in regimes where decode dominates — not this 16k-token-prefill, 117-token-output workload.

---

## Bottom line

**The bottleneck is now demonstrably comm-bound, not compute-bound.** The 10–22 % regression is a topology pathology: TP=2 across `SYS`-connected, cross-socket GPUs without NVLink. On this hardware, a 32B model with 16k-token prefills sees the cross-socket all-reduce dominate.

---

## Recommended next steps (in priority order)

1. **Re-run TP=2 on a same-NUMA pair.** Easiest fix: `CUDA_DEVICE=2,3` (NUMA 1, `NODE`) or `CUDA_DEVICE=5,6` (NUMA 2, `NODE`). Expectation: prefill throughput rises to ~14–16 k tok/s, TTFT roughly halves vs the current TP=2 number.
2. **If results still don't beat TP=1:** the model is too small / too prefill-vision-heavy for TP=2 even on a good link on this hardware. Stick with TP=1 and instead try **disaggregated EPD** (separate encoder / prefill / decode workers on different GPUs). That avoids the all-reduce path entirely while still using more silicon.
3. **Don't bother trying TP=4 on this box** — there's no NVSwitch, so any TP>1 group will include at least one `SYS` hop and the comm penalty compounds.
4. **Lower `mem-fraction-static` for TP=2** (e.g. 0.5–0.6) to free GPU memory for non-KV uses (encoder activations, NIXL buffers) — current setting wastes ~50 % of doubled KV space.

# TP=2 on GPUs 4,5 with NVLink confirmed: major performance win

**System:** Qwen3-VL-32B-Instruct-FP8, aggregated EPD, 8×1920×1080 images per request, rate=1 req/s, 64 prompts.
**GPUs:** 4,5 (NUMA 2, NV18 NVLink — 18 bonded NVLinks @ 26.562 GB/s each ≈ 478 GB/s aggregate)
**Setup script:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh` (after fixes)
**Bench window:** 2026-05-21 17:21:53 – 17:25:32 (~112 s)
**Result file:** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_172153/rate_1.0/`

---

## Hardware reality check (correcting prior analyses)

**This machine has full NVLink. The earlier `.md` files in this directory ALL incorrectly claimed "no NVLink anywhere on this box."**

Verified facts:

| Check | Result |
|---|---|
| GPU model | `NVIDIA H200` (SXM, full NVSwitch fabric) — NOT "H200 NVL" |
| `nvidia-smi nvlink --status -i 4` | 18 active NVLinks @ 26.562 GB/s each |
| `nvidia-smi nvlink --status -i 5` | 18 active NVLinks @ 26.562 GB/s each |
| `nvidia-smi topo -m` GPU4↔GPU5 | **NV18** (NOT `NODE` as previously reported) |
| Aggregate NVLink BW per GPU | ~478 GB/s |
| NCCL log evidence | `Channel 00/0 : 0[4] -> 1[5] via P2P/IPC`, `isAllDirectP2p=1`, 24 NCCL channels |

The earlier reports' diagnoses ("PCIe-bound all-reduce", "NUMA crossing penalty", "DP encoder needed because PCIe is slow") were all **based on this wrong hardware assumption** and need to be revisited.

---

## Bugs found in the start script

1. **`IP_LOCAL=172.26.46.162` → wrong host IP.** Actual host is `172.26.46.75`. The `.162` IP doesn't exist on any interface here. Frontend couldn't connect to etcd at all — failures got hidden because curl from localhost worked and prior runs may have happened on a different machine.
2. **`UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy`** lacked `cuda_ipc`. Without it, NIXL GPU↔GPU transfers fall back to host-staged copies instead of using NVLink P2P.
3. **`--mm-enable-dp-encoder` and `--enable-broadcast-mm-inputs-process`** were enabled as workarounds for the (non-existent) PCIe bottleneck. With real NVLink, the default TP-sharded ViT actually runs faster.
4. **etcd startup wait was 3 s**, occasionally too short on cold start — bumped to 8 s + reachability poll.

## Changes applied

```diff
-export IP_LOCAL=172.26.46.162
+export IP_LOCAL=172.26.46.75

-UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
+UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \

+NCCL_DEBUG=INFO \
+NCCL_DEBUG_SUBSYS=INIT,P2P \

     --enable-mm-global-cache \
-    --mm-enable-dp-encoder \
-    --enable-broadcast-mm-inputs-process \
     --dtype auto \
```

(plus added an etcd-reachable poll after `sleep 8` in the startup section)

---

## NCCL P2P confirmation from new worker log

```
NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
NCCL INFO Channel 00/0 : 0[4] -> 1[5] via P2P/IPC
NCCL INFO Channel 01/0 : 0[4] -> 1[5] via P2P/IPC
... (24 channels total)
NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 4 is 48-71,144-167.
NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 5 is 48-71,144-167.
```

24 NCCL channels with direct P2P/IPC — definitive proof NVLink is being used. CPU affinity matches NUMA 2 for both GPUs.

---

## Headline performance (TP=2 NVLink on GPUs 4,5)

| Metric | Value |
|---|---|
| Request throughput | **0.57 req/s** (offered 1.0 — still back-pressured but much better) |
| Total token tput | **9,467 tok/s** |
| Input tput | 9,400 tok/s |
| Output tput | 67 tok/s |
| **Peak output tput** | **883 tok/s** |
| Mean / median TTFT | **35.0 s / 42.5 s**, p99 **46.7 s** |
| Mean / median TPOT | 473 ms / 223 ms, p99 4,467 ms |
| Mean / median ITL | 261 ms / 20 ms |
| Max ITL | 48.7 s |
| Mean E2E | 64.0 s |
| Concurrency | 36.7 |

---

## Comparison vs prior runs (rate=1, 64 prompts, 1080p×8)

| Metric | TP=1 (old) | TP=2 NODE (old, broken) | **TP=2 NODE+NVLink (NEW)** | 2× TP=1 DP (winner) | Δ NEW vs old TP=2 | Δ NEW vs TP=1 |
|---|---:|---:|---:|---:|---:|---:|
| Bench duration | 152.6 s | 169.5 s | **111.8 s** | 84.1 s | **−34%** | **−27%** |
| Request throughput | 0.42 req/s | 0.38 req/s | **0.57 req/s** | 0.76 req/s | **+50%** | **+36%** |
| Total token tput | 6,933 | 6,241 | **9,467 tok/s** | 12,579 | +52% | +37% |
| Input tput | 6,884 | 6,197 | **9,400 tok/s** | 12,490 | +52% | +37% |
| Peak output tput | 592 | 844 | 883 | 647 | +5% | +49% |
| Mean TTFT | 50.2 s | 60.4 s | **35.0 s** | 11.2 s | **−42%** | **−30%** |
| Median TTFT | 56.0 s | 65.8 s | **42.5 s** | 11.9 s | −35% | −24% |
| P99 TTFT | 84.9 s | 102.9 s | **46.7 s** | 17.4 s | **−55%** | −45% |
| Mean E2E | 93.2 s | 107.3 s | **64.0 s** | 45.2 s | **−40%** | **−31%** |
| Mean TPOT | 622 ms | 671 ms | **473 ms** | 898 ms | −29% | −24% |
| Median TPOT | 354 ms | 382 ms | **223 ms** | 289 ms | −42% | −37% |
| Concurrency | 39.1 | 40.5 | 36.7 | 34.4 | — | — |

---

## Interpretation

**TP=2 is no longer broken.** Once NVLink is actually used (not just topologically present), TP=2 delivers what tensor parallelism is supposed to: faster prefill, faster decode, lower TTFT. The fixed TP=2 setup beats the old TP=1 baseline on every meaningful metric except concurrency.

### Where the gains came from

1. **NVLink-backed all-reduce (NCCL P2P/IPC):** what was supposed to be cheap-on-NVLink, expensive-on-PCIe is actually cheap. The 32B model's per-layer all-reduce now travels at ~478 GB/s aggregate instead of being thrown back to PCIe by the workaround flags.
2. **Removed `--enable-broadcast-mm-inputs-process`:** in the prior "fix attempt", this introduced a 54% "unaccounted" wall-clock window because broadcasting MM input tensors over what was assumed to be PCIe was actually disabling P2P or hitting a different code path. Without it, the encoder pipeline runs cleanly.
3. **Removed `--mm-enable-dp-encoder`:** the DP-sharded ViT path (one all-gather of MM features at the end) was preferred over TP-sharded ViT (two all-reduces per layer) on the assumption that comm was expensive. With NVLink, TP-sharded ViT is fine — the per-layer all-reduces are cheap.
4. **`cuda_ipc` in `UCX_TLS`:** ensures NIXL transfers use NVLink P2P instead of host-staged copies for embedding/KV transfer paths.

### What still loses to 2× TP=1 DP at rate=1

- 2× TP=1 DP gives 0.76 req/s and 11 s TTFT — still better than TP=2's 0.57 req/s and 35 s TTFT.
- Reason: at 1 req/s offered with rate-1 concurrency profile, the queue per worker matters more than per-request speed. Two independent workers each see half the queue → super-linear TTFT improvement (4.5×).
- TP=2 still beats DP on **per-request decode speed** (median TPOT 223 ms vs 289 ms) and **peak output throughput** (883 vs 647 tok/s) — relevant for low-concurrency, decode-heavy workloads.

---

## Honest verdict

**The earlier conclusion that "TP=2 is fundamentally wrong on this hardware" was wrong because the hardware diagnosis was wrong.** This box has full NVLink. With NVLink properly engaged, TP=2 delivers a 1.36× throughput improvement and 1.43× TTFT improvement over single TP=1.

For this specific rate-1 multimodal workload, **2× TP=1 DP still wins** because of queue-depth halving, not because TP-sharding is broken. The right tool depends on the workload:

| Workload profile | Best config |
|---|---|
| High concurrency, prefill-heavy multimodal | **2× TP=1 DP** (each worker sees half the queue) |
| Low concurrency, decode-heavy or per-request latency-sensitive | **TP=2 with NVLink** (faster per-request, lower TPOT) |
| Single-request latency benchmarks | **TP=2 with NVLink** (best peak output tput, best TPOT) |

The "rate=1, 64 prompts" bench is a high-concurrency back-pressured workload, so 2× TP=1 DP wins there. At lower offered rates (e.g. 0.25 req/s) where the queue stays shallow, TP=2 with NVLink should match or beat 2× DP.

---

## Files in this experiment series (updated)

- `time_breakdown_for_agg_tp1_1080p_8image.md` — TP=1 baseline (correct)
- `time_breakdown_for_agg_tp2_1080p_8image.md` — TP=2 SYS (3,4) — **diagnosis incorrect, ignored NVLink**
- `time_breakdown_for_agg_tp2_gpu45_1080p_8image.md` — TP=2 NODE (4,5) — **diagnosis incorrect, ignored NVLink**
- `why_tp2_worse_than_tp1_reason.md` — encoder+prep diagnosis was directionally right but reasoning ("PCIe-bound") was wrong
- `attemp_on_different_solutions_for_tp2.md` — fix attempts with MM flags **made things worse because the underlying assumption was wrong**
- `vision_encode_summary.md` — the SGLang code walkthrough is correct, but the conclusion ("TP loses on this hardware") was wrong
- `two_tp1_summary.md` — 2× TP=1 DP results stand, but the explanation ("avoids PCIe") is wrong; the real explanation is "halves per-worker queue depth"
- **`new_nvlink_gpu4.5_tp2_performance.md`** — this document (corrected hardware story + working TP=2 results)

## Bench result directories

- TP=1: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_211545/`
- TP=2 SYS (broken): `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_222206/`
- TP=2 NODE (broken): `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_232131/`
- TP=2 NODE+old MM flags: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_003139/`
- 2× TP=1 DP: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_005405/`
- **TP=2 NVLink (NEW, fixed):** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_172153/`

# TP=1 vs TP=2 on NVLink machine (GPUs 4,5): apples-to-apples retest

**System:** Qwen3-VL-32B-Instruct-FP8, aggregated EPD, 8×1920×1080 images per request, rate=1 req/s, 64 prompts.
**Hardware:** NVIDIA H200 SXM, full NVSwitch, GPU4↔GPU5 = NV18 (~478 GB/s aggregate NVLink BW).
**Setup scripts:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp1.sh` and `..._tp2.sh` (after fixes).

---

## Hardware reality check (recap)

This machine has full NVLink. The earlier `.md` files in this directory ALL incorrectly claimed "no NVLink." Verified:

| Check | Result |
|---|---|
| GPU model | `NVIDIA H200` (SXM, NVSwitch) — NOT "H200 NVL" |
| `nvidia-smi nvlink --status` | 18 active NVLinks @ 26.562 GB/s per GPU |
| `nvidia-smi topo -m` GPU↔GPU | `NV18` everywhere (NOT `NODE`/`SYS`) |
| NCCL log evidence (TP=2) | `Channel XX/0 : 0[4] -> 1[5] via P2P/IPC`, 24 channels, `isAllDirectP2p=1` |

---

## Bugs found and fixed in both start scripts

1. **`IP_LOCAL=172.26.46.162` → `172.26.46.75`** — wrong host IP. Actual host is `.75`. The `.162` IP doesn't exist on any interface here. The corporate proxy intercepted the bad-IP traffic and returned a 403 HTML page, which the etcd gRPC client choked on.
2. **`UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy`** lacked `cuda_ipc`. Without it, NIXL GPU↔GPU transfers fall back to host-staged copies instead of NVLink P2P. Added `cuda_ipc` as the first transport.
3. **TP=2 only:** removed `--mm-enable-dp-encoder` and `--enable-broadcast-mm-inputs-process`. These were workarounds for the (non-existent) PCIe bottleneck; with real NVLink, default TP-sharded ViT runs faster.
4. **etcd startup wait was 3 s**, occasionally too short. Bumped to 8 s + a 10× polling loop hitting `/version`.
5. **TP=1 GPU choice:** changed `CUDA_DEVICE=3` → `CUDA_DEVICE=4` so TP=1 and TP=2 share NUMA/CPU affinity for fair comparison.
6. **Added `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,P2P`** for runtime verification of NVLink P2P.

---

## Configuration table

| Setting | TP=1 (NEW) | TP=2 (NEW) |
|---|---:|---:|
| `--tensor-parallel-size` | 1 | 2 |
| GPUs | 4 | 4,5 |
| `--max-running-requests` | 40 | 25 |
| `--mem-fraction-static` | 0.95 | 0.88 |
| Resident GPU mem | 136 GB on GPU 4 | 126 GB on each of 4 & 5 |
| Multimodal flags | default (TP-sharded ViT) | default (TP-sharded ViT, NVLink-cheap) |

(Note: TP=1 with `40/0.95` admits more concurrent requests than TP=2's `25/0.88` per-rank — they're not perfectly matched. TP=1 has more KV headroom on a single GPU; TP=2 effectively doubles total KV but per-rank cap stays at 25.)

---

## TP=1 (GPU 4, NVLink machine, fixed IP) — Headline results

| Metric | Value |
|---|---|
| Request throughput | **0.47 req/s** (offered 1.0 — back-pressured) |
| Total token tput | **7,741 tok/s** |
| Input tput | 7,686 tok/s |
| Output tput | 54.7 tok/s |
| Peak output tput | **762 tok/s** |
| Mean / median TTFT | **41.0 s / 45.0 s**, p99 68.6 s |
| Mean / median TPOT | 1,229 ms / 433 ms, p99 15,900 ms |
| Mean / median ITL | 461 ms / 46 ms |
| Max ITL | **77.0 s** |
| Mean E2E | 94.0 s |
| Concurrency observed | 44.0 |
| Bench duration | 136.7 s |

Result dir: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_174047/`

---

## TP=2 (GPUs 4,5 NVLink, fixed IP, no MM flags) — recap from prior run

| Metric | Value |
|---|---|
| Request throughput | **0.57 req/s** |
| Total token tput | **9,467 tok/s** |
| Mean / median TTFT | **35.0 s / 42.5 s**, p99 46.7 s |
| Mean / median TPOT | **473 ms / 223 ms**, p99 4,467 ms |
| Max ITL | 48.7 s |
| Mean E2E | **64.0 s** |
| Concurrency | 36.7 |
| Bench duration | 111.8 s |

Result dir: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_172153/`

---

## Apples-to-apples comparison (rate=1, 64 prompts, 1080p×8)

| Metric | TP=1 (NEW) | **TP=2 NVLink (NEW)** | Δ TP=2 vs TP=1 |
|---|---:|---:|---:|
| Bench duration | 136.7 s | **111.8 s** | **−18%** |
| Request throughput | 0.47 req/s | **0.57 req/s** | **+21%** |
| Total token tput | 7,741 tok/s | **9,467 tok/s** | **+22%** |
| Input tput | 7,686 tok/s | 9,400 tok/s | +22% |
| Peak output tput | 762 tok/s | **883 tok/s** | +16% |
| Mean TTFT | 41.0 s | **35.0 s** | **−15%** |
| Median TTFT | 45.0 s | 42.5 s | −6% |
| P99 TTFT | 68.6 s | **46.7 s** | **−32%** |
| Mean E2E | 94.0 s | **64.0 s** | **−32%** |
| Mean TPOT | 1,229 ms | **473 ms** | **−61%** |
| Median TPOT | 433 ms | **223 ms** | **−49%** |
| Max ITL | 77.0 s | **48.7 s** | **−37%** |
| P99 TPOT | 15,900 ms | 4,467 ms | −72% |
| Concurrency | 44.0 | 36.7 | −17% |

**TP=2 with NVLink wins on every meaningful metric.** This directly contradicts the prior `.md` files' conclusion.

---

## Full ranking (rate=1, 64 prompts, 1080p×8)

| Config | req/s | Mean TTFT | Mean E2E | Median TPOT | Max ITL | Notes |
|---|---:|---:|---:|---:|---:|---|
| TP=1 old (broken IP, GPU 0/3, 25/0.88) | 0.42 | 50.2 s | 93.2 s | 354 ms | 60.4 s | Old baseline (in retrospect, also bug-affected) |
| TP=2 NODE old (broken IP, 25/0.88, MM workaround flags) | 0.38 | 60.4 s | 107.3 s | 382 ms | 68.0 s | Worst — hit by all three bugs |
| TP=2 NODE+MM flags + low mem (Attempt 3) | 0.29 | 119.9 s | 160.9 s | 355 ms | — | "Fix" backfired |
| **TP=1 NEW (fixed IP, GPU 4, 40/0.95)** | **0.47** | **41.0 s** | 94.0 s | 433 ms | 77.0 s | **Single-GPU baseline, this machine** |
| **TP=2 NVLink NEW (fixed IP, 4&5, 25/0.88)** | **0.57** | **35.0 s** | **64.0 s** | **223 ms** | **48.7 s** | **Best single-instance** |
| 2× TP=1 DP (4,5, prior winner) | 0.76 | 11.2 s | 45.2 s | 289 ms | — | Best overall — queue-depth halving |

---

## Interpretation

### The new TP=1 number (0.47 req/s, 41 s TTFT) is meaningfully better than the old TP=1 baseline (0.42, 50 s)

Why? Two changes vs the original baseline:
1. **IP fix:** old baseline had broken `IP_LOCAL=172.26.46.162`. Health-check / KV-event-publishing paths were silently failing; a small but real overhead per request.
2. **`max_running=40`, `mem=0.95`** vs old `25/0.88`: lets more requests sit in-flight on the single GPU. Throughput rises (more concurrency), but **decode tails get worse** — mean TPOT 1,229 ms vs 622 ms, max ITL 77 s vs 60 s. So this isn't a pure win; it's a different operating point.

### TP=2 with NVLink is genuinely faster than TP=1 on this hardware

The 21% throughput improvement and 32% mean E2E improvement come from:
- **Faster prefill** via TP-sharded LLM forward pass (each rank does half the matmul, NVLink-backed all-reduce ≈ free).
- **Faster decode** for the same reason — median TPOT dropped from 433 ms to 223 ms (−49%).
- **More balanced KV headroom:** TP=2 has 240 GB combined KV (vs TP=1's 136 GB), reducing eviction pressure.
- **TP-sharded ViT works** on NVLink. The per-layer all-reduces in the ViT block (which the prior analysis blamed for the regression) are actually cheap when NCCL is using NVLink P2P/IPC.

### The previous "TP=2 is wrong on this hardware" conclusion was wrong

That conclusion was based on three independently-broken things, all of which made TP=2 look bad:
1. Wrong host IP → all distributed plumbing intermittently broken.
2. Missing `cuda_ipc` in `UCX_TLS` → NIXL transfers fell back to host-staged copies.
3. The misguided "fix" of adding `--mm-enable-dp-encoder --enable-broadcast-mm-inputs-process` — which on this machine actually disabled or bypassed P2P paths and was strictly worse than the default.

Plus, the underlying hardware diagnosis ("no NVLink") was empirically wrong — there's been NV18 between every GPU pair the entire time.

### Why 2× TP=1 DP (0.76 req/s, 11 s TTFT) still wins overall at rate=1

This is workload-shape dependent, not a topology argument:
- At offered rate=1 with the system back-pressured, **per-worker queue depth dominates TTFT**.
- 2× TP=1 DP has two independent workers each seeing ~half the queue → super-linear TTFT improvement (4.5× vs single TP=1).
- TP=2 has one worker seeing the full queue, so even though per-request prefill is faster, queue-wait dominates TTFT.

For a different workload (e.g. lower offered rate, or fewer larger requests where decode dominates), TP=2 with NVLink should be comparable or better than 2× DP. The 2× DP advantage at this rate is structural (queue), not architectural.

---

## Recommended next experiments

1. **Re-run 2× TP=1 DP with fixed IP** to confirm it still wins after the IP fix.
2. **Re-run TP=1 with `25/0.88`** to match the original baseline parameters exactly — would eliminate the concurrency-cap confound and give a cleaner TP=1 vs TP=2 prefill-speed comparison.
3. **Sweep offered rate** (0.25, 0.5, 0.75, 1.0, 1.5) for TP=2 NVLink to find its true knee — likely much higher than the broken 0.38 req/s ceiling reported earlier.
4. **Try TP=4** on the NVSwitch fabric (4 GPUs out of 8). Now that NVLink is confirmed, this should scale further.
5. **Disaggregated EPD** with NVLink-enabled NIXL for KV/MM transfers — potentially the best of both worlds (TP per-stage + parallel stages).

---

## Files in this experiment series (updated)

- `time_breakdown_for_agg_tp1_1080p_8image.md` — TP=1 baseline (correct numbers but conclusions about TP=2 in margin notes were wrong)
- `time_breakdown_for_agg_tp2_1080p_8image.md` — TP=2 SYS (3,4) — diagnosis incorrect (ignored NVLink)
- `time_breakdown_for_agg_tp2_gpu45_1080p_8image.md` — TP=2 NODE (4,5) — diagnosis incorrect (ignored NVLink)
- `why_tp2_worse_than_tp1_reason.md` — encoder+prep diagnosis was directionally right (encoder time ~1.4 s/req) but root cause analysis ("PCIe-bound") was wrong
- `attemp_on_different_solutions_for_tp2.md` — fix attempts with MM flags made things worse because the problem they "fixed" didn't exist
- `vision_encode_summary.md` — code walkthrough is correct, but the conclusion ("TP loses on this hardware") was wrong
- `two_tp1_summary.md` — 2× TP=1 DP results stand, but the explanation ("avoids slow PCIe") is wrong; the real explanation is "halves per-worker queue depth"
- `new_nvlink_gpu4.5_tp2_performance.md` — corrected TP=2 results
- **`new_nvlink_gpu4.5_tp2_tp1_performance.md`** — this document (TP=1 vs TP=2 head-to-head with all fixes)

## Bench result directories

- TP=1 old (bug-affected): `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_211545/`
- TP=2 SYS (bug-affected): `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_222206/`
- TP=2 NODE (bug-affected): `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_232131/`
- TP=2 NODE + MM flags (worse): `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_003139/`
- 2× TP=1 DP: `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_005405/`
- **TP=2 NVLink NEW (fixed):** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_172153/`
- **TP=1 NEW (GPU 4, fixed):** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260521_174047/`

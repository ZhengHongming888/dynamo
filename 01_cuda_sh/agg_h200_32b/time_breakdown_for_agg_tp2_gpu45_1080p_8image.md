# Time-breakdown analysis: Aggregated EPD, TP=2 on same-NUMA NODE pair (GPUs 4,5), 32B FP8, 1080p×8 images, rate=1 req/s

**System under test:** `01_cuda_sh/agg_h200_32b/start_h200_aggregate_epd_server_32b_tp2.sh`
**Model:** Qwen3-VL-32B-Instruct-FP8
**GPU pair:** GPUs **4,5** (NUMA 2, `NODE` connection — same-socket, single PCIe host-bridge hop)
**Benchmark:** `sglang.bench_serving --dataset-name image --image-count 8 --image-resolution 1920x1080 --random-input-len 128 --random-output-len 256 --num-prompts 64 --request-rate 1.0`
**Bench window:** 2026-05-20 23:23:27 – 23:26:18 (~171.5 s analysis window)
**Result file:** `/hongming/res3/h200_agg_tp2_32b_image8_1080p_np_rates/test_sglang_multi_rates_1080p_20260520_232131/rate_1.0/`

---

## Motivation for this run

The previous TP=2 run used GPUs 3,4 — a **cross-socket `SYS`-connected** pair (NUMA 1 ↔ NUMA 2 over QPI/UPI). That gave −22 % TTFT vs TP=1, attributed to the cross-socket all-reduce penalty. This run repeats the same bench on GPUs **4,5**, both in NUMA 2 with a `NODE` (same socket, PCIe-only) connection, to isolate the NUMA crossing as a contributor.

Topology recap from `nvidia-smi topo -m`:
```
        GPU3   GPU4   GPU5
GPU3     X    SYS    SYS
GPU4    SYS    X    NODE      ← previous run used 3,4 (SYS)
GPU5    SYS  NODE    X        ← this run uses 4,5 (NODE)
NUMA     1     2     2
```
**No NVLink anywhere on the box** — `nvidia-smi nvlink --status` empty for all GPUs. All inter-GPU traffic still goes over PCIe.

---

## Headline performance (TP=2 on GPUs 4,5)

| Metric | Value |
|---|---|
| Request throughput | 0.38 req/s (offered 1.0 — back-pressured) |
| Total token tput | 6,241 tok/s |
| Input tput | 6,197 tok/s |
| Output tput | 44.1 tok/s |
| **Peak output tput** | **844 tok/s** ← highest of all three runs |
| Mean / median TTFT | **60.4 s / 65.8 s**, p99 102.9 s |
| Mean / median TPOT | 671 ms / 382 ms, p99 5,202 ms |
| Mean ITL | 406 ms (median 26 ms — best-case decode is fast) |
| Max ITL | 68.0 s |
| Concurrency observed | 40.5 |

---

## Three-way comparison (rate=1, 64 prompts, 1080p×8 images)

| Metric | TP=1 (1 GPU) | TP=2 SYS (3,4) | **TP=2 NODE (4,5)** | 4,5 vs 3,4 | 4,5 vs TP=1 |
|---|---:|---:|---:|---:|---:|
| Bench duration | 152.6 s | 168.4 s | **169.5 s** | +0.7 % | +11 % |
| Request throughput | 0.42 req/s | 0.38 req/s | **0.38 req/s** | ~0 | −10 % |
| Total token tput | 6,933 tok/s | 6,283 tok/s | **6,241 tok/s** | −0.7 % | −10 % |
| Mean TTFT | 50.2 s | 61.0 s | **60.4 s** | −1 % | +20 % |
| Median TTFT | 56.0 s | 66.7 s | **65.8 s** | −1 % | +18 % |
| P99 TTFT | 84.9 s | 101.8 s | **102.9 s** | +1 % | +21 % |
| Mean E2E | 93.2 s | 107.4 s | **107.3 s** | ~0 | +15 % |
| Mean TPOT | 622 ms | 650 ms | **671 ms** | +3 % | +8 % |
| Median TPOT | 354 ms | 407 ms | **382 ms** | −6 % | +8 % |
| Max ITL | 60.4 s | 69.2 s | **68.0 s** | −2 % | +13 % |
| Peak output tput | 592 tok/s | 785 tok/s | **844 tok/s** | +8 % | +43 % |
| Concurrency | 39.1 | 40.8 | **40.5** | ~0 | +4 % |
| **Prefill 8192-chunk median tput** | **9,518** | 8,077 | **7,784** | **−4 %** | **−18 %** |
| Prefill 8192-chunk mean tput | 8,707 | 8,579 | **8,275** | −4 % | −5 % |
| Prefill 8192-chunk max tput | 11,829 | 13,503 | **13,092** | −3 % | +11 % |
| Decode gen tput median | 71 | 84 | **84** | ~0 | +18 % |
| Decode gen tput max | 599 | 702 | **706** | +1 % | +18 % |
| Wall-clock prefill % | 60 % | 91 % | **91 %** | ~0 | +31 pp |
| Wall-clock decode % | 11 % | 9 % | **9 %** | ~0 | −2 pp |
| Wall-clock unaccounted % | 29 % | 0 % | **0 %** | ~0 | −29 pp |
| KV usage | 0.71-0.74 | 0.22-0.24 | **0.22-0.24** | ~0 | (TP=2 doubles cap.) |

---

## Interpretation

**Same-NUMA `NODE` was not the win we hoped for.** Switching from cross-socket `SYS` (3,4) to same-socket `NODE` (4,5):
- Improved median TPOT by 6 % and peak output tput by 8 %
- Made **no measurable difference to the prefill bottleneck**: prefill chunk throughput stayed at ~7.8k tok/s (essentially flat vs 8.1k on SYS, both ~18 % below the TP=1 baseline of 9.5k)
- Same TTFT, same E2E, same overall throughput

### Why didn't `NODE` help?

The H200 NVL on this box has **no NVLink at all**. `NODE` means "PCIe + single PCIe Host Bridge, same NUMA" — i.e. all-reduce still goes over PCIe between two NIC-attached GPUs. Specifically:
- GPU 4 ↔ NIC3: `PIX` (single PCIe bridge)
- GPU 5 ↔ NIC3: `NODE` (different bridge)
- GPU 4 ↔ GPU 5: `NODE`

PCIe Gen5 x16 ≈ ~50 GB/s practical per direction. On a 32B FP8 model the per-chunk all-reduce volume is enough that the link is the binder regardless of whether you cross a socket. The CPU↔CPU UPI hop adds a few microseconds of latency and a small bandwidth tax; for the multi-MB collectives we issue per layer, latency isn't dominant — aggregate PCIe throughput is. So `NODE` only saved ~1 % overall.

### Where `NODE` did help

Decode improved slightly on `NODE` (peak 844 tok/s vs 785, median TPOT 382 ms vs 407 ms), consistent with decode being more latency-sensitive — lots of small all-reduces per token, so reducing per-collective latency helps.

### Why TP=2 still loses to TP=1

1. **Without NVLink/NVSwitch, TP all-reduce is PCIe-bound** at ~50 GB/s. For a 32B model on 16k-token prefill chunks, the per-chunk inter-rank traffic saturates the link.
2. The vision encoder dominates per-request time (gigantic prefill: ~16k vision tokens out of ~16.4k total). TP=2 helps the encoder only modestly while doubling the comm cost.
3. KV cache is barely used (0.22-0.24 with TP=2 doubling capacity to ~240 GB combined) — memory was never the binder; compute + comm were.
4. **Net effect:** TP=2 makes prefill chunk throughput drop ~18 %, and since prefill consumes 91 % of wall-clock, total throughput drops ~10 %.

---

## Bottom line

**TP=2 is not a win for this workload on this hardware, regardless of GPU pair choice.**
- TP=1: 0.42 req/s, 50 s TTFT
- TP=2 across SYS: 0.38 req/s, 61 s TTFT
- TP=2 across NODE: 0.38 req/s, 60 s TTFT

The bottleneck is **PCIe all-reduce bandwidth**, not NUMA crossing. NUMA pairing matters at the ~1 % level; PCIe-vs-NVLink would matter at the 30-50 % level — but there is no NVLink on this box.

TP=2 only wins on **peak output token rate** (+33-43 %), which is useful for low-concurrency decode-heavy workloads, not for this prefill-vision-bound bench.

---

## Recommended next steps (in priority order)

1. **Stay on TP=1** for serving 1080p×8-image requests (0.42 req/s, 50 s TTFT, 9.5 k tok/s prefill).
2. **Use the second GPU as a separate worker** (data-parallel: 2 × single-GPU instances behind the frontend's KV router). Expected: ~0.84 req/s aggregate throughput at the same per-request TTFT, vs 0.38 req/s with TP=2.
3. **Try disaggregated EPD** (encoder on one GPU, prefill+decode on another) — avoids the all-reduce entirely while still parallelizing across two GPUs. Should reduce the 91 % prefill share and especially help TTFT/ITL tails.
4. **Lower per-request vision token count** if the application allows (fewer images, lower resolution). At 8×1080p the workload is wildly prefill-skewed.
5. **Don't try TP=4 or higher** on this box — without NVSwitch, every additional rank adds another PCIe collective without a fast link to amortize it.

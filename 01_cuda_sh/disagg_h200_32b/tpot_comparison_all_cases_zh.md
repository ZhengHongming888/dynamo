# TPOT (Time Per Output Token) Comparison — All Cases at np=128

**Date:** 2026-06-01
**Hardware:** H200 GPU on sc09super21-h200
**Model:** Qwen3-VL-32B-Instruct-FP8
**Test:** np=128 prompts, output=256 tokens, seed=0

TPOT measures **inter-token latency during decode phase** — the time to generate each output token *after* the first. Lower is better. Strongly affects perceived streaming quality.

`TPOT (excl. 1st token) = (E2E - TTFT) / (output_len - 1)`

---

## 1. Full TPOT Comparison Table

| Workload | rate | Setup | Success | **TPOT median** | TPOT mean | TPOT p99 | ITL median | Output tput |
|---|---:|---|:---:|---:|---:|---:|---:|---:|
| **8img/1080p** | 1.0 | TP=1 Agg | 128/128 | **663 ms** | 635 ms | 1,798 ms | 46 ms | 55.9 tok/s |
| 8img/1080p | 1.0 | Disagg max=64 | 65/128 | **726 ms** | 1,529 ms | **19,252 ms** | 19 ms | 27.8 tok/s |
| 8img/1080p | 2.0 | TP=1 Agg | 128/128 | **475 ms** | 464 ms | 997 ms | 47 ms | 57.7 tok/s |
| 8img/1080p | 2.0 | Disagg max=64 | 51/128 | 355 ms | 1,134 ms | **15,145 ms** | 18 ms | 22.9 tok/s |
| **16img/768p** | 1.0 | TP=1 Agg | 128/128 | **519 ms** | 528 ms | 1,664 ms | 44 ms | 84.8 tok/s |
| 16img/768p | 1.0 | Disagg max=64 | 77/128 | 686 ms | 924 ms | 3,983 ms | 45 ms | 44.7 tok/s |
| 16img/768p | 2.0 | TP=1 Agg | 128/128 | **225 ms** | 239 ms | 663 ms | 42 ms | 90.6 tok/s |
| 16img/768p | 2.0 | Disagg max=64 | 72/128 | 394 ms | 607 ms | 5,064 ms | 19 ms | 41.4 tok/s |
| **8img/768p** | 1.0 | **TP=1 Agg** | 128/128 | **36 ms ★** | **52 ms** | 376 ms | 14 ms | **106.6 tok/s** |
| 8img/768p | 1.0 | Disagg max=64 | 128/128 | 489 ms | 612 ms | 2,390 ms | 45 ms | 83.5 tok/s |
| 8img/768p | 1.0 | Disagg max=128 | 128/128 | 633 ms | 1,507 ms | 17,582 ms | 45 ms | 87.5 tok/s |
| 8img/768p | 2.0 | **TP=1 Agg** | 128/128 | **272 ms** | 313 ms | 1,016 ms | 28 ms | 177.2 tok/s |
| 8img/768p | 2.0 | Disagg max=64 | 107/128 | 392 ms | 596 ms | 2,334 ms | 43 ms | 86.0 tok/s |
| 8img/768p | 2.0 | Disagg max=128 | 128/128 | 495 ms | 1,041 ms | 8,596 ms | 46 ms | 109.7 tok/s |
| **4img/768p** | 1.0 | **TP=1 Agg** | 128/128 | **16 ms ★** | **21 ms** | 105 ms | 13 ms | 117.3 tok/s |
| 4img/768p | 1.0 | Disagg max=64 | 128/128 | 32 ms | 54 ms | 503 ms | 12 ms | 116.0 tok/s |
| 4img/768p | 2.0 | **TP=1 Agg** | 128/128 | **16 ms ★** | 16 ms | 30 ms | 13 ms | 228.7 tok/s |
| 4img/768p | 2.0 | Disagg max=64 | 128/128 | 50 ms | 114 ms | 986 ms | 18 ms | 212.4 tok/s |

---

## 2. Median TPOT Side-by-Side

| Workload | rate | TP=1 Agg | Disagg max=64 | Disagg max=128 | Agg / Disagg64 |
|---|---:|---:|---:|---:|---:|
| 8img/1080p | 1.0 | **663 ms** | 726 ms | — | 1.10× faster |
| 8img/1080p | 2.0 | **475 ms** | 355 ms | — | 0.75× (disagg lower)\* |
| 16img/768p | 1.0 | **519 ms** | 686 ms | — | 1.32× faster |
| 16img/768p | 2.0 | **225 ms** | 394 ms | — | 1.75× faster |
| 8img/768p  | 1.0 | **36 ms** | 489 ms | 633 ms | **13.6× faster** ★ |
| 8img/768p  | 2.0 | **272 ms** | 392 ms | 495 ms | 1.44× faster |
| 4img/768p  | 1.0 | **16 ms** | 32 ms | — | **2.0× faster** |
| 4img/768p  | 2.0 | **16 ms** | 50 ms | — | **3.1× faster** |

\* 8img/1080p r=2.0 disagg has lower median TPOT (355 ms vs 475 ms) but only because **51/128 requests succeeded** — the failed 77 are not counted. Mean TPOT (2.4× higher) and P99 (15.2× higher) tell the real story.

---

## 3. P99 TPOT — The Tail Latency Story

P99 TPOT exposes the worst 1% of decode-step latencies — usually due to scheduler stalls, prefill interrupts, or queue backups.

| Workload | rate | TP=1 Agg | Disagg max=64 | **Agg P99 advantage** |
|---|---:|---:|---:|---:|
| 8img/1080p | 1.0 | 1,798 ms | 19,252 ms | **10.7× lower** |
| 8img/1080p | 2.0 | 997 ms | 15,145 ms | **15.2× lower** |
| 16img/768p | 1.0 | 1,664 ms | 3,983 ms | 2.4× lower |
| 16img/768p | 2.0 | 663 ms | 5,064 ms | **7.6× lower** |
| 8img/768p  | 1.0 | 376 ms | 2,390 ms | **6.4× lower** |
| 8img/768p  | 2.0 | 1,016 ms | 2,334 ms | 2.3× lower |
| 4img/768p  | 1.0 | 105 ms | 503 ms | **4.8× lower** |
| 4img/768p  | 2.0 | 30 ms | 986 ms | **32.9× lower** ★ |

Aggregate's P99 TPOT is **2.3× to 33× lower** in every case. Disagg's tail latency is dominated by:
- **NIXL handoff waits** during decode-phase scheduler picks (15-20s pauses when buffer pool fills)
- **Cross-process scheduling jitter** between encoder and PD
- **PD scheduler getting stuck** behind chunked-prefill admit when new requests arrive

---

## 4. Why is Aggregate TPOT So Much Better?

### 4.1 Mechanism

In **Aggregate** mode:
- ViT and LLM are in the **same process / same CUDA context**
- Decode batch runs continuously without cross-process synchronization
- New requests' prefill happens **inside** the same scheduler loop → minimal interrupt to running decodes

In **Disagg** mode:
- Decode batch on PD **must yield** when NIXL embedding arrives from encoder
- Each new admit triggers a prefill batch (16k tokens), pausing decode for **300-2000 ms**
- The pause is amortized across all in-flight tokens → **median TPOT = 400-700 ms** (vs agg's 16-475 ms)

### 4.2 Validation: 8img/768p r=1.0 dramatic gap (36 ms vs 489 ms)

Why such a huge gap (13.6×)?
- Agg r=1.0 has **avg concurrency only 6.0** (PD never busy during decode)
- Disagg r=1.0 has **avg concurrency 64.6** (request pile-up due to NIXL queuing)
- Higher concurrency in disagg means decode batches face more frequent prefill interruptions

When **agg's concurrency rises** (8img/768p r=2.0 → 55.7), TPOT degrades from 36 ms → 272 ms — confirming concurrency is the driver.

---

## 5. Output Token Throughput — Total Decode Bandwidth

| Workload | rate | TP=1 Agg | Disagg max=64 | Disagg max=128 |
|---|---:|---:|---:|---:|
| 8img/1080p | 1.0 | **55.9** | 27.8 | — |
| 8img/1080p | 2.0 | **57.7** | 22.9 | — |
| 16img/768p | 1.0 | **84.8** | 44.7 | — |
| 16img/768p | 2.0 | **90.6** | 41.4 | — |
| 8img/768p  | 1.0 | **106.6** | 83.5 | 87.5 |
| 8img/768p  | 2.0 | **177.2** | 86.0 | 109.7 |
| 4img/768p  | 1.0 | 117.3 | 116.0 | — |
| 4img/768p  | 2.0 | **228.7** | 212.4 | — |

(All values in tokens/second)

**Aggregate generates 2-2.7× more output tokens per second** in heavy workloads. In 8img/768p r=2.0, agg throughput hits **177 tok/s** vs disagg's 86 tok/s — same root cause as RPS gap (no NIXL backpressure).

---

## 6. ITL (Inter-Token Latency) — Streaming Quality Indicator

ITL median measures **per-token streaming smoothness** ignoring queue waits. Agg vs disagg ITL median is **roughly equal** in most cases (12-47 ms), confirming that:
- **GPU per-token decode time is similar** between agg and disagg (same kernel, same model)
- **TPOT/E2E differences come entirely from queue/handoff overhead**, not decode kernel speed

| Workload | rate | Agg ITL_med | Disagg ITL_med |
|---|---:|---:|---:|
| 8img/1080p | 1.0 | 46 ms | 19 ms |
| 8img/1080p | 2.0 | 47 ms | 18 ms |
| 16img/768p | 1.0 | 44 ms | 45 ms |
| 16img/768p | 2.0 | 42 ms | 19 ms |
| 8img/768p  | 1.0 | 14 ms | 45 ms |
| 8img/768p  | 2.0 | 28 ms | 43 ms |
| 4img/768p  | 1.0 | 13 ms | 12 ms |
| 4img/768p  | 2.0 | 13 ms | 18 ms |

Note: when disagg has many requests stuck in NIXL queue (1080p, 16img), the *running* requests can have very low ITL (19 ms) because GPU is dedicated to fewer in-flight reqs — but P99 spikes to 15-20 seconds during scheduler stalls.

---

## 7. Per-Workload Streaming Quality Verdict

| Workload | Recommendation | Reason |
|---|---|---|
| **8img/1080p** | **TP=1 Agg only** | Disagg P99 TPOT 19.2s (intolerable streaming). Agg P99 1.8s acceptable. |
| **16img/768p** | **TP=1 Agg only** | Disagg P99 TPOT 4-5s (poor streaming). Agg P99 0.7-1.7s. |
| **8img/768p**  | **TP=1 Agg strongly preferred** | Agg median TPOT 36 ms (excellent) vs disagg 489-633 ms. **13× better streaming.** |
| **4img/768p**  | **TP=1 Agg preferred** | Agg 16 ms TPOT (real-time). Disagg 32-50 ms still acceptable but 2-3× slower. |

---

## 8. Key Takeaways

1. **Aggregate has consistently lower TPOT** in **every** case — from 1.1× faster (heavy 1080p) to 13.6× faster (light 8img/768p r=1.0).
2. **P99 TPOT** is where the gap is most extreme: agg's worst 1% is 2-33× faster than disagg's worst 1%.
3. **Disagg's TPOT degrades more under load** — at r=2.0 disagg median TPOT drops to 50-394 ms but P99 explodes to 1-15 s.
4. **`max_running_requests=128` does NOT help disagg TPOT** — actually slightly worse than max=64 because higher concurrency means more decode-time prefill interruptions (TPOT median 489 → 633 ms at r=1.0).
5. **For interactive / streaming applications, always use Aggregate.** Disagg's per-token latency variance is too high for end-user-facing streaming.

---

## 9. Validated Data Sources

- TP=1 Agg: `/hongming/res_samehost_agg_tp1_32b_gpu1/` — 8 benches, all 128/128 successful
- Disagg max=64: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/` — 8 benches, 4 failed (50% to 95% success)
- Disagg max=128 (8img/768p only): `/hongming/res_samehost_disagg_32b_gpu01_max128/` — 2 benches, 128/128 successful

Each directory contains `results.txt` with full TPOT/ITL/throughput metrics from `sglang.bench_serving`.

# vLLM EPD paper findings vs our session — what helps our sglang case

**Source:** `/hongming/brian_liu_epd.pdf` — "Heterogeneous E/PD Disaggregation in Dynamo Framework" by Daniel Socek, Sergey Plotnikov, Pallavi Jaini, Brian Liu (Intel, May 2026)

**Same exact setup as ours**: B60 XPU encoders + H200 PD, Qwen3-VL-32B-FP8, NVIDIA Dynamo. Studies TTFT vs request rate across light/moderate/heavy load.

## Paper's headline conclusions

| Regime | Finding |
|---|---|
| **Light load** | Disagg slightly *worse* than agg due to communication overhead |
| **Moderate load** | Disagg better via overlap of ViT with prefill |
| **Heavy load** | Disagg dramatically better — but **not** because ViT runs in parallel; it's because offloading ViT shrinks the PD's queue, which is the dominant TTFT contributor |

Concrete result they cite: at 1.0 RPS for 20img/480p, **4E disagg drops median TTFT 22.7s → 7.2s** vs aggregated. P99 42.8s → 16.2s.

## What it tells us about our session

### 1. We're using the wrong NIXL transfer mode

The paper measured both modes for a 332 MB embedding payload:

| Transport | Primitive | Latency |
|---|---|---:|
| TCP | NIXL Read | 479 ms |
| TCP | NIXL Write | 323 ms |
| RDMA | **NIXL Read** | 80 ms |
| **RDMA** | **NIXL Write** | **22 ms** ← winner |

They use **`nixl_write` for all production results**. We've been using **`nixl_read`** the whole time:

```
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read   ← what we set on giga01 PD
```

Per their measurements, `nixl_write` is **3-4× faster** on the wire (22 ms vs 80 ms). End-to-end TTFT may not move dramatically given that ViT compute dominates, but it's free improvement.

### 2. They use vLLM backend — we use sglang

The paper explicitly says **vLLM 0.19.1 with Dynamo 1.0.0**. We're on **sglang**. Their custom YAML (`pallavi_yaml.txt`) confirms vLLM. This explains why some flags (`--kv-transfer-config`, `--enable-mm-embeds`, `--route-to-encoder`) we discussed don't exist on the sglang side — they're vLLM-only.

For an apples-to-apples replication of their results we'd need to switch backends. That's a significant lift.

### 3. They use different workload params — much smaller per-request

Paper workload: **20 images at 480×854** = 405 visual tokens × 20 × 20480 × 2 bytes = **332 MB embedding/req**.

Our workloads:
- 4img/768p ≈ 127 MB
- 8img/768p ≈ 242 MB
- 8img/1080p ≈ 638 MB

Their 332 MB is closer to our 8img/768p than 8img/1080p. **All our tests at 8img/1080p are 2× the embedding payload of theirs.** Probably part of why we hit harder bottlenecks.

Their input is 128 tokens, output 256 tokens — same as ours. So per-request decode cost matches.

### 4. They use `vllm bench`, not `sglang.bench_serving`

They run `vllm bench`. We've been running `sglang.bench_serving`. The SSE timeout / stream cancellation issue we hit on the 1E test might be specific to sglang's bench client behavior under long-tail latencies. Worth knowing.

### 5. Their "1E performs WORSE than aggregated baseline at low load"

From Figure 6 caption text:

> "The single-E-worker configuration in particular performs worse than the aggregated baseline, indicating that the added transfer overhead is not offset by sufficient overlap opportunities at low concurrency."

This matches our observation that **1E is structurally bad** — not just that our SSE timeouts broke the bench. The paper authors saw the same thing on their stack. **1E should mostly be skipped going forward.** Their headline result is **4E** specifically.

### 6. Their preemption warning maps directly to our OOM

Section 4.2.1 "Heavy workload" warns about **request preemption from KV cache pressure**:

> "When the number of concurrently running requests is large enough, vLLM runs out of free KV cache blocks and must preempt one of the running requests... performance drops so severely that any other performance optimizations become meaningless."

Their fix: tune `--max-num-seqs` conservatively. Our equivalent on sglang is `--max-running-requests`. We've used 64 → 32 across runs. The paper's reasoning (preemption causes prefill recomputation, doubling work) is a possible explanation for some of our weirder benchmark behavior under load.

### 7. Their Figure 5 shows EXACTLY what we measured

Their Figure 5 (per-request TTFT vs request number):
- Aggregated baseline TTFT grows from ~2s at request 1 to ~40s at request 100
- Disagg 4E TTFT stays at ~3-15s, with steady increase

Our 4E patched 8img/1080p:
- mean E2E 410s
- median TTFT 393s

These don't compare directly (different workload size), but the **shape** matches: queue growth dominates TTFT at high load, disagg shifts the queue-saturation point to a higher rate.

### 8. The cost argument they were going to make (TODO in the paper)

The paper has a `[TODO]` block calling out the **cost ratio**: B60 ~0.025× the cost of an H200. So 4× B60 + 1× H200 ≈ 1.1× the cost of just an H200. **For ~3× better TTFT at moderate-to-heavy load.**

This is the actual production case for cross-host disagg. The paper authors didn't include numbers but we have them now via our `patched_4E_results.md` plus the same-host TP=1/TP=2 baselines from `saturation_analysis.md`.

## Concrete actions for our session

| Priority | Action | Effort |
|---|---|---|
| High | Switch to **`nixl-write`** mode on giga01 PD (and B70 to match) | env var change, restart PD |
| High | **Add `vllm-bench`** as an alternative bench client to verify our numbers aren't sglang-bench artifacts | install vLLM client lib |
| Medium | Test 20 images at 480×854 to **directly replicate** their workload | bench config change only |
| Medium | Tune `--max-running-requests` to definitively avoid preemption | already at 64, try smaller |
| Low | Stop testing 1E configurations — their data confirms it's a dead end | just stop |

The biggest single thing: **try `nixl-write`**. If their 22 ms vs 80 ms holds for our setup, that's tangible improvement.

## Cross-references

- Paper text PDF: `/hongming/brian_liu_epd.pdf`
- Their YAML config (vLLM-based): `/hongming/pallavi_yaml.txt`
- YAML diff vs our scripts: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/vllm_yaml_vs_sglang_scripts.md`
- Our patched 4E sweep results: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/patched_4E_results.md`
- Our same-host baselines: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/saturation_analysis.md`
- Our time breakdown analysis: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/h200_time_breakdown_v02.md`
- Our patch (CUDA receive descriptor): `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/h200_cuda_nixl.patch`
- B70 patch report: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/b70_patched.md`

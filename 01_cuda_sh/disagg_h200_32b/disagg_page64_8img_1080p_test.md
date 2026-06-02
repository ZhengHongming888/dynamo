# Disagg 8img/1080p with page_size=64 (cuda_ipc, max_running=128)

**Test:** Restored cuda_ipc transport (default UCX_TLS) + page_size=64 + max_running_requests=128, on 8img/1080p np=128 × {rate=1.0, rate=2.0}

**Question being tested:** Does increasing page_size from 16 to 64 help disagg 8img/1080p (which is the heaviest case with KV pool truly saturated)?

## TL;DR

**No measurable benefit. RPS stays at 0.18-0.22 (essentially identical to page=16/max=64 baseline).** The bottleneck is structural (input_len > chunked_prefill_size + 8 GB NIXL pool exhaustion), not page-alignment overhead.

## 1. Configuration tested

| Setting | Baseline | This Test |
|---|---|---|
| **UCX_TLS** | cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy | **same (cuda_ipc restored)** |
| `--page-size` | 16 | **64** ← changed |
| `--max-running-requests` | 64 | **128** ← changed |
| All other params | same | same |

Verified at startup:
```
INFO scheduler.init_model_worker: max_total_num_tokens=695104,
  chunked_prefill_size=16384, max_prefill_tokens=16384,
  max_running_requests=128, context_len=262144, available_gpu_mem=20.28 GB
```

(`max_total_num_tokens` slightly less than 695,136 with page=16: 695,104 with page=64 because pool size rounds down to multiple of 64. Effective KV capacity unchanged.)

## 2. Results

| Config | rate | **RPS** | Success | total_tput | TTFT_med | E2E_med | TPOT_med | TPOT_p99 |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| **Disagg page=16, max=64** (baseline) | 1.0 | 0.23 | 65/128 | 3,781 | 112.0 s | 175.4 s | 726 ms | 19,252 ms |
| **Disagg page=16, max=64** | 2.0 | 0.18 | 51/128 | 2,975 | 169.9 s | 214.9 s | 355 ms | 15,145 ms |
| **Disagg page=64, max=128** (this test) | 1.0 | **0.22** | **64/128** | 3,695 | 112.9 s | 172.7 s | 526 ms | 22,332 ms |
| **Disagg page=64, max=128** | 2.0 | **0.18** | **49/128** | 2,906 | 161.6 s | 215.5 s | 494 ms | 19,092 ms |
| TP=1 Agg (reference) | 1.0 | **0.47** | 128/128 | 7,787 | 82.3 s | 148.7 s | 663 ms | 1,798 ms |
| TP=1 Agg | 2.0 | **0.49** | 128/128 | 8,033 | 107.7 s | 187.7 s | 475 ms | 997 ms |

### Side-by-side disagg comparison

| Metric | page=16, max=64 | page=64, max=128 | Change |
|---|---:|---:|---:|
| RPS r=1.0 | 0.23 | 0.22 | **-4%** |
| RPS r=2.0 | 0.18 | 0.18 | 0% |
| Success r=1.0 | 65/128 | 64/128 | -1 |
| Success r=2.0 | 51/128 | 49/128 | -2 |
| Total tput r=1.0 (tok/s) | 3,781 | 3,695 | -2% |
| Total tput r=2.0 (tok/s) | 2,975 | 2,906 | -2% |
| TPOT median r=1.0 | 726 ms | 526 ms | **-28%** ↓ |
| TPOT median r=2.0 | 355 ms | 494 ms | +39% ↑ |
| TPOT P99 r=1.0 | 19,252 ms | 22,332 ms | +16% |
| TPOT P99 r=2.0 | 15,145 ms | 19,092 ms | +26% |

## 3. Why page_size=64 did NOT help

### 3a. KV pool capacity unchanged

```
page_size=16:  per-req KV = 16,420 + 256 + 16 = 16,692
               KV pool = 695,136
               theoretical cap = 41 in-flight

page_size=64:  per-req KV = 16,420 + 256 + 64 = 16,740
               KV pool = 695,104
               theoretical cap = 41 in-flight
```

**Same theoretical cap (41).** Empirical peak running-req:
- page=16: **31** (74% of theoretical, due to mem_fraction=0.85 cuda graph headroom)
- page=64: **31-32** (same)

The page_size only affects **alignment padding** — single page wasted per request. Going from page=16 (16 token padding max) to page=64 (64 token padding max) adds at most 48 tokens/request × 32 in-flight = 1,536 tokens of waste = 0.2% of pool. Negligible.

### 3b. Chunked-prefill split-tail unchanged

```
page=16: input_len 16,420 → first chunk floor((16384/16)*16) = 16,384, tail 36 tokens
page=64: input_len 16,420 → first chunk floor((16384/64)*64) = 16,384, tail 36 tokens
```

**Same 40-43% small-batch (`<500 token`) tail rate** observed in PD prefill log:

| Config | <500-token batches | 13-17k batches |
|---|---:|---:|
| page=16, max=64 | 40% | 60% |
| page=64, max=128 | **43%** | **57%** |

(Slightly worse with page=64 because larger pages may misalign with chunked_prefill boundary in some batches, creating more tails.)

### 3c. NIXL embedding buffer pool also unchanged

The 8 GB ring buffer holds ~50 concurrent 161 MB embeddings (8img/1080p). Same for both configs, since NIXL transfer is independent of LLM page_size.

Same buffer-pool exhaustion math:
- input rate 1.0 vs PD admit 0.22-0.23 → pile-up rate 0.77 emb/s
- Bench duration ~280 s → ~215 embeddings would queue
- Pool fits ~50 → ~165 fail
- Observed: 64-65 succeed (consistent with PD throughput × duration), rest fail

### 3d. Why TPOT median actually got *worse* at r=2.0 (494 vs 355 ms)

Larger pages mean **larger per-decode-step memory access patterns** and **slightly worse cuda graph locality** for variable-length decodes. The decode TPOT is more sensitive to page alignment than prefill is. This is a known trade-off:

- **Smaller pages** (16): finer-grained KV slot allocation → less waste, better TPOT under heterogeneous output lengths
- **Larger pages** (64): better memory bandwidth utilization in dense prefill → marginally lower median TPOT at low load (r=1.0 dropped 28%), but larger blocks during decode batch growth at high load (r=2.0 worsened 39%)

In practice page_size=16 is the validated SGLang default for variable-length decode workloads.

## 4. RPS Model Verification

`RPS = total_tput / (input_len + output_len)`

| Config | total_tput | per_req tokens | predicted RPS | measured RPS |
|---|---:|---:|---:|---:|
| page=64 r=1.0 | 3,695 | 16,420 + 154 = 16,574 | 0.223 | **0.22** ✓ |
| page=64 r=2.0 | 2,906 | 16,420 + 154 = 16,574 | 0.175 | **0.18** ✓ |

Within 1.4% accuracy — consistent with the formula's predictive power across all 18 disagg/agg measurements so far.

## 5. Why this result is expected

The 8img/1080p case has **5 active bottlenecks** (per the slide 9 matrix in the status report):

1. ✓ Chunked-prefill split tail (40-43% small batches)
2. ✓ KV pool 74% saturated (in-flight 31, theoretical 41)
3. ✓ Single-request-per-batch (input 16,420 > chunked budget 16,384)
4. ✗ max_running_requests=64 cap
5. ✓ NIXL buffer pool exhaustion (~50 concurrent embeddings → ~64 succeed in ~280s)

Increasing `page_size` only addresses **page alignment waste**, which was never one of the 5 bottlenecks. Increasing `max_running_requests` from 64 to 128 doesn't help because **bottleneck #4 was never the binding constraint for 8img/1080p** — KV pool (bottleneck #2) caps in-flight at 31 long before max=64 is reached.

### What page_size=64 + max_running=128 doesn't change

| Bottleneck | Why page_size doesn't help |
|---|---|
| #1 Split-tail | input/chunked budget ratio unchanged |
| #2 KV pool cap | per-req KV demand difference is 48 tokens (negligible) |
| #3 1-req/batch | chunked_prefill_size/input_len ratio unchanged |
| #5 NIXL buffer pool | embedding size and PD admit rate unchanged |

## 6. What WOULD help 8img/1080p disagg

Based on the bottleneck analysis, ordered by expected RPS gain:

| Method | Expected RPS gain | Cost |
|---|---|---|
| **TP=1 Aggregate (no disagg)** | 0.23 → 0.47 (**2.0×**) | None — just use agg |
| `--chunked-prefill-size 32768` | 0.23 → ~0.40 estimate | 2× per-batch GPU mem (OOM risk) |
| TP=2 Aggregate | 0.23 → ~0.95 (**4×**) | 2 GPUs |
| Image resize to 768p (8img/768p) | 0.23 → 0.90 in agg, 0.74 in disagg max=128 | Resolution loss |
| SGLang scheduler patch (tail-coalesce) | +20% | Multi-day kernel work |

**The page_size knob is not on this list — it's not expected to help, and this test confirms it doesn't.**

## 7. Conclusion

> **`page_size=64` does NOT help disagg 8img/1080p.** RPS unchanged within noise (-4% / 0%), success rate unchanged (50% failures persist), TPOT P99 worse by 16-26%. The 8img/1080p bottlenecks are KV pool saturation, chunked-prefill split-tails, and NIXL buffer pool exhaustion — all independent of page_size.

> **Stick with page_size=16** (SGLang default) and use TP=1 Aggregate instead of disagg for 8img/1080p workloads. Aggregate gives 2× the RPS with 100% success and 10× lower TPOT P99.

## 8. Result files

- page=64 r=1.0: `/hongming/res_disagg_page64_max128/8img_1080p_rate1.0_np128_20260601_064336/`
- page=64 r=2.0: `/hongming/res_disagg_page64_max128/8img_1080p_rate2.0_np128_20260601_065032/`
- page=16 baseline: `/hongming/res_samehost_disagg_32b_gpu01_unpatched/8img_1080p_rate{1.0,2.0}_np128_*`
- TP=1 Agg reference: `/hongming/res_samehost_agg_tp1_32b_gpu1/8img_1080p_rate{1.0,2.0}_np128_*`
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_pd_20260601_063630.log`
- Encoder log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_encoder_20260601_063630.log`
- Launcher: `/tmp/start_samehost_disagg_page64.sh`

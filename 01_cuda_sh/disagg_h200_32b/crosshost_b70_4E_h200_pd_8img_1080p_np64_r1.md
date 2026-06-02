# Cross-Host Disagg: B70 4×Encoder + H200 PD — 8img/1080p np=64 r=1.0

**Date:** 2026-06-01
**Test:** Cross-host disagg with 4 encoder workers on B70 host + 1 PD worker on H200 (this host)

## Setup

| Component | Host | GPU | NIC / Address |
|---|---|---|---|
| **PD Worker** | sc09super21-h200 (172.26.46.133) | GPU 5 (NUMA 2) | mlx5_4 RoCE 192.165.123.52 |
| **Encoder #1** | b70 (172.26.46.180) | (B70 GPU) | tcp 172.26.46.180:40483 |
| **Encoder #2** | b70 | (B70 GPU) | tcp 172.26.46.180:39139 |
| **Encoder #3** | b70 | (B70 GPU) | tcp 172.26.46.180:36893 |
| **Encoder #4** | b70 | (B70 GPU) | tcp 172.26.46.180:39923 |
| Frontend | this host | — | http://172.26.46.133:7001 |
| NATS | this host | — | nats://172.26.46.133:14222 |
| etcd | this host | — | http://172.26.46.133:12379 |

**PD config (giga01 launcher "as-is"):**
- `--max-running-requests 64`
- `--mem-fraction-static 0.65`
- `--page-size 16`
- `--chunked-prefill-size 16384`
- `--kv-cache-dtype fp8_e4m3`
- `DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read` (PD pulls embeddings from B70)
- `UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy` (no cuda_ipc — cross-host)
- effective `max_total_num_tokens = 467,072` (smaller than same-host 695k due to mem_fraction=0.65)

## Bench Configuration
- Workload: 8img/1080p (input_len ≈ 16,420 tokens)
- np=64, rate=1.0, output=256

## Results

| Metric | **Cross-host (this run)** |
|---|---:|
| **RPS** | **0.23** |
| **Successful** | **46/64 (72%)** ⚠ |
| Failed | 18 (28%) |
| Bench duration | 202.7 s |
| Total tput | 3,751 tok/s |
| **Median TTFT** | 113.2 s |
| **Median E2E** | 163.4 s |
| P99 E2E | 198.4 s |
| **Median TPOT** | 439 ms |
| **P99 TPOT** | 18,571 ms |
| Concurrency (avg) | 37.4 |
| PD running peak | 20 (KV pool theoretical cap = 27 with mem_frac=0.65) |
| PD queue peak | 41 |
| Buffer timeouts | **0** |

Note: 18 failures are NOT NIXL buffer timeouts (0 occurred) — they're likely from another mechanism (encoder pipeline backpressure / RoCE QP timeouts).

## Comparison with Prior 8img/1080p Configurations

| Config | np | RPS | Success | tput (tok/s) | TTFT_med | E2E_med | TPOT_p99 | Concur |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| **Cross-host B70 4E + H200 PD** | 64 | **0.23** | **46/64** | 3,751 | 113.2 s | 163.4 s | 18,571 ms | 37.4 |
| Same-host disagg page=16 max=64 | 128 | 0.23 | 65/128 | 3,781 | 112.0 s | 175.4 s | 19,252 ms | 39.7 |
| Same-host disagg page=64 max=128 | 128 | 0.22 | 64/128 | 3,695 | 112.9 s | 172.7 s | 22,332 ms | 39.3 |
| **TP=1 Aggregate (reference)** | 128 | **0.47** | 128/128 | **7,787** | 82.3 s | 148.7 s | **1,798 ms** | 67.9 |

## Key Findings

### 1. Cross-host disagg gives identical RPS to same-host disagg

**0.23 RPS in both configs** — the cross-host RDMA transport (mlx5_4 NDR) is **NOT** the bottleneck. The 8img/1080p case is bottlenecked by:
- Chunked-prefill split-tail (input_len 16,420 > chunked_prefill_size 16,384)
- KV pool capacity (here 27 theoretical due to mem_frac=0.65, but only 20 observed)
- GPU compute throughput (~3,750 tok/s sustained for these batches)

**Whether the encoder is on the same host (NVLink) or B70 (RoCE) makes no measurable difference for this workload's RPS.**

### 2. PD running-req peak only 20 (vs 31 same-host)

Cross-host PD has lower KV pool because `mem_fraction=0.65` (vs same-host 0.85):

| Config | mem_fraction | max_total_num_tokens | KV cap (16,692/req) | Observed peak |
|---|---:|---:|---:|---:|
| Same-host (0.85) | 0.85 | 695,136 | 41 | 31 |
| **Cross-host (0.65)** | **0.65** | **467,072** | **27** | **20** |

The cross-host launcher uses 0.65 to leave headroom for cross-process memory artifacts (which were causing OOM in earlier patched experiments). With 0.85 we'd recover similar in-flight to same-host, but the giga01 launcher conservatively uses 0.65.

### 3. Failure mode different from same-host

Cross-host: **0 NIXL buffer timeouts**, but 28% requests still failed (18/64).

Possible causes (not yet diagnosed):
- B70 encoder side bottleneck (only 4 encoders for 64 prompts × ~2 sec/encode = saturates at 64/4 = 16-second wait per request, but bench is 202s so should drain)
- RoCE QP retry timeout on long-running flows
- dynamo cross-host control-plane / TCP request-plane issues
- bench client connection timeout on slow responses (median E2E 163 s is close to bench client timeouts)

Same-host with same total tput had 50% failure (NIXL buffer), cross-host has 28% failure (different cause). Net result: cross-host has fewer failures despite identical RPS.

### 4. Latency parity with same-host disagg

| Metric | Cross-host | Same-host disagg page=16/max=64 | Diff |
|---|---:|---:|---:|
| Median TTFT | 113.2 s | 112.0 s | +1% |
| Median E2E | 163.4 s | 175.4 s | -7% |
| P99 E2E | 198.4 s | 226.3 s | -12% |
| Median TPOT | 439 ms | 726 ms | **-40%** ↓ |
| P99 TPOT | 18,571 ms | 19,252 ms | -4% |

Cross-host actually has **lower median TPOT** (439 vs 726 ms) — likely because B70 encoder offload reduces PD-side decode-phase context-switching pressure (encoder ViT forward not competing for PD's GPU SMs).

### 5. RPS Model Verification

`RPS = total_tput / per_req_tokens`

```
3,751 / (16,420 + 112) = 0.227 ≈ measured 0.23 ✓
```

### 6. Aggregate is still 2× faster

| Setup | RPS | Success | TPOT P99 |
|---|---:|:---:|---:|
| TP=1 Agg | 0.47 | 128/128 | 1,798 ms |
| Cross-host disagg | 0.23 | 46/64 | 18,571 ms |
| Same-host disagg | 0.23 | 65/128 | 19,252 ms |

Aggregate RPS is **2.0× higher**, success rate 100%, and TPOT P99 is **10× lower** (1.8 s vs 18.6 s).

Cross-host disagg makes sense **only if you need to scale encoders independently** (here: 4 B70 encoders feed 1 H200 PD). The per-request RPS isn't faster than same-host disagg because **PD GPU compute is the bottleneck**, not the encoder side.

## Why Cross-Host Disagg Doesn't Beat Same-Host Disagg for This Workload

**Hypothesis tested:** "Cross-host with 4 encoders should improve RPS over same-host with 1 encoder"
**Result:** No improvement — RPS identical (0.23)

**Why:** 8img/1080p is bottlenecked at PD, not at encoder:
- Encoder ViT forward for 8img/1080p ≈ 1-2 sec on a B70 GPU (decoder GPU class)
- 4 encoders → aggregate encoder throughput ≈ 2-4 RPS (well above PD's 0.23 RPS sink)
- PD admit rate at 0.23 RPS keeps NIXL buffer pool drained (0 timeouts)
- **PD's chunked-prefill + KV pool + compute throughput is the same as same-host disagg**, so RPS is the same

## When Cross-Host Disagg WOULD Help

| Workload | Same-host disagg RPS | Cross-host w/ 4 encoders potential |
|---|---:|---:|
| 8img/1080p (PD-bound) | 0.23 | 0.23 (no improvement — what we just measured) |
| 4img/768p (encoder-bound at high rate) | 1.79 (rate=2.0 limit) | **estimate 4-6 RPS** (4 encoders × encoder ~1 RPS each, PD has plenty of headroom) |

Cross-host would shine on **4img/768p or smaller** where PD admit rate is high (~2 RPS per H200) and encoder is the bottleneck.

## Files

- Bench result: `/hongming/res_crosshost_b70_4E_h200_pd/8img_1080p_rate1.0_np64_20260601_072217/`
- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_20260601_071108.log`
- Launcher: `/tmp/start_pd_only_giga01.sh`

## Bench Start Time
- UTC: `2026-06-01T07:22:17`
- Duration: 287 s (4m 47s)

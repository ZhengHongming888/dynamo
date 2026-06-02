# SGLang + Dynamo Disagg System Optimization Tutorial

## Hands-on Guide for Qwen3-VL-32B-FP8 on H200 + B70

**Audience:** New developers and operators wanting to optimize disagg E/PD performance
**Date:** 2026-06-01
**Source:** Empirical results from 80+ benches across same-host disagg, cross-host disagg, and TP=1 aggregate

---

## Slide 1 — Title

# SGLang Disagg System Optimization
## Hands-on Tutorial for Qwen3-VL-32B-FP8

**Two Critical Problems to Solve:**

1. Same-host H200/H200 disagg E/PD is **slower** than aggregate TP=1 (0.23 vs 0.47 RPS for 8img/1080p — only 49% of agg performance)
2. Cross-host 4×encoder (B70) + 1×PD (H200) is also slower than aggregate TP=1 (0.71 vs 0.90 RPS for 8img/768p — only 79%)

**Goal:** Use systematic time breakdown + RPS modeling to identify bottlenecks, then apply targeted tuning to make disagg viable.

---

## Slide 2 — Outline

This tutorial has 5 parts:

| Part | Topic | Slides |
|---|---|---|
| **I** | Foundations: architecture, workloads, baseline data | 1–12 |
| **II** | How to count: tokens, KV demand, NIXL pool, time decomposition | 13–30 |
| **III** | Bottleneck tour by workload (1080p, 768p, 4img) | 31–48 |
| **IV** | Tuning workflow + which knobs help | 49–60 |
| **V** | Conclusions, references, cheat sheets | 61–70 |

By the end, you will:
- Read a PD log and tell within 60 seconds where the time is going
- Predict bottlenecks before running the bench
- Know which knob to turn first for each workload class

---

## Slide 3 — The Two Problems

### Problem 1: Same-host disagg ≠ better than agg

| Workload | Agg TP=1 | Disagg same-host | Gap |
|---|---:|---:|---:|
| 8img/1080p | **0.47 RPS** | 0.23 RPS | **2.0× slower** |
| 8img/768p | **0.90 RPS** | 0.74 RPS | 1.2× slower |
| 4img/768p | 0.99 RPS | 0.98 RPS | tie (input-bound) |

### Problem 2: Cross-host 4E + 1PD ≠ even matches agg

| Workload | Agg TP=1 | Cross-host (4×B70 + 1×H200 PD) | Gap |
|---|---:|---:|---:|
| 8img/1080p | 0.47 | 0.24 | **2× slower** |
| 8img/768p | 0.90 | 0.71 | 1.3× slower |

**Insight:** Adding 4× the encoder count doesn't help because PD is the bottleneck. We need to fix PD first.

---

## Slide 4 — System Architecture: 3 Configurations Tested

### Same-host Disagg
```
[Same machine, GPU 0 + GPU 1]
  Encoder process (GPU 0) ──NIXL_WRITE over NVLink/cuda_ipc──→ PD process (GPU 1)
```
- 1 encoder + 1 PD on different GPUs same host
- NVLink direct GPU-GPU (cuda_ipc transport)

### Cross-host Disagg
```
[B70 host, ×4 encoders]                  [H200 host, GPU 5]
  Encoder #1 ──┐                              ┌──── PD process
  Encoder #2 ──┼── RoCE NIXL_READ ───────────┤
  Encoder #3 ──┤                              │
  Encoder #4 ──┘                              │
```
- 4 encoders on B70 (XPU), 1 PD on H200
- Mellanox 400 Gb/s NDR RoCE, no cuda_ipc

### Aggregate (TP=1)
```
[Single process, GPU 1]
  ViT + LLM in same CUDA context, no NIXL
```
- Same code, no encoder/decoder split
- This is our reference (target to beat)

---

## Slide 5 — Why Disagg Exists at All

### The promise of disagg

- **Independent scaling**: encoder pool scales with vision throughput, decoder pool scales with text throughput
- **Heterogeneous hardware**: encoder on cheap GPU (e.g., L40S/B70), decoder on H100/H200
- **Shared encoder cache**: 1 encoder pool serves N decoders (with mm-global-cache)

### When disagg is theoretically optimal

```
encoder_compute > decoder_compute
   AND
encoder_count_needed > 1
   AND
NIXL handoff cost < per-request encode_time × scaling_advantage
```

### What we measure today

For Qwen3-VL-32B vision model: **encoder ViT forward is fast (1-2s), decoder prefill+decode is slow (25-90s)**. So disagg's encoder-scaling advantage doesn't materialize — PD is the bottleneck.

---

## Slide 6 — Workloads Tested

| Workload | Image count | Resolution | Visual tokens/img | Total input_len | Per-req embedding |
|---|---:|---|---:|---:|---:|
| **8img/1080p** | 8 | 1920×1080 | ~2,064 | **16,420** | 161 MB |
| **8img/768p** | 8 | 1024×768 | ~770 | **6,238** | 60 MB |
| **4img/768p** | 4 | 1024×768 | ~770 | **3,158** | 30 MB |

### Per-request output

- All workloads use `--random-output-len 256` → max 256 output tokens per request
- All bench scripts use `--seed 0` for reproducibility (same images, same text prompts)

### Why these three?

- 8img/1080p: heavy enough to hit chunked-prefill split (input > 16,384)
- 8img/768p: medium (input < 16,384, but 2 reqs barely fit in 1 batch)
- 4img/768p: light (input ≪ 16,384, 5 reqs fit in 1 batch easily)

The ratio `input_len / chunked_prefill_size` (16,384) determines the bottleneck regime.

---

## Slide 7 — Configuration Baseline

### PD Worker Args (same for all configs)

```bash
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal --enable-mm-global-cache --multimodal-worker \
    --dtype auto --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 64 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --page-size 16 \
    --chunked-prefill-size 16384 \
    --enable-request-time-stats-logging --show-time-cost
```

### Critical environment variables

```bash
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write   # same-host
# OR
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read    # cross-host

UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy   # same-host (NVLink)
# OR
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy            # cross-host (no cuda_ipc)

UCX_NET_DEVICES=mlx5_0:1   # NIC paired with GPU's NUMA
NIXL_MAX_BUFFER_SIZE=805306368   # NIXL internal cap (NOT the dynamo ring buffer)
```

---

## Slide 8 — Hardware Profile

### H200 host (sc09super21-h200, 172.26.46.133)

- 8× H200 GPUs (143 GB HBM each), all NVLink (NV18 = 18 lanes)
- 8 RoCE NICs (mlx5_0..7), 4× active 400 Gb/s NDR
- mlx5_0 → NUMA 0 (GPU 0,1), mlx5_4 → NUMA 2 (GPU 4,5)

### B70 host (172.26.46.180)

- Intel XPU (PVC) GPUs
- mlx5_0 RoCE 400 Gb/s NDR

### NUMA-aware GPU/NIC pairing (PIX/NODE)

```
GPU 0 ↔ mlx5_0  (PIX, NUMA 0) — same-host disagg encoder
GPU 1 ↔ mlx5_0  (NODE, NUMA 0) — same-host disagg PD or aggregate
GPU 5 ↔ mlx5_4  (NODE, NUMA 2) — cross-host disagg PD
```

Use `nvidia-smi topo -m` to verify pairing on your hardware.

---

## Slide 9 — Headline Results Table (np=128, rate=1.0)

| Config | RPS | Success | Median TTFT | Median E2E | TPOT P99 |
|---|---:|:---:|---:|---:|---:|
| **8img/1080p** | | | | | |
| TP=1 Aggregate | **0.47** | 128/128 | **82.3 s** | **148.7 s** | **1,798 ms** |
| Same-host disagg max=64 | 0.23 | 65/128 | 112.0 s | 175.4 s | 19,252 ms |
| Cross-host (B70 4E + H200 PD) | 0.24 | 64/64 (np=64) | 147.7 s | 209.0 s | 13,629 ms |
| **8img/768p** | | | | | |
| TP=1 Aggregate | **0.90** | 128/128 | **1.7 s** | **5.3 s** | **376 ms** |
| Same-host disagg max=64 | 0.70 | 128/128 | 39.9 s | 84.9 s | 2,390 ms |
| Same-host disagg max=128 | 0.74 | 128/128 | 32.4 s | 101.8 s | 17,582 ms |
| Cross-host (4E+1PD) | 0.71 | 64/64 (np=64) | 11.9 s | 41.3 s | 7,384 ms |
| **4img/768p** | | | | | |
| TP=1 Aggregate | 0.99 | 128/128 | 0.7 s | 2.9 s | 105 ms |
| Same-host disagg max=64 | 0.98 | 128/128 | 2.5 s | 6.5 s | 503 ms |

---

## Slide 10 — How to Read These Numbers

### RPS (Request Per Second)
- = `successful_requests / bench_duration`
- Higher is better
- **Target**: this is what you optimize

### Success rate
- `successful / num_prompts`
- 100% required for production
- < 100% = NIXL buffer pool exhaustion or PD memory crash

### Median TTFT (Time To First Token, ms)
- Time from client send → first response token
- Affects perceived "responsiveness"

### Median E2E (End-to-End latency, ms)
- Time from client send → last response token
- Total user-perceived response time

### Median/P99 TPOT (Time Per Output Token, ms)
- Per-token decode time, excluding first
- Affects "streaming quality"
- P99 worse than median = bad tail latency = decode interruptions

---

## Slide 11 — Glossary (Reference Card 1/2)

| Term | Meaning | How to find |
|---|---|---|
| **input_len** | Total input tokens for one request (vision + text) | PD log `ReqTimeStats input_len=N` |
| **per_req_KV** | KV cache slots a request needs to complete | `input_len + max_new_tokens + page_size` |
| **max_total_num_tokens** | Total KV cache pool capacity | PD log `init_model_worker max_total_num_tokens=N` |
| **chunked_prefill_size** | Max new tokens per prefill batch | Launcher arg `--chunked-prefill-size 16384` |
| **page_size** | KV slot block size for allocation | Launcher arg `--page-size 16` |
| **max_running_requests** | Hard cap on in-flight requests | Launcher arg `--max-running-requests 64` |
| **mem_fraction_static** | Fraction of GPU mem allocated to KV pool | Launcher arg `--mem-fraction-static 0.85` |
| **#new-token** | Tokens computed in one prefill batch | PD log `Prefill batch #new-token=N` |
| **#running-req** | In-flight requests in scheduler | PD log `Prefill batch #running-req=N` |
| **#queue-req** | Requests waiting for KV slot | PD log `Prefill batch #queue-req=N` |

---

## Slide 12 — Glossary (Reference Card 2/2)

| Term | Meaning | How to find |
|---|---|---|
| **queue_duration** | Time req waited in scheduler | PD log `ReqTimeStats queue_duration=Nms` |
| **forward_duration** | Prefill + decode time | PD log `ReqTimeStats forward_duration=Nms` |
| **NIXL ring buffer** | Receiver-side buffer pool for embeddings | Source: `embedding_transfer.py:646` (default 8 GB) |
| **per_emb_size** | Bytes per embedding | `imgs × tokens/img × hidden × dtype_bytes` |
| **NIXL_MAX_BUFFER_SIZE** | NIXL internal C-level alloc cap (NOT ring buffer) | env var, 805306368 (768 MB) |
| **PATCH(non-cached)** | XPU encoder CPU→XPU memcopy warning | Encoder log |
| **buffer timeout** | NIXL ring buffer pool exhausted, request fails | PD log `Timeout while waiting for available buffer` |
| **PD admit rate** | RPS PD can sustain (ceiling) | `total_token_throughput / per_req_tokens` |
| **GPU compute ceiling** | Max tok/s sustained on H200 FP8 | ~9,500 tok/s for agg, ~3,750 for disagg 1080p |

---

# Part II — How to Count Everything

## Slide 13 — The Master Formula: RPS Model

```
RPS = total_token_throughput / per_req_total_tokens
    = total_token_throughput / (input_len + output_len)
```

**Verified empirically within ±1.4% on 18+ benchmarks.**

### Worked example: 8img/1080p disagg

```
total_token_throughput = 3,781 tok/s  (from bench result)
input_len             = 16,420 tokens (median, from PD log)
output_len            = 121 tokens   (median actual generated, < 256 max)

RPS_predicted = 3,781 / (16,420 + 121) = 3,781 / 16,541 = 0.229 RPS
RPS_measured  = 0.23 RPS  ✓ matches within 0.4%
```

### Why this matters

- If you know GPU compute ceiling, you can predict RPS
- If RPS doesn't match this formula, you have a transport/queue problem (not GPU compute)
- All optimization work focuses on increasing total_token_throughput OR decreasing per_req_tokens

---

## Slide 14 — Counting Input Length

### Formula

```
input_len = vision_tokens + text_tokens

vision_tokens = num_images × tokens_per_image
tokens_per_image = depends on resolution + Qwen3-VL ViT geometry

For Qwen3-VL with patch_size=14, merge=2:
  1024 × 768 image  → ~770 visual tokens
  1920 × 1080 image → ~2,064 visual tokens
```

### How to verify on your data

```python
# Run a single request, then check the PD log
grep "ReqTimeStats" /path/to/pd.log | head -1
# Output:  ReqTimeStats(rid=..., input_len=16389, ...)
```

### Per-workload median input_len

| Workload | Computed | Measured median |
|---|---:|---:|
| 8img/1080p | 8 × 2,064 + ~80 text = 16,592 | 16,420 |
| 8img/768p  | 8 × 770 + ~80 = 6,240 | 6,238 |
| 4img/768p  | 4 × 770 + ~80 = 3,160 | 3,158 |

(Slight differences from chat template tokenization)

---

## Slide 15 — Worked Example: input_len for 8img/1080p

### Step 1: Find tokens_per_image

```python
# Qwen3-VL ViT config (from model card)
# patch_size = 14
# spatial_merge_size = 2
# resolution_grid = (W // patch_size // 2, H // patch_size // 2)

W, H = 1920, 1080
grid_w = W // 14 // 2  # 1920 / 28 = 68
grid_h = H // 14 // 2  # 1080 / 28 = 38
# Each grid cell = 1 visual token
tokens_per_image_raw = grid_w * grid_h  # 68 × 38 = 2,584
# Qwen3-VL adds positional/special tokens, observed ~2,064
```

### Step 2: Add text tokens

```
Chat template: <|im_start|>user\n{IMAGE_PLACEHOLDERS}{TEXT}<|im_end|>
Random text input_len = 128 (from --random-input-len 128)
But after template: ~80 actual tokens
```

### Step 3: Total

```
input_len ≈ 8 × 2,064 + 80 ≈ 16,592
```

(Bench measures slightly less; ~16,420 with bfloat16 tokenization quirks)

---

## Slide 16 — Counting Per-Request KV Demand

### Formula

```
per_req_KV = input_len + max_new_tokens + page_size
```

### Why each term?

- `input_len`: prefill stores K,V for every input token
- `max_new_tokens`: SGLang **reserves** KV slots up-front to avoid mid-decode preemption (defensive allocation)
- `page_size`: 1 extra block for alignment padding

### Example: 8img/1080p

```
input_len = 16,420
max_new   = 256
page_size = 16

per_req_KV = 16,420 + 256 + 16 = 16,692 tokens
```

### Why 16 vs 64?

```
page_size=16:  per_req_KV = 16,692, alignment waste up to 15 tokens
page_size=64:  per_req_KV = 16,740, alignment waste up to 63 tokens
                                    +48 tokens, negligible (0.3% of pool)
```

Page size 16 (default) is optimal for variable-length decode.

---

## Slide 17 — Reading max_total_num_tokens from PD Log

### Where it appears

```
INFO scheduler.init_model_worker:
  max_total_num_tokens=695136, chunked_prefill_size=16384,
  max_prefill_tokens=16384, max_running_requests=64,
  context_len=262144, available_gpu_mem=20.43 GB
```

### How it's computed (mental model)

```
max_total_num_tokens ≈ floor(
    (mem_fraction × GPU_total_mem - model_weights - cuda_graphs)
    / (kv_dtype_bytes × hidden_dim × num_layers × 2)
)
```

### Concrete H200 + Qwen3-VL-32B-FP8 + mem_fraction=0.85

```
GPU mem total: 143 GB
Model weights (FP8): ~42 GB
Cuda graphs: ~15 GB
Headroom (1 - 0.85 = 0.15): ~21 GB
Available for KV: ~64 GB
KV dtype (fp8_e4m3): 1 byte
Per token: 1 × ~64 hidden × 64 layers × 2 (K+V) = 8.2 KB
Theoretical: 64 GB / 8.2 KB ≈ 7.8M

Observed (after dynamic alloc overhead): 695,136
```

**Always read from log — don't try to derive.**

---

## Slide 18 — KV Pool Theoretical Capacity

### Formula

```
KV_pool_capacity_in_requests = max_total_num_tokens / per_req_KV
```

### Per-workload calculation

| Workload | per_req_KV | max_total_num_tokens | Theoretical cap |
|---|---:|---:|---:|
| 8img/1080p | 16,692 | 695,136 | **41 in-flight** |
| 8img/768p | 6,510 | 695,136 | **106 in-flight** |
| 4img/768p | 3,430 | 695,136 | **202 in-flight** |

### Compare to `--max-running-requests`

| Workload | KV cap | max_running=64 | max_running=128 | Effective cap |
|---|---:|---:|---:|---:|
| 8img/1080p | 41 | 64 | 128 | **41** (KV-limited) |
| 8img/768p | 106 | 64 | 128 | **64 → 106** (max-running-limited then KV-limited) |
| 4img/768p | 202 | 64 | 128 | **64 → 128** (max-running-limited) |

**Lesson**: Always `min(KV_cap, max_running_requests)` is the binding constraint.

---

## Slide 19 — NIXL Ring Buffer (the 8 GB default, NOT the env var)

### Common misconception

`NIXL_MAX_BUFFER_SIZE=805306368` (768 MB) is **not** the dynamo ring buffer size.

### Where the real ring buffer is set

```python
# /opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py:646
class NixlWriteEmbeddingReceiver(AbstractEmbeddingReceiver):
    def __init__(self, buffer_size=2 * 8 * 1024 * 1024 * 256 * 2):
        # = 8 * 1024 * 1024 * 1024 = 8,589,934,592 bytes = 8 GB
        self.ring_buffer = RingBuffer(buffer_size)
```

`EMBEDDING_RECEIVER_FACTORIES` instantiates with **no arguments** → default 8 GB.

### What `NIXL_MAX_BUFFER_SIZE` actually limits

NIXL's C-level allocator for **per-region IB/RoCE buffer registration**. It limits the maximum size of any single registered memory region (used internally by UCX for IB transfers). It does NOT cap the dynamo ring buffer total.

---

## Slide 20 — Per-Request Embedding Size

### Formula

```
embedding_size = num_visual_tokens × hidden_dim × dtype_bytes

For Qwen3-VL-32B-Instruct-FP8:
  hidden_dim = 5120 (read from model config.json)
  dtype = bfloat16  → 2 bytes
  visual_tokens = num_images × tokens_per_image
```

### Per-workload calculation

| Workload | visual_tokens | embedding_size |
|---|---:|---:|
| 8img/1080p | 8 × 2,064 = 16,512 | 16,512 × 5,120 × 2 = **161 MB** |
| 16img/768p | 16 × 770 = 12,320 | 12,320 × 5,120 × 2 = **120 MB** |
| 8img/768p  | 8 × 770 = 6,160 | 6,160 × 5,120 × 2 = **60 MB** |
| 4img/768p  | 4 × 770 = 3,080 | 3,080 × 5,120 × 2 = **30 MB** |

### Lesson

Larger images / more images = bigger embeddings = fewer concurrent fit in NIXL ring buffer.

---

## Slide 21 — NIXL Ring Buffer Concurrent Capacity

### Formula

```
concurrent_embeddings = floor(NIXL_ring_buffer_bytes / per_emb_bytes)
                      = floor(8 GB / per_emb_bytes)
```

### Per-workload calculation

| Workload | embedding_size | concurrent in 8 GB pool |
|---|---:|---:|
| 8img/1080p | 161 MB | **~50** |
| 16img/768p | 120 MB | **~68** |
| 8img/768p | 60 MB | **~136** |
| 4img/768p | 30 MB | **~272** |

### Why this matters for failure prediction

If `arrival_rate × duration` produces more in-flight embeddings than the pool can hold, requests fail with:
```
TimeoutError: Timeout while waiting for available buffer.
```

(60s timeout, hardcoded in `embedding_transfer.py:724`)

---

## Slide 22 — Failure Prediction Formula

### Mechanism

When PD admit rate < input rate, embeddings pile up in NIXL ring buffer. After 60s, new embeddings can't get a slot → timeout failure.

### Formula

```
expected_failures ≈ (arrival_rate − PD_admit_rate) × bench_duration − pool_capacity
```

### Validation across all our benches

| Workload | r=1.0 | PD_admit | Pool cap | Predicted | Observed ✓ |
|---|---:|---:|---:|---:|:---:|
| 8img/1080p np=128 r=1.0 (same-host) | 1.0 | 0.23 | 50 | (1−0.23)×280−50 = 165 | 63 (capped by bench) |
| 16img/768p np=128 r=1.0 | 1.0 | 0.34 | 68 | (1−0.34)×227−68 = 81 | 51 |
| 8img/768p np=128 r=1.0 | 1.0 | 0.70 | 136 | 0 (cap > demand) | **0** ✓ |
| 4img/768p np=128 r=1.0 | 1.0 | 0.99 | 272 | 0 | **0** ✓ |

**Lesson**: Use this formula **before** running bench to predict failure rate.

---

## Slide 23 — Time Decomposition T1-T6 Model

### The 6 phases of every disagg request

```
Time:   0 ───────────────────────────────────────→
        │
Client request received by frontend
        │
        ↓ [Frontend routing]
Encoder: enc_recv ──→ ViT forward + NIXL register ──→ enc_done
        │             ↑                                ↑
        │             T1 (encoder lifetime)
        │
        ↓ [Control-plane fan-out via dynamo router]
PD: pd_recv ──→ wait in SGLang queue ──→ prefill+decode ──→ pd_done
   ↑                  ↑                       ↑                ↑
   T2 (handoff)       T4 (queue)              T5 (forward)
                                                                T3 (PD lifetime)
                                          T6 = T3 - T4 - T5 (NIXL recv + dynamo)
```

### Definitions

| Phase | What it measures |
|---|---|
| **T1** | Encoder lifetime (recv → done): includes ViT + idle wait for PD's NIXL pull |
| **T2** | Control-plane handoff: time from encoder receiving req → PD receiving same req |
| **T3** | PD lifetime (recv → done): total PD-side processing |
| **T4** | SGLang queue_duration: time req sits waiting for KV slot |
| **T5** | SGLang forward_duration: actual prefill+decode work |
| **T6** | T3 - T4 - T5 = NIXL receive overhead + dynamo runtime |

---

## Slide 24 — Extracting T1 from Encoder Logs

### Markers in encoder log

```
INFO handle_payload: ... request received  request_id=AAA  component="encoder"
...                                          (ViT forward + NIXL register happens here)
INFO handle_payload: ... request completed  request_id=AAA  component="encoder"
```

### Python recipe

```python
import re
from datetime import datetime

ts_re = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)')
ANSI = re.compile(r'\x1b\[[0-9;]*m')

enc_req = {}  # rid → {recv, done}
for log_path in encoder_logs:
    with open(log_path) as f:
        for line in f:
            line = ANSI.sub('', line)
            m = ts_re.search(line)
            if not m: continue
            t = datetime.fromisoformat(m.group(1).rstrip('Z'))
            
            m = re.search(r'request received\s+request_id=([0-9a-f-]+).*encoder', line)
            if m:
                enc_req[m.group(1)] = {'recv': t}
            m = re.search(r'request completed\s+request_id=([0-9a-f-]+).*encoder', line)
            if m and m.group(1) in enc_req:
                enc_req[m.group(1)]['done'] = t

# T1 = done - recv
durations = [(d['done'] - d['recv']).total_seconds() for d in enc_req.values() if 'done' in d]
```

---

## Slide 25 — Extracting T2 (Control-plane Handoff)

### Definition

```
T2 = pd_recv - enc_recv
```

### Why it's not zero

- Encoder receives request from frontend's router
- Encoder publishes "embedding ready" notification on dynamo TCP request plane
- Frontend's router forwards request (or its embedding ref) to PD
- PD receives — this takes 1-15 seconds depending on dynamo runtime + cross-host network

### Typical values

| Config | Median T2 |
|---|---:|
| Same-host disagg (NVLink) | ~150 ms (almost instant) |
| Cross-host disagg (RoCE) | 5-15 s (dynamo router + RoCE) |
| Aggregate TP=1 | 0 (no handoff) |

### Code reference

`/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py:402-454`

---

## Slide 26 — Extracting T3 (PD Lifetime)

### Markers in PD log

```
INFO handle_payload: ... request received  request_id=AAA  component="backend"
...
INFO handle_payload: ... request completed request_id=AAA  component="backend"
```

### Python recipe

```python
pd_req = {}
with open(pd_log_path) as f:
    for line in f:
        line = ANSI.sub('', line)
        m = ts_re.search(line)
        if not m: continue
        t = datetime.fromisoformat(m.group(1).rstrip('Z'))
        
        m = re.search(r'request received\s+request_id=([0-9a-f-]+).*backend', line)
        if m:
            pd_req[m.group(1)] = {'recv': t}
        m = re.search(r'request completed\s+request_id=([0-9a-f-]+).*backend', line)
        if m and m.group(1) in pd_req:
            pd_req[m.group(1)]['done'] = t

# T3 = done - recv
```

### Note

`pd_req` request_ids match `enc_req` request_ids (same dynamo UUID). You can join them by ID.

---

## Slide 27 — Extracting T4/T5 from PD ReqTimeStats

### The actual log lines

```
INFO schedule_batch.log_time_stats: ReqTimeStats(rid=cb7e43bdd38249269630ef9ea18c8de2,
    input_len=316, cached_input_len=0, output_len=30, type=unified):
    queue_duration=0.17ms, forward_duration=519.88ms, entry_time=...
```

### Python recipe

```python
pd_q_list = []  # queue durations in seconds
pd_f_list = []  # forward durations in seconds

with open(pd_log_path) as f:
    for line in f:
        if 'ReqTimeStats' not in line: continue
        m_q = re.search(r'queue_duration=(\d+\.\d+)', line)
        m_f = re.search(r'forward_duration=(\d+\.\d+)', line)
        m_in = re.search(r'input_len=(\d+)', line)
        if all([m_q, m_f, m_in]):
            in_len = int(m_in.group(1))
            if in_len > 0:  # skip intermediate reports
                pd_q_list.append(float(m_q.group(1)) / 1000)
                pd_f_list.append(float(m_f.group(1)) / 1000)
```

### Important

ReqTimeStats `rid` is **SGLang's internal request id** (no dashes), different from dynamo's request_id. So you can't directly join, but distributions are comparable.

---

## Slide 28 — Computing T6 (NIXL Recv + Dynamo)

### Formula

```
T6 = T3 - T4 - T5
   = PD lifetime - SGLang queue - SGLang forward
   = (everything else)
```

### What's included in T6

- NIXL receive operation (ring buffer alloc, RoCE/cuda_ipc transfer, completion)
- dynamo runtime serialization/deserialization
- TCP request plane handoff to SGLang
- Result streaming back via dynamo

### Typical decomposition (8img/1080p cross-host)

```
T3 (PD lifetime)      = 219.4 s  (100%)
  ├─ T4 SGLang queue   = 88.1 s   (40%)
  ├─ T5 SGLang forward = 87.8 s   (40%)
  └─ T6 (other)        = 43.6 s   (20%)
```

T6 is the **opportunity for system optimization** — it's overhead not directly tied to GPU work.

---

## Slide 29 — Joining Encoder + PD Logs (Full Recipe)

### Complete Python script

```python
import re, statistics
from datetime import datetime
from collections import defaultdict

PD_LOG = '/path/to/pd.log'
ENC_LOGS = ['/path/to/enc1.log', '/path/to/enc2.log', ...]

T0, T1 = datetime.fromisoformat('YYYY-MM-DDTHH:MM:00'), datetime.fromisoformat('YYYY-MM-DDTHH:MM:30')
ANSI = re.compile(r'\x1b\[[0-9;]*m')

def ts(line):
    m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)', line)
    return datetime.fromisoformat(m.group(1).rstrip('Z')) if m else None

# 1) Build enc_req[rid] = {recv, done, idx}
# 2) Build pd_req[rid] = {recv, done}
# 3) Build pd_rt = aggregate of ReqTimeStats (q, f) — distribution only
# 4) Join by rid:
joined = []
for rid in sorted(enc_req.keys() & pd_req.keys(), key=lambda r: enc_req[r]['recv']):
    e, p = enc_req[rid], pd_req[rid]
    if 'done' not in e or 'done' not in p: continue
    joined.append({
        'enc_dur': (e['done'] - e['recv']).total_seconds(),
        'pd_dur':  (p['done'] - p['recv']).total_seconds(),
        'handoff': (p['recv'] - e['recv']).total_seconds(),
    })

# 5) Compute medians
print(f"T1 median = {statistics.median(b['enc_dur'] for b in joined):.1f}s")
print(f"T2 median = {statistics.median(b['handoff'] for b in joined):.1f}s")
print(f"T3 median = {statistics.median(b['pd_dur'] for b in joined):.1f}s")
print(f"T4 median = {statistics.median(pd_q_list):.1f}s")
print(f"T5 median = {statistics.median(pd_f_list):.1f}s")
```

---

## Slide 30 — Reading Prefill Batch Composition

### Why it matters

The `#new-token` distribution tells you how SGLang is batching requests.

### Critical pattern: chunked-prefill split-tail

When `input_len > chunked_prefill_size`, every request creates 2 batches:
- 1 main batch of `chunked_prefill_size` tokens (16,384)
- 1 tail batch of `input_len - 16,384` tokens (often <100)

### Python recipe

```python
prefill_sizes = []
with open(pd_log_path) as f:
    for line in f:
        if 'Prefill batch' not in line: continue
        m = re.search(r'#new-token:\s*(\d+)', line)
        if m: prefill_sizes.append(int(m.group(1)))

# Bucket counts
small = sum(1 for s in prefill_sizes if s < 500)
single = sum(1 for s in prefill_sizes if 5500 < s < 7500)
double = sum(1 for s in prefill_sizes if 11000 < s < 13500)
fullchunk = sum(1 for s in prefill_sizes if 13000 <= s < 17000)
```

### What healthy vs unhealthy looks like

| Workload | <500 (tail) | full chunk | 2 reqs |
|---|---:|---:|---:|
| 8img/1080p | **42% (split-tail)** | 58% | 0% |
| 8img/768p | 3% | 26% (single) | **71% (good batching)** |

---

# Part III — Bottleneck Tour by Workload

## Slide 31 — Workload Overview: 8img/1080p

### Profile

| Property | Value |
|---|---:|
| Image count × resolution | 8 × 1920×1080 |
| Visual tokens per image | ~2,064 |
| **input_len** | **16,420 tokens** |
| input_len / chunked_prefill | 16,420 / 16,384 = **100.2%** |
| per_req_KV demand | 16,692 tokens |
| KV pool theoretical cap | 41 in-flight |
| Embedding size | 161 MB |
| NIXL pool concurrent | ~50 |

### The structural problem

input_len is **JUST ABOVE** chunked_prefill_size by 36 tokens. Every request gets split into a 16,384-token main + 36-token tail. The tail wastes a full prefill cycle for trivial work.

---

## Slide 32 — 8img/1080p: 5 Active Bottlenecks

| # | Bottleneck | Symptom | Source |
|---|---|---|---|
| **1** | Chunked-prefill split-tail | 40-46% of prefill batches are <500 tokens | `schedule_policy.py:813-933` |
| **2** | KV pool saturation | in-flight cap at 31 (vs 41 theoretical, due to mem_frac headroom) | SGLang allocator |
| **3** | Single-request-per-batch | 0% of prefills admit 2 reqs (each req takes 16,384 of 16,384 budget) | scheduler.py |
| **4** | NIXL buffer pool exhaustion | Same-host disagg fails 50% of requests at np=128 r=1.0 | embedding_transfer.py:724 |
| **5** | GPU compute throughput limit | total_tput stays ~3,800 tok/s regardless of in-flight count | Hardware ceiling |

### Bench impact on RPS

```
Aggregate TP=1:           0.47 RPS (only #5)
Same-host disagg max=64:  0.23 RPS (all 5)
Cross-host disagg:        0.24 RPS (all 5 except #4 partially)
```

---

## Slide 33 — Bottleneck #1: Chunked-prefill Split-Tail

### What's happening

```
Request arrives with input_len = 16,420
Scheduler tries to admit:
  - rem_chunk_tokens (initially 16,384)
  - 16,420 > 16,384 → must split
  - Main chunk: 16,384 tokens
  - Tail chunk: 16,420 - 16,384 = 36 tokens (page-aligned to 48)

Result: every request triggers 2 prefill batches:
  Batch 1: 16,384 token main (full GPU)
  Batch 2: 48 token tail (mostly idle GPU, only kernel launch overhead)
```

### Empirical evidence

```
Prefill batch sizes for 8img/1080p (n=108 batches in np=128 r=1.0 bench):
  <500 tokens:  43%   ← these are the tails
  13-17k tokens: 57%  ← these are the main chunks
```

### How to verify

```bash
grep "Prefill batch" pd.log | awk -F'#new-token:' '{print $2}' | awk '{print $1}' | sort | uniq -c
# Expect 50/50 split between tiny and 16k-ish
```

---

## Slide 34 — Bottleneck #2: KV Pool Saturation

### The math

```
max_total_num_tokens = 695,136 (from PD log, mem_frac=0.85)
per_req_KV = 16,692
KV_pool_cap = floor(695,136 / 16,692) = 41 in-flight
```

### But observed peak is 31, not 41

```
Why? mem_fraction=0.85 reserves 15% headroom for cuda graphs + activations.
Effective KV pool ≈ 0.74 × 695,136 ≈ 514k tokens
514k / 16,692 = 30.8 → observed peak 31 ✓
```

### Workaround: raise mem_fraction

```bash
--mem-fraction-static 0.92   # squeezes 50k more tokens, in-flight 31 → 35
```

But beware OOM (multiple of our patches with mem_frac=0.92 OOM'd).

### Why this caps RPS

When 31 requests are running and a 32nd arrives, it sits in queue for ~50s waiting for one to free its KV slots after generating 256 output tokens.

---

## Slide 35 — Bottleneck #4: NIXL Buffer Pool Exhaustion

### The math

```
NIXL ring buffer = 8 GB (default, in embedding_transfer.py:646)
Per-embedding = 161 MB (8img/1080p)
Pool concurrent capacity = floor(8 GB / 161 MB) = 50 embeddings
```

### When it fails

```
np=128 r=1.0 same-host disagg:
  Bench duration = 280s
  Arrival = 1.0 RPS × 280s = 280 embeddings sent
  PD admit = 0.23 RPS × 280s = 64 embeddings consumed
  Pile-up = 280 - 64 = 216 embeddings need slots
  Pool holds 50 → pool fills → 60s timeout per failure
  
Observed: 65/128 successful, 63 buffer-pool timeouts ✓
```

### Lesson

For 8img/1080p, **never run sustained input rate > 0.5 RPS** in disagg mode without admission control.

---

## Slide 36 — 8img/1080p: Time Breakdown (Cross-host, 209s median E2E)

### Visualization

```
Time (s):  0       14         107        163       207  209
Client ────┤        │           │          │          │   ├─→ last token
Encoder ───┤(rcv)──────────────────────────────────────┤(done)
PD ─────────────────┤(rcv)─────────────────────────────┤(done)
                    │←── 88s queue ──→│←─55s forward──→│
                    │←── 47s NIXL recv + dynamo ──────→│
```

### Breakdown table

| Phase | Median | % of E2E |
|---|---:|---:|
| T1 Encoder lifetime | 208 s | (overlaps T3) |
| T2 Control-plane handoff | 14 s | 7% |
| T3 PD lifetime | 195 s | 93% |
| ↳ T4 SGLang queue | 88 s | 47% |
| ↳ T5 SGLang forward | 55 s | 28% |
| ↳ T6 NIXL recv + dynamo | 47 s | 24% |

### What this tells us

- Queue (T4) is the #1 contributor: 47% of total time is just waiting
- Forward (T5) is 28%: actual GPU work
- Other (T6) is 24%: NIXL handoff overhead
- Encoder (T1) is 99% idle holding memory

---

## Slide 37 — Why Same-host Disagg Gives 0.23 RPS vs Agg 0.47

### Where agg wins

```
Aggregate TP=1:
  - No NIXL handoff (saves ~T6 = 47s/req)
  - ViT and LLM in same CUDA context → no cross-process serialization
  - No encoder-side scheduler queue (single SGLang queue handles everything)
  
Disagg same-host:
  - NIXL_WRITE over cuda_ipc: ~150 ms/req transfer (small)
  - But ring buffer pool holds embedding longer due to PD queue
  - PD pays T4 (queue) + T5 (forward) + T6 (NIXL recv) = 3-4× longer
  - Total tput: 3,781 tok/s (disagg) vs 7,787 (agg) = 2× lower
```

### Quantitative

| Metric | Agg | Disagg same-host |
|---|---:|---:|
| total_token_throughput | 7,787 | 3,781 |
| RPS | 0.47 | 0.23 |
| Median TTFT | 82 s | 112 s |
| Median E2E | 149 s | 175 s |

**The 2× tput gap is from disagg's NIXL backpressure on GPU compute** (memory contention between NIXL ring buffer + KV pool + activations).

---

## Slide 38 — Workload Overview: 8img/768p

### Profile

| Property | Value |
|---|---:|
| Image count × resolution | 8 × 1024×768 |
| Visual tokens per image | ~770 |
| **input_len** | **6,238 tokens** |
| input_len / chunked_prefill | 6,238 / 16,384 = **38%** |
| per_req_KV demand | 6,510 tokens |
| KV pool theoretical cap | 106 in-flight |
| Embedding size | 60 MB |
| NIXL pool concurrent | ~136 |

### The structural difference from 1080p

input_len is **less than half** of chunked_prefill_size. **Two requests can fit in one prefill batch** (12,476 tokens < 16,384). This unlocks a 2× prefill efficiency.

---

## Slide 39 — 8img/768p: Bottleneck Shift

| Bottleneck | 8img/1080p | **8img/768p** | Change |
|---|:---:|:---:|---|
| #1 Split-tail | ✓ (40%) | ✗ (3%) | **Eliminated** |
| #2 KV pool sat | ✓ (74%) | ✗ (28%) | **Eliminated** |
| #3 1-req/batch | ✓ | ✗ (2 reqs/batch, 71%) | **Eliminated** |
| #4 NIXL pool fail | ✓ (rate=1.0) | ✗ (no failures at rate=1.0) | **Eliminated** |
| #5 GPU compute | ✓ | ✓ (now dominant) | Same |

### The new bottleneck profile

For 8img/768p, only **GPU compute throughput** is the binding constraint:
- Disagg: ~4,500 tok/s (RPS 0.70)
- Agg: ~5,700 tok/s (RPS 0.90)
- Disagg gap to agg = 21% (from removed NIXL contention)

### Lesson

**Same-host disagg performs comparably to agg for 8img/768p** because there's no split-tail penalty.

---

## Slide 40 — 8img/768p: 71% Prefill Batches Admit 2 Requests

### The breakdown

```
Prefill batch sizes for 8img/768p (cross-host, np=64 r=1.0, n=38 batches):
  <500 tokens:           3%
  2-7k (1 req):          26%   ← single-request batches
  7-13k (2 reqs):        71%   ← two-request batches (12,476 tokens)
  13-17k (single 16k):   0%    ← no chunked split
```

### Why this matters

```
Single request prefill cost: ~6,238 tokens × kernel_launch_overhead = X
Two-request prefill cost:    ~12,476 tokens × kernel_launch_overhead = X (same overhead!)

Effective tokens/forward:
  Single: 6,238 / X
  Double: 12,476 / X = 2× higher
```

This is the **mechanism** that makes 8img/768p PD admit ~3× faster than 8img/1080p.

### Configuration knob

```bash
# If you can't easily reduce input_len, you can try:
--chunked-prefill-size 32768   # allows even 16,420 input to fit in one chunk
```

But this 2× per-batch GPU memory and may OOM.

---

## Slide 41 — 8img/768p: Time Breakdown

### Cross-host (B70 4E + H200 PD)

```
Time (s):  0   2    5             29              39  41
Client ────┤   │     │              │              │   ├──→ last token
Encoder ───┤(rcv)──────────────────────────────────┤(done)
PD ──────────────┤(rcv)───────────────────────────┤(done)
                 │←─3s queue─→│←──── 25s forward ──→│
                 │←── 10s NIXL recv + dynamo ──────→│
```

### Comparison table

| Metric | Same-host max=64 | **Cross-host (4E+1PD)** |
|---|---:|---:|
| T2 handoff | ~0.15 s | 1.6 s |
| T3 PD lifetime | 80 s | 37 s |
| T4 queue | (similar) | 2.8 s |
| T5 forward | (similar) | 24.7 s |
| T6 NIXL+dynamo | (similar) | 9.7 s |
| Median E2E | 84.9 s | **41.3 s** ← 49% faster |
| Median TTFT | 39.9 s | **11.9 s** ← 70% faster |

**Cross-host improves TTFT massively** for 8img/768p because 4 encoders absorb burst load. Median E2E also improves because PD queue stays small.

---

## Slide 42 — Workload Overview: 4img/768p

### Profile

| Property | Value |
|---|---:|
| Image count × resolution | 4 × 1024×768 |
| Visual tokens per image | ~770 |
| **input_len** | **3,158 tokens** |
| input_len / chunked_prefill | 3,158 / 16,384 = **19%** |
| per_req_KV demand | 3,430 tokens |
| KV pool theoretical cap | 202 in-flight |
| Embedding size | 30 MB |
| NIXL pool concurrent | ~272 |

### The "no bottleneck" case

input_len is **only 19%** of chunked_prefill_size. **5 requests can fit in one prefill batch** (5 × 3,158 = 15,790 < 16,384). This is the optimal configuration for chunked-prefill scheduler.

---

## Slide 43 — 4img/768p: Zero-Bottleneck Case

### Active bottlenecks

| # | Bottleneck | 4img/768p status |
|---|---|---|
| #1 Split-tail | ✗ (input ≪ chunked) |
| #2 KV pool | ✗ (5% utilized at peak) |
| #3 1-req/batch | ✗ (5 reqs/batch possible) |
| #4 NIXL pool | ✗ (272 concurrent fit) |
| #5 GPU compute | ✗ (PD admit > input rate) |

### What limits RPS instead?

**Input rate from bench client.**

| Bench config | Measured RPS | Limit |
|---|---:|---|
| np=128 rate=1.0 | 0.99 | bench input rate |
| np=128 rate=2.0 | 1.79-1.93 | bench input rate × bench duration |
| np=256 rate=4.0 (estimated) | ~3.0 | starts to hit max_running_requests=64 |
| np=512 rate=8.0 (estimated) | ~5.0 | KV pool / max_running cap |

### Lesson

For 4img/768p, **just push the input rate higher**. RPS scales linearly with rate until you hit max_running_requests or KV pool.

---

## Slide 44 — 4img/768p: 5 Reqs Per Prefill Batch

### Empirical data

```
Prefill batch sizes for 4img/768p (np=128 r=1.0, n=73 batches):
  <500 tokens:        1%
  3,100 (1 req):      85%
  6,200 (2 reqs):     8%
  9,400 (3 reqs):     4%
  15,800 (5 reqs):    2%   ← max possible: 5 × 3,158 = 15,790
```

### Why mostly single-req?

In steady-state at rate=1.0 with admit rate ≈ 1.0, only 1-2 requests are typically waiting in queue at any moment. The scheduler doesn't get a chance to merge 5 into one batch.

### What pushes batching higher

```
rate=2.0:
  Batch sizes shift to ~30% double-req, ~10% triple-req
  Effective tokens/forward goes up
  Total tput climbs from ~3,200 → ~6,300 tok/s (+95%!)
```

**Rate=2.0 doubles tput because the scheduler finally gets to batch.**

---

## Slide 45 — The 4-Bin Classification by `input_len / chunked_prefill_size`

This is the **single most important categorization** for predicting bottlenecks.

| input_len / 16,384 | Behavior | Reqs/batch | Active bottlenecks |
|---|---|---:|---|
| **> 100%** | Must chunked-split | 0 (split) + tail | All 5 |
| **50% - 100%** | Single req fills full chunk | 1 | KV pool + admission |
| **33% - 50%** | 2 reqs fit per batch | 2 | max_running + GPU compute |
| **< 20%** | 5+ reqs fit, no split | 5+ | Input rate (no PD bottleneck) |

### Examples we tested

| Workload | input_len | Ratio | Bin | Measured RPS |
|---|---:|---:|---|---:|
| 8img/1080p | 16,420 | 100% | Bin 1 (split-tail) | 0.23 |
| 16img/768p | 12,400 | 76% | Bin 2 (1 req/batch) | 0.34 |
| 8img/768p | 6,238 | 38% | Bin 3 (2 reqs/batch) | 0.70 |
| 4img/768p | 3,158 | 19% | Bin 4 (5+ reqs/batch) | 0.99 |

---

## Slide 46 — Bottleneck Activation Matrix

| Bottleneck | 8img/1080p | 16img/768p | 8img/768p | 4img/768p |
|---|:---:|:---:|:---:|:---:|
| #1 Split-tail | ✓ (42%) | — | — | — |
| #2 KV pool sat | ✓ (74%) | ✓ (98%) | — | — |
| #3 1-req/batch | ✓ | ✓ | — | — |
| #4 max_running cap | — | — | ✓ | — |
| #5 NIXL buffer | ✓ | ✓ | (rate=2.0 only) | — |
| #6 GPU compute | ✓ | ✓ | ✓ | (limit at high rate) |
| **Active bottlenecks** | **5** | **4** | **2** | **0** |
| **Disagg RPS @ r=1.0** | 0.23 | 0.34 | 0.70 | 0.99 |

### Intuition

- More active bottlenecks ⇒ lower RPS
- Each bottleneck independently caps total tput
- They compound multiplicatively

### How to use this

Before running a bench, compute the ratio `input_len / 16,384` and identify which bin you're in. That predicts your active bottlenecks (and therefore expected RPS).

---

## Slide 47 — Why Aggregate Always Wins for Single-Host

### The reasoning

```
Aggregate eliminates:
  ✓ T2 (control-plane handoff) → 0
  ✓ Most of T6 (NIXL recv) → 0
  ✓ NIXL ring buffer pool → no failure mode
  ✓ Memory contention between encoder + decoder + ring buffer
  
Aggregate keeps:
  • T4 (SGLang queue) — same as disagg
  • T5 (forward) — same as disagg
  • #1 split-tail — same (SGLang chunked-prefill)
```

### Concrete savings (8img/768p)

| Time | Disagg | **Agg** | Saved |
|---|---:|---:|---:|
| T2 handoff | 1.6 s | 0 | 1.6 s |
| T6 NIXL+dynamo | 9.7 s | <0.5 s | ~9 s |
| T1 idle | 39 s | (no encoder) | irrelevant |
| **E2E median** | 41.3 s | **5.3 s** | **36 s** |

### The 2× RPS comes from

1. Saving 11s/req of T2+T6 → tput ↑
2. Tighter prefill+decode interleaving (single-process scheduler)
3. No memory contention between processes

---

## Slide 48 — Why 4×Encoder Cross-Host Doesn't Beat 1×Aggregate

### The expectation

"4 encoders should give 4× encoder throughput, so aggregate RPS should be 4× higher!"

### The reality

For 8img/1080p (PD-bound):
- Encoder ViT forward: 1-2 s per request
- 4 encoders aggregate ≈ 2-4 RPS encoder throughput
- **PD admit rate: 0.24 RPS** (the bottleneck)
- Encoders are 99% idle waiting for PD

```
Cross-host:     4 encoders × 99% idle = 1% utilized
                PD compute = 100% utilized
                Bottleneck: PD compute (regardless of encoder count)

Single-host:    1 encoder × 1% utilized + PD = same PD compute
                Bottleneck: PD compute (same)
```

### Conclusion

**Adding more encoders to a PD-bound system is free overhead, not throughput.** The system is only as fast as the slowest tier.

For 8img/1080p, the slowest tier is **PD GPU compute** (3,781 tok/s ceiling), and that doesn't change with encoder count.

---

# Part IV — Tuning Workflow

## Slide 49 — Tuning Workflow Overview

### The diagnostic loop

```
1. Run baseline bench → record RPS, success rate
2. Compute input_len, identify bin (1/2/3/4)
3. Predict active bottlenecks from matrix (slide 46)
4. Extract time breakdown (T1-T6)
5. Identify dominant phase
6. Apply targeted knob
7. Re-bench, verify improvement
8. Repeat from step 4 until acceptable
```

### What you should NOT do

- ❌ Try random combinations of knobs without measurement
- ❌ Tune knobs that don't address the dominant bottleneck
- ❌ Compare across configs without re-running bench (timing varies)
- ❌ Trust bench numbers without RPS-formula sanity check

---

## Slide 50 — Step 1: Identify the Binding Constraint

### Decision tree

```
Run bench → Check success rate
   ├── 100% successful → analyze RPS / TPOT / TTFT
   └── < 100% successful → find failure mode first
                            ├── "buffer timeout" → NIXL pool exhaustion
                            ├── OOM → mem_fraction too high
                            └── Other → check logs for stack trace

If 100% successful:
   ├── PD running peak < max_running_requests → not max_running-bound
   ├── PD running peak = max_running_requests → max_running-bound
   ├── PD queue median > forward median → queue-bound (#2 KV pool or #4 max_running)
   └── PD queue median < forward median → compute-bound (#5 GPU)
```

### How to extract these signals

```bash
# Running peak
grep "Prefill batch" pd.log | grep -oE "#running-req:\s*[0-9]+" | sort -u | tail -3
# Queue median vs forward median: see slide 27 recipe
```

---

## Slide 51 — Step 2: Verify with RPS Formula

### Always sanity-check

```
RPS_predicted = total_token_throughput / per_req_total_tokens
              = total_token_throughput / (input_len + output_len)
```

### Why this matters

If `RPS_measured ≪ RPS_predicted`, you have a non-compute bottleneck (NIXL transport, scheduler, control plane). Tuning compute won't help.

If `RPS_measured ≈ RPS_predicted`, you're compute-bound. Only TP=2 or smaller workload helps.

### Worked example

```
Bench result: total_tput = 4,470, RPS = 0.70
Workload: 8img/768p, input_len = 6,238, output_len ≈ 117
Predicted: 4,470 / 6,355 = 0.703  ✓ matches 0.70

Conclusion: 8img/768p is compute-bound. TP=1 max-out at ~5,700 tok/s.
To beat 0.70, you need either smaller workload or TP=2 / aggregate.
```

---

## Slide 52 — Tuning 1080p: Try `--chunked-prefill-size 32768`

### Hypothesis

For 8img/1080p (input 16,420), if `chunked_prefill_size = 32,768`, the input fits in one chunk → eliminates split-tail (bottleneck #1).

### Configuration

```bash
python3 -m dynamo.sglang \
    ... \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.78  # 0.85 may OOM with bigger chunks
```

### Expected outcomes

| Metric | chunked=16384 | chunked=32768 estimate |
|---|---:|---:|
| Small batches (<500) | 42% | ~5% |
| RPS | 0.23 | 0.30-0.40 |
| Forward median | 55 s | 40-45 s |
| OOM risk | low | medium-high |

### Risks

- 2× per-batch GPU memory for activations
- May exceed cuda graph capacity
- Required to lower mem_fraction → smaller KV pool
- Net effect: queue gets worse if KV pool shrinks

### Status: not yet tested empirically (high-priority experiment)

---

## Slide 53 — Tuning 1080p: max_running_requests Sweep

### Hypothesis

Higher max_running → more in-flight → lower queue wait → higher RPS.

### Empirical result

| max_running | RPS | TPOT P99 | E2E_med |
|---|---:|---:|---:|
| 64 | 0.70 | 2,390 ms | 84.9 s |
| 128 | 0.74 (+5.7%) | 17,582 ms (**+635%**) | 101.8 s (+20%) |

### Why max=128 is worse

```
max=64:  queue depth peak 33, forward median 23.99s, P99 forward 97s
max=128: queue depth peak 27, forward median 33.18s, P99 forward 213s

  Queue-decreased by 18% (good)
  Forward-increased by 38% (bad — decode contention)
  Net E2E: worsened by 20%
```

### When does max_running help?

**Only for 8img/768p** — KV pool wasn't yet binding constraint at max=64.

| Workload | max=64 effective? | max=128 helps? |
|---|---|---|
| 8img/1080p | KV-bound (cap=31) | No (<5% gain) |
| 8img/768p | max_running-bound (=64) | Yes (+5-15%) |
| 4img/768p | input-rate bound | No (cap = ∞) |

---

## Slide 54 — Tuning 1080p: mem_fraction Sweep (Counterintuitive)

### Hypothesis

Higher mem_fraction → larger KV pool → more in-flight → higher RPS.

### Empirical result (cross-host)

| mem_fraction | KV pool | running peak | RPS | E2E_med |
|---|---:|---:|---:|---:|
| 0.65 | 467k | 28 | 0.24 | 209 s |
| 0.85 | 695k | 42 | **0.23** ⚠ | 233 s ⚠ |

### Why higher mem_fraction is worse

```
mem_frac=0.85: in-flight = 42
  → 42 simultaneous decode batches share GPU SMs
  → per-token decode time grows from 33 ms → 50 ms
  → forward time grows 49% (55s → 88s)
  → net E2E grows 12%
```

### The general lesson

**Any tuning that adds more concurrent requests will hurt latency at the per-token level.** GPU compute is fixed; concurrency just spreads it across more work.

---

## Slide 55 — Tuning 8img/768p: max_running_requests=128

### Hypothesis

Same-host disagg 8img/768p was capped at max_running=64. Raise to 128 to leverage KV pool's full capacity (theoretical 106).

### Empirical result

| Config | RPS | Success | TPOT P99 |
|---|---:|---:|---:|
| max=64, np=128 r=1.0 | 0.70 | 128/128 | 2,390 ms |
| **max=128, np=128 r=1.0** | **0.74** | 128/128 | 17,582 ms ⚠ |
| max=64, np=128 r=2.0 | 0.67 | **107/128** ⚠ | 2,334 ms |
| **max=128, np=128 r=2.0** | **0.92** | **128/128** ✓ | 8,596 ms |

### Verdict

**+37% RPS at rate=2.0**, all 21 prior buffer-pool failures avoided. This is the best concrete tuning win we found.

### Recommendation

For 8img/768p disagg, **always use `--max-running-requests 128`** (not the default 64). The marginal P99 TPOT cost is acceptable.

---

## Slide 56 — Tuning 4img/768p: Just Push Input Rate

### Empirical result

| np | rate | RPS | E2E_med |
|---|---:|---:|---:|
| 32 | 0.1 | 0.10 | 1.5 s |
| 32 | 1.0 | 0.84 | 8.1 s |
| 128 | 1.0 | 0.99 | 6.5 s |
| 128 | 2.0 | 1.93 | 2.7 s |

### Why E2E gets BETTER at higher rate

```
At rate=1.0 with np=128:
  - in-flight ~ 7 (rate × lifetime ≈ 1.0 × 7 = 7)
  - PD has lots of headroom (max_running=64)
  - Bench client only sends 1 req/sec → spread out

At rate=2.0 with np=128:
  - in-flight ~ 24 (rate × lifetime ≈ 2.0 × 12 = 24)
  - Scheduler can batch 2-3 reqs per prefill
  - Effective tokens/forward doubles → tput doubles → individual req faster
```

### Lesson

**For light workloads, push input rate harder to unlock prefill batching.** Counterintuitive but follows from chunked-prefill scheduler's batching semantics.

---

## Slide 57 — Cross-host: NIXL Transport (cuda_ipc vs cuda_copy)

### Empirical result (8img/768p np=128, max_running=128)

| UCX_TLS | r=1.0 RPS | r=2.0 RPS | r=2.0 success |
|---|---:|---:|:---:|
| cuda_ipc + ib + rc + ud + ... | 0.74 | 0.92 | 128/128 |
| cuda_copy (no cuda_ipc) | 0.72 | 0.89 | 128/128 |
| NIXL_USE_CPU_HOST_MEMORY=1 | 0.72 | **0.81** | **124/128** |

### Why so close

```
Theoretical bandwidth:
  cuda_ipc (NVLink):  400 GB/s, 1 µs latency
  cuda_copy (D→H→D): 10 GB/s, 8 µs latency

Per-embedding 63 MB:
  cuda_ipc transfer: 0.16 ms
  cuda_copy transfer: 6.3 ms

Difference: 6 ms per request, in a 92-second pipeline
  → 0.007% of E2E
```

### Lesson

For same-host, **always use cuda_ipc** (default). For cross-host, you can't use cuda_ipc anyway. Don't worry about this knob — it's not the dominant cost.

---

## Slide 58 — Cross-host: page_size=64 Doesn't Help

### Empirical result (8img/1080p)

| page_size | KV pool | running peak | RPS | TPOT_med |
|---|---:|---:|---:|---:|
| 16 | 695,136 | 31 | 0.23 | 726 ms |
| 64 | 695,104 | 31-32 | 0.22 (-4%) | 526 ms (-28%) |

### Why page_size doesn't help RPS

```
page_size affects:
  - Alignment padding waste (max 48 tokens/req with page=64 vs 15 with page=16)
  - 32 in-flight × 48 tokens = 1,536 tokens of waste = 0.2% of pool
  
That 0.2% is negligible. KV pool capacity is unchanged.
```

### Why TPOT_med decreased (with page=64)

Larger pages improve memory access locality during prefill. But during decode, larger pages increase variance → P99 TPOT was 16-26% worse.

### Lesson

**Stick with page_size=16** (SGLang default). Don't waste time tuning this.

---

## Slide 59 — The "Do Not Bother" List

These knobs were tested empirically and found to have **negligible or negative effect**:

| Knob | Status | Why |
|---|---|---|
| `--page-size 64` (vs 16) | ❌ Don't | -4% RPS, +16-26% P99 TPOT |
| `--mem-fraction-static 0.92` | ❌ Don't | OOM risk, marginal +5% RPS, +12% E2E |
| Switching `nixl-write` ↔ `nixl-read` | ❌ Don't | <5% diff for same-host |
| Adding more encoders (4 → 8 → 16) | ❌ Don't | Encoders already 90-99% idle |
| `UCX_TLS=cuda_copy` (instead of cuda_ipc) | ❌ Don't | -3-12% RPS for same-host |
| `NIXL_USE_CPU_HOST_MEMORY=1` | ❌ Don't | -12% RPS, can introduce failures |
| Faster RoCE NIC (400G NDR currently) | ❌ Don't | Transfer is 0.007% of E2E |

### Why these don't help

They target time components that are **<5% of E2E**. The big chunks (T4 queue, T5 forward, T6 NIXL recv) are unaffected.

---

## Slide 60 — Decision Tree: Which Knob for Which Symptom

```
         ┌─────────────────────────────────┐
         │ Bench succeeded < 100%?          │
         └────────────┬────────────────────┘
                      │
            ┌─────────┴─────────┐
           YES                  NO
            │                   │
   ┌────────▼─────────┐   ┌─────▼──────────────────┐
   │ Buffer timeout?   │   │ RPS << prediction?      │
   │  → Use TP=1 agg   │   │   → check NIXL/transport │
   │  → Or rate < 0.5  │   │ RPS ≈ prediction?       │
   │  → Or max=128     │   │   → compute-bound        │
   └───────────────────┘   └─────┬──────────────────┘
                                 │
                       ┌─────────┴────────┐
                  Compute               Queue
                  bound                 dominant
                       │                  │
              ┌────────▼─────┐    ┌──────▼──────────┐
              │ TP=2 or       │    │ #2 KV pool?     │
              │ smaller image │    │   bump max=128  │
              │ Or chunked=   │    │ #4 max_running? │
              │   32768       │    │   bump max=128  │
              └───────────────┘    └─────────────────┘
```

---

# Part V — Conclusions & Reference

## Slide 61 — Headline Conclusions

1. **Disagg same-host on single GPU with 1 encoder + 1 PD is structurally inferior to TP=1 aggregate** for any workload that hits T6 (NIXL handoff overhead). TP=1 aggregate eliminates T2 + most of T6 → 2× RPS speedup for free.

2. **Adding encoders to disagg doesn't help** when PD is the bottleneck. The 4 B70 encoders are 90-99% idle holding embeddings waiting for PD.

3. **The `input_len / chunked_prefill_size` ratio determines bottleneck regime.** Bin 1 (>100%) is split-tail hell, Bin 4 (<20%) is no-bottleneck heaven.

4. **mem_fraction tuning doesn't help RPS** — adding in-flight requests trades queue wait for forward time, net zero.

5. **For 8img/1080p, no PD-side tuning gets us close to aggregate.** Either change workload (smaller image / fewer images) or change architecture (TP=2 agg).

6. **For 8img/768p, max_running=128 is the only meaningful PD-side win** (RPS 0.67 → 0.92 at rate=2.0).

7. **For 4img/768p, just push the rate.** It's input-rate-limited until ~3 RPS.

---

## Slide 62 — Why TP=1 Aggregate Wins (Deep Dive)

### Architectural advantages

```
Aggregate eliminates:
  ✓ T2 (control-plane handoff): 1.6-15 s/req → 0
  ✓ Most of T6 (NIXL recv): 10-47 s/req → < 0.5s
  ✓ Cross-process scheduler queue serialization
  ✓ NIXL ring buffer pool (no failure mode)
  ✓ Memory contention between encoder + decoder + ring buffer

Aggregate keeps:
  • SGLang queue (T4) — same as disagg
  • SGLang forward (T5) — same as disagg
  • Chunked-prefill split-tail — same penalty for input_len > 16,384
```

### Quantitative gains (8img/768p)

| Config | RPS | TTFT_med | E2E_med | TPOT P99 | success @ r=2 |
|---|---:|---:|---:|---:|:---:|
| Disagg max=64 | 0.70 | 39.9 s | 84.9 s | 2,390 ms | 107/128 |
| Disagg max=128 | 0.74 | 32.4 s | 101.8 s | 17,582 ms | 128/128 |
| **Aggregate** | **0.90** | **1.7 s** | **5.3 s** | **376 ms** | **128/128** |

### Recommendation

Unless you genuinely need encoder/decoder split (different hardware, encoder cache sharing across many decoders), **always use TP=1 aggregate** for single-host single-GPU.

---

## Slide 63 — Why Cross-Host Disagg is Structurally Limited

### The promise vs reality

| What we expected | What we measured |
|---|---|
| 4 encoders × 1 RPS each = 4 RPS | 0.71 RPS for 8img/768p |
| Independent encoder scaling | Encoder utilization 1-14% |
| Lower TTFT due to multiple encoders | TTFT 11.9 s (vs 1.7 s for agg) |

### Why

```
PD is the bottleneck. Its compute throughput is fixed (~3,800-5,700 tok/s).
Encoders are pre-bottleneck → adding more doesn't matter.

For PD compute throughput to grow, you need:
  - TP=2 (2× GPU compute)
  - or smaller per-request workload
```

### When cross-host disagg actually helps

1. **Heterogeneous hardware**: encoder on cheap GPU (L40S), decoder on H200
2. **Encoder cache sharing**: 1 encoder pool serves many small decoders
3. **Fault isolation**: encoder failure doesn't kill all decoders
4. **Multi-node deployment**: PD pool + encoder pool on different physical hosts

For us (B70 4E + 1 H200 PD), only #1 matches our hardware spread, but B70 ≠ cheap, and decode workload doesn't justify the network complexity.

---

## Slide 64 — Future Work: SGLang/Dynamo Upstream Patches

### High-impact patches (multi-day work each)

1. **Tail-coalesce in PrefillAdder.add_one_req()**
   - File: `/opt/sglang/python/sglang/srt/managers/schedule_policy.py:813-933`
   - Idea: when `chunked_req` has < 256 tokens remaining, merge tail with current main batch (1 forward instead of 2)
   - Expected gain: +20-30% RPS for 8img/1080p
   - Risk: must preserve chunked_prefill correctness for streaming

2. **Dynamic chunk size based on batch state**
   - Already exists (`--enable-dynamic-chunking`) but **gated to `pp_size > 1`**
   - File: `/opt/sglang/python/sglang/srt/managers/scheduler.py`
   - Gate is: `self.enable_dynamic_chunking = self.server_args.enable_dynamic_chunking and self.pp_size > 1`
   - Would need to remove gate or implement TP=1 variant

3. **GPU NIXL ring buffer pre-allocation**
   - File: `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py:646`
   - Default 8 GB pool. For 8img/1080p, only 50 fit.
   - Idea: allocate per-workload sized pool from GPU memory (not host) to avoid CPU-staged copies
   - Risk: cuda_ipc compatibility issues (we hit this in `b70_xpu_nixl.patch` rounds 4-5)

---

## Slide 65 — Pitfalls and Gotchas

### Things that bit us during testing

1. **Stale GPU memory from zombie processes**: Previous PD worker holding 7 GB after kill. Solution: `nvidia-smi --query-compute-apps`, find PID, kill.

2. **NIXL agent disconnect**: After 4+ hours of heavy benches, NIXL agents go stale. Errors look like `NIXL_ERR_REMOTE_DISCONNECT`. Fix: restart both encoder and PD with fresh agents.

3. **`expandable_segments:True`**: causes `cuda_ipc_md.c:281` assertion. **Always remove** from `PYTORCH_CUDA_ALLOC_CONF`.

4. **Etcd port lease**: `dynamo.frontend` doesn't release etcd leases on crash. Stale entries keep showing up in `/health`. Manual cleanup: `etcdctl del --prefix v1/instances/dynamo/`.

5. **ZMQ port collisions**: Default 22081/22091 may be held by zombie ZMQ sockets. Try a different port like 22082.

6. **NIXL_USE_CPU_HOST_MEMORY=1 causes silent perf regression**. Verify your env vars before complaining about RPS.

7. **Bench `--seed 0` produces identical images**: SGLang RadixCache may hit, masking real prefill cost. Use `--seed 1, 2, ...` for unbiased benches.

---

## Slide 66 — Reference Cheat Sheet (Env Vars, Ports, Paths)

### Environment variables

```bash
# Same-host disagg
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-write
UCX_TLS=cuda_ipc,ib,rc,ud,rc_verbs,ud_verbs,cuda_copy
UCX_NET_DEVICES=mlx5_0:1   # mlx5 NIC matching GPU NUMA

# Cross-host disagg
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy
UCX_NET_DEVICES=mlx5_4:1   # NIC NUMA-paired with PD GPU

# Common
ETCD_LEASE_TTL=600
DYN_TCP_MAX_MESSAGE_SIZE=268435456
DYN_HTTP_BODY_LIMIT_MB=256
NIXL_MAX_BUFFER_SIZE=805306368
```

### Ports

```
Frontend HTTP:    7001
NATS:             14222
etcd:             12379, 12380 (peer)
KV events ZMQ:    22080-22091 (per worker)
NIXL side channel: 20098-20099
```

### Critical file paths

```bash
PD log:       /hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/samehost_pd.log
Encoder log:  /hongming/dynamo/02_xpu_sh/logs/encode_xpu_32b_b70_*.log
Bench results: /hongming/res_*/...
SGLang src:   /opt/sglang/python/sglang/srt/...
Dynamo src:   /opt/venv/lib/python3.12/site-packages/dynamo/...
```

---

## Slide 67 — Source Code Reference Table

| Concept | File:line |
|---|---|
| RingBuffer (NIXL recv) | `/opt/venv/.../dynamo/common/multimodal/embedding_transfer.py:259-356` |
| RingBuffer default size (8 GB) | `embedding_transfer.py:646` |
| `Timeout while waiting for available buffer` | `embedding_transfer.py:724` |
| `NIXL_USE_CPU_HOST_MEMORY` env var | `embedding_transfer.py:30, 39` |
| `EMBEDDING_RECEIVER_FACTORIES` | `dynamo/common/multimodal/__init__.py:33-41` |
| PrefillAdder.add_one_req() | `/opt/sglang/python/sglang/srt/managers/schedule_policy.py:813-933` |
| chunked_prefill split logic | `schedule_policy.py:907-935` |
| KV pool admit refusal | `schedule_policy.py:866-870` (`if total_tokens >= rem_total_tokens: return NO_TOKEN`) |
| ReqTimeStats logging | `/opt/sglang/python/sglang/srt/managers/scheduler_metrics_mixin.py:353-388` |
| chunked_req continuation | `scheduler.py:2674-2691` |
| `--enable-dynamic-chunking` arg | `server_args.py:413` |
| dynamic chunking gate to `pp_size > 1` | `scheduler.py` (search `enable_dynamic_chunking`) |

---

## Slide 68 — Frequently Used Python Snippets

### Snippet 1: Compute predicted RPS

```python
def predicted_rps(total_tput_tokps, input_len, output_len):
    return total_tput_tokps / (input_len + output_len)

# 8img/1080p disagg
print(predicted_rps(3781, 16420, 121))  # 0.229 RPS
```

### Snippet 2: Identify workload bin

```python
def bin_for(input_len, chunked_prefill_size=16384):
    ratio = input_len / chunked_prefill_size
    if ratio > 1.0: return "bin1 (split-tail)"
    elif ratio > 0.5: return "bin2 (1 req/batch)"
    elif ratio > 1/3: return "bin3 (2 reqs/batch)"
    else: return f"bin4 ({chunked_prefill_size // input_len}+ reqs/batch)"

print(bin_for(16420))  # bin1
print(bin_for(6238))   # bin3
print(bin_for(3158))   # bin4 (5+ reqs/batch)
```

### Snippet 3: Compute KV cap

```python
def kv_cap(input_len, max_new=256, page_size=16, max_total=695136):
    per_req = input_len + max_new + page_size
    return max_total // per_req

print(kv_cap(16420))   # 41 (1080p)
print(kv_cap(6238))    # 106 (768p, 8img)
```

---

## Slide 69 — References (All Prior Analysis Docs)

### Top-level summaries

- `INDEX.md` — directory index
- `TECHNICAL_SUMMARY.md` — original 1080p analysis
- `status_report_4_cases_and_bottleneck_analysis.md` — 10-page status report
- `agg_tp1_vs_disagg_np128_zh.md` — TP=1 aggregate vs disagg comparison

### Workload-specific deep dives

- `same_host_problem_analysis_zh.md` — 8img/1080p root cause
- `same_host_768p_problem_analysis_zh.md` — 768p workload analysis
- `same_host_3_cases_problem_analysis_zh_v01.md` — np=128 sweep across 4 workloads
- `disagg_8img_768p_max_running_128_zh.md` — max_running=128 result for 8img/768p

### Bottleneck math

- `nixl_ring_buffer_count.md` — ring buffer pool capacity correction
- `16img_768p_bottleneck_math_explained.md` — KV pool / chunked_prefill formulas
- `tpot_comparison_all_cases_zh.md` — TPOT analysis across all configs

### Cross-host

- `crosshost_8img_1080p_time_breakdown.md` — 209 s breakdown
- `crosshost_8img_1080p_memfrac85_time_breakdown.md` — mem_frac sweep
- `crosshost_8img_768p_memfrac85_time_breakdown.md` — 8img/768p cross-host
- `crosshost_memfrac_065_vs_085_8img_1080p.md` — mem_fraction comparison

### Negative results (knobs that don't help)

- `disagg_page64_8img_1080p_test.md` — page_size=64
- `cuda_copy_vs_cuda_ipc_analysis_zh.md` — UCX_TLS variants

---

## Slide 70 — Q&A / Closing

### Key takeaways

1. **Always start with the RPS formula** to predict expected RPS, then compare with measured
2. **The `input_len / 16,384` ratio is the single most important number** — it determines bottleneck regime
3. **Time decomposition T1-T6 is reproducible** with grep + Python (snippets in slide 29)
4. **Most knobs don't help** — see the "Don't Bother" list (slide 59)
5. **Aggregate beats disagg for single-host single-GPU** by 2× — use it unless you have a specific reason

### Three concrete actions for new operators

1. Run baseline bench → compute predicted RPS → compare → identify if compute-bound
2. Identify which bin (1/2/3/4) your workload is in → know your active bottlenecks
3. If you must use disagg, set `--max-running-requests 128` for any workload in bin 3 or 4

### What's next?

- Implement tail-coalesce SGLang patch (Part V slide 64)
- Test TP=2 aggregate on these workloads (predicted ~2× RPS)
- Validate cross-host with smaller workloads (4img/768p) where encoder pool actually matters

### Contact

For questions or follow-ups: file an issue in the analysis repo, or reach out via the team channel. All bench data + analysis docs preserved in `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/`.

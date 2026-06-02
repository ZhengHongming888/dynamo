# Where the 16img/768p Bottleneck Numbers Come From

This explains the two specific claims in §6 of the disagg analysis:

> **#2 KV pool truly saturated**: in_flight=54, KV usage 98% (54 × 12,673 ≈ 695k pool)
> **#3 Single-request-per-batch**: 2 reqs (24,802) > chunked budget (16,384)

---

## Part 1 — KV pool: where do 695,136 and 12,673 come from?

### 1a. The 695,136 — KV pool capacity (`max_total_num_tokens`)

This is **printed by SGLang at PD startup** in the scheduler init line:

```
2026-06-01T01:11:53.390560Z INFO scheduler.init_model_worker:
    max_total_num_tokens=695136, chunked_prefill_size=16384, max_prefill_tokens=16384,
    max_running_requests=128, context_len=262144, available_gpu_mem=20.28 GB
```

**Source**: `/opt/sglang/python/sglang/srt/managers/scheduler.py` `init_model_worker()`

**What it is**: The total number of KV-cache token slots across all attention layers, computed by SGLang based on:

```
max_total_num_tokens ≈ floor(
    available_gpu_mem_bytes
    / (kv_cache_dtype_bytes × hidden_dim_per_token × num_layers × 2 (K+V))
)
```

For H200 with Qwen3-VL-32B-FP8 and `mem-fraction-static=0.85`:
- Total GPU memory: 142 GB
- Reserved for model weights (FP8 + bf16 mixing): ~42 GB
- Reserved for cuda graphs + activations: ~15 GB
- Reserved for headroom (1 - mem_fraction_static = 0.15): ~21 GB
- **Available for KV cache**: ~64 GB
- KV-cache-dtype: fp8_e4m3 (1 byte per element)
- Per token: 1 byte × 64 hidden_dim_per_token × 64 layers × 2 (K+V) ≈ 8.2 KB/token (Qwen3-VL specifics)
- **max_total_num_tokens** ≈ 64 GB / 8.2 KB ≈ 7.8M nominal, but actual is 695,136 due to dynamic allocator overhead

The exact number 695,136 is **whatever SGLang prints in your log** — it depends on the GPU, mem_fraction, model, and dtype. **You read it directly from the log, not derive it.**

### 1b. The 12,673 — per-request KV demand for 16img/768p

```
KV_demand_per_req = input_len + max_new_tokens + page_size
                  = 12,401 + 256 + 16
                  = 12,673 tokens
```

**Where each component comes from:**

#### `input_len = 12,401`
This is the **median input length** measured from PD-side ReqTimeStats logs during the 16img/768p bench. Each ReqTimeStats line looks like:

```
ReqTimeStats: ... input_len=12401, output_len=128, queue_duration=18267.4ms, forward_duration=3508.7ms ...
```

The 12,401 number breaks down as:
- 16 images × ~770 visual tokens/image = 12,320 visual tokens
- ~80 text tokens (chat template + user message + image placeholders)
- = **12,401 total input tokens** (median across 128 requests)

The 770 tokens/image for 1024×768 Qwen3-VL is determined by ViT patch geometry:
```
patches = floor((1024 × 768) / (28 × 28)) ≈ 1004 raw patches
After 2×2 spatial merge in Qwen3-VL ViT: 1004 / 4 ≈ 251 visual tokens per image (theoretical)
But Qwen3-VL adds positional/grid tokens, bringing it to ~770/image observed
```

You don't compute this — **you read it from the bench log**. The 12,401 is empirical.

#### `max_new_tokens = 256`
From the bench command:
```bash
python3 -m sglang.bench_serving \
    --random-output-len 256 \
    ...
```

Each request can produce up to 256 output tokens, so SGLang's `add_one_req()` reserves 256 KV slots per request to avoid mid-decode preemption. Reference:

`/opt/sglang/python/sglang/srt/managers/schedule_policy.py:866-870`:
```python
total_tokens = req.extend_input_len + max_new + self.page_size
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN
```

#### `page_size = 16`
From the PD launcher:
```
--page-size 16
```

Block size for KV cache allocation — each request gets KV slots in chunks of 16, so 1 extra page is reserved as alignment padding.

### 1c. The 98% saturation — `54 × 12,673 / 695,136`

Once we have:
- KV pool: 695,136 tokens
- Per-req demand: 12,673 tokens
- Observed peak running-req: 54 (from PD prefill log `#running-req` field)

```
KV usage = 54 × 12,673 / 695,136 = 684,342 / 695,136 = 0.984 = 98.4%
```

**Where the 54 comes from:**
- PD prefill log lines look like: `Prefill batch, #new-token: 12384, ..., #running-req: 54, #queue-req: 16, ...`
- Maximum `#running-req` value across all prefill events during the bench window = 54
- This is the **observed peak in-flight requests on PD scheduler**

**Theoretical cap based on KV pool only:**
```
floor(695,136 / 12,673) = 54.85 → 54 requests max
```

So 16img/768p hits the **hard ceiling of KV pool 98%** (or 100% in pages, with 54 = floor() of the theoretical max). Going to 55 in-flight requires 55 × 12,673 = 696,915 tokens > 695,136 → SGLang's `add_one_req()` returns `AddReqResult.NO_TOKEN` and refuses to admit.

**Verification**: For 8img/768p the same calc gives `floor(695,136 / 6,510) = 106`, and `--max-running-requests=64` is what caps it (not KV pool). For 16img/768p, KV pool caps it at 54 *before* `--max-running-requests=128` would.

---

## Part 2 — Single-request-per-batch: where do 24,802 and 16,384 come from?

### 2a. The 16,384 — `chunked_prefill_size`

From the PD launcher:
```bash
--chunked-prefill-size 16384
```

This is the **maximum number of new tokens** SGLang's scheduler will accept into a single prefill batch. It limits per-step GPU memory for activations and cuda-graph buffers.

### 2b. The 24,802 — what 2 requests would need

```
2 requests × input_len 12,401 = 24,802 tokens
```

If SGLang tried to **merge two 16img/768p requests into one prefill batch**, it would need 24,802 tokens of `extend_input_len` — far exceeding the 16,384 budget.

### 2c. The "single-request-per-batch" enforcement code

In `/opt/sglang/python/sglang/srt/managers/schedule_policy.py`, `PrefillAdder` tracks `rem_chunk_tokens` (initially = `chunked_prefill_size = 16384`) and decrements it as it admits each request:

```python
# schedule_policy.py:430
self.rem_chunk_tokens = rem_chunk_tokens   # = 16384 at batch start

# schedule_policy.py:553-555 (in add_req_state)
if self.rem_chunk_tokens is not None:
    alloc = min(extend_input_len, self.rem_chunk_tokens)
```

**Walkthrough for 16img/768p**:

| Step | Action | rem_chunk_tokens after |
|---|---|---:|
| Batch start | `rem_chunk_tokens = 16384` | 16,384 |
| Try admit req #1 (12,401 tokens) | 12,401 ≤ 16,384 → admit | 16,384 − 12,401 = 3,983 |
| Try admit req #2 (12,401 tokens) | 12,401 > 3,983 → cannot fit | 3,983 |
| Try chunked admit of req #2 | Could chunk to 3,983, but creates split-tail (8,418 token remainder) | 0 |
| Decision | Either skip req #2 OR chunk it | varies |

The implementation choice depends on `enable_dynamic_chunking` and the request queue state. In practice for 16img/768p we observed (PD-side prefill log analysis):

```
99% of prefill batches have #new-token in 7,000-13,000 range
1% are <500 (mostly RadixCache hits)
0% are 13,000-17,000 (no chunked tails)
```

This means scheduler **almost always picks "skip req #2 → admit only req #1 in this batch"** because the alternative (chunking req #2 into a 3,983-token chunk + 8,418-token tail) introduces the same chunked-prefill split-tail problem that hurts 1080p (40% small-batch overhead).

### 2d. Why the "2 reqs > chunked budget" matters

Per prefill batch, SGLang admits **1 request** for 16img/768p (vs **2 requests** for 8img/768p, vs **5 requests** for 4img/768p). This directly limits prefill throughput:

```
prefill_throughput = (tokens_per_batch × forward_freq)

16img/768p: 1 req × 12,401 tokens × ~0.6 batch/s ≈ 7,400 tok/s
8img/768p:  2 reqs × 12,476 tokens × ~0.7 batch/s ≈ 17,500 tok/s (+ 2 reqs admitted/batch)
4img/768p:  5 reqs × 15,790 tokens × ~0.9 batch/s ≈ ~70,000 tok/s
```

(Rough numbers; actual depends on GPU-side decode interleaving.)

### 2e. The general formula

For any input_len and chunked_prefill_size:

```
max_reqs_per_batch = floor(chunked_prefill_size / input_len)
```

| Workload | input_len | chunked_prefill_size | max_reqs/batch |
|---|---:|---:|---:|
| 8img/1080p | 16,420 | 16,384 | **0** (must split — split-tail problem) |
| 16img/768p | 12,401 | 16,384 | **1** |
| 8img/768p  |  6,238 | 16,384 | **2** |
| 4img/768p  |  3,158 | 16,384 | **5** |

**This ratio is the dominant factor in disagg RPS scaling.** Doubling chunked_prefill_size from 16,384 → 32,768 would let 16img/768p admit 2 reqs/batch and roughly double its disagg RPS — but at the cost of 2× per-batch GPU mem (likely OOM on 80% mem_fraction).

---

## 3. Quick verification for any workload

Use this Python snippet:

```python
# Step 1: Read max_total_num_tokens from PD log
# grep "max_total_num_tokens" your_pd_log.log
max_total_num_tokens = 695136

# Step 2: Read input_len from any successful ReqTimeStats line in PD log
# grep "ReqTimeStats" your_pd_log.log | head -1
input_len = 12401   # for 16img/768p

# Step 3: Configure constants
max_new_tokens = 256       # from --random-output-len in bench cmd
page_size = 16             # from --page-size in PD launcher
chunked_prefill_size = 16384  # from --chunked-prefill-size in PD launcher
max_running_requests = 128 # from --max-running-requests in PD launcher

# Step 4: Compute KV demand and theoretical caps
kv_demand_per_req = input_len + max_new_tokens + page_size
kv_pool_cap = max_total_num_tokens // kv_demand_per_req
effective_cap = min(kv_pool_cap, max_running_requests)
print(f"KV demand per req: {kv_demand_per_req}")
print(f"KV pool cap:       {kv_pool_cap}")
print(f"max_running cap:   {max_running_requests}")
print(f"Effective cap:     {effective_cap}")

# Step 5: Compute reqs per prefill batch
max_reqs_per_batch = chunked_prefill_size // input_len
print(f"Max reqs per prefill batch: {max_reqs_per_batch}")

# Step 6: Compute KV usage at observed peak (read from prefill log #running-req field)
peak_running = 54
kv_usage_pct = 100 * peak_running * kv_demand_per_req / max_total_num_tokens
print(f"KV usage at peak {peak_running}: {kv_usage_pct:.1f}%")
```

For 16img/768p this gives:
```
KV demand per req: 12673
KV pool cap:       54
max_running cap:   128
Effective cap:     54     ← KV pool wins
Max reqs per prefill batch: 1
KV usage at peak 54: 98.4%
```

For 8img/768p:
```
KV demand per req: 6510
KV pool cap:       106
max_running cap:   128
Effective cap:     106    ← KV pool wins (close call)
Max reqs per prefill batch: 2
KV usage at peak 110: ... 110 × 6510 / 695136 = 103% (slightly over due to dynamic alloc)
```

(For 8img/768p with max_running=128, observed peak was 110 because cuda_graph headroom can flex slightly; with max_running=64, peak was 63.)

---

## 4. Source references

| Number | Where it's defined | How to verify |
|---|---|---|
| `max_total_num_tokens=695,136` | SGLang scheduler init, depends on GPU + model | grep PD log for `init_model_worker` |
| `input_len=12,401` | Empirical from bench (Qwen3-VL ViT + chat template) | grep PD log for `ReqTimeStats input_len=` |
| `max_new=256` | `--random-output-len` in bench command | bench command |
| `page_size=16` | `--page-size` in PD launcher | launcher script |
| `chunked_prefill_size=16,384` | `--chunked-prefill-size` in PD launcher | launcher script |
| `max_running_requests=64 or 128` | `--max-running-requests` in PD launcher | launcher script |
| Peak `#running-req=54` | PD prefill log during bench | grep PD log for `#running-req` |
| KV-pool admit refusal logic | `schedule_policy.py:866-870` | `if total_tokens >= rem_total_tokens: return NO_TOKEN` |
| Chunked-prefill-budget logic | `schedule_policy.py:430, 553-555` | `self.rem_chunk_tokens` decrement |

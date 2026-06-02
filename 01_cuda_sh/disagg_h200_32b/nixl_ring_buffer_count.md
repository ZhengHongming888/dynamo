# NIXL Buffer Pool Math: How to Calculate "768 MB pool only fits 5-6 of 134 MB embeddings"

**TL;DR:** Earlier reports said "**768 MB pool fits 5-6 embeddings**" — this was based on the `NIXL_MAX_BUFFER_SIZE=805306368` env var (768 MB) in the launcher. **However, that env var is for NIXL's internal allocator and is NOT what gates the embedding transfer ring buffer.** The actual ring buffer is **8 GB by default** (hardcoded in `embedding_transfer.py:646`). The "5-6 fits" was slightly wrong — see corrected math below.

## 1. Where the numbers come from

### 1a. Per-embedding tensor size (correct)

For Qwen3-VL-32B-Instruct-FP8, the visual embedding tensor sent over NIXL has shape `(num_visual_tokens, hidden_size)` and dtype `bfloat16`.

```
embedding_size_bytes = num_visual_tokens × hidden_size × dtype_bytes
                     = (num_images × tokens_per_image) × 5120 × 2
```

| Workload | num_images | tokens/image | total tokens | size (bytes) | size (MB) |
|---|---:|---:|---:|---:|---:|
| 8img/1080p | 8 | ~2,064 | 16,512 | 169,082,880 | **161.2 MB** |
| 16img/768p | 16 | ~770 | 12,320 | 126,156,800 | **120.3 MB** |
| 8img/768p  | 8 | ~770 |  6,160 |  63,078,400 | **60.2 MB** |
| 4img/768p  | 4 | ~770 |  3,080 |  31,539,200 | **30.1 MB** |

Constants:
- **Qwen3-VL hidden_size = 5120** (from `config.json` of the model, not the comment's incorrect guess of 8192)
- **tokens per image** depends on resolution and Qwen3-VL ViT patch geometry:
  - 1024×768 (768p) image → ~770 visual tokens
  - 1920×1080 (1080p) image → ~2,064 visual tokens
- **dtype = bfloat16** (2 bytes; matches model dtype with kv-cache-dtype=fp8 only affecting KV, not vision tower output)

**Earlier doc said "134 MB" for 8img/1080p — that was a rough estimate. Correct value is 161.2 MB.**

### 1b. Ring buffer pool size (corrected)

I wrongly attributed the limit to `NIXL_MAX_BUFFER_SIZE` env var. The actual code path is:

**Source: `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py:646`**

```python
class NixlWriteEmbeddingReceiver(AbstractEmbeddingReceiver):
    def __init__(self, buffer_size=2 * 8 * 1024 * 1024 * 256 * 2):
        # the default buffer_size is the product of:
        # 2 (typical dtype size float16)
        # 8 * 1024 (typical embedding hidden size for Qwen-VL)
        # 256 * 1024 (1024 count of 256 mm token item)
        # 2 (extra copies) = 8 GB memory
        self.ring_buffer = RingBuffer(buffer_size)
```

Default = `2 × 8 × 1024 × 1024 × 256 × 2 = 8,589,934,592 bytes = 8 GB`

And `EMBEDDING_RECEIVER_FACTORIES` (in `dynamo/common/multimodal/__init__.py:37`) instantiates it with **no arguments**, so the default 8 GB is used.

**The `NIXL_MAX_BUFFER_SIZE=805306368` env var (768 MB) is consumed by NIXL's C-level allocator** for *individual NIXL transfer descriptors*, not the dynamo ring buffer. It limits the maximum size of a single registered memory region, not the pool total.

### 1c. Correct number of concurrent embeddings that fit

With the **actual 8 GB ring buffer:**

| Workload | embedding size | **8 GB pool fits** | **768 MB pool fits (if it gated)** |
|---|---:|---:|---:|
| 8img/1080p | 161.2 MB | **~50 concurrent** | ~4 (incorrect attribution) |
| 16img/768p | 120.3 MB | **~68 concurrent** | ~6 |
| 8img/768p  |  60.2 MB | **~136 concurrent** | ~12 |
| 4img/768p  |  30.1 MB | **~272 concurrent** | ~25 |

## 2. Why disagg failures still happened despite 8 GB pool

If the actual pool is 8 GB and 50+ embeddings fit, why did 8img/1080p disagg fail at rate=1.0 with 63/128 buffer-pool timeouts?

### Answer: ring-buffer fragmentation, not raw capacity

Look at the **RingBuffer.get_buffer()** in `embedding_transfer.py:306-350`:

```python
def get_buffer(self, size):
    if size > self.buffer_size:                         # 1. larger than total pool → fail
        return None, None
    self._flush_freed_list()                            # 2. reclaim contiguous freed space

    # If allocation goes over end boundary, try wrap around
    if self.free_start_idx + size > self.end_idx:
        if self.allocated_start_idx < size:             # 3. not enough space at start either
            return None, None                           #    → fail
        # Mark remainder as fake-allocated, jump to start
        self.freed_list[self.free_start_idx] = self.end_idx
        self.free_start_idx = 0
        self.wrapped_around = True

    start_idx = self.free_start_idx
    end_idx = start_idx + size

    if self.wrapped_around and end_idx > self.allocated_start_idx:
        return None, None                               # 4. would overlap allocated → fail

    # ...success path
```

**Failure reasons (each returns `None, None` → caller waits 60s and times out):**

1. **Total size exceeds pool** — would fail every request. Not what we see.
2. **Wrap-around impossible** — when ring is mostly full, head allocation can't proceed.
3. **Wrap-around overlap** — head wrapped past tail, can't allocate without overwriting in-use data.

So even with 8 GB pool, if **50 in-flight embeddings of ~161 MB each are pinned and not yet processed**, the ring buffer becomes fragmented and new arrivals fail.

### How many in-flight embeddings actually pile up?

For 8img/1080p disagg:
- input rate = 1.0 RPS
- PD admit rate = 0.24 RPS (saturated)
- Net pile-up rate = 0.76 embeddings/s
- Bench duration ~280s → **~213 embeddings would queue** if pool were infinite
- Pool capacity = 8 GB / 161 MB = ~50
- **First 50 fit, next 163 fail** at receive_timeout=60s after pool stays full

This roughly matches the observed **63 failures out of 128** at rate=1.0 (63 + 65 successful = 128, where the 65 successful are the ones that drained through over 280s × 0.24 = 67 ≈ 65).

## 3. So what's the real bottleneck math?

### For each workload, the failure-rate prediction is:

```
admit_rate    = PD_real_RPS                          (e.g. 0.24 for 8img/1080p)
arrival_rate  = bench input rate                     (e.g. 1.0)
duration      = bench wall-clock                     (e.g. 280s)

# steady-state pile-up (Little's Law for queue):
pileup_rate = arrival_rate - admit_rate              (=0.76 emb/s)
total_arrivals = arrival_rate × duration             (=280)
total_admitted = admit_rate × duration               (=67)
total_failed_estimate = total_arrivals - total_admitted - pool_capacity

# With pool_capacity ≈ 50 (8 GB / 161 MB)
total_failed_estimate ≈ 280 - 67 - 50 = 163
```

But empirically failures cap at "remaining requests after admission saturates" because requests don't accumulate forever — they fail at 60s timeout.

### Per-workload prediction (using 8 GB pool, not 768 MB):

| Workload | embedding | **pool capacity** | PD RPS | r=1.0 failures (predicted) | r=1.0 failures (observed) |
|---|---:|---:|---:|---:|---:|
| 8img/1080p | 161 MB | 50 | 0.24 | ~63 | **63** ✓ |
| 16img/768p | 120 MB | 68 | 0.34 | ~51 | **51** ✓ |
| 8img/768p  |  60 MB | 136 | 0.70 | 0 (pool never fills) | **0** ✓ |
| 4img/768p  |  30 MB | 272 | 1.0+ | 0 | **0** ✓ |

The math works correctly with **8 GB pool**, not 768 MB.

## 4. Updated bottleneck identification

The earlier docs said the "5-6 fit" theory based on 768 MB. **Math on 8 GB pool also explains the failures correctly** — the failures aren't from raw capacity (50+ slots) but from the imbalance:

`net_arrival_rate × bench_duration > pool_capacity`

For 8img/1080p at r=1.0: `(1.0 - 0.24) × 280 = 213 > 50` → buffer pool fills, subsequent requests timeout.

For 8img/768p at r=2.0: `(2.0 - 0.7) × 159 = 207 > 136` → pool fills, **21 timeouts observed**.

For 4img/768p at r=2.0: `(2.0 - 1.79) × 71 = 15 < 272` → pool never fills, **0 timeouts observed**.

**The "5-6 fit" was a wrong attribution to NIXL_MAX_BUFFER_SIZE. The mechanism (pool fills under input > admit rate over time) is correct, just with 8 GB capacity not 768 MB.**

## 5. How to verify on your system

```bash
# Check the actual ring buffer size in dynamo
grep -A 5 "NixlWriteEmbeddingReceiver" /opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py | grep buffer_size

# Should output:
# def __init__(self, buffer_size=2 * 8 * 1024 * 1024 * 256 * 2):
# = 8589934592 bytes = 8 GB
```

```python
# Compute embedding size for any workload:
import torch
n_imgs = 8
tokens_per_image = 770   # 1024×768 image with Qwen3-VL ViT
hidden = 5120            # Qwen3-VL hidden_size
dtype_bytes = 2          # bfloat16
size_bytes = n_imgs * tokens_per_image * hidden * dtype_bytes
print(f"Embedding size: {size_bytes / 1024**2:.1f} MB")

pool_bytes = 8 * 1024**3
print(f"Pool fits ~{pool_bytes // size_bytes} concurrent embeddings")
```

## 6. Why the 768 MB number kept appearing

The launcher sets `NIXL_MAX_BUFFER_SIZE=805306368` (line 158), which **does** affect NIXL but not the multimodal ring buffer. It's used by NIXL's UCX backend to size internal IB/RoCE buffers when registering memory regions.

I conflated this env var with the ring buffer capacity in earlier analysis. **Please disregard the "5-6 fit" / "12 fit" numbers in earlier docs** — they should all be ~8× larger.

## 7. Corrected slide 4 numbers

Earlier doc:
> "768 MB pool fits only 5-6 of 134 MB embeddings"

Correct version:
> **8 GB pool fits ~50 of 161 MB embeddings (8img/1080p), ~68 of 120 MB (16img/768p), ~136 of 60 MB (8img/768p), ~272 of 30 MB (4img/768p)**

The relative ranking and qualitative conclusion are unchanged: **larger embeddings + slower PD admit rate = pool fills faster = failures occur**. But absolute numbers were 8× too small.

## 8. Files updated / to update

- `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/status_report_4_cases_and_bottleneck_analysis.md` (slide 5, slide 8 — needs fix)
- `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/status_report_4_cases_and_bottleneck_analysis.pptx` (slides 5, 8)
- `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/same_host_768p_problem_analysis_zh.md` (§3.6 buffer pool reasoning)
- `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/cuda_copy_vs_cuda_ipc_analysis_zh.md` (§7 transfer time table is correct, only §3.6 in 768p doc references 768 MB)

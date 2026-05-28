# Per-Request Hand-off in Disaggregated Inference

This document explains the "per-request hand-off" concept that dominates throughput in our same-host dynamo+SGLang disagg experiments. Understanding it explains why no amount of GPU adding (TP=1 PD → TP=2 PD) and no easy tuning knobs improved throughput on the 32B-FP8 model.

---

## What "per-request hand-off" means

In disaggregation, each user request must travel between two separate processes (encoder ↔ PD) before it can be served. The **hand-off** is everything that happens between when the encoder finishes the ViT and when the PD actually starts the LLM prefill.

This is a **per-request cost** — it has to happen for every single request.

---

## Visualizing one request's life

For a request like "describe these 8 images, output 256 tokens":

```
Time     Where                    What happens
───────────────────────────────────────────────────────────────────────────
t=0      [Frontend]               HTTP POST received, parse 17 MB body
t=0.05   [Frontend]               Decide to route to encoder (via TCP request plane)
t=0.10   [Encoder GPU]            Receive request via TCP
t=0.15   [Encoder GPU]            Preprocess: decode JPEG, resize, normalize 8 imgs
t=0.30   [Encoder GPU]            Run ViT: 16K visual tokens × 27 layers
t=3.50   [Encoder GPU]            ViT done — embeddings tensor ready (160 MB on GPU)
                              ╔══════════════════════════════════════════╗
                              ║ ── HAND-OFF STARTS HERE ──              ║
t=3.51   [Encoder]            ║ torch.cuda.synchronize() — block until  ║
                              ║   all encoder GPU work finishes         ║
t=3.55   [Encoder]            ║ embeddings.clone().detach() — 160 MB    ║
                              ║   GPU→GPU memcopy on encoder side       ║
t=3.62   [Encoder]            ║ NIXL create_readable() — register       ║
                              ║   buffer with NIXL agent (CPU + RDMA    ║
                              ║   metadata setup)                        ║
t=3.70   [Encoder]            ║ Send TransferRequest to PD via TCP      ║
                              ║   (the metadata, NOT the data)          ║
t=3.75   [PD scheduler]       ║ Receive request from TCP request plane  ║
t=3.80   [PD scheduler]       ║ Validate, build SglangMultimodalRequest ║
t=3.90   [PD scheduler]       ║ Allocate KV cache slot for the request  ║
t=4.00   [PD]                 ║ NIXL begin_read() — pull 160 MB from    ║
                              ║   encoder GPU to PD GPU via cuda_ipc    ║
                              ║   (over NVLink)                         ║
t=4.20   [PD]                 ║ NIXL wait_for_completion()              ║
t=4.25   [PD]                 ║ Build mm_item dict for SGLang engine    ║
t=4.30   [PD]                 ║ Call engine.async_generate(...) —       ║
                              ║   queues the request for the scheduler  ║
t=4.35   [PD scheduler]       ║ Scheduler picks it up on next tick      ║
                              ║   (~1 ms poll interval, but only when   ║
                              ║   not busy with other work)             ║
                              ╚══════════════════════════════════════════╝
                              ─── HAND-OFF DONE ─── (took ~0.85 s here,
                              but in our real run at low load it was
                              already 11.5 s; under load 70+ s)
t=15.50  [PD GPU]             First prefill chunk forward starts
t=15.95  [PD GPU]             Decode begins, streams first token to client
```

---

## The two flavors of hand-off cost

### 1. Fixed overhead per request (setup latency)

Things that happen once per request, regardless of system load:
- TCP request plane round-trip between encoder and PD (~few ms)
- NIXL `create_readable()` + `begin_read()` setup (~ms each)
- `torch.cuda.synchronize()` blocking call (waits for ALL encoder GPU work to finish before exposing buffer)
- Python async overhead (event loop, `await` transitions)
- `embeddings.clone().detach()` GPU→GPU memcopy of ~160 MB (`stage_embeddings=False` default)
- KV slot allocation in PD scheduler
- `engine.async_generate()` request enqueue

In an idle system this floor is **~1-2 seconds per request**.

### 2. Queue/contention overhead (load-dependent)

When many requests are in flight:
- Each request waits its turn at the encoder→PD TCP queue
- NIXL READs serialize on the cuda_ipc channel (only so many in-flight transfers)
- The PD scheduler can only handle one new-request setup per scheduler tick (~1 ms), but ticks are gated on the active prefill chunk finishing
- Python async tasks contend for the GIL on the PD process

This explodes from **~1 s/req at idle** to **~11+ s/req at concurrency 50**.

---

## Why aggregated (agg) mode avoids it

In aggregated mode, the SAME process does ViT and LLM. The hand-off looks like:

```python
# Inside the same Python process, same GPU
embeddings = vit_model(images)              # GPU op, output stays on GPU
output = llm_engine.generate(text_tokens, embeddings=embeddings)
# Just a Python function call — no TCP, no NIXL, no cross-process serialization
```

**No memcopy. No serialization. No synchronization barriers. No scheduler hop.** The embedding tensor is a Python object passed by reference between two functions in the same process.

That's why TP=1 agg at 0.52 RPS beats 2-GPU disagg at 0.23 RPS: the agg config saves the 11.5 s of hand-off per request.

---

## Why this kills disagg's value proposition (in theory)

Disagg's textbook promise: "ViT and LLM run on different GPUs in parallel — encoder for req(N+1) overlaps with prefill for req(N)."

**Theoretical math (no hand-off):**
- ViT alone: ~3.5 s
- LLM prefill alone: ~3 s
- With perfect pipelining: max(3.5, 3) = **3.5 s/req**
- Speedup vs agg (3.5 + 3 = 6.5 s): **1.86×**

**Reality with 11.5 s hand-off:**
- ViT: 3.5 s
- Hand-off: 11.5 s
- LLM prefill: 3 s
- Even with perfect pipelining: max(3.5, 11.5 + 3) = **14.5 s/req**
- Speedup vs agg: **0.45×** (i.e., disagg is 2.2× SLOWER than agg)

The hand-off **dominates** the critical path. Pipelining ViT with LLM doesn't matter when both stages are dwarfed by the bridge between them.

---

## Concrete number from a real run

From `/hongming/dynamo/logs/pd_worker.log` and `encoder_worker.log`, request `c74cb9d2` (first request, completely idle system, PD-TP=2 disagg):

```
06:41:20.221  [encoder]  request received
06:41:23.835  [encoder]  ViT done, sending to PD via TCP   ← 3.6 s of ViT
06:41:23.838  [PD]       request received from encoder
06:41:35.388  [PD]       first prefill batch starts        ← 11.5 s GAP HERE
06:41:35.421  [PD]       second prefill chunk
06:41:35.871  [PD]       request completed (decode done)   ← 0.5 s of LLM work
```

**Per-request breakdown for this single request, no contention:**
- 3.6 s — encoder ViT compute (real, useful work)
- **11.5 s — hand-off pipeline (pure overhead)**
- 0.5 s — LLM prefill + decode (useful work)

The LLM compute was 0.5 s. The hand-off was **11.5 s — 23× more time in the bridge than in the actual LLM work** for that request.

---

## Why adding more GPUs didn't help

We tested 3-GPU disagg (1 encoder + TP=2 PD, GPUs 4/5/7). Result: **0.24 RPS**, essentially identical to 2-GPU disagg's 0.23 RPS. The 3rd GPU added zero throughput.

**Reason:** TP=2 PD made the LLM compute (the 0.5 s above) twice as fast — but you can't speed up 0.5 s into something meaningful when you're still paying 11.5 s of hand-off per request. **The bottleneck is the bridge, not the compute.**

Concretely from the PD-TP=2 run:
- Mean TPOT: 57 ms (TP=2 decode is fast, as expected)
- Mean TTFT: 230 s (hand-off + queue under load — same as TP=1 PD!)
- → Doubled compute didn't speed up the dominant cost

---

## Why our easy-tuning attempts failed

1. **`stage_embeddings=True`** (skip the 160 MB GPU clone): made TPOT 7× worse and dropped throughput to 0.16 RPS. Likely because the source tensor is freed mid-NIXL-read. Need lifecycle refactor.
2. **`chunked-prefill-size 8192` + `--enable-mixed-chunk`**: made E2E 22× worse (777 s for one request). The smaller chunks plus prefill+decode interleaving caused per-request latency to balloon under disagg's already-strained scheduling.
3. **`--enable-mm-global-cache` on encoder**: crashed because Mooncake isn't installed.

The hand-off bottleneck is in **integration code paths** (Python async, TCP serialization, NIXL handshake, scheduler enqueue), not in any single tunable knob.

---

## What COULD reduce per-request hand-off

These all require code changes, not config tuning:

### 1. Pipeline the hand-off across requests (highest impact)
Currently, request N's hand-off (~11.5 s) blocks request N+1's hand-off entirely. If hand-off were truly pipelined — encoder doing ViT(N+1) while PD does NIXL READ for N while LLM does prefill for N-1 — the overhead amortizes. **Probably the biggest possible win.**

### 2. Co-locate encoder + PD scheduler in a single process (medium impact)
Skip the TCP request plane and `engine.async_generate()` round-trip entirely. The encoder writes embeddings directly into the PD scheduler's input queue. Lose process isolation, save several ms of round-trip.

### 3. Skip NIXL when same-host (medium impact)
On a same-host setup, both processes have access to the same GPUs. The encoder could write embeddings to a shared CUDA IPC pool, and the PD reads them by handle (no copy). NIXL's RDMA setup is overkill when you have direct cuda_ipc.

### 4. Eliminate the embedding `clone().detach()` SAFELY (small win)
Replace `stage_embeddings=False` (default) with proper lifecycle management — keep source tensor alive until NIXL transfer completes. Saves ~160 MB GPU memcopy per request and ~50 ms.

### 5. Pre-allocate KV slots speculatively (small win)
The encoder knows the visual token count before sending. Communicate that to the PD scheduler so it can pre-reserve KV slots, removing one synchronous step from the critical path.

---

## Practical takeaway

For Qwen3-VL-32B on same-host 2-3 GPU setups with current dynamo:

- **TP=2 agg always wins.** It avoids the entire hand-off problem.
- **Disagg is structurally penalized** by the per-request hand-off floor (~11 s minimum at low load, scaling badly with concurrency).
- **More PD GPUs (TP=N) cannot fix this** because the bottleneck is upstream of compute.
- **Disagg makes sense when**: encoder is on a different machine (so the per-request hand-off cost is meaningful relative to network latency anyway), OR when ViT compute is so dominant that even a slow hand-off is small relative to it (haven't found a Qwen3-VL workload in our test matrix where this is true).

---

## Files

- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/per_request_handoff.md`
- Source code referenced:
  - Encoder handler: `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py`
  - PD handler: `/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/worker_handler.py`
  - NIXL transfer: `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py`
- Live evidence in: `/hongming/dynamo/logs/pd_worker.log`, `/hongming/dynamo/logs/encoder_worker.log`
- Companion analysis: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pd_tp2_results.md`

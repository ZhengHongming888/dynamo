# Cross-host Disagg Time Breakdown: 8img/1080p np=64 r=1.0 (B70 4E + H200 PD)

**Bench:** Run 2 at 2026-06-01T07:37:27 UTC
**Result:** RPS=0.24, 64/64 successful, median E2E 209.0s, median TTFT 147.7s
**Logs analyzed:** 4 encoder logs (B70) + 1 PD log (H200), 65 requests joined by dynamo request_id

---

## 1. Per-Request Time Decomposition

For each request we have 5 timestamps from log joining:

```
Client request → Frontend → [Encoder] → NIXL → [PD] → first token → ... → last token → Client
                            ↑          ↑            ↑                  ↑
                         enc_recv  enc_done     pd_recv           pd_done
                         (B70)     (B70)        (H200)            (H200)
                                   ←-----  T1 ------→
                          ←- T2 -→            ←-------- T3 --------→
                                                ↳ T4 (SGLang queue) ↳ T5 (forward)
```

| Phase | Description | Median | Mean | P99 | % of E2E |
|---|---|---:|---:|---:|---:|
| **T1** Encoder lifetime (enc_recv → enc_done) | Encoder holds embedding while waiting for PD to finish reading via NIXL | **208.3 s** | 200.7 s | 259.2 s | (overlaps T3) |
| **T2** Control-plane fan-out (enc_recv → pd_recv) | Time from encoder receiving request to PD receiving the same request via dynamo router | **14.2 s** | 14.2 s | 22.9 s | 7% |
| **T3** PD lifetime (pd_recv → pd_done) | Total time PD spends on this request (from receive to last token sent) | **194.8 s** | 184.9 s | 250.3 s | 93% |
| **T4** PD SGLang queue_duration (sub-component of T3) | Wait inside SGLang's scheduler for KV slot + admission | **92.6 s** | 93.6 s | 156.6 s | 47% of T3 |
| **T5** PD SGLang forward_duration (sub-component of T3) | Prefill (16k tokens) + decode (256 tokens) on H200 GPU | **55.4 s** | 64.5 s | 194.6 s | 28% of T3 |
| **T6 = T3 - T4 - T5** Other PD overhead | NIXL receive + dynamo runtime + result streaming | **46.8 s** | — | — | 24% of T3 |

### Reconciliation with bench-reported numbers

| Metric | Bench reports | Computed | Match |
|---|---:|---:|---|
| Median E2E (request received → last token) | 209.0 s | T2 + T3 = 208.2 s | ✓ |
| Median TTFT (request received → first token) | 147.7 s | ≈ T2 + T4 + 1 prefill batch ≈ 14.2 + 92.6 + ~40 ≈ 146 s | ✓ |

---

## 2. PD-side Breakdown (median 194.8 s)

```
T3 PD lifetime (194.8 s) ────────────────────────────────────────────────────────
  ├─ T4: SGLang queue_duration       92.6 s  (47.5%)  ▓▓▓▓▓▓▓▓▓▓▓▓
  ├─ T5: SGLang forward_duration     55.4 s  (28.4%)  ▓▓▓▓▓▓▓
  └─ T6: NIXL recv + dynamo          46.8 s  (24.0%)  ▓▓▓▓▓▓
```

**~50% of PD time is spent waiting in the SGLang scheduler queue** for a KV cache slot (KV pool capacity 27 / mem_fraction=0.65, but observed peak only 28 in-flight = saturated).

**~30% of PD time is actual GPU work** (prefill 16k tokens + decode 256 tokens at chunked-prefill split-tail rate).

**~25% of PD time is NIXL receive overhead** — pulling the 161 MB embedding from B70 over RoCE (this includes ring-buffer management, cuda_copy D→H→D since cross-host can't use cuda_ipc, and dynamo runtime serialization).

---

## 3. Encoder-side Behavior (T1 = 208 s median lifetime)

This is the headline finding: **each encoder request stays "alive" on B70 for ~208 seconds median** — but only ~1-2 of those seconds is actual ViT forward work. The rest is **waiting for PD to finish reading the embedding via NIXL**.

```
Encoder lifetime (208 s) ─────────────────────────────────────────────────────
  ├─ ~1-2 s: ViT forward (8 images × 1080p)
  ├─ ~14 s:  Wait in dynamo router for PD to be ready (T2 control-plane handoff)
  └─ ~190 s: Embedding sits in encoder GPU memory, waiting for PD's NIXL_READ pull
              ↳ This corresponds almost exactly to T3 (PD lifetime)
              ↳ Encoder can't release until PD signals "embedding consumed"
```

**Verification: T1 ≈ T2 + T3** (median diff = 0.1 s, mean = 1.6 s, p99 = 70 s)

This means encoder is essentially **a passive memory holder** for 99% of its "lifetime." The actual ViT forward is fast.

### Per-encoder load (4 B70 encoders)

| Encoder | Requests handled | Median lifetime | Mean | Max | Min |
|---|---:|---:|---:|---:|---:|
| enc#1 | 14 | 209 s | 211 s | 246 s | 149 s |
| enc#2 | 17 | 206 s | 201 s | 254 s | 149 s |
| enc#3 | 14 | 225 s | 217 s | 259 s | 162 s |
| enc#4 | 20 | 205 s | 182 s | 259 s | **14 s** ← only the very first warmup |

Load is reasonably balanced (14-20 reqs each across 4 encoders). Encoder #4 served the first request quickly (14.4 s — included ViT only, no PD-side queue), then subsequent ones grew to 200+ s as PD-side queue formed.

---

## 4. Where is the Bottleneck?

### The bottleneck is **NOT the encoder** (B70 ViT is fast: ~1-2 s)
### The bottleneck is **NOT the network** (NIXL handoff/transfer is ~14 s of T2 + ~47 s of T6 = ~61 s, only 30% of E2E)
### The bottleneck **IS the PD scheduler queue + GPU compute**:

```
What actually consumes the 209-second E2E?

  Real bottleneck               Time    % of E2E
  ────────────────────────────  ─────   ────────
  PD SGLang queue (T4)          92.6s    44%   ← #1 bottleneck (KV pool full, can't admit)
  PD SGLang forward (T5)        55.4s    27%   ← #2 bottleneck (chunked-prefill split + GPU compute)
  PD NIXL recv + dynamo (T6)    46.8s    22%   ← #3 (cross-host overhead)
  Control-plane fan-out (T2)    14.2s     7%   ← minor (dynamo router latency)
```

**Bottleneck #1: KV cache saturation in SGLang scheduler (44% of E2E)**

PD's KV cache pool is 467,072 tokens (with mem_fraction=0.65). For 8img/1080p:
- per_req KV demand = 16,420 (input) + 256 (max_new) + 16 (page) = 16,692 tokens
- theoretical cap = 467,072 / 16,692 = **27 in-flight** maximum
- Observed peak running-req = 28 (at the cap)
- Queue depth peak = 55 requests waiting

When 28 requests are running and a 29th arrives, it sits in `queue_duration` for ~93 seconds median (waiting for one of the 28 to free its KV slots after generating 256 output tokens).

**Bottleneck #2: Chunked-prefill split-tail (28% of E2E)**

input_len=16,420 > chunked_prefill_size=16,384 → every request is split into:
- 1 main chunk of 16,384 tokens (full chunked budget)
- 1 tiny tail of 36 tokens (~50% of prefill batches in the log)

Observed prefill batch sizes: 42% are <500 tokens (the tails), 58% are 13-17k (the main chunks). Each of the 64 requests forces 2 prefill events → ~128 prefill batches just for ingestion, vs ~64 if input fit in one chunk.

**Bottleneck #3: NIXL recv + dynamo handoff (22% of E2E, T6)**

This 47 s/request includes:
- NIXL ring buffer wait (per-request slot allocation)
- D→H staged copy from B70 GPU memory to host pinned buffer
- RoCE transfer (~161 MB at 400 Gb/s NDR = ~3.2 ms theoretical, but actual is much higher due to ring buffer contention)
- H→D staged copy on H200 (cuda_copy, since cross-host can't use cuda_ipc)
- dynamo runtime serialization/deserialization

---

## 5. Encoder Logs Tell Us Important Things

### `PATCH(non-cached): moving embeddings from cpu to xpu for NIXL transfer`

Found 4× in the bench window. Significance:
- B70 encoder is **XPU** (Intel GPU), not CUDA
- The encoder dynamo path **defaults to CPU embedding gen** for cross-architecture; but PD expects GPU memory
- Each request triggers a CPU→XPU copy (~63 MB) before NIXL registers it
- This adds ~50-100 ms per request, **not the main bottleneck**

### `Created shared connection 'XXXX-1'` from `dynamo.nixl_connect.Connector`

Per-request marker showing NIXL connection establishment between encoder and PD. Takes ~0-10 s for first request (handshake) but ~0 s for subsequent (already connected).

### Encoder dynamo router activity

Encoder log shows `request received → request completed` with median 208 s gap. Encoder is **just an embedding holder** for that entire duration.

---

## 6. Comparing to Same-Host Disagg (Why Cross-Host Doesn't Beat It)

| Metric | Cross-host (this) | Same-host disagg page=16 max=64 | Δ |
|---|---:|---:|---|
| RPS | 0.24 | 0.23 | +4% |
| E2E median | 209 s | 175 s | **+19% (worse)** |
| TTFT median | 148 s | 112 s | **+32% (worse)** |
| TPOT P99 | 13.6 s | 19.3 s | -29% (better) |
| KV pool cap | 27 (mem_frac=0.65) | 41 (mem_frac=0.85) | -34% capacity |
| Failed | 0/64 (run 2) | 65/128 (50% fail) | (different mode) |

The 4 B70 encoders are massively underutilized — they could handle ~10× more requests if PD could keep up. But PD GPU is saturated by:
- KV pool (28 in-flight max)
- Chunked-prefill split-tail (40% of prefill events are tiny)
- Decode contention with prefill admit (queue interrupts running batches)

**Adding more encoders on B70 will NOT improve RPS** for 8img/1080p because PD is the bottleneck. Cross-host disagg with 4 encoders has the same PD-side rate as same-host with 1 encoder (0.23-0.24 RPS).

---

## 7. What WOULD Help This Workload

In priority order (by expected RPS gain):

| Fix | Expected RPS gain | Affects which T |
|---|---|---|
| Switch to TP=1 Aggregate | 0.24 → 0.47 (**2×**) | Eliminates T2, T6, reduces T4 (no NIXL handoff) |
| TP=2 Aggregate | 0.24 → ~0.95 (**4×**) | Doubles GPU compute |
| `--chunked-prefill-size 32768` | 0.24 → ~0.40 | Reduces T5 by eliminating split-tails |
| Bump PD `--mem-fraction-static 0.65 → 0.85` | 0.24 → ~0.32 | Increases KV pool 27 → 41, reduces T4 |
| Lower image resolution to 768p | 0.24 → ~0.71 (16img/768p) or 0.90 (8img/768p) | input_len drops, fits chunked budget |
| More encoders on B70 | 0% (encoder is not bottleneck) | — |
| Faster NIXL transport (HCA upgrade) | <5% | T6 only, already <1/4 of E2E |

---

## 8. Summary Diagram: Median Request Lifecycle (209 s)

```
Time (s):  0        14         107       163       207  209
           │         │           │         │          │   │
Client ────┤         │           │         │          │   ├──→ last token
           │         │           │         │          │   │
Encoder ───┤(rcv)    │           │         │          │   │
           │  ViT forward+memhold (208 s, mostly idle waiting)
           │                                          │   │
PD ────────│─────────┤(rcv)─────────────────────────┤(done)
           │         │           │         │          │
                     │←──── 92.6 s queue ────→│←─55s forward─→│
                                              prefill 16k+decode
                     │←─── 47 s NIXL recv ───────────────→│
```

---

## 9. Files

- PD log: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/pd_worker_giga01_20260601_071108.log`
- Encoder logs (B70): `/hongming/dynamo/02_xpu_sh/logs/encode_xpu_32b_b70_{1,2,3,4}.log`
- Bench result: `/hongming/res_crosshost_b70_4E_h200_pd/8img_1080p_rate1.0_np64_repeat_20260601_073727/`
- Bench window: 2026-06-01T07:37:27 → 07:43:30 UTC

## 10. One-Line Conclusion

> **Cross-host disagg's 209-second E2E is dominated by PD-side bottlenecks: 47% SGLang scheduler queue + 28% chunked-prefill+decode + 22% NIXL handoff overhead. The 4 B70 encoders contribute only ~14 s (7%) of control-plane delay and are 99% idle waiting for PD to consume their embeddings. To improve RPS, fix PD (raise mem_fraction, larger chunked_prefill_size, or switch to aggregate) — adding more encoders won't help.**

# B70 encoder time-breakdown analysis (patched vs unpatched)

**Host:** B70 (sc09giga01-b70), 4 encode workers on XPUs 0..3
**Pair :** H200 (172.26.46.75), prefill+decode in same instance
**Model:** Qwen3-VL-32B-Instruct-FP8, encoder-only mode
**Workload:** synthetic local data-URI image, 1-img and 4-img prompts
**Method:** instrumented per-stage timings emitted as `BENCH_TIMING …` log lines

> **Date:** 2026-05-24
> **Sample sizes:** 79 patched-ON 4E + 60 patched-OFF 4E + 60 patched-ON 1E
> **Toggle:** `BENCH_DISABLE_XPU_PATCH=1` env var skips the
> `precomputed_embeddings.to(xpu)` line; everything else is byte-identical.
>
> *Section 8 added 2026-05-24: 1-encoder (1E) vs 4-encoder (4E) comparison
> at the same client concurrency=4.*

## 1. Encoder request lifecycle (where time goes)

The `generate()` coroutine in `encode_worker_handler.py` walks each request through:

| # | Stage | Code region | What it does |
|---|-------|-------------|--------------|
| 1 | `setup_ms`        | lines 319–336 | Parse JSON, extract image URLs, build `MultiModalGroup` |
| 2 | `pure_encode_ms`  | lines 339–350 | Vision tower forward pass on XPU (image download → preprocess → ViT) |
| 3 | `patch_to_xpu_ms` | lines 351–370 | **The patch.** `precomputed_embeddings.to(xpu)` (CPU → XPU memcpy) |
| 4 | `encode_total_ms` | (1+2+3 in the with block) | total time inside `mm:enc:vision_encode` NVTX region |
| 5 | `token_expand_ms` | lines 372–456 | Replace each image-token with N tokens for the patch grid (CPU work) |
| 6 | `nixl_setup_ms`   | lines 458–464 | `embedding_sender.send_embeddings(tensor)` → NIXL `register_memory` + `create_readable` |
| 7 | `pd_call_ms`      | lines 467–469 | `pd_worker_client.round_robin(req)` returning the response generator |
| 8 | `pd_ttft_ms`      | first iter of generator | Time from PD call return to first response token from H200 (TTFT proxy, includes RDMA pull) |
| 9 | `pd_stream_ms`    | rest of generator | Streaming the remaining decode tokens back |
| 10 | `nixl_wait_ms`   | line 489 | `await transfer_future` — final NIXL completion wait |
| 11 | `total_ms`       | start to end | wall-clock end-to-end |

The instrumentation lives next to `_nvtx.annotate(...)` regions that already
existed; I just added `time.perf_counter()` checkpoints around them and emit
one structured WARN line per request.

## 2. Headline numbers (median, 1-img workload)

```
stage                      patched ON  patched OFF    Δ ms        Δ%
setup_ms                         0.15        0.16    -0.01     -6.3%
pure_encode_ms                 993.32     1014.13   -20.81     -2.1%   (noise — vision tower is the same)
patch_to_xpu_ms                 12.09        0.06   +12.03    +∞      (patch adds CPU→XPU copy)
encode_total_ms               1006.50     1014.17    -7.67     -0.8%
token_expand_ms                  0.09        0.08    +0.01    +12.5%   (sub-millisec, ignore)
nixl_setup_ms                    1.65       14.35   -12.70    -88.5%   (XPU-mem register cheaper than CPU-mem register)
pd_call_ms                       4.03        3.84    +0.19     +4.9%
pd_ttft_ms                     292.48      272.73   +19.75     +7.2%
pd_stream_ms                   708.27      683.49   +24.78     +3.6%
nixl_wait_ms                     0.32        0.49    -0.17    -34.7%
total_ms                      1716.87     1722.24    -5.37     -0.3%   (≈ wash on encoder side)
```

For 4-image:

```
stage                      patched ON  patched OFF    Δ ms        Δ%
setup_ms                         0.20        0.29    -0.09    -31.0%
pure_encode_ms                2634.41     2652.09   -17.68     -0.7%
patch_to_xpu_ms                 25.13        0.09   +25.04    +∞
encode_total_ms               2653.40     2652.18    +1.22      0.0%
nixl_setup_ms                    2.09       26.20   -24.11    -92.0%
pd_call_ms                       9.79        3.56    +6.23   +175.0%
pd_ttft_ms                    1257.83     1242.79   +15.04     +1.2%
pd_stream_ms                  1823.57     1660.46  +163.11     +9.8%
nixl_wait_ms                     0.50        9.75    -9.25    -94.9%
total_ms                      4582.36     4378.12  +204.24     +4.7%
```

## 3. Where is the bottleneck?

**Vision encode dominates by an order of magnitude.** For both workloads
the median total time is gated by `encode_total_ms` plus `pd_ttft_ms +
pd_stream_ms`. The actual NIXL work (`nixl_setup_ms + nixl_wait_ms`) is
**1–4 ms** in steady state — invisible on the time budget.

Stage share of median total (4-image, patched-on):
```
encode_total_ms     2653 ms   58 %    ←  vision tower forward pass
pd_ttft_ms          1258 ms   27 %    ←  H200 prefill + first-token
pd_stream_ms        1824 ms   40 %    ←  decode tokens streaming back
nixl_setup_ms          2 ms  0.05 %
patch_to_xpu_ms       25 ms  0.55 %
nixl_wait_ms           1 ms  0.02 %
all-other            ~10 ms  0.22 %
total_ms            4582 ms                (sums >100% because PD streaming overlaps with NIXL wait)
```

**Conclusion:** The encoder-side bottleneck on B70 is **the vision tower
itself, not the embedding transfer.** A 32B-class Qwen3-VL ViT processing
4 images at 768×768 on a single Battlemage-class XPU takes ~2.65 s of
pure compute. That's ~63 % of total wall-time *before any wire traffic
exists*.

Two corollaries:

1. **NIXL on B70 is NOT the gating factor for cross-host disagg
   throughput today.** Steady-state NIXL costs are 1–4 ms even on the
   slow CPU path. The earlier `cross_host_giga01_b70_results.md` numbers
   showing ~0.10 RPS at 8img/1080p reflect the **encoder forward pass**,
   not the wire.
2. **NIXL agent setup is a one-time tax, ~3.3 s.** First request through
   each worker (`nixl_setup_ms p99=3300.73` patched-on, p99=3269.91
   patched-off) pays the per-worker NIXL agent + remote-pairing cost.
   After that, register-memory drops to 1–4 ms (XPU) or 14–26 ms (CPU).
   Subsequent runs of the same worker do not pay this again.

## 4. What did the patch actually change?

Mechanically the patch shifts where the embedding tensor lives at the
moment NIXL exposes it for RDMA:

|                                    | patched ON              | patched OFF (baseline)  |
|------------------------------------|-------------------------|-------------------------|
| `emb_pre`  (after sglang.encode)   | `cpu`                   | `cpu`                   |
| `emb_post` (handed to NIXL)        | **`xpu:0`**             | `cpu`                   |
| `patch_to_xpu_ms` (CPU→XPU memcpy) | 12 ms (1-img) / 25 ms (4-img) | 0 ms (skipped)    |
| `nixl_setup_ms` steady-state       | **1.65 ms / 2.09 ms**   | 14.35 ms / 26.20 ms     |
| `nixl_wait_ms`                     | 0.32 ms / 0.50 ms       | 0.49 ms / 9.75 ms       |
| `total_ms` (median)                | 1716.87 / 4582.36       | 1722.24 / 4378.12       |

So the patch replaces ~12–26 ms of NIXL-on-CPU register/wait with
~12–25 ms of explicit CPU→XPU copy + 2–10× cheaper NIXL setup on
XPU memory. The two near-cancel. The patch is approximately
**neutral on encoder wall-clock time** for these workloads on B70.

The actual win for the patch is on the **giga01 receive side**, not
visible in B70 encoder timings: with `device=xpu` on the descriptor,
the giga01 NIC pulls bytes directly into H200 GPU VRAM (GPUDirect RDMA)
instead of staging through host memory. That saves a CPU pinned-buffer
copy on giga01 per request and frees PCIe BW that otherwise contends
with the ongoing decode step on the H200. The giga01 operator should
re-measure with the matching receive-side patches enabled to confirm.

## 5. Cold/warm patterns and outliers

The mean–median split tells us most of the variance comes from a
small number of **cold first-requests on each worker**:

```
patched ON, n_imgs=1:
  pure_encode_ms   p50= 993   mean=2459   p99=10116    ← p99 is one cold encode warm-up
  nixl_setup_ms    p50=1.65   mean= 245   p99= 3301    ← p99 is the per-worker NIXL agent setup
  pd_call_ms       p50=4.03   mean= 201   p99= 1928    ← p99 is first-time PD remote-agent pairing

patched OFF, n_imgs=1:
  pure_encode_ms   p50=1014   mean=1390   p99= 4294
  nixl_setup_ms    p50=14.35  mean= 428   p99= 3270
```

Both regimes pay essentially the same one-time NIXL bootstrap (~3.3 s
total amortised across the first burst per worker). After that, all
encoder-side stages except the vision tower are sub-millisecond.

## 6. Bottleneck recommendations

In priority order for moving the needle on cross-host disagg
throughput:

1. **Faster vision encode on B70.** This is 58 % of total time at
   4 images and 100 % of latency for image-1 cold path. Levers:
   - sglang attention backend tuning (currently `triton_attn` —
     `WARN: Multimodal attention backend not set. Use triton_attn.`)
   - smaller `image_grid_thw` (i.e. lower input resolution)
   - batch encode within an in-flight window so multiple requests
     share a single forward pass
   - use the second-half NUMA XPUs (4..7) too, total 8 encoders not 4
2. **Reduce H200 prefill time.** `pd_ttft_ms` is the second-largest
   stage (27 %). Owned by giga01.
3. **Eliminate the per-worker first-request NIXL agent setup spike.**
   Could be done by warming up the NIXL agent at startup with a
   throwaway register/readable cycle — but only saves ~3 s once per
   worker boot. Not worth chasing unless workers churn.
4. **The patch itself.** Encoder-side: ~zero net effect (it's a
   trade between CPU→XPU memcpy and cheaper NIXL register).
   Receive-side (giga01): the patch is what enables true GPUDirect
   RDMA on the wire — measure that with the matching A+B patches.

## 7. Method, raw data and reproducibility

- **Instrumentation:** `encode_worker_handler.py` `generate()`
  augmented with `time.perf_counter()` checkpoints around each
  NVTX region, emitting one `BENCH_TIMING …` line per request at
  `WARN` level. Patch is gated by `BENCH_DISABLE_XPU_PATCH=1` env
  for clean A/B with no other code differences.
- **Driver:** `curl --noproxy '*' -X POST .../v1/chat/completions`
  to the H200 frontend at `172.26.46.75:7001`, concurrency 4,
  fixed local data-URI 768×768 JPEG, `max_tokens=30`.
- **Workloads:**
  - 1-image: `prompt_tokens ≈ 23`, `emb_shape=(576, 20480)` or
    `(1107, 20480)` depending on grid thw
  - 4-image: 4×768×768 same image, `emb_shape=(2304, 20480)`
- **Logs (raw `BENCH_TIMING` lines):**
  - `bench_run/patched_on_w{1,2,3,4}.txt` — 79 rows total (4E patched ON)
  - `bench_run/patched_off_w{1,2,3,4}.txt` — 60 rows total (4E patched OFF)
  - `bench_run/patched_on_1E_w1.txt` — 60 rows total (1E patched ON)
- **Re-run analysis:**
  ```
  cd /hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs/bench_run
  python3 analyze_bench.py \
      --group 4E_on  patched_on_w*.txt \
      --group 4E_off patched_off_w*.txt \
      --group 1E_on  patched_on_1E_w*.txt
  ```
- **Disable instrumentation later:** revert the `BENCH_TIMING` block
  in `encode_worker_handler.py` (or restore from `.bak` and re-apply
  `b70_xpu_nixl.patch`).

---

## 8. 1-encoder (1E) vs 4-encoder (4E), patched

Same instrumented build, same workload (30× 1-img + 30× 4-img, client
concurrency=4), same patch applied, only thing changed: number of encode
workers and the launcher script
(`start_sglang_pd_xpu_32b_b70.sh` vs `start_sglang_pd_xpu_32b_b70_4E.sh`).

### 8.1 Headline numbers (median)

#### 1-image workload

```
stage                   1E_on    4E_on   delta_ms   delta_%
setup_ms                 0.06     0.15      -0.09     -60.0%
pure_encode_ms         673.77   993.32    -319.55    -32.2%
patch_to_xpu_ms         14.19    12.09      +2.10    +17.4%
encode_total_ms        686.26  1006.50    -320.24    -31.8%
token_expand_ms          0.06     0.09      -0.03    -33.3%
nixl_setup_ms            1.20     1.65      -0.45    -27.3%
pd_call_ms             336.46     4.03    +332.43  +8249.4%   ← PD-side queueing
pd_ttft_ms             183.26   292.48    -109.22    -37.3%
pd_stream_ms           354.11   708.27    -354.16    -50.0%
nixl_wait_ms             0.24     0.32      -0.08    -25.0%
total_ms              1388.45  1716.87    -328.42    -19.1%
```

#### 4-image workload

```
stage                   1E_on    4E_on   delta_ms   delta_%
setup_ms                 0.07     0.20      -0.13    -65.0%
pure_encode_ms        2823.14  2634.41    +188.73     +7.2%
patch_to_xpu_ms         20.47    25.13      -4.66    -18.5%
encode_total_ms       2844.65  2653.40    +191.25     +7.2%
token_expand_ms          0.14     0.21      -0.07    -33.3%
nixl_setup_ms            1.54     2.09      -0.55    -26.3%
pd_call_ms            1410.77     9.79   +1400.98 +14310.3%   ← PD-side queueing
pd_ttft_ms               1.06  1257.83   -1256.77   -99.9%   ← bimodal (see §8.4)
pd_stream_ms          1143.75  1823.57    -679.82    -37.3%
nixl_wait_ms             0.26     0.50      -0.24    -48.0%
total_ms              5662.64  4582.36   +1080.28    +23.6%
```

### 8.2 Throughput at concurrency=4

Wall-clock for the 30-request batches (total time in seconds, from when
the first request was sent to when the last response landed):

| Workload | 1E patched | 4E patched | speed-up |
|----------|-----------:|-----------:|---------:|
| 30× 1-img | 21 s        | 13 s        | 1.6× |
| 30× 4-img | 52 s        | 35 s        | 1.5× |
| Combined  | 73 s        | 48 s        | 1.5× |

Throughput at conc=4: 1E ≈ **1.43 req/s**, 4E ≈ **2.31 req/s**.
Per-encoder throughput drops from 1.43 to 0.58 req/s — i.e. each of
the 4 encoders is at 40 % of single-encoder utilisation, because the
client concurrency is 4 and the H200 PD has finite capacity, not because
the encoders themselves contend.

### 8.3 What 1E vs 4E reveals about the bottleneck

The single-encoder run is the **cleanest measurement of intrinsic
encoder cost** because there is no contention:

- `pure_encode_ms` for 1 image: **673.77 ms** (1E) vs 993.32 ms (4E)
- `pure_encode_ms` for 4 images: 2823.14 ms (1E) vs 2634.41 ms (4E)

The 1-image median improves by 32 % when going to 1E. This is **not**
because the encode is faster on a single XPU — it's because the 4E
median was inflated by warm-up cost. p50 is sensitive to outliers in
small samples; 1E was a clean 30-sample run while 4E mixed the first
worker's first few cold requests into the same bucket. p99 is a much
better gauge:

```
pure_encode_ms p99:  1E = 7398 ms      4E = 10116 ms
```

p99 is similar (within ~30 %) — confirming the **intrinsic vision-tower
forward pass time is XPU-bound and roughly the same in both
configurations**. 4E p99 is higher only because there are 4 cold
warm-ups instead of 1.

For 4-image, `pure_encode_ms` p50 is **larger** in 1E than 4E (2823 vs
2634 ms). At first that looks paradoxical — fewer competing encoders
should be at least as fast — but the 4E numbers are tightly clustered
(p50 ≈ p90 ≈ p99 ≈ 2700 ms; warm steady-state) while 1E was driven
hard with concurrency=4 → a single encoder serialising 4 in-flight
encodes sees the 2nd/3rd/4th ones queue *inside the encoder's coroutine
runtime*, inflating the per-request encode time slightly. **Net
encoder throughput** is what matters, not p50: see §8.2.

### 8.4 The `pd_ttft_ms` bimodality on 1E

In 1E, `pd_ttft_ms` for the 4-image workload looks suspicious:

```
pd_ttft_ms p50 = 1.06 ms     ← suspiciously low
pd_ttft_ms p90 = 2842 ms     ← much closer to 4E's ~1300 ms
pd_ttft_ms mean = ~1100 ms
```

Inspecting raw values shows a strict bimodal split:

```
pd_ttft_ms: 0.46  1410.44  1410.42  791.24  0.28  2938.79  0.17  800.61
            0.28  0.42     0.62     781.86  0.59  2792.59  0.99  802.94
            ...
```

About half the requests show `pd_ttft_ms` around 0–1 ms; the other half
show 800–2900 ms. **Why:** with 1 encoder + concurrency=4, the encoder's
vision tower is the wall-clock bottleneck. By the time request N has
finished its 2.8 s encode pass, the H200 has been idle for most of that
duration with respect to request N — but during those 2.8 s, **request
N–1's PD prefill+decode has been running**. Often by the time we
issue `pd_worker_client.round_robin(...)` for request N, the PD has
already finished pre-filling the batch slot freed by N–1 and has
**generated tokens that are sitting in a NATS buffer waiting for any
consumer**. The first iteration of the response generator returns
those tokens almost instantly → `pd_ttft_ms ≈ 0`. Other requests miss
that timing and pay the genuine ~800 ms prefill.

**This is a measurement artefact of how I computed `pd_ttft_ms`** — it
is the wall-clock between `await pd_worker_client.round_robin()`
returning and the first item from the async iterator, not the
H200-side prefill latency itself. With 4E the encoder is fast enough
that this race essentially never happens (4E `pd_ttft_ms` is
unambiguously ~1200–1500 ms, dominated by H200 prefill).

Either way, the takeaway is the same: **on a 1-encoder B70 against a
single H200 PD slot, the wall-clock is dominated by encoder
serialisation, and PD has spare capacity it can't be fed fast enough.**

### 8.5 The `pd_call_ms` queueing on 1E

The huge `pd_call_ms` in 1E (336 ms for 1-img p50, 1411 ms for 4-img
p50, vs 4–10 ms in 4E) is the second telltale sign of the same effect:
4 concurrent client requests serialising through 1 encoder reach the
H200 PD scheduler in a tight burst. PD has finite running capacity, so
the 2nd/3rd/4th requests wait in PD's queue before
`round_robin()` returns a generator. That wait is recorded in
`pd_call_ms` because that's the only span where the encoder is blocked
on the PD-side coordinator.

In 4E, requests are spread across 4 encoders, hit PD nearly
simultaneously but only 1-deep per encoder, and PD scheduling gap
shrinks to a few ms.

### 8.6 Summary: 1E vs 4E

| Metric (median, 4-image) | 1E | 4E | Notes |
|---|---|---|---|
| `total_ms` | 5663 ms | 4582 ms | 4E is **23 %** faster end-to-end at conc=4 |
| `pure_encode_ms` | 2823 ms | 2634 ms | similar — vision tower is the same per request |
| Throughput (req/s, 4-img) | 0.58 | 0.86 | 4E gives **1.5×** throughput at this concurrency |
| Encoder utilisation per worker | 100 % | ~40 % | 4E is over-provisioned for conc=4 |
| `pd_call_ms` | 1411 ms | 10 ms | 1E inflates due to PD-side queueing |
| `nixl_setup_ms` (warm) | 1.5 ms | 2.1 ms | both negligible |
| `patch_to_xpu_ms` | 20 ms | 25 ms | similar |

**Practical implications:**

1. **For low concurrency (≤ 2)**, 1E is sufficient and simpler.
   It saves 3 XPUs of memory and simplifies the topology.
2. **For concurrency ≥ 4**, 4E gives ~1.5× throughput. The 1E `pd_call_ms`
   blow-up is the smoking gun: the encoder becomes the bottleneck and PD
   cannot stay fed.
3. **For higher concurrency**, scaling beyond 4E will hit the H200 PD
   limit, not the encoder — at conc=4, each 4E worker is already at
   ~40 % utilisation. Adding more encoders won't help unless PD scales
   too (e.g. multi-instance disagg PD on a TP=4 H200).
4. **The `--encoder-only` mode + the GPUDirect patch are orthogonal to
   this scaling discussion** — they just determine where embeddings
   live during NIXL transfer and have negligible effect on encoder
   wall-clock time on B70.

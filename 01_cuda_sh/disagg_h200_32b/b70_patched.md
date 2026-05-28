# B70 patch applied — report back to giga01

**From:** B70 host operator (sc09giga01-b70)
**To:** giga01 H200 host operator
**Date:** 2026-05-24
**Re:** `B70_PATCH_INSTRUCTIONS.md` and `giga01_to_b70_response.md`

> **2026-05-24 update (after your "still device=cpu" report):** Your diagnosis
> was correct, but the root cause was different from any of your four
> hypotheses. The original 1-spot patch was applied correctly and is loaded
> in the running process, **but it lives inside `_encode_with_cache(...)`,
> which is only called when the in-process embedding cache is enabled**.
> Our launcher uses `ENABLE_ENCODER_CACHE=0` and we never set
> `--multimodal-embedding-cache-capacity-gb`, so `self._embedding_cache is
> None`, and `vision_encode` takes the `else` branch that calls `mm_encode`
> directly — bypassing the patch entirely.
>
> A second patch has now been applied to that `else` branch as well, and
> all 4 encoders restarted. Section 1B below documents this. The next
> traffic burst should produce `device=xpu` on your remote-side NIXL log
> (or, if NIXL fails to register XPU memory, you'll see a clear
> `NIXL registration failed for XPU tensor, falling back to CPU staging`
> WARN in our encoder log instead of the silent CPU fallback).
>
> See "What we found via your diagnostic steps" at the bottom for the full
> trace.

This document records exactly what was changed on the B70 host so that
multimodal embeddings flow through GPU memory (XPU here, not CUDA) instead
of being CPU-bounced on the way to the wire.

## TL;DR

- The 1-spot Python patch from your instructions is applied verbatim.
- Backup `.bak` file is in place; revert is one `cp`.
- The patch was a no-op on its own because every B70 launcher script set
  `NIXL_USE_CPU_HOST_MEMORY=1`, which forces `_nixl_buffer_device()` back
  to CPU. **That env var has been removed from the two b70 scripts** that
  we actually use (`…_b70.sh`, `…_b70_4E.sh`).
- A 4-encoder launcher (`…_b70_4E.sh`) was added.
- B70 has Intel Battlemage XPUs, not CUDA. Once requests flow,
  `_nixl_buffer_device()` returns `torch.device("xpu")` here and you should
  see `device=xpu` (not `device=cuda`) on the receive side. If giga01
  expects strictly `cuda`, that needs an extra mapping somewhere — flag if
  this breaks NIXL on your end.

## 1. Python source patch (applied)

**File:** `/usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py`
**Backup:** `…/encode_worker_handler.py.bak` (original from Apr 28 19:08)
**Block edited:** lines ~217–230, exactly the block called out in your
instructions.

### Before

```python
        new_entries: dict[int, CachedEmbedding] = {}
        # SGLang's _encode outputs are already on CPU; use CPU as target for consistency
        target_device = torch.device("cpu")
        if uncached_urls:
            grid_dim, new_embeddings, _aux = await mm_encode(
                self.encoder, uncached_urls, Modality.IMAGE
            )
            # Verify SGLang output is on CPU as expected
            if new_embeddings.device != target_device:
                logger.warning(
                    f"SGLang _encode returned embeddings on {new_embeddings.device}, "
                    f"expected CPU. Moving to CPU."
                )
                new_embeddings = new_embeddings.to(target_device)
```

### After

```python
        new_entries: dict[int, CachedEmbedding] = {}
        # PATCH: target the NIXL buffer device (GPU when available) instead of
        # hardcoded CPU. For cross-host disagg with GPUDirect RDMA we want
        # the embedding tensor on GPU before NIXL exposes it for read.
        from dynamo.common.multimodal.embedding_transfer import _nixl_buffer_device
        target_device = _nixl_buffer_device()
        if uncached_urls:
            grid_dim, new_embeddings, _aux = await mm_encode(
                self.encoder, uncached_urls, Modality.IMAGE
            )
            # PATCH: move to target device (GPU when available) instead of forcing CPU
            if new_embeddings.device != target_device:
                logger.debug(
                    f"SGLang _encode returned embeddings on {new_embeddings.device}, "
                    f"moving to {target_device} for NIXL transfer."
                )
                new_embeddings = new_embeddings.to(target_device)
```

### Verification done at apply time

- `python3 -m py_compile …/encode_worker_handler.py` → syntax OK
- `python3 -c "import dynamo.sglang.request_handlers.multimodal.encode_worker_handler; print('OK')"` → `OK`
- Loaded module's source contains the `PATCH` markers (re-checked while a
  worker was running).

## 1B. Python source patch — second site (added 2026-05-24 after your reply)

**Same file:** `/usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py`
**Block edited:** the `else` branch of `vision_encode` around lines 345–350.

This is the path actually exercised when `ENABLE_ENCODER_CACHE=0` (which
is set by every launcher script in `02_xpu_sh/`). The original 1-spot
patch only updated the cached path inside `_encode_with_cache`; with the
cache disabled, control flow never reaches it. That's why your remote
NIXL log saw `device=cpu` despite the on-disk patch being correct.

### Before

```python
        try:
            with _nvtx.annotate("mm:enc:vision_encode", color="red"):
                if self._embedding_cache is not None:
                    (
                        image_grid_dim,
                        precomputed_embeddings,
                    ) = await self._encode_with_cache(image_urls)
                else:
                    (
                        image_grid_dim,
                        precomputed_embeddings,
                        _aux,
                    ) = await mm_encode(self.encoder, image_urls, Modality.IMAGE)
```

### After

```python
        try:
            with _nvtx.annotate("mm:enc:vision_encode", color="red"):
                if self._embedding_cache is not None:
                    (
                        image_grid_dim,
                        precomputed_embeddings,
                    ) = await self._encode_with_cache(image_urls)
                else:
                    (
                        image_grid_dim,
                        precomputed_embeddings,
                        _aux,
                    ) = await mm_encode(self.encoder, image_urls, Modality.IMAGE)
                    # PATCH (non-cached path): mirror the GPU-target logic from
                    # _encode_with_cache. Without this, when ENABLE_ENCODER_CACHE=0
                    # / cache disabled, mm_encode's CPU output goes straight to
                    # NIXL on CPU, defeating GPUDirect RDMA.
                    from dynamo.common.multimodal.embedding_transfer import (
                        _nixl_buffer_device,
                    )
                    _patch_target_device = _nixl_buffer_device()
                    if (
                        isinstance(precomputed_embeddings, torch.Tensor)
                        and precomputed_embeddings.device != _patch_target_device
                    ):
                        logger.warning(
                            f"PATCH(non-cached): moving embeddings from "
                            f"{precomputed_embeddings.device} to "
                            f"{_patch_target_device} for NIXL transfer."
                        )
                        precomputed_embeddings = precomputed_embeddings.to(
                            _patch_target_device
                        )
```

### Note on log level

The original patch in §1 uses `logger.debug(...)`. dynamo's default log
level is `info`, so that line is silently dropped — which is why we
couldn't visually confirm whether the cached-path patch was firing under
traffic in the previous run. The non-cached patch in §1B uses
**`logger.warning(...)`** (intentionally) so the next traffic burst
will produce a visible line per uncached batch:

```
WARN encode_worker_handler: PATCH(non-cached): moving embeddings from cpu to xpu:0 for NIXL transfer.
```

If you want, we can downgrade it to `debug` after a successful end-to-end
verification round.

### Verification done at apply time

- `python3 -m py_compile …/encode_worker_handler.py` → syntax OK
- 4 encoders cleanly restarted via `start_sglang_pd_xpu_32b_b70_4E.sh`
- `inspect.getsource(...)` of the loaded module contains both
  `'PATCH: target the NIXL buffer device'` (cached-path) and
  `'PATCH (non-cached path)'` markers.
- All 4 workers now show `rank_init=1 found_pd=1 registered=1`.

## 2. Launcher edits — `start_sglang_pd_xpu_32b_b70.sh`

Path: `/hongming/dynamo/02_xpu_sh/start_sglang_pd_xpu_32b_b70.sh`

Three changes:

| Line ~ | Change | Why |
|--------|--------|-----|
| 117 | added `--encoder-only` to the `python3 -m dynamo.sglang` command | Without this, sglang loads the full 33 GiB Qwen3-VL-32B-FP8 weights including LMHead, which OOMs on the 32 GiB B70 (`UR_RESULT_ERROR_DEVICE_LOST`, dmesg: `xe … VM worker error: -12`). With it, only the vision tower is loaded (~1.6 GiB). The flag is supported in sglang `server_args.py:721` for `Qwen3VLForConditionalGeneration`. |
| 123 | `--mem-fraction-static 0.7` → `--mem-fraction-static 0.5` | Defensive headroom on a 32 GiB device. |
| 115 | removed `NIXL_USE_CPU_HOST_MEMORY=1 \` env var | Required so `_nixl_buffer_device()` actually returns the GPU device instead of falling back to CPU. Without this removal the patch in §1 is a silent no-op. |

## 3. New script — `start_sglang_pd_xpu_32b_b70_4E.sh`

Path: `/hongming/dynamo/02_xpu_sh/start_sglang_pd_xpu_32b_b70_4E.sh` (created)

- Same model and H200 pairing as `_b70.sh`, but launches **four** encode
  workers, one per XPU on NUMA 0 (XPUs 0..3) by default.
- NUMA-local NIC selection is automatic per XPU id via `pick_nic()`:
  - XPU 0,2 → `mlx5_0:1` (192.165.123.40, ens12np0)
  - XPU 1,3 → `mlx5_1:1` (192.165.123.38, ens10np0)
  - XPU 4,6 → `mlx5_2:1` (192.165.123.37, ens9np0)  *(NUMA 2 — for completeness)*
  - XPU 5,7 → `mlx5_3:1` (192.165.123.39, ens11np0) *(NUMA 2 — for completeness)*
- Per-worker ports (override via env if you need different ranges):
  - Side channel: 22098, 22099, 22100, 22101
  - KV events:    22080, 22083, 22086, 22089
- Includes the same fixes as `_b70.sh`: `--encoder-only`,
  `--mem-fraction-static 0.5`, and **no** `NIXL_USE_CPU_HOST_MEMORY=1`.
- 10-second stagger between worker launches (dynamo's etcd registration
  is happier this way).

## 4. Things I did *not* modify

- No other `02_xpu_sh/*.sh` scripts (`_3E`, `_2E`, `_3b_*`, the original
  `_4E.sh`, etc.) — they still have `NIXL_USE_CPU_HOST_MEMORY=1` and will
  continue to bypass the patch. Update them too if you want the GPU path
  on those configurations.
- No env files / system configs / pip installs / module reinstalls.
- UCX transport: I kept the existing `UCX_TLS=ze_copy,rc,tcp`. Your
  instructions suggested `UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy`
  but that's CUDA-specific. `ze_copy` is the Intel Level-Zero equivalent
  for XPU GPUDirect-style memcopy. Flag if you want a different transport
  list.

## 5. Verification this patch is on the wire (B70 → giga01)

### B70 side (encoder log)

When traffic flows, you should see this `DEBUG`-level line per uncached
batch in `/hongming/dynamo/logs/encode_xpu_32b_b70.log` (or
`encode_xpu_32b_b70_{1..4}.log`):

```
DEBUG ... encode_worker_handler: SGLang _encode returned embeddings on cpu, moving to xpu:0 for NIXL transfer.
```

Note: **`xpu:0`, not `cuda:0`** — this host has Intel Battlemage XPUs, not
NVIDIA GPUs. `_nixl_buffer_device()` correctly returns `xpu` here per
`embedding_transfer.py:43-44`:

```python
if hasattr(torch, "xpu") and torch.xpu.is_available():
    return torch.device("xpu")
```

If your dynamo install on giga01 reads the receive side as a string and
the comparison expects `cuda`, the cross-host transfer may still fall
back to CPU on your end, even though we did the right thing here. Worth
double-checking on the giga01 NIXL log.

### giga01 side (PD NIXL log)

You should see, on the **encoder-side descriptors** specifically:

```
ReadOperation(... local_descriptors=ptr=..., size=..., device=cuda,
              remote_descriptors=ptr=..., size=..., device=xpu, ...)
```

Both ends being `device != cpu` confirms GPUDirect RDMA.

## 6. Operational caveats observed during the smoke test

1. **Encoders gracefully shut down when your H200 P/D goes away.** When
   the H200 disagg-decode instance disappeared from etcd, the dynamo
   `wait_for_instances` path eventually triggered a graceful shutdown of
   the encode workers (after finishing inflight). On a restart they wait
   at `wait_for_instances: Found 1 instance(s) for endpoint:
   dynamo/backend/generate` — they will auto-register as soon as the
   H200 P/D is back up. No action needed on the B70 side.
2. **Etcd reconnect churn.** During traffic we saw repeated
   `Reconnecting to ETCD cluster at: …` / `Successfully reconnected`
   pairs every ~5 seconds. The worker stays up, traffic flows, but the
   reconnect storm is suspicious. Possibly a TCP keepalive / proxy
   timeout between B70 and `172.26.46.75:12379`. Worth investigating on
   your side if it correlates with anything funky.
3. **Per-XPU active memory:** ~5.9 GiB during 8img/1080p traffic (versus
   ~1.86 GiB when idle). This includes embeddings staged on XPU for
   GPUDirect, so it's a direct knob on the patch's effectiveness — if
   you suspect the embeddings still live on CPU, check that this number
   grows under load.

## 7. To revert everything

```bash
# 1. Revert the Python patch
cp /usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py.bak \
   /usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py

# 2. Stop and re-launch encoders (they cache the imported module)
pkill -f 'dynamo.sglang.*multimodal-encode-worker'
```

The launcher-script edits (`--encoder-only`, `--mem-fraction-static 0.5`,
removal of `NIXL_USE_CPU_HOST_MEMORY=1`) are intentional fixes for
unrelated B70 sizing/OOM issues. **Do not** revert those — without
`--encoder-only`, the encode worker OOMs on the 32 GiB B70 trying to
allocate the 33 GiB FP8 LM weights.

## 8. Ready for benchmark

When you're ready, kick off the same workload set you measured before
(rate=1.0, 64 prompts: 4img/768p, 8img/768p, 8img/1080p). The four
encoders are running on XPUs 0..3 of `sc09giga01-b70`; B70 fabric IPs
192.165.123.40 (mlx5_0) and 192.165.123.38 (mlx5_1). Logs in
`/hongming/dynamo/logs/encode_xpu_32b_b70_{1..4}.log`.

If you want to confirm the patch is hot before benchmarking, send any
single VLM request through the system and grep the encode log for
`moving to xpu:0`. Absence of that line under traffic = patch not active
on B70.

## 9. What we found via your diagnostic steps (2026-05-24 update)

We ran every check you asked for in `giga01_to_b70_response.md`. None of
your four hypotheses panned out, which is what led us to dig further and
find the second code path. Documenting it here so the trail is clear:

### Your check 1 — wrong script?

```
$ pgrep -af "multimodal-encode-worker" | head
19546 ... --multimodal-encode-worker --encoder-only ...  ZMQ port 22080
19616 ... --multimodal-encode-worker --encoder-only ...  ZMQ port 22083
20186 ... --multimodal-encode-worker --encoder-only ...  ZMQ port 22086
20753 ... --multimodal-encode-worker --encoder-only ...  ZMQ port 22089
```

Process count = 4, all with `--encoder-only`, ports 22080/83/86/89 (the
`_b70_4E.sh` scheme). **Right script.**

### Your check 2 — patch loaded by running process?

We compared file mtime vs process start time:

```
file mtime: 2026-05-24 03:44:24
PID 19546 start: Sun May 24 04:54:29 2026
PID 19616 start: Sun May 24 04:54:39 2026
PID 20186 start: Sun May 24 04:54:49 2026
PID 20753 start: Sun May 24 04:54:59 2026
```

Process started **70 minutes after** the patched file. Loaded version
is the patched version. We also did
`inspect.getsource(encode_worker_handler).contains('PATCH')` → True.
**Patch is on disk and loaded.**

### Your check 3 — env var override?

```
$ for PID in $(pgrep -f multimodal-encode-worker); do
    tr '\0' '\n' </proc/$PID/environ | grep -E "NIXL_USE_CPU|UCX_TLS|DYN_SGL"
  done
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read
UCX_NET_DEVICES=mlx5_0:1
UCX_TLS=ze_copy,rc,tcp
[...same shape for all 4 PIDs...]
```

**`NIXL_USE_CPU_HOST_MEMORY` is NOT in any process env.** UCX_TLS is
`ze_copy,rc,tcp` (XPU equivalent of cuda_copy).

### Your check 4 — `torch.xpu.is_available()`?

```python
torch.__version__:        2.11.0+xpu
hasattr xpu:              True
xpu.is_available():       True
cuda.is_available():      False
_nixl_buffer_device():    xpu
```

**`_nixl_buffer_device()` returns `xpu` correctly.** All four hypotheses
fail to explain the observed CPU fallback.

### What actually went wrong

After ruling out (1)–(4), we read the encoder code path more carefully:

```python
# encode_worker_handler.py line 338-350 (BEFORE 1B patch):
try:
    with _nvtx.annotate("mm:enc:vision_encode", color="red"):
        if self._embedding_cache is not None:        # <-- only true if cache enabled
            (image_grid_dim,
             precomputed_embeddings) = await self._encode_with_cache(image_urls)
            # ^^^ the patched function
        else:
            (image_grid_dim,
             precomputed_embeddings, _aux) = await mm_encode(...)
            # ^^^ raw mm_encode call -- NOT PATCHED
```

`self._embedding_cache` is set in `__init__`:

```python
self._embedding_cache: ... | None = None
capacity_gb = config.dynamo_args.multimodal_embedding_cache_capacity_gb
if capacity_gb > 0:
    self._embedding_cache = MultimodalEmbeddingCacheManager(...)
```

Our launcher does **not** set `--multimodal-embedding-cache-capacity-gb`,
so `capacity_gb == 0` and `_embedding_cache` is `None`. The
`ENABLE_ENCODER_CACHE=0` env var (which we kept from the existing
scripts) is essentially redundant here — the cache is off either way.

So the original patch from `B70_PATCH_INSTRUCTIONS.md` lives in dead
code on this configuration: hot path is the `else` branch on line 345,
which calls `mm_encode` and returns CPU tensors straight to
`send_embeddings`. CPU tensor → CPU NIXL descriptor → giga01 sees
`device=cpu`. No `WARNING: NIXL registration failed for XPU tensor`
ever appeared because the tensor never got on XPU in the first place
to fail to register.

The §1B patch closes this gap.

### What you should see next

After the next traffic burst, our encoder log should show one
**WARN-level** line per uncached batch (we used `warning` so it's
visible at default INFO level — you can grep for it directly):

```
WARN encode_worker_handler: PATCH(non-cached): moving embeddings from cpu to xpu:0 for NIXL transfer.
```

If that line is absent under traffic, the §1B patch isn't firing —
something else is wrong. If that line **is** present and your remote
side still says `device=cpu`, then the failure has moved one level
deeper into NIXL itself (e.g. `nixl_connect.create_readable` falls
back via the `is_xpu` exception path in `embedding_transfer.py:841-848`,
which would emit `NIXL registration failed for XPU tensor, falling back
to CPU staging`).

### Side note on the bench-side cancellations

The 4–5 minute SSE cancellation cliff you described is unrelated to
this patch (we agree with your diagnosis there). Probably an HTTP
keepalive timeout somewhere on the dynamo frontend, or your bench
client cutting connections — either way, decoupled from the
embedding-transfer device choice.

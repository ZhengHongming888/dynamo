# Patch instructions for B70 encoder (cross-host disagg)

**Recipient:** B70 host operator
**From:** giga01 host operator
**Date:** 2026-05-24
**Goal:** Apply a 1-spot Python patch on B70's encoder install so multimodal embeddings flow through GPU memory (GPUDirect RDMA) instead of getting bounced through CPU memory on the way to the wire.

## Why this patch

We measured cross-host disagg (your B70 encoder → giga01 H200 PD) and found
the bottleneck is per-request CPU-bounce in the NIXL embedding transfer:

```
Cross-host disagg @ rate=1.0, 8img/1080p:  0.13 RPS   (vs 0.95 same-host TP=2 agg)
Cross-host disagg @ rate=1.0, 4img/768p:   1.02 RPS   (vs 3.1 same-host TP=2 agg)
```

The PD log shows every NIXL transfer using `device=cpu` for both ends, so the
RoCE wire is doing CPU→CPU instead of GPU→GPU. The hardcoded CPU target in the
sglang encoder handler is the encoder-side cause:

```python
# /opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py:218-219
new_entries: dict[int, CachedEmbedding] = {}
# SGLang's _encode outputs are already on CPU; use CPU as target for consistency
target_device = torch.device("cpu")           # <-- this line forces CPU
```

I've already patched the receive side on giga01 (in
`embedding_transfer.py:882, 915`). For the full GPU path to work end-to-end,
the encoder side on B70 needs the matching change.

## Step 1 — Locate the file

The exact path depends on how dynamo was installed. Common locations:

```bash
# Most common (pip install in venv):
/opt/venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py

# Alternative (system install):
/usr/lib/python3.12/site-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py

# Find it on B70:
python3 -c "import dynamo.sglang.request_handlers.multimodal.encode_worker_handler as m; print(m.__file__)"
```

## Step 2 — Make a backup

```bash
cp <path>/encode_worker_handler.py <path>/encode_worker_handler.py.bak
```

## Step 3 — Apply the patch

Edit `encode_worker_handler.py`. Find this block (around line 217-230):

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

Replace with:

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

The two changes are:
1. `target_device = torch.device("cpu")` → `target_device = _nixl_buffer_device()` (with import added)
2. `expected CPU. Moving to CPU.` → `moving to {target_device} for NIXL transfer.` (cosmetic, less alarming logging)

## Step 4 — Verify the patch loads

```bash
python3 -c "import dynamo.sglang.request_handlers.multimodal.encode_worker_handler; print('OK')"
```

Should print `OK` with no traceback.

## Step 5 — Restart your encoder workers

The encoder workers cache the imported Python module, so they need a full process
restart for the patch to take effect. Stop and re-launch each encoder worker
process (or scale your replica set down to 0 and back up).

## Step 6 — Reconnect to giga01

The giga01 control plane is currently up:
- `NATS_SERVER=nats://172.26.46.75:14222`
- `ETCD_ENDPOINTS=http://172.26.46.75:12379`

Make sure each restarted encoder still has these envs (and the rest from your
existing setup):
```
NATS_SERVER=nats://172.26.46.75:14222
ETCD_ENDPOINTS=http://172.26.46.75:12379
VLLM_NIXL_SIDE_CHANNEL_HOST=192.165.123.40   # B70 mlx5_0 RoCE IP (or your matching NIC)
VLLM_NIXL_SIDE_CHANNEL_PORT=20098
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy
UCX_NET_DEVICES=mlx5_0:1
UCX_MEMTYPE_CACHE=0
DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read
DYN_TCP_MAX_MESSAGE_SIZE=268435456
ENABLE_ENCODER_CACHE=0
PYTHONHASHSEED=0
TRANSFER_LOCAL=0
```

## How to confirm the patch is working

Once the encoder reconnects, look for this in your encoder log:

```
DEBUG ... encode_worker_handler: SGLang _encode returned embeddings on cpu, moving to cuda:0 for NIXL transfer.
```

This confirms the patch is in effect (the move-to-cuda happens). Old behavior
would have been `expected CPU. Moving to CPU.` (a `WARNING`).

On giga01 we'll then see in the NIXL log:

```
ReadOperation(... local_descriptors=ptr=..., size=668467200, device=cuda, remote_descriptors=ptr=..., size=668467200, device=cuda, ...)
```

Both sides showing `device=cuda` instead of `device=cpu` means we're getting
true GPUDirect RDMA.

## Risks / known issues

1. **GPU OOM**: Moving embeddings to GPU adds ~640 MB per in-flight request to encoder
   GPU memory pressure. If you used `--mem-fraction-static 0.95` before, drop to
   `0.85` first to leave headroom. Watch for OOM in encoder log:
   ```
   torch.cuda.OutOfMemoryError: CUDA out of memory.
   ```
2. **Cupy fallback**: dynamo's `nixl_connect` warns at startup if `cupy` is not
   installed. The patch still works without cupy (uses raw torch tensor on GPU);
   but installing cupy makes some intermediate path slightly faster. To install:
   ```
   pip install cupy-cuda13x   # match your CUDA major version
   ```
3. **Not for production**: This is an unmerged ad-hoc patch. Don't ship it. It's
   a measurement scaffold to confirm the bottleneck is the CPU bounce. The real
   fix should be a config-driven knob in dynamo upstream.

## To revert

```bash
mv <path>/encode_worker_handler.py.bak <path>/encode_worker_handler.py
# then restart encoders
```

## What we'll measure once you're back up

Same as before: rate=1.0, 64 prompts, three workloads (4img/768p, 8img/768p, 8img/1080p).
We expect the biggest improvement on the largest workload (8img/1080p) where the
CPU bounce dominates.

Reach out when encoders are back online. I'll re-run the bench from giga01 and
share before/after numbers.

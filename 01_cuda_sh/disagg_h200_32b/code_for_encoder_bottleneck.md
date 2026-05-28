# B70 encoder bottleneck — code walkthrough

> **Companion to:** `b70_encoder_time_breakdown.md`
> **Date:** 2026-05-24
> **Question answered:** *Show me the actual Python code for the encoder bottleneck.*
>
> This is a literal source-code map of the path that consumes ~58 % of
> total encoder request time on B70 (median 2.6 s for a 4-image
> 768×768 prompt). Time stages here correspond 1:1 to the
> `pure_encode_ms` field in the `BENCH_TIMING` log lines.

## TL;DR

If you want **one line** as "the bottleneck", it is:

- **`/opt/sglang/python/sglang/srt/disaggregation/encode_server.py:1078`**
  → `mm_embedding: torch.Tensor = get_feature_fn([mm_item])`

…which dispatches into:

- **`/opt/sglang/python/sglang/srt/models/qwen3_vl.py:1210`**
  → `return self.visual(pixel_values, grid_thw=image_grid_thw)`

That's the ~600 M-parameter Qwen3-VL Vision Transformer running on
the B70 (Battlemage) XPU through sglang's `triton_attn` multimodal
attention backend. For a 4-image 768×768 batch it produces a
`(2304, 20480)` embedding tensor and accounts for the bulk of the
~2.6 s `pure_encode_ms` budget.

## Execution order

```
encoder_handler.generate (our patched file)
      └── mm_encode (thin wrapper)                                                (Step 1)
              └── MMEncoder._encode                                              (Step 2)
                      ├── _process_mm_items                                       (Step 3)
                      │       ├── _flatten_and_load_images   ← HTTP + PIL decode
                      │       └── self.image_processor(...)  ← HF resize/normalize/patchify
                      └── get_feature_fn([mm_item])                               (Step 4)
                              └── self.visual(pixel_values, grid_thw=…)  ← THE ViT
                              └── mm_embedding.cpu()  ← unconditional XPU→CPU memcpy
      └── PATCH: precomputed_embeddings.to(xpu)                                  (Step 5)
```

`pure_encode_ms` measured in our instrumentation covers all of Step 1
through end of Step 4. Step 5 is `patch_to_xpu_ms`.

## Step 1 — `mm_encode` (thin wrapper)

**File:** `/usr/local/lib/python3.12/dist-packages/dynamo/sglang/_compat.py:117`

```python
async def mm_encode(encoder: Any, mm_items: Any, modality: Any) -> tuple:
    """Version-safe wrapper around MMEncoder._encode().

    Always returns (grid_dim, embedding, aux_data). On sglang 0.5.9
    _encode takes no modality arg and returns a 2-tuple; on 0.5.10+ it
    takes modality and returns a 3-tuple. We try the new signature first
    and fall back to the old one.
    """
    try:
        result = await encoder._encode(mm_items, modality)
    except TypeError:
        # sglang 0.5.9: _encode(mm_items) -> (grid_dim, embedding)
        result = await encoder._encode(mm_items)

    if len(result) == 2:
        return (*result, None)
    return result
```

No real work here — just a compat shim. All measured time is downstream.

## Step 2 — `MMEncoder._encode` (orchestrator)

**File:** `/opt/sglang/python/sglang/srt/disaggregation/encode_server.py:1045`

This is the function the encoder handler calls. Everything inside is
counted in our `pure_encode_ms`.

```python
async def _encode(self, mm_items, modality: Modality) -> torch.Tensor:
    try:
        # === A: image download + preprocess (CPU + per-image net I/O) ===
        mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)  # 1047
    except NotImplementedError as e:
        raise InternalError(f"Not implemented error: {str(e)}")
    except Exception as e:
        raise BadRequestError(f"Failed to process mm items: {str(e)}")

    try:
        # support mm_cache
        mm_embedding = None
        mm_hash = None

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": modality,
                "feature": _convert(_get_mm_feature(mm_inputs, modality)),
            }
        )
        for k, v in mm_inputs.items():
            if k in _mm_feature_attrs[modality]:
                continue
            mm_item.set(k, _convert(v))

        if self.server_args.enable_prefix_mm_cache:        # disabled in our config
            mm_item.set_pad_value()
            mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
            async with self.mm_cache_lock:
                mm_cache = self.mm_cache.get([mm_item.hash])
                if mm_cache is not None:
                    mm_embedding = mm_cache.embedding

        if mm_embedding is None:
            # === B: THE HOT PATH — ViT forward + final .cpu() ===
            with torch.inference_mode():
                mm_embedding: torch.Tensor = get_feature_fn([mm_item])   # 1078  ← ViT compute on XPU
                mm_embedding = mm_embedding.cpu()                         # 1079  ← forces XPU sync + memcpy
            if len(mm_embedding.shape) != 2:
                mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

        if self.server_args.enable_prefix_mm_cache:
            async with self.mm_cache_lock:
                self.mm_cache.set(mm_hash, EmbeddingResult(embedding=mm_embedding))
        if self.profiler is not None:
            self.profiler.step()

        aux_data = _build_mm_aux_data(mm_inputs)
        return (
            _get_mm_grid_dim(mm_inputs, modality, self.model_type),
            mm_embedding,
            aux_data,
        )
    except BadRequestError as e:
        raise BadRequestError(f"Bad request error: {str(e)}")
    except Exception as e:
        raise InternalError(f"Internal encoding error: {str(e)}")
```

Two interesting things in the hot block:

- **Line 1078 `get_feature_fn([mm_item])`** is the synchronous ViT
  forward pass on the XPU. **Dominant cost.**
- **Line 1079 `mm_embedding.cpu()`** unconditionally copies the
  embedding back to host memory. This is the line our patch
  effectively *un-does* later — we move it back to `xpu` in the
  encoder handler before NIXL exposes it. A cleaner upstream fix
  would delete this `.cpu()` here, but that's an sglang change.

## Step 3 — `_process_mm_items` (download + preprocess)

**File:** `/opt/sglang/python/sglang/srt/disaggregation/encode_server.py:952`

This runs first inside `_encode`. For image inputs it does network I/O,
PIL decoding, and the HF image-processor's resize/normalize/patchify.

```python
async def _process_mm_items(self, mm_items, modality):
    if modality == Modality.IMAGE and self.image_processor:
        images = await self._flatten_and_load_images(mm_items)        # 954  ← HTTP download + PIL decode
        image_config = self.vision_config.get("image", {})
        if self.model_type in ["kimi_k25", "kimi_vl"]:
            images = self._normalize_kimi_encoder_images(images)
        processor_input = self.image_processor(images=images, **image_config)  # 958  ← HF resize + normalize + patchify
        if hasattr(self.model, "thinker"):  # for omni models
            get_feature_method = self.model.thinker.get_image_feature
        else:
            get_feature_method = self.model.get_image_feature          # 962  ← ViT entry point selected
    elif modality == Modality.VIDEO and self.video_processor:
        # ... video branch (not exercised in our benchmark)
        ...
    elif modality == Modality.AUDIO and self.audio_processor:
        # ... audio branch (not exercised in our benchmark)
        ...
```

Significant CPU cost here:

- `_flatten_and_load_images` blocks per URL on `aiohttp` fetch +
  `PIL.Image.open` decode.
- `self.image_processor(...)` runs the Hugging Face image processor:
  resize to grid resolution, normalize, patchify. For 768×768 with
  `merge_kernel_size=2`, this produces ~1107 patches per image.

This is why even our 1-image cold median was ~700–1000 ms — most of
that is pre-processing, not the ViT.

## Step 4 — `get_image_feature` (the actual ViT forward)

**File:** `/opt/sglang/python/sglang/srt/models/qwen3_vl.py:1193`

```python
def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
    # in qwen-vl, last dim is the same
    pixel_values = torch.cat([item.feature for item in items], dim=0).type(
        self.visual.dtype
    )
    image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
    assert pixel_values.dim() == 2, pixel_values.dim()
    assert image_grid_thw.dim() == 2, image_grid_thw.dim()

    if self.use_data_parallel:
        return run_dp_sharded_mrope_vision_model(
            self.visual,
            pixel_values,
            image_grid_thw.tolist(),
            rope_type="rope_3d",
        )
    else:
        return self.visual(pixel_values, grid_thw=image_grid_thw)  # 1210  ← THE ViT forward
```

Notes:

- We are NOT in the `use_data_parallel` branch — sglang's vision tower
  runs as a single forward pass on the bound XPU (`ZE_AFFINITY_MASK=N`).
- `self.visual` is the Qwen3-VL ViT (~600 M params).
- Multimodal attention backend on this host is `triton_attn`
  (the `WARN: Multimodal attention backend not set. Use triton_attn.`
  in the encoder log confirms this). There may be wins from a tuned
  XPU-native attention kernel.

For our 4-image 768×768 benchmark this produces a `(2304, 20480)`
tensor — 2304 vision tokens (4 × 576 patches each) × 20480 hidden
dim — and that single line accounts for **~2.2–2.4 s** of the
2.6 s `pure_encode_ms`.

## Step 5 — back to our patched handler (post-encode `.to(xpu)`)

**File:** `/usr/local/lib/python3.12/dist-packages/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py:339`

This is what the BENCH instrumentation actually wraps:

```python
_bt["before_encode"] = _bench_time.perf_counter()
with _nvtx.annotate("mm:enc:vision_encode", color="red"):
    if self._embedding_cache is not None:
        (
            image_grid_dim,
            precomputed_embeddings,
        ) = await self._encode_with_cache(image_urls)
    else:
        # All of Step 2..4 happens inside this single await call.
        # On return, precomputed_embeddings is on CPU because of
        # encode_server.py line 1079's mm_embedding.cpu().
        (
            image_grid_dim,
            precomputed_embeddings,
            _aux,
        ) = await mm_encode(self.encoder, image_urls, Modality.IMAGE)

        _bt["before_patch_to_xpu"] = _bench_time.perf_counter()
        if isinstance(precomputed_embeddings, torch.Tensor):
            _emb_device_pre = str(precomputed_embeddings.device)        # 'cpu'

        # PATCH (non-cached path): mirror the GPU-target logic from
        # _encode_with_cache. Without this, when ENABLE_ENCODER_CACHE=0
        # / cache disabled, mm_encode's CPU output goes straight to
        # NIXL on CPU, defeating GPUDirect RDMA.
        # BENCH gate: BENCH_DISABLE_XPU_PATCH=1 skips this step
        # for unpatched-baseline measurements (timing instrumentation
        # remains on so traces are comparable).
        import os as _bench_os
        _bench_skip_patch = (
            _bench_os.environ.get("BENCH_DISABLE_XPU_PATCH", "0") == "1"
        )
        from dynamo.common.multimodal.embedding_transfer import (
            _nixl_buffer_device,
        )
        _patch_target_device = _nixl_buffer_device()
        if (
            not _bench_skip_patch
            and isinstance(precomputed_embeddings, torch.Tensor)
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
        if isinstance(precomputed_embeddings, torch.Tensor):
            _emb_device_post = str(precomputed_embeddings.device)       # 'xpu:0'
_bt["after_encode"] = _bench_time.perf_counter()
```

So `pure_encode_ms = before_patch_to_xpu - before_encode` corresponds
exactly to lines 952-1079 of `encode_server.py` — image download,
preprocess, ViT forward, and the unconditional `.cpu()`.

## Where each ms is spent

| Code location | Stage we measured | Median (4-img) |
|---|---|---|
| `_flatten_and_load_images` (`encode_server.py:954`) | part of `pure_encode_ms` | ~50–200 ms |
| `self.image_processor(...)` (`encode_server.py:958`) | part of `pure_encode_ms` | ~100–300 ms |
| **`self.visual(pixel_values)` (`qwen3_vl.py:1210`)** | **most of** `pure_encode_ms` | **~2200–2400 ms** |
| `mm_embedding.cpu()` (`encode_server.py:1079`) | end of `pure_encode_ms` | ~10–40 ms (XPU sync) |
| `precomputed_embeddings.to(xpu)` (handler:368) | `patch_to_xpu_ms` | ~25 ms |
| `_nvtx.annotate("mm:enc:embedding_transfer")` (handler:458) | `nixl_setup_ms` | ~2 ms warm |

## Practical implications

1. **The encoder bottleneck is XPU compute on the ViT.** Image
   download + HF preprocess take a few hundred ms, but the dominant
   ~2 s is `self.visual(...)`. On a 32 GiB single-die Battlemage,
   there's no easy way to make that faster without a tuned attention
   kernel.
2. **Lines 1078-1079 of `encode_server.py` are wasteful for cross-host
   disagg.** sglang allocates the embedding on XPU (line 1078), then
   immediately moves it to CPU (line 1079), and then our patch moves
   it back to XPU. A clean upstream fix would skip the `.cpu()` when
   the caller wants a device tensor — that would save ~25 ms per
   request and remove the patch's net cost.
3. **Tuning the multimodal attention backend** is the highest-leverage
   knob. The encoder currently logs:
   ```
   common.print_info_once: Multimodal attention backend not set. Use triton_attn.
   common.print_info_once: Using triton_attn as multimodal attention backend.
   ```
   sglang has alternative MM attention backends; `triton_attn` is the
   safe default. If a Battlemage-tuned kernel exists, even a 20 %
   speed-up on this single line saves ~440 ms per 4-image request
   (roughly 10 % of total).
4. **NIXL is already a non-issue at the encoder level** (median
   `nixl_setup_ms ≈ 2 ms` warm, `nixl_wait_ms ≈ 0.5 ms`). Optimising
   it further is wasted effort until the ViT path is faster.
5. **Concurrency / sharding wins beat single-request optimisation
   wins.** As shown in §8 of `b70_encoder_time_breakdown.md`, going
   from 1E to 4E gave 1.5× throughput at conc=4 with no code
   changes. There is room to extend to 8E (XPUs 4-7) once those
   stale memory allocations are reclaimed.

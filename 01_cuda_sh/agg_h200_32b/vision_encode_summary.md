# SGLang vision encoder implementation: TP=1 vs TP=2

A walkthrough of how the Qwen3-VL vision encoder is implemented in SGLang, and what changes (or doesn't) between tensor-parallel configurations.

Source files:
- `/opt/sglang/python/sglang/srt/models/qwen3_vl.py`
- `/opt/sglang/python/sglang/srt/multimodal/mm_utils.py`

---

## 1. Architecture of Qwen3-VL vision encoder

```
Qwen3VLForConditionalGeneration (qwen3_vl.py:1070)
├── self.visual = Qwen3VLMoeVisionModel       ← the vision encoder
│   ├── Qwen3VLVisionPatchEmbed                ← Conv3d that turns pixels → patches
│   ├── pos_embed (VocabParallelEmbedding)     ← positional embeddings
│   ├── blocks: list of Qwen3_VisionBlock      ← N transformer blocks
│   │   ├── norm1
│   │   ├── attn = VisionAttention             ← MHSA over vision tokens
│   │   ├── norm2
│   │   └── mlp = Qwen3_VisionMLP              ← FFN
│   └── merger = Qwen3VLMoeVisionPatchMerger   ← projects ViT dim → LLM dim
└── self.model = Qwen3LLMModel                 ← the language model
```

The vision encoder takes **pixel patches** → outputs **embedding tensors** that are spliced into the LLM's input sequence at image-token positions.

---

## 2. The interesting code: each linear layer takes a `use_data_parallel` flag

`Qwen3_VisionMLP.__init__` (qwen3_vl.py:110-134):

```python
def __init__(self, ..., use_data_parallel: bool = False):
    self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
    self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
    self.linear_fc1 = ColumnParallelLinear(
        in_features, hidden_features,
        tp_size=self.tp_size,    # ← TP-shard FC1 by output dim
        tp_rank=self.tp_rank,
    )
    self.linear_fc2 = RowParallelLinear(
        hidden_features, in_features,
        tp_size=self.tp_size,    # ← TP-shard FC2 by input dim
        tp_rank=self.tp_rank,
    )
```

The same pattern appears in `Qwen3VLMoeVisionPatchMerger` (line 273).

`VisionAttention` (called from `Qwen3_VisionBlock`, line 195) has `use_qkv_parallel=True`, which means QKV projection and output projection are also TP-sharded.

The key controller flag at the top level (line 1097):
```python
self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
self.visual = Qwen3VLMoeVisionModel(
    vision_config,
    ...
    use_data_parallel=self.use_data_parallel,
)
```

---

## 3. What happens at TP=1 vs TP=2

### TP=1
- `tp_size = 1`, `tp_rank = 0`
- `ColumnParallelLinear` and `RowParallelLinear` with `tp_size=1` are basically **identity wrappers** around regular `nn.Linear` — no sharding, no all-reduce.
- The whole vision encoder runs as a single dense ViT on one GPU.
- For 8 images × 1080p, this is ~5-8 GFLOPs of work; ~1.0-1.5 s on one H200.

### TP=2 (default — `use_data_parallel=False`)
- `tp_size = 2`, `tp_rank = 0 or 1`
- Every `ColumnParallelLinear` shards weights across 2 ranks: rank 0 owns columns 0..N/2, rank 1 owns N/2..N. After matmul, each rank has half the output.
- Every `RowParallelLinear` shards weights across rows. After matmul, each rank has a *partial sum*; an **`all_reduce`** is needed to combine.
- `VisionAttention` with TP=2: each rank computes a different head subset, then output projection is a `RowParallelLinear` → all-reduce.
- **Per ViT block, there are typically 2 all-reduces** (one after attention, one after MLP).

So when running TP=2 *without* `mm_enable_dp_encoder`, the ViT becomes a TP-sharded model that sends a lot of all-reduce traffic per layer. On hardware with NVLink, this is fast. On hardware without NVLink (this box), each all-reduce crosses PCIe → expensive.

### TP=2 with `--mm-enable-dp-encoder`

Each ViT runs as TP=1 on each rank (so `tp_size=1`, no TP sharding inside ViT layers). Instead, **the input images are split**:
- Rank 0 processes images A, B, C, D
- Rank 1 processes images E, F, G, H
- Then both ranks `all_gather` to collect everyone's image embeddings.

Code at `qwen3_vl.py:1202`:
```python
if self.use_data_parallel:
    return run_dp_sharded_mrope_vision_model(self.visual, pixel_values, ...)
else:
    return self.visual(pixel_values, grid_thw=image_grid_thw)
```

`run_dp_sharded_mrope_vision_model` at `mm_utils.py:474` does:
1. Compute load-balancing assignment (which images go to which rank).
2. Gather local pixel values for this rank's assigned images.
3. Run the vision model on local images only.
4. Pad outputs to a common length.
5. **`tensor_model_parallel_all_gather`** to collect all rank's embeddings.

---

## 4. Side-by-side: three modes

| Mode | Per-layer ViT compute per rank | Per-layer comm | Total comm per request (8×1080p) |
|---|---|---|---|
| TP=1 (single GPU) | full ViT on 1 GPU | none | 0 |
| TP=2 default (TP-sharded ViT) | ½ of each layer | 2 all-reduce/layer × N layers | ~10 GB cross-rank |
| TP=2 + `mm-enable-dp-encoder` | full ViT on each rank, ½ images each | 1 all-gather at end of ViT | ~few hundred MB |

---

## 5. Why this matched the bench results

On this hardware (no NVLink, only PCIe):
- **TP=1 single agg, encoder time ≈ 1.18 s/req** (the `post-flush → next 8192` gap from earlier analysis).
- **TP=2 NODE default, encoder time ≈ 1.47 s/req** — *worse* than TP=1 because the all-reduce-per-layer cost dominates the small saving from halving each layer's compute.
- **TP=2 + `mm-enable-dp-encoder`, encoder time ≈ 1.23 s/req** — better than TP=2 default (one all-gather instead of N all-reduces), almost matches TP=1, but the `--enable-broadcast-mm-inputs-process` flag added at the same time introduced a bigger penalty elsewhere that overwhelmed the gain.

---

## 6. Code-level evidence: what each layer actually does

### `Qwen3_VisionBlock.forward` (qwen3_vl.py:219)

```python
def forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, ...):
    hidden_states = self.norm1(x)
    hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
    attn = self.attn(hidden_states, ...)            # ← TP-sharded, internal all-reduce
    attn = rearrange(attn, "b s ... -> s b ...")
    x += attn
    norm2 = self.norm2(x)
    mlp = self.mlp(norm2)                           # ← TP-sharded, internal all-reduce
    x += mlp
    return x
```

Two TP-sharded sub-modules per block; each does one all-reduce internally on TP>1.

### `Qwen3_VisionMLP.forward` (qwen3_vl.py:136)

```python
def forward(self, x):
    x_fc1, _ = self.linear_fc1(x)         # ColumnParallelLinear: split output dim
    mlp_output, _ = self.linear_fc2(self.act(x_fc1))  # RowParallelLinear: split input dim, all-reduce
    return mlp_output
```

`ColumnParallelLinear` then `RowParallelLinear` = classic Megatron-style 2-GPU MLP sharding. The all-reduce happens inside `RowParallelLinear`.

### `run_dp_sharded_mrope_vision_model` (mm_utils.py:474)

```python
def run_dp_sharded_mrope_vision_model(vision_model, pixel_values, grid_thw_list, ...):
    tp_size = get_attention_tp_size()
    if tp_size == 1:
        return vision_model(pixel_values, grid_thw=...)      # ← TP=1 fast path

    tp_rank_local = get_attention_tp_rank()
    # ... load-balancing: assign images to ranks ...
    image_idxs_local = ...
    pixel_values_local = ...                                  # ← only this rank's images

    # Run the vision model on the local pixel_values_local only
    image_embeds_local = vision_model(pixel_values_local, ...)

    # Pad to max length so all_gather works
    image_embeds_local_padded = ...

    # All-gather across all ranks to assemble final embeddings
    gathered_embeds = get_attention_tp_group().all_gather(
        image_embeds_local_padded, dim=0
    )
    # ... un-pad and return ...
```

**Note:** when `mm-enable-dp-encoder=True`, the ViT model itself is constructed with `use_data_parallel=True`, so internally each layer's `tp_size=1` (no all-reduce per layer). The only cross-rank communication is the one final `all_gather` of the ViT outputs. That's the fundamental trade-off vs default TP-sharded mode.

---

## 7. Mental model

For **vision encoders**:
- **TP shards each layer** → reduces per-layer compute, adds per-layer comm. Good when comm is cheap (NVLink).
- **DP shards the input batch** → keeps each layer monolithic, only one comm at the end. Good when comm is expensive (PCIe).

For **LLMs**:
- TP is typically necessary for memory (a 32B model doesn't fit on one GPU at high concurrency without sharding).
- DP for LLMs means running independent model copies (what we did with 2× TP=1 DP, the winning config).

So `mm-enable-dp-encoder` is essentially "use TP for the LLM but DP for the encoder" — a hybrid that's well-suited to multimodal workloads on weak-interconnect hardware.

---

## 8. The two key files for further reading

- **`/opt/sglang/python/sglang/srt/models/qwen3_vl.py`** — the actual ViT model definition; trace `Qwen3VLMoeVisionModel.forward` to see the per-block flow.
- **`/opt/sglang/python/sglang/srt/multimodal/mm_utils.py:474`** — `run_dp_sharded_mrope_vision_model`, the DP-sharded path.

---

## 9. How to verify this hardware has no NVLink

Three independent commands prove it:

### a) Topology matrix — should show `NV#` entries if NVLink exists
```bash
nvidia-smi topo -m
```
On this box: every cell is `NODE` or `SYS`, never `NV#`. Legend keys:
- `NV#` = bonded NVLinks (would show speed in GB/s)
- `PIX` = single PCIe bridge
- `NODE` = same NUMA, multiple PCIe host bridges
- `SYS` = cross-NUMA via QPI/UPI (slowest)

### b) Per-GPU NVLink status
```bash
nvidia-smi nvlink --status -i 4
nvidia-smi nvlink --status -i 5
```
On an NVLinked box this prints "Link 0: 26.562 GB/s" etc. On this box: empty output.

### c) GPU model name
```bash
nvidia-smi --query-gpu=name --format=csv
```
On this box: `NVIDIA H200 NVL`. The "NVL" suffix indicates PCIe form factor (vs "SXM" which has the full NVSwitch fabric).

### Bandwidth comparison

| Path | Practical bandwidth |
|---|---:|
| H100/H200 SXM NVLink (full mesh) | ~900 GB/s aggregate |
| H100/H200 NVL 2-card NVLink bridge | ~900 GB/s between paired pair only |
| **PCIe Gen5 x16** (this box, `NODE`) | **~50 GB/s practical** |
| **PCIe Gen5 x16 + UPI** (this box, `SYS`) | **~30-40 GB/s practical** |

Inter-GPU bandwidth on this box is **15-30× slower** than on an NVLinked H200/H100 SXM system. That's why TP-style sharding loses on this hardware.

---

## 10. Summary

- The vision encoder in SGLang is built from layers that **each accept** a `use_data_parallel` flag. Setting it sets the internal `tp_size=1` per layer.
- With **TP>1 default** (no `mm-enable-dp-encoder`), the ViT is TP-sharded **per layer** — N layers × ~2 all-reduces = expensive on PCIe.
- With **TP>1 + `mm-enable-dp-encoder`**, the ViT runs unsharded on each rank but **images are batch-split**, with one final `all_gather` — much cheaper on PCIe.
- For our workload on this NVLink-less hardware, TP-sharded ViT is the wrong choice; DP-sharded ViT recovers most of the loss; and **simply running independent TP=1 workers** (no inter-GPU traffic at all) is the actual win.

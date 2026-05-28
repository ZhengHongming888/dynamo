# Applying sglang PR #26460 (xpu_attn vision backend) to local sglang

**PR:** https://github.com/sgl-project/sglang/pull/26460
**Title:** [Intel GPU][Encoder] Add xpu_attn backend for encoder vision attention
**Author:** jianan-gu (Intel)
**Upstream commit:** `186c058d2906a6cc5db320af208ebdb1cc123f87`
**Upstream parent:**  `21d0e74aff8877abdb513e013ba4a5364e53d13a`
**Date applied:** 2026-05-28

This document records the gap between local sglang and the upstream PR, the manual
port required because `git apply` failed, and the verification steps performed.
The combined local patch is also saved alongside this file as `pr26460_local.patch`.

## What the PR does

Adds an `xpu_attn` vision attention backend that dispatches to
`sgl_kernel.flash_attn.flash_attn_varlen_func` when running on Intel XPU and
`use_intel_xpu_backend()` returns true. Without this PR, multimodal encoders on
XPU fall through to `triton_attn`, which is ~10–30× slower than a tuned
attention kernel.

Mechanics:

1. New `VisionIntelXPUAttention` class wrapping `flash_attn_varlen_func`
2. Registered in the `QKV_BACKEND_IMPL` dispatch dict as `"xpu_attn"`
3. `_determine_attention_backend` picks `xpu_attn` over `triton_attn` on XPU when
   `use_intel_xpu_backend()` is on
4. CLI flag `--mm-attention-backend` accepts `xpu_attn`

## Gap analysis: local sglang vs upstream PR

**Local sglang:** `/opt/sglang`, `git rev-parse HEAD = 4a04a9818e335a4496f6ab8cade377fdc6997fd0`
("【hicache】Optimize HiCache prefetch logic …" #16370)

**Upstream PR parent:** `21d0e74aff8877abdb513e013ba4a5364e53d13a`

### Overall gap

- Local is **~1,125 commits behind** the PR's parent on `main`.
- `git apply --check` and `git apply --3way --check` both **failed**
  (`patch does not apply` / `does not match index`). Manual port was required.

### Per-file gap

#### `python/sglang/srt/layers/attention/vision.py`

- 4 commits between local and PR parent touch this file:
  - `87c3171aa [CPU] Add support for Qwen3-vl and Qwen3-omni (#12662)`
  - `d8e66e54e fix: use triton_attn as default vision attention on B300 (SM103) (#25570)`
  - `71e89e900 [MUSA][19/N] Support qwen series models (#23654)`
  - `651af06a0 [Feature] Xiaomi MiMo-V2.5 day0 support (#23811)`
- Diff: **+148 / −30 lines** between local and PR parent
- Notable structural drift that broke `git apply`:
  - `VisionAMXAttention` class **exists upstream but not locally**. The PR's diff
    context (`@@ -800`) anchors on this; local context is at `@@ -722` and lacks
    the `VisionAMXAttention` block.
  - `"amx_attn": VisionAMXAttention` entry in `QKV_BACKEND_IMPL` (upstream)
  - New `"amx_attn"` branch in `_determine_attention_backend` (upstream ~L1013)

#### `python/sglang/srt/server_args.py`

- **93 commits** between local and PR parent touch this file
- Diff: **+1,170 / −754 lines** (heavy churn)
- Local `--mm-attention-backend` choices block sits at L5170 vs upstream's L5398

### What PR 26460 itself adds (the change to apply)

| # | Where (upstream line) | Local line (post-port) | Change |
|---|---|---|---|
| 1 | `vision.py:31` | `vision.py:27` | Add `use_intel_xpu_backend` to existing `from sglang.srt.utils import (...)` import block |
| 2 | `vision.py:60` | `vision.py:47` | Add `if _is_xpu: from sgl_kernel.flash_attn import flash_attn_varlen_func` next to the `_is_npu` block |
| 3 | `vision.py:797` | `vision.py:725` | Insert `VisionIntelXPUAttention` class (52 lines) immediately after `VisionAiterAttention.forward()` returns |
| 4 | `vision.py:858` | `vision.py:782` | Add `"xpu_attn": VisionIntelXPUAttention,` entry in `QKV_BACKEND_IMPL` dict |
| 5 | `vision.py:1089` | `vision.py:983` | Replace `backend = "triton_attn"` with `backend = "triton_attn" if not use_intel_xpu_backend() else "xpu_attn"` in the `elif _is_xpu:` branch of `_determine_attention_backend` |
| 6 | `server_args.py:5401` | `server_args.py:5181` | Add `"xpu_attn"` to `--mm-attention-backend` `choices` list |

Local prerequisites already satisfied (no extra work needed):

- `use_intel_xpu_backend` exists at `/opt/sglang/python/sglang/srt/utils/common.py:304`
  and is re-exported via `sglang.srt.utils`
- `_is_xpu`, `is_xpu`, `print_info_once` already imported in vision.py
- `SingletonCache` and `resolve_seqlens` (used by the new class) already defined
  at `vision.py:96` and `vision.py:126` respectively
- `get_attention_tp_size` already imported

## Manual port — exact edits

Source line numbers below refer to the **local file before any edits**. After
each edit the line numbers shift; the file content shown is the surrounding
context kept stable across each edit.

### Backups

```bash
cp /opt/sglang/python/sglang/srt/layers/attention/vision.py \
   /opt/sglang/python/sglang/srt/layers/attention/vision.py.bak.pre-pr26460
cp /opt/sglang/python/sglang/srt/server_args.py \
   /opt/sglang/python/sglang/srt/server_args.py.bak.pre-pr26460
```

### Edit 1 — `vision.py` imports (around L18–27)

**Before:**
```python
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_blackwell_supported,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
    print_info_once,
)
```

**After:**
```python
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_blackwell_supported,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
    print_info_once,
    use_intel_xpu_backend,
)
```

### Edit 2 — `vision.py` `_is_xpu` import block (around L45–46)

**Before:**
```python
if _is_npu:
    import torch_npu
```

**After:**
```python
if _is_npu:
    import torch_npu
if _is_xpu:
    from sgl_kernel.flash_attn import flash_attn_varlen_func
```

### Edit 3 — `vision.py` add `VisionIntelXPUAttention` class (around L722, right after `VisionAiterAttention` returns)

**Before:**
```python
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=output,
        )
        return output
```

**After:**
```python
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=output,
        )
        return output


class VisionIntelXPUAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not (_is_xpu):
            raise Exception("VisionIntelXPUAttention is only available for Intel XPU")
        super().__init__()
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        window_size = kwargs.get("window_size", (-1, -1))
        s_aux = kwargs.get("s_aux", None)

        cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        fa_kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
        if s_aux is not None:
            fa_kwargs["sinks"] = s_aux
        output = flash_attn_varlen_func(q, k, v, **fa_kwargs)

        return output
```

### Edit 4 — `vision.py` register in `QKV_BACKEND_IMPL` (around L730)

**Before:**
```python
QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
    "fa4": VisionFlash4Attention,
    "flashinfer_cudnn": VisionFlashInferAttention,
    "ascend_attn": VisionAscendAttention,
    "aiter_attn": VisionAiterAttention,
}
```

**After:**
```python
QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
    "fa4": VisionFlash4Attention,
    "flashinfer_cudnn": VisionFlashInferAttention,
    "ascend_attn": VisionAscendAttention,
    "aiter_attn": VisionAiterAttention,
    "xpu_attn": VisionIntelXPUAttention,
}
```

### Edit 5 — `vision.py` `_determine_attention_backend` `_is_xpu` branch (around L927–928)

**Before:**
```python
        elif _is_xpu:
            backend = "triton_attn"
```

**After:**
```python
        elif _is_xpu:
            backend = "triton_attn" if not use_intel_xpu_backend() else "xpu_attn"
```

### Edit 6 — `server_args.py` CLI choices (around L5173–5181)

**Before:**
```python
            choices=[
                "sdpa",
                "fa3",
                "fa4",
                "triton_attn",
                "ascend_attn",
                "aiter_attn",
                "flashinfer_cudnn",
            ],
```

**After:**
```python
            choices=[
                "sdpa",
                "fa3",
                "fa4",
                "triton_attn",
                "ascend_attn",
                "aiter_attn",
                "flashinfer_cudnn",
                "xpu_attn",
            ],
```

## Combined patch artifact

A single combined diff capturing all six edits relative to the local pre-port
files is saved at:

```
/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_local.patch
```

To re-apply this on a fresh local sglang at the same baseline commit
(`4a04a9818`):

```bash
cd /opt/sglang
git apply --check /hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_local.patch
git apply       /hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_local.patch
```

To revert:

```bash
cp /opt/sglang/python/sglang/srt/layers/attention/vision.py.bak.pre-pr26460 \
   /opt/sglang/python/sglang/srt/layers/attention/vision.py
cp /opt/sglang/python/sglang/srt/server_args.py.bak.pre-pr26460 \
   /opt/sglang/python/sglang/srt/server_args.py
```

## Verification

```bash
$ python3 -m py_compile /opt/sglang/python/sglang/srt/layers/attention/vision.py
$ python3 -m py_compile /opt/sglang/python/sglang/srt/server_args.py
# both: clean
```

```bash
$ python3 -c "
import sglang.srt.layers.attention.vision as v
print('VisionIntelXPUAttention class:', v.VisionIntelXPUAttention)
print('xpu_attn in QKV_BACKEND_IMPL:', 'xpu_attn' in v.QKV_BACKEND_IMPL)
print('mapped to:', v.QKV_BACKEND_IMPL.get('xpu_attn'))
print('use_intel_xpu_backend imported:', hasattr(v, 'use_intel_xpu_backend'))
"
VisionIntelXPUAttention class: <class 'sglang.srt.layers.attention.vision.VisionIntelXPUAttention'>
xpu_attn in QKV_BACKEND_IMPL: True
mapped to: <class 'sglang.srt.layers.attention.vision.VisionIntelXPUAttention'>
use_intel_xpu_backend imported: True
```

```bash
$ python3 -c "
from sglang.srt.server_args import ServerArgs
import argparse
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
for action in p._actions:
    if any('mm-attention-backend' in s for s in action.option_strings):
        print('mm-attention-backend choices:', action.choices)
        break
"
mm-attention-backend choices: ['sdpa', 'fa3', 'fa4', 'triton_attn', 'ascend_attn', 'aiter_attn', 'flashinfer_cudnn', 'xpu_attn']
```

All four checks pass: both files compile, the class is importable and registered,
and the CLI advertises `xpu_attn` as a valid choice.

## Usage

On an Intel XPU host:

```bash
# Explicit selection
python3 -m sglang.launch_server ... --mm-attention-backend xpu_attn

# Or rely on _determine_attention_backend auto-detection by toggling
# whatever env var use_intel_xpu_backend() reads
# (defined at /opt/sglang/python/sglang/srt/utils/common.py:304).
```

## Caveats

- The PR was flagged by gemini-code-assist for two issues that were not
  addressed before this port:
  1. `flash_attn_varlen_func` from `sgl_kernel.flash_attn` may not accept
     `sinks` or `window_size` kwargs — runtime errors are possible if a model
     passes those.
  2. The `cu_seqlens.to(dtype=torch.int32).to(q.device)` is two passes that
     could be one.
  Both are upstream concerns and not addressed by this local port.

- This port is against local commit `4a04a9818`. If local sglang is rebased
  forward (e.g. once `VisionAMXAttention` lands), Edit 4 (`QKV_BACKEND_IMPL`)
  may need to be re-anchored.

- The H200 PD investigation in the parent directory does not exercise the
  `xpu_attn` path — the encoder side runs on B70 Battlemage XPU but the
  installation there is `/usr/local/lib/python3.12/dist-packages/dynamo/...`
  on the B70 host, which is a separate sglang build. To activate this PR for
  the cross-host disagg encoder, the same edits must be made on the B70 host's
  sglang install.

## Files

- This document: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_apply.md`
- Combined local patch: `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/pr26460_local.patch`
- Backups:
  - `/opt/sglang/python/sglang/srt/layers/attention/vision.py.bak.pre-pr26460`
  - `/opt/sglang/python/sglang/srt/server_args.py.bak.pre-pr26460`
- Companion docs in this directory:
  - `b70_patched.md`, `b70_xpu_nixl.patch` (the B70-side dynamo patch chain
    that the bench would also need for end-to-end XPU acceleration)
  - `b70_encoder_time_breakdown.md` (showing that `triton_attn` is the
    encoder-side bottleneck on B70 — exactly what this PR is meant to address)

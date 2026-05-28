# vLLM K8s YAML vs our sglang scripts — diff & analysis

**Source:** `/hongming/pallavi_yaml.txt` (Pallavi's vLLM K8s test deployment)
**Comparison target:** our cross-host PD script `/hongming/dynamo/01_cuda_sh/disagg_h200_32b/start_sglang_pd_cuda_32b_fp8_giga01.sh` and the B70 encoder side
**Code referenced:** `/opt/venv/lib/python3.12/site-packages/dynamo/{common,vllm,sglang}/...`

## Topology comparison

| | vLLM YAML | Our sglang setup |
|---|---|---|
| Frontend node | giga01 (sc09super21-h200) | giga01 |
| **Encoder node** | **B60** (sc09rvp03-b60, **Intel XPU**) | **B70** (NVIDIA H200) |
| Encoder count | 4 replicas | 1 (later 4) |
| PD/Decode node | giga01 (NVIDIA H200) | giga01 (NVIDIA H200) |
| Backend | vLLM | sglang |
| Container image | `dynamo_xpu:275-d1ab4c5` (encoder) / `dynamo_gpu:277-d1ab4c5` (PD) | bare `/opt/venv` install on host |
| Control-plane KV store | `DYN_STORE_KV=mem` (in-memory) | etcd at port 12379 |

**Critical**: The vLLM YAML's encoder runs on Intel B60 XPUs, NOT NVIDIA. So this is a hetero-GPU disagg test (XPU encoder → H200 PD), not the same as our H200 → H200 setup.

## Key env-var differences

### Our sglang has these — YAML doesn't

| Env var | Our value | Why |
|---|---|---|
| `DYN_TCP_MAX_MESSAGE_SIZE` | 268435456 (256MB) | Needed for big multimodal payloads through dynamo TCP plane |
| `DYN_HTTP_BODY_LIMIT_MB` | 256 | Frontend HTTP body limit |
| `DYN_SGL_EMBEDDING_TRANSFER_MODE` | nixl-read | sglang-specific (vLLM uses `DYN_VLLM_*`) |
| `UCX_NET_DEVICES` | mlx5_4:1 | Pin to specific NIC |

The YAML doesn't need `DYN_TCP_MAX_MESSAGE_SIZE` or `DYN_HTTP_BODY_LIMIT_MB` — likely because dynamo K8s deployments default to higher limits, or because vLLM's pipeline doesn't push as much through the dynamo TCP plane.

### YAML has these — our sglang doesn't

| Env var | YAML value | What it does | Applies to sglang? |
|---|---|---|---|
| `DYN_STORE_KV` | mem | Use in-memory KV store instead of etcd | yes (could try) |
| **`NIXL_USE_CPU_HOST_MEMORY`** | **0** | Force GPU descriptors (only on encoder pod) | **NO — see analysis** |
| **`VISION_ENCODE_SERIALIZE`** | **1** | Serialize vision encoding within encoder pod | **NO — vLLM-only flag** |
| `UCX_IB_ROCE_REACHABILITY_MODE` | all | UCX considers all RoCE-reachable peers | yes (UCX-level) |
| `DYN_VLLM_KV_EVENT_PORT` | 20080 | (vLLM equivalent of our 22081) | n/a |

### Both set, with same/similar values

| Env var | Both | Note |
|---|---|---|
| `DYN_REQUEST_PLANE` | tcp | match |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | 20098 | match |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | pod IP / RoCE IP | match (different mechanism — K8s downward API vs hardcoded) |
| `UCX_TLS` | `ib,rc,ud,rc_verbs,ud_verbs,cuda_copy` (PD) / `ze_copy` for XPU | YAML matches our PD; XPU encoder uses `ze_copy` instead of `cuda_copy` |
| `UCX_MEMTYPE_CACHE` | 0 | match |
| `ENABLE_ENCODER_CACHE` | 0 | match |
| `PYTHONHASHSEED` | 0 | match |
| `DYN_*EMBEDDING_TRANSFER_MODE` | nixl-read | match (different env name per backend) |

## Worker launch arg differences

### Encoder: vLLM YAML

```bash
python -m dynamo.vllm \
  --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --enable-multimodal --multimodal-encode-worker --enable-mm-embeds \
  --dtype bfloat16 \
  --enforce-eager \
  --block-size 64 \
  --gpu-memory-utilization 0.95 \
  --kv-events-config '{...}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"xpu","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
```

### Encoder: our sglang (from `start_disagg_h200_32b_combined.sh`, applicable to B70 side)

```bash
python3 -m dynamo.sglang \
  --model /mnt/.../Qwen3-VL-32B-Instruct-FP8 \
  --enable-multimodal --multimodal-encode-worker \
  --multimodal-embedding-cache-capacity-gb 16 \
  --chat-template qwen2-vl \
  --dtype auto --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --page-size 16 \
  --enable-request-time-stats-logging --show-time-cost \
  --kv-events-config '{...}'
```

Differences worth noting:
- **vLLM has `--enable-mm-embeds`** flag to opt in to embedding transfer pipeline; sglang doesn't have this (its multimodal flow always supports it via `embedding-transfer-mode`)
- **vLLM has `--kv-transfer-config`** (NixlConnector); sglang uses `DYN_SGL_EMBEDDING_TRANSFER_MODE` env var instead. Different mechanisms internally but same intent.
- **vLLM uses `--enforce-eager`** on encoder (skip CUDA graph capture, makes sense for variable-shape ViT)
- **vLLM uses `--block-size 64`** vs sglang's `--page-size 16` (KV cache page size; affects memory granularity)
- **vLLM uses `--dtype bfloat16`** vs sglang's `--dtype auto`. For Qwen3-VL-FP8 model, `auto` will pick bf16 too, so equivalent.

### PD/Decode: vLLM YAML

```bash
python -m dynamo.vllm --model Qwen/Qwen3-VL-32B-Instruct-FP8 --enable-multimodal \
  --multimodal-worker --enable-mm-embeds --route-to-encoder \
  --mm-prompt-template "<|im_start|>system\n..." \
  --dtype bfloat16 --max-num-seqs 40 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --dyn-tool-call-parser hermes \
  --kv-events-config '{...}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
```

### PD/Decode: our sglang (giga01 PD)

```bash
python3 -m dynamo.sglang --model /mnt/.../Qwen3-VL-32B-Instruct-FP8 \
  --enable-multimodal --enable-mm-global-cache --multimodal-worker \
  --dtype auto --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 64 --tensor-parallel-size 1 \
  --mem-fraction-static 0.92 --page-size 16 --chunked-prefill-size 16384 \
  --enable-request-time-stats-logging --show-time-cost \
  --kv-events-config '{...}'
```

Differences:
- **vLLM has `--route-to-encoder`** — explicitly opts into routing MM data to a separate encoder
- **vLLM has `--mm-prompt-template`** — passes the chat template manually (sglang reads it from the model dir)
- **vLLM uses `--max-num-seqs 40`** vs our **`--max-running-requests 64`** — sglang has higher concurrent request cap
- **vLLM uses 0.95 mem util** vs our **0.92**
- **vLLM has `--dyn-tool-call-parser hermes`** (tool calling support, not relevant for vision benchmarks)
- **Our sglang has `--chunked-prefill-size 16384` and `--enable-mm-global-cache`** — sglang-specific tuning we kept

## The `NIXL_USE_CPU_HOST_MEMORY` finding (most important)

**This env var is set to `0` only on the vLLM YAML's encoder pod. Setting it on our sglang setup would NOT have the same effect**, because:

### How it actually works in dynamo code

**In `/opt/venv/lib/python3.12/site-packages/dynamo/common/multimodal/embedding_transfer.py`:**

Lines 28-45 define `_nixl_buffer_device()` which returns CUDA/XPU/CPU based on the env var:

```python
NIXL_USE_CPU_HOST_MEMORY = bool(int(os.getenv("NIXL_USE_CPU_HOST_MEMORY", 0)))

def _nixl_buffer_device() -> torch.device:
    if NIXL_USE_CPU_HOST_MEMORY:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
```

**But this helper is NEVER CALLED in the receive path.** Lines 882 and 915 allocate
NIXL receive buffers without device kwarg, defaulting to CPU:

```python
# Line 882 — warmed-up descriptor pool init
encodings_tensor = torch.zeros(
    max_item_mm_token * embedding_hidden_size, dtype=torch.int8
)  # ← no device=, defaults to CPU

# Line 915 — fallback when warmed pool is empty
encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)
# ← no device=, defaults to CPU
```

So the **PD-side receive buffer is always CPU**, regardless of `NIXL_USE_CPU_HOST_MEMORY`.

### vLLM encoder handler honors the env var; sglang's hardcodes CPU

**vLLM's encoder handler** at `dynamo/vllm/multimodal_handlers/encode_worker_handler.py:330`:

```python
emb_tensor = split_tensor.cpu() if _needs_cpu else split_tensor
# ↑ honors NIXL_USE_CPU_HOST_MEMORY env var
```

**sglang's encoder handler** at `dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py:218-230`:

```python
# SGLang's _encode outputs are already on CPU; use CPU as target for consistency
target_device = torch.device("cpu")  # ← hardcoded CPU!
...
if new_embeddings.device != target_device:
    logger.warning(...)
    new_embeddings = new_embeddings.to(target_device)  # ← forces to CPU
```

The comment says "SGLang's `_encode` outputs are already on CPU" — meaning the sglang ViT
encoder itself stages embeddings to CPU memory before returning them. The dynamo handler
then defensively re-pins to CPU. **No env var bypasses this.**

### Combined result for sglang disagg

The sglang multimodal disagg path is **structurally CPU-bound, twice over**:

1. **Encoder side**: SGLang's `_encode` returns tensors on CPU (and dynamo handler re-pins to CPU)
2. **Receive side**: NIXL receive descriptor allocated on CPU at line 915 (and warmed pool at 882)

Wire RDMA goes CPU→CPU regardless of `NIXL_USE_CPU_HOST_MEMORY` value. The only way to fix
this for sglang is the structural patch from SESSION_MEMORY round-4 — i.e., modify
`embedding_transfer.py:915` (and 882) to use `_nixl_buffer_device()`, AND modify
`encode_worker_handler.py:218-230` to NOT force CPU. Both code edits required.

For vLLM, only setting `NIXL_USE_CPU_HOST_MEMORY=0` on the encoder side fixes half the problem;
the receive side at 915/882 is still CPU-bound. So even vLLM's setup doesn't fully exploit
GPUDirect RDMA — the wire bytes still flow GPU→CPU on encoder, RDMA, CPU on receive, then
CPU→GPU. Better than sglang's path because at least the encoder doesn't do an extra `.cpu()`,
but not optimal.

## Other minor findings

1. **`VISION_ENCODE_SERIALIZE=1`** — defined at `dynamo/vllm/multimodal_handlers/encode_worker_handler.py:52`. Wraps the encoder's vision-encode call in an asyncio lock so requests serialize within a pod. Default is also 1, so the YAML is just being explicit. **Not in sglang's encode_worker_handler** — sglang has different concurrency logic.

2. **`DYN_STORE_KV=mem`** — uses an in-process KV store instead of etcd. Could simplify our setup, but we already have a working etcd-based stack and shouldn't change it mid-investigation.

3. **`UCX_IB_ROCE_REACHABILITY_MODE=all`** — relaxes UCX's IB reachability check. Not strictly needed for our setup (we have direct routes), but harmless. Could add for robustness.

4. **`enforce_handshake_compat: false`** in `kv_connector_extra_config` — bypasses NIXL handshake compat check. Relevant when crossing different NIXL versions or GPU vendors (their XPU↔CUDA case). Not needed for our homogeneous CUDA setup.

5. **`--enforce-eager`** on encoder side — vLLM-specific flag to skip CUDA graph capture. Sglang's encoder doesn't use CUDA graphs in the same way, so no equivalent needed.

## Actionable conclusions

1. **The `NIXL_USE_CPU_HOST_MEMORY=0` env var doesn't help our sglang setup.** Sglang's encoder hardcodes CPU output (line 219), and the receive descriptor allocation also hardcodes CPU (line 915). Setting the env var has no effect on either path.

2. **The fix for sglang must be code-level**, matching SESSION_MEMORY's round-4 patch direction:
   - `embedding_transfer.py:882, 915` — add `device=_nixl_buffer_device()` to `torch.zeros(...)` calls
   - `encode_worker_handler.py:219` — drop the hardcoded `target_device = torch.device("cpu")` and let SGLang's output device flow through

3. **Even vLLM's setup has the same receive-side bug** (line 915 / 882). They might be hitting it less because their encoder cache (line 337-344) means many requests hit the cache and skip the dynamic descriptor path. Worth verifying.

4. **The vLLM K8s test isn't directly comparable to our cross-host benchmarks** because:
   - Different backend (vLLM vs sglang)
   - Different encoder hardware (Intel XPU B60 vs NVIDIA H200 B70)
   - Different memory utilization (0.95 vs 0.85/0.92)
   - Different multimodal pipeline (`--enable-mm-embeds` + `--route-to-encoder` vs sglang's mode)

5. **Things from the YAML we could/should adopt for our sglang scripts:**
   - `UCX_IB_ROCE_REACHABILITY_MODE=all` — defensive, no downside
   - `--enforce-eager` for encoder side (would skip our DeepGEMM warmup overhead on encoder)
   - Try `DYN_STORE_KV=mem` if we want simpler control plane (but only after current bench cycle)

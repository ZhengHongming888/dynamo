# Current Embedding-Transfer Flow (post-patch, dell06 ↔ giga01)

This is the actual code path for one multimodal request flowing **frontend → encoder → PD**, after both encoder-side (B70 `b70_xpu_nixl.patch` / dell06's matching version) and PD-side (`h200_cuda_nixl.patch`) patches are applied.

## Pictorial overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  FRONTEND  (giga01:7001)                                                 │
│  POST /v1/chat/completions  →  round-robin to encoder via dynamo TCP     │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  ENCODE WORKER  (dell06 H200, encode_worker_handler.py)                  │
│                                                                          │
│  1. parse request, fetch images, build MultimodalGroup            (~0 s) │
│  2. mm_encode(...) → vision tower forward, returns                       │
│       (image_grid_dim, precomputed_embeddings: torch.Tensor)             │
│       precomputed_embeddings is on cuda:0  (encoder patch)        (~1 s) │
│  3. token-expand: replace each image-token with N tokens          (~ms)  │
│                                                                          │
│  4. NixlReadEmbeddingSender.send_embeddings(precomputed_embeddings):     │
│       4a. transfer_buf = embeddings.clone().detach()    [GPU memcpy]     │
│           (because stage_embeddings=False default;                       │
│            bypassed only if stage_embeddings=True)                       │
│       4b. torch.cuda.synchronize()                       [global stream] │
│       4c. descriptor = nixl_connect.Descriptor(transfer_buf)             │
│           → ptr=<gpu addr>, size=<bytes>, device=cuda:0                  │
│       4d. readable_op = await connector.create_readable(descriptor)      │
│           → registers GPU memory with NIXL + UCX                         │
│           → builds RdmaMetadata = (ptr, size, agent_name, device)        │
│       returns (TransferRequest, transfer_future)                         │
│                                                                          │
│  5. request.transfer_payload = TransferRequest(                          │
│         embeddings_shape=[N_tokens, hidden],                             │
│         embedding_dtype_str="torch.bfloat16",                            │
│         serialized_request=RdmaMetadata.model_dump()  ← ptr, size, dev   │
│     )                                                                    │
│  6. response_generator = pd_worker_client.round_robin(request)           │
│     → dynamo TCP plane sends request to PD; metadata is small (~1 KB)    │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  ↓ TCP control plane (~ms)
┌──────────────────────────────────────────────────────────────────────────┐
│  PD WORKER  (giga01 H200, worker_handler.py)                             │
│                                                                          │
│  7. EmbeddingsProcessor.process_embeddings(request):                     │
│       7a. tensor_id, embeddings =                                        │
│             await embedding_receiver.receive_embeddings(transfer_req)    │
│                                                                          │
│  NixlReadEmbeddingReceiver.receive_embeddings(transfer_req):             │
│       7b. shape = transfer_req.embeddings_shape                          │
│           dtype = bf16                                                   │
│           readable_metadata = parse(serialized_request)                  │
│       7c. # PATCHED PATH (line 919):                                     │
│           encodings_tensor = torch.zeros(                                │
│               *shape, dtype=bf16, device=_nixl_buffer_device())          │
│               ← LANDS ON cuda:0  (was cpu pre-patch)                     │
│       7d. descriptor = nixl_connect.Descriptor(encodings_tensor)         │
│           → ptr=<gpu addr>, size, device=cuda:0                          │
│                                                                          │
│       7e. read_op = await connector.begin_read(                          │
│                       readable_metadata, descriptor)                     │
│           → ReadOperation:                                               │
│             local_descriptors  = device=cuda:0  ← us (PD)                │
│             remote_descriptors = device=cuda:0  ← encoder                │
│           → NIXL submits RDMA READ via UCX over RoCE NIC mlx5_4          │
│             RoCE wire pulls bytes  encoder GPU → PD GPU                  │
│             (GPUDirect RDMA: NIC reads directly from sender's HBM,       │
│              writes directly to receiver's HBM)                          │
│       7f. await read_op.wait_for_completion()                            │
│           ← ~11 ms for 64 MB embedding (8img/1080p)                      │
│       returns (tensor_id, encodings_tensor)  ← tensor lives on cuda:0    │
│                                                                          │
│  8. mm_items = [create_multimodal_item(embeddings, grid_thw)]            │
│       ← packs into SGLang's "precomputed_embeddings" mm_item format      │
│                                                                          │
│  9. results = await engine.async_generate(                               │
│         input_ids=request.token_ids,                                     │
│         image_data=mm_items,         ← NO get_image_feature() call,      │
│         sampling_params=...           SGLang sees pre-encoded embeddings │
│     )                                                                    │
│       → chunked-prefill of 16k visual tokens through 35B decoder         │
│       → autoregressive decode of N output tokens                         │
│       → streams generated tokens back via async iterator                 │
│                                                                          │
│  10. release_tensor(tensor_id):                                          │
│        — if dynamic descriptor: NIXL deregister + free GPU mem           │
│        — if from warmedup pool: return descriptor to pool                │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  ↓ token stream over dynamo TCP
                                  ↓ (back through encoder for streaming)
                                  ↓
                              FRONTEND → SSE → client
```

## What lives where (post-patch)

| Component | Code location | Device |
|---|---|---|
| Encoder vision tower output (`precomputed_embeddings`) | `encode_worker_handler.py:347` | encoder GPU (cuda:0 on dell06, xpu:0 on B70) |
| Sender's transfer_buf | `embedding_transfer.py:824/826` | **cuda:0** (no `.cpu()` call anywhere) |
| Sender's NIXL descriptor (encoder side) | `embedding_transfer.py:835` | **cuda:0** |
| Receiver's NIXL receive buffer (PD side) | **`embedding_transfer.py:921-925` (patched)** | **cuda:0** |
| RDMA wire path | UCX over mlx5_4 (RoCE) | NIC → NIC |
| PD's view of received tensor | returned from `receive_embeddings()` | **cuda:0** (no copy needed) |
| `async_generate` call | `worker_handler.py:430/628` | passes the cuda:0 tensor directly |

## Per-request data movement (8img/1080p, ~64 MB embedding)

```
Encoder GPU (dell06 cuda:0):
  embeddings.clone()                        ← 64 MB device-internal copy
  cuda.synchronize()                        ← stream barrier
  nixl_connect.Descriptor(transfer_buf)     ← register-with-NIXL
  create_readable                           ← ucp_mem_map + advertise to peer
                                              ← NIC's GPUDirect path armed

  ↓  RoCE NIXL READ (one shot)              [11 ms wire]

PD GPU (giga01 cuda:0):
  torch.zeros(..., device='cuda:0')         ← allocate 64 MB on PD GPU
  Descriptor(encodings_tensor)              ← register dest with NIXL
  begin_read → wait_for_completion          ← UCX submits RDMA READ;
                                              data lands directly in HBM
  → 64 MB now in PD's cuda:0
  → mm_item['precomputed_features'] = embeddings.to(bf16)   ← view, no copy
  → engine.async_generate() ingests directly
```

**No CPU staging on either side.** Pre-patch the line 921 allocation defaulted to CPU memory, which forced the NIC to write to host memory then a separate CPU→GPU `cudaMemcpy` before `async_generate` could use the embedding. The patch removes that staging step.

## What still defaults to CPU (i.e. potential next optimisations)

1. **Sender-side `clone()` at line 826** — when `stage_embeddings=False` (default for SGLang backend), the sender does a 64 MB device-to-device clone. The vLLM backend uses `stage_embeddings=True` to skip this. SGLang would need a lifecycle queue to do the same safely (per `patches_for_one_request_handoff.md` patches 1a-1c, with caveat that 1a alone was unsafe).

2. **Sender-side `cuda.synchronize()` at line 833** — global device barrier rather than stream-scoped. `patches_for_one_request_handoff.md` round 1c proposed `torch.cuda.current_stream(_dev_type).synchronize()` instead. Probably negligible at current concurrency.

3. **`max_items=0` factory** at `dynamo/common/multimodal/__init__.py:40` — disables the warmedup descriptor pool, so every receive uses the dynamic-allocation path (line 919). The warmedup-pool patch at line 882 is therefore dead code currently. To activate it, the factory needs to be changed AND the variable-shape descriptor problem needs to be solved (the `[gluo FIXME]` comment).

4. **No coordinated `mem_fraction_static` accounting** — patched dynamic GPU descriptors are charged against the same pool as KV cache + Mamba state. At 40 in-flight × 64 MB = 2.5 GB just from active descriptors, plus deregistration latency. The 32B-FP8 docs document OOMs at higher concurrency (see `patches_for_one_request_handoff.md` round 4). Currently fine for 35B at np=32; might hurt at np>=40 with 8img/1080p.

5. **Encoder-side cleanup** — `release_tensor(tensor_id)` does NIXL `deregister_with_connector()` which frees the descriptor's GPU memory. If this is async-late, in-flight memory peaks higher than steady-state. Not currently observed as a problem.

## Summary

The post-patch flow is **end-to-end GPU-resident**: encoder vision tower → NIXL `create_readable` (cuda:0) → RoCE GPUDirect RDMA → NIXL receive descriptor (cuda:0, was cpu pre-patch) → SGLang `async_generate(image_data=...)` directly. Verified by `device=cuda:0` on both `local_descriptors` and `remote_descriptors` in 33/33 ReadOperations during the bench, and by the +1.4 GB GPU memory bump on PD during request bursts (NIXL receive buffers materialising on the H200).

The remaining structural improvements (lifecycle queue for `stage_embeddings=True`, coordinated `mem_fraction_static` for descriptor pool, fixing `max_items=0`) are documented in the 32B docs but not yet applied.

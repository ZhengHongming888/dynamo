# SGLang E_PD encoder router — what selects which encoder?

## TL;DR
**Round-robin via atomic counter.** Not load-aware, not KV-aware, not sticky. The kv_router
scheduler doesn't even have a notion of "encoder" worker type — encoders use the generic
`PushRouter::round_robin` path.

## Two dispatch hops, both round-robin

| Hop | Selector | Configurable? |
|---|---|---|
| **Frontend → encoder** (sglang E_PD where encoder is the registered model) | `PushRouter` with `router_mode` (defaults RoundRobin) | Yes — `--router-mode` / `DYN_ROUTER_MODE` |
| **PD worker → encoder** (vLLM / TRT-LLM E_PD) | Hard-coded `.round_robin()` | No |

## The actual algorithm

`/opt/dynamo/lib/runtime/src/pipeline/network/egress/push_router.rs:358`:

```rust
pub async fn round_robin(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
    let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
    let instance_ids = self.client.instance_ids_avail();
    let count = instance_ids.len();
    instance_ids[counter % count]
    ...
}
```

Atomic counter modulo number of available instances. **Per-process** (the frontend's counter is
independent of the PD worker's counter).

## For our setup (sglang E_PD on giga01 + B70)

The frontend on giga01 sees the **encoder** as the registered model and routes via
`PushRouter::generate` with router_mode = whatever the frontend launch args set.

In our launcher script `start_sglang_pd_cuda_32b_fp8_giga01.sh`:
```
python3 -m dynamo.frontend --http-port 7001 --router-mode kv --router-reset-states
```

So we set `--router-mode kv`. **But** for the encoder hop, kv mode is meaningless (encoders
don't publish KV events). What likely happens internally is: kv-mode falls back to round-robin
for encoder targets, OR the encoder gets handled by the chat pipeline's own dispatcher which
uses round-robin by default. In the frontend logs we only saw `worker_type=decode` selected —
there was no `worker_type=encoder` log line because the kv_router scheduler doesn't track
encoders.

## Knobs you could change

**Frontend-side** (affects how giga01 picks among 4 B70 encoders), via
`/opt/dynamo/components/src/dynamo/frontend/frontend_args.py:237-256`:

- `--router-mode round-robin` (default)
- `--router-mode random`
- `--router-mode power-of-two` (load-aware: pick 2 random, choose less-busy)
- `--router-mode least-loaded`
- `--router-mode device-aware-weighted`

`power-of-two` would be the smartest pick if encoder load varies — but in our case all 4 B70
XPUs are identical and ViT-bound, so round-robin is roughly optimal for our workload anyway.

**PD-worker→encoder hop** (vLLM/TRT-LLM only): no knob, would need a source patch. Doesn't
apply to our SGLang setup.

## Why this matters for our 4E results

The 30%-of-1E (instead of 25%) ratio in `patched_1E_results.md` is **not** due to a smart
router — round-robin with 4 identical encoders should give exactly 4× throughput if encoders
are the only bottleneck. The fact that we see 4E getting only ~3.3× implies something else
(NIXL setup overhead, PD-side prefill backlog, ZMQ sched delays) is taking a slice. So the
router policy isn't a meaningful tuning knob for this workload — even the smartest router
can't help when all 4 encoders are equally loaded.

## Files involved

### Rust core (the actual selection algorithm)

- `/opt/dynamo/lib/runtime/src/pipeline/network/egress/push_router.rs`
  - `enum RouterMode` at line 161 — `#[default] RoundRobin` (line 162-163)
  - `PushRouter::round_robin` at line 359 — atomic counter modulo instance count
  - `PushRouter::generate` at line 862 — dispatches based on `router_mode`
- `/opt/dynamo/lib/bindings/python/rust/lib.rs`
  - `Endpoint::client` at line 880 — Python `endpoint.client()` returns a `PushRouter`.
    **Default `router_mode = RoundRobin`** (line 886).
  - `Client::round_robin` at line 991 — what Python's `client.round_robin(...)` actually calls.

### Frontend (Rust): how the encoder is picked when it is the model

- `/opt/dynamo/lib/llm/src/discovery/watcher.rs` lines 462–655 — the `ModelInput::Tokens`
  + supports_chat path that the sglang encode worker registers under. At line 589, the chat
  pipeline is built with `self.router_config.router_mode`. The encode worker has no special
  branch — it is treated as any other Tokens+Chat model.
- `/opt/dynamo/lib/llm/src/entrypoint/input/common.rs` lines 285–367 —
  `build_routed_pipeline_with_preprocessor`. The `service_backend` for any non-KV mode
  (line 348-356) is just the `PushRouter`.

### Encoder registration (so you know how it shows up in discovery)

- `/opt/dynamo/components/src/dynamo/sglang/args.py:271-272` — encoder advertises itself at
  `dyn://{namespace}.encoder.generate`
- `/opt/dynamo/components/src/dynamo/sglang/init_multimodal.py:39-43, 65` —
  `init_multimodal_encode_worker` serves the endpoint and registers a model with
  `ModelInput.Tokens` (line 78)

### PD-worker → encoder dispatch (vLLM / TRT-LLM E_PD)

- `/opt/dynamo/components/src/dynamo/trtllm/workers/llm_worker.py:147-157` — constructs
  `encode_client = await runtime.endpoint(...).client()` (no router_mode → RoundRobin)
- `/opt/dynamo/components/src/dynamo/trtllm/multimodal/embedding_fetcher.py:99` —
  `await encode_client.round_robin(request, context=trace_context)`
- `/opt/dynamo/components/src/dynamo/trtllm/request_handlers/handlers.py:118` —
  `PrefillHandler.remote_encode_with_nixl` does
  `await self.encode_client.round_robin(request, context=context)`
- `/opt/dynamo/components/src/dynamo/vllm/worker_factory.py:641-654` —
  `_maybe_get_encode_worker_client` (gated by `config.route_to_encoder` /
  `--route-to-encoder`)
- `/opt/dynamo/components/src/dynamo/vllm/multimodal_utils/prefill_worker_utils.py:138-191` —
  `_fetch_from_encode_workers`. It uses `encode_worker_client.instance_ids()` to count
  workers, splits image URLs into batches, and dispatches each batch via
  `encode_worker_client.round_robin(payload, context=context)` (lines 182, 190).
- `/opt/dynamo/components/src/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py:420` —
  encoder→PD direction, also `round_robin`.

## Things explicitly NOT in the codebase (verified)

- No file/function named `select_encoder`, `pick_encoder`, `choose_encoder` anywhere in
  `/opt/dynamo` or `/opt/venv/.../dynamo`.
- No "encoder" code path in `/opt/dynamo/lib/llm/src/kv_router/` — `grep -i 'encoder|encode'`
  in that subtree returns only references to a Prometheus `TextEncoder` (`metrics.rs:489-491`),
  unrelated to worker selection.
- The kv_router scheduler only uses `WORKER_TYPE_DECODE` and `WORKER_TYPE_PREFILL`
  (`/opt/dynamo/lib/llm/src/discovery/worker_monitor.rs:29`,
  `/opt/dynamo/lib/llm/src/discovery/watcher.rs:486`). The "Selected worker: worker_type=decode"
  log line we saw can never say `worker_type=encoder` — that string isn't anywhere in the
  kv_router selection paths.
- `/opt/dynamo/lib/llm/src/kv_router/sticky_sessions.rs` does not reference encoders.
- The multimodal hashing in `embedding_fetcher.py:156` and `encode_utils.py:41` is for
  **caching the embedding output** keyed by URL, not for selecting which encoder gets the
  request.

## Configuration summary

### Frontend → encoder (sglang E_PD where encoder is the registered model)

- CLI flag: `--router-mode`
  (`/opt/dynamo/components/src/dynamo/frontend/frontend_args.py:237-256`)
- Env var: `DYN_ROUTER_MODE`
- Choices: `round-robin` (default), `random`, `power-of-two`, `kv`, `direct`, `least-loaded`,
  `device-aware-weighted` (full list in `frontend_args.py:248-256`)
- Mapping done at `/opt/dynamo/components/src/dynamo/frontend/main.py:239-258`
- This setting is what gets handed to `PushRouter` in `discovery/watcher.rs` for the
  encoder's chat pipeline. So if you set `DYN_ROUTER_MODE=power-of-two` on the frontend, the
  encoder selection in this hop becomes load-aware (in-flight count).

### PD worker → encoder (vLLM / TRT-LLM E_PD)

- **There is no knob.** The Python code calls `runtime.endpoint(...).client()` with no
  `router_mode` arg (defaulting to RoundRobin) and then explicitly invokes the
  `.round_robin(...)` method. To change it you'd have to patch the source.
- Related but orthogonal CLI flags:
  - vLLM: `--route-to-encoder`
    (`/opt/dynamo/components/src/dynamo/vllm/backend_args.py:69`) — toggles whether the PD
    worker calls a remote encoder at all.
  - TRT-LLM: `--encode-endpoint`
    (`/opt/dynamo/components/src/dynamo/trtllm/backend_args.py:181`) — sets which
    `namespace.component.endpoint` to call.
  - vLLM: env-controlled `SPLIT_ENCODE` (used at `prefill_worker_utils.py:159`) controls
    whether multi-image requests get fanned out across N encoders (one batch per round_robin
    call) vs sent to a single encoder.

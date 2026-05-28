# 2E 8img/768p np=32 rate=1.0: partial result + routing discoveries

**Date:** 2026-05-28
**Setup:** super21 PD GPU 5 (mem-frac=0.50) + 2 encoders on dell06 GPUs 0/1
**Workload:** Qwen3-VL-32B-Instruct-FP8, 8 imgs × 1024×768, in=128 / out=256, rate=1.0 RPS, np=32

**Result:** **13/32 successful (40%)**, partial run — encoder #1 overloaded by skewed router balance.

Result paths:
- 2E 768p: `/hongming/res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_2enc_20260528_062928/`
- 2E 1080p baseline: `/hongming/res_xhost_dell06_super21/32b_8img_1080p_rate1.0_np32_2enc_20260528_050453/`
- 1E 768p np=16 reference: `/hongming/res19_1E_patched_sweep/8img_768p/rate_1.0_np16/`

## Numbers

| Metric | 2E 768p (13/32) | 2E 1080p (32/32) | 1E 768p np=16 (16/16) |
|---|---:|---:|---:|
| Successful | 13/32 (40%) | 32/32 (100%) | 16/16 |
| Duration (s) | 26.1 | 131.5 | 78.3 |
| **RPS** (partial) | **0.50** | 0.24 | 0.20 |
| Mean TTFT (ms) | 10 582 | 93 480 | 56 099 |
| Mean TPOT (ms) | 38 | 86 | 110 |
| P99 TPOT (ms) | 96 | 1 004 | — |
| Concurrency | 7.9 | 24.0 | 13.7 |
| Peak decode (tok/s) | 226 | 461 | 871 |

## What worked

When requests landed and ran, **768p was much faster than 1080p**:
- TTFT: 10.6 s vs 93.5 s — 9× faster
- TPOT: 38 ms vs 86 ms — 2× faster
- Per-req embedding payload: ~80 MB vs ~250 MB — 3× smaller NIXL transfer

This confirms the prediction from `2E_vs_1E_dell06_super21_32b_8img_1080p.md`: at lower
resolution, the bottleneck shifts off PD prefill and onto encoder vision tower compute,
so 2E starts to actually scale.

**Apparent 2.5× RPS gain over 1E** (0.50 vs 0.20) is in the right direction but unreliable
given only 13 reqs completed.

## What failed: skewed encoder routing

KV router worker selection during bench (33 routings):
- Encoder #1 (`bb7a`): **25 routings** (76%)
- Encoder #2 (`bb82`): 8 routings (24%)

Encoder #1 hit ~25 concurrent requests on a single H200 GPU running the vision tower.
Each request's vision compute holds GPU memory and sequential queue position. Result:
19 requests failed with `internal server error during processing` in 400-1000 ms each
(too fast for real compute, suggesting fast-fail at ingress: queue full or model load
contention).

PD on super21 received 13 prefill events during the bench, exactly matching the 13
successes — so PD never saw the failed reqs. **Failure happened at encoder before
forwarding.**

### Why is routing skewed?

Both encoders advertise identical ModelCard config (same model, same context, same
block_size). KV router's `Formula = prefill_blocks + decode_blocks` returns the same
score for both, so the tie-breaker picks **tree size** (request history). Once a few
requests land on encoder #1, its tree depth grows, and the tie-breaker keeps preferring
it for prefix-cache reasons that don't actually apply (no shared prompts in random JPEGs).

This is a router policy issue at low cache-hit-rate workloads.

## Architectural finding: encoders are the entry point, not preprocessor

While debugging routing, I confirmed the actual cross-host architecture:

```
┌──────────┐  chat/completions   ┌──────────────┐  embedding pull (NIXL)   ┌────────────────┐
│ frontend │─────────────────────→│ encoder      │←─────────────────────────│ PD on super21  │
│ super21  │                      │  (dell06)    │  PD-initiated READ       │  (multimodal-  │
└──────────┘                      │  bb7a / bb82 │                          │   worker)      │
                                  └──────┬───────┘                          └────────────────┘
                                         │  forward SglangMultimodalRequest envelope
                                         │  (TCP request plane → PD on super21:35341)
                                         ↓
                                  PD runs LLM forward, streams tokens back
```

So the two H200 GPUs on dell06 are **chat-completion entry points** that:
1. Receive the chat completion HTTP body from frontend
2. Tokenize + apply chat template
3. Run the vision tower locally to extract image embeddings
4. Publish embeddings via NIXL
5. Forward a `SglangMultimodalRequest` envelope to PD on super21
6. PD pulls embeddings from dell06 via NIXL (RoCE 192.165.123.x)
7. PD runs full LLM prefill+decode, streams tokens back to encoder
8. Encoder relays stream to frontend → client

**Key implication:** when comparing "2E vs 1E", we're measuring **2 entry-point-encoders**
vs **1 entry-point-encoder**. Both still funnel through one PD bottleneck for LLM forward.

The bench duration in 2E 1080p (131 s for 32 reqs at rate=1.0) was determined by PD
prefill capacity. At 768p, PD is fast enough that encoder vision compute starts to
matter — 2× more entry points helps if load is balanced.

## What was investigated and ruled out

During this session I went down several wrong paths trying to make 768p smoke pass:

1. **Frontend restart caching state** — first 1080p bench worked because the original
   frontend (started 01:04) had been alive when PD `bb09` first registered. After
   restart, cached routing state was lost.

2. **PD missing ModelCard in etcd** — confirmed PD's `dynamo.sglang --multimodal-worker`
   mode does NOT register an MDC. Only encoders register MDCs. This is by design:
   the encoders are the chat entry point, PD is internal.

3. **Manual MDC injection for PD** (option C from prior session) — synthesized a
   backend MDC, frontend picked it up, requests routed directly to PD bypassing the
   encoder, and PD rejected with `pydantic_core.ValidationError: missing field 'request'`
   because PD's `SglangMultimodalRequest` validator requires the encoder-wrapped envelope.
   This proved encoders are required intermediaries.

4. **NIXL_ERR_REMOTE_DISCONNECT** — first 768p attempt at 05:43 hit this because the
   encoders had been idle ~80 min after the 05:07 1080p bench. Their NIXL agents went
   stale. After dell06-side encoder restart, fresh NIXL handshake worked.

After restoring the encoder MDCs and removing the synthetic backend MDC, routing went
back to encoder-mediated flow and 768p smoke + bench succeeded (with skew problem).

## Recommendations

1. **For reliable 2E benchmarking**: explicitly balance encoders. Either:
   - Run with `--router-mode random` instead of `--router-mode kv` (eliminates
     prefix-cache tie-breaker that overweights one encoder)
   - Or run with `--max-concurrency 16` per request batch so neither encoder oversaturates

2. **For comparing encoder regimes**: a dedicated `1E 768p np=32` baseline would be
   apples-to-apples (current `1E 768p np=16` baseline isn't the same workload size)

3. **Address router skew**: file a dynamo issue — KV router tie-breaker shouldn't use
   tree size for workloads with no shared prefix (vision-heavy benchmarks). Could use
   round-robin or in-flight-request count instead.

4. **For deeper analysis**: instrument the encoder side on dell06 to capture the 19
   fast-fail reqs. Currently we only see them as `internal server error` in frontend.

## Files
- This doc: `disagg_h200_32b/2E_768p_np32_partial_routing_findings.md`
- 2E 768p result: `res_xhost_dell06_super21/32b_8img_768p_rate1.0_np32_2enc_20260528_062928/`
- PD log (DYN_LOG=debug): `disagg_h200_32b/logs/pd_worker_giga01_lowmem_20260528_060548.log`
- Frontend log: `disagg_h200_32b/logs/frontend_giga01_20260528_062823.log`
- Prior 2E 1080p analysis: `disagg_h200_32b/2E_vs_1E_dell06_super21_32b_8img_1080p.md`

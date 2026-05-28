# Dynamo KV-router 5-minute hardcoded request expiry — root cause and fix

**Date:** 2026-05-24
**Severity:** Functional bug for any disagg config where per-request E2E latency exceeds 5 minutes
**Affected:** All cross-host disagg configs with slow-per-request workloads (1E + 8img/1080p, P/D split with slow KV transfer, etc.)

## Symptom

When testing 1E (single encoder) cross-host disagg at 8img/1080p rate=1.0:
- Bench reports 1/32 successful (just the warmup)
- PD-side log shows 25 requests with valid `ReqTimeStats` (PD compute completed)
- Frontend log shows `WARN: Expiring stale request: <rid>` for all stuck requests starting at exactly 4-5 minutes after the bench started
- Eventually frontend logs `ERROR ... error_type=cancelled error_detail=cancelled before completion`

PD does the work, frontend kills the responses anyway.

## Root cause

`/opt/dynamo/lib/kv-router/src/sequences/single.rs` lines 33-39:

```rust
/// Duration after which stale requests may be expired (5 minutes).
const EXPIRY_DURATION: Duration = Duration::from_secs(300);

/// How often we *check* for stale requests (30 seconds). This is not
/// the expiration time, that is EXPIRY_DURATION.
const CHECK_EXPIRY_FREQUENCY: Duration = Duration::from_secs(30);
```

The `force_expiry()` function at lines 429-452 runs every 30 seconds, and forcibly cancels any request that's been alive in the router for >300 seconds. There's no env var to override these constants. They're hardcoded.

## Why this kills 1E but not 4E for 8img/1080p

- **4E**: 4 parallel encoders → ~9s per request (encoder ViT) + ~5s (PD compute) = ~14s per-request inherent latency. Even with 32 prompts at rate=1.0, mean E2E ~210s, P99 ~242s. **All under 300s.** ✓

- **1E**: serialized encoder → ~33s per request encoder time. At rate=1.0 with 32 prompts, request #N reaches PD at t=N×33s. Request #10 reaches PD at t=330s, **already past the 300s expiry**. ✗

Even with patching, request 10+ would expire. With 1E, only request 1-9 could possibly succeed.

## Why warmup and smoke test work

The warmup is sent in isolation before the rate=1.0 batch starts:
- Smoke test (1 image): 8s end-to-end → easily under 300s ✓
- Warmup (1 image, separate from main batch): 33s end-to-end → easily under 300s ✓

The main batch's first prompt also makes it through (similar timing). Everything after queues up at the encoder and ages past 300s.

## Specific evidence from the run

| Request | PD-completed at | Frontend stale-expired at | Outcome |
|---|---|---|---|
| `fce5a553` (smoke test, 1img) | 18:35:38 | (never expired) | ✓ success |
| `c7093953` (bench warmup, 1 prompt rate=0.1) | 18:38:21 | (never expired) | ✓ success |
| `503fa1c2` (bench prompt #1) | 18:38:55 | **18:43:33** (4m 38s after PD) | ✗ stuck → killed |
| `b7283c93` (bench prompt #2) | 18:39:25 | **18:43:33** (same sweep) | ✗ stuck → killed |
| `c3ffe4da` (bench prompt #3) | 18:39:52 | **18:43:33** (same sweep) | ✗ stuck → killed |
| ... | ... | ... | (32 total expirations) |

The first stale-expiry sweep at **18:43:33** wiped out everything that was alive for >300s. Subsequent sweeps at ~30s intervals expired more requests as they aged past 300s.

## The bug path

1. Bench client sends HTTP POST → frontend receives it
2. Frontend allocates a request_id and calls into KV router
3. KV router's `RequestSequence::add_request()` records `started_at = Instant::now()`
4. Request is routed via TCP to encoder, then PD
5. Encoder runs ViT (~33s for 8img/1080p on B70 XPU)
6. NIXL transfers embedding to PD
7. PD runs prefill+decode (~5s)
8. PD streams response tokens back through Dynamo TCP plane → frontend → HTTP SSE → bench
9. **(parallel)** every 30s, KV router runs `force_expiry()`. Any request older than 300s gets `tracing::warn!("Expiring stale request...")` and is forcibly freed
10. The freed state breaks the response stream — frontend can't deliver further tokens, eventually times out the HTTP connection

The bug: the expiry doesn't check whether the request is **actually making progress** — it just looks at wall-clock age.

## Fix options

### Option 1: Patch the constant (quick fix)

Edit `/opt/dynamo/lib/kv-router/src/sequences/single.rs` line 34:

```diff
- const EXPIRY_DURATION: Duration = Duration::from_secs(300);
+ const EXPIRY_DURATION: Duration = Duration::from_secs(1800);  // 30 minutes
```

Then rebuild the dynamo-py3 binding:

```bash
cd /opt/dynamo/lib/bindings/python
maturin build --release --features default
# or alternatively
cargo build --release
# then copy the resulting _core.abi3.so to /opt/venv/lib/python3.12/site-packages/dynamo/
```

Trade-off: requests that legitimately got stuck (e.g., dead workers, lost network) take 30 min to clean up instead of 5 min. Acceptable for benchmarking.

### Option 2: Make it env-var configurable (proper fix)

```rust
const DEFAULT_EXPIRY_DURATION_SECS: u64 = 300;

fn get_expiry_duration() -> Duration {
    let secs = std::env::var("DYN_KV_ROUTER_EXPIRY_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_EXPIRY_DURATION_SECS);
    Duration::from_secs(secs)
}
```

Then call this in the constructor. Slightly bigger change but proper engineering.

### Option 3: Don't expire requests that have published progress

Track "last activity time" separately from "started_at" — only expire requests that have been **silent** for >5 min, not just *alive* for >5 min. The proper architectural fix but requires more code surgery.

## Workarounds without code change

For benchmarking 1E (or any slow-per-req config) without modifying dynamo:

1. **Use `--max-concurrency 1` on bench**: bench sends one at a time, never queues at frontend, every request stays under 5 min. Loses throughput-under-load measurement but gets per-request latency.
2. **Use shorter workloads**: smaller images / fewer images / shorter output → per-request E2E < 300s
3. **Match 4E or higher encoder count**: parallelize encoder enough that per-request stays under 300s

## Files referenced

- Bug location: `/opt/dynamo/lib/kv-router/src/sequences/single.rs:34`
- Compiled binary containing the bug: `/opt/venv/lib/python3.12/site-packages/dynamo/_core.abi3.so` (100 MB, dated May 15 2026)
- Build cache: `/opt/dynamo/lib/bindings/python/target/` (2.4 GB)
- Dynamo workspace: `/opt/dynamo/`
- Cargo.toml: `/opt/dynamo/Cargo.toml`
- Rust toolchain: rustc 1.93.1, cargo 1.93.1, maturin 1.13.3 (all installed)

## Test that exposed this

- Date: 2026-05-24 18:37
- Config: 1E patched, 8img/1080p, rate=1.0, np=32
- Bench result: 1/32 successful before kill
- Stale-expiry events: 32 (every request after the warmup got expired)
- PD-side completions: 25 (PD did its job for 25 requests before bench was killed)
- PD measured throughput (independent of frontend): ~0.030 req/s

## Related session findings

This bug also explains some earlier confusion:
- `cross_host_giga01_b70_results.md` — earlier 1E runs with similar symptom of "0 frontend completions" but PD log showing healthy completions
- `patched_results_b70_h200_v01.md` — initial cross-host runs that timed out at the 4-5 minute mark
- `time_breakdown_analysis.md` / `h200_time_breakdown_v02.md` — measured PD compute is fast; bottleneck is encoder, but that pushes E2E past the 5-min threshold under load

This is **not** specific to our patches. It would affect any dynamo + slow-encoder + heavy-image workload combination, including the unpatched baseline. The Brian Liu paper (`vllm_epd_paper_v04.md`) used 20img/480p (smaller embedding, faster ViT) on 4E specifically, which is fast enough to never hit 300s. They probably didn't notice this bug because their workloads stay well under it.

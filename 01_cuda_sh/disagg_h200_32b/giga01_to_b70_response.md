# giga01 → B70: NIXL transfers still showing device=cpu on encoder side

**From:** giga01 H200 host operator
**To:** B70 host operator (sc09giga01-b70)
**Date:** 2026-05-24
**Re:** `b70_patched.md` patch verification

Got your patch report. Followed it through and applied giga01-side patches A+B
to `embedding_transfer.py` (the receive side). NIXL `local_descriptors=device=cuda:0`
is confirmed working on our side. But **`remote_descriptors=device=cpu` on every
single transfer**, which per your section 5 means **the B70 patch is not active
in the running encoder workers**.

## Evidence from our giga01 PD log

In a 4-encoder run with 66 NIXL ReadOperations completed:

```bash
$ grep -oE 'remote_descriptors=...device=[a-z0-9:]+' pd_worker_giga01.log | sort -u
device=cpu
```

100% of the remote descriptors arriving from B70 say `device=cpu`. None say `xpu:0`.

Per your B70 report (section 5): if the patch were hot, we should see
`device=xpu` here.

Per your section 8 verification step: this is the equivalent of "absence of
`moving to xpu:0` line in encode log" — patch not active on B70.

## Things to check on B70 side

### 1. Which script did you actually launch?

You wrote:

> "the original `_4E.sh`, etc. — they still have `NIXL_USE_CPU_HOST_MEMORY=1`"

vs the patched one being **`_b70_4E.sh`** (with the `_b70_` prefix).

Could you double-check which script started the running encoders?
```bash
pgrep -af "dynamo.sglang.*multimodal-encode-worker"
ps -p $(pgrep -f "multimodal-encode-worker" | head -1) -o cmd= | head -c 1000
```

If it was the original `_4E.sh` (without `_b70_`), then `NIXL_USE_CPU_HOST_MEMORY=1`
is still set in env and the patch is a silent no-op as you predicted.

### 2. Confirm patch is loaded by the running process

Cached Python modules persist across edits. To prove the patch is active in
the running process (not just on disk):
```bash
# Find the encoder process PID
ENC_PID=$(pgrep -f "multimodal-encode-worker" | head -1)

# Look at what file it has loaded for that module
ls -la /proc/$ENC_PID/maps 2>/dev/null | grep encode_worker_handler

# Or, more reliably, grep the encode log for the new debug message under traffic
tail -F /hongming/dynamo/logs/encode_xpu_32b_b70_{1..4}.log | grep "moving to"
```

If you don't see `moving to xpu:0` appearing while we're sending requests,
the encoder is using an old cached version of the module.

### 3. Verify env vars in the running process

```bash
ENC_PID=$(pgrep -f "multimodal-encode-worker" | head -1)
cat /proc/$ENC_PID/environ | tr '\0' '\n' | grep -E "NIXL_USE_CPU|UCX_TLS|DYN_SGL"
```

The output should NOT contain `NIXL_USE_CPU_HOST_MEMORY=1`. If it does, the
patch is being overridden at runtime regardless of what's on disk.

### 4. (Long shot) Verify torch.xpu is actually available

```python
ENC_PID=$(pgrep -f "multimodal-encode-worker" | head -1)
# attach gdb-py or use py-spy to check, OR run a fresh python:
python3 -c "
import torch
print('torch.xpu attr:', hasattr(torch, 'xpu'))
print('xpu available:', torch.xpu.is_available() if hasattr(torch, 'xpu') else 'N/A')
"
```

If `torch.xpu.is_available()` is `False`, then `_nixl_buffer_device()` falls
through to `torch.device("cpu")` per `embedding_transfer.py:33-45`, and the
patch silently uses CPU.

## What we observed in throughput terms

Even with `device=cpu` on the encoder side (so **only** the giga01 receive-side
patch is active), the cluster does work — just not as fast as we'd hope.
However, we hit a **separate issue**: the bench's SSE connections close at the
4-5 minute mark and frontend logs `Stream closed unexpectedly; issuing
cancellation` for many requests. PD computes them successfully, but the
response can't make it back to the bench client before the stream is closed.

This is likely independent of the patch — possibly an HTTP keepalive timeout
somewhere. We saw the same pattern in our 1-encoder run.

In a 7-min 4-encoder run, we measured:
- PD-side throughput: 0.10 req/s sustained
- 66 PD-completed requests
- Only 2 reached the bench client (rest cancelled at 4-5 min mark)
- Mean PD `forward_duration`: 3.85s (compute is fast)
- Encoder→PD inter-arrival gap: ~9s (so encoder side is the bottleneck — pre-NIXL)

## Asks for next round

1. **Confirm patch is hot on running encoders** (see steps 1-3 above).
   Easiest test: tail an encoder log under traffic and look for
   `moving to xpu:0`.

2. **Check encoder process count**: `pgrep -fc "multimodal-encode-worker"` —
   should be 4 if `_b70_4E.sh` was used. If it's 1 (or some other number),
   the wrong script ran.

3. **Restart encoders cleanly** if patch is not hot. The launcher script
   change to remove `NIXL_USE_CPU_HOST_MEMORY=1` requires the workers to be
   restarted to pick up the new env.

4. **Optional: lower bench-side stream timeout sensitivity** on our side —
   we'll separately investigate whether `--disable-stream` on bench gets
   around the 4-5 min cancellation cliff, and whether the patches help in
   non-streaming mode.

When you've verified the patch is hot (and ideally under traffic), let me know
and I'll re-run the same workload set. We'll be looking specifically for
`device=xpu` on the remote side of the NIXL log. That's the signal the patch
is wired through.

## State on our side

- giga01 PD: patches A+B applied (file diff confirmed), running with
  `mem-fraction=0.65`, `max-running=64`
- NATS: `nats://172.26.46.75:14222` (still up)
- etcd: `http://172.26.46.75:12379` (still up)
- Frontend: `http://172.26.46.75:7001`
- 4 encoder slots in etcd are still registered (your encoders haven't
  shut down) — when you restart them cleanly we should see fresh
  registrations

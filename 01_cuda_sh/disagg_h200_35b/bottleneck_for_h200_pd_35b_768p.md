# Analysis written to `/hongming/dynamo/01_cuda_sh/disagg_h200_35b/4img_768p_4E_bottleneck.md`

## Key findings

**The bottleneck has shifted from where 1E had it.**

In 1E (`35b_bottleneck_analysis.md`): the bottleneck was a single B70 XPU running `self.visual()`. PD was idle 96%. Forward = 0.82 s, ViT = 1.41 s, ratio 1.4 s of "wait for the next embedding."

In 4E (4img/768p): the bottleneck is now **the encoder→PD pipeline cadence as a whole**, not any single component:

| Component | Time | Verdict |
|---|---|---|
| 4 parallel B70 ViTs | ~0.35 s/img amortized | Not bottleneck — well-fed |
| PD `queue_duration` | 0.26 ms | Negligible |
| NIXL fetch + dispatch + ZMQ | 125-180 ms | **Per-request overhead floor** |
| PD `forward_duration` | 1055 ms | Now genuinely concurrent (1-3 deep) |
| PD lifetime per req | 1.18 s | Locked at this floor |

Saturation = `1 / (mean_inter_arrival on PD)` = `1 / 0.658 s` = **1.52 RPS** (matches measured 1.55).

## Three observations that surprised me

1. **No bursting.** I expected 4 encoders to deliver requests in clusters of 4. They don't — every PD ingress is >100 ms from any neighbor. dynamo's round-robin dispatcher serializes them.

2. **PD now batches.** First time in any 35B disagg run: `running-req=1` is only 46% at saturation. `running-req=2` is 40%, `running-req=3` is 14%. Effective concurrency ~1.7.

3. **The cross-host RoCE NIXL handoff is only ~150 ms**, vs the ~11.5 s "per-request handoff floor" documented for same-host 32B in `per_request_handoff.md`. The cross-host RoCE path quietly side-stepped the CPU-bounce that plagued same-host disagg.

## Why 1.55 RPS instead of perfect 4× = 1.76 RPS

12% loss = ~150 ms × 1.55 RPS = 0.23 s/s of un-overlapped per-request PD overhead. 87% efficient. Not transformative to fix.

## Things confirmed not the bottleneck

- PD compute (max queue=0, util ~60%, running-req never hits 4)
- PD scheduler (q_med=0.26 ms — instant enqueue)
- NIXL ring buffers (no timeouts, no back-pressure)
- KV cache (<0.1% utilization)
- max-running-requests (40 configured, 3 actually used)

Stack remains up. Want me to sweep 8img/768p next (between the two extremes), tear down, or something else?

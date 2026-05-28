#!/usr/bin/env python3
"""Deep-dive into the saturation regime (rate=2.0, 3.0) for 4img/768p_4E.
Goals:
  1. Walk through a single saturation-rate request's PD-side timeline.
  2. Compute what fraction of time PD is "doing prefill" vs "doing decode" vs idle.
  3. Identify if there's any wait between encoder→PD that's still significant.
  4. Check if 4E throughput is limited by:
     (a) PD prefill compute (visual token integration)
     (b) PD decode compute (text generation)
     (c) PD scheduler queue / serialization
     (d) encoder→PD handoff floor
"""
import re
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

PD_LOG = Path("/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01.log")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")
REQ_RE = re.compile(
    r"ReqTimeStats\(rid=(?P<rid>[0-9a-f]+), input_len=(?P<input>\d+), "
    r"cached_input_len=(?P<cached>\d+), output_len=(?P<output>\d+), type=(?P<type>\w+)\):"
    r" queue_duration=(?P<q>[\d.]+)ms, forward_duration=(?P<f>[\d.]+)ms,"
    r" entry_time=(?P<entry>[\d.]+)"
)
PREFILL_RE = re.compile(
    r"Prefill batch, #new-seq: (?P<seq>\d+), #new-token: (?P<newtok>\d+), "
    r"#cached-token: (?P<cached>\d+),.*?#running-req: (?P<run>\d+), "
    r"#queue-req: (?P<queue>\d+).*?cuda graph: (?P<cg>\w+).*?input throughput "
    r"\(token/s\): (?P<tput>[\d.]+)"
)
DECODE_RE = re.compile(
    r"Decode batch, #running-req: (?P<run>\d+),.*?gen throughput "
    r"\(token/s\): (?P<tput>[\d.]+),.*?#queue-req: (?P<queue>\d+)"
)

def iso_to_unix(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()

def main():
    raw = ANSI.sub("", PD_LOG.read_text())
    lines = raw.splitlines()

    events = []  # [(ts, type, dict)]
    for ln in lines:
        ts_m = TS_RE.search(ln)
        if not ts_m: continue
        ts = iso_to_unix(ts_m.group(1))
        if "ReqTimeStats" in ln:
            m = REQ_RE.search(ln)
            if m:
                d = m.groupdict()
                events.append((ts, "req", {
                    "rid": d["rid"],
                    "input_len": int(d["input"]),
                    "cached": int(d["cached"]),
                    "output": int(d["output"]),
                    "q": float(d["q"]),
                    "f": float(d["f"]),
                    "entry": float(d["entry"]),
                }))
        elif "Prefill batch" in ln:
            m = PREFILL_RE.search(ln)
            if m:
                d = m.groupdict()
                events.append((ts, "P", {
                    "seq": int(d["seq"]),
                    "newtok": int(d["newtok"]),
                    "cached": int(d["cached"]),
                    "run": int(d["run"]),
                    "queue": int(d["queue"]),
                    "cg": d["cg"]=="True",
                    "tput": float(d["tput"]),
                }))
        elif "Decode batch" in ln:
            m = DECODE_RE.search(ln)
            if m:
                d = m.groupdict()
                events.append((ts, "D", {
                    "run": int(d["run"]),
                    "tput": float(d["tput"]),
                    "queue": int(d["queue"]),
                }))

    events.sort(key=lambda x: x[0])

    # Saturation rate=2.0 window: 23:43:40 - 23:44:22
    s = iso_to_unix("2026-05-25T23:43:40.000000Z")
    e = iso_to_unix("2026-05-25T23:44:22.000000Z")

    print(f"=== Saturation rate=2.0 window: 42 s, RPS=1.54, 32/32 OK ===\n")

    in_win = [(t,k,d) for t,k,d in events if s <= t <= e+2]
    print(f"Events in window: {sum(1 for _,k,_ in in_win if k=='P')} prefills, "
          f"{sum(1 for _,k,_ in in_win if k=='D')} decode batches, "
          f"{sum(1 for _,k,_ in in_win if k=='req')} req completions")
    print()

    # Print a chronological dump of first 20 events
    print("=== First 20 events of saturation window (relative time in s) ===")
    for t, k, d in in_win[:30]:
        rel = t - s
        if k == "P":
            print(f"  +{rel:5.2f}s  P  #seq={d['seq']} #new={d['newtok']:>5} #cached={d['cached']:>5}  "
                  f"run={d['run']} queue={d['queue']} cg={d['cg']} tput={d['tput']:.0f} tok/s")
        elif k == "D":
            print(f"  +{rel:5.2f}s  D  run={d['run']} queue={d['queue']} tput={d['tput']:.2f} tok/s")
        elif k == "req":
            print(f"  +{rel:5.2f}s  REQ_DONE input={d['input_len']:>5} out={d['output']:>3} q={d['q']:.2f}ms fwd={d['f']:.0f}ms")

    print()
    print("=== Decode-batch running-req distribution (rate=2.0) ===")
    dec_run = Counter([d["run"] for t,k,d in in_win if k=="D"])
    total = sum(dec_run.values())
    for r in sorted(dec_run.keys()):
        n = dec_run[r]
        print(f"  running={r}:  {n:>4} batches  ({n/total*100:.1f}%)")

    print()
    print("=== Prefill-batch shape (rate=2.0) ===")
    pre_in = [d for t,k,d in in_win if k=="P"]
    new_toks = sorted([p["newtok"] for p in pre_in])
    cached_toks = sorted([p["cached"] for p in pre_in])
    print(f"  #new-token p50={new_toks[len(new_toks)//2]}  p99={new_toks[-1]}  min={new_toks[0]}")
    print(f"  #cached-token p50={cached_toks[len(cached_toks)//2]}  p99={cached_toks[-1]}")
    print(f"  prefill #seq distribution: {Counter([p['seq'] for p in pre_in])}")
    print(f"  prefill running-req distribution: {Counter([p['run'] for p in pre_in])}")
    print(f"  prefill queue-req distribution: {Counter([p['queue'] for p in pre_in])}")

    print()
    print("=== ReqTimeStats for sat rate=2.0 ===")
    reqs_in = [d for t,k,d in in_win if k=="req"]
    if reqs_in:
        f_vals = sorted([r["f"] for r in reqs_in])
        q_vals = sorted([r["q"] for r in reqs_in])
        print(f"  forward_duration: min={f_vals[0]:.0f} p50={f_vals[len(f_vals)//2]:.0f} "
              f"p99={f_vals[-1]:.0f}  ms")
        print(f"  queue_duration:   min={q_vals[0]:.2f} p50={q_vals[len(q_vals)//2]:.2f} "
              f"p99={q_vals[-1]:.2f}  ms")
        print(f"  Total forward time across {len(reqs_in)} reqs: "
              f"{sum(r['f'] for r in reqs_in)/1000:.1f}s")
        print(f"  Window duration: 42s")
        print(f"  PD parallelism factor = sum(f)/window = {sum(r['f'] for r in reqs_in)/1000/42:.2f}x")

    # Now compute "time on PD doing prefill" vs "time on decode" vs "idle"
    # Approach: for each adjacent (P or D) batch event, the GAP from prev event to this event
    # represents work completing in that interval. We don't have explicit start times,
    # only the LOG time of each batch tick. SGLang logs every step (~10ms).
    # If consecutive events are <50ms apart, treat the gap as "active".
    # If >100ms, treat as "idle".

    print()
    print("=== PD activity profile (rate=2.0, between consecutive P/D events) ===")
    pd_events = [(t,k) for t,k,_ in in_win if k in ("P","D")]
    if len(pd_events) >= 2:
        gaps = [(pd_events[i+1][0]-pd_events[i][0], pd_events[i][1]) for i in range(len(pd_events)-1)]
        active = sum(g for g,_ in gaps if g < 0.1)
        idle = sum(g for g,_ in gaps if g >= 0.1)
        print(f"  Active (gap<100ms): {active:.2f}s ({active/(active+idle)*100:.1f}%)")
        print(f"  Idle   (gap>100ms): {idle:.2f}s ({idle/(active+idle)*100:.1f}%)")
        long_gaps = sorted([g for g,_ in gaps], reverse=True)[:10]
        print(f"  10 longest gaps: {[f'{g:.2f}s' for g in long_gaps]}")

    # Compare same metrics for rate=3.0 window
    print()
    print("=" * 70)
    print("=== Repeating for rate=3.0 (same conclusions expected) ===")
    s2 = iso_to_unix("2026-05-25T23:45:52.000000Z")
    e2 = iso_to_unix("2026-05-25T23:46:34.000000Z")
    in_win2 = [(t,k,d) for t,k,d in events if s2 <= t <= e2+2]
    dec_run2 = Counter([d["run"] for t,k,d in in_win2 if k=="D"])
    total2 = sum(dec_run2.values())
    print("Decode running-req:")
    for r in sorted(dec_run2.keys()):
        print(f"  running={r}:  {dec_run2[r]} batches ({dec_run2[r]/total2*100:.1f}%)")
    pd_events2 = [(t,k) for t,k,_ in in_win2 if k in ("P","D")]
    gaps2 = [(pd_events2[i+1][0]-pd_events2[i][0], pd_events2[i][1]) for i in range(len(pd_events2)-1)]
    active2 = sum(g for g,_ in gaps2 if g < 0.1)
    idle2 = sum(g for g,_ in gaps2 if g >= 0.1)
    if (active2+idle2) > 0:
        print(f"  Active: {active2:.2f}s ({active2/(active2+idle2)*100:.1f}%)")
        print(f"  Idle  : {idle2:.2f}s ({idle2/(active2+idle2)*100:.1f}%)")

    # Compare to 1E baseline by looking at the *original* 4img/768p log windows
    # from the 1E sweep. Those were earlier runs to the same log file before the 4E start;
    # but the current pd_worker_giga01.log was started fresh at 22:21, so the 1E 4img/768p
    # results are NOT in this log. We rely on numbers from disagg_35b_results.md instead.

if __name__ == "__main__":
    main()

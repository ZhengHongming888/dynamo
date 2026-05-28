#!/usr/bin/env python3
"""Analyze 4img/768p 4E PD-side timing per-rate, mirroring the methodology in
35b_bottleneck_analysis.md.

For each rate window we extract:
  - ReqTimeStats events that fall in the window:
      queue_duration_ms (q), forward_duration_ms (f), input_len, output_len
  - Prefill batch events: #running-req, #new-tok, #cached-tok
  - Decode batch events: #running-req, gen_throughput

Then compute:
  - PD utilization = sum(forward_duration) / window_duration
  - Inter-completion gap p50 (= 1/measured RPS, sanity)
  - Per-batch concurrency distribution (running-req histogram)
  - Implied encoder time = inter_completion_gap - PD_forward_time
  - Whether PD's #running-req ever exceeds 1 (vs always-1 in 1E sweep)
"""
import re
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

PD_LOG = Path("/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01.log")
RESULT_BASE = Path("/hongming/res22_disagg_h200_35b_sweep/4img_768p_4E")
RATES = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

# Sweep-master windows for 4img/768p_4E (UTC, from sweep_4img_768p_4E log)
WINDOWS = {
    0.10: ("23:23:41", "23:30:03"),
    0.25: ("23:31:33", "23:34:18"),
    0.50: ("23:35:48", "23:37:22"),
    1.00: ("23:38:52", "23:39:54"),
    1.50: ("23:41:24", "23:42:10"),
    2.00: ("23:43:40", "23:44:22"),
    3.00: ("23:45:52", "23:46:34"),
}
DAY_PREFIX = "2026-05-25T"

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

def parse_window(s, e):
    """Convert HH:MM:SS to (unix_start, unix_end)."""
    s_unix = iso_to_unix(f"{DAY_PREFIX}{s}.000000Z")
    e_unix = iso_to_unix(f"{DAY_PREFIX}{e}.000000Z")
    return s_unix, e_unix

def main():
    raw = PD_LOG.read_text()
    raw = ANSI.sub("", raw)
    lines = raw.splitlines()

    # Parse all events with timestamps
    reqs = []      # list of dicts with ts + req fields
    prefills = []
    decodes = []
    for ln in lines:
        ts_m = TS_RE.search(ln)
        if not ts_m:
            continue
        ts = iso_to_unix(ts_m.group(1))
        if "ReqTimeStats" in ln:
            m = REQ_RE.search(ln)
            if m:
                d = m.groupdict()
                reqs.append({
                    "ts": ts,
                    "rid": d["rid"],
                    "input_len": int(d["input"]),
                    "cached": int(d["cached"]),
                    "output_len": int(d["output"]),
                    "q": float(d["q"]),
                    "f": float(d["f"]),
                    "entry": float(d["entry"]),
                })
        elif "Prefill batch" in ln:
            m = PREFILL_RE.search(ln)
            if m:
                d = m.groupdict()
                prefills.append({
                    "ts": ts,
                    "seq": int(d["seq"]),
                    "newtok": int(d["newtok"]),
                    "cached": int(d["cached"]),
                    "running": int(d["run"]),
                    "queue": int(d["queue"]),
                    "cg": d["cg"] == "True",
                    "tput": float(d["tput"]),
                })
        elif "Decode batch" in ln:
            m = DECODE_RE.search(ln)
            if m:
                d = m.groupdict()
                decodes.append({
                    "ts": ts,
                    "running": int(d["run"]),
                    "tput": float(d["tput"]),
                    "queue": int(d["queue"]),
                })

    print(f"Parsed: {len(reqs)} ReqTimeStats, {len(prefills)} Prefill, {len(decodes)} Decode events")
    print(f"Time range: {datetime.fromtimestamp(reqs[0]['ts'], timezone.utc).isoformat()} ..")
    print(f"            {datetime.fromtimestamp(reqs[-1]['ts'], timezone.utc).isoformat()}")
    print()

    # Per-rate analysis
    print(f"{'Rate':>5} {'measured_RPS':>12} {'window_s':>8} {'n_req':>5} "
          f"{'q_med':>7} {'q_p99':>7} {'f_med':>7} {'f_p99':>7} "
          f"{'output_med':>10} {'inter_compl':>11} {'PD_util':>8} {'enc_implied':>11}")
    print(f"{'':>5} {'':>12} {'':>8} {'':>5} "
          f"{'(ms)':>7} {'(ms)':>7} {'(ms)':>7} {'(ms)':>7} "
          f"{'(tok)':>10} {'p50_(s)':>11} {'(%)':>8} {'(s)':>11}")

    rate_summary = {}
    for rate in RATES:
        s_str, e_str = WINDOWS[rate]
        s_unix, e_unix = parse_window(s_str, e_str)
        win_secs = e_unix - s_unix

        bench = json.load(open(RESULT_BASE / f"rate_{rate}_np32" / "benchmark_output.json"))
        measured_rps = bench["request_throughput"]
        bench_dur = bench["duration"]

        # Filter ReqTimeStats inside the window. The ReqTimeStats line is logged
        # AT request COMPLETION; entry_time + (q+f)/1000 ≈ event ts. So filter on
        # ts within [s_unix, e_unix + small grace].
        req_in = [r for r in reqs if s_unix <= r["ts"] <= e_unix + 5]
        # Prefill / Decode events in window
        pre_in = [p for p in prefills if s_unix <= p["ts"] <= e_unix]
        dec_in = [d for d in decodes if s_unix <= d["ts"] <= e_unix]

        if not req_in:
            print(f"  rate={rate}  no ReqTimeStats in window")
            continue

        q_vals = sorted([r["q"] for r in req_in])
        f_vals = sorted([r["f"] for r in req_in])
        out_vals = sorted([r["output_len"] for r in req_in])

        def q_pct(arr, p):
            if not arr: return 0
            i = max(0, min(len(arr)-1, int(p/100 * (len(arr)-1))))
            return arr[i]

        # Inter-completion gap p50
        completion_ts = sorted([r["ts"] for r in req_in])
        if len(completion_ts) >= 2:
            gaps = [completion_ts[i+1]-completion_ts[i] for i in range(len(completion_ts)-1)]
            gaps.sort()
            ic_p50 = gaps[len(gaps)//2]
        else:
            ic_p50 = float("nan")

        # PD utilization = sum(forward) / actual bench duration
        sum_fwd_s = sum(r["f"] for r in req_in) / 1000
        pd_util = (sum_fwd_s / bench_dur) * 100  # naive: serial-equivalent

        # Implied encoder = ic_p50 - f_med (only valid when serial)
        f_med_s = statistics.median(f_vals) / 1000
        enc_implied = ic_p50 - f_med_s if completion_ts else float("nan")

        rate_summary[rate] = {
            "measured_rps": measured_rps,
            "win_secs": win_secs,
            "bench_dur": bench_dur,
            "n_req": len(req_in),
            "n_prefill": len(pre_in),
            "n_decode": len(dec_in),
            "q_med": statistics.median(q_vals),
            "q_p99": q_pct(q_vals, 99),
            "f_med": statistics.median(f_vals),
            "f_p99": q_pct(f_vals, 99),
            "output_med": statistics.median(out_vals),
            "ic_p50": ic_p50,
            "pd_util": pd_util,
            "enc_implied": enc_implied,
            "input_len_med": statistics.median(sorted([r["input_len"] for r in req_in])),
            "cached_med": statistics.median(sorted([r["cached"] for r in req_in])),
            "running_max_pre": max([p["running"] for p in pre_in], default=0),
            "running_max_dec": max([d["running"] for d in dec_in], default=0),
            "running_dec_dist": {},  # filled below
            "running_pre_dist": {},
            "queue_max_pre": max([p["queue"] for p in pre_in], default=0),
            "queue_max_dec": max([d["queue"] for d in dec_in], default=0),
            "cuda_graph_pre": sum(1 for p in pre_in if p["cg"]),
            "cuda_graph_pre_total": len(pre_in),
        }

        # Histogram of running-req in this window (decode events)
        from collections import Counter
        cdec = Counter([d["running"] for d in dec_in])
        cpre = Counter([p["running"] for p in pre_in])
        rate_summary[rate]["running_dec_dist"] = dict(sorted(cdec.items()))
        rate_summary[rate]["running_pre_dist"] = dict(sorted(cpre.items()))

        print(f"{rate:>5} {measured_rps:>12.4f} {win_secs:>7.0f}s {len(req_in):>5} "
              f"{statistics.median(q_vals):>6.2f}  {q_pct(q_vals,99):>6.2f}  "
              f"{statistics.median(f_vals):>6.1f}  {q_pct(f_vals,99):>6.1f}  "
              f"{statistics.median(out_vals):>9.0f}  {ic_p50:>10.3f}  "
              f"{pd_util:>7.1f}  {enc_implied:>10.3f}")

    print()
    print("=== Detailed per-rate concurrency distribution ===")
    for rate in RATES:
        if rate not in rate_summary: continue
        r = rate_summary[rate]
        # Top 3 most common decode running-req values
        top_dec = sorted(r["running_dec_dist"].items(), key=lambda x: -x[1])[:5]
        cg_frac = (r["cuda_graph_pre"]/r["cuda_graph_pre_total"]*100) if r["cuda_graph_pre_total"] else 0
        print(f"  rate={rate:>4}: n_pre={r['n_prefill']:>3} n_dec={r['n_decode']:>4}  "
              f"max_running_pre={r['running_max_pre']:>2}  max_running_dec={r['running_max_dec']:>2}  "
              f"max_queue_dec={r['queue_max_dec']:>2}  prefill_with_cuda_graph={cg_frac:.0f}%  "
              f"top_dec_running={top_dec}")

    print()
    print("=== Saturation check (rate>=2.0): is PD really maxed? ===")
    print("If PD truly bottlenecks, util should be ~100% AND running-req frequently > 1")
    print("If encoder still bottlenecks, util << 100% AND running-req frequently == 1")

if __name__ == "__main__":
    main()

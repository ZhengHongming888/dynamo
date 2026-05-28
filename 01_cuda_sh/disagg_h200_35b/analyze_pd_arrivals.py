#!/usr/bin/env python3
"""Compute inter-arrival of requests on PD side (= encoder→PD handoff cadence).
For 4E we expect requests to arrive in groups of 4 from the 4 parallel encoders.
"""
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import statistics

PD_LOG = Path("/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01.log")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")

def iso_to_unix(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()

def main():
    raw = ANSI.sub("", PD_LOG.read_text())
    # Match dynamo "request received" at PD ingress — these are the encoder→PD handoff timestamps
    # Format: "request received [3mrequest_id[0m=02070af08aa44030..."
    rcv_re = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?request received.*?request_id=(?P<rid>[0-9a-f-]+)")
    cmp_re = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?request completed.*?request_id=(?P<rid>[0-9a-f-]+)")

    received = []
    completed = []
    for ln in raw.splitlines():
        m = rcv_re.search(ln)
        if m:
            received.append((iso_to_unix(m.group(1)), m.group("rid")))
            continue
        m = cmp_re.search(ln)
        if m:
            completed.append((iso_to_unix(m.group(1)), m.group("rid")))

    print(f"Total received={len(received)} completed={len(completed)}")
    print()

    windows = {
        0.10: (iso_to_unix("2026-05-25T23:23:41.000000Z"), iso_to_unix("2026-05-25T23:30:03.000000Z")),
        0.25: (iso_to_unix("2026-05-25T23:31:33.000000Z"), iso_to_unix("2026-05-25T23:34:18.000000Z")),
        0.50: (iso_to_unix("2026-05-25T23:35:48.000000Z"), iso_to_unix("2026-05-25T23:37:22.000000Z")),
        1.00: (iso_to_unix("2026-05-25T23:38:52.000000Z"), iso_to_unix("2026-05-25T23:39:54.000000Z")),
        1.50: (iso_to_unix("2026-05-25T23:41:24.000000Z"), iso_to_unix("2026-05-25T23:42:10.000000Z")),
        2.00: (iso_to_unix("2026-05-25T23:43:40.000000Z"), iso_to_unix("2026-05-25T23:44:22.000000Z")),
        3.00: (iso_to_unix("2026-05-25T23:45:52.000000Z"), iso_to_unix("2026-05-25T23:46:34.000000Z")),
    }

    print(f"{'Rate':>5} {'measRPS':>8} {'recvN':>5} {'cmplN':>5} "
          f"{'arrival_p50_s':>14} {'arrival_p90_s':>14} {'arrival_min_s':>14} "
          f"{'lifetime_p50_s':>15} {'PD_busy_total_s':>16} {'win_s':>7} "
          f"{'PD_idle_%':>10}")

    for rate, (s, e) in windows.items():
        bench = json.load(open(f"/hongming/res22_disagg_h200_35b_sweep/4img_768p_4E/rate_{rate}_np32/benchmark_output.json"))
        rps = bench["request_throughput"]
        bench_dur = bench["duration"]

        rcv_in = [t for t, _ in received if s <= t <= e + 5]
        cmp_in = [t for t, _ in completed if s <= t <= e + 5]
        rcv_in.sort(); cmp_in.sort()

        if len(rcv_in) < 2:
            continue
        # Inter-arrival on PD side
        arr = sorted([rcv_in[i+1] - rcv_in[i] for i in range(len(rcv_in)-1)])
        # Lifetime per-request: completed_ts - received_ts
        # Match by min count
        n = min(len(rcv_in), len(cmp_in))
        lifetimes = sorted([cmp_in[i] - rcv_in[i] for i in range(n) if cmp_in[i] >= rcv_in[i]])

        arr_p50 = arr[len(arr)//2]
        arr_p90 = arr[int(len(arr)*0.9)]
        arr_min = arr[0]
        life_p50 = lifetimes[len(lifetimes)//2] if lifetimes else float("nan")

        # PD busy = sum of intervals where PD was actively running ≥1 req
        # Approximation: union of [rcv_i, cmp_i] intervals
        intervals = sorted(zip(rcv_in[:n], cmp_in[:n]))
        merged = []
        for st, en in intervals:
            if not merged or st > merged[-1][1]:
                merged.append([st, en])
            else:
                merged[-1][1] = max(merged[-1][1], en)
        busy = sum(en - st for st, en in merged)
        win = e - s
        idle_pct = (1 - busy/win) * 100

        print(f"{rate:>5} {rps:>8.4f} {len(rcv_in):>5} {len(cmp_in):>5} "
              f"{arr_p50:>13.3f}s {arr_p90:>13.3f}s {arr_min:>13.3f}s "
              f"{life_p50:>14.3f}s {busy:>15.1f}s {win:>6.0f}s "
              f"{idle_pct:>9.1f}%")

    print()
    print("Notes:")
    print("  arrival_p50: time between consecutive request-received events on PD")
    print("    1/RPS would predict steady-state arrival; bursty arrivals show <<1/RPS")
    print("  lifetime_p50: time from PD receives request to PD completes it (E2E proxy)")
    print("  PD_busy_%: fraction of window when at least one request was being processed")

if __name__ == "__main__":
    main()

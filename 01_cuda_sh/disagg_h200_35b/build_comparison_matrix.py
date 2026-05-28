#!/usr/bin/env python3
"""Build comprehensive comparison across 5 topologies × 3 workloads × 7 rates for 35B."""
import json
import glob
import os

# Topology -> workload -> rate -> json path
sources = {
    "B70_1E":     "/hongming/res22_disagg_h200_35b_sweep/{wl}/rate_{rate}_np32/benchmark_output.json",
    "B70_4E":     "/hongming/res22_disagg_h200_35b_sweep/{wl}_4E/rate_{rate}_np32/benchmark_output.json",
    "dell06_1E":  "/hongming/res22_disagg_h200_35b_sweep/{wl}_dell06_1E/rate_{rate}_np32/benchmark_output.json",
    # agg sweeps have a different glob, find latest sweep dir
    "agg_TP1":    "/hongming/res20_agg_h200_35b/tp1/{wl}_np32_sweep_*/rate_{rate}/benchmark_output.json",
    "agg_TP2":    "/hongming/res20_agg_h200_35b/tp2/{wl}_np32_sweep_*/rate_{rate}/benchmark_output.json",
}
WORKLOADS = ["4img_768p", "8img_768p", "8img_1080p"]
RATES = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
# Some agg sweeps also have rate_4.0; collect that too for completeness
EXTRA_RATES = [4.0]

def load(path_pat, wl, rate):
    p = path_pat.format(wl=wl, rate=rate)
    if "*" in p:
        cands = glob.glob(p)
        if not cands:
            return None
        p = sorted(cands)[-1]
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except:
        return None

# Build a flat dict: (topology, workload, rate) -> dict of metrics
data = {}
for topo, pat in sources.items():
    for wl in WORKLOADS:
        for rate in RATES + EXTRA_RATES:
            d = load(pat, wl, rate)
            if d:
                data[(topo, wl, rate)] = d

# Print presence matrix
print("=== Presence matrix (topology × workload × rate) ===")
print(f"{'rate':>5}", end="")
for topo in sources.keys():
    for wl in WORKLOADS:
        print(f" {topo[:3]}_{wl[:8]:>8}", end="")
print()
for rate in RATES + EXTRA_RATES:
    print(f"{rate:>5}", end="")
    for topo in sources.keys():
        for wl in WORKLOADS:
            present = "Y" if (topo, wl, rate) in data else "."
            print(f"  {present:>11}", end="")
    print()

print()
print("=== Detailed metrics ===")
for wl in WORKLOADS:
    print(f"\n--- Workload: {wl} ---")
    print(f"{'rate':>5} {'topology':>11} {'RPS':>7} {'dur(s)':>7} {'E2E_p50':>9} {'TTFT_p50':>10} {'TPOT_p50':>10} {'ITL_p50':>9} {'tok/s':>9} {'OK':>4}")
    for rate in RATES + EXTRA_RATES:
        for topo in sources.keys():
            d = data.get((topo, wl, rate))
            if not d: continue
            print(f"{rate:>5} {topo:>11} "
                  f"{d.get('request_throughput',0):>7.4f} "
                  f"{d.get('duration',0):>6.0f}s "
                  f"{d.get('median_e2e_latency_ms',0)/1000:>7.2f}s "
                  f"{d.get('median_ttft_ms',0)/1000:>9.2f}s "
                  f"{d.get('median_tpot_ms',0):>9.2f}ms "
                  f"{d.get('median_itl_ms',0):>8.2f}ms "
                  f"{d.get('total_token_throughput',0):>8.0f} "
                  f"{d.get('completed',0):>4}")

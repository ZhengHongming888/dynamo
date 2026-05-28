#!/usr/bin/env python3
"""Comprehensive per-request time-breakdown for the patched + B70 4E sweep.

Per request, we extract:
  T0 = request received at PD
  T1 = process_embeddings starts (first embedding_transfer log line for this req)
  T2 = NIXL receive descriptor allocated (Created Descriptor cuda:0 from torch.Tensor)
  T3 = ReadOperation Created (NIXL READ submitted)
  T4 = Wire transfer DONE (encoder→PD wire complete)
  T5 = first Prefill batch starts after request
  T6 = ReqTimeStats (forward complete)
  T7 = request completed (response done, back to client)

From these we compute:
  pd_dispatch_ms        = T1 - T0
  cuda_alloc_ms         = T2 - T1   (allocate cuda:0 receive buffer + register with NIXL)
  nixl_setup_ms         = T3 - T2   (build descriptors, register, create handle)
  nixl_wire_ms          = T4 - T3   (the actual RoCE READ transfer)
  prep_to_prefill_ms    = T5 - T4   (PD scheduler enqueue + image processing)
  pd_forward_ms         = T6 - T5   (chunked-prefill + decode)
  egress_ms             = T7 - T6   (response stream wrap-up)
  total_lifetime_ms     = T7 - T0

The "encoder ViT time" is NOT directly measurable from PD logs alone, but we can estimate
it from inter-arrival of completions on PD side at saturation:
  encoder_vit_estimate = inter_completion_gap × encoder_count - PD_lifetime
  (because at saturation, gap = 1/RPS and encoder pool produces in parallel)
"""
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import statistics

PD_LOG = Path("/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs/pd_worker_giga01_h200_patched_debug_20260526_050419.log")

# Bench windows from sweep master log
WINDOWS = {
    0.10: ("05:43:02", "05:49:17"),
    0.25: ("05:50:47", "05:55:47"),
    0.50: ("05:57:17", "06:02:07"),
    1.00: ("06:03:37", "06:08:23"),
    1.50: ("06:09:53", "06:14:39"),
    2.00: ("06:16:09", "06:20:56"),
    3.00: ("06:22:26", "06:27:12"),
}

DAY = "2026-05-26T"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")

def iso_unix(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()

def hms_unix(hms):
    return iso_unix(f"{DAY}{hms}.000000Z")

# Patterns
RCV = re.compile(r'request received.*?request_id=([0-9a-f-]+)')
CMP = re.compile(r'request completed.*?request_id=([0-9a-f-]+)')
PROC_EMB = re.compile(r'Processing embeddings with shape')
DESC_CREATE_CUDA = re.compile(
    r'Descriptor: Created Descriptor\(ptr=0x[0-9a-f]+, size=(\d+), device=cuda:0\) from `torch\.Tensor`'
)
READ_CREATE = re.compile(
    r'Created ReadOperation\(operation_kind=READ.*?local_descriptors=ptr=0x[0-9a-f]+, '
    r'size=(\d+), device=cuda:0.*?remote_descriptors=ptr=0x[0-9a-f]+, '
    r'size=\d+, device=([^,]+),'
)
NIXL_DONE = re.compile(r"NIXL reported transfer state: DONE")
PREFILL = re.compile(
    r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), '
    r'.*?#running-req: (\d+), #queue-req: (\d+).*?cuda graph: (\w+)'
)
DECODE = re.compile(r'Decode batch, #running-req: (\d+),.*?gen throughput \(token/s\): ([\d.]+),.*?#queue-req: (\d+)')
RTS = re.compile(
    r'ReqTimeStats\(rid=([0-9a-f]+), input_len=(\d+), cached_input_len=(\d+), '
    r'output_len=(\d+).*?queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms,'
    r' entry_time=([\d.]+)'
)
QWENVL = re.compile(
    r"\[QwenVLProcessor Perf\] rid='([0-9a-f]+)', load_time: ([\d.]+) ms, "
    r"preprocess_time: ([\d.]+) ms, process_time: ([\d.]+) ms, "
    r"get_rope_index_time: ([\d.]+) ms, total_time: ([\d.]+) ms"
)

def main():
    print("Reading PD log...")
    raw = ANSI.sub("", PD_LOG.read_text())
    lines = raw.splitlines()
    print(f"  {len(lines):,} lines")

    # Build a stream of timestamped events
    events = []
    for ln in lines:
        m = TS.search(ln)
        if not m: continue
        ts = iso_unix(m.group(1))
        events.append((ts, ln))

    # Index events that we need by their patterns
    # Pass 1: build per-request lifecycle
    req_lifecycle = {}   # uuid -> dict of timestamps
    pending_req = None   # last-received UUID, to pin subsequent events to it
    rid_short_to_uuid = {}  # SGLang rid (no dashes) -> UUID

    descriptor_create_iter = []
    readop_create_iter = []
    nixl_done_iter = []
    prefill_events = []
    decode_events = []
    rts_events = []
    qwenvl_events = []

    for ts, ln in events:
        if "request received" in ln:
            m = RCV.search(ln)
            if m:
                uuid = m.group(1)
                req_lifecycle[uuid] = {"T0_received": ts}
                pending_req = uuid
        elif "request completed" in ln:
            m = CMP.search(ln)
            if m:
                uuid = m.group(1)
                if uuid in req_lifecycle:
                    req_lifecycle[uuid]["T7_completed"] = ts
        elif "Processing embeddings with shape" in ln:
            if pending_req and "T1_proc_emb" not in req_lifecycle.get(pending_req, {}):
                req_lifecycle[pending_req]["T1_proc_emb"] = ts
        elif "Descriptor: Created Descriptor" in ln and "device=cuda:0" in ln and "from `torch.Tensor`" in ln:
            m = DESC_CREATE_CUDA.search(ln)
            if m and pending_req and "T2_cuda_desc" not in req_lifecycle.get(pending_req, {}):
                req_lifecycle[pending_req]["T2_cuda_desc"] = ts
                req_lifecycle[pending_req]["embedding_size"] = int(m.group(1))
        elif "Created ReadOperation" in ln:
            m = READ_CREATE.search(ln)
            if m and pending_req and "T3_readop" not in req_lifecycle.get(pending_req, {}):
                req_lifecycle[pending_req]["T3_readop"] = ts
                req_lifecycle[pending_req]["remote_device"] = m.group(2)
        elif NIXL_DONE.search(ln):
            if pending_req and "T4_nixl_done" not in req_lifecycle.get(pending_req, {}):
                req_lifecycle[pending_req]["T4_nixl_done"] = ts
        elif "Prefill batch" in ln:
            m = PREFILL.search(ln)
            if m:
                prefill_events.append({
                    "ts": ts,
                    "seq": int(m.group(1)), "newtok": int(m.group(2)),
                    "cached": int(m.group(3)), "running": int(m.group(4)),
                    "queue": int(m.group(5)), "cuda_graph": m.group(6) == "True",
                })
        elif "Decode batch" in ln:
            m = DECODE.search(ln)
            if m:
                decode_events.append({
                    "ts": ts, "running": int(m.group(1)),
                    "tput": float(m.group(2)), "queue": int(m.group(3)),
                })
        elif "ReqTimeStats" in ln:
            m = RTS.search(ln)
            if m:
                rts_events.append({
                    "ts": ts, "rid": m.group(1),
                    "input_len": int(m.group(2)), "cached_input_len": int(m.group(3)),
                    "output_len": int(m.group(4)),
                    "q": float(m.group(5)), "f": float(m.group(6)),
                    "entry_time": float(m.group(7)),
                })
        elif "QwenVLProcessor Perf" in ln:
            m = QWENVL.search(ln)
            if m:
                qwenvl_events.append({
                    "ts": ts, "rid": m.group(1),
                    "load_time": float(m.group(2)),
                    "preprocess_time": float(m.group(3)),
                    "process_time": float(m.group(4)),
                    "get_rope_index_time": float(m.group(5)),
                    "total_time": float(m.group(6)),
                })

    print(f"  {len(req_lifecycle)} request UUIDs traced")
    print(f"  {len(prefill_events)} prefill events, {len(decode_events)} decode events")
    print(f"  {len(rts_events)} ReqTimeStats events")
    print(f"  {len(qwenvl_events)} QwenVLProcessor events")
    print()

    # Pair each req UUID with its T5_first_prefill (next prefill batch event after T0_received)
    # and its T6_rts (next ReqTimeStats event after T0).
    # Simpler approach: for each request, find smallest event ts > T0 of each type
    # But better: ReqTimeStats is logged immediately after forward completes, and the
    # 'request completed' event is logged a few ms later. We use the time-ordered
    # 'request completed' as T7, and ReqTimeStats with closest ts <= T7 as T6.
    #
    # The ReqTimeStats rid format is the SGLang internal rid (no dashes).
    # We have to match by time-locality rather than by rid.
    rts_by_ts = sorted(rts_events, key=lambda e: e["ts"])
    prefill_by_ts = sorted(prefill_events, key=lambda e: e["ts"])

    # For each lifecycle entry, find the first prefill batch event AFTER T4_nixl_done
    # (where the request's first chunked-prefill chunk lands), and the first
    # ReqTimeStats event AFTER T4 with input_len matching the embedding shape.
    for uuid, life in req_lifecycle.items():
        if "T4_nixl_done" not in life: continue
        t4 = life["T4_nixl_done"]
        # First prefill after t4
        for p in prefill_by_ts:
            if p["ts"] > t4 and "T5_first_prefill" not in life:
                life["T5_first_prefill"] = p["ts"]
                life["first_prefill_newtok"] = p["newtok"]
                life["first_prefill_cached"] = p["cached"]
                life["first_prefill_running"] = p["running"]
                break
        # First ReqTimeStats with ts in [t4, T7]
        t7 = life.get("T7_completed", t4 + 60)  # 60s ceiling
        for r in rts_by_ts:
            if t4 <= r["ts"] <= t7 + 0.5:
                life["T6_rts"] = r["ts"]
                life["rts_q_ms"] = r["q"]
                life["rts_f_ms"] = r["f"]
                life["rts_input_len"] = r["input_len"]
                life["rts_output_len"] = r["output_len"]
                break

    # Now per-rate aggregation
    def in_window(ts, s, e): return s <= ts <= e

    print("=" * 130)
    print(f"PER-RATE TIME BREAKDOWN (median per stage, ms)")
    print("=" * 130)
    print(f"{'Rate':>5} {'n':>4} {'RPS':>7} {'rcv→proc':>10} {'cuda_alloc':>11} {'nixl_setup':>11} "
          f"{'nixl_wire':>10} {'prep→pf':>9} {'pd_forward':>11} {'egress':>8} {'lifetime':>10} "
          f"{'rts_f':>9} {'rts_q':>9}")
    print(f"{'':>5} {'':>4} {'':>7} {'(T1-T0)':>10} {'(T2-T1)':>11} {'(T3-T2)':>11} "
          f"{'(T4-T3)':>10} {'(T5-T4)':>9} {'(T6-T5)':>11} {'(T7-T6)':>8} {'(T7-T0)':>10} "
          f"{'forward':>9} {'queue':>9}")
    print("-" * 130)

    for rate, (s_str, e_str) in WINDOWS.items():
        s = hms_unix(s_str)
        e = hms_unix(e_str)
        in_win = [(uuid, life) for uuid, life in req_lifecycle.items()
                  if "T0_received" in life and s <= life["T0_received"] <= e + 5]

        bench = json.load(open(f"/hongming/res22_disagg_h200_35b_sweep/8img_1080p_h200_patched_b70_4E/rate_{rate}_np32/benchmark_output.json"))
        rps = bench["request_throughput"]

        def med(arr):
            arr = sorted(x for x in arr if x is not None and x >= 0)
            return arr[len(arr)//2] if arr else float("nan")

        rcv_proc = []
        cuda_alloc = []
        nixl_setup = []
        nixl_wire = []
        prep_pf = []
        pd_fwd = []
        egress = []
        lifetime = []
        rts_f = []
        rts_q = []

        for uuid, life in in_win:
            if "T1_proc_emb" in life: rcv_proc.append((life["T1_proc_emb"]-life["T0_received"])*1000)
            if "T2_cuda_desc" in life and "T1_proc_emb" in life:
                cuda_alloc.append((life["T2_cuda_desc"]-life["T1_proc_emb"])*1000)
            if "T3_readop" in life and "T2_cuda_desc" in life:
                nixl_setup.append((life["T3_readop"]-life["T2_cuda_desc"])*1000)
            if "T4_nixl_done" in life and "T3_readop" in life:
                nixl_wire.append((life["T4_nixl_done"]-life["T3_readop"])*1000)
            if "T5_first_prefill" in life and "T4_nixl_done" in life:
                prep_pf.append((life["T5_first_prefill"]-life["T4_nixl_done"])*1000)
            if "T6_rts" in life and "T5_first_prefill" in life:
                pd_fwd.append((life["T6_rts"]-life["T5_first_prefill"])*1000)
            if "T7_completed" in life and "T6_rts" in life:
                egress.append((life["T7_completed"]-life["T6_rts"])*1000)
            if "T7_completed" in life and "T0_received" in life:
                lifetime.append((life["T7_completed"]-life["T0_received"])*1000)
            if "rts_f_ms" in life: rts_f.append(life["rts_f_ms"])
            if "rts_q_ms" in life: rts_q.append(life["rts_q_ms"])

        n = len(in_win)
        print(f"{rate:>5} {n:>4} {rps:>7.4f} {med(rcv_proc):>9.2f}  {med(cuda_alloc):>10.2f}  "
              f"{med(nixl_setup):>10.2f}  {med(nixl_wire):>9.2f}  {med(prep_pf):>8.2f}  "
              f"{med(pd_fwd):>10.1f}  {med(egress):>7.2f}  {med(lifetime):>9.1f}  "
              f"{med(rts_f):>8.1f}  {med(rts_q):>8.2f}")

    print()
    print("=" * 130)
    print(f"INTER-ARRIVAL ON PD (= encoder→PD cadence) and IMPLIED encoder ViT time")
    print("=" * 130)
    print(f"{'Rate':>5} {'RPS':>7} {'arr_p50':>9} {'arr_p99':>9} {'enc_pool':>9} {'enc_vit_per_req':>16}  notes")
    for rate, (s_str, e_str) in WINDOWS.items():
        s = hms_unix(s_str)
        e = hms_unix(e_str)
        arrivals = sorted([life["T0_received"] for uuid, life in req_lifecycle.items()
                           if "T0_received" in life and s <= life["T0_received"] <= e + 5])
        if len(arrivals) < 2: continue
        deltas = sorted([arrivals[i+1]-arrivals[i] for i in range(len(arrivals)-1)])
        bench = json.load(open(f"/hongming/res22_disagg_h200_35b_sweep/8img_1080p_h200_patched_b70_4E/rate_{rate}_np32/benchmark_output.json"))
        rps = bench["request_throughput"]
        # At sat: encoder_vit_per_req = arrival_p50 × encoder_count
        # (because 4 encoders produce in parallel, so per-encoder cycle = arrival × 4)
        encoder_count = 4  # B70 4E
        enc_vit_estimate = deltas[len(deltas)//2] * encoder_count
        is_sat = "SAT" if rate >= 0.5 else "    "
        print(f"{rate:>5} {rps:>7.4f} {deltas[len(deltas)//2]:>8.3f}s {deltas[-1]:>8.3f}s "
              f"{encoder_count:>8} {enc_vit_estimate:>15.2f}s  {is_sat} (gap×{encoder_count})")

    print()
    print("=" * 130)
    print(f"NIXL WIRE TRANSFER STATISTICS (post-patch)")
    print("=" * 130)
    wires = []
    for uuid, life in req_lifecycle.items():
        if "T4_nixl_done" in life and "T3_readop" in life:
            wires.append((life["T4_nixl_done"]-life["T3_readop"])*1000)
    wires.sort()
    if wires:
        n = len(wires)
        print(f"  count            : {n}")
        print(f"  min              : {wires[0]:.2f} ms")
        print(f"  p50              : {wires[n//2]:.2f} ms")
        print(f"  p99              : {wires[int(n*0.99)]:.2f} ms")
        print(f"  max              : {wires[-1]:.2f} ms")
        # compute effective bandwidth assuming 64 MB per transfer
        emb_size_mb = 66846720 / (1024*1024)
        med_wire_s = wires[n//2] / 1000
        print(f"  embedding size   : {emb_size_mb:.1f} MB")
        print(f"  effective BW p50 : {emb_size_mb / med_wire_s / 1000:.2f} GB/s")

    print()
    print("=" * 130)
    print(f"PD CONCURRENCY DURING SATURATION (rate ≥ 1.0)")
    print("=" * 130)
    from collections import Counter
    for rate in [1.0, 1.5, 2.0, 3.0]:
        s_str, e_str = WINDOWS[rate]
        s = hms_unix(s_str); e = hms_unix(e_str)
        dec = [d["running"] for d in decode_events if s <= d["ts"] <= e]
        pre = [p["running"] for p in prefill_events if s <= p["ts"] <= e]
        c_dec = Counter(dec); c_pre = Counter(pre)
        print(f"  rate={rate:>4}  decode running-req: {dict(sorted(c_dec.items()))}   "
              f"prefill running-req: {dict(sorted(c_pre.items()))}")

if __name__ == "__main__":
    main()

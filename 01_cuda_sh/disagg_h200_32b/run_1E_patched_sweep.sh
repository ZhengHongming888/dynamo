#!/bin/bash
# 1E patched cross-host sweep runner v3
# 3 workloads × 7 rates = 21 runs
# Changes from v2:
#   - np=16 (down from 32) to reduce B70 XPU memory pressure
#   - reordered: small→large (4img_768p → 8img_768p → 8img_1080p)
#   - 90s drain between runs (B70 XPU needs time to release memory)
#   - separate output dir np16/ to avoid mixing with prior np=32 result

set -u

MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8
BASE_URL=http://localhost:7001
RESULT_BASE=/hongming/res19_1E_patched_sweep
LOG_DIR=/hongming/dynamo/01_cuda_sh/disagg_h200_32b/logs
NP=16
RATES=(0.1 0.25 0.5 1.0 1.5 2.0 3.0)

mkdir -p "$RESULT_BASE" "$LOG_DIR"

# Smallest first to warm up the pipeline gracefully
WORKLOADS=(
  "4img_768p  4 1024x768"
  "8img_768p  8 1024x768"
  "8img_1080p 8 1920x1080"
)

PER_RUN_TIMEOUT=1800   # 30 min cap (np=16 ⇒ 16 reqs × ~27s = ~7 min for 1080p sat)
DRAIN_SECONDS=90       # wait after each run for B70 XPU to release memory
MAX_WARMUP_RETRIES=4

is_done () {
  local outfile=$1
  [ -s "$outfile" ] && python3 -c "
import json, sys
try:
    d = json.load(open('$outfile'))
    sys.exit(0 if d.get('completed', 0) > 0 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

run_one () {
  local wl_name=$1 imgs=$2 res=$3 rate=$4
  local outdir="$RESULT_BASE/$wl_name/rate_${rate}_np${NP}"
  local outfile="$outdir/benchmark_output.json"
  local logfile="$LOG_DIR/bench_1E_patched_${wl_name}_r${rate}_np${NP}.log"
  mkdir -p "$outdir"
  if is_done "$outfile"; then
    echo "[$(date -u +%H:%M:%S)] SKIP $wl_name rate=$rate np=$NP — already done"
    return 0
  fi
  echo "[$(date -u +%H:%M:%S)] >>> 1E $wl_name rate=$rate np=$NP"

  local attempt=0
  while [ $attempt -lt $MAX_WARMUP_RETRIES ]; do
    attempt=$((attempt+1))
    timeout $PER_RUN_TIMEOUT python3 -m sglang.bench_serving \
      --backend sglang-oai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --apply-chat-template \
      --dataset-name image \
      --image-count "$imgs" \
      --image-resolution "$res" \
      --image-format jpeg \
      --image-content random \
      --num-prompts $NP \
      --random-input-len 128 \
      --random-output-len 256 \
      --request-rate "$rate" \
      --warmup-requests 1 \
      --seed 0 \
      --output-file "$outfile" \
      > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && is_done "$outfile"; then
      echo "[$(date -u +%H:%M:%S)]   OK (attempt $attempt)"
      grep -E "Successful requests:|Request throughput \(req/s\):|Mean E2E|Median TTFT|Median TPOT|Total token throughput" "$logfile" 2>/dev/null | head -6 | sed 's/^/    /'
      return 0
    fi
    if grep -q "Warmup failed\|UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY\|level_zero" "$logfile" 2>/dev/null; then
      echo "[$(date -u +%H:%M:%S)]   Warmup/encoder transient (attempt $attempt) — drain 90s and retry"
      sleep 90
      continue
    fi
    echo "[$(date -u +%H:%M:%S)]   FAILED rc=$rc (attempt $attempt) — see $logfile"
    if [ $rc -eq 124 ]; then
      echo "[$(date -u +%H:%M:%S)]   timeout — workload saturated, accepting partial result"
      return 0
    fi
    sleep 30
  done
  echo "[$(date -u +%H:%M:%S)]   GIVE UP after $MAX_WARMUP_RETRIES attempts"
  return 1
}

echo "==== 1E patched sweep v3 (np=$NP, small→large, 90s drain) starting at $(date -u) ===="
echo

for wl in "${WORKLOADS[@]}"; do
  read -r wl_name imgs res <<<"$wl"
  echo "==== workload: $wl_name (imgs=$imgs res=$res) ===="
  for rate in "${RATES[@]}"; do
    run_one "$wl_name" "$imgs" "$res" "$rate"
    echo "[$(date -u +%H:%M:%S)]   draining ${DRAIN_SECONDS}s..."
    sleep $DRAIN_SECONDS
  done
  echo
done

echo "==== 1E patched sweep v3 done at $(date -u) ===="

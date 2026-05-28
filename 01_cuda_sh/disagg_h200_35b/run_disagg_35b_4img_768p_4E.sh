#!/bin/bash
# Disagg 35B 4E sweep — 4img/768p only, all 7 rates, np=32.
# Companion to run_disagg_35b_8img_1080p_4E.sh; results to 4img_768p_4E/

set -u

MODEL=/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
BASE_URL=http://localhost:7001
RESULT_BASE=/hongming/res22_disagg_h200_35b_sweep/4img_768p_4E
LOG_DIR=/hongming/dynamo/01_cuda_sh/disagg_h200_35b/logs
NP=32
RATES=(0.1 0.25 0.5 1.0 1.5 2.0 3.0)

mkdir -p "$RESULT_BASE" "$LOG_DIR"

PER_RUN_TIMEOUT=2700
DRAIN_SECONDS=90
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
  local rate=$1
  local outdir="$RESULT_BASE/rate_${rate}_np${NP}"
  local outfile="$outdir/benchmark_output.json"
  local logfile="$LOG_DIR/bench_disagg_35b_4E_4img_768p_r${rate}_np${NP}.log"
  mkdir -p "$outdir"
  if is_done "$outfile"; then
    echo "[$(date -u +%H:%M:%S)] SKIP rate=$rate np=$NP — already done"
    return 0
  fi
  echo "[$(date -u +%H:%M:%S)] >>> 4E disagg 35B 4img/768p rate=$rate np=$NP"

  local attempt=0
  while [ $attempt -lt $MAX_WARMUP_RETRIES ]; do
    attempt=$((attempt+1))
    timeout $PER_RUN_TIMEOUT python3 -m sglang.bench_serving \
      --backend sglang-oai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --apply-chat-template \
      --dataset-name image \
      --image-count 4 \
      --image-resolution 1024x768 \
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
      grep -E "Successful requests:|Request throughput \(req/s\):|Mean E2E|Median TTFT|Median TPOT|Total token throughput" "$logfile" 2>/dev/null | head -8 | sed 's/^/    /'
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

echo "==== 4E disagg 35B 4img/768p sweep starting at $(date -u) ===="
echo "Result base: $RESULT_BASE"
echo

for rate in "${RATES[@]}"; do
  run_one "$rate"
  echo "[$(date -u +%H:%M:%S)]   draining ${DRAIN_SECONDS}s..."
  sleep $DRAIN_SECONDS
done

echo "==== 4E disagg 35B 4img/768p sweep done at $(date -u) ===="

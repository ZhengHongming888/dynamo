#!/bin/bash
# Run multiple rate tests for SGLang benchmark - 32B FP8 model with 8 images at 1080p
# Uses dynamic num_prompts per rate for better measurement
# Usage: ./test_sglang_mult_rates_32b_1080p_np_over_rates.sh [rates...]
# Example: ./test_sglang_mult_rates_32b_1080p_np_over_rates.sh 0.1 0.2 0.3
#          ./test_sglang_mult_rates_32b_1080p_np_over_rates.sh  # uses default rates

MODEL="/mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8"
SEED=0
PORT=7001
INPUT_LEN=128
OUTPUT_LEN=256
NUM_IMAGES=8
IMAGE_RESOLUTION="1920x1080"
#IMAGE_RESOLUTION="1024x768"

# Default rates for 32B model with 8 images at 1080p
# Fixed num_prompts=64 for ALL rates (apples-to-apples sweep)
DEFAULT_RATES=(0.1 0.25 0.5 1.0 1.25)
FIXED_NUM_PROMPTS=32
declare -A NUM_PROMPTS_MAP
NUM_PROMPTS_MAP[0.1]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[0.2]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[0.25]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[0.4]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[0.5]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[0.6]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[0.8]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[1.0]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[1.2]=$FIXED_NUM_PROMPTS
NUM_PROMPTS_MAP[1.25]=$FIXED_NUM_PROMPTS

# Use provided rates or default
if [ $# -eq 0 ]; then
    RATES=("${DEFAULT_RATES[@]}")
else
    RATES=("$@")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# RESULT_BASE can be overridden via env to keep each config's results separate
RESULT_BASE="${RESULT_BASE:-/hongming/res10_pd_tp2_rdma/h200_disagg_pdtp2_rdma_32b_image8_1080p_np64_rates}"
TEST_DIR="${RESULT_BASE}/test_sglang_multi_rates_1080p_${TIMESTAMP}"
mkdir -p "$TEST_DIR"

echo "=========================================="
echo "SGLang Multi-Rate Benchmark - 32B FP8 (1080p)"
echo "=========================================="
echo "Model:       $MODEL"
echo "Port:        $PORT"
echo "Images:      $NUM_IMAGES (8 images per request)"
echo "Resolution:  $IMAGE_RESOLUTION"
echo "Input Len:   $INPUT_LEN"
echo "Output Len:  $OUTPUT_LEN"
echo "Rates:       ${RATES[*]}"
echo "Num Prompts: $FIXED_NUM_PROMPTS (fixed for all rates)"
echo "Test Dir:    $TEST_DIR"
echo "=========================================="
echo ""

# Check if server is ready
echo "Checking if server is ready..."
HEALTH_CHECK_SUCCESS=0
for i in {1..5}; do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "✓ Server is ready (attempt $i)"
        HEALTH_CHECK_SUCCESS=1
        break
    fi
    echo "  Server check attempt $i failed, retrying..."
    sleep 1
done

if [ $HEALTH_CHECK_SUCCESS -eq 0 ]; then
    echo "❌ Server is not responding at http://localhost:${PORT}"
    echo "Please start the server first"
    exit 1
fi

# Save server info
curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool > "$TEST_DIR/server_info.json" 2>&1

# Create results CSV file with header
RESULTS_CSV="$TEST_DIR/results_summary.csv"
echo "target_rate,actual_rps,successful_requests,duration_s,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,median_itl_ms,p95_itl_ms,p99_itl_ms,max_itl_ms,mean_e2e_ms,median_e2e_ms,p90_e2e_ms,p99_e2e_ms,input_throughput_toks,output_throughput_toks,peak_output_throughput_toks,concurrency" > "$RESULTS_CSV"

# Arrays to store results for table display
declare -a TARGET_RATES
declare -a ACTUAL_RPS
declare -a MEAN_TTFT
declare -a MEDIAN_TTFT
declare -a MEAN_TPOT
declare -a MEDIAN_TPOT
declare -a MEAN_E2E
declare -a MEDIAN_E2E
declare -a P99_E2E
declare -a SUCCESS_COUNT

# Run benchmark for each rate
RATE_COUNT=0
for RATE in "${RATES[@]}"; do
    # Determine num_prompts for this rate
    if [ -n "${NUM_PROMPTS_MAP[$RATE]}" ]; then
        NUM_PROMPTS=${NUM_PROMPTS_MAP[$RATE]}
    else
        # Default for custom rates not in map
        NUM_PROMPTS=32
    fi

    echo ""
    echo "=========================================="
    echo "Testing Rate: $RATE RPS [$(($RATE_COUNT + 1))/${#RATES[@]}]"
    echo "Num Prompts: $NUM_PROMPTS"
    echo "=========================================="

    RATE_DIR="$TEST_DIR/rate_${RATE}"
    mkdir -p "$RATE_DIR"

    # Run warmup (use 5 prompts regardless of rate)
    echo "Running warmup..."
    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --base-url http://localhost:${PORT} \
        --model "$MODEL" \
        --dataset-name image \
        --image-count $NUM_IMAGES \
        --image-resolution $IMAGE_RESOLUTION \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts 5 \
        --request-rate 0.1 \
        --seed $SEED \
        > "$RATE_DIR/warmup.log" 2>&1

    sleep 2

    # Run actual benchmark
    echo "Running benchmark at $RATE RPS..."
    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --base-url http://localhost:${PORT} \
        --model "$MODEL" \
        --dataset-name image \
        --image-count $NUM_IMAGES \
        --image-resolution $IMAGE_RESOLUTION \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --request-rate $RATE \
        --seed $SEED \
        --output-file "$RATE_DIR/benchmark_output.json" \
        > "$RATE_DIR/results.txt" 2>&1

    BENCHMARK_EXIT=$?

    if [ $BENCHMARK_EXIT -eq 0 ]; then
        echo "✓ Benchmark completed successfully"

        # Extract metrics from results
        ACTUAL_RPS_VAL=$(grep "Request throughput (req/s):" "$RATE_DIR/results.txt" | awk '{print $4}')
        SUCCESS_VAL=$(grep "Successful requests:" "$RATE_DIR/results.txt" | awk '{print $3}')
        DURATION_VAL=$(grep "Benchmark duration (s):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MEAN_TTFT_VAL=$(grep "Mean TTFT (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MEDIAN_TTFT_VAL=$(grep "Median TTFT (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        P99_TTFT_VAL=$(grep "P99 TTFT (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MEAN_TPOT_VAL=$(grep "Mean TPOT (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MEDIAN_TPOT_VAL=$(grep "Median TPOT (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        P99_TPOT_VAL=$(grep "P99 TPOT (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MEAN_ITL_VAL=$(grep "Mean ITL (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MEDIAN_ITL_VAL=$(grep "Median ITL (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        P95_ITL_VAL=$(grep "P95 ITL (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        P99_ITL_VAL=$(grep "P99 ITL (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MAX_ITL_VAL=$(grep "Max ITL (ms):" "$RATE_DIR/results.txt" | awk '{print $4}')
        MEAN_E2E_VAL=$(grep "Mean E2E Latency (ms):" "$RATE_DIR/results.txt" | awk '{print $5}')
        MEDIAN_E2E_VAL=$(grep "Median E2E Latency (ms):" "$RATE_DIR/results.txt" | awk '{print $5}')
        P90_E2E_VAL=$(grep "P90 E2E Latency (ms):" "$RATE_DIR/results.txt" | awk '{print $5}')
        P99_E2E_VAL=$(grep "P99 E2E Latency (ms):" "$RATE_DIR/results.txt" | awk '{print $5}')
        INPUT_THROUGHPUT_VAL=$(grep "Input token throughput" "$RATE_DIR/results.txt" | awk '{print $5}')
        OUTPUT_THROUGHPUT_VAL=$(grep "Output token throughput" "$RATE_DIR/results.txt" | awk '{print $5}')
        PEAK_OUTPUT_THROUGHPUT_VAL=$(grep "Peak output token throughput" "$RATE_DIR/results.txt" | awk '{print $6}')
        CONCURRENCY_VAL=$(grep "Concurrency:" "$RATE_DIR/results.txt" | awk '{print $2}')

        # Store in arrays
        TARGET_RATES[$RATE_COUNT]=$RATE
        ACTUAL_RPS[$RATE_COUNT]=$ACTUAL_RPS_VAL
        MEAN_TTFT[$RATE_COUNT]=$MEAN_TTFT_VAL
        MEDIAN_TTFT[$RATE_COUNT]=$MEDIAN_TTFT_VAL
        MEAN_TPOT[$RATE_COUNT]=$MEAN_TPOT_VAL
        MEDIAN_TPOT[$RATE_COUNT]=$MEDIAN_TPOT_VAL
        MEAN_E2E[$RATE_COUNT]=$MEAN_E2E_VAL
        MEDIAN_E2E[$RATE_COUNT]=$MEDIAN_E2E_VAL
        P99_E2E[$RATE_COUNT]=$P99_E2E_VAL
        SUCCESS_COUNT[$RATE_COUNT]=$SUCCESS_VAL

        # Append to CSV
        echo "$RATE,$ACTUAL_RPS_VAL,$SUCCESS_VAL,$DURATION_VAL,$MEAN_TTFT_VAL,$MEDIAN_TTFT_VAL,$P99_TTFT_VAL,$MEAN_TPOT_VAL,$MEDIAN_TPOT_VAL,$P99_TPOT_VAL,$MEAN_ITL_VAL,$MEDIAN_ITL_VAL,$P95_ITL_VAL,$P99_ITL_VAL,$MAX_ITL_VAL,$MEAN_E2E_VAL,$MEDIAN_E2E_VAL,$P90_E2E_VAL,$P99_E2E_VAL,$INPUT_THROUGHPUT_VAL,$OUTPUT_THROUGHPUT_VAL,$PEAK_OUTPUT_THROUGHPUT_VAL,$CONCURRENCY_VAL" >> "$RESULTS_CSV"

        echo "  Actual RPS: $ACTUAL_RPS_VAL"
        echo "  Mean TTFT: $MEAN_TTFT_VAL ms"
        echo "  Mean TPOT: $MEAN_TPOT_VAL ms"
        echo "  Mean E2E: $MEAN_E2E_VAL ms"
    else
        echo "⚠ Benchmark failed (exit code: $BENCHMARK_EXIT)"
        echo "$RATE,FAILED,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
    fi

    RATE_COUNT=$((RATE_COUNT + 1))

    # Small delay between tests
    sleep 3
done

# Generate summary table
echo ""
echo ""
echo "=========================================="
echo "BENCHMARK SUMMARY - ALL RATES (32B FP8 1080p)"
echo "=========================================="
echo ""

# Main metrics table
echo "=== Request Throughput & Latency ==="
printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
    "Target_RPS" "Actual_RPS" "Success" "Mean_E2E" "Median_E2E" "P99_E2E" "Concurrency"
printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
    "(req/s)" "(req/s)" "(count)" "(ms)" "(ms)" "(ms)" ""
echo "--------------------------------------------------------------------------------------------"

for i in "${!TARGET_RATES[@]}"; do
    RATE_VAL=$(grep "^${TARGET_RATES[$i]}," "$RESULTS_CSV" | tail -1)
    if [ ! -z "$RATE_VAL" ]; then
        TARGET=$(echo "$RATE_VAL" | cut -d',' -f1)
        ACTUAL=$(echo "$RATE_VAL" | cut -d',' -f2)
        SUCCESS=$(echo "$RATE_VAL" | cut -d',' -f3)
        MEAN_E2E_V=$(echo "$RATE_VAL" | cut -d',' -f11)
        MEDIAN_E2E_V=$(echo "$RATE_VAL" | cut -d',' -f12)
        P99_E2E_V=$(echo "$RATE_VAL" | cut -d',' -f14)
        CONCURRENCY=$(echo "$RATE_VAL" | cut -d',' -f18)

        printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
            "$TARGET" "$ACTUAL" "$SUCCESS" "$MEAN_E2E_V" "$MEDIAN_E2E_V" "$P99_E2E_V" "$CONCURRENCY"
    fi
done

echo ""
echo "=== Time to First Token (TTFT) ==="
printf "%-12s %-12s %-12s %-12s\n" \
    "Target_RPS" "Mean_TTFT" "Median_TTFT" "P99_TTFT"
printf "%-12s %-12s %-12s %-12s\n" \
    "(req/s)" "(ms)" "(ms)" "(ms)"
echo "----------------------------------------------------"

for i in "${!TARGET_RATES[@]}"; do
    RATE_VAL=$(grep "^${TARGET_RATES[$i]}," "$RESULTS_CSV" | tail -1)
    if [ ! -z "$RATE_VAL" ]; then
        TARGET=$(echo "$RATE_VAL" | cut -d',' -f1)
        MEAN_TTFT_V=$(echo "$RATE_VAL" | cut -d',' -f5)
        MEDIAN_TTFT_V=$(echo "$RATE_VAL" | cut -d',' -f6)
        P99_TTFT_V=$(echo "$RATE_VAL" | cut -d',' -f7)

        printf "%-12s %-12s %-12s %-12s\n" \
            "$TARGET" "$MEAN_TTFT_V" "$MEDIAN_TTFT_V" "$P99_TTFT_V"
    fi
done

echo ""
echo "=== Time per Output Token (TPOT) ==="
printf "%-12s %-12s %-12s %-12s\n" \
    "Target_RPS" "Mean_TPOT" "Median_TPOT" "P99_TPOT"
printf "%-12s %-12s %-12s %-12s\n" \
    "(req/s)" "(ms)" "(ms)" "(ms)"
echo "----------------------------------------------------"

for i in "${!TARGET_RATES[@]}"; do
    RATE_VAL=$(grep "^${TARGET_RATES[$i]}," "$RESULTS_CSV" | tail -1)
    if [ ! -z "$RATE_VAL" ]; then
        TARGET=$(echo "$RATE_VAL" | cut -d',' -f1)
        MEAN_TPOT_V=$(echo "$RATE_VAL" | cut -d',' -f8)
        MEDIAN_TPOT_V=$(echo "$RATE_VAL" | cut -d',' -f9)
        P99_TPOT_V=$(echo "$RATE_VAL" | cut -d',' -f10)

        printf "%-12s %-12s %-12s %-12s\n" \
            "$TARGET" "$MEAN_TPOT_V" "$MEDIAN_TPOT_V" "$P99_TPOT_V"
    fi
done

echo ""
echo "=== Token Throughput ==="
printf "%-12s %-15s %-15s %-15s\n" \
    "Target_RPS" "Input_tok/s" "Output_tok/s" "Peak_Out_tok/s"
echo "------------------------------------------------------------"

for i in "${!TARGET_RATES[@]}"; do
    RATE_VAL=$(grep "^${TARGET_RATES[$i]}," "$RESULTS_CSV" | tail -1)
    if [ ! -z "$RATE_VAL" ]; then
        TARGET=$(echo "$RATE_VAL" | cut -d',' -f1)
        INPUT_TP=$(echo "$RATE_VAL" | cut -d',' -f15)
        OUTPUT_TP=$(echo "$RATE_VAL" | cut -d',' -f16)
        PEAK_TP=$(echo "$RATE_VAL" | cut -d',' -f17)

        printf "%-12s %-15s %-15s %-15s\n" \
            "$TARGET" "$INPUT_TP" "$OUTPUT_TP" "$PEAK_TP"
    fi
done

echo ""
echo "=========================================="
echo "Summary saved to: $TEST_DIR"
echo "CSV file: $RESULTS_CSV"
echo "=========================================="
echo ""

# Also save the table output to a file
{
    echo "BENCHMARK SUMMARY - ALL RATES (32B FP8 1080p)"
    echo "Generated: $(date)"
    echo "Configuration: 8 images at 1024x768, dynamic prompts (32@0.1, 64@0.25, 128@0.5, 256@1.0 QPS)"
    echo ""
    echo "=== Request Throughput & Latency ==="
    printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
        "Target_RPS" "Actual_RPS" "Success" "Mean_E2E" "Median_E2E" "P99_E2E" "Concurrency"
    echo "--------------------------------------------------------------------------------------------"
    for i in "${!TARGET_RATES[@]}"; do
        RATE_VAL=$(grep "^${TARGET_RATES[$i]}," "$RESULTS_CSV" | tail -1)
        if [ ! -z "$RATE_VAL" ]; then
            TARGET=$(echo "$RATE_VAL" | cut -d',' -f1)
            ACTUAL=$(echo "$RATE_VAL" | cut -d',' -f2)
            SUCCESS=$(echo "$RATE_VAL" | cut -d',' -f3)
            MEAN_E2E_V=$(echo "$RATE_VAL" | cut -d',' -f11)
            MEDIAN_E2E_V=$(echo "$RATE_VAL" | cut -d',' -f12)
            P99_E2E_V=$(echo "$RATE_VAL" | cut -d',' -f14)
            CONCURRENCY=$(echo "$RATE_VAL" | cut -d',' -f18)
            printf "%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
                "$TARGET" "$ACTUAL" "$SUCCESS" "$MEAN_E2E_V" "$MEDIAN_E2E_V" "$P99_E2E_V" "$CONCURRENCY"
        fi
    done
    echo ""
    echo "=== Time to First Token (TTFT) ==="
    printf "%-12s %-12s %-12s %-12s\n" "Target_RPS" "Mean_TTFT" "Median_TTFT" "P99_TTFT"
    echo "----------------------------------------------------"
    for i in "${!TARGET_RATES[@]}"; do
        RATE_VAL=$(grep "^${TARGET_RATES[$i]}," "$RESULTS_CSV" | tail -1)
        if [ ! -z "$RATE_VAL" ]; then
            TARGET=$(echo "$RATE_VAL" | cut -d',' -f1)
            MEAN_TTFT_V=$(echo "$RATE_VAL" | cut -d',' -f5)
            MEDIAN_TTFT_V=$(echo "$RATE_VAL" | cut -d',' -f6)
            P99_TTFT_V=$(echo "$RATE_VAL" | cut -d',' -f7)
            printf "%-12s %-12s %-12s %-12s\n" "$TARGET" "$MEAN_TTFT_V" "$MEDIAN_TTFT_V" "$P99_TTFT_V"
        fi
    done
    echo ""
    echo "=== Time per Output Token (TPOT) ==="
    printf "%-12s %-12s %-12s %-12s\n" "Target_RPS" "Mean_TPOT" "Median_TPOT" "P99_TPOT"
    echo "----------------------------------------------------"
    for i in "${!TARGET_RATES[@]}"; do
        RATE_VAL=$(grep "^${TARGET_RATES[$i]}," "$RESULTS_CSV" | tail -1)
        if [ ! -z "$RATE_VAL" ]; then
            TARGET=$(echo "$RATE_VAL" | cut -d',' -f1)
            MEAN_TPOT_V=$(echo "$RATE_VAL" | cut -d',' -f8)
            MEDIAN_TPOT_V=$(echo "$RATE_VAL" | cut -d',' -f9)
            P99_TPOT_V=$(echo "$RATE_VAL" | cut -d',' -f10)
            printf "%-12s %-12s %-12s %-12s\n" "$TARGET" "$MEAN_TPOT_V" "$MEDIAN_TPOT_V" "$P99_TPOT_V"
        fi
    done
} > "$TEST_DIR/summary_table.txt"

echo "✓ All tests complete!"
echo "Summary table saved to: $TEST_DIR/summary_table.txt"

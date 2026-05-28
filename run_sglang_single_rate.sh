#!/bin/bash
# Run a single rate test for SGLang benchmark (client only)
# Usage: ./run_sglang_single_rate.sh <rate>
# Note: Server must be running separately before executing this script

RATE=$1
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
NUM_PROMPTS=32
SEED=0
PORT=7001
INPUT_LEN=128
OUTPUT_LEN=128
NUM_IMAGES=8
IMAGE_RESOLUTION="640x480"

if [ -z "$RATE" ]; then
    echo "Usage: $0 <rate>"
    echo "Example: $0 0.5"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_DIR="/workspace/test_sglang_rate${RATE}_${TIMESTAMP}"
mkdir -p "$TEST_DIR"

echo "=========================================="
echo "SGLang Benchmark @ Rate $RATE RPS"
echo "=========================================="
echo "Model:       $MODEL"
echo "Port:        $PORT"
echo "Images:      $NUM_IMAGES"
echo "Resolution:  $IMAGE_RESOLUTION"
echo "Input Len:   $INPUT_LEN"
echo "Output Len:  $OUTPUT_LEN"
echo "Num Prompts: $NUM_PROMPTS"
echo "Test Dir:    $TEST_DIR"
echo "=========================================="

# Quick health check
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
    echo "Please start the server first (e.g., using multimodal_epd_xpu.sh)"
    exit 1
fi

# Get server info
curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool > "$TEST_DIR/server_info.json" 2>&1
echo "Server info saved to $TEST_DIR/server_info.json"

# Warmup phase
echo ""
echo "Running warmup (5 prompts at 0.5 RPS)..."
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
    --request-rate 0.5 \
    --seed $SEED \
    > "$TEST_DIR/warmup.log" 2>&1

WARMUP_EXIT=$?
if [ $WARMUP_EXIT -eq 0 ]; then
    echo "✓ Warmup completed successfully"
else
    echo "⚠ Warmup had issues (exit code: $WARMUP_EXIT)"
    echo "Warmup log (last 30 lines):"
    tail -30 "$TEST_DIR/warmup.log"
fi

sleep 2

# Run actual benchmark
echo ""
echo "Running benchmark at $RATE RPS with $NUM_PROMPTS prompts..."
echo "(Progress bar will show real-time status)"
echo ""
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
    --output-file "$TEST_DIR/benchmark_output.json" \
    2>&1 | tee "$TEST_DIR/results.txt"

BENCHMARK_EXIT_CODE=$?

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "Status: ✓ SUCCESS"
else
    echo "Status: ⚠ COMPLETED WITH ISSUES (exit code: $BENCHMARK_EXIT_CODE)"
fi
echo "Test Directory: $TEST_DIR"
echo "Files created:"
echo "  - warmup.log            : Warmup phase logs"
echo "  - results.txt           : Benchmark results"
echo "  - benchmark_output.json : Detailed benchmark metrics (JSON)"
echo "  - server_info.json      : Server model information"
echo ""

# Show quick results summary if available
if [ -f "$TEST_DIR/results.txt" ]; then
    echo "Quick Results:"
    grep -E "Successful requests|Request throughput|Input token throughput|Output token throughput|Mean E2E Latency|Median E2E Latency|Mean TTFT|Mean TPOT" "$TEST_DIR/results.txt" || true
fi

echo ""
if [ -f "$TEST_DIR/benchmark_output.json" ]; then
    echo "Detailed metrics saved to: $TEST_DIR/benchmark_output.json"
fi

echo ""
echo "✓ Test complete: $TEST_DIR"

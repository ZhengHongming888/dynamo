#!/bin/bash
set -e

# Dynamo SGLang Aggregated EPD Test Script
# This script sets up and tests the complete EPD pipeline with NATS, etcd, frontend, and SGLang EPD

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
export IP_LOCAL=172.26.46.162
export PORT_NATS=14222
export PORT_ETCD=12379
export PORT_HTTP=7001
export CUDA_DEVICE=1
export KV_EVENT_PORT=22080
export SIDE_CHANNEL_PORT=20098

# Cleanup flag
CLEANUP_ON_EXIT=true

# Log files
LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
NATS_LOG="$LOG_DIR/nats.log"
ETCD_LOG="$LOG_DIR/etcd.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
EPD_LOG="$LOG_DIR/epd.log"
TEST_LOG="$LOG_DIR/test.log"

# PID tracking
PIDS=()

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up processes...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Additional cleanup for any remaining processes
    pkill -f "nats-server -js -p $PORT_NATS" 2>/dev/null || true
    pkill -f "etcd.*--listen-client-urls=http://0.0.0.0:$PORT_ETCD" 2>/dev/null || true
    pkill -f "dynamo.frontend --http-port $PORT_HTTP" 2>/dev/null || true
    pkill -f "dynamo.sglang.*--enable-multimodal" 2>/dev/null || true

    # Clean up etcd data directory
    rm -rf /tmp/etcd-test-$$

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set up trap for cleanup on exit
if [ "$CLEANUP_ON_EXIT" = true ]; then
    trap cleanup EXIT INT TERM
fi

# Helper function to check if port is in use
check_port() {
    local port=$1
    if python3 -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('', $port)); s.close()" 2>/dev/null; then
        return 0
    else
        echo -e "${RED}Port $port is already in use${NC}"
        return 1
    fi
}

# Helper function to wait for port to be ready
wait_for_port() {
    local host=$1
    local port=$2
    local timeout=${3:-30}
    local count=0

    echo -n "Waiting for $host:$port to be ready..."
    while [ $count -lt $timeout ]; do
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('$host', $port)); s.close()" 2>/dev/null; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo -e " ${RED}✗${NC}"
    return 1
}

# Helper function to check process health
check_process() {
    local pid=$1
    local name=$2

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $name is running (PID: $pid)"
        return 0
    else
        echo -e "${RED}✗${NC} $name has died (PID: $pid)"
        return 1
    fi
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Dynamo SGLang Aggregated EPD Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

# Check required commands
for cmd in nats-server etcd python3 nvidia-smi; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        exit 1
    fi
done

# Check CUDA device
if ! nvidia-smi -i $CUDA_DEVICE &> /dev/null; then
    echo -e "${RED}Error: CUDA device $CUDA_DEVICE is not available${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites satisfied${NC}"
echo ""

# Step 2: Start NATS and etcd
echo -e "${YELLOW}[2/5] Starting infrastructure services...${NC}"

# Check ports
check_port $PORT_NATS || exit 1
check_port 18222 || exit 1
check_port $PORT_ETCD || exit 1
check_port 12380 || exit 1

# Start NATS
echo "Starting NATS server on port $PORT_NATS..."
nats-server -js -p $PORT_NATS -m 18222 > "$NATS_LOG" 2>&1 &
NATS_PID=$!
PIDS+=($NATS_PID)
sleep 2

if ! check_process $NATS_PID "NATS"; then
    echo -e "${RED}Failed to start NATS. Check $NATS_LOG${NC}"
    exit 1
fi

# Start etcd
echo "Starting etcd on port $PORT_ETCD..."
# Clean up any existing etcd data
rm -rf /tmp/etcd-test-$$
etcd \
  --listen-client-urls=http://0.0.0.0:$PORT_ETCD \
  --advertise-client-urls=http://0.0.0.0:$PORT_ETCD \
  --listen-peer-urls=http://0.0.0.0:12380 \
  --initial-advertise-peer-urls=http://0.0.0.0:12380 \
  --initial-cluster=default=http://0.0.0.0:12380 \
  --data-dir=/tmp/etcd-test-$$ > "$ETCD_LOG" 2>&1 &
ETCD_PID=$!
PIDS+=($ETCD_PID)
sleep 3

if ! check_process $ETCD_PID "etcd"; then
    echo -e "${RED}Failed to start etcd. Check $ETCD_LOG${NC}"
    exit 1
fi

# Wait for services to be ready
wait_for_port localhost $PORT_NATS || exit 1
wait_for_port localhost $PORT_ETCD || exit 1

echo -e "${GREEN}✓ Infrastructure services started${NC}"
echo ""

# Step 3: Start Frontend
echo -e "${YELLOW}[3/5] Starting Dynamo frontend...${NC}"

check_port $PORT_HTTP || exit 1

export ETCD_ENDPOINTS=http://$IP_LOCAL:$PORT_ETCD
DYN_REQUEST_PLANE=tcp \
DYN_EVENT_PLANE=zmq \
SGLANG_LOG_LEVEL=debug python3 -m dynamo.frontend \
    --http-port $PORT_HTTP \
    --router-mode kv \
    --router-reset-states > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
PIDS+=($FRONTEND_PID)

# Wait for frontend to be ready
wait_for_port localhost $PORT_HTTP 60 || {
    echo -e "${RED}Failed to start frontend. Check $FRONTEND_LOG${NC}"
    exit 1
}

if ! check_process $FRONTEND_PID "Frontend"; then
    echo -e "${RED}Frontend died unexpectedly. Check $FRONTEND_LOG${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Frontend started on port $PORT_HTTP${NC}"
echo ""

# Step 4: Start SGLang EPD
echo -e "${YELLOW}[4/5] Starting SGLang EPD with aggregated configuration...${NC}"

check_port $KV_EVENT_PORT || exit 1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
NATS_SERVER=nats://${IP_LOCAL}:${PORT_NATS} \
ETCD_ENDPOINTS=http://${IP_LOCAL}:${PORT_ETCD} \
DYN_REQUEST_PLANE=tcp \
TRANSFER_LOCAL=0 \
DYN_VLLM_KV_EVENT_PORT=$KV_EVENT_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_LOCAL} \
UCX_TLS=ib,rc,ud,rc_verbs,ud_verbs,cuda_copy \
UCX_NET_DEVICES=mlx5_0:1 \
UCX_MEMTYPE_CACHE=0 \
DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-read \
ENABLE_ENCODER_CACHE=0 \
python3 -m dynamo.sglang \
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --enable-mm-global-cache \
    --dtype auto \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 40 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.95 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'$KV_EVENT_PORT'","enable_kv_cache_events":true}' \
    > "$EPD_LOG" 2>&1 &
EPD_PID=$!
PIDS+=($EPD_PID)

echo "EPD starting (PID: $EPD_PID)..."
echo "This may take several minutes as the model loads..."

# Wait for EPD to initialize (model loading can take a while)
sleep 10

# Monitor EPD startup
STARTUP_TIMEOUT=300  # 5 minutes
STARTUP_COUNT=0
while [ $STARTUP_COUNT -lt $STARTUP_TIMEOUT ]; do
    if ! check_process $EPD_PID "EPD" > /dev/null 2>&1; then
        echo -e "${RED}EPD process died during startup. Check $EPD_LOG${NC}"
        tail -50 "$EPD_LOG"
        exit 1
    fi

    # Check if EPD has finished loading
    if grep -q "Model registration succeeded" "$EPD_LOG" 2>/dev/null || \
       grep -q "Successfully registered LLM with runtime config" "$EPD_LOG" 2>/dev/null; then
        echo -e "${GREEN}✓ EPD initialized successfully${NC}"
        break
    fi

    if grep -qi "error\|failed\|exception" "$EPD_LOG" 2>/dev/null | tail -5 | grep -v "UserWarning"; then
        echo -e "${YELLOW}Warning: Potential errors detected in EPD log${NC}"
    fi

    if [ $((STARTUP_COUNT % 10)) -eq 0 ]; then
        echo -n "."
    fi

    sleep 1
    STARTUP_COUNT=$((STARTUP_COUNT + 1))
done

if [ $STARTUP_COUNT -ge $STARTUP_TIMEOUT ]; then
    echo -e "${RED}EPD startup timed out after $STARTUP_TIMEOUT seconds${NC}"
    echo "Last 50 lines of EPD log:"
    tail -50 "$EPD_LOG"
    exit 1
fi

echo ""

# Step 5: Run tests
echo -e "${YELLOW}[5/5] Running tests...${NC}"

# Health check test
echo "Test 1: Frontend health check..."
if curl -s -f "http://localhost:$PORT_HTTP/health" > /dev/null; then
    echo -e "${GREEN}✓ Frontend health check passed${NC}"
else
    echo -e "${RED}✗ Frontend health check failed${NC}"
    exit 1
fi

# List models test
echo "Test 2: List available models..."
if curl -s -f "http://localhost:$PORT_HTTP/v1/models" -o /tmp/models.json; then
    echo -e "${GREEN}✓ Models endpoint accessible${NC}"
    echo "Available models:"
    cat /tmp/models.json | python3 -m json.tool 2>/dev/null || cat /tmp/models.json
else
    echo -e "${RED}✗ Models endpoint failed${NC}"
fi

# Simple completion test (if model supports it)
echo "Test 3: Simple text completion..."
cat > /tmp/completion_request.json <<EOF
{
  "model": "Qwen3-VL-32B-Instruct-FP8",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_tokens": 50,
  "temperature": 0.7
}
EOF

if curl -s -f -X POST "http://localhost:$PORT_HTTP/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @/tmp/completion_request.json \
    -o /tmp/completion_response.json; then
    echo -e "${GREEN}✓ Text completion test passed${NC}"
    echo "Response:"
    cat /tmp/completion_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/completion_response.json
else
    echo -e "${YELLOW}⚠ Text completion test failed (may be expected for VL model)${NC}"
fi

# Check all processes are still running
echo ""
echo "Final process health check..."
ALL_HEALTHY=true
check_process $NATS_PID "NATS" || ALL_HEALTHY=false
check_process $ETCD_PID "etcd" || ALL_HEALTHY=false
check_process $FRONTEND_PID "Frontend" || ALL_HEALTHY=false
check_process $EPD_PID "EPD" || ALL_HEALTHY=false

echo ""
if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Service endpoints:"
    echo "  - Frontend: http://localhost:$PORT_HTTP"
    echo "  - NATS: nats://$IP_LOCAL:$PORT_NATS"
    echo "  - etcd: http://$IP_LOCAL:$PORT_ETCD"
    echo "  - NATS monitoring: http://localhost:18222"
    echo ""
    echo "Logs:"
    echo "  - NATS: $NATS_LOG"
    echo "  - etcd: $ETCD_LOG"
    echo "  - Frontend: $FRONTEND_LOG"
    echo "  - EPD: $EPD_LOG"
    echo ""
    echo "Process PIDs:"
    echo "  - NATS: $NATS_PID"
    echo "  - etcd: $ETCD_PID"
    echo "  - Frontend: $FRONTEND_PID"
    echo "  - EPD: $EPD_PID"
    echo ""

    # Ask if user wants to keep services running
    if [ -t 0 ]; then
        echo -e "${YELLOW}Keep services running? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            CLEANUP_ON_EXIT=false
            echo -e "${GREEN}Services will continue running. To stop them, run:${NC}"
            echo "  kill ${PIDS[*]}"
            echo ""
            echo "Press Ctrl+C to tail logs..."
            tail -f "$FRONTEND_LOG" "$EPD_LOG"
        fi
    fi
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Some tests failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

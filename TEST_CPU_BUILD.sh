#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script for CPU docker build verification

set -e

IMAGE="${1:-dynamo:cpu-vllm-runtime-new}"

echo "========================================="
echo "CPU Docker Build Test Suite"
echo "Image: $IMAGE"
echo "========================================="
echo ""

# Test 1: Verify image exists
echo "[1/7] Checking if image exists..."
if docker images "$IMAGE" --format "{{.Repository}}:{{.Tag}}" | grep -q "$IMAGE"; then
    echo "✓ Image found: $IMAGE"
    docker images "$IMAGE" --format "  Size: {{.Size}}, Created: {{.CreatedSince}}"
else
    echo "✗ Image not found: $IMAGE"
    exit 1
fi
echo ""

# Test 2: Check NIXL packages installed
echo "[2/7] Checking installed NIXL packages..."
NIXL_PACKAGES=$(docker run --rm "$IMAGE" ls /opt/dynamo/venv/lib/python3.12/site-packages/ 2>/dev/null | grep "^nixl" || true)
echo "$NIXL_PACKAGES"

if echo "$NIXL_PACKAGES" | grep -q "nixl_cpu"; then
    echo "✓ nixl_cpu is installed"
else
    echo "✗ nixl_cpu is NOT installed"
    exit 1
fi

if echo "$NIXL_PACKAGES" | grep -q "nixl_cu12"; then
    echo "⚠ WARNING: nixl_cu12 is also installed (should not be present for CPU-only build)"
else
    echo "✓ nixl_cu12 is NOT installed (correct for CPU-only build)"
fi
echo ""

# Test 3: Verify NIXL wheels in wheelhouse
echo "[3/7] Checking NIXL wheels in wheelhouse..."
docker run --rm "$IMAGE" ls -lh /opt/dynamo/wheelhouse/nixl/ 2>/dev/null
echo ""

# Test 4: Test NIXL import
echo "[4/7] Testing NIXL import..."
docker run --rm "$IMAGE" python -c "
import sys
from nixl._api import nixl_agent, nixl_agent_config
print('✓ NIXL import successful')
print('  Module:', __import__('nixl').__file__)
nixl_mods = [m for m in sys.modules if 'nixl' in m and 'nixl_c' in m]
if nixl_mods:
    print('  Device module loaded:', nixl_mods[0])
" 2>&1
echo ""

# Test 5: Check vLLM installation
echo "[5/7] Checking vLLM installation..."
VLLM_VERSION=$(docker run --rm "$IMAGE" python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
if [ -n "$VLLM_VERSION" ]; then
    echo "✓ vLLM installed: version $VLLM_VERSION"
else
    echo "✗ vLLM not found or failed to import"
    exit 1
fi
echo ""

# Test 6: Test vLLM help (basic functionality check)
echo "[6/7] Testing vLLM CLI..."
docker run --rm "$IMAGE" python -m vllm.entrypoints.api_server --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ vLLM CLI works"
else
    echo "✗ vLLM CLI failed"
    exit 1
fi
echo ""

# Test 7: Check dynamo installation
echo "[7/7] Checking Dynamo installation..."
DYNAMO_VERSION=$(docker run --rm "$IMAGE" python -c "import dynamo; print(dynamo.__version__)" 2>/dev/null || echo "N/A")
if [ "$DYNAMO_VERSION" != "N/A" ]; then
    echo "✓ Dynamo installed: version $DYNAMO_VERSION"
else
    echo "⚠ Dynamo version check failed (may need source mount)"
fi
echo ""

echo "========================================="
echo "✓ All basic tests passed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Test PD disaggregation:"
echo "   ./container/run.sh --image $IMAGE --mount-workspace"
echo ""
echo "2. Start prefill worker:"
echo "   python -m dynamo.vllm --role prefill ..."
echo ""
echo "3. Start decode worker:"
echo "   python -m dynamo.vllm --role decode ..."
echo ""

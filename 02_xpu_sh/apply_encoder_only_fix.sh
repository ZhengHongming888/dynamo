#!/bin/bash
# Quick patch to apply PR #9292 encoder_only fix
# This adds the missing encoder_only flag for multimodal encode workers

set -e

ARGS_FILE="/usr/local/lib/python3.12/dist-packages/dynamo/sglang/args.py"

echo "=================================================="
echo "Applying encoder_only fix (PR #9292)"
echo "=================================================="
echo ""

# Check if file exists
if [ ! -f "$ARGS_FILE" ]; then
    echo "❌ Error: File not found: $ARGS_FILE"
    exit 1
fi

# Check if already patched
if grep -q "parsed_args.encoder_only = True" "$ARGS_FILE"; then
    echo "✅ Fix is already applied!"
    exit 0
fi

# Backup original
BACKUP="${ARGS_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
echo "Creating backup: $BACKUP"
cp "$ARGS_FILE" "$BACKUP"

# Apply the patch
echo "Applying patch..."

# Find the line number where we need to insert
LINE_NUM=$(grep -n "if dynamo_config.embedding_worker:" "$ARGS_FILE" | head -1 | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo "❌ Error: Could not find insertion point (if dynamo_config.embedding_worker:)"
    echo "Restoring backup..."
    mv "$BACKUP" "$ARGS_FILE"
    exit 1
fi

# Calculate insertion point (after the is_embedding block)
INSERT_LINE=$((LINE_NUM + 2))

echo "Inserting fix at line $INSERT_LINE..."

# Create patched file
{
    head -n $((INSERT_LINE - 1)) "$ARGS_FILE"
    cat << 'EOF'

    # Enable encoder_only mode for multimodal encode workers to load only vision encoder
    # This significantly reduces memory usage by avoiding loading the full LLM weights
    if dynamo_config.multimodal_encode_worker:
        parsed_args.encoder_only = True
EOF
    tail -n +$INSERT_LINE "$ARGS_FILE"
} > "${ARGS_FILE}.new"

# Replace original with patched version
mv "${ARGS_FILE}.new" "$ARGS_FILE"

echo ""
echo "✅ Patch applied successfully!"
echo ""
echo "Backup saved at: $BACKUP"
echo ""

# Verify the fix
echo "Verifying fix..."
if python3 << 'VERIFY'
import inspect
from dynamo.sglang.args import DynamoConfig
source = inspect.getsource(DynamoConfig.parse_server_args)
if 'encoder_only' in source and 'multimodal_encode_worker' in source:
    print('✅ Fix verified: encoder_only flag is now present!')
    exit(0)
else:
    print('❌ Fix verification failed!')
    exit(1)
VERIFY
then
    echo ""
    echo "=================================================="
    echo "✅ SUCCESS!"
    echo "=================================================="
    echo ""
    echo "You can now run your 32B model script:"
    echo "  ./start_sglang_pd_xpu_32b.sh"
    echo ""
    echo "Expected behavior:"
    echo "  - Much lower memory usage (~2-4GB instead of ~20GB)"
    echo "  - No 'Detected fp8 checkpoint' message"
    echo "  - Faster startup (only vision encoder loads)"
    echo ""
else
    echo ""
    echo "❌ Verification failed. Restoring backup..."
    mv "$BACKUP" "$ARGS_FILE"
    echo "Backup restored. Please check the file manually."
    exit 1
fi

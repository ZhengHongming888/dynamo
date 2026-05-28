# 32B Model OOM Issue - Root Cause Analysis

## Issue Summary
The 32B model fails with `RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` while the 3B model works fine.

## Root Cause: MISSING CRITICAL FIX

**PR #9292** (merged on May 8, 2026) fixes this exact issue but **IS NOT in your installed package**.

### What PR #9292 Does:
- **Problem**: Multimodal encode workers load the ENTIRE model (vision encoder + LLM weights)
- **Solution**: Sets `encoder_only = True` flag to load ONLY the vision encoder
- **Impact**: Dramatically reduces memory usage for multimodal encode workers

### Evidence:

#### ✅ Fix IS in your git repo:
```bash
$ cd /hongming/dynamo && git log --oneline | head -20
...
a5f3459ad7 fix: enable encoder_only mode for sglang multimodal encode workers (#9292)
...
```

#### ❌ Fix is NOT in installed package:
```bash
$ grep "encoder_only" /usr/local/lib/python3.12/dist-packages/dynamo/sglang/args.py
# No results - the fix is missing!
```

### What Should Be There:

From PR #9292 in `components/src/dynamo/sglang/args.py`:

```python
# If --embedding-worker is set, also set SGLang's --is-embedding flag
if dynamo_config.embedding_worker:
    parsed_args.is_embedding = True

# Enable encoder_only mode for multimodal encode workers to load only vision encoder
# This significantly reduces memory usage by avoiding loading the full LLM weights
if dynamo_config.multimodal_encode_worker:
    parsed_args.encoder_only = True   # <-- THIS LINE IS MISSING IN YOUR INSTALL
```

## Why This Causes Your Issue:

### Without the fix (current state):
1. 32B encode worker tries to load:
   - Vision encoder (ViT) ~2-4GB
   - **Full 32B LLM weights** ~16-20GB FP8
   - **Total: ~20-24GB** → OOM on 30GB XPU with 0.7 mem fraction

2. 3B encode worker tries to load:
   - Vision encoder (ViT) ~2-4GB
   - **Full 3B LLM weights** ~6GB BF16
   - **Total: ~8-10GB** → Works fine

### With the fix (encoder_only=True):
1. 32B encode worker loads:
   - Vision encoder ONLY ~2-4GB
   - **No LLM weights**
   - **Total: ~2-4GB** → Should work!

2. 3B encode worker loads:
   - Vision encoder ONLY ~2-4GB
   - **No LLM weights**
   - **Total: ~2-4GB** → Works (but also loads LLM by accident)

## Solution: Rebuild/Reinstall Dynamo

### Option 1: Rebuild from source (Recommended)
```bash
cd /hongming/dynamo

# If using Docker container, rebuild the image
# or reinstall the package:
pip install -e components/  # Install in editable mode

# Verify the fix is applied
python3 -c "
import inspect
from dynamo.sglang.args import DynamoConfig
source = inspect.getsource(DynamoConfig.parse_server_args)
if 'encoder_only' in source:
    print('✅ Fix is now installed!')
else:
    print('❌ Fix still missing')
"
```

### Option 2: Quick manual patch (Temporary)
```bash
# Backup original
cp /usr/local/lib/python3.12/dist-packages/dynamo/sglang/args.py \
   /usr/local/lib/python3.12/dist-packages/dynamo/sglang/args.py.backup

# Apply patch - find the line with:
#   if dynamo_config.embedding_worker:
#       parsed_args.is_embedding = True
# And add after it:
#   
#   if dynamo_config.multimodal_encode_worker:
#       parsed_args.encoder_only = True
```

### Option 3: Use environment variable workaround
If rebuilding is not immediately possible, you can pass the flag directly:
```bash
# In your start_sglang_pd_xpu_32b.sh, add to the python command:
python3 -m dynamo.sglang \
    --encoder-only \              # <-- ADD THIS LINE
    --model /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
    --enable-multimodal \
    --multimodal-encode-worker \
    # ... rest of args
```

## Verification Steps

After applying the fix:

1. **Check the logs**: Should see much lower memory usage
2. **No more "Detected fp8 checkpoint"**: Because LLM head won't be loaded
3. **Faster startup**: Only vision encoder loads, not full model
4. **Should succeed**: Memory usage ~2-4GB instead of ~20GB

## Additional Notes

- The 3B model "works" but is also loading unnecessary LLM weights (wasting ~6GB)
- After applying the fix, both 3B and 32B should use similar memory (~2-4GB)
- This is the SAME fix that was applied to vLLM multimodal encode workers

## References
- PR: https://github.com/ai-dynamo/dynamo/pull/9292
- Issue: #9291
- Commit: a5f3459ad76ac645aa5bf2677cc7de3549c0c4d4
- Author: Daniel Socek <daniel.socek@intel.com>
- Merged: May 8, 2026

# Fixes for 32B Model XPU Error

## Issue
`RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`

## Root Cause
XPU device memory issue during large FP8 model loading (32B parameters)

## Recommended Solutions (Try in order):

### 1. Clean XPU Device 4 Memory
```bash
# Reset the XPU device before running
ZE_AFFINITY_MASK=4 python3 -c "import torch; torch.xpu.empty_cache(); print('Cleared')"
```

### 2. Use a Different XPU Device
Change in start_sglang_pd_xpu_32b.sh:
```bash
# Try device 2 (which works for 3B) or another clean device
export XPU_DEVICE=2  # or 0, 1, 3, 5, 6, 7
```

### 3. Reduce Memory Fraction
Change line 84:
```bash
--mem-fraction-static 0.5 \    # Reduced from 0.7
```

### 4. Force BF16 Instead of Auto
Change lines 82-83:
```bash
--dtype bfloat16 \              # Instead of auto
# Remove: --kv-cache-dtype auto \
```

### 5. Add Memory Management Environment Variables
Add before ZE_AFFINITY_MASK:
```bash
IPEX_TILE_AS_DEVICE=0 \
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
```

### 6. Verify Model Path
Check if the FP8 model exists and is complete:
```bash
ls -lh /mnt/weka/data/llm-d-models-pv/models--Qwen--Qwen3-VL-32B-Instruct-FP8/
```

## Quick Test Command
```bash
# Test minimal model load on device 4
ZE_AFFINITY_MASK=4 python3 << 'EOPYTHON'
import torch
print("Testing XPU 4 large allocation...")
try:
    # Allocate ~16GB to simulate 32B model
    x = torch.randn(4096, 1024, 1024, device='xpu', dtype=torch.bfloat16)
    print(f"Success! Allocated {x.element_size() * x.nelement() / 1024**3:.2f} GB")
    del x
    torch.xpu.empty_cache()
except Exception as e:
    print(f"Failed: {e}")
EOPYTHON
```

# CPU Docker Build Fixes

## Problem
When running dynamo vLLM on CPU, the system fails with:
```
ImportError: /opt/dynamo/venv/lib/python3.12/site-packages/nixl_cu12/_bindings.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZNK9nixlAgent13prepXferDlistERK12nixlDescListI13nixlBasicDescERP10nixlDlistHPK21nixlAgentOptionalArgs
```

This occurs because the NIXL meta package tries to import `nixl_cu12` instead of `nixl_cpu`.

## Root Causes

### 1. NIXL Meta Package Not Finding CPU Wheel
The NIXL meta package (`nixl/__init__.py`) only looked for CUDA wheels (`nixl_cu13`, `nixl_cu12`), not CPU or XPU wheels.

### 2. vLLM CPU Build Missing L2 Cache Workaround
PyTorch 2.10+ CPU builds removed the `at::cpu::L2_cache_size()` function, causing vLLM compilation to fail.

### 3. Meta Package Wheel Not Built from Patched Source
The meta package wheel was being copied from the meson build directory instead of being explicitly built after source patches were applied.

## Solutions Applied

### 1. Fixed NIXL Meta Package Detection (`wheel_builder.Dockerfile`)
**Lines 510-514**: Added sed commands to patch NIXL source for non-CUDA devices:
- Update `meson.build` to use device-specific wheel directory name (e.g., `nixl_cpu` instead of `nixl_cu12`)
- Extend candidates list in `nixl-meta/nixl/__init__.py` to include `nixl_cpu` and `nixl_xpu`
- Update error message to be device-agnostic

**Lines 556-571**: Explicitly build meta package wheel:
```dockerfile
uv build . --wheel --out-dir /opt/dynamo/dist/nixl --python $PYTHON_VERSION && \
cd src/bindings/python/nixl-meta && \
uv build . --wheel --out-dir /opt/dynamo/dist/nixl --python $PYTHON_VERSION
```

This ensures the meta package is built from the patched source with CPU/XPU support.

### 2. Applied vLLM L2 Cache Patch (`install_vllm.sh`)
**Lines 210-215**: Added patch application before building vLLM for CPU:
```bash
if [ -f /tmp/deps/vllm/cpu_l2_cache_fix.patch ]; then
    echo "Applying CPU L2 cache fix patch..."
    git apply --ignore-whitespace /tmp/deps/vllm/cpu_l2_cache_fix.patch
fi
```

The patch (`cpu_l2_cache_fix.patch`) replaces the missing `at::cpu::L2_cache_size()` call with a hardcoded 1MB default.

### 3. Simplified Runtime Wheel Copy (`vllm_runtime.Dockerfile`)
**Line 256**: Removed duplicate meta package wheel copy since both wheels are now in `/opt/dynamo/dist/nixl/`:
```dockerfile
# Before:
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# After:
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
```

## Wheel Structure

After these changes, the CPU build produces two wheels in `/opt/dynamo/dist/nixl/`:

1. **Device-specific wheel**: `nixl_cpu-<version>.whl`
   - Contains C++ bindings built with meson/ninja
   - Installed to Python as `nixl_cpu` module
   - Contains `_bindings.so` compiled for CPU

2. **Meta package wheel**: `nixl-<version>.whl`
   - Pure Python package
   - Tries to import from candidates: `nixl_cu13`, `nixl_cu12`, `nixl_cpu`, `nixl_xpu`
   - Finds and imports `nixl_cpu` for CPU builds
   - Re-exports all NIXL APIs

## Installation Flow

1. Runtime image installs both wheels: `uv pip install /opt/dynamo/wheelhouse/nixl/nixl*.whl`
2. User code does: `from nixl._api import nixl_agent, nixl_agent_config`
3. `nixl/__init__.py` iterates through candidates and successfully imports `nixl_cpu`
4. All NIXL APIs work correctly on CPU

## Files Modified

1. `container/templates/wheel_builder.Dockerfile` - NIXL build and meta package generation
2. `container/templates/vllm_runtime.Dockerfile` - Runtime wheel installation
3. `container/deps/vllm/install_vllm.sh` - vLLM CPU build with L2 cache patch
4. `container/deps/vllm/cpu_l2_cache_fix.patch` - Workaround for missing PyTorch API

## Testing Required

- Build CPU docker image: `make build-cpu` or similar
- Run dynamo vLLM with CPU device
- Test prefill/decode disaggregation with pure CPU setup
- Verify NIXL imports work correctly: `python -c "from nixl._api import nixl_agent; print('OK')"`
- Verify vLLM starts without L2 cache errors

## Related Issues

- PR #7891: CPU build support base changes
- PyTorch issue: `at::cpu::L2_cache_size()` removed in 2.10+
- vLLM issue #33991: Non-AVX512 CPU build handling

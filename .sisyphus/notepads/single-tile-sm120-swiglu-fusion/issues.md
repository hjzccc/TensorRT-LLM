## Task 1 Script Fixes - Resolved Issues

### Issue 1: Module-Level Library Loading Breaks --help
**Status**: RESOLVED
- **Root Cause**: `load_trtllm_library()` called at line 34 (module import time)
- **Impact**: Any import of the module fails if libnvinfer.so.10 is unavailable
- **Fix**: Removed module-level call; added calls to `harness_main()` and `branch_main()` instead
- **Files**: `test_single_tile_swiglu_fusion.py`

### Issue 2: Guard Script Import Fails
**Status**: RESOLVED
- **Root Cause**: Guard script imports from fusion script, triggering module-level library loading
- **Impact**: `test_single_tile_swiglu_guard.py --help` fails
- **Fix**: Fixing Issue 1 automatically fixes this (no changes needed to guard script)
- **Files**: `test_single_tile_swiglu_guard.py`

### Issue 3: Row/Col Indexing Type Error
**Status**: RESOLVED
- **Root Cause**: `divmod(flat_index.item(), candidate.shape[1])` where shape[1] is a torch.Size element
- **Impact**: Potential type mismatch in error reporting
- **Fix**: Explicitly convert both operands to int: `divmod(int(flat_index.item()), int(candidate.shape[1]))`
- **Files**: `test_single_tile_swiglu_fusion.py` line 353

### Verification Results
- ✅ `PYTHONDONTWRITEBYTECODE=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --help` → exit 0
- ✅ `PYTHONDONTWRITEBYTECODE=1 python3 added_benchmark/test_single_tile_swiglu_guard.py --help` → exit 0
- ✅ `PYTHONPYCACHEPREFIX=/tmp/trtllm_pycache python3 -m py_compile` → OK


## Task 1 Runtime Blocker
- Structured script verification passed, but full harness execution is blocked locally because `libnvinfer.so.10` is unavailable on the loader path and `/usr/local/tensorrt` is absent.
- Proceeding to Task 2 implementation while leaving Task 1 unchecked until runtime verification is possible.

## Task 2 Verification Blockers
- `lsp_diagnostics` cannot validate `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` here because no LSP server is configured for `.cu` files in this environment.
- `cmake --build "cpp/build" -j4 --target _moe_gemm_launcher` does not reach compilation locally because the checked-in `cpp/build/CMakeCache.txt` points at `/code/tensorrt_llm/cpp/build`, the generated Makefile expects `/usr/local/cmake/bin/cmake`, and `/usr/local/tensorrt` is absent.

## Task 4 Verification Blockers
- `lsp_diagnostics` on `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` still fails immediately because this environment has no LSP server configured for `.cu` files.
- `cmake --build "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/cpp/build" -j4 --target _moe_gemm_launcher` still stops before compilation because `cpp/build/CMakeCache.txt` was generated for `/code/tensorrt_llm/cpp/build` and the generated Makefile invokes missing `/usr/local/cmake/bin/cmake`.

## Task 5 Verification Blockers
- `lsp_diagnostics` on `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp` cannot be made clean here because the configured CUDA clang invocation rejects `sm_120a` and aborts before semantic analysis (`Unsupported CUDA gpu architecture: sm_120a`, `Unable to handle compilation, expected exactly one compiler job in ''`).
- `cmake --build "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/cpp/build" -j4 --target _moe_gemm_launcher` still does not reach compilation because the checked-in cache targets `/code/tensorrt_llm/cpp/build` and the generated Makefile depends on missing `/usr/local/cmake/bin/cmake`.
- Full runtime validation of the fused M128 store path remains blocked locally until a usable TensorRT/CUDA build environment is available.

## Task 6 Fallback Observability - RESOLVED
- **Objective**: Encode and verify unsupported-case fallback behavior in harnesses.
- **Implementation**: Added explicit gate-state reporting surface to both harnesses.
- **Status**: ✅ COMPLETE
  - Fusion harness: `--print-gate-state` flag reports `FUSION_GATE=0/1` without library loading.
  - Guard harness: Explicit fallback matrix verification before subprocess execution.
  - Fallback matrix: No opt-in, FORCE_UNFUSED, and tactics 1/2/3 all correctly report `FUSION_GATE=0`.
  - Deterministic: Gate state depends only on env vars and tactic, not runtime conditions.
- **Verification**: All 6 fallback cases pass deterministically (no opt-in, opt-in+tactic0, forced unfused, tactic 1/2/3).
- **Files Modified**: `added_benchmark/test_single_tile_swiglu_fusion.py`, `added_benchmark/test_single_tile_swiglu_guard.py`

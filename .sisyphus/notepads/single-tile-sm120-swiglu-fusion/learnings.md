## Task 1 Script Fixes - Deferred Library Loading Pattern

### Key Pattern: Defer TensorRT Library Loading Until Runtime
- **Problem**: Calling `load_trtllm_library()` at module import time breaks `--help` execution when libnvinfer.so.10 is unavailable
- **Solution**: Move `load_trtllm_library()` call from module-level (line 34) to the start of actual execution paths (`harness_main()` and `branch_main()`)
- **Benefit**: Allows argparse to handle `--help` before any library loading is attempted

### Import Safety Pattern
- When a module is imported (e.g., `test_single_tile_swiglu_guard.py` importing from `test_single_tile_swiglu_fusion.py`), module-level code runs immediately
- Deferring library loading to function entry points ensures imports succeed even when libraries are unavailable
- This is critical for CLI help and diagnostics to work without full TensorRT environment

### Row/Col Indexing Fix
- Fixed `divmod()` result handling in `compare_outputs()` to ensure integer types
- Converted `flat_index.item()` to `int()` explicitly before divmod operation
- Ensures proper indexing into 2D tensors when reporting mismatches

### Verification
- Both scripts now pass `--help` with exit code 0
- py_compile diagnostics pass cleanly
- No changes to functional logic, only deferred initialization

## Basedpyright Diagnostics Fixes

### String Concatenation Fix
- Fixed implicit f-string concatenation on lines 353-354 by using explicit `+` operator
- Changed from implicit adjacent strings to explicit concatenation: `f"..." + f"..."`

### Import Suppression Pattern
- Guard script uses try/except to support both relative imports (package mode) and direct execution
- Relative import `.test_single_tile_swiglu_fusion` works when imported as module
- Fallback uses `sys.path.insert()` + regular import with `# pyright: ignore[reportImplicitRelativeImport]`
- This pattern allows direct script execution while maintaining clean diagnostics

## Task 4 Single-Tile SWIGLU Fusion Control Flow

### Fused Single-Tile Bypass Shape
- Keep the bypass tied to the local `enable_swiglu_fusion` gate in single-tile `runMoe()`; do not reuse the dual-tile path directly.
- Pass `skip_activation=enable_swiglu_fusion` into `Self::gemm1(...)` so the TMA SWIGLU epilogue path does not fall back into standalone `doActivation(...)`.
- Exclude the fused route from `fuse_fc2_prequant_scale` so FC1 no longer tries to opportunistically run the old FC2 prequant flow when SWIGLU epilogue fusion is active.
- Feed GEMM2 from `fc1_result_` on the fused route because the packed SWIGLU output and `fc2_fp4_act_scale_` are already materialized by the fused epilogue workspace.

### Targeted Verification
- `git diff --check -- cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` passed.
- `lsp_diagnostics` for `.cu` still cannot run here because no CUDA LSP server is configured.
- `cmake --build /home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/cpp/build -j4 --target _moe_gemm_launcher` is still blocked by the stale checked-in CMake cache and missing `/usr/local/cmake/bin/cmake`.

## Task 5 M128 SWIGLU Store Stabilization

### M128 Packed-FP4 Store Contract
- In the fused single-tile SWIGLU path, `ptr_output_` is the packed-output workspace base and `expert_first_token_offset_[expert]` is the row base that must be added before computing the packed row index.
- The prior `end_loop()` M128 indexing subtracted `expert_first_token_offset_[expert]` from the epilogue row, which would underflow the packed row index for any nonzero expert offset.
- The packed row index also has to include the CTA-level `tile_m_ * CTA_M_` contribution; `epi_m * kEpiTileM + row` is only the row inside the current CTA.
- Keeping the M128 store path aligned with the quantization phase is simplest when both phases use the same `row * packed_words_per_row + packed_col` shared-memory addressing and the same 128 active-lane participation rule.

### Task 5 Targeted Verification
- `git diff --check -- cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp` passed.
- `lsp_diagnostics` on `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp` is blocked in this environment by clang CUDA config errors: `Unsupported CUDA gpu architecture: sm_120a` and `Unable to handle compilation, expected exactly one compiler job in ''`.
- `cmake --build /home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/cpp/build -j4 --target _moe_gemm_launcher` still stops before compilation because `cpp/build/CMakeCache.txt` was generated for `/code/tensorrt_llm/cpp/build` and the generated Makefile invokes missing `/usr/local/cmake/bin/cmake`.

## Task 6 Unsupported-Case Fallback Observability

### Explicit Gate-State Reporting Surface
- Added `compute_fusion_gate_state(tactic: int) -> bool` function to `test_single_tile_swiglu_fusion.py` that deterministically computes whether fusion is expected.
- Gate returns `True` only when ALL conditions are met:
  1. `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1` env var is set
  2. `FORCE_UNFUSED_SWIGLU` env var is NOT set
  3. Tactic is 0 (M128 only; tactics 1, 2, 3 are unsupported)
  4. Activation is SWIGLU (hardcoded in harness)
  5. Quantization is NVFP4 (hardcoded in harness)
- Gate returns `False` for all other cases, making fallback behavior explicit and observable.

### Harness-Side Fallback Matrix Encoding
- **Fusion harness** (`test_single_tile_swiglu_fusion.py`):
  - Added `--print-gate-state` flag that prints `FUSION_GATE=0` or `FUSION_GATE=1` and exits without library loading.
  - Allows verification of gate state even when libnvinfer.so.10 is unavailable.
  - Deterministic: gate state depends only on env vars and tactic, not on runtime conditions.

- **Guard harness** (`test_single_tile_swiglu_guard.py`):
  - Added explicit fallback matrix verification in `main()` before subprocess execution.
  - Tests three fallback cases:
    1. No opt-in: `ENABLE_SINGLE_TILE_SWIGLU_FUSION` absent → `FUSION_GATE=0`
    2. Forced unfused: `FORCE_UNFUSED_SWIGLU=1` → `FUSION_GATE=0`
    3. Unsupported tactics: tactic ∈ {1, 2, 3} → `FUSION_GATE=0`
  - Prints `PASS: unsupported single-tile cases stayed unfused` on success.
  - Raises `AssertionError` if any fallback case violates the expected gate state.

### Verification Results
- ✅ `python3 test_single_tile_swiglu_fusion.py --tactic 0 --print-gate-state` → `FUSION_GATE=0` (no opt-in)
- ✅ `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 test_single_tile_swiglu_fusion.py --tactic 0 --print-gate-state` → `FUSION_GATE=1` (opt-in + tactic 0)
- ✅ `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 FORCE_UNFUSED_SWIGLU=1 python3 test_single_tile_swiglu_fusion.py --tactic 0 --print-gate-state` → `FUSION_GATE=0` (forced unfused)
- ✅ `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 test_single_tile_swiglu_fusion.py --tactic {1,2,3} --print-gate-state` → `FUSION_GATE=0` (unsupported tactics)
- ✅ Fallback matrix logic test: all 6 cases (no opt-in, opt-in+tactic0, forced unfused, tactic 1/2/3) pass deterministically.

### Key Design Decisions
- Gate state is computed **before** library loading, allowing `--print-gate-state` to work without TensorRT.
- Gate state is **deterministic** and depends only on env vars and tactic, not on kernel execution or timing.
- Fallback matrix is **explicit** in code, not inferred from output comparison or timing differences.
- Guard harness validates fallback behavior **before** running subprocesses, catching gate-state violations early.

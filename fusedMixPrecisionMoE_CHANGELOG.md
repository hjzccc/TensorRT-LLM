# Fused Mixed-Precision MoE — Changelog

All changes for the fused mixed-precision MoE feature (bf16 hot experts + nvfp4 cold experts in a single kernel call).

**Design doc**: `fusedMixPrecisionMoE.txt`
**Implementation plan**: `fusedMixPrecisionMoE_plan.md` (618 lines, locked)

---

## Summary

| Metric | Value |
|--------|-------|
| Files created | 4 |
| Files modified | 7 |
| Total lines across all touched files | ~16,565 |
| C++ test cases added | 7 |
| Python test cases added | 13 |
| Benchmark scripts added | 1 |

---

## Phase 1: C++ Kernel Implementation

### Step 0 — Header data structures

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` (1133 lines total)

Added:
- `MixedPrecisionMoeState` struct holding per-call state: precision assignments, per-group expert maps, per-group first-token offsets, expand buffers, scratch output, workspace pointers.
- `runMixedPrecisionMoe()` pure virtual in `CutlassMoeFCRunnerInterface`.
- `getMixedPrecisionWorkspaceSize()` pure virtual in `CutlassMoeFCRunnerInterface`.
- Override declarations in `CutlassMoeFCRunner<T, WeightType, OutputType, Enable>`.
- Private workspace helper declarations:
  - `getWorkspaceDeviceBufferSizesMixedPrecision()`
  - `configureWsPtrsMixedPrecision()`
- 16+ new workspace member pointers (`mixed_prec_*` prefix) for dual-group buffers.

### Step 0b — Utility header declarations

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h` (131 lines total)

Added:
- `sortExpertsByTokenCount()` declaration — assigns precision (bf16 vs fp4) to each expert based on token load, builds per-group `expert_first_token_offset` arrays.
- `expandInputRowsMixedPrecisionKernelLauncher()` template declaration — dual-buffer expand that writes bf16-expert rows to bf16 buffer and quantizes fp4-expert rows to fp4 buffer with scaling factors.

### Step 1.1 — sortExpertsByTokenCount kernel

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` (lines ~921–1048)

Added:
- `sortExpertsByTokenCountKernel` — single-thread CUDA kernel with PDL guards (`cudaTriggerProgrammaticLaunchCompletion`).
- Computes `tokens_per_expert[E]` from `expert_first_token_offset`.
- Top-K selection: picks top `num_high_precision_experts` by token count → bf16, rest → fp4.
- Builds per-group `bf16_expert_first_token_offset[E+1]` and `fp4_expert_first_token_offset[E+1]`.
- Host launcher `sortExpertsByTokenCount()` with `<<<1, 1>>>` launch.

### Step 1.2 — expandInputRowsMixedPrecision kernel

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` (lines ~1824–2114)

Added:
- `expandInputRowsMixedPrecisionKernel<T>` — CUDA kernel iterating permuted rows.
  - Checks `expert_precision_assignment[expert_id]` for each row.
  - bf16 rows: direct copy to bf16 output buffer.
  - fp4 rows: quantize to fp4 with per-block scaling factors (`__nv_fp4_e2m1` packing).
  - PDL guards for programmatic launch completion.
- `expandInputRowsMixedPrecisionKernelLauncher<T>` — host launcher with grid sizing.
- Template instantiations for `__nv_bfloat16` and `half`.

### Step 1.3–1.5 — Dual gemm1, Activation, Dual gemm2 + Finalize

Handled inside `runMixedPrecisionMoe()` orchestrator (Step 1.7).

- **gemm1**: `computeStridesTmaWarpSpecializedDispatch()` for TMA descriptors, then `this->gemm1()` for bf16 group and `fp4_runner->gemm1()` for fp4 group.
- **Activation**: `doActivation<bf16>()` for bf16 group, `doActivation<fp4>()` for fp4 group.
- **gemm2 + Finalize**: Fused finalize via TMA epilogue writing to `mixed_prec_final_output_scratch_`. Each group's result accumulated into `final_output` by `accumulateMixedPrecisionOutput()`.

### Step 1.6 — Workspace management

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` (lines ~3304–3666)

Added three methods:
- `getWorkspaceDeviceBufferSizesMixedPrecision()` — computes buffer sizes for all dual-group workspace allocations (precision assignment arrays, per-group offsets, dual expand buffers, dual GEMM intermediates, scratch output, permutation maps).
- `getMixedPrecisionWorkspaceSize()` — sums all buffer sizes + alignment padding, returns total workspace bytes.
- `configureWsPtrsMixedPrecision()` — slices a flat workspace pointer into individual buffer pointers with 256-byte alignment.

### Step 1.7 — runMixedPrecisionMoe orchestrator

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` (lines ~4622–4848)

Added `CutlassMoeFCRunner::runMixedPrecisionMoe()` — the main entry point:
1. `configureWsPtrsMixedPrecision()` — set up workspace.
2. `sortExpertsByTokenCount()` — assign bf16/fp4 to each expert by token load.
3. `expandInputRowsMixedPrecisionKernelLauncher()` — dual-buffer input expansion.
4. bf16 gemm1 → bf16 activation.
5. fp4 gemm1 → fp4 activation.
6. Zero-initialize `final_output`.
7. bf16 gemm2 with fused finalize writing to scratch → `accumulateMixedPrecisionOutput()`.
8. fp4 gemm2 with fused finalize writing to scratch → `accumulateMixedPrecisionOutput()`.

### accumulateMixedPrecisionOutput kernel

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` (lines ~2911–2952)

Added:
- `accumulateMixedPrecisionOutputKernel<T>` — element-wise `output[i] += scratch[i]` with PDL guards.
- Host launcher `accumulateMixedPrecisionOutput<T>()`.

---

## Phase 2: Python Integration

### Step 2.1 — moeOp.cpp (torch C++ op)

**File**: `cpp/tensorrt_llm/thop/moeOp.cpp` (1544 lines total)

Added:
- `MoeOpRunner::runMixedPrecisionMoe()` — public method (~170 lines) that:
  - Validates input tensors and dimensions.
  - Lazily creates `mFp4Runner` (`CutlassMoeFCRunner<bf16, fp4_e2m1>`).
  - Shares GEMM tactics from primary runner to fp4 runner.
  - Allocates mixed-precision workspace.
  - Builds `QuantParams::FP4(...)` from 6 NVFP4 scale tensors.
  - Calls `mRunner->runMixedPrecisionMoe(...)`.
- `mFp4Runner` and `mMixedPrecStreamWorkspaces` member variables.
- `getMixedPrecWorkspaceInfo()` and `buildFp4QuantParams()` private helpers.
- `TORCH_LIBRARY` registration: `.def("run_mixed_precision_moe", ...)`.

### Step 2.2 — torch_custom_ops.py (Python op wrapper)

**File**: `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` (2200 lines total)

Added:
- `@torch.library.custom_op("trtllm::fused_moe_mixed_precision", ...)` (lines ~345–483):
  - Accepts bf16 weights, fp4 packed weights, 6 NVFP4 scale tensors, routing tensors, dimension params.
  - Calls `torch.ops.trtllm.run_mixed_precision_moe(...)`.
- `@torch.library.register_fake(...)` (lines ~486–517):
  - Returns `torch.empty(num_rows, hidden_size, dtype=torch.bfloat16)` for torch.compile/export.

### Step 2.3 — moe_op_cutlass.py (Op abstraction)

**File**: `tensorrt_llm/_torch/modules/fused_moe/ops/moe_op_cutlass.py` (387 lines total)

Added:
- `CutlassMixedPrecisionMoEOp` class (lines ~297–387):
  - Extends the op abstraction pattern.
  - `forward()` routes to `trtllm::fused_moe_mixed_precision` custom op.
  - Manages weight layout validation (bf16 contiguous, fp4 packed as int64).

### Step 2.4 — fused_moe_mixed_precision.py (nn.Module) [NEW FILE]

**File**: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_mixed_precision.py` (286 lines)

Created:
- `FusedMixedPrecisionMoE(nn.Module)` — high-level module:
  - Registers bf16 and fp4 weight buffers.
  - Registers 6 NVFP4 scale tensor buffers (`fc1_weight_sf`, `fc1_act_sf`, `fc1_global_scale`, `fc2_weight_sf`, `fc2_act_sf`, `fc2_global_scale`).
  - `forward()` calls `CutlassMixedPrecisionMoEOp.forward()`.
  - `from_heter_moe()` classmethod for migration from `HeterCutlassFusedMoE`.
- `_ModuleProxy` helper for weight loading.

---

## Phase 3: Testing

### Step 3.1 — C++ unit tests

**File**: `cpp/tests/unit_tests/kernels/mixtureOfExpertsTest.cu` (3239 lines total, was 2728)

Added ~511 lines with `MixedPrecisionMoETest` fixture and 7 test cases:

| Test | Description |
|------|-------------|
| `SortExpertsByTokenCount_Basic` | 8 experts, 3 bf16 — verifies correct assignment and per-group offsets |
| `SortExpertsByTokenCount_AllBF16` | All experts assigned bf16 when `num_high_prec == num_experts` |
| `SortExpertsByTokenCount_AllFP4` | All experts assigned fp4 when `num_high_prec == 0` |
| `SortExpertsByTokenCount_TieBreaking` | Equal token counts — deterministic tie-breaking by expert ID |
| `SortExpertsByTokenCount_EmptyExperts` | Experts with 0 tokens handled correctly |
| `SortExpertsByTokenCount_DeepSeekV3Scale` | 256 experts, top-12 bf16 — mirrors DeepSeek-V3 config |
| `WorkspaceSize_LargerThanStandard` | Mixed-precision workspace ≥ standard single-precision workspace |

All guarded by `#if defined(ENABLE_BF16) && defined(ENABLE_FP4)` and SM ≥ 100 runtime skip.

### Step 3.2 — Python correctness tests [NEW FILES]

**File**: `tests/unittest/_torch/modules/moe/mixed_precision_moe_utils.py` (493 lines)

Created shared test utilities:
- `create_unquantized_weights()` — generates random bf16 weights for E experts.
- `pack_bf16_weights_for_fused_module()` / `pack_fp4_weights_for_fused_module()` — weight packing.
- `quantize_bf16_to_nvfp4()` — reference fp4 quantization.
- `create_bf16_reference_backend()` / `create_fused_mixed_precision_module()` — backend factories.
- `run_reference_forward()` / `run_mixed_precision_forward()` — forward pass helpers.
- `flush_l2_cache()` — L2 cache flush for accurate timing.
- `run_with_cuda_graph()` — CUDA graph wrapper for correctness and timing.
- `mixed_precision_supported()` — runtime capability check.
- Small config constants: `SMALL_NUM_EXPERTS=16`, `SMALL_TOP_K=4`, `SMALL_HIDDEN_SIZE=512`, `SMALL_INTER_SIZE=512`.

**File**: `tests/unittest/_torch/modules/moe/test_mixed_precision_moe.py` (892 lines)

Created 13 test cases across 6 test classes:

| Class | Test | Type | Description |
|-------|------|------|-------------|
| `TestWeightPacking` | `test_bf16_weight_shape` | CPU | Verifies bf16 weight buffer shapes |
| | `test_fp4_packed_as_int64` | CPU | Verifies fp4 packing: 16 elements per int64 |
| `TestModuleConstruction` | `test_module_init_parameters` | CPU | Module creation, buffer registration |
| | `test_module_accepts_valid_config` | CPU | Valid config accepted without error |
| | `test_module_rejects_invalid_high_prec` | CPU | `num_high_prec > num_experts` raises ValueError |
| `TestMixedPrecisionForwardEquivalence` | `test_all_bf16_matches_reference` | GPU | All experts bf16 → output matches pure bf16 reference |
| | `test_half_bf16_half_fp4` | GPU | 50% bf16 + 50% fp4 → output within tolerance of reference |
| | `test_all_fp4_within_tolerance` | GPU | All experts fp4 → output within relaxed tolerance |
| `TestCUDAGraphCompatibility` | `test_forward_under_cuda_graph` | GPU | Forward works inside CUDA graph capture/replay |
| | `test_cuda_graph_no_extra_allocs` | GPU | No per-call allocations during graph replay |
| `TestTorchCompileCompatibility` | `test_compile_no_graph_breaks` | GPU | `torch.compile` produces no graph breaks |
| `TestDeepSeekV3Scale` | `test_deepseek_v3_256e_top8` | GPU | 256 experts, top-8, 12 bf16 — DeepSeek-V3 config |
| `TestEdgeCases` | `test_single_token` | GPU | Single token input |
| | `test_experts_per_token_equals_num_experts` | GPU | top_k == num_experts (all experts selected) |

All GPU tests are L2-cache-aware and use CUDA graphs per task constraints.

### Step 3.3 — Benchmark [NEW FILE]

**File**: `tests/microbenchmarks/bench_mixed_precision_moe.py` (373 lines)

Created:
- Sweeps `seq_len × num_high_precision_experts` configurations.
- Compares HeterMoE (separate per-group `fused_moe()` calls) vs FusedMixedPrecMoE (single C++ call).
- L2-cache-aware timing with `flush_l2_cache()` between iterations.
- CUDA graph warmup + timed replay.
- CSV output with columns: `seq_len, num_high_prec, heter_ms, fused_ms, speedup`.
- Default config: E=16, top_k=4, hidden_size=512, inter_size=512 (fits RTX 5070).

---

## Files Summary

### New Files (4)

| File | Lines | Description |
|------|-------|-------------|
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_mixed_precision.py` | 286 | `FusedMixedPrecisionMoE` nn.Module |
| `tests/unittest/_torch/modules/moe/mixed_precision_moe_utils.py` | 493 | Test utilities (weight packing, backends, L2 flush, CUDA graph) |
| `tests/unittest/_torch/modules/moe/test_mixed_precision_moe.py` | 892 | 13 correctness test cases |
| `tests/microbenchmarks/bench_mixed_precision_moe.py` | 373 | Performance benchmark (fused vs HeterMoE) |

### Modified Files (7)

| File | Total Lines | What Changed |
|------|-------------|--------------|
| `cpp/.../include/moe_kernels.h` | 1133 | `MixedPrecisionMoeState` struct, virtual methods, workspace members |
| `cpp/.../include/moe_util_kernels.h` | 131 | `sortExpertsByTokenCount`, `expandInputRowsMixedPrecisionKernelLauncher` declarations |
| `cpp/.../moe_gemm/moe_kernels.cu` | 5887 | All kernel implementations + orchestrator + workspace management |
| `cpp/.../thop/moeOp.cpp` | 1544 | `runMixedPrecisionMoe()`, fp4 runner, torch op registration |
| `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` | 2200 | `fused_moe_mixed_precision` custom op + fake registration |
| `tensorrt_llm/_torch/modules/fused_moe/ops/moe_op_cutlass.py` | 387 | `CutlassMixedPrecisionMoEOp` class |
| `cpp/tests/unit_tests/kernels/mixtureOfExpertsTest.cu` | 3239 | 7 C++ test cases (~511 lines added) |

---

## Key Architecture Decisions

1. **Dual Runner Pattern**: Primary `CutlassMoeFCRunner<bf16, bf16>` for bf16 experts. Lazily-created `CutlassMoeFCRunner<bf16, fp4_e2m1>` passed as `fp4_runner` parameter.

2. **Finalize Strategy**: Fused finalize (TMA epilogue) writes to `mixed_prec_final_output_scratch_`, then `accumulateMixedPrecisionOutput` kernel adds each group's result to a zero-initialized `final_output`.

3. **FP4 Runner Tactic Sharing**: Both runners share GEMM tactics via `mFp4Runner->setTactic(gemm1_profile, gemm2_profile)`.

4. **Quant Params**: bf16 group uses empty `QuantParams{}`. fp4 group uses `QuantParams::FP4(...)` built from 6 NVFP4 scale tensors.

5. **FP4 Packing**: 16 fp4 elements per `int64` (pack factor = 16). Buffer sizes use `hidden_size / 16` for fp4.

---

## Deferred Items

| Item | Reason |
|------|--------|
| Phase 3.4: Buffer aliasing optimization | Deferred — optimization pass, not required for correctness |
| Phase 3.5–3.6: Short-pass optimization | Deferred — high complexity, requires profiling data |

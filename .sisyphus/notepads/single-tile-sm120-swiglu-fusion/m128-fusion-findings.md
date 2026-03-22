# SM120 Single-Tile M128 SwiGLU Fusion Findings

## Scope

- Target path: SM120, NVFP4, single-tile `runMoe()`, tactic `0`, `CtaShape128x128x128B`, SwiGLU, opt-in via `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1`.
- Non-goals: dual-tile, M64/M256 fusion bring-up, unsupported tactics, non-SwiGLU activations, non-NVFP4 quantization, and the pre-existing M32 runtime-compare gap.

## What Changed To Make M128 Single-Tile Fusion Work

### 1. Added a strict runtime gate for the supported path

- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`
- `enable_swiglu_fusion` is only true when all of the following are true:
  - single-tile path
  - opt-in env var enabled
  - activation is SwiGLU
  - block scaling type is NVFP4
  - prequant-scale path is disabled
  - forced-unfused path is disabled
  - GEMM1 is TMA warp-specialized
  - SM120 tile config is `CtaShape128x128x128B`
- This keeps unsupported cases on the old unfused route.

### 2. Routed GEMM1 through a fused SwiGLU epilogue path

- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`
- For the fused path, GEMM1 writes to `fc1_result_`, which is used as the packed FP4 SwiGLU output workspace.
- The old standalone `doGatedActivationKernel` path is bypassed for this case.
- GEMM2 reads directly from that fused packed output, along with `fc2_fp4_act_scale_` produced by the epilogue.

### 3. Added fused SwiGLU epilogue plumbing to the TMA workspace input

- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu`
- Added `setSwigluFusionParams(...)` to pass the fused output pointer, FC2 activation-scale buffer, expert row offsets, global scale factor, and shape info into the CUTLASS epilogue callback.
- The fused epilogue gets all data it needs without changing the unfused path.

### 4. Implemented the fused M128 epilogue callback

- `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp`
- The M128 fused visitor now does the full sequence inside the epilogue:
  1. read the GEMM1 epilogue fragment
  2. apply SwiGLU pairwise
  3. stage bf16 results in shared memory
  4. quantize to packed FP4
  5. write packed FP4 and scale bytes for GEMM2 input
- This is the core change that removes the separate activation kernel launch.

### 5. Fixed the weight/layout assumptions for fused gate-up pairing

- FC1 weights are interleaved so the GEMM output columns line up as `[up_0, gate_0, up_1, gate_1, ...]` in the fused path.
- The fused visitor then pairs adjacent fragment values as gate/up neighbors for SwiGLU.
- This makes the fusion work without changing the underlying accumulator math for the cooperative path.

### 6. Re-enabled pingpong for fused SM120 M128 and fixed its thread-local decode

- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`
- The fused SM120 M128 blockscaled path now selects pingpong again.
- `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp`
- The M128 visitor coordinate cache was extended from 8 to 16 entries and now fills `cute::size(tRS_epi0)` entries instead of assuming the old cooperative-sized thread-local fragment bundle.
- This is what made the pingpong path numerically correct again.

## Why We Do Not Have The Hang / Instability Issue Anymore

### Root cause of the old M128 store failure

- The old `end_loop()` M128 store path used the expert row offset incorrectly.
- The prior logic effectively subtracted `expert_first_token_offset_[expert]` from the epilogue row when computing the packed output row.
- For any expert with a nonzero offset, that could underflow or miscompute the packed row index and corrupt the final packed-FP4 store path.

### Correct packed-row computation now

- `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp`
- The fixed M128 `end_loop()` now uses:
  - `expert_row_base = expert_first_token_offset_[expert_idx_]`
  - `token_in_expert = tile_m_ * CTA_M_ + m_cta`
  - `packed_row = expert_row_base + token_in_expert`
- This keeps the global packed row aligned with expert-local routing and CTA-local row progress.

### Explicit synchronization contract in the fused M128 path

- The fused M128 visitor now uses explicit barriers between all three phases:
  1. before/after the shared-memory SwiGLU staging
  2. before/after the FP4 quantization phase
  3. before `end_loop()` consumes packed output and scale bytes
- The path no longer depends on implicit ordering or comments about harmless garbage reads to stay safe.

### Full warp participation is preserved for shuffle-based quantization

- The M128 quantization phase keeps all 128 active quantization threads participating in the quantize helper.
- Bounds checks guard only the global writes after quantization, not the shuffle-based quantize call itself.
- This avoids warp-level participation mismatches in the helper's internal shuffle operations.

### Global stores are now predicated cleanly

- `end_loop()` only writes packed FP4 and scale bytes for rows/columns that are in bounds.
- Out-of-bounds rows and half-columns return early before touching global memory.
- This prevents tail tiles from performing invalid packed writes.

## Current Validation Summary

- Build passes with `-j2` in the SM120 container.
- Supported fused correctness passed for tactic `0` with token counts `{1, 4, 8, 16, 64, 65, 128, 129}` and seeds `1234` and `5678`.
- Guard coverage confirms unsupported tactics stay unfused; tactic `2` still has the pre-existing M32 runtime-compare gap.
- End-to-end performance is positive but modest; the gain comes mainly from removing `doActivationKernel`, while the long fused GEMM kernel remains slightly slower than the unfused baseline.

## Practical Bottom Line

- M128 single-tile fusion works because the fused epilogue now has the right runtime gate, workspace plumbing, weight/layout contract, packed FP4 store logic, and pingpong-aware per-thread coordinate decode.
- The old hang/instability is gone because the M128 `end_loop()` row math is now correct and the quantize/store path obeys an explicit synchronization and full-warp participation contract.

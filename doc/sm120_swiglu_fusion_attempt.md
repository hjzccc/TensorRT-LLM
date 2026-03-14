# SM120 Single-Tile NVFP4 SwiGLU Design

This note describes the design that is active in the current codebase for single-tile SwiGLU fusion on SM120.

## Supported scope

- Architecture: `SM120`
- Quantization: `NVFP4`
- Activation: `SwiGLU`
- Path: single-tile `runMoe()`
- Supported fused tactics:
  - `tactic 0`: `CtaShape128x128x128B` (`M128`)
  - `tactic 1`: `CtaShape64x128x64B` (`M64`)
- Unsupported / forced unfused:
  - `tactic 2`: `M256`
  - `M32` fused path remains disabled because its epilogue fragment ordering is still unresolved

Fusion is opt-in and only enabled when `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1` is set.

## High-level execution flow

The current fused path replaces:

1. `GEMM1`
2. `doActivation` / `doGatedActivation`
3. `GEMM2`

with:

1. `GEMM1` with a fused SwiGLU + NVFP4 epilogue
2. direct packed-FP4 handoff into `GEMM2`

The active single-tile fused dataflow is:

```text
interleaved FC1 weights
  -> GEMM1 accumulators
     -> fused epilogue callback
        -> SwiGLU in registers
        -> bf16 staging in shared memory
        -> NVFP4 quantization
        -> packed FP4 + SF writes to global memory
           -> GEMM2 input
              -> GEMM2
```

## Weight layout contract

FC1 weights are physically interleaved before quantization so the GEMM1 output columns appear as:

```text
[up_0, gate_0, up_1, gate_1, ...]
```

That is what allows the fused epilogue to consume adjacent accumulator values as one SwiGLU pair.

The interleaving contract is enforced in the benchmark and validation harnesses so fused and unfused comparisons use equivalent logical weights.

## Runtime gate

The runtime fusion gate lives in `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`.

Fusion is enabled only when all of the following are true:

- single-tile path
- `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1`
- activation is `SwiGLU`
- block scaling type is `NVFP4`
- prequant-scale path is disabled
- `FORCE_UNFUSED_SWIGLU` is not set
- GEMM1 uses the TMA warp-specialized grouped GEMM path
- tile config is one of:
  - `CtaShape128x128x128B`
  - `CtaShape64x128x64B`

Everything else falls back to the stable unfused path.

## Core implementation pieces

### 1. Fused epilogue callback

The fused epilogue lives in:

- `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp`

It does three things:

1. uses epilogue-partition-derived coordinates to interpret the accumulator fragments
2. computes SwiGLU and stages bf16 results in shared memory
3. quantizes to packed NVFP4 and writes packed FP4 plus scale-factor bytes for `GEMM2`

The current callback supports the active `64x32` epilogue-tile family used by fused `M128` and `M64`.

### 2. Workspace and parameter plumbing

The TMA workspace input and MoE runtime wire the following into the callback:

- fused packed-output pointer
- FC2 activation-scale buffer
- expert-first-token offsets
- global scale factor
- output shape information

Key files:

- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu`
- `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h`
- `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h`

### 3. Dispatch and schedule selection

Single-tile fused SM120 dispatch is wired through:

- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h`
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h`
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`

Current schedule behavior:

- fused `M128`: pingpong
- fused `M64`: pingpong
- unfused paths may still use a mix of cooperative and pingpong, depending on tile and epilogue

### 4. SM120 grouped-FP4 candidate set

The current active grouped FP4 SM120 candidate list is:

- `M128`
- `M64`
- `M256`

with `M256` intentionally remaining outside the fusion gate.

Key file:

- `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp`

## Why the fused path is stable now

The main stability fixes in the current design are:

- correct packed-row computation using `expert_first_token_offset_` plus CTA-local token progress
- explicit synchronization between the bf16 staging phase and the quantization/store phase
- full-warp participation for the shuffle-based NVFP4 quantization helper
- bounds-checked packed-FP4 and scale-factor writes
- pingpong-aware per-thread coordinate decoding for the supported fused shapes

These fixes are what made the `M128` and `M64` fused paths numerically correct.

## Current limitation: M32 is still unresolved

The codebase contains structural SM120 `M32` support in generation and CUTLASS patching work, but the fused `M32` path is not part of the supported runtime design.

Why it is still disabled:

- the live fused `M32` kernel uses a `32x32` epilogue subtile with `FragmentSize=4`
- the remaining issue is the exact `visit_results(epi_v)` ordering for that epilogue path
- multiple attempts to reconstruct the mapping from raw MMA ownership and manual `epi_v` remaps still produced catastrophic wrong results
- because of that, the safe current code keeps `M32` outside the fusion gate

So the current design should be understood as:

```text
supported fused:   M128, M64
unsupported fused: M32, M256
```

## Validation status for the current code

### Correctness

- `M128` fused passes deterministic fused-vs-unfused checks
- `M64` fused passes deterministic fused-vs-unfused checks
- unsupported `M256` stays unfused and matches forced-unfused

### Performance

- `M128` fused shows a modest end-to-end win, mainly from removing `doActivationKernel`
- `M64` fused shows a stronger win for larger batches
- best observed `M64` speedup so far is about `7.4%` at `batch=4096`, validated by CUDA-event timing and `nsys` NVTX GPU projection

## Benchmark and validation harnesses

The current single-tile validation entry points are:

- `added_benchmark/test_single_tile_swiglu_fusion.py`
- `added_benchmark/test_single_tile_swiglu_guard.py`
- `added_benchmark/ncu_tile_roofline.py`
- `added_benchmark/compare_single_tile_swiglu_profile.py`

These scripts are the authoritative checks for the current single-tile fused design.

## Practical bottom line

The current codebase has a stable single-tile fused SM120 NVFP4 SwiGLU implementation for `M128` and `M64`.

That design is based on:

- interleaved FC1 gate/up layout
- fused GEMM1 epilogue computation
- packed NVFP4 output plus scale-factor writes from the epilogue
- direct GEMM2 consumption of the fused output
- pingpong scheduling for the supported fused tiles

`M32` is not part of the supported design yet and should still be treated as unresolved.

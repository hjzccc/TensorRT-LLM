# Fused Mixed Precision MoE — Implementation Plan

## 1. Motivation

Different MoE experts exhibit different arithmetic intensity (AI) based on their token load:
- **Hot experts** (high token count) → high AI → benefit from bf16 precision
- **Cold experts** (low token count) → low AI → precision gain from loading bf16 weights is diminishing; nvfp4 is sufficient

The `huanchen-twoGemmMoe` branch implements this at the **Python level** via `HeterCutlassFusedMoE`, making separate `fused_moe()` calls per precision group. This works but has overhead:
- Redundant token routing/sorting per call
- Separate activation kernels per group
- No opportunity to fuse the activation step across groups

**This plan describes a single fused C++ kernel** (`runMixedPrecisionMoe`) that handles both precision groups within one `runMoe`-like function, achieving:
1. **One fused activation call** across both precision groups
2. **Minimized quantization overhead** by co-locating quant steps with data movement
3. Full CUDA graph, compilation, and PDL compatibility

---

## 2. Architecture Overview

### 2.1 Current `runMoe()` Flow (single precision)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ runMoe() in moe_kernels.cu                                             │
│                                                                         │
│  1. Sort: fusedBuildExpertMapsSortFirstToken()                          │
│     → expert_first_token_offset[E+1], permuted_row_to_unpermuted_row   │
│                                                                         │
│  2. Expand: expandInputRowsKernelLauncher()                             │
│     → permuted_data (reordered input activations)                       │
│     → fc1_fp4_act_scale (if NVFP4 quantization)                        │
│                                                                         │
│  3. GEMM1: gemm1() → fc1_result (intermediate activations)             │
│                                                                         │
│  4. Activation: SwiGLU (fused in GEMM1 epilogue or separate kernel)    │
│     + optional quantization for GEMM2 input                             │
│                                                                         │
│  5. GEMM2: gemm2() → final_output (with fused finalize on Hopper)      │
│     Finalize combines expert outputs weighted by router scales          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Proposed `runMixedPrecisionMoe()` Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ runMixedPrecisionMoe()                                                           │
│                                                                                  │
│  0. Extra arguments:                                                             │
│     - fc1_expert_weights_bf16, fc2_expert_weights_bf16 (bf16 weight set)         │
│     - fc1_expert_weights_fp4,  fc2_expert_weights_fp4  (nvfp4 weight set)        │
│     - quant_params_bf16, quant_params_fp4 (per-group quant params)               │
│     - num_high_precision_experts (how many top-loaded experts get bf16)           │
│                                                                                  │
│  1. Sort + Precision Assignment:                                                 │
│     a. fusedBuildExpertMapsSortFirstToken() [existing, unchanged]                │
│        → expert_first_token_offset[E+1], permutation maps                        │
│     b. NEW: sortExpertsByTokenCount()                                            │
│        Input:  expert_first_token_offset[E+1], num_high_precision_experts        │
│        Output: expert_precision_assignment[E] ∈ {0=fp4, 1=bf16}                  │
│                bf16_expert_list[], fp4_expert_list[]                              │
│                bf16_token_boundary, fp4_token_boundary                            │
│        Method: Compute tokens_per_expert[e] = prefix[e+1] - prefix[e],           │
│                sort by count descending, top-K → bf16, rest → fp4                │
│                Reuse cub::DeviceRadixSort or simple selection (E is small)        │
│                                                                                  │
│  2. Dual-Buffer Expand Input + Quantization:                                     │
│     NEW: expandInputRowsMixedPrecisionKernelLauncher()                           │
│     Two passes over permuted tokens using expert_precision_assignment:            │
│       Pass 1 (bf16 experts): write to bf16_expanded_buffer                       │
│       Pass 2 (fp4 experts):  quantize to nvfp4, write to fp4_expanded_buffer     │
│                               + compute fp4 scaling factors                       │
│     Output: One contiguous buffer [bf16_region | fp4_region]                     │
│             with boundary index stored alongside                                  │
│     Buffer sizing: worst-case (all tokens to one group) for CUDA graph safety    │
│                                                                                  │
│  3. Dual GEMM1:                                                                  │
│     a. groupgemm_bf16: gemm1() with bf16 weights on bf16_region                  │
│        → bf16_gemm1_output (bf16 intermediate)                                   │
│     b. groupgemm_fp4:  gemm1() with fp4 weights on fp4_region                    │
│        → fp4_gemm1_output (cast to bf16 by CUTLASS epilogue)                     │
│     Output: One contiguous buffer [bf16_gemm1_out | fp4_gemm1_out]               │
│             boundary index precomputed from step 1                                │
│                                                                                  │
│  4. Fused Activation + Quantization:                                             │
│     SINGLE activation kernel over the entire [bf16_out | fp4_out] buffer         │
│     (Both outputs are bf16 at this point — fp4 gemm output is cast in epilogue)  │
│     a. SwiGLU activation on full buffer                                          │
│     b. For fp4_region only: quantize activation output to nvfp4 for gemm2        │
│        (use boundary index to know which part to quantize)                        │
│     Output: [bf16_act_out | fp4_act_out_quantized]                               │
│             + fp4 scaling factors for gemm2                                       │
│                                                                                  │
│  5. Dual GEMM2 + Finalize:                                                       │
│     a. groupgemm_bf16: gemm2() with bf16 weights on bf16_act_out                 │
│     b. groupgemm_fp4:  gemm2() with fp4 weights on fp4_act_out_quantized         │
│     Both write to the SAME final_output tensor at correct token positions         │
│     The finalize (weighted combination) handles cross-group assembly              │
│     using the inverse permutation from step 1                                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Step-by-Step Implementation

### Step 0: New Arguments & Data Structures

**File**: `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h`

Add to `CutlassMoeFCRunnerInterface`:

```cpp
virtual void runMixedPrecisionMoe(
    // Input
    void const* input_activations,
    void const* input_sf,
    bool const swizzled_input_sf,
    int const* token_selected_experts,
    float const* token_final_scales,

    // Dual weight sets — bf16 group
    void const* fc1_expert_weights_bf16,
    void const* fc1_expert_biases_bf16,
    void const* fc2_expert_weights_bf16,
    void const* fc2_expert_biases_bf16,

    // Dual weight sets — nvfp4 group
    void const* fc1_expert_weights_fp4,
    void const* fc2_expert_weights_fp4,

    // Quantization params per group
    QuantParams quant_params_bf16,
    QuantParams quant_params_fp4,

    // Mixed precision control
    int const num_high_precision_experts,  // top-N experts by load → bf16

    // Standard params (same as runMoe)
    ActivationParams fc1_activation_type,
    int64_t const num_rows,
    int64_t const num_valid_rows,
    int64_t const hidden_size,
    int64_t const unpadded_hidden_size,
    int64_t const inter_size,
    int const num_experts,
    int const experts_per_token,
    char* workspace_ptr,
    void* final_output,
    int* unpermuted_row_to_permuted_row,
    MOEParallelismConfig parallelism_config,
    bool const enable_alltoall,
    cudaStream_t stream
) = 0;
```

New internal data structure for precision assignment:

```cpp
struct MixedPrecisionMoeState {
    int* expert_precision_assignment;  // [E]: 0=fp4, 1=bf16
    int* bf16_expert_indices;          // list of bf16 expert IDs
    int* fp4_expert_indices;           // list of fp4 expert IDs
    int  num_bf16_experts;             // count
    int  num_fp4_experts;              // count
    int64_t bf16_token_count;          // total tokens routed to bf16 group
    int64_t fp4_token_count;           // total tokens routed to fp4 group
    int64_t boundary_index;            // split point in contiguous buffers
};
```

**Key decisions**:
- Biases for fp4 group: since nvfp4 is weight-only quantization, biases remain in bf16 format (same as existing `use_wfp4a16` pattern in the codebase). Reuse `fc1_expert_biases_bf16` / `fc2_expert_biases_bf16` for both groups, indexed by expert ID.
- Weights are stored contiguously **per precision group**: all bf16 expert weights together, all fp4 expert weights together. This is required by groupgemm which expects contiguous weight layout.

---

### Step 1: Sort + Precision Assignment

**Files to modify**:
- `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h` — new function declaration
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` — new kernel implementation

**What exists (reuse as-is)**:
- `fusedBuildExpertMapsSortFirstToken()` — produces `expert_first_token_offset[E+1]` and permutation maps
- `threeStepBuildExpertMapsSortFirstToken()` — fallback 3-kernel pipeline

**What's new**:

```cpp
// New kernel: assign precision based on token load
void sortExpertsByTokenCount(
    int64_t const* expert_first_token_offset,  // [E+1] from existing sort
    int const num_experts_per_node,
    int const num_high_precision_experts,       // user-specified threshold
    int* expert_precision_assignment,           // output: [E] → {0=fp4, 1=bf16}
    int* bf16_expert_indices,                   // output: sorted list
    int* fp4_expert_indices,                    // output: sorted list
    int64_t* boundary_index,                    // output: split point
    cudaStream_t stream
);
```

**Implementation approach**:
1. Compute `tokens_per_expert[e] = expert_first_token_offset[e+1] - expert_first_token_offset[e]` — trivial, E elements
2. Since E is small (typically 8-256), a simple selection kernel suffices:
   - Each thread handles one expert
   - Use `cub::BlockRadixSort` to sort (token_count, expert_id) pairs descending
   - Top `num_high_precision_experts` → bf16 (assignment=1)
   - Remainder → fp4 (assignment=0)
   - Write out `bf16_expert_indices`, `fp4_expert_indices`, `boundary_index`
3. The `boundary_index` is computed as: sum of token counts for all fp4 experts (or bf16 experts), which tells us where the fp4 region starts in the contiguous output buffers

**PDL**: Guard with `cudaGridDependencySynchronize()` / `cudaTriggerProgrammaticLaunchCompletion()` on `__CUDA_ARCH__ >= 900` (existing pattern).

**Inverse mapping requirement**: The finalize step in GEMM2 needs to combine results from both groups back to original token positions. The existing `permuted_row_to_unpermuted_row` and `unpermuted_row_to_permuted_row` maps already provide this. The new precision assignment adds an additional level: within each group, we need a **group-local** row mapping. This can be computed alongside the precision assignment as:
- `bf16_permuted_row_to_global_row[i]` — maps bf16 group row i to global expanded row
- `fp4_permuted_row_to_global_row[i]` — maps fp4 group row i to global expanded row

These are derived from combining `expert_precision_assignment` with the existing permutation.

---

### Step 2: Dual-Buffer Expand Input + Quantization

**Files to modify**:
- `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h` — new function declaration
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` — new kernel implementation

**What exists (reference)**:
- `expandInputRowsKernelLauncher()` — single-precision expand with optional quantization
- Pattern: iterates over `permuted_row_to_unpermuted_row`, copies rows, optionally computes fp4 scaling factors

**What's new**:

```cpp
template <class InputActivationsType>
void expandInputRowsMixedPrecisionKernelLauncher(
    InputActivationsType const* unpermuted_input,  // original input [num_rows, hidden_size]

    // Output buffers — one contiguous region, split by boundary
    __nv_bfloat16* bf16_expanded_output,           // bf16 group output
    __nv_fp4_e2m1* fp4_expanded_output,            // fp4 group output (quantized)

    // Scaling
    float const* unpermuted_scales,
    float* bf16_permuted_scales,
    float* fp4_permuted_scales,
    TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF* fp4_act_sf,  // fp4 act scaling factors

    // Routing
    int const* permuted_row_to_unpermuted_row,
    int const* expert_precision_assignment,         // [E]: 0=fp4, 1=bf16
    int64_t const* expert_first_token_offset,

    // Dimensions
    int64_t const num_rows,
    int64_t const hidden_size,
    int const experts_per_token,
    int const num_experts_per_node,

    // Group row mappings
    int* bf16_row_counter,                          // atomic counter for bf16 rows written
    int* fp4_row_counter,                           // atomic counter for fp4 rows written

    cudaStream_t stream
);
```

**Implementation approach**:
1. Each thread block processes a chunk of expanded rows
2. For each row `i`:
   - Determine which expert it belongs to via `expert_first_token_offset` (binary search or direct lookup)
   - Check `expert_precision_assignment[expert_id]`:
     - If bf16 → copy row to `bf16_expanded_output` at next available bf16 slot (atomic increment)
     - If fp4 → quantize row to nvfp4 and write to `fp4_expanded_output` + compute scaling factors
3. Permuted scales are written to the corresponding group's scale buffer

**Optimization notes**:
- The nvfp4 quantization path is almost fully memory-bound even for small batches (as noted in design doc). Reuse the per-16-group quantization pattern from `quantize_nvfp4_sharedmem()` in `fusedMoeCommKernels.cu`.
- Consider: could the two passes (bf16 copy + fp4 quant) be done as a single pass with warp-level divergence, or is it better as two separate passes? A single pass avoids re-reading `expert_precision_assignment` but introduces warp divergence. Two passes keep warps coherent. **Recommendation**: Start with two passes (simpler, no divergence), optimize later if profiling shows overhead.

**Buffer layout**:
```
  bf16_expanded_output: [max_expanded_rows × hidden_size] in bf16
  fp4_expanded_output:  [max_expanded_rows × hidden_size] in fp4 (sizeof(__nv_fp4_e2m1) per element)
```
Both are **worst-case sized** (all tokens could route to one group) for CUDA graph compatibility. The actual used sizes are determined by `bf16_token_count` and `fp4_token_count` from Step 1.

**Consideration** (from design doc): Whether to store the boundary index inline in the buffer vs. as a separate value:
- **Recommendation**: Separate `int64_t boundary_index` value. It's already computed in Step 1 and used throughout. Embedding it in the buffer adds complexity for no benefit. The scaling factors for fp4 are inherently alongside the fp4 data (8 bytes e2m1 + 1 byte scale per 16-element group).

---

### Step 3: Dual GEMM1

**Files to modify**:
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` — orchestration in `runMixedPrecisionMoe()`

**What exists (reuse)**:
- `CutlassMoeFCRunner::gemm1()` static method — already generic over types
- `MoeGemmRunner::moeGemm()` / `moeGemmBiasAct()` — dispatches to CUTLASS
- `computeStridesTmaWarpSpecialized()` — sets up TMA descriptors

**Approach**:
Since each precision group only activates a subset of experts (never both activate the majority simultaneously, as the design doc notes), we invoke two sequential groupgemms:

```cpp
// 3a. bf16 groupgemm
// Create bf16-specific expert_first_token_offset (only bf16 experts have nonzero counts)
// This requires remapping the offset array for the bf16 subset
Self::gemm1(moe_gemm_runner_bf16_, ...,
    bf16_expanded_input,
    bf16_gemm1_output,
    bf16_expert_first_token_offset,
    fc1_expert_weights_bf16,
    ...);

// 3b. fp4 groupgemm
Self::gemm1(moe_gemm_runner_fp4_, ...,
    fp4_expanded_input,
    fp4_gemm1_output,
    fp4_expert_first_token_offset,
    fc1_expert_weights_fp4,
    ...);
```

**Key detail**: The existing `gemm1()` already supports both bf16 and fp4 weight types via template specialization (see `moe_gemm_kernels_bf16_bf16.cu` and `moe_gemm_kernels_bf16_fp4.cu`). We need **two `MoeGemmRunner` instances** — one for `<bf16, bf16>` and one for `<bf16, fp4>` (or `<fp4, fp4>` depending on activation precision).

**Output**: Both GEMM1 outputs are written to a single contiguous buffer:
```
gemm1_output: [bf16_gemm1_region | fp4_gemm1_region]
```
The boundary is precomputed from Step 1. The fp4 GEMM output is cast to bf16 by the CUTLASS epilogue (existing behavior for nvfp4 — `OutputType` is always `>=16-bit`).

**Short-pass optimization** (deferred): The design doc mentions releasing the threshold to ≤128 tokens for the fp4 groupgemm (min-latency mode). This is a future optimization — for now, use the standard groupgemm path for both. The min-latency path currently `TLLM_THROW("Min latency mode is no longer supported")` anyway.

---

### Step 4: Fused Activation + Quantization

**Files to modify**:
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` — new activation kernel or extension of `doActivation`

**What exists (reference)**:
- `SwigluBiasAdaptor`, `GLUAdaptor` in `moe_kernels.cuh` — activation functors
- Activation is typically fused into GEMM1 epilogue on Hopper (TMA warp-specialized)
- When not fused: `doActivation()` / `doGatedActivation()` separate kernels

**Why fusion is possible**:
Both GEMM1 outputs are in bf16 at this point (fp4 GEMM casts output to bf16 in epilogue). Therefore a **single activation kernel** can process the entire contiguous buffer.

**What's new**:

```cpp
template <class ActivationAdaptor>
__global__ void doMixedPrecisionActivationKernel(
    __nv_bfloat16 const* gemm1_output,      // [total_expanded_rows × inter_size*2] (GLU)
    __nv_bfloat16* activation_output_bf16,   // bf16 region output
    __nv_fp4_e2m1* activation_output_fp4,    // fp4 region output (quantized for gemm2)
    TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF* fp4_gemm2_act_sf,  // fp4 scaling factors
    int64_t const boundary_index,            // split point: rows [0, boundary) = bf16, [boundary, end) = fp4
    int64_t const total_rows,
    int64_t const inter_size,
    ActivationAdaptor activation
);
```

**Implementation approach**:
1. Process the full `gemm1_output` buffer with SwiGLU activation (same as existing `doGatedActivation`)
2. For rows in `[0, boundary_index)` (bf16 group): write activated result in bf16 to `activation_output_bf16`
3. For rows in `[boundary_index, total_rows)` (fp4 group): activate **and quantize** to nvfp4, write to `activation_output_fp4` + scaling factors
4. The quantization for the fp4 group uses the per-16-group pattern from `quantize_nvfp4_sharedmem()`:
   - Compute warp-level absmax
   - Quantize to fp4 with per-group scaling
   - Write packed fp4 data + scale bytes

**Benefits** (from design doc):
- Single kernel launch for activation (instead of two separate activation calls)
- Quantization for gemm2 fp4 input is fused with activation (no separate quant kernel)

**Output buffer layout**:
```
bf16 activation output:  [bf16_token_count × inter_size]
fp4 activation output:   [fp4_token_count × inter_size] in fp4 + scaling factors
```

---

### Step 5: Dual GEMM2 + Finalize

**Files to modify**:
- `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` — orchestration

**What exists (reuse)**:
- `CutlassMoeFCRunner::gemm2()` — supports fused finalize epilogue on Hopper
- `finalizeMoeRoutingKernelLauncher()` — standalone finalize for non-fused path

**Key insight**: The finalize is already per-expert. The output buffer is fixed (`final_output[num_rows × hidden_size]`), and each groupgemm's finalize writes its experts' weighted contributions to the correct positions. The precision of the group doesn't matter since the output is bf16 regardless (fp4 GEMM casts output to bf16 in epilogue). The only differences from single-precision are:

1. **Called twice** — once per precision group, sequentially (no races)
2. **Expert mapping** — each call's finalize must know which experts it operates on (group-local-to-global expert ID mapping)

**Approach**: Call the existing GEMM2 + fused finalize twice, writing to the same `final_output`:

```cpp
// Zero-initialize final_output (both groups accumulate into it)
cudaMemsetAsync(final_output, 0, num_rows * hidden_size * sizeof(bf16), stream);

// 5a. bf16 groupgemm2 + finalize → accumulate bf16 experts' contributions
Self::gemm2(moe_gemm_runner_bf16_, ...,
    bf16_activation_output,
    /*gemm_output=*/bf16_gemm2_scratch,  // intermediate before finalize
    /*final_output=*/final_output,        // finalize writes here
    bf16_expert_first_token_offset,       // only bf16 experts have nonzero counts
    fc2_expert_weights_bf16,
    // permutation maps: group-local row → global unpermuted row
    bf16_unpermuted_row_to_permuted_row,
    bf16_permuted_row_to_unpermuted_row,
    ...);

// 5b. fp4 groupgemm2 + finalize → accumulate fp4 experts' contributions
Self::gemm2(moe_gemm_runner_fp4_, ...,
    fp4_activation_output,
    /*gemm_output=*/fp4_gemm2_scratch,
    /*final_output=*/final_output,        // same output buffer — accumulates
    fp4_expert_first_token_offset,
    fc2_expert_weights_fp4,
    fp4_unpermuted_row_to_permuted_row,
    fp4_permuted_row_to_unpermuted_row,
    ...);
```

**How the existing finalize handles this**:
- The finalize epilogue (or standalone kernel) iterates over `experts_per_token` slots per output row
- For each slot, it looks up `unpermuted_row_to_permuted_row[row * experts_per_token + k]` to find the source expanded row, multiplies by the router scale, and accumulates into `final_output[row]`
- When a group doesn't own a particular expert slot, that expert's token count in the group's `expert_first_token_offset` is zero — no work is done for those slots
- Since the two calls are sequential, the second call's writes to `final_output` don't race with the first

**Per-group expert offset arrays**: Each group needs its own `expert_first_token_offset[E+1]` where only the group's experts have nonzero token counts and the others are zero. This is computed in Step 1 alongside the precision assignment.

**No new finalize kernel needed** — the existing `finalizeMoeRoutingKernelLauncher` or fused finalize epilogue works as-is, provided we pass the correct per-group offset/mapping arrays.

---

## 4. File Change Summary

### New Files

| File | Purpose |
|------|---------|
| (none anticipated — all changes in existing files) | |

### Modified Files

| File | Changes |
|------|---------|
| `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` | Add `runMixedPrecisionMoe()` virtual method to `CutlassMoeFCRunnerInterface`; add `MixedPrecisionMoeState` struct; extend `CutlassMoeFCRunner` with dual `MoeGemmRunner` instances and mixed-precision workspace pointers |
| `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h` | Add `sortExpertsByTokenCount()` declaration; add `expandInputRowsMixedPrecisionKernelLauncher()` declaration |
| `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` | Implement `runMixedPrecisionMoe()`, `sortExpertsByTokenCount()`, `expandInputRowsMixedPrecisionKernelLauncher()`, `doMixedPrecisionActivationKernel()`; add `getWorkspaceSize` overload for mixed precision; add `configureWsPtrs` overload for mixed precision buffers |
| `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cuh` | (Potentially) Add mixed-precision activation adaptor if the fused activation+quantize kernel needs a new functor |
| `cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp` | Add new plugin configuration for mixed-precision mode; pass dual weight pointers |
| `cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h` | Extend plugin interface for mixed-precision arguments |
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py` | Add `compute_mixed_precision_moe()` path; pass dual weight tensors + precision config to C++ |
| `tensorrt_llm/_torch/modules/fused_moe/interface.py` | Extend weight registration for dual-weight storage (bf16 + fp4) |
| `tensorrt_llm/_torch/modules/fused_moe/quantization.py` | Add mixed-precision quantization method that manages dual scale sets |

### Test Files

| File | Purpose |
|------|---------|
| `tests/unittest/_torch/modules/moe/test_mixed_precision_moe.py` | Correctness: compare fused mixed-precision output vs. reference (two separate fused_moe calls); test with varying num_high_precision_experts, token distributions |
| `cpp/tests/unit_tests/kernels/mixtureOfExpertsTest.cu` | Add C++ unit tests for `sortExpertsByTokenCount`, `expandInputRowsMixedPrecision`, mixed-precision `runMoe` |

---

## 5. Buffer / Workspace Management

All buffers must be **preallocated** for CUDA graph compatibility (existing pattern in `getWorkspaceSize()` / `configureWsPtrs()`).

### New Workspace Buffers

| Buffer | Size | Type | Purpose |
|--------|------|------|---------|
| `expert_precision_assignment_` | `E × sizeof(int)` | `int*` | Per-expert precision flag |
| `bf16_expert_indices_` | `E × sizeof(int)` | `int*` | List of bf16 expert IDs |
| `fp4_expert_indices_` | `E × sizeof(int)` | `int*` | List of fp4 expert IDs |
| `bf16_expanded_data_` | `expanded_rows × hidden_size × sizeof(bf16)` | `bf16*` | bf16 group expanded input |
| `fp4_expanded_data_` | `expanded_rows × hidden_size × sizeof(__nv_fp4_e2m1)` | `fp4*` | fp4 group expanded input |
| `fp4_expanded_sf_` | `expanded_rows × hidden_size / 16 × sizeof(ElementSF)` | `ElementSF*` | fp4 scaling factors for expand |
| `bf16_gemm1_output_` | `expanded_rows × inter_size × sizeof(bf16)` | `bf16*` | bf16 GEMM1 output |
| `fp4_gemm1_output_` | `expanded_rows × inter_size × sizeof(bf16)` | `bf16*` | fp4 GEMM1 output (bf16 after epilogue cast) |
| `bf16_activation_output_` | `expanded_rows × inter_size × sizeof(bf16)` | `bf16*` | bf16 group activation output |
| `fp4_activation_output_` | `expanded_rows × inter_size × sizeof(__nv_fp4_e2m1)` | `fp4*` | fp4 group activation output (quantized) |
| `fp4_gemm2_act_sf_` | `expanded_rows × inter_size / 16 × sizeof(ElementSF)` | `ElementSF*` | fp4 gemm2 act scaling factors |
| `bf16_gemm2_scratch_` | `expanded_rows × hidden_size × sizeof(bf16)` | `bf16*` | bf16 GEMM2 scratch (before finalize writes to final_output) |
| `fp4_gemm2_scratch_` | `expanded_rows × hidden_size × sizeof(bf16)` | `bf16*` | fp4 GEMM2 scratch (before finalize writes to final_output) |
| `bf16_expert_first_token_offset_` | `(E+1) × sizeof(int64_t)` | `int64_t*` | bf16 group offset array |
| `fp4_expert_first_token_offset_` | `(E+1) × sizeof(int64_t)` | `int64_t*` | fp4 group offset array |
| `bf16_permuted_scales_` | `expanded_rows × sizeof(float)` | `float*` | bf16 group permuted router scales |
| `fp4_permuted_scales_` | `expanded_rows × sizeof(float)` | `float*` | fp4 group permuted router scales |
| `boundary_index_` | `sizeof(int64_t)` | `int64_t*` | Split point between groups |

**Note**: `expanded_rows = num_rows × experts_per_token` (worst-case). Many buffers can be aliased/shared if lifetimes don't overlap (e.g., `bf16_expanded_data_` can be reused as `bf16_activation_output_` since they don't overlap in time).

**GLU intermediate**: The existing `glu_inter_result_` buffer is needed for both groups' GEMM1. Since the groups run sequentially, this buffer can be shared.

---

## 6. Implementation Order

### Phase 1: Core Kernel (C++)

| Order | Task | Complexity | Dependencies |
|-------|------|------------|--------------|
| 1.1 | `sortExpertsByTokenCount()` kernel | Low | None — standalone kernel, E is small |
| 1.2 | `expandInputRowsMixedPrecisionKernelLauncher()` | Medium | Step 1.1 output; reuse `expandInputRowsKernel` pattern |
| 1.3 | Dual `gemm1()` orchestration | Low | Step 1.2 buffers; reuse existing `gemm1()` — just call it twice with different weights/buffers |
| 1.4 | `doMixedPrecisionActivationKernel()` | Medium | Step 1.3 output; extend existing activation kernel with fused fp4 quantization |
| 1.5 | Dual `gemm2()` + finalize orchestration | Low | Step 1.4 output; call existing `gemm2()` twice with per-group offset arrays, both write to `final_output` |
| 1.6 | Workspace management (`getWorkspaceSize` + `configureWsPtrs`) | Medium | All buffer sizes finalized |
| 1.7 | `runMixedPrecisionMoe()` orchestrator | Low | Steps 1.1–1.6 composed |

### Phase 2: Integration (Python + Plugin)

| Order | Task | Complexity | Dependencies |
|-------|------|------------|--------------|
| 2.1 | Plugin interface extension | Medium | Phase 1 complete |
| 2.2 | Python torch op registration | Medium | Phase 1 complete |
| 2.3 | `fused_moe_cutlass.py` mixed-precision path | Medium | Step 2.2 |
| 2.4 | Weight management (dual-weight loading) | Medium | Step 2.3 |

### Phase 3: Testing & Optimization

| Order | Task | Complexity | Dependencies |
|-------|------|------------|--------------|
| 3.1 | C++ unit tests | Medium | Phase 1 |
| 3.2 | Python correctness tests | Medium | Phase 2 |
| 3.3 | Benchmark vs. separate-call HeterMoE | Low | Phase 2 |
| 3.4 | Buffer aliasing optimization | Low | Phase 3.1 passing |
| 3.5 | Short-pass optimization for fp4 (deferred) | High | Phase 3.3 data |
| 3.6 | Short-pass optimization for fp4 (deferred) | High | Phase 3.3 data |

---

## 7. Cross-Cutting Concerns

### 7.1 CUDA Graph Compatibility
- **All buffers preallocated** in `getWorkspaceSize()` — no per-call mallocs
- **No host-device sync** — precision assignment is fully on-device
- **Atomic counters** for dual-buffer expand must be reset to 0 each call (use `cudaMemsetAsync`)

### 7.2 PDL (Programmatic Dependency Launch)
- All new kernels must include PDL guards:
  ```cuda
  #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      cudaGridDependencySynchronize();
      // ... kernel body ...
      cudaTriggerProgrammaticLaunchCompletion();
  #endif
  ```
- Use `cudaLaunchKernelEx` with `cudaLaunchAttributeProgrammaticStreamSerialization` (see `computeStridesTmaWarpSpecialized()` pattern at line ~3960 of `moe_kernels.cu`)
- Check `tensorrt_llm::common::getEnvEnablePDL()` for runtime control

### 7.3 Compilation Compatibility
- Use `constexpr` type checks and `if constexpr` for precision-specific code paths
- Follow existing pattern: `static constexpr bool use_fp4 = ...` (moe_kernels.h line ~575)
- Template the `CutlassMoeFCRunner` for mixed precision — may need a new specialization or a runtime dispatch

### 7.4 torch.compile Compatibility
- All CUDA operations must go through registered torch ops
- No Python-level control flow that depends on tensor values (precision assignment is GPU-only)
- Workspace sizes are deterministic from static parameters

---

## 8. Open Questions & Risks

| # | Question | Impact | Proposed Resolution |
|---|----------|--------|---------------------|
| 1 | **Two MoeGemmRunner instances**: The `CutlassMoeFCRunner` currently holds one `moe_gemm_runner_`. Mixed precision needs two (one for bf16×bf16, one for bf16×fp4). How to manage tactic selection for both? | Medium | Add a second `moe_gemm_runner_fp4_` member. Tactic profiling runs independently for each. |
| 2 | **Expert ID remapping**: Groupgemm expects contiguous expert indices [0, N). When we split into two groups, each group needs its own [0, N_group) indexing. How does this interact with `expert_first_token_offset`? | High | Build per-group offset arrays in `sortExpertsByTokenCount()`. Map global expert IDs to group-local IDs. |
| 3 | **Finalize across groups**: When `experts_per_token > 1`, a single token may have experts in both groups. The finalize must sum contributions from both GEMM2 outputs. | Low | Not an issue — each groupgemm's finalize writes its experts' contributions to `final_output` sequentially. Zero-init output, both calls accumulate. No new kernel needed. |
| 4 | **LoRA support**: The design doc doesn't mention LoRA. Should mixed-precision MoE support LoRA? | Low | Initially no. Add `TLLM_CHECK(!use_lora)` in `runMixedPrecisionMoe`. |
| 5 | **Expert parallelism (EP)**: With EP, each node handles a subset of experts. Mixed precision assignment must be per-node. | Medium | `sortExpertsByTokenCount` operates on `num_experts_per_node` (not `full_num_experts`), same as existing sort functions. |
| 6 | **Memory overhead**: Storing two full weight sets doubles weight memory. Is there a way to reduce? | Low (accepted) | This is by design — the design doc explicitly states "we simply keep two versions of model weight in memory." The trade-off is memory for quality+speed. |
| 7 | **Dynamic precision reassignment**: Token distribution changes throughout serving. How often should precision assignment be recomputed? | Low | Every `runMixedPrecisionMoe` call recomputes assignment based on current `expert_first_token_offset`. No caching needed — the sort kernel is cheap (E elements). |
| 8 | **Workspace size increase**: The dual-buffer approach roughly doubles workspace memory. | Medium | Buffer aliasing (Phase 3.4) can significantly reduce this. Sequential GEMM execution means many buffers have non-overlapping lifetimes. |

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **AI (Arithmetic Intensity)** | Ratio of compute operations to memory bytes accessed. Higher AI → more compute-bound, benefits from higher precision. |
| **Hot expert** | Expert receiving many tokens in the current batch → high AI |
| **Cold expert** | Expert receiving few tokens → low AI → diminishing returns from bf16 precision |
| **NVFP4** | NVIDIA FP4 (e2m1) format — 4-bit floating point with per-16-element block scaling |
| **GroupGEMM** | Grouped GEMM — batched matrix multiply where each "group" (expert) has different dimensions/counts |
| **PDL** | Programmatic Dependency Launch — CUDA 12+ feature for fine-grained kernel dependencies |
| **Fused finalize** | Combining expert outputs weighted by router scores in the GEMM2 epilogue (Hopper optimization) |
| **Short pass** | Min-latency code path for sparsely activated MoE (≤128 tokens total) |
| **TMA** | Tensor Memory Accelerator — Hopper hardware feature for efficient memory copies |

---

## 10. Target Hardware & Testing

| Hardware | Role |
|----------|------|
| RTX 5070 | Development — single-layer simulation (limited by memory) |
| RTX 6000 Pro | Evaluation benchmark — full model testing (deferred until implementation complete) |

**Testing strategy**: Implementation first, then benchmark on RTX 6000 Pro. Unit tests on available GPU.

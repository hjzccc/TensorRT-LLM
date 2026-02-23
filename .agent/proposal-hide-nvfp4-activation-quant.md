# Proposal: Hide NVFP4 Activation Quantization in Heterogeneous MoE

**Author**: huanchen  
**Date**: 2026-02-22  
**Status**: Draft  
**Scope**: `fused_moe_heter.py`, `torch_custom_ops.py`, C++ quantization kernels, CUTLASS epilogue
**Related**: [Bucket-Aware GroupGEMM Tactics](./proposal-bucket-aware-groupgemm-tactics.md)

---

## 1. Problem

### Current Flow

In heterogeneous MoE, the BF16 precision group and NVFP4 precision group
run **sequentially**. Before the NVFP4 GroupGEMM can start, activations
must be quantized from BF16 → FP4:

```
run_moe():
  for group_idx, (grp_experts, grp_scales) in dispatches:
      qi = _quantize_input_for_group(x, desc.quant_algo, ws)   ← BLOCKING
      group_result = torch.ops.trtllm.fused_moe(qi.x, ...)     ← GroupGEMM
      accumulated += group_result
```

For the NVFP4 group, `_quantize_input_for_group` calls:
```python
qx, qx_sf = torch.ops.trtllm.fp4_quantize(
    x,                          # BF16 input [N, hidden_dim]
    weight_set.fc31_input_scale, # global scale factor
    weight_set.scaling_vector_size,  # 16 for NVFP4
    False,                      # input sf not swizzled
    True,                       # output sf swizzled
)
```

### What the Kernel Does

**Location**: `cpp/tensorrt_llm/kernels/quantization.cu` → `invokeFP4Quantization()`

The kernel (`quantize_with_block_size<FP16_TO_FP4>`) does:
1. Read BF16 values (8 elements per thread)
2. Compute block-level scale factors (per 16 elements)
3. Quantize BF16 → FP4 E2M1 (pack 2 values per byte)
4. Swizzle scale factors into the layout expected by CUTLASS GEMM

**This is the same generic kernel used for weight quantization.** It is NOT
optimized for the activation quantization use case (latency-sensitive,
smaller M dimension, could be overlapped).

### Cost Breakdown

For a typical heter MoE inference step (e.g., N=128 tokens, hidden=7168):
- `fp4_quantize`: ~15–40 µs (memory-bound: read N×K BF16, write N×K/2 FP4 + scales)
- BF16 GroupGEMM (GEMM1): ~100–500 µs (compute-bound for high-traffic experts)
- NVFP4 GroupGEMM (GEMM1): ~50–200 µs

The FP4 quantization sits on the **critical path** between the BF16 group's
output accumulation and the NVFP4 group's GEMM1 start. It is entirely
memory-bound and could run concurrently with compute.

---

## 2. Investigation: Current Quantization Path

### ⚠️ FIRST STEP: Verify Kernel Identity

**Before implementing any optimization, confirm the hypothesis that the
activation quantization kernel is suboptimal.**

Check items:

| # | Question | How to Check | File |
|---|----------|-------------|------|
| 1 | Is `fp4_quantize` really the same kernel for weights and activations? | Compare call paths: `_quantize_input_for_group()` vs weight loading in `post_load_weights()` | `fused_moe_heter.py`, `fused_moe_cutlass.py` |
| 2 | Is the swizzle step fused or separate? | Check if `isSfSwizzledLayout=True` does swizzle in-kernel or needs a second pass | `quantization.cu` line 90–91 |
| 3 | What fraction of MoE layer time does quantization take? | Profile with `TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO=1` or nsys | Runtime profiling |
| 4 | Does the CuTE DSL backend's fused `moe_swiglu_nvfp4_quantize` apply here? | Check if CUTLASS backend could use a similar fused kernel | `cuteDslMoeUtilsOp.cpp` line 387 |
| 5 | Is the global scale (`fc31_input_scale`) precomputed or dynamic? | Check if it's a static parameter or recomputed per-step | `fused_moe_heter.py`, weight loading |

### Current Kernel Details

```
File: cpp/tensorrt_llm/kernels/quantization.cu
Function: invokeFP4Quantization<__nv_bfloat16, 16>()

Grid:  min(m_padded, SM_count * blocks_per_SM)
Block: min(K / 8, 512)  ← 8 elements per thread
Shared memory: none
Swizzle: fused when isSfSwizzledLayout=true (line 90)

Input:  [M, K] BF16        (M=num_tokens, K=hidden_dim)
Output: [M, K/2] FP4_E2M1  (packed 2-per-byte)
Scales: swizzled 1D tensor  (padded to 128-row blocks)
```

**Key observation**: The kernel IS memory-bound — it reads K BF16 values
and writes K/2 FP4 values per row. No reduction across rows. Each row is
independent. This makes it a perfect candidate for overlapping with compute.

### Existing Fused Kernel (CuTE DSL Only)

The CuTE DSL MoE backend already has `moe_swiglu_nvfp4_quantize`
(`cuteDslMoeUtilsOp.cpp:387`) that fuses:
- SwiGLU activation (gate * up, with swish)
- BF16 → FP4 quantization
- Scale factor computation

into a single kernel. This is used **between GEMM1 and GEMM2** in the
CuTE DSL path. The CUTLASS backend does NOT use this — it has a separate
activation + separate quantization.

---

## 3. Proposed Solutions (Three Options)

### Option A: Overlap FP4 Quantize with BF16 GroupGEMM (Aux Stream)

**Lowest effort, highest immediate impact.**

The BF16 group's GroupGEMM (GEMM1+activation+GEMM2) is compute-bound.
The FP4 quantization is memory-bound. They access different memory regions.
Run them concurrently on separate CUDA streams.

```
Current (sequential):
  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐
  │ BF16 GroupGEMM│  │ FP4 Quantize│  │ FP4 GroupGEMM  │
  └──────────────┘  └─────────────┘  └────────────────┘
  ████████████████  ████             ████████████████
                    ↑ critical path overhead

Proposed (overlapped):
  Stream 0: ┌──────────────┐          ┌────────────────┐
            │ BF16 GroupGEMM│  ─sync─  │ FP4 GroupGEMM  │
            └──────────────┘          └────────────────┘
  Stream 1:     ┌─────────────┐
                │ FP4 Quantize│
                └─────────────┘
                ↑ launched early, hidden behind BF16 GEMM compute
```

**Implementation:**

```python
# fused_moe_heter.py :: run_moe()
bf16_group_idx = ...  # group with quant_algo=None
fp4_group_idx = ...   # group with quant_algo=NVFP4

# 1. Launch BF16 GroupGEMM on main stream
bf16_result = torch.ops.trtllm.fused_moe(x, bf16_experts, ...)  # main stream

# 2. Concurrently: launch FP4 quantization on aux stream
fp4_quant_event = torch.cuda.Event()
with torch.cuda.stream(self.aux_stream):
    # Wait for x to be ready (it's the attention output, already computed)
    qi = _quantize_input_for_group(x, NVFP4, ws)  # aux stream
    fp4_quant_event.record()  # signal: FP4 input is ready

# 3. Main stream waits for FP4 quantization to finish
torch.cuda.current_stream().wait_event(fp4_quant_event)

# 4. Launch FP4 GroupGEMM on main stream
fp4_result = torch.ops.trtllm.fused_moe(qi.x, fp4_experts, ...)

accumulated = bf16_result + fp4_result
```

**Constraints:**
- Requires `x` (BF16 input) to be available before both groups start —
  it IS, since `x` is the attention output passed into `run_moe()`.
- The aux stream infrastructure already exists: `AuxStreamType.MoeChunkingOverlap`
  is used by `configurable_moe.py`, `fused_moe_cutlass.py`, etc.
- CUDA graph compatible: stream events are capturable.

**Expected latency savings**: ~15–40 µs per MoE layer (the full FP4
quantization cost, minus any memory bandwidth contention).

### Option B: Fuse Activation + FP4 Quantize for CUTLASS Backend

**Medium effort, applicable between GEMM1 and GEMM2.**

The CuTE DSL backend already has `moe_swiglu_nvfp4_quantize` that
fuses SwiGLU + FP4 quant into one kernel. The CUTLASS backend does NOT
have this — it runs activation and quantization as separate steps inside
the C++ `FusedMoeRunner::run_moe()`.

However, this optimization applies to the **inter-GEMM quantization**
(between GEMM1 output and GEMM2 input), NOT the **input quantization**
we're discussing. In the CUTLASS backend, GEMM1 already produces FP4
output internally when the weight dtype is FP4 — the activation+quant
fusion is handled inside the CUTLASS kernel's epilogue.

**Relevance to our problem**: This is about the **input** to GEMM1.
The input comes from attention as BF16 and must be quantized to FP4
before GEMM1 can start. Option B doesn't directly help here unless we
can fuse the quantization into the GEMM1 prologue (which CUTLASS doesn't
support).

**Verdict**: Option B helps the inter-GEMM path but NOT the input quant
path. Worth investigating separately.

### Option C: Hide in BF16 GEMM Epilogue (Custom Epilogue Fusion)

**High effort, maximum theoretical benefit.**

The BF16 GroupGEMM's epilogue writes BF16 results back to global memory.
During the first few loads of the NEXT GroupGEMM (FP4), the pipeline is
memory-bound. We could extend the BF16 GEMM's epilogue to simultaneously:
1. Write BF16 output
2. Read BF16 input (x) and quantize to FP4
3. Write FP4 quantized output

This exploits the fact that the BF16 epilogue has **spare compute during
memory stores**, and the FP4 quantization only needs **memory reads +
simple arithmetic**.

**Implementation path:**
- Extend `EpilogueFusionType` in `gemm_configs.h` (currently only `NONE`
  and `FINALIZE`) to add `QUANTIZE_FP4`
- Write a custom CUTLASS epilogue visitor that:
  a. Stores the GEMM output (normal epilogue)
  b. Reads a separate input tensor
  c. Quantizes it to FP4 with block scaling
  d. Writes the FP4 output + swizzled scales
- Wire this through `moe_kernels.cu` as a new epilogue option

**Pros**: Zero additional kernel launch, truly hidden latency.
**Cons**: Deep CUTLASS surgery, SM90/SM100 epilogue visitors are complex,
maintains two code paths, increases binary size.

**Verdict**: Phase 3 optimization. Only pursue if Option A proves
insufficient and profiling shows the quant overhead is >5% of MoE time.

---

## 4. Recommended Approach: Option A First

### Why Option A Wins

| Criteria | Option A (Overlap) | Option B (Fused Kernel) | Option C (Epilogue) |
|----------|-------------------|-------------------------|---------------------|
| Effort | Small (Python only) | Medium (C++ kernel) | Very Large (CUTLASS) |
| Latency hidden | ~100% of quant cost | N/A for input quant | ~100% of quant cost |
| C++ changes | None | Port CuTE DSL kernel | Deep CUTLASS surgery |
| CUDA graph safe | Yes (events) | Yes | Yes |
| Risk | Low (proven pattern) | Medium | High |

The aux stream pattern is **already used throughout the MoE codebase**:
- `configurable_moe.py` uses `AuxStreamType.MoeChunkingOverlap` for
  chunked execution overlap
- `fused_moe_cutlass.py`, `fused_moe_deepgemm.py`, `fused_moe_wide_ep.py`
  all have `self.aux_stream` for multi-stream overlap
- `moe_load_balancer.py` uses aux streams for stats computation overlap

Adding another aux stream usage in `fused_moe_heter.py` follows an
established pattern.

---

## 5. Implementation Plan

### Phase 1: Overlap FP4 Quantize with BF16 GroupGEMM

| Step | Description | Files | Effort |
|------|-------------|-------|--------|
| 1.0  | **Profile baseline**: measure `fp4_quantize` latency vs GroupGEMM latency with nsys | Runtime | Small |
| 1.1  | Add `aux_stream` to `HeterCutlassFusedMoE.__init__()` (from `aux_stream_dict`) | `fused_moe_heter.py` | Small |
| 1.2  | Reorder `run_moe()` loop: launch BF16 group first, start FP4 quant on aux stream concurrently | `fused_moe_heter.py` | Medium |
| 1.3  | Add CUDA event synchronization between aux stream quant and main stream FP4 GEMM | `fused_moe_heter.py` | Small |
| 1.4  | Handle >2 groups: generalize to pipeline quant(group N+1) while GEMM(group N) runs | `fused_moe_heter.py` | Medium |
| 1.5  | CUDA graph compatibility: verify stream events capture correctly | `fused_moe_heter.py` | Medium |
| 1.6  | Benchmark: latency comparison with/without overlap | benchmarks | Small |

### Phase 2: Investigate Fused Activation Quantize for CUTLASS Backend

| Step | Description | Files | Effort |
|------|-------------|-------|--------|
| 2.1  | Profile inter-GEMM quantization (between GEMM1→GEMM2) in CUTLASS path | Runtime | Small |
| 2.2  | Evaluate porting `moe_swiglu_nvfp4_quantize` from CuTE DSL to CUTLASS backend | `moe_kernels.cu`, `moeOp.cpp` | Large |
| 2.3  | If applicable: fuse SwiGLU + FP4 quant between GEMM1 and GEMM2 | `moe_kernels.cu` | Large |

### Phase 3: CUTLASS Epilogue Fusion (Only if Needed)

| Step | Description | Files | Effort |
|------|-------------|-------|--------|
| 3.1  | Extend `EpilogueFusionType` to support `QUANTIZE_FP4` | `gemm_configs.h` | Small |
| 3.2  | Write custom CUTLASS epilogue visitor for FP4 quantization | New file in `cutlass_extensions/` | Very Large |
| 3.3  | Wire through tactic selection and profiling | `moe_kernels.cu`, `moeOp.cpp` | Large |

---

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory bandwidth contention between BF16 GEMM and FP4 quant | Reduced overlap benefit | Profile: FP4 quant is small (reads N×K BF16 only); BF16 GEMM reads N×K×2 weights → different memory regions |
| FP4 quantization depends on `x` which might be modified | Correctness bug | `x` is the attention output, read-only in MoE; BF16 GEMM reads `x` too → no conflict |
| CUDA graph capture with multi-stream events | Graph capture failure | Test explicitly; existing MoE code already uses this pattern successfully |
| Group ordering assumption (BF16 first, FP4 second) | Breaks if groups are reordered | Assert group ordering or sort groups by quant cost (BF16 < FP4) |
| >2 precision groups | Overlap becomes complex | Generalize: quant(group_i+1) overlaps with GEMM(group_i) in a pipeline |

---

## 7. Expected Impact

### Latency Savings

For DeepSeek-V3 heterogeneous MoE (50% BF16, 50% NVFP4 experts):

| Component | Baseline | With Overlap | Savings |
|-----------|----------|-------------|---------|
| FP4 input quantize | 15–40 µs | ~0 µs (hidden) | 15–40 µs |
| BF16 GroupGEMM | 100–500 µs | unchanged | — |
| FP4 GroupGEMM | 50–200 µs | unchanged | — |
| **Total per layer** | **165–740 µs** | **150–700 µs** | **~5–10%** |

Combined with [Bucket-Aware Tactics](./proposal-bucket-aware-groupgemm-tactics.md):
- Bucket-aware tactics: **15–30%** reduction in GroupGEMM time
- FP4 quant overlap: **5–10%** reduction in total MoE time
- **Combined: 20–35% MoE layer latency reduction**

---

## 8. Key File References

| File | Role |
|------|------|
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_heter.py` | `run_moe()` — main modification point; `_quantize_input_for_group()` at line 110 |
| `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` | `fused_moe` custom op — `fp4_quantize` call at line 141 |
| `tensorrt_llm/_torch/utils.py` | `Fp4QuantizedTensor` dataclass (line 143); `swizzle_sf()` (line 159) |
| `cpp/tensorrt_llm/thop/fp4Quantize.cpp` | C++ wrapper for `invokeFP4Quantization` — takes [M,K] BF16, returns [M,K/2] FP4 + swizzled scales |
| `cpp/tensorrt_llm/kernels/quantization.cu` | CUDA kernel: `quantize_with_block_size<FP16_TO_FP4>` — 8 elements/thread, memory-bound, no shared memory |
| `cpp/tensorrt_llm/thop/cuteDslMoeUtilsOp.cpp` | Reference: `moe_swiglu_nvfp4_quantize` — fused activation+FP4 quant (CuTE DSL only) |
| `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h` | `EpilogueFusionType` — currently only `NONE` and `FINALIZE` |
| `tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py` | Reference: existing aux_stream usage pattern for MoE overlap |

---

## 9. Open Questions

1. **Global scale factor (`fc31_input_scale`)**: Is this static (per-layer) or
   dynamic (per-step)? If static, the quant could even be precomputed for
   known input shapes.
2. **Swizzle cost**: What fraction of `fp4_quantize` time is the scale factor
   swizzle vs the actual quantization? If swizzle dominates, a non-swizzled
   quantize + async swizzle might be better.
3. **Multiple MoE layers**: With pipeline parallelism, can we quant layer N+1's
   FP4 input while layer N's FP4 GEMM runs?
4. **Interaction with bucket-aware tactics**: If we split each precision group
   into sub-GroupGEMMs (from the other proposal), the overlap becomes:
   `quant(FP4) || [BF16_bucket_0, BF16_bucket_1, ...]` — even more compute
   to hide behind.
5. **Is `fp4_quantize` the bottleneck at all?** Must profile first (Step 1.0).
   If it's <5 µs, the overlap infrastructure isn't worth it.
# SM120 Small M-Tile CTA Sizes for NVFP4 MoE Grouped GEMM

## Problem

Mixture-of-Experts (MoE) models route each token to a subset of experts via top-k selection. The grouped GEMM kernel processes all experts in a single launch, where each expert's GEMM has M = number of tokens routed to that expert. With many experts and moderate batch sizes, per-expert M is small:

```
per-expert M ≈ total_tokens × top_k / num_experts
```

For Qwen3-30B-A3B (128 experts, top_k=8), a batch of 256 tokens yields per-expert M ≈ 16 on average. The default SM120 CTA tile uses M=128 rows, meaning 112 of 128 rows are zero-padded — 87.5% wasted compute.

## Solution

Add M=64 and M=32 CTA tile sizes for SM120 NVFP4 grouped GEMM kernels. The runtime auto-tuner profiles all available tile sizes and selects the fastest for the current workload. This eliminates the need for manual tile selection or the dual-tile mechanism.

### Kernel-level results (RTX 5090, Qwen3-30B-A3B dimensions, 8 experts)

| Tokens | M128    | M64     | M32     | M64 speedup | M32 speedup |
|--------|---------|---------|---------|-------------|-------------|
| 4      | 0.037ms | 0.029ms | 0.026ms | 1.28x       | 1.40x       |
| 16     | 0.035ms | 0.028ms | 0.026ms | 1.25x       | 1.34x       |
| 64     | 0.035ms | 0.027ms | 0.027ms | 1.26x       | 1.30x       |
| 128    | 0.035ms | 0.029ms | 0.034ms | 1.18x       | 1.02x       |
| 256    | 0.043ms | 0.040ms | 0.047ms | 1.07x       | 0.92x       |

M32 wins at per-expert M ≤ 64 (30-40% kernel speedup). M64 wins at per-expert M ≤ 128 (18-28% kernel speedup). M128 only wins when per-expert M > 128.

### Auto-tuner integration

The runtime auto-tuner already profiles all tile sizes (M32, M64, M128) and selects the best one per-call. Instrumentation confirms the tuner picks M32 for decode-phase workloads across all batch sizes tested (1–512):

```
[TACTIC] gemm1=2/6, gemm2=5/12, num_tokens=128   # tactic 2 = M32
[TACTIC] gemm1=2/6, gemm2=5/12, num_tokens=64
[TACTIC] gemm1=2/6, gemm2=5/12, num_tokens=32
[TACTIC] gemm1=3/6, gemm2=3/12, num_tokens=8192  # tactic 3 = M128-swap (large prefill)
```

### End-to-end impact

E2E benchmarks on full Qwen3-30B-A3B show negligible overall speedup (<1%) because MoE kernels are a small fraction of total inference time (attention, layer norm, and embedding dominate). The dual-tile mechanism (running two GEMM passes with different tile sizes) adds overhead that offsets tile-size savings. The single-tile auto-tuner path is the recommended approach.

## Architecture Background

### SM120 MMA Atom for NVFP4

The fundamental compute unit for SM120 FP4 is the MMA atom:

```
MMA Atom Shape: M=16, N=8, K=64 (elements)
AtomLayoutMNK (Pingpong): Shape<_2, _2, _1>
→ Effective atom block: M=32, N=16, K=64
```

The Pingpong schedule groups 2 atoms in the M direction, making M=32 the architectural minimum. The cooperative schedule uses `Shape<_4, _2, _1>`, requiring M=64 minimum. M=16 is not feasible on SM120.

### Scale Factor A (SFA) Layout

NVFP4 uses block-scaled quantization with a dual-granularity SFA system:

```
Blk_MN  = 128    (rows per SFA block in GMEM)
Blk_SF  = 4      (scale factor groups per block)
Blk_rows_per_SF = 32  (rows per scale factor group)
SFVecSize = 16   (elements per scale factor)
```

TMA always loads 128-row SFA blocks (matching GMEM layout). The MMA operates on TileM-sized fragments. For M < 128, these two layouts differ — the core challenge of this work.

## Changes

### 1. Dual SFA Layout in CUTLASS Builder

**File**: `cutlass-src/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`

For M < 128, the TMA SFA layout (128-row blocks) and MMA SFA layout (TileM-row fragments) must coexist. We compute separate layout atoms and pass both via a 3-tuple:

```cpp
// TileM_SFA: round up to 128 for TMA
static constexpr int TileM_SFA = ceil_div(TileM_actual, Blk_MN) * Blk_MN;

// MMA SFA dimensions based on actual tile size
static constexpr int MMA_SFA_inner = min(Blk_SF, TileM_actual / Blk_rows_per_SF);
static constexpr int MMA_SFA_outer = max(1, TileM_actual / Blk_MN);

// 3-tuple: (SmemLayoutAtomA, SmemLayoutAtomSFA_TMA, SmemLayoutAtomSFA_MMA)
using SmemLayoutAtomsA = cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}, SmemLayoutAtomSFA_MMA{});
```

For M=128: TileM_SFA=128, MMA_SFA_inner=4, MMA_SFA_outer=1 (degenerate case, TMA=MMA).
For M=64: TileM_SFA=128, MMA_SFA_inner=2, MMA_SFA_outer=1.
For M=32: TileM_SFA=128, MMA_SFA_inner=1, MMA_SFA_outer=1.

### 2. Dual SFA Tensor in CUTLASS Collective

**File**: `cutlass-src/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`

The collective MMA implementation extracts the MMA SFA layout from the 3-tuple and creates a separate `sSFA_mma` tensor for MMA fragment partitioning:

- `SmemLayoutAtomSFA_MMA = get<2>(SmemLayoutAtomsA{})` — MMA-sized SFA layout
- `SmemLayoutSFA_MMA` — adds pipeline stages using TMA's stage stride for correct indexing
- `TileShapeSFA = Shape<Int<TileM_SFA>, TileN, TileK>` — always 128 in M for TMA descriptors
- `SFA_M_ratio = TileM_SFA / TileM_actual` — coordinate conversion in `load()`
- MMA uses `sSFA_mma` tensor for `partition_fragment_SFA` and `smem_tiled_copy_SFA`
- TMA uses `sSFA` tensor (original 128-row layout) for loads and SMEM allocation

### 3. Kernel Schedule Selection

**File**: `moe_gemm_tma_ws_launcher.inl`

For M < 128, we force Pingpong schedule (which supports AtomLayout `Shape<_2, _2, _1>` with minimum M=32):

```cpp
using KernelScheduleSM120 = std::conditional_t<(CTA_M_ < 128),
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
    cutlass::gemm::collective::KernelScheduleAuto>;
```

### 4. Epilogue Tile Override

**File**: `moe_gemm_tma_ws_launcher.inl`

SM120's default epilogue tile `Shape<_64, _32>` causes a static assert failure when CTA_M < 64 (since 32 % 64 ≠ 0). We override for small tiles:

```cpp
using EpilogueSubTile = std::conditional_t<IsSM120 && (CTA_M_ < 64),
    cute::Shape<cute::Int<CTA_M_>, cute::_32>,
    cutlass::epilogue::collective::EpilogueTileAuto>;
```

### 5. Tile Config Registration

**File**: `gemm_configs.h`

New enum values using the existing `shape_tuple_to_enum` macro:

```cpp
CtaShape64x128x64B = shape_tuple_to_enum(64, 128, 64),
CtaShape32x128x64B = shape_tuple_to_enum(32, 128, 64),
```

Note: enum names use bytes for K dimension. FP4 packs 2 elements per byte, so `64B` = 128 elements.

### 6. Candidate Config Lists

**File**: `cutlass_heuristic.cpp`

Both FAST_BUILD and non-FAST_BUILD paths include M64 and M32 for SM120 grouped GEMM:

```cpp
// FAST_BUILD: 3 configs
{CtaShape128x128x128B, CtaShape64x128x64B, CtaShape32x128x64B}

// Full build: 6 configs
{CtaShape128x128x128B, CtaShape128x128x64B, CtaShape128x256x64B,
 CtaShape256x128x64B, CtaShape64x128x64B, CtaShape32x128x64B}
```

### 7. FAST_BUILD Filter

**File**: `cutlass_heuristic.h`

The compile-time filter allows M64 and M32 tiles through:

```cpp
using SupportedCtaShape64 = cute::Shape<cute::_64, cute::_128, decltype(cute::get<2>(TileShape{}))>;
using SupportedCtaShape32 = cute::Shape<cute::_32, cute::_128, decltype(cute::get<2>(TileShape{}))>;

constexpr static bool is_supported_tile = is_same_v<SupportedCtaShape128, TileShape>
    || is_same_v<SupportedCtaShape64, TileShape>
    || is_same_v<SupportedCtaShape32, TileShape>;
```

### 8. Dispatch Switch Cases

**File**: `moe_gemm_template_dispatch_tma_ws.h`

New SHAPE_CASE entries for SM120:

```cpp
SHAPE_CASE(120, 64, 128, 64)
SHAPE_CASE(120, 32, 128, 64)
```

### 9. Kernel Code Generation

**File**: `generate_kernels.py`

Added M64 and M32 to SM120 FP4 grouped GEMM shapes:

```python
cta_shapes_mnk = [[128, 128, 128], [128, 128, 256], [256, 128, 128],
                  [128, 256, 128], [64, 128, 128], [32, 128, 128]]
```

After running `generate_kernels.py`, two new .cu files are produced:
- `cutlass_kernel_file_gemm_grouped_sm120_M64_BS_group0.generated.cu`
- `cutlass_kernel_file_gemm_grouped_sm120_M32_BS_group0.generated.cu`

## File Summary

### TRT-LLM files (git-tracked)

| File | Change |
|------|--------|
| `gemm_configs.h` | `CtaShape64x128x64B`, `CtaShape32x128x64B` enums |
| `cutlass_heuristic.h` | `SupportedCtaShape64`, `SupportedCtaShape32` FAST_BUILD filters |
| `cutlass_heuristic.cpp` | M64/M32 in FAST_BUILD and full candidate lists |
| `moe_gemm_template_dispatch_tma_ws.h` | SHAPE_CASE entries for SM120 M64/M32 |
| `moe_gemm_tma_ws_launcher.inl` | Pingpong schedule for M<128, EpilogueSubTile for M<64 |
| `generate_kernels.py` | `[64, 128, 128]` and `[32, 128, 128]` shapes |

### CUTLASS headers (not git-tracked, reset by cmake reconfigure)

| File | Change |
|------|--------|
| `sm120_blockscaled_mma_builder.inl` | Dual SFA layout (3-tuple SmemLayoutAtomsA) |
| `sm120_blockscaled_mma_array_tma.hpp` | SmemLayoutSFA_MMA, TileShapeSFA, SFA_M_ratio, sSFA_mma tensor |

### Build notes

- CUTLASS headers under `build/_deps/cutlass-src/` are reset by `cmake ..`. Modifications must be re-applied after cmake reconfigure.
- After changing CUTLASS headers, manually `touch` the .cu files since cmake does not track header dependencies:
  ```bash
  touch cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu
  ```
- After adding new generated .cu files, re-run `cmake ..` (which triggers the CUTLASS reset).

## Architectural Limits

M=32 is the minimum feasible tile for SM120 NVFP4:

- **MMA atom constraint**: Pingpong AtomLayout `Shape<_2, _2, _1>` requires minimum M = 2 × 16 = 32.
- **SFA constraint**: `Blk_rows_per_SF = 32`. For M=32, `MMA_SFA_inner = 1` (one SF group). M=16 would give `MMA_SFA_inner = 0`, which is invalid.
- **Cooperative schedule** uses AtomLayout `Shape<_4, _2, _1>`, requiring minimum M=64. Not usable for M=32.

## Tactic Indices (FAST_BUILD, SM120 FP4 Grouped GEMM)

With FAST_BUILD and 3 base CTA configs (M128, M64, M32), each with swap/no-swap variants:

**GEMM1 (6 tactics)**: 0=M128, 1=M64, 2=M32, 3=M128-swap, 4=M64-swap, 5=M32-swap

**GEMM2 (12 tactics)**: Similar pattern with additional epilogue variants.

## Dual-Tile Mechanism

TRT-LLM includes a dual-tile mechanism (`set_dual_tile_profiles` + `run_moe_dual_tile`) that splits experts into two groups based on a per-expert token threshold and runs separate GEMM passes with different tile sizes for each group.

E2E benchmarks showed this mechanism does not improve performance because:

1. The auto-tuner already selects the optimal single tile per call.
2. The two-pass approach adds overhead: output buffer zeroing, input re-expansion, double kernel launches, and FINALIZE epilogue atomics.
3. MoE kernel time is a small fraction of total inference time.

The recommended approach is single-tile with auto-tuning, which this work enables by providing M32 and M64 as additional tactic options.

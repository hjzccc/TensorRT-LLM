# CuTe Layout Representation and Algebra

## Paper Metadata
- **Title**: CuTe Layout Representation and Algebra
- **Authors**: Cris Cecka
- **Venue**: Preprint (arXiv)
- **arXiv ID**: arXiv:2603.02298v1 [cs.MS]
- **Year**: 2026

---

## Problem

Modern GPU kernels for GEMM, tensor contractions, and convolution require expressing dozens of distinct memory and thread layouts: data arrangements in HBM/shared memory, thread-value partitions for Tensor Core instructions, swizzle patterns to avoid bank conflicts, and TMA/TMEM coordinate mappings for Hopper/Blackwell. Traditional flat shape+stride tensor representations cannot capture these hierarchical, folded, or non-integral layouts. CUTLASS v2 addressed this by implementing nearly 300 separate layout classes across 87 files and ~55,000 lines of code, which is unmaintainable and error-prone.

---

## Approach

CuTe introduces a **hierarchical layout algebra** where a layout `L` is a pair of congruent hierarchical tuples: a shape `S` (nested positive integers) and a stride `D` (nested integers). Layout evaluation maps a coordinate to a memory offset via a generalized inner product:

```
L(c) = D · c_tilde
```

where `c_tilde` is the natural coordinate induced by `S` via the mapping:

```
idx2crd(i) = ( i mod |S0|,  floor(i/|S0|) mod |S1|,  ...,  floor(i / prod_{k<r-1} |Sk|) )
```

This makes layouts behave like generalized matrix-vector products over integers, enabling a small set of algebraic operations to replace hundreds of ad-hoc implementations.

The core algebra consists of five operations:

1. **Coalesce**: flatten/simplify a layout while preserving its evaluation function, collapsing compatible adjacent modes.
2. **Composition** `A o B`: compose two layouts so `(A o B)(c) = A(B(c))`. The base-case formula for `B = s:d` produces a result layout under divisibility conditions on `Sbar_r | d` or `d | Sbar_r`.
3. **Right-inverse** `L‡`: recovers coordinates from offsets; used to detect contiguous subvectors and vectorization opportunities.
4. **Left-inverse** `L†`: checks admissibility ("does this instruction's accessed offsets exist in this data layout?").
5. **Complement** `L*`: generates codomain elements not hit by `L`, used to build:
   - **Logical product** `A ⊗ B = (A, A* o B)`: tiles layout `A` over a grid defined by `B`.
   - **Logical divide** `A ⊘ B = A o (B, B*_{|A|})`: splits a layout into the part selected by tiler `B` and the complementary rest.

Tensors bind a data accessor to a layout: `T(c) = *(base + L(c))`. The paper shows how to fold arbitrary tensor contractions into a canonical batched-GEMM form with row, column, reduction, and batch modes, and how to use composition to express per-thread/per-block subtensors for any instruction's prescribed layout.

Swizzle patterns (shared-memory bank-conflict avoidance) are modeled by replacing integer strides with F2-valued strides, keeping the same algebra. Blackwell TMEM instructions (e.g., `tcgen05.ld.32x32b.x1` with layout `(1,128):(1,16384)`) are expressed and verified using the same left-inverse admissibility check.

---

## Key Results

- CuTe's core layout library requires **~3,000 lines of code** and can represent all ~300 layouts found in CUTLASS v2 (which needed ~55,000 lines across 87 files), plus additional layouts not previously expressible.
- CuTe is the foundation of **CUTLASS v3, CUTLASS v4, and the CuTe DSL**; performance evaluations in CUTLASS and FlashAttention show **no runtime overhead** versus hand-tuned implementations.
- Ampere FP64 Tensor Core thread-value partitioning of an 8x8 C matrix is expressed as a single layout `((4,8),2):((16,1),8)`, replacing what previously required a bespoke class.

---

## Relevance

CuTe's layout algebra is the foundational abstraction used in CUTLASS v3/v4 kernels for Blackwell GPUs, making it directly relevant to implementing mixed-precision MoE GEMM kernels that must correctly express FP4/FP8/BF16 weight layouts, TMEM access patterns, and thread-value partitions for Blackwell Tensor Core instructions.

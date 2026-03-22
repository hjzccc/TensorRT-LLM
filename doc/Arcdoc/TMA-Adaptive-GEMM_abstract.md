# TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hopper

## Paper Metadata

- **Title**: TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hopper
- **Authors**: Zhongling Su, Rong Fu, Weihan Cao, Jianfei Gao, Minxi Jin, Zhilin Pei, Hui Wang
- **Venue**: arXiv preprint
- **arXiv ID**: arXiv:2508.16584v1
- **Year**: 2025

---

## Problem

FP8 grouped GEMM on Hopper requires each group's row count to be aligned to a fixed boundary (e.g., 128 rows) because TMA descriptors are statically configured and Hopper's TMA hardware enforces strict alignment constraints. In MoE inference and training, expert token counts `M_g` vary dynamically and are rarely multiples of the tile size, so existing implementations pad each group to the next alignment boundary. This wastes both memory (up to 23.8% in tested configurations) and compute FLOPs. The paper solves how to run FP8 grouped GEMM with arbitrary variable group sizes on Hopper TMA without any padding, while remaining bitwise equivalent to the padded baseline.

---

## Approach

The core insight is that TMA descriptors cannot be reconfigured at runtime, but a small precomputed pool of descriptors can cover all possible residual row counts using a two-phase overlapping load/store strategy.

**Descriptor pool construction.** For a tile height `blockM`, the method predefines a pool of `floor(log2(blockM)) + 1` TMA descriptors, one for each power-of-two height up to `blockM`:

```
D_pool = { [2^i, blockN] | 0 <= i <= floor(log2(blockM)) }
```

This logarithmic coverage means only ~7 descriptors are needed for `blockM=128`.

**Runtime residual handling.** For each group `g`, the residual row count is `res_g = M_g mod blockM`. The kernel selects the largest power-of-two descriptor not exceeding the residual: `D_opt^g = [2^(floor(log2(res_g))), blockN]`. The residual block is then written in exactly two TMA operations:

- First op: shared-memory rows `[0, 2^(floor(log2(res_g))) - 1]` to global rows `[M_g - res_g, ...]`
- Second op: shared-memory rows `[res_g - 2^(floor(log2(res_g))), res_g - 1]` to global rows `[M_g - 2^(floor(log2(res_g))), M_g - 1]`

The two writes intentionally overlap on a small region, which safely covers all valid residual rows without any out-of-bounds access. For example, with `blockM=128` and `res_g=125`, the chosen descriptor height is 64, and the two writes cover rows 128-191 and 189-252 respectively, with a 3-row overlap.

**Alignment fixes.** Two secondary alignment issues arise. For shared-memory output `C_g`, the second TMA phase may start at a non-128-byte-aligned row; this is resolved by constraining `blockN` to be a multiple of 64 so the per-row offset `2*blockN` bytes stays aligned. For the FP8 scale tensor `S_A^g`, the per-row stride `4*ceil(K/128)` bytes may violate 16-byte global-memory alignment; this is resolved by boundary-aligned over-fetching: the kernel prefetches a small number of rows from the preceding group so the TMA descriptor starts at a 16-byte-aligned address, then uses only the central `blockM` valid rows for computation.

---

## Key Results

- **1.7% to 20.4% end-to-end speedup** over explicit-padding + DeepGEMM baseline across `N, K in {3072..8192}`, group counts in `{4..32}`, and total sequence lengths `M in {8192..65536}` on H800.
- **Up to 23.8% memory reduction**, with the largest savings at small `M` (8192) with many groups (32).
- Speedup correlates strongly with `N` (r = -0.899): smaller `N` gives larger gains because padding waste is proportionally higher; correlation with group count is weak (r = 0.096). Numerical results are bitwise identical to the padded baseline for all valid data.

---

## Relevance

This technique directly addresses the padding overhead in FP8 grouped GEMM for MoE inference on Hopper and Blackwell GPUs, where variable expert token counts make alignment-free TMA loading essential for achieving full throughput without wasted compute or memory.

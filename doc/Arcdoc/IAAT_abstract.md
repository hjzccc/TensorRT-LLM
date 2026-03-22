# IAAT: A Input-Aware Adaptive Tuning Framework for Small GEMM

## Paper Metadata

- **Title**: IAAT: A Input-Aware Adaptive Tuning framework for Small GEMM
- **Authors**: Jianyu Yao, Boqian Shi, Chunyang Xiang, Haipeng Jia, Chendi Li, Hang Cao, Yunquan Zhang
- **Venue**: arXiv preprint
- **arXiv ID**: arXiv:2208.09822v1
- **Year**: 2022

---

## Problem

Standard GEMM libraries (OpenBLAS, ARMPL, BLIS) perform poorly on small matrices because they were designed around large-matrix assumptions. The dominant cost for large GEMM is the pack step, which copies A and B into contiguous micro-panels to enable stride-1 access; for small matrices this pack overhead can reach 67% of total execution time. Additionally, boundary-processing overhead (handling non-multiple-of-tile-size dimensions) is negligible for large matrices but significant for small ones. The paper targets ARMv8 CPUs and defines "small GEMM" as cube-root(MNK) <= 80 for NN/NT/TT transpositions and <= 32 for TN.

---

## Approach

IAAT is a two-stage framework: an install-time kernel generator that produces pack-free assembly kernels for specific sizes, and a runtime tiling planner that selects and sequences those kernels to minimize memory traffic.

**Install-time stage.** A Computational Template Designer abstracts ARMv8 NEON FMLA instructions into reusable templates (e.g., `sfmlas(out, in1, in2, index)` for `fmla out.4s, in1.4s, in2.s[index]`). A Kernel Generator uses these templates to produce block-level kernels computing `Cc = Ac * Bc + Cc` for specific `(mc, nc, kc)` block sizes. Each kernel uses two ping-pong subkernels: the first loads the next column of A into a second register group while computing with the first, and the second does the reverse, hiding load latency behind compute. A Register Allocator assigns the 32 ARMv8 128-bit SIMD registers across three groups (A, B, C accumulators) using strategies like `ANTwoCC` (two columns of A, `2*ceil(mc/elenum)` registers) or `ATEachCOne` (one register per column of transposed A). A Kernel Optimizer then intersperses load and compute instructions and prefers `ldp`/`ldr` and FMA to minimize pipeline stalls.

**Runtime stage.** Given input dimensions `(M, N, K)`, the runtime planner builds a kernel execution plan by tiling M and N into blocks that minimize total L2-to-register data movement. The objective is to minimize `sum(m_i + n_i)` across all block pairs, since total data moved is `(sum(m_i + n_i)) * K + 2*M*N`. For SGEMM NN, if `N <= 13` the planner sets `nc = N` and finds the largest feasible `mc`; for larger N it case-splits on M, tries 8-based and 16-based tilings, and picks the one with fewer memory operations via `CompareLessMemops`. The resulting plan connects pre-generated kernels without any pack step.

---

## Key Results

- **SGEMM NN (M=N=K <= 80) on Kunpeng920 ARMv8.2 at 2.6 GHz**: IAAT is **1.81x faster than OpenBLAS**, 2.3x faster than ARMPL, and 20.17x faster than BLIS.
- **DGEMM NN (M=N=K <= 80)**: 1.48x over OpenBLAS, 1.66x over ARMPL, 15.0x over BLIS.
- **CGEMM/ZGEMM**: 1.09-1.37x over OpenBLAS across transpositions; TN transposition is the hardest case (1.16-1.32x) because discontinuous memory access prevents effective vectorization.

---

## Relevance

IAAT's pack-free, input-aware tiling strategy for small GEMM is relevant to MoE inference on Blackwell GPUs when expert token counts are very small (decode-phase batch sizes), where the overhead of standard grouped GEMM padding and packing dominates and size-specific kernel selection can recover significant throughput.

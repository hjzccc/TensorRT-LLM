# The Cambrian Explosion of Mixed-Precision Matrix Multiplication for Quantized Deep Learning Inference

## Paper Metadata

- **Title**: The Cambrian Explosion of Mixed-Precision Matrix Multiplication for Quantized Deep Learning Inference
- **Authors**: Hector Martinez, Adrian Castello, Francisco D. Igual, Enrique S. Quintana-Orti
- **Venue**: Future Generation Computer Systems (preprint submitted)
- **arXiv ID**: arXiv:2506.11728v1
- **Year**: 2025

---

## Problem

Modern CPUs have shifted from FP32 FMA-oriented SIMD toward mixed-precision integer DOT-product units and tile/matrix engines (ARM NEON DOT, SVE2 I8MM, Intel AMX, ARM SME, RISC-V IME). These units offer 4-65x higher throughput than FP32 FMA for INT8 inputs with INT32 accumulation, but the classic GotoBLAS2 GEMM framework was designed around AXPY-style rank-1 updates and does not map cleanly onto DOT-product or matrix-engine semantics. The paper asks how to redesign the micro-kernel and packing layers of high-performance GEMM to exploit these new ISA primitives for quantized deep learning inference across x86_64, ARM, and RISC-V.

---

## Approach

The paper starts from the standard 5-loop GotoBLAS2 macro-kernel structure (loops over `jc`, `pc`, `ic`, `jr`, `ir`) with packing of A and B into contiguous micro-panels `Ac` and `Bc`. The macro-kernel is kept unchanged; only the micro-kernel and packing layouts are redesigned per ISA.

**Quantization formulation.** The quantized approximation is `C ≈ sA * sB * Aq * Bq`, where `Aq` and `Bq` are INT8 matrices and `sA`, `sB` are per-tensor scaling factors. The dominant cost is the integer GEMM `Cq = Aq * Bq` with INT32 accumulation; rescaling is negligible. This preserves the linear structure of GEMM and allows the same macro-kernel loop nest to be reused.

**ISA-specific micro-kernels.** For each target, the authors redesign the innermost loop to match the hardware's DOT or matrix-engine operand layout:

- *ARMv8.0-A NEON (no native DOT):* DOT is emulated using `vmull_s8` (INT8 multiply to INT16), `vmlal_s8` (multiply-accumulate to INT16), `vpadalq_s16` (pairwise add INT16 to INT32), and `vpaddq_s32` (pairwise add INT32). Micro-kernel is `mr x nr = 2 x 8` with `kr = 16`.
- *ARMv8.2 NEON (native DOT):* Uses `vdotq_laneq_s32(v0, v1, v2, lane)` which computes four INT8 dot products into one INT32 lane in a single instruction. Micro-kernel is `mr x nr = 4 x 16` with `kr = 16`, greatly simplifying the ARMv8.0 design.
- *ARM SVE2:* Vector-length-agnostic version using `svdot_lane_s32`, supporting 128-2048-bit registers with the same DOT-oriented structure.
- *SpacemiT K1 IME (RISC-V matrix engine):* Uses `vmadot v0, v10, v11` which computes a `4 x 4 x 8` INT8 mixed-precision GEMM in one instruction. Micro-kernel is `mr x nr = 4 x 8` with `kr = 8`; packing is redesigned to match the matrix engine's operand layout exactly.
- *Intel AMX:* Uses `_tile_dpbssd(c, a, b)` for `16 x 16 x 64` INT8+INT32 tile GEMM. The micro-kernel becomes very simple (one tile multiply-accumulate per `kr = 64` step) but packing is more specialized to match AMX tile layout.
- *ARM SME:* Uses `smopa <ZAda>.S, <Zn>.B, <Zm>.B` which computes a sum of four INT8 outer products widened to INT32 and accumulated into ZA tile storage. The existing AXPY-style logic ports with minor changes since operands remain vector-like.

In all cases, packing is redesigned so that the micro-kernel reads stride-1 from `Ac` and `Bc`, and the micro-tile dimensions `mr x nr` are chosen to maximize register utilization while holding `Cr` plus at least one column of `Ar` and one row of `Br` in registers.

---

## Key Results

- **ISA throughput ratios vs. FP32 FMA baseline**: Intel AMX INT8 achieves 65.36x; ARM SME2 INT8 achieves 33.94x; SpacemiT K1 IME INT8 achieves 15.33x; ARMv8.2 NEON INT8 DOT achieves 4.0x.
- **End-to-end model speedups**: ResNet50v1.5 on ARM Cortex-A78AE runs **2.0x faster** with INT8 MIP vs. FP32; BERT-Large on ARM Cortex-A72 runs **2.25x faster**; SpacemiT K1 (8 cores) achieves **2.1-2.32x** across both models. Energy consumption drops **5.1x** for ResNet50 and **2.72x** for BERT on Cortex-A78AE.
- **Accuracy impact under ~1%**: ResNet50 Top-1 drops from 0.752 to 0.741 (3600 samples); BERT SST-2 accuracy drops from 0.944 to 0.932.

---

## Relevance

This paper's systematic framework for adapting GEMM micro-kernels to mixed-precision DOT and matrix-engine ISAs provides a direct template for designing FP8/BF16 mixed-precision GEMM micro-kernels for Blackwell's tensor core and TMA hardware, particularly for the small-M expert GEMM shapes that arise during MoE decode-phase inference.

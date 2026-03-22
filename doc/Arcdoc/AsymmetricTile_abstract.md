# Can Asymmetric Tile Buffering Be Beneficial?

## Paper Metadata

- **Title**: Can Asymmetric Tile Buffering Be Beneficial?
- **Authors**: Chengyue Wang, Wesley Pang, Xinrui Wu, Gregory Jun, Luis Romero, Endri Taka, Diana Marculescu, Tony Nowatzki, Pranathi Vasireddy, Joseph Melber, Deming Chen, Jason Cong
- **Venue**: arXiv preprint (placeholder conference header in PDF; actual venue not specified)
- **arXiv ID**: arXiv:2511.16041v1
- **Year**: 2025

---

## Problem

On memory-constrained AI accelerators like AMD XDNA2 (Strix Point NPU), the on-chip L1 buffer per core is only 63 KB. Conventional GEMM tiling uses symmetric buffering where the M-dimension of the input A tile and the output C tile are equal (`T_MA = T_MC`), which wastes buffer space because A rows only need to live until their contribution to C is accumulated. This constraint limits the achievable tile sizes and therefore the arithmetic intensity. The paper asks whether decoupling `T_MA` from `T_MC` (asymmetric tile buffering, ATB) can improve GEMM throughput despite the added kernel-switching overhead it introduces.

---

## Approach

ATB introduces a tile parameterization `(T_MA, T_MC, T_K, T_N)` with asymmetry ratio `rho = T_MC / T_MA >= 1`. The A buffer holds `T_MA` rows while the C accumulator holds `T_MC = rho * T_MA` rows, so A is reloaded `rho` times per C tile.

**Arithmetic intensity model.** For an output-stationary tile, arithmetic intensity is:

```
AI_rho = 2 / (a/T_N + b/T_MC + c/K)
```

where `a`, `b`, `c` are per-element byte costs for A, B, and C respectively. Larger `T_MC` and `T_N` increase AI; smaller `T_K` reduces buffer pressure and allows larger tiles. The L1 capacity constraint with double-buffered A and B is:

```
2a/rho * T_MC * T_K + 2b * T_K * T_N + c * T_MC * T_N <= 63 KB
```

Dividing the A buffer by `rho` is what makes larger `T_MC` feasible within the same L1 budget.

**Microkernel efficiency model.** ATB requires switching between A-loading microkernels `rho` times per C tile, each switch costing ~50 cycles on XDNA2. The effective core efficiency accounting for switching overhead is:

```
Eff_core = 1 / (1/Eff_micro + delta * rho * Perf_core^peak / (2 * T_MC * T_N * T_K))
```

where `delta ~ 50` cycles is the measured switching cost and `Eff_micro` is the microkernel efficiency without switching. This creates a tradeoff: larger `rho` improves AI (via larger `T_MC`) but hurts `Eff_core` (more switches). Larger `T_K` amortizes the switching cost and improves `Eff_micro` by hiding load latency through ping-pong register scheduling.

**Microkernel design.** The authors hand-schedule BFP16 VLIW microkernels for XDNA2 AIE cores using: double-buffered input registers (ping-pong), shared inputs across parallel accumulator chains to reduce loads per VMAC, and overlapped prolog/epilog across successive chain clusters. The steady-state lower bound is `T_steady >= II_parallel * (N_accum - C)` where `C` is the number of parallel chains and `II_parallel = max(P + 1 - C, ceil(R_load / U_LD)) / C`.

**Final performance model.** Array-level performance is:

```
Perf_array = min(AI_array * BW_offchip, Eff_core * Perf_core^peak * N_core)
```

with `BW_offchip ~ 65 GB/s`, `N_core = 32`, `Perf_core^peak = 1.84 TFLOPS`. The optimal `(T_K, rho)` is found by jointly maximizing `AI_array` and `Eff_core`.

---

## Key Results

- **Up to 24.6 TFLOPS** on mixed-precision BFP16-BF16 GEMM on AMD XDNA2 (Strix Point NPU), a **4.54x speedup** over the prior state-of-the-art MLIR-AIE baseline of 4.8 TFLOPS.
- ATB alone provides **up to 40% throughput gain** over the best symmetric-tile kernel: the symmetric-feasible tile `64x64x128` achieves 17.3 TFLOPS, while ATB-enabled `128x64x128` with `rho=4` achieves 24.3 TFLOPS (the symmetric version would require 91 KB, exceeding the 63 KB L1 budget).
- Single-core efficiency improves from 0.32 TFLOPS (MLIR-AIE) to **0.92 TFLOPS** (2.88x) with the authors' optimized microkernel at `T_K=64, rho=4`.

---

## Relevance

ATB's insight that decoupling input and output tile M-dimensions can unlock higher arithmetic intensity within a fixed buffer budget applies directly to Blackwell GPU GEMM tile design for MoE inference, where SMEM capacity constraints similarly limit achievable tile sizes for mixed-precision (FP8/BF16) expert kernels.

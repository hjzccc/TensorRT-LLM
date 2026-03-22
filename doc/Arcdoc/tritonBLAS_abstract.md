# tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection

## Paper Metadata

- **Title**: tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection
- **Authors**: Ryan Swann, Muhammad Osama, Xiaohu Guo, Bryant Nelson, Lixun Zhang, Alex Brown, Yen Ong, Ali Yazdani, Sean Siddens, Ganesh Dasika, Alex Underwood
- **Venue**: arXiv preprint
- **arXiv ID**: arXiv:2512.04226v1
- **Year**: 2025

---

## Problem

Triton's exhaustive autotuning for GEMM kernel parameter selection is prohibitively expensive: it benchmarks dozens to hundreds of candidate tile configurations at runtime, taking tens of seconds to over 20 minutes for large matrices. This cost is impractical for production deployments where GEMM shapes change frequently. The paper asks whether a deterministic analytical model can predict near-optimal tile configurations from hardware and problem characteristics alone, eliminating all runtime tuning overhead.

---

## Approach

tritonBLAS builds a hierarchical latency model that ranks candidate tiling configurations by estimated total GEMM execution time, then selects the minimum without any benchmarking.

The model decomposes GEMM into a five-level tile hierarchy: matrix instruction, register/wavefront, shared-memory/workgroup, cache, and global. For a shared-memory tile of dimensions `(MTM, MTN, MTK)`, compute latency per tile is `LMT = LMI * NMI,M * NMI,N * NMI,K`, where `LMI` is the latency of one matrix instruction and `NMI,*` are the instruction counts along each dimension. Wave quantization is captured by computing the number of output tiles `Tout = ceil(M/MTM) * ceil(N/MTN)` and the number of waves `omega = ceil(Tout / NCU)`, which exposes tail-occupancy waste. Cache reuse is modeled via a hit-rate formula `h = 1 - U/R` derived from the ratio of uncached to total reads for a given cache-tile shape, with a capacity correction when the working set exceeds the effective cache size.

Memory latency is computed by routing the per-CU load volume through the bandwidth hierarchy (L1, L2, HBM) and taking the maximum across levels: `Lmem = max(LCU_lat, L1, L2, LMEM)`. The per-iteration exposed latency is then `Lloopiter = max(Lcompute, Lmem)`, capturing whether a tile is compute-bound, bandwidth-bound, or occupancy-limited. Total tile latency adds a prologue (one memory prefetch), a steady-state loop over K iterations, and an epilogue (output writeback). Total GEMM latency is `Ltotal = Nwaves * Ltile`. The model is parameterized entirely by measurable hardware constants (bandwidths, access latencies, matrix-core instruction shapes) and requires no runtime calibration per problem size. The entire selection runs in Triton and takes 50-80 microseconds regardless of matrix dimensions.

---

## Key Results

- **94.7% selection efficiency** relative to exhaustive autotuning across 150,000 random GEMM shapes (dimensions multiples of 128, up to 8192).
- **>5 orders of magnitude faster** than autotuning: 0.000055-0.000075 s vs. 12-1384 s for Triton exhaustive search on the same 75-candidate set.
- On average **3% better than PyTorch `torch.matmul()`** on AMD MI300X; on Llama 3 matrix sizes, tritonBLAS is 13.9% slower on average but achieves up to 1.10x speedup on individual shapes.

---

## Relevance

tritonBLAS demonstrates that a lightweight analytical model can replace runtime autotuning for GEMM tile selection, which is directly applicable to selecting mixed-precision (e.g., FP8/BF16) tile configurations for MoE expert GEMMs on Blackwell GPUs without incurring per-deployment tuning cost.

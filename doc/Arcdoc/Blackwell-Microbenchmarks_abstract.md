# Microbenchmarking NVIDIA's Blackwell Architecture: An In-Depth Architectural Analysis

## Paper Metadata

- **Title**: Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis
- **Authors**: Aaron Jarmusch, Sunita Chandrasekaran
- **Venue**: arXiv preprint (no conference venue listed)
- **arXiv ID**: arXiv:2512.02189v3
- **Year**: 2026

---

## Problem

NVIDIA's Blackwell B200 GPU introduces several new microarchitectural features (5th-generation tensor cores, Tensor Memory, a hardware Decompression Engine, dual-die NV-HBI interconnect, and native FP4/FP6 support) that have no prior systematic characterization. Without concrete latency, throughput, and bandwidth numbers, practitioners cannot make informed decisions about how to exploit these features for LLM inference, scientific computing, or mixed-precision training. This paper builds an open-source PTX/CUDA microbenchmark suite to fill that gap and compares B200 against H200 across a range of workloads.

---

## Approach

The authors construct targeted PTX-level and CUDA microbenchmarks for each Blackwell-specific feature, validating PTX-to-SASS mappings to ensure measurements reflect actual hardware behavior. For Tensor Memory (TMEM), they use dependency-controlled pointer-chase and data-movement tests, varying tile sizes and strides to find latency and bandwidth saturation points. For the hardware Decompression Engine (DE), they pre-compress 100 MB datasets across seven formats (LZ4, Snappy, Zstandard, GZIP, Cascaded, Bitcomp, ANS) and sweep chunk sizes (32-256 KB) and concurrency (1-1024) to characterize throughput, latency, and pipeline depth. For 5th-generation tensor cores, they use `tcgen05.mma` with dependency chains for latency and independent MMAs for throughput, covering all supported precisions including the new FP4 (E2M1) and FP6 (E2M3, E3M2, E2M3) formats.

These microarchitectural measurements are then connected to end-to-end workloads: LLM inference on Mistral-7B and Mixtral-8x7B (dense and MoE), FP64 DGEMM and SpMV for scientific computing, and mixed-precision training of ResNet-50 and GPT-1.3B. All results are compared against H200 to quantify generational improvement. The paper also derives practical tuning guidance, such as targeting 64x64 tiles for TMEM efficiency and scaling DE concurrency by chunk size.

---

## Key Results

- **Tensor core throughput on B200**: FP4 reaches 7,700 TFLOPS, FP6 reaches 5,134 TFLOPS, FP8 reaches 3,851 TFLOPS, and BF16/FP16 reach ~1,928 TFLOPS. Single-instruction `tcgen05.mma` latency is 11.0-11.4 cycles, a 2.9x-11.2x reduction versus Hopper's `wgmma`.
- **MoE inference (Mixtral-8x7B)**: FP4 reaches 76,900 tok/s on B200 (2.69x vs B200 FP16 baseline), outpacing the dense Mistral-7B FP4 gain of 2.50x. FP8 reaches 51,200 tok/s (1.58x vs H200 FP8). FP4 perplexity degradation is +9.1% for Mixtral-8x7B.
- **Training speedup**: B200 achieves 1.85x ResNet-50 and 1.55x GPT-1.3B training throughput versus H200, with 32% better energy efficiency.

---

## Relevance

This paper provides the ground-truth hardware numbers for Blackwell B200 tensor core throughput, TMEM bandwidth, and FP4/FP8 inference performance on both dense and MoE models (Mixtral-8x7B), making it the essential reference for calibrating mixed-precision MoE inference kernel design and roofline analysis on Blackwell GPUs.

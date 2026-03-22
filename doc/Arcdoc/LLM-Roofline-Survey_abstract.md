# LLM Inference Unveiled: Survey and Roofline Model Insights

## Paper Metadata
- **Title**: LLM Inference Unveiled: Survey and Roofline Model Insights
- **Authors**: Zhihang Yuan, Yuzhang Shang, Yang Zhou, Zhen Dong, Zhe Zhou, Chenhao Xue, Bingzhe Wu, Zhikai Li, Qingyi Gu, Yong Jae Lee, Yan Yan, Beidi Chen, Guangyu Sun, Kurt Keutzer
- **Venue**: arXiv preprint
- **arXiv ID**: arXiv:2402.16363
- **Year**: 2024

---

## Problem

LLM inference is slow and expensive, but practitioners lack a principled framework for understanding *why* a given model is slow on a given GPU and *which* optimization will actually help. Prefill and decode have fundamentally different bottlenecks, different operators within a layer have different arithmetic intensities, and the effectiveness of quantization, batching, or FlashAttention depends on where the workload sits relative to the hardware's compute and memory-bandwidth limits. Without a unified analysis tool, optimization choices are made by trial and error.

---

## Approach

The paper combines a survey of LLM inference optimization methods with a practical analysis framework called **LLM-Viewer**, built on the Roofline model.

The Roofline model characterizes hardware by two limits: the compute roof `P_peak` (peak FLOP/s) and the memory-bandwidth roof `B_peak * I`, where `I` is arithmetic intensity (FLOPs/byte). The turning point `I_bar = P_peak / B_peak` separates memory-bound from compute-bound operation. Any operator with `I < I_bar` is memory-bound; its throughput scales with bandwidth, not compute.

The paper formalizes the two inference stages. In **prefill**, the input is a matrix `X_pre in R^(n x d)`, and attention computes `O_pre = softmax(Q_pre K_pre^T / sqrt(d)) V_pre W_o + X_pre`. With sequence length n, the QK matmul has `O(n^2 d)` FLOPs and `O(n^2 + nd)` bytes, giving arithmetic intensity that grows with n. In **decode**, the input is a single vector `X_dec in R^(1 x d)`, and attention uses the cached `K_cat = [K_cache, X_dec W_k]`. All linear projections reduce to matrix-vector products with arithmetic intensity ~1 FLOPs/byte, making decode almost entirely memory-bound regardless of sequence length.

LLM-Viewer takes as input: (1) model layer specs (shapes, ops), (2) hardware specs (peak compute, peak bandwidth), (3) inference config (batch size, prompt length, generation length), and (4) optimization config (quantization bitwidth, FlashAttention on/off, decoding method). It computes per-layer arithmetic intensity, maps each layer to the roofline, identifies the bottleneck, and aggregates to network-level throughput, memory footprint, and batch-size/sequence-length performance curves.

The framework explains optimization effects analytically. Quantization reduces bytes moved per weight, increasing arithmetic intensity and shifting memory-bound operators toward the compute roof. It also raises the compute roof itself if the hardware supports lower-precision compute (e.g., INT8 doubles peak FLOP/s on A6000 from 155 to 310 TOP/s). FlashAttention fuses attention into a single kernel, reducing intermediate memory traffic and increasing effective arithmetic intensity. Batching increases the number of tokens processed per weight load, raising arithmetic intensity for decode. KV-cache quantization reduces the bytes read per attention step, which matters most when context length is long enough that KV cache dominates memory.

---

## Key Results

- On Nvidia A6000 with Llama-2-7B at sequence length 2048, batch size 1: **prefill is compute-bound** for linear projections (arithmetic intensity ~1024-1215, above the turning point), while **decode is entirely memory-bound** for all operators (arithmetic intensity ~1 FLOPs/byte).
- Quantization from FP16 to INT8 doubles peak compute throughput on A6000 (155 to 310 TOP/s); decode latency decreases monotonically as weight precision drops from FP16 to W4 for small batch sizes, but gains plateau at large batch sizes when the workload becomes compute-bound.
- FlashAttention reduces memory access and inference time by roughly **25-38%** depending on stage and batch size; the reduction is larger in decode (where attention is memory-bound) than in prefill (where some ops are already compute-bound).

---

## Relevance

The Roofline-based analysis framework directly applies to characterizing mixed-precision MoE expert GEMM kernels on Blackwell GPUs, where the interplay between FP4/FP8/BF16 arithmetic intensity, HBM bandwidth, and Tensor Core throughput determines whether each expert dispatch is compute-bound or memory-bound and which precision regime yields the best throughput per watt.

# SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

## Paper Metadata
- **Title**: SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
- **Authors**: Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, Dan Alistarh
- **Venue**: arXiv preprint (arXiv:2306.03078v1 [cs.CL])
- **arXiv ID**: 2306.03078
- **Year**: 2023

## Problem

Post-training quantization of LLMs to 3-4 bits causes moderate to severe perplexity degradation with existing methods (RTN, GPTQ), especially for smaller deployable models like 7B-13B. The core issue is that a small fraction of weights, when quantized, cause disproportionately large output errors even after GPTQ-style compensation. SpQR targets near-lossless compression: preserve 16-bit model quality while reducing memory enough to run larger LLMs on consumer hardware (e.g., a 33B model on a single 24 GB GPU).

## Approach

SpQR defines per-weight quantization sensitivity using an Optimal Brain Surgeon-style derivation. For weight `w_ij` in a layer with calibration inputs `X`, the sensitivity is the minimum output error when that weight is forced to its quantized value while all others can compensate: `s_ij = (w_ij - quant(w_ij))^2 / [2 (X X^T)^(-1)]_jj`. This measures how much rounding a single weight costs even with optimal compensation from correlated weights.

The method splits weights into two groups. A small fraction (typically 0.2-0.5%) whose sensitivity exceeds a threshold are stored as **outliers** in FP16 with a CSR-like sparse structure (16-bit value, 16-bit column index, 32-bit per-row prefix). The remaining weights are quantized aggressively to 3 or 4 bits using very small first-level groups (size `beta_1`, typically 16) with asymmetric min-max quantization: scale `s_i = (max - min) / (2^b - 1)`, zero-point `z_i = -min / s_i`. Small groups improve precision but create many scale/zero-point values, so SpQR applies a second level of quantization to the quantization statistics themselves using second-level groups of size `beta_2`. This **bilevel quantization** keeps the overhead manageable. The average bits per parameter is approximately `b_w + (b_s + b_z)/beta_1 + 64/(beta_1 * beta_2) + 32 * r_o`, where `r_o` is the outlier fraction. For typical settings (3-bit weights, 3-bit statistics, `beta_1=16`, `beta_2=32`, `r_o=0.004`), this gives ~3.63 bits/parameter.

The quantization procedure follows GPTQ's calibration-based layerwise reconstruction: compute the Hessian `H = 2 X X^T`, detect outliers within each group, fit first- and second-level quantizers excluding outliers, quantize a column, compute the residual error, and propagate it to remaining unquantized weights. Inference uses a custom hybrid kernel: dense low-bit weights are decoded and multiplied with activations, while outlier weights are handled with a sparse matrix-vector kernel optimized for autoregressive (batch-1) decoding.

## Key Results

- **Near-lossless at ~4.6 bits**: SpQR compresses LLaMA-65B to 4.71 bits/parameter with WikiText-2 perplexity **3.57** vs. FP16 baseline **3.53**, less than 1.1% relative degradation; similar near-lossless results hold for all LLaMA sizes and Falcon models.
- **Better than GPTQ at ~4 bits**: On LLaMA-7B at ~3.94 bits, SpQR achieves WikiText-2 **5.87** vs. GPTQ 4-bit **6.13** and RTN 4-bit **6.43**, roughly halving the error gap to FP16 compared to GPTQ.
- **Inference speedup**: The optimized SpQR kernel achieves **57 tokens/s** on LLaMA-7B (vs. 47 for FP16) and **22 tokens/s** on LLaMA-30B (vs. 19 for FP16) on a single A100, with LLaMA-65B running on a single GPU where FP16 is OOM.

## Relevance

SpQR's bilevel quantization of both weights and their quantization statistics, combined with sparse FP16 storage for the most sensitive weights, is a direct template for mixed-precision MoE inference on Blackwell: expert weight matrices can be compressed with aggressive low-bit quantization while retaining a small outlier set in higher precision, matching Blackwell's support for mixed INT4/FP8/FP16 within a single kernel.

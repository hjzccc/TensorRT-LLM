# QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

## Paper Metadata
- **Title**: QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- **Authors**: Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa
- **Venue**: ICML 2024 (Proceedings of the 41st International Conference on Machine Learning, PMLR 235)
- **arXiv ID**: 2402.04396
- **Year**: 2024

## Problem

Extreme low-bit quantization of LLMs (2-3 bits per weight) causes severe accuracy loss with standard methods, and prior approaches that achieve good quality at 2 bits are either too slow for practical inference (large learned codebooks that don't fit in GPU L1 cache) or lack principled outlier suppression. QuIP# aims to simultaneously achieve state-of-the-art quantization quality at 2, 3, and 4 bits, fast hardware-friendly inference, and practical fine-tuning, all within a unified framework.

## Approach

QuIP# minimizes a per-layer proxy loss `l(W_hat) = tr((W_hat - W) H (W_hat - W)^T)` where `H = E[xx^T]` is the proxy Hessian from calibration data. The method has three main components.

**Incoherence processing via Randomized Hadamard Transform (RHT).** Outliers hurt quantization because they concentrate quantization error. Instead of handling outliers heuristically, QuIP# applies a structured random orthogonal transform to both weights and Hessian before quantization: sample random sign vectors `S_V in {+/-1}^n` and `S_U in {+/-1}^m`, then transform `W_tilde = Had(diag(S_U) Had(diag(S_V) W^T)^T)` and similarly for `H`. After transformation, the weight matrix is incoherent with high probability (max entry bounded by `O(sqrt(log(mn)/mn)) * ||W||_F`), so no single entry dominates and scalar quantization works well. At inference, the transform is applied online: `y = Had(S_V * x)`, multiply by quantized `W_hat`, then `y = Had(S_U * y)`. The RHT costs `O(n log n)` vs. `O(n sqrt(n))` for QuIP's Kronecker transform, and has better incoherence bounds.

**Adaptive vector quantization with BlockLDLQ.** Rather than quantizing each weight independently, QuIP# quantizes groups of `g` weights jointly using a block LDL decomposition of the Hessian `H = L^T D L`. The block rounding rule is `W_hat_k = Q(W_k + (W_{:,<k} - W_hat_{:,<k}) A_k)` where `A_k` is the k-th block of `L^T - I` and `Q` is a vector quantizer. This is a block generalization of LDLQ adaptive rounding that provably reduces quantization error when the Hessian is incoherent.

**E8P lattice codebook.** After RHT, transformed weights are approximately Gaussian/ball-shaped. Scalar quantization uses a hypercube-shaped representable set, which wastes capacity. QuIP# uses an 8-dimensional lattice codebook based on the E8 lattice (the densest 8D sphere packing): `E8 = (Z^8 union (Z^8 + 1/2)) intersect {x | 1^T x is even}`. The practical codebook E8P uses 2 bits/weight (16 bits total per 8D vector, 2^16 entries) but is compressed to a 1 KiB lookup table using symmetry. For 3-bit and 4-bit quantization, residual vector quantization (RVQ) stacks multiple E8P stages. Fine-tuning is added as a final stage: within each transformer block, quantize one layer, freeze it, fine-tune remaining parameters to minimize activation MSE, then proceed to the next layer; after all layers are quantized, fine-tune remaining unquantized parameters end-to-end on cross-entropy.

## Key Results

- **2-bit quality**: On Llama 2 13B at 2 bits, QuIP# achieves WikiText-2 perplexity **5.35** vs. OmniQuant **17.2** and QuIP **13.5**; on Llama 2 70B, **3.91** vs. QuIP **5.90**, making 2-bit models genuinely usable.
- **Scaling**: QuIP# is the first PTQ method where 3-bit models scale better than 4-bit models; on Llama 2, 3-bit QuIP# outperforms a theoretical lossless 4-bit model in scaling plots.
- **Inference throughput**: On RTX 4090, QuIP# 2-bit Llama 2 7B achieves **106.3 tokens/s** vs. AQLM 2-bit **20.6 tokens/s** and FP16 **33.1 tokens/s**, because the 1 KiB E8P codebook fits in L1 cache while AQLM's 1 MiB per-layer codebooks do not.

## Relevance

QuIP#'s Randomized Hadamard Transform for principled outlier suppression and its hardware-efficient lattice codebook design are directly relevant to mixed-precision MoE inference on Blackwell: the RHT approach eliminates the need for per-channel outlier bookkeeping, and the E8P codebook's tiny decode cost makes sub-4-bit expert weights practical without sacrificing throughput on Blackwell's memory-bandwidth-limited MoE decode path.

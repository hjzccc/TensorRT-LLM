# AQLM: Extreme Compression of Large Language Models via Additive Quantization

## Paper Metadata
- **Title**: Extreme Compression of Large Language Models via Additive Quantization
- **Authors**: Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh
- **Venue**: arXiv preprint (arXiv:2401.06118v4 [cs.LG])
- **arXiv ID**: 2401.06118
- **Year**: 2024

## Problem

Post-training quantization of LLMs at very low precision (2-3 bits per parameter) causes major accuracy drops with scalar quantization methods. The challenge is that 2-bit scalar quantization has only 4 representable values per weight, which is far too coarse for the continuous weight distributions in large models. AQLM addresses this by adapting multi-codebook additive quantization from approximate nearest neighbor search to LLM weight compression, targeting Pareto-optimal accuracy vs. model size below 3 bits/parameter.

## Approach

AQLM represents each group of `g` consecutive weights in a row as a sum of `M` codewords, one from each of `M` codebooks `C_1, ..., C_M`, where each codebook has `2^B` vectors of dimension `g`. The full row is `W_{c,i} = concat_{j} sum_{m=1}^M C_m b_{i,j,m}` where `b_{i,j,m}` is a one-hot code selection. The joint optimization objective is `arg min_{C,b} ||W X - W_c X||_2^2`, minimizing layer output error over both codebooks and discrete codes simultaneously.

Initialization uses residual K-means: run K-means on weight groups, store nearest clusters, compute residuals, run K-means on residuals, and repeat to initialize successive codebooks so each compensates prior error.

**Phase 1: discrete code optimization via beam search.** Expanding the objective, the loss decomposes into unary potentials `<W, C_m b_m>_{XX^T}` and pairwise potentials `<C_i b_i, C_j b_j>_{XX^T}`, forming a fully connected discrete MRF. Exact inference is intractable, so AQLM uses beam search: maintain a beam of top-k code assignments, replace one code at a time trying all `2^B` alternatives, keep the best k under the MSE objective. The key computational trick is precomputing `XX^T` so the objective evaluation is independent of the number of calibration samples `n`. Output units are processed in parallel since their objectives decompose.

**Phase 2: codebook optimization via gradient descent.** With codes fixed, codebooks are updated by minimizing `||(W - W_c) X||_2^2 = <(W - W_c) XX^T, (W - W_c)>_F` using Adam. A per-output-unit scale `s_i` (initialized as `||W_i||_2`) is learned jointly with codebooks.

**Phase 3: block-wise fine-tuning.** After quantizing all linear layers in a transformer block, the codebooks, scales, and non-quantized parameters (e.g., RMSNorm) are fine-tuned to minimize `||block(X_{block}) - Y_{block}||_2^2` against the original block outputs, using autograd. This captures intra-block interactions missed by layer-wise optimization. An optional end-to-end fine-tuning stage uses knowledge distillation: `L = (1/N) sum_i D_KL(p_s(x_i), p_t(x_i))` between student and teacher output distributions.

The average bits per parameter is `(16 g M 2^B + d_out (d_in/g) B M + 16 d_out) / (d_out d_in)`. For typical settings (M=2 codebooks, B=8 bits, g=8), this gives ~2 bits/parameter.

## Key Results

- **2-bit quality**: On Llama 2 13B at ~2 bits, AQLM achieves WikiText-2 perplexity **5.60** vs. QuIP# **6.06** and QuIP **13.48**; on Llama 2 70B, **3.94** vs. QuIP# **4.16**, establishing Pareto-optimality below 3 bits.
- **3-bit quality**: On Llama 2 70B at 3.01 bits, AQLM achieves **3.36** WikiText-2 perplexity vs. GPTQ **4.40** and SpQR **3.85**, the best reported 3-bit result at the time.
- **Inference throughput**: On RTX 3090, AQLM 2x8-bit achieves **114.1 tokens/s** for Llama 2 7B (vs. FP16 **54.2 tokens/s**) and **14.3 tokens/s** for 70B (vs. FP16 **5.8 tokens/s**), with 2.75-3.69x CPU speedups on Intel i9.

## Relevance

AQLM's multi-codebook additive quantization achieves the best known accuracy at 2-3 bits/parameter, making it relevant for MoE inference on Blackwell where expert weights can be compressed to 2-3 bits to fit more experts in GPU memory, with the beam-search code optimization and block-wise fine-tuning providing a blueprint for high-quality low-bit expert compression.

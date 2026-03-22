# QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

## Paper Metadata
- **Title**: QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- **Authors**: Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman
- **Venue**: NeurIPS 2024 (38th Conference on Neural Information Processing Systems)
- **arXiv ID**: 2404.00456
- **Year**: 2024

## Problem

End-to-end 4-bit inference for LLMs requires quantizing not just weights but also activations and the KV cache. The main obstacle is activation outliers: a small number of channels have magnitudes far larger than the rest, making 4-bit quantization inaccurate. Prior methods handle this by retaining outlier channels in higher precision, requiring calibration, or retraining. QuaRot aims to eliminate outliers entirely through a mathematically equivalent model transformation, so all matrix multiplications, activations, and KV caches can be quantized to 4 bits with zero special-cased high-precision channels.

## Approach

QuaRot exploits a key property of transformers: orthogonal rotations can be fused into adjacent weight matrices without changing model outputs. The method applies randomized Hadamard transforms at multiple points in the network, making activations incoherent (no dominant outlier channels) while preserving exact mathematical equivalence.

**Global rotation (Stage 1a).** RMSNorm satisfies `RMSNorm(X) = RMSNorm(X Q^T) Q` because rotations preserve vector norms. This means the hidden state can be globally rotated by a randomized Hadamard matrix `Q` and the rotation absorbed into all adjacent weight matrices offline. For example, the key projection becomes `W_k <- Q^T diag(alpha) W_k`. This removes outliers from inter-block activations.

**FFN internal rotation (Stage 1b).** Outliers also appear inside the FFN before the down-projection. An online Hadamard transform is inserted before the down-projection and its inverse is fused into the weight: `W_down <- H W_down`. The effective down-projection becomes `H W_down Q`, making FFN internal activations outlier-free.

**Attention value rotation (Stage 1c).** For each attention head `h`, a headwise Hadamard `H_{d_h}` is inserted between the value projection and output projection: `W_v^(h) <- W_v^(h) H_{d_h}` and `W_out^(h) <- H_{d_h} W_out^(h)`. An additional online transform `Z <- Z (H_{n_h} ⊗ I)` is applied between heads before the output projection. This removes outliers in value paths and enables 4-bit value cache quantization.

**Key/query rotation (Stage 1d).** Because RoPE positional embeddings sit between projections and attention, keys and queries are rotated online headwise: `Q <- Pos(X W_q)(I ⊗ H_{d_h})` and `K <- Pos(X W_k)(I ⊗ H_{d_h})`. Since both are rotated identically, attention scores are unchanged. Keys are cached post-RoPE and post-Hadamard to avoid repeated inverse rotations during decoding.

After all rotations are applied, weights are quantized with GPTQ (or RTN for ablations). Activations use symmetric per-token INT4 quantization with scale `max(|row|) / 7`. The KV cache uses asymmetric group-wise quantization with group size 128. Dequantization multiplies the INT32 accumulation result by row and column scales, casting back to FP16. The Hadamard transform overhead is small: at most ~7% extra latency per linear layer.

## Key Results

- **End-to-end 4-bit (A4W4KV4)**: QuaRot-GPTQ on Llama 2 70B achieves WikiText-2 perplexity **3.79** vs. FP16 **3.32**, retaining ~99% of zero-shot accuracy (avg 75.98 vs. 77.07); prior methods like SmoothQuant RTN give 83.12 on 7B and QUIK-4B with 256 outlier features gives 6.91 on 70B.
- **Group-wise 4-bit**: QuaRot-128G on Llama 2 7B achieves **5.93** perplexity, slightly better than Atom-128G (**6.03**) which retains 128 outlier features in FP16.
- **Prefill speedup**: Up to **3.33x** prefill speedup on Llama 2 70B (RTX 3090, batch 32, seq 2048); decoding memory savings of **3.89-3.92x** on 70B.

## Relevance

QuaRot's rotation-based outlier elimination enables true end-to-end INT4 inference (weights, activations, and KV cache) without any FP16 fallback channels, which maps directly onto Blackwell's INT4 Tensor Core path for MoE inference: by rotating expert weight matrices offline and applying online Hadamard transforms, all expert GEMMs can use INT4 with no mixed-precision bookkeeping overhead.

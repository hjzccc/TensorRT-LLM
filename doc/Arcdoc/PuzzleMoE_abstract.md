# PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-Packed Inference

## Paper Metadata

- **Title**: PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-Packed Inference
- **Authors**: Yushu Zhao, Zheng Wang, Minjia Zhang
- **Venue**: arXiv preprint (no conference venue listed)
- **arXiv ID**: arXiv:2511.04805v1 [cs.LG]
- **Year**: 2025

---

## Problem

MoE models store all expert weights in memory even though only a small fraction activate per token, creating a memory bottleneck that forces large models like Mixtral-8x7B onto multiple GPUs. Existing compression methods either prune weights globally (ignoring expert structure) or merge experts naively (losing expert-specific information). The challenge is compressing expert weights aggressively while preserving the distinct behavior of each expert and keeping inference practical without retraining.

---

## Approach

PuzzleMoE compresses MoE models by merging expert pairs into a shared weight tensor, then reconstructing individual experts at inference time using lightweight metadata packed directly into the weight bits.

**Pairwise sparse expert merging** operates on two experts with weights `W_i, W_j in R^{d x h}`. The method avoids merging three or more experts at once because the masking choice space grows as `2^k - 1` per weight entry. For each pair, two types of masks are computed:

A **similarity mask** identifies weight entries whose magnitudes are close enough to safely average:
```
Delta := ||W_i| - |W_j|| / (|W_i| + |W_j|)
M_sim := 1{Delta <= tau_sim}
```

A **saliency mask** uses activation-weighted importance to decide which expert's value to keep where magnitudes differ:
```
A_i = |W_i| ⊙ ||X_i||_2,   A_j = |W_j| ⊙ ||X_j||_2
M_i^sal := 1{A_i >= A_j},   M_j^sal := 1 - M_i^sal
```

The final merged weight combines both:
```
W_merged = M_sim ⊙ (|W_i| + |W_j|)/2 + (1 - M_sim) ⊙ (M_i^sal ⊙ |W_i| + M_j^sal ⊙ |W_j|)
```

At inference, expert `i` is reconstructed element-wise using its sign pattern `S_i` and combined mask `M_i = M_i^sal ∨ M_sim`:
```
W_i^c = (-1)^{S_i} ⊙ M_i ⊙ W_merged
```

**Bit-packed inference** eliminates the overhead of storing `M_i, M_j, S_i, S_j` as separate tensors. The key observation is that BF16 exponent values in MoE weights cluster in a narrow range (112 to 128 for Mixtral-8x7B). By clamping exponents below 112 to 112 and shifting by 112, the exponent fits in 5 bits instead of 8, freeing 3 bits per weight element. These freed bits store the mask and sign metadata directly inside the BF16 representation. A custom CUDA GEMV kernel decodes the packed weights on the fly during matrix multiplication, adding no storage overhead and no measurable perplexity change.

---

## Key Results

- **Accuracy at 50% compression (Mixtral-8x7B)**: PuzzleMoE achieves average zero-shot accuracy of 72.6 vs. baseline 74.1, with MMLU at 65.7 compared to Wanda's 62.0 and HC-SMoE's 49.0, a gain of up to 16.7% over prior MoE compression methods.
- **Combined with quantization**: PuzzleMoE + quantization reaches 4.8x compression at 3.35 average bits, with only 1.7% accuracy drop for Mixtral-8x7B and 1.0% for DeepSeek-MoE-16B. Compressed Mixtral-8x7B fits on a single A100-80GB instead of two.
- **Compression speed**: Mixtral-8x7B to 50% sparsity takes 2 minutes vs. 55 minutes for the next best method (D2), a 45x speedup. Inference throughput improves 1.28x on Mixtral-8x7B.

---

## Relevance

PuzzleMoE's bit-packing technique, which embeds expert reconstruction metadata into unused BF16 exponent bits, is directly applicable to mixed-precision MoE inference on Blackwell GPUs where BF16 and FP8 expert weights coexist and minimizing metadata overhead per expert is critical for memory-bandwidth-bound decoding.

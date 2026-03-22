# Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques

## Paper Metadata

- **Title**: Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques
- **Authors**: Shwai He, Daize Dong, Liang Ding, Ang Li
- **Venue**: Transactions on Machine Learning Research (TMLR), published March 2025
- **arXiv ID**: arXiv:2406.02500v3 [cs.LG]
- **Year**: 2025

---

## Problem

MoE language models like Mixtral-8x7B require 87.7 GB of memory for deployment despite activating only a fraction of their parameters per token. Existing compression approaches treat MoE models like dense models, applying expert-level pruning that reduces parameter count but delivers negligible inference speedup because the MoE layer structure and inter-expert communication remain intact. The paper asks which combination of compression techniques actually improves both memory footprint and real inference speed without destroying model quality.

---

## Approach

The paper organizes MoE compression into two orthogonal axes: **Expert Slimming** (compressing individual expert weights) and **Expert Trimming** (removing entire experts or layers). The standard MoE forward pass is:

```
K = TopK(Softmax(G(x)), k)
y = sum_{i in K} G(x)_i * E_i(x | W_i)
```

**Expert Slimming** applies a transformation `f` to each expert's weights independently:
```
y = sum_{i in T'} G_i * E_i(x | f(W_i))
```
where `f` is either pruning (`W_hat_i = M_i ⊙ W_i`) or quantization (`W_hat_i = Quant(W_i)`). The paper finds 4-bit AWQ quantization is the most effective slimming method, reducing Mixtral-8x7B memory from 87.7 GB to 24.4 GB with only 0.7 average score drop.

**Expert Trimming** reduces the retained expert set `T` to a subset `T'`. Standard Expert Drop scores experts by routing frequency and removes the least-used ones, but the authors show this gives under 1.06x speedup because the MoE layer itself remains. They propose two more aggressive variants:

**Layer Drop** removes entire MoE layers (all experts in a layer), scored by cosine similarity between the layer's input and output. Because transformer blocks include normalization and residual connections, they score the combined Norm+MoE behavior:
```
S^(NM) = (x' · y') / (||x'||_2 ||y'||_2),  where y' = x' + MoE(Norm(x'))
```
High similarity means the layer is redundant. Dropping 8 of 32 MoE layers in Mixtral-8x7B causes only a 7-point average score drop, compared to a 24-point drop for the equivalent operation on a dense Mistral-7B, confirming MoE layers are more structurally redundant.

**Block Drop** extends this to remove entire transformer blocks (attention + MoE together), scored by input-output similarity at the block level. This also reduces KV-cache size, which Layer Drop alone cannot achieve.

The recommended recipe combines Layer Drop or Block Drop with AWQ quantization. After optional supervised fine-tuning on the compressed model, performance gaps shrink to under 1%.

---

## Key Results

- **Best combined recipe (Mixtral-8x7B, Layer Drop + AWQ)**: 6.05x inference speedup, memory reduced from 87.7 GB to 20.0 GB (77.1% reduction), average benchmark score 66.1 vs. baseline 71.5 (92.4% retention). The compressed model fits on a single RTX 3090.
- **Expert Drop vs. Layer Drop**: Expert Drop at 25% removal causes a 23% MMLU drop with only 1.06x speedup. Layer Drop at 25% removal causes a 7-point average drop with 1.19x speedup and much better memory reduction.
- **Post-finetuning recovery (DeepSeek-MoE-16B)**: After SFT, Block Drop + AWQ recovers to within 0.6% of the uncompressed baseline average score (62.7 vs. 63.3).

---

## Relevance

The Layer Drop and Block Drop findings directly inform which MoE layers are candidates for lower-precision treatment in mixed-precision inference on Blackwell GPUs: layers with high input-output similarity are both safe to drop and likely safe to quantize more aggressively, providing a principled criterion for per-layer precision assignment.

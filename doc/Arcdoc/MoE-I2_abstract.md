# MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition

## Paper Metadata

- **Title**: MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition
- **Authors**: Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Yuanlin Duan, Wenqi Jia, Miao Yin, Yu Cheng, Bo Yuan
- **Venue**: arXiv preprint (no conference venue listed)
- **arXiv ID**: arXiv:2411.01016
- **Year**: 2024

---

## Problem

MoE LLMs like Mixtral-8x7B are large and memory-hungry because they carry many experts, most of which are rarely activated. Post-training compression of MoE models is challenging because redundancy exists at two levels: across experts (some experts are nearly interchangeable) and within experts (individual expert weight matrices are low-rank). Existing methods address only one level at a time, and naive uniform pruning ratios ignore the fact that different layers have different sensitivity to expert removal. MoE-I2 targets both levels jointly with a calibration-driven, non-uniform compression strategy.

---

## Approach

MoE-I2 proceeds in two sequential stages. In the first stage (inter-expert pruning), it measures each expert's importance by the loss increase when that expert is removed from its layer on a calibration set:

`I_{i,j} = sum_B L(X, {E_i} \ {e_{i,j}})`

Layer importance is the sum of expert importances in that layer, and this is used to assign non-uniform pruning ratios: more important layers lose fewer experts. Within each layer, a genetic search finds the best combination of experts to prune. The population (N=100 combinations) is scored by layer-output Frobenius distance, evolved over 50 generations via selection, union-based crossover, and random mutation. To capture cross-layer interactions, a block-wise KT-Receptive Field step groups T consecutive layers, keeps the top-K candidate combinations per layer, and brute-forces over K^T cross-layer choices to minimize block output loss.

In the second stage (intra-expert decomposition), the remaining experts are compressed with non-uniform low-rank SVD. Each expert's target rank is:

`R_{i,j} = floor( ((I_{i,j} + eps)^alpha / sum_j (I_{i,j} + eps)^alpha) * R_a * M_i' )`

where `alpha=0.15` controls sensitivity weighting, `R_a` is the target average rank, and `M_i'` is the number of surviving experts. More important experts receive higher rank budgets. Finally, LoRA fine-tuning on a small calibration set recovers accuracy lost during both compression stages.

---

## Key Results

- **Mixtral-8x7B at ~50% expert-parameter reduction**: MoE-I2 achieves avg zero-shot 64.55 with 43.49 GB memory and 1.28x inference speedup, versus EEP (the best prior method) at 61.33 with 45.78 GB and 1.20x speedup.
- **25% pruning + LoRA fine-tuning**: Average accuracy recovers to 67.65 on Mixtral-8x7B (slightly above the 67.53 baseline), showing that short LoRA fine-tuning is nearly lossless at moderate compression.
- **DeepSeek-V2-Lite**: After 53.98% expert-parameter reduction, average accuracy is 59.62 vs baseline 61.49, with memory cut from 29.26 GB to 15.03 GB.

---

## Relevance

MoE-I2's importance-driven, non-uniform expert pruning and rank assignment directly informs the expert-level granularity decisions in mixed-precision MoE inference on Blackwell: the same importance scores that guide pruning ratios could guide per-expert bit-width assignment, allocating FP8 to high-importance experts and FP4 to low-importance ones.

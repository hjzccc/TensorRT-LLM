# QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts

## Paper Metadata

- **Title**: QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts
- **Authors**: Pingzhi Li, Xiaolong Jin, Zhen Tan, Yu Cheng, Tianlong Chen
- **Venue**: arXiv preprint (ACM template placeholder; no confirmed venue)
- **arXiv ID**: arXiv:2406.08155v2
- **Year**: 2024 (v2: February 2025)

---

## Problem

Uniform bit-width PTQ treats all MoE components identically, but different components (attention, shared experts, token-conditioned experts, early vs. late blocks) have very different sensitivity to quantization. Applying the same low bit-width everywhere wastes precision on robust components while under-protecting fragile ones. There is no systematic benchmark or principled heuristic set for deciding which MoE structures deserve more bits, and no lightweight predictor for block-level importance that avoids expensive search.

---

## Approach

The paper builds on GPTQ weight-only quantization (group size 128, calibration on 512 WikiText sequences) and systematically evaluates a set of structure-aware mixed-precision heuristics on Mixtral-8x7B and DeepSeek-MoE-16B-base. The core GPTQ objective minimizes `||WX - W_hat X||_2^2` using a Hessian `H = 2XX^T` for greedy column-wise updates.

Four structural heuristics are evaluated independently and in combination. (1) **Attention priority**: attention layers get 4-bit or 8-bit while FFN experts stay at 2-bit, motivated by the observation that attention is far more sensitive than FFN weights. (2) **Shared-expert priority**: in DeepSeek-style architectures, shared experts that process every token get higher precision than token-conditioned experts. (3) **Frequency-based expert priority**: expert activation frequency is measured on calibration data as `usage = normalize(sum_i G(W^l X_i))`; more frequently activated experts receive 4-bit while tail experts get 2-bit. (4) **First-block priority**: earlier MoE blocks are assigned higher precision because they show larger accuracy impact when quantized.

Beyond these heuristics, the paper introduces two data-driven selectors. An **outlier-aware linear layer scorer** ranks each FFN linear layer by `max_j(max(|W_{:,j}|) / mean(|W_{:,j}|))`, the ratio of the maximum column element to the column mean, and assigns 4-bit to the top-k highest-scoring layers (those with the largest within-column outliers). A **block importance predictor** trains a lightweight 2-layer FFNN to predict the cosine similarity between a block's input and output tokens; blocks predicted to have high input-output similarity (i.e., they change the representation little) are assigned 2-bit, while blocks predicted to be more transformative get 4-bit.

---

## Key Results

- **Mixtral-8x7B** with combined heuristics (`+Attn+Freq+FirstL`) at 3.51 average bits: 72.83 average zero-shot accuracy vs. 70.69 for uniform 3-bit GPTQ (+2.14 pts).
- **DeepSeek-MoE-16B-base** with combined heuristics (`+Attn+Shared+Freq+FirstL`) at 3.06 average bits: 65.35 average accuracy vs. 64.30 for uniform 3-bit GPTQ (+1.05 pts); the outlier-aware linear scorer alone adds +2.82 pts over random layer selection on DeepSeek.
- Shared-expert priority is the single biggest win on DeepSeek: adding it to attention-priority raises accuracy from 56.77 to 61.55 (+4.78 pts) at only 2.12 average bits.

---

## Relevance

QuantMoE-Bench provides a concrete set of structure-aware bit-allocation heuristics (attention > shared experts > frequent experts > early blocks) that directly inform the mixed-precision assignment policy for MoE inference on Blackwell GPUs, where different precision tiers map to different GEMM kernels.

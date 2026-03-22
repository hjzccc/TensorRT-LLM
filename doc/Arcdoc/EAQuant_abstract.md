# EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization

## Paper Metadata

- **Title**: EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization
- **Authors**: Zhongqian Fu, Tianyi Zhao, Ning Ding, Xianzhi Yu, Xiaosong Li, Yehui Tang, Yunhe Wang
- **Venue**: Preprint (arXiv)
- **arXiv ID**: arXiv:2506.13329v3
- **Year**: 2026

---

## Problem

Post-training quantization (PTQ) for MoE LLMs degrades badly under aggressive bit-widths (W4A4, W3A4, W3A3, W2A4) for three compounding reasons. First, activations carry severe outliers that vary across experts and layers, making a single smoothing transform hard to apply. Second, router logits are highly sensitive to quantization noise: small perturbations shift top-k expert selection, cascading errors through the entire forward pass. Third, expert usage is heavy-tailed, so rarely activated experts receive too few calibration tokens to estimate good quantization parameters.

---

## Approach

EAQuant introduces three expert-aware components that each target one of the above failure modes, all applied at calibration time with no inference overhead.

**Expert-Aware Smoothing Aggregation (EA-SA)** builds a single unified activation-scaling vector shared across all experts in a layer. Standard per-expert smoothing computes a scale `s_j^i = max(|x_j^i|)^alpha / max(|W_j^i|)^(1-alpha)` for each expert `i` and channel `j`, but these per-expert scales cannot all be fused into the single preceding RMSNorm. EAQuant instead takes the channel-wise maximum across all experts:

```
s_j = max_i( max(|x_j^i|)^alpha / max(|W_j^i|)^(1-alpha) )
```

The router gate is included in the same aggregation so the shared scale suppresses outliers for both routing and computation. The scale is absorbed into RMSNorm as `gamma' = gamma / s`, preserving mathematical equivalence for every expert and the router while adding zero runtime cost.

**Expert-Aware Routing Consistency Alignment (EA-RCA)** protects the router from quantization-induced routing changes. It optimizes router quantization parameters by minimizing a combined loss: the standard weight reconstruction error plus a KL divergence between full-precision and quantized routing probabilities, restricted to the top-k experts plus a fraction of the next-ranked candidates (`m = k + floor(alpha * (n - k))`). This KL-Top term directly penalizes shifts in the routing probability mass near the top-k boundary, keeping expert selection stable.

**Expert-Aware Calibration Data Balance (EA-CDB)** addresses the sparse-expert calibration problem. Starting from 128 WikiText2 sequences, it profiles expert activation counts and oversamples new batches for any expert that falls below a threshold `r * kN / n` (where `r` is a magnification ratio, `k` is top-k, `N` is total tokens). The augmented calibration set is used only for the local expert quantization step, improving parameter estimates for tail experts without synthetic data.

---

## Key Results

- **W4A4 on Mixtral-8x7B**: EAQuant improves average zero-shot accuracy from 73.06 to 74.21 (+1.15 pts) over DuQuant, with WikiText2 PPL dropping from 4.47 to 4.44.
- **W3A3 on Mixtral-8x7B**: accuracy jumps from 53.05 to 62.46 (+9.41 pts); WikiText2 PPL from 11.18 to 7.46.
- **W2A4 on Mixtral-8x22B**: DuQuant collapses to PPL 108.66; EAQuant holds at 7.07, a 15x improvement in perplexity at 2-bit weights.

---

## Relevance

EAQuant's unified cross-expert smoothing and router-consistency regularization directly address the two biggest accuracy hazards in mixed-precision MoE quantization, making it a strong calibration-time baseline for selecting per-expert bit-widths in a Blackwell-deployed MoE inference stack.

# MoQE: Improve Quantization Model Performance via Mixture of Quantization Experts

## Paper Metadata

- **Title**: MoQE: Improve Quantization Model Performance via Mixture of Quantization Experts
- **Authors**: Jinhao Zhang, Yunquan Zhang, Boyang Zhang, Zeyu Liu, Daning Cheng
- **Venue**: Under review at ICLR 2026
- **arXiv ID**: arXiv:2508.09204v2 [cs.LG]
- **Year**: 2025

---

## Problem

Quantizing LLMs degrades accuracy, and no single quantization method (GPTQ, SmoothQuant, AWQ, K-Quants, imatrix) is uniformly best across all inputs. Different quantization schemes produce models that excel on different data subsets, with per-sample accuracy gaps reaching up to 4.7% in vision tasks. The challenge is exploiting this heterogeneity at inference time without paying the cost of running multiple quantized models.

---

## Approach

MoQE treats multiple quantized variants of the same base model as a "mixture of quantization experts." Given a full-precision model, the authors produce N frozen quantized copies using different methods (e.g., GPTQ, SmoothQuant, K-Quants, imatrix). A lightweight router is then trained to predict, for each input, which quantized expert will produce the lowest loss. The router is supervised by labeling each training sample with its optimal expert:

```
j*(x_i) = argmin_{j in [M]} l_j(x_i, y_i)
```

Only the router is trained; all quantized experts remain frozen throughout. The training objective combines cross-entropy with a load-balancing regularizer to prevent expert collapse:

```
L = L_CE + alpha_dyn * L_bal
L_bal = N * sum_{i=1}^{N} P_i * F_i
alpha_dyn = alpha * (1 + sigma_t)
```

where `P_i` is the mean routing probability for expert `i`, `F_i` is the fraction of samples dispatched to it, and `sigma_t` is the relative standard deviation of expert usage (which drives `alpha_dyn` higher when load is imbalanced). The balancing weight `alpha` starts at 0.02 and decays linearly to zero near the end of training.

For NLP, the router reuses the frozen full-precision embedding layer, passes tokens through a Transformer encoder and self-attention refinement module, then an MLP to produce routing logits. For vision, a 3-layer MLP combined with a 3-stage SEResNet-8 and 8-head self-attention serves as the router. At inference, only the selected expert and the router reside on GPU; remaining experts stay in CPU RAM and are loaded on demand, keeping peak VRAM close to a single quantized model.

---

## Key Results

- **NLP (Int8, perplexity on WikiText-2)**: MoQE with 4 experts outperforms the best single quantized expert on every tested model. On LLaMA-3B, MoQE achieves 11.01 perplexity vs. the best single expert at 11.82, and even beats FP16 (12.46) on OpenWebText.
- **Vision (Int8, ImageNet top-1)**: MoQE reaches 78.01% on ResNet-50 vs. 77.09% for the best single method (QAT), closing 87% of the gap to FP16 (80.21%).
- **Inference overhead**: Extra inference time from routing and expert loading stays below 8% in all tested configurations on a V100S, with most cases under 3%. Peak VRAM overhead vs. a single expert is under 500 MB for all tested models.

---

## Relevance

MoQE's per-request expert selection and CPU-offload strategy directly informs mixed-precision MoE inference on Blackwell GPUs, where different quantization formats (FP8, INT4, INT8) can be assigned to different expert slots and selected dynamically based on input characteristics.

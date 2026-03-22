# ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization

## Paper Metadata

- **Title**: ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization
- **Authors**: Zechun Liu, Changsheng Zhao, Hanxian Huang, Sijia Chen, Jing Zhang, Jiawei Zhao, Scott Roy, Lisa Jin, Yunyang Xiong, Yangyang Shi, Lin Xiao, Yuandong Tian, Bilge Soran, Raghuraman Krishnamoorthi, Tijmen Blankevoort, Vikas Chandra
- **Venue**: NeurIPS 2025 (39th Conference on Neural Information Processing Systems)
- **arXiv ID**: arXiv:2502.02631v2
- **Year**: 2025

---

## Problem

Prior work on extremely low-bit LLM quantization (1-bit through 4-bit) produced inconsistent conclusions about which bit-width offers the best accuracy-vs-model-size trade-off, because comparisons ignored the interaction between model size, training token budget, and the quantization function itself. There was no unified framework that jointly optimized training strategy and quantizer design across bit-widths, making apples-to-apples scaling-law comparisons impossible. ParetoQ fills this gap by formalizing the problem and providing bit-specific quantizers and training schedules that enable fair Pareto-frontier analysis.

---

## Approach

ParetoQ formalizes quantized model loss as a function of five factors: model size N, training tokens D, bit precision P, training strategy S_train, and quantization function F. The first key finding is that the optimal training budget split is roughly 90% full-precision pretraining followed by 10% QAT, and that fine-tuning from a pretrained FP checkpoint consistently beats training quantized models from scratch. Token requirements differ by regime: 3-bit and 4-bit QAT saturates around 10B tokens, while 1-bit, 1.58-bit, and 2-bit need around 30B tokens because they undergo "reconstruction" (weights deviate ~40% from initialization) rather than the "compensation" behavior seen at higher bits (~10-20% deviation).

The second key contribution is a unified, bit-specific quantizer called ParetoQ. For binary (1-bit), it uses elastic binarization: `W^Q = alpha * Sign(W^R)` with `alpha = ||W^R||_1 / n`. For ternary (1.58-bit) and 2-bit, it introduces SEQ (Stretched Elastic Quant): `W^Q = alpha * (floor(Clip(W/alpha, -1, 1) * k/2 - 0.5) + 0.5) / k * 2`, which ensures balanced quantization levels and range coverage. For 3-bit and 4-bit, it uses LSQ: `W^Q = alpha * floor(Clip(W/alpha, n, p))`. All variants are trained with straight-through estimators (STE) for both weight and scale gradients. The result is a single framework that produces Pareto-optimal models at each bit-width, enabling the conclusion that 1.58-bit, 2-bit, and 3-bit quantization generally dominate 4-bit on the accuracy-per-effective-model-size frontier.

---

## Key Results

- **2-bit vs 4-bit Pareto dominance**: A 2-bit MobileLLM-1B is 1.8 accuracy points better than a 4-bit MobileLLM-600M while also being smaller in effective size.
- **LLaMA-3 8B, 2-bit**: ParetoQ achieves avg zero-shot 71.2 (WikiText2 PPL 8.0), only 3.4 points below full precision and +5.7 points over the best prior QAT method (EfficientQAT).
- **GPU kernel speedup**: For a 16384x16384 weight matrix, the 2-bit kernel is 4.13x faster than BF16 and 1.24x faster than the 4-bit Machete kernel, implemented on a CUTLASS mixed-precision W2A16 backbone.

---

## Relevance

ParetoQ's finding that 2-bit weights with 16-bit activations (W2A16) sit on the Pareto frontier, combined with its CUTLASS-based mixed-precision kernel implementation, directly informs bit-width selection and kernel design for mixed-precision MoE inference on Blackwell, where expert weights are the dominant memory bottleneck and per-expert bit-width assignment is a key degree of freedom.

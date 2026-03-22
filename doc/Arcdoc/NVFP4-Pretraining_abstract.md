# Pretraining Large Language Models with NVFP4

## Paper Metadata

- **Title**: Pretraining Large Language Models with NVFP4
- **Authors**: NVIDIA (Anjulie Agrusa, Muya Chang, Mike Chrzanowski, Eric Chung, Steve Dai, Bita Darvish Rouhani, Carlo del Mundo, Brucek Khailany, and many others)
- **Venue**: NVIDIA technical report on arXiv (no conference venue listed)
- **arXiv ID**: arXiv:2509.25149v2
- **Year**: 2026

---

## Problem

Training large language models in 4-bit floating point is unstable at scale: FP4's limited dynamic range causes outlier saturation, quantization bias accumulates over trillions of tokens, and inconsistency between forward and backward quantization breaks the chain rule. Prior FP4 formats (MXFP4) use coarse block sizes and integer-only block scales that exacerbate these problems. This paper demonstrates stable, accurate multi-trillion-token LLM pretraining using NVIDIA's NVFP4 format, closing the quality gap to FP8 while achieving up to 6x higher throughput on Blackwell hardware.

---

## Approach

NVFP4 uses E2M1 elements with block size 16 (half of MXFP4's 32), FP8 E4M3 block scales, and an additional FP32 per-tensor scale. This two-level scaling scheme quantizes each block as:

`s_enc,b = 1 / (fp32(e4m3(amax_b / 6 * s_enc)) * s_dec)`

where `s_enc = (6 * 448) / amax_x` is the global tensor encode scale and `amax_b` is the per-block maximum. The finer block size better isolates outliers and reduces the saturation that plagues MXFP4.

Four techniques make training stable at the 12B-parameter, 10T-token scale. First, sensitive layers are kept in higher precision: for the 12B model, the first 2 and last 8 transformer blocks remain in BF16, covering about 16% of linear layers. Embeddings, normalization, attention softmax, and optimizer states stay in BF16/FP32 throughout. Second, Random Hadamard Transforms (RHT) are applied to weight-gradient GEMM inputs only, using a 16x16 Hadamard matrix to spread outliers toward a more Gaussian distribution and reduce FP4 gradient error. Third, weights use 2D block scaling (16x16 blocks) so the same quantized representation is used in both forward and backward passes, preserving chain-rule consistency. Activations and gradients use 1x16 scaling. Fourth, stochastic rounding is applied to gradients (but not to weights or activations), reducing deterministic quantization bias in the backward pass.

An optional hybrid schedule switches from NVFP4 to BF16 at 8.2T tokens (using ~6% of total compute in higher precision), recovering most of the residual loss gap.

---

## Key Results

- **12B model, 10T tokens**: NVFP4 validation loss tracks FP8 with relative error below 1% during the stable phase and slightly above 1.5% near end-of-training LR decay. Downstream accuracy is comparable: MMLU 76.57 (NVFP4) vs 77.36 (FP8), with NVFP4 actually outperforming FP8 on several benchmarks (GSM8K CoT: 92.27 vs 89.08).
- **NVFP4 vs MXFP4 (8B, 1T tokens)**: NVFP4 achieves ~1.5% relative loss error vs BF16; MXFP4 needs 36% more tokens (1.36T) to match NVFP4 trained on 1T tokens.
- **Hardware throughput**: NVFP4 delivers 4x speedup over BF16 on GB200 and 6x on GB300, with ~half the memory footprint of FP8.

---

## Relevance

NVFP4 pretraining is the upstream foundation for deploying FP4-quantized MoE models on Blackwell: models pretrained natively in NVFP4 avoid post-training quantization accuracy loss, and the paper's mixed-precision recipe (FP4 for most linear layers, BF16 for sensitive early/late blocks) directly maps to the per-expert precision assignment problem in mixed-precision MoE inference on GB200/GB300.

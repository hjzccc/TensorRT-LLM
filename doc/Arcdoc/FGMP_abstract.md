# FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference

## Paper Metadata

- **Title**: FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference
- **Authors**: Coleman Hooper, Charbel Sakr, Ben Keller, Rangharajan Venkatesan, Kurt Keutzer, Sophia Shao, Brucek Khailany
- **Venue**: arXiv preprint (no conference venue listed)
- **arXiv ID**: arXiv:2504.14152v1
- **Year**: 2025

---

## Problem

Aggressive W4A4-style quantization of LLMs causes accuracy loss because a small fraction of weight and activation values are highly sensitive to quantization error, and outliers distort the quantization range for entire channels or layers. Existing mixed-precision methods operate at coarse granularity (channel, row, or layer level), which cannot adapt to the unstructured distribution of sensitive values. FGMP proposes fine-grained mixed-precision assignment at the 1D block level, keeping only the most sensitive blocks in FP8 while quantizing the rest to NVFP4, and co-designs hardware to execute this efficiently.

---

## Approach

FGMP assigns each 1D block of 16 elements to either FP8 (high precision) or NVFP4 (low precision) based on a sensitivity score derived from first-order Taylor expansion of the loss. For a block v with gradient g, the impact of switching from high to low precision is:

`I'_L(v) = sum_i g_i^2 * (Delta_{p_h -> p_l} v_i)^2`

where `Delta_{p_h -> p_l} v_i = Q_{p_l}(v_i) - Q_{p_h}(v_i)` is the additional quantization error from downgrading precision. For weights, squared gradients are averaged over a 512-sample calibration set (Wikitext-103). For activations, per-channel squared gradients are pre-computed offline and used as a runtime sensitivity proxy, since true gradients are unavailable during inference. A single global threshold (separately for weights and activations) is applied across the entire model, so more sensitive layers automatically receive more FP8 blocks without per-layer tuning.

For low-precision weight blocks, FGMP additionally optimizes the NVFP4 microscaling factor by brute-force search over FP8 candidate scales to minimize sensitivity-weighted quantization error. Activations are quantized online by a dedicated post-processing unit (PPU) that computes FP4 and FP8 quantization errors per block, weights them by the pre-calibrated channel sensitivity, compares against a threshold, and emits FP4 or FP8 accordingly. The hardware datapath supports four dot-product modes (FP4xFP4, FP8xFP8, FP4xFP8, FP8xFP4) with a single metadata bit per block indicating precision. The PPU adds only ~0.20 fJ/op amortized overhead, under 1% of dot-product energy.

---

## Key Results

- **Energy vs accuracy on Llama-2-7B**: FGMP achieves less than 1% perplexity degradation relative to all-FP8 while consuming 14% less inference energy and requiring 30% less weight memory (70% of blocks in FP4).
- **MMLU accuracy recovery**: With 70% of blocks in FP4, FGMP recovers 58-89% of the MMLU accuracy gap between FP8 and all-FP4 across Llama-2, GPT3, and Nemotron model families.
- **Calibration cost**: Fisher information computation for Llama-2-7B takes under 3 minutes on one A100 GPU using 512 samples of length 512.

---

## Relevance

FGMP's block-level FP4/FP8 mixed-precision framework with hardware co-design is directly applicable to Blackwell's native NVFP4 and FP8 tensor core support, and its sensitivity-weighted block selection strategy could be applied per-expert in MoE models to allocate higher precision to the most critical expert weight blocks.

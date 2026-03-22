# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

## Paper Metadata
- **Title**: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **Authors**: Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Venue**: NeurIPS 2022 (36th Conference on Neural Information Processing Systems)
- **arXiv ID**: 2208.07339
- **Year**: 2022

## Problem

Post-training 8-bit quantization of large language models degrades perplexity and zero-shot accuracy at multi-billion-parameter scale. Prior methods worked reasonably below ~350M parameters but failed at 6.7B and above. The paper identifies the root cause: beyond ~6B parameters, transformer hidden states develop a small number of feature dimensions (typically 6-7) with magnitudes up to 20x larger than the rest. These "emergent outlier features" appear in nearly every layer and sequence position, and they destroy the dynamic range of standard int8 quantization. The goal is to quantize the feed-forward and attention projection weights (95% of parameters, 65-85% of compute) to 8 bits without any accuracy loss, up to 175B parameters.

## Approach

LLM.int8() combines two techniques: vector-wise quantization and mixed-precision decomposition.

**Vector-wise quantization** treats matrix multiplication as a collection of independent inner products rather than a single tensor-wide operation. Each row of the activation matrix gets its own absmax scaling constant `c_x`, and each column of the weight matrix gets its own constant `c_w`. The output is dequantized using the outer product of scales: `C_f16 = (1 / (c_x ⊗ c_w)) * C_i32 = S * A_i8 * B_i8`. This improves precision for normal-valued weights and activations and works well up to about 2.7B parameters.

**Mixed-precision decomposition** handles the outlier dimensions that vector-wise scaling cannot fix. Let `O` be the set of hidden dimensions where any activation exceeds a threshold `alpha = 6.0`. The matrix multiplication is split into two parts: `C_f16 ≈ sum_{h in O} X^h_f16 * W^h_f16 + S_f16 * sum_{h not in O} X^h_i8 * W^h_i8`. The outlier columns (at most 7 for models up to 13B) are multiplied in FP16; everything else is multiplied in INT8. Since outliers comprise only ~0.1% of all values, the extra memory overhead is negligible. The two partial results are accumulated in FP16.

The method requires no retraining or calibration data. Weights are quantized at load time; activations are quantized on the fly during inference. The outlier threshold `alpha = 6.0` is fixed across all models and layers.

## Key Results

- **Perplexity preservation**: LLM.int8() exactly matches 32-bit C4 perplexity through 13B parameters (e.g., **12.45** for both FP32 and LLM.int8() at 13B), while plain vector-wise absmax degrades to 16.48.
- **Memory reduction**: BLOOM-176B memory footprint is reduced by **1.96x**, enabling the model to run on 4x A100 80GB GPUs instead of 8x, with similar generation latency (~246 ms/token vs. 239 ms/token).
- **Throughput**: Raw matrix multiplication speedup reaches **1.81x** over FP16 at 175B scale; end-to-end generation latency is within ~6% of bfloat16 baseline on 8x A100.

## Relevance

LLM.int8()'s discovery that activation outliers are concentrated in a tiny number of feature dimensions, and its mixed-precision decomposition to handle them, is foundational for understanding why naive INT8/INT4 quantization fails on large LLMs and motivates the outlier-aware mixed-precision strategies used in MoE inference on Blackwell.

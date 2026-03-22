# OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models

## Paper Metadata
- **Title**: OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models
- **Authors**: Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, Eunhyeok Park
- **Venue**: AAAI 2024
- **arXiv ID**: 2306.02272
- **Year**: 2024

## Problem

Quantizing LLM weights to 3 bits or below causes significant accuracy degradation with existing methods like GPTQ/OPTQ. The core issue is that activation outliers in LLMs make certain weight columns far more sensitive to quantization than others: a weight column aligned with a high-magnitude activation feature incurs disproportionately large output error when rounded. GPTQ treats all columns uniformly, missing this structure. OWQ also addresses efficient task-specific fine-tuning of quantized LLMs, where methods like QLoRA require many trainable parameters and significant memory.

## Approach

OWQ formulates layer-wise quantization as minimizing output activation error: `arg min_{W_hat} ||W X - W_hat X||_2^2`. The Hessian of this objective is `H = 2 X X^T`, and the quantization error for weight column `j` can be approximated as `E_i ≈ ΔW_{i,:} H ΔW_{i,:}^T`. Because LLM activations have outliers in specific feature dimensions, the diagonal elements `λ_j` of `H` vary enormously across columns. OWQ defines a per-column sensitivity metric `sensitivity_j = λ_j * ||ΔW_{:,j}||_2^2` that combines Hessian curvature with the actual quantization perturbation.

The top-k most sensitive columns (the "weak columns") are kept in FP16 and excluded from low-bit quantization. The remaining dense weights are quantized with GPTQ-style post-training quantization. Crucially, weak columns are reordered to the end of the quantization sequence so that GPTQ's sequential error compensation can flow into them without being destroyed by subsequent quantization. OWQ also performs a 2D grid search over quantization step size and zero-point (with truncation) to further reduce reconstruction error for the dense portion.

For fine-tuning, OWQ introduces **Weak Column Tuning (WCT)**: after quantization, only the FP16 weak columns are updated for downstream tasks while the low-precision dense matrix is frozen. For LLaMA, WCT uses only `(5d + 2D)k` learnable parameters per decoder block vs. `(11d + 3D)r` for QLoRA, where `d` is hidden size, `D` is FFN intermediate size, `k` is weak column count, and `r` is LoRA rank. With `k = 8` and `r = 64`, WCT uses just 6.8% of QLoRA's parameters.

## Key Results

- **3-bit vs. 4-bit parity**: OWQ at **3.1 bits** matches 4-bit OPTQ perplexity on multiple models (e.g., OPT-6.7B: both 11.14 WikiText-2; LLaMA-65B: OWQ 3.1-bit 4.09 vs. OPTQ 4-bit 4.10).
- **Fine-tuning quality**: WCT with only 8 weak columns (6.8% of QLoRA parameters) **outperforms QLoRA** in GPT-4 pairwise evaluation on LLaMA-7B (wins 68, ties 26, loses 66 out of 160 comparisons).
- **Overhead**: Custom CUDA kernel adds only **~3.2% latency** over OPTQ on LLaMA-7B; a 66B model can be quantized in under 3 hours on a single A100.

## Relevance

OWQ's activation-outlier-aware sensitivity metric and the idea of keeping a tiny fraction of weight columns in higher precision while quantizing the rest aggressively is directly applicable to mixed-precision MoE inference on Blackwell, where expert weight matrices have heterogeneous sensitivity profiles and the cost of per-column precision decisions must be weighed against kernel overhead.

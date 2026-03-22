# MoPEQ: Mixture of Mixed Precision Quantized Experts

## Paper Metadata

- **Title**: MoPEQ: Mixture of Mixed Precision Quantized Experts
- **Authors**: Krishna Teja Chitty-Venkata, Jie Ye, Murali Emani
- **Venue**: arXiv preprint (no confirmed venue)
- **arXiv ID**: arXiv:2509.02512
- **Year**: 2025

---

## Problem

Mixed-precision quantization for MoE models typically assigns higher bit-widths to more frequently activated experts. This works poorly for load-balanced MoE architectures (e.g., DeepSeek-VL2) where routing frequencies are nearly uniform across experts, leaving no signal to differentiate precision. The method also fails to capture experts that are infrequently activated but highly sensitive to quantization error. MoPEQ targets vision-language MoE models (VLM-MoEs) specifically, where memory reduction is critical for deployment while accuracy across diverse multimodal tasks must be preserved.

---

## Approach

MoPEQ focuses quantization decisions on the MoE expert layers while quantizing all other layers uniformly. It assigns per-expert bit-widths from the set {2, 3, 4} based on an importance score, then clusters experts by importance to determine the final mixed-precision assignment.

The key innovation is replacing activation frequency with a **data-free Hessian trace sensitivity** estimate. For each expert, the Hessian trace is approximated via the Hutchinson estimator:

```
Tr(H) ≈ (1/m) * sum_{i=1}^{m} v_i^T H v_i
```

where each `v_i` is a random Gaussian or Rademacher vector. The Hessian-vector product `Hv` is computed as `grad_W(grad_W(||W||_F)^T v)`, requiring only two backward passes per sample with no calibration data. The total sensitivity for expert `i` sums the Hessian traces of its Gate, Up, and Down projection layers: `H_i^G + H_i^D + H_i^U`.

Optionally, a **hybrid importance score** combines normalized activation frequency and normalized Hessian sensitivity multiplicatively:

```
I_i = norm(AF_i) * norm(H_i)
```

where each term is min-max normalized across all experts. This hybrid score is useful when routing is imbalanced enough to provide a meaningful frequency signal.

Expert bit-widths are then assigned via **K-means clustering**. Experts are partitioned into C clusters (one per available precision level), sorted by cluster mean importance, and the highest-importance cluster receives the highest bit-width. The paper evaluates two assignment granularities: **layer-wise** (cluster within each layer independently) and **model-wise** (cluster all experts globally across the entire model). Model-wise assignment generally outperforms layer-wise because it avoids over-fitting the clustering to local layer statistics.

Quantization itself uses AutoRound (SignRound), and the method is evaluated on MolmoE-1B, DeepSeek-VL2-Tiny, DeepSeek-VL2-Small, and DeepSeek-VL2-Base across nine multimodal benchmarks (MME, TextVQA, DocVQA, AI2D, MMMU, InfoVQA, RealWorldQA, ScienceQA, BLINK).

---

## Key Results

- **MolmoE-1B**: Hessian model-wise MoPEQ achieves MME-Perception 1338.09 at 3.41 GB, beating uniform 4-bit (1300.09 at 4.08 GB) with a smaller model.
- **DeepSeek-VL2-Tiny**: hybrid model-wise MoPEQ reaches MME-Perception 1585.69 at 2.35 GB vs. uniform 4-bit at 1563.25 at 2.64 GB.
- **DeepSeek-VL2-Base**: Hessian model-wise reduces model size from 14.35 GB to 10.49 GB while improving MME-Reasoning (576.43 vs. 539.29) and MME-Perception (1623.45 vs. 1614.61) over uniform 4-bit; model-wise assignment wins in 63 of 105 evaluated scenarios vs. 42 for layer-wise.

---

## Relevance

MoPEQ's data-free Hessian sensitivity scoring provides a calibration-free way to rank expert quantization sensitivity, directly applicable to assigning INT4/INT8/FP8 precision tiers to MoE experts in a Blackwell inference engine without needing representative input data.

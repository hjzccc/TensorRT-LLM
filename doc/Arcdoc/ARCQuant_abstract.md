# ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs

## Paper Metadata

- **Title**: ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs
- **Authors**: Haoqian Meng, Yilun Luo, Yafei Zhao, Wenyuan Liu, Peng Zhang, Xindian Ma
- **Venue**: arXiv preprint (no conference venue listed)
- **arXiv ID**: arXiv:2601.07475v1
- **Year**: 2026

---

## Problem

Post-training W4A4 quantization under NVFP4 suffers from large activation errors because outlier channels dominate the quantization range. Rotation-based methods (QuaRot, SpinQuant) spread outliers across channels but destroy NVFP4's fine-grained block isolation. Smoothing is too weak under 4-bit activation error. Mixed-precision methods (Atom, FGMP) conflict with Blackwell Tensor Core constraints: NVFP4 uses group size 16 while higher-precision microscaling formats (MXFP6/MXFP8) use group size 32, so mixing them in a single GEMM requires non-standard data paths. ARCQuant achieves mixed-precision-like accuracy recovery while keeping a strictly unified NVFP4 execution path.

---

## Approach

ARCQuant identifies the top-S outlier activation channels using an adaptive threshold: channels whose absolute maximum exceeds `tau = 2^(-3) * M` (where M is the layer maximum) are flagged as outliers. The threshold is motivated by the fact that E5M2 FP8 has 3 more exponent bits than E2M1 FP4, so channels below tau are already in a range where NVFP4 precision is close to FP8 and need no special treatment.

For the flagged outlier channels, ARCQuant computes a quantized residual online:

`R_o = X_o - s_{X_o} * Q_{X_o}`

and quantizes this residual to a second NVFP4 block. The augmented activation tensor is then:

`QX_aug = [QX | QR_o]`  with scales  `sX_aug = [sX | sR_o]`

On the weight side, the corresponding outlier weight columns are simply duplicated (not residualized):

`QW_aug = [QW | QW_o]`

This means the augmented GEMM computes `Q(X) Q(W)^T + Q(R_o) Q(W_o)^T` in a single extended GEMM of shape `(N, K_in + S, M)` rather than two separate GEMMs. No mixed-precision branch is needed; the entire computation runs through standard NVFP4 tensor cores. A fused kernel combines channel reordering, RMSNorm, primary quantization, and residual quantization in one pass, using an interleaved channel layout that keeps primary and residual blocks GEMM-friendly.

The paper also provides an error-bound analysis showing that ARCQuant's two-stage NVFP4 worst-case error bound (`alpha_1 * alpha_2 * M * eps_4^2 = alpha_1 * alpha_2 * M * eps_8`, with `sup(alpha_1 * alpha_2) ~= 1.266`) is tighter than single-stage MXFP8 (`sup(alpha_mx) = 2`), giving a theoretical justification for why the residual correction is effective.

---

## Key Results

- **Llama 3.1-8B**: ARCQuant achieves avg zero-shot 70.90 and WikiText2 PPL 6.87, the best among all NVFP4-only methods and competitive with W4A8 baselines (RTN: 70.59, FlatQuant: 70.51, Atom: 67.74).
- **Qwen2.5-Coder-7B-Instruct**: ARCQuant scores 86.0 on HumanEval, exceeding FP16 (84.1) and far above Atom (80.5), while adding only 4.9% end-to-end prefill latency overhead versus uncompensated NVFP4.
- **RTX 5090 speedup**: Llama 3.1-8B prefill (batch 4, len 2048) runs in 270.50 ms with ARCQuant vs 917.05 ms for FP16, a 3.4x speedup, with memory reduced from 21.92 GB to 10.27 GB.

---

## Relevance

ARCQuant is directly relevant to mixed-precision MoE inference on Blackwell: it solves the hardware-compatibility problem of achieving mixed-precision accuracy within a unified NVFP4 execution path, and its augmented-channel approach could be applied per-expert to handle the activation outliers that are especially pronounced in MoE models with heterogeneous expert activation patterns.

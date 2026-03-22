# HAWQ-V3: Dyadic Neural Network Quantization

## Paper Metadata
- **Title**: HAWQ-V3: Dyadic Neural Network Quantization
- **Authors**: Zhewei Yao, Zhen Dong, Zhangcheng Zheng, Amir Gholami, Jiali Yu, Eric Tan, Leyuan Wang, Qijing Huang, Yida Wang, Michael W. Mahoney, Kurt Keutzer
- **Venue**: ICML 2021 (Proceedings of the 38th International Conference on Machine Learning, PMLR 139)
- **arXiv ID**: 2011.10680
- **Year**: 2021

## Problem

Existing quantization methods often retain hidden floating-point operations during inference, for example in requantization, batch normalization, or residual addition. This means the model cannot be deployed on integer-only hardware and the claimed speedups are not fully realized. HAWQ-V3 targets fully integer-only inference, including INT4 and mixed INT4/INT8, with no FP32 casting anywhere in the forward pass. A secondary problem is choosing per-layer bitwidths in a hardware-aware way that balances accuracy against actual latency on the target device, not just proxy metrics like FLOPs.

## Approach

The core quantization formula maps a real value `r` to an integer via `Q(r) = Int(r / S) - Z`, where `S` is the scale and `Z` is the zero-point. For a convolution with quantized activation `S_h * q_h` and weight `S_w * q_w`, the output before requantization is `a = S_w * S_h * (q_w * q_h)`, where the inner product is integer arithmetic. Requantization then needs to multiply by `S_w * S_h / S_a`, which would normally require floating-point division. HAWQ-V3 constrains this ratio to a **dyadic number** `b / 2^c` (integers `b` and `c`), so requantization becomes an integer multiply followed by a bit shift, with no floating-point at all.

Batch normalization is fused into the preceding convolution to produce a fused weight `W_bar` and bias `b_bar`. The bias is stored in INT32 and the fused weight is quantized to 4 or 8 bits. Residual addition is handled by dyadically rescaling each branch: `q_a = DN(S_m / S_a) * q_m + DN(S_r / S_a) * q_r`, where `DN(.)` denotes dyadic approximation. The paper shows that fake quantization (which accumulates residuals in FP32) introduces O(1) errors that propagate through the network, with final feature-map differences exceeding 95% for 4-bit ResNet50.

For mixed-precision selection, HAWQ-V3 uses a Hessian-based sensitivity metric to estimate per-layer quantization perturbation, then solves an integer linear program (ILP) to minimize total perturbation subject to constraints on model size, BOPS, or latency. Layer latencies are measured directly on the target hardware (NVIDIA T4 Tensor Cores) rather than estimated analytically. The ILP is solved with PuLP in under one second. Deployment uses Apache TVM extended to support INT4 and mixed INT4/INT8 inference, with all accuracy results verified by direct hardware execution.

## Key Results

- **INT8 accuracy**: HAWQ-V3 INT8 achieves **77.58% top-1** on ResNet50 (vs. 74.90% for prior integer-only INT8 work by Jacob et al.), a +2.68 point improvement; on InceptionV3, **78.76%** vs. 74.20% (+4.56 points).
- **INT4 speedup**: Uniform INT4 gives an average **1.45x speedup** over INT8 on ResNet50 on T4 GPUs.
- **Mixed INT4/INT8**: Mixed-precision ResNet50 achieves **76.73% top-1** (with distillation) at 1.23x speedup over INT8, recovering 2.49 points over uniform INT4 with only modest extra memory.

## Relevance

HAWQ-V3's dyadic integer-only inference and hardware-aware ILP for mixed INT4/INT8 selection directly inform mixed-precision MoE inference on Blackwell GPUs, where INT4 Tensor Core support and the need for true integer-only execution (no FP32 fallback) are central design constraints.

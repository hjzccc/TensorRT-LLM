# HAQ: Hardware-Aware Automated Quantization with Mixed Precision

## Paper Metadata
- **Title**: HAQ: Hardware-Aware Automated Quantization with Mixed Precision
- **Authors**: Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, Song Han
- **Venue**: arXiv preprint (arXiv:1811.08886v3 [cs.CV])
- **arXiv ID**: 1811.08886
- **Year**: 2019

## Problem

Neural network quantization typically assigns the same bitwidth to every layer, or relies on human heuristics to pick per-layer precision. This ignores two key facts: different layers have different redundancy, and the optimal bitwidth depends heavily on the target hardware (e.g., a memory-bound edge FPGA behaves very differently from a compute-bound cloud accelerator). HAQ addresses the problem of automatically finding per-layer weight and activation bitwidths that satisfy a hardware resource constraint (latency, energy, or model size) on a specific target device, without manual tuning.

## Approach

HAQ frames mixed-precision quantization as a reinforcement learning problem. An agent processes the network layer by layer. For each layer it makes two sequential decisions: one for weight bitwidth and one for activation bitwidth. The state fed to the agent for layer k encodes layer geometry (input/output channels, kernel size, stride, feature map size, parameter count, depthwise flag) plus a flag indicating whether the current decision is for weights or activations, and the previous action. All dimensions are normalized to [0, 1].

The agent outputs a continuous action `a_k in [0, 1]`, which maps to a discrete bitwidth via `b_k = round(b_min - 0.5 + a_k * (b_max - b_min + 1))`, with `b_min = 2` and `b_max = 8`. Using a continuous action space preserves bitwidth ordering, which matters for stable learning. After all layers are assigned, the resource usage is measured directly on hardware or a cycle-accurate simulator (BISMO or BitFusion on Xilinx FPGAs). If the policy exceeds the budget, layer bitwidths are sequentially reduced until the constraint is met. The reward is purely accuracy-based: `R = lambda * (acc_quant - acc_origin)`, since resource constraints are enforced through action restriction rather than reward shaping.

The agent is trained with DDPG (deep deterministic policy gradient). One episode covers all layers of the network. After each episode, the quantized model is finetuned for one epoch on a 100-category ImageNet subset before evaluating reward. For the final policy, the model is finetuned on the full dataset. Weights use linear quantization with KL-divergence-based clipping threshold selection; for model-size experiments, k-means quantization (following Deep Compression) is used instead.

A key empirical finding is that the optimal policy differs substantially across hardware targets. On edge accelerators (memory-bound), HAQ assigns fewer activation bits to depthwise convolutions because activations dominate memory traffic. On cloud accelerators (more bandwidth), it gives more bits to depthwise layers and fewer to pointwise layers.

## Key Results

- **Latency**: On MobileNet-V2 with BISMO edge accelerator, HAQ reaches near-8-bit accuracy (70.90% top-1) at 66.92 ms vs. 115.84 ms for 8-bit, a **1.73x speedup** with negligible accuracy loss.
- **Energy**: On MobileNet-V1 with BitFusion, HAQ achieves 70.37% top-1 at 16.30 mJ vs. 31.03 mJ for 8-bit, roughly a **1.9x energy reduction**.
- **Model size**: On MobileNet-V1 at ~1.09 MB, HAQ reaches **57.14% top-1** vs. 37.62% for Deep Compression at the same size, a 19.5-point improvement.

## Relevance

HAQ's hardware-in-the-loop RL framework for per-layer mixed-precision selection is a direct conceptual ancestor of mixed-precision MoE inference on Blackwell GPUs, where different expert layers and projection types have different sensitivity and different hardware utilization profiles that a single global bitwidth cannot capture.

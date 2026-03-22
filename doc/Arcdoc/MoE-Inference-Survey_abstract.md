# A Survey on Inference Optimization Techniques for Mixture of Experts Models

## Paper Metadata
- **Title**: A Survey on Inference Optimization Techniques for Mixture of Experts Models
- **Authors**: Jiacheng Liu, Peng Tang, Wenfeng Wang, Yuhang Ren, Xiaofeng Hou, Pheng-Ann Heng, Minyi Guo, Chao Li
- **Venue**: Manuscript submitted to ACM (preprint)
- **arXiv ID**: arXiv:2412.14219v2 [cs.LG]
- **Year**: 2025 (submitted January 22, 2025)

---

## Problem

MoE models are theoretically efficient because they activate only a subset of experts per token, but deploying them efficiently is hard. Dynamic routing creates load imbalance across experts, all-to-all communication dominates inter-node bandwidth, and the total parameter count (even with sparse activation) creates severe memory pressure. These problems compound across model-level design, system-level scheduling, and hardware-level execution, and no single prior work synthesizes the full optimization landscape.

---

## Approach

This is a survey paper. Its contribution is a three-level taxonomy of MoE inference optimization methods, synthesizing results across dozens of systems and papers.

**Model-level optimization** covers techniques that reduce the cost of the model itself before deployment. Expert pruning removes low-importance experts entirely. Quantization assigns lower bit-widths to less-critical experts, with methods like MC-MoE using access frequency `phi_i = n_i / N` and activation weight `w_i = sum_j sigma_j / N` to solve an integer program for per-expert bit allocation. Decomposition methods like MoE-I2 allocate rank `r_{i,j}` to each expert proportional to its importance score `I_{i,j}` raised to a power alpha, then apply low-rank factorization. Dynamic gating methods skip experts whose gate scores fall below a threshold, reducing active FLOPs without retraining.

**System-level optimization** covers runtime scheduling and memory management. Expert parallelism strategies partition experts across GPUs and must solve load balancing (unequal token counts per expert) and all-to-all communication overhead. Systems like ScheMoE overlap computation with communication by pipelining expert dispatch. Expert offloading systems (Fiddler, MoE-Lightning, MoE-Infinity) keep only a subset of experts in GPU memory and prefetch the next layer's experts from CPU DRAM while the GPU computes the current layer, using caching policies to exploit expert reuse across requests.

**Hardware-level optimization** covers custom accelerators: near-data processing (PIM/AiM), FPGA implementations, and mobile-targeted sparse MoE accelerators that co-design routing and compute to match the sparse, dynamic access pattern of MoE inference.

---

## Key Results

- **Quantization**: MC-MoE achieves 4.27x memory reduction and 1.80x speedup with 3.8% accuracy drop; QMoE achieves 20x memory reduction with 6.7% accuracy drop.
- **Offloading on Mixtral-8x7B**: Fiddler achieves 8.20x throughput improvement; MoE-Lightning achieves 3.50x; MoE-Infinity achieves 1.96x over baseline offloading.
- **Distributed inference systems**: DeepSpeed-MoE achieves 7.25x over PyTorch; Brainstorm achieves 3.33x; Prophet achieves 1.75x to 12.06x load-balance improvement.

---

## Relevance

This survey maps the full design space for MoE inference optimization, providing a structured reference for choosing quantization schemes, offloading strategies, and parallelism configurations when targeting mixed-precision MoE inference on Blackwell GPUs.

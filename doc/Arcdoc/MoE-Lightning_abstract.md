# MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

## Paper Metadata
- **Title**: MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
- **Authors**: Shiyi Cao, Shu Liu, Tyler Griggs, Peter Schafhalter, Xiaoxuan Liu, Ying Sheng, Joseph E. Gonzalez, Matei Zaharia, Ion Stoica
- **Venue**: ACM (DOI: 10.1145/3669940.3707267, pp. 715-730)
- **arXiv ID**: arXiv:2411.11217
- **Year**: 2024

> **Note**: The PDF at `/tmp/papers/MoE-Lightning.pdf` contained a different paper (arXiv:2411.17217, SAM anomaly segmentation). This abstract was reconstructed from the arXiv HTML version of the correct paper.

---

## Problem

Large MoE models like Mixtral 8x22B require over 256 GB of memory for expert FFN weights alone, 4-5x more than dense models with equivalent active FLOPs. This makes them inaccessible on consumer or low-cost GPUs. The standard workaround is to offload weights to CPU DRAM and load them layer-by-layer, but existing offloading systems (e.g., FlexGen) leave GPU and CPU idle while waiting for data transfers, achieving poor resource utilization. The core challenge is scheduling GPU computation, CPU computation, and CPU-to-GPU transfers so that no resource blocks another.

---

## Approach

MoE-Lightning introduces two components: **CGOPipe**, a CPU-GPU-I/O pipeline schedule, and **HRM**, a Hierarchical Roofline Model used as a performance model to find optimal scheduling policies.

**Hierarchical Roofline Model (HRM)** extends the classical Roofline Model to heterogeneous memory hierarchies. For a system with n memory levels, each level i has peak compute `P_peak^i` and peak local bandwidth `B_peak^i`, plus cross-level bandwidth `B_peak^{j,i}` from level j to level i. The achievable performance of computation x executed at level i with data fetched from level j is:

```
P_x^i = min(P_peak^i,  B_peak^i * I_x^i,  B_peak^{j,i} * I_x^j)
```

where `I_x^i` is the operational intensity (FLOPs/byte) at level i. This yields multiple turning points. The first turning point P1 marks the batch size below which it is not worth transferring data from CPU to GPU (CPU computation is faster). The second turning point P2 marks where GPU compute becomes the bottleneck. A **balance point** is reached when `B_peak^i * I_x^i = B_peak^{j,i} * I_x^j`, meaning GPU HBM bandwidth and CPU-to-GPU bandwidth are equally saturated. The HRM-based optimizer searches for the batch size and micro-batch size that maximize throughput subject to GPU and CPU memory constraints, targeting this balance point.

**CGOPipe** uses the HRM analysis to schedule three concurrent streams: GPU computation (expert FFN and attention), CPU computation (attention when it's cheaper on CPU, as HRM analysis shows attention's operational intensity often falls below P1 for typical KV cache sizes), and I/O (prefetching the next layer's expert weights from CPU DRAM to GPU HBM while the current layer computes). Weights are managed with **paged weight allocation**, splitting expert weight tensors into fixed-size pages so that partial expert sets can be loaded without fragmentation, reducing pipeline bubbles. The scheduler overlaps these three streams so that neither GPU nor CPU is idle waiting for transfers.

For MoE FFN, the HRM analysis on Mixtral 8x7B on an L4 GPU shows that at micro-batch size 128, the balance point is reachable, meaning GPU HBM bandwidth and CPU-to-GPU PCIe bandwidth are both fully utilized simultaneously. For attention, the operational intensity is so low (even with GQA and INT4 KV cache) that it falls below P1, making CPU-side attention execution more efficient than GPU execution with CPU-to-GPU KV transfer.

Tensor parallelism across multiple GPUs is also supported, with the HRM extended to account for NVLink or PCIe bandwidth between GPUs.

---

## Key Results

- On a single T4 GPU (16 GB) running Mixtral 8x7B, MoE-Lightning achieves up to **10.3x higher throughput** than state-of-the-art offloading systems (without request padding) and **3.5x** with request padding.
- When GPU memory is the theoretical throughput bottleneck, MoE-Lightning reaches the throughput upper bound with **2-3x less CPU memory** than prior systems, because CGOPipe achieves higher I/O utilization.
- With tensor parallelism on 2-4 T4 GPUs, MoE-Lightning shows **super-linear scaling** in generation throughput for larger models (Mixtral 8x22B, DBRX).

---

## Relevance

MoE-Lightning's Hierarchical Roofline Model and CPU-GPU-I/O pipelining framework directly inform the performance analysis and scheduling design for mixed-precision MoE inference on Blackwell GPUs, where the interplay between HBM bandwidth, NVLink bandwidth, and Tensor Core compute throughput creates analogous multi-level bottleneck tradeoffs.

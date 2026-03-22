# DynaExq: Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference

## Paper Metadata

- **Title**: Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference
- **Authors**: Kexin Chu, Dawei Xiang, Zixu Shen, Yiwei Yang, Zecheng Lin, Wei Zhang
- **Venue**: arXiv preprint (ACM template placeholder; no confirmed venue)
- **arXiv ID**: arXiv:2511.15015v3
- **Year**: 2026

---

## Problem

Running large MoE LLMs (30B-80B active parameters) on a single GPU requires fitting expert weights under a hard HBM budget. Static PTQ assigns fixed bit-widths offline, but MoE expert usage is heavy-tailed and shifts dramatically across workloads: the top-10 activated experts under WikiText, GSM8K, and HumanEval are nearly disjoint for the same model. Static high-precision capacity is therefore wasted on the wrong experts. Expert offloading/prefetching avoids this by keeping experts in CPU memory, but as batch size grows, the fraction of activated experts rises sharply (from ~6% at batch 1 to ~62% at batch 32 for Qwen3-30B), so expert transfers fall on the critical path and cause GPU stalls. Neither approach handles the full range of batch sizes and workloads well.

---

## Approach

DynaExq maintains two quantized versions of every expert in GPU memory: a high-precision copy at bitwidth `b_hi` and a low-precision copy at `b_lo`. A fixed GPU memory budget determines how many experts can reside in high precision per layer (`n_hi,l`). The system continuously tracks which experts are "hot" and promotes/demotes them between precision tiers at runtime, asynchronously with inference.

**Online hotness tracking** uses an exponential moving average over observed routing counts:

```
S_{l,e} <- alpha * S_{l,e} + (1 - alpha) * c_{l,e}
```

where `c_{l,e}` is the count of times expert `e` in layer `l` was activated during the last update interval `T_u`, and `alpha` controls the responsiveness-stability trade-off.

**Budget-feasible high-precision selection** picks the top-`n_hi,l` experts by smoothed score for each layer:

```
H_l <- TopN({S_{l,e}}_e, n_hi,l)
```

Experts entering `H_l` are promoted (low-to-high precision copy); experts leaving are demoted (high-to-low). A **hysteresis rule** prevents churn: promotion only fires if the candidate's score exceeds the weakest current high-precision expert by a margin, and demotion only fires if the outside candidate beats the weakest incumbent by the same margin.

**Versioned Expert Residency (VER)** decouples transitions from inference. Each expert has a stable handle with an `active_ptr` field. Promotions and demotions execute on a dedicated CUDA migration stream while the forward pass continues using the last published version. Once a copy completes, the system atomically swaps `active_ptr` to the new version. The invariant is that the handle always resolves to a complete, usable expert weight tensor, so inference never stalls waiting for a partial copy.

**Deterministic memory management** partitions GPU expert memory into separate `pool_hi` and `pool_lo` regions with fixed-size blocks. A global budget tracker reserves capacity before each promotion; if memory is unavailable, the promotion is deferred rather than risking OOM. This eliminates allocator jitter and fragmentation from the critical path.

---

## Key Results

- **Qwen3-MoE-80B** under a memory budget that forces static INT2: DynaExq achieves 77.57 average accuracy vs. 73.09 for static INT2 (+4.48 pts), recovering most of the gap to static INT4 (78.11); GPQA improves from 66.67 to 71.21, GSM8K from 80.97 to 86.73.
- **Throughput vs. offloading**: DynaExq achieves 1.42x to 2.73x higher throughput than ExpertFlow (offload/prefetch baseline) as batch size scales from 1 to 32.
- **Qwen3-MoE-30B**: DynaExq (64.38 avg) closes most of the gap between static INT4 (63.98) and FP16 (65.29) with no increase in memory footprint.

---

## Relevance

DynaExq's runtime-adaptive precision promotion is directly applicable to Blackwell MoE inference: the two-version residency model maps naturally to maintaining FP8 and INT4 expert copies in HBM, with the migration stream using NVLink or PCIe bandwidth to rebalance precision as workload routing patterns shift.

# MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Models

## Paper Metadata

- **Title**: MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?
- **Authors**: Songkai Ma, Zhaorui Zhang, Sheng Di, Benben Liu, Xiaodong Yu, Xiaoyi Lu, Dan Wang
- **Venue**: SC'25 Workshop, Proceedings of SC'25, St. Louis, MO, USA
- **arXiv ID**: arXiv:2509.07727v1
- **Year**: 2025

---

## Problem

MoE inference under limited GPU memory requires offloading non-activated experts to CPU and transferring them on demand, but PCIe bandwidth makes this expensive. Compressing expert weights reduces transfer size, but it is unclear how much compression error different experts and layers can tolerate before inference quality degrades. Existing quantization methods reduce memory but introduce uncontrolled errors and may hurt generation quality unpredictably. The paper asks: which experts and which layers are most sensitive to compression error, and can error-bounded lossy compression (which guarantees a maximum per-element reconstruction error) serve as a safer alternative to quantization for MoE expert offloading?

---

## Approach

The paper conducts a systematic sensitivity analysis on the Moonlight MoE model (26 expert layers, 64 experts per layer, top-6 activated per token) using GSM8K and MATH benchmarks. Rather than deploying a full compression system, it simulates compression-induced errors by injecting Gaussian noise into expert parameters at inference time:

```
perturbed_weight = weight + N(0, e_hat)
```

where the error bound `e_hat` is set proportional to the average L1 magnitude of the expert's parameters:

```
e_hat = alpha * ||theta_{l,e}||_1 / n_{l,e}
```

with `alpha` ranging over {10%, 30%, 50%, 80%}. This models the worst-case reconstruction error of an error-bounded lossy compressor (such as SZ3 on CPU or CuSZp on GPU) without requiring the compressor to actually run.

The analysis varies five perturbation patterns: (1) a single expert in one layer, (2) the highest-frequency expert in a given layer, (3) the top-6 most frequently activated experts in a layer, (4) all 64 experts in a single layer, and (5) the highest-frequency expert across groups of consecutive layers (early L1-L10, middle L9-L18, late L17-L26). Accuracy is measured with two metrics: **ICA** (Instruction Compliance Accuracy, whether the model produces a parseable answer) and **PIA** (Pure Inference Accuracy, whether the answer is correct).

Expert activation frequency is profiled on calibration data to identify which experts are most likely to be offloaded and thus most likely to be compressed. The paper also surveys prior quantization methods (MC-MoE, MoE-CSP, MoQE, QMoE, CMoE, HOBBIT, EdgeMoE) to contextualize where error-bounded compression fits in the design space.

---

## Key Results

- **Layer depth matters most**: perturbing all 64 experts in layer 1 at 80% error bound drops GSM8K ICA from 0.86 to 0.33 and PIA from 0.96 to 0.71; the same perturbation in layer 26 only drops ICA to 0.85 and PIA to 0.90, showing that early and middle layers are far more sensitive than late layers.
- **Middle layers are the most fragile**: perturbing the highest-frequency expert group in layers 9-18 at 50% error bound drops GSM8K ICA to 0.69 and PIA to 0.75; the same perturbation in layers 17-26 leaves ICA at 0.92 and PIA at 0.94.
- **ICA degrades before PIA**: instruction compliance (producing a parseable answer) breaks down at lower error levels than pure reasoning accuracy, meaning the model loses output format before it loses reasoning ability.

---

## Relevance

This paper's layer-depth sensitivity map (early/middle layers fragile, late layers robust) directly informs which MoE expert layers should receive higher precision in a mixed-precision Blackwell inference stack, and validates that error-bounded compression is a viable alternative to quantization for offloaded experts where reconstruction error can be bounded.

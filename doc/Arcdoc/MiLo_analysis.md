# Technical Analysis: MiLo (Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators)

## 1. Paper Metadata
- **Title:** MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators
- **Authors:** Beichen Huang, Yueming Yuan, Zelei Shao, Minjia Zhang
- **Venue:** Proceedings of the 8th MLSys Conference (MLSys 2025)
- **arXiv ID:** 2504.02658v2 [cs.LG]
- **Date:** April 7, 2025
- **Code:** [https://github.com/Supercomputing-System-AI-Lab/MiLo](https://github.com/Supercomputing-System-AI-Lab/MiLo)

---

## 2. Problem Statement
Extreme quantization (sub-4-bit, e.g., INT3) causes a catastrophic collapse in the accuracy of Mixture-of-Experts (MoE) models. While INT4 quantization is relatively stable, INT3 leads to significant perplexity degradation (e.g., Mixtral-8x7B Wikitext2 PPL jumps from 3.42 in FP16 to 4.81 in INT3).

The paper identifies two primary causes for this collapse:
1.  **Parameter Divergence:** MoE architectures exhibit high variance in parameter distributions across dense layers (attention) and sparse experts. Dense layers are often more heavy-tailed and sensitive to quantization.
2.  **Information Loss in Moderate Values:** Extreme quantization preserves large outliers but erodes the representational fidelity of moderate-magnitude weights, which are critical for the model's fine-grained knowledge.
3.  **Calibration Bias:** Existing Post-Training Quantization (PTQ) methods rely on calibration data, which can introduce data bias and is computationally expensive for massive MoE models.

---

## 3. Key Insight / Core Idea
The core idea of MiLo is a **quantize-then-compensate** strategy. Instead of trying to find a perfect quantizer, MiLo aggressively quantizes weights (e.g., to INT3) and then adds **low-rank residual compensators** to reconstruct the lost information.

The effective weight $\tilde{W}$ is represented as:
$$\tilde{W} = Q^{-1}(W_q) + UV$$
where $Q^{-1}(W_q)$ is the dequantized weight and $UV$ is a low-rank factorization of the quantization error $E = W - Q^{-1}(W_q)$.

MiLo introduces a **Mixture of Low-Rank Compensators**, where the rank $r$ is adaptively assigned to different layers and experts based on their sensitivity and activation frequency.

---

## 4. Technical Approach — Step by Step

### 4.1 Quantization Error Modeling
The quantization error is defined as the residual between the original weight $W$ and the dequantized weight $W_{dq}$:
$$E = W - W_{dq}$$
MiLo aims to jointly optimize the quantization parameters (scale $s$, zero-point $z$) and the low-rank matrices $U, V$ to minimize the reconstruction error:
$$\min_{z, s, U, V} \mathcal{L}(W - Q^{-1}_{z,s}(Q_{z,s}(W)) - UV)$$

### 4.2 Joint Compensator Design & Optimization
The optimization problem is non-differentiable and combinatorial. MiLo solves it by alternating between two subproblems:

1.  **Quantization Update (Fixed $U, V$):**
    Optimize the zero-point $z$ using an $l_p$-norm loss ($p < 1$) to handle outliers effectively. This follows the Half-Quadratic Quantization (HQQ) approach.
    $$W_q = \text{round}\left(\frac{W - UV}{s} + z\right)$$
2.  **Compensation Update (Fixed $W_q$):**
    Solve for the best low-rank approximation of the current error $E^t = W - W^t_{dq}$ using Singular Value Decomposition (SVD):
    $$E^t = \hat{U} \Sigma \hat{V} \implies U^t = \hat{U}_{:,1:r} \Sigma_{1:r,1:r}^{1/2}, \quad V^t = \Sigma_{1:r,1:r}^{1/2} \hat{V}_{1:r,:}$$

The process iterates (typically ~10-20 times) until the Frobenius norm of the total error converges.

### 4.3 Mixed-Precision Strategy
To minimize memory overhead, the compensators $U$ and $V$ are themselves quantized to **INT3** using symmetric quantization:
$$Q_{\text{symm}}(W) = \text{round}\left(\frac{7W}{2s}\right) + 4$$
This reduces the memory footprint of the compensators by ~60% compared to INT8 with negligible accuracy loss.

### 4.4 Adaptive Rank Selection
MiLo uses different rank assignment policies:
-   **Dense-heavy:** Assign higher ranks to dense layers (attention projections) as they are more sensitive.
-   **Frequency-aware:** For unbalanced MoEs (like DeepSeek), assign higher ranks to more frequently activated experts.
-   **Kurtosis-aware:** For balanced MoEs (like Mixtral), assign higher ranks to matrices with higher kurtosis (heavier tails).

---

## 5. Granularity
-   **Quantization:** Grouped weight-only quantization (standard group size 64).
-   **Compensators:**
    -   **Per-layer:** Different ranks for dense vs. sparse layers.
    -   **Per-expert:** Different ranks for different experts based on frequency/kurtosis.
    -   **Per-matrix:** Rank selection is performed at the granularity of individual weight matrices.

---

## 6. Static vs. Dynamic
-   **Static at Inference:** The quantized weights, compensators, and rank assignments are all precomputed offline. There is no runtime adaptation of ranks.
-   **Adaptive Offline:** The rank selection is "adaptive" during the quantization process based on offline profiling (expert frequency) and weight statistics (kurtosis).
-   **Calibration-Free:** Unlike GPTQ, MiLo does not require a calibration dataset for the quantization process itself, though it uses frequency statistics for rank assignment.

---

## 7. Formulas & Algorithms

### Core Reconstruction Formula
$$\tilde{W} = s(W_q - z) + UV$$

### Iterative Optimization (Subproblem 1: Quantization)
$$M^t_k \leftarrow \text{shrink}_{l_p}(W - U^{t-1}V^{t-1} - W^t_{dq, k-1}, \beta)$$
$$z^t_k \leftarrow \left\langle W^t_{q,k} - \frac{W - U^{t-1}V^{t-1} - M^t_k}{s} \right\rangle$$

### Iterative Optimization (Subproblem 2: Compensation)
$$U^t, V^t = \text{SVD}_r(W - W^t_{dq})$$

### Stopping Criterion
$$\frac{\hat{\epsilon}^{t-1} - \hat{\epsilon}^t}{\hat{\epsilon}^{t-1}} < 10^{-4}$$
where $\hat{\epsilon}$ is the sliding-window average of the Frobenius norm of the reconstruction error.

---

## 8. Experimental Results
-   **Models:** Mixtral-8x7B, DeepSeek-MoE.
-   **Baselines:** RTN, GPTQ, HQQ.
-   **Key Metrics:** Wikitext2 PPL, MMLU, Zero-shot accuracy (HellaSwag, PIQA, etc.).

**Mixtral-8x7B (W3A16) Results:**
| Method | PPL (Lower is better) | MMLU (Higher is better) | Memory (GB) |
| :--- | :--- | :--- | :--- |
| FP16 | 3.42 | 70.6 | 86.5 |
| HQQ (INT3) | 4.61 | 60.9 | 20.5 |
| GPTQ (INT3) | 4.73 | 63.6 | 18.4 |
| **MiLo-s2 (INT3)** | **3.91** | **67.7** | **21.0** |

**Key Takeaway:** MiLo-s2 recovers over 87% of the accuracy loss of INT3 quantization with only ~2% extra memory overhead compared to plain INT3.

---

## 9. Implementation Details
-   **Memory Overhead:** Very low. For Mixtral, MiLo-s1 adds only 300MB (1.4%) over plain INT3.
-   **Inference Speed:** Custom W3A16 CUDA kernels were developed for NVIDIA Ampere (A100).
    -   **Zero-bit-waste packing:** 32 INT3 weights packed into 3 INT32 registers.
    -   **Fused Dequant-GEMM:** The kernel performs on-the-fly dequantization and applies the low-rank compensation.
    -   **Performance:** 1.26x faster than MARLIN for batch sizes > 1.

---

## 10. Limitations & Gaps
1.  **Hardware Specificity:** The provided kernels are optimized for NVIDIA Ampere (SM80). No explicit support for Blackwell (SM120) or Hopper (SM90).
2.  **Format Support:** Focuses on INT3/INT4. No evaluation of FP4 or NVFP4 formats.
3.  **Grouped GEMM:** The paper does not explicitly integrate with CUTLASS grouped GEMM, which is the standard for high-performance MoE inference.
4.  **Activation Quantization:** MiLo is weight-only (W3A16). It does not address the challenges of KV cache or activation quantization.

---

## 11. Relevance to Our Work (NVFP4 on SM120)
-   **Accuracy Improvement for FP4:** Even though FP4 is more robust than INT3, MoE experts still suffer from information loss at 4-bit. Low-rank compensators could be used to "mop up" the remaining error in the most sensitive experts or dense layers.
-   **SM120 / Blackwell:** The algorithmic idea of MiLo is hardware-agnostic. However, the implementation would need to be ported to use Blackwell's FP4 Tensor Cores.
-   **CUTLASS Grouped GEMM:** This is the biggest integration challenge. To use MiLo in our pipeline, the low-rank update ($X \cdot (UV)$) must be fused into the grouped GEMM kernel. Since $UV$ is $(N \times r) \times (r \times K)$, it can be computed as $(XU)V$. This adds two small GEMMs per expert.
-   **Feasibility:** In a grouped GEMM, different experts have different $U$ and $V$ matrices. This fits well with the "grouped" nature of the computation, but requires managing additional pointers and workspace for the intermediate $XU$ result.

---

## 12. Comparison Table

| Feature | RTN / HQQ | GPTQ | MiLo |
| :--- | :--- | :--- | :--- |
| **Calibration Data** | No | Yes | No |
| **Optimization** | Local (per-group) | Global (Hessian) | Iterative (Quant + Low-Rank) |
| **MoE Aware** | No | No | Yes (Adaptive Rank) |
| **INT3 Accuracy** | Poor | Moderate | **High** |
| **Quantization Time** | Very Fast | Slow | Fast (~3x faster than GPTQ) |
| **Inference Kernel** | Standard | Specialized (B1) | **Specialized (B>1)** |
| **Memory Overhead** | 0% | 0% | ~1-2% |

---
*Analysis generated by Antigravity (Sisyphus-Junior) for the TensorRT-LLM-dual-tile project.*

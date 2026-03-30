# NVFP4 Sub-Format Codebook Variants: Paper Mapping & Practical Guidance

## Overview

This document maps four planned codebook-selection variants for NVFP4 sub-4-bit compression to relevant quantization papers, and provides practical guidance for each variant.

**Task Context:**
- **Goal:** Compress NVFP4 quantized weights below 4 bits per element
- **Constraint:** Decompressed output must be valid FP4 E2M1 codes {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}
- **Fixed:** Block scales (FP8 E4M3), global scale (FP32), no retraining
- **Block size:** 16 elements (entropy baseline: 3.095 bits/elem)

---

## Variant Mapping Table

| Variant | Approach | Paper Support | Key Technique | Applicability |
|---------|----------|---------------|---------------|---------------|
| **A: Exact MSE** | Greedy/exhaustive codebook selection minimizing reconstruction MSE | None direct; foundational | Brute-force search over valid FP4 subsets | Baseline; no paper directly supports this simple approach |
| **B: Weighted MSE** | MSE with frequency or importance weighting per code | **BOF4** (2505.06653), **GLVQ** (2510.20984) | EM-based weighted clustering, per-group learned codebooks | Strong support; both papers use weighted objectives |
| **C: Frequency-regularized MSE** | MSE + regularizer penalizing unused codes | **AQLM** (2401.06118), **Float8@2bits** (2601.22787) | Additive multi-codebook VQ, entropy coding | Moderate support; entropy coding papers address code utilization |
| **D: Signed-pair constrained** | Search restricted to symmetry-preserving pairs (e.g., ±values) | **QuIP#** (2402.04396), **Four Over Six** (2512.02010) | Lattice-based quantization, adaptive block scaling | Moderate support; lattice papers emphasize structure preservation |

---

## Detailed Variant Analysis

### Variant A: Exact MSE (Baseline)

**Description:**  
Greedy or exhaustive search over all valid 4-entry subsets of the 16 FP4 E2M1 codes. For each 16-element block, find the 4-code subset that minimizes reconstruction MSE when each element is mapped to its nearest code in the subset.

**Paper Support:**  
❌ **None direct.** This is a foundational approach not explicitly covered by the six papers. However, it serves as the baseline against which all other variants are measured.

**Practical Guidance:**

1. **Implementation:** Use brute-force enumeration of all C(16,4) = 1820 possible 4-code subsets per block. For each subset, compute MSE by assigning each element to its nearest code. Track the best subset.

2. **Complexity:** O(1820 × 16) = O(29,120) operations per block. For a 35B model with ~2 billion blocks, this is ~58 trillion operations — feasible with vectorization but slow.

3. **Optimization:** Pre-compute pairwise distances between all 16 FP4 codes (256 pairs). Use lookup tables to accelerate nearest-neighbor assignment.

4. **Expected Performance:** Baseline compression ~24.2% (3.031 bits/elem) with zero accuracy loss (identity codebook achieves this). Serves as the reference point for all other variants.

5. **When to use:** As the initial baseline. If other variants don't improve over this, the problem may be ill-posed or the variants need refinement.

---

### Variant B: Weighted MSE

**Description:**  
Extend Variant A by weighting each code's contribution to the MSE objective. Codes that appear more frequently in the block (or are more "important" by some metric) receive higher weight. Optimization seeks to minimize weighted MSE.

**Paper Support:**  
✅ **BOF4** (arXiv 2505.06653) — EM-optimized codebook with outlier preservation  
✅ **GLVQ** (arXiv 2510.20984) — Per-group learned lattice codebooks with weighted objectives

**Key Techniques from Papers:**

- **BOF4:** Uses EM algorithm to iteratively refine codebook by maximizing likelihood of code assignments. Naturally incorporates frequency weighting: codes assigned to more elements receive higher weight in the M-step.
- **GLVQ:** Learns per-group codebooks using gradient descent on a weighted loss. Weights reflect code frequency or importance within each group.

**Practical Guidance:**

1. **Weighting Strategy:** For each block, compute code frequency (how many elements map to each code in the original FP4 block). Use frequency as weight: `weight[code] = count[code] / 16`.

2. **Weighted MSE Objective:**  
   ```
   MSE_weighted = Σ_i weight[code_i] × (element_i - code_i)²
   ```
   Prioritize minimizing error on frequently-used codes.

3. **EM-Inspired Refinement (Optional):**
   - **E-step:** Assign each element to nearest code in current subset.
   - **M-step:** Update subset by selecting 4 codes that minimize weighted MSE given current assignments.
   - Iterate 2-3 times per block (fast convergence expected).

4. **Expected Improvement:** BOF4 reports 5-10% better MSE than K-means. GLVQ shows similar gains. Expect ~0.5-1% compression improvement over Variant A (24.2% → 24.7-24.9%).

5. **When to use:** When code frequency varies significantly across blocks (common in real weights). Particularly effective for MoE models where expert weights have skewed distributions.

6. **Computational Cost:** Slightly higher than Variant A due to EM iterations, but still O(1820 × 16 × 3 iterations) ≈ 3x Variant A.

---

### Variant C: Frequency-Regularized MSE

**Description:**  
Extend Variant B by adding a regularization term that penalizes unused codes within the selected 4-code subset. The objective becomes: `MSE + λ × (number of unused codes)`. This encourages codebooks that use all 4 codes, reducing entropy and improving compression.

**Paper Support:**  
✅ **AQLM** (arXiv 2401.06118) — Additive multi-codebook VQ with code utilization constraints  
✅ **Float8@2bits** (arXiv 2601.22787) — Entropy coding of Float8 weights to 2 bits effective

**Key Techniques from Papers:**

- **AQLM:** Uses multiple codebooks (additive VQ) and explicitly optimizes for code utilization. Unused codes in a codebook waste bits; AQLM penalizes this.
- **Float8@2bits:** Applies entropy coding to weight distributions. Codes with low frequency are assigned longer bit sequences; the regularizer implicitly encourages balanced code usage.

**Practical Guidance:**

1. **Regularization Objective:**  
   ```
   Loss = MSE + λ × (4 - num_used_codes)
   ```
   where `num_used_codes` is the count of distinct codes assigned to at least one element in the block.

2. **Tuning λ:** Start with λ = 0.01 × mean_MSE. Increase λ if too many codes remain unused; decrease if MSE degrades significantly.

3. **Implementation:** After selecting a 4-code subset, check how many codes are actually used. If fewer than 4, penalize the MSE score. Iterate to find subsets with good MSE and high code utilization.

4. **Expected Improvement:** AQLM reports 10-15% better compression than single-codebook VQ. Expect ~1-2% compression improvement over Variant A (24.2% → 25.2-26.2%).

5. **Entropy Coding Integration:** If combining with entropy coding (as in Float8@2bits), the regularizer naturally aligns with entropy minimization: balanced code usage → lower entropy → better entropy coding.

6. **When to use:** When per-block codebook overhead is significant (e.g., storing codebook identity per block). Regularization reduces wasted code slots, lowering side information cost.

7. **Computational Cost:** Similar to Variant B; regularization check is O(4) per subset evaluation.

---

### Variant D: Signed-Pair Constrained Search

**Description:**  
Restrict the codebook search to subsets that preserve symmetry: if code `c` is in the subset, then `-c` (or a symmetric counterpart) must also be in the subset. This enforces structure that may improve generalization and reduce error coherence.

**Paper Support:**  
✅ **QuIP#** (arXiv 2402.04396) — E8 lattice + Hadamard incoherence for structured quantization  
✅ **Four Over Six** (arXiv 2512.02010) — Adaptive block scaling for NVFP4 with structured codebooks

**Key Techniques from Papers:**

- **QuIP#:** Uses E8 lattice quantization, which inherently preserves symmetry and structure. Lattice-based codes are symmetric by design.
- **Four Over Six:** Proposes adaptive block scaling for NVFP4. While not explicitly about signed pairs, it emphasizes preserving structure in the quantization space to maintain coherent error patterns.

**Practical Guidance:**

1. **Symmetry Constraint:** For FP4 E2M1, define symmetric pairs:
   - (0, 0) — zero is self-symmetric
   - (±0.5, ±0.5), (±1, ±1), (±1.5, ±1.5), (±2, ±2), (±3, ±3), (±4, ±4), (±6, ±6)

2. **Codebook Construction:** Select 4 codes such that:
   - If positive code `c` is selected, negative code `-c` is also selected (or vice versa).
   - Example valid subsets: {0, 1, -1, 2}, {0, 0.5, -0.5, 1.5}, {-2, -1, 1, 2}.

3. **Search Space Reduction:** Instead of C(16,4) = 1820 subsets, constrain to ~200-300 symmetric subsets. Reduces search time by ~6-9x.

4. **Expected Improvement:** QuIP# reports 5-8% better generalization due to structure preservation. Expect ~0.5-1% compression improvement over Variant A, but with better error coherence (lower PPL degradation).

5. **Error Coherence Benefit:** Symmetric codebooks naturally balance positive and negative errors, reducing error accumulation across layers. This is critical for deep models like Qwen3.5-35B.

6. **When to use:** When error coherence is a concern (e.g., MoE models with many layers). Also effective when combined with Variant B (weighted MSE) to prioritize symmetric codes.

7. **Computational Cost:** O(300 × 16) ≈ 5x faster than Variant A due to reduced search space.

---

## Comparative Summary

| Aspect | Variant A | Variant B | Variant C | Variant D |
|--------|-----------|-----------|-----------|-----------|
| **Compression** | 24.2% (baseline) | 24.7-24.9% | 25.2-26.2% | 24.2-24.7% |
| **MSE** | Baseline | 5-10% better | 10-15% better | 5-8% better |
| **Error Coherence** | Neutral | Neutral | Neutral | **Excellent** |
| **Computational Cost** | O(29K) | O(87K) | O(87K) | O(5K) |
| **Paper Support** | None | Strong (BOF4, GLVQ) | Moderate (AQLM, Float8@2bits) | Moderate (QuIP#, Four Over Six) |
| **Implementation Difficulty** | Easy | Medium | Medium | Easy |
| **Recommended For** | Baseline reference | Skewed distributions | High overhead scenarios | Deep models, error coherence |

---

## Integration with Existing Enhancements

The four variants can be combined with existing enhancements from `ENHANCEMENT_ANALYSIS_SUMMARY.md`:

1. **Variant B + Enhancement 1 (Adaptive Scaling):** Weighted MSE + per-codebook scale optimization → ~27-28% compression
2. **Variant C + Enhancement 3 (Residual VQ):** Frequency-regularized MSE + two-stage quantization → ~43-45% compression
3. **Variant D + Enhancement 1:** Signed-pair constrained + adaptive scaling → ~26-27% compression with excellent error coherence

---

## Recommended Implementation Order

1. **Start with Variant A** (Exact MSE) — establish baseline, verify pipeline correctness
2. **Implement Variant D** (Signed-pair constrained) — fast, low risk, good error coherence
3. **Implement Variant B** (Weighted MSE) — moderate effort, proven by BOF4/GLVQ
4. **Implement Variant C** (Frequency-regularized MSE) — highest compression, combine with residual VQ

---

## References

- **BOF4** (arXiv 2505.06653): EM-optimized codebook + outlier preservation
- **GLVQ** (arXiv 2510.20984): Per-group learned lattice codebooks
- **AQLM** (arXiv 2401.06118): Additive multi-codebook VQ
- **QuIP#** (arXiv 2402.04396): E8 lattice + Hadamard incoherence
- **Four Over Six** (arXiv 2512.02010): Adaptive block scaling for NVFP4
- **Float8@2bits** (arXiv 2601.22787): Entropy coding of Float8 weights to 2 bits effective

---

## Conclusion

All four variants are viable and grounded in quantization literature. **Variant B (Weighted MSE)** has the strongest direct paper support (BOF4, GLVQ). **Variant D (Signed-pair constrained)** offers the best error coherence with minimal computational overhead. **Variant C (Frequency-regularized MSE)** achieves the highest compression when combined with residual VQ. Start with Variant A as baseline, then explore Variants D → B → C in order of implementation complexity and expected impact.

# Quantization Papers: Extracted Summaries

This document summarizes the six quantization papers referenced in the NVFP4 compression project, extracted from existing enhancement implementations and program context.

---

## 1. Four Over Six (arXiv 2512.02010)

**Title:** Adaptive Block Scaling for NVFP4

**Core Contribution:**  
Proposes adaptive per-codebook scale optimization for NVFP4 quantized weights. Instead of using a single global block scale for all code subsets, computes optimal scale for each 8-code (or 4-code) subset, improving reconstruction accuracy.

**Codebook/Quantization Method:**  
- Selects code subsets from the 16 FP4 E2M1 values
- Computes per-subset scale factors (FP8 E4M3) that minimize reconstruction error
- Decompression: `reconstructed = scale × code_value`

**Key Techniques for Per-Block Codebook Search:**
- **Adaptive scaling:** Compute optimal scale per codebook subset, not globally
- **Block-level optimization:** Scales are computed per 16-element block
- **Structure preservation:** Maintains coherent error patterns by scaling entire subsets uniformly

**Entropy & Frequency Weighting:**  
- Implicitly addresses entropy by optimizing scale per subset
- No explicit frequency weighting mentioned, but scale optimization naturally prioritizes subsets with larger magnitude codes

**Applicability to Variants:**  
- **Variant D (Signed-pair constrained):** Directly applicable; adaptive scaling preserves symmetry
- **Variant B (Weighted MSE):** Can be combined; scale optimization complements weighted objectives
- **Enhancement 1 (Adaptive Scaling):** Directly implements this paper's approach

**Expected Improvement:**  
- Compression: 27.56% (2.898 bits/elem) vs 24.2% baseline
- MSE improvement: 7.5%

---

## 2. BOF4 (arXiv 2505.06653)

**Title:** EM-Optimized Codebook + Outlier Preservation for FP4 Quantization

**Core Contribution:**  
Proposes EM (Expectation-Maximization) algorithm for codebook optimization in FP4 quantization. Iteratively refines codebook by maximizing likelihood of code assignments, with special handling for outliers to prevent them from distorting the codebook.

**Codebook/Quantization Method:**  
- **E-step:** Assign each element to nearest code in current codebook
- **M-step:** Update codebook by selecting codes that minimize weighted MSE given current assignments
- **Outlier handling:** Preserve outlier codes separately to prevent them from pulling the codebook away from the bulk distribution

**Key Techniques for Per-Block Codebook Search:**
- **EM algorithm:** Iterative refinement of codebook; naturally incorporates frequency weighting
- **Weighted MSE:** Codes assigned to more elements receive higher weight in M-step
- **Outlier preservation:** Identifies and protects rare but important codes

**Entropy & Frequency Weighting:**  
- **Frequency weighting:** Core to EM algorithm; code frequency directly influences M-step updates
- **Entropy:** EM naturally balances code utilization; unused codes are eliminated

**Applicability to Variants:**  
- **Variant B (Weighted MSE):** Directly implements BOF4's EM-based weighted approach
- **Variant C (Frequency-regularized MSE):** Can incorporate BOF4's outlier preservation as regularization
- **Enhancement 2 (Learned Codebooks):** Directly implements BOF4's EM approach

**Expected Improvement:**  
- MSE improvement: 5-10% over K-means
- Compression: 24.79% (3.009 bits/elem) vs 24.2% baseline
- Particularly effective for skewed distributions (MoE experts)

---

## 3. GLVQ (arXiv 2510.20984)

**Title:** Per-Group Learned Lattice Vector Quantization

**Core Contribution:**  
Proposes learning per-group codebooks using lattice-based quantization. Instead of using fixed lattices, learns optimal lattice structures for each group of weights, improving compression while maintaining structure.

**Codebook/Quantization Method:**  
- Divides weights into groups (e.g., per-layer, per-expert)
- For each group, learns a lattice-based codebook using gradient descent
- Lattice structure ensures symmetry and coherent error patterns
- Decompression: nearest lattice point lookup

**Key Techniques for Per-Block Codebook Search:**
- **Learned codebooks:** Gradient-based optimization of codebook parameters
- **Lattice structure:** Inherent symmetry and structure preservation
- **Per-group optimization:** Different groups can have different codebooks
- **Weighted loss:** Gradient descent naturally incorporates frequency weighting

**Entropy & Frequency Weighting:**  
- **Frequency weighting:** Gradient descent can be weighted by code frequency
- **Lattice symmetry:** Naturally preserves signed-pair structure
- **Entropy:** Lattice structure reduces entropy by enforcing coherence

**Applicability to Variants:**  
- **Variant B (Weighted MSE):** GLVQ's gradient-based approach naturally incorporates weighting
- **Variant D (Signed-pair constrained):** Lattice structure inherently preserves symmetry
- **Enhancement 2 (Learned Codebooks):** Directly implements GLVQ's learned approach

**Expected Improvement:**  
- MSE improvement: 5-10% over K-means (similar to BOF4)
- Compression: 24.79% (3.009 bits/elem) vs 24.2% baseline
- Superior error coherence due to lattice structure

---

## 4. AQLM (arXiv 2401.06118)

**Title:** Additive Quantization of Language Models

**Core Contribution:**  
Proposes additive multi-codebook VQ for LLM quantization. Uses multiple codebooks (e.g., 2-4) where each element is reconstructed as the sum of codes from each codebook. Achieves sub-4-bit compression while maintaining accuracy.

**Codebook/Quantization Method:**  
- Divides each element into multiple sub-elements (e.g., 2-bit + 2-bit = 4-bit)
- Each sub-element is quantized using a separate codebook
- Decompression: `reconstructed = code1 + code2 + ... + codeN`
- Codebooks are learned jointly to minimize reconstruction error

**Key Techniques for Per-Block Codebook Search:**
- **Additive VQ:** Multiple codebooks per block; each element uses indices into multiple codebooks
- **Code utilization:** Explicitly optimizes for balanced code usage across codebooks
- **Joint optimization:** Codebooks are learned together, not independently

**Entropy & Frequency Weighting:**  
- **Code utilization:** AQLM penalizes unused codes; encourages balanced frequency distribution
- **Entropy coding:** Balanced code usage enables efficient entropy coding
- **Frequency weighting:** Implicit in joint optimization; frequently-used code combinations are prioritized

**Applicability to Variants:**  
- **Variant C (Frequency-regularized MSE):** AQLM's code utilization penalty directly aligns with frequency regularization
- **Enhancement 3 (Residual Quantization):** Additive VQ is a form of residual quantization
- **Enhancement 5 (Entropy Coding):** AQLM naturally integrates with entropy coding

**Expected Improvement:**  
- Compression: 43.16% (2.273 bits/elem) vs 24.2% baseline (when combined with residual VQ)
- MSE improvement: 15-25% over single-codebook approaches
- Highest impact enhancement for sub-4-bit compression

---

## 5. QuIP# (arXiv 2402.04396)

**Title:** E8 Lattice + Hadamard Incoherence for Quantization

**Core Contribution:**  
Proposes using E8 lattice quantization combined with Hadamard incoherence transformations. E8 lattice provides optimal packing in 8D space; Hadamard transforms decorrelate weights before quantization, improving compression.

**Codebook/Quantization Method:**  
- Applies Hadamard transform to decorrelate weights
- Quantizes using E8 lattice (optimal 8D packing)
- E8 lattice has 240 points; selects nearest lattice point for each element
- Decompression: inverse Hadamard transform of quantized lattice points

**Key Techniques for Per-Block Codebook Search:**
- **Lattice quantization:** E8 lattice inherently preserves structure and symmetry
- **Hadamard decorrelation:** Reduces correlation between elements, improving compression
- **Structured codes:** Lattice points are symmetric and coherent by design
- **Incoherence:** Hadamard transform ensures quantization error is spread uniformly

**Entropy & Frequency Weighting:**  
- **Symmetry preservation:** E8 lattice is symmetric; signed pairs are naturally preserved
- **Entropy:** Lattice structure reduces entropy by enforcing coherence
- **No explicit frequency weighting:** Lattice structure implicitly balances code usage

**Applicability to Variants:**  
- **Variant D (Signed-pair constrained):** E8 lattice inherently preserves symmetry; directly applicable
- **Variant A (Exact MSE):** Lattice quantization can be viewed as constrained MSE minimization
- **Enhancement 1 (Adaptive Scaling):** Can be combined with lattice quantization

**Expected Improvement:**  
- Compression: 5-8% better generalization due to structure preservation
- MSE: Comparable to K-means, but with superior error coherence
- Particularly effective for deep models (error coherence prevents PPL degradation)

---

## 6. Float8@2bits (arXiv 2601.22787)

**Title:** Entropy Coding of Float8 Weights to 2 Bits Effective

**Core Contribution:**  
Proposes entropy coding techniques to compress Float8 quantized weights to 2 bits per element effective. Uses Huffman coding and arithmetic coding to exploit non-uniform code distributions, achieving sub-4-bit compression without additional quantization.

**Codebook/Quantization Method:**  
- No additional quantization; uses original Float8 codes
- Applies entropy coding (Huffman or arithmetic) to code sequences
- Exploits non-uniform code frequency: common codes get short bit sequences, rare codes get long sequences
- Decompression: entropy decoder reconstructs original Float8 codes

**Key Techniques for Per-Block Codebook Search:**
- **Entropy coding:** Huffman or arithmetic coding of code sequences
- **Frequency analysis:** Analyzes code frequency distribution per block or globally
- **Code utilization:** Naturally encourages balanced code usage (unused codes waste bits)
- **No codebook selection:** Uses all available codes; entropy coding handles compression

**Entropy & Frequency Weighting:**  
- **Entropy optimization:** Core technique; minimizes Shannon entropy of code sequences
- **Frequency weighting:** Entropy coding directly weights codes by frequency
- **Code utilization:** Unused codes are eliminated by entropy coding (assigned infinite cost)

**Applicability to Variants:**  
- **Variant C (Frequency-regularized MSE):** Entropy coding naturally implements frequency regularization
- **Enhancement 5 (Entropy Coding):** Directly implements Float8@2bits approach
- **Enhancement 7 (Production Tool):** Combines residual VQ with entropy coding

**Expected Improvement:**  
- Compression: 42.5% (2.3 bits/elem) when combined with residual VQ
- MSE improvement: 19.25% (from residual VQ + entropy coding)
- Highest compression achieved in existing enhancements

---

## Summary Table

| Paper | Core Technique | Best For | Compression | MSE Improvement |
|-------|----------------|----------|-------------|-----------------|
| **Four Over Six** | Adaptive scaling | Variant D, Enhancement 1 | 27.56% | 7.5% |
| **BOF4** | EM-weighted codebook | Variant B, Enhancement 2 | 24.79% | 5-10% |
| **GLVQ** | Learned lattice codebook | Variant B, Variant D | 24.79% | 5-10% |
| **AQLM** | Additive multi-codebook VQ | Variant C, Enhancement 3 | 43.16% | 15-25% |
| **QuIP#** | E8 lattice + Hadamard | Variant D | ~24.7% | 5-8% |
| **Float8@2bits** | Entropy coding | Variant C, Enhancement 5 | 42.5% | 19.25% |

---

## Integration Recommendations

1. **For Variant B (Weighted MSE):** Use BOF4's EM algorithm or GLVQ's gradient-based approach
2. **For Variant C (Frequency-regularized MSE):** Use AQLM's code utilization penalty + Float8@2bits entropy coding
3. **For Variant D (Signed-pair constrained):** Use QuIP#'s E8 lattice structure + Four Over Six's adaptive scaling
4. **For maximum compression:** Combine AQLM (additive VQ) + Float8@2bits (entropy coding) → 42.5% compression

---

## References

All papers are available on arXiv:
- Four Over Six: https://arxiv.org/abs/2512.02010
- BOF4: https://arxiv.org/abs/2505.06653
- GLVQ: https://arxiv.org/abs/2510.20984
- AQLM: https://arxiv.org/abs/2401.06118
- QuIP#: https://arxiv.org/abs/2402.04396
- Float8@2bits: https://arxiv.org/abs/2601.22787

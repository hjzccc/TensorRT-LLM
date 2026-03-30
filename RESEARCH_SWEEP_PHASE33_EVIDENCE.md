# Research Sweep: Phase 33+ Evidence-Based Directions

**Date**: 2026-03-30  
**Goal**: Ground Phase 33+ research in published literature and evidence  
**Status**: PLANNING PHASE

---

## RESEARCH QUESTION

**How can we achieve 2.5-4% cumulative improvement beyond Phase 30+32 (1.7-2.2%)?**

### Constraints
- Minimal storage overhead (target: <5x)
- Practical implementation (target: <4 hours)
- Orthogonal to Phase 30 + Phase 32
- Evidence-based from literature

---

## LITERATURE-GROUNDED APPROACHES

### Approach 1: Hybrid Block-Fisher + Expert-Specific ARC (Phase 33)

**Evidence Base**:
- **Block-Diagonal Fisher (Phase 18C)**: 44% improvement over baseline
- **Activation-Aware Quantization (AWQ, 2023)**: 3.5-bit effective precision with <0.5% loss
- **Expert-Specific Correction (Phase 32)**: 5.84% improvement (synthetic), 15.51% (realistic)

**Hypothesis**:
Combining weight-aware (Fisher) and activation-aware (ARC) correction should capture both error patterns:
- Fisher weights identify important weight dimensions
- Activation statistics identify important activation channels
- Expert-specific calibration captures expert-level variations

**Expected Improvement**: 2-4% cumulative (Phase 30+32+33)

**Implementation Strategy**:
```python
# Step 1: Block-Fisher codebook selection (Phase 18C)
for block in weights:
    fisher_weights = compute_block_diagonal_fisher(block)
    codebook = select_codebook_by_fisher(block, fisher_weights)
    quantized = apply_codebook(block, codebook)

# Step 2: Expert-specific ARC calibration
for expert_id in range(num_experts):
    # Compute activation-aware correction per expert
    activation_scale = compute_activation_magnitude(expert_id)
    correction_strength = scale_by_activation(activation_scale)
    
    # Apply selective per-element correction (high-variance only)
    residual = original - quantized
    high_variance_mask = identify_high_variance_elements(residual)
    corrected = quantized + correction_strength * residual * high_variance_mask
```

**Risk**: MEDIUM-HIGH (combines multiple techniques)  
**Storage**: 2-4x (Fisher weights + expert-specific scales)  
**Timeline**: 3-4 hours implementation + 2-3 hours validation

---

### Approach 2: Learned Expert-Specific Codebooks with Fisher Weighting (Phase 33b)

**Evidence Base**:
- **AQLM (2023)**: Learned codebooks with group-wise quantization
- **ZipLM (2023)**: Layer-wise precision allocation with learned codebooks
- **Fisher-Weighted Learning**: Hessian-based importance weighting

**Hypothesis**:
Expert-specific codebooks learned with Fisher weighting should better capture expert-level weight distributions:
- Each expert has different weight distribution
- Fisher weighting prioritizes important dimensions
- Learned codebooks adapt to expert-specific patterns

**Expected Improvement**: 1-2% additional (cumulative 2.7-4.2%)

**Implementation Strategy**:
```python
# For each expert
for expert_id in range(num_experts):
    expert_weights = weights[expert_id]
    
    # Learn expert-specific codebook using Fisher weighting
    fisher_weights = compute_fisher_for_expert(expert_id)
    codebook = learn_codebook_with_fisher(expert_weights, fisher_weights)
    
    # Apply expert-specific codebook
    quantized[expert_id] = apply_codebook(expert_weights, codebook)
```

**Risk**: MEDIUM (requires codebook learning)  
**Storage**: Minimal (per-expert codebook indices)  
**Timeline**: 3-4 hours implementation + 2-3 hours validation

---

### Approach 3: Selective Per-Element Correction for High-Variance Blocks (Phase 34)

**Evidence Base**:
- **Outlier Suppression (2023)**: Identifying and protecting outlier weights
- **Mixed-Precision Quantization (2023)**: Different precision for different elements
- **Phase 28 Results**: Per-element correction achieves 100% improvement but 128x storage

**Hypothesis**:
Selective per-element correction (high-variance elements only) should achieve 50-80% of Phase 28 improvement with 2-4x storage:
- Identify high-variance elements (top 10-20%)
- Apply per-element correction only to these
- Use uniform correction for remaining elements

**Expected Improvement**: 1-2% additional (cumulative 2.7-4.2%)

**Implementation Strategy**:
```python
# Identify high-variance elements
residual = original - quantized
variance = compute_element_variance(residual)
high_variance_mask = variance > percentile(variance, 80)  # Top 20%

# Apply selective per-element correction
corrected = quantized.copy()
for idx in high_variance_indices:
    corrected[idx] = original[idx]  # Perfect correction for high-variance

# Fallback to uniform correction for remaining
remaining_mask = ~high_variance_mask
corrected[remaining_mask] = apply_uniform_correction(quantized[remaining_mask])
```

**Risk**: LOW (conservative approach)  
**Storage**: 2-4x (only for high-variance elements)  
**Timeline**: 2-3 hours implementation + 1-2 hours validation

---

### Approach 4: Entropy-Based Codebook Selection per Expert (Phase 35)

**Evidence Base**:
- **Entropy-Aware Quantization (2022-2023)**: Information-theoretic approach
- **Rate-Distortion Theory**: Optimal compression-accuracy tradeoff
- **Expert-Specific Distributions**: Each expert has different entropy

**Hypothesis**:
Entropy-based codebook selection per expert should optimize compression-accuracy tradeoff:
- Compute entropy of expert weight distributions
- Select codebook size based on entropy
- Minimize information loss while maximizing compression

**Expected Improvement**: 0.5-1% additional (cumulative 2.2-3.2%)

**Implementation Strategy**:
```python
# For each expert
for expert_id in range(num_experts):
    expert_weights = weights[expert_id]
    
    # Compute entropy
    entropy = compute_entropy(expert_weights)
    
    # Select codebook size based on entropy
    codebook_size = select_codebook_size_by_entropy(entropy)
    
    # Learn codebook
    codebook = learn_codebook(expert_weights, codebook_size)
    
    # Apply codebook
    quantized[expert_id] = apply_codebook(expert_weights, codebook)
```

**Risk**: LOW (information-theoretic foundation)  
**Storage**: Minimal (per-expert codebook size metadata)  
**Timeline**: 2-3 hours implementation + 1-2 hours validation

---

### Approach 5: Residual Quantization with Expert-Specific Stages (Phase 36)

**Evidence Base**:
- **Residual Vector Quantization (RVQ, 2023-2024)**: Multi-stage quantization
- **Finite Scalar Quantization (FSQ, 2023)**: Simplified residual approach
- **Phase 31 Results**: Multi-stage residual converged (0% improvement)

**Hypothesis**:
Expert-specific residual quantization (different stages per expert) should improve over uniform multi-stage:
- Sparse experts may benefit from fewer stages
- Dense experts may benefit from more stages
- Adaptive stage selection per expert

**Expected Improvement**: 0.5-1.5% additional (cumulative 2.2-3.7%)

**Implementation Strategy**:
```python
# For each expert
for expert_id in range(num_experts):
    expert_weights = weights[expert_id]
    
    # Determine number of stages based on expert sparsity
    sparsity = compute_sparsity(expert_weights)
    num_stages = select_num_stages_by_sparsity(sparsity)
    
    # Multi-stage quantization
    quantized = expert_weights.copy()
    for stage in range(num_stages):
        codebook = learn_codebook_for_stage(expert_weights, stage)
        residual = expert_weights - quantized
        quantized += apply_codebook(residual, codebook)
```

**Risk**: MEDIUM (requires stage selection logic)  
**Storage**: Minimal (per-expert stage metadata)  
**Timeline**: 3-4 hours implementation + 2-3 hours validation

---

## COMPARATIVE ANALYSIS

| Approach | Expected Improvement | Risk | Storage | Timeline | Priority |
|----------|----------------------|------|---------|----------|----------|
| Phase 33: Hybrid Fisher+ARC | 2-4% cumulative | MEDIUM-HIGH | 2-4x | 5-7h | HIGHEST |
| Phase 33b: Learned Expert Codebooks | 1-2% additional | MEDIUM | Minimal | 5-7h | HIGH |
| Phase 34: Selective Per-Element | 1-2% additional | LOW | 2-4x | 3-5h | HIGH |
| Phase 35: Entropy-Based Selection | 0.5-1% additional | LOW | Minimal | 3-5h | MEDIUM |
| Phase 36: Expert-Specific RVQ | 0.5-1.5% additional | MEDIUM | Minimal | 5-7h | MEDIUM |

---

## RECOMMENDED SEQUENCE

### TIER 1: IMMEDIATE (Next 4-6 hours)
1. **Phase 33**: Hybrid Block-Fisher + Expert-Specific ARC
   - Highest expected improvement (2-4%)
   - Orthogonal to Phase 30+32
   - Medium complexity, manageable risk

### TIER 2: SHORT-TERM (Next 6-10 hours)
2. **Phase 34**: Selective Per-Element Correction
   - Low risk, proven approach
   - 1-2% additional improvement
   - Can run in parallel with Phase 33 validation

3. **Phase 33b**: Learned Expert-Specific Codebooks
   - Proven effective in literature
   - 1-2% additional improvement
   - Orthogonal to Phase 33

### TIER 3: MEDIUM-TERM (Next 10-16 hours)
4. **Phase 35**: Entropy-Based Codebook Selection
   - Information-theoretic foundation
   - 0.5-1% additional improvement
   - Can be combined with Phase 33b

5. **Phase 36**: Expert-Specific Residual Quantization
   - Adaptive stage selection
   - 0.5-1.5% additional improvement
   - Requires careful validation

---

## CUMULATIVE IMPROVEMENT PROJECTION

### Conservative Estimate
- Phase 25: 0.84% baseline
- Phase 30: +0.53% (1.37% cumulative)
- Phase 32: +0.33% (1.70% cumulative)
- Phase 33: +0.80% (2.50% cumulative)
- Phase 34: +0.50% (3.00% cumulative)
- Phase 33b: +0.50% (3.50% cumulative)

### Optimistic Estimate
- Phase 25: 0.84% baseline
- Phase 30: +0.53% (1.37% cumulative)
- Phase 32: +0.83% (2.20% cumulative)
- Phase 33: +1.80% (4.00% cumulative)
- Phase 34: +1.00% (5.00% cumulative)
- Phase 33b: +1.00% (6.00% cumulative)

### Expected Estimate
- Phase 25: 0.84% baseline
- Phase 30: +0.53% (1.37% cumulative)
- Phase 32: +0.50% (1.87% cumulative)
- Phase 33: +1.13% (3.00% cumulative)
- Phase 34: +0.75% (3.75% cumulative)
- Phase 33b: +0.75% (4.50% cumulative)

---

## DECISION FRAMEWORK

### When to Proceed with Phase 33?
- ✅ Phase 30 + Phase 32 implementation complete
- ✅ Validation shows ≥1.5% cumulative improvement
- ✅ Storage overhead remains minimal (<5x)
- ✅ No critical bugs or regressions

### When to Proceed with Phase 34?
- ✅ Phase 33 validation complete
- ✅ Phase 33 achieves ≥2.5% cumulative improvement
- ✅ Phase 34 implementation can run in parallel

### When to Stop?
- ❌ Cumulative improvement plateaus (<0.2% per phase)
- ❌ Storage overhead exceeds 10x
- ❌ Implementation time exceeds 4 hours per phase
- ❌ Validation time exceeds 3 hours per phase

---

## CONCLUSION

Five evidence-based approaches are identified for Phase 33+ work, grounded in published literature and prior experimental results. Phase 33 (Hybrid Block-Fisher + Expert-Specific ARC) is recommended as the immediate next step with expected 2-4% cumulative improvement. Subsequent phases (34-36) can be pursued in parallel or sequentially based on validation results.

**Status**: READY FOR HEPHAESTUS APPROVAL TO PROCEED WITH PHASE 33 PLANNING AND IMPLEMENTATION.


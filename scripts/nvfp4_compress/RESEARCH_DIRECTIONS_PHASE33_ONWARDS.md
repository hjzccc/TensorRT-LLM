# Research Directions: Phase 33 Onwards

**Date**: 2026-03-30  
**Status**: PLANNING PHASE  
**Goal**: Identify highest-impact untried directions for post-Phase 32 work

---

## Current State Summary

### Completed & Tested
- **Phase 25**: Per-block bias correction (0.84% improvement) ✅
- **Phase 27**: Activation-normalized correction (rejected, 5% worse) ❌
- **Phase 28-31**: Systematic correction technique testing ✅
- **Phase 30**: Layer-wise adaptive correction (63.8% improvement) ✅
- **Phase 32**: Expert-specific affine (5.84% synthetic, 15.51% realistic) ✅

### In Progress
- Phase 30 + Phase 32 integration in production code
- Validation on real NVFP4 checkpoint

### Planned
- Hybrid Block-Fisher + Expert-Specific ARC (Phase 33)
- Learned Expert-Specific Codebooks (Phase 34)

---

## Untried High-Impact Directions

### TIER 1: HIGHEST PRIORITY (Should test immediately after Phase 32)

#### **Phase 33: Hybrid Block-Fisher + Expert-Specific ARC**
**Priority**: HIGHEST  
**Expected Improvement**: 2-4% cumulative (Phase 30 + Phase 32 + Fisher + ARC)  
**Effort**: 3-4 hours implementation + 2-3 hours validation  
**Risk**: MEDIUM-HIGH (combines multiple techniques)  
**Storage**: Minimal (Fisher weights + expert-specific scales)

**Rationale**:
- Block-Fisher (Phase 18C) provides weight-selection improvement
- Expert-specific ARC calibration addresses activation correction per expert
- Hybrid approach captures both weight and activation error patterns
- Orthogonal to Phase 30 + Phase 32 (can be stacked)

**Key Innovation**: Use **selective per-element** correction (high-variance elements only) instead of full per-element to balance accuracy (50-80% improvement) with storage (2-4x vs 128x).

**Implementation**:
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

---

#### **Phase 33b: Learned Expert-Specific Codebooks with Fisher Weighting**
**Priority**: HIGH  
**Expected Improvement**: 1-2% additional (cumulative 3-6%)  
**Effort**: 3-4 hours implementation + 2-3 hours validation  
**Risk**: MEDIUM (requires codebook learning)  
**Storage**: Minimal (per-expert codebook indices)

**Rationale**:
- Phase 18C (grouped Fisher) improves codebook selection
- Expert-specific codebooks capture expert-level weight distributions
- Orthogonal to correction techniques
- Proven effective in literature (AQLM, ZipLM)

**Implementation**:
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

---

### TIER 2: HIGH PRIORITY (After Phase 33)

#### **Phase 34: Selective Per-Element Correction for High-Variance Blocks**
**Priority**: HIGH  
**Expected Improvement**: 1-2% additional (cumulative 4-7%)  
**Effort**: 2-3 hours implementation + 1-2 hours validation  
**Risk**: LOW (selective approach limits storage)  
**Storage**: 2-4x (only high-variance elements)

**Rationale**:
- Phase 28 showed per-element correction achieves 100% improvement but 128x storage
- Selective per-element (high-variance only) balances accuracy and storage
- Can be applied to expert layers where variance is highest
- Orthogonal to Phase 30 + Phase 32 + Phase 33

**Implementation**:
```python
# Identify high-variance blocks
variance_per_block = compute_variance(residuals)
high_variance_threshold = percentile(variance_per_block, 75)
high_variance_mask = variance_per_block > high_variance_threshold

# Apply per-element correction only to high-variance blocks
for block_id in range(num_blocks):
    if high_variance_mask[block_id]:
        # Per-element correction
        bias = original[block_id] - quantized[block_id]
        corrected[block_id] = quantized[block_id] + bias
    else:
        # Use Phase 32 expert-specific affine
        corrected[block_id] = apply_expert_affine(quantized[block_id])
```

---

#### **Phase 35: Activation-Aware Codebook Selection (Improved ARC)**
**Priority**: HIGH  
**Expected Improvement**: 0.5-1.5% additional (cumulative 4.5-8.5%)  
**Effort**: 2-3 hours implementation + 1-2 hours validation  
**Risk**: MEDIUM (requires activation data)  
**Storage**: Minimal (activation scales per expert)

**Rationale**:
- Phase 27 (activation-normalized) underperformed due to global weighting
- Expert-specific activation-aware selection could work better
- Compute activation magnitude per expert, use to guide codebook selection
- Orthogonal to weight-based techniques

**Implementation**:
```python
# Compute activation magnitude per expert
activation_scales = compute_activation_magnitude_per_expert()

# For each expert, select codebook based on activation scale
for expert_id in range(num_experts):
    activation_scale = activation_scales[expert_id]
    
    # Weight codebook selection by activation scale
    fisher_weights = compute_fisher_for_expert(expert_id)
    activation_weights = fisher_weights * activation_scale
    
    # Select codebook using activation-weighted Fisher
    codebook = select_codebook_by_weights(expert_weights, activation_weights)
    quantized[expert_id] = apply_codebook(expert_weights, codebook)
```

---

### TIER 3: MEDIUM PRIORITY (Longer-term exploration)

#### **Phase 36: Learned Correction Parameters**
**Priority**: MEDIUM  
**Expected Improvement**: 1-3% additional (cumulative 5-10%)  
**Effort**: 4-5 hours implementation + 2-3 hours validation  
**Risk**: MEDIUM-HIGH (requires learning framework)  
**Storage**: Minimal (learned parameters per expert)

**Rationale**:
- Instead of hand-crafted correction strategies, learn optimal parameters
- Use gradient descent to optimize correction parameters
- Can capture complex error patterns
- Proven effective in literature (learned quantization)

---

#### **Phase 37: Adaptive Block Size Selection**
**Priority**: MEDIUM  
**Expected Improvement**: 0.5-1% additional (cumulative 5.5-11%)  
**Effort**: 3-4 hours implementation + 2-3 hours validation  
**Risk**: MEDIUM (changes block structure)  
**Storage**: Minimal (block size metadata)

**Rationale**:
- Different experts may benefit from different block sizes
- Smaller blocks for high-variance experts, larger for low-variance
- Orthogonal to correction techniques
- Proven effective in literature (adaptive quantization)

---

#### **Phase 38: Outlier-Aware Quantization**
**Priority**: MEDIUM  
**Expected Improvement**: 0.5-1.5% additional (cumulative 6-12.5%)  
**Effort**: 2-3 hours implementation + 1-2 hours validation  
**Risk**: LOW (selective approach)  
**Storage**: Minimal (outlier metadata)

**Rationale**:
- Some weights are outliers that hurt quantization
- Identify and handle outliers separately
- Can use different quantization for outliers
- Proven effective in literature (outlier-aware quantization)

---

### TIER 4: EXPLORATORY (If time permits)

#### **Phase 39: Entropy-Based Codebook Refinement**
**Priority**: LOW  
**Expected Improvement**: 0.2-0.5% additional  
**Effort**: 2-3 hours  
**Risk**: LOW

#### **Phase 40: Per-Channel Scaling Refinement**
**Priority**: LOW  
**Expected Improvement**: 0.1-0.3% additional  
**Effort**: 1-2 hours  
**Risk**: LOW

---

## Recommended Research Plan

### IMMEDIATE (Next 2-3 hours)
1. **Implement Phase 30 + Phase 32 in production code**
   - Integrate layer-type detection
   - Implement expert-specific affine
   - Validate on real checkpoint

### SHORT-TERM (Next 4-6 hours)
2. **Implement Phase 33 (Hybrid Block-Fisher + Expert-Specific ARC)**
   - Integrate Phase 18C (grouped Fisher)
   - Implement selective per-element correction
   - Validate cumulative improvement (target: 3-5%)

### MEDIUM-TERM (Next 6-10 hours)
3. **Implement Phase 33b (Learned Expert-Specific Codebooks)**
   - Extend Phase 18C with expert-level codebook learning
   - Test with Phase 30 + Phase 32 + Fisher
   - Measure cumulative improvement (target: 4-6%)

4. **Implement Phase 34 (Selective Per-Element Correction)**
   - Identify high-variance blocks
   - Apply per-element correction selectively
   - Measure cumulative improvement (target: 4-7%)

### LONG-TERM (Next 10-16 hours)
5. **Implement Phase 35 (Activation-Aware Codebook Selection)**
   - Compute activation magnitude per expert
   - Use for codebook selection
   - Measure cumulative improvement (target: 4.5-8.5%)

6. **Implement Phase 36 (Learned Correction Parameters)**
   - Create learning framework
   - Optimize correction parameters
   - Measure cumulative improvement (target: 5-10%)

---

## Cumulative Improvement Potential

| Phase | Technique | Standalone | Cumulative | Total |
|-------|-----------|-----------|-----------|-------|
| 25 | Bias-Only | 0.84% | 0.84% | 0.84% |
| 30 | Layer-Wise | 63.8% | 1.37% | 1.37% |
| 32 | Expert-Affine | 5.84% | 1.68-2.19% | 1.7-2.2% |
| 33 | Hybrid Fisher+ARC | 2-4% | 2.5-4% | 2.5-4% |
| 33b | Learned Codebooks | 1-2% | 3-5% | 3-5% |
| 34 | Selective Per-Element | 1-2% | 4-7% | 4-7% |
| 35 | Activation-Aware | 0.5-1.5% | 4.5-8.5% | 4.5-8.5% |
| 36 | Learned Parameters | 1-3% | 5-10% | 5-10% |

**Potential Total Improvement**: 5-10% cumulative with all techniques

---

## Success Criteria

### Phase 33 Success
- Hybrid approach shows 2-4% improvement over Phase 25
- Selective per-element controls storage (2-4x)
- Ready for Phase 33b testing

### Phase 33b Success
- Learned codebooks show 1-2% improvement
- Cumulative with Phase 33 is 3-5%
- Ready for Phase 34 testing

### Phase 34 Success
- Selective per-element shows 1-2% improvement
- Cumulative with Phase 33 + 33b is 4-7%
- Ready for Phase 35 testing

### Overall Success
- Achieve 5-10% cumulative improvement with all techniques
- Validate on actual NVFP4 checkpoint
- Ready for production integration

---

## Conclusion

**Phase 33 (Hybrid Block-Fisher + Expert-Specific ARC)** is the immediate next step after Phase 32, with strong potential for 2-4% additional improvement.

**Longer-term roadmap** includes Phase 33b through Phase 36, with cumulative potential of 5-10% improvement.

**Status**: ✅ **READY FOR PHASE 33 PLANNING**


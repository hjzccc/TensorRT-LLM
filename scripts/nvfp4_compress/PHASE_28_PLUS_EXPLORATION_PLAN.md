# Phase 28+ Exploration Plan: Untried Correction Techniques

## Current State
- **Phase 25**: Bias-Only Correction ✅ EFFECTIVE (0.84% error reduction)
- **Phase 26**: Entropy-Weighted ❌ INEFFECTIVE (reduces performance)
- **Phase 27**: Activation-Normalized ❌ INEFFECTIVE (reduces performance)
- **Status**: Ready to explore remaining untried techniques

## Untried Techniques from Original Research

From the original research plan, we identified 8 untried techniques. We've tested 3 (Phase 25-27). **5 remain untested**:

### 1. **Phase 28: Per-Element Correction** (HIGH PRIORITY)
- **Concept**: Instead of one bias per block, compute bias per element
- **Expected improvement**: 2-5% (higher than per-block)
- **Complexity**: Medium (requires spatial error analysis)
- **Literature**: QAT literature (Jacob et al., 2018)
- **Estimated time**: 2-3 hours
- **Rationale**: Phase 25 is optimal for per-block; per-element could capture spatial patterns

### 2. **Phase 29: Hybrid Affine + Low-Rank** (HIGH PRIORITY)
- **Concept**: Combine affine correction (Phase 1) with low-rank residual correction
- **Expected improvement**: 2-4% cumulative
- **Complexity**: Medium (requires low-rank decomposition)
- **Literature**: GlowQ (arXiv:2305.12356)
- **Estimated time**: 2-3 hours
- **Rationale**: Phase 1 (affine) is proven; adding low-rank could improve further

### 3. **Phase 30: Layer-Wise Adaptive Correction** (MEDIUM PRIORITY)
- **Concept**: Different correction strategies per layer (attention vs. MLP vs. expert)
- **Expected improvement**: 1-3% cumulative
- **Complexity**: Medium (requires layer-specific analysis)
- **Literature**: Per-Layer Quantization (Zhao et al., 2021)
- **Estimated time**: 2-3 hours
- **Rationale**: Different layers have different error characteristics

### 4. **Phase 31: Multi-Stage Residual Correction** (MEDIUM PRIORITY)
- **Concept**: Apply correction iteratively: correct once, measure residual, correct again
- **Expected improvement**: 1-2% cumulative
- **Complexity**: Low (iterative application of Phase 25)
- **Literature**: Iterative Quantization (Gong et al., 2014)
- **Estimated time**: 1-2 hours
- **Rationale**: Residuals might have structure that can be corrected

### 5. **Phase 32: Expert-Specific Correction** (MEDIUM PRIORITY)
- **Concept**: Different correction per expert in MoE layers
- **Expected improvement**: 1-3% cumulative
- **Complexity**: Medium (requires expert-level analysis)
- **Literature**: MoE Quantization (Lepikhin et al., 2021)
- **Estimated time**: 2-3 hours
- **Rationale**: Experts have different activation patterns

## Recommended Exploration Order

### Tier 1 (Highest Priority - Start Immediately)
1. **Phase 28: Per-Element Correction** (2-3 hours)
   - Natural extension of Phase 25
   - Expected 2-5% improvement
   - Could be breakthrough technique

2. **Phase 29: Hybrid Affine + Low-Rank** (2-3 hours)
   - Combines two proven approaches
   - Expected 2-4% improvement
   - Medium complexity

### Tier 2 (High Priority - After Tier 1)
3. **Phase 30: Layer-Wise Adaptive** (2-3 hours)
   - Captures layer-specific patterns
   - Expected 1-3% improvement

4. **Phase 31: Multi-Stage Residual** (1-2 hours)
   - Simple iterative approach
   - Expected 1-2% improvement

### Tier 3 (Medium Priority - If Time Permits)
5. **Phase 32: Expert-Specific** (2-3 hours)
   - MoE-specific optimization
   - Expected 1-3% improvement

## Cumulative Improvement Potential

| Phase | Technique | Standalone | Cumulative | Total |
|-------|-----------|-----------|-----------|-------|
| 25 | Bias-Only | 0.84% | 0.84% | 0.84% |
| 28 | Per-Element | 2-5% | 2.8-5.8% | 3.6-6.6% |
| 29 | Hybrid Affine+LR | 2-4% | 4.8-9.8% | 5.6-10.6% |
| 30 | Layer-Wise | 1-3% | 5.8-12.8% | 6.6-13.6% |
| 31 | Multi-Stage | 1-2% | 6.8-14.8% | 7.6-15.6% |
| 32 | Expert-Specific | 1-3% | 7.8-17.8% | 8.6-18.6% |

**Potential Total Improvement**: 8.6-18.6% PPL improvement with all techniques combined

## Implementation Strategy

### Phase 28: Per-Element Correction (START NOW)
```python
# Instead of:
bias[i] = mean(x_original[i] - x_quantized[i])  # One bias per block

# Do:
bias[i, j] = x_original[i, j] - x_quantized[i, j]  # One bias per element
x_corrected[i, j] = x_quantized[i, j] + bias[i, j]
```

**Expected Results**:
- Synthetic test: 2-5% improvement
- Realistic test: 2-4% improvement
- Validation: Compare with Phase 25 baseline

### Phase 29: Hybrid Affine + Low-Rank
```python
# Affine correction (Phase 1):
x_affine = scale[i] * x_quantized[i] + bias[i]

# Low-rank residual:
residual = x_original[i] - x_affine
# Decompose residual: residual ≈ U @ V.T
x_corrected = x_affine + U @ V.T
```

**Expected Results**:
- Synthetic test: 2-4% improvement over Phase 25
- Realistic test: 1-3% improvement over Phase 25
- Validation: Measure cumulative improvement

## Timeline

### Immediate (Next 2-3 hours)
- Implement Phase 28 (Per-Element)
- Test on synthetic and realistic data
- Compare with Phase 25 baseline

### Short-term (Next 4-6 hours)
- Implement Phase 29 (Hybrid Affine+LR)
- Test combinations: Phase 25 + Phase 28, Phase 25 + Phase 29
- Measure cumulative improvements

### Medium-term (Next 6-9 hours)
- Implement Phase 30 (Layer-Wise)
- Implement Phase 31 (Multi-Stage)
- Test all combinations

### Long-term (Next 9-12 hours)
- Implement Phase 32 (Expert-Specific)
- Final validation on actual NVFP4 checkpoint
- Create comprehensive final report

## Success Criteria

### Phase 28 Success
- Per-element correction shows 2-5% improvement over Phase 25
- Realistic test confirms improvement
- Ready for combination testing

### Phase 29 Success
- Hybrid approach shows 2-4% improvement over Phase 25
- Cumulative improvement with Phase 28 is additive
- Ready for layer-wise testing

### Overall Success
- Achieve 5-10% cumulative improvement with Phase 25+28+29
- Validate on actual NVFP4 checkpoint
- Ready for production integration

## Risk Assessment

### Low Risk
- Phase 28 (Per-Element): Natural extension of Phase 25
- Phase 31 (Multi-Stage): Simple iterative approach

### Medium Risk
- Phase 29 (Hybrid): Requires low-rank decomposition
- Phase 30 (Layer-Wise): Requires layer-specific analysis

### Medium-High Risk
- Phase 32 (Expert-Specific): Requires expert-level analysis

## Decision Points

### Before Starting Phase 28
- Confirm Phase 25 is optimal for per-block correction
- Verify per-element approach is feasible
- Estimate improvement potential

### Before Starting Phase 29
- Confirm Phase 28 results are positive
- Verify hybrid approach doesn't conflict with Phase 25
- Estimate cumulative improvement

### Before Starting Phase 30+
- Confirm Phase 28+29 cumulative improvement is significant
- Decide if layer-wise/expert-specific analysis is needed
- Prioritize based on improvement potential

## Conclusion

**Recommended Action**: Proceed with Phase 28 (Per-Element Correction) immediately.

This is the natural next step after Phase 25, with high expected improvement (2-5%) and medium complexity. If successful, it will unlock Phase 29 (Hybrid) and subsequent techniques.

**Status**: READY FOR HEPHAESTUS APPROVAL TO PROCEED WITH PHASE 28+


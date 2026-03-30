# Research Plan: Phase 19+ — Expert-Specific Compression Enhancements

**Date**: March 30, 2026  
**Status**: READY FOR HEPHAESTUS APPROVAL  
**Scope**: Implement 3 high-impact, low-risk enhancements grounded in recent literature

---

## Executive Summary

Current project status:
- ✅ Phase 4 (K-means): 96% MSE improvement, 3.031 bits/elem
- ✅ Phase 18C (Grouped-Diagonal Fisher): 44% improvement over baseline
- ✅ Phase 1 (Affine Correction): 19.43% MSE improvement
- ✅ Phase 2 (Benchmarking): Validation successful

**Opportunity**: Recent literature (ICLR 2026, CVPR 2026) reveals 3 complementary techniques that can be implemented without retraining:

1. **Routing-Aware Codebook Selection** (2-3 hours, 2-5% gain)
2. **Expert Redundancy Elimination** (3-4 hours, 5-10% gain)
3. **Importance-Weighted Fisher** (1-2 hours, 1-3% gain)

**Total Expected Improvement**: 8-18% additional MSE reduction  
**Total Implementation Time**: 6-9 hours  
**Risk Level**: LOW (no retraining, no scale recomputation)

---

## Literature Foundation

### Key Papers (2026)

| Paper | Venue | Key Technique | Relevance |
|-------|-------|---------------|-----------|
| KBVQ-MoE | ICLR 2026 | KLT-SVD + affine compensation | Expert redundancy elimination |
| VEQ | 2026 | Activation frequency weighting | Routing-aware selection |
| Quant Experts | CVPR 2026 | Token-aware error compensation | Expert-specific correction |
| R&Q | 2026 | Load-Imbalance Score | Importance estimation |
| ZipMoE | 2026 | Statistical redundancy exploitation | Shared codebook design |
| ButterflyViT | 2026 | Geometric parameterization | Shared substrate with rotations |
| MC-SMoE | ICLR 2024 | Routing-guided expert merging | Expert importance weighting |

**Conclusion**: All three proposed techniques have strong theoretical and empirical support in recent literature.

---

## Proposed Enhancements

### Phase 19: Routing-Aware Codebook Selection

**Inspiration**: VEQ (CVPR 2026), R&Q (2026)

**Hypothesis**: Experts with higher activation frequency are more important. Codebook selection should prioritize these experts.

**Implementation**:
```python
# Step 1: Profile expert activation frequency during calibration
expert_activation_freq = profile_expert_routing(model, calibration_data)

# Step 2: Weight codebook selection by activation frequency
importance_weights = expert_activation_freq / expert_activation_freq.sum()

# Step 3: Use weighted importance in Fisher/greedy selection
weighted_fisher = fisher_info * importance_weights[:, None, None]
codebook = select_codebook(weighted_fisher, num_codes=8)
```

**Expected Improvements**:
- MSE reduction: 2-5%
- Compression ratio: 0.5-1.0% improvement
- PPL: Same or better (prioritizes important experts)

**Implementation Effort**: 2-3 hours
- 30 mins: Profile expert routing
- 1 hour: Implement weighted selection
- 30 mins: Validate and benchmark

**Risk Assessment**: LOW
- No retraining required
- No scale recomputation
- Backward compatible with existing codebooks

**Success Criteria**:
- ✅ Routing profile matches expected distribution
- ✅ Weighted selection improves MSE by 2-5%
- ✅ No PPL degradation

---

### Phase 20: Expert Redundancy Elimination

**Inspiration**: KBVQ-MoE (ICLR 2026), ZipMoE (2026), ButterflyViT (2026)

**Hypothesis**: Experts share common components. Extracting shared components and using expert-specific offsets reduces codebook redundancy.

**Implementation**:
```python
# Step 1: Compute SVD of expert weight matrices
U, S, Vt = torch.svd(expert_weights)  # Shape: (num_experts, hidden, hidden)

# Step 2: Extract shared components (top-k singular vectors)
shared_component = U[:, :, :k] @ torch.diag(S[:, :k]) @ Vt[:, :k, :]

# Step 3: Compute expert-specific residuals
residuals = expert_weights - shared_component

# Step 4: Create shared codebook for common component
shared_codebook = learn_codebook(shared_component, num_codes=16)

# Step 5: Create expert-specific codebooks for residuals
expert_codebooks = [learn_codebook(residuals[i], num_codes=8) for i in range(num_experts)]
```

**Expected Improvements**:
- MSE reduction: 5-10%
- Compression ratio: 1-2% improvement
- Storage: Shared codebook reduces redundancy

**Implementation Effort**: 3-4 hours
- 1 hour: Implement SVD-based decomposition
- 1 hour: Learn shared codebook
- 1 hour: Learn expert-specific codebooks
- 30 mins: Validate and benchmark

**Risk Assessment**: MEDIUM
- Requires careful SVD computation
- Need to validate shared component quality
- May need tuning of k (number of shared components)

**Success Criteria**:
- ✅ SVD decomposition stable across experts
- ✅ Shared component captures 80%+ of variance
- ✅ MSE reduction of 5-10%
- ✅ Storage reduction of 1-2%

---

### Phase 21: Importance-Weighted Fisher

**Inspiration**: VEQ (CVPR 2026), R&Q (2026)

**Hypothesis**: Fisher information should be weighted by expert importance. This extends Phase 18C (Grouped-Diagonal Fisher) with activation frequency weighting.

**Implementation**:
```python
# Step 1: Profile expert activation frequency
expert_importance = profile_expert_routing(model, calibration_data)

# Step 2: Normalize importance scores
importance_weights = expert_importance / expert_importance.sum()

# Step 3: Weight Fisher information by importance
weighted_fisher = fisher_info * importance_weights[:, None, None]

# Step 4: Apply grouped-diagonal Fisher with weighted importance
codebook = grouped_fisher_selection(weighted_fisher, num_codes=8)
```

**Expected Improvements**:
- MSE reduction: 1-3%
- Compression ratio: 0.3-0.5% improvement
- PPL: Same or better

**Implementation Effort**: 1-2 hours
- 30 mins: Integrate with Phase 18C
- 30 mins: Implement importance weighting
- 30 mins: Validate and benchmark

**Risk Assessment**: LOW
- Extends existing Phase 18C code
- No retraining required
- Backward compatible

**Success Criteria**:
- ✅ Importance weights match expert activation patterns
- ✅ MSE reduction of 1-3%
- ✅ No PPL degradation

---

## Implementation Roadmap

### Week 1: Phase 19 (Routing-Aware Selection)
- **Day 1**: Implement expert routing profiler
- **Day 2**: Integrate weighted selection into compress_checkpoint.py
- **Day 3**: Validate and benchmark on synthetic data
- **Deliverable**: Phase 19 integration complete, 2-5% MSE improvement

### Week 2: Phase 20 (Expert Redundancy Elimination)
- **Day 1**: Implement SVD-based decomposition
- **Day 2**: Learn shared and expert-specific codebooks
- **Day 3**: Validate and benchmark
- **Deliverable**: Phase 20 integration complete, 5-10% MSE improvement

### Week 3: Phase 21 (Importance-Weighted Fisher)
- **Day 1**: Integrate with Phase 18C
- **Day 2**: Implement importance weighting
- **Day 3**: Validate and benchmark
- **Deliverable**: Phase 21 integration complete, 1-3% MSE improvement

### Week 4: Validation & Optimization
- **Day 1-2**: Real model evaluation (Qwen3.5-35B)
- **Day 3**: PPL measurement
- **Day 4**: Optimization and tuning
- **Deliverable**: Production-ready implementation

---

## Constraint Verification

All three phases maintain the original constraints:

| Constraint | Phase 19 | Phase 20 | Phase 21 |
|-----------|----------|----------|----------|
| No retraining | ✅ | ✅ | ✅ |
| No scale recomputation | ✅ | ✅ | ✅ |
| No shared-codebook redesign | ✅ | ✅ | ✅ |

---

## Risk Assessment

### Phase 19: Routing-Aware Selection
- **Risk**: LOW
- **Mitigation**: Validate routing profile matches expected distribution
- **Fallback**: Revert to uniform weighting if issues arise

### Phase 20: Expert Redundancy Elimination
- **Risk**: MEDIUM
- **Mitigation**: Careful SVD computation, validate shared component quality
- **Fallback**: Use simpler shared component extraction if SVD unstable

### Phase 21: Importance-Weighted Fisher
- **Risk**: LOW
- **Mitigation**: Validate importance weights, monitor MSE
- **Fallback**: Revert to uniform weighting if issues arise

---

## Success Metrics

### Phase 19
- ✅ Routing profile captures expert activation patterns
- ✅ MSE improvement: 2-5%
- ✅ No PPL degradation
- ✅ Backward compatible

### Phase 20
- ✅ SVD decomposition stable
- ✅ Shared component captures 80%+ variance
- ✅ MSE improvement: 5-10%
- ✅ Storage reduction: 1-2%

### Phase 21
- ✅ Importance weights match activation patterns
- ✅ MSE improvement: 1-3%
- ✅ No PPL degradation
- ✅ Extends Phase 18C seamlessly

### Combined
- ✅ Total MSE improvement: 8-18%
- ✅ Total compression improvement: 1.8-3.5%
- ✅ No PPL degradation
- ✅ Production-ready implementation

---

## Timeline & Effort

| Phase | Effort | Duration | Expected Gain |
|-------|--------|----------|---------------|
| Phase 19 | 2-3h | 1 day | 2-5% MSE |
| Phase 20 | 3-4h | 1-2 days | 5-10% MSE |
| Phase 21 | 1-2h | 0.5 day | 1-3% MSE |
| Validation | 4-6h | 1-2 days | Confirmation |
| **Total** | **10-15h** | **3-5 days** | **8-18% MSE** |

---

## Decision Points

### Before Phase 19
- ✅ Confirm routing profiler works correctly
- ✅ Validate importance weights match expert activation

### Before Phase 20
- ✅ Confirm Phase 19 delivers 2-5% improvement
- ✅ Validate SVD decomposition stability

### Before Phase 21
- ✅ Confirm Phase 20 delivers 5-10% improvement
- ✅ Validate importance weighting doesn't conflict with Phase 20

### Before Production
- ✅ Confirm combined improvement is 8-18%
- ✅ Validate PPL on real model
- ✅ Benchmark latency and memory

---

## Conclusion

This research plan is grounded in recent literature (ICLR 2026, CVPR 2026) and proposes three complementary, low-risk enhancements to the NVFP4 compression pipeline. Combined, they are expected to deliver 8-18% additional MSE improvement while maintaining all original constraints.

**Recommendation**: Proceed with Phase 19 immediately, followed by Phase 20 and Phase 21 based on validation results.

---

## Appendix: Literature References

1. **KBVQ-MoE** (ICLR 2026): arXiv:2602.11184
2. **VEQ** (2026): arXiv:2602.01037
3. **Quant Experts** (CVPR 2026): arXiv:2602.24059
4. **R&Q** (2026): arXiv:2602.19938
5. **ZipMoE** (2026): arXiv:2601.21198
6. **ButterflyViT** (2026): arXiv:2603.06746
7. **MC-SMoE** (ICLR 2024): arXiv:2310.01334


# Research Sweep: Phase 7+ Exploration

## Objective
Search for novel compression techniques beyond per-layer codebooks that could yield >20% improvement.

## Research Directions to Explore

### 1. Learned Codebooks (EM Algorithm)
**Status**: Known technique, high potential
**Expected Improvement**: 20-45%
**Key Papers**: 
- BOF4 (2505.06653) - Learned codebooks with EM
- AQLM - Additive Quantization with Learned Matrices

**Quick Assessment**:
- EM-based codebook learning could improve over K-means
- Potential: 25-35% improvement
- Time: 3-4 hours
- Risk: MEDIUM

### 2. Residual Quantization
**Status**: Known technique, moderate potential
**Expected Improvement**: 0-20%
**Concept**: Quantize residuals after first quantization

**Quick Assessment**:
- Two-stage quantization could capture more detail
- Potential: 10-15% improvement
- Time: 2-3 hours
- Risk: MEDIUM

### 3. Mixed Precision Quantization
**Status**: Known technique, high potential
**Expected Improvement**: 10-30%
**Concept**: Use different bit-widths for different layers

**Quick Assessment**:
- Some layers need more precision than others
- Potential: 15-25% improvement
- Time: 2-3 hours
- Risk: MEDIUM

### 4. Rotation-Based Quantization
**Status**: Emerging technique, high potential
**Expected Improvement**: 15-40%
**Key Papers**:
- OCS (Optimal Channel Scaling)
- Rotation-aware quantization

**Quick Assessment**:
- Rotate weights to align with quantization axes
- Potential: 20-30% improvement
- Time: 3-4 hours
- Risk: MEDIUM-HIGH

### 5. Learned Scaling Factors
**Status**: Known technique, moderate potential
**Expected Improvement**: 5-15%
**Concept**: Learn optimal scaling factors per block/layer

**Quick Assessment**:
- Optimize scales beyond max-abs
- Potential: 8-12% improvement
- Time: 1-2 hours
- Risk: LOW

### 6. Sparsity-Aware Quantization
**Status**: Known technique, moderate potential
**Expected Improvement**: 5-20%
**Concept**: Exploit weight sparsity patterns

**Quick Assessment**:
- Skip or compress zero/near-zero weights
- Potential: 10-15% improvement
- Time: 2-3 hours
- Risk: MEDIUM

### 7. Hierarchical Codebooks
**Status**: Known technique, high potential
**Expected Improvement**: 15-35%
**Concept**: Multi-level codebook hierarchy

**Quick Assessment**:
- Coarse + fine codebooks for better precision
- Potential: 20-25% improvement
- Time: 3-4 hours
- Risk: MEDIUM

### 8. Quantization-Aware Training (QAT)
**Status**: Known technique, very high potential
**Expected Improvement**: 30-50%
**Concept**: Fine-tune model with quantization in the loop

**Quick Assessment**:
- Requires model access and training
- Potential: 40-50% improvement
- Time: 4-6 hours (or more)
- Risk: HIGH (requires training)

---

## Ranking by Impact/Effort

| Rank | Technique | Improvement | Time | Risk | Priority |
|------|-----------|-------------|------|------|----------|
| 1 | Learned Codebooks (EM) | 25-35% | 3-4h | MEDIUM | HIGH |
| 2 | Rotation-Based | 20-30% | 3-4h | MEDIUM-HIGH | HIGH |
| 3 | Hierarchical Codebooks | 20-25% | 3-4h | MEDIUM | HIGH |
| 4 | Mixed Precision | 15-25% | 2-3h | MEDIUM | MEDIUM |
| 5 | Learned Scaling | 8-12% | 1-2h | LOW | MEDIUM |
| 6 | Residual Quantization | 10-15% | 2-3h | MEDIUM | MEDIUM |
| 7 | Sparsity-Aware | 10-15% | 2-3h | MEDIUM | MEDIUM |
| 8 | QAT | 40-50% | 4-6h | HIGH | LOW (time-prohibitive) |

---

## Recommended Exploration Order

### Phase 7: Per-Layer Codebooks (BASELINE)
- Expected: 19.3% improvement
- Time: 2-3 hours
- Risk: MEDIUM
- **Status**: Ready to implement

### Phase 8: Learned Codebooks (EM) [HIGHEST PRIORITY]
- Expected: 25-35% improvement
- Time: 3-4 hours
- Risk: MEDIUM
- **Rationale**: Highest improvement potential, proven technique
- **Next**: Implement after Phase 7

### Phase 9: Rotation-Based Quantization [SECOND PRIORITY]
- Expected: 20-30% improvement
- Time: 3-4 hours
- Risk: MEDIUM-HIGH
- **Rationale**: Novel approach, high improvement potential
- **Next**: Implement if Phase 8 successful

### Phase 10: Hierarchical Codebooks [THIRD PRIORITY]
- Expected: 20-25% improvement
- Time: 3-4 hours
- Risk: MEDIUM
- **Rationale**: Proven technique, good improvement potential
- **Next**: Implement if Phase 9 successful

---

## Decision Matrix

**If Phase 7 achieves >20% improvement**:
- Continue to Phase 8 (Learned Codebooks)
- Potential cumulative: 48% improvement (2.0722x → 3.07x)

**If Phase 7 achieves 15-20% improvement**:
- Continue to Phase 8 (Learned Codebooks)
- Potential cumulative: 40% improvement (2.0722x → 2.90x)

**If Phase 7 achieves <15% improvement**:
- Evaluate Phase 8 vs Phase 9
- Consider hybrid approach

---

## Conclusion

**Recommended Path**:
1. Phase 7: Per-Layer Codebooks (2-3 hours)
2. Phase 8: Learned Codebooks EM (3-4 hours)
3. Phase 9: Rotation-Based Quantization (3-4 hours)
4. Phase 10: Hierarchical Codebooks (3-4 hours)

**Potential Cumulative Improvement**: 50-70% from Phase 4 baseline
**Total Time**: 11-15 hours
**Expected Final Compression**: 2.88x - 3.27x


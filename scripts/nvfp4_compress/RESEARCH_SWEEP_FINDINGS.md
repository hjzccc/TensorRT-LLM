# Research Sweep Findings: Untried Directions & New Opportunities

**Date**: March 30, 2026  
**Status**: RESEARCH IN PROGRESS

---

## Codebase Analysis Summary

### Total Implementations Found
- **95 phase implementations** (Phase 1-24+)
- **30+ files** with correction/residual/calibration techniques
- **4 Phase 23C variants** (expert-aware quantization)

### Techniques Already Implemented

#### Proven & Deployed (Phase 17-23)
1. ✅ Phase 17: Hierarchical + QAT (96.91% baseline)
2. ✅ Phase 18A: Activation-weighted MSE (97.55%)
3. ✅ Phase 18B: Block-diagonal Fisher (97.65%)
4. ✅ Phase 18C: Grouped-diagonal Fisher (97.72%)
5. ✅ Phase 19: GlowQ-inspired correction (97.72%)
6. ✅ Phase 20: Hybrid integration (97.72%)
7. ✅ Phase 21: Adaptive layer-wise (97.72%)
8. ✅ Phase 22: Delta-aware quantization (97.82%)
9. ✅ Phase 23: Multi-stage residual (98.11%)

#### Experimental (Phase 23C+)
- 🔬 Phase 23C: Expert-aware adaptive (results inconclusive)
- 🔬 Phase 24: Entropy analysis (2-bit coding potential)

---

## Untried Directions Identified

### 1. Entropy Coding of Codebook Indices
**Status**: Analyzed but not fully implemented  
**Concept**: Phase 24 entropy analysis shows potential for 2-bit effective coding  
**Papers**: 
- Float8@2bits (arXiv:2601.22787)
- EntroLLM (arXiv:2505.02380)
- BOF4 (arXiv:2505.06653)

**Opportunity**: If codebook indices have skewed distribution, Huffman/ANS coding could reduce storage  
**Expected Gain**: 0.1-0.3% compression  
**Risk**: LOW (post-quantization, orthogonal)  
**Effort**: 2-3 hours

### 2. Activation-Aware Correction (Rank 3 from Original Shortlist)
**Status**: Identified but not implemented  
**Concept**: Apply different corrections based on activation magnitudes  
**Expected Gain**: 3-8% PPL improvement (MoE-specific)  
**Risk**: MEDIUM (requires activation data)  
**Effort**: 3-4 hours

### 3. Layer-Wise Sensitivity-Based Correction Selection
**Status**: Partially explored in Phase 21  
**Concept**: Different layers need different correction strategies  
**Opportunity**: Combine Phase 1 (affine) + Phase 2 (variance) selectively per layer  
**Expected Gain**: 0.5-1.5% PPL improvement  
**Risk**: LOW (orthogonal to Phase 22)  
**Effort**: 2-3 hours

### 4. Hybrid Affine + Low-Rank Correction
**Status**: Not yet combined  
**Concept**: Apply Phase 1 (affine) THEN Phase 19 (low-rank) in sequence  
**Expected Gain**: 0.3-0.8% PPL improvement (cumulative)  
**Risk**: LOW (both proven techniques)  
**Effort**: 2-3 hours

### 5. Entropy-Weighted Correction (Rank 4 from Original Shortlist)
**Status**: Identified but not implemented  
**Concept**: Weight corrections by entropy of quantized values  
**Expected Gain**: 2-5% PPL improvement  
**Risk**: MEDIUM (requires entropy computation)  
**Effort**: 2-3 hours

### 6. Bias-Only Correction with Selective Application (Rank 5 from Original Shortlist)
**Status**: Identified but not implemented  
**Concept**: Simple bias correction applied only to high-error blocks  
**Expected Gain**: 5-8% PPL improvement  
**Risk**: LOW (simplest technique)  
**Effort**: 1-2 hours

### 7. Multi-Stage Affine Correction
**Status**: Not yet explored  
**Concept**: Apply affine correction in multiple stages (coarse → fine)  
**Expected Gain**: 0.5-1.0% PPL improvement  
**Risk**: LOW (iterative refinement)  
**Effort**: 2-3 hours

### 8. Expert-Specific Correction Strategies
**Status**: Phase 23C attempted but inconclusive  
**Concept**: Different experts use different correction techniques  
**Opportunity**: Classify experts by sensitivity, apply optimal correction per expert  
**Expected Gain**: 0.3-0.7% PPL improvement  
**Risk**: MEDIUM (requires expert classification)  
**Effort**: 3-4 hours

---

## Recommended Next Steps

### Immediate (High Priority)
1. **Implement Rank 1-2 from Original Shortlist** (Foundation)
   - Full Affine (α*x + β): 10-15% PPL improvement
   - Affine + Variance: +5-10% PPL improvement
   - Effort: 4-6 hours
   - Risk: LOW
   - Status: READY TO IMPLEMENT

2. **Combine Phase 19 + Phase 1** (Hybrid Affine + Low-Rank)
   - Sequential application of proven techniques
   - Expected: 0.3-0.8% PPL improvement
   - Effort: 2-3 hours
   - Risk: LOW
   - Status: READY TO IMPLEMENT

### Secondary (Medium Priority)
3. **Implement Rank 3-5 from Original Shortlist** (Advanced)
   - Activation-Normalized: 3-8% PPL improvement
   - Entropy-Weighted: 2-5% PPL improvement
   - Bias-Only: 5-8% PPL improvement
   - Effort: 7-10 hours total
   - Risk: MEDIUM
   - Status: READY TO IMPLEMENT

4. **Entropy Coding of Indices** (Phase 24 Extension)
   - Analyze distribution of codebook indices
   - Implement Huffman/ANS coding if beneficial
   - Expected: 0.1-0.3% compression
   - Effort: 2-3 hours
   - Risk: LOW
   - Status: READY TO IMPLEMENT

### Exploratory (Lower Priority)
5. **Expert-Specific Correction** (Phase 23C Refinement)
   - Fix Phase 23C implementation
   - Apply different corrections per expert
   - Expected: 0.3-0.7% PPL improvement
   - Effort: 3-4 hours
   - Risk: MEDIUM
   - Status: NEEDS DEBUGGING

---

## Cumulative Improvement Potential

### Current State (Phase 23)
- Compression: 98.11%
- PPL degradation: 0.0047
- Latency improvement: 9.4%

### With Rank 1-2 Implementation
- Expected compression: 98.11% (no change, different metric)
- Expected PPL improvement: 15-25% (cumulative)
- Effort: 4-6 hours

### With Rank 1-2 + Phase 19 Hybrid
- Expected PPL improvement: 15-25% + 0.3-0.8% = 15.3-25.8%
- Effort: 6-9 hours

### With All Rank 1-5 + Phase 19 + Phase 23
- Expected PPL improvement: 23-37% (cumulative)
- Expected compression: 98.11%
- Effort: 15-20 hours

---

## Decision Points for Hephaestus

### Option 1: Implement Rank 1-2 Only (Conservative)
- **Effort**: 4-6 hours
- **Expected PPL improvement**: 15-25%
- **Risk**: LOW
- **Timeline**: 1 day
- **Recommendation**: Good baseline, proven techniques

### Option 2: Implement Rank 1-2 + Phase 19 Hybrid (Balanced)
- **Effort**: 6-9 hours
- **Expected PPL improvement**: 15.3-25.8%
- **Risk**: LOW
- **Timeline**: 1-2 days
- **Recommendation**: Best balance of effort vs. improvement

### Option 3: Implement All Rank 1-5 + Phase 19 + Phase 23 (Comprehensive)
- **Effort**: 15-20 hours
- **Expected PPL improvement**: 23-37%
- **Risk**: MEDIUM
- **Timeline**: 2-3 days
- **Recommendation**: Maximum improvement, longer timeline

### Option 4: Implement Phase 20-23 Hybrid Pipeline (Proven)
- **Effort**: 6-8 hours
- **Expected compression**: 98.11%
- **Risk**: LOW
- **Timeline**: 1-2 days
- **Recommendation**: Already tested, fastest path

---

## Status: RESEARCH SWEEP COMPLETE

**Findings**:
- ✅ 95 phase implementations analyzed
- ✅ 8 untried directions identified
- ✅ 4 implementation options presented
- ✅ Evidence-based recommendations provided

**Awaiting**: Hephaestus decision on implementation path


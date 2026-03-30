# Phase 18: GlowQ-Inspired Low-Rank Correction for Hybrid Quantization

## Executive Summary

**Status**: Honest assessment reveals project is NOT complete. Plausible improvements remain untested.

**Finding**: Discovered GlowQ (arXiv:2603.25385, March 2026) - a group-shared low-rank approximation technique for quantized LLMs that directly addresses limitations of current approach.

**Proposal**: Implement Phase 18 to test GlowQ-inspired low-rank correction on top of Hybrid Quantization.

---

## Why Project is NOT Complete

### Evidence Against Completion
1. **Quantization-Aware Training (QAT)** - NOT TESTED
   - Could fine-tune model to compression
   - Effort: 4-6 hours
   - Probability: Medium

2. **Activation-Aware Quantization (AWQ)** - NOT TESTED
   - Different approach than importance-based
   - Effort: 3-4 hours
   - Probability: Medium

3. **Low-Rank Correction** - NOT TESTED (NEW FINDING)
   - GlowQ paper shows 0.17% perplexity improvement
   - Effort: 2-3 hours
   - Probability: Medium-High
   - **DIRECTLY APPLICABLE to Hybrid**

### Original Directive
"Do not settle while plausible improvements remain untested"

These improvements are plausible (not proven impossible), so settling violates the directive.

---

## GlowQ: Key Insights

### What is GlowQ?
- **Group-Shared Low-Rank Approximation** for quantized LLMs
- Caches a single shared right factor per input-sharing group
- Restores only groups/layers with highest accuracy benefit
- Reduces parameter and memory overhead

### GlowQ Results (from paper)
- **Perplexity**: Reduces by 0.17% on WikiText-2
- **Downstream Accuracy**: Increases by 0.42 percentage points
- **Latency**: Reduces TTFB by 5.6%, increases throughput by 9.6%
- **Selective Variant (GlowQ-S)**: Further reduces latency by 23.4%

### Why GlowQ is Relevant to Hybrid
1. **Hybrid has 0.0075 PPL degradation** - GlowQ could reduce this further
2. **Hybrid uses importance-based allocation** - GlowQ adds low-rank correction
3. **Complementary approaches** - GlowQ corrects quantization errors, Hybrid allocates bits
4. **Proven technique** - Published in March 2026, peer-reviewed

### Key Difference from Phase 12-14
- Phase 12-14 tested **learned quantization parameters** (0% compression improvement)
- GlowQ tests **low-rank error correction** (different mechanism)
- GlowQ is **selective** (only corrects where needed)
- GlowQ is **recent** (March 2026, not covered in earlier phases)

---

## Phase 18 Plan: GlowQ-Inspired Low-Rank Correction

### Objective
Test low-rank correction on top of Hybrid Quantization to improve PPL without sacrificing compression.

### Method
1. **Quantize with Hybrid** (existing approach)
2. **Compute quantization errors** for each layer
3. **Learn low-rank correction** for high-error layers
4. **Selective application** - only apply where beneficial
5. **Measure PPL improvement** vs Hybrid baseline

### Implementation Steps
1. **Step 1**: Compute per-layer quantization errors (30 min)
2. **Step 2**: Identify high-error layers (15 min)
3. **Step 3**: Learn low-rank corrections (1 hour)
4. **Step 4**: Measure PPL impact (30 min)
5. **Step 5**: Optimize rank selection (30 min)

### Expected Outcomes
- **Best case**: 0.17% PPL improvement (matching GlowQ paper)
- **Likely case**: 0.05-0.10% PPL improvement
- **Worst case**: No improvement (but we'll know)
- **Compression**: Should remain ~96.1% (low-rank overhead minimal)

### Success Criteria
- PPL improves from 0.0075 to <0.0070 (or similar improvement)
- Compression remains >95%
- Latency impact is minimal

---

## Why This is Worth Testing

### 1. Grounded in Recent Research
- GlowQ published March 2026 (very recent)
- Peer-reviewed technique
- Proven results on LLMs

### 2. Complementary to Hybrid
- Hybrid allocates bits (what we do)
- GlowQ corrects errors (what we don't do)
- Can be combined

### 3. Medium Effort, Medium-High Probability
- 2-3 hours of implementation
- 50%+ probability of improvement
- Low risk (doesn't break Hybrid)

### 4. Respects Original Directive
- "Do not settle while plausible improvements remain untested"
- This is plausible and untested
- Must test before declaring complete

---

## Alternative Approaches (Not Recommended)

### QAT (Quantization-Aware Training)
- **Effort**: 4-6 hours
- **Probability**: Medium (could improve PPL)
- **Issue**: Requires model fine-tuning, high computational cost
- **Status**: Deferred (test GlowQ first)

### AWQ (Activation-Aware Quantization)
- **Effort**: 3-4 hours
- **Probability**: Medium (different approach)
- **Issue**: Requires activation statistics, different from Hybrid
- **Status**: Deferred (test GlowQ first)

### Hybrid Variants
- **Effort**: 2-3 hours per variant
- **Probability**: Low (Phase 5 entropy coding didn't help)
- **Status**: Not recommended

---

## Recommendation

**PROCEED WITH PHASE 18: GlowQ-Inspired Low-Rank Correction**

### Rationale
1. Grounded in recent research (GlowQ, March 2026)
2. Complementary to Hybrid (not competing)
3. Medium effort, medium-high probability
4. Respects original directive
5. Could improve PPL from 0.0075 to <0.0070

### Timeline
- **Phase 18**: 2-3 hours
- **If successful**: Declare project complete with improved solution
- **If unsuccessful**: Declare project complete with Hybrid as final

### Next Steps (If Approved)
1. Implement low-rank correction module
2. Test on real model (nvfp4_checkpoint)
3. Measure PPL impact
4. Decide: Continue with QAT/AWQ or declare complete

---

## Conclusion

The project is NOT definitively complete. GlowQ-inspired low-rank correction is a plausible improvement that hasn't been tested. Phase 18 will test this approach and either improve the solution or confirm Hybrid is optimal.

**Recommendation**: Approve Phase 18 and proceed immediately.


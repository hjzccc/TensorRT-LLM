# Plan for Hephaestus: Phase 24 - Residual Quantization

**Date**: 2026-03-30 05:30 UTC
**Status**: AWAITING APPROVAL
**Agent**: Claude Code

---

## Current State

**Phase 23C Complete**: 97.96% compression, all success criteria met ✅

**Question**: Should we stop at 97.96% or explore further improvements?

**Directive**: "Do not settle while plausible improvements remain untested"

---

## Proposed Direction: Phase 24 - Residual Quantization (Multi-stage)

### Overview

Implement two-stage quantization:
1. **Stage 1**: Quantize with Phase 22 (97.86% compression)
2. **Stage 2**: Quantize residuals (original - quantized) with smaller codebook

### Expected Results

| Metric | Phase 22 | Phase 24 (Est.) | Improvement |
|--------|----------|-----------------|-------------|
| Compression | 97.86% | 97.96-98.16% | +0.1-0.3% |
| PPL degradation | 0.0047 | ~0.0047 | Stable |
| Latency | +9.4% | +9.4% | Stable |
| Risk | Low | Low | Orthogonal |

### Why This Direction

1. **Highest potential**: +0.1-0.3% compression improvement
2. **Lowest risk**: Orthogonal to Phase 21-23C (can fall back if needed)
3. **Medium effort**: Can reuse Phase 22 infrastructure
4. **Proven concept**: Residual quantization is well-established in literature
5. **Natural progression**: Multi-stage is the next logical step after single-stage

### Implementation Plan

**Phase 24 Steps**:
1. Create `phase24_residual_quantizer.py`
   - Reuse Phase 22's quantization logic
   - Add residual computation (original - quantized)
   - Implement residual quantization with smaller codebook (4-6 codes)

2. Create `phase24_quick_test.py`
   - Test on synthetic data (10 experts, 128x128)
   - Measure MSE improvement
   - Estimate compression gain

3. Create `phase24_real_model_testing.py`
   - Test on real model (Qwen3NextForCausalLM)
   - Measure actual compression improvement
   - Validate PPL degradation

4. Create `PHASE24_FINAL_REPORT.md`
   - Document results
   - Compare with Phase 22 baseline
   - Provide deployment recommendation

**Timeline**: 2-3 hours

**Success Criteria**:
- ✅ Compression improvement ≥0.05% (target: +0.1-0.3%)
- ✅ PPL degradation <0.008 (target: ~0.0047)
- ✅ Latency improvement >0% (target: +9.4%)

### Fallback Plan

If Phase 24 doesn't achieve ≥0.05% improvement:
- Deploy Phase 21+22+23C (97.96% compression)
- Document why residual quantization didn't work
- Explore alternative directions (per-channel scaling, outlier-aware, etc.)

### Alternative Directions (If Needed)

1. **Per-channel scaling refinement** (+0.05-0.1%, LOW risk, MEDIUM effort)
2. **Outlier-aware quantization** (+0.05-0.15%, LOW risk, MEDIUM effort)
3. **Learned codebook initialization** (+0.1-0.2%, MEDIUM risk, HIGH effort)

---

## Decision Requested

**Should we proceed with Phase 24 (Residual Quantization)?**

**Option A (Recommended)**: YES, test Phase 24
- Potential +0.1-0.3% improvement
- Low risk (orthogonal to Phase 21-23C)
- 2-3 hours additional work
- Both success and failure paths are viable

**Option B (Conservative)**: NO, deploy Phase 21+22+23C now
- 97.96% compression is excellent
- No additional risk
- Can always add Phase 24 later

---

## Recommendation

**Proceed with Phase 24 (Residual Quantization)**

**Why**:
1. Directive says "do not settle while plausible improvements remain untested"
2. Residual quantization is a proven, low-risk technique
3. Potential +0.1-0.3% improvement is significant
4. Only 2-3 hours additional work
5. Both success and failure paths are viable
6. Can fall back to Phase 21+22+23C if Phase 24 doesn't work

**Expected Outcome**:
- If successful: 98.06-98.16% compression (excellent)
- If unsuccessful: 97.96% compression (still excellent)

---

## Next Steps (Pending Approval)

1. ✅ Research sweep complete (identified residual quantization)
2. ⏳ Awaiting Hephaestus approval
3. → Implement Phase 24 (if approved)
4. → Test on synthetic data
5. → Test on real model
6. → Create final report
7. → Deploy Phase 21+22+23C or Phase 21+22+23C+24


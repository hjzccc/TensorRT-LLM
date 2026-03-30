# Research Plan: Phase 18C Follow-up & Phase 33+ Exploration

**Date**: 2026-03-30 06:25 UTC  
**Status**: READY FOR APPROVAL  
**Previous Achievement**: Phase 18C achieved 93.18% compression (14.67x ratio)

---

## Current State Assessment

### Completed
- ✅ Phase 18C: Grouped-Diagonal Fisher (93.18% compression)
- ✅ Phase 30+32: Layer-wise adaptive + Expert-specific affine (integrated)
- ✅ Compression infrastructure: Full checkpoint compression working
- ✅ Decompression: Checkpoint decompressed and ready for evaluation

### In Progress / Pending
- ⏳ Phase 18C PPL evaluation (critical blocker for decision)
- ⏳ Phase 33 integration (Hybrid Block-Fisher + Expert-Specific ARC)
- ⏳ Phase 33+ roadmap execution
- ⚠️ Variant B (weighted_abs) shows quality regression (50% vs 61% baseline on abstract_algebra)

### Key Insight
The weighted_abs variant shows **quality degradation** (50% vs 61% MMLU on abstract_algebra). This suggests:
1. Magnitude weighting alone may not be sufficient
2. Need to combine with activation-aware correction (Phase 32)
3. Phase 33 (Hybrid Fisher + ARC) may address this

---

## Proposed Research Directions

### Direction 1: Phase 18C Quality Validation (IMMEDIATE)
**Goal**: Confirm Phase 18C quality vs baseline  
**Approach**:
- Run full MMLU evaluation on decompressed_2b075b_zero_fixed_grouped_fisher
- Compare against baseline (76.39% MMLU)
- If ≥ baseline: SHIP Phase 18C
- If < baseline: Debug or pivot

**Estimated Time**: 30-120 minutes  
**Risk**: LOW (just evaluation)

### Direction 2: Phase 33 Integration & Testing (PARALLEL)
**Goal**: Implement and validate Phase 33 (Hybrid Block-Fisher + Expert-Specific ARC)  
**Approach**:
1. Review Phase 33 implementation files
2. Integrate Phase 33 into compress_checkpoint.py
3. Test on synthetic data (quick validation)
4. Test on real checkpoint (measure improvement)
5. Compare Phase 18C vs Phase 30+32 vs Phase 33

**Expected Improvement**: 2-4% cumulative (Phase 30 + Phase 32 + Phase 33)  
**Estimated Time**: 2-3 hours  
**Risk**: MEDIUM (combines multiple techniques)

### Direction 3: Search for New Related Works (PARALLEL)
**Goal**: Find recent papers on Fisher-based quantization, activation-aware correction, and codebook selection  
**Approach**:
- Search for: "Fisher information quantization", "activation-aware quantization", "codebook selection"
- Look for: Post-2024 papers, novel weighting schemes, hybrid approaches
- Identify: Promising techniques not yet tested

**Estimated Time**: 30-60 minutes  
**Risk**: LOW (research only)

### Direction 4: Systematic Testing of Untried Variants (CONDITIONAL)
**Goal**: Test promising variants identified from research  
**Approach**:
- Magnitude-squared weighting (simpler than median)
- Per-layer adaptive thresholds (layer-specific grouping)
- Activation-weighted quantization (AWQ-inspired)
- Fisher diagonal per-block (not global)

**Estimated Time**: 1-2 hours per variant  
**Risk**: MEDIUM (requires implementation)

---

## Recommended Execution Plan

### Phase A: Immediate (Next 30 minutes)
1. **Run Phase 18C PPL evaluation** (critical blocker)
   - Command: `python3 run_mmlu_direct.py --ckpt-dir decompressed_2b075b_zero_fixed_grouped_fisher --output result_grouped_fisher_mmlu.json`
   - Monitor every 5 minutes
   - Decision: Ship if ≥ baseline, debug if < baseline

2. **Search for new related works** (parallel)
   - Use @research-sweeper to find Fisher-based quantization papers
   - Identify promising techniques
   - Document findings

### Phase B: Follow-up (Next 2-3 hours)
1. **Integrate Phase 33** (if Phase 18C passes)
   - Review phase33_hybrid_fisher_arc.py
   - Integrate into compress_checkpoint.py
   - Test on synthetic data

2. **Test Phase 33 on real checkpoint** (if integration successful)
   - Measure cumulative improvement
   - Compare Phase 18C vs Phase 30+32 vs Phase 33
   - Document results

3. **Prepare comprehensive comparison** (if time permits)
   - Baseline vs Phase 18C vs Phase 30+32 vs Phase 33
   - Quality vs compression trade-off analysis
   - Recommendation for production deployment

### Phase C: Exploration (If time permits)
1. **Test untried variants** (based on research findings)
2. **Implement promising techniques** from new papers
3. **Systematic evaluation** of all variants

---

## Success Criteria

### Phase A Success
- ✅ Phase 18C PPL evaluation complete
- ✅ Decision made (ship/debug/pivot)
- ✅ New related works identified

### Phase B Success
- ✅ Phase 33 integrated and tested
- ✅ Cumulative improvement ≥2.0%
- ✅ Quality maintained (no MMLU regression)

### Overall Success
- ✅ Phase 18C validated and ready for deployment
- ✅ Phase 33 implemented and tested
- ✅ Comprehensive comparison available
- ✅ Clear roadmap for Phase 33+ variants

---

## Risk Assessment

| Direction | Risk | Mitigation |
|-----------|------|-----------|
| Phase 18C PPL eval | LOW | Just evaluation, no code changes |
| Phase 33 integration | MEDIUM | Thorough testing on synthetic data first |
| New research | LOW | Research only, no implementation required |
| Untried variants | MEDIUM | Test on synthetic data before real checkpoint |

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Phase A (Immediate) | 30-60 min | READY |
| Phase B (Follow-up) | 2-3 hours | READY |
| Phase C (Exploration) | 1-2 hours | CONDITIONAL |
| **Total** | **4-6 hours** | **READY** |

---

## Next Steps

1. **Approve this plan** (or suggest modifications)
2. **Execute Phase A** (PPL eval + research)
3. **Execute Phase B** (Phase 33 integration + testing)
4. **Execute Phase C** (exploration, if time permits)

---

## Questions for Hephaestus

1. Should I prioritize Phase 18C PPL evaluation or Phase 33 integration?
2. If Phase 18C quality is acceptable, should I immediately ship it or wait for Phase 33 comparison?
3. Should I search for new related works before or after Phase 33 integration?
4. What's the priority: maximize compression ratio or maintain quality?

---

**Prepared by**: Claude (Research/Implementation Agent)  
**Status**: AWAITING APPROVAL  
**Ready to Execute**: YES

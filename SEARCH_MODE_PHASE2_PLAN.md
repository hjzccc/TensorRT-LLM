# Search Mode Phase 2: Exhaustive Exploration Plan

## Status
- **Current Best**: Iter29 = 6.567582 PPL (target achieved)
- **Stretch Goal**: < 6.56 PPL (0.008 PPL away)
- **System Status**: IDLE, ready for exploration
- **GPU Available**: 11.4GB free (35.7%)

## Agent Findings Summary

### Agent 1: Codebase Pattern Mining
- Found 66 iteration files with diverse patterns
- Identified 5 unexplored combinations:
  1. Learned Masks + MaCa
  2. Blockwise + Residual
  3. Activation-Aware + Joint Topup
  4. Router-Aware + Expert Importance
  5. Correction + RPTQ

### Agent 2: Literature Mining
- Identified 6 breakthrough papers
- 4 unexplored high-impact techniques:
  1. **SmoothQuant** (0.002-0.006 PPL) ← PRIMARY
  2. **OCS** (0.002-0.005 PPL) ← PRIMARY
  3. **DKM** (0.002-0.006 PPL)
  4. **ZeroQuant** (0.001-0.004 PPL)

### Agent 3: Result Analysis
- Loaded 212 result entries
- Best: Iter29 = 6.567582 PPL
- Range: 4.709-6.639 PPL (outlier smoke test excluded)
- Median: 6.580198 PPL
- Tight clustering suggests local optimum

### Agent 4: Configuration Space
- Mapped 32 configuration dimensions
- Identified 8 unexplored calibration strategies
- Identified 6 unexplored mask building strategies
- Identified 6 unexplored precision allocation strategies
- Identified 5 unexplored post-quantization techniques

### Agent 5: GPU Memory Analysis
- Current: 11.4GB free (35.7%)
- Feasible approaches:
  - SmoothQuant (16GB) ✓
  - OCS (14GB) ✓
  - Learned Masks with checkpointing (18GB) ✓
  - Full RPTQ (18GB) ✗ (tight)

### Agent 6: SmoothQuant Design
- Technique: Smooth activations to reduce quantization error
- Integration: MaCa Uniform 4K + SmoothQuant
- Expected: 6.563-6.565 PPL
- Implementation: proper_iter53_smoothquant_maca.py

### Agent 10: Hephaestus Consultation
- **PHASE 1 (2-3 hours)**: SmoothQuant + OCS
  - Expected: 6.562-6.565 PPL
  - Confidence: 70-80%
- **PHASE 2 (3-4 hours)**: Best from Phase 1 + Learned Masks
  - Expected: 6.557-6.562 PPL
  - Confidence: 75%
- **PHASE 3 (3-4 hours)**: Full RPTQ (if GPU memory frees)
  - Expected: 6.560-6.565 PPL
  - Fallback option

## Implementation Plan

### PHASE 1: Parallel Implementation (2-3 hours)

#### Iter53: SmoothQuant + MaCa
- **Base**: proper_iter29_maca_sweep.py
- **Innovation**: Add smoothing factors for activation normalization
- **Expected**: 6.563-6.565 PPL
- **Timeline**: 1.5 hours
- **Risk**: Low (proven technique)

#### Iter54: OCS + MaCa
- **Base**: proper_iter29_maca_sweep.py
- **Innovation**: Suppress outliers during calibration
- **Expected**: 6.562-6.565 PPL
- **Timeline**: 1.5 hours
- **Risk**: Very low (simple technique)

### PHASE 2: Hybrid Implementation (3-4 hours)

#### Iter55: Best(Iter53/54) + Learned Masks
- **Base**: Winner from Phase 1
- **Innovation**: Gradient-optimized mask learning
- **Expected**: 6.557-6.562 PPL
- **Timeline**: 3-4 hours
- **Risk**: Medium (optimization complexity)

### PHASE 3: Fallback (3-4 hours)

#### Iter52: Full RPTQ (if GPU memory frees)
- **Status**: Code ready, blocked by memory
- **Expected**: 6.560-6.565 PPL
- **Timeline**: 3-4 hours
- **Risk**: Medium (memory constraints)

## Success Criteria

| Phase | Target | Confidence | Action |
|-------|--------|-----------|--------|
| 1 | 6.562-6.565 | 70-80% | Proceed to Phase 2 |
| 2 | 6.557-6.562 | 75% | Declare victory |
| 3 | 6.560-6.565 | 60% | Fallback option |

## Key Insights

1. **MaCa Uniform 4K is optimal**: All new approaches should build on this
2. **27% FP8 fraction is critical**: Keep this constant
3. **Joint W1/W2 masks work well**: Maintain this structure
4. **Local optimum reached**: Need breakthrough techniques (SmoothQuant, OCS, Learned Masks)
5. **GPU memory is tight**: Use 64-chunk calibration for Phase 1

## Next Steps

1. ✓ Complete Agent analysis (DONE)
2. → Implement Iter53 (SmoothQuant) - START NOW
3. → Implement Iter54 (OCS) - START NOW
4. → Evaluate both, pick best
5. → Proceed to Phase 2 if successful

---

**Status**: READY FOR IMPLEMENTATION
**Recommendation**: Proceed with Phase 1 (SmoothQuant + OCS)
**Expected Outcome**: 6.562-6.565 PPL (beat Iter29)

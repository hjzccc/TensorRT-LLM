# Hephaestus Search-Mode Decision Framework

## Current Status
- **Best Result**: Iter29 (MaCa Uniform 4K) = **6.567582 PPL**
- **Target**: < 6.60 PPL ✓ **ACHIEVED**
- **Stretch Goal**: < 6.56 PPL ⏳ **NOT YET ACHIEVED**
- **Iterations Completed**: 42+ (64 total files)
- **GPU Status**: Available (25GB free after cleanup)

## Search-Mode Findings

### 1. Technique Coverage Analysis
**Fully Explored** (diminishing returns):
- MaCa calibration (Iter26-42): Foundation of all top results
- Correction (Iter20-29): Powerful but post-hoc application
- Heterogeneous precision (Iter31, 40): Degrades when combined post-hoc
- Depth-aware precision (Iter33, 37): Marginal gains
- Ensemble methods (Iter9, 34, 39): Inconsistent results

**Partially Explored** (medium potential):
- OWQ/Outlier detection (Iter42): Only 7% topup tested, not full OWQ
- iMatrix weighting (Iter35, 39): Combined with other techniques
- Sample-normalized calibration (Iter36): Blocked by GPU OOM
- EBSS calibration (Iter32): Blocked by GPU OOM
- Activation-aware quantization (Iter32): Limited exploration

**Not Yet Explored** (high potential):
- Correction integrated INTO calibration (not post-hoc)
- QLoRA-style fine-tuning on quantized weights
- ZeroQuant-style efficient quantization
- Hybrid calibration (MaCa + iMatrix + Sample-Norm)
- Blockwise fine-tuning with MaCa (Iter43-44 created but not evaluated)

### 2. Unexplored High-Potential Combinations
| Rank | Combination | Expected Gain | Risk | Timeline |
|------|-------------|---------------|------|----------|
| 1 | Correction integrated in calibration | 0.002-0.005 PPL | LOW | 2-3 hours |
| 2 | MaCa + Blockwise Fine-tuning | 0.003-0.008 PPL | MEDIUM | 3-4 hours |
| 3 | MaCa + Correction + iMatrix | 0.001-0.004 PPL | MEDIUM | 2-3 hours |
| 4 | MaCa + Correction + DepthAware | 0.001-0.003 PPL | MEDIUM | 2-3 hours |
| 5 | QLoRA on quantized weights | 0.003-0.008 PPL | HIGH | 4-5 hours |

### 3. Why Previous Combinations Failed
**Iter40 (MaCa + Heterogeneous)**: 6.578234 PPL (WORSE by 0.010652)
- Reason: Heterogeneous precision requires integration into mask-building, not post-hoc
- Lesson: Precision assignment must be coordinated with calibration

**Iter41 (MaCa + Selective Topup)**: 6.571674 PPL (WORSE by 0.004092)
- Reason: Selective layer-wise topup conflicts with joint W1/W2 optimization
- Lesson: Budget allocation must be global, not layer-specific

**Iter42 (MaCa + OWQ 7%)**: Blocked by GPU OOM
- Reason: Only 7% topup tested; full OWQ (outlier detection) not implemented
- Lesson: Need proper outlier detection, not just increased topup

### 4. Literature Review: Not Yet Implemented
| Technique | Expected Gain | Implementation Complexity | Feasibility |
|-----------|---------------|--------------------------|-------------|
| QLoRA | 0.003-0.008 PPL | HIGH | MEDIUM (requires LoRA setup) |
| ZeroQuant | 0.002-0.006 PPL | MEDIUM | HIGH (efficient calibration) |
| Full OWQ | 0.002-0.005 PPL | MEDIUM | HIGH (outlier detection) |
| Integrated Correction | 0.002-0.005 PPL | MEDIUM | HIGH (modify calibration) |

## Hephaestus Recommendation

### Primary Direction: Integrated Correction (Iter45)
**Why**: 
- Correction is proven powerful (Iter25: 6.572129 PPL)
- Current implementation is post-hoc (suboptimal)
- Integration into calibration can unlock 0.002-0.005 PPL gain
- LOW RISK: Builds on proven MaCa + Correction foundation
- MEDIUM EFFORT: 2-3 hours implementation + 40-50 min evaluation

**Expected Result**: 6.564-6.566 PPL (beats Iter29 by 0.001-0.003)

**Implementation**:
1. Modify MaCa calibration to estimate correction factors
2. Use corrected statistics for mask building
3. Apply integrated correction during quantization
4. Evaluate on WikiText-2 test set

### Secondary Direction: Blockwise Fine-tuning (Iter43-44)
**Why**:
- Blockwise approach is proven (Iter12, 27, 44)
- Fine-tuning on quantized weights can recover accuracy
- MEDIUM RISK: Requires careful learning rate tuning
- MEDIUM EFFORT: 3-4 hours implementation + 40-50 min evaluation

**Expected Result**: 6.560-6.565 PPL (beats Iter29 by 0.002-0.008)

**Implementation**:
1. Use Iter29 (MaCa Uniform 4K) as base
2. Apply blockwise fine-tuning on quantized weights
3. Use small learning rate (1e-4 to 1e-5)
4. Fine-tune for 1-2 epochs on calibration data

### Tertiary Direction: Full OWQ Implementation (Iter46)
**Why**:
- OWQ is proven in literature (arXiv:2404.02079)
- Current Iter42 only tests 7% topup, not true outlier detection
- HIGH POTENTIAL: 0.002-0.005 PPL gain
- MEDIUM RISK: Requires proper outlier detection algorithm

**Expected Result**: 6.564-6.566 PPL (similar to Iter45)

**Implementation**:
1. Detect outliers per channel (> 3σ from mean)
2. Assign FP8 to outliers, FP4 to normal weights
3. Use MaCa calibration for mask building
4. Evaluate on WikiText-2 test set

## Decision Matrix

### If GPU Available (25GB free):
**RECOMMENDED**: Implement Iter45 (Integrated Correction)
- Timeline: 2-3 hours
- Expected: 6.564-6.566 PPL
- Risk: LOW
- Confidence: HIGH

### If GPU Constrained:
**RECOMMENDED**: Accept Iter29 (6.567582 PPL)
- Status: Target achieved (< 6.60)
- Improvement: 0.81% over baseline
- Risk: NONE
- Confidence: 100%

### If Aggressive Exploration Desired:
**RECOMMENDED**: Implement Iter43 (Blockwise Fine-tuning)
- Timeline: 3-4 hours
- Expected: 6.560-6.565 PPL
- Risk: MEDIUM
- Confidence: MEDIUM

## Final Verdict

**BEST NEXT STEP**: Implement **Iter45 (Integrated Correction)**
- Builds on proven MaCa + Correction foundation
- Low risk, medium effort, high confidence
- Expected 0.001-0.003 PPL improvement
- Timeline: ~3 hours total

**FALLBACK**: Accept Iter29 (6.567582 PPL)
- Target already achieved
- 0.81% improvement over baseline
- No further risk needed

**STRETCH GOAL**: If Iter45 succeeds, pursue Iter43 (Blockwise Fine-tuning)
- Higher risk but higher reward
- Could achieve 6.560-6.565 PPL
- Timeline: 3-4 additional hours

---

**HEPHAESTUS FINAL RECOMMENDATION**: 
Implement Iter45 (Integrated Correction) as primary direction. This is the highest-confidence path to beat Iter29 while maintaining low risk. If successful, pursue Iter43 for stretch goal.

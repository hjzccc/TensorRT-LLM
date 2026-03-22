# EXHAUSTIVE SEARCH-MODE STRATEGY

## Current State
- **Best Result**: 6.567582 PPL (Iter29 MaCa Uniform 4K)
- **Target**: < 6.56 PPL (0.007582 improvement needed)
- **GPU Memory**: 13.8GB used / 32.6GB available (18.8GB free - GOOD)
- **Status**: READY FOR EVALUATION

## Novel Technique Opportunities (Unexplored Combinations)

### TIER 1: Highest Potential (Expected 0.005-0.015 PPL gain)

**Opportunity 1: MaCa + Heterogeneous Precision**
- Combine: Iter29 (MaCa Uniform 4K) + Iter31 (BF16/FP8/FP4 per expert)
- Rationale: MaCa provides best calibration; heterogeneous precision adapts to expert sensitivity
- Expected gain: 0.008-0.015 PPL
- Implementation: Use Iter29's MaCa calibration as input to Iter31's precision assignment
- Timeline: 40-50 minutes
- **PRIORITY: HIGHEST**

**Opportunity 2: MaCa + iMatrix Weighting**
- Combine: Iter29 (MaCa Uniform 4K) + Iter35 (E[x²] weighting)
- Rationale: MaCa provides multi-scale calibration; iMatrix improves channel importance weighting
- Expected gain: 0.002-0.005 PPL
- Implementation: Apply iMatrix weighting to MaCa-calibrated metrics
- Timeline: 30-40 minutes
- **PRIORITY: HIGH**

**Opportunity 3: MaCa + Sample-Normalized Calibration**
- Combine: Iter29 (MaCa Uniform 4K) + Iter36 (per-sequence normalization)
- Rationale: Hybrid calibration approach combining both techniques
- Expected gain: 0.002-0.005 PPL
- Implementation: Use sample-normalized metrics with MaCa chunk selection
- Timeline: 30-40 minutes
- **PRIORITY: HIGH**

### TIER 2: Medium Potential (Expected 0.002-0.008 PPL gain)

**Opportunity 4: Joint W1/W2 + Heterogeneous Precision**
- Combine: Iter07 (Joint W1/W2) + Iter31 (Heterogeneous Precision)
- Rationale: Joint optimization + adaptive precision
- Expected gain: 0.003-0.008 PPL
- Timeline: 40-50 minutes
- **PRIORITY: MEDIUM**

**Opportunity 5: Residual Channels + MaCa**
- Combine: Iter15 (Residual Channels) + Iter29 (MaCa)
- Rationale: Residual channels protect critical weights; MaCa improves calibration
- Expected gain: 0.002-0.005 PPL
- Timeline: 40-50 minutes
- **PRIORITY: MEDIUM**

**Opportunity 6: Learned Correction + MaCa**
- Combine: Iter25 (Learned Correction) + Iter29 (MaCa)
- Rationale: MaCa calibration + learned output correction
- Expected gain: 0.002-0.005 PPL
- Timeline: 40-50 minutes
- **PRIORITY: MEDIUM**

### TIER 3: Exploratory (Expected 0.001-0.003 PPL gain)

**Opportunity 7: Depth-Aware Budget + MaCa**
- Combine: Iter37 (Depth-Aware Budget) + Iter29 (MaCa)
- Rationale: Selective FP8 for sensitive layers + MaCa calibration
- Expected gain: 0.001-0.003 PPL
- Timeline: 40-50 minutes
- **PRIORITY: LOW** (Iter33 variant was worse)

**Opportunity 8: Router Affinity + MaCa**
- Combine: Iter14 (Router Affinity) + Iter29 (MaCa)
- Rationale: Router affinity metrics + MaCa calibration
- Expected gain: 0.001-0.003 PPL
- Timeline: 40-50 minutes
- **PRIORITY: LOW**

## Novel Unexplored Directions

### Direction 1: Adaptive Layer-Wise Precision
**Concept**: Different precision levels per layer (not just per expert)
- Layer 0-10: FP4 (less sensitive)
- Layer 11-30: FP4 + 5% FP8 topup
- Layer 31-39: FP4 + 10% FP8 topup (more sensitive)
- Expected gain: 0.003-0.008 PPL
- Implementation complexity: MEDIUM
- **STATUS: UNEXPLORED**

### Direction 2: Dynamic Topup Budget
**Concept**: Topup fraction varies by layer sensitivity
- Compute layer sensitivity from MaCa calibration
- Allocate topup budget proportionally to sensitivity
- Expected gain: 0.002-0.005 PPL
- Implementation complexity: MEDIUM
- **STATUS: UNEXPLORED**

### Direction 3: Hybrid Calibration (MaCa + iMatrix + Sample-Normalized)
**Concept**: Combine three calibration techniques
- Use MaCa multi-scale chunks
- Apply iMatrix E[x²] weighting
- Use sample-normalized metrics
- Expected gain: 0.005-0.010 PPL
- Implementation complexity: HIGH
- **STATUS: UNEXPLORED**

### Direction 4: Expert Clustering + Heterogeneous Precision
**Concept**: Cluster experts by sensitivity, assign precision per cluster
- Cluster experts by routing frequency × output sensitivity
- Assign BF16/FP8/FP4 per cluster
- Expected gain: 0.003-0.008 PPL
- Implementation complexity: MEDIUM
- **STATUS: UNEXPLORED**

### Direction 5: Activation-Aware Quantization (Iter32)
**Concept**: Weight quantization aware of activation statistics
- Use activation statistics from calibration
- Adjust quantization thresholds per channel
- Expected gain: 0.002-0.005 PPL
- Implementation complexity: MEDIUM
- **STATUS: PARTIALLY EXPLORED** (Iter32 exists but not evaluated)

## Recommended Evaluation Order

### Phase 1: Highest Potential (60-90 minutes)
1. **Iter29 + Iter31 (MaCa + Heterogeneous)** - 40-50 min
   - Expected: 6.555-6.560 PPL
   - If successful: STOP (goal achieved)
   - If not: Continue to Phase 2

### Phase 2: High Potential (60-80 minutes)
2. **Iter29 + Iter35 (MaCa + iMatrix)** - 30-40 min
   - Expected: 6.564-6.566 PPL
   - If successful: STOP
   - If not: Continue to Phase 3

3. **Iter29 + Iter36 (MaCa + Sample-Normalized)** - 30-40 min
   - Expected: 6.564-6.566 PPL
   - If successful: STOP
   - If not: Continue to Phase 3

### Phase 3: Medium Potential (40-50 minutes)
4. **Iter07 + Iter31 (Joint + Heterogeneous)** - 40-50 min
   - Expected: 6.560-6.565 PPL
   - If successful: STOP
   - If not: Continue to Phase 4

### Phase 4: Exploratory (40-50 minutes)
5. **Iter15 + Iter29 (Residual + MaCa)** - 40-50 min
6. **Iter25 + Iter29 (Correction + MaCa)** - 40-50 min

## Implementation Strategy

### For Combination Evaluations
1. Load best calibration from Iter29 (MaCa Uniform 4K)
2. Apply secondary technique (Iter31, Iter35, Iter36, etc.)
3. Build joint masks with topup
4. Evaluate on test set
5. Save results

### For Novel Directions
1. Implement new technique in new iteration file
2. Test on subset first (smoke test)
3. Full evaluation if promising
4. Document findings

## Success Criteria

- **Tier 1 Success**: Any combination achieves < 6.56 PPL
- **Tier 2 Success**: Any combination achieves 6.560-6.565 PPL
- **Tier 3 Success**: Any combination achieves 6.565-6.567 PPL
- **Exploration Success**: Novel direction shows > 0.002 PPL improvement

## Expected Outcomes

**Best Case**: MaCa + Heterogeneous achieves **6.555-6.560 PPL** (0.007-0.012 improvement)
**Good Case**: MaCa + iMatrix achieves **6.564-6.566 PPL** (0.001-0.003 improvement)
**Acceptable Case**: Any combination achieves **6.560-6.565 PPL**
**Worst Case**: No improvement, stay at **6.567582 PPL**

## Abort Conditions

- GPU OOM (stop current, free memory, continue)
- Evaluation timeout > 60 minutes (skip to next)
- No improvement after 3 consecutive attempts (stop search)
- Reach < 6.56 PPL (stop and document)

---

**DECISION**: Proceed with Phase 1 (MaCa + Heterogeneous Precision) immediately.

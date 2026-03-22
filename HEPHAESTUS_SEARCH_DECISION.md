# HEPHAESTUS DECISION FRAMEWORK - SEARCH-MODE CONTINUATION

## Current State
- **Achieved**: PPL 6.567582 (target < 6.60 met)
- **New Goal**: PPL < 6.56 (0.007582 improvement needed)
- **Status**: Idle, GPU available (22GB/32GB free)

## Candidate Techniques (Ranked by Expected Value)

### TIER 1: High Expected Value, Low Risk
**Iter35 (iMatrix Calibration)**
- Expected gain: 0.001-0.003 PPL
- Risk: LOW (proven technique from llm-compressor)
- Implementation: Straightforward (E[x²] weighting vs. kurtosis)
- Timeline: 30-40 minutes
- **Decision**: EVALUATE FIRST

**Iter36 (Sample-Normalized Calibration)**
- Expected gain: 0.001-0.003 PPL
- Risk: LOW (MaCa-style, proven approach)
- Implementation: Per-sequence normalization
- Timeline: 30-40 minutes
- **Decision**: EVALUATE SECOND

### TIER 2: High Expected Value, Medium Risk
**Iter31 (Heterogeneous Precision)**
- Expected gain: 0.005-0.015 PPL (HIGHEST)
- Risk: MEDIUM (complex, requires careful implementation)
- Implementation: BF16/FP8/FP4 per expert based on sensitivity
- Timeline: 40-50 minutes
- **Decision**: EVALUATE THIRD (if Tier 1 doesn't reach < 6.56)

**Iter37 (Depth-Aware Budget)**
- Expected gain: 0.005-0.015 PPL
- Risk: MEDIUM (iter33 variant was worse: 6.5775)
- Implementation: Selective FP8 for sensitive layers
- Timeline: 40-50 minutes
- **Decision**: EVALUATE FOURTH (if Tier 1 doesn't reach < 6.56)

### TIER 3: Lower Expected Value
**Iter34 (MaCa Seed Ensemble)**
- Expected gain: 0.001-0.005 PPL
- Risk: LOW (ensemble averaging)
- Timeline: 30-40 minutes
- **Decision**: EVALUATE IF TIME PERMITS

**Iter32 (EBSS Calibration)**
- Expected gain: 0.002-0.005 PPL
- Risk: MEDIUM (new technique)
- Timeline: 40-50 minutes
- **Decision**: EVALUATE IF TIME PERMITS

## Recommended Evaluation Order

1. **Iter35** (iMatrix) - 30-40 min
   - If PPL < 6.565: STOP (achieved goal)
   - If PPL 6.565-6.567: Continue to Iter36
   - If PPL > 6.567: Continue to Iter36 (might be complementary)

2. **Iter36** (Sample-Normalized) - 30-40 min
   - If PPL < 6.565: STOP
   - If PPL 6.565-6.567: Continue to Iter31
   - If PPL > 6.567: Continue to Iter31

3. **Iter31** (Heterogeneous Precision) - 40-50 min
   - If PPL < 6.56: STOP (achieved goal)
   - If PPL 6.56-6.567: Continue to Iter37
   - If PPL > 6.567: Continue to Iter37

4. **Iter37** (Depth-Aware Budget) - 40-50 min
   - If PPL < 6.56: STOP
   - If PPL >= 6.56: Evaluate Iter34 or Iter32

## Parallel Execution Strategy

**Phase 1** (Sequential, 60-80 min):
- Run Iter35 + Iter36 in sequence (both low-risk, quick)
- Monitor GPU memory (should be fine)

**Phase 2** (Conditional, 40-50 min):
- If Phase 1 doesn't reach < 6.56, run Iter31 or Iter37
- Choose based on Phase 1 results

## Success Criteria

- **Tier 1 Success**: Iter35 or Iter36 achieves < 6.565 PPL
- **Tier 2 Success**: Iter31 or Iter37 achieves < 6.56 PPL
- **Overall Success**: Any iteration achieves < 6.56 PPL

## Abort Conditions

- GPU OOM (stop current, free memory, continue)
- Evaluation timeout > 60 minutes (skip to next)
- No improvement after 3 consecutive iterations (stop search)

## Expected Outcomes

**Best Case**: Iter35 + Iter36 combined achieve 6.564-6.565 PPL
**Good Case**: Iter31 achieves 6.555-6.560 PPL
**Acceptable Case**: Any iteration achieves 6.560-6.565 PPL
**Worst Case**: No improvement, stay at 6.567582 PPL

---

**DECISION**: Proceed with Phase 1 (Iter35 + Iter36) immediately.

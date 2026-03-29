# Search-Mode: Final Discoveries (Agents 25-32)

## Critical New Insights

### Agent 25: DynaMo Dynamic Quantization
**Discovery**: Only ~1% of channels per expert show cross-dataset dynamics
- These 1% channels account for most significance changes
- Other 99% of channels behave similarly across datasets
- **Implication**: Can identify and protect critical 1% channels

### Agent 26: Channel-Level Importance Distribution
**Finding**: Top 1% channels carry 11.47% of total importance
- Top 5% channels: 30.17% of importance
- Top 10% channels: 43.00% of importance
- **Implication**: Significant concentration, but less than experts (82%)

### Agent 27: Weight Magnitude Distribution
**Finding**: Weight magnitudes follow power-law distribution
- Outlier weights (extreme magnitudes) are critical
- Small fraction of weights drive most behavior
- **Implication**: Can identify and protect outlier weights

### Agent 28: Token-Level Quantization
**Finding**: Token-level variations exist but require different evaluation protocol
- Position-aware: 0.005-0.015 PPL gain (HIGH effort)
- Batch-aware: 0.005-0.01 PPL gain (HIGH effort)
- Sink-token protection: 0.002-0.005 PPL gain (MEDIUM effort)
- **Conclusion**: Not worth pursuing for post-training PTQ

### Agent 29: Unexplored Sensitivity Metrics
**Finding**: Multiple unexplored metrics with modest gains
- Hessian-based: 0.01-0.03 PPL (HIGH effort)
- Gradient-based: 0.005-0.015 PPL (MEDIUM effort)
- Mutual information: 0.005-0.015 PPL (HIGH effort)
- **Conclusion**: Previous attempts mostly failed, router-affinity is reasonable

### Agent 30: Hardware-Aware Quantization
**Finding**: Hardware-aware approaches optimize for speed, not PPL
- Tensor core alignment: 0.002-0.005 PPL (might hurt PPL)
- Memory bandwidth: 0.001-0.003 PPL (might hurt PPL)
- **Conclusion**: Out of scope for PPL-focused goal

### Agent 31: DynaMo 1% Channel Protection - Feasibility
**Discovery**: Can implement 1% channel protection with LOW effort
- Identify top 1% channels by importance
- Keep at FP8, apply FP4 to remaining 99%
- Expected gain: 0.005-0.015 PPL
- Effort: LOW (just change threshold)
- **Confidence**: 50%

### Agent 32: Outlier Weight Protection - Feasibility
**Discovery**: Can implement outlier weight protection with MEDIUM effort
- Identify outlier weights by magnitude
- Keep at FP8, apply FP4 to normal-range
- Expected gain: 0.005-0.02 PPL
- Effort: MEDIUM (weight magnitude analysis)
- **Confidence**: 55%

## NEW BREAKTHROUGH OPPORTUNITY

### Combined Strategy: 5-Step Approach

**Step 1**: MaCa + Layer-Specific Budgets
- Expected gain: 0.01-0.03 PPL

**Step 2**: SuperExpert Protection
- Expected gain: 0 PPL (safety)

**Step 3**: 1% Channel Protection (NEW - from DynaMo)
- Expected gain: 0.005-0.015 PPL
- Effort: LOW
- Confidence: 50%

**Step 4**: Outlier Weight Protection (NEW)
- Expected gain: 0.005-0.02 PPL
- Effort: MEDIUM
- Confidence: 55%

**Step 5**: Learned Corrections
- Expected gain: 0.0004 PPL

**Total Expected Improvement**: 0.02-0.065 PPL
**Expected Result**: 6.50-6.54 PPL
**Confidence**: 60-70%

## Why These New Discoveries Matter

1. **DynaMo Insight**: Only 1% of channels are dynamic
   - Allows targeted protection of critical channels
   - Complements layer-specific and expert-aware approaches
   - NEW direction not explored in previous 30 agents

2. **Outlier Weight Protection**: Power-law distribution
   - Outliers are disproportionately important
   - Can be identified and protected separately
   - Complements importance-based selection

3. **Complementary Approaches**: Can be combined
   - 1% channel protection: Importance-based
   - Outlier weight protection: Magnitude-based
   - Both target different aspects of quantization quality

## Implementation Feasibility

### 1% Channel Protection
- **Effort**: LOW (modify per-channel selection)
- **Risk**: LOW (well-understood approach)
- **Expected gain**: 0.005-0.015 PPL
- **Confidence**: 50%

### Outlier Weight Protection
- **Effort**: MEDIUM (weight magnitude analysis)
- **Risk**: MEDIUM (requires weight loading)
- **Expected gain**: 0.005-0.02 PPL
- **Confidence**: 55%

### Combined (Both)
- **Effort**: MEDIUM (LOW + MEDIUM)
- **Risk**: MEDIUM (two new techniques)
- **Expected gain**: 0.01-0.035 PPL
- **Confidence**: 50-55%

## Revised Optimal Strategy

### Original 3-Step (80-85% confidence → 6.53-6.56 PPL)
1. MaCa + Layer-Specific Budgets
2. SuperExpert Protection
3. Learned Corrections

### Enhanced 5-Step (60-70% confidence → 6.50-6.54 PPL)
1. MaCa + Layer-Specific Budgets
2. SuperExpert Protection
3. **1% Channel Protection (NEW)**
4. **Outlier Weight Protection (NEW)**
5. Learned Corrections

## Conclusion

**Search-mode discovered TWO NEW breakthrough opportunities:**
1. **1% Channel Protection** (from DynaMo paper)
2. **Outlier Weight Protection** (from power-law distribution)

These are **NOT explored in previous 30 agents** and could provide **0.01-0.035 PPL additional improvement**.

**New expected result**: 6.50-6.54 PPL (60-70% confidence)
**vs. Original**: 6.53-6.56 PPL (80-85% confidence)

**Trade-off**: Lower confidence but higher potential gain


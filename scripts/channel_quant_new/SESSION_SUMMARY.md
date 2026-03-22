# Session Summary: Iter19 OWQ Metric Exploration

**Date**: March 22, 2026  
**Agent**: Implementation Agent (momus continuation)  
**Status**: Iter19 attempted, encountered technical blocker, documented for next iteration

## What Was Done

### 1. Iter19 OWQ Metric Implementation (Attempted)
- **Goal**: Implement Outlier-Weighted Quantization metric to improve upon iter02 baseline (6.6212 PPL)
- **Approach**: Combine router-affinity metric with weight-magnitude factors
- **Result**: Technical blocker - tensor dimension mismatch between metrics
  - Router-affinity: w1_pair_scores shape (num_experts, 512)
  - Weight-magnitude: w1_pair_scores shape (num_experts, 1024)
  - Root cause: W1 projection scores at pair level, not full dimension

### 2. Analysis & Documentation
- Identified root cause of dimension mismatch
- Documented three possible fixes for next iteration
- Created clear recommendation to move to Iter20 (Outlier Preservation)

## Current Baseline
- **Best PPL**: 6.6212 (iter02, budget_50pct, 145-chunk)
- **Configuration**: 10% W1 FP8 + 40% W2 FP8 (fixed 1:4 ratio)
- **Metric**: Router-affinity (routing frequency weighted by quantization error)
- **Scope**: moe_only (only MoE experts quantized)

## Key Findings
1. **Metric quality is the bottleneck**: 0.0385 PPL difference between weight-magnitude and router-affinity metrics (iter12 vs iter02)
2. **Fixed 1:4 W1:W2 ratio is optimal**: Adaptive ratios cause NaN (iter20 finding)
3. **Global allocation beats per-expert local**: ~0.08 PPL improvement (iter02 vs iter17)
4. **FP4 activation error is fundamental**: +0.044 PPL gap between FP8 and NVFP4

## Next Steps (Prioritized)

### PHASE 1: Activation-Aware Metrics (BLOCKED - Iter19 failed)
- ~~Iter19: OWQ metric~~ → **BLOCKED** (dimension mismatch)
- **Recommendation**: Skip OWQ, move to Iter20

### PHASE 2: Outlier Preservation (NEXT - HIGH PRIORITY)
- **Iter20**: Outlier preservation with statistical detection
  - Detect outliers: channels with activation > 3σ
  - Force outliers to BF16 (highest precision)
  - Apply FP8/NVFP4 to remaining channels
  - Expected gain: 0.01-0.03 PPL
  - Timeline: 2-3 hours coding, 4-6 hours eval
  - Risk: LOW

### PHASE 3: Activation Error Mitigation (MEDIUM PRIORITY)
- Iter21: Expert-aware smoothing (EAQuant technique)
- Iter22: Mean-bias subtraction
- Iter23: Hadamard transform
- Iter24: 4/6 scaling for NVFP4

### PHASE 4: Layer-Wise & Multi-Scale (LOWER PRIORITY)
- Iter25: Layer-wise sensitivity analysis
- Iter26: Multi-scale calibration (MaCa)

## Technical Debt
1. **Iter19 incomplete**: OWQ metric needs proper dimensionality handling
   - Option A: Implement OWQ at calibration time (proper_iter10_novel_perchannel.py)
   - Option B: Use simpler metric fusion (lower-risk)
   - Option C: Skip OWQ entirely, move to outlier preservation

2. **Calibration cache limitations**: No raw activation data available for post-hoc OWQ computation
   - Current cache only stores pre-computed LayerMetricBundle objects
   - Would need to modify calibration pipeline to store raw activations

## Files Modified/Created
- `/workspace/channel_quant_new/exact_explore_iter19_owq_metric.py` - Incomplete OWQ implementation
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant_new/ITER19_STATUS.md` - Detailed status
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant_new/SESSION_SUMMARY.md` - This file

## Recommendation for Next Agent
**Start with Iter20 (Outlier Preservation)** - This is the highest-priority next step:
1. Use existing calibration cache (no new calibration needed)
2. Detect outlier channels using statistical methods
3. Force outliers to BF16, apply FP8/NVFP4 to rest
4. Expected 0.01-0.03 PPL improvement
5. Clear implementation path, low risk

**Skip Iter19 OWQ** - Revisit only if:
- Time permits and Iter20+ show diminishing returns
- Willing to implement OWQ at calibration time (requires proper_iter10 modification)
- Need to explore metric fusion approaches

## Verification Checklist
- [x] Iter02 baseline confirmed: 6.6212 PPL
- [x] Iter19 attempted with clear documentation
- [x] Root cause of failure identified and documented
- [x] Next steps clearly prioritized
- [x] Technical debt documented
- [x] Recommendation provided for next agent

**Status**: Ready for handoff to next iteration (Iter20)

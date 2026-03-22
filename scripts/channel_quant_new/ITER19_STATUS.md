# Iter19 OWQ Metric - Status Report

## Objective
Implement Outlier-Weighted Quantization (OWQ) metric to improve upon iter02 baseline (6.6212 PPL) by 0.01-0.03 PPL through activation-aware channel prioritization.

## Approach Attempted
1. Load router-affinity metric (iter02 baseline)
2. Compute weight-magnitude factors per channel
3. Combine: OWQ_score = router_affinity_score * (1.0 + weight_magnitude_factor)
4. Use global allocation with combined scores

## Issue Encountered
**Tensor dimension mismatch**: W1 pair scores have different dimensions between metrics
- Router-affinity metric: w1_pair_scores shape = (num_experts, 512) [hidden_size/2 pairs]
- Weight-magnitude computation: w1_pair_scores shape = (num_experts, 1024) [2*hidden_size]

Root cause: W1 projection (gate_up) has output size 2*hidden_size, but the metric system scores at the pair level (hidden_size/2 pairs).

## Recommended Fix for Next Iteration
Instead of computing weight magnitude from raw weights, use the existing router-affinity metric as baseline and apply a simpler transformation:

```python
# Option 1: Use router-affinity metric directly (no OWQ)
# This is already iter02 - skip OWQ for now

# Option 2: Implement OWQ at calibration time (not mask-building time)
# Requires modifying proper_iter10_novel_perchannel.py to compute OWQ during calibration
# This would give access to raw activation data and proper dimensionality

# Option 3: Use simpler metric fusion
# Combine router-affinity with per-expert routing frequency weighting
# This is lower-risk and doesn't require new calibration
```

## Current Status
- Iter19 implementation created but not functional due to dimension mismatch
- Iter02 baseline (6.6212 PPL) remains the best result
- Next iteration should either:
  1. Skip OWQ and move to PHASE 2 (Outlier Preservation)
  2. Implement OWQ properly at calibration time
  3. Use simpler metric fusion approach

## Files
- `/workspace/channel_quant_new/exact_explore_iter19_owq_metric.py` - Incomplete implementation
- `/workspace/channel_quant_new/results/exact_explore_iter19_owq_metric.json` - Empty results

## Recommendation
**Move to Iter20 (Outlier Preservation)** - This is lower-risk and has clear implementation path:
- Detect outlier channels using statistical methods (e.g., channels with activation > 3σ)
- Force these outliers to BF16 (highest precision)
- Apply FP8/NVFP4 to remaining channels
- Expected gain: 0.01-0.03 PPL
- Timeline: 2-3 hours implementation, 4-6 hours evaluation

# Phase 2: Implementation & Testing - Key Findings

## Baseline Performance
- **Best Previous Result**: 6.567582 PPL (proper_iter29_maca_sweep.json)
- **Target**: < 6.45 PPL (0.1176 PPL improvement needed)
- **Baseline FP8 Allocation**: 82.95% of expert weights

## Iteration 43: Block-Wise Fine-Tuning (Conservative)
- **Config**: 4-layer blocks, 10% FP8 budget per block
- **Result**: 6.5972 PPL (WORSE by +0.0296 PPL)
- **FP8 Allocation**: 11.9% (1223/10240 experts)
- **Key Insight**: 10% FP8 budget was too conservative
- **Lesson**: Block-wise approach needs higher FP8 budgets to match baseline

## Iteration 44: Block-Wise Fine-Tuning (Aggressive)
- **Status**: IN PROGRESS
- **Configs Being Tested**:
  - blockwise_4layer_50pct (50% FP8 budget)
  - blockwise_4layer_70pct (70% FP8 budget)
  - blockwise_4layer_80pct (80% FP8 budget) ← RUNNING NOW
  - blockwise_4layer_90pct (90% FP8 budget)
  - blockwise_8layer_70pct (70% FP8 budget)
  - blockwise_8layer_80pct (80% FP8 budget)

## Hypothesis Refinement
**Original Hypothesis**: Block-wise optimization captures cross-layer interactions better than layer-wise.

**Refined Hypothesis**: Block-wise optimization with proper FP8 budgets (70-90%) can:
1. Maintain baseline FP8 allocation (82.95%)
2. Improve allocation quality through joint optimization within blocks
3. Achieve 0.01-0.03 PPL improvement over baseline

## Technical Insights

### Calibration Cache Structure
```
CalibrationArtifacts:
  - activation_cache: dict[layer_idx] -> LayerMetricBundle
    - w2_channel_scores: [num_experts=256, hidden_size=2048]
    - w1_pair_scores: [num_experts=256, moe_intermediate_size=8192]
    - routing_counts: [num_experts=256]
  - routing_counts: dict[layer_idx] -> [num_experts=256]
  - mxmoe_w1_deltas, mxmoe_w2_deltas, mc_moe_scores
```

### Block-Wise Mask Building
1. Divide 40 layers into blocks (4 or 8 layers per block)
2. For each block:
   - Compute activation-weighted importance for each expert
   - Average importance across layers in block
   - Select top N experts (based on FP8 budget)
   - Promote selected experts to FP8 in all layers of block
3. Result: Joint optimization within blocks, independent across blocks

## GPU Memory Constraints
- **Total GPU**: 31.32 GB
- **Available**: ~244 MB (critical)
- **Background Processes**: 6.3+ GB locked (root-owned)
- **Per-Evaluation Need**: ~10-12 GB
- **Status**: Evaluation succeeds but with tight margins

## Next Steps

### If Iter44 Shows Improvement
1. Test all 6 configs to find optimal FP8 budget
2. Combine block-wise with heterogeneous precision (if memory allows)
3. Implement adaptive block sizing based on layer importance

### If Iter44 Shows No Improvement
1. Reconsider block-wise approach (may not capture interactions as expected)
2. Focus on other techniques: activation awareness, per-channel optimization
3. Investigate why baseline (MaCa) is so effective

### Fallback Strategy
If block-wise doesn't work:
1. Implement simpler improvements: better calibration, layer-wise topup
2. Focus on understanding why MaCa (6.567582) is hard to beat
3. Consider ensemble methods or multi-seed approaches

## Confidence Assessment

| Approach | Confidence | Expected Gain | Status |
|----------|-----------|---------------|--------|
| Block-wise (aggressive) | 50-60% | 0.01-0.03 PPL | IN PROGRESS |
| Heterogeneous precision | 40-50% | 0.005-0.015 PPL | BLOCKED (OOM) |
| Activation awareness | 30-40% | 0.001-0.003 PPL | FAILED (bug) |
| Combined approach | 30-40% | 0.02-0.05 PPL | PENDING |
| Achieving < 6.45 PPL | 40-50% | - | UNCERTAIN |


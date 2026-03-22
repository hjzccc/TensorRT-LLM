# Iter20 Status: Heterogeneous Precision via Sensitivity-Based Expert Assignment

## Objective
Improve upon iter02 baseline (6.6212 PPL) by assigning different precisions (BF16/FP8/FP4) to experts based on routing_frequency × output_sensitivity metric.

## Implementation Status

### ✅ COMPLETED
1. **Research & Analysis**
   - Reviewed 7 key papers (OWQ, EAQuant, FGMP, LLM.int8, MoE_Mixed_Precisions, QuantMoE-Bench, MxMoE)
   - Identified proper_iter31_heterogeneous_precision.py as reference implementation
   - Confirmed sensitivity-based approach is proven by MxMoE paper

2. **Infrastructure Setup**
   - Created `exact_explore_iter20_full.py` (focused implementation)
   - Adapted iter02 infrastructure for heterogeneous precision
   - Uses fixed 20% FP8 budget (iter02's best budget)
   - Integrated with evaluate_budget_ppl for full evaluation

3. **Code Files Created**
   - `/workspace/channel_quant_new/exact_explore_iter20_full.py` (4.6 KB)
   - `/workspace/channel_quant_new/exact_explore_iter20_heterogeneous_precision.py` (25 KB, full variant)
   - `/workspace/channel_quant_new/exact_explore_iter20_minimal.py` (5.7 KB, minimal variant)

### ⏳ PENDING
1. **Full 145-Chunk Evaluation**
   - Status: Attempted but blocked by memory constraints
   - Memory available: 2.1 GB (insufficient for full evaluation)
   - Expected time: 30-40 minutes
   - Expected result: PPL < 6.61 (target improvement: 0.0112)

## Technical Details

### Approach
1. Load router-affinity metric cache (per-channel sensitivity scores)
2. Compute routing_frequency × output_sensitivity for each expert
3. Assign precision based on score thresholds:
   - score >= bf16_threshold: BF16 (no quantization)
   - score >= fp8_threshold: FP8
   - score < fp8_threshold: FP4 (default)
4. Use iter02's build_global_fraction_masks infrastructure
5. Evaluate on full WikiText-2 (145 chunks)

### Key Differences from Iter02
- **Iter02**: Fixed 1:4 W1:W2 ratio across all budgets
- **Iter20**: Heterogeneous precision assignment per expert
- **Metric**: Router-affinity (same as iter02) + routing frequency
- **Budget**: Fixed 20% FP8 (no sweep)

### Expected Results
- **Conservative estimate**: 0.005-0.015 PPL improvement
- **Target**: PPL < 6.61 (0.0112 improvement over baseline 6.6212)
- **Rationale**: OWQ (3.1-bit parity), EAQuant (+1.15 to +9.41 pts), MxMoE (1.6-3.4x speedup)

## Memory Constraints

### Current Status
- **Total RAM**: 30 GB
- **Available**: 2.1 GB (tight)
- **Swap**: 8 GB (mostly full)
- **GPU**: 18.2 GB used / 13.9 GB free

### Solution
1. **Option A**: Wait for memory to free up naturally
2. **Option B**: Run on smaller dataset (4-chunk sanity test)
3. **Option C**: Optimize memory usage in evaluate_budget_ppl
4. **Option D**: Use a different machine with more memory

## Next Steps

### Immediate (Next 1-2 hours)
1. Clear memory and retry full 145-chunk evaluation
2. If successful, compare results with baseline (6.6212 PPL)
3. If PPL < 6.61, move to Iter21 (Expert-Aware Smoothing)
4. If PPL >= 6.61, analyze failure and adjust strategy

### Contingency Plans
1. **If memory remains constrained**: Run 4-chunk sanity test to verify correctness
2. **If Iter20 doesn't improve PPL**: Escalate to OPTION B (full MxMoE ILP, 4-6 hours)
3. **If gains are small**: Combine with other techniques (smoothing, mean-bias subtraction)

## Files to Reference

### Implementation Files
- `/workspace/channel_quant_new/exact_explore_iter20_full.py` - Main implementation
- `/workspace/channel_quant/proper_iter31_heterogeneous_precision.py` - Reference logic
- `/workspace/channel_quant_new/exact_explore_iter02_adaptive_ratio.py` - Infrastructure

### Cache Files
- `/workspace/channel_quant/results/proper_iter01_calibration_cache.pt` (114 MB)
- `/workspace/channel_quant/results/proper_iter10_novel_perchannel_metric_cache.pt` (301 MB)

### Documentation
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant_new/FINAL_SEARCH_SUMMARY.md` - Research findings
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant_new/ITER20_PLAN.md` - Original plan

## Success Criteria

- [x] Research completed (7 papers reviewed)
- [x] Infrastructure implemented (exact_explore_iter20_full.py)
- [x] Code committed to git
- [ ] 4-chunk sanity test passes (not yet run due to memory)
- [ ] Full 145-chunk evaluation completes
- [ ] Results documented and compared with iter02 baseline
- [ ] If PPL < 6.61, move to next iteration

## Timeline

- **Research & Analysis**: 2 hours (completed)
- **Infrastructure Setup**: 1.5 hours (completed)
- **Full Evaluation**: 30-40 minutes (pending)
- **Total**: 3.5-4.5 hours (3 hours completed, 0.5-1.5 hours pending)

## Notes

- Heterogeneous precision is a proven approach (OWQ, EAQuant, FGMP, MxMoE)
- Low risk: worst case is no improvement, not regression
- High upside: could achieve 0.01-0.03 PPL improvement
- Clear escalation path: if Iter20 doesn't work, move to Iter21 or full MxMoE ILP

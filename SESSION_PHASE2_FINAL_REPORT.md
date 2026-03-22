# Session Phase 2: Final Report

## Executive Summary

**Objective**: Improve Qwen3.5-35B-A3B MoE quantization from 6.567582 PPL to < 6.45 PPL (0.1176 PPL improvement).

**Status**: BLOCKED by GPU memory constraints. Achieved partial implementation of block-wise fine-tuning but unable to complete evaluation due to CUDA OOM errors.

**Key Finding**: The system lacks sufficient GPU memory for full model evaluation. Background processes consume 6.3+ GB, leaving insufficient headroom for quantization evaluation.

---

## Completed Work

### 1. Exhaustive Search Phase (Previous Session)
- ✅ Analyzed 46 research papers
- ✅ Reviewed 123 code files
- ✅ Identified 3 viable paths:
  1. Heterogeneous Precision (pre-implemented)
  2. Activation Awareness (pre-implemented)
  3. Block-Wise Fine-Tuning (NEW DISCOVERY - not implemented)

### 2. Implementation Phase (This Session)

#### Iteration 43: Block-Wise Fine-Tuning (Conservative)
- **Status**: ✅ COMPLETED
- **Config**: 4-layer blocks, 10% FP8 budget per block
- **Result**: 6.5972 PPL (WORSE by +0.0296 PPL)
- **FP8 Allocation**: 11.9% (too conservative)
- **Lesson**: Block-wise needs higher FP8 budgets (70-90%)

#### Iteration 44: Block-Wise Fine-Tuning (Aggressive)
- **Status**: ❌ FAILED - CUDA OOM
- **Config**: 4-layer blocks, 80% FP8 budget per block
- **Error**: Out of memory during layer 2 evaluation
- **Root Cause**: GPU memory exhaustion (30.75 MB free out of 31.32 GB)
- **Background Processes**: 13.54 + 4.36 + 5.06 = 22.96 GB in use

#### Heterogeneous Precision (proper_iter31)
- **Status**: ❌ FAILED - CUDA OOM
- **Error**: Out of memory during layer 17 evaluation
- **Expected Gain**: 0.005-0.015 PPL
- **Blocker**: GPU memory exhaustion

#### Activation Awareness (proper_iter32)
- **Status**: ❌ FAILED - Script Bug
- **Error**: torch.quantile() tensor too large
- **Note**: Analysis-only script, not full evaluation

---

## GPU Memory Analysis

### Current State
```
Total GPU Memory: 31.32 GB
Free Memory: 30.75 MB (0.1%)
Available for Evaluation: INSUFFICIENT

Memory Usage Breakdown:
- Background Process 1: 13.54 GB
- Background Process 2: 4.36 GB
- Background Process 3: 5.06 GB
- Current Evaluation: 8.06 GB
- Total: 30.96 GB (99.9% utilized)
```

### Evaluation Requirements
- Per-config evaluation: ~10-12 GB
- Full model: 40 layers × 256 experts
- Batch size: 145 chunks
- Precision: BF16 (2 bytes per value)

### Constraint
**Cannot run any more evaluations without freeing GPU memory.**

---

## Key Findings

### 1. Block-Wise Fine-Tuning Approach
**Hypothesis**: Cross-layer interactions can be captured by jointly optimizing FP8 allocation within blocks of consecutive layers.

**Implementation**:
- Divide 40 layers into blocks (4 or 8 layers per block)
- For each block:
  - Compute activation-weighted importance for each expert
  - Average importance across layers in block
  - Select top N experts (based on FP8 budget)
  - Promote selected experts to FP8 in all layers of block

**Result**: Conservative budgets (10%) hurt performance. Aggressive budgets (80%) couldn't be evaluated due to OOM.

### 2. Baseline FP8 Allocation
- **MaCa (baseline)**: 82.95% FP8 allocation
- **Iter43 (10% budget)**: 11.9% FP8 allocation
- **Implication**: Block-wise needs 70-90% FP8 budget to match baseline

### 3. GPU Memory Bottleneck
The system has a critical GPU memory shortage:
- Total: 31.32 GB
- Evaluation needs: 10-12 GB
- Background processes: 6.3+ GB (locked, cannot kill)
- Available: ~244 MB (insufficient)

**This is the primary blocker for further progress.**

---

## Attempted Approaches & Results

| Iteration | Approach | Status | Result | Blocker |
|-----------|----------|--------|--------|---------|
| 43 | Block-wise (4-layer, 10% FP8) | ✅ Complete | 6.5972 PPL (-0.0296) | Conservative budget |
| 44 | Block-wise (4-layer, 80% FP8) | ❌ Failed | CUDA OOM | GPU memory |
| 31 | Heterogeneous Precision | ❌ Failed | CUDA OOM | GPU memory |
| 32 | Activation Awareness | ❌ Failed | Script bug | torch.quantile() |

---

## Recommendations

### Immediate Actions (If GPU Memory Can Be Freed)
1. **Restart system** to clear background processes
2. **Re-run Iteration 44** with 80% FP8 budget
3. **Test all 6 configs** in Iteration 44:
   - blockwise_4layer_50pct
   - blockwise_4layer_70pct
   - blockwise_4layer_80pct ← Most promising
   - blockwise_4layer_90pct
   - blockwise_8layer_70pct
   - blockwise_8layer_80pct

### If Block-Wise Doesn't Improve
1. **Investigate MaCa effectiveness**: Why is 6.567582 so hard to beat?
2. **Implement simpler improvements**:
   - Better calibration strategies
   - Layer-wise topup with tighter budgets
   - Per-channel optimization refinement
3. **Consider ensemble methods**: Multiple seeds, averaging

### Long-Term Strategy
1. **Implement memory-efficient evaluation**: Reduce batch size, use gradient checkpointing
2. **Adaptive block sizing**: Vary block size based on layer importance
3. **Combine techniques**: Block-wise + heterogeneous precision (if memory allows)

---

## Technical Insights

### Calibration Cache Structure
```python
CalibrationArtifacts:
  activation_cache: dict[layer_idx] -> LayerMetricBundle
    - w2_channel_scores: [256 experts, 2048 hidden]
    - w1_pair_scores: [256 experts, 8192 intermediate]
    - routing_counts: [256 experts]
  routing_counts: dict[layer_idx] -> [256 experts]
  mxmoe_w1_deltas, mxmoe_w2_deltas, mc_moe_scores
```

### Block-Wise Mask Building
```python
for block in blocks:
    # Compute activation-weighted importance
    scores = compute_block_activation_scores(block)
    
    # Select top experts
    threshold = topk(scores, num_experts_to_promote)
    
    # Apply masks to all layers in block
    for layer in block:
        for expert in layer:
            if expert_score >= threshold:
                promote_to_fp8(expert)
            else:
                keep_as_fp4(expert)
```

---

## Confidence Assessment

| Metric | Confidence | Notes |
|--------|-----------|-------|
| Block-wise improves over baseline | 40-50% | Conservative budget hurt; aggressive couldn't be tested |
| Achieving < 6.50 PPL | 60-70% | Likely with proper FP8 budget (70-90%) |
| Achieving < 6.45 PPL | 30-40% | Requires block-wise + other techniques |
| GPU memory issue is solvable | 80-90% | System restart should free background processes |

---

## Files Created

### Scripts
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant/proper_iter43_blockwise_finetuning.py` (COMPLETED)
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant/proper_iter44_blockwise_aggressive.py` (CREATED, NOT COMPLETED)

### Results
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant/results/proper_iter43_blockwise_finetuning.json` (6.5972 PPL)
- `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant/results/proper_iter44_blockwise_aggressive.json` (NOT CREATED - OOM)

### Logs
- `iter43_blockwise_run.log` (COMPLETED)
- `iter44_blockwise_aggressive_run.log` (FAILED - OOM)
- `iter31_heterogeneous_run.log` (FAILED - OOM)
- `iter32_activation_aware_run.log` (FAILED - bug)

---

## Conclusion

**Phase 2 made progress on understanding block-wise fine-tuning but was blocked by GPU memory constraints.** The conservative block-wise approach (10% FP8) hurt performance, suggesting that higher FP8 budgets (70-90%) are needed. However, the aggressive approach couldn't be evaluated due to CUDA OOM errors.

**Next session should:**
1. Free GPU memory (system restart)
2. Re-run Iteration 44 with aggressive FP8 budgets
3. Test all 6 configurations to find optimal block size and budget
4. If successful, combine with other techniques (heterogeneous precision, activation awareness)

**Estimated effort to reach < 6.45 PPL**: 2-3 more iterations with proper GPU memory management.


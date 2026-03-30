# Variant B Integration Plan - Immediate Actions

**Date**: March 30, 2026  
**Status**: Ready to Execute  
**Objective**: Integrate Variant B (Weighted-MSE) into production pipeline and validate

---

## Current State

### ✅ Completed
- Variant B implementation: `variant_b_weighted_mse.py` (263 lines)
- Synthetic validation: 100% improvement in weighted MSE
- Analysis document: `VARIANT_B_ANALYSIS.md`
- Phase 4 production implementation: `phase4_variant_b_production.py`
- Checkpoint integration framework: `phase4_2_checkpoint_integration.py`

### ⏳ Pending
- Full-model compression with Variant B
- MMLU/GSM8K validation
- PPL degradation measurement
- Comparison with Variant A baseline

---

## Immediate Action Items (Next 2-3 hours)

### Step 1: Verify Checkpoint Structure (15 min)
```bash
# Check baseline checkpoint
ls -lh /home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint/
# Expected: config.json + safetensors shards

# Check decompressed checkpoint
ls -lh /home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/decompressed_2b075b_zero_fixed_weighted_abs/
# Expected: config.json + model weights
```

### Step 2: Create Variant B Full-Model Compression Script (30 min)
- Load baseline checkpoint
- Apply Variant B codebook selection to all MoE expert weights
- Save compressed checkpoint
- Measure compression ratio

### Step 3: Decompress & Validate (30 min)
- Load compressed checkpoint
- Decompress using stored codebooks
- Verify reconstruction accuracy
- Compare with Variant A baseline

### Step 4: Run MMLU Evaluation (60 min)
- Use `lm_eval_nvfp4.py` with decompressed checkpoint
- Measure accuracy on MMLU (subset)
- Compare with baseline PPL
- Target: <0.03 PPL degradation

### Step 5: Run GSM8K Evaluation (30 min)
- Measure accuracy on GSM8K
- Verify consistency with MMLU results
- Document final results

---

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Compression Ratio | ≥97.5% | TBD |
| PPL Degradation | <0.03 | TBD |
| MMLU Accuracy | >baseline-0.5% | TBD |
| GSM8K Accuracy | >baseline-0.5% | TBD |
| Execution Time | <4 hours | TBD |

---

## Files to Create/Modify

1. **variant_b_full_model_compression.py** (NEW)
   - Load checkpoint
   - Apply Variant B to all MoE weights
   - Save compressed checkpoint

2. **variant_b_decompression.py** (NEW)
   - Load compressed checkpoint
   - Decompress using stored codebooks
   - Validate reconstruction

3. **variant_b_mmlu_eval.py** (NEW)
   - Run MMLU evaluation
   - Compare with baseline
   - Document results

---

## Decision Points

### After Step 3 (Decompression):
- **Decision**: Proceed with MMLU evaluation?
- **Criteria**: Reconstruction MSE < 0.01
- **If yes**: Continue to Step 4
- **If no**: Debug decompression or adjust parameters

### After Step 4 (MMLU):
- **Decision**: Variant B ready for production?
- **Criteria**: PPL degradation < 0.03
- **If yes**: Proceed to Variant D implementation
- **If no**: Investigate root cause or adjust compression parameters

---

## Risk Mitigation

1. **Checkpoint Backup**: Always keep baseline checkpoint
2. **Incremental Validation**: Test on sample weights first
3. **Easy Rollback**: Can revert to Variant A if needed
4. **Progress Tracking**: Log all intermediate results

---

## Next Phase (After Variant B)

Once Variant B is validated:
1. Implement Variant D (Signed-Pair Constrained)
2. Implement Variant C (Frequency-Regularized MSE)
3. Implement Adaptive Block Scaling
4. Combine all techniques for maximum compression

---

**Estimated Total Time**: 2-3 hours  
**Start Time**: Now  
**Target Completion**: Within 3 hours

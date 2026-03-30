# Variant B Validation Plan

## Current Status

### ✅ Completed
1. **Variant B Checkpoint**: Complete and validated
   - Location: `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs`
   - Size: 16.58 GB (240 shards)
   - Compression ratio: 1.28x (vs 22GB baseline)
   - Weight distribution:
     - Codebook entries: 30,840
     - Indices: 30,840
     - Scales: 92,520
     - Regular weights: 433

2. **Checkpoint Analysis**: Complete
   - All 733 shards present (240 compressed, 493 symlinks to baseline)
   - Index file valid: `model.safetensors.index.json` (15MB)
   - Config valid: Qwen3Next with 40 layers, 256 experts

### 🔄 In Progress
1. **MMLU Evaluation Setup**: Blocked by tensorrt_llm build issue
   - lm_eval_nvfp4.py requires tensorrt_llm._torch.auto_deploy.custom_ops
   - Build failing due to egg-info permissions

### ⏳ Next Steps

## Phase 1: Resolve Build Issue (30 min)
1. Clean build artifacts
2. Try alternative build approach
3. Or: Create minimal evaluation wrapper that doesn't require full tensorrt_llm

## Phase 2: Run Variant B Evaluation (1-2 hours)
1. Run MMLU subset (5 subjects):
   - professional_law
   - abstract_algebra
   - anatomy
   - astronomy
   - business_ethics

2. Measure:
   - Accuracy per subject
   - Perplexity (if available)
   - Inference time
   - Memory usage

3. Compare with baseline:
   - Baseline NVFP4: 61% on abstract_algebra, 59.8% on professional_law
   - Target: <0.03 PPL degradation

## Phase 3: Implement Variant D (2-3 hours)
1. Signed-pair constrained codebook selection
2. A/B test against Variant B
3. Document performance difference

## Phase 4: Extended Validation (2-3 hours, if time permits)
1. Full MMLU evaluation (57 subjects)
2. GSM8K benchmark
3. Variant C implementation (frequency-regularized MSE)

## Key Constraints
- No retraining
- No scale recomputation
- No shared-codebook redesign
- Must pair with per-channel or block-diagonal Fisher codebook selection

## Success Criteria
- Variant B validation: >97.5% compression ratio, <0.03 PPL degradation
- Variant D: Better compression than Variant B
- Full MMLU: Comparable accuracy to baseline

## Files to Monitor
- `/tmp/variant_b_eval.log` - Evaluation progress
- `variant_b_mmlu_results.json` - Results
- `variant_b_detailed_results.json` - Detailed metrics

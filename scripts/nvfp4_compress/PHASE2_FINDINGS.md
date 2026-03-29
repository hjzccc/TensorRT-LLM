# Phase 2 Findings & Critical Discovery

## Summary

Phase 2 evaluation completed successfully, but revealed a **critical architectural issue** that invalidates the initial results.

## Key Finding: Weights are BF16, Not FP4

**Discovery**: The model weights in safetensors files are stored as **BFloat16**, not FP4.

- Total weight keys: 1811
- Expert weight keys: 848 (gate_up_proj + down_proj)
- Format: BFloat16 (not FP4)

**Implication**: FP4 quantization happens **during the forward pass** via `torch.ops.trtllm.fp4_quantize()`, not during weight loading.

## Phase 2 Results (Invalid)

All 5 codebook strategies produced **identical PPL: 6.6976**

| ID | Codebook | Codes | PPL | Status |
|----|----------|-------|-----|--------|
| 2.1 | identity | 16 | 6.6976 | ✗ Invalid |
| 2.2 | 3bit_uniform | 8 | 6.6976 | ✗ Invalid |
| 2.3 | 3bit_adaptive | 8 | 6.6976 | ✗ Invalid |
| 2.4 | 2bit_uniform | 4 | 6.6976 | ✗ Invalid |
| 2.5 | 2bit_optimal | 4 | 6.6976 | ✗ Invalid |

**Why Invalid**: The CodebookWeightStore was trying to apply codebook mapping to BF16 weights, which had no effect. All evaluations ran with original BF16 weights, producing identical results.

## Root Cause

The CodebookWeightStore implementation assumed:
1. Weights are loaded as FP4 packed bytes
2. Codebook mapping can be applied to packed FP4 codes
3. Mapped codes are repacked and used in inference

**Reality**:
1. Weights are loaded as BF16 floats
2. FP4 quantization happens in the forward pass via `fp4_quantize()`
3. Codebook mapping must happen AFTER quantization, not before

## Correct Approach for Phase 2 (Revised)

To properly evaluate codebook compression, we need to:

1. **Intercept the FP4 quantization** in the forward pass
2. **Apply codebook mapping** to the quantized FP4 codes
3. **Repack and use** the mapped codes in tensor core operations

This requires modifying the evaluation pipeline to:
- Hook into `moe_forward_exact()` or `exact_linear()`
- Capture the FP4 quantized codes
- Apply codebook mapping
- Continue with inference

## Next Steps

### Option A: Modify exact_docker_eval.py (Recommended)
- Add a `codebook` parameter to `evaluate_ppl()`
- Modify `moe_forward_exact()` to apply codebook mapping after FP4 quantization
- Re-run Phase 2 with correct implementation

### Option B: Create a wrapper around fp4_quantize
- Monkey-patch `torch.ops.trtllm.fp4_quantize()` to apply codebook mapping
- Less invasive but harder to debug

### Option C: Implement Phase 3 directly
- Skip Phase 2 (per-block-16 codebook)
- Jump to Phase 3 (per-block optimal codebook selection)
- Use the correct approach from the start

## Recommendation

**Proceed with Option A**: Modify exact_docker_eval.py to properly apply codebook mapping after FP4 quantization. This will give us correct Phase 2 results and establish the foundation for Phase 3.

## Files Affected

- `/code/tensorrt_llm/scripts/nvfp4_compress/phase2_full_eval.py` - needs rewrite
- `/code/tensorrt_llm/scripts/channel_quant_new/exact_docker_eval.py` - needs modification
- `/code/tensorrt_llm/scripts/nvfp4_compress/fp4_utils.py` - no changes needed
- `/code/tensorrt_llm/scripts/nvfp4_compress/eval_pipeline.py` - no changes needed

## Timeline Impact

- Phase 2 (corrected): ~1 hour to implement + 1 hour to run
- Phase 3: Can proceed after Phase 2 results are validated
- Total delay: ~2 hours

## Lessons Learned

1. **Always verify data formats** - assumed FP4 but weights were BF16
2. **Understand the quantization pipeline** - FP4 quantization is a forward-pass operation, not a weight-loading operation
3. **Test with small examples first** - would have caught this issue immediately

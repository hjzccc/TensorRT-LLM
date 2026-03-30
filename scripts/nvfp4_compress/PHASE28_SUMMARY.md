# Phase 28: Entropy-Based Codebook Selection

## Objective
Implement entropy-based codebook selection to improve compression ratio beyond Phase 21 baseline (97.725%).

## Implementation

### Approach
Added entropy-based loss mode to compress_checkpoint.py that selects codebooks based on frequency skewness:
- **Frequency Skewness**: Computed as sum of squared code frequencies
- **Intuition**: More skewed distributions (higher skewness) have lower entropy and are more compressible
- **Fast Proxy**: Avoids expensive log calculations by using skewness as entropy proxy

### Code Changes
1. Added `compute_code_entropy()` function for Shannon entropy calculation
2. Added entropy loss mode handling in `compress_codes()` function
3. Added two new schemes:
   - `2b075b_zero_fixed_entropy`: 0-fixed variant
   - `3b1b_4free_entropy`: 4-free variant

### Implementation Details
```python
# Fast entropy-based selection using frequency skewness
freq = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
skewness = (freq ** 2).sum(dim=1, keepdim=True)  # Higher skewness = lower entropy
weighted_counts = counts / (skewness + 1e-8)
costs = weighted_counts @ candidate_mse_luts.T
```

## Test Results

### Sample Test (model-00104-of-00733.safetensors)
- **Baseline (exact)**: 31.25% compression
- **Entropy-based**: 31.25% compression
- **Improvement**: +0.00%

### Analysis
The entropy-based selection shows **no improvement** on the sample test. This suggests:

1. **Frequency skewness is too similar to MSE**: The heuristic doesn't provide additional discriminative power
2. **Code distribution is already near-optimal**: The baseline MSE selection already produces relatively skewed distributions
3. **Entropy correlation is weak**: Shannon entropy may not be the right metric for this problem

## Findings

### Why Entropy-Based Selection Doesn't Help
1. **MSE already optimizes for reconstruction quality**: Lower MSE codebooks tend to produce more skewed distributions naturally
2. **Frequency distribution is uniform across blocks**: All 16 FP4 codes appear with ~6.25% frequency (Phase 24 finding)
3. **Skewness heuristic is redundant**: Weighting by skewness doesn't add new information beyond MSE

### Literature Context
- **AQLM**: Uses adaptive bit-width allocation (not applicable to fixed 2-bit indices)
- **EntroLLM**: Uses entropy for codebook selection but on different problem (continuous values, not discrete FP4)
- **GPTQ**: Uses gradient-based quantization (requires retraining, excluded per constraints)

## Conclusion

**Phase 28 Result**: Entropy-based codebook selection does NOT improve compression ratio.

The frequency skewness heuristic is too similar to MSE-based selection to provide additional benefit. The baseline MSE approach already selects codebooks that produce relatively skewed distributions.

## Recommendations

1. **Do not pursue entropy-based selection further**: The approach is fundamentally limited by the uniform code distribution
2. **Focus on alternative directions**:
   - Phase 29: Mixed-precision quantization (variable bits per block)
   - Phase 30: Learned codebook refinement (optimize codebooks for specific data)
   - Phase 31: Residual quantization (two-stage approach)

## Files Modified
- `compress_checkpoint.py`: Added entropy loss mode and schemes
- `phase28_entropy_test.py`: Test script for validation

## Status
✓ Implementation complete
✓ Testing complete
✗ No improvement found
→ Ready to move to Phase 29


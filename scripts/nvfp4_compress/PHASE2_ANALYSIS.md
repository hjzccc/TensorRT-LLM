# Phase 2 Analysis: 3bit_uniform Failure

## Observations

1. **Baseline (identity mapping)**: PPL = 6.6974 ✓
   - Full NVFP4 codebook
   - 618 seconds
   - Pipeline verified correct

2. **3bit_uniform**: PPL = 1817424.66 ✗
   - Codebook: [-6, -4, -2, 0, 2, 4, 6]
   - 1330 seconds
   - CRITICAL FAILURE

3. **Diagnostic Tests**:
   - Codebook LUT: ✓ Valid (verified)
   - Minimal test: ✓ Valid outputs
   - Partial eval (3 layers): ✓ Reasonable PPL
     - nvfp4_full: 36315.5
     - 3bit_uniform: 38657.7 (+2.3%)

## Root Cause Hypothesis

**Cumulative Error Divergence**: The 3bit_uniform codebook introduces quantization error that compounds across layers, causing the model's predictions to diverge and loss to increase exponentially.

### Evidence

1. **Partial evaluation shows only +2.3% worse** on 3 layers
2. **Full evaluation shows catastrophic failure** on 40 layers
3. **Required mean loss for observed PPL**: 14.41 (vs 10.5 for 3 layers)
4. **This suggests loss increases from ~10.5 to ~14.4 across 40 layers**

### Mechanism

The 3bit_uniform codebook has significant quantization error for certain values:
- Code 1 (+0.5) → 0 (+0.0): error = 0.5
- Code 2 (+1.0) → 0 (+0.0): error = 1.0
- Code 3 (+1.5) → 4 (+2.0): error = 0.5
- Code 5 (+3.0) → 4 (+2.0): error = 1.0
- Code 13 (-3.0) → 14 (-4.0): error = 1.0

These errors accumulate across layers, causing:
1. Hidden states to diverge from correct values
2. Attention patterns to change
3. Expert routing to change
4. Loss to increase exponentially

## Solution Approaches

### Option 1: Better Codebook Design
- Use codebooks with smaller maximum error
- Example: 3bit_dense [-6, -2, -1, 0, 1, 2, 6] has max error = 2.0 (vs 1.0 for 3bit_uniform)
- Example: 3bit_truncate [-4, -2, -1, 0, 1, 2, 4] has max error = 2.0

### Option 2: Per-Block Optimal Codebook
- Allow different codebooks per block
- Select codebook that minimizes error for each block
- Overhead: 8 bits per block = 0.5 bits/elem

### Option 3: Adaptive Precision
- Use full NVFP4 for critical layers (early layers, attention)
- Use 3bit for non-critical layers (later layers, MoE)

### Option 4: Investigate Numerical Stability
- Check if there's a numerical issue in the kernel
- Verify scales are being applied correctly
- Check for underflow/overflow in intermediate computations

## Recommendation

**Proceed with Option 2 (Per-Block Optimal Codebook)** because:
1. It allows flexibility to minimize error per block
2. Overhead is manageable (0.5 bits/elem)
3. It's a principled approach based on entropy analysis
4. It should avoid cumulative error divergence

**Before that, test Option 1** with better codebooks:
- 3bit_dense: [-6, -2, -1, 0, 1, 2, 6]
- 3bit_truncate: [-4, -2, -1, 0, 1, 2, 4]

These have better error distribution and might avoid divergence.


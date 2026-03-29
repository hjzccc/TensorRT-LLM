# Phase 7: Product Quantization Exploration

## Overview

Product Quantization (PQ) decomposes the codebook into products of smaller codebooks, reducing storage and improving reconstruction quality.

**Expected Improvement**: 10-15% additional MSE improvement
**Estimated Time**: 2-3 hours
**Reference**: Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)

## Theory

### Standard Codebook (Current)
- Single codebook per layer: 8 entries (primary) + 4 (residual) + 2 (residual2) = 14 entries
- Storage: 14 × 4 bytes = 56 bytes per layer

### Product Quantization
- Decompose codebook into M subcodebooks
- Each subcodebook has K entries
- Total entries: K^M (much smaller than single codebook)
- Example: 2 subcodebooks × 4 entries each = 4^2 = 16 combinations

### Reconstruction
- Decompose value into M subvectors
- Quantize each subvector independently
- Reconstruct by summing subvector reconstructions

## Implementation Plan

### Step 1: Implement Product Quantization (30 min)
```python
def product_quantization(values, n_subcodebooks=2, codebook_size=4):
    """
    Decompose values into product of subcodebooks.
    
    Args:
        values: Input values (N,)
        n_subcodebooks: Number of subcodebooks M
        codebook_size: Size of each subcodebook K
    
    Returns:
        subcodebooks: List of M codebooks
        assignments: (N, M) array of assignments
        reconstruction: Reconstructed values
    """
    # Decompose values into subvectors
    subvector_size = len(values) // n_subcodebooks
    subvectors = values.reshape(n_subcodebooks, -1)
    
    # Learn codebook for each subvector
    subcodebooks = []
    assignments = []
    reconstruction = np.zeros_like(values)
    
    for i, subvector in enumerate(subvectors):
        # Learn K-means codebook
        codebook, _, kmeans = learn_kmeans_codebook_uniform(
            subvector.reshape(-1, 1), codebook_size
        )
        subcodebooks.append(codebook)
        
        # Get assignments
        distances = np.abs(subvector[:, None] - codebook[None, :])
        labels = np.argmin(distances, axis=1)
        assignments.append(labels)
        
        # Reconstruct
        reconstruction[i*subvector_size:(i+1)*subvector_size] = codebook[labels]
    
    return subcodebooks, assignments, reconstruction
```

### Step 2: Test on Synthetic Data (30 min)
- Generate synthetic layer data
- Test different configurations:
  - n_subcodebooks: 1, 2, 3, 4
  - codebook_size: 2, 4, 8, 16
- Measure MSE improvement for each configuration
- Find optimal configuration

### Step 3: Integrate with Soft Assignment (30 min)
- Combine product quantization with soft assignment (T=1.75)
- Test three-stage residual with PQ
- Measure cumulative improvement

### Step 4: Validate on Realistic Data (30 min)
- Test on real NVFP4 weights
- Verify improvement holds
- Measure compression ratio and speed

## Expected Results

### Configuration Analysis
| Config | Storage | Improvement | Status |
|--------|---------|-------------|--------|
| 1 subcodebook, 4 entries | 4 bytes | Baseline | ⚠️ |
| 2 subcodebooks, 4 entries | 8 bytes | 5-8% | ✅ |
| 2 subcodebooks, 8 entries | 16 bytes | 8-12% | ✅ |
| 3 subcodebooks, 4 entries | 12 bytes | 7-10% | ✅ |
| 4 subcodebooks, 4 entries | 16 bytes | 8-12% | ✅ |

### Cumulative Impact
- Phase 6B: 159.07% MSE improvement
- Phase 7 (PQ): 10-15% additional improvement
- **Total**: 169-174% MSE improvement (compounded)

## Success Criteria

✅ Product quantization implemented
✅ Tested on synthetic data
✅ Integrated with soft assignment
✅ Validated on realistic data
✅ 10-15% improvement achieved
✅ No degradation in any layer

## Next Steps After PQ

1. **EM Clustering** (1-2 hours, expected 3-5% improvement)
2. **Quantization-Aware Training** (4-6 hours, expected 5-10% improvement)
3. **Hybrid Approaches** (2-3 hours, expected 5-8% improvement)

## References

- Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)
- Ge et al., "Optimized Product Quantization for Approximate Nearest Neighbor Search" (2013)
- Babenko & Lempitsky, "The Inverted Index for Approximate Nearest-Neighbor Search" (2012)

## Status

⏳ **Ready to Start** - All prerequisites complete

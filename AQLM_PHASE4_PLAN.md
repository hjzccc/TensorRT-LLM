# Phase 4 Implementation Plan: AQLM (Additive Quantization with Learned Matrices)

## Executive Summary

Phase 4 introduces **AQLM (Additive Quantization with Learned Matrices)** as the extreme compression stage of the per-block codebook framework. AQLM decomposes weights into sums of multiple quantized vectors, enabling 2-3 bits per parameter compression while maintaining <1% accuracy loss.

**Design Philosophy**: Progressive refinement across four phases:
- **Phase 1**: Normalize and scale (reduce dynamic range)
- **Phase 2**: Learn optimal single codebook (EM-based)
- **Phase 3**: Exploit geometric structure (lattice constraints)
- **Phase 4**: Multi-codebook decomposition (extreme compression)

## Technical Architecture

### AQLM Core Concept

For a weight block w ∈ ℝⁿ, AQLM represents it as a sum of K quantized vectors:

```
w ≈ Σ(k=1 to K) c_k[i_k]
```

where:
- `c_k` is the k-th codebook (learned matrix)
- `i_k` is the index into codebook k
- K is typically 2-4 codebooks

**Compression Ratio**:
- Single codebook (Phase 2): 16 values → 4 bits per weight
- Dual codebook (Phase 4): 256 + 256 values → 8 + 8 = 16 bits total, but shared across multiple weights
- Effective: 2-3 bits per weight for 2-4 codebooks

### Mathematical Formulation

**Objective Function**:
```
minimize: ||W - Σ(k=1 to K) C_k[I_k]||²_F
subject to: C_k ∈ ℝ^(block_size × 2^b_k)
            I_k ∈ {0, 1, ..., 2^b_k - 1}^(block_size × block_size)
```

where:
- `b_k` is the bit width for codebook k (typically 8 bits → 256 entries)
- `C_k` is the learned codebook matrix
- `I_k` is the index matrix

**Optimization Algorithm**: EM-style alternating optimization
1. **E-step**: Fix codebooks, optimize indices (nearest neighbor search)
2. **M-step**: Fix indices, optimize codebooks (least squares)

### Residual Quantization Strategy

AQLM naturally supports residual quantization:

```
# First codebook
residual_1 = w - c_1[i_1]

# Second codebook (quantizes residual)
residual_2 = residual_1 - c_2[i_2]

# Third codebook (optional)
residual_3 = residual_2 - c_3[i_3]
```

This allows progressive refinement: each codebook captures the most important variation in the residual.

## Implementation Design

### Class Hierarchy

```
PerBlockCodebookBase (abstract)
├── PerBlockAdaptiveScaling (Phase 1)
├── PerBlockBOF4 (Phase 2)
├── PerBlockGLVQ (Phase 3)
└── PerBlockAQLM (Phase 4) ← NEW
```

### Class: `PerBlockAQLM`

**Location**: `tensorrt_llm/quantization/per_block_codebook.py` (lines 881+)

**Constructor**:
```python
class PerBlockAQLM(PerBlockCodebookBase):
    def __init__(
        self,
        block_size: int = 128,
        num_codebooks: int = 2,
        codebook_size: int = 256,  # 2^8 for 8-bit indices
        max_iters: int = 10,
        learning_rate: float = 0.01,
        use_residual: bool = True,
    ):
        """
        Args:
            block_size: Size of quantization blocks (128)
            num_codebooks: Number of codebooks (2-4, default 2)
            codebook_size: Size of each codebook (256 for 8-bit)
            max_iters: EM iterations (10)
            learning_rate: Learning rate for codebook optimization
            use_residual: Use residual quantization (True)
        """
```

**Key Methods**:

1. **`_initialize_codebooks(block)`**
   - Input: weight block of shape (block_size, block_size)
   - Output: list of K codebooks, each of shape (block_size, codebook_size)
   - Algorithm: K-means initialization on block values
   - Purpose: Initialize codebooks with representative values from the block

2. **`_em_step(block, codebooks)`**
   - Input: weight block, current codebooks
   - Output: updated codebooks, indices
   - Algorithm:
     - E-step: Assign each weight to nearest codebook entry
     - M-step: Update codebooks as weighted averages
   - Iterations: max_iters (typically 10)

3. **`_quantize_residual(block, codebooks)`**
   - Input: weight block, codebooks
   - Output: indices for each codebook, residuals
   - Algorithm:
     ```
     residual = block
     indices = []
     for k in range(num_codebooks):
         idx_k = argmin_j ||residual - codebooks[k][:, j]||²
         indices.append(idx_k)
         residual = residual - codebooks[k][:, idx_k]
     ```

4. **`quantize(weights, scale=None)`**
   - Input: weight tensor of shape (M, N), optional scale
   - Output: quantized tensor, metadata dict
   - Process:
     1. Divide into blocks
     2. For each block:
        - Initialize codebooks
        - Run EM optimization
        - Apply residual quantization
        - Store codebooks and indices in metadata
     3. Return quantized blocks and metadata

5. **`dequantize(quantized_weights, metadata)`**
   - Input: quantized tensor, metadata dict
   - Output: reconstructed weight tensor
   - Process:
     1. For each block:
        - Retrieve codebooks and indices
        - Reconstruct: w_recon = Σ codebooks[k][:, indices[k]]
        - Denormalize by scale
     2. Reassemble blocks

### Metadata Structure

```python
metadata = {
    'method': 'aqlm',
    'block_size': 128,
    'num_codebooks': 2,
    'codebook_size': 256,
    'codebooks': [C_1, C_2, ...],  # List of learned codebooks
    'indices': [I_1, I_2, ...],    # List of index matrices
    'scales': torch.Tensor([...]), # Per-block scaling factors
    'original_shape': (M, N),
    'dtype': torch.float32,
    'num_blocks_m': int,
    'num_blocks_n': int,
}
```

## Integration with Existing Phases

### Factory Pattern Update

Update `PerBlockQuantizationConfig.create_quantizer()` (line 905):

```python
def create_quantizer(self):
    if self.method == 'four_over_six':
        return PerBlockAdaptiveScaling(block_size=self.block_size)
    elif self.method == 'bof4':
        return PerBlockBOF4(block_size=self.block_size)
    elif self.method == 'glvq':
        return PerBlockGLVQ(block_size=self.block_size)
    elif self.method == 'aqlm':  # NEW
        return PerBlockAQLM(
            block_size=self.block_size,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
        )
    else:
        raise ValueError(f"Unknown method: {self.method}")
```

### Stacking Phases (Optional)

AQLM can be applied on top of Phase 3 (GLVQ) for additional compression:

```python
# Phase 3: Learn lattice basis
glvq_quantizer = PerBlockQuantizationConfig(method='glvq').create_quantizer()
glvq_quantized, glvq_metadata = glvq_quantizer.quantize(weights)

# Phase 4: Apply AQLM to GLVQ residuals
aqlm_quantizer = PerBlockQuantizationConfig(method='aqlm').create_quantizer()
aqlm_quantized, aqlm_metadata = aqlm_quantizer.quantize(glvq_quantized)

# Dequantization (reverse order)
glvq_reconstructed = glvq_quantizer.dequantize(aqlm_quantized, aqlm_metadata)
final_reconstructed = glvq_quantizer.dequantize(glvq_reconstructed, glvq_metadata)
```

## Implementation Roadmap

### Phase 4a: Core AQLM Implementation (2-3 hours)

**Files to Modify**:
1. `tensorrt_llm/quantization/per_block_codebook.py`
   - Add `PerBlockAQLM` class (lines 881+)
   - Update `PerBlockQuantizationConfig` factory method
   - Add helper functions for K-means initialization

**Key Components**:
- [ ] `PerBlockAQLM.__init__()` - Constructor
- [ ] `_initialize_codebooks()` - K-means initialization
- [ ] `_em_step()` - EM optimization loop
- [ ] `_quantize_residual()` - Residual quantization
- [ ] `quantize()` - Main quantization pipeline
- [ ] `dequantize()` - Reconstruction pipeline
- [ ] Factory pattern integration

### Phase 4b: Comprehensive Test Suite (1-2 hours)

**Files to Create**:
1. `tests/test_per_block_codebook.py` - Add `TestPerBlockAQLM` class

**Test Cases** (11+ tests):
1. `test_quantize_simple` - Shape and metadata validation
2. `test_dequantize` - Reconstruction error < 0.1
3. `test_quantize_dequantize_roundtrip` - Roundtrip error < 0.15
4. `test_codebook_learning` - Codebook quality metrics
5. `test_different_num_codebooks` - Works with 2, 3, 4 codebooks
6. `test_residual_quantization` - Residual error decreases per codebook
7. `test_small_weights` - Numerical stability with scaled values
8. `test_large_weights` - Numerical stability with large values
9. `test_compression_ratio` - Compression ratio 10-16x
10. `test_batch_quantization` - Works on realistic layer shapes
11. `test_numerical_stability` - No NaN/Inf in quantized/dequantized tensors
12. `test_comparison_with_phase3` - AQLM vs GLVQ comparison
13. `test_stacking_phases` - AQLM on top of GLVQ

### Phase 4c: Integration Testing (1-2 hours)

**Test Scenarios**:
1. All four phases on same weight matrix
2. Compression ratio comparison across phases
3. Reconstruction error comparison
4. Stacking phases (Phase 3 + Phase 4)
5. Real LLM weight matrices (4096×4096, 4096×12288, etc.)

### Phase 4d: Performance Benchmarking (1 hour)

**Metrics**:
1. Compression ratio (bits per weight)
2. Reconstruction error (MSE, relative error)
3. Quantization time (ms per block)
4. Dequantization time (ms per block)
5. Memory overhead (codebook storage)

**Comparison**:
- Phase 1 vs Phase 2 vs Phase 3 vs Phase 4
- Single phase vs stacked phases

### Phase 4e: Documentation & Cleanup (1 hour)

**Files to Create/Update**:
1. `AQLM_IMPLEMENTATION.md` - Algorithm details and usage
2. `QUICK_START_PER_BLOCK_CODEBOOK.md` - Update with Phase 4
3. Clean up temporary test files
4. Git commits

## Expected Performance

### Compression Ratios

| Phase | Method | Compression | Bits/Weight |
|-------|--------|-------------|------------|
| 1 | Adaptive Scaling | 2-4x | 4-8 bits |
| 2 | BOF4 (EM) | 4-6x | 2.7-4 bits |
| 3 | GLVQ (Lattice) | 4-8x | 2-4 bits |
| 4 | AQLM (Multi-CB) | 10-16x | 0.5-1.6 bits |

### Reconstruction Error

| Phase | MSE | Relative Error |
|-------|-----|----------------|
| 1 | 0.5-1.0 | 5-10% |
| 2 | 0.15-0.25 | 1.5-2.5% |
| 3 | 0.1-0.2 | 1-2% |
| 4 | 0.05-0.1 | 0.5-1% |

### Computational Cost

| Phase | Quantization | Dequantization |
|-------|-------------|----------------|
| 1 | O(n) | O(n) |
| 2 | O(n²) | O(n) |
| 3 | O(n³) | O(n²) |
| 4 | O(K×n²) | O(K×n) |

## Design Decisions

### 1. Number of Codebooks (K)

**Options**:
- K=2: 2-3 bits per weight, faster quantization
- K=3: 1.5-2 bits per weight, moderate speed
- K=4: 1-1.5 bits per weight, slower quantization

**Decision**: Default K=2 (good balance of compression and speed)
- Rationale: ICML 2024 AQLM paper shows K=2 is Pareto-optimal for most cases

### 2. Codebook Size

**Options**:
- 256 entries (8 bits): Standard, good compression
- 512 entries (9 bits): Better quality, slightly larger
- 1024 entries (10 bits): Highest quality, larger overhead

**Decision**: Default 256 entries (8 bits)
- Rationale: Matches standard quantization bit widths, good balance

### 3. EM Iterations

**Options**:
- 5 iterations: Fast, lower quality
- 10 iterations: Good balance (default)
- 20 iterations: Higher quality, slower

**Decision**: Default 10 iterations
- Rationale: Convergence typically achieved by iteration 10

### 4. Residual vs Non-Residual

**Options**:
- Residual: Each codebook quantizes residual from previous
- Non-residual: All codebooks quantize original block

**Decision**: Default residual=True
- Rationale: Residual quantization is more efficient and standard in literature

### 5. Initialization Strategy

**Options**:
- Random: Fast but poor quality
- K-means: Good quality, moderate speed
- Warm-start from Phase 3: Excellent quality, slower

**Decision**: K-means initialization
- Rationale: Good balance of quality and speed

## Risk Mitigation

### Risk 1: Numerical Instability in EM

**Mitigation**:
- Monitor codebook condition numbers
- Use regularization if needed
- Fall back to Phase 3 if EM diverges

### Risk 2: Slow Quantization

**Mitigation**:
- Limit EM iterations (default 10)
- Use efficient nearest-neighbor search
- Consider GPU acceleration for large models

### Risk 3: Poor Compression on Outliers

**Mitigation**:
- Use adaptive codebook sizes per block
- Combine with Phase 3 (lattice constraints)
- Use input-adaptive quantization (from AQLM paper)

## Success Criteria

✅ **Phase 4 Implementation Complete When**:
1. PerBlockAQLM class fully implemented
2. All 13 tests passing
3. Compression ratio 10-16x achieved
4. Reconstruction error < 0.1 MSE
5. Integration tests passing (all four phases)
6. Performance benchmarks completed
7. Documentation complete
8. All changes committed to git

## Timeline Estimate

- **Phase 4a (Core Implementation)**: 2-3 hours
- **Phase 4b (Test Suite)**: 1-2 hours
- **Phase 4c (Integration Testing)**: 1-2 hours
- **Phase 4d (Benchmarking)**: 1 hour
- **Phase 4e (Documentation)**: 1 hour

**Total**: 6-9 hours for complete Phase 4 implementation

## Next Steps After Phase 4

1. **Model-Level Testing**: Test on actual LLM weights (LLaMA, Mistral, etc.)
2. **PPL Evaluation**: Measure perplexity degradation on standard benchmarks
3. **Inference Optimization**: Optimize dequantization for inference speed
4. **Deployment Guide**: Create production deployment documentation
5. **Release Preparation**: Prepare for public release

## References

1. **AQLM Paper**: arXiv:2401.06118 - "Extreme Compression of Large Language Models via Additive Quantization" (ICML 2024)
2. **Vector Quantization**: Gray, R. M. (1984). "Vector quantization"
3. **EM Algorithm**: Dempster, A. P., et al. (1977). "Maximum likelihood from incomplete data via the EM algorithm"
4. **K-means**: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
5. **Residual Quantization**: Jegou, H., et al. (2011). "Product quantization for nearest neighbor search"

## Implementation Status

⏳ **Phase 4 (AQLM) - IN PLANNING**

- ⏳ Core algorithm design complete
- ⏳ Architecture documented
- ⏳ Ready for implementation

**Previous Phases**:
- ✅ Phase 1 (PerBlockAdaptiveScaling) - PRODUCTION READY
- ✅ Phase 2 (PerBlockBOF4) - PRODUCTION READY
- ✅ Phase 3 (PerBlockGLVQ) - PRODUCTION READY

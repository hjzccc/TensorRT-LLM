# Research Exploration Plan: Beyond Phase 4

## Current Achievement
- **Phase 1**: Adaptive scaling (error: 0.073)
- **Phase 2**: EM-optimized codebook (error: 0.082)
- **Phase 4**: AQLM multi-codebook (error: 0.0039) ✅ **EXCELLENT**

## Research Directions to Explore

### 1. Entropy Coding for AQLM Indices
**Hypothesis**: AQLM indices are not uniformly distributed. Entropy coding can reduce storage.

**Papers to Review**:
- Float8@2bits (arXiv 2601.22787) - Entropy coding of Float8 to 2 bits
- Huffman coding for quantization indices
- Arithmetic coding for better compression

**Expected Benefit**: 10-20% additional compression on indices

**Implementation Approach**:
1. Analyze distribution of AQLM indices
2. Build Huffman tree per block
3. Encode indices using variable-length codes
4. Store codebook + Huffman tree in metadata

### 2. Hybrid AQLM + Entropy Coding
**Hypothesis**: Combining AQLM with entropy coding gives best compression.

**Implementation**:
```python
# Phase 4a: AQLM quantization
aqlm_quantized, aqlm_meta = aqlm.quantize(weights)

# Phase 4b: Entropy code the indices
entropy_coded, entropy_meta = entropy_coder.encode(aqlm_meta['indices'])

# Result: Compressed indices + codebooks
```

**Expected Compression**: 2-3 bits/parameter → 1-2 bits/parameter

### 3. Adaptive Codebook Sizes
**Hypothesis**: Different blocks need different codebook sizes.

**Approach**:
- Analyze reconstruction error per block
- Use larger codebooks for high-error blocks
- Use smaller codebooks for low-error blocks
- Store codebook size in metadata

**Expected Benefit**: Better accuracy with same compression

### 4. Input-Adaptive Quantization (from AQLM Paper)
**Hypothesis**: Quantization parameters should depend on input distribution.

**Approach**:
- Learn per-block quantization parameters
- Adapt codebook initialization based on weight distribution
- Use learned scaling factors

**Expected Benefit**: Better compression on diverse weight distributions

### 5. Residual Entropy Coding
**Hypothesis**: Residuals after AQLM have structure that can be exploited.

**Approach**:
- Analyze residual distribution after each codebook
- Apply entropy coding to residuals
- Store residual codebook

**Expected Benefit**: Capture remaining structure

### 6. Learned Quantization Schedules
**Hypothesis**: Different layers benefit from different quantization strategies.

**Approach**:
- Profile each layer's weight distribution
- Learn optimal quantization parameters per layer
- Use heterogeneous bit-widths

**Expected Benefit**: Better accuracy with same compression

## Priority Ranking

1. **HIGH**: Entropy coding for AQLM indices (easy, high impact)
2. **HIGH**: Hybrid AQLM + entropy coding (builds on #1)
3. **MEDIUM**: Adaptive codebook sizes (moderate complexity, good benefit)
4. **MEDIUM**: Input-adaptive quantization (requires research)
5. **LOW**: Residual entropy coding (complex, uncertain benefit)
6. **LOW**: Learned quantization schedules (requires model training)

## Immediate Next Steps

1. **Analyze AQLM index distribution** (30 min)
   - Check if indices are uniformly distributed
   - Compute entropy of index distribution
   - Estimate compression potential

2. **Implement entropy coding** (1-2 hours)
   - Huffman coding for indices
   - Integration with AQLM
   - Test on various weight matrices

3. **Benchmark hybrid approach** (1 hour)
   - Measure compression ratio
   - Measure reconstruction error
   - Compare with pure AQLM

4. **Document findings** (30 min)
   - Create research summary
   - Propose Phase 5 (Entropy Coding)

## Success Criteria

- ✅ Entropy coding reduces index storage by 10-20%
- ✅ Hybrid approach achieves 1-2 bits/parameter
- ✅ Reconstruction error remains < 0.01
- ✅ Implementation is practical and fast

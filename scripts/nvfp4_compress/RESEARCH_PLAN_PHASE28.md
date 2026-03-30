# Research Plan: Phase 28 - New Directions

## Current State
- **Best Compression**: Phase 23c (97.96%)
- **Best MMLU**: 2b075b_zero_fixed_exact (76.39%, 2.75 bits/elem)
- **Baseline**: NVFP4 (4.0 bits/elem)

## Explored Approaches
1. ✅ Exhaustive codebook search (Phase 2)
2. ✅ DAQ metrics (Phase 22)
3. ✅ Hierarchical codebook (Phase 23)
4. ✅ Expert-aware quantization (Phase 23c)
5. ✅ Loss mode variations (Phase 24)
6. ✅ Bias correction (Phase 25-27)

## Unexplored Directions

### Direction 1: Adaptive Block Size
- **Hypothesis**: Different layers need different block sizes
- **Approach**: Use smaller blocks for high-sensitivity layers, larger for low-sensitivity
- **Expected Improvement**: +0.2-0.5%
- **Effort**: Medium (2-3 hours)

### Direction 2: Mixed-Precision Quantization
- **Hypothesis**: Different layers can use different bit widths
- **Approach**: Use 2-bit for low-sensitivity, 3-bit for high-sensitivity
- **Expected Improvement**: +0.5-1.0%
- **Effort**: High (4-5 hours)

### Direction 3: Learned Codebook Refinement
- **Hypothesis**: Refine codebook entries using gradient descent
- **Approach**: Use activation data to optimize codebook entries
- **Expected Improvement**: +0.3-0.7%
- **Effort**: High (4-5 hours)

### Direction 4: Residual Quantization
- **Hypothesis**: Quantize residuals after first-pass quantization
- **Approach**: Two-stage quantization with residual correction
- **Expected Improvement**: +0.5-1.5%
- **Effort**: High (5-6 hours)

### Direction 5: Entropy-Based Codebook Selection
- **Hypothesis**: Select codebooks based on entropy of weight distribution
- **Approach**: Compute entropy per block, use as selection criterion
- **Expected Improvement**: +0.1-0.3%
- **Effort**: Low (1-2 hours)

### Direction 6: Activation-Aware Quantization
- **Hypothesis**: Use activation statistics to guide quantization
- **Approach**: Compute activation ranges, adjust quantization accordingly
- **Expected Improvement**: +0.2-0.5%
- **Effort**: Medium (2-3 hours)

## Recommended Next Steps

1. **Immediate** (Phase 28): Test Adaptive Block Size
   - Low effort, medium potential improvement
   - Can be done in parallel with other work

2. **Short-term** (Phase 29): Test Mixed-Precision Quantization
   - Higher effort but higher potential improvement
   - Builds on Phase 28 results

3. **Medium-term** (Phase 30): Test Learned Codebook Refinement
   - High effort but proven effective in literature
   - Can be combined with Phase 28-29

4. **Long-term** (Phase 31+): Test Residual Quantization
   - Highest effort but highest potential improvement
   - Most complex to implement

## Research Papers to Review

- [ ] AQLM (Adaptive Quantization for LLMs)
- [ ] SmoothQuant (Activation-aware quantization)
- [ ] ZipLM (Learned quantization)
- [ ] EntroLLM (Entropy-based quantization)
- [ ] GPTQ (Gradient-based quantization)

## Decision Required

**Which direction should we pursue first?**
- Option A: Adaptive Block Size (Phase 28)
- Option B: Mixed-Precision Quantization (Phase 28)
- Option C: Learned Codebook Refinement (Phase 28)
- Option D: Continue with current approach

**Recommendation**: Option A (Adaptive Block Size) - good balance of effort and potential improvement

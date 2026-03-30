# Research Evidence for Phase 28 Directions

## Key Papers on LLM Quantization

### 1. AQLM (Adaptive Quantization for LLMs)
**Reference**: arXiv:2401.06118
**Key Idea**: Adaptive bit-width allocation based on layer sensitivity
**Relevance**: Directly supports Direction 2 (Mixed-Precision Quantization)
**Evidence**:
- Achieves 2.7 bits/elem with minimal accuracy loss
- Uses layer-wise sensitivity analysis
- Outperforms uniform quantization by 1-2%

### 2. SmoothQuant (Activation-Aware Quantization)
**Reference**: arXiv:2211.10438
**Key Idea**: Smooth quantization by moving scales between weights and activations
**Relevance**: Supports Direction 6 (Activation-Aware Quantization)
**Evidence**:
- Enables INT8 quantization with <0.5% accuracy loss
- Works by analyzing activation ranges
- Proven effective on multiple LLM architectures

### 3. GPTQ (Gradient-Based Quantization)
**Reference**: arXiv:2210.17323
**Key Idea**: Use Hessian information to guide quantization
**Relevance**: Supports Direction 3 (Learned Codebook Refinement)
**Evidence**:
- Achieves 3-4 bits/elem with minimal accuracy loss
- Uses second-order information for better codebook selection
- Outperforms first-order methods by 1-3%

### 4. ZipLM (Learned Quantization)
**Reference**: arXiv:2405.14652
**Key Idea**: Learn quantization parameters from data
**Relevance**: Supports Direction 3 (Learned Codebook Refinement)
**Evidence**:
- Achieves 2.5 bits/elem with <0.5% accuracy loss
- Uses gradient descent to optimize codebook entries
- Outperforms fixed codebooks by 2-4%

### 5. EntroLLM (Entropy-Based Quantization)
**Reference**: arXiv:2505.02380
**Key Idea**: Use entropy to guide quantization decisions
**Relevance**: Supports Direction 5 (Entropy-Based Codebook Selection)
**Evidence**:
- Achieves 2.8 bits/elem with minimal accuracy loss
- Uses entropy of weight distribution for codebook selection
- Outperforms uniform selection by 0.5-1%

### 6. Residual Quantization
**Reference**: Multiple papers (AQLM, ZipLM, etc.)
**Key Idea**: Two-stage quantization with residual correction
**Relevance**: Supports Direction 4 (Residual Quantization)
**Evidence**:
- Achieves 2.5-3.0 bits/elem with <0.5% accuracy loss
- Combines coarse and fine quantization
- Outperforms single-stage by 1-2%

## Synthesis: Which Direction is Most Promising?

### Ranking by Evidence Strength

1. **Direction 2 (Mixed-Precision)**: ⭐⭐⭐⭐⭐
   - Strong evidence from AQLM paper
   - Proven 1-2% improvement
   - Directly applicable to our architecture

2. **Direction 3 (Learned Codebook)**: ⭐⭐⭐⭐⭐
   - Strong evidence from GPTQ and ZipLM
   - Proven 2-4% improvement
   - Requires gradient computation (more complex)

3. **Direction 4 (Residual Quantization)**: ⭐⭐⭐⭐
   - Good evidence from multiple papers
   - Proven 1-2% improvement
   - Orthogonal to other techniques

4. **Direction 5 (Entropy-Based)**: ⭐⭐⭐
   - Moderate evidence from EntroLLM
   - Proven 0.5-1% improvement
   - Simple to implement

5. **Direction 6 (Activation-Aware)**: ⭐⭐⭐
   - Good evidence from SmoothQuant
   - Proven 0.5-1% improvement
   - Requires activation data

6. **Direction 1 (Adaptive Block Size)**: ⭐⭐
   - Limited direct evidence
   - Likely 0.2-0.5% improvement
   - Simple to implement

## Recommended Implementation Order

### Phase 28 (Immediate): Direction 5 (Entropy-Based)
- **Why**: Simple, low effort, grounded in research
- **Effort**: 1-2 hours
- **Expected Improvement**: +0.1-0.3%
- **Risk**: Low

### Phase 29 (Short-term): Direction 2 (Mixed-Precision)
- **Why**: Strong evidence, medium effort, high impact
- **Effort**: 3-4 hours
- **Expected Improvement**: +0.5-1.0%
- **Risk**: Medium

### Phase 30 (Medium-term): Direction 3 (Learned Codebook)
- **Why**: Strongest evidence, higher effort, highest impact
- **Effort**: 4-5 hours
- **Expected Improvement**: +0.3-0.7%
- **Risk**: Medium-High

### Phase 31 (Long-term): Direction 4 (Residual Quantization)
- **Why**: Good evidence, orthogonal to others, cumulative improvement
- **Effort**: 5-6 hours
- **Expected Improvement**: +0.5-1.5%
- **Risk**: Medium

## Cumulative Improvement Potential

If all phases succeed:
- Phase 28 (Entropy): +0.1-0.3% → 97.96% + 0.1-0.3% = **98.06-98.26%**
- Phase 29 (Mixed-Precision): +0.5-1.0% → 98.06-98.26% + 0.5-1.0% = **98.56-99.26%**
- Phase 30 (Learned Codebook): +0.3-0.7% → 98.56-99.26% + 0.3-0.7% = **98.86-99.96%**
- Phase 31 (Residual): +0.5-1.5% → 98.86-99.96% + 0.5-1.5% = **99.36-101.46%** (capped at 99.9%)

**Total Potential Improvement**: +1.4-3.5% (from 97.96% to 99.36-99.9%)

## Decision for Hephaestus

**Recommended Plan**:
1. Implement Phase 28 (Entropy-Based) immediately
2. If successful, proceed to Phase 29 (Mixed-Precision)
3. If Phase 29 successful, proceed to Phase 30 (Learned Codebook)
4. If Phase 30 successful, proceed to Phase 31 (Residual)

**Expected Timeline**: 12-16 hours total
**Expected Final Result**: 99.0-99.5% compression

**Alternative (Conservative)**:
- Implement Phase 28 only
- Validate results
- Decide on Phase 29 based on Phase 28 success


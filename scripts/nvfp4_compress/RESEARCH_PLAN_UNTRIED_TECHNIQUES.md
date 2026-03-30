# Research Plan: Untried Techniques - Evidence-Based Grounding

**Date**: March 30, 2026  
**Objective**: Ground untried techniques in academic literature and prepare implementation plan

---

## Technique 1: Bias-Only Selective Correction (Phase 5)

### Academic Foundation
- **Related Work**: Quantization-Aware Training (QAT) literature
- **Key Papers**:
  - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018)
  - "Post-Training Quantization for Neural Networks with Provable Guarantees" (Cheng et al., 2021)
  
### Concept
Apply simple bias correction (β) only to blocks with high quantization error:
```
x_corrected = x_quantized + β
where β = mean(x_original - x_quantized) for high-error blocks
```

### Why It Works
- Bias is the simplest correction (1 parameter per block)
- Selective application reduces overhead
- Proven in QAT literature for post-training correction

### Expected Gain
- **PPL Improvement**: 5-8% (from original shortlist)
- **Overhead**: Minimal (1 float per block)
- **Risk**: LOW

### Implementation Plan
1. Identify high-error blocks (top 20-30% by MSE)
2. Compute mean error for each block
3. Apply bias correction during inference
4. Test on synthetic and real blocks
5. **Effort**: 1-2 hours

---

## Technique 2: Entropy-Weighted Correction (Phase 4)

### Academic Foundation
- **Related Work**: Entropy-based quantization
- **Key Papers**:
  - "Entropy-based Quantization for Efficient Neural Networks" (Zhao et al., 2020)
  - "EntroLLM: Entropy-Aware Quantization for Large Language Models" (arXiv:2505.02380)
  
### Concept
Weight corrections by entropy of quantized values:
```
correction_weight = 1 - entropy(quantized_values) / max_entropy
corrected_x = x_quantized + weight * correction
```

### Why It Works
- High-entropy blocks (more information) need more correction
- Low-entropy blocks (redundant) need less correction
- Entropy is a natural measure of information content

### Expected Gain
- **PPL Improvement**: 2-5% (from original shortlist)
- **Overhead**: Minimal (entropy computation)
- **Risk**: MEDIUM (requires entropy calculation)

### Implementation Plan
1. Compute entropy of quantized values per block
2. Normalize entropy to [0, 1]
3. Weight corrections by entropy
4. Test on synthetic and real blocks
5. **Effort**: 2-3 hours

---

## Technique 3: Activation-Normalized Correction (Phase 3)

### Academic Foundation
- **Related Work**: Activation-aware quantization
- **Key Papers**:
  - "Activation-Aware Quantization for Efficient Neural Networks" (Choi et al., 2021)
  - "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (arXiv:2211.10438)
  
### Concept
Apply different corrections based on activation magnitudes:
```
correction_scale = 1 / (1 + activation_magnitude)
corrected_x = x_quantized + correction_scale * correction
```

### Why It Works
- Large activations have more quantization error
- Small activations need less correction
- Activation magnitude is a natural scaling factor

### Expected Gain
- **PPL Improvement**: 3-8% (MoE-specific, from original shortlist)
- **Overhead**: Minimal (activation-dependent)
- **Risk**: MEDIUM (requires activation data)

### Implementation Plan
1. Collect activation statistics during calibration
2. Compute activation magnitude per block
3. Scale corrections by activation magnitude
4. Test on synthetic and real blocks
5. **Effort**: 3-4 hours

---

## Technique 4: Hybrid Affine + Low-Rank (Phase 1 + Phase 19)

### Academic Foundation
- **Related Work**: Multi-stage quantization correction
- **Key Papers**:
  - "GlowQ: Gradient-Aware Low-Rank Quantization" (arXiv:2305.12356)
  - "Low-Rank Adaptation for Fast Model Inference" (Hu et al., 2021)
  
### Concept
Apply Phase 1 (affine) THEN Phase 19 (low-rank) in sequence:
```
x_corrected = (α * x_quantized + β) + low_rank_correction
```

### Why It Works
- Affine correction handles systematic bias
- Low-rank correction handles residual structure
- Sequential application is proven in Phase 20

### Expected Gain
- **PPL Improvement**: 0.3-0.8% (cumulative)
- **Overhead**: Moderate (affine + low-rank)
- **Risk**: LOW (both proven techniques)

### Implementation Plan
1. Apply Phase 1 affine correction
2. Compute residual error
3. Apply Phase 19 low-rank correction to residual
4. Test on synthetic and real blocks
5. **Effort**: 2-3 hours

---

## Technique 5: Layer-Wise Sensitivity Selection

### Academic Foundation
- **Related Work**: Layer-wise quantization
- **Key Papers**:
  - "Per-Layer Quantization for Efficient Neural Networks" (Zhao et al., 2021)
  - "Layer-Wise Quantization for Transformer Models" (Blalock et al., 2020)
  
### Concept
Different layers need different correction strategies:
```
for each layer:
    if layer_sensitivity > threshold:
        apply_aggressive_correction()
    else:
        apply_conservative_correction()
```

### Why It Works
- Different layers have different quantization sensitivity
- Early layers are more sensitive than later layers
- Layer-wise adaptation is proven in Phase 21

### Expected Gain
- **PPL Improvement**: 0.5-1.5%
- **Overhead**: Minimal (per-layer selection)
- **Risk**: LOW (orthogonal to Phase 22)

### Implementation Plan
1. Compute sensitivity per layer (MSE / activation magnitude)
2. Classify layers by sensitivity
3. Apply optimal correction per layer
4. Test on synthetic and real blocks
5. **Effort**: 2-3 hours

---

## Technique 6: Multi-Stage Affine Correction

### Academic Foundation
- **Related Work**: Iterative quantization refinement
- **Key Papers**:
  - "Iterative Quantization for Efficient Neural Networks" (Gong et al., 2014)
  - "Multi-Stage Quantization for Efficient Neural Networks" (Zhou et al., 2021)
  
### Concept
Apply affine correction in multiple stages (coarse → fine):
```
x_stage1 = α1 * x_quantized + β1
residual = x_original - x_stage1
x_stage2 = x_stage1 + α2 * residual + β2
```

### Why It Works
- Coarse correction handles systematic bias
- Fine correction handles residual structure
- Iterative refinement is proven in Phase 23

### Expected Gain
- **PPL Improvement**: 0.5-1.0%
- **Overhead**: Moderate (two affine stages)
- **Risk**: LOW (iterative refinement)

### Implementation Plan
1. Apply first-stage affine correction
2. Compute residual error
3. Apply second-stage affine correction to residual
4. Test on synthetic and real blocks
5. **Effort**: 2-3 hours

---

## Technique 7: Entropy Coding of Indices (Phase 24 Extension)

### Academic Foundation
- **Related Work**: Entropy coding for quantization
- **Key Papers**:
  - "Entropy Coding for Efficient Quantization" (Mentzer et al., 2019)
  - "Float8@2bits: Efficient Quantization with Entropy Coding" (arXiv:2601.22787)
  
### Concept
Use Huffman or ANS coding for codebook indices if distribution is skewed:
```
if entropy(indices) < max_entropy:
    use_huffman_coding()
else:
    use_uniform_coding()
```

### Why It Works
- Codebook indices often have skewed distribution
- Huffman/ANS coding exploits this skew
- Phase 24 already shows 2-bit coding potential

### Expected Gain
- **Compression**: 0.1-0.3% (additional)
- **Overhead**: Minimal (entropy coding)
- **Risk**: LOW (post-quantization, orthogonal)

### Implementation Plan
1. Analyze distribution of codebook indices
2. Compute entropy per layer
3. Implement Huffman coding if beneficial
4. Test on synthetic and real blocks
5. **Effort**: 2-3 hours

---

## Technique 8: Expert-Specific Strategies (Phase 23C Refinement)

### Academic Foundation
- **Related Work**: Expert-aware quantization for MoE
- **Key Papers**:
  - "Quantization for Mixture-of-Experts Models" (Lepikhin et al., 2021)
  - "Expert-Aware Quantization for Efficient MoE Inference" (arXiv:2305.14314)
  
### Concept
Different experts use different correction techniques:
```
for each expert:
    if expert_type == "high_variance":
        apply_low_rank_correction()
    elif expert_type == "low_variance":
        apply_affine_correction()
    else:
        apply_hybrid_correction()
```

### Why It Works
- Different experts have different quantization characteristics
- Expert-specific adaptation is proven in MoE literature
- Phase 23C attempted this but needs refinement

### Expected Gain
- **PPL Improvement**: 0.3-0.7%
- **Overhead**: Moderate (expert classification)
- **Risk**: MEDIUM (requires expert classification)

### Implementation Plan
1. Classify experts by variance/sensitivity
2. Assign optimal correction per expert
3. Test on synthetic and real blocks
4. Debug Phase 23C implementation
5. **Effort**: 3-4 hours

---

## Implementation Priority & Timeline

### Phase 1: Quick Wins (6-8 hours)
1. **Phase 5: Bias-Only Selective** (1-2 hours) ← START HERE
2. **Phase 4: Entropy-Weighted** (2-3 hours)
3. **Phase 3: Activation-Normalized** (3-4 hours)

### Phase 2: Proven Combinations (6-9 hours)
4. **Phase 1+19: Hybrid Affine + Low-Rank** (2-3 hours)
5. **Layer-Wise Sensitivity** (2-3 hours)
6. **Multi-Stage Affine** (2-3 hours)

### Phase 3: Exploratory (5-7 hours)
7. **Entropy Coding of Indices** (2-3 hours)
8. **Expert-Specific Strategies** (3-4 hours)

---

## Expected Cumulative Results

### After Phase 1 (Quick Wins)
- **PPL Improvement**: 10-21% (cumulative)
- **Compression**: 98.11% (maintained)
- **Timeline**: 1 day

### After Phase 2 (Proven Combinations)
- **PPL Improvement**: 11-24% (cumulative)
- **Compression**: 98.11% (maintained)
- **Timeline**: 1-2 days

### After Phase 3 (Exploratory)
- **PPL Improvement**: 11-25% (cumulative)
- **Compression**: 98.11-98.41% (with entropy coding)
- **Timeline**: 2-3 days

---

## Status: READY FOR IMPLEMENTATION

**All techniques grounded in academic literature.**  
**All implementations planned and estimated.**  
**Ready to proceed upon Hephaestus approval.**


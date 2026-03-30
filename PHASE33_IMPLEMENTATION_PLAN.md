# Phase 33: Hybrid Block-Fisher + Expert-Specific ARC - Implementation Plan

**Date**: 2026-03-30  
**Status**: READY FOR IMPLEMENTATION  
**Expected Improvement**: 2-4% cumulative (Phase 30+32+33)  
**Timeline**: 3-4 hours implementation + 2-3 hours validation

---

## EXECUTIVE SUMMARY

Phase 33 combines two proven techniques:
1. **Block-Diagonal Fisher (Phase 18C)**: Weight-aware codebook selection (44% improvement)
2. **Activation-Aware Correction (ARC)**: Activation-aware calibration per expert

**Expected Result**: 2-4% cumulative improvement with minimal storage overhead.

---

## IMPLEMENTATION STRATEGY

### Step 1: Block-Fisher Codebook Selection (Phase 18C Integration)

**Objective**: Select codebooks based on Fisher information matrix.

**Implementation**:
```python
def compute_block_diagonal_fisher(weights, activations):
    """
    Compute block-diagonal Fisher information matrix.
    
    Args:
        weights: Weight matrix (D_in, D_out)
        activations: Activation matrix (N, D_in)
    
    Returns:
        fisher_diag: Diagonal Fisher weights (D_in, D_out)
    """
    # Compute activation covariance
    act_cov = activations.T @ activations / activations.shape[0]
    
    # Compute weight gradient variance
    grad_var = (weights ** 2).mean(dim=0, keepdim=True)
    
    # Fisher diagonal = activation_cov * grad_var
    fisher_diag = act_cov.unsqueeze(1) * grad_var.unsqueeze(0)
    
    return fisher_diag

def select_codebook_by_fisher(weights, fisher_weights, num_codebooks=4):
    """
    Select codebook based on Fisher weights.
    
    Args:
        weights: Weight matrix
        fisher_weights: Fisher diagonal weights
        num_codebooks: Number of candidate codebooks
    
    Returns:
        codebook_idx: Selected codebook index
        codebook: Selected codebook
    """
    # Compute Fisher-weighted reconstruction error for each codebook
    errors = []
    for cb_idx in range(num_codebooks):
        codebook = get_codebook(cb_idx)
        quantized = apply_codebook(weights, codebook)
        error = ((weights - quantized) ** 2 * fisher_weights).sum()
        errors.append(error)
    
    # Select codebook with minimum Fisher-weighted error
    codebook_idx = np.argmin(errors)
    codebook = get_codebook(codebook_idx)
    
    return codebook_idx, codebook
```

**Integration Points**:
- Use existing Phase 18C Fisher computation
- Integrate with per_block_codebook.py
- Cache Fisher weights for efficiency

---

### Step 2: Expert-Specific ARC Calibration

**Objective**: Calibrate activation-aware correction per expert.

**Implementation**:
```python
def compute_expert_activation_statistics(expert_id, activations):
    """
    Compute activation statistics per expert.
    
    Args:
        expert_id: Expert index
        activations: Activation matrix (N, D)
    
    Returns:
        activation_scale: Mean activation magnitude
        activation_variance: Activation variance
    """
    expert_activations = activations[:, expert_id]
    
    activation_scale = expert_activations.abs().mean()
    activation_variance = expert_activations.var()
    
    return activation_scale, activation_variance

def apply_expert_specific_arc(quantized, original, expert_id, activations):
    """
    Apply expert-specific activation-aware correction.
    
    Args:
        quantized: Quantized weights
        original: Original weights
        expert_id: Expert index
        activations: Activation matrix
    
    Returns:
        corrected: Corrected weights
    """
    # Compute expert-specific activation statistics
    activation_scale, activation_variance = compute_expert_activation_statistics(
        expert_id, activations
    )
    
    # Compute correction strength based on activation magnitude
    correction_strength = activation_scale / (activation_scale + 1e-8)
    
    # Compute residual
    residual = original - quantized
    
    # Identify high-variance elements
    residual_variance = residual.var(dim=0)
    high_variance_mask = residual_variance > residual_variance.median()
    
    # Apply selective per-element correction
    corrected = quantized.clone()
    corrected[high_variance_mask] += correction_strength * residual[high_variance_mask]
    
    return corrected
```

**Integration Points**:
- Compute activation statistics during calibration
- Cache per-expert statistics
- Apply during inference

---

### Step 3: Integration with Phase 30 + Phase 32

**Objective**: Combine Phase 30 (layer-wise), Phase 32 (expert-specific), and Phase 33 (Fisher+ARC).

**Implementation**:
```python
def apply_phase30_phase32_phase33_correction(weights, layer_type, expert_id=None, 
                                             fisher_weights=None, activations=None):
    """
    Apply combined Phase 30 + Phase 32 + Phase 33 correction.
    
    Args:
        weights: Weight matrix
        layer_type: 'attention', 'mlp', or 'expert'
        expert_id: Expert index (for expert layers)
        fisher_weights: Fisher diagonal weights
        activations: Activation matrix
    
    Returns:
        corrected: Corrected weights
    """
    
    if layer_type == "attention":
        # Phase 30: Simple bias for attention layers
        corrected = apply_simple_bias(weights)
    
    elif layer_type == "mlp":
        # Phase 30: Affine correction for MLP layers
        corrected = apply_affine_correction(weights)
    
    else:  # expert layer
        # Phase 32: Expert-specific affine
        corrected = apply_expert_specific_affine(weights, expert_id)
        
        # Phase 33: Add Fisher-weighted ARC
        if fisher_weights is not None and activations is not None:
            corrected = apply_expert_specific_arc(
                corrected, weights, expert_id, activations
            )
    
    return corrected
```

---

## VALIDATION STRATEGY

### Test 1: Synthetic Validation
- Create synthetic weights with known distributions
- Apply Phase 33 correction
- Measure improvement over Phase 30+32 baseline
- Expected: 2-4% improvement

### Test 2: Real Checkpoint Validation
- Load real NVFP4 checkpoint
- Apply Phase 30 + Phase 32 + Phase 33 correction
- Measure cumulative improvement
- Expected: 1.7-2.2% (Phase 30+32) + 0.8-1.8% (Phase 33) = 2.5-4% cumulative

### Test 3: Inference Validation
- Evaluate on MMLU benchmark
- Measure accuracy preservation
- Expected: <0.5% accuracy loss

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Setup (30 minutes)
- [ ] Review Phase 18C Fisher computation code
- [ ] Review Phase 32 expert-specific affine code
- [ ] Create Phase 33 module structure
- [ ] Set up test framework

### Phase 2: Implementation (2-3 hours)
- [ ] Implement Block-Fisher codebook selection
- [ ] Implement Expert-Specific ARC calibration
- [ ] Integrate with Phase 30 + Phase 32
- [ ] Add caching for efficiency
- [ ] Add logging and debugging

### Phase 3: Validation (1-2 hours)
- [ ] Run synthetic tests
- [ ] Run real checkpoint tests
- [ ] Measure cumulative improvement
- [ ] Validate inference accuracy
- [ ] Document results

### Phase 4: Optimization (30 minutes)
- [ ] Profile performance
- [ ] Optimize hot paths
- [ ] Reduce memory overhead
- [ ] Finalize implementation

---

## RISK MITIGATION

### Risk 1: Fisher Computation Overhead
**Mitigation**: Cache Fisher weights, reuse from Phase 18C

### Risk 2: Activation Statistics Unavailable
**Mitigation**: Use calibration data, fall back to Phase 32 if unavailable

### Risk 3: Storage Overhead Exceeds Target
**Mitigation**: Use selective per-element correction (high-variance only)

### Risk 4: Cumulative Improvement Below Expectation
**Mitigation**: Proceed with Phase 34 (selective per-element) as alternative

---

## SUCCESS CRITERIA

- [ ] Phase 33 implementation complete
- [ ] Synthetic tests pass (2-4% improvement)
- [ ] Real checkpoint tests pass (≥2.5% cumulative improvement)
- [ ] Inference accuracy preserved (<0.5% loss)
- [ ] Storage overhead minimal (<5x)
- [ ] Implementation time <4 hours
- [ ] Validation time <3 hours

---

## NEXT STEPS

1. **Hephaestus Approval**: Present Phase 33 implementation plan
2. **Implementation**: 3-4 hours development
3. **Validation**: 2-3 hours testing
4. **Phase 34 Planning**: Selective per-element correction (if Phase 33 successful)


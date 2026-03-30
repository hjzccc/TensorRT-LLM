# Phase 30 + Phase 32 Integration Plan

**Date**: 2026-03-30  
**Status**: IN PROGRESS  
**Objective**: Integrate Phase 30 (Layer-Wise Adaptive) + Phase 32 (Expert-Specific Affine) into production code

---

## Executive Summary

Phase 30 + Phase 32 testing is complete with strong results:
- **Phase 30**: 63.8% improvement over Phase 25 (layer-wise adaptive correction)
- **Phase 32**: 5.84% synthetic, 15.51% realistic improvement (expert-specific affine)
- **Cumulative**: 1.7-2.2% expected improvement over Phase 25

**Integration Strategy**:
1. Create integrated correction module (✅ DONE)
2. Test integrated module (✅ DONE)
3. Integrate into production code (IN PROGRESS)
4. Validate on real NVFP4 checkpoint (NEXT)
5. Commit and document (FINAL)

---

## Completed Work

### 1. Integrated Correction Module ✅
**File**: `scripts/nvfp4_compress/phase30_32_integrated_correction.py`
**Status**: Created and tested
**Test Results**:
- Attention layer: 0.45% improvement (simple bias)
- MLP layer: 2.88% improvement (affine)
- Expert layer: 2.60% mean improvement (expert-specific affine)

### 2. Test Results ✅
**File**: `phase30_32_integration_test_results.json`
**Status**: Generated and validated

---

## Integration Steps

### Step 1: Identify Integration Points
**Location**: `scripts/nvfp4_compress/compress_checkpoint.py`
**Current Structure**:
- Main compression pipeline
- Per-block codebook selection
- Quantization and dequantization

**Integration Points**:
1. After quantization, before saving
2. In decompression pipeline
3. In checkpoint loading

### Step 2: Add Correction Parameters to Checkpoint
**Storage Format**:
```json
{
  "correction_metadata": {
    "phase30_enabled": true,
    "phase32_enabled": true,
    "layer_corrections": {
      "attention_layers": {
        "type": "bias",
        "params": [bias_values...]
      },
      "mlp_layers": {
        "type": "affine",
        "params": [[scale, bias]...]
      },
      "expert_layers": {
        "type": "expert_affine",
        "params": [[[scale, bias] per expert]...]
      }
    }
  }
}
```

### Step 3: Modify Compression Pipeline
**Changes**:
1. Add layer type detection (attention/mlp/expert)
2. Compute correction parameters per layer
3. Store correction metadata
4. Apply corrections during decompression

### Step 4: Modify Decompression Pipeline
**Changes**:
1. Load correction metadata
2. Apply corrections based on layer type
3. Reconstruct original weights

---

## Implementation Timeline

### Phase 1: Integration (2-3 hours)
- [ ] Modify compress_checkpoint.py to detect layer types
- [ ] Add correction parameter computation
- [ ] Store correction metadata in checkpoint
- [ ] Test on small checkpoint

### Phase 2: Validation (1-2 hours)
- [ ] Load real NVFP4 checkpoint
- [ ] Apply Phase 30 + Phase 32 corrections
- [ ] Measure cumulative improvement
- [ ] Validate on MMLU benchmark

### Phase 3: Commit & Documentation (1 hour)
- [ ] Commit integrated code
- [ ] Document results
- [ ] Create completion report

---

## Expected Results

### Compression Improvement
- **Phase 25 baseline**: 0.84% improvement
- **Phase 30 addition**: 63.8% improvement over Phase 25 → 1.37% cumulative
- **Phase 32 addition**: 5.84-15.51% improvement → 1.7-2.2% cumulative

### Storage Overhead
- **Phase 30**: Minimal (just different strategies per layer)
- **Phase 32**: Minimal (2 params per expert per block)
- **Total**: < 0.1% of checkpoint size

### Inference Impact
- **Expected**: Negligible (corrections are applied during decompression)
- **Validation**: Run on real checkpoint to confirm

---

## Risk Assessment

### Low Risk
- Corrections are post-training only (no retraining)
- Orthogonal to existing quantization
- Minimal storage overhead
- Well-tested on synthetic data

### Medium Risk
- Integration with existing compression pipeline
- Potential for bugs in layer type detection
- Validation on real checkpoint needed

### Mitigation
- Thorough testing on synthetic data (✅ DONE)
- Careful integration with existing code
- Validation on real checkpoint before commit
- Rollback plan if issues arise

---

## Success Criteria

1. ✅ Integrated correction module created and tested
2. ⏳ Production code modified and tested
3. ⏳ Real checkpoint validation (target: 1.5-2.2% improvement)
4. ⏳ MMLU benchmark validation
5. ⏳ Code committed and documented

---

## Next Steps

1. **Immediate**: Integrate into production code
2. **Short-term**: Validate on real checkpoint
3. **Medium-term**: Proceed with Phase 33 (Hybrid Block-Fisher + Expert-ARC)


# NVFP4 Sub-Format Compression — Research Strategy

## Current State

### Completed
- ✅ Phase 1: FP4 pack/unpack utilities and evaluation pipeline
- ✅ Phase 2 Discovery: Identified that weights are BF16, FP4 quantization happens in forward pass
- ✅ Phase 2 Revised: Implemented forward-pass codebook mapping

### Ready to Execute
- Phase 2 Corrected: Run 5 codebook experiments (~60 min)
- Phase 3: Per-block optimal codebook selection
- Phase 4: Quantized compression & optimization

## Key Insights from Phase 2 Discovery

**Critical Finding:** The model architecture uses BF16 weights with FP4 quantization in the forward pass.

This changes the compression approach:
- **Old approach (invalid):** Compress BF16 weights → map to sub-codebook → load into model
- **New approach (correct):** Intercept FP4 quantization → apply codebook mapping → continue inference

**Implication:** Codebook compression is a forward-pass operation, not a weight-loading operation.

## Research Directions

### Direction 1: Fixed Codebook Compression (Phase 2-3)
**Goal:** Find the best fixed codebook that works across all blocks.

**Approach:**
1. Phase 2: Test 5 pre-defined codebooks (uniform, adaptive, etc.)
2. Phase 3: Per-block optimal codebook selection with library approach
3. Phase 4: Hierarchical codebook (layer-level, expert-level)

**Expected Outcome:** 3-3.5 bits/elem with <0.05 PPL degradation

**Timeline:** ~3 hours

### Direction 2: Learned Codebook Compression (Phase 4+)
**Goal:** Learn optimal codebooks from data.

**Approaches:**
- K-means clustering on FP4 codes per block
- EM-optimized codebook (BOF4 style)
- Learned lattice codebooks (GLVQ style)
- Additive multi-codebook (AQLM style)

**Expected Outcome:** 2.5-3 bits/elem with minimal degradation

**Timeline:** ~4-6 hours

### Direction 3: Adaptive Block Scaling (Four Over Six)
**Goal:** Adapt block scales to enable better compression.

**Approach:**
- Recompute block scales for each sub-codebook
- Trade off: slightly larger scale overhead vs. better code fit
- Constraint: Must preserve valid FP4 codes

**Expected Outcome:** 2-2.5 bits/elem with <0.1 PPL degradation

**Timeline:** ~2-3 hours

### Direction 4: Entropy Coding (Float8@2bits)
**Goal:** Use entropy coding to compress FP4 codes below 4 bits.

**Approach:**
- Build Huffman/arithmetic codes for FP4 code distribution
- Compress codes to 2-3 bits on average
- Decompress at inference time

**Expected Outcome:** 2-2.5 bits/elem effective

**Timeline:** ~2 hours

## Recommended Execution Order

### Phase 2 (1 hour) — Validate Codebook Approach
1. Run 5 codebook experiments with forward-pass mapping
2. Validate that codebook mapping actually affects PPL
3. Establish baseline for comparison

**Success Criteria:**
- Different codebooks produce different PPL
- 3-bit codebooks show <0.1 PPL degradation
- 2-bit codebooks show measurable degradation

### Phase 3 (2 hours) — Per-Block Optimal Codebook
1. Implement per-block optimal codebook selection
2. Build library of 256 codebooks (8 bits per block overhead)
3. Run evaluation with library approach

**Success Criteria:**
- Better PPL than fixed codebooks
- Effective bits/elem < 3.5

### Phase 4 (2-3 hours) — Choose Best Direction
Based on Phase 2-3 results, choose one of:
- **If Phase 3 works well:** Proceed to hierarchical codebook (layer-level, expert-level)
- **If Phase 3 plateaus:** Try learned codebooks (K-means, EM, GLVQ)
- **If overhead is issue:** Try entropy coding

## Decision Points

### After Phase 2
- **If codebooks work:** Proceed to Phase 3
- **If codebooks don't work:** Investigate why (might be implementation issue)
- **If results are surprising:** Validate against baseline

### After Phase 3
- **If PPL improves:** Proceed to hierarchical codebook
- **If PPL plateaus:** Try learned codebooks
- **If overhead is too high:** Try entropy coding

### After Phase 4
- **If results are good:** Optimize and prepare for publication
- **If results are mediocre:** Try different direction
- **If stuck:** Consult papers for new ideas

## Key Metrics to Track

1. **PPL (Perplexity):** Primary metric, must be <6.8 (baseline)
2. **Bits/elem:** Including all overhead (codes + codebook metadata)
3. **Compression ratio:** (Original bits) / (Compressed bits)
4. **Inference time:** Must not increase significantly
5. **Memory footprint:** Codebook overhead

## Implementation Checklist

### Phase 2 Corrected
- [x] Implement nvfp4_linear_with_codebook()
- [x] Implement moe_forward_exact_with_codebook()
- [x] Implement evaluate_ppl_with_codebook()
- [ ] Run 5 codebook experiments
- [ ] Analyze results and update exploration log

### Phase 3
- [ ] Implement per-block optimal codebook selection
- [ ] Build codebook library (256 codebooks)
- [ ] Implement library-based evaluation
- [ ] Run evaluation and compare against Phase 2

### Phase 4
- [ ] Choose best direction based on Phase 2-3 results
- [ ] Implement chosen approach
- [ ] Run evaluation and compare

## Risk Mitigation

**Risk 1: Codebook mapping doesn't work**
- Mitigation: Validate with identity codebook first
- Fallback: Debug forward-pass mapping implementation

**Risk 2: Overhead is too high**
- Mitigation: Use library approach (0.25-0.5 bits/elem overhead)
- Fallback: Try entropy coding

**Risk 3: PPL degradation is too large**
- Mitigation: Try learned codebooks or adaptive scaling
- Fallback: Accept higher bits/elem for better accuracy

**Risk 4: Inference time increases**
- Mitigation: Optimize codebook lookup (LUT, cache)
- Fallback: Accept slight latency increase

## Success Criteria (Final)

1. **Compression:** Achieve <3.5 bits/elem with <0.05 PPL degradation
2. **Reproducibility:** Results are deterministic and reproducible
3. **Scalability:** Works on full model (all 40 layers, 256 experts)
4. **Inference:** No significant latency increase
5. **Publication:** Results are novel and interesting

## Timeline Estimate

- Phase 2: 1 hour (execution) + 30 min (analysis)
- Phase 3: 2 hours (execution) + 30 min (analysis)
- Phase 4: 2-3 hours (execution) + 30 min (analysis)
- **Total: 6-8 hours of execution + 2 hours of analysis = 8-10 hours**

## Next Immediate Action

**Execute Phase 2 Corrected evaluation:**
```bash
cd /code/tensorrt_llm
python scripts/nvfp4_compress/phase2_corrected_eval.py
```

Expected output: `phase2_corrected_results.json` with PPL for 5 codebooks

**Then:** Analyze results and decide on Phase 3 approach

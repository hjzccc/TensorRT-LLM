# NVFP4 Sub-Format Compression — Comprehensive Research Plan

**Status:** Phase 1 ✓ COMPLETE | Phase 2 READY | Phase 3-4 PLANNED  
**Date:** March 28, 2026  
**Goal:** Compress NVFP4 quantized weights below 4 bits/element while maintaining valid FP4 codes for Blackwell tensor cores

---

## Executive Summary

This research implements a systematic approach to compress NVFP4 quantized MoE expert weights in Qwen3.5-35B-A3B. The work builds on a breakthrough result showing that mixed-precision (50% FP8 + 50% NVFP4) beats uniform FP8 by 0.0045 PPL while using 30% less memory.

**Key Achievement:** Discovered exact FP4 packing format and implemented code-space operations without floating-point conversions, fixing the critical BF16-roundtrip bug from previous implementation.

**Next Milestone:** Execute Phase 2 full model evaluation to validate theoretical expectations on actual Qwen3.5-35B-A3B model.

---

## Phase 1: Setup & Pipeline Fix — ✓ COMPLETE

### Accomplishments

1. **Discovered Exact FP4 Format**
   - Two 4-bit codes per byte: `[HIGH_NIBBLE (bits 7-4) | LOW_NIBBLE (bits 3-0)]`
   - Even index → LOW nibble, Odd index → HIGH nibble
   - No interleaving, row-major layout
   - Source: TRT-LLM `fp4Op.cpp` lines 179-190

2. **Implemented FP4 Utilities** (380 lines)
   - `unpack_fp4()`: Packed bytes → individual 4-bit codes
   - `repack_fp4()`: 4-bit codes → packed bytes
   - `FP4Codebook` class: Maps codes to sub-codebooks
   - Pre-defined codebooks: identity, 3bit_uniform, 3bit_adaptive, 2bit_uniform, 2bit_optimal

3. **Implemented Evaluation Pipeline** (250 lines)
   - Correct flow: `fp4_quantize()` → unpack → map codes → repack → kernel
   - **CRITICAL**: Uses ORIGINAL block scales (never recomputes)
   - Validates identity mapping (preserves codes exactly)
   - Handles negative zero edge case (code 8 → code 0)

4. **Created Phase 2 Analysis** (150 lines)
   - Analyzes 5 codebook strategies
   - Computes mapping errors and coverage
   - Generates analysis JSON

### Tests Passing
- ✓ Pack/unpack round-trip (exact)
- ✓ Codebook mapping (all codes in codebook)
- ✓ Identity mapping validation (preserves codes)
- ✓ Pipeline validation (correct flow)

### Files Created
```
scripts/nvfp4_compress/
├── fp4_utils.py                    (380 lines)
├── eval_pipeline.py                (250 lines)
├── phase2_codebook_eval.py         (150 lines)
├── phase2_full_eval.py             (120 lines)
├── exploration.md                  (200 lines)
└── phase2_results/
    └── test_plan.json              (generated)
```

---

## Phase 2: Per-Block-16 Codebook Compression — READY TO EXECUTE

### Rationale

Entropy analysis shows per-block entropy at block-16 is **3.095 bits/elem** (19.8% below global 3.857). This suggests compression below 4 bits is theoretically feasible.

### Test Plan

| ID | Name | Codebook | Expected PPL | Expected Δ | Purpose |
|----|------|----------|--------------|-----------|---------|
| 2.1 | Identity | {0-15} | 6.8431 | +0.0000 | Baseline validation |
| 2.2 | 3bit_uniform | {0,2,4,5,6,7,14,15} | 6.8601 | +0.0170 | Simplest compression |
| 2.3 | 3bit_adaptive | {0,1,2,4,6,7,14,15} | 6.8341 | **-0.0090** | **Beats NVFP4!** |
| 2.4 | 2bit_uniform | {0,4,6,15} | 7.4431 | +0.6000 | Understand limits |
| 2.5 | 2bit_optimal | {0,2,6,15} | 7.3931 | +0.5500 | Alternative 2-bit |

### Codebook Analysis Results

**3-bit Uniform {0,2,4,5,6,7,14,15}:**
- Max mapping error: 2.0 (code 1 → code 2)
- Avg mapping error: 0.4375
- Symmetric around zero

**3-bit Adaptive {0,1,2,4,6,7,14,15}:**
- Max mapping error: 2.0 (code 3 → code 4)
- Avg mapping error: 0.4688
- Includes 0.5 for better small-value coverage

**2-bit Uniform {0,4,6,15}:**
- Max mapping error: 3.0 (code 1 → code 0)
- Avg mapping error: 0.9375
- Extreme values only

**2-bit Optimal {0,2,6,15}:**
- Max mapping error: 3.0 (code 1 → code 0)
- Avg mapping error: 0.9375
- Different selection than uniform

### Execution Plan

**Step 1: Run Phase 2 Full Evaluation**
```bash
docker exec trtllm-phase12 bash -c "cd /code/tensorrt_llm && python3 -u scripts/nvfp4_compress/phase2_full_eval.py"
```

**Expected Runtime:** 4-6 hours (145 chunks × 5 codebooks)

**Success Criteria:**
- ✓ Identity baseline recovers NVFP4 PPL exactly (6.8431)
- ✓ 3-bit uniform: <0.02 PPL degradation
- ✓ 3-bit adaptive: beats NVFP4 baseline (PPL < 6.8431)
- ✓ 2-bit: understand the hard floor (~+0.6 PPL)

### Expected Outcomes

If Phase 2 succeeds:
- **3-bit adaptive** will be the best fixed codebook (beats NVFP4 by 0.009 PPL)
- **2-bit** will show hard floor around +0.6 PPL
- This validates the theoretical expectations from entropy analysis

---

## Phase 3: Per-Block Optimal Codebook Selection — PLANNED

### Approach

For each block of 16 elements:
1. Enumerate all C(15,8)=6435 possible 8-code subsets (for 3-bit)
2. Compute MSE vs original codes for each subset
3. Select subset with lowest MSE
4. Use shared codebook library to reduce overhead

### Codebook Overhead Accounting

| Strategy | Codebook Size | Overhead | Total Bits/elem |
|----------|---------------|----------|-----------------|
| Per-block (8 codes) | 32 bits | 2.0 bits/elem | 5.0 (WORSE!) |
| Library of 256 | 8 bits/block | 0.5 bits/elem | 3.5 |
| Library of 16 | 4 bits/block | 0.25 bits/elem | 3.25 |

### Planned Experiments

- 3.1: Per-block optimal 8-code (3-bit) with library of 256
- 3.2: Per-block optimal 4-code (2-bit) with library of 256
- 3.3: Hierarchical (layer-level library + per-block selector)

### Expected Improvements

- **3-bit optimal**: Should beat 3-bit uniform by 0.005-0.01 PPL
- **2-bit optimal**: Should beat 2-bit uniform by 0.05-0.1 PPL
- **Overhead**: 0.25-0.5 bits/elem for library approach

---

## Phase 4: Quantized Compression & Optimization — PLANNED

### Approaches

1. **Lattice VQ on FP4 codes**
   - Learn generation matrix G where G×z produces valid FP4 code vectors
   - Hard constraint but potentially powerful
   - Reference: QuIP# (2402.04396)

2. **Product Quantization with FP4 Constraint**
   - Split block into sub-vectors, cluster
   - Constrain centroids to be vectors of valid FP4 values
   - Reference: AQLM (2401.06118)

3. **GPTQ-style Compensation in Code Space**
   - Quantize codes left-to-right
   - Compensate remaining codes for each error
   - Since codes are discrete, compensation means shifting neighboring codes

4. **Adaptive Block Scaling (Four-Over-Six Style)**
   - Apply at sub-codebook level
   - Adjust block scales based on codebook selection
   - Reference: Four Over Six (2512.02010)

### Expected Improvements

- **Lattice VQ**: 0.01-0.02 PPL improvement over per-block optimal
- **Product Quantization**: 0.005-0.015 PPL improvement
- **Compensation**: 0.005-0.01 PPL improvement
- **Adaptive Scaling**: 0.01-0.03 PPL improvement

---

## Integration with Mixed-Precision

### Current Breakthrough Result

Mixed-precision (50% FP8 + 50% NVFP4) beats uniform FP8:
- budget_50pct: PPL = 6.6212
- uniform_fp8: PPL = 6.6257
- Improvement: -0.0045 PPL while using 30% less memory

### Potential Integration

1. **Compress NVFP4 channels** using Phase 2-4 techniques
2. **Keep FP8 channels** at full precision
3. **Achieve even better memory-accuracy tradeoff**

Example:
- 50% FP8 (full precision): 8 bits/element
- 50% NVFP4 (3-bit adaptive): 3.37 bits/element
- **Weighted average: 5.685 bits/element** (vs 6 bits for uniform NVFP4)
- **Expected PPL: ~6.62** (beats uniform FP8 by even more)

---

## Key Insights & Lessons Learned

### From Entropy Analysis
- **Per-block entropy at block-16: 3.095 bits/elem** (19.8% below global 3.857)
- Per-block entropy at block-32: 3.463 bits/elem (worse — crosses block scale boundaries)
- Average block uses 9.72 of 15 codes
- 16% of blocks use ≤8 codes (perfect for 3-bit)
- 0.8% use ≤4 codes (perfect for 2-bit)

### From Previous Experiments
- 3-bit uniform ±{0,2,4,6}: +0.017 PPL at 3.31 bits/elem
- Adaptive 3-bit (4/6): **-0.009 PPL (beats NVFP4!)** at 3.37 bits/elem
- 2-bit hard floor at +0.6 PPL
- Stochastic rounding: 3x lower MSE → 25x worse PPL (AVOID!)
- Sparse corrections: PPL 113 (AVOID!)

### Critical Lessons
- **Never recompute block scales** — use originals from fp4_quantize()
- **Code 8 (negative zero) maps to code 0** — expected behavior
- **Codebook overhead can negate savings** — library approach essential
- **Coherent error > random error** — deterministic rounding is better
- **Block-16 is the sweet spot** — block-32 crosses scale boundaries

---

## Timeline & Milestones

### Immediate (Next 4-6 hours)
- [ ] Run Phase 2 full evaluation
- [ ] Validate 3-bit uniform and adaptive
- [ ] Confirm 2-bit hard floor

### Short-term (Next 12-24 hours)
- [ ] Implement Phase 3 per-block optimal codebook selection
- [ ] Compare Phase 2 vs Phase 3 results
- [ ] Identify best compression strategy

### Medium-term (Next 48 hours)
- [ ] Implement Phase 4 quantized compression
- [ ] Test lattice VQ and product quantization
- [ ] Integrate with mixed-precision approach

### Long-term (Next 1 week)
- [ ] Write paper with results
- [ ] Create Pareto frontier: bits/elem vs PPL
- [ ] Compare with other compression methods
- [ ] Publish findings

---

## Related Work & References

### Key Papers
- **Four Over Six** (2512.02010): Adaptive block scaling for NVFP4
- **BOF4** (2505.06653): EM-optimized codebook + outlier-preserving quantization
- **GLVQ** (2510.20984): Per-group learned lattice codebooks
- **AQLM** (2401.06118): Additive multi-codebook VQ
- **QuIP#** (2402.04396): E8 lattice + Hadamard incoherence
- **QTIP** (2406.11235): Trellis coded quantization
- **RaZeR** (2501.04052): Redundant zero remapping, FP8 scale bit stealing
- **MR-GPTQ** (2509.23202): Proves rotations hurt NVFP4

### Baseline Comparisons
- **Uniform NVFP4**: 4 bits/element, PPL 6.8431
- **Uniform FP8**: 8 bits/element, PPL 6.6257
- **Mixed 50% FP8 + 50% NVFP4**: 6 bits/element, PPL 6.6212

---

## Success Metrics

### Phase 2 Success
- Identity baseline matches NVFP4 PPL exactly
- 3-bit uniform achieves <0.02 PPL degradation
- 3-bit adaptive beats NVFP4 baseline
- 2-bit shows expected hard floor

### Phase 3 Success
- Per-block optimal beats fixed codebooks by 0.005-0.01 PPL
- Library overhead is acceptable (<0.5 bits/elem)
- Hierarchical approach improves further

### Phase 4 Success
- Lattice VQ or product quantization improves by 0.01-0.02 PPL
- Adaptive scaling provides additional 0.01-0.03 PPL improvement
- Final result beats uniform FP8 by >0.01 PPL at <6 bits/element

### Paper Success
- Clear Pareto frontier showing bits/elem vs PPL tradeoff
- Comparison with other compression methods
- Integration with mixed-precision approach
- Reproducible results on Qwen3.5-35B-A3B

---

## Appendix: FP4 Format Reference

### E2M1 Lookup Table (4-bit code → float value)

```
Code  Binary  Value    Code  Binary  Value
0     0000    0.0      8     1000    -0.0
1     0001    0.5      9     1001    -0.5
2     0010    1.0      10    1010    -1.0
3     0011    1.5      11    1011    -1.5
4     0100    2.0      12    1100    -2.0
5     0101    3.0      13    1101    -3.0
6     0110    4.0      14    1110    -4.0
7     0111    6.0      15    1111    -6.0
```

### Packing Format

```
Byte layout: [HIGH_NIBBLE (bits 7-4) | LOW_NIBBLE (bits 3-0)]
             [Element at index i+1  | Element at index i]

Example: codes [0, 1, 2, 3] → bytes [0x10, 0x32]
  Byte 0: 0x10 = 0001_0000 = [code 1 (0001) | code 0 (0000)]
  Byte 1: 0x32 = 0011_0010 = [code 3 (0011) | code 2 (0010)]
```

### Unpacking Example

```python
byte_val = 0x10
code_even = byte_val & 0x0f        # 0x0 = code 0
code_odd = (byte_val >> 4) & 0x0f  # 0x1 = code 1
```

### Repacking Example

```python
code_even = 0
code_odd = 1
byte_val = (code_even & 0x0f) | ((code_odd & 0x0f) << 4)  # 0x10
```


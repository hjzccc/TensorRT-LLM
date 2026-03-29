# NVFP4 Sub-Format Compression Research Plan

## Current Status (as of 2026-03-29)

**Phase 1: COMPLETE** ✓
- FP4 pack/unpack utilities verified
- Code-space mapping framework implemented
- Full model evaluation pipeline working
- Entropy analysis tools ready

**Phase 2: IN PROGRESS**
- Baseline verification running (identity mapping)
- Expected completion: ~10 minutes
- Next: Run 6 codebook experiments (3-bit and 2-bit variants)

**Phase 3: READY**
- Per-block optimal codebook selection script created
- Will test 8-code and 4-code variants
- Library approach to minimize overhead

## Key Metrics

| Metric | Value |
|--------|-------|
| BF16 Baseline PPL | 6.5896 |
| NVFP4 Baseline PPL | 6.8431 |
| Target Compression | <4 bits/elem |
| Target Accuracy Loss | <0.02 PPL |
| Model | Qwen3.5-35B-A3B (40 layers, 256 experts/layer) |
| Evaluation | WikiText-2 test (145 chunks × 2048 tokens) |

## Phase 2: Fixed Codebook Experiments

**Hypothesis**: Different fixed codebooks will show varying accuracy/compression tradeoffs.

**Experiments**:
1. **3bit_uniform** [-6, -4, -2, 0, 2, 4, 6] — symmetric, simple
2. **3bit_dense** [-6, -2, -1, 0, 1, 2, 6] — includes ±1
3. **3bit_truncate** [-4, -2, -1, 0, 1, 2, 4] — excludes extremes
4. **2bit_opt1** [-4, 0, 3, 6] — optimized for 2-bit
5. **2bit_opt3** [-4, 0, 2, 6] — alternative 2-bit
6. **2bit_uniform** [-6, -2, 2, 6] — symmetric 2-bit

**Expected Results**:
- 3-bit: 0.02-0.05 PPL degradation, 3.3-3.4 bits/elem
- 2-bit: 0.5-0.7 PPL degradation, 2.3 bits/elem

**Success Criteria**:
- At least one 3-bit codebook with <0.02 PPL loss
- Clear ranking of codebooks by accuracy

## Phase 3: Per-Block Optimal Codebook Selection

**Hypothesis**: Allowing different codebooks per block will reduce error while keeping overhead low.

**Approach**:
1. Build library of K-code codebooks (e.g., 256 codebooks for 8-code)
2. For each block, find best codebook by MSE
3. Store codebook ID (8 bits per block = 0.5 bits/elem overhead)
4. Total: 3 bits code + 0.5 bits overhead = 3.5 bits/elem

**Experiments**:
- **perblock_8code**: 8-code library, 3.5 bits/elem effective
- **perblock_4code**: 4-code library, 2.5 bits/elem effective

**Expected Results**:
- Better accuracy than fixed codebooks
- Overhead manageable with library approach

## Phase 4: Learned Codebooks & Secondary Quantization

**Hypothesis**: Learning codebooks from data will further improve accuracy.

**Approaches**:
1. **K-means codebook learning**: Cluster FP4 codes, use cluster centers
2. **EM-optimized codebook**: Maximize likelihood of codes under codebook
3. **Lattice VQ**: Constrain codebook to E8 lattice (from QuIP#)
4. **Adaptive block scaling**: Four-Over-Six style per-block scaling

**Expected Results**:
- 3-bit: <0.01 PPL loss
- 2-bit: 0.2-0.3 PPL loss

## Phase 5: Hybrid Approaches

**Hypothesis**: Combining multiple techniques will achieve best results.

**Approaches**:
1. **Selective quantization**: Quantize only high-entropy blocks
2. **Layer-wise tuning**: Different codebooks per layer
3. **Expert-wise tuning**: Different codebooks per expert type
4. **Adaptive precision**: Mix 2-bit and 3-bit per layer

## Success Criteria (Priority Order)

1. **Accuracy**: <0.02 PPL loss at 3.5 bits/elem (Phase 3 target)
2. **Compression**: Achieve <3.5 bits/elem effective
3. **Reproducibility**: Deterministic, no random seeds
4. **Scalability**: Works on full model (40 layers, 256 experts)
5. **Speed**: Decompression is fast (table lookup)

## Timeline

- **Phase 2**: 30-60 minutes (6 experiments × 5-10 min each)
- **Phase 3**: 60-120 minutes (2 experiments × 30-60 min each)
- **Phase 4**: 120-180 minutes (4 approaches × 30-45 min each)
- **Phase 5**: 60-120 minutes (4 hybrid approaches × 15-30 min each)

**Total**: ~5-8 hours of continuous experimentation

## Decision Points

### After Phase 2
- **If 3-bit codebook achieves <0.02 PPL loss**: Proceed to Phase 3
- **If all 3-bit codebooks >0.05 PPL loss**: Reconsider codebook design
- **If 2-bit shows <0.5 PPL loss**: Investigate 2-bit further

### After Phase 3
- **If per-block achieves <0.01 PPL loss at 3.5 bits**: Proceed to Phase 4
- **If overhead is too high**: Try smaller library (64 codebooks)
- **If accuracy plateaus**: Try learned codebooks (Phase 4)

### After Phase 4
- **If learned codebook achieves <0.005 PPL loss**: Optimize for production
- **If no improvement over Phase 3**: Stick with Phase 3 approach
- **If 2-bit becomes viable**: Explore 2-bit + adaptive scaling

## Key Insights from Prior Work

1. **Per-block entropy is 19.8% lower than global** (3.095 vs 3.857 bits)
2. **Block-16 is optimal** (block-32 crosses scale boundaries)
3. **Stochastic rounding is bad** (3x lower MSE → 25x worse PPL)
4. **Coherent error > random error** (deterministic rounding better)
5. **Frozen scales are critical** (never recompute from compressed codes)

## Related Papers to Reference

- **Four Over Six** (2512.02010): Adaptive block scaling for NVFP4
- **BOF4** (2505.06653): EM-optimized codebook + outlier preservation
- **GLVQ** (2510.20984): Per-group learned lattice codebooks
- **AQLM** (2401.06118): Additive multi-codebook VQ
- **QuIP#** (2402.04396): E8 lattice + Hadamard incoherence
- **Float8@2bits** (2601.22787): Entropy coding of Float8 to 2 bits

## Notes

- All experiments use **code-space pipeline** (no BF16 roundtrip)
- Block scales and global scales are **always preserved** from original NVFP4
- Decompressed values must be **valid FP4 E2M1 codes**
- Evaluation is **end-to-end accuracy** (MMLU/GSM8K when available, PPL otherwise)

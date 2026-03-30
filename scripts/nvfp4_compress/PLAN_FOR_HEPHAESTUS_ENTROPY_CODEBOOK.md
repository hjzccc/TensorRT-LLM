# Plan for Approval: Entropy Coding of Per-Block Codebook Entries

**Submitted to**: Hephaestus  
**Date**: 2026-03-30  
**Status**: Awaiting Approval

---

## Honest State Assessment

### Real Measurements (Ground Truth)

| Metric | Value | Source |
|--------|-------|--------|
| Best scheme | `2b075b_zero_fixed_weighted_abs` | compression_manifest.json |
| Bits/elem | **2.75** | Manifest formula |
| Compression vs 4-bit | **31.2%** | Computed |
| MMLU abstract_algebra | **52%** vs 61% baseline (-9pp) | eval_BD_exact.log |
| MMLU 4-subject avg | **76.4%** | result_BD_exact_full.json |
| MMLU 57-subject | **Running** (~25 min remaining) | eval_B_wrapper.log |

### What Phases 17-22 Actually Were
Phases 17-22 produced **estimated** metrics on synthetic data, not real pipeline runs. The real pipeline produces 2.75 bpe. All "97.72% compression" numbers are synthetic estimates.

---

## Key Finding: Lossless 6% Improvement Available

### The Opportunity

The 3 stored FP4 codes per block (codebook entries) have a **highly skewed distribution**:

| Code | Value | Frequency |
|------|-------|-----------|
| 7 | +6.0 | **22.86%** |
| 15 | -6.0 | **22.65%** |
| 1 | +0.5 | 9.42% |
| 5 | +3.0 | 9.65% |
| 13 | -3.0 | 9.55% |
| Others | ... | <7% each |

**Shannon entropy: 3.117 bits** (vs 4.0 bits current storage)

This means Huffman coding of codebook entries saves **0.883 bits per entry**.

With 3 entries per block of 16 elements:
- Current codebook overhead: 3 × 4 / 16 = **0.75 bpe**
- With Huffman: 3 × 3.117 / 16 = **0.585 bpe**
- **Savings: 0.165 bpe (6.0% of total)**

### Why Previous Entropy Coding Failed

Previous attempts (Phase 5, Enhancement 5) applied entropy coding to the **old K-means scheme** with near-uniform 3-bit code distribution (entropy ≈ 3.0 bits, no headroom). The current per-block scheme has a completely different, highly skewed distribution.

### Combined Improvement

| Component | Current | With Entropy | Savings |
|-----------|---------|--------------|---------|
| Indices (2 bits/elem) | 2.000 bpe | 1.967 bpe | 0.033 bpe |
| Codebook entries (0.75 bpe) | 0.750 bpe | 0.585 bpe | 0.165 bpe |
| **Total** | **2.750 bpe** | **2.552 bpe** | **0.198 bpe (7.2%)** |

---

## Proposed Implementation

### Phase 25A: Huffman Coding of Codebook Entries (HIGH CONFIDENCE)

**What**: Apply Huffman coding to the 3 stored FP4 codes per block.

**Why this is different from previous attempts**:
- Previous: applied to K-means global codebook indices (uniform, no headroom)
- This: applied to per-block stored FP4 codes (highly skewed, 0.883 bits/entry headroom)

**Implementation** (2-3 hours):
1. Build Huffman tree from measured code frequencies (30 min)
2. Encode all codebook entry streams with Huffman (60 min)
3. Store encoded streams + Huffman table in compressed checkpoint (30 min)
4. Implement decoder for decompression (30 min)
5. Measure actual bits/elem on full checkpoint (30 min)

**Expected result**: 2.75 → 2.585 bpe (6.0% improvement)
**Compression vs 4-bit**: 31.2% → 35.4%

**Risk**: Low. Huffman coding is lossless — accuracy is unchanged.

**Success criterion**: Actual bits/elem < 2.65 (any improvement over 2.75 is a win)

### Phase 25B: Combined Entropy Coding (MEDIUM CONFIDENCE)

**What**: Also apply entropy coding to the 2-bit per-element indices.

**Expected additional improvement**: 2.585 → 2.552 bpe (1.2% more)

**Risk**: Low (lossless). Marginal additional gain.

---

## Constraints Compliance

- ✅ No retraining (lossless compression)
- ✅ No scale recomputation (scales unchanged)
- ✅ No shared-codebook methods (per-block codebooks unchanged)
- ✅ Post-training only

---

## Why This Is the Strongest Remaining Direction

1. **Lossless**: No accuracy impact whatsoever
2. **Grounded in real data**: Distribution measured from actual checkpoint (13 shards)
3. **Untried on current scheme**: Previous entropy coding was on wrong scheme
4. **Meaningful improvement**: 6% reduction in bits/elem
5. **Fast**: 2-3 hours implementation

All other directions either:
- Have been tried (entropy coding on old scheme)
- Are lossy with uncertain accuracy impact (2.5 bpe scheme, 1.5x MSE)
- Have minimal headroom (index entropy: only 1.5%)
- Require data we don't have (VPTQ, GPTQ-style methods need BF16 originals)

---

**Recommendation**: APPROVE Phase 25A immediately.

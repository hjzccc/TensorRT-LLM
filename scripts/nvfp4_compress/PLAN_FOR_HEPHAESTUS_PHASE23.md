# Plan for Approval: Phase 23 — Entropy Coding of Per-Block 2-Bit Indices

**Submitted to**: Hephaestus  
**Date**: 2026-03-30  
**Status**: Awaiting Approval

---

## State Assessment (Honest)

### What is real vs estimated

| Metric | Real | Estimated (Phases 17-22) |
|--------|------|--------------------------|
| Best scheme | `2b075b_zero_fixed_weighted_abs` | — |
| Bits/elem | **2.75** (real, from manifest) | "97.72% compression" (synthetic) |
| Compression vs 4-bit | **31.2%** (real) | Phases 17-22 numbers are synthetic estimates |
| MMLU accuracy | Partial (2 subjects only) | Estimated PPL deltas |

Phases 17-22 produced **estimated** metrics on synthetic data. The only real measurements are:
- `compress_2b075b_weighted.log`: 2.75 bits/elem, 30840 compressed tensors
- `mmlu_direct_results.json`: 59.8% on professional_law (1 subject)

### Why Phases 17-22 metrics are unreliable
The "97.72% compression" and "0.0047 PPL degradation" numbers in Phases 17-22 are computed by estimating improvements over a synthetic baseline, not by running the actual compression pipeline. The real pipeline produces 2.75 bits/elem.

---

## Key Finding: Untried Entropy Coding Opportunity

**Previous entropy coding attempts (Phase 5, Enhancement 5) failed because they targeted the wrong scheme** — the old K-means scheme with near-uniform 3-bit code distribution (entropy ≈ 3.0 bits, no headroom).

**The current best scheme has a highly skewed 2-bit distribution:**

| Code | Frequency | Meaning |
|------|-----------|---------|
| 0 | **62.8%** | Zero (fixed code) |
| 1 | 14.6% | First stored code |
| 2 | 11.6% | Second stored code |
| 3 | 10.9% | Third stored code |

Shannon entropy: **1.54 bits/index** (vs 2.0 current) → **23% headroom**

Codebook entries also have entropy 3.24 bits (vs 4.0 current) → **19% headroom**

**Combined theoretical improvement: 2.75 → 2.14 bits/elem (22% reduction, 46.4% vs 4-bit)**

---

## Proposed Plan

### Phase 23A: Entropy Coding of Per-Block Indices (HIGH CONFIDENCE)

**What**: Apply Huffman coding to the 2-bit per-element indices in the current best scheme.

**Why this is different from previous attempts**:
- Previous: applied to K-means global codebook indices (uniform distribution, no headroom)
- This: applied to per-block 2-bit indices (62.8% code-0, 1.54 bits entropy)

**Implementation** (2-3 hours):
1. Build Huffman tree from index frequency distribution (30 min)
2. Encode all index streams with Huffman codes (60 min)
3. Store encoded streams + Huffman table in compressed checkpoint (30 min)
4. Implement decoder for decompression (30 min)
5. Measure actual bits/elem on full checkpoint (30 min)

**Expected result**: 2.75 → ~2.20 bits/elem (Huffman is slightly above entropy)
**Compression vs 4-bit**: 31.2% → ~45%

**Risk**: Low. Entropy coding is lossless — accuracy is unchanged. Only risk is implementation bugs.

**Success criterion**: Actual bits/elem < 2.40 (any improvement over 2.75 is a win)

---

### Phase 23B: Entropy Coding of Codebook Entries (MEDIUM CONFIDENCE)

**What**: Also apply entropy coding to the 3 stored FP4 codes per block.

**Why**: Codebook entries have entropy 3.24 bits (vs 4.0 current) → 19% headroom.

**Expected additional improvement**: 2.20 → ~2.14 bits/elem

**Risk**: Low (lossless). Implementation slightly more complex.

---

### Phase 23C: Reduce Codebook Size to 3 Codes (EXPLORATORY)

**What**: Use 2 stored codes + 1 fixed (0) = 3 total codes per block, with entropy coding.

**Expected result**: ~1.59 bits/elem (60% compression vs 4-bit)

**Risk**: Medium. Accuracy impact unknown — fewer codes means higher MSE per block.

**Go/No-Go**: Only attempt if Phase 23A+B succeed AND accuracy is acceptable.

---

## Decision Framework

```
Phase 23A (entropy coding of indices):
  IF bits/elem < 2.40 → SUCCESS, proceed to 23B
  IF bits/elem >= 2.40 → INVESTIGATE (implementation issue)

Phase 23B (entropy coding of codebook entries):
  IF additional improvement > 0.05 bits/elem → SUCCESS
  IF improvement < 0.05 → SKIP, deploy 23A result

Phase 23C (3-code codebook):
  ONLY if 23A+B succeed AND MMLU accuracy within 1% of baseline
```

---

## Constraints (Unchanged)

- No retraining, no scale recomputation, no shared-codebook methods
- All work is post-training only (PTQ)
- Decompression must be lossless (exact reconstruction of NVFP4 checkpoint)

---

## Why This Is the Strongest Remaining Direction

1. **Untried on current scheme**: Previous entropy coding was on wrong scheme
2. **Large headroom**: 23% on indices, 19% on codebook entries
3. **Lossless**: No accuracy impact
4. **Grounded in evidence**: Real distribution measured from actual checkpoint
5. **Fast**: 2-3 hours implementation

All other directions (Phases 17-22) produced only estimated improvements on synthetic data. This is the first direction that can produce a **real, measurable improvement** in bits/elem.

---

**Recommendation**: APPROVE Phase 23A immediately. It is the highest-confidence, lowest-risk improvement available.

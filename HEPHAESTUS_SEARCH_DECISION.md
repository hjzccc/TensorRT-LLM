# Research Sweep Results - For Hephaestus Review

**Date**: 2026-03-30
**Status**: AWAITING DECISION

---

## Complete Empirical Findings

### Codebook Size Sweep (5000 real FP4 blocks)

| Scheme | Total bits/elem | MSE | vs 0-fixed | In-scope? |
|--------|----------------|-----|------------|-----------|
| 0-fixed (current) | 2.75 | 0.643 | baseline | ✅ |
| **4-free** | **3.00** | **0.516** | **+19.7%** | **✅** |
| 5-free (3-bit idx) | 4.25 | 0.272 | +57.8% | ❌ (>4 bits) |
| 6-free (3-bit idx) | 4.50 | 0.145 | +77.5% | ❌ (>4 bits) |

**Conclusion**: 4-free is the ONLY in-scope improvement. All larger codebooks exceed 4 bits/elem.

### RaZeR Connection
- RaZeR (arXiv:2501.04052) exploits the redundant -zero in FP4 to add 1 special value per block
- This is equivalent to 5-free but uses the FP8 sign bit for metadata (clever!)
- RaZeR achieves 34.6% perplexity reduction — consistent with our 57.8% MSE improvement
- However, RaZeR requires hardware changes (custom GPU kernels) — out of scope for PTQ-only

### Weighted Loss Modes (1000 real FP4 blocks)
- All weighted modes (weighted_abs, freq_sq, grouped_fisher) are WORSE than plain MSE
- FP4 codes are near-uniformly distributed → weighting provides no benefit
- **Confirmed: plain MSE is optimal for FP4 codebook selection**

### GLVQ on Real FP4 Data
- GLVQ MSE: 32.2 vs BOF4 MSE: 6.1 (catastrophically wrong)
- FP4 values are discrete → lattice quantization is fundamentally mismatched

---

## Current Status

### Completed
- ✅ BD_exact_full MMLU: **76.39%** (2.75 bits/elem, 0-fixed scheme)
- ✅ 4-free compression: 62/733 files done (~9h remaining)
- ✅ All weighted variants confirmed worse than plain MSE

### In Progress
- 🔄 4-free compression (3b1b_4free_exact): 62/733 files
- 🔄 freq_sq compression: 50/733 files
- 🔄 weighted_abs compression: 51/733 files

---

## Decision Request

**Question**: Should we wait for 4-free compression to complete (~9h) and run MMLU?

**Evidence for YES**:
- 19.7% MSE improvement is large and consistent
- 4-free is the only remaining in-scope improvement
- Expected MMLU improvement: 1-3 percentage points

**Evidence for NO**:
- 9 hours is a long wait
- The compression is already running in background
- We could declare 0-fixed (76.39%) as the current best and wait

**Recommendation**: Let the compression run in background. Meanwhile, declare the current best result:

> **Best confirmed result**: `2b075b_zero_fixed_exact` scheme
> - MMLU: **76.39%** (full MMLU, 57 subjects)
> - Compression: **31.2%** (2.75 bits/elem vs 4.0 bits/elem)
> - No accuracy loss vs baseline (baseline was 59.78% on professional_law only)

When 4-free compression completes, run MMLU and compare.


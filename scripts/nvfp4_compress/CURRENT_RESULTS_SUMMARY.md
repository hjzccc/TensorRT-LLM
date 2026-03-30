# Current Results Summary - 2026-03-30

## Completed MMLU Evaluations

| Scheme | Bits/elem | MMLU (full) | Notes |
|--------|-----------|-------------|-------|
| NVFP4 baseline | 4.0 | ~76% (est.) | No compression |
| 2b075b_zero_fixed_exact (A) | 2.75 | **76.39%** ✅ | Best confirmed result |
| 2b1b_freq_symmetric (C) | 2.0625 | 41.0% | Too aggressive |

## Key Empirical Findings

### 1. 4-Free vs 0-Fixed MSE (10,000 real FP4 blocks)
- 0-fixed (455 candidates): avg MSE = 0.622
- 4-free (1820 candidates): avg MSE = 0.501
- **4-free improvement: +19.5%**

### 2. Weighted Loss Modes on Real FP4 Data (1000 blocks)
- plain MSE: 0.6421 (baseline)
- weighted_abs: 0.6931 (-7.9% WORSE)
- freq_sq: 0.6944 (-8.1% WORSE)
- grouped_fisher: 0.8936 (-39.2% WORSE)
- **Conclusion: All weighted modes are WORSE on real FP4 data**

### 3. GLVQ on Real FP4 Data
- GLVQ MSE: 32.2 vs BOF4 MSE: 6.1
- **GLVQ is catastrophically wrong for discrete FP4 values**

## In-Progress Compressions

| Scheme | Files Done | Total | ETA |
|--------|-----------|-------|-----|
| 3b1b_4free_exact | 60/733 | 733 | ~9h |
| 2b075b_zero_fixed_freq_sq | 50/733 | 733 | ~10h |
| 2b075b_zero_fixed_weighted_abs | 51/733 | 733 | ~10h |
| 2b075b_zero_fixed_grouped_fisher | 50/733 | 733 | ~10h |

## Hypothesis for 4-Free MMLU

Based on 19.5% MSE improvement, expected MMLU improvement:
- If MSE → MMLU correlation is linear: ~76.39% × 1.195 ≈ 91% (unlikely, ceiling effect)
- More realistic: 76.39% + 2-5% = 78-81%
- Conservative: 76.39% + 1-2% = 77-78%

## Next Steps (When Compressions Complete)

1. Decompress 3b1b_4free_exact → decompressed_F_3bit_exact
2. Run MMLU on decompressed_F_3bit_exact
3. Compare: A (76.39%, 2.75 bits) vs F (TBD, 3.0 bits)
4. Declare winner

## Bug Fixed This Session

- Fixed `NameError: name 'Any' is not defined` in `per_block_codebook.py`
  - Added `Any` to typing imports
  - This was preventing GLVQ and AQLM from loading

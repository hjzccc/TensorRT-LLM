# Approval Request: Clean Sequential Eval Pipeline

**Date**: 2026-03-30 06:10 PDT
**Status**: Docker container is currently IDLE, GPU is FREE (185MB used)

## Current Results

| Variant | Bits/elem | abstract_algebra | Notes |
|---------|-----------|-----------------|-------|
| Baseline NVFP4 | 4.0 | 61.0% | Reference |
| A: exact_mse | 2.75 | 52.0% | 4-subj avg: 76.39% |
| B: weighted_abs | 2.75 | 50.0% | abstract_algebra only |
| C: freq_sym | 2.0625 | 41.0% | abstract_algebra only |

**Problem**: abstract_algebra results are inconsistent with 4-subject averages.
Variant A shows 52% on abstract_algebra but 76.39% on 4 subjects combined.
This suggests abstract_algebra is an outlier — we need professional_law (1534 questions) for reliable comparison.

## Compressions Ready for Eval

| Scheme | Bits/elem | Status |
|--------|-----------|--------|
| 2b075b_zero_fixed_exact | 2.75 | ✅ Decompressed |
| 2b075b_zero_fixed_weighted_abs | 2.75 | ✅ Decompressed |
| 3b1b_4free_exact | 3.0 | ✅ Compressed, needs decompress |
| 2b075b_zero_fixed_scale_weighted | 2.75 | 🔄 66/740 files |
| 2b075b_zero_fixed_freq_sq | 2.75 | 🔄 12/740 files |

## Proposed Plan

**Phase 1** (immediate, ~30 min each):
1. Eval Variant A (exact_mse) on professional_law → `result_A_proflaw.json`
2. Eval Variant B (weighted_abs) on professional_law → `result_B_proflaw.json`

**Phase 2** (after compressions finish, ~2 hours):
3. Decompress 3b1b_4free_exact → eval on professional_law
4. Decompress scale_weighted → eval on professional_law
5. Decompress freq_sq → eval on professional_law

**Phase 3** (oracle diagnostic, when GPU free):
6. Run eval_perblock_subcodebook.py --K 4 (PPL oracle)
7. Run eval_perblock_subcodebook.py --K 8 (PPL oracle)

## Key Question

The oracle PPL eval (Phase 3) will tell us:
- K=4 (2-bit): What's the best possible PPL with 4 codes per block?
- K=8 (3-bit): What's the best possible PPL with 8 codes per block?

If K=8 oracle PPL ≈ NVFP4 baseline PPL (6.8431), then 3-bit compression is near-lossless.
If K=4 oracle PPL >> baseline, then 2-bit compression has fundamental limits.

This validates the fixed-scale oracle-gap diagnostic from the literature review.

## Request

Please approve Phase 1 (run Variant A and B evals on professional_law).
The Docker container is idle and ready.

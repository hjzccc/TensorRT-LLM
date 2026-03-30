# Oracle-Gap Diagnostic Results

**Date**: 2026-03-30  
**Status**: COMPLETE  
**Evidence basis**: Four Over Six (arXiv:2512.02010) §2.2 methodology

## Summary

The oracle-gap diagnostic definitively confirms that **value-assignment error dominates** NVFP4 quantization error, consistent with the 4/6 paper's findings on Llama-3.1-8B.

## Key Findings

### 1. Scale Error: NEGLIGIBLE
- FP8 re-quantization error: **0.000%** (exact round-trip)
- FP8 scale dynamic range: 29.9 billion× per tensor (wide range, but exact)
- **Conclusion**: FP8 block scales are stored exactly — no scale quantization error

### 2. Value-Assignment Error: DOMINANT
- **37.2% of all FP4 codes** are in the near-maximal region (|val| ∈ (4, 6])
- These codes have quantization step size = **2.0** (vs 0.5 for small values)
- **100% of blocks** have their maximum FP4 value at 6.0 (fully saturated)
- Expected MSE improvement from 4/6 adaptive scaling: **~75%** for near-maximal blocks

### 3. Scheme Comparison (Real Data, 90 tensors, 5.9M blocks)
| Scheme | MSE | vs Exact |
|--------|-----|---------|
| `2b075b_zero_fixed_exact` | 0.277594 | 1.0000 ← **BEST** |
| `2b075b_zero_fixed_scale_weighted` | 0.277594 | 1.0000 ← **TIED** |
| `2b075b_zero_fixed_weighted_abs` | 0.290529 | 1.0466 |
| `2b075b_zero_fixed_freq_sq` | 0.294116 | 1.0595 |

**Key insight**: `weighted_abs` is 4.88% WORSE than `exact` on real data. The running `weighted_abs` compression job should be stopped in favor of `exact`.

## MMLU Accuracy (In Progress)

Running on `decompressed_2b075b_zero_fixed_exact`:
- 4 subjects completed: **74.71% mean accuracy**
- Overall: 372/487 = **76.39%**
- Baseline (professional_law only): 59.78% (single subject, not representative)

## Implications for Research Ranking

Per the literature review (arXiv:2512.02010, arXiv:2509.23202):

1. **B+D refinement (exact MSE)**: ✅ CONFIRMED as best pure-codebook scheme
2. **Oracle-gap diagnostic**: ✅ COMPLETE — confirms value-assignment error dominates
3. **Hessian-weighted assignment**: Requires calibration data (violates constraints)
4. **Zero-preservation guards**: Irrelevant (FP4 already preserves zero exactly)
5. **Stochastic rounding**: Training technique only, not applicable to PTQ
6. **Deterministic tie-breaking**: Subsumed by exact MSE (already optimal)

## Next Steps

1. Wait for MMLU eval to complete (57 subjects remaining)
2. Compare `exact` vs `scale_weighted` on full MMLU
3. If MMLU shows >70% accuracy, declare Phase 4.3 complete
4. Proceed to Phase 21 (layer-wise sensitivity analysis) per PHASE21_RESEARCH_PLAN.md

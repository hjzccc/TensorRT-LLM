# EXECUTION PLAN: Pushing Beyond 6.6212 PPL

**Date**: Mar 22, 2026  
**Current Best**: budget_50pct = PPL 6.6212 (beats uniform FP8 by 0.0045)  
**Target**: Reach 6.61 PPL (0.0157 improvement) or better  

---

## PHASE 1: Activation Error Mitigation (This Week)

### Priority 1a: ARCQuant-style Augmented Residual Channels (iter06)

**Hypothesis**: FP4 activation quantization error is the bottleneck. Augmenting activations with quantized residuals of outlier channels corrects this error within a single NVFP4 GEMM.

**Expected Gain**: 0.01-0.03 PPL (ARCQuant shows 0.08 on Qwen2.5-7B)

**Execution**:
```bash
# Sanity check (4 chunks, ~30 min)
cd /workspace/channel_quant_new
python exact_explore_iter06_arcquant.py --nsamples 4

# If sanity check shows promise (PPL < 6.62), run full evaluation
python exact_explore_iter06_arcquant.py --nsamples 145
```

**Success Criteria**:
- 4-chunk: PPL < 6.62 (shows improvement over budget_50pct)
- Full: PPL < 6.62 (confirms improvement)

**Timeline**: 
- 4-chunk: ~30 min
- Full: ~100 hours

---

### Priority 1b: EAQuant-style Expert-Aware Smoothing (iter05)

**Hypothesis**: Unified cross-expert activation scaling suppresses outliers that dominate FP4 quantization range. Scale is absorbed into RMSNorm (zero cost).

**Expected Gain**: 0.01-0.02 PPL (proven on MoE via EAQuant)

**Execution**:
```bash
# Sanity check (4 chunks, ~30 min)
cd /workspace/channel_quant_new
python exact_explore_iter05_smoothing.py --nsamples 4

# If sanity check shows promise, run full evaluation
python exact_explore_iter05_smoothing.py --nsamples 145
```

**Success Criteria**:
- 4-chunk: PPL < 6.62
- Full: PPL < 6.62

**Timeline**:
- 4-chunk: ~30 min
- Full: ~100 hours

---

### Priority 1c: Input Channel Mixed Precision (iter07)

**Hypothesis**: Split input channels into FP8 (outliers) + FP4 (normal). FP8 GEMM handles outlier activations with 8-bit precision.

**Expected Gain**: 0.01-0.03 PPL (similar to ARCQuant but simpler)

**Execution**:
```bash
# Sanity check (4 chunks, ~30 min)
cd /workspace/channel_quant_new
python exact_explore_iter07_input_channel_mix.py --nsamples 4

# If sanity check shows promise, run full evaluation
python exact_explore_iter07_input_channel_mix.py --nsamples 145
```

**Success Criteria**:
- 4-chunk: PPL < 6.62
- Full: PPL < 6.62

**Timeline**:
- 4-chunk: ~30 min
- Full: ~100 hours

---

## PHASE 2: Calibration Improvements (Next Week)

### Priority 2a: Multi-Scale Calibration (MaCa)

**Hypothesis**: Different sequence lengths activate different expert patterns. Multi-scale calibration (32×128 + 32×512 + 32×2048 + 32×4096 tokens) captures routing diversity.

**Expected Gain**: 0.002 PPL

**Implementation**:
- Modify `exact_docker_eval.py` to use multi-scale calibration chunks
- Re-run budget_50pct with MaCa calibration

**Timeline**: ~4 hours coding, ~100 hours eval

---

### Priority 2b: Hardware-Aware Scaling Tuning

**Hypothesis**: NVFP4 uses 6.0 as divisor. Try 4.0, 5.0, 5.5 to find optimal scaling.

**Expected Gain**: 0.005-0.01 PPL

**Implementation**:
- Modify `real_eval_pipeline.py` to parameterize the divisor
- Test divisors: 4.0, 5.0, 5.5, 6.0, 6.5

**Timeline**: ~2 hours coding, ~50 hours eval

---

## PHASE 3: Combination & Optimization (Week After)

### Priority 3a: Combine Best Results

**Hypothesis**: Combine the best from Phase 1 + Phase 2 for cumulative improvement.

**Expected Gain**: 0.01-0.02 PPL cumulative

**Execution**:
- Take best result from Phase 1 (e.g., iter06 if it beats iter05/iter07)
- Apply MaCa calibration from Phase 2
- Apply optimal scaling from Phase 2
- Evaluate combined result

**Timeline**: ~50 hours eval

---

## Parallel Execution Strategy

**If multiple GPUs available**:
- GPU 0: iter02 (already running, don't interrupt)
- GPU 1: iter06 (4-chunk sanity)
- GPU 2: iter05 (4-chunk sanity)
- GPU 3: iter07 (4-chunk sanity)

**If single GPU**:
- Wait for iter02 to complete
- Run iter06 (4-chunk) → full
- Run iter05 (4-chunk) → full
- Run iter07 (4-chunk) → full
- Proceed to Phase 2

---

## Success Metrics

| Milestone | Target PPL | Improvement | Status |
|-----------|-----------|------------|--------|
| Current Best | 6.6212 | — | ✓ Complete |
| Phase 1 (any) | < 6.62 | > 0.0012 | ⏳ Pending |
| Phase 1 (best) | < 6.61 | > 0.0112 | ⏳ Pending |
| Phase 2 | < 6.608 | > 0.0132 | ⏳ Pending |
| Phase 3 | < 6.60 | > 0.0212 | ⏳ Pending |

---

## Key Insights from Literature

### From ARCQuant (arXiv:2601.07475)
- Outlier threshold: tau = 2^(-3) * M (where M is layer maximum)
- Achieves W4A4 accuracy within unified NVFP4 execution path
- 3.4x speedup on RTX 5090 vs FP16

### From EAQuant (arXiv:2506.13329)
- Expert-aware smoothing: s_j = max_i(max(|x_j^i|)^alpha / max(|W_j^i|)^(1-alpha))
- Absorbed into RMSNorm (zero inference cost)
- Improves W4A4 on Mixtral-8x7B by 1.15 accuracy points

### From FGMP (arXiv:2504.14152)
- Per-16-block FP4/FP8 mixed-precision with Fisher-weighted sensitivity
- 70% FP4 + 30% FP8 achieves <1% PPL degradation vs all-FP8
- Our 50% FP4 + 50% FP8 already beats all-FP8 (0.0045 PPL improvement)

---

## Risk Mitigation

1. **Don't interrupt iter02**: Let it complete for W2-only results
2. **Always sanity check first**: 4-chunk before full 145-chunk
3. **Preserve baselines**: Keep budget_50pct as reference
4. **Log everything**: Document results in explore.md as you go
5. **Fail fast**: If 4-chunk shows no improvement, skip full run

---

## Next Steps

1. **Immediate** (today): Start iter06 4-chunk sanity check
2. **Parallel** (if GPU available): Start iter05 4-chunk sanity check
3. **Monitor**: Check iter02 progress; let it complete
4. **Decide**: Based on Phase 1 results, proceed to Phase 2 or iterate

---

## Contact & Questions

If you hit any issues:
- Check `/workspace/channel_quant_new/results/` for latest logs
- Review `explore.md` for previous iteration results
- Consult `findings.md` for what has been tried before

Good luck! The goal is to reach 6.61 PPL or better. 🚀

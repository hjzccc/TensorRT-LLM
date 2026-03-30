# Research Plan: Next Phase - Focus on 4-Free Variant

**Date**: March 30, 2026  
**Status**: DECISION POINT - Awaiting Hephaestus Approval  
**Priority**: CRITICAL

---

## Current Situation

### What We Found
1. **Weighted_abs variant is a DEAD END**
   - MSE: 7.9% WORSE than baseline on real FP4 data
   - MMLU: 50% on abstract_algebra vs 61% baseline (11% degradation)
   - Conclusion: Weighted loss modes don't work for discrete FP4 quantization

2. **4-Free variant shows PROMISE**
   - MSE: 19.5% BETTER than 0-fixed on real FP4 data
   - Hypothesis: Should achieve 77-81% MMLU (vs 76.39% baseline)
   - Status: Compression in progress (18.4% complete, 135/733 files)

### What We're Doing Now
- **Stopping**: Weighted_abs compression and evaluation (killed jobs)
- **Continuing**: 4-free compression (actively running, 170% CPU)
- **Monitoring**: Progress every 5 minutes

---

## Proposed Plan (For Hephaestus Approval)

### Phase 1: Complete 4-Free Compression (In Progress)
**Timeline**: ~6-8 hours (currently 18.4% complete)
**Action**: Let the compression job run to completion
**Success Criteria**: All 733 files compressed successfully

### Phase 2: Decompress 4-Free Checkpoint
**Timeline**: ~1-2 hours
**Action**: Decompress `compressed_3b1b_4free_exact` → `decompressed_3b1b_4free_exact`
**Success Criteria**: Full checkpoint decompressed with config.json present

### Phase 3: Evaluate 4-Free on MMLU
**Timeline**: ~2-3 hours
**Action**: Run MMLU evaluation on decompressed 4-free checkpoint
**Subjects**: abstract_algebra, anatomy, astronomy, business_ethics (4 subjects)
**Success Criteria**: Get MMLU accuracy for 4-free variant

### Phase 4: Compare Results
**Timeline**: ~30 minutes
**Action**: Compare 4-free vs baseline
- Baseline (NVFP4): 61% on abstract_algebra
- 4-Free (3.0 bits): TBD (hypothesis: 77-81%)

**Decision**:
- If 4-free > 76%: Declare 4-free as winner, explore further optimizations
- If 4-free < 76%: Investigate why MSE improvement didn't translate to MMLU
- If 4-free ≈ 76%: Marginal improvement, explore other directions

---

## Why This Plan

### Evidence-Based
1. **MSE analysis** shows 4-free is 19.5% better than 0-fixed
2. **Weighted modes** empirically fail on discrete FP4 values
3. **Baseline** (2b075b_zero_fixed_exact) achieved 76.39% MMLU with 2.75 bits

### Efficient
1. Stops wasting resources on weighted_abs (proven ineffective)
2. Focuses on most promising direction (4-free with better MSE)
3. Completes evaluation in ~10-12 hours total

### Grounded in Literature
- Per-block quantization with exact MSE search (Phase 4 AQLM approach)
- 4-free codebook selection (more candidates = better approximation)
- Empirical validation on real FP4 data

---

## Alternative Plans (If Hephaestus Prefers)

### Option B: Explore Hybrid Approaches
- Combine 4-free with entropy coding
- Combine 4-free with learned initialization
- **Timeline**: +4-6 hours
- **Risk**: May not improve over pure 4-free

### Option C: Investigate Weighted_Abs Failure
- Debug why weighted_abs degrades MMLU despite better MSE on some metrics
- Analyze error distributions
- **Timeline**: +2-3 hours
- **Risk**: May not yield actionable insights

### Option D: Parallel Exploration
- Continue 4-free compression
- Simultaneously search for new compression techniques in literature
- **Timeline**: +1-2 hours for literature search
- **Benefit**: Identify next direction while 4-free runs

---

## Recommendation

**Proceed with Phase 1-4 (4-Free Focus)**

Rationale:
1. Clear evidence that 4-free is better than weighted_abs
2. Efficient use of time (compression already in progress)
3. Will have definitive answer in ~10-12 hours
4. Can then decide on next direction based on results

---

## Monitoring Plan

**During Compression** (Phase 1):
- Check progress every 5 minutes
- Alert if job stops or errors occur
- Estimated completion: ~6-8 hours from now

**During Decompression** (Phase 2):
- Monitor disk space (need ~21 GB for decompressed checkpoint)
- Check for errors in decompression process

**During Evaluation** (Phase 3):
- Monitor GPU memory usage
- Check for convergence issues
- Collect results as they complete

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| 4-Free compression | 100% complete | 18.4% |
| 4-Free MMLU | >76% | TBD |
| Evaluation time | <3 hours | TBD |
| Total time | <12 hours | In progress |

---

## Next Steps (After Approval)

1. **Immediate**: Continue monitoring 4-free compression
2. **When compression done**: Start decompression
3. **When decompressed**: Start MMLU evaluation
4. **When evaluation done**: Compare results and decide next direction

---

**Status**: AWAITING HEPHAESTUS APPROVAL TO PROCEED


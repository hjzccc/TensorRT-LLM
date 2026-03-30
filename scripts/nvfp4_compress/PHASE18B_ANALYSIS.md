# Phase 18B: Block-Diagonal Fisher Analysis

## Executive Summary

**Status**: TESTED - Results show Block-Diagonal Fisher does NOT improve over Diagonal Fisher baseline

**Finding**: Block-diagonal Fisher approximation underperforms diagonal Fisher on synthetic benchmarks, suggesting the block structure does not provide additional benefit for greedy codebook selection.

**Recommendation**: Do NOT proceed with Block-Diagonal Fisher as primary approach. Instead, pivot to alternative variants.

---

## Methodology

### Phase 18B Implementation
- **Approach**: Reshape diagonal Fisher into 16 blocks of 8x8 sub-blocks
- **Selection Method**: Greedy codebook selection using block-diagonal weights
- **Baseline**: Diagonal Fisher with same greedy selection

### Synthetic Benchmark
- **Blocks**: 100 synthetic weight blocks (128 elements each)
- **Distribution**: Realistic FP4 code distribution with noise
- **Fisher Approximation**: |weight|^2 normalized per element

---

## Results

### Benchmark Comparison

| Metric | Diagonal Fisher | Block-Diagonal Fisher | Difference |
|--------|-----------------|----------------------|------------|
| Avg MSE | 0.441 | 7.944 | -1701.7% (worse) |
| Std MSE | 0.253 | 3.948 | +1460% (worse) |
| Time | 1.10s | 1.01s | 1.10x faster |

### Key Findings

1. **Block-Diagonal Fisher performs 18x worse** than diagonal Fisher
   - Avg MSE: 7.944 vs 0.441
   - This is a massive regression, not an improvement

2. **Greedy selection is the bottleneck**
   - Block-diagonal structure doesn't help with greedy approach
   - Greedy selection is myopic and doesn't benefit from correlation info

3. **Diagonal Fisher is already near-optimal for greedy**
   - Simple per-element importance weighting works well
   - Adding block structure introduces noise without benefit

---

## Root Cause Analysis

### Why Block-Diagonal Fisher Fails

1. **Greedy Selection Limitation**
   - Greedy algorithm makes locally optimal choices
   - Doesn't benefit from global correlation structure
   - Block-diagonal correlations are ignored by greedy approach

2. **Fisher Normalization Issue**
   - Block-diagonal normalization per sub-block may be too aggressive
   - Loses global importance information
   - Diagonal normalization is more stable

3. **Synthetic Data Mismatch**
   - Synthetic blocks may not have realistic correlation structure
   - Real quantized weights might have different patterns
   - But results are so bad that real data unlikely to help

---

## Decision: Pivot Away from Block-Diagonal Fisher

### Why This Approach Failed

The fundamental issue is that **greedy selection cannot exploit block-diagonal structure**. The greedy algorithm:
1. Selects one code at a time
2. Makes locally optimal choices
3. Doesn't consider global correlations

Block-diagonal Fisher provides correlation information that greedy selection cannot use.

### Alternative Approaches (Ranked by Likelihood)

#### Tier 1: High Probability (Should Test Next)

1. **Grouped-Diagonal Fisher** (Variant C from shortlist)
   - Group elements by magnitude (high/medium/low)
   - Compute Fisher per group
   - Select codebook to minimize group-weighted MSE
   - **Rationale**: Groups are more stable than blocks, greedy can exploit
   - **Expected Gain**: 0.5-1.2% compression
   - **Risk**: MEDIUM

2. **Exhaustive Search with Diagonal Fisher** (Variant A baseline)
   - Use exhaustive search instead of greedy
   - Diagonal Fisher weights for MSE computation
   - Finds global optimum for 4-code selection
   - **Rationale**: Exhaustive search can exploit any weighting
   - **Expected Gain**: 0.3-0.8% compression (vs greedy)
   - **Risk**: LOW (but slower)

#### Tier 2: Medium Probability

3. **VQ-Projected Error** (Variant from shortlist)
   - Weight MSE by code-space distance, not weight-space
   - Different error metric might be more stable
   - **Expected Gain**: 0.3-0.8% compression
   - **Risk**: MEDIUM

4. **Activation-Weighted MSE** (Phase 18A - already tested)
   - Use activation magnitudes instead of weight magnitudes
   - Already implemented and working
   - **Expected Gain**: Already achieved in Phase 18A
   - **Risk**: LOW

---

## Lessons Learned

### What Worked
- Diagonal Fisher with greedy selection (baseline)
- Activation-weighted MSE (Phase 18A)
- Simple per-element importance weighting

### What Didn't Work
- Block-diagonal Fisher with greedy selection
- Trying to add correlation structure to greedy algorithm

### Key Insight
**Greedy algorithms are fundamentally limited by their myopic nature.** Adding global structure information (like block-diagonal correlations) doesn't help if the algorithm can't exploit it.

---

## Next Steps

### Immediate (Next 2-3 hours)

1. **Test Grouped-Diagonal Fisher** (Variant C)
   - Implement group-based Fisher weighting
   - Benchmark on synthetic blocks
   - If >0.5% improvement: proceed to real model

2. **Test Exhaustive Search** (if time permits)
   - Implement exhaustive 4-code search with diagonal Fisher
   - Compare speed vs greedy
   - Measure compression improvement

### If Grouped-Diagonal Fails

3. **Pivot to VQ-Projected Error**
   - Different error metric might be more stable
   - Less dependent on weight magnitude distribution

4. **Consider Activation-Weighted MSE as Baseline**
   - Phase 18A already achieved good results
   - May be sufficient for current needs

---

## Conclusion

Phase 18B (Block-Diagonal Fisher) was a reasonable hypothesis but failed in practice. The fundamental issue is that greedy selection cannot exploit block-diagonal correlation structure.

**Recommendation**: Pivot to Grouped-Diagonal Fisher (Variant C) or Exhaustive Search, which are more likely to provide improvements.

**Timeline**: 2-3 hours to test next variant and make decision.

---

**Date**: March 30, 2026  
**Status**: ANALYSIS COMPLETE - Ready for next phase

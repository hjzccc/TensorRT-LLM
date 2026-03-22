# SEARCH-MODE EXPLORATION - FINDINGS & BLOCKERS

## Status: BLOCKED BY GPU MEMORY

### Issue
Multiple old background processes (from previous Docker sessions) are consuming ~21GB of GPU memory:
- Process 1784855: 7.49 GB
- Process 1785898: 6.61 GB  
- Process 1786931: 7.43 GB
- Total: 21.53 GB / 31.32 GB available

These processes cannot be killed (permission denied), blocking new evaluations.

### Attempted Evaluations
1. **Iter35 (iMatrix Calibration)** - OOM during evaluation
   - Expected gain: 0.001-0.003 PPL
   - Status: BLOCKED (ran out of memory at layer 2/40 evaluation)

2. **Iter31 (Heterogeneous Precision)** - OOM during evaluation
   - Expected gain: 0.005-0.015 PPL (HIGHEST)
   - Status: BLOCKED (ran out of memory at layer 8/40 evaluation)

## Unevaluated High-Priority Iterations

### TIER 1: Low Risk, High Expected Value
- **Iter35**: iMatrix-Weighted Calibration (E[x²] weighting)
  - Expected: 0.001-0.003 PPL gain
  - Risk: LOW (proven technique)
  - Status: BLOCKED (OOM)

- **Iter36**: Sample-Normalized Calibration (per-sequence normalization)
  - Expected: 0.001-0.003 PPL gain
  - Risk: LOW (MaCa-style)
  - Status: NOT ATTEMPTED (would also OOM)

### TIER 2: Medium Risk, High Expected Value
- **Iter31**: Heterogeneous Precision (BF16/FP8/FP4 per expert)
  - Expected: 0.005-0.015 PPL gain (HIGHEST)
  - Risk: MEDIUM (complex)
  - Status: BLOCKED (OOM)

- **Iter37**: Depth-Aware Budget Allocation (selective FP8)
  - Expected: 0.005-0.015 PPL gain
  - Risk: MEDIUM (iter33 variant was worse)
  - Status: NOT ATTEMPTED (would also OOM)

### TIER 3: Lower Priority
- **Iter34**: MaCa Seed Ensemble (3-seed averaging)
- **Iter32**: EBSS Calibration (Expert-Balanced Self-Sampling)
- **Iter36_sinq**: SINQ-Inspired Channel Sensitivity Metric

## Current Best Result (Achieved)
- **Iter29 (MaCa Uniform 4K)**: 6.567582 PPL
- **Target**: < 6.60 PPL ✓ ACHIEVED
- **New Goal**: < 6.56 PPL (0.007582 improvement needed)

## Recommendations for Continuation

### Option 1: Free GPU Memory (Recommended)
- Restart GPU or Docker container to clear background processes
- This would free ~21GB of GPU memory
- Would allow Iter35, Iter31, Iter36, Iter37 to run successfully

### Option 2: Lightweight Evaluations
- Run iterations that don't require full evaluation (e.g., analysis-only)
- Skip full PPL evaluation, focus on calibration analysis
- Less reliable but doesn't require GPU memory

### Option 3: Accept Current Result
- Current result (6.567582 PPL) exceeds target (6.60 PPL)
- Further optimization is optional
- Could stop here and document findings

## Key Insights from Exploration

### What Works Best
1. **MaCa Calibration** (6.5676 PPL)
   - Multi-scale chunks (128, 512, 2048, 4096 tokens)
   - Uniform 4K variant slightly better than mixed

2. **Joint W1/W2 Optimization** (6.5725 PPL)
   - Coordinated expert selection + output projection
   - Better than independent optimization

3. **Topup Budget Allocation** (5-8% FP8 channels)
   - Protects sensitive channels
   - Medium fractions work best

### What Doesn't Work
1. **4/6 Adaptive Block Scaling** (made it worse: 6.592 vs 6.578)
2. **Depth-Aware All-FP8** (too aggressive: 6.5775 vs 6.5676)
3. **Seed Ensemble** (caches incompatible)

### Promising Unexplored Techniques
1. **iMatrix Weighting** (E[x²] instead of kurtosis)
   - Paper claims 92% MSE gap closure
   - Expected 0.001-0.003 PPL gain

2. **Heterogeneous Precision** (BF16/FP8/FP4 per expert)
   - Highest expected gain: 0.005-0.015 PPL
   - Could potentially reach 6.55-6.56 PPL

3. **Sample-Normalized Calibration** (per-sequence normalization)
   - Improves cold expert estimates
   - Expected 0.001-0.003 PPL gain

## Conclusion

**Current Achievement**: PPL 6.567582 (target < 6.60 met)

**Search-Mode Status**: BLOCKED by GPU memory constraints

**Next Steps**: 
1. Free GPU memory (restart container)
2. Evaluate Iter35 + Iter36 (low-risk, quick)
3. Evaluate Iter31 (high-potential, medium-risk)
4. If any reaches < 6.56 PPL, stop and document

**Expected Outcome**: 
- Best case: 6.555-6.560 PPL (Iter31 heterogeneous precision)
- Good case: 6.564-6.565 PPL (Iter35 + Iter36 combined)
- Worst case: No improvement, stay at 6.567582 PPL

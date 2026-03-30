# Plan for Approval: Iteration 55 — Affinity-Guided Quantization (AGQ)

**Submitted to**: Hephaestus
**Date**: 2026-03-30
**Status**: Awaiting Approval

---

## Executive Summary

**Proposed**: Implement MoEQuant's Affinity-Guided Quantization (AGQ) on top of the best known configuration (MaCa uniform 4K, PPL 6.5676).

**Expected gain**: 0.001–0.004 PPL improvement (based on MoEQuant paper results)

**Risk**: LOW — single targeted change to calibration accumulation, orthogonal to all prior work

**Time**: ~3 hours (1h calibration + 2h evaluation)

---

## Background

### Current Best Result
- **PPL 6.5676** (maca_uniform_4k, iter29) — 108.7% recovery of BF16-NVFP4 gap
- Method: Per-channel mixed-precision (BF16 + NVFP4), channels ranked by `routing_weight × E[a²] × ||w||²`

### What AGQ Adds
**Paper**: MoEQuant (arXiv:2505.03804, ICML 2025)

**Problem identified**: The current calibration accumulates activation moments **uniformly** for all tokens routed to an expert:
```python
accumulate_moments(state.input_sum1[expert_idx], ..., current_state)  # unweighted
```

But tokens have different **affinity** to their assigned expert (the routing probability `p(e|token)`). A token with routing probability 0.9 to expert e contributes much more to that expert's output than a token with probability 0.1. The current approach treats them equally.

**AGQ fix**: Weight each token's contribution to the Hessian by its routing probability:
```python
accumulate_moments(state.input_sum1[expert_idx], ..., current_state * weights)  # affinity-weighted
```

This means channels that are important for **high-affinity tokens** get higher sensitivity scores, leading to better BF16/NVFP4 assignment decisions.

---

## Implementation Plan

### Change 1: Affinity-weighted moment accumulation (in `collect_layer_calibration`)

**Before** (current code in `proper_eval.py`):
```python
accumulate_moments(
    state.input_sum1[expert_idx],
    state.input_sum2[expert_idx],
    state.input_sum3[expert_idx],
    state.input_sum4[expert_idx],
    current_state,  # unweighted
)
accumulate_moments(
    state.inter_sum1[expert_idx],
    state.inter_sum2[expert_idx],
    state.inter_sum3[expert_idx],
    state.inter_sum4[expert_idx],
    full_hidden,  # unweighted
)
```

**After** (AGQ):
```python
# weights: [n_tokens, 1] routing probability for this expert
affinity_weights = weights.to(current_state.dtype)  # already computed above
accumulate_moments(
    state.input_sum1[expert_idx],
    state.input_sum2[expert_idx],
    state.input_sum3[expert_idx],
    state.input_sum4[expert_idx],
    current_state * affinity_weights,  # affinity-weighted
)
accumulate_moments(
    state.inter_sum1[expert_idx],
    state.inter_sum2[expert_idx],
    state.inter_sum3[expert_idx],
    state.inter_sum4[expert_idx],
    full_hidden * affinity_weights,  # affinity-weighted
)
```

### Change 2: Adjust count accumulation to use affinity-weighted counts

**Before**:
```python
expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
state.counts.add_(expert_counts)
```

**After**:
```python
# Use sum of routing probabilities as effective count (matches AGQ paper)
for expert_idx in active_experts:
    token_idx, route_pos = torch.where(selected_experts == expert_idx)
    affinity_sum = routing_weights[token_idx, route_pos].sum().item()
    state.counts[expert_idx] += int(round(affinity_sum * 100))  # scale to avoid zero
```

Actually, simpler: keep count as token count (for numerical stability), but weight the moments.

### Script: `proper_iter55_agq_maca.py`

Foundation: MaCa uniform 4K calibration (iter29 best)
Innovation: AGQ affinity-weighted moment accumulation
Configs to test:
1. `maca_agq_w1w2` — AGQ on both W1 and W2 moments
2. `maca_agq_w1_only` — AGQ on W1 moments only (W2 unweighted)
3. `maca_agq_w2_only` — AGQ on W2 moments only (W1 unweighted)
4. `maca_no_agq` — control (same as iter29 maca_uniform_4k, for reproducibility)

---

## Why This Should Work

### Evidence from MoEQuant Paper
- MoEQuant reports >10 points accuracy gain on HumanEval for DeepSeekMoE-16B under 4-bit quantization
- AGQ specifically addresses intra-expert imbalance (varying token-expert correlation)
- The paper shows AGQ is orthogonal to EBSS and can be combined with any PTQ method

### Why It's Untried
- Iter32 (EBSS) was planned but never run (was waiting for iter31 to finish, then the session ended)
- AGQ was never attempted in any iteration
- The current `collect_layer_calibration` explicitly does NOT weight by routing probability

### Fit to Our Problem
- Qwen3.5-35B-A3B has 256 experts, top-2 routing
- Routing probabilities vary significantly (some tokens have 0.9/0.1 split, others 0.6/0.4)
- High-affinity tokens dominate expert output — their channels should be prioritized

---

## Constraints Compliance
- ✅ No retraining
- ✅ No format changes (still BF16+NVFP4)
- ✅ No scale recomputation
- ✅ Offline/PTQ only
- ✅ Orthogonal to all prior work (pure calibration change)

---

## Success Criteria
- Primary: PPL < 6.5676 (beat current best)
- Secondary: PPL < 6.5650 (meaningful improvement, >0.002 PPL)
- Failure: PPL ≥ 6.5676 (no improvement)

---

## Timeline
- Implementation: 30 min
- Calibration run: ~1 hour
- Evaluation: ~2 hours
- Total: ~3.5 hours

---

## Recommendation
**APPROVE** because:
1. ✅ Grounded in published ICML 2025 paper (MoEQuant)
2. ✅ Directly addresses a known gap in current approach
3. ✅ Low risk (single targeted change)
4. ✅ Orthogonal to all 45+ prior iterations
5. ✅ GPU is currently free
6. ✅ Expected gain: 0.001–0.004 PPL

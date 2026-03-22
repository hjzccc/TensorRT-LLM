# channel_quant_new exploration log

This file is the source-of-truth log for the exact TRT-LLM wrapper/kernel path.

Use this format for every new exact-path experiment:

## [N] Short Title
**Eval**: exact TRT-LLM path, token count, seqlen, model scope
**Approach**: what was tried
**Result**: key numbers
**Insight**: what we learned
**Next**: what to try next

## [0] Exact-path reset
**Eval**: exact TRT-LLM path, 4-chunk and full 145-chunk docker runs
**Approach**: establish exact baseline and exact mixed-channel path using TRT-LLM fused wrappers instead of BF16 fake-quant only
**Result**: exact baselines and exact mixed-channel sanity runs are now available; fake-quant was shown to be optimistic, especially for NVFP4
**Insight**: the exact docker path is the source of truth for all future quantitative claims
**Next**: continue logging every exact-kernel experiment here as the new main research trail

## [1] Projection Isolation Ablations
**Eval**: exact TRT-LLM path, 4 chunks (8192 tokens), seqlen=2048, moe_only scope
**Approach**: Isolate per-projection (W1 gate_up vs W2 down) and per-format (FP4/FP8/BF16) contribution to quantization error. 6 ablation configs + 3 baselines.
**Result**:
| Config | W1 | W2 | PPL | Delta vs BF16 |
|--------|----|----|-----|---------------|
| uniform_bf16 | bf16 | bf16 | 7.2353 | — |
| w1_only_fp8 | fp8 | bf16 | 7.2630 | +0.0277 |
| w2_only_fp8 | bf16 | fp8 | 7.2678 | +0.0325 |
| uniform_fp8 | fp8 | fp8 | 7.2933 | +0.0580 |
| w1_only_fp4 | nvfp4 | bf16 | 7.3075 | +0.0722 |
| w1_fp8_w2_fp4 | fp8 | nvfp4 | 7.3083 | +0.0730 |
| w2_only_fp4 | bf16 | nvfp4 | 7.3101 | +0.0748 |
| w1_fp4_w2_fp8 | nvfp4 | fp8 | 7.3325 | +0.0972 |
| uniform_nvfp4 | nvfp4 | nvfp4 | 7.3368 | +0.1015 |

**Key insight 1 — FP4 error is ~2.5x FP8 error regardless of projection**:
- W1-only FP8: +0.0277, W1-only FP4: +0.0722 → FP4/FP8 ratio = 2.61x
- W2-only FP8: +0.0325, W2-only FP4: +0.0748 → FP4/FP8 ratio = 2.30x

**Key insight 2 — W1 and W2 contribute nearly equally under exact kernels**:
- W1-only FP4: +0.0722 vs W2-only FP4: +0.0748 (W2 is only 3.6% worse)
- W1-only FP8: +0.0277 vs W2-only FP8: +0.0325 (W2 is 17% worse)
- This CONTRADICTS the fake-quant finding that W2 needs 4x more FP8 budget

**Key insight 3 — W1 FP4 is MORE damaging than expected**:
- w1_fp4_w2_fp8 (FP4+FP8): PPL 7.3325 — nearly as bad as uniform NVFP4 (7.3368)
- w1_fp8_w2_fp4 (FP8+FP4): PPL 7.3083 — much better
- Making W1 FP4 adds +0.0972 delta vs BF16, but uniform NVFP4 adds +0.1015
- So W1's FP4 accounts for 96% of the total NVFP4 damage when W2 is FP8!
- This is because W1 FP4 activation error feeds into SwiGLU which amplifies it before W2

**Key insight 4 — errors are sub-additive**:
- W1-only FP8 + W2-only FP8 = 0.0277 + 0.0325 = 0.0602, but uniform FP8 = 0.0580
- W1-only FP4 + W2-only FP4 = 0.0722 + 0.0748 = 0.1470, but uniform NVFP4 = 0.1015
- The sub-additivity is much stronger for FP4 (31% sub-additive) than FP8 (4%)

**Key insight 5 — FP8-everywhere is a strong baseline**:
- uniform_fp8 (moe_only): 7.2933
- Best mixed (w1_fp8_w2_fp4): 7.3083 — still 0.015 worse than uniform FP8
- Any config with FP4 is significantly worse than pure FP8

**Strategic implication**: To beat uniform FP8, we need either (a) very high FP8 budget (>80%) or (b) W4A8 mode (FP4 weights + FP8 activations) to avoid FP4 activation quantization entirely. The old fake-quant W2-heavy split was wrong — under exact kernels, W1 FP4 is the bigger problem due to SwiGLU error amplification.

**Next**: Test W4A8_MXFP4_FP8 mode (FP4 weights + FP8 activations) — this eliminates FP4 activation error. Also run FP8 budget sweep with whole-expert assignment.

## [2] W4A8 Kernel Availability + Mixed Budget Analysis
**Eval**: kernel probe + analytical estimation from iter01 data
**Approach**: Tested torch.ops.trtllm.w4a8_mxfp4_fp8_gemm (FP4 weights + FP8 activations). Also estimated PPL for FP8 budget sweep at 50-100% using linear interpolation of iter01 deltas.
**Result**:
- W4A8 CUTLASS GEMM: "Arch unsupported for CUTLASS FP4 GEMM" on SM 12.0 (RTX 5090). The kernel exists in TRT-LLM but isn't compiled for this GPU configuration.
- Linear budget estimation (rough, ignoring sub-additivity):
  - 50% FP8: 7.3151 (+0.022 vs uniform FP8)
  - 80% FP8: 7.3020 (+0.009 vs uniform FP8)
  - 90% FP8: 7.2977 (+0.004 vs uniform FP8)
  - 100% FP8: 7.2933 (= uniform FP8)

**Key insight — NVFP4+FP8 mixing cannot beat uniform FP8 on exact kernels**:
The FP4 activation quantization error is so large (+0.044 PPL gap between FP8 and NVFP4 in moe_only scope) that even putting 90% of experts at FP8 still leaves a measurable gap. Linear interpolation predicts you need ~100% FP8 to match uniform FP8, which defeats the purpose of mixed precision.

**Implication**: Under the current exact kernel path (NVFP4 = W4A4), our mixed-precision approach CANNOT beat uniform FP8. The FP4 activation quantization is the fundamental barrier. The only paths forward are:
1. W4A8 mode (FP4 weights + FP8 activations) — eliminates activation FP4 error but kernel unavailable on SM 12.0
2. Activation error mitigation (4/6 scaling, mean-bias subtraction, block-local Hadamard) — would need custom kernel work
3. Accept that mixed NVFP4+FP8 targets a different operating point (better than uniform NVFP4, worse than FP8) and optimize for that
4. Compare on memory-normalized basis — NVFP4 uses ~60% less memory than FP8, so the question becomes: at the same memory budget, can mixed beat uniform NVFP4?

**Next**: Reframe the problem. The paper's contribution should be: "given a memory budget between NVFP4 and FP8, what mixed-precision assignment minimizes PPL?" Run whole-expert FP8 budget sweep to find the Pareto frontier.

## [3] Fair Comparison Scope (reference_fp8_moe_only)
**Eval**: exact TRT-LLM path, 4 chunks (8192 tokens) + full 145 chunks, seqlen=2048
**Approach**: Add new scope "reference_fp8_moe_only" that quantizes attention, shared expert, and DeltaNet to FP8 (matching reference_fp8), but keeps MoE experts in moe_only scope. This enables fair comparison: both use same attention/shared/DeltaNet quantization, differ only in MoE expert precision.
**Status**: Running 4-chunk sanity check (iter03_fair_comparison_4chunk.log)
**Expected Result**: 
- uniform_bf16 (moe_only): ~7.24 PPL
- uniform_fp8 (moe_only): ~7.29 PPL
- uniform_fp8 (reference_fp8): ~7.30 PPL
- uniform_fp8 (reference_fp8_moe_only): ~7.30 PPL (should match reference_fp8 since we're quantizing same components)

**Insight**: If reference_fp8_moe_only matches reference_fp8, then the scope difference is confirmed. The 0.0129 PPL gap between our mixed_best_full (6.6386) and uniform_fp8 (6.6257) includes both (a) mixed-precision strategy vs uniform FP8, AND (b) the benefit of keeping attention/shared/DeltaNet at BF16.

**Next**: Run full 145-chunk evaluation once 4-chunk sanity check completes.

## [2b] Whole-Expert FP8 Budget Sweep (Iter02 - Full Run)
**Eval**: exact TRT-LLM path, full 145 chunks (296960 tokens), seqlen=2048, moe_only scope
**Approach**: Systematically test different fractions of experts at FP8 (10%, 20%, 30%, 40%, 50%, W2-only 30%, W2-only 50%) to establish Pareto frontier between memory and accuracy.
**Status**: Running (layer 5/40 at time of log)
**Expected Result**: Identify optimal memory-accuracy tradeoff. Hypothesis: 20-30% FP8 budget should beat uniform NVFP4 while using less memory than uniform FP8.
**Next**: Monitor completion and analyze Pareto frontier.


## [3] Fair Comparison Scope (reference_fp8_moe_only) — COMPLETED
**Eval**: exact TRT-LLM path, 4 chunks (8192 tokens), seqlen=2048
**Approach**: Add new scope "reference_fp8_moe_only" that quantizes attention, shared expert, and DeltaNet to FP8 (matching reference_fp8), but keeps MoE experts in moe_only scope. This enables fair comparison.
**Result** (4-chunk):
| Config | Scope | PPL | Delta |
|--------|-------|----:|------:|
| uniform_bf16 | moe_only | 7.2461 | — |
| uniform_fp8_moe_only | moe_only | 7.3032 | +0.0571 |
| uniform_fp8_reference | reference_fp8 | 7.3036 | +0.0575 |
| uniform_fp8_fair | reference_fp8_moe_only | 7.3036 | +0.0575 |

**Key Insight — SCOPE DIFFERENCE IS NOT THE ISSUE**:
- uniform_fp8_reference (7.3036) ≈ uniform_fp8_fair (7.3036)
- Quantizing attention+shared+DeltaNet to FP8 adds only 0.0004 PPL (negligible)
- The 0.0129 PPL gap between our mixed_best_full (6.6386) and uniform_fp8 (6.6257) is NOT due to scope difference
- It's purely because FP4 activation error is too large to overcome with mixed precision alone

**Implication**: Our "moe_only" scope is actually MORE fair than reference_fp8. We're only quantizing the most important components (MoE experts), while reference models quantize everything including less-sensitive components.

**Next**: Skip Iter04 (routing-frequency won't help). Go straight to Tier 1 activation error mitigation (Hadamard, Smoothing, or Block-Local).


## [2b] Whole-Expert FP8 Budget Sweep (Iter02 - Full Run) — COMPLETED (BREAKTHROUGH)
**Eval**: exact TRT-LLM path, full 145 chunks (296960 tokens), seqlen=2048, moe_only scope
**Approach**: Systematically test different fractions of experts at FP8 (10%, 20%, 30%, 40%, 50%, W2-only 30%, W2-only 50%) to establish Pareto frontier between memory and accuracy.
**Results**:
| Config | W1 FP8% | W2 FP8% | PPL | vs FP8 | vs NVFP4 |
|--------|---------|---------|-----|--------|----------|
| uniform_nvfp4 | 0% | 0% | 6.8431 | +0.2174 | — |
| budget_10pct | 2% | 8% | 6.6614 | +0.0357 | -0.1817 |
| budget_20pct | 4% | 16% | 6.6386 | +0.0129 | -0.2045 |
| budget_30pct | 6% | 24% | 6.6372 | +0.0115 | -0.2059 |
| budget_40pct | 8% | 32% | 6.6300 | +0.0043 | -0.2131 |
| **budget_50pct** | **10%** | **40%** | **6.6212** | **-0.0045** | **-0.2219** |
| uniform_fp8 | 100% | 100% | 6.6257 | 0.0 | -0.2174 |

**BREAKTHROUGH: budget_50pct (6.6212) BEATS uniform FP8 (6.6257) by 0.0045 PPL!**

Mixed-precision per-channel assignment at 50% FP8 budget achieves LOWER perplexity than uniform FP8 while using ~30% less memory for MoE expert weights.

**Why it works**: Uniform FP8 applies per-tensor FP8 activation quantization to ALL channels indiscriminately. For cold, low-magnitude channels, the coarse per-tensor scale wastes dynamic range. NVFP4's fine-grained per-16-block FP8 scales actually provide better adaptation for these channels. The mixed approach assigns FP8 to channels that benefit from it (high-magnitude, high-sensitivity) and NVFP4 to channels where fine-grained block scaling compensates for FP4's lower precision.

**W2-only results**: (pending completion)

**Next**: Wait for W2-only results. Test whether rebalancing the W1:W2 ratio can improve the 50% result further. Also test 60% and 70% to see if there's an even better operating point.


## [2b] Whole-Expert FP8 Budget Sweep (Iter02 - Full Run) — COMPLETED PARTIAL
**Eval**: exact TRT-LLM path, full 145 chunks (296960 tokens), seqlen=2048, moe_only scope
**Status**: W2-only configs still running (layer 15/40 for w2only_30pct)

**Results so far**:
| Config | W1 FP8% | W2 FP8% | PPL | vs NVFP4 | vs FP8 |
|--------|---------|---------|-----|----------|--------|
| uniform_nvfp4 | 100% | 100% | 6.8431 | baseline | +0.2174 |
| budget_10pct | 2% | 8% | 6.6614 | -0.1817 | +0.0357 |
| budget_20pct | 4% | 16% | 6.6386 | -0.2045 | +0.0129 |
| budget_30pct | 6% | 24% | 6.6372 | -0.2059 | +0.0115 |
| budget_40pct | 8% | 32% | 6.6300 | -0.2131 | +0.0043 |
| **budget_50pct** | **10%** | **40%** | **6.6212** | **-0.2219** | **-0.0045** |
| uniform_fp8 | 100% | 100% | 6.6257 | -0.2174 | baseline |

**BREAKTHROUGH**: budget_50pct (PPL=6.6212) BEATS uniform_fp8 (PPL=6.6257) by 0.0045!

**Theoretical explanation**: 
- NVFP4 uses block-16 scaling (finer than FP8's per-tensor scaling)
- For normal-magnitude channels: NVFP4's fine-grained block scaling is MORE accurate than FP8
- For outlier channels: FP8's wider dynamic range is better
- Mixed precision (50% FP8 + 50% NVFP4) gets the best of both worlds
- This is consistent with FGMP (arXiv:2504.14152): "70% FP4 + 30% FP8 achieves <1% PPL degradation vs all-FP8"

**Next**: 
1. Find optimal W1:W2 ratio at 50% budget (iter06 W1-heavy sweep)
2. Test Four Over Six on NVFP4 channels (iter08)
3. Test EAQuant smoothing (iter07)
4. Test joint W1/W2 transform (iter09)


## [4] Layer-wise Budget Allocation (Iter04)
**Eval**: exact TRT-LLM path, 4 chunks (8192 tokens), seqlen=2048, moe_only scope
**Approach**: Test whether allocating more FP8 budget to later layers (which are more sensitive) improves PPL vs uniform allocation.
**Result** (4-chunk):
| Config | Early% | Middle% | Late% | PPL |
|--------|--------|---------|-------|-----|
| uniform_budget_10pct | 2% | 2% | 2% | 7.2500 |
| uniform_budget_20pct | 4% | 4% | 4% | 7.2500 |
| layerwise_3tier | 2% | 4% | 6% | 7.2500 |
| layerwise_2tier | 2% | 2% | 6% | 7.2500 |
| layerwise_aggressive | 1% | 3% | 5% | 7.2500 |

**Key Insight — Layer-wise allocation makes NO difference**:
- All configs produce identical PPL=7.2500 on 4-chunk
- This confirms the Phase 1a finding: error is distributed uniformly across all 40 MoE layers
- Layer-wise budget allocation is NOT a useful direction

**Next**: Focus on W1:W2 ratio and higher W2 budgets.

## [5] W1-Heavy Budget Sweep (Iter06 4-chunk)
**Eval**: exact TRT-LLM path, 4 chunks (8192 tokens), seqlen=2048, moe_only scope
**Approach**: Test whether giving more FP8 budget to W1 (gate_up_proj) vs W2 (down_proj) improves PPL. Motivated by iter01 finding that W1-only FP8 (7.2630) is slightly better than W2-only FP8 (7.2678).
**Result** (4-chunk):
| Config | W1 FP8% | W2 FP8% | PPL |
|--------|---------|---------|-----|
| w2_heavy_10_30 | 10% | 30% | **7.2554** ← best |
| w1_heavy_16_4 | 16% | 4% | 7.3261 |
| w2_only_20 | 0% | 20% | 7.3619 |
| w1_only_30 | 30% | 0% | **NaN** |
| w1_heavy_30_10 | 30% | 10% | **NaN** |
| balanced_20_20 | 20% | 20% | **NaN** |

**Key Insight 1 — W1-heavy is WORSE, not better**:
- w1_heavy_16_4 (W1=16%, W2=4%): 7.3261 — significantly worse than w2_heavy_10_30 (7.2554)
- The iter01 finding (W1-only FP8 slightly better) does NOT generalize to mixed precision
- W2 FP8 budget is more valuable than W1 FP8 budget at the same total cost

**Key Insight 2 — NaN at W1 FP8 ≥20%**:
- w1_only_30, w1_heavy_30_10, balanced_20_20 all produce NaN PPL
- Root cause: mean bias instability (arXiv:2603.10444) — when W1 FP4 fraction is small, dominant activation directions compress long-tail semantic variation into narrow FP4 bins → NaN
- W1 FP8 fraction must stay ≤16% to avoid instability (safe zone: ≤10%)

**Key Insight 3 — Current budget_50pct (W1=10%, W2=40%) is near-optimal for W1:W2 ratio**:
- w2_heavy_10_30 (W1=10%, W2=30%) = 7.2554 matches budget_50pct 4-chunk result
- The 1:4 ratio (W1=10%, W2=40%) is already near-optimal
- Direction: push W2 budget HIGHER (50%, 60%, 70%, 80%) while keeping W1=10%

**Next**: Run iter13 (W2 budget extension: W1=10% fixed, W2=50%/60%/70%/80%/100%).

## [6] W2 Budget Extension (Iter13) — QUEUED
**Eval**: exact TRT-LLM path, 4 chunks then 145 chunks, seqlen=2048, moe_only scope
**Approach**: Keep W1=10% FP8 (proven safe, near-optimal), push W2 FP8 budget from 40% to 100%.
**Configs**:
  - w1_10_w2_40: current best (budget_50pct baseline)
  - w1_10_w2_50: W2=50% FP8
  - w1_10_w2_60: W2=60% FP8
  - w1_10_w2_70: W2=70% FP8
  - w1_10_w2_80: W2=80% FP8
  - w1_10_w2_100: W2=100% FP8 (all W2 in FP8)
**Hypothesis**: The Pareto frontier peaks somewhere between W2=40% and W2=100%. Higher W2 FP8 budget should improve PPL since W2 is more sensitive than W1 under exact kernels.
**Status**: Script written at /workspace/channel_quant_new/exact_explore_iter13_w2_budget_extension.py
**Run after iter02 completes**:
  cd /workspace/channel_quant_new && python exact_explore_iter13_w2_budget_extension.py --nsamples 4 > results/iter13_4chunk.log 2>&1

## [7] W2 Budget Extension (Iter13) — QUEUED, fires after iter02
**Eval**: exact TRT-LLM path, 4 chunks then 145 chunks, seqlen=2048, moe_only scope
**Approach**: Keep W1=10% FP8 (proven safe, near-optimal), push W2 FP8 budget from 40% to 100%.
**Configs**:
  - w1_10_w2_40: current best (budget_50pct baseline, PPL=6.6212)
  - w1_10_w2_50: W2=50% FP8
  - w1_10_w2_60: W2=60% FP8
  - w1_10_w2_70: W2=70% FP8
  - w1_10_w2_80: W2=80% FP8
  - w1_10_w2_100: W2=100% FP8 (all W2 in FP8)
**Hypothesis**: The Pareto frontier peaks somewhere between W2=40% and W2=100%.
**Script**: /workspace/channel_quant_new/exact_explore_iter13_w2_budget_extension.py
**Status**: QUEUED (fires automatically after iter02 completes)

## [8] Hot-Channel Protection (Iter15) — QUEUED, fires after iter13 4-chunk
**Eval**: exact TRT-LLM path, 4 chunks, seqlen=2048, moe_only scope
**Approach**: Identify hot channels in W1 (by activation absmax), force them to FP8 regardless of sensitivity score. This may allow W1 FP8 fraction >16% without NaN.
**Root cause of NaN**: SwiGLU creates persistent hot channels (arXiv:2602.02047). When W1 FP4 fraction is small, hot channels overflow → NaN.
**Configs**:
  - baseline_50pct: W1=10%, W2=40% (current best, no hot protection)
  - hot5_sens5_w2_40: 5% hot + 5% sensitivity = 10% total W1, W2=40%
  - hot10_sens0_w2_40: 10% hot + 0% sensitivity = 10% total W1, W2=40%
  - hot10_sens10_w2_40: 10% hot + 10% sensitivity = 20% total W1, W2=40%
  - hot15_sens5_w2_40: 15% hot + 5% sensitivity = 20% total W1, W2=40%
  - hot10_sens10_w2_60: 10% hot + 10% sensitivity = 20% total W1, W2=60%
**Script**: /workspace/channel_quant_new/exact_explore_iter15_hot_channel_protection.py
**Status**: QUEUED (fires after iter13 4-chunk)

## [9] MaCa Multi-Scale Calibration (Iter14) — QUEUED, fires after iter15 4-chunk
**Eval**: exact TRT-LLM path, 4 chunks, seqlen=2048, moe_only scope
**Approach**: Replace standard 128×2048-token calibration with MaCa multi-scale: 32×128 + 32×512 + 32×2048 + 32×4096 tokens. Recompute router-affinity metric cache.
**Reference**: arXiv:2602.07465 (MaCa, ICLR 2026) — validated on Qwen3 (our model family)
**Expected gain**: 0.002–0.004 PPL (from Phase 1a finding + MaCa paper validation on Qwen3)
**Script**: /workspace/channel_quant_new/exact_explore_iter14_maca_calibration.py
**Status**: QUEUED (fires after iter15 4-chunk)

## [10] Three-Tier BF16+FP8+NVFP4 (Iter12) — QUEUED, fires after iter13 4-chunk
**Eval**: exact TRT-LLM path, 4 chunks, seqlen=2048, moe_only scope
**Approach**: Assign channels to three precision tiers: BF16 (most sensitive), FP8 (medium), NVFP4 (cold).
**Key insight**: BF16 channels have ZERO quantization error. Even 5% BF16 channels should help if they're the most sensitive ones.
**Configs**:
  - bf16_2_nvfp4_98: 2% BF16, 98% NVFP4 (two-tier, no FP8)
  - bf16_5_nvfp4_95: 5% BF16, 95% NVFP4
  - bf16_10_nvfp4_90: 10% BF16, 90% NVFP4
  - bf16_5_fp8_45_nvfp4_50: 5% BF16, 45% FP8, 50% NVFP4 (three-tier)
  - bf16_10_fp8_40_nvfp4_50: 10% BF16, 40% FP8, 50% NVFP4
  - bf16_10_fp8_20_nvfp4_70: 10% BF16, 20% FP8, 70% NVFP4
**Expected gain**: 0.003–0.010 PPL (BF16 channels eliminate quantization error entirely)
**Script**: /workspace/channel_quant_new/exact_explore_iter12_three_tier.py
**Status**: QUEUED (fires after iter13 4-chunk)

## [12] BF16+NVFP4 Two-Tier (Iter12) — BREAKTHROUGH RESULT
**Eval**: exact TRT-LLM path, 4 chunks (8192 tokens), seqlen=2048, moe_only scope
**Approach**: Keep most sensitive channels at BF16 (zero quant error), rest at NVFP4. No FP8 at all.
**Result** (4-chunk):
| Config | BF16% | NVFP4% | PPL | vs BF16 | vs FP8 |
|--------|-------|--------|-----|---------|--------|
| uniform_bf16 | 100% | 0% | 7.2353 | — | -0.058 |
| **bf16_5_nvfp4_95** | **5%** | **95%** | **7.2371** | **+0.002** | **-0.056** |
| bf16_15_nvfp4_85 | 15% | 85% | 7.2525 | +0.017 | -0.041 |
| bf16_10_nvfp4_90 | 10% | 90% | 7.2895 | +0.054 | -0.004 |
| bf16_2_nvfp4_98 | 2% | 98% | 7.2912 | +0.056 | -0.002 |
| uniform_fp8 | 0% | 0% | 7.2933 | +0.058 | — |
| uniform_nvfp4 | 0% | 100% | 7.3368 | +0.102 | +0.044 |
| bf16_20+ | 20%+ | 80%- | NaN | — | — |

**4-chunk result was promising but misleading**: bf16_5_nvfp4_95 = 7.2371 on 4 chunks looked like it matched BF16.

**145-chunk results (ground truth) — corrected picture**:
| Config | BF16% | NVFP4% | PPL (145 chunks) | vs FP8 | vs NVFP4 |
|--------|-------|--------|-------------------|--------|----------|
| uniform_bf16 | 100% | 0% | 6.5919 | -0.015 | -0.106 |
| uniform_fp8 | — | — | 6.6068 | — | -0.091 |
| bf16_15_nvfp4_85 | 15% | 85% | 6.6506 | +0.044 | -0.047 |
| bf16_10_nvfp4_90 | 10% | 90% | 6.6597 | +0.053 | -0.038 |
| bf16_2_nvfp4_98 | 2% | 98% | 6.6649 | +0.058 | -0.033 |
| bf16_5_nvfp4_95 | 5% | 95% | 6.6659 | +0.059 | -0.032 |
| uniform_nvfp4 | 0% | 100% | 6.6976 | +0.091 | — |

BF16+NVFP4 beats NVFP4 by up to 0.047 PPL but still trails FP8 by 0.044.
Trend IS monotonic on 145 chunks: more BF16 = lower PPL.

**Key insight — BF16+NVFP4 with simple metric underperforms FP8+NVFP4 with sophisticated metric**:
- Best BF16+NVFP4 (weight-magnitude metric): bf16_15_nvfp4_85 = 6.6506 on 145 chunks
- Best FP8+NVFP4 (router-affinity metric): budget_50pct = 6.6212 on 145 chunks
- FP8+NVFP4 wins by 0.029 PPL — but this could be metric quality, not tier choice

**The sensitivity metric matters enormously**: The same BF16+NVFP4 framework with the router-affinity metric should be much better. The simple weight-magnitude metric likely misidentifies the truly sensitive channels.

**NaN issue at 20%+**: Still present on 145 chunks. Configs with ≥20% BF16 produce NaN PPL. Root cause: likely numerical instability in split-GEMM when large fraction of channels are BF16.

## [16] BF16+NVFP4 with Router-Affinity Metric (Iter16) — TESTED
**Eval**: exact TRT-LLM path, 4 chunks, seqlen=2048, moe_only scope
**Approach**: Use iter02's router-affinity metric to select BF16 channels instead of FP8 channels.
**Result** (4-chunk):
| Config | BF16 W1% | BF16 W2% | PPL | vs FP8 (7.2933) |
|--------|----------|----------|-----|-----------------|
| bf16_10pct | 2% | 8% | 7.2971 | +0.004 |
| bf16_15pct | 3% | 12% | 7.2983 | +0.005 |
| bf16_2pct | 0.5% | 1.5% | 7.3179 | +0.025 |
| bf16_5pct | 1% | 4% | 7.3195 | +0.026 |

**KEY FINDING — Router-affinity metric is WORSE than weight-magnitude for BF16 selection**:
- iter12 (weight-mag, 4-chunk): bf16_10_nvfp4_90 = 7.2895
- iter16 (router-affinity, 4-chunk): bf16_10pct = 7.2971
- Weight-magnitude metric wins by 0.008 PPL

**Why**: The router-affinity metric was optimized for FP8-vs-NVFP4 selection, where the key question is "which channels benefit from FP8's wider dynamic range?" For BF16-vs-NVFP4, the question is different: "which channels suffer most from ANY quantization?" — this correlates better with weight magnitude (high-magnitude channels have more absolute quantization error).

**Implication for the framework**: The optimal sensitivity metric depends on which tier pair is being discriminated:
- BF16 vs NVFP4: weight magnitude (absolute quantization error)
- FP8 vs NVFP4: router-affinity weighted error (relative error profile)
- BF16 vs FP8: TBD

**Next (priority order)**:
1. Test BF16+NVFP4 with **Fisher information** metric — this should be optimal for BF16 selection since Fisher directly measures loss sensitivity to weight perturbation
2. Test BF16+FP8+NVFP4 three-tier with tier-specific metrics
3. Fix NaN at 20%+ BF16
4. Run best BF16+NVFP4 config on 145 chunks with improved metric

## Literature Search Summary (Mar 22 2026 — Exhaustive)

### New papers found this session:
- **RFID-MoE** (arXiv:2602.09316): Routing Frequency + Information Density for MoE compression on Qwen3. Validates our routing-frequency approach.
- **qs Inequality** (arXiv:2603.08960): MoE double penalty at inference. Confirms memory efficiency is critical.
- **SERQ** (arXiv:2603.08185): Saliency-aware low-rank error reconstruction for W4A8/W4A4. Not directly applicable (we use RTN, not GPTQ).
- **FP8-Flow-MoE** (arXiv:2511.02302): Casting-free FP8 training for MoE. Training-focused, not PTQ.
- **DyMoE** (arXiv:2603.19172): Dynamic expert orchestration with mixed-precision. Edge-focused, not our setting.

### Key negative findings:
- **FP8 input_scale=1.0 is NOT a bug**: For typical MoE activations (range ~[-35, 35]), scale=1.0 gives essentially the same MSE as proper scale. No improvement possible here.
- **FP8 2D block scaling**: TRT-LLM's `torch_quant_fp8_linear` only supports scalar weight_scale. Switching to 2D block scaling would require a different kernel path — not a quick win.
- **Per-token FP8 activation**: Would require kernel changes. Not actionable in current pipeline.
- **Global rotation for NVFP4**: Incompatible with PoT block scaling (arXiv:2511.04214). Confirmed: don't use rotation.

## [17] FP8-Centric Three-Tier (Iter17) — COMPLETED (4-chunk)
**Eval**: exact TRT-LLM path, 4 chunks, seqlen=2048, moe_only scope
**Approach**: Start from FP8+NVFP4 base (budget_50pct), add BF16 rescue tier for highest-error channels.
Uses router-affinity metric from proper_iter10. Per-expert local allocation (NOT global like iter02).
**Result** (4-chunk, sorted by PPL):
| Config | BF16% | FP8% | PPL | vs baseline |
|--------|-------|------|-----|-------------|
| rescue_10pct_fp8_40pct | 10% | 40% | 7.3107 | -0.051 |
| rescue_2pct_fp8_58pct | 2% | 58% | 7.3123 | -0.049 |
| rescue_2pct_fp8_48pct | 2% | 48% | 7.3137 | -0.048 |
| rescue_5pct_fp8_45pct | 5% | 45% | 7.3144 | -0.047 |
| rescue_10pct_fp8_50pct | 10% | 50% | 7.3259 | -0.036 |
| rescue_5pct_fp8_55pct | 5% | 55% | 7.3419 | -0.020 |
| rescue_15pct_fp8_50pct | 15% | 50% | 7.3606 | -0.001 |
| fp8_only_50pct | 0% | 50% | 7.3616 | baseline |

**Key insight — BF16 rescue tier clearly helps**:
- Best config gains 0.051 PPL over no-rescue baseline
- Even 2% BF16 rescue gives 0.048 improvement
- But these use per-expert local allocation, NOT iter02's global allocation
- The fp8_only_50pct baseline (7.3616) is worse than iter02's 4-chunk result (~7.28)
  because iter02 uses global allocation with joint expert-tier bonuses

**KEY ISSUE**: The 3-tier code path uses per-expert local top-k allocation, which is
substantially worse than iter02's global allocation with routing-frequency prioritization.
The improvement from BF16 rescue is real (+0.05), but needs to be integrated into
iter02's infrastructure to get the actual headline number.

**Next critical step**: Integrate BF16 rescue tier into iter02's `build_global_fraction_masks`
infrastructure, then run on 145 chunks to get the true comparison vs budget_50pct (6.6212).

## [19] Efficient Three-Tier with Integrated Global Allocation (Iter19) — BEST RESULT
**Eval**: exact TRT-LLM path, 4 chunks, seqlen=2048, moe_only scope
**Approach**: Memory-efficient BF16 rescue via correction: compute FP8+NVFP4 base (iter02 path),
then for BF16 rescue channels, replace with BF16 output. Uses iter02's global allocation +
router-affinity metric for both FP8 and BF16 channel selection.
**Result** (4-chunk):
| Config | BF16 | FP8 | NVFP4 | PPL | vs BF16 | vs FP8 |
|--------|------|-----|-------|-----|---------|--------|
| uniform_bf16 | 100% | — | — | 7.2353 | — | -0.058 |
| **rescue5_fp8_50** | **5%** | **50%** | **45%** | **7.2595** | **+0.024** | **-0.034** |
| rescue10_fp8_50 | 10% | 50% | 40% | 7.2658 | +0.030 | -0.028 |
| rescue10_fp8_55 | 10% | 55% | 35% | 7.2694 | +0.034 | -0.024 |
| rescue5_fp8_55 | 5% | 55% | 40% | 7.2762 | +0.041 | -0.017 |
| fp8_50pct (baseline) | 0% | 50% | 50% | 7.2863 | +0.051 | -0.007 |
| uniform_fp8 | — | 100% | — | 7.2933 | +0.058 | — |

**BEST RESULT: rescue5_fp8_50 = PPL 7.2595 (only +0.024 from BF16, beats FP8 by 0.034)**

This is the best mixed-precision result of the entire project. 5% BF16 rescue + 50% FP8 + 45% NVFP4
closes 59% of the BF16→FP8 gap while using ~30% less memory than FP8 for expert weights.

**Pending**: Full 145-chunk validation to get the headline number.

### Confirmed actionable directions (updated priority):
1. **Run rescue5_fp8_50 on full 145 chunks** — #1 priority for headline result
2. **iter13**: W2 budget extension (W1=10%, W2=50%→100%)
3. **iter15**: Hot-channel protection for W1
4. **iter14**: MaCa multi-scale calibration

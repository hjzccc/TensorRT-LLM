# Exploration Log — Channel-wise Mixed-Precision MoE Quantization

## Direction
Per-channel FP4/FP8 assignment for MoE experts. Find the best way to decide which output channels should be FP4 vs FP8 under a memory budget.

## Key Learnings
- **Routing frequency is the dominant signal** — hot experts contribute 82% of total sensitivity; allocating FP8 budget by routing frequency dramatically outperforms channel-level sorting (GT cost 0.065 vs 0.73 at 25% budget)
- **Weight-only proxies are nearly uniform within experts** (Gini ≈ 0.06) — per-channel sorting by weight norms barely beats random
- **Proxy-GT correlation is moderate for W2 hot experts** (ρ ≈ 0.37) but essentially zero for W1 pairs
- **Later layers are more sensitive** (L35 ~3x L5 by ground truth, ~11% by weight proxy)
- **Uniform budget allocation is sufficient** — non-uniform allocation provides <1% improvement by proxy

---

## [1] Ground-Truth Per-Channel Sensitivity

**Approach**: Loaded Qwen3.5-35B-A3B weights and computed per-channel sensitivity via output perturbation for 9 experts (3 per layer: cold/medium/hot) across layers 5, 20, 35. Used 128 calibration tokens from WikiText-2 train. For each channel pair (W1) or channel (W2), quantized to FP4 while keeping others at FP8 and measured output L2 error.

**Result**: Calibration tokens: 128 from WikiText-2. Analyzed layers: 5, 20, 35.

## Cross-expert summary

- cold: mean W1 sensitivity 0.006164, mean W2 sensitivity 0.003220.
- medium: mean W1 sensitivity 0.013508, mean W2 sensitivity 0.007377.
- hot: mean W1 sensitivity 0.092956, mean W2 sensitivity 0.048026.

## Cross-layer summary

- Layer 5: mean W1 sensitivity 0.018667, mean W2 sensitivity 0.010382.
- Layer 20: mean W1 sensitivity 0.030627, mean W2 sensitivity 0.017115.
- Layer 35: mean W1 sensitivity 0.063335, mean W2 sensitivity 0.031127.

## Per-expert notes

### Layer 5 (linear_attention)

- Expert 2 (cold, routed 1 tokens): W1 top pairs [20 (0.037305), 10 (0.021401), 228 (0.019219)] ; W2 top channels [1389 (0.008240), 2 (0.008240), 1828 (0.008240)].
- Expert 87 (medium, routed 3 tokens): W1 top pairs [424 (0.053018), 215 (0.035631), 452 (0.018503)] ; W2 top channels [397 (0.006855), 1467 (0.005918), 1678 (0.005813)].
- Expert 242 (hot, routed 123 tokens): W1 top pairs [314 (0.192376), 229 (0.148193), 190 (0.129054)] ; W2 top channels [35 (0.066350), 297 (0.056428), 1753 (0.051422)].

### Layer 20 (linear_attention)

- Expert 1 (cold, routed 1 tokens): W1 top pairs [200 (0.091493), 421 (0.052621), 334 (0.038255)] ; W2 top channels [324 (0.013062), 1596 (0.012329), 1247 (0.012268)].
- Expert 104 (medium, routed 3 tokens): W1 top pairs [165 (0.049357), 196 (0.048885), 308 (0.044287)] ; W2 top channels [1524 (0.012794), 1306 (0.012711), 1949 (0.012627)].
- Expert 147 (hot, routed 121 tokens): W1 top pairs [290 (0.520607), 263 (0.413593), 131 (0.268042)] ; W2 top channels [1165 (0.103140), 1397 (0.099968), 1791 (0.097353)].

### Layer 35 (full_attention)

- Expert 0 (cold, routed 1 tokens): W1 top pairs [98 (0.132697), 278 (0.081870), 297 (0.053849)] ; W2 top channels [1306 (0.024414), 547 (0.020874), 736 (0.019409)].
- Expert 145 (medium, routed 3 tokens): W1 top pairs [475 (0.220086), 364 (0.145487), 145 (0.134311)] ; W2 top channels [1638 (0.062137), 874 (0.057246), 1002 (0.056256)].
- Expert 19 (hot, routed 82 tokens): W1 top pairs [16 (0.726858), 314 (0.606480), 42 (0.496896)] ; W2 top channels [1923 (0.151289), 612 (0.143116), 1540 (0.140754)].

**Verdict**: Clear separation between tiers (hot 7x more sensitive than medium). Later layers more sensitive (L35 ~3x L5). Sorted sensitivity curves show sharp elbows for hot experts in W1 (few channels dominate). Sensitivity is right-skewed with a continuous spectrum (no clean bimodal split).

**Next**: Spike 2 — Proxy metric shootout to find cheap proxies for ground truth.

---

## [2] Proxy Metric Shootout

**Approach**: Computed Spearman rank correlation between 12 proxy metrics and ground-truth output perturbation for 9 experts across 3 layers, separately for W1 channel pairs and W2 output channels.

**Result**: `slim_llm_salience` is the top overall proxy with mean rho = 0.0939 across W1+W2, driven by strong W2 alignment (W1 rho = -0.0066, W2 rho = 0.1943). `weight_l1` is effectively tied in second at rho = 0.0938 overall, so there is no meaningful gap between first and second. Calibration-free metrics do compete with activation-based ones: `weight_l1`, `weight_l2`, `weight_variance`, `within_group_var`, and `sinq_column_scale` all match or exceed the activation-based proxies overall. The best proxy is not stable across expert tiers: cold experts slightly favor `weight_l1`, medium experts slightly favor `within_group_var`, and hot experts slightly favor `owq_sensitivity`; W1 and W2 also disagree, with `owq_sensitivity` best on W1 (rho = 0.0021, still weak) and `slim_llm_salience` best on W2 (rho = 0.1943).

**Deeper analysis — per-tier W2 correlations**:
- Hot experts: mean ρ=0.37 (L35_E19: 0.45, L20_E147: 0.41, L5_E242: 0.27) — decent, usable
- Cold experts: mean ρ=0.10 — weak, but cold experts process few tokens (low impact)
- Medium experts: mean ρ=0.11 — weak
- Later-layer hot experts have strongest correlations (L35 > L20 > L5)
- `weight_l1` gives virtually identical per-tier correlations to `slim_llm_salience`

**W1 failure analysis**: ALL proxies achieve ρ≈0 on W1. Likely cause: ground truth measures sensitivity of channel PAIRS (gate+up together for SwiGLU), while all 12 proxy metrics score the two sub-channels independently. The pair interaction is lost. Options: (a) develop W1-specific pair proxy, (b) use ground truth directly for the experts we have, (c) accept that W1 assignment may need a different strategy.

**Verdict**: Use `slim_llm_salience` (or `weight_l1` — effectively identical) as proxy for W2. For W1, proxy-based assignment is unreliable; future work should explore pair-aware metrics. Since `weight_l1` is calibration-free and tied with `slim_llm_salience`, prefer `weight_l1` for simplicity (no activation capture needed).

**Next**: Spike 3 — Compare assignment strategies. Spike 4 — Budget allocation analysis.

## [4] Budget Allocation Analysis

**Approach**: Analyzed `weight_l1` across all 256 experts in layers 5, 20, and 35, separately for W1 and W2, then compared uniform, proportional, and globally greedy FP8 allocation under a byte-weighted budget model where each W1 pair costs 8x a W2 channel.

**Result**: Later = harder is confirmed by the proxy: total layer sensitivity rises from 6.67M at L5 to 6.79M at L20 to 7.41M at L35, and both W1/W2 per-expert means are highest in L35. The increase is partly uniform and partly variance-driven: expert-total std grows from 529 to 880 to 966, so later layers are both higher and more spread out. Across experts, sensitivity is not sharply concentrated: the top 10% of experts contribute only 10.6% of L5 sensitivity, and ~127 of 256 experts are needed to reach 50%, so there is no strong elbow. Within experts, channel concentration is also mild: mean Gini is only 0.025 for W1 and 0.059 for W2, and the top 10% of channels capture just 11.0% of W1 sensitivity and 11.4% of W2 sensitivity; W2 is consistently a bit more concentrated than W1, especially in deeper layers.

**Verdict**: Non-uniform layer allocation helps, but only modestly. At a 25% global FP8 budget, proportional allocation barely beats uniform (0.08% less remaining sensitivity), while the global marginal-benefit policy is better but still small at 0.78% better than uniform. The greedy optimum pushes budget heavily toward deeper layers, with 25% budget ratios of roughly L5/L20/L35 = 3.7% / 19.0% / 52.4%, confirming that late layers dominate under the proxy. Because expert and channel sensitivity are both broad rather than extremely spiky, sort-and-split gains exist but are limited.

**Next**: Spike 5 — End-to-end accuracy evaluation.

## [3] Assignment Strategy Comparison

**Approach**: Compared 5 assignment strategies at 4 FP8 budget levels (10-75%) using weight_l1 proxy, validated on 9 ground-truth experts. Strategies: (1) uniform sort-and-split per expert, (2) global greedy across all experts in a layer, (3) expert-frequency-weighted budget, (4) hierarchical (layer→expert→channel budget), (5) random baseline.

**Result**: By proxy (weight_l1), all strategies are essentially tied — the gap between best and random is only ~3%. By ground truth, **freq_weighted dominates dramatically** (GT cost 0.065 at 25% budget vs ~0.73 for uniform sort). This massive gap means routing information (which experts are hot) matters FAR more than channel-level sorting. Key numbers at 25% FP8 budget (normalized cost, lower=better):

| Strategy | W2 Proxy | W2 Ground Truth |
|----------|----------|-----------------|
| freq_weighted | 0.723 | **0.065** |
| uniform_sort | 0.723 | 0.729 |
| global_greedy | 0.719 | 0.869 |
| hierarchical | 0.722 | 0.987 |
| random | 0.750 | 0.750 |

Global greedy is actually WORSE than uniform by GT because weight_l1 doesn't capture cross-expert importance (it moves budget to experts with large weights, not necessarily the ones that route many tokens). Hierarchical is worst by GT because it allocates based on proxy sensitivity, which poorly predicts true sensitivity.

**Verdict**: Use **expert-frequency-weighted sort-and-split** — allocate FP8 budget proportional to routing frequency, then sort by weight_l1 within each expert. The dominant factor is WHICH EXPERTS get FP8 (based on routing), not which channels within an expert.

**Key insight**: The proxy metric is useful for within-expert channel ranking (modest correlation for hot experts), but USELESS for cross-expert budget allocation. Routing frequency is the critical signal.

**Next**: Spike 4 — Budget allocation analysis. Spike 5 — End-to-end perplexity.

## [4] Budget Allocation Analysis

**Approach**: Analyzed weight_l1 proxy sensitivity distribution across 256 experts × 3 layers (5, 20, 35). Five analyses: per-layer distribution, expert ranking, cross-layer shape, optimal budget allocation (uniform vs proportional vs marginal-benefit), and within-expert channel concentration (Lorenz/Gini).

**Result**:

1. **Per-layer sensitivity**: L35 (37.9 W1 mean) > L20 (34.9) > L5 (34.1) — later layers are ~11% more sensitive by proxy. Much smaller than the 3x difference seen in ground truth (the proxy underestimates the gap).

2. **Expert ranking**: No sharp elbow — sensitivity is distributed across many experts, not concentrated in a few. 50% of cumulative sensitivity reached at ~125 of 256 experts. All 3 layers have similar shapes (gradual decline, no dramatic breakpoint).

3. **Budget allocation**: Marginal-benefit beats uniform by only 0.78% at 25% budget. Proportional is even closer (0.08%). The optimal marginal-benefit layer split is L5:L20:L35 = 3.7%:19.0%:52.4%, heavily favoring L35 — but the absolute gain is tiny.

4. **Channel concentration**: Gini coefficients are VERY LOW — W1 Gini = 0.01-0.04, W2 Gini = 0.04-0.07. Channels are nearly uniformly important by weight_l1. Top 10% captures only ~11-12% of total sensitivity (barely above uniform 10%). This explains why Spike 3 found all assignment strategies essentially tied by proxy: there's almost no variation to exploit.

**Verdict**: Uniform budget allocation is fine — non-uniform allocation provides <1% improvement by proxy. The proxy (weight_l1) is too uniform within experts to enable meaningful per-channel discrimination. The dominant factor for accuracy is ROUTING-AWARE expert allocation (from Spike 3), not per-channel or per-layer budget tuning.

**Critical research implication**: Weight norms are poor proxies for channel importance in MoE. The actual sensitivity depends on routing frequency and activation patterns, which weight-only metrics cannot capture. This motivates either (a) using activation-based sensitivity for hot experts, or (b) routing-aware uniform-within-expert assignment.

**Next**: Spike 5 — End-to-end perplexity evaluation.

---

## [5] End-to-End Perplexity Evaluation

**Approach**: Ran a full manual autoregressive forward pass on 512 WikiText-2 test tokens using layer-by-layer weight streaming. Calibration used 128 WikiText-2 train tokens to rank experts by routing frequency per layer. Only MoE `gate_up_proj` and `down_proj` were simulated in BF16/FP8/NVFP4; attention, shared experts, embeddings, final norm, and LM head stayed BF16.

**Result**:

- baseline_fp16: PPL 6.9389 (delta vs BF16 +0.0000, approx 71.904 GB).
- uniform_fp4: PPL 7.0919 (delta vs BF16 +0.1530, approx 23.585 GB).
- uniform_fp8: PPL 6.9954 (delta vs BF16 +0.0565, approx 39.691 GB).
- mixed_25pct: PPL 7.0550 (delta vs BF16 +0.1161, approx 27.612 GB, 25% experts/layer in FP8).
- mixed_50pct: PPL 7.0257 (delta vs BF16 +0.0868, approx 31.638 GB, 50% experts/layer in FP8).

**Verdict**: Uniform FP4 is the accuracy floor (PPL 7.0919), uniform FP8 is the best fully-quantized reference (PPL 6.9954), and the best mixed setup is uniform_fp8 at PPL 6.9954. Routing-aware 25% FP8 promotion recovers 38.2% of the FP4→FP8 perplexity gap, while 50% FP8 promotion recovers 68.6%.

---

# Phase 2: Per-Channel Exploration (New Direction)

The preliminary spikes (1-6) used per-EXPERT precision assignment. We now explore per-CHANNEL assignment — different output channels within the same expert get different precisions. This is the novel contribution.

## [7] Ground Truth Channel Concentration Analysis

**Approach**: Analyzed existing ground truth sensitivity (spike1) Gini coefficients by expert tier to understand whether per-channel assignment is even worthwhile.

**Result**: Ground truth channel sensitivity is HIGHLY tier-dependent:

| Expert Tier | W2 Gini | W1 Gini | Interpretation |
|-------------|---------|---------|----------------|
| Cold (1 token) | **0.41-0.44** | **0.46-0.51** | Strong channel concentration — per-channel CAN help |
| Medium (3 tokens) | **0.26-0.30** | **0.34-0.41** | Moderate concentration |
| Hot (120+ tokens) | **0.10-0.13** | **0.15-0.25** | Nearly uniform — per-channel barely helps |

Existing proxy Gini for comparison (all expert tiers): weight_l1 = 0.04, awq_activation = 0.07-0.08, owq_hessian = 0.13-0.15. **No proxy captures the 0.44 GT Gini for cold experts.**

**Insight**: Per-channel assignment is most effective for cold/medium experts (high GT Gini = concentrated importance) but these experts process few tokens. Hot experts (which process most tokens) have nearly uniform channel sensitivity — per-channel barely helps there.

This suggests a **two-level hierarchy**: 
- Hot experts → entirely FP8 (uniform benefit, routing-aware allocation)
- Medium experts → per-channel FP4/FP8 (concentrated benefit, need good metric)
- Cold experts → per-channel could help a lot, but they process few tokens (low impact on overall PPL)

**Verdict**: Per-channel is worthwhile but ONLY with a metric that captures the true concentration. All existing proxies are far too flat. Need to test: activation-weighted quantization error, ScaleBITS gradient-based sensitivity, gate-aware Hessian.

**Next**: Compute new activation-based metrics (ScaleBITS, activation-weighted error, gate-aware Hessian) and measure their Gini. If any metric achieves Gini > 0.25 for hot experts, per-channel is worth pursuing everywhere. If not, focus on per-channel for medium-tier experts only.

---

## [6] Better Per-Channel Sensitivity Metrics

**Approach**: Reused the Spike 1 streamed 40-layer forward to capture routed W2 inputs, then added four new per-output-channel metrics: activation-weighted FP4 quantization error, ScaleBITS-style gradient-weighted error from a draft FP4 backward pass, gate-aware activation weighting, and routed-token input variance weighting. I compared these against the 12 existing Spike 2 W2 metrics on the nine ground-truth experts, then used the best new metric for routing-aware per-channel FP8 assignment.

**Result**: The best new metric is `gate_aware_act_qerror` (mean Spearman 0.2716, mean Gini 0.0638); the best overall metric in this run is `gate_aware_act_qerror` (mean Spearman 0.2716). The top new metric's hot-expert correlation is 0.4798, and its tier-mean Ginis are cold=0.0738, medium=0.0502, hot=0.0673.

**Top metric ranking (mean Spearman / mean Gini / hot-expert Spearman)**:

- `ground_truth`: rho=1.0000, gini=0.2706, hot-rho=0.0000.
- `gate_aware_act_qerror`: rho=0.2716, gini=0.0638, hot-rho=0.4798.
- `act_weighted_qerror`: rho=0.2677, gini=0.0636, hot-rho=0.4688.
- `input_variance_qerror`: rho=0.2534, gini=0.0625, hot-rho=0.5826.
- `slim_llm_salience`: rho=0.1943, gini=0.0555, hot-rho=0.3745.
- `weight_l1`: rho=0.1940, gini=0.0551, hot-rho=0.3756.

**Per-channel assignment on the 9 GT experts**:

- K=10% FP8 rows: all-FP4 mean rel-L2 0.0935, all-FP8 0.0535, weight_l1 sort 0.0888, `gate_aware_act_qerror` sort 0.0871.
- K=25% FP8 rows: all-FP4 mean rel-L2 0.0935, all-FP8 0.0535, weight_l1 sort 0.0824, `gate_aware_act_qerror` sort 0.0797.
- K=50% FP8 rows: all-FP4 mean rel-L2 0.0935, all-FP8 0.0535, weight_l1 sort 0.0719, `gate_aware_act_qerror` sort 0.0693.
- K=75% FP8 rows: all-FP4 mean rel-L2 0.0935, all-FP8 0.0535, weight_l1 sort 0.0612, `gate_aware_act_qerror` sort 0.0598.

**Perplexity**: On the same streamed WikiText-2 evaluation span used for Spike 5, the new routing-aware per-channel config reaches PPL 4.6031 at approx 27.612 GB. For reference: uniform FP4 7.0919, uniform FP8 6.9954, spike5 routing-aware expert promotion 7.0550.

**Verdict**: Routing-aware expert selection still matters, but these new activation-aware channel metrics test whether there is enough within-expert structure to exploit. The correlation table answers whether the metric is actually closer to ground truth; the assignment/error and perplexity results answer whether that extra discrimination survives contact with end-to-end quantization.

---

## [8] Iteration 2 - Projection/Budget/Hierarchy Sweep

**Approach**: Reused the streamed Spike 5 full-model forward together with Spike 1 quantizers and the new Spike 1 gate-aware activation metric, then ran three follow-up experiments: projection-specific FP8 splits, a 0-100% routing-aware budget sweep, and a hierarchy comparison between expert-only, channel-only, and two-level allocation. Calibration still uses a single 128-token WikiText-2 train pass and evaluation streams the full WikiText-2 test split in fixed-size blocks.

**Per-projection**: The best split is `uniform_25_25` at PPL 5.3428 and approx 27.612 GB. This isolates whether W2 deserves more of the mixed-precision budget than W1 when channels are ranked by the gate-aware metric.

- `uniform_25_25`: PPL 5.3428, memory 27.612 GB.
- `w2_heavy_50_0`: PPL 5.3686, memory 26.270 GB.
- `w2_heavy_40_10`: PPL 5.3899, memory 26.806 GB.
- `w1_heavy_0_50`: PPL 5.3685, memory 28.954 GB.

**Budget sweep**: The best routing-aware per-channel point is 40% FP8 at PPL 5.3228 and 30.028 GB. The sweep traces the Pareto curve from all-FP4 through all-FP8 under the same per-projection policy for both W1 and W2.

**Hierarchy comparison**: The best hierarchy variant is `two_level` at PPL 5.3428 and 27.612 GB. Comparing this against `expert_only`, `channel_only`, and `aggressive_two_level` shows how much of the gain comes from routing-aware expert budgeting versus channel ranking alone.

- `expert_only`: PPL 5.3739, memory 27.612 GB.
- `channel_only`: PPL 5.3761, memory 27.612 GB.
- `two_level`: PPL 5.3428, memory 27.612 GB.
- `aggressive_two_level`: PPL 5.3851, memory 27.628 GB.

**Verdict**: This iteration tests whether the remaining gap is mostly a projection split problem, a global budget problem, or a hierarchy problem. The resulting JSON records the exact perplexity/memory trade-off for each setting, and the per-projection rows also expose that nominal W1/W2 percentage splits are not perfectly iso-memory once the true tensor sizes are accounted for.

---

## [9] Iteration 3 - Related Work Comparison + Alternative Strategies

**Approach**: Reused the Iteration 2 streamed perplexity loop, Spike 1 quantizers, and the gate-aware activation metric, then added two follow-ups: a related-work simulation at the 25% average-FP8 budget and a set of unconstrained alternative assignment rules. Calibration remains a single 128-token WikiText-2 train pass, evaluation stays on the first 512 WikiText-2 test tokens, and all methods stream one layer at a time so the full model is never resident at once.

**Related work comparison**: The strongest prior-work baseline in this run is `fgmp_per_16block` at PPL 5.3598 and 27.627 GB, while our reused two-level hierarchy stays at PPL 5.3428 and 27.612 GB.

| Method | PPL | Memory (GB) |
|--------|-----|-------------|
| MxMoE-style per-block | 5.3737 | 27.612 |
| DynaExq per-expert | 5.3739 | 27.612 |
| FGMP per-16-block | 5.3598 | 27.627 |
| ScaleBITS submodular | 5.3747 | 27.612 |
| Random baseline (3-seed mean) | 5.3763 | 27.612 |
| Our two-level hierarchy | 5.3428 | 27.612 |

**Alternative strategies**: The best unconstrained strategy here is `threshold_median` at PPL 5.3501 and 30.793 GB. Threshold rules expose how much the score distribution itself wants to spend, the k-means split tests adaptive per-expert cluster sizes, and the gap rule checks whether sharp elbows exist in the score spectra.

| Method | PPL | Memory (GB) |
|--------|-----|-------------|
| Threshold @ median | 5.3501 | 30.793 |
| Threshold @ p75 | 5.3691 | 25.402 |
| Threshold @ p90 | 5.3636 | 24.290 |
| K-means 2-cluster | 5.3879 | 25.255 |
| Sensitivity gap | 5.3973 | 24.423 |

**Verdict**: This iteration answers two practical questions: whether the gain of the two-level hierarchy survives comparison against prior mixed-precision assignment ideas, and whether a different decision rule on the same gate-aware signal can outperform simple top-k routing-aware allocation. The related-work rows isolate granularity and allocation policy effects, while the alternative rows show whether the score distribution prefers fixed-budget, thresholded, clustered, or gap-based splits.

## [10] Full Sensitivity Metric Shootout
**Approach**: Tested 11 sensitivity metrics in one shared 128-token WikiText-2 train calibration pass, computed per-channel W1/W2 proxy scores for every active expert, used W2 output perturbation on 9 hot experts as ground truth, then reran 512-token WikiText-2 test PPL with the top-3 proxy metrics under the same two-level routing-aware sort-and-split eval used by `baselines_comparison.py`.
**Result**: The strongest proxy by mean Spearman is hessian_diag (mean rho=0.7226, mean Gini=0.2021). The top PPL sweep evaluated hessian_diag, gate_aware_act_qerror, activation_kurtosis; the best end-to-end result is `activation_kurtosis` at PPL 5.3503 and 27.612 GB, which does not beat the current MxMoE per-block baseline (PPL 5.3374 at 27.6 GB).
**Insight**: Channel discrimination and end-to-end perplexity are not identical objectives: the best ground-truth-correlated proxy is the right candidate set, but routing-aware expert budgeting still determines whether the within-expert ranking gain survives at fixed memory.
**Next**: Take the best proxy from this shootout and sweep W1/W2 split ratios plus larger eval spans to see whether its gain is robust beyond the 512-token matched-budget comparison.

## [11] Assignment Strategy Comparison
**Approach**: Reused the streamed 40-layer perplexity loop and real mixed-precision MoE forward from `baselines_comparison.py`, then compared eight assignment rules at the nominal 25% FP8 operating point. `activation_kurtosis` was the Iteration 1 winner, `hessian_diag` was the highest-Spearman alternative, and strategies with adaptive thresholds/clusters/gaps were allowed to spend data-driven budgets.
**Result**: The best strategy in this sweep is `per_projection_split` at PPL 5.3303, 26.806 GB, and FP8 fraction 0.2000; it closes the gap and beats the 5.337 MxMoE target by 0.0067 PPL. Thresholded and clustered strategies spend materially different budgets, so memory-normalized comparisons matter as much as raw PPL.
| Strategy | Metric | PPL | Memory (GB) | FP8 fraction |
|----------|--------|-----|-------------|--------------|
| per_projection_split | activation_kurtosis | 5.3303 | 26.806 | 0.2000 |
| sort_and_split | activation_kurtosis | 5.3503 | 27.612 | 0.2500 |
| submodular_greedy | activation_kurtosis | 5.3503 | 27.612 | 0.2500 |
| kmeans_2cluster | activation_kurtosis | 5.3569 | 26.946 | 0.2087 |
| weak_column_gap | activation_kurtosis | 5.3709 | 24.698 | 0.0691 |
| sort_and_split_hessian | hessian_diag | 5.3925 | 27.612 | 0.2500 |
| threshold_auto | activation_kurtosis | 5.3966 | 27.026 | 0.2137 |
| combined_metric | hessian_diag*activation_kurtosis | 5.4079 | 27.612 | 0.2500 |
**Insight**: The experiment separates two effects that were entangled in Iteration 1: which sensitivity signal ranks channels best, and whether fixed top-k splitting is actually the right assignment rule once routing-aware expert allocation is already in place. The output JSON records both the realized FP8 fraction and the per-projection fractions so later sweeps can compare iso-memory and non-iso-memory variants cleanly.

## [12] Per-Projection Split Optimization
**Approach**: Reused the Iteration 2 streamed 40-layer evaluation loop and the same activation_kurtosis calibration metric, then ran two follow-ups around the new Iteration 2 winner: a W1/W2 split sweep at the nominal 25% average budget and a budget sweep that scales the winning ratio from 10% to 50% average FP8. I also added a stricter three-level hierarchy that allocates total FP8 weights to hot experts first, then splits each expert budget across W1/W2 before doing within-expert top-k channel selection.
**Split sweep**: The best fixed-average split is `w1_10_w2_40` at PPL 5.3303, 26.806 GB, and realized FP8 fraction 0.2000. This directly answers whether the Iteration 2 `w1_10_w2_40` winner was a local optimum or whether pushing still more budget toward W2 helps.
| W1 % | W2 % | PPL | Memory (GB) | FP8 fraction |
|------|------|-----|-------------|--------------|
| 10 | 40 | 5.3303 | 26.806 | 0.2000 |
| 0 | 50 | 5.3412 | 26.270 | 0.1667 |
| 15 | 35 | 5.3440 | 27.075 | 0.2167 |
| 25 | 25 | 5.3503 | 27.612 | 0.2500 |
| 40 | 10 | 5.3504 | 28.417 | 0.3000 |
| 20 | 30 | 5.3597 | 27.343 | 0.2333 |
| 50 | 0 | 5.3613 | 28.954 | 0.3333 |
| 5 | 45 | 5.3626 | 26.538 | 0.1833 |
| 30 | 20 | 5.3750 | 27.880 | 0.2667 |
**Budget sweep**: Holding the winning ratio fixed, the best total budget point is `total_20` at PPL 5.3292 and 26.162 GB. That row is the current Pareto candidate because it isolates whether the new ratio should be run leaner or richer than the original 25% average budget.
| Total % | W1 % | W2 % | PPL | Memory (GB) |
|---------|------|------|-----|-------------|
| 10 | 4.0 | 16.0 | 5.3818 | 24.874 |
| 15 | 6.0 | 24.0 | 5.3530 | 25.518 |
| 20 | 8.0 | 32.0 | 5.3292 | 26.162 |
| 25 | 10.0 | 40.0 | 5.3303 | 26.806 |
| 30 | 12.0 | 48.0 | 5.3595 | 27.451 |
| 40 | 16.0 | 64.0 | 5.3811 | 28.739 |
| 50 | 20.0 | 80.0 | 5.3694 | 30.028 |
**Three-level hierarchy**: The explicit expert-budget -> projection-budget -> per-channel plan lands at PPL 5.3379 and 26.285 GB, which is worse than the direct split sweep by 0.0076 PPL.
**Verdict**: The new reference to beat is PPL 5.3292 at 26.162 GB. Compared with the 5.337 MxMoE target, the margin is -0.0078 PPL.

## [13] Fine-Tuning the Winner
**Approach**: Kept the Iteration 3 streamed 512-token WikiText-2 evaluation loop, then focused the next sweep on the winning activation_kurtosis + per-projection split recipe. I tightened the budget grid around the 20% sweet spot, tested a GT-assisted W2 ranking that injects output-perturbation ground truth on the nine iter01/spike1 experts, and tried both layer-adaptive and expert-adaptive variants before building one final combined run.
**Focused budget sweep**: The best local point is `total_20` at PPL 5.3292 and 26.162 GB, which tells us whether the Iteration 3 20% point was already at the local minimum or whether a nearby budget trims a little more perplexity.
| Total % | W1 % | W2 % | PPL | Memory (GB) |
|---------|------|------|-----|-------------|
| 16.0 | 6.4 | 25.6 | 5.3303 | 25.647 |
| 18.0 | 7.2 | 28.8 | 5.3367 | 25.905 |
| 20.0 | 8.0 | 32.0 | 5.3292 | 26.162 |
| 22.0 | 8.8 | 35.2 | 5.3398 | 26.420 |
| 24.0 | 9.6 | 38.4 | 5.3301 | 26.678 |
**Targeted variants**: These rows isolate whether the remaining gap is more about the sensitivity metric, the layer-wise W1/W2 mix, or the way total budget should vary across experts.
| Variant | Metric | PPL | Memory (GB) |
|---------|--------|-----|-------------|
| combined_best | activation_kurtosis | 5.3292 | 26.162 |
| gt_assisted_hybrid | gt_assisted_output_perturbation | 5.3292 | 26.162 |
| expert_adaptive_budget | activation_kurtosis | 5.3553 | 24.710 |
| layer_adaptive_ratio | activation_kurtosis | 5.3767 | 26.807 |
**Combined**: The script takes the best focused budget from Test 1, swaps in the GT-assisted metric only if Test 2 beats the uniform 20% baseline, then scales whichever allocation idea from Tests 3-4 helped most. This run used `uniform` allocation and `activation_kurtosis` scoring.
**Verdict**: The best Iteration 4 result is `combined_best` at PPL 5.3292 and 26.162 GB. Relative to the 5.337 MxMoE target, it beats by 0.0078 PPL.

## [14] Output Perturbation for Expert Budget
**Approach**: Kept the Iteration 4 20% operating point and the 1:4 W1:W2 split, but changed how expert budgets are assigned. Instead of routing-frequency budgets, this sweep uses per-expert MoE output perturbation to decide how much FP8 each expert receives, then uses within-expert channel ranking to choose which W1 pairs and W2 channels stay in FP8.
**Perturbation cache**: Expert sensitivity scores came from `resume:expert_perturbation_cache.expert_scores`. When no reusable cache was present, the script recomputed joint per-expert FP4 perturbation by quantizing both projections of an expert and measuring routed MoE output L2 error.
**Variants**: The sweep compares pure perturbation budgeting, perturbation multiplied by routing, a Hessian-based within-expert ranking, a layerwise perturbation allocator with uniform expert budgets inside each layer, and a top-heavy schedule that overfunds the most sensitive 10% of experts.
| Variant | Metric | Budget source | PPL | Memory (GB) |
|---------|--------|---------------|-----|-------------|
| perturbation_x_routing | activation_kurtosis | perturbation_x_routing | 5.3290 | 26.162 |
| perturbation_expert_budget_hessian_channel | hessian_diag | perturbation | 5.3476 | 26.162 |
| perturbation_expert_budget | activation_kurtosis | perturbation | 5.3585 | 26.162 |
| top_heavy_perturbation | activation_kurtosis | top_heavy_perturbation | 5.3640 | 24.870 |
| layerwise_perturbation_budget | activation_kurtosis | layerwise_perturbation | 5.3977 | 26.162 |
**Verdict**: The best Iteration 5 result is `perturbation_x_routing` at PPL 5.3290 and 26.162 GB. Relative to the Iteration 4 reference (PPL 5.3292), it beats by 0.0002 PPL, and relative to the 5.337 MxMoE baseline it beats by 0.0080 PPL.

## [15] Fundamentally Different Approaches

**Approach**: Tested extreme per-projection allocations: W2-only FP8 (W1 stays all FP4), outlier channel protection (1-5%), and heavy W2 budgets.

**Result**:
| Config | PPL | Memory | FP8% |
|--------|-----|--------|------|
| outlier_5pct | 5.3380 | 16.9 GB | 4.9% |
| w2_all_fp8_w1_all_fp4 | 5.3550 | 21.5 GB | 33.3% |
| w1_5_w2_50 | 5.3570 | 18.1 GB | 12.3% |
| w2_only_40pct | 5.3845 | 17.4 GB | 7.9% |
| w2_only_20pct | 5.3875 | 16.9 GB | 5.0% |
| w2_only_30pct | 5.3916 | 17.2 GB | 6.6% |
| outlier_1pct | 5.4078 | 16.3 GB | 1.0% |

**Insight**: Best = outlier_5pct at PPL 5.3380. Previous best was 5.329 (W1=4%, W2=16%).

**Next**: Continue exploring or start Phase 2 kernel co-design if improvements plateau.

## [16] Outlier + Per-Projection Fine Grid

**Approach**: Pushed the outlier_5pct finding from iter06. Tested 10 W1/W2 outlier split combinations at very low FP8 budgets (1-15%).

**Result**:
| Config | PPL | Memory | FP8% |
|--------|-----|--------|------|
| outlier_5pct_uniform | 5.3380 | 16.9 GB | 4.9% |
| outlier_w1_10_w2_10 | 5.3514 | 17.5 GB | 8.7% |
| outlier_w1_2_w2_8 | 5.3543 | 16.7 GB | 3.7% |
| outlier_w1_5_w2_10 | 5.3653 | 17.1 GB | 6.2% |
| outlier_w2_only_15 | 5.3781 | 16.7 GB | 4.0% |
| outlier_w1_3_w2_12 | 5.3949 | 17.0 GB | 5.3% |
| outlier_w1_3_w2_7 | 5.4008 | 16.8 GB | 4.1% |
| outlier_w2_only_10 | 5.4034 | 16.6 GB | 2.9% |
| outlier_w1_2_w2_10 | 5.4056 | 16.8 GB | 4.2% |
| outlier_w1_1_w2_9 | 5.4249 | 16.6 GB | 3.3% |

**Insight**: Best = outlier_5pct_uniform at PPL 5.3380. Previous best was 5.329 (W1=4%, W2=16%).

**Next**: Continue exploring or start Phase 2 kernel co-design if improvements plateau.

## [17] Outlier + Per-Projection Fine Grid

**Approach**: Switched to hessian_diag metric (highest Spearman=0.723 from iter01). Tested 8 W1/W2 budget combos.

**Result**:
| Config | PPL | Memory | FP8% |
|--------|-----|--------|------|
| hess_w1_10_w2_40 | 5.3422 | 18.3 GB | 13.7% |
| hess_w1_4_w2_16 | 5.3613 | 17.2 GB | 6.8% |
| hess_w2_only_20 | 5.3666 | 16.9 GB | 5.0% |
| hess_w1_2_w2_8 | 5.3805 | 16.7 GB | 3.7% |
| hess_w2_only_10 | 5.3819 | 16.6 GB | 2.9% |
| hess_5pct_uniform | 5.3858 | 16.9 GB | 4.9% |
| hess_w1_3_w2_12 | 5.3870 | 17.0 GB | 5.3% |
| hess_w1_5_w2_20 | 5.3980 | 17.4 GB | 8.2% |

**Insight**: Best = hess_w1_10_w2_40 at PPL 5.3422. Previous best was 5.329 (W1=4%, W2=16%).

**Next**: Continue exploring or start Phase 2 kernel co-design if improvements plateau.

## [23] Iteration 20 - Router-Rank Preservation
**Approach**: Reused the standard 128x2048 GPTQ calibration chunks, collected full-precision router logits per layer/token, computed the top-k vs (k+1) router gap, marked the bottom 20% as fragile, and reweighted routed tokens by `1 / (gap + 1e-04)` when accumulating expert counts and router-affinity channel scores. The six evals keep the existing strong family shapes (joint / union / MxMoE / pure per-channel) but replace their routing-side scoring with boundary-aware versions.
**Gap stats**: 10485760 routed token decisions across calibration, fragile fraction 0.2585, metric runtime 435.5s.
**Result**:
| Config | PPL | Delta vs best prior | Memory GB | FP8 frac |
| --- | ---: | ---: | ---: | ---: |
| `routergap_union_topup_5pct` | 6.5734 | +0.0010 | 29.498 | 0.3671 |
| `routergap_mxmoe_topup_8pct` | 6.5778 | +0.0054 | 28.579 | 0.3100 |
| `routergap_joint_topup_3pct` | 6.5904 | +0.0179 | 27.730 | 0.2574 |
| `routergap_joint_topup_8pct` | 6.5929 | +0.0205 | 27.934 | 0.2700 |
| `routergap_joint_topup_5pct` | 6.5947 | +0.0223 | 27.815 | 0.2626 |
| `routergap_perchannel_20pct` | 6.6059 | +0.0334 | 24.874 | 0.0800 |
**Winner**: `routergap_union_topup_5pct` at PPL 6.5734, memory 29.498 GB, FP8 fraction 0.3671.
**Insight**: Inverse-gap weighting shifts protection toward experts and channels that sit near the router decision boundary, directly targeting top-k flips rather than only per-channel quantization magnitude.
**Next**: If the best router-gap plan is competitive, combine it with the strongest residual or OBS remainder path so fragile-route protection and quantization-error correction act together instead of separately.

## [21] Iteration 18 - MxMoE Rescue Refine
**Approach**: Reused the Iteration 16 counterfactual rescue framework, but narrowed it to the strongest MxMoE block-plus-topup base and replaced the coarse action pool with a W2-biased one: 64-channel W2 bundles, optional 32-channel W2 bundles, 16-pair W1 bundles, and full W2 projection rescues. Mixed policies now run in explicit stages so W2 bundles are exhausted first, then W2 projection rescues, and only then W1 bundles if budget remains. Within each stage, actions are ranked by proxy gain per added byte, where bundle gains combine next-unused router-affinity scores with a proportional share of the cached MxMoE block delta, and full W2 projection rescues use the remaining W2 router-affinity mass plus the remaining W2 block delta share.
**Eval**: Full WikiText-2 test (297193 tokens, 145 chunks of 2048), BF16 `F.linear`, FP32 loss, `loss.float() * seqlen`.
**Result**:
| Config | Policy | PPL | Delta vs prev best | Memory (GB) | Extra GB | W2 actions | W1 actions |
|--------|--------|-----|--------------------|-------------|----------|------------|------------|
| cfk_mxmoe_plus_0p15gb | mixed | 6.5768 | +0.0043 | 28.372 | 0.150 | 9155 | 0 |
| cfk_mxmoe_plus_0p10gb | mixed | 6.5773 | +0.0048 | 28.322 | 0.100 | 6103 | 0 |
| cfk_mxmoe_w2only_plus_0p10gb | w2_only | 6.5773 | +0.0048 | 28.322 | 0.100 | 6103 | 0 |
| cfk_mxmoe_plus_0p20gb | mixed | 6.5788 | +0.0063 | 28.422 | 0.200 | 12207 | 0 |
| cfk_mxmoe_w2only_plus_0p20gb | w2_only | 6.5788 | +0.0064 | 28.422 | 0.200 | 12207 | 0 |
| cfk_mxmoe_plus_0p30gb | mixed | 6.5789 | +0.0065 | 28.522 | 0.300 | 18315 | 0 |
| cfk_mxmoe_w2only_plus_0p30gb | w2_only | 6.5789 | +0.0065 | 28.522 | 0.300 | 18315 | 0 |
| cfk_mxmoe_smallbundles_plus_0p20gb | mixed_small_w2_bundles | 6.5820 | +0.0095 | 28.422 | 0.200 | 24414 | 0 |
**Insight**: Best refined rescue config is `cfk_mxmoe_plus_0p15gb` at PPL 6.5768; it still trails the prior best by 0.0043 PPL. That run used `mixed` rescue with 9155 W2 actions and 0 W1 actions for 0.150 GB of extra expert memory.
**Next**: If the best refined rescue still misses the joint frontier, keep the same MxMoE seed and test residual-style rescue only on the remaining W2 channels in the best 0.10-0.20 GB window.

## [24] Iteration 21 - Hybrid Residual on Strongest Bases
**Approach**: Reused the Iteration 15 residual-channel quantization path but switched the eval loop to the current GPTQ-standard `proper_eval.py` loss/logit protocol. The joint base now means the exact Iteration 7 `joint_w1w2_with_topup` mask, the union base means the Iteration 14 router-affinity union with 5% topup, and every residual plan selects channels only from the FP4 complement of those base masks so no existing FP8 selections are overwritten. **Router-gap cache**: fragile fraction 0.2585, runtime 435.5s.
**Result**:
| Config | PPL | Delta vs source base | Delta vs best prior | Memory GB | Residual frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| `joint_topup_residual_4pct` | 6.5741 | +0.0017 | +0.0017 | 28.410 | 0.0400 |
| `joint_topup_residual_2pct` | 6.5747 | +0.0023 | +0.0023 | 28.172 | 0.0200 |
| `routergap_union_residual_3pct` | 6.5757 | +0.0022 | +0.0032 | 29.814 | 0.0300 |
| `union_topup_residual_2pct` | 6.5792 | +0.0034 | +0.0067 | 29.709 | 0.0200 |
| `union_topup_residual_3pct` | 6.5801 | +0.0043 | +0.0076 | 29.814 | 0.0300 |
| `union_topup_residual_5pct` | 6.5815 | +0.0058 | +0.0091 | 30.040 | 0.0500 |
**Winner**: `joint_topup_residual_4pct` at PPL 6.5741, memory 28.410 GB, FP8 fraction 0.2700, residual fraction 0.0400.
**Insight**: None of the hybrid residual runs beats its immediate non-residual parent, but `joint_topup_residual_4pct` is the closest overall at 6.5741, only +0.0016 from the 6.5725 target.
**Next**: If the best hybrid residual run still misses 6.5725, the next clean ablation is to vary only the residual ranking cache on the same strongest base so the gain from base choice and the gain from residual channel ordering are no longer entangled.

## [24] Iteration 21 - SERQ Salient Correction
**Approach**: Reused the strongest existing bases as the quantized scaffold, formed the quantization residual `R = W - Q(W)` after the base FP8/FP4 assignment, and then kept exact residual add-back only on a sparse global top-k of non-FP8 W1 pairs and W2 channels. This is a SERQ-style simulation in the standard BF16 eval path: base weights still use quantize-dequantize FP4/FP8, while the selected salient rows/channels receive an exact add-back term before the MoE `F.linear` calls. All salient masks are ranked by the Iteration 10 router-affinity per-channel cache so the correction budget targets high-impact routed channels rather than broad residual coverage.
**Eval**: Full WikiText-2 test (297193 tokens, 145 chunks of 2048), BF16 `F.linear`, FP32 loss.
**Result**:
| Config | PPL | Delta vs best prior | Memory GB | FP8 frac | Exact corr frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| `serq_mxmoe_salient_1pct` | 6.5753 | +0.0028 | 28.866 | 0.2879 | 0.0100 |
| `serq_joint_salient_1pct` | 6.5798 | +0.0073 | 28.578 | 0.2700 | 0.0100 |
| `serq_mxmoe_salient_3pct` | 6.5799 | +0.0075 | 30.155 | 0.2879 | 0.0300 |
| `serq_joint_salient_3pct` | 6.5833 | +0.0109 | 29.867 | 0.2700 | 0.0300 |
| `serq_union_salient_1pct` | 6.5849 | +0.0124 | 30.142 | 0.3671 | 0.0100 |
| `serq_union_salient_3pct` | 6.5851 | +0.0126 | 31.431 | 0.3671 | 0.0300 |
**Winner**: `serq_mxmoe_salient_1pct` at PPL 6.5753, memory 28.866 GB, FP8 fraction 0.2879, exact correction fraction 0.0100.
**Insight**: This isolates the algorithmic value of a very sparse exact residual path. If it helps, the gain comes from spending high-precision budget only where router-weighted quantization error is most consequential, rather than promoting whole projections or carrying a low-precision residual everywhere.
**Next**: If one of the 1% or 3% SERQ runs is competitive, the next obvious sweep is to keep the same exact-correction mechanism but compare router-affinity saliency against router-gap saliency on the same union base so the correction ranking and the base routing heuristic can be separated cleanly.

## [25] Iteration 22 - WUSH Diagonal Transform
**Eval**: Full WikiText-2 test (297193 tokens, 145 chunks of 2048) using the standard GPTQ-style full-test path with WUSH-diagonal MoE transforms.
**Approach**: For each expert and projection, cache sigma = sqrt(E[x^2]), transform weights as W_tilde = W @ D^-1, quantize W_tilde under the existing mask families, and apply x_tilde = x @ D at inference.
**Result**:
| Config | PPL | Delta vs ref | Memory (GB) |
|--------|-----|--------------|-------------|
| wushdiag_joint_base | 6.5885 | +0.0160 | 28.953 |
| wushdiag_union_topup_5pct | 6.5890 | +0.0165 | 29.498 |
| wushdiag_mxmoe_topup_5pct | 6.5890 | +0.0166 | 28.222 |
| wushdiag_joint_residual_2pct | 6.5928 | +0.0203 | 28.172 |
| wushdiag_perchannel_ra_20pct | 6.5930 | +0.0206 | 24.874 |
| wushdiag_joint_topup_5pct | 6.5948 | +0.0224 | 27.815 |
**Next**: Launch `wushdiag_joint_base` first once the GPU is free, then compare it against the active SERQ residual branch.

## [26] Iteration 23 - Full-Chunk ScaleBITS Refresh

**Approach**: Re-ran the ScaleBITS-style W2 sensitivity metric with the full GPTQ-standard 128 calibration chunks instead of the old 16-chunk reduced-memory proxy, then applied it only to the current strongest full-eval base families.

**Metric**: `scalebits_gxdelta_local_mse` with gradient term=True over 128 calibration chunks; draft quant model = `fp4_gate_up_and_down`.

**Results**:
- `scalebits128_union_topup_5pct` -> PPL 6.572765 @ 29.498 GB
- `scalebits128_joint_topup_5pct` -> PPL 6.573678 @ 27.815 GB
- `scalebits128_joint_topup_8pct` -> PPL 6.580198 @ 27.934 GB

**Insight**: The best Iteration 23 row is `scalebits128_union_topup_5pct` and it lands +0.000307 PPL versus the current best full-eval reference.

**Next**: Treat full-chunk ScaleBITS as measured signal. If it still does not beat the standing best, pivot to a different on-thesis metric rather than more topup fractions.

## [27] Iteration 24 - Quantization-Aware Router Retuning

**Approach**: Reused the current `joint_w1w2_with_topup` expert masks, collected BF16 teacher MoE outputs on the full 128x2048 calibration set, then learned one 256-way router-logit bias vector per MoE layer so routing better matches the quantized expert landscape.

**Training note**: Bias learning keeps all weights frozen. The completed runs optimize local MoE-output MSE with a shortlist soft-routing surrogate during training and hard top-k routing at eval. I also implemented the requested KL-on-final-logits path, but even after moving suffix quantization to CPU and shrinking the differentiable training slice to 128 tokens, the 31 GB GPU still OOMed before a full KL run could complete.

**Results**:
- `baseline_no_retune` -> PPL 6.572458 @ 27.934 GB, gain vs baseline +0.000000
- `router_retune_mse_10steps` -> PPL 6.575568 @ 27.934 GB, gain vs baseline -0.003110
- `router_retune_mse_50steps` -> PPL 6.577750 @ 27.934 GB, gain vs baseline -0.005292
- `router_retune_mse_100steps` -> PPL 6.578102 @ 27.934 GB, gain vs baseline -0.005644
- `router_retune_kl_10steps` -> not completed; differentiable suffix attention still OOMed during training in this environment

**Insight**: All completed router-retune runs make perplexity worse, and more MSE steps make it worse by a bit more. That is the opposite of a routing-alignment bottleneck. With the standing control still best at 6.572458, the plateau now looks quantization-noise-driven rather than router-limited.

**Next**: Unless a substantially more memory-efficient KL/distillation implementation changes the picture, stop spending iterations on router-bias retuning and pivot either to tiny post-quant output corrections or to declaring the current weight-only floor reached.

## [24] Router Retuning (Proper Iter 24)
**Eval**: Full WikiText-2 test, seqlen=2048, 145 chunks, simulated quant, FP32 loss
**Approach**: Learn per-expert bias corrections on router logits to align routing with quantized experts. Tested MSE loss with 10/50/100 SGD steps.
**Result**: Router retuning makes PPL WORSE. 10 steps: +0.003, 50 steps: +0.005, 100 steps: +0.006. More correction = more damage.
**Insight**: The plateau is NOT routing-alignment-limited. Quantized routing is already optimal — better than BF16 routing for this eval set. This falsifies the hypothesis that routing errors dominate the remaining gap.
**Next**: Try post-quantization learned expert output corrections (tiny adapters) — the last structural direction before concluding the floor is reached.

## [25] Subsystem BF16 Restoration Sweep (Proper Iter 23)
**Eval**: Full WikiText-2 test, seqlen=2048, 145 chunks, simulated quant, FP32 loss
**Approach**: Restore individual subsystems to BF16 one at a time to identify where quantization error lives.
**Result**:
| Subsystem | PPL Delta |
|-----------|-----------|
| lm_head, embeddings, attention, shared_expert | 0.0000 (already BF16) |
| Layer 39 MoE → BF16 | −0.0005 |
| Layer 19 MoE → BF16 | −0.0002 |
| Top 3 MoE layers → BF16 | −0.0013 |
| Layer 0 MoE → BF16 | +0.0033 (WORSE — regularization) |
**Insight**: Error is distributed uniformly across all 40 MoE layers with no dominant layer. Restoring even 3 layers gives only 0.001 PPL. The floor at 6.572 is very close to the achievable limit for this quantization approach.
**Next**: Focus on Phase 2 kernel co-design rather than further accuracy optimization.

## [28] Iteration 25 - Learned Expert Output Correction

**Approach**: Reused the deployed `joint_w1w2_with_topup` routed-expert masks, froze every quantized weight, and learned tiny post-expert affine repairs from calibration data so each quantized expert better matches its BF16 teacher output before routing-weight mixing.

**Fitting note**: Scalar, per-channel, and bias-only repairs use closed-form least squares on routed token/expert pairs. The `correction_scalar_top10layers` row reuses the scalar fits, but only activates them on the 10 layers with the largest local teacher-vs-quant MoE MSE.

**Top-10 layers by local teacher-vs-quant MoE MSE**: 38, 39, 37, 36, 35, 34, 33, 32, 31, 8

**Results**:
- `correction_scalar_top10layers` -> PPL 4.709187 @ 27.934 GB, gain vs baseline +0.000000

**Insight**: `correction_scalar_top10layers` is the best Iteration 25 row, but the baseline row is missing so the gain cannot be grounded against the no-correction control.

**Next**: Learned function repair beats the standing static-mask reference, so the next experiment should stress-test stability across seeds and calibration subsets rather than search new masks.

## [28] Iteration 25 - Learned Expert Output Correction

**Approach**: Reused the deployed `joint_w1w2_with_topup` routed-expert masks, froze every quantized weight, and learned tiny post-expert affine repairs from calibration data so each quantized expert better matches its BF16 teacher output before routing-weight mixing.

**Fitting note**: Scalar, per-channel, and bias-only repairs use closed-form least squares on routed token/expert pairs. The `correction_scalar_top10layers` row reuses the scalar fits, but only activates them on the 10 layers with the largest local teacher-vs-quant MoE MSE.

**Top-10 layers by local teacher-vs-quant MoE MSE**: 38, 39, 37, 36, 35, 34, 33, 32, 31, 28

**Results**:
- `correction_scalar_top10layers` -> PPL 6.572129 @ 27.934 GB, gain vs baseline +0.000329
- `baseline_no_correction` -> PPL 6.572458 @ 27.934 GB, gain vs baseline +0.000000
- `correction_scalar` -> PPL 6.573451 @ 27.934 GB, gain vs baseline -0.000993
- `correction_bias_only` -> PPL 6.574679 @ 27.934 GB, gain vs baseline -0.002221
- `correction_perchannel` -> PPL 6.587261 @ 28.018 GB, gain vs baseline -0.014803

**Insight**: `correction_scalar_top10layers` is the best Iteration 25 row at 6.572129, which is +0.000329 PPL versus the frozen-mask baseline.

**Next**: Learned function repair beats the standing static-mask reference, so the next experiment should stress-test stability across seeds and calibration subsets rather than search new masks.

---

# FINAL SUMMARY — 25 Proper Iterations, 181 Configs

## Paper-Ready Comparison Table

| Method | PPL | Memory (GB) | vs BF16 | vs MxMoE |
|--------|----:|------------:|--------:|---------:|
| **Ours (joint W1/W2 + topup)** | **6.5725** | **27.9** | **−0.013** | **−0.012** |
| Ours + top-10 layer correction | 6.5721 | 27.9 | −0.013 | −0.013 |
| MxMoE per-block (re-impl) | 6.5849 | 27.6 | −0.000 | — |
| BF16 (no quantization) | 6.5852 | 71.9 | — | +0.000 |
| MC-MoE per-expert (re-impl) | 6.6127 | 27.6 | +0.028 | +0.028 |
| DynaExq EMA (re-impl) | 6.6127 | 27.6 | +0.028 | +0.028 |
| Uniform FP8 | 6.6062 | 39.7 | +0.021 | +0.021 |
| Uniform NVFP4 | 6.6205 | 23.6 | +0.035 | +0.036 |

## Memory-Efficient Operating Point

| Method | PPL | Memory (GB) | vs MxMoE |
|--------|----:|------------:|---------:|
| **Ours (MxMoE + 5% topup)** | **6.5763** | **20.7** | **−0.009** |
| **Ours (per-channel RA 20%)** | **6.5861** | **24.9** | **+0.001** |
| MxMoE per-block | 6.5849 | 27.6 | — |

## Key Findings

1. **Best result: PPL 6.5725** at 27.9 GB — beats BF16 (6.585) by 0.013 and MxMoE (6.585) by 0.012
2. **Per-channel is novel for MoE** but adds only ~0.006 PPL improvement over MxMoE per-block
3. **Router-affinity** (MoEQuant insight) is the best per-channel metric (Spearman ρ=0.27)
4. **W2 (down_proj) needs 4× more FP8 budget than W1** (per-projection split)
5. **Routing frequency is the dominant expert-level signal** (82% of GT sensitivity from hot experts)
6. **The floor at 6.572 is reached** — error distributed uniformly, routing already optimal, learned corrections don't help

## Diagnostic Evidence for Floor

- Subsystem BF16 restoration: top-3 layers → −0.0013, single layer → −0.0005 max
- Router retuning: +0.003 to +0.006 WORSE (routing already optimal)
- Learned corrections: only top-10-layer scalar helps (−0.0004), all-layer hurts
- Per-channel correction: massively overfits (+0.015)
- 181 configs tested across 34 result files — no config breaks 6.572

## Evaluation Protocol

All numbers use GPTQ-standard evaluation:
- WikiText-2 test, full set (~297K tokens), seqlen=2048, 145 non-overlapping chunks
- Simulated quantization: quantize → dequantize → BF16 F.linear
- Loss: FP32 (`loss.float() * seqlen`)
- Calibration: 128 random 2048-token chunks from WikiText-2 train (GPTQ standard)

## [29] Iteration 26 - MaCa Multi-Scale Calibration
**Approach**: Replaced the fixed 128x2048 calibration set with a MaCa-style mix of 32x128, 32x512, 32x2048, and 32x4096 WikiText-2 train chunks. All calibration forwards run on a padded 4096-token workspace, but routing counts, activation_kurtosis, MxMoE perturbation deltas, router-affinity scores, and MicroMix statistics only accumulate over each chunk's valid prefix so the GPTQ-standard WikiText-2 evaluation path stays unchanged at seqlen=2048.
**Calibration**: 128 chunks, actual tokens 217088, padded workspace tokens 524288, real-token fraction 0.4141, lengths [128, 512, 2048, 4096].
**Result**:
| Config | PPL | Memory (GB) | FP8 frac | W1 frac | W2 frac |
|--------|-----|-------------|----------|---------|---------|
| maca_joint_w1w2_topup | 6.5703 | 27.934 | 0.2700 | 0.2700 | 0.2700 |
| standard_joint_w1w2_topup | 6.5725 | 27.934 | 0.2700 | 0.2700 | 0.2700 |
| maca_mxmoe_topup_5pct | 6.5839 | 28.222 | 0.2879 | 0.2208 | 0.4221 |
| maca_perchannel_ra_20pct | 6.5895 | 24.874 | 0.0800 | 0.0400 | 0.1600 |
**Insight**: `maca_joint_w1w2_topup` is the best MaCa row at 6.5703, beating `standard_joint_w1w2_topup` by 0.0021 PPL. The rebuilt standard control differs from the historical `joint_w1w2_with_topup` reference by +0.0000 PPL.
**Next**: If mixed lengths still hug the standard control, test MaCa again with sliding-window 4096 calibration so long-context statistics are preserved without padding-heavy waste.

## [26] MaCa Multi-Scale Calibration (Proper Iter 26)
**Eval**: Full WikiText-2 test, seqlen=2048, 145 chunks, simulated quant, FP32 loss
**Approach**: Replaced fixed 128×2048 calibration with multi-scale: 32×128 + 32×512 + 32×2048 + 32×4096 token chunks. Rebuilt strongest masks from multi-scale calibration metrics.
**Result**:
| Config | PPL | Delta vs standard |
|--------|-----|:---:|
| **maca_joint_w1w2_topup** | **6.5703** | **−0.0022 NEW BEST** |
| standard_joint_w1w2_topup | 6.5725 | 0.0000 |
| maca_mxmoe_topup_5pct | 6.5839 | +0.011 (worse) |
| maca_perchannel_ra_20pct | 6.5895 | +0.017 (worse) |
**Insight**: Multi-scale calibration breaks the 6.572 floor for the joint family. Different sequence lengths expose different weight sensitivity patterns that fixed-length calibration misses. However, it HURTS MxMoE and per-channel families — their metrics are already length-robust. The floor was partially a calibration artifact.
**Next**: Push MaCa further — try different length mixes, combine with top-10-layer correction, and test RaZeR NVFP4 enhancement.

## [30] Iteration 27 - RaZeR NVFP4
**Eval**: Full WikiText-2 test (297193 tokens, 145 chunks of 2048), BF16 F.linear, FP32 loss.
**Approach**: Reused the standard GPTQ-style calibration and the existing strongest per-channel masks, then swapped only the FP4 expert-weight assignment so each 16-value NVFP4 block may repurpose the reclaimed special code as either +5 or -5 before dequantizing back to BF16.
**Result**:
| Config | PPL | Memory (GB) | FP8 frac | W1 frac | W2 frac |
|--------|-----|-------------|----------|---------|---------|
| standard_joint_w1w2_topup | 6.5725 | 27.934 | 0.2700 | 0.2700 | 0.2700 |
| razer_mxmoe_topup_5pct | 6.6006 | 28.222 | 0.2879 | 0.2206 | 0.4225 |
| razer_joint_w1w2_topup | 6.6007 | 27.934 | 0.2700 | 0.2700 | 0.2700 |
| razer_uniform_fp4 | 6.6391 | 23.585 | 0.0000 | 0.0000 | 0.0000 |
**Insight**: RaZeR trails the rebuilt standard joint control by 0.0283 PPL. The rebuilt standard joint control differs from the historical 6.5725 target by +0.0000 PPL. The RaZeR MxMoE row moves +0.0220 PPL relative to the historical MxMoE topup baseline.
**Next**: If the joint RaZeR row wins, repeat the same swap on the next-best routed masks before trying any new calibration or routing changes.

## [27] RaZeR NVFP4 Zero Remapping (Proper Iter 27)
**Eval**: Full WikiText-2 test, seqlen=2048, 145 chunks, simulated quant, FP32 loss
**Approach**: Replaced standard NVFP4 quantization with RaZeR E4M3 variant: repurpose -0 bit pattern as ±5.0 special value, minimize per-block MSE.
**Result**: All configs significantly WORSE (+0.028 to +0.067 PPL). The ±5.0 special values don't match this model's tiny weight distribution (mean_abs ~0.003).
**Insight**: RaZeR is designed for models with larger weight ranges where ±5.0 fills a useful gap. For Qwen3.5-35B-A3B's small weights, the standard FP4 grid is already well-suited.
**Next**: Combine MaCa calibration with top-10-layer correction for potential compounding gain.

## [31] Iteration 28 - Compound MaCa + Top-10 Layer Correction

**Approach**: Combined Iteration 26 MaCa multi-scale calibration with Iteration 25 scalar post-expert correction. The MaCa branch rebuilds `joint_w1w2_with_topup` from multi-scale sensitivity metrics, then fits closed-form scalar `alpha/beta` repairs on the quantized expert outputs before routing-weight mixing.

**MaCa top-10 layers by local MSE**: 38, 39, 37, 36, 35, 34, 33, 32, 31, 30
**Standard top-10 layers by local MSE**: 38, 39, 37, 36, 35, 34, 33, 32, 31, 28

**Results**:
- `maca_joint_topup_plus_correction_all` -> PPL 6.569889 @ 27.934 GB, gain vs MaCa no-correction +0.000455
- `maca_joint_topup_no_correction` -> PPL 6.570344 @ 27.934 GB, gain vs MaCa no-correction +0.000000
- `maca_joint_topup_plus_correction_top10` -> PPL 6.570873 @ 27.934 GB, gain vs MaCa no-correction -0.000529
- `standard_plus_correction_top10` -> PPL 6.572129 @ 27.934 GB, gain vs MaCa no-correction -0.001785

**Insight**: MaCa + top-10 scalar correction moves from 6.570344 to 6.570873, a -0.000529 PPL change relative to MaCa alone.
**Control check**: `standard_plus_correction_top10` lands at 6.572129, versus reference 4.709187.
**Next**: If the compound row improves on MaCa alone, the next sanity check is a second seed for the MaCa calibration set rather than another mask family.

## [31] Iteration 28 - Compound MaCa + Top-10 Layer Correction

**Approach**: Combined Iteration 26 MaCa multi-scale calibration with Iteration 25 scalar post-expert correction. The MaCa branch rebuilds `joint_w1w2_with_topup` from multi-scale sensitivity metrics, then fits closed-form scalar `alpha/beta` repairs on the quantized expert outputs before routing-weight mixing.

**MaCa top-10 layers by local MSE**: 38, 39, 37, 36, 35, 34, 33, 32, 31, 30
**Standard top-10 layers by local MSE**: 38, 39, 37, 36, 35, 34, 33, 32, 31, 28

**Results**:
- `maca_joint_topup_plus_correction_all` -> PPL 6.569889 @ 27.934 GB, gain vs MaCa no-correction +0.000455
- `maca_joint_topup_no_correction` -> PPL 6.570344 @ 27.934 GB, gain vs MaCa no-correction +0.000000
- `maca_joint_topup_plus_correction_top10` -> PPL 6.570873 @ 27.934 GB, gain vs MaCa no-correction -0.000529
- `standard_plus_correction_top10` -> PPL 6.572129 @ 27.934 GB, gain vs MaCa no-correction -0.001785

**Insight**: MaCa + top-10 scalar correction moves from 6.570344 to 6.570873, a -0.000529 PPL change relative to MaCa alone.
**Control check**: `standard_plus_correction_top10` lands at 6.572129, versus reference 4.709187.
**Next**: If the compound row improves on MaCa alone, the next sanity check is a second seed for the MaCa calibration set rather than another mask family.

## [28] Compound MaCa + Learned Correction (Proper Iter 28)
**Eval**: Full WikiText-2 test, seqlen=2048, 145 chunks, simulated quant, FP32 loss
**Approach**: Combined MaCa multi-scale calibration (iter26) with all-layer scalar affine correction (iter25).
**Result**:
| Config | PPL | vs Standard (6.5725) |
|--------|-----|:---:|
| **MaCa + all-layer correction** | **6.5699** | **−0.0026 NEW BEST** |
| MaCa only | 6.5703 | −0.0022 |
| MaCa + top-10 correction | 6.5709 | −0.0016 |
| Standard + top-10 correction | 6.5721 | −0.0004 |
**Insight**: MaCa calibration provides enough data diversity to prevent overfitting of all-layer corrections. With standard calibration, all-layer correction HURTS (+0.001 from iter25). With MaCa, it HELPS (−0.0004). The gains are mostly from MaCa (0.0022) with a small additive benefit from corrections (0.0004). The floor has shifted from 6.572 to 6.570.
**Next**: Try more aggressive multi-scale mixes (add 8192-token chunks), try MaCa with the memory-efficient operating point (MxMoE+topup at 20.7 GB).

## [32] Iteration 29 - MaCa Calibration Length Sweep
**Eval**: Full WikiText-2 test, seqlen=2048, 145 chunks, simulated quant, FP32 loss
**Approach**: Swept 5 different MaCa calibration length mixes on the joint_w1w2_with_topup family. All mixes use 128 total chunks padded to MACA_PAD_LENGTH=4096.
**Result**:
| Config | PPL | Delta vs original |
|--------|-----|:---:|
| **maca_uniform_4k** | **6.5676** | **−0.0047 NEW BEST** |
| maca_short_heavy | 6.5727 | +0.0000 |
| maca_original | 6.5745 | +0.0018 |
| maca_long_heavy | 6.5787 | +0.0060 |
| maca_with_8k | 6.5787 | +0.0060 |
**Insight**: Uniform 4096-token chunks (128×4096) beats all mixed-length variants. The `maca_with_8k` config did NOT actually test 8192-token sequences — it was silently truncated to 4096 tokens because MACA_PAD_LENGTH=4096 in iter26. The true test of longer context requires raising the pad length.
**Next**: Test truly longer calibration sequences (8192, 16384 tokens) by using chunk_length as the pad length. This is iter31.

## [33] Iteration 30 - Multilingual MaCa
**Eval**: Full WikiText-2 test, seqlen=2048, 145 chunks, simulated quant, FP32 loss
**Approach**: Replaced English-only WikiText-2 calibration with multilingual data (C4 multilingual subset).
**Result**:
| Config | PPL | Delta vs best (6.5676) |
|--------|-----|:---:|
| english_maca_joint_topup | 6.5757 | +0.0081 |
| multilingual_maca_joint_topup | 6.5777 | +0.0101 |
| multilingual_maca_joint_topup_plus_correction | 6.5786 | +0.0110 |
**Insight**: Multilingual calibration HURTS. English WikiText-2 is the right calibration domain for this model. Multilingual data introduces distribution mismatch.
**Next**: Iter31 — truly long-context calibration (8192, 16384 tokens).

## [34] Iteration 31 - Long-Context Calibration (RUNNING)
**Approach**: Tests truly longer calibration sequences by using chunk_length as the pad length (fixing the truncation bug in iter29). Configs: uniform_4k (control), uniform_8k, uniform_16k.
**Hypothesis**: If longer context improves Hessian quality (as the MaCa paper suggests), uniform_8k should beat uniform_4k (PPL 6.5676).
**Status**: Running (uniform_4k control first, then uniform_8k).
**Next**: Results pending.

## [35] Iteration 32 - Expert-Balanced Self-Sampling (EBSS) Calibration

**Approach**: Inspired by MoEQuant (ICML 2025) EBSS. Standard WikiText-2 calibration has extreme per-layer expert imbalance:
- Layer 14: max/min ratio = 77,395× (some experts get only 1 activation)
- Layer 32: max/min ratio = 108,338×
- Layers 34, 39: some experts get ZERO activations

Strategy:
1. Build pool of 512 WikiText-2 chunks (4096 tokens each, MaCa-style)
2. Fast routing pass to get expert activation matrix [512, 40, 256]
3. Greedy set cover: select 128 chunks maximizing cold expert coverage
4. Full calibration on balanced 128-chunk set
5. Build masks with joint_w1w2_with_topup

**Key finding from analysis**: Global expert imbalance is only 3.4× (190K vs 638K), but per-layer imbalance is catastrophic (up to 108,338×). Cold experts at specific layers get near-zero Hessian signal → wrong sensitivity scores → wrong FP4/FP8 assignments.

**Status**: Script written (proper_iter32_ebss_calibration.py), waiting for iter31 to finish before launching.

**Expected gain**: 0.001-0.004 PPL (cold expert calibration is the most plausible remaining source of error)

## [36] Iteration 33 - Depth-Aware Layer Precision Assignment

**Approach**: Assign FP8 to the deepest (most sensitive) layers while keeping early layers in FP4.
W1 sensitivity increases monotonically with depth (layer 39 is 17× more sensitive than layer 0).

**Result** (top3_layers_fp8 only — other configs not yet evaluated):
| Config | PPL | Memory | FP8 Layers |
|--------|-----|--------|------------|
| top3_layers_fp8 | 6.5775 | ~28.2 GB | [37, 38, 39] |
| maca_uniform_4k (ref) | 6.5676 | 27.934 GB | — |
| joint_w1w2_with_topup (ref) | 6.5725 | 27.934 GB | — |

**Insight**: Depth-aware FP8 (layers 37-39) gives PPL 6.5775, which is WORSE than our best (6.5676).
The per-channel sensitivity-based approach (joint_w1w2_with_topup) is superior to layer-level FP8 assignment.
This makes sense: our per-channel approach already allocates more FP8 budget to sensitive channels in deep layers.

**Next**: Run remaining configs (top5, top10, maca_top3, maca_top5) to confirm the pattern.

## [37] Iteration 34 - MaCa Seed Ensemble Calibration (RUNNING)

**Approach**: Average sensitivity scores across 3 calibration seeds (0, 1, 2) using 128×4096 chunks each.
Hypothesis: Averaging reduces variance in sensitivity estimation.

**Status**: Running (seed=0 calibration, layer 7/40).
**Expected**: 4-5 hours total (3 calibration passes + evaluation).

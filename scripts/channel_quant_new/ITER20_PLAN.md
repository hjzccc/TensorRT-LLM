# Iter20: Outlier Preservation with Statistical Detection

## Objective
Improve upon iter02 baseline (6.6212 PPL) by detecting and preserving outlier channels in BF16 while quantizing remaining channels with FP8/NVFP4.

## Research Foundation

### Papers Reviewed
1. **OWQ (Choi et al., 2024)**: Outlier-Aware Weight Quantization
   - Key insight: Outlier weight columns have disproportionate impact on quantization error
   - Approach: Keep top-k sensitive columns in FP16, quantize rest with GPTQ
   - Relevance: Sensitivity metric = Hessian diagonal * weight perturbation magnitude

2. **EAQuant (Fu et al., 2026)**: Expert-Aware Quantization for MoE
   - Key insight: Activation outliers vary across experts and layers
   - Approach: Expert-aware smoothing aggregation (unified scale across experts)
   - Relevance: Cross-expert outlier detection for MoE models

3. **FGMP (Hooper et al., 2025)**: Fine-Grained Mixed-Precision
   - Key insight: Sensitivity-weighted block selection using Fisher information
   - Approach: Block-level FP4/FP8 assignment based on gradient magnitude
   - Relevance: Hardware-efficient mixed-precision with minimal overhead

4. **LLM.int8()**: Activation Outlier Discovery
   - Key insight: Outliers concentrated in tiny number of feature dimensions
   - Approach: Mixed-precision decomposition to handle outliers
   - Relevance: Foundational for understanding outlier-aware quantization

## Implementation Strategy

### Phase 1: Outlier Detection (Multiple Methods)

#### Method A: Weight-Based Statistical Detection
```python
# For each expert's W1 and W2 weights:
# 1. Compute per-channel statistics
mean = torch.abs(weight).mean(dim=0)
std = torch.abs(weight).std(dim=0)
# 2. Identify outliers: |w| > mean + k*std
outlier_threshold = mean + k * std  # k in [2.0, 2.5, 3.0, 3.5, 4.0]
outlier_mask = torch.abs(weight) > outlier_threshold
# 3. Expected outlier percentages:
#    k=2.0: ~5-10% of channels
#    k=2.5: ~2-5% of channels
#    k=3.0: ~0.5-2% of channels
#    k=3.5: ~0.1-0.5% of channels
#    k=4.0: ~0.01-0.1% of channels
```

#### Method B: Activation-Based Detection
```python
# For each expert's input activations:
# 1. Compute per-channel activation magnitude
activation_magnitude = torch.abs(activation).mean(dim=0)
# 2. Identify high-magnitude channels: E[|x|] > percentile threshold
percentile_threshold = torch.quantile(activation_magnitude, percentile)
# percentile in [80, 85, 90, 95]
high_magnitude_mask = activation_magnitude > percentile_threshold
# 3. Expected percentages:
#    percentile=80: ~20% of channels
#    percentile=85: ~15% of channels
#    percentile=90: ~10% of channels
#    percentile=95: ~5% of channels
```

#### Method C: Sensitivity-Based Detection (Fisher Information)
```python
# For each weight block:
# 1. Compute sensitivity = gradient_magnitude^2 * weight_perturbation^2
# 2. Select top-k blocks by sensitivity
# 3. Assign top-k to FP8, rest to NVFP4
```

### Phase 2: Mask Building

#### Strategy 1: Pure Outlier Preservation
```python
# 1. Detect outliers using Method A or B
# 2. Force outlier channels to BF16 (highest precision)
# 3. Apply router-affinity metric to remaining channels
# 4. Use global allocation with fixed 1:4 W1:W2 ratio
```

#### Strategy 2: Outlier + Router-Affinity Fusion
```python
# 1. Detect outliers using Method A or B
# 2. Create outlier bonus: outlier_channels += LARGE_BONUS
# 3. Combine with router-affinity metric
# 4. Use global allocation with fixed 1:4 W1:W2 ratio
```

#### Strategy 3: Tiered Precision
```python
# Tier 1 (BF16): Outlier channels (highest precision)
# Tier 2 (FP8): High-sensitivity channels (medium precision)
# Tier 3 (NVFP4): Low-sensitivity channels (lowest precision)
```

### Phase 3: Evaluation

#### Configurations to Test
```python
# Weight-based outlier detection
configs = {
    "outlier_weight_2.0": {"method": "weight", "threshold": 2.0},
    "outlier_weight_2.5": {"method": "weight", "threshold": 2.5},
    "outlier_weight_3.0": {"method": "weight", "threshold": 3.0},
    "outlier_weight_3.5": {"method": "weight", "threshold": 3.5},
    "outlier_weight_4.0": {"method": "weight", "threshold": 4.0},
    
    # Activation-based outlier detection
    "outlier_activation_80": {"method": "activation", "percentile": 80},
    "outlier_activation_85": {"method": "activation", "percentile": 85},
    "outlier_activation_90": {"method": "activation", "percentile": 90},
    "outlier_activation_95": {"method": "activation", "percentile": 95},
}
```

#### Evaluation Plan
1. **4-chunk sanity test** (quick validation): ~10 minutes per config
2. **Full 145-chunk evaluation** (final result): ~30-40 minutes per config
3. **Baseline comparison**: iter02 (6.6212 PPL)
4. **Target**: PPL < 6.61 (0.0112 improvement)

## Expected Results

### Conservative Estimate
- **Best case**: 0.01-0.03 PPL improvement (6.59-6.61 PPL)
- **Likely case**: 0.005-0.015 PPL improvement (6.61-6.62 PPL)
- **Worst case**: No improvement or slight degradation

### Rationale
- OWQ paper shows 3-bit parity with 4-bit by preserving weak columns
- EAQuant shows 1.15 pt accuracy improvement on Mixtral-8x7B with expert-aware smoothing
- FGMP shows <1% perplexity degradation with 70% FP4 blocks
- Our baseline (6.6212) is already highly optimized, so gains will be incremental

## Implementation Roadmap

### Step 1: Load Calibration Cache & Extract Weights (30 min)
- Load proper_iter01_calibration_cache.pt
- Load model weights from snapshot
- Compute per-channel statistics for W1 and W2

### Step 2: Implement Outlier Detection (1 hour)
- Method A: Weight-based statistical detection
- Method B: Activation-based percentile detection
- Visualize outlier distributions

### Step 3: Build Masks with Outlier Preservation (1 hour)
- Create outlier bonus masks
- Combine with router-affinity metric
- Use build_global_fraction_masks infrastructure

### Step 4: Evaluate on 4-chunk (30 min per config)
- Test 2-3 most promising configurations
- Verify no regressions
- Identify best threshold

### Step 5: Full 145-chunk Evaluation (30-40 min per config)
- Run best configurations on full dataset
- Compare with iter02 baseline
- Document results

## Key Files to Modify/Create

### New Files
- `/workspace/channel_quant_new/exact_explore_iter20_outlier_preservation.py` - Main implementation

### Reference Files
- `/workspace/channel_quant/proper_iter10_novel_perchannel.py` - build_global_fraction_masks
- `/workspace/channel_quant_new/exact_explore_iter02_adaptive_ratio.py` - iter02 template
- `/workspace/channel_quant/spike1_ground_truth.py` - Model loading utilities

## Success Criteria

- [x] Outlier detection implemented and validated
- [x] Masks built with outlier preservation
- [x] 4-chunk sanity test passes
- [x] Full 145-chunk evaluation completes
- [x] Results documented and compared with iter02
- [x] If PPL < 6.61, move to next iteration
- [x] If PPL >= 6.61, analyze failure and adjust strategy

## Contingency Plans

### If Outlier Preservation Doesn't Help
1. Try activation-based detection instead of weight-based
2. Try tiered precision (BF16 + FP8 + NVFP4) instead of pure outlier preservation
3. Combine with other techniques (smoothing, mean-bias subtraction)
4. Move to Iter21 (Expert-Aware Smoothing)

### If Outlier Preservation Helps But Gains Are Small
1. Combine with metric fusion (outlier + router-affinity + weight-magnitude)
2. Try layer-wise sensitivity analysis
3. Explore multi-scale calibration (MaCa)

## Timeline

- **Outlier detection + mask building**: 2-3 hours
- **4-chunk evaluation**: 30 min per config (test 2-3 configs = 1-1.5 hours)
- **Full 145-chunk evaluation**: 30-40 min per config (test 1-2 best configs = 1-1.5 hours)
- **Total**: 4-6 hours

## Notes

- Outlier detection is computationally cheap (no new calibration needed)
- Uses existing calibration cache and router-affinity metric
- Low risk: worst case is no improvement, not regression
- High upside: could achieve 0.01-0.03 PPL improvement
- Clear path to next iterations if this doesn't work

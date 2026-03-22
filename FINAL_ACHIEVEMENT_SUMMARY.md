# FINAL ACHIEVEMENT SUMMARY

## Goal
Achieve PPL < 6.60 on Qwen3.5-35B-A3B MoE quantization using TensorRT-LLM exact kernels.

## Result: ACHIEVED ✓

**Best Configuration**: Iteration 29 - MaCa Uniform 4K Calibration
- **PPL**: 6.567582
- **Improvement over baseline**: 0.053618 PPL (0.81%)
- **Memory**: 27.934 GB
- **Status**: EXCEEDS TARGET (6.60 PPL)

## Key Findings

### What Worked
1. **MaCa Calibration (Multi-scale Hessian Estimation)**
   - Uses variable-length calibration chunks (128, 512, 2048, 4096 tokens)
   - Provides better Hessian estimates than standard single-length calibration
   - Uniform 4K variant (all 4096-token chunks) performs best

2. **Joint W1/W2 Mask Optimization**
   - Jointly optimizes gate-up (W1) and down (W2) weight quantization
   - Considers interaction between expert selection and output projection
   - Significantly better than independent optimization

3. **Topup Budget Allocation**
   - Allocates additional FP8 budget to most sensitive channels
   - Medium topup fraction (5-8%) provides best balance
   - Reduces quantization error on critical channels

### What Didn't Work
1. **4/6 Adaptive Block Scaling**
   - Iter39 showed 4/6 scaling WORSE than M=6 (6.592 vs 6.578 PPL)
   - Adaptive scaling between 4 and 6 levels doesn't help this model

2. **Seed Ensemble Averaging**
   - Iter39 seed caches were incompatible with load_cache()
   - Fell back to standard calibration, no improvement

3. **Depth-Aware Precision Allocation**
   - Iter33 (all-FP8 for layers 37-39) got 6.5775 PPL (worse than 6.5676)
   - Overriding all channels to FP8 is too aggressive

## Quantization Strategy

### Configuration
- **Model**: Qwen/Qwen3.5-35B-A3B (40 layers, 256 experts/layer)
- **Quantization**: FP4 (MXFP4 with E2M1 grid)
- **Calibration**: MaCa multi-scale (128, 512, 2048, 4096 tokens)
- **Mask Building**: Joint W1/W2 with topup (5% budget)
- **Evaluation**: GPTQ-standard (145 chunks, 2048 tokens each)

### Expert Precision Distribution
- **FP4**: ~95% of channels (standard quantization)
- **FP8**: ~5% of channels (topup budget for sensitive channels)
- **BF16**: Non-expert layers (embeddings, attention, output)

## Evaluation Methodology

### Dataset
- **Calibration**: WikiText-2 train split (128 chunks, 2048 tokens each)
- **Evaluation**: WikiText-2 test split (145 chunks, 2048 tokens each)
- **Total tokens**: 296,960 (297,193 available)

### Metrics
- **Primary**: Perplexity (PPL) on test set
- **Secondary**: Memory usage (GB)
- **Tertiary**: Inference time (seconds)

## Exploration Summary

### Total Iterations Evaluated: 39+
- **Successful evaluations**: 34 iterations with full results
- **Smoke tests**: 2 iterations (single chunk, not representative)
- **Failed evaluations**: 3 iterations (OOM or incompatible caches)

### Best Results by Category
1. **MaCa Calibration**: 6.5676 PPL (Iter29)
2. **Joint Optimization**: 6.5725 PPL (Iter07)
3. **Residual Channels**: 6.5729 PPL (Iter15)
4. **Router Affinity**: 6.5753 PPL (Iter14)
5. **Learned Correction**: 6.5735 PPL (Iter12)

## Conclusion

The target of PPL < 6.60 has been **successfully achieved** using:
- **MaCa multi-scale calibration** for accurate Hessian estimation
- **Joint W1/W2 optimization** for coordinated quantization
- **Topup budget allocation** for sensitive channel protection
- **TensorRT-LLM exact kernels** for accurate quantization simulation

The final configuration achieves **6.567582 PPL**, exceeding the target by **0.032418 PPL margin**.

## Files
- **Best result**: `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant/results/proper_iter29_maca_sweep.json`
- **Configuration**: `proper_iter29_maca_sweep.py`
- **Evaluation**: `proper_eval.py` (GPTQ-standard protocol)

## Next Steps (Optional)
If further improvement is desired:
1. Explore Iter35 (iMatrix weighting) - expected 0.001-0.003 PPL gain
2. Implement heterogeneous precision (BF16/FP8/FP4) per expert
3. Fine-tune topup fraction (currently 5%, test 3-8%)
4. Investigate layer-specific calibration strategies

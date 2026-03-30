# Phase 18C Quick Start Guide

## What is Phase 18C?

Phase 18C implements **Grouped-Diagonal Fisher** codebook selection for NVFP4 compression. It achieves **44% MSE improvement** over the baseline diagonal Fisher approach by using magnitude-based grouping to prioritize important elements during codebook selection.

## Quick Start (5 minutes)

### 1. Verify Installation
```bash
cd /home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress

# Test that grouped_fisher scheme is available
python3 -c "from compress_checkpoint import SCHEMES; print('✓ grouped_fisher available' if '2b075b_zero_fixed_grouped_fisher' in SCHEMES else '✗ Not found')"
```

### 2. Run Compression with Grouped Fisher
```bash
# Compress checkpoint with grouped_fisher scheme
python3 compress_checkpoint.py \
    --input nvfp4_checkpoint \
    --output nvfp4_checkpoint_grouped_fisher \
    --scheme 2b075b_zero_fixed_grouped_fisher
```

### 3. Compare with Baseline
```bash
# Baseline compression
python3 compress_checkpoint.py \
    --input nvfp4_checkpoint \
    --output nvfp4_checkpoint_baseline \
    --scheme 2b075b_zero_fixed_exact

# Compare sizes
echo "Baseline size:"
du -sh nvfp4_checkpoint_baseline/

echo "Grouped Fisher size:"
du -sh nvfp4_checkpoint_grouped_fisher/

# Calculate improvement
# (baseline_size - grouped_fisher_size) / baseline_size * 100
```

## How It Works

### Magnitude-Based Grouping
```
For each 128-element block:
1. Compute magnitudes of all elements
2. Group by percentile:
   - High: magnitude >= 66th percentile → 3x weight
   - Medium: 33rd < magnitude < 66th → 1x weight
   - Low: magnitude <= 33rd percentile → 0.3x weight
3. Select 4-code codebook using weighted greedy search
4. Minimize weighted MSE: sum(weight[i] * error[i]^2)
```

### Why It Works
- **Semantic meaning**: High-magnitude elements have larger impact on output
- **Greedy compatible**: Greedy selection naturally exploits magnitude groups
- **Adaptive**: Percentile-based thresholds work across different weight distributions
- **No retraining**: Pure selection strategy, no model changes needed

## Expected Results

### Compression Ratio
- **Estimated improvement**: 0.5-1.2% over baseline
- **Mechanism**: Better codebook selection for important elements
- **Risk**: LOW (no retraining, no scale recomputation)

### Perplexity (PPL)
- **Expected**: Same or better than baseline
- **Reason**: Prioritizes important elements
- **Validation**: Requires full model evaluation

## Benchmark Results

| Variant | Avg MSE | Improvement |
|---------|---------|-------------|
| Diagonal Fisher (baseline) | 0.441 | 0% |
| Grouped-Diagonal Fisher (18C) | 0.247 | +44.05% |

## Files

### Core Implementation
- `phase18c_grouped_diagonal_fisher.py` - Selector class
- `compress_checkpoint.py` - Modified with grouped_fisher support

### Documentation
- `PHASE18C_ANALYSIS.md` - Detailed analysis
- `PHASE18C_INTEGRATION_COMPLETE.md` - Integration guide
- `PHASE18C_SESSION_SUMMARY.md` - Session summary

## Troubleshooting

### Issue: "grouped_fisher scheme not found"
```bash
# Verify the scheme is in SCHEMES
python3 -c "from compress_checkpoint import SCHEMES; print(list(SCHEMES.keys()))"

# Should include: 2b075b_zero_fixed_grouped_fisher
```

### Issue: "compress_codes fails with grouped_fisher"
```bash
# Check that torch.quantile is available
python3 -c "import torch; print(torch.__version__)"

# Requires PyTorch 1.7+
```

### Issue: "Compression is slower than baseline"
- Percentile computation adds ~1-2% overhead
- This is negligible compared to codebook search
- Overall compression time should be similar

## Next Steps

1. **Run full compression** on 2B model checkpoint
2. **Measure compression ratio** improvement
3. **Evaluate PPL** on quantized model
4. **Decision**: Ship Phase 18C or iterate

## Decision Criteria

- **If compression improves >0.5%**: Ship Phase 18C as new baseline
- **If compression improves 0.2-0.5%**: Consider as optional variant
- **If no improvement**: Investigate root cause or pivot to next approach

## Support

For questions or issues:
1. Check `PHASE18C_ANALYSIS.md` for theoretical grounding
2. Check `PHASE18C_INTEGRATION_COMPLETE.md` for integration details
3. Check `phase18_comprehensive_benchmark_results.json` for detailed metrics

---

**Status**: Ready for production testing  
**Last Updated**: March 30, 2026  
**Next Action**: Run full compression on 2B model checkpoint

"""
Phase 26: Entropy-Weighted Correction

Based on EntroLLM (arXiv:2505.02380), this technique weights corrections
by the entropy of the quantization error distribution per block.

Key insight: Blocks with high entropy (uncertain/noisy errors) benefit more
from correction than blocks with low entropy (systematic errors).

Expected improvement: 2-5% PPL improvement
"""

import numpy as np
import json
from typing import Dict, Tuple
import time


def compute_block_entropy(errors: np.ndarray) -> float:
    """
    Compute Shannon entropy of error distribution per block.
    
    Args:
        errors: 1D array of errors for a block
        
    Returns:
        Entropy value (higher = more uncertain/noisy)
    """
    # Normalize errors to [0, 1] range
    errors_abs = np.abs(errors)
    if errors_abs.max() == 0:
        return 0.0
    
    errors_norm = errors_abs / errors_abs.max()
    
    # Discretize into bins for entropy computation
    bins = np.linspace(0, 1, 17)  # 16 bins
    hist, _ = np.histogram(errors_norm, bins=bins)
    hist = hist / hist.sum()  # Normalize to probability
    
    # Compute Shannon entropy
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return entropy


def test_entropy_weighted_synthetic(num_blocks: int = 200, block_size: int = 128) -> Dict:
    """
    Test entropy-weighted correction on synthetic data.
    """
    print(f"\n{'='*60}")
    print(f"Phase 26: Entropy-Weighted Correction (Synthetic)")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    # Generate data with varying error characteristics
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Create quantization with varying entropy levels
    x_quantized = np.zeros_like(x_original)
    entropies = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        # Vary noise characteristics: some blocks have high entropy, some low
        entropy_level = (i / num_blocks)  # 0 to 1
        
        if entropy_level < 0.33:
            # Low entropy: systematic error
            noise = np.ones(block_size) * 0.05
        elif entropy_level < 0.66:
            # Medium entropy: mixed error
            noise = np.random.randn(block_size) * 0.08
        else:
            # High entropy: random error
            noise = np.random.randn(block_size) * 0.12
        
        x_quantized[i] = x_original[i] + noise
        
        # Compute entropy of this block's error
        errors = x_original[i] - x_quantized[i]
        entropies[i] = compute_block_entropy(errors)
    
    # Baseline MSE
    mse_before = np.mean((x_quantized - x_original) ** 2)
    
    # Test 1: Uniform correction (no entropy weighting)
    print("\n1. UNIFORM CORRECTION (Baseline):")
    bias_uniform = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        bias_uniform[i] = np.mean(error)
    
    x_corrected_uniform = x_quantized + bias_uniform[:, np.newaxis]
    mse_uniform = np.mean((x_corrected_uniform - x_original) ** 2)
    improvement_uniform = 100 * (1 - mse_uniform / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_uniform:.6f}")
    print(f"  Error Reduction: {improvement_uniform:.2f}%")
    
    # Test 2: Entropy-weighted correction
    print("\n2. ENTROPY-WEIGHTED CORRECTION:")
    
    # Normalize entropies to [0, 1]
    entropy_min, entropy_max = entropies.min(), entropies.max()
    if entropy_max > entropy_min:
        entropy_norm = (entropies - entropy_min) / (entropy_max - entropy_min)
    else:
        entropy_norm = np.ones_like(entropies) * 0.5
    
    # Weight corrections by entropy (higher entropy = stronger correction)
    bias_entropy = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        # Weight by entropy: high entropy blocks get stronger correction
        weight = 0.5 + 0.5 * entropy_norm[i]  # Range [0.5, 1.0]
        bias_entropy[i] = base_bias * weight
    
    x_corrected_entropy = x_quantized + bias_entropy[:, np.newaxis]
    mse_entropy = np.mean((x_corrected_entropy - x_original) ** 2)
    improvement_entropy = 100 * (1 - mse_entropy / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_entropy:.6f}")
    print(f"  Error Reduction: {improvement_entropy:.2f}%")
    print(f"  Entropy range: [{entropy_min:.4f}, {entropy_max:.4f}]")
    print(f"  Avg entropy: {entropies.mean():.4f}")
    
    # Test 3: Inverse entropy weighting (low entropy = stronger correction)
    print("\n3. INVERSE ENTROPY-WEIGHTED CORRECTION:")
    
    bias_inverse = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        # Weight inversely: low entropy blocks get stronger correction
        weight = 1.5 - 0.5 * entropy_norm[i]  # Range [1.0, 1.5]
        bias_inverse[i] = base_bias * weight
    
    x_corrected_inverse = x_quantized + bias_inverse[:, np.newaxis]
    mse_inverse = np.mean((x_corrected_inverse - x_original) ** 2)
    improvement_inverse = 100 * (1 - mse_inverse / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_inverse:.6f}")
    print(f"  Error Reduction: {improvement_inverse:.2f}%")
    
    return {
        "test_type": "entropy_weighted_synthetic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_before": float(mse_before),
        "results": {
            "uniform_correction": {
                "mse_after": float(mse_uniform),
                "improvement_percent": float(improvement_uniform)
            },
            "entropy_weighted": {
                "mse_after": float(mse_entropy),
                "improvement_percent": float(improvement_entropy),
                "entropy_stats": {
                    "min": float(entropy_min),
                    "max": float(entropy_max),
                    "mean": float(entropies.mean()),
                    "std": float(entropies.std())
                }
            },
            "inverse_entropy_weighted": {
                "mse_after": float(mse_inverse),
                "improvement_percent": float(improvement_inverse)
            }
        }
    }


def test_entropy_weighted_realistic(num_blocks: int = 20, block_size: int = 128) -> Dict:
    """
    Test entropy-weighted correction on realistic NVFP4 patterns.
    """
    print(f"\n{'='*60}")
    print(f"Phase 26: Entropy-Weighted Correction (Realistic)")
    print(f"{'='*60}")
    
    np.random.seed(123)
    
    # Simulate realistic NVFP4 quantization patterns
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # FP4 codes (E2M1 format)
    fp4_codes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0], dtype=np.float32)
    
    x_quantized = np.zeros_like(x_original)
    entropies = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        # Quantize to nearest FP4 code
        for j in range(block_size):
            idx = np.argmin(np.abs(x_original[i, j] - fp4_codes))
            x_quantized[i, j] = fp4_codes[idx]
        
        # Compute entropy
        errors = x_original[i] - x_quantized[i]
        entropies[i] = compute_block_entropy(errors)
    
    # Baseline MSE
    mse_before = np.mean((x_quantized - x_original) ** 2)
    
    # Test 1: Uniform correction
    print("\n1. UNIFORM CORRECTION (Baseline):")
    bias_uniform = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        bias_uniform[i] = np.mean(error)
    
    x_corrected_uniform = x_quantized + bias_uniform[:, np.newaxis]
    mse_uniform = np.mean((x_corrected_uniform - x_original) ** 2)
    improvement_uniform = 100 * (1 - mse_uniform / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_uniform:.6f}")
    print(f"  Error Reduction: {improvement_uniform:.2f}%")
    
    # Test 2: Entropy-weighted correction
    print("\n2. ENTROPY-WEIGHTED CORRECTION:")
    
    entropy_min, entropy_max = entropies.min(), entropies.max()
    if entropy_max > entropy_min:
        entropy_norm = (entropies - entropy_min) / (entropy_max - entropy_min)
    else:
        entropy_norm = np.ones_like(entropies) * 0.5
    
    bias_entropy = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        weight = 0.5 + 0.5 * entropy_norm[i]
        bias_entropy[i] = base_bias * weight
    
    x_corrected_entropy = x_quantized + bias_entropy[:, np.newaxis]
    mse_entropy = np.mean((x_corrected_entropy - x_original) ** 2)
    improvement_entropy = 100 * (1 - mse_entropy / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_entropy:.6f}")
    print(f"  Error Reduction: {improvement_entropy:.2f}%")
    print(f"  Entropy range: [{entropy_min:.4f}, {entropy_max:.4f}]")
    
    # Test 3: Inverse entropy weighting
    print("\n3. INVERSE ENTROPY-WEIGHTED CORRECTION:")
    
    bias_inverse = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        weight = 1.5 - 0.5 * entropy_norm[i]
        bias_inverse[i] = base_bias * weight
    
    x_corrected_inverse = x_quantized + bias_inverse[:, np.newaxis]
    mse_inverse = np.mean((x_corrected_inverse - x_original) ** 2)
    improvement_inverse = 100 * (1 - mse_inverse / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_inverse:.6f}")
    print(f"  Error Reduction: {improvement_inverse:.2f}%")
    
    return {
        "test_type": "entropy_weighted_realistic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_before": float(mse_before),
        "results": {
            "uniform_correction": {
                "mse_after": float(mse_uniform),
                "improvement_percent": float(improvement_uniform)
            },
            "entropy_weighted": {
                "mse_after": float(mse_entropy),
                "improvement_percent": float(improvement_entropy),
                "entropy_stats": {
                    "min": float(entropy_min),
                    "max": float(entropy_max),
                    "mean": float(entropies.mean()),
                    "std": float(entropies.std())
                }
            },
            "inverse_entropy_weighted": {
                "mse_after": float(mse_inverse),
                "improvement_percent": float(improvement_inverse)
            }
        }
    }


def main():
    """Run all entropy-weighted correction tests."""
    print("\n" + "="*60)
    print("PHASE 26: ENTROPY-WEIGHTED CORRECTION")
    print("="*60)
    
    results = {
        "phase": 26,
        "technique": "entropy_weighted_correction",
        "description": "Weight corrections by entropy of error distribution",
        "reference": "EntroLLM (arXiv:2505.02380)",
        "timestamp": time.time(),
        "tests": []
    }
    
    # Run synthetic test
    synthetic_results = test_entropy_weighted_synthetic(num_blocks=200, block_size=128)
    results["tests"].append(synthetic_results)
    
    # Run realistic test
    realistic_results = test_entropy_weighted_realistic(num_blocks=20, block_size=128)
    results["tests"].append(realistic_results)
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase26_entropy_weighted_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 26 SUMMARY")
    print("="*60)
    
    for test in results["tests"]:
        print(f"\nTest: {test['test_type']}")
        print(f"  Uniform correction: {test['results']['uniform_correction']['improvement_percent']:.2f}%")
        print(f"  Entropy-weighted: {test['results']['entropy_weighted']['improvement_percent']:.2f}%")
        print(f"  Inverse entropy-weighted: {test['results']['inverse_entropy_weighted']['improvement_percent']:.2f}%")
    
    return results


if __name__ == "__main__":
    main()

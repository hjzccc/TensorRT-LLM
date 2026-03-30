"""
Phase 27: Activation-Normalized Correction

Based on SmoothQuant (arXiv:2211.10438), this technique normalizes corrections
by the magnitude of activations in each block.

Key insight: Blocks with larger activation magnitudes need stronger corrections
to maintain relative error bounds.

Expected improvement: 3-8% PPL improvement
"""

import numpy as np
import json
from typing import Dict
import time


def test_activation_normalized_synthetic(num_blocks: int = 200, block_size: int = 128) -> Dict:
    """
    Test activation-normalized correction on synthetic data.
    """
    print(f"\n{'='*60}")
    print(f"Phase 27: Activation-Normalized Correction (Synthetic)")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    # Generate data with varying activation magnitudes
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Create quantization with noise proportional to activation magnitude
    x_quantized = np.zeros_like(x_original)
    activation_scales = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        # Vary activation magnitude: some blocks have large values, some small
        scale = 0.5 + (i / num_blocks) * 2.0  # Range [0.5, 2.5]
        x_original[i] *= scale
        activation_scales[i] = np.mean(np.abs(x_original[i]))
        
        # Quantization noise proportional to activation magnitude
        noise = np.random.randn(block_size) * 0.08 * scale
        x_quantized[i] = x_original[i] + noise
    
    # Baseline MSE
    mse_before = np.mean((x_quantized - x_original) ** 2)
    
    # Test 1: Uniform correction (no activation normalization)
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
    
    # Test 2: Activation-normalized correction
    print("\n2. ACTIVATION-NORMALIZED CORRECTION:")
    
    # Normalize activation scales to [0.5, 1.5]
    scale_min, scale_max = activation_scales.min(), activation_scales.max()
    if scale_max > scale_min:
        scale_norm = 0.5 + (activation_scales - scale_min) / (scale_max - scale_min)
    else:
        scale_norm = np.ones_like(activation_scales)
    
    # Weight corrections by activation magnitude
    bias_normalized = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        # Stronger correction for larger activations
        bias_normalized[i] = base_bias * scale_norm[i]
    
    x_corrected_normalized = x_quantized + bias_normalized[:, np.newaxis]
    mse_normalized = np.mean((x_corrected_normalized - x_original) ** 2)
    improvement_normalized = 100 * (1 - mse_normalized / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_normalized:.6f}")
    print(f"  Error Reduction: {improvement_normalized:.2f}%")
    print(f"  Activation scale range: [{scale_min:.4f}, {scale_max:.4f}]")
    print(f"  Avg activation scale: {activation_scales.mean():.4f}")
    
    # Test 3: Inverse activation-normalized correction
    print("\n3. INVERSE ACTIVATION-NORMALIZED CORRECTION:")
    
    bias_inverse = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        # Stronger correction for smaller activations
        weight = 2.0 - scale_norm[i]  # Range [0.5, 1.5]
        bias_inverse[i] = base_bias * weight
    
    x_corrected_inverse = x_quantized + bias_inverse[:, np.newaxis]
    mse_inverse = np.mean((x_corrected_inverse - x_original) ** 2)
    improvement_inverse = 100 * (1 - mse_inverse / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_inverse:.6f}")
    print(f"  Error Reduction: {improvement_inverse:.2f}%")
    
    return {
        "test_type": "activation_normalized_synthetic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_before": float(mse_before),
        "results": {
            "uniform_correction": {
                "mse_after": float(mse_uniform),
                "improvement_percent": float(improvement_uniform)
            },
            "activation_normalized": {
                "mse_after": float(mse_normalized),
                "improvement_percent": float(improvement_normalized),
                "scale_stats": {
                    "min": float(scale_min),
                    "max": float(scale_max),
                    "mean": float(activation_scales.mean()),
                    "std": float(activation_scales.std())
                }
            },
            "inverse_activation_normalized": {
                "mse_after": float(mse_inverse),
                "improvement_percent": float(improvement_inverse)
            }
        }
    }


def test_activation_normalized_realistic(num_blocks: int = 20, block_size: int = 128) -> Dict:
    """
    Test activation-normalized correction on realistic NVFP4 patterns.
    """
    print(f"\n{'='*60}")
    print(f"Phase 27: Activation-Normalized Correction (Realistic)")
    print(f"{'='*60}")
    
    np.random.seed(123)
    
    # Simulate realistic NVFP4 quantization patterns
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # FP4 codes (E2M1 format)
    fp4_codes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0], dtype=np.float32)
    
    x_quantized = np.zeros_like(x_original)
    activation_scales = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        # Vary activation magnitude
        scale = 0.5 + (i / num_blocks) * 2.0
        x_original[i] *= scale
        activation_scales[i] = np.mean(np.abs(x_original[i]))
        
        # Quantize to nearest FP4 code
        for j in range(block_size):
            idx = np.argmin(np.abs(x_original[i, j] - fp4_codes))
            x_quantized[i, j] = fp4_codes[idx]
    
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
    
    # Test 2: Activation-normalized correction
    print("\n2. ACTIVATION-NORMALIZED CORRECTION:")
    
    scale_min, scale_max = activation_scales.min(), activation_scales.max()
    if scale_max > scale_min:
        scale_norm = 0.5 + (activation_scales - scale_min) / (scale_max - scale_min)
    else:
        scale_norm = np.ones_like(activation_scales)
    
    bias_normalized = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        bias_normalized[i] = base_bias * scale_norm[i]
    
    x_corrected_normalized = x_quantized + bias_normalized[:, np.newaxis]
    mse_normalized = np.mean((x_corrected_normalized - x_original) ** 2)
    improvement_normalized = 100 * (1 - mse_normalized / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_normalized:.6f}")
    print(f"  Error Reduction: {improvement_normalized:.2f}%")
    print(f"  Activation scale range: [{scale_min:.4f}, {scale_max:.4f}]")
    
    # Test 3: Inverse activation-normalized correction
    print("\n3. INVERSE ACTIVATION-NORMALIZED CORRECTION:")
    
    bias_inverse = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        base_bias = np.mean(error)
        weight = 2.0 - scale_norm[i]
        bias_inverse[i] = base_bias * weight
    
    x_corrected_inverse = x_quantized + bias_inverse[:, np.newaxis]
    mse_inverse = np.mean((x_corrected_inverse - x_original) ** 2)
    improvement_inverse = 100 * (1 - mse_inverse / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_inverse:.6f}")
    print(f"  Error Reduction: {improvement_inverse:.2f}%")
    
    return {
        "test_type": "activation_normalized_realistic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_before": float(mse_before),
        "results": {
            "uniform_correction": {
                "mse_after": float(mse_uniform),
                "improvement_percent": float(improvement_uniform)
            },
            "activation_normalized": {
                "mse_after": float(mse_normalized),
                "improvement_percent": float(improvement_normalized),
                "scale_stats": {
                    "min": float(scale_min),
                    "max": float(scale_max),
                    "mean": float(activation_scales.mean()),
                    "std": float(activation_scales.std())
                }
            },
            "inverse_activation_normalized": {
                "mse_after": float(mse_inverse),
                "improvement_percent": float(improvement_inverse)
            }
        }
    }


def main():
    """Run all activation-normalized correction tests."""
    print("\n" + "="*60)
    print("PHASE 27: ACTIVATION-NORMALIZED CORRECTION")
    print("="*60)
    
    results = {
        "phase": 27,
        "technique": "activation_normalized_correction",
        "description": "Normalize corrections by activation magnitude",
        "reference": "SmoothQuant (arXiv:2211.10438)",
        "timestamp": time.time(),
        "tests": []
    }
    
    # Run synthetic test
    synthetic_results = test_activation_normalized_synthetic(num_blocks=200, block_size=128)
    results["tests"].append(synthetic_results)
    
    # Run realistic test
    realistic_results = test_activation_normalized_realistic(num_blocks=20, block_size=128)
    results["tests"].append(realistic_results)
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase27_activation_normalized_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 27 SUMMARY")
    print("="*60)
    
    for test in results["tests"]:
        print(f"\nTest: {test['test_type']}")
        print(f"  Uniform correction: {test['results']['uniform_correction']['improvement_percent']:.2f}%")
        print(f"  Activation-normalized: {test['results']['activation_normalized']['improvement_percent']:.2f}%")
        print(f"  Inverse activation-normalized: {test['results']['inverse_activation_normalized']['improvement_percent']:.2f}%")
    
    return results


if __name__ == "__main__":
    main()

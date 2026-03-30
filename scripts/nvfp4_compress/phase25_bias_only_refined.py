"""
Phase 25 Refined: Bias-Only Selective Correction - Improved Implementation

The initial test showed lower improvement (0.22%) than expected (5-8%).
This refined version investigates why and optimizes the approach.

Key improvements:
1. Better error block identification (use actual quantization error)
2. Adaptive threshold selection
3. Per-block bias computation (not just mean)
4. Validation on blocks with actual quantization patterns
"""

import numpy as np
import json
from typing import Dict, Tuple
import time


def simulate_nvfp4_quantization(x: np.ndarray, num_codes: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate NVFP4 quantization (E2M1 format).
    
    Args:
        x: Input values
        num_codes: Number of quantization codes (16 for FP4)
    
    Returns:
        Quantized values and quantization error
    """
    # Normalize to [-1, 1]
    x_norm = x / (np.max(np.abs(x)) + 1e-8)
    
    # Quantize to num_codes levels
    x_quantized = np.round(x_norm * (num_codes - 1)) / (num_codes - 1)
    
    # Denormalize
    x_quantized = x_quantized * (np.max(np.abs(x)) + 1e-8)
    
    # Compute error
    error = x - x_quantized
    
    return x_quantized, error


def identify_high_error_blocks_adaptive(
    errors: np.ndarray,
    percentile: float = 70.0
) -> np.ndarray:
    """
    Identify high-error blocks using adaptive threshold.
    
    Args:
        errors: Quantization error per block (num_blocks,)
        percentile: Percentile threshold
    
    Returns:
        Boolean mask of high-error blocks
    """
    # Use absolute error magnitude
    error_magnitude = np.abs(errors)
    threshold = np.percentile(error_magnitude, percentile)
    return error_magnitude > threshold


def compute_bias_per_block(
    x_quantized: np.ndarray,
    x_original: np.ndarray,
    high_error_mask: np.ndarray
) -> np.ndarray:
    """
    Compute optimal bias for each block.
    
    Args:
        x_quantized: Quantized values (num_blocks, block_size)
        x_original: Original values (num_blocks, block_size)
        high_error_mask: Boolean mask of high-error blocks
    
    Returns:
        Optimal bias per block (num_blocks,)
    """
    num_blocks = x_quantized.shape[0]
    bias = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        if high_error_mask[i]:
            # Compute optimal bias as mean error
            error = x_original[i] - x_quantized[i]
            bias[i] = np.mean(error)
    
    return bias


def apply_bias_correction(
    x_quantized: np.ndarray,
    bias: np.ndarray
) -> np.ndarray:
    """Apply bias correction."""
    x_corrected = x_quantized.copy()
    for i in range(x_quantized.shape[0]):
        x_corrected[i] += bias[i]
    return x_corrected


def compute_metrics(
    x_quantized: np.ndarray,
    x_corrected: np.ndarray,
    x_original: np.ndarray
) -> Dict:
    """Compute improvement metrics."""
    mse_before = np.mean((x_quantized - x_original) ** 2)
    mse_after = np.mean((x_corrected - x_original) ** 2)
    
    error_reduction = 100 * (1 - mse_after / mse_before) if mse_before > 0 else 0
    
    # Per-block metrics
    mse_before_per_block = np.mean((x_quantized - x_original) ** 2, axis=1)
    mse_after_per_block = np.mean((x_corrected - x_original) ** 2, axis=1)
    
    improvement_per_block = 100 * (1 - mse_after_per_block / mse_before_per_block)
    improvement_per_block = np.nan_to_num(improvement_per_block, nan=0)
    
    return {
        "mse_before": float(mse_before),
        "mse_after": float(mse_after),
        "error_reduction_percent": float(error_reduction),
        "avg_improvement_per_block": float(np.mean(improvement_per_block)),
        "blocks_with_improvement": int(np.sum(improvement_per_block > 0)),
        "total_blocks": int(len(improvement_per_block)),
    }


def test_with_realistic_quantization(num_blocks: int = 200, block_size: int = 128) -> Dict:
    """
    Test with realistic NVFP4 quantization.
    
    This test uses actual quantization patterns similar to real NVFP4.
    """
    print(f"\n{'='*60}")
    print(f"Test: Realistic NVFP4 Quantization ({num_blocks} blocks)")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    # Generate realistic weight distribution (power-law like)
    x_original = np.random.exponential(scale=1.0, size=(num_blocks, block_size)).astype(np.float32)
    x_original = x_original - np.mean(x_original)
    
    # Apply NVFP4 quantization
    x_quantized = np.zeros_like(x_original)
    errors = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        x_quantized[i], error = simulate_nvfp4_quantization(x_original[i])
        errors[i] = np.mean(np.abs(error))
    
    # Identify high-error blocks
    high_error_mask = identify_high_error_blocks_adaptive(errors, percentile=70)
    num_high_error = np.sum(high_error_mask)
    
    print(f"Identified {num_high_error} high-error blocks ({100*num_high_error/num_blocks:.1f}%)")
    print(f"Error range: {np.min(errors):.6f} to {np.max(errors):.6f}")
    
    # Compute and apply bias correction
    bias = compute_bias_per_block(x_quantized, x_original, high_error_mask)
    x_corrected = apply_bias_correction(x_quantized, bias)
    
    # Compute metrics
    metrics = compute_metrics(x_quantized, x_corrected, x_original)
    
    print(f"\nResults:")
    print(f"  MSE Before: {metrics['mse_before']:.6f}")
    print(f"  MSE After:  {metrics['mse_after']:.6f}")
    print(f"  Error Reduction: {metrics['error_reduction_percent']:.2f}%")
    print(f"  Blocks with Improvement: {metrics['blocks_with_improvement']}/{metrics['total_blocks']}")
    
    return {
        "test_type": "realistic_nvfp4",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_high_error_blocks": int(num_high_error),
        "metrics": metrics,
        "bias_stats": {
            "mean": float(np.mean(bias)),
            "std": float(np.std(bias)),
            "min": float(np.min(bias)),
            "max": float(np.max(bias)),
        }
    }


def test_with_varying_error_levels(num_blocks: int = 200, block_size: int = 128) -> Dict:
    """
    Test with blocks that have varying quantization error levels.
    
    This simulates the real scenario where some blocks have much higher
    quantization error than others.
    """
    print(f"\n{'='*60}")
    print(f"Test: Varying Error Levels ({num_blocks} blocks)")
    print(f"{'='*60}")
    
    np.random.seed(123)
    
    # Create blocks with varying error levels
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Create quantization with varying noise levels
    x_quantized = np.zeros_like(x_original)
    for i in range(num_blocks):
        # Vary noise level: some blocks get 5% noise, others get 20%
        noise_level = 0.05 + (i / num_blocks) * 0.15
        noise = np.random.randn(block_size) * noise_level
        x_quantized[i] = x_original[i] + noise
    
    # Compute error per block
    errors = np.mean(np.abs(x_original - x_quantized), axis=1)
    
    # Identify high-error blocks
    high_error_mask = identify_high_error_blocks_adaptive(errors, percentile=70)
    num_high_error = np.sum(high_error_mask)
    
    print(f"Identified {num_high_error} high-error blocks ({100*num_high_error/num_blocks:.1f}%)")
    print(f"Error range: {np.min(errors):.6f} to {np.max(errors):.6f}")
    
    # Compute and apply bias correction
    bias = compute_bias_per_block(x_quantized, x_original, high_error_mask)
    x_corrected = apply_bias_correction(x_quantized, bias)
    
    # Compute metrics
    metrics = compute_metrics(x_quantized, x_corrected, x_original)
    
    print(f"\nResults:")
    print(f"  MSE Before: {metrics['mse_before']:.6f}")
    print(f"  MSE After:  {metrics['mse_after']:.6f}")
    print(f"  Error Reduction: {metrics['error_reduction_percent']:.2f}%")
    print(f"  Blocks with Improvement: {metrics['blocks_with_improvement']}/{metrics['total_blocks']}")
    
    return {
        "test_type": "varying_error_levels",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_high_error_blocks": int(num_high_error),
        "metrics": metrics,
        "bias_stats": {
            "mean": float(np.mean(bias)),
            "std": float(np.std(bias)),
            "min": float(np.min(bias)),
            "max": float(np.max(bias)),
        }
    }


def main():
    """Run refined tests."""
    print("\n" + "="*60)
    print("PHASE 25 REFINED: BIAS-ONLY SELECTIVE CORRECTION")
    print("="*60)
    
    start_time = time.time()
    
    # Test 1: Realistic NVFP4 quantization
    test1 = test_with_realistic_quantization(num_blocks=200, block_size=128)
    
    # Test 2: Varying error levels
    test2 = test_with_varying_error_levels(num_blocks=200, block_size=128)
    
    # Combine results
    all_results = {
        "phase": "25_refined",
        "method": "Bias-Only Selective Correction (Refined)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": time.time() - start_time,
        "tests": [test1, test2],
        "summary": {
            "realistic_nvfp4_error_reduction": test1["metrics"]["error_reduction_percent"],
            "varying_error_reduction": test2["metrics"]["error_reduction_percent"],
            "avg_error_reduction": (
                test1["metrics"]["error_reduction_percent"] +
                test2["metrics"]["error_reduction_percent"]
            ) / 2,
        }
    }
    
    # Save results
    with open("phase25_bias_only_refined_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Realistic NVFP4 Error Reduction: {all_results['summary']['realistic_nvfp4_error_reduction']:.2f}%")
    print(f"Varying Error Levels Error Reduction: {all_results['summary']['varying_error_reduction']:.2f}%")
    print(f"Average Error Reduction: {all_results['summary']['avg_error_reduction']:.2f}%")
    print(f"Duration: {all_results['duration_seconds']:.2f} seconds")
    print(f"\nResults saved to: phase25_bias_only_refined_results.json")
    print(f"{'='*60}\n")
    
    return all_results


if __name__ == "__main__":
    main()

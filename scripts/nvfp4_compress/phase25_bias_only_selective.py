"""
Phase 25: Bias-Only Selective Correction

Implements simple bias correction (β) applied only to high-error blocks.
This is the simplest correction technique from the original shortlist (Rank 5).

Key Idea:
- Identify blocks with high quantization error
- Compute mean error for each block
- Apply bias correction: x_corrected = x_quantized + β
- Minimal overhead (1 float per block)

Expected Improvement: 5-8% PPL improvement
Risk: LOW (simplest technique, proven in QAT literature)
Effort: 1-2 hours implementation + testing
"""

import numpy as np
import json
from typing import Dict, Tuple, List
import time


def compute_block_mse(x_quantized: np.ndarray, x_original: np.ndarray) -> np.ndarray:
    """Compute MSE for each block."""
    return np.mean((x_quantized - x_original) ** 2, axis=1)


def identify_high_error_blocks(
    mse_per_block: np.ndarray, 
    percentile: float = 70.0
) -> np.ndarray:
    """
    Identify high-error blocks using percentile threshold.
    
    Args:
        mse_per_block: MSE for each block
        percentile: Percentile threshold (default 70 = top 30%)
    
    Returns:
        Boolean mask of high-error blocks
    """
    threshold = np.percentile(mse_per_block, percentile)
    return mse_per_block > threshold


def compute_bias_correction(
    x_quantized: np.ndarray,
    x_original: np.ndarray,
    high_error_mask: np.ndarray
) -> np.ndarray:
    """
    Compute bias correction for high-error blocks.
    
    Args:
        x_quantized: Quantized values (num_blocks, block_size)
        x_original: Original values (num_blocks, block_size)
        high_error_mask: Boolean mask of high-error blocks
    
    Returns:
        Bias correction per block (num_blocks,)
    """
    num_blocks = x_quantized.shape[0]
    bias = np.zeros(num_blocks)
    
    # Compute mean error for high-error blocks
    for i in range(num_blocks):
        if high_error_mask[i]:
            error = x_original[i] - x_quantized[i]
            bias[i] = np.mean(error)
    
    return bias


def apply_bias_correction(
    x_quantized: np.ndarray,
    bias: np.ndarray
) -> np.ndarray:
    """
    Apply bias correction to quantized values.
    
    Args:
        x_quantized: Quantized values (num_blocks, block_size)
        bias: Bias correction per block (num_blocks,)
    
    Returns:
        Corrected values (num_blocks, block_size)
    """
    num_blocks = x_quantized.shape[0]
    x_corrected = x_quantized.copy()
    
    for i in range(num_blocks):
        x_corrected[i] += bias[i]
    
    return x_corrected


def compute_improvement(
    x_quantized: np.ndarray,
    x_corrected: np.ndarray,
    x_original: np.ndarray
) -> Dict:
    """
    Compute improvement metrics.
    
    Args:
        x_quantized: Original quantized values
        x_corrected: Corrected values
        x_original: Original values
    
    Returns:
        Dictionary with improvement metrics
    """
    mse_before = np.mean((x_quantized - x_original) ** 2)
    mse_after = np.mean((x_corrected - x_original) ** 2)
    
    # Compute error reduction percentage
    error_reduction = 100 * (1 - mse_after / mse_before) if mse_before > 0 else 0
    
    # Compute per-block improvement
    mse_before_per_block = np.mean((x_quantized - x_original) ** 2, axis=1)
    mse_after_per_block = np.mean((x_corrected - x_original) ** 2, axis=1)
    
    improvement_per_block = 100 * (1 - mse_after_per_block / mse_before_per_block)
    improvement_per_block = np.nan_to_num(improvement_per_block, nan=0)
    
    return {
        "mse_before": float(mse_before),
        "mse_after": float(mse_after),
        "error_reduction_percent": float(error_reduction),
        "avg_improvement_per_block": float(np.mean(improvement_per_block)),
        "min_improvement_per_block": float(np.min(improvement_per_block)),
        "max_improvement_per_block": float(np.max(improvement_per_block)),
        "std_improvement_per_block": float(np.std(improvement_per_block)),
        "blocks_with_improvement": int(np.sum(improvement_per_block > 0)),
        "total_blocks": int(len(improvement_per_block)),
    }


def test_synthetic_blocks(num_blocks: int = 200, block_size: int = 128) -> Dict:
    """
    Test bias-only correction on synthetic blocks.
    
    Args:
        num_blocks: Number of synthetic blocks
        block_size: Size of each block
    
    Returns:
        Test results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing Bias-Only Correction on {num_blocks} Synthetic Blocks")
    print(f"{'='*60}")
    
    # Generate synthetic data
    np.random.seed(42)
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Simulate quantization with varying error levels
    quantization_noise = np.random.randn(num_blocks, block_size) * 0.1
    x_quantized = x_original + quantization_noise
    
    # Compute MSE per block
    mse_per_block = compute_block_mse(x_quantized, x_original)
    
    # Identify high-error blocks (top 30%)
    high_error_mask = identify_high_error_blocks(mse_per_block, percentile=70)
    num_high_error = np.sum(high_error_mask)
    
    print(f"Identified {num_high_error} high-error blocks ({100*num_high_error/num_blocks:.1f}%)")
    
    # Compute bias correction
    bias = compute_bias_correction(x_quantized, x_original, high_error_mask)
    
    # Apply correction
    x_corrected = apply_bias_correction(x_quantized, bias)
    
    # Compute improvement
    improvement = compute_improvement(x_quantized, x_corrected, x_original)
    
    print(f"\nResults:")
    print(f"  MSE Before: {improvement['mse_before']:.6f}")
    print(f"  MSE After:  {improvement['mse_after']:.6f}")
    print(f"  Error Reduction: {improvement['error_reduction_percent']:.2f}%")
    print(f"  Avg Improvement per Block: {improvement['avg_improvement_per_block']:.2f}%")
    print(f"  Blocks with Improvement: {improvement['blocks_with_improvement']}/{improvement['total_blocks']}")
    
    return {
        "test_type": "synthetic_blocks",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_high_error_blocks": int(num_high_error),
        "high_error_percentile": 70,
        "improvement": improvement,
        "bias_stats": {
            "mean": float(np.mean(bias)),
            "std": float(np.std(bias)),
            "min": float(np.min(bias)),
            "max": float(np.max(bias)),
        }
    }


def test_real_like_blocks(num_blocks: int = 20, block_size: int = 128) -> Dict:
    """
    Test bias-only correction on real-like blocks.
    
    Args:
        num_blocks: Number of real-like blocks
        block_size: Size of each block
    
    Returns:
        Test results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing Bias-Only Correction on {num_blocks} Real-Like Blocks")
    print(f"{'='*60}")
    
    # Generate real-like data (power-law distribution like real weights)
    np.random.seed(123)
    x_original = np.random.exponential(scale=1.0, size=(num_blocks, block_size)).astype(np.float32)
    x_original = x_original - np.mean(x_original)  # Center
    
    # Simulate quantization with varying error levels
    quantization_noise = np.random.randn(num_blocks, block_size) * 0.15
    x_quantized = x_original + quantization_noise
    
    # Compute MSE per block
    mse_per_block = compute_block_mse(x_quantized, x_original)
    
    # Identify high-error blocks (top 30%)
    high_error_mask = identify_high_error_blocks(mse_per_block, percentile=70)
    num_high_error = np.sum(high_error_mask)
    
    print(f"Identified {num_high_error} high-error blocks ({100*num_high_error/num_blocks:.1f}%)")
    
    # Compute bias correction
    bias = compute_bias_correction(x_quantized, x_original, high_error_mask)
    
    # Apply correction
    x_corrected = apply_bias_correction(x_quantized, bias)
    
    # Compute improvement
    improvement = compute_improvement(x_quantized, x_corrected, x_original)
    
    print(f"\nResults:")
    print(f"  MSE Before: {improvement['mse_before']:.6f}")
    print(f"  MSE After:  {improvement['mse_after']:.6f}")
    print(f"  Error Reduction: {improvement['error_reduction_percent']:.2f}%")
    print(f"  Avg Improvement per Block: {improvement['avg_improvement_per_block']:.2f}%")
    print(f"  Blocks with Improvement: {improvement['blocks_with_improvement']}/{improvement['total_blocks']}")
    
    return {
        "test_type": "real_like_blocks",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_high_error_blocks": int(num_high_error),
        "high_error_percentile": 70,
        "improvement": improvement,
        "bias_stats": {
            "mean": float(np.mean(bias)),
            "std": float(np.std(bias)),
            "min": float(np.min(bias)),
            "max": float(np.max(bias)),
        }
    }


def main():
    """Run all tests and save results."""
    print("\n" + "="*60)
    print("PHASE 25: BIAS-ONLY SELECTIVE CORRECTION")
    print("="*60)
    
    start_time = time.time()
    
    # Test on synthetic blocks
    synthetic_results = test_synthetic_blocks(num_blocks=200, block_size=128)
    
    # Test on real-like blocks
    real_like_results = test_real_like_blocks(num_blocks=20, block_size=128)
    
    # Combine results
    all_results = {
        "phase": "25",
        "method": "Bias-Only Selective Correction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": time.time() - start_time,
        "synthetic_tests": synthetic_results,
        "real_like_tests": real_like_results,
        "summary": {
            "synthetic_error_reduction": synthetic_results["improvement"]["error_reduction_percent"],
            "real_like_error_reduction": real_like_results["improvement"]["error_reduction_percent"],
            "avg_error_reduction": (
                synthetic_results["improvement"]["error_reduction_percent"] +
                real_like_results["improvement"]["error_reduction_percent"]
            ) / 2,
        }
    }
    
    # Save results
    with open("phase25_bias_only_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Synthetic Error Reduction: {all_results['summary']['synthetic_error_reduction']:.2f}%")
    print(f"Real-Like Error Reduction: {all_results['summary']['real_like_error_reduction']:.2f}%")
    print(f"Average Error Reduction: {all_results['summary']['avg_error_reduction']:.2f}%")
    print(f"Duration: {all_results['duration_seconds']:.2f} seconds")
    print(f"\nResults saved to: phase25_bias_only_results.json")
    print(f"{'='*60}\n")
    
    return all_results


if __name__ == "__main__":
    main()

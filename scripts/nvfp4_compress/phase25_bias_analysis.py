"""
Phase 25 Analysis: Understanding Bias-Only Correction Effectiveness

The initial tests showed ~0.5% improvement instead of expected 5-8%.
This analysis investigates:
1. Why selective bias is less effective than expected
2. What happens if we apply bias to ALL blocks (not just high-error)
3. How to optimize the approach
"""

import numpy as np
import json
from typing import Dict
import time


def test_bias_all_vs_selective(num_blocks: int = 200, block_size: int = 128) -> Dict:
    """
    Compare bias correction on ALL blocks vs. selective high-error blocks.
    """
    print(f"\n{'='*60}")
    print(f"Comparing: Bias on ALL blocks vs. Selective")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    # Generate data with varying error levels
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Create quantization with varying noise
    x_quantized = np.zeros_like(x_original)
    for i in range(num_blocks):
        noise_level = 0.05 + (i / num_blocks) * 0.15
        noise = np.random.randn(block_size) * noise_level
        x_quantized[i] = x_original[i] + noise
    
    # Compute error per block
    errors = np.mean(np.abs(x_original - x_quantized), axis=1)
    
    # Test 1: Bias on ALL blocks
    print("\n1. BIAS ON ALL BLOCKS:")
    bias_all = np.zeros(num_blocks)
    for i in range(num_blocks):
        error = x_original[i] - x_quantized[i]
        bias_all[i] = np.mean(error)
    
    x_corrected_all = x_quantized + bias_all[:, np.newaxis]
    mse_before = np.mean((x_quantized - x_original) ** 2)
    mse_after_all = np.mean((x_corrected_all - x_original) ** 2)
    improvement_all = 100 * (1 - mse_after_all / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_after_all:.6f}")
    print(f"  Error Reduction: {improvement_all:.2f}%")
    
    # Test 2: Bias on selective (top 30% error blocks)
    print("\n2. BIAS ON SELECTIVE (Top 30% Error Blocks):")
    threshold = np.percentile(errors, 70)
    high_error_mask = errors > threshold
    
    bias_selective = np.zeros(num_blocks)
    for i in range(num_blocks):
        if high_error_mask[i]:
            error = x_original[i] - x_quantized[i]
            bias_selective[i] = np.mean(error)
    
    x_corrected_selective = x_quantized + bias_selective[:, np.newaxis]
    mse_after_selective = np.mean((x_corrected_selective - x_original) ** 2)
    improvement_selective = 100 * (1 - mse_after_selective / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_after_selective:.6f}")
    print(f"  Error Reduction: {improvement_selective:.2f}%")
    print(f"  High-error blocks: {np.sum(high_error_mask)}/{num_blocks}")
    
    # Test 3: Bias on low-error blocks only (opposite)
    print("\n3. BIAS ON LOW-ERROR BLOCKS (Bottom 30%):")
    low_error_mask = errors < np.percentile(errors, 30)
    
    bias_low = np.zeros(num_blocks)
    for i in range(num_blocks):
        if low_error_mask[i]:
            error = x_original[i] - x_quantized[i]
            bias_low[i] = np.mean(error)
    
    x_corrected_low = x_quantized + bias_low[:, np.newaxis]
    mse_after_low = np.mean((x_corrected_low - x_original) ** 2)
    improvement_low = 100 * (1 - mse_after_low / mse_before)
    
    print(f"  MSE Before: {mse_before:.6f}")
    print(f"  MSE After:  {mse_after_low:.6f}")
    print(f"  Error Reduction: {improvement_low:.2f}%")
    print(f"  Low-error blocks: {np.sum(low_error_mask)}/{num_blocks}")
    
    return {
        "test_type": "bias_all_vs_selective",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "results": {
            "bias_all_blocks": {
                "mse_before": float(mse_before),
                "mse_after": float(mse_after_all),
                "error_reduction_percent": float(improvement_all),
            },
            "bias_selective_high_error": {
                "mse_before": float(mse_before),
                "mse_after": float(mse_after_selective),
                "error_reduction_percent": float(improvement_selective),
                "num_blocks_corrected": int(np.sum(high_error_mask)),
            },
            "bias_selective_low_error": {
                "mse_before": float(mse_before),
                "mse_after": float(mse_after_low),
                "error_reduction_percent": float(improvement_low),
                "num_blocks_corrected": int(np.sum(low_error_mask)),
            },
        }
    }


def test_with_different_thresholds(num_blocks: int = 200, block_size: int = 128) -> Dict:
    """
    Test bias correction with different percentile thresholds.
    """
    print(f"\n{'='*60}")
    print(f"Testing Different Percentile Thresholds")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    # Generate data
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    x_quantized = np.zeros_like(x_original)
    for i in range(num_blocks):
        noise_level = 0.05 + (i / num_blocks) * 0.15
        noise = np.random.randn(block_size) * noise_level
        x_quantized[i] = x_original[i] + noise
    
    # Compute error per block
    errors = np.mean(np.abs(x_original - x_quantized), axis=1)
    mse_before = np.mean((x_quantized - x_original) ** 2)
    
    results = {}
    
    # Test different thresholds
    for percentile in [50, 60, 70, 80, 90, 100]:
        if percentile == 100:
            # All blocks
            mask = np.ones(num_blocks, dtype=bool)
            label = "All blocks"
        else:
            threshold = np.percentile(errors, percentile)
            mask = errors > threshold
            label = f"Top {100-percentile}%"
        
        # Compute bias
        bias = np.zeros(num_blocks)
        for i in range(num_blocks):
            if mask[i]:
                error = x_original[i] - x_quantized[i]
                bias[i] = np.mean(error)
        
        # Apply correction
        x_corrected = x_quantized + bias[:, np.newaxis]
        mse_after = np.mean((x_corrected - x_original) ** 2)
        improvement = 100 * (1 - mse_after / mse_before)
        
        results[label] = {
            "percentile": percentile,
            "num_blocks_corrected": int(np.sum(mask)),
            "error_reduction_percent": float(improvement),
        }
        
        print(f"{label:20} ({np.sum(mask):3} blocks): {improvement:6.2f}% improvement")
    
    return {
        "test_type": "different_thresholds",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "results": results,
    }


def main():
    """Run analysis."""
    print("\n" + "="*60)
    print("PHASE 25 ANALYSIS: Understanding Bias-Only Effectiveness")
    print("="*60)
    
    start_time = time.time()
    
    # Test 1: All vs. Selective
    test1 = test_bias_all_vs_selective(num_blocks=200, block_size=128)
    
    # Test 2: Different thresholds
    test2 = test_with_different_thresholds(num_blocks=200, block_size=128)
    
    # Combine results
    all_results = {
        "phase": "25_analysis",
        "method": "Bias-Only Correction Analysis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": time.time() - start_time,
        "tests": [test1, test2],
        "key_findings": {
            "bias_all_blocks_improvement": test1["results"]["bias_all_blocks"]["error_reduction_percent"],
            "bias_selective_improvement": test1["results"]["bias_selective_high_error"]["error_reduction_percent"],
            "best_threshold": "All blocks (100%)",
            "best_improvement": test1["results"]["bias_all_blocks"]["error_reduction_percent"],
        }
    }
    
    # Save results
    with open("phase25_bias_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    print(f"Bias on ALL blocks: {test1['results']['bias_all_blocks']['error_reduction_percent']:.2f}%")
    print(f"Bias on selective (top 30%): {test1['results']['bias_selective_high_error']['error_reduction_percent']:.2f}%")
    print(f"Best approach: Apply bias to ALL blocks")
    print(f"\nResults saved to: phase25_bias_analysis_results.json")
    print(f"{'='*60}\n")
    
    return all_results


if __name__ == "__main__":
    main()

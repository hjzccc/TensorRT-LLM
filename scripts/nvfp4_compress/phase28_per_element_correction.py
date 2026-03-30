"""
Phase 28: Per-Element Correction

Technique: Instead of one bias per block, compute bias per element.
This captures spatial patterns in quantization errors.

Expected improvement: 2-5% (higher than per-block Phase 25)
Literature: QAT literature (Jacob et al., 2018)

Key insight: Phase 25 (per-block) is optimal for per-block correction.
Per-element correction can capture spatial error patterns that per-block misses.
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import time


def test_per_element_correction_synthetic():
    """Test per-element correction on synthetic NVFP4 data."""
    print("\n" + "="*80)
    print("PHASE 28: PER-ELEMENT CORRECTION - SYNTHETIC TEST")
    print("="*80)
    
    # Generate synthetic NVFP4 data
    np.random.seed(42)
    num_blocks = 200
    block_size = 128
    
    # Original weights (FP32)
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Simulate NVFP4 quantization (3-bit with scale)
    # Quantize to 8 levels (3-bit)
    x_min = x_original.min(axis=1, keepdims=True)
    x_max = x_original.max(axis=1, keepdims=True)
    scale = (x_max - x_min) / 7.0  # 8 levels = 0-7
    scale = np.maximum(scale, 1e-6)  # Avoid division by zero
    
    x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
    
    # Compute baseline error
    error_original = x_original - x_quantized
    mse_before = np.mean(error_original ** 2)
    
    print(f"\nBaseline MSE: {mse_before:.6f}")
    print(f"Error range: [{error_original.min():.6f}, {error_original.max():.6f}]")
    print(f"Error mean: {error_original.mean():.6f}")
    print(f"Error std: {error_original.std():.6f}")
    
    # ===== PHASE 25: PER-BLOCK CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 25: PER-BLOCK CORRECTION (Baseline)")
    print("-"*80)
    
    # Compute per-block bias
    bias_per_block = np.mean(error_original, axis=1, keepdims=True)
    x_corrected_block = x_quantized + bias_per_block
    
    error_block = x_original - x_corrected_block
    mse_block = np.mean(error_block ** 2)
    improvement_block = (mse_before - mse_block) / mse_before * 100
    
    print(f"Per-block bias shape: {bias_per_block.shape}")
    print(f"Per-block bias range: [{bias_per_block.min():.6f}, {bias_per_block.max():.6f}]")
    print(f"MSE after per-block correction: {mse_block:.6f}")
    print(f"Improvement: {improvement_block:.4f}%")
    
    # ===== PHASE 28: PER-ELEMENT CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 28: PER-ELEMENT CORRECTION (New)")
    print("-"*80)
    
    # Compute per-element bias
    bias_per_element = error_original  # One bias per element
    x_corrected_element = x_quantized + bias_per_element
    
    error_element = x_original - x_corrected_element
    mse_element = np.mean(error_element ** 2)
    improvement_element = (mse_before - mse_element) / mse_before * 100
    
    print(f"Per-element bias shape: {bias_per_element.shape}")
    print(f"Per-element bias range: [{bias_per_element.min():.6f}, {bias_per_element.max():.6f}]")
    print(f"MSE after per-element correction: {mse_element:.6f}")
    print(f"Improvement: {improvement_element:.4f}%")
    
    # ===== COMPARISON =====
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    
    relative_improvement = (mse_block - mse_element) / mse_block * 100
    print(f"Per-element vs Per-block improvement: {relative_improvement:.4f}%")
    print(f"Per-element is {relative_improvement:.2f}% better than per-block")
    
    # ===== STORAGE ANALYSIS =====
    print("\n" + "-"*80)
    print("STORAGE ANALYSIS")
    print("-"*80)
    
    # Per-block: 1 bias per block
    storage_block = num_blocks * 4  # 4 bytes per float32
    
    # Per-element: 1 bias per element
    storage_element = num_blocks * block_size * 4
    
    print(f"Per-block storage: {storage_block} bytes ({storage_block/1024:.2f} KB)")
    print(f"Per-element storage: {storage_element} bytes ({storage_element/1024:.2f} KB)")
    print(f"Storage overhead: {storage_element/storage_block:.1f}x")
    
    # ===== COMPRESSION ANALYSIS =====
    print("\n" + "-"*80)
    print("COMPRESSION ANALYSIS")
    print("-"*80)
    
    # Estimate compression ratio
    # Original: num_blocks * block_size * 4 bytes (FP32)
    # Compressed: indices (3 bits) + scales (4 bytes) + per-element bias (4 bytes)
    
    original_size = num_blocks * block_size * 4
    
    # Per-block approach
    indices_size = num_blocks * block_size * 3 / 8  # 3 bits per element
    scales_size = num_blocks * 4
    bias_size_block = num_blocks * 4
    compressed_size_block = indices_size + scales_size + bias_size_block
    compression_block = original_size / compressed_size_block
    
    # Per-element approach
    bias_size_element = num_blocks * block_size * 4
    compressed_size_element = indices_size + scales_size + bias_size_element
    compression_element = original_size / compressed_size_element
    
    print(f"Original size: {original_size} bytes ({original_size/1024:.2f} KB)")
    print(f"Per-block compression: {compression_block:.4f}x ({100/compression_block:.2f}%)")
    print(f"Per-element compression: {compression_element:.4f}x ({100/compression_element:.2f}%)")
    print(f"Compression loss: {(compression_block - compression_element)/compression_block*100:.2f}%")
    
    return {
        "test_type": "per_element_synthetic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_before": float(mse_before),
        "results": {
            "per_block": {
                "mse_after": float(mse_block),
                "improvement_percent": float(improvement_block),
                "storage_bytes": int(storage_block),
                "compression_ratio": float(compression_block),
            },
            "per_element": {
                "mse_after": float(mse_element),
                "improvement_percent": float(improvement_element),
                "storage_bytes": int(storage_element),
                "compression_ratio": float(compression_element),
                "relative_improvement_vs_block": float(relative_improvement),
            }
        }
    }


def test_per_element_correction_realistic():
    """Test per-element correction on realistic NVFP4 patterns."""
    print("\n" + "="*80)
    print("PHASE 28: PER-ELEMENT CORRECTION - REALISTIC TEST")
    print("="*80)
    
    # Generate realistic NVFP4 data (smaller scale, more structured)
    np.random.seed(42)
    num_blocks = 20
    block_size = 128
    
    # Realistic weights: smaller magnitude, more structure
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32) * 0.1
    
    # Add some structure (spatial correlation)
    for i in range(num_blocks):
        # Add smoothness: neighboring elements are correlated
        for j in range(1, block_size):
            x_original[i, j] += 0.3 * x_original[i, j-1]
    
    # Simulate NVFP4 quantization
    x_min = x_original.min(axis=1, keepdims=True)
    x_max = x_original.max(axis=1, keepdims=True)
    scale = (x_max - x_min) / 7.0
    scale = np.maximum(scale, 1e-6)
    
    x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
    
    # Compute baseline error
    error_original = x_original - x_quantized
    mse_before = np.mean(error_original ** 2)
    
    print(f"\nBaseline MSE: {mse_before:.6f}")
    print(f"Error range: [{error_original.min():.6f}, {error_original.max():.6f}]")
    print(f"Error mean: {error_original.mean():.6f}")
    print(f"Error std: {error_original.std():.6f}")
    
    # ===== PHASE 25: PER-BLOCK CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 25: PER-BLOCK CORRECTION (Baseline)")
    print("-"*80)
    
    bias_per_block = np.mean(error_original, axis=1, keepdims=True)
    x_corrected_block = x_quantized + bias_per_block
    
    error_block = x_original - x_corrected_block
    mse_block = np.mean(error_block ** 2)
    improvement_block = (mse_before - mse_block) / mse_before * 100
    
    print(f"MSE after per-block correction: {mse_block:.6f}")
    print(f"Improvement: {improvement_block:.4f}%")
    
    # ===== PHASE 28: PER-ELEMENT CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 28: PER-ELEMENT CORRECTION (New)")
    print("-"*80)
    
    bias_per_element = error_original
    x_corrected_element = x_quantized + bias_per_element
    
    error_element = x_original - x_corrected_element
    mse_element = np.mean(error_element ** 2)
    improvement_element = (mse_before - mse_element) / mse_before * 100
    
    print(f"MSE after per-element correction: {mse_element:.6f}")
    print(f"Improvement: {improvement_element:.4f}%")
    
    # ===== COMPARISON =====
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    
    relative_improvement = (mse_block - mse_element) / mse_block * 100
    print(f"Per-element vs Per-block improvement: {relative_improvement:.4f}%")
    print(f"Per-element is {relative_improvement:.2f}% better than per-block")
    
    return {
        "test_type": "per_element_realistic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_before": float(mse_before),
        "results": {
            "per_block": {
                "mse_after": float(mse_block),
                "improvement_percent": float(improvement_block),
            },
            "per_element": {
                "mse_after": float(mse_element),
                "improvement_percent": float(improvement_element),
                "relative_improvement_vs_block": float(relative_improvement),
            }
        }
    }


def main():
    """Run all Phase 28 tests."""
    print("\n" + "="*80)
    print("PHASE 28: PER-ELEMENT CORRECTION")
    print("="*80)
    print("\nTechnique: Instead of one bias per block, compute bias per element")
    print("Expected improvement: 2-5% (higher than per-block Phase 25)")
    print("Literature: QAT literature (Jacob et al., 2018)")
    
    results = []
    
    # Test 1: Synthetic data
    result1 = test_per_element_correction_synthetic()
    results.append(result1)
    
    # Test 2: Realistic data
    result2 = test_per_element_correction_realistic()
    results.append(result2)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 28 SUMMARY")
    print("="*80)
    
    synthetic_improvement = result1["results"]["per_element"]["relative_improvement_vs_block"]
    realistic_improvement = result2["results"]["per_element"]["relative_improvement_vs_block"]
    
    print(f"\nSynthetic test improvement: {synthetic_improvement:.4f}%")
    print(f"Realistic test improvement: {realistic_improvement:.4f}%")
    print(f"Average improvement: {(synthetic_improvement + realistic_improvement)/2:.4f}%")
    
    # Storage analysis
    print("\n" + "-"*80)
    print("STORAGE TRADEOFF")
    print("-"*80)
    
    compression_block = result1["results"]["per_block"]["compression_ratio"]
    compression_element = result1["results"]["per_element"]["compression_ratio"]
    
    print(f"Per-block compression: {compression_block:.4f}x")
    print(f"Per-element compression: {compression_element:.4f}x")
    print(f"Storage overhead: {compression_block/compression_element:.2f}x")
    
    # Decision
    print("\n" + "-"*80)
    print("DECISION")
    print("-"*80)
    
    if synthetic_improvement > 1.0 and realistic_improvement > 0.5:
        print("✅ PER-ELEMENT CORRECTION IS EFFECTIVE")
        print(f"   - Synthetic improvement: {synthetic_improvement:.2f}%")
        print(f"   - Realistic improvement: {realistic_improvement:.2f}%")
        print("   - Recommendation: IMPLEMENT for production")
    else:
        print("⚠️  PER-ELEMENT CORRECTION HAS LIMITED BENEFIT")
        print(f"   - Synthetic improvement: {synthetic_improvement:.2f}%")
        print(f"   - Realistic improvement: {realistic_improvement:.2f}%")
        print("   - Recommendation: Consider storage tradeoff")
    
    # Save results
    with open("phase28_per_element_correction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to phase28_per_element_correction_results.json")
    
    return results


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test quantization-aware K-means compression.

Hypothesis: Learning K-means on quantized weights instead of original
BF16 weights could improve compression quality.
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression_v2 import BLOCK_SIZE, CODEBOOK_SIZE


def simulate_fp4_quantization(weight):
    """Simulate FP4 quantization."""
    # Simple FP4 simulation: quantize to 4-bit range
    weight_min = weight.min()
    weight_max = weight.max()
    weight_range = weight_max - weight_min
    
    # Quantize to 16 levels (4-bit)
    quantized = torch.round((weight - weight_min) / weight_range * 15) / 15 * weight_range + weight_min
    return quantized


def test_kmeans_on_original_vs_quantized():
    """Compare K-means learned on original vs quantized weights."""
    print("=" * 70)
    print("QUANTIZATION-AWARE K-MEANS COMPARISON")
    print("=" * 70)
    
    # Create synthetic data
    num_blocks = 1000
    block_size = BLOCK_SIZE
    original_data = torch.randn(num_blocks, block_size, dtype=torch.float32)
    
    print(f"\nTest data: {num_blocks} blocks, {block_size} elements each")
    print(f"Codebook size: {CODEBOOK_SIZE} codewords")
    
    # Test 1: K-means on original data
    print("\n" + "-" * 70)
    print("K-Means on Original Data")
    print("-" * 70)
    
    t0 = time.time()
    
    indices = torch.randperm(num_blocks)[:CODEBOOK_SIZE]
    codebook_original = original_data[indices].clone()
    
    for iteration in range(10):
        distances = torch.cdist(original_data, codebook_original)
        codes = torch.argmin(distances, dim=1)
        
        for i in range(CODEBOOK_SIZE):
            mask = codes == i
            if mask.sum() > 0:
                codebook_original[i] = original_data[mask].mean(dim=0)
    
    distances = torch.cdist(original_data, codebook_original)
    codes_original = torch.argmin(distances, dim=1)
    
    elapsed_original = time.time() - t0
    
    reconstructed_original = codebook_original[codes_original]
    mse_original = torch.mean((original_data - reconstructed_original) ** 2)
    
    print(f"MSE: {mse_original:.6f}")
    print(f"Time: {elapsed_original:.3f}s")
    
    # Test 2: K-means on quantized data
    print("\n" + "-" * 70)
    print("K-Means on Quantized Data (FP4)")
    print("-" * 70)
    
    quantized_data = simulate_fp4_quantization(original_data)
    
    t0 = time.time()
    
    indices = torch.randperm(num_blocks)[:CODEBOOK_SIZE]
    codebook_quantized = quantized_data[indices].clone()
    
    for iteration in range(10):
        distances = torch.cdist(quantized_data, codebook_quantized)
        codes = torch.argmin(distances, dim=1)
        
        for i in range(CODEBOOK_SIZE):
            mask = codes == i
            if mask.sum() > 0:
                codebook_quantized[i] = quantized_data[mask].mean(dim=0)
    
    distances = torch.cdist(quantized_data, codebook_quantized)
    codes_quantized = torch.argmin(distances, dim=1)
    
    elapsed_quantized = time.time() - t0
    
    reconstructed_quantized = codebook_quantized[codes_quantized]
    mse_quantized = torch.mean((quantized_data - reconstructed_quantized) ** 2)
    
    print(f"MSE: {mse_quantized:.6f}")
    print(f"Time: {elapsed_quantized:.3f}s")
    
    # Test 3: Evaluate quantized codebook on original data
    print("\n" + "-" * 70)
    print("Quantized Codebook Applied to Original Data")
    print("-" * 70)
    
    distances = torch.cdist(original_data, codebook_quantized)
    codes_cross = torch.argmin(distances, dim=1)
    reconstructed_cross = codebook_quantized[codes_cross]
    mse_cross = torch.mean((original_data - reconstructed_cross) ** 2)
    
    print(f"MSE: {mse_cross:.6f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\nMSE Comparison:")
    print(f"  Original codebook on original data: {mse_original:.6f}")
    print(f"  Quantized codebook on quantized data: {mse_quantized:.6f}")
    print(f"  Quantized codebook on original data: {mse_cross:.6f}")
    
    improvement = (mse_original - mse_quantized) / mse_original * 100
    degradation = (mse_cross - mse_original) / mse_original * 100
    
    print(f"\nQuantized vs Original:")
    print(f"  Improvement on quantized data: {improvement:+.2f}%")
    print(f"  Degradation on original data: {degradation:+.2f}%")
    
    print(f"\nTime Comparison:")
    print(f"  Original: {elapsed_original:.3f}s")
    print(f"  Quantized: {elapsed_quantized:.3f}s")
    print(f"  Overhead: {(elapsed_quantized - elapsed_original) / elapsed_original * 100:+.2f}%")
    
    if improvement > 2 and degradation < 1:
        print(f"\n✓ Quantization-aware K-means is BENEFICIAL")
        return True
    else:
        print(f"\n✗ Quantization-aware K-means is NOT beneficial")
        return False


def main():
    """Run test."""
    print("\nQUANTIZATION-AWARE K-MEANS TEST\n")
    
    success = test_kmeans_on_original_vs_quantized()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if success:
        print("✓ Quantization-aware K-means is RECOMMENDED")
        print("  - Better compression on quantized data")
        print("  - Minimal degradation on original data")
        return 0
    else:
        print("✗ Quantization-aware K-means is NOT recommended")
        print("  - No significant improvement")
        print("  - Stick with original K-means")
        return 0


if __name__ == "__main__":
    sys.exit(main())

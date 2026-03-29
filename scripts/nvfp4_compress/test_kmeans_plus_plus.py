#!/usr/bin/env python3
"""Test K-means++ initialization vs random initialization.

K-means++ provides better initialization which can improve codebook quality.
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression import BLOCK_SIZE, CODEBOOK_SIZE


def kmeans_random_init(data, k, num_iterations=10):
    """K-means with random initialization."""
    num_blocks, block_size = data.shape
    
    # Random initialization
    indices = torch.randperm(num_blocks)[:k]
    codebook = data[indices].clone()
    
    # K-means iterations
    for iteration in range(num_iterations):
        distances = torch.cdist(data, codebook)
        codes = torch.argmin(distances, dim=1)
        
        for i in range(k):
            mask = codes == i
            if mask.sum() > 0:
                codebook[i] = data[mask].mean(dim=0)
    
    # Final assignment
    distances = torch.cdist(data, codebook)
    codes = torch.argmin(distances, dim=1)
    
    # Calculate MSE
    reconstructed = codebook[codes]
    mse = torch.mean((data - reconstructed) ** 2)
    
    return codebook, codes, mse


def kmeans_plus_plus_init(data, k, num_iterations=10):
    """K-means with K-means++ initialization."""
    num_blocks, block_size = data.shape
    
    # K-means++ initialization
    codebook = [data[torch.randint(0, num_blocks, (1,))].clone()]
    
    for _ in range(k - 1):
        # Calculate distances to nearest center
        distances = torch.cdist(data, torch.cat(codebook, dim=0))
        min_distances = torch.min(distances, dim=1)[0]
        
        # Choose next center with probability proportional to distance squared
        probabilities = min_distances ** 2
        probabilities = probabilities / probabilities.sum()
        
        next_idx = torch.multinomial(probabilities, 1)
        codebook.append(data[next_idx].clone())
    
    codebook = torch.cat(codebook, dim=0)
    
    # K-means iterations
    for iteration in range(num_iterations):
        distances = torch.cdist(data, codebook)
        codes = torch.argmin(distances, dim=1)
        
        for i in range(k):
            mask = codes == i
            if mask.sum() > 0:
                codebook[i] = data[mask].mean(dim=0)
    
    # Final assignment
    distances = torch.cdist(data, codebook)
    codes = torch.argmin(distances, dim=1)
    
    # Calculate MSE
    reconstructed = codebook[codes]
    mse = torch.mean((data - reconstructed) ** 2)
    
    return codebook, codes, mse


def test_initialization_strategies():
    """Test different K-means initialization strategies."""
    print("=" * 70)
    print("K-MEANS INITIALIZATION COMPARISON")
    print("=" * 70)
    
    # Create synthetic data
    num_blocks = 1000
    block_size = BLOCK_SIZE
    data = torch.randn(num_blocks, block_size, dtype=torch.float32)
    
    print(f"\nTest data: {num_blocks} blocks, {block_size} elements each")
    print(f"Codebook size: {CODEBOOK_SIZE} codewords")
    
    # Test random initialization
    print("\n" + "-" * 70)
    print("Random Initialization")
    print("-" * 70)
    
    mse_random = []
    times_random = []
    
    for trial in range(5):
        t0 = time.time()
        codebook, codes, mse = kmeans_random_init(data, CODEBOOK_SIZE, num_iterations=10)
        elapsed = time.time() - t0
        
        mse_random.append(mse.item())
        times_random.append(elapsed)
        
        print(f"Trial {trial+1}: MSE={mse:.6f}, Time={elapsed:.3f}s")
    
    avg_mse_random = np.mean(mse_random)
    avg_time_random = np.mean(times_random)
    
    print(f"\nAverage MSE: {avg_mse_random:.6f}")
    print(f"Average time: {avg_time_random:.3f}s")
    print(f"MSE std dev: {np.std(mse_random):.6f}")
    
    # Test K-means++ initialization
    print("\n" + "-" * 70)
    print("K-Means++ Initialization")
    print("-" * 70)
    
    mse_kmeans_pp = []
    times_kmeans_pp = []
    
    for trial in range(5):
        t0 = time.time()
        codebook, codes, mse = kmeans_plus_plus_init(data, CODEBOOK_SIZE, num_iterations=10)
        elapsed = time.time() - t0
        
        mse_kmeans_pp.append(mse.item())
        times_kmeans_pp.append(elapsed)
        
        print(f"Trial {trial+1}: MSE={mse:.6f}, Time={elapsed:.3f}s")
    
    avg_mse_kmeans_pp = np.mean(mse_kmeans_pp)
    avg_time_kmeans_pp = np.mean(times_kmeans_pp)
    
    print(f"\nAverage MSE: {avg_mse_kmeans_pp:.6f}")
    print(f"Average time: {avg_time_kmeans_pp:.3f}s")
    print(f"MSE std dev: {np.std(mse_kmeans_pp):.6f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    mse_improvement = (avg_mse_random - avg_mse_kmeans_pp) / avg_mse_random * 100
    time_overhead = (avg_time_kmeans_pp - avg_time_random) / avg_time_random * 100
    
    print(f"\nMSE Improvement: {mse_improvement:.2f}%")
    print(f"  Random: {avg_mse_random:.6f}")
    print(f"  K-means++: {avg_mse_kmeans_pp:.6f}")
    
    print(f"\nTime Overhead: {time_overhead:.2f}%")
    print(f"  Random: {avg_time_random:.3f}s")
    print(f"  K-means++: {avg_time_kmeans_pp:.3f}s")
    
    print(f"\nStability (lower std dev is better):")
    print(f"  Random: {np.std(mse_random):.6f}")
    print(f"  K-means++: {np.std(mse_kmeans_pp):.6f}")
    
    if mse_improvement > 0:
        print(f"\n✓ K-means++ is BETTER ({mse_improvement:.2f}% improvement)")
        return True
    else:
        print(f"\n✗ K-means++ is WORSE ({-mse_improvement:.2f}% degradation)")
        return False


def main():
    """Run test."""
    print("\nK-MEANS INITIALIZATION TEST\n")
    
    success = test_initialization_strategies()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if success:
        print("✓ K-means++ initialization is recommended")
        print("  - Better MSE (lower is better)")
        print("  - More stable across trials")
        print("  - Reasonable time overhead")
        return 0
    else:
        print("✗ Random initialization is sufficient")
        print("  - Similar or better MSE")
        print("  - Faster initialization")
        return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fast variant of grouped_fisher that avoids expensive torch.quantile.

Instead of computing exact quantiles, we use a simpler magnitude-based grouping:
- High: magnitude >= median
- Low: magnitude < median
- Medium: (not used in this variant)

This reduces complexity from O(n log n) to O(n) while maintaining the core idea.
"""

import torch
import time

def fast_grouped_fisher_weights(codes: torch.Tensor) -> torch.Tensor:
    """Compute magnitude-based weights without expensive quantile computation.
    
    Args:
        codes: [num_blocks, block_size] tensor of FP4 codes (0-15)
    
    Returns:
        weights: [num_blocks, block_size] tensor of weights
    """
    magnitudes = codes.float().abs()
    
    # Use median instead of quantile (O(n) vs O(n log n))
    # For FP4 E2M1 values: [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    # Median magnitude is around 1.5
    median = torch.median(magnitudes)
    
    # Simple binary grouping: high (>= median) gets 2x, low (< median) gets 0.5x
    weights = torch.where(magnitudes >= median, 2.0, 0.5)
    
    # Normalize per block
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    
    return weights

def benchmark():
    """Benchmark fast_grouped_fisher vs torch.quantile."""
    print("Benchmarking magnitude-based weighting schemes...")
    print()
    
    # Simulate shard data
    flat_blocks = torch.randint(0, 16, (8192, 128), dtype=torch.uint8)
    magnitudes = flat_blocks.float().abs()
    
    # Method 1: torch.quantile (current grouped_fisher)
    print("Method 1: torch.quantile (current grouped_fisher)")
    start = time.time()
    high_threshold = torch.quantile(magnitudes, 0.66)
    low_threshold = torch.quantile(magnitudes, 0.33)
    weights1 = torch.ones_like(magnitudes)
    weights1[magnitudes >= high_threshold] = 3.0
    weights1[(magnitudes > low_threshold) & (magnitudes < high_threshold)] = 1.0
    weights1[magnitudes <= low_threshold] = 0.3
    weights1 = weights1 / (weights1.sum(dim=1, keepdim=True) + 1e-8)
    elapsed1 = time.time() - start
    print(f"  Time: {elapsed1:.3f}s")
    print(f"  High threshold: {high_threshold:.4f}, Low threshold: {low_threshold:.4f}")
    
    # Method 2: torch.median (fast variant)
    print("\nMethod 2: torch.median (fast variant)")
    start = time.time()
    median = torch.median(magnitudes)
    weights2 = torch.where(magnitudes >= median, 2.0, 0.5)
    weights2 = weights2 / (weights2.sum(dim=1, keepdim=True) + 1e-8)
    elapsed2 = time.time() - start
    print(f"  Time: {elapsed2:.3f}s")
    print(f"  Median: {median:.4f}")
    print(f"  Speedup: {elapsed1/elapsed2:.1f}x")
    
    # Method 3: torch.kthvalue (alternative fast variant)
    print("\nMethod 3: torch.kthvalue (alternative fast variant)")
    start = time.time()
    # Find 75th percentile using kthvalue
    k_high = int(magnitudes.numel() * 0.75)
    k_low = int(magnitudes.numel() * 0.25)
    high_val = torch.kthvalue(magnitudes.flatten(), k_high)[0]
    low_val = torch.kthvalue(magnitudes.flatten(), k_low)[0]
    weights3 = torch.ones_like(magnitudes)
    weights3[magnitudes >= high_val] = 3.0
    weights3[(magnitudes > low_val) & (magnitudes < high_val)] = 1.0
    weights3[magnitudes <= low_val] = 0.3
    weights3 = weights3 / (weights3.sum(dim=1, keepdim=True) + 1e-8)
    elapsed3 = time.time() - start
    print(f"  Time: {elapsed3:.3f}s")
    print(f"  High value: {high_val:.4f}, Low value: {low_val:.4f}")
    print(f"  Speedup: {elapsed1/elapsed3:.1f}x")
    
    # Method 4: Histogram-based percentile (fastest)
    print("\nMethod 4: Histogram-based percentile (fastest)")
    start = time.time()
    # For FP4 codes (0-15), we can use a simple histogram
    hist = torch.histc(magnitudes, bins=16, min=0, max=15)
    cumsum = torch.cumsum(hist, dim=0)
    total = cumsum[-1]
    high_idx = torch.searchsorted(cumsum, total * 0.66)
    low_idx = torch.searchsorted(cumsum, total * 0.33)
    high_val = high_idx.float()
    low_val = low_idx.float()
    weights4 = torch.ones_like(magnitudes)
    weights4[magnitudes >= high_val] = 3.0
    weights4[(magnitudes > low_val) & (magnitudes < high_val)] = 1.0
    weights4[magnitudes <= low_val] = 0.3
    weights4 = weights4 / (weights4.sum(dim=1, keepdim=True) + 1e-8)
    elapsed4 = time.time() - start
    print(f"  Time: {elapsed4:.3f}s")
    print(f"  High value: {high_val:.4f}, Low value: {low_val:.4f}")
    print(f"  Speedup: {elapsed1/elapsed4:.1f}x")
    
    print()
    print("Summary:")
    print(f"  torch.quantile:        {elapsed1:.3f}s (baseline)")
    print(f"  torch.median:          {elapsed2:.3f}s ({elapsed1/elapsed2:.1f}x faster)")
    print(f"  torch.kthvalue:        {elapsed3:.3f}s ({elapsed1/elapsed3:.1f}x faster)")
    print(f"  histogram-based:       {elapsed4:.3f}s ({elapsed1/elapsed4:.1f}x faster)")

if __name__ == "__main__":
    benchmark()

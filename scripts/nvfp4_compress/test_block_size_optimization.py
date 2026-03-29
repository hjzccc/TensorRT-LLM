#!/usr/bin/env python3
"""Test different block sizes for K-means compression.

Block size affects compression ratio and quality.
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression import CODEBOOK_SIZE


def test_block_sizes():
    """Test different block sizes."""
    print("=" * 70)
    print("BLOCK SIZE OPTIMIZATION")
    print("=" * 70)
    
    # Create synthetic data
    num_elements = 16384
    data = torch.randn(num_elements, dtype=torch.float32)
    
    print(f"\nTest data: {num_elements} elements")
    print(f"Codebook size: {CODEBOOK_SIZE} codewords")
    
    block_sizes = [8, 16, 32, 64]
    results = {}
    
    for block_size in block_sizes:
        print(f"\n" + "-" * 70)
        print(f"Block Size: {block_size}")
        print("-" * 70)
        
        # Reshape data into blocks
        num_blocks = num_elements // block_size
        blocks = data[:num_blocks * block_size].reshape(num_blocks, block_size)
        
        # K-means
        t0 = time.time()
        
        # Random initialization
        indices = torch.randperm(num_blocks)[:CODEBOOK_SIZE]
        codebook = blocks[indices].clone()
        
        # K-means iterations
        for iteration in range(10):
            distances = torch.cdist(blocks, codebook)
            codes = torch.argmin(distances, dim=1)
            
            for i in range(CODEBOOK_SIZE):
                mask = codes == i
                if mask.sum() > 0:
                    codebook[i] = blocks[mask].mean(dim=0)
        
        # Final assignment
        distances = torch.cdist(blocks, codebook)
        codes = torch.argmin(distances, dim=1)
        
        elapsed = time.time() - t0
        
        # Calculate MSE
        reconstructed = codebook[codes]
        mse = torch.mean((blocks - reconstructed) ** 2)
        
        # Calculate compression metrics
        original_bits = 32  # float32
        codes_bits = 3  # 3-bit codes
        codebook_bits = codebook.numel() * 32 / num_elements
        total_bits = codes_bits + codebook_bits
        compression_ratio = original_bits / total_bits
        
        results[block_size] = {
            "mse": mse.item(),
            "time": elapsed,
            "compression_ratio": compression_ratio,
            "codes_bits": codes_bits,
            "codebook_bits": codebook_bits,
            "total_bits": total_bits
        }
        
        print(f"MSE: {mse:.6f}")
        print(f"Time: {elapsed:.3f}s")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"  Codes: {codes_bits:.2f} bits/elem")
        print(f"  Codebook: {codebook_bits:.2f} bits/elem")
        print(f"  Total: {total_bits:.2f} bits/elem")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Block Size':>12} {'MSE':>12} {'Compression':>12} {'Time':>10}")
    print("-" * 70)
    
    for block_size in block_sizes:
        r = results[block_size]
        print(f"{block_size:>12} {r['mse']:>12.6f} {r['compression_ratio']:>12.2f}x {r['time']:>10.3f}s")
    
    # Find best
    best_mse = min(results.items(), key=lambda x: x[1]['mse'])
    best_compression = max(results.items(), key=lambda x: x[1]['compression_ratio'])
    
    print(f"\nBest MSE: Block size {best_mse[0]} ({best_mse[1]['mse']:.6f})")
    print(f"Best compression: Block size {best_compression[0]} ({best_compression[1]['compression_ratio']:.2f}x)")
    
    return results


def main():
    """Run test."""
    print("\nBLOCK SIZE OPTIMIZATION TEST\n")
    
    results = test_block_sizes()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Current block size is 16
    current_mse = results[16]['mse']
    current_compression = results[16]['compression_ratio']
    
    print(f"\nCurrent block size (16):")
    print(f"  MSE: {current_mse:.6f}")
    print(f"  Compression: {current_compression:.2f}x")
    
    # Check if other sizes are better
    improvements = []
    for block_size, r in results.items():
        if block_size != 16:
            mse_diff = (current_mse - r['mse']) / current_mse * 100
            comp_diff = (r['compression_ratio'] - current_compression) / current_compression * 100
            
            if mse_diff > 1 or comp_diff > 1:
                improvements.append((block_size, mse_diff, comp_diff))
    
    if improvements:
        print(f"\n✓ Better block sizes found:")
        for block_size, mse_diff, comp_diff in improvements:
            print(f"  Block size {block_size}: MSE {mse_diff:+.2f}%, Compression {comp_diff:+.2f}%")
    else:
        print(f"\n✗ Block size 16 is optimal")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Phase 3: Fast per-block codebook analysis (no exhaustive search).

Uses greedy/heuristic approaches instead of exhaustive search for speed.
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter

import torch
import numpy as np

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def find_greedy_codebook(codes: torch.Tensor, k: int) -> list[int]:
    """Find K-element codebook using greedy approach.
    
    Greedy: Start with most frequent codes, then add codes that minimize MSE.
    """
    code_counts = Counter(codes.tolist())
    
    # Start with most frequent codes
    sorted_codes = sorted(code_counts.items(), key=lambda x: -x[1])
    codebook = [code for code, _ in sorted_codes[:k]]
    
    return codebook


def compute_mse_fast(codes: torch.Tensor, codebook: list[int]) -> float:
    """Fast MSE computation."""
    codebook_values = E2M1_TABLE[codebook]
    
    mse = 0.0
    for code in codes:
        src_val = E2M1_TABLE[code.item()].item()
        dists = (codebook_values - src_val).abs()
        nearest_val = codebook_values[dists.argmin()].item()
        mse += (src_val - nearest_val) ** 2
    
    return mse / len(codes)


def analyze_distribution_fast(codes: torch.Tensor, block_size: int = 16) -> dict:
    """Fast analysis without exhaustive search."""
    num_blocks = len(codes) // block_size
    
    stats = {
        "total_codes": len(codes),
        "num_blocks": num_blocks,
        "block_size": block_size,
        "unique_codes_per_block": [],
        "entropy_per_block": [],
        "mse_3bit": [],
        "mse_2bit": [],
    }
    
    for block_idx in range(min(num_blocks, 100)):  # Sample first 100 blocks
        start = block_idx * block_size
        end = start + block_size
        block_codes = codes[start:end]
        
        # Unique codes
        unique = len(torch.unique(block_codes))
        stats["unique_codes_per_block"].append(unique)
        
        # Entropy
        counts = torch.bincount(block_codes, minlength=16)
        probs = counts[counts > 0].float() / len(block_codes)
        entropy = -(probs * torch.log2(probs)).sum().item()
        stats["entropy_per_block"].append(entropy)
        
        # Greedy codebooks
        codebook_3bit = find_greedy_codebook(block_codes, 8)
        mse_3bit = compute_mse_fast(block_codes, codebook_3bit)
        stats["mse_3bit"].append(mse_3bit)
        
        codebook_2bit = find_greedy_codebook(block_codes, 4)
        mse_2bit = compute_mse_fast(block_codes, codebook_2bit)
        stats["mse_2bit"].append(mse_2bit)
    
    stats["mean_unique_codes"] = np.mean(stats["unique_codes_per_block"])
    stats["mean_entropy"] = np.mean(stats["entropy_per_block"])
    stats["mean_mse_3bit"] = np.mean(stats["mse_3bit"])
    stats["mean_mse_2bit"] = np.mean(stats["mse_2bit"])
    
    return stats


def main():
    print("Phase 3: Fast Codebook Analysis")
    print("=" * 60)
    
    # Generate synthetic codes
    print("\nGenerating synthetic FP4 codes...")
    np.random.seed(42)
    codes_list = []
    
    for _ in range(1000):
        block = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=12, p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05])
        block = np.concatenate([block, np.random.choice([8, 9, 10, 11, 12, 13, 14, 15], size=4)])
        codes_list.extend(block)
    
    codes = torch.tensor(codes_list, dtype=torch.long)
    print(f"Generated {len(codes)} codes")
    
    # Analyze
    print("\nAnalyzing distribution...")
    start_time = time.time()
    stats = analyze_distribution_fast(codes)
    elapsed = time.time() - start_time
    
    print(f"Analysis completed in {elapsed:.2f}s")
    print(f"  Mean unique codes/block: {stats['mean_unique_codes']:.2f}")
    print(f"  Mean entropy: {stats['mean_entropy']:.3f} bits")
    print(f"  Mean MSE (3-bit): {stats['mean_mse_3bit']:.6f}")
    print(f"  Mean MSE (2-bit): {stats['mean_mse_2bit']:.6f}")
    
    # Estimate compression
    print("\nCompression Estimates:")
    print("-" * 60)
    
    # 3-bit: 3 bits/code + 0.5 bits/block overhead = 3.03 bits/elem
    bits_3bit = 3.0 + 0.5/16
    print(f"3-bit: {bits_3bit:.3f} bits/elem (3 bits code + 0.5 bits overhead)")
    
    # 2-bit: 2 bits/code + 0.5 bits/block overhead = 2.03 bits/elem
    bits_2bit = 2.0 + 0.5/16
    print(f"2-bit: {bits_2bit:.3f} bits/elem (2 bits code + 0.5 bits overhead)")
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase3_fast_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "analysis": {
                "mean_unique_codes": stats["mean_unique_codes"],
                "mean_entropy": stats["mean_entropy"],
                "mean_mse_3bit": stats["mean_mse_3bit"],
                "mean_mse_2bit": stats["mean_mse_2bit"],
            },
            "compression_estimates": {
                "3bit": {"bits_per_elem": bits_3bit, "mse": stats["mean_mse_3bit"]},
                "2bit": {"bits_per_elem": bits_2bit, "mse": stats["mean_mse_2bit"]},
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

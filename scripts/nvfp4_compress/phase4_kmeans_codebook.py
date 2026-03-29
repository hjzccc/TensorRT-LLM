#!/usr/bin/env python3
"""Phase 4: K-Means Codebook Learning.

Uses K-means clustering to find optimal codebooks for FP4 codes.
This is a proven approach from AQLM and other papers.
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from sklearn.cluster import KMeans

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def codes_to_values(codes: torch.Tensor) -> np.ndarray:
    """Convert FP4 codes to float values."""
    return E2M1_TABLE[codes].numpy().reshape(-1, 1)


def values_to_nearest_codes(values: np.ndarray) -> list[int]:
    """Convert float values to nearest FP4 codes."""
    codes = []
    for val in values.flatten():
        dists = np.abs(E2M1_TABLE.numpy() - val)
        code = np.argmin(dists)
        codes.append(int(code))
    return codes


def kmeans_codebook(codes: torch.Tensor, k: int, max_iter: int = 100) -> tuple[list[int], float]:
    """Find K-means codebook for a block of codes.
    
    Args:
        codes: [N] tensor of FP4 codes
        k: number of clusters
        max_iter: max iterations for K-means
    
    Returns:
        (codebook_codes, mse)
    """
    # Convert codes to values
    values = codes_to_values(codes)
    
    # Run K-means
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=42)
    kmeans.fit(values)
    
    # Convert cluster centers back to nearest FP4 codes
    codebook_codes = values_to_nearest_codes(kmeans.cluster_centers_)
    
    # Compute MSE
    labels = kmeans.labels_
    mse = 0.0
    for i, code in enumerate(codes):
        src_val = E2M1_TABLE[code.item()].item()
        cluster_center = kmeans.cluster_centers_[labels[i], 0]
        mse += (src_val - cluster_center) ** 2
    mse /= len(codes)
    
    return codebook_codes, mse


def analyze_with_kmeans(codes: torch.Tensor, block_size: int = 16) -> dict:
    """Analyze FP4 codes using K-means codebooks."""
    num_blocks = len(codes) // block_size
    
    stats = {
        "total_codes": len(codes),
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_3bit_kmeans": [],
        "mse_2bit_kmeans": [],
        "mse_3bit_greedy": [],
        "mse_2bit_greedy": [],
    }
    
    for block_idx in range(min(num_blocks, 100)):  # Sample first 100 blocks
        start = block_idx * block_size
        end = start + block_size
        block_codes = codes[start:end]
        
        # K-means codebooks
        _, mse_3bit_km = kmeans_codebook(block_codes, 8)
        stats["mse_3bit_kmeans"].append(mse_3bit_km)
        
        _, mse_2bit_km = kmeans_codebook(block_codes, 4)
        stats["mse_2bit_kmeans"].append(mse_2bit_km)
        
        # Greedy codebooks (for comparison)
        code_counts = Counter(block_codes.tolist())
        greedy_3bit = [code for code, _ in code_counts.most_common(8)]
        greedy_2bit = [code for code, _ in code_counts.most_common(4)]
        
        # Compute MSE for greedy
        mse_3bit_greedy = 0.0
        for code in block_codes:
            src_val = E2M1_TABLE[code.item()].item()
            dists = np.abs(E2M1_TABLE[greedy_3bit].numpy() - src_val)
            nearest_val = E2M1_TABLE[greedy_3bit[np.argmin(dists)]].item()
            mse_3bit_greedy += (src_val - nearest_val) ** 2
        mse_3bit_greedy /= len(block_codes)
        stats["mse_3bit_greedy"].append(mse_3bit_greedy)
        
        mse_2bit_greedy = 0.0
        for code in block_codes:
            src_val = E2M1_TABLE[code.item()].item()
            dists = np.abs(E2M1_TABLE[greedy_2bit].numpy() - src_val)
            nearest_val = E2M1_TABLE[greedy_2bit[np.argmin(dists)]].item()
            mse_2bit_greedy += (src_val - nearest_val) ** 2
        mse_2bit_greedy /= len(block_codes)
        stats["mse_2bit_greedy"].append(mse_2bit_greedy)
    
    # Aggregate statistics
    stats["mean_mse_3bit_kmeans"] = np.mean(stats["mse_3bit_kmeans"])
    stats["mean_mse_2bit_kmeans"] = np.mean(stats["mse_2bit_kmeans"])
    stats["mean_mse_3bit_greedy"] = np.mean(stats["mse_3bit_greedy"])
    stats["mean_mse_2bit_greedy"] = np.mean(stats["mse_2bit_greedy"])
    
    # Improvement
    stats["improvement_3bit"] = (stats["mean_mse_3bit_greedy"] - stats["mean_mse_3bit_kmeans"]) / stats["mean_mse_3bit_greedy"] * 100
    stats["improvement_2bit"] = (stats["mean_mse_2bit_greedy"] - stats["mean_mse_2bit_kmeans"]) / stats["mean_mse_2bit_greedy"] * 100
    
    return stats


def main():
    print("Phase 4: K-Means Codebook Learning")
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
    
    # Analyze with K-means
    print("\nAnalyzing with K-means...")
    start_time = time.time()
    stats = analyze_with_kmeans(codes)
    elapsed = time.time() - start_time
    
    print(f"Analysis completed in {elapsed:.2f}s")
    
    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON: K-Means vs Greedy")
    print("=" * 60)
    
    print("\n3-BIT CODEBOOK (8 codes):")
    print(f"  Greedy MSE:  {stats['mean_mse_3bit_greedy']:.6f}")
    print(f"  K-Means MSE: {stats['mean_mse_3bit_kmeans']:.6f}")
    print(f"  Improvement: {stats['improvement_3bit']:.1f}%")
    
    print("\n2-BIT CODEBOOK (4 codes):")
    print(f"  Greedy MSE:  {stats['mean_mse_2bit_greedy']:.6f}")
    print(f"  K-Means MSE: {stats['mean_mse_2bit_kmeans']:.6f}")
    print(f"  Improvement: {stats['improvement_2bit']:.1f}%")
    
    # Compression estimates
    print("\n" + "=" * 60)
    print("COMPRESSION ESTIMATES")
    print("=" * 60)
    
    bits_3bit = 3.0 + 0.5/16
    bits_2bit = 2.0 + 0.5/16
    
    print(f"\n3-bit: {bits_3bit:.3f} bits/elem")
    print(f"  MSE (K-means): {stats['mean_mse_3bit_kmeans']:.6f}")
    print(f"  MSE (Greedy):  {stats['mean_mse_3bit_greedy']:.6f}")
    
    print(f"\n2-bit: {bits_2bit:.3f} bits/elem")
    print(f"  MSE (K-means): {stats['mean_mse_2bit_kmeans']:.6f}")
    print(f"  MSE (Greedy):  {stats['mean_mse_2bit_greedy']:.6f}")
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase4_kmeans_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "kmeans_analysis": {
                "mean_mse_3bit_kmeans": stats["mean_mse_3bit_kmeans"],
                "mean_mse_2bit_kmeans": stats["mean_mse_2bit_kmeans"],
                "mean_mse_3bit_greedy": stats["mean_mse_3bit_greedy"],
                "mean_mse_2bit_greedy": stats["mean_mse_2bit_greedy"],
                "improvement_3bit_percent": stats["improvement_3bit"],
                "improvement_2bit_percent": stats["improvement_2bit"],
            },
            "compression_estimates": {
                "3bit": {
                    "bits_per_elem": bits_3bit,
                    "mse_kmeans": stats["mean_mse_3bit_kmeans"],
                    "mse_greedy": stats["mean_mse_3bit_greedy"],
                },
                "2bit": {
                    "bits_per_elem": bits_2bit,
                    "mse_kmeans": stats["mean_mse_2bit_kmeans"],
                    "mse_greedy": stats["mean_mse_2bit_greedy"],
                }
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

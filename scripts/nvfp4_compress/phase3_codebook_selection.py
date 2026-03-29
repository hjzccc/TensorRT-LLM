#!/usr/bin/env python3
"""Phase 3: Per-block optimal codebook selection.

This script implements per-block optimal codebook selection without requiring
the full evaluation pipeline. It analyzes the FP4 code distribution and finds
the best K-element subset for each block.

Key insight: Instead of running full PPL evaluation, we can:
1. Analyze FP4 code distribution per block
2. Find optimal K-element subsets using exhaustive search or clustering
3. Build a library of codebooks
4. Estimate compression ratio and overhead

This is a pure algorithmic approach that doesn't require docker runtime.
"""

import sys
import json
import time
from pathlib import Path
from itertools import combinations
from collections import Counter

import torch
import numpy as np

sys.path.insert(0, "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress")

# E2M1 code → float value (16 entries, codes 0-15)
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # codes 0-7  (positive)
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # codes 8-15 (negative)
], dtype=torch.float32)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] uint8 → [M, K] uint8 with values 0-15."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def compute_mse_for_codebook(codes: torch.Tensor, codebook_indices: list[int]) -> float:
    """Compute MSE when mapping codes to a sub-codebook.
    
    Args:
        codes: [N] tensor of FP4 codes (0-15)
        codebook_indices: list of code indices in the sub-codebook
    
    Returns:
        MSE of mapping all codes to nearest codebook code
    """
    codebook_values = E2M1_TABLE[codebook_indices]
    
    mse = 0.0
    for code in codes:
        src_val = E2M1_TABLE[code.item()].item()
        # Find nearest codebook value
        dists = (codebook_values - src_val).abs()
        nearest_val = codebook_values[dists.argmin()].item()
        mse += (src_val - nearest_val) ** 2
    
    return mse / len(codes)


def find_optimal_codebook(codes: torch.Tensor, k: int, max_candidates: int = 1000) -> tuple[list[int], float]:
    """Find optimal K-element codebook for a block of codes.
    
    Uses exhaustive search for small K, or greedy approach for large K.
    
    Args:
        codes: [N] tensor of FP4 codes
        k: number of codes in sub-codebook
        max_candidates: max candidates to evaluate (for performance)
    
    Returns:
        (codebook_indices, mse)
    """
    # Get unique codes in this block
    unique_codes = torch.unique(codes).tolist()
    
    # If block has <= k unique codes, use all of them
    if len(unique_codes) <= k:
        mse = compute_mse_for_codebook(codes, unique_codes)
        return unique_codes, mse
    
    # For small k, use exhaustive search
    if k <= 4:
        best_codebook = None
        best_mse = float('inf')
        
        # Enumerate all C(15, k) combinations
        for combo in combinations(range(15), k):  # Exclude code 8 (negative zero)
            mse = compute_mse_for_codebook(codes, list(combo))
            if mse < best_mse:
                best_mse = mse
                best_codebook = list(combo)
        
        return best_codebook, best_mse
    
    # For larger k, use greedy approach
    # Start with most frequent codes
    code_counts = Counter(codes.tolist())
    sorted_codes = [code for code, _ in code_counts.most_common(k)]
    
    mse = compute_mse_for_codebook(codes, sorted_codes)
    return sorted_codes, mse


def analyze_block_distribution(codes: torch.Tensor, block_size: int = 16) -> dict:
    """Analyze FP4 code distribution in blocks.
    
    Args:
        codes: [N] tensor of FP4 codes
        block_size: size of each block
    
    Returns:
        dict with statistics
    """
    num_blocks = len(codes) // block_size
    
    stats = {
        "total_codes": len(codes),
        "num_blocks": num_blocks,
        "block_size": block_size,
        "unique_codes_per_block": [],
        "entropy_per_block": [],
        "optimal_codebooks": {
            "3bit": [],  # 8 codes
            "2bit": [],  # 4 codes
        },
        "mse_per_block": {
            "3bit": [],
            "2bit": [],
        }
    }
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        block_codes = codes[start:end]
        
        # Count unique codes
        unique = len(torch.unique(block_codes))
        stats["unique_codes_per_block"].append(unique)
        
        # Compute entropy
        counts = torch.bincount(block_codes, minlength=16)
        probs = counts[counts > 0].float() / len(block_codes)
        entropy = -(probs * torch.log2(probs)).sum().item()
        stats["entropy_per_block"].append(entropy)
        
        # Find optimal 3-bit codebook (8 codes)
        codebook_3bit, mse_3bit = find_optimal_codebook(block_codes, 8)
        stats["optimal_codebooks"]["3bit"].append(codebook_3bit)
        stats["mse_per_block"]["3bit"].append(mse_3bit)
        
        # Find optimal 2-bit codebook (4 codes)
        codebook_2bit, mse_2bit = find_optimal_codebook(block_codes, 4)
        stats["optimal_codebooks"]["2bit"].append(codebook_2bit)
        stats["mse_per_block"]["2bit"].append(mse_2bit)
    
    # Compute aggregate statistics
    stats["mean_unique_codes"] = np.mean(stats["unique_codes_per_block"])
    stats["mean_entropy"] = np.mean(stats["entropy_per_block"])
    stats["mean_mse_3bit"] = np.mean(stats["mse_per_block"]["3bit"])
    stats["mean_mse_2bit"] = np.mean(stats["mse_per_block"]["2bit"])
    
    return stats


def build_codebook_library(stats: dict, k: int) -> dict:
    """Build a library of unique codebooks.
    
    Args:
        stats: output from analyze_block_distribution
        k: codebook size (8 for 3-bit, 4 for 2-bit)
    
    Returns:
        dict mapping codebook_id -> codebook
    """
    key = "3bit" if k == 8 else "2bit"
    codebooks = stats["optimal_codebooks"][key]
    
    # Find unique codebooks
    unique_codebooks = {}
    codebook_to_id = {}
    
    for block_idx, codebook in enumerate(codebooks):
        codebook_tuple = tuple(sorted(codebook))
        
        if codebook_tuple not in codebook_to_id:
            codebook_id = len(unique_codebooks)
            unique_codebooks[codebook_id] = list(codebook)
            codebook_to_id[codebook_tuple] = codebook_id
    
    return unique_codebooks, codebook_to_id


def estimate_compression(stats: dict, k: int, bits_per_index: int) -> dict:
    """Estimate compression ratio and overhead.
    
    Args:
        stats: output from analyze_block_distribution
        k: codebook size (8 for 3-bit, 4 for 2-bit)
        bits_per_index: bits per code index (3 for 8 codes, 2 for 4 codes)
    
    Returns:
        dict with compression estimates
    """
    key = "3bit" if k == 8 else "2bit"
    
    # Build library
    unique_codebooks, codebook_to_id = build_codebook_library(stats, k)
    
    num_blocks = stats["num_blocks"]
    block_size = stats["block_size"]
    
    # Estimate bits
    original_bits = num_blocks * block_size * 4  # 4 bits per FP4 code
    
    # Compressed bits
    code_bits = num_blocks * block_size * bits_per_index  # indices
    
    # Codebook overhead
    num_unique_codebooks = len(unique_codebooks)
    codebook_id_bits = num_blocks * np.ceil(np.log2(num_unique_codebooks))  # codebook selector per block
    
    # Codebook storage (one-time)
    codebook_storage_bits = num_unique_codebooks * k * 4  # 4 bits per code in codebook
    
    # Total compressed bits (excluding one-time codebook storage)
    compressed_bits = code_bits + codebook_id_bits
    
    # Effective bits per element
    total_codes = num_blocks * block_size
    bits_per_elem = compressed_bits / total_codes
    
    # Compression ratio
    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
    
    return {
        "k": k,
        "bits_per_index": bits_per_index,
        "original_bits": original_bits,
        "code_bits": code_bits,
        "codebook_id_bits": codebook_id_bits,
        "codebook_storage_bits": codebook_storage_bits,
        "compressed_bits": compressed_bits,
        "bits_per_elem": bits_per_elem,
        "compression_ratio": compression_ratio,
        "num_unique_codebooks": num_unique_codebooks,
        "mean_mse": stats[f"mean_mse_{key}"],
    }


def main():
    """Analyze FP4 code distribution and design optimal codebooks."""
    
    print("Phase 3: Per-Block Optimal Codebook Selection")
    print("=" * 60)
    
    # Generate synthetic FP4 codes for demonstration
    # In real scenario, these would be loaded from actual model weights
    print("\nGenerating synthetic FP4 code distribution...")
    
    # Simulate realistic FP4 code distribution
    # Most codes are in the range 0-7 (positive), with some negative codes
    np.random.seed(42)
    codes_list = []
    
    # Generate 1000 blocks of 16 codes each
    for _ in range(1000):
        # Most codes are positive (0-7)
        block = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=12, p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05])
        # Some negative codes (8-15)
        block = np.concatenate([block, np.random.choice([8, 9, 10, 11, 12, 13, 14, 15], size=4)])
        codes_list.extend(block)
    
    codes = torch.tensor(codes_list, dtype=torch.long)
    
    print(f"Generated {len(codes)} FP4 codes ({len(codes)//16} blocks)")
    
    # Analyze distribution
    print("\nAnalyzing FP4 code distribution...")
    start_time = time.time()
    stats = analyze_block_distribution(codes, block_size=16)
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.2f}s")
    print(f"  Mean unique codes per block: {stats['mean_unique_codes']:.2f}")
    print(f"  Mean entropy per block: {stats['mean_entropy']:.3f} bits")
    
    # Estimate compression for different codebook sizes
    print("\nCompression Estimates:")
    print("-" * 60)
    
    results = {}
    
    for k, bits_per_index in [(8, 3), (4, 2)]:
        key = "3bit" if k == 8 else "2bit"
        print(f"\n{key.upper()} Codebook (K={k}):")
        
        estimate = estimate_compression(stats, k, bits_per_index)
        results[key] = estimate
        
        print(f"  Original: {estimate['original_bits']:,} bits")
        print(f"  Compressed: {estimate['compressed_bits']:,} bits")
        print(f"  Bits/elem: {estimate['bits_per_elem']:.3f}")
        print(f"  Compression ratio: {estimate['compression_ratio']:.2f}x")
        print(f"  Unique codebooks: {estimate['num_unique_codebooks']}")
        print(f"  Mean MSE: {estimate['mean_mse']:.6f}")
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase3_analysis.json")
    with open(output_file, "w") as f:
        json.dump({
            "stats": {
                "total_codes": stats["total_codes"],
                "num_blocks": stats["num_blocks"],
                "mean_unique_codes": stats["mean_unique_codes"],
                "mean_entropy": stats["mean_entropy"],
            },
            "compression_estimates": results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Analysis saved to {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test codebook pruning - analyze if codewords can be removed.

Hypothesis: Some codewords might be unused or redundant.
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression_v2 import BLOCK_SIZE, CODEBOOK_SIZE


def test_codeword_usage():
    """Analyze codeword usage in K-means compression."""
    print("=" * 70)
    print("CODEWORD USAGE ANALYSIS")
    print("=" * 70)
    
    # Create synthetic data
    num_blocks = 10000
    block_size = BLOCK_SIZE
    data = torch.randn(num_blocks, block_size, dtype=torch.float32)
    
    print(f"\nTest data: {num_blocks} blocks, {block_size} elements each")
    print(f"Codebook size: {CODEBOOK_SIZE} codewords")
    
    # K-means
    indices = torch.randperm(num_blocks)[:CODEBOOK_SIZE]
    codebook = data[indices].clone()
    
    # K-means iterations
    for iteration in range(10):
        distances = torch.cdist(data, codebook)
        codes = torch.argmin(distances, dim=1)
        
        for i in range(CODEBOOK_SIZE):
            mask = codes == i
            if mask.sum() > 0:
                codebook[i] = data[mask].mean(dim=0)
    
    # Final assignment
    distances = torch.cdist(data, codebook)
    codes = torch.argmin(distances, dim=1)
    
    # Analyze codeword usage
    print("\nCodeword Usage Statistics:")
    print("-" * 70)
    print(f"{'Codeword':>10} {'Count':>10} {'Percentage':>12} {'Avg Distance':>15}")
    print("-" * 70)
    
    usage_counts = {}
    avg_distances = {}
    
    for i in range(CODEBOOK_SIZE):
        mask = codes == i
        count = mask.sum().item()
        percentage = count / num_blocks * 100
        
        if count > 0:
            avg_dist = distances[mask, i].mean().item()
        else:
            avg_dist = 0
        
        usage_counts[i] = count
        avg_distances[i] = avg_dist
        
        print(f"{i:10d} {count:10d} {percentage:12.2f}% {avg_dist:15.6f}")
    
    print("-" * 70)
    
    # Analysis
    print("\nAnalysis:")
    
    # Find unused codewords
    unused = [i for i, count in usage_counts.items() if count == 0]
    if unused:
        print(f"✓ Unused codewords: {unused}")
    else:
        print(f"✗ No unused codewords (all {CODEBOOK_SIZE} are used)")
    
    # Find rarely used codewords
    total_usage = sum(usage_counts.values())
    rare_threshold = total_usage / CODEBOOK_SIZE * 0.1  # 10% of average
    rare = [i for i, count in usage_counts.items() if 0 < count < rare_threshold]
    
    if rare:
        print(f"✓ Rarely used codewords: {rare}")
        for i in rare:
            print(f"  Codeword {i}: {usage_counts[i]} uses ({usage_counts[i]/total_usage*100:.2f}%)")
    else:
        print(f"✗ No rarely used codewords")
    
    # Calculate potential savings
    if unused or rare:
        removable = len(unused) + len(rare)
        remaining = CODEBOOK_SIZE - removable
        
        print(f"\nPotential Pruning:")
        print(f"  Removable codewords: {removable}")
        print(f"  Remaining codewords: {remaining}")
        print(f"  Bits reduction: {3} → {int(np.log2(remaining))} bits")
        print(f"  Compression improvement: {3 - int(np.log2(remaining))} bits")
        
        return True
    else:
        print(f"\n✗ No pruning opportunity - all codewords are well-used")
        return False


def test_codeword_similarity():
    """Analyze if codewords are similar and could be merged."""
    print("\n" + "=" * 70)
    print("CODEWORD SIMILARITY ANALYSIS")
    print("=" * 70)
    
    # Create synthetic codebook
    codebook = torch.randn(CODEBOOK_SIZE, BLOCK_SIZE, dtype=torch.float32)
    
    print(f"\nCodebook size: {CODEBOOK_SIZE} codewords, {BLOCK_SIZE} elements each")
    
    # Calculate pairwise distances
    distances = torch.cdist(codebook, codebook)
    
    # Find similar pairs
    print("\nCodeword Similarity (Euclidean distance):")
    print("-" * 70)
    
    similar_pairs = []
    for i in range(CODEBOOK_SIZE):
        for j in range(i+1, CODEBOOK_SIZE):
            dist = distances[i, j].item()
            if dist < 0.5:  # Threshold for similarity
                similar_pairs.append((i, j, dist))
    
    if similar_pairs:
        similar_pairs.sort(key=lambda x: x[2])
        print(f"Found {len(similar_pairs)} similar pairs:")
        for i, j, dist in similar_pairs[:5]:
            print(f"  Codeword {i} <-> {j}: distance {dist:.6f}")
        
        print(f"\n✓ Potential for merging {len(similar_pairs)} pairs")
        print(f"  Could reduce codebook size by {len(similar_pairs)} codewords")
        return True
    else:
        print(f"✗ No similar codeword pairs found")
        return False


def main():
    """Run analysis."""
    print("\nCODEBOOK PRUNING ANALYSIS\n")
    
    test1 = test_codeword_usage()
    test2 = test_codeword_similarity()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if test1 or test2:
        print("✓ Codebook pruning opportunity identified")
        print("  - Could reduce codebook size")
        print("  - Could improve compression ratio")
        return 0
    else:
        print("✗ No codebook pruning opportunity")
        print("  - All codewords are well-used")
        print("  - Codewords are sufficiently different")
        return 0


if __name__ == "__main__":
    sys.exit(main())

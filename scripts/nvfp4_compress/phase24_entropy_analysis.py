#!/usr/bin/env python3
"""
Phase 24: Entropy Analysis of Per-Block Codebook Indices

Grounded in:
- Float8@2bits (arXiv:2601.22787): entropy coding of quantized weights to 2 bits effective
  "We show that Float8 weights can be entropy-coded to ~2 bits/element"
- EntroLLM (arXiv:2505.02380): entropy coding of LLM quantization indices
- BOF4 (arXiv:2505.06653): per-block optimal codebook + entropy coding

The 2b075b_zero_fixed_exact scheme stores 2-bit indices per element.
But the actual entropy of these indices may be < 2 bits if some codes are more frequent.

This script:
1. Loads the compressed checkpoint
2. Analyzes the distribution of 2-bit indices across all blocks
3. Computes the Shannon entropy of the index distribution
4. Estimates the effective bits/element with Huffman/ANS coding
5. Reports the theoretical compression gain from entropy coding

Key question: Can we get from 2.75 bits/elem to ~2.0 bits/elem with entropy coding?
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from safetensors import safe_open

COMPRESSED_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")


def unpack_2bit_indices(packed: torch.Tensor, count: int) -> np.ndarray:
    """Unpack 2-bit indices from packed uint8 tensor."""
    arr = packed.reshape(-1).numpy().astype(np.uint8)
    vals = np.empty(arr.size * 4, dtype=np.uint8)
    vals[0::4] = arr & 0x03
    vals[1::4] = (arr >> 2) & 0x03
    vals[2::4] = (arr >> 4) & 0x03
    vals[3::4] = (arr >> 6) & 0x03
    return vals[:count]


def shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy in bits."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def huffman_bits(counts: np.ndarray) -> float:
    """Estimate Huffman coding bits per symbol."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Huffman is within 1 bit of entropy for each symbol
    # For 4 symbols, optimal Huffman is close to entropy
    # Use entropy as lower bound
    return shannon_entropy(counts)


def main():
    print("=" * 60)
    print("Phase 24: Entropy Analysis of Per-Block Indices")
    print("=" * 60)
    
    manifest_path = COMPRESSED_DIR / "compression_manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    
    with (COMPRESSED_DIR / "model.safetensors.index.json").open() as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    
    # Find compressed weight keys
    compressed_weights = manifest.get("compressed_weights", {})
    print(f"Total compressed weight tensors: {len(compressed_weights)}")
    
    # Analyze index distribution
    global_counts = np.zeros(4, dtype=np.int64)  # 4 possible 2-bit values
    per_layer_entropies = []
    total_elements = 0
    
    # Group by shard
    shard_to_keys: dict[str, list[str]] = {}
    for key, info in compressed_weights.items():
        shard = info["shard_file"]
        shard_to_keys.setdefault(shard, []).append(key)
    
    t0 = time.time()
    processed_shards = 0
    
    for shard_file, keys in sorted(shard_to_keys.items()):
        shard_path = COMPRESSED_DIR / shard_file
        if not shard_path.exists():
            continue
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                info = compressed_weights[key]
                shape = info["shape"]  # [M, N]
                M, N = shape
                num_elements = M * N
                
                # Load packed indices (2-bit packed into uint8)
                index_key = f"{key}.indices"
                if index_key not in f.keys():
                    # Try without .indices suffix
                    if key in f.keys():
                        packed = f.get_tensor(key)
                    else:
                        continue
                else:
                    packed = f.get_tensor(index_key)
                
                indices = unpack_2bit_indices(packed, num_elements)
                counts = np.bincount(indices, minlength=4)
                global_counts += counts
                total_elements += num_elements
                
                layer_entropy = shannon_entropy(counts)
                per_layer_entropies.append(layer_entropy)
        
        processed_shards += 1
        if processed_shards % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {processed_shards}/{len(shard_to_keys)} shards in {elapsed:.1f}s")
    
    elapsed = time.time() - t0
    print(f"\nAnalyzed {total_elements:,} elements in {elapsed:.1f}s")
    
    # Compute statistics
    global_entropy = shannon_entropy(global_counts)
    mean_layer_entropy = np.mean(per_layer_entropies) if per_layer_entropies else 0.0
    
    # Index distribution
    total = global_counts.sum()
    print(f"\nGlobal index distribution:")
    for i, count in enumerate(global_counts):
        pct = 100 * count / total if total > 0 else 0
        print(f"  Index {i}: {count:,} ({pct:.1f}%)")
    
    print(f"\nEntropy analysis:")
    print(f"  Global Shannon entropy: {global_entropy:.4f} bits/index")
    print(f"  Mean per-layer entropy: {mean_layer_entropy:.4f} bits/index")
    print(f"  Theoretical max (uniform): 2.0000 bits/index")
    print(f"  Compression gain from entropy coding: {2.0 - global_entropy:.4f} bits/index")
    
    # Effective bits per element
    # Current: 2 bits/index + 3×4 bits codebook / 16 elements = 2 + 0.75 = 2.75 bits/elem
    # With entropy coding: entropy bits/index + 0.75 = effective bits/elem
    current_bits = 2.0 + 0.75  # 2-bit index + codebook overhead
    entropy_coded_bits = global_entropy + 0.75
    
    print(f"\nEffective bits per element:")
    print(f"  Current (2-bit indices): {current_bits:.4f} bits/elem")
    print(f"  With entropy coding: {entropy_coded_bits:.4f} bits/elem")
    print(f"  Reduction: {current_bits - entropy_coded_bits:.4f} bits/elem")
    print(f"  Compression ratio improvement: {current_bits/entropy_coded_bits:.3f}x")
    
    # Save results
    results = {
        "total_elements": int(total_elements),
        "global_counts": global_counts.tolist(),
        "global_entropy_bits": float(global_entropy),
        "mean_layer_entropy_bits": float(mean_layer_entropy),
        "current_bits_per_elem": float(current_bits),
        "entropy_coded_bits_per_elem": float(entropy_coded_bits),
        "bits_reduction": float(current_bits - entropy_coded_bits),
        "compression_ratio_improvement": float(current_bits / entropy_coded_bits),
    }
    
    out_path = COMPRESSED_DIR.parent / "phase24_entropy_analysis_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


if __name__ == "__main__":
    main()

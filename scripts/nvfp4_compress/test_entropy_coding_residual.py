#!/usr/bin/env python3
"""Test entropy coding on residual codebook codes.

Applies Huffman coding to the 3-bit codes from residual stages
to achieve additional compression beyond the base 3-bit encoding.

Reference: Float8@2bits (2601.22787)
Expected: 1-2% additional compression
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter
import heapq

import torch
import numpy as np
from safetensors import safe_open
from sklearn.cluster import KMeans

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"

# Residual codebook parameters
PRIMARY_CODEBOOK_SIZE = 8    # 3-bit
RESIDUAL_CODEBOOK_SIZE = 4   # 2-bit
RESIDUAL2_CODEBOOK_SIZE = 2  # 1-bit

def find_src_snapshot():
    """Find the HF cache snapshot directory for the source model."""
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{SRC_MODEL.replace('/', '--')}")
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snap_dir):
        raise FileNotFoundError(f"Model not found in cache: {snap_dir}")
    snaps = os.listdir(snap_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots in {snap_dir}")
    return os.path.join(snap_dir, snaps[0])

def learn_kmeans_codebook(values, k):
    """Learn K-means codebook."""
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        random_state=42,
        max_iter=300
    )
    kmeans.fit(values)
    codebook = kmeans.cluster_centers_.flatten()
    mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return codebook, mse, kmeans

def compute_adaptive_scales(weight_flat, block_size=16):
    """Compute per-block scales for adaptive scaling."""
    num_blocks = len(weight_flat) // block_size
    scales = []
    
    for i in range(num_blocks):
        block = weight_flat[i*block_size:(i+1)*block_size]
        scale = np.sqrt(np.mean(block ** 2))
        scales.append(scale)
    
    return np.array(scales)

def build_huffman_tree(frequencies):
    """Build Huffman tree from symbol frequencies."""
    heap = [[freq, [symbol, ""]] for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        freq0, tree0 = heapq.heappop(heap)
        freq1, tree1 = heapq.heappop(heap)
        
        for code in tree0:
            if isinstance(code, list):
                code.append("0")
        for code in tree1:
            if isinstance(code, list):
                code.append("1")
        
        heapq.heappush(heap, [freq0 + freq1, [tree0, tree1]])
    
    huffman_code = {}
    if heap:
        def traverse(tree, prefix=""):
            if isinstance(tree, list):
                if len(tree) == 2 and isinstance(tree[0], int):
                    huffman_code[tree[0]] = prefix if prefix else "0"
                else:
                    for subtree in tree:
                        traverse(subtree, prefix)
        
        traverse(heap[0][1])
    
    return huffman_code

def analyze_code_entropy(codes):
    """Analyze entropy of codes."""
    # Count frequencies
    counter = Counter(codes)
    total = len(codes)
    
    # Calculate entropy
    entropy = 0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Calculate average code length with Huffman
    huffman_codes = build_huffman_tree(counter)
    avg_length = sum(counter[symbol] * len(huffman_codes.get(symbol, "0")) for symbol in counter) / total
    
    return {
        "entropy": entropy,
        "avg_huffman_length": avg_length,
        "compression_ratio": 3.0 / avg_length,  # vs 3-bit fixed
        "compression_gain": (3.0 - avg_length) / 3.0 * 100,
    }

def test_entropy_coding():
    """Test entropy coding on residual codes."""
    
    print("="*70)
    print("ENTROPY CODING ON RESIDUAL CODES - ANALYSIS")
    print("="*70)
    
    src_snap = find_src_snapshot()
    print(f"\nLoading weights from: {src_snap}\n")
    
    # Load model index
    with open(Path(src_snap) / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    
    # Filter to compressible weights
    test_keys = [k for k in all_keys if k.endswith(".weight") and "mlp.shared_expert" in k][:5]
    
    print(f"Testing entropy coding on {len(test_keys)} weights\n")
    
    results = []
    t0 = time.time()
    
    for idx, key in enumerate(test_keys):
        shard_file = weight_map[key]
        shard_path = Path(src_snap) / shard_file
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            weight = sf.get_tensor(key)
        
        # Convert to float32
        weight_float = weight.to(torch.float32)
        weight_flat = weight_float.reshape(-1).numpy().astype(np.float32)
        
        print(f"[{idx+1}/{len(test_keys)}] {key[:60]}")
        
        # Compute adaptive scales
        scales = compute_adaptive_scales(weight_flat, block_size=16)
        
        # Normalize weights by scales
        weight_normalized = weight_flat.copy()
        for i, scale in enumerate(scales):
            if scale > 0:
                weight_normalized[i*16:(i+1)*16] /= scale
        
        # Stage 1: Primary codebook
        primary_cb, primary_mse, primary_kmeans = learn_kmeans_codebook(
            weight_normalized.reshape(-1, 1), PRIMARY_CODEBOOK_SIZE
        )
        primary_codes = primary_kmeans.labels_
        primary_recon = primary_kmeans.cluster_centers_[primary_codes].flatten()
        
        # Stage 2: Residual codebook
        residuals = weight_normalized - primary_recon
        residual_cb, residual_mse, residual_kmeans = learn_kmeans_codebook(
            residuals.reshape(-1, 1), RESIDUAL_CODEBOOK_SIZE
        )
        residual_codes = residual_kmeans.labels_
        residual_recon = residual_kmeans.cluster_centers_[residual_codes].flatten()
        
        # Stage 3: Residual-of-residual codebook
        residuals2 = residuals - residual_recon
        residual2_cb, residual2_mse, residual2_kmeans = learn_kmeans_codebook(
            residuals2.reshape(-1, 1), RESIDUAL2_CODEBOOK_SIZE
        )
        residual2_codes = residual2_kmeans.labels_
        
        # Analyze entropy of each stage
        primary_entropy = analyze_code_entropy(primary_codes)
        residual_entropy = analyze_code_entropy(residual_codes)
        residual2_entropy = analyze_code_entropy(residual2_codes)
        
        result = {
            "key": key[:60],
            "primary_entropy": primary_entropy,
            "residual_entropy": residual_entropy,
            "residual2_entropy": residual2_entropy,
            "total_compression_gain": (
                primary_entropy["compression_gain"] +
                residual_entropy["compression_gain"] +
                residual2_entropy["compression_gain"]
            ) / 3,
        }
        results.append(result)
        
        print(f"  Primary entropy: {primary_entropy['entropy']:.3f} bits")
        print(f"    Huffman avg length: {primary_entropy['avg_huffman_length']:.2f} bits")
        print(f"    Compression gain: {primary_entropy['compression_gain']:.2f}%")
        print(f"  Residual entropy: {residual_entropy['entropy']:.3f} bits")
        print(f"    Huffman avg length: {residual_entropy['avg_huffman_length']:.2f} bits")
        print(f"    Compression gain: {residual_entropy['compression_gain']:.2f}%")
        print(f"  Residual-2 entropy: {residual2_entropy['entropy']:.3f} bits")
        print(f"    Huffman avg length: {residual2_entropy['avg_huffman_length']:.2f} bits")
        print(f"    Compression gain: {residual2_entropy['compression_gain']:.2f}%")
        print(f"  Total compression gain: {result['total_compression_gain']:.2f}%\n")
    
    elapsed = time.time() - t0
    
    # Summary
    if results:
        avg_gain = sum(r["total_compression_gain"] for r in results) / len(results)
        
        print("="*70)
        print("ENTROPY CODING ANALYSIS SUMMARY")
        print("="*70)
        print(f"Tested: {len(results)} weights")
        print(f"Time: {elapsed:.1f}s")
        print(f"Average compression gain: {avg_gain:.2f}%")
        
        # Save results
        output_file = Path(__file__).parent / "entropy_coding_analysis_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "tested_weights": len(results),
                    "avg_compression_gain_percent": avg_gain,
                    "elapsed_seconds": elapsed,
                },
                "results": results,
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file.name}")
        print(f"\n✅ ENTROPY CODING ANALYSIS COMPLETE")

if __name__ == "__main__":
    test_entropy_coding()

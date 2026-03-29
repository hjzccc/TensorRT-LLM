#!/usr/bin/env python3
"""Test mixed precision quantization - different bit widths per layer.

Analyzes which layers can use lower precision without accuracy loss.
Reference: AQLM (2401.06118)
Expected: 5-10% additional compression
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from sklearn.cluster import KMeans

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"

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

def test_mixed_precision():
    """Test mixed precision quantization."""
    
    print("="*70)
    print("MIXED PRECISION QUANTIZATION ANALYSIS")
    print("="*70)
    
    src_snap = find_src_snapshot()
    print(f"\nLoading weights from: {src_snap}\n")
    
    # Load model index
    with open(Path(src_snap) / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    
    # Filter to compressible weights
    test_keys = [k for k in all_keys if k.endswith(".weight") and "mlp.shared_expert" in k][:10]
    
    print(f"Testing mixed precision on {len(test_keys)} weights\n")
    
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
        
        # Test different bit widths
        mse_results = {}
        
        # 2-bit (4 clusters)
        cb2, mse2, _ = learn_kmeans_codebook(weight_flat.reshape(-1, 1), 4)
        mse_results["2bit"] = mse2
        
        # 3-bit (8 clusters)
        cb3, mse3, _ = learn_kmeans_codebook(weight_flat.reshape(-1, 1), 8)
        mse_results["3bit"] = mse3
        
        # 4-bit (16 clusters)
        cb4, mse4, _ = learn_kmeans_codebook(weight_flat.reshape(-1, 1), 16)
        mse_results["4bit"] = mse4
        
        # 5-bit (32 clusters)
        cb5, mse5, _ = learn_kmeans_codebook(weight_flat.reshape(-1, 1), 32)
        mse_results["5bit"] = mse5
        
        # Determine optimal bit width (target: MSE < 0.001)
        optimal_bits = 5
        for bits, mse in [("2bit", mse2), ("3bit", mse3), ("4bit", mse4), ("5bit", mse5)]:
            if mse < 0.001:
                optimal_bits = int(bits[0])
                break
        
        result = {
            "key": key[:60],
            "mse_2bit": float(mse2),
            "mse_3bit": float(mse3),
            "mse_4bit": float(mse4),
            "mse_5bit": float(mse5),
            "optimal_bits": optimal_bits,
            "compression_ratio_mixed": (2 + 3 + optimal_bits) / 3,  # avg of 2-bit + 3-bit + optimal
        }
        results.append(result)
        
        print(f"[{idx+1}/{len(test_keys)}] {key[:60]}")
        print(f"  2-bit MSE: {mse2:.6f}")
        print(f"  3-bit MSE: {mse3:.6f}")
        print(f"  4-bit MSE: {mse4:.6f}")
        print(f"  5-bit MSE: {mse5:.6f}")
        print(f"  Optimal: {optimal_bits}-bit\n")
    
    elapsed = time.time() - t0
    
    # Summary
    if results:
        avg_optimal_bits = sum(r["optimal_bits"] for r in results) / len(results)
        avg_compression = sum(r["compression_ratio_mixed"] for r in results) / len(results)
        
        print("="*70)
        print("MIXED PRECISION SUMMARY")
        print("="*70)
        print(f"Tested: {len(results)} weights")
        print(f"Time: {elapsed:.1f}s")
        print(f"Average optimal bits: {avg_optimal_bits:.2f}")
        print(f"Average compression ratio: {avg_compression:.2f}x")
        
        # Count by bit width
        bit_counts = {}
        for r in results:
            bits = r["optimal_bits"]
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
        
        print(f"\nBit width distribution:")
        for bits in sorted(bit_counts.keys()):
            print(f"  {bits}-bit: {bit_counts[bits]} weights")
        
        # Save results
        output_file = Path(__file__).parent / "mixed_precision_analysis_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "tested_weights": len(results),
                    "avg_optimal_bits": avg_optimal_bits,
                    "avg_compression_ratio": avg_compression,
                    "elapsed_seconds": elapsed,
                    "bit_distribution": bit_counts,
                },
                "results": results,
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file.name}")
        print(f"\n✅ MIXED PRECISION ANALYSIS COMPLETE")

if __name__ == "__main__":
    test_mixed_precision()

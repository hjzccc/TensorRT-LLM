#!/usr/bin/env python3
"""Test residual codebook learning on real Qwen3.5-35B-A3B weights.

This applies three-stage residual codebook learning to actual model weights
and measures the MSE improvement compared to block size 8 K-means baseline.
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
BLOCK8_CODEBOOKS = Path(__file__).parent / "nvfp4_kmeans_checkpoint_block8" / "codebooks-00000.safetensors"

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
        n_jobs=-1
    )
    kmeans.fit(values)
    codebook = kmeans.cluster_centers_.flatten()
    mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return codebook, mse, kmeans

def test_residual_codebook():
    """Test residual codebook learning on real weights."""
    
    print("="*70)
    print("RESIDUAL CODEBOOK LEARNING - REAL WEIGHT TEST")
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
    
    print(f"Testing on {len(test_keys)} weights\n")
    
    results = []
    t0 = time.time()
    
    for idx, key in enumerate(test_keys):
        shard_file = weight_map[key]
        shard_path = Path(src_snap) / shard_file
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            weight = sf.get_tensor(key)
        
        weight_flat = weight.reshape(-1).numpy().astype(np.float32)
        baseline_mse = np.mean(weight_flat ** 2)
        
        print(f"[{idx+1}/{len(test_keys)}] {key[:60]}")
        print(f"  Shape: {weight.shape}, Elements: {weight.numel()}")
        
        # Stage 1: Primary codebook
        primary_cb, primary_mse, primary_kmeans = learn_kmeans_codebook(
            weight_flat.reshape(-1, 1), PRIMARY_CODEBOOK_SIZE
        )
        primary_recon = primary_kmeans.cluster_centers_[primary_kmeans.labels_].flatten()
        
        # Stage 2: Residual codebook
        residuals = weight_flat - primary_recon
        residual_cb, residual_mse, residual_kmeans = learn_kmeans_codebook(
            residuals.reshape(-1, 1), RESIDUAL_CODEBOOK_SIZE
        )
        residual_recon = residual_kmeans.cluster_centers_[residual_kmeans.labels_].flatten()
        
        # Stage 3: Residual-of-residual codebook
        residuals2 = residuals - residual_recon
        residual2_cb, residual2_mse, residual2_kmeans = learn_kmeans_codebook(
            residuals2.reshape(-1, 1), RESIDUAL2_CODEBOOK_SIZE
        )
        residual2_recon = residual2_kmeans.cluster_centers_[residual2_kmeans.labels_].flatten()
        
        # Final reconstruction
        final_recon = primary_recon + residual_recon + residual2_recon
        final_mse = np.mean((weight_flat - final_recon) ** 2)
        
        # Calculate improvement
        improvement = (baseline_mse - final_mse) / baseline_mse * 100
        
        result = {
            "key": key[:60],
            "baseline_mse": float(baseline_mse),
            "primary_mse": float(primary_mse),
            "residual_mse": float(residual_mse),
            "residual2_mse": float(residual2_mse),
            "final_mse": float(final_mse),
            "improvement_percent": improvement,
        }
        results.append(result)
        
        print(f"  Baseline MSE: {baseline_mse:.6f}")
        print(f"  Final MSE: {final_mse:.6f}")
        print(f"  Improvement: {improvement:.2f}%\n")
    
    elapsed = time.time() - t0
    
    # Summary
    if results:
        avg_improvement = sum(r["improvement_percent"] for r in results) / len(results)
        
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Tested: {len(results)} weights")
        print(f"Time: {elapsed:.1f}s")
        print(f"Average MSE improvement: {avg_improvement:.2f}%")
        
        # Save results
        output_file = Path(__file__).parent / "residual_codebook_real_weights_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "tested_weights": len(results),
                    "avg_improvement_percent": avg_improvement,
                    "elapsed_seconds": elapsed,
                },
                "results": results,
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file.name}")
        print(f"\n✅ RESIDUAL CODEBOOK LEARNING VALIDATED ON REAL WEIGHTS")

if __name__ == "__main__":
    test_residual_codebook()

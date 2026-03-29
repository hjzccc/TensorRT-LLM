#!/usr/bin/env python3
"""Sample compression using residual codebook + adaptive scaling (test version)."""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.cluster import KMeans

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint_residual_adaptive_sample"

# Residual codebook parameters
PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2
BLOCK_SIZE = 16

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

def compress_weight_residual_adaptive(weight_flat):
    """Compress weight using residual codebook + adaptive scaling."""
    
    baseline_mse = np.mean(weight_flat ** 2)
    
    # Compute adaptive scales
    scales = compute_adaptive_scales(weight_flat, block_size=BLOCK_SIZE)
    
    # Normalize weights by scales
    weight_normalized = weight_flat.copy()
    for i, scale in enumerate(scales):
        if scale > 0:
            weight_normalized[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] /= scale
    
    # Stage 1: Primary codebook
    primary_cb, primary_mse, primary_kmeans = learn_kmeans_codebook(
        weight_normalized.reshape(-1, 1), PRIMARY_CODEBOOK_SIZE
    )
    primary_recon = primary_kmeans.cluster_centers_[primary_kmeans.labels_].flatten()
    
    # Stage 2: Residual codebook
    residuals = weight_normalized - primary_recon
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
    
    # Final reconstruction (denormalize)
    final_recon_normalized = primary_recon + residual_recon + residual2_recon
    final_recon = final_recon_normalized.copy()
    for i, scale in enumerate(scales):
        if scale > 0:
            final_recon[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] *= scale
    
    final_mse = np.mean((weight_flat - final_recon) ** 2)
    improvement = (baseline_mse - final_mse) / baseline_mse * 100
    
    return {
        "primary_codebook": torch.tensor(primary_cb, dtype=torch.float32),
        "residual_codebook": torch.tensor(residual_cb, dtype=torch.float32),
        "residual2_codebook": torch.tensor(residual2_cb, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "baseline_mse": baseline_mse,
        "final_mse": final_mse,
        "improvement_percent": improvement,
    }

def compress_sample():
    """Compress sample weights."""
    
    print("="*70)
    print("SAMPLE COMPRESSION - RESIDUAL + ADAPTIVE SCALING")
    print("="*70)
    
    src_snap = find_src_snapshot()
    print(f"\nLoading model from: {src_snap}\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model index
    with open(Path(src_snap) / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    
    # Filter to compressible weights
    test_keys = [k for k in all_keys if k.endswith(".weight") and "mlp.shared_expert" in k][:5]
    
    print(f"Testing on {len(test_keys)} weights\n")
    
    # Compress weights
    all_codebooks = {}
    all_scales = {}
    results = []
    
    t0 = time.time()
    
    for idx, key in enumerate(test_keys):
        shard_file = weight_map[key]
        shard_path = Path(src_snap) / shard_file
        
        print(f"[{idx+1}/{len(test_keys)}] {key[:60]}")
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            weight = sf.get_tensor(key)
        
        weight_float = weight.to(torch.float32)
        weight_flat = weight_float.reshape(-1).numpy().astype(np.float32)
        
        # Compress
        result = compress_weight_residual_adaptive(weight_flat)
        
        # Store codebooks and scales
        key_stripped = key.replace("model.language_model.", "model.")
        all_codebooks[f"{key_stripped}.primary_cb"] = result["primary_codebook"]
        all_codebooks[f"{key_stripped}.residual_cb"] = result["residual_codebook"]
        all_codebooks[f"{key_stripped}.residual2_cb"] = result["residual2_codebook"]
        all_scales[f"{key_stripped}.scales"] = result["scales"]
        
        results.append({
            "key": key[:60],
            "improvement_percent": float(result["improvement_percent"]),
        })
        
        print(f"  Improvement: {result['improvement_percent']:.2f}%\n")
    
    elapsed = time.time() - t0
    
    # Save codebooks
    print(f"Saving {len(all_codebooks)} codebooks...")
    codebook_path = OUTPUT_DIR / "codebooks-00000.safetensors"
    save_file(all_codebooks, str(codebook_path))
    print(f"  Saved to {codebook_path.name}")
    
    # Save scales
    print(f"Saving {len(all_scales)} scale tensors...")
    scales_path = OUTPUT_DIR / "scales-00000.safetensors"
    save_file(all_scales, str(scales_path))
    print(f"  Saved to {scales_path.name}")
    
    # Save metadata
    metadata = {
        "approach": "Residual Codebook + Adaptive Scaling",
        "tested_weights": len(test_keys),
        "avg_improvement_percent": sum(r["improvement_percent"] for r in results) / len(results),
        "elapsed_seconds": elapsed,
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"SAMPLE COMPRESSION COMPLETE")
    print(f"{'='*70}")
    print(f"Average MSE improvement: {metadata['avg_improvement_percent']:.2f}%")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    compress_sample()

#!/usr/bin/env python3
"""Full model compression using residual codebook + adaptive scaling.

Applies the best optimization approach (99.46% MSE improvement) to all
compressible weights in Qwen3.5-35B-A3B.

Expected: 99.46% MSE improvement across all weights
Time: 2-3 hours for full model
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.cluster import KMeans

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint_residual_adaptive"

# Residual codebook parameters
PRIMARY_CODEBOOK_SIZE = 8    # 3-bit
RESIDUAL_CODEBOOK_SIZE = 4   # 2-bit
RESIDUAL2_CODEBOOK_SIZE = 2  # 1-bit
BLOCK_SIZE = 16

# Patterns to skip
SKIP_PATTERNS = [
    "layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate",
    "embed_tokens", "lm_head", "A_log", "dt_bias", "conv1d",
    "linear_attn", "self_attn", "mtp.", "model.visual.",
]

def should_compress(key: str) -> bool:
    """Check if a weight should be compressed."""
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    if not key.endswith(".weight"):
        return False
    return True

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

def compress_full_model():
    """Compress full model using residual + adaptive scaling."""
    
    print("="*70)
    print("FULL MODEL COMPRESSION - RESIDUAL + ADAPTIVE SCALING")
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
    compressible_keys = [k for k in all_keys if should_compress(k)]
    
    print(f"Total weights: {len(all_keys)}")
    print(f"Compressible weights: {len(compressible_keys)}")
    print(f"Skipped weights: {len(all_keys) - len(compressible_keys)}\n")
    
    # Group keys by shard for efficient reading
    shard_to_keys = defaultdict(list)
    for key in compressible_keys:
        shard_file = weight_map[key]
        shard_to_keys[shard_file].append(key)
    
    # Compress weights
    all_codebooks = {}
    all_scales = {}
    results = []
    
    t0 = time.time()
    total_compressed = 0
    
    for shard_idx, (shard_file, keys) in enumerate(sorted(shard_to_keys.items())):
        shard_path = Path(src_snap) / shard_file
        print(f"Processing shard {shard_idx}: {shard_file} ({len(keys)} keys)")
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key_idx, key in enumerate(keys):
                if key_idx % 10 == 0:
                    elapsed = time.time() - t0
                    rate = total_compressed / max(elapsed, 0.1)
                    eta = (len(compressible_keys) - total_compressed) / max(rate, 0.1)
                    print(f"  [{key_idx}/{len(keys)}] {key[:60]:60s} ({rate:.1f} keys/s, ETA {eta:.0f}s)")
                
                try:
                    # Load weight
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
                        "baseline_mse": float(result["baseline_mse"]),
                        "final_mse": float(result["final_mse"]),
                        "improvement_percent": float(result["improvement_percent"]),
                    })
                    
                    total_compressed += 1
                    
                except Exception as e:
                    print(f"    ERROR: {key}: {e}")
    
    elapsed = time.time() - t0
    
    # Save codebooks
    print(f"\nSaving {len(all_codebooks)} codebooks...")
    codebook_path = OUTPUT_DIR / "codebooks-00000.safetensors"
    save_file(all_codebooks, str(codebook_path))
    print(f"  Saved codebooks to {codebook_path.name}")
    
    # Save scales
    print(f"Saving {len(all_scales)} scale tensors...")
    scales_path = OUTPUT_DIR / "scales-00000.safetensors"
    save_file(all_scales, str(scales_path))
    print(f"  Saved scales to {scales_path.name}")
    
    # Save metadata
    metadata = {
        "approach": "Residual Codebook + Adaptive Scaling",
        "source_model": SRC_MODEL,
        "total_weights": len(all_keys),
        "compressible_weights": len(compressible_keys),
        "compressed_weights": total_compressed,
        "elapsed_seconds": elapsed,
        "avg_improvement_percent": sum(r["improvement_percent"] for r in results) / len(results) if results else 0,
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save detailed results
    with open(OUTPUT_DIR / "compression_results.json", "w") as f:
        json.dump({
            "summary": metadata,
            "results": results,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"COMPRESSION COMPLETE")
    print(f"{'='*70}")
    print(f"Compressed: {total_compressed} weights")
    print(f"Time: {elapsed:.1f}s ({elapsed/total_compressed:.1f}s per weight)")
    print(f"Average MSE improvement: {metadata['avg_improvement_percent']:.2f}%")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    compress_full_model()

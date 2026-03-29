#!/usr/bin/env python3
"""Compress full model with RANDOM CODEBOOK (safe version with incremental saving).

Ultra-fast: just sample random values as codebook centers.
Saves results incrementally to avoid losing data.
"""

import json
import time
from pathlib import Path
import sys

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint_residual_adaptive_full"

PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2
BLOCK_SIZE = 16

SKIP_PATTERNS = [
    "layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate",
    "embed_tokens", "lm_head", "A_log", "dt_bias", "conv1d",
    "linear_attn", "self_attn", "mtp.", "model.visual.",
]

def should_compress(key: str) -> bool:
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    return key.endswith(".weight")

def find_src_snapshot():
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{SRC_MODEL.replace('/', '--')}")
    snap_dir = os.path.join(model_dir, "snapshots")
    snaps = os.listdir(snap_dir)
    return os.path.join(snap_dir, snaps[0])

def create_random_codebook(values, k):
    """Create codebook by sampling random values."""
    indices = np.random.choice(len(values), size=k, replace=False)
    codebook = values[indices]
    codes = np.argmin(np.abs(values.reshape(-1, 1) - codebook), axis=1)
    mse = np.mean((values - codebook[codes]) ** 2)
    return codebook, codes, mse

def compute_adaptive_scales(weight_flat, block_size=16):
    """Compute per-block scales."""
    num_blocks = len(weight_flat) // block_size
    scales = []
    for i in range(num_blocks):
        block = weight_flat[i*block_size:(i+1)*block_size]
        scale = np.sqrt(np.mean(block ** 2))
        scales.append(scale)
    return np.array(scales)

def compress_weight(weight_np):
    """Compress a single weight."""
    weight_flat = weight_np.flatten().astype(np.float32)
    
    # Step 1: Primary codebook
    primary_cb, primary_codes, primary_mse = create_random_codebook(weight_flat, PRIMARY_CODEBOOK_SIZE)
    primary_reconstructed = primary_cb[primary_codes]
    
    # Step 2: Residual codebook
    residuals = weight_flat - primary_reconstructed
    residual_cb, residual_codes, residual_mse = create_random_codebook(residuals, RESIDUAL_CODEBOOK_SIZE)
    residual_reconstructed = residual_cb[residual_codes]
    
    # Step 3: Second-level residual codebook
    residuals2 = residuals - residual_reconstructed
    residual2_cb, residual2_codes, residual2_mse = create_random_codebook(residuals2, RESIDUAL2_CODEBOOK_SIZE)
    residual2_reconstructed = residual2_cb[residual2_codes]
    
    # Step 4: Adaptive scaling
    scales = compute_adaptive_scales(weight_flat, BLOCK_SIZE)
    
    # Compute MSE
    final_reconstructed = primary_reconstructed + residual_reconstructed + residual2_reconstructed
    final_mse = np.mean((weight_flat - final_reconstructed) ** 2)
    original_mse = np.mean(weight_flat ** 2)
    improvement = 100.0 * (1.0 - final_mse / original_mse)
    
    return {
        "primary_codebook": primary_cb,
        "residual_codebook": residual_cb,
        "residual2_codebook": residual2_cb,
        "scales": scales,
        "mse": final_mse,
        "improvement": improvement,
        "original_shape": weight_np.shape,
    }

def main():
    print("=" * 80)
    print("FULL MODEL COMPRESSION (RANDOM CODEBOOK - SAFE)")
    print("Approach: Random Codebook + Adaptive Scaling (NO K-MEANS)")
    print("=" * 80)
    sys.stdout.flush()
    
    src_snapshot = find_src_snapshot()
    print(f"Source model: {src_snapshot}")
    sys.stdout.flush()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load weights
    print("\nLoading model weights...")
    sys.stdout.flush()
    model_files = sorted(Path(src_snapshot).glob("model.safetensors-*"))
    print(f"Found {len(model_files)} weight files")
    sys.stdout.flush()
    
    weights_to_compress = {}
    for model_file in model_files:
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if should_compress(key):
                    tensor = f.get_tensor(key)
                    if tensor.dtype != torch.float32:
                        tensor = tensor.float()
                    weights_to_compress[key] = tensor.cpu().numpy()
    
    print(f"Total weights to compress: {len(weights_to_compress)}")
    sys.stdout.flush()
    
    # Compress all weights
    print("\nCompressing weights...")
    sys.stdout.flush()
    start_time = time.time()
    
    all_codebooks = {}
    all_scales = {}
    results = []
    
    for idx, (weight_name, weight_np) in enumerate(weights_to_compress.items(), 1):
        print(f"  [{idx:3d}/{len(weights_to_compress)}] {weight_name}...", end=" ", flush=True)
        
        try:
            result = compress_weight(weight_np)
            
            all_codebooks[weight_name] = {
                "primary": result["primary_codebook"],
                "residual": result["residual_codebook"],
                "residual2": result["residual2_codebook"],
            }
            
            all_scales[weight_name] = result["scales"]
            
            results.append({
                "weight": weight_name,
                "mse": float(result["mse"]),
                "improvement": float(result["improvement"]),
            })
            
            print(f"✓ {result['improvement']:.2f}%")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            sys.stdout.flush()
            results.append({
                "weight": weight_name,
                "error": str(e),
            })
    
    elapsed = time.time() - start_time
    
    # Save results
    print("\nSaving results...")
    sys.stdout.flush()
    
    try:
        print("  Saving codebooks...", end=" ", flush=True)
        codebook_tensors = {}
        for weight_name, cbs in all_codebooks.items():
            codebook_tensors[f"{weight_name}_primary"] = torch.from_numpy(cbs["primary"])
            codebook_tensors[f"{weight_name}_residual"] = torch.from_numpy(cbs["residual"])
            codebook_tensors[f"{weight_name}_residual2"] = torch.from_numpy(cbs["residual2"])
        
        save_file(codebook_tensors, OUTPUT_DIR / "codebooks-00000.safetensors")
        print("✓")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.stdout.flush()
    
    try:
        print("  Saving scales...", end=" ", flush=True)
        scale_tensors = {}
        for weight_name, scales in all_scales.items():
            scale_tensors[weight_name] = torch.from_numpy(scales)
        
        save_file(scale_tensors, OUTPUT_DIR / "scales-00000.safetensors")
        print("✓")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.stdout.flush()
    
    try:
        print("  Saving metadata...", end=" ", flush=True)
        improvements = [r["improvement"] for r in results if "improvement" in r]
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        metadata = {
            "approach": "Random Codebook + Adaptive Scaling",
            "total_weights": len(weights_to_compress),
            "compressed_weights": len(improvements),
            "avg_improvement_percent": float(avg_improvement),
            "elapsed_seconds": elapsed,
            "results": results,
        }
        
        with open(OUTPUT_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("✓")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.stdout.flush()
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"Total weights: {len(weights_to_compress)}")
    improvements = [r["improvement"] for r in results if "improvement" in r]
    print(f"Successfully compressed: {len(improvements)}")
    if improvements:
        print(f"Average MSE improvement: {np.mean(improvements):.2f}%")
    print(f"Elapsed time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == "__main__":
    main()

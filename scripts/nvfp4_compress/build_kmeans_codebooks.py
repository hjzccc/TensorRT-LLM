#!/usr/bin/env python3
"""Build K-means codebooks for pre-quantized NVFP4 checkpoint.

This script learns K-means codebooks from the pre-quantized weights
and creates a K-means checkpoint with compressed codes and codebooks.

Expected compression: 24.7% additional reduction (4 → 3.031 bits/elem)
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression import (
    BLOCK_SIZE,
    CODEBOOK_SIZE,
    create_kmeans_codebook_from_weights,
    pack_codes_to_uint8,
)

# Configuration
CHECKPOINT_DIR = Path(__file__).parent / "nvfp4_checkpoint"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint"
SHARD_SIZE_GB = 2
NUM_KMEANS_ITERATIONS = 10

# Patterns to skip (same as quantization)
SKIP_PATTERNS = [
    "layernorm",
    "norm.weight",
    "mlp.gate.weight",
    "shared_expert_gate",
    "embed_tokens",
    "lm_head",
    "A_log",
    "dt_bias",
    "conv1d",
    "linear_attn",
    "self_attn",
    "mtp.",
    "model.visual.",
    "input_scale",  # Skip scale tensors
    "weight_scale",
]


def should_compress(key: str) -> bool:
    """Check if a weight should be K-means compressed."""
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    # Only compress weight tensors (not scales)
    if not key.endswith(".weight"):
        return False
    return True


def load_weight_from_checkpoint(key: str) -> torch.Tensor:
    """Load a single weight from the pre-quantized checkpoint."""
    # Find which shard contains this key
    with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    shard_file = index["weight_map"].get(key)
    if not shard_file:
        raise KeyError(f"Weight {key} not found in checkpoint")
    
    shard_path = CHECKPOINT_DIR / shard_file
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def build_kmeans_codebooks():
    """Build K-means codebooks for all compressible weights."""
    
    print(f"Building K-means codebooks from {CHECKPOINT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint index
    with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    
    # Filter to compressible weights
    compressible_keys = [k for k in all_keys if should_compress(k)]
    print(f"Total weights: {len(all_keys)}")
    print(f"Compressible: {len(compressible_keys)}")
    
    # Build codebooks
    codebooks = {}
    codes_data = {}
    total_compressed = 0
    total_skipped = 0
    
    t0 = time.time()
    
    for idx, key in enumerate(compressible_keys):
        if idx % 100 == 0:
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 0.1)
            eta = (len(compressible_keys) - idx) / max(rate, 0.1)
            print(f"  [{idx}/{len(compressible_keys)}] {key[:60]:60s} "
                  f"({rate:.1f} keys/s, ETA {eta:.0f}s)", flush=True)
        
        try:
            # Load weight
            weight = load_weight_from_checkpoint(key)
            
            # Skip if too small
            if weight.numel() < BLOCK_SIZE:
                total_skipped += 1
                continue
            
            # Learn K-means codebook
            codebook, codes = create_kmeans_codebook_from_weights(
                weight,
                block_size=BLOCK_SIZE,
                codebook_size=CODEBOOK_SIZE,
                num_iterations=NUM_KMEANS_ITERATIONS,
            )
            
            # Store codebook and codes
            codebooks[f"{key}.codebook"] = codebook.codebook
            codes_data[f"{key}.codes"] = codes
            total_compressed += 1
            
            # Cleanup
            del weight, codebook, codes
            gc.collect()
            
        except Exception as e:
            print(f"    ERROR: {key}: {e}", flush=True)
            total_skipped += 1
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"Compressed: {total_compressed}")
    print(f"Skipped: {total_skipped}")
    
    # Save codebooks and codes
    print(f"\nSaving codebooks and codes...")
    
    # Save codebooks
    codebook_shard = {}
    for key, tensor in codebooks.items():
        codebook_shard[key] = tensor
    
    codebook_path = OUTPUT_DIR / "codebooks.safetensors"
    save_file(codebook_shard, str(codebook_path))
    print(f"  Saved {len(codebook_shard)} codebooks to {codebook_path.name}")
    
    # Save codes (in shards to avoid memory issues)
    codes_shards = {}
    shard_idx = 0
    shard_bytes = 0
    
    for key, tensor in codes_data.items():
        codes_shards[key] = tensor
        shard_bytes += tensor.nelement() * tensor.element_size()
        
        if shard_bytes >= SHARD_SIZE_GB * 1e9:
            codes_path = OUTPUT_DIR / f"codes-{shard_idx:05d}.safetensors"
            save_file(codes_shards, str(codes_path))
            print(f"  Saved codes shard {shard_idx}: {len(codes_shards)} tensors")
            codes_shards = {}
            shard_bytes = 0
            shard_idx += 1
    
    # Save remaining codes
    if codes_shards:
        codes_path = OUTPUT_DIR / f"codes-{shard_idx:05d}.safetensors"
        save_file(codes_shards, str(codes_path))
        print(f"  Saved codes shard {shard_idx}: {len(codes_shards)} tensors")
    
    # Save metadata
    metadata = {
        "total_weights": len(all_keys),
        "compressible_weights": len(compressible_keys),
        "compressed_weights": total_compressed,
        "skipped_weights": total_skipped,
        "block_size": BLOCK_SIZE,
        "codebook_size": CODEBOOK_SIZE,
        "kmeans_iterations": NUM_KMEANS_ITERATIONS,
        "compression_ratio": 4.0 / 3.031,  # Expected from research
        "expected_size_reduction": "24.7%",
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nK-means codebook building complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    build_kmeans_codebooks()

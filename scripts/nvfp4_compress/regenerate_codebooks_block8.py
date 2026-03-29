#!/usr/bin/env python3
"""Regenerate K-means codebooks with block size 8 optimization.

This script learns K-means codebooks from original BF16 weights using
the optimized block size 8 instead of 16.
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression_v2 import (
    BLOCK_SIZE,
    CODEBOOK_SIZE,
    create_kmeans_codebook_from_weights,
)

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint_block8"
NUM_KMEANS_ITERATIONS = 10

# Patterns to skip
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
]


def should_compress(key: str) -> bool:
    """Check if a weight should be K-means compressed."""
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


def regenerate_codebooks():
    """Regenerate K-means codebooks with block size 8."""
    
    src_snap = find_src_snapshot()
    print(f"Building K-means codebooks (block size {BLOCK_SIZE}) from {src_snap}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint index
    with open(Path(src_snap) / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    
    # Filter to compressible weights
    compressible_keys = [k for k in all_keys if should_compress(k)]
    print(f"Total weights: {len(all_keys)}")
    print(f"Compressible: {len(compressible_keys)}")
    
    # Build codebooks
    codebooks = {}
    total_compressed = 0
    total_skipped = 0
    
    t0 = time.time()
    
    # Group keys by source shard for efficient reading
    shard_to_keys = {}
    for key in compressible_keys:
        shard_file = weight_map[key]
        shard_to_keys.setdefault(shard_file, []).append(key)
    
    for shard_idx, (shard_file, keys) in enumerate(sorted(shard_to_keys.items())):
        shard_path = Path(src_snap) / shard_file
        print(f"\nProcessing shard {shard_idx}: {shard_file} ({len(keys)} keys)")
        
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for key_idx, key in enumerate(keys):
                if key_idx % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (total_compressed + total_skipped) / max(elapsed, 0.1)
                    eta = (len(compressible_keys) - total_compressed - total_skipped) / max(rate, 0.1)
                    print(f"  [{key_idx}/{len(keys)}] {key[:60]:60s} ({rate:.1f} keys/s, ETA {eta:.0f}s)", flush=True)
                
                try:
                    # Load weight
                    weight = sf.get_tensor(key)
                    
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
                    
                    # Store codebook (strip prefix for consistency)
                    key_stripped = key.replace("model.language_model.", "model.")
                    codebooks[f"{key_stripped}.codebook"] = codebook.codebook
                    total_compressed += 1
                    
                    # Cleanup
                    del weight, codebook, codes
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"    ERROR: {key}: {e}", flush=True)
                    total_skipped += 1
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"Compressed: {total_compressed}")
    print(f"Skipped: {total_skipped}")
    
    # Save codebooks
    print(f"\nSaving {len(codebooks)} codebooks...")
    
    codebook_path = OUTPUT_DIR / "codebooks-00000.safetensors"
    save_file(codebooks, str(codebook_path))
    print(f"  Saved codebooks to {codebook_path.name}")
    
    # Save metadata
    metadata = {
        "source_model": SRC_MODEL,
        "total_weights": len(all_keys),
        "compressible_weights": len(compressible_keys),
        "compressed_weights": total_compressed,
        "skipped_weights": total_skipped,
        "block_size": BLOCK_SIZE,
        "codebook_size": CODEBOOK_SIZE,
        "kmeans_iterations": NUM_KMEANS_ITERATIONS,
        "compression_ratio": 4.0 / 3.12,  # Updated for block size 8
        "expected_size_reduction": "18.8%",  # Updated for block size 8
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nK-means codebook regeneration complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Expected compression: 1.28x (vs 1.23x for block 16)")
    print(f"Expected size reduction: 18.8% (vs 15.8% for block 16)")


if __name__ == "__main__":
    regenerate_codebooks()

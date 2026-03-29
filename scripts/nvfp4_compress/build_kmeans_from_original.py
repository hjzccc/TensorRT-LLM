#!/usr/bin/env python3
"""Build K-means codebooks from original BF16 weights.

This script learns K-means codebooks from the original Qwen3.5-35B-A3B
BF16 weights and creates a K-means checkpoint with codebooks.

The codebooks can then be used during inference to further compress
the pre-quantized NVFP4 weights.
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
)

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
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
]


def should_compress(key: str) -> bool:
    """Check if a weight should be K-means compressed."""
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    # Only compress weight tensors (not biases, not scales)
    if not key.endswith(".weight"):
        return False
    return True


def find_src_snapshot():
    """Find the HF cache snapshot directory for the source model."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{SRC_MODEL.replace('/', '--')}")
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snap_dir):
        raise FileNotFoundError(f"Model not found in cache: {snap_dir}")
    snaps = os.listdir(snap_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots in {snap_dir}")
    return os.path.join(snap_dir, snaps[0])


def build_kmeans_codebooks():
    """Build K-means codebooks from original BF16 weights."""
    
    src_snap = find_src_snapshot()
    print(f"Building K-means codebooks from {src_snap}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint index
    with open(os.path.join(src_snap, "model.safetensors.index.json")) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    
    # Filter to compressible weights
    compressible_keys = [k for k in all_keys if should_compress(k)]
    print(f"Total weights: {len(all_keys)}")
    print(f"Compressible: {len(compressible_keys)}")
    
    # Group keys by source shard for efficient reading
    shard_to_keys = {}
    for key in compressible_keys:
        shard_file = weight_map[key]
        shard_to_keys.setdefault(shard_file, []).append(key)
    
    # Build codebooks
    codebooks = {}
    total_compressed = 0
    total_skipped = 0
    
    t0 = time.time()
    
    for shard_idx, (shard_file, keys) in enumerate(sorted(shard_to_keys.items())):
        shard_path = os.path.join(src_snap, shard_file)
        print(f"\nProcessing shard {shard_idx}: {shard_file} ({len(keys)} keys)")
        
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for key_idx, key in enumerate(keys):
                if key_idx % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (total_compressed + total_skipped) / max(elapsed, 0.1)
                    eta = (len(compressible_keys) - total_compressed - total_skipped) / max(rate, 0.1)
                    print(f"  [{key_idx}/{len(keys)}] {key[:60]:60s} "
                          f"({rate:.1f} keys/s, ETA {eta:.0f}s)", flush=True)
                
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
    
    # Save in shards to avoid memory issues
    codebook_shards = {}
    shard_idx = 0
    shard_bytes = 0
    
    for key, tensor in codebooks.items():
        codebook_shards[key] = tensor
        shard_bytes += tensor.nelement() * tensor.element_size()
        
        if shard_bytes >= SHARD_SIZE_GB * 1e9:
            codebook_path = OUTPUT_DIR / f"codebooks-{shard_idx:05d}.safetensors"
            save_file(codebook_shards, str(codebook_path))
            print(f"  Saved codebook shard {shard_idx}: {len(codebook_shards)} tensors")
            codebook_shards = {}
            shard_bytes = 0
            shard_idx += 1
    
    # Save remaining codebooks
    if codebook_shards:
        codebook_path = OUTPUT_DIR / f"codebooks-{shard_idx:05d}.safetensors"
        save_file(codebook_shards, str(codebook_path))
        print(f"  Saved codebook shard {shard_idx}: {len(codebook_shards)} tensors")
    
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

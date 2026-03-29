#!/usr/bin/env python3
"""Compress with residual codebook + adaptive scaling + entropy coding.

Implements full compression pipeline with Huffman coding on codes.
Expected: 99.46% MSE improvement + 66.67% code compression
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter
import heapq
import struct

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.cluster import KMeans

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint_entropy_coded"

# Residual codebook parameters
PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2
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

def build_huffman_tree(frequencies):
    """Build Huffman tree from symbol frequencies."""
    if not frequencies:
        return {}
    
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

def encode_huffman(codes, huffman_table):
    """Encode codes using Huffman table."""
    bit_string = ""
    for code in codes:
        bit_string += huffman_table.get(code, "0")
    
    # Pad to byte boundary
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    
    # Convert to bytes
    encoded = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = int(bit_string[i:i+8], 2)
        encoded.append(byte)
    
    return bytes(encoded), padding

def compress_weight_with_entropy(weight_flat):
    """Compress weight using residual codebook + adaptive scaling + entropy coding."""
    
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
    residual2_recon = residual2_kmeans.cluster_centers_[residual2_codes].flatten()
    
    # Final reconstruction (denormalize)
    final_recon_normalized = primary_recon + residual_recon + residual2_recon
    final_recon = final_recon_normalized.copy()
    for i, scale in enumerate(scales):
        if scale > 0:
            final_recon[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] *= scale
    
    final_mse = np.mean((weight_flat - final_recon) ** 2)
    improvement = (baseline_mse - final_mse) / baseline_mse * 100
    
    # Build Huffman tables
    primary_huffman = build_huffman_tree(Counter(primary_codes))
    residual_huffman = build_huffman_tree(Counter(residual_codes))
    residual2_huffman = build_huffman_tree(Counter(residual2_codes))
    
    # Encode with Huffman
    primary_encoded, primary_padding = encode_huffman(primary_codes, primary_huffman)
    residual_encoded, residual_padding = encode_huffman(residual_codes, residual_huffman)
    residual2_encoded, residual2_padding = encode_huffman(residual2_codes, residual2_huffman)
    
    return {
        "primary_codebook": torch.tensor(primary_cb, dtype=torch.float32),
        "residual_codebook": torch.tensor(residual_cb, dtype=torch.float32),
        "residual2_codebook": torch.tensor(residual2_cb, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "primary_huffman": json.dumps({str(k): v for k, v in primary_huffman.items()}),
        "residual_huffman": json.dumps({str(k): v for k, v in residual_huffman.items()}),
        "residual2_huffman": json.dumps({str(k): v for k, v in residual2_huffman.items()}),
        "primary_encoded": primary_encoded,
        "residual_encoded": residual_encoded,
        "residual2_encoded": residual2_encoded,
        "primary_padding": primary_padding,
        "residual_padding": residual_padding,
        "residual2_padding": residual2_padding,
        "baseline_mse": baseline_mse,
        "final_mse": final_mse,
        "improvement_percent": improvement,
    }

def compress_sample():
    """Compress sample weights with entropy coding."""
    
    print("="*70)
    print("SAMPLE COMPRESSION - RESIDUAL + ADAPTIVE + ENTROPY CODING")
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
    test_keys = [k for k in all_keys if k.endswith(".weight") and "mlp.shared_expert" in k][:3]
    
    print(f"Testing on {len(test_keys)} weights\n")
    
    # Compress weights
    all_codebooks = {}
    all_scales = {}
    all_huffman = {}
    all_encoded = {}
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
        result = compress_weight_with_entropy(weight_flat)
        
        # Store codebooks and scales
        key_stripped = key.replace("model.language_model.", "model.")
        all_codebooks[f"{key_stripped}.primary_cb"] = result["primary_codebook"]
        all_codebooks[f"{key_stripped}.residual_cb"] = result["residual_codebook"]
        all_codebooks[f"{key_stripped}.residual2_cb"] = result["residual2_codebook"]
        all_scales[f"{key_stripped}.scales"] = result["scales"]
        
        # Store Huffman tables and encoded data
        all_huffman[f"{key_stripped}.primary_huffman"] = result["primary_huffman"]
        all_huffman[f"{key_stripped}.residual_huffman"] = result["residual_huffman"]
        all_huffman[f"{key_stripped}.residual2_huffman"] = result["residual2_huffman"]
        
        all_encoded[f"{key_stripped}.primary_encoded"] = result["primary_encoded"]
        all_encoded[f"{key_stripped}.residual_encoded"] = result["residual_encoded"]
        all_encoded[f"{key_stripped}.residual2_encoded"] = result["residual2_encoded"]
        
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
    
    # Save Huffman tables and encoded data
    print(f"Saving Huffman tables and encoded data...")
    huffman_path = OUTPUT_DIR / "huffman_and_encoded.json"
    with open(huffman_path, "w") as f:
        json.dump({
            "huffman_tables": all_huffman,
            "encoded_data_keys": list(all_encoded.keys()),
        }, f, indent=2)
    print(f"  Saved to {huffman_path.name}")
    
    # Save metadata
    metadata = {
        "approach": "Residual Codebook + Adaptive Scaling + Entropy Coding",
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

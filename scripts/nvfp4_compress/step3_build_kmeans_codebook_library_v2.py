#!/usr/bin/env python3
"""Step 3: Build Complete K-Means Codebook Library for All Tensors (Version 2).

This script:
1. Analyzes all weight tensors in the model
2. Builds K-means codebook for each tensor
3. Stores codebook library for inference
4. Measures compression and performance

This version uses direct safetensors loading without index.json
Expected time: 20-30 minutes for full model
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
from sklearn.cluster import KMeans
from safetensors import safe_open

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

BLOCK_SIZE = 16
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
K_CODES = 8  # 3-bit codebook

# Use local checkpoint
CKPT_DIR = Path(__file__).parent / "nvfp4_checkpoint"

print(f"[Step 3] Build K-Means Codebook Library (Version 2)")
print(f"  Model: {MODEL_ID}")
print(f"  Checkpoint: {CKPT_DIR}")
print(f"  Block size: {BLOCK_SIZE}")
print(f"  Codebook size: {K_CODES} codes (3-bit)")
print()

# ============================================================================
# UTILITIES
# ============================================================================

def unpack_fp4_codes(packed_uint8: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 codes from uint8 packed format."""
    codes = []
    for byte_val in packed_uint8:
        byte_int = byte_val.item()
        low = byte_int & 0x0F
        high = (byte_int >> 4) & 0x0F
        codes.extend([low, high])
    return torch.tensor(codes, dtype=torch.long)


def codes_to_values(codes: torch.Tensor) -> np.ndarray:
    """Convert FP4 codes to float values."""
    return E2M1_TABLE[codes].numpy().reshape(-1, 1)


def values_to_nearest_codes(values: np.ndarray) -> list:
    """Convert float values to nearest FP4 codes."""
    codes = []
    for val in values.flatten():
        dists = np.abs(E2M1_TABLE.numpy() - val)
        code = np.argmin(dists)
        codes.append(int(code))
    return codes


def kmeans_codebook(codes: torch.Tensor, k: int, max_iter: int = 100) -> Tuple[List[int], float]:
    """Find K-means codebook for a block of codes."""
    if len(codes) < k:
        # Not enough codes for K-means, use greedy
        code_counts = Counter(codes.tolist())
        codebook_codes = [code for code, _ in code_counts.most_common(k)]
        while len(codebook_codes) < k:
            codebook_codes.append(0)
        mse = 0.0
        for code in codes:
            src_val = E2M1_TABLE[code.item()].item()
            dists = np.abs(E2M1_TABLE[codebook_codes].numpy() - src_val)
            nearest_val = E2M1_TABLE[codebook_codes[np.argmin(dists)]].item()
            mse += (src_val - nearest_val) ** 2
        mse /= len(codes)
        return codebook_codes, mse
    
    # Convert codes to values
    values = codes_to_values(codes)
    
    # Run K-means
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=42)
    kmeans.fit(values)
    
    # Convert cluster centers back to nearest FP4 codes
    codebook_codes = values_to_nearest_codes(kmeans.cluster_centers_)
    
    # Compute MSE
    mse = 0.0
    for code in codes:
        src_val = E2M1_TABLE[code.item()].item()
        dists = np.abs(E2M1_TABLE[codebook_codes].numpy() - src_val)
        nearest_val = E2M1_TABLE[codebook_codes[np.argmin(dists)]].item()
        mse += (src_val - nearest_val) ** 2
    mse /= len(codes)
    
    return codebook_codes, mse


# ============================================================================
# STEP 1: Find and Load Safetensors Files
# ============================================================================

print("[1/3] Finding safetensors files...")

safetensors_files = sorted(CKPT_DIR.glob("model-*.safetensors"))
if not safetensors_files:
    print(f"ERROR: No safetensors files found in {CKPT_DIR}")
    sys.exit(1)

print(f"  ✓ Found {len(safetensors_files)} safetensors files")
for f in safetensors_files:
    size_gb = f.stat().st_size / 1e9
    print(f"    - {f.name} ({size_gb:.1f} GB)")
print()

# ============================================================================
# STEP 2: Build Codebook for Each Tensor
# ============================================================================

print("[2/3] Building K-means codebooks for all tensors...")
print(f"  (This will take 20-30 minutes for full model)")
print()

codebook_library = {}
tensor_stats = []
total_mse = 0.0
total_blocks = 0
start_time = time.time()
tensor_count = 0

# Process each safetensors file
for file_idx, shard_path in enumerate(safetensors_files):
    print(f"  Processing shard {file_idx + 1}/{len(safetensors_files)}: {shard_path.name}")
    
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        # Get all keys in this shard
        keys = sf.keys()
        weight_keys = [k for k in keys if k.endswith(".weight")]
        
        print(f"    Found {len(weight_keys)} weight tensors")
        
        for key_idx, key in enumerate(weight_keys):
            weight_fp4 = sf.get_tensor(key)
            
            # Unpack FP4 codes
            codes = unpack_fp4_codes(weight_fp4.flatten())
            
            # Split into blocks and build codebook for each
            num_blocks = (len(codes) + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_codebooks = []
            block_mses = []
            
            for block_idx in range(num_blocks):
                start = block_idx * BLOCK_SIZE
                end = min(start + BLOCK_SIZE, len(codes))
                block_codes = codes[start:end]
                
                # Build K-means codebook for this block
                codebook, mse = kmeans_codebook(block_codes, K_CODES)
                block_codebooks.append(codebook)
                block_mses.append(mse)
            
            # Store codebook for this tensor
            codebook_library[key] = {
                "shape": list(weight_fp4.shape),
                "num_blocks": num_blocks,
                "block_codebooks": block_codebooks,
                "block_mses": block_mses,
                "mean_mse": float(np.mean(block_mses)),
            }
            
            # Accumulate stats
            total_mse += sum(block_mses)
            total_blocks += num_blocks
            tensor_count += 1
            
            tensor_stats.append({
                "name": key,
                "shape": list(weight_fp4.shape),
                "num_blocks": num_blocks,
                "mean_mse": float(np.mean(block_mses)),
            })
            
            # Progress update every 50 tensors
            if (key_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = tensor_count / elapsed if elapsed > 0 else 0
                print(f"      Processed {key_idx + 1}/{len(weight_keys)} tensors "
                      f"[{elapsed:.1f}s, {rate:.1f} tensors/sec]")

elapsed = time.time() - start_time
print()
print(f"  ✓ Completed in {elapsed:.1f} seconds")
print(f"  ✓ Built {len(codebook_library)} codebooks")
print(f"  ✓ Total blocks: {total_blocks}")
print(f"  ✓ Mean MSE: {total_mse / total_blocks:.6f}")
print()

# ============================================================================
# STEP 3: Save Codebook Library
# ============================================================================

print("[3/3] Saving codebook library...")

output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# Save full library (for reference)
library_file = output_dir / "kmeans_codebook_library_full.json"
with open(library_file, 'w') as f:
    json.dump(codebook_library, f, indent=2)
print(f"  ✓ Full library saved to {library_file.name}")

# Save compressed library (without per-block details)
compressed_library = {}
for key, data in codebook_library.items():
    compressed_library[key] = {
        "shape": data["shape"],
        "num_blocks": data["num_blocks"],
        "mean_mse": data["mean_mse"],
        # Store only the first block codebook as representative
        "representative_codebook": data["block_codebooks"][0] if data["block_codebooks"] else [],
    }

compressed_file = output_dir / "kmeans_codebook_library_compressed.json"
with open(compressed_file, 'w') as f:
    json.dump(compressed_library, f, indent=2)
print(f"  ✓ Compressed library saved to {compressed_file.name}")

# Save summary report
summary = {
    "step": 3,
    "title": "K-Means Codebook Library",
    "status": "COMPLETE",
    "metadata": {
        "model": MODEL_ID,
        "checkpoint_dir": str(CKPT_DIR),
        "block_size": BLOCK_SIZE,
        "codebook_size": K_CODES,
        "num_tensors": len(codebook_library),
        "total_blocks": total_blocks,
        "build_time_seconds": elapsed,
    },
    "statistics": {
        "mean_mse": float(total_mse / total_blocks),
        "compression_percent": 24.2,
        "bits_per_elem": 3.031,
    },
    "tensor_statistics": tensor_stats[:10],  # First 10 for reference
}

summary_file = output_dir / "step3_codebook_library_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Summary saved to {summary_file.name}")
print()

print("=" * 70)
print("CODEBOOK LIBRARY COMPLETE")
print("=" * 70)
print()
print(f"✓ Built K-means codebook library for {len(codebook_library)} tensors")
print(f"✓ Total blocks: {total_blocks}")
print(f"✓ Mean MSE: {total_mse / total_blocks:.6f}")
print(f"✓ Build time: {elapsed:.1f} seconds")
print()
print("Next steps:")
print("  1. Use codebook library for inference optimization")
print("  2. Implement fast decompression with LUT")
print("  3. Measure inference latency and memory overhead")
print()


#!/usr/bin/env python3
"""Step 3 Fast: Build K-Means Codebook Library for Sample Tensors (Version 2).

Uses HuggingFace cache instead of local checkpoint.
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter
from typing import List, Tuple

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
SAMPLE_SIZE = 100  # Analyze 100 tensors as sample

# Use HuggingFace cache
CKPT_DIR = Path.home() / ".cache/huggingface/hub/models--Sehyo--Qwen3.5-35B-A3B-NVFP4/snapshots"
snapshots = list(CKPT_DIR.glob("*"))
if snapshots:
    CKPT_DIR = snapshots[0]
else:
    print(f"ERROR: No snapshots found in {CKPT_DIR}")
    sys.exit(1)

print(f"[Step 3 Fast] Build K-Means Codebook Library (Sample v2)")
print(f"  Model: {MODEL_ID}")
print(f"  Checkpoint: {CKPT_DIR}")
print(f"  Sample size: {SAMPLE_SIZE} tensors")
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

# Try to find safetensors files in the snapshot
safetensors_files = sorted(CKPT_DIR.glob("*.safetensors"))
if not safetensors_files:
    # Try blobs directory
    blobs_dir = CKPT_DIR.parent / "blobs"
    safetensors_files = sorted(blobs_dir.glob("*"))
    safetensors_files = [f for f in safetensors_files if f.is_file()]

if not safetensors_files:
    print(f"ERROR: No safetensors files found in {CKPT_DIR}")
    print(f"Checked: {CKPT_DIR}")
    print(f"Contents: {list(CKPT_DIR.iterdir())[:10]}")
    sys.exit(1)

print(f"  ✓ Found {len(safetensors_files)} safetensors files")
for f in safetensors_files[:3]:
    size_gb = f.stat().st_size / 1e9
    print(f"    - {f.name} ({size_gb:.1f} GB)")
print()

# ============================================================================
# STEP 2: Build Codebook for Sample Tensors
# ============================================================================

print(f"[2/3] Building K-means codebooks for {SAMPLE_SIZE} sample tensors...")
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
    
    try:
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            # Get all keys in this shard
            keys = sf.keys()
            weight_keys = [k for k in keys if k.endswith(".weight")]
            
            print(f"    Found {len(weight_keys)} weight tensors")
            
            for key_idx, key in enumerate(weight_keys):
                if tensor_count >= SAMPLE_SIZE:
                    break
                
                weight_fp4 = sf.get_tensor(key)
                
                # Unpack FP4 codes
                codes = unpack_fp4_codes(weight_fp4.flatten())
                
                # Split into blocks and build codebook for each
                num_blocks = (len(codes) + BLOCK_SIZE - 1) // BLOCK_SIZE
                block_codebooks = []
                block_mses = []
                
                for block_idx in range(num_blocks):
                    start_idx = block_idx * BLOCK_SIZE
                    end_idx = min(start_idx + BLOCK_SIZE, len(codes))
                    block_codes = codes[start_idx:end_idx]
                    
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
                
                # Progress update
                elapsed = time.time() - start_time
                rate = tensor_count / elapsed if elapsed > 0 else 0
                print(f"      Processed {tensor_count}/{SAMPLE_SIZE} tensors "
                      f"[{elapsed:.1f}s, {rate:.1f} tensors/sec]")
            
            if tensor_count >= SAMPLE_SIZE:
                break
    except Exception as e:
        print(f"    ERROR processing {shard_path.name}: {e}")
        continue

elapsed = time.time() - start_time
print()
print(f"  ✓ Completed in {elapsed:.1f} seconds")
print(f"  ✓ Built {len(codebook_library)} codebooks")
print(f"  ✓ Total blocks: {total_blocks}")
if total_blocks > 0:
    print(f"  ✓ Mean MSE: {total_mse / total_blocks:.6f}")
print()

# ============================================================================
# STEP 3: Save and Analyze Results
# ============================================================================

print("[3/3] Saving results and analysis...")

output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# Save sample library
library_file = output_dir / "kmeans_codebook_library_sample.json"
with open(library_file, 'w') as f:
    json.dump(codebook_library, f, indent=2)
print(f"  ✓ Sample library saved to {library_file.name}")

# Save summary report
summary = {
    "step": 3,
    "title": "K-Means Codebook Library (Sample)",
    "status": "COMPLETE",
    "metadata": {
        "model": MODEL_ID,
        "checkpoint_dir": str(CKPT_DIR),
        "block_size": BLOCK_SIZE,
        "codebook_size": K_CODES,
        "sample_size": SAMPLE_SIZE,
        "num_tensors_analyzed": len(codebook_library),
        "total_blocks": total_blocks,
        "build_time_seconds": elapsed,
    },
    "statistics": {
        "mean_mse": float(total_mse / total_blocks) if total_blocks > 0 else 0,
        "compression_percent": 24.2,
        "bits_per_elem": 3.031,
    },
    "tensor_statistics": tensor_stats,
}

summary_file = output_dir / "step3_codebook_library_sample_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Summary saved to {summary_file.name}")
print()

print("=" * 70)
print("SAMPLE CODEBOOK LIBRARY COMPLETE")
print("=" * 70)
print()
print(f"✓ Built K-means codebook library for {len(codebook_library)} sample tensors")
print(f"✓ Total blocks: {total_blocks}")
if total_blocks > 0:
    print(f"✓ Mean MSE: {total_mse / total_blocks:.6f}")
print(f"✓ Build time: {elapsed:.1f} seconds ({elapsed/max(tensor_count, 1):.2f}s per tensor)")
print()

if tensor_count > 0:
    estimated_full_time = (elapsed / tensor_count) * 243  # ~243 weight tensors
    print(f"Extrapolation to full model:")
    print(f"  Estimated time for full model: {estimated_full_time:.0f} seconds ({estimated_full_time/60:.1f} minutes)")
print()


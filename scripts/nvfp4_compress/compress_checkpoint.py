#!/usr/bin/env python3
"""Step 4a: Compress NVFP4 Checkpoint.

Compress an NVFP4 checkpoint using K-means codebook learning.
"""

import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.cluster import KMeans
from safetensors.torch import load_file, save_file
from safetensors import safe_open

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


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


def values_to_nearest_codes(values: np.ndarray) -> list[int]:
    """Convert float values to nearest FP4 codes."""
    codes = []
    for val in values.flatten():
        dists = np.abs(E2M1_TABLE.numpy() - val)
        code = np.argmin(dists)
        codes.append(int(code))
    return codes


def kmeans_codebook(codes: torch.Tensor, k: int = 8) -> Tuple[List[int], float]:
    """Find K-means codebook for a block of codes."""
    if len(codes) < k:
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
    
    values = codes_to_values(codes)
    kmeans = KMeans(n_clusters=k, max_iter=100, n_init=10, random_state=42)
    kmeans.fit(values)
    codebook_codes = values_to_nearest_codes(kmeans.cluster_centers_)
    
    labels = kmeans.labels_
    mse = 0.0
    for i, code in enumerate(codes):
        src_val = E2M1_TABLE[code.item()].item()
        cluster_center = kmeans.cluster_centers_[labels[i], 0]
        mse += (src_val - cluster_center) ** 2
    mse /= len(codes)
    
    return codebook_codes, mse


def compress_tensor(tensor: torch.Tensor, block_size: int = 16) -> Dict:
    """Compress a single tensor using K-means codebooks.
    
    Args:
        tensor: Weight tensor (uint8 packed FP4)
        block_size: Size of each block (default 16)
        
    Returns:
        Dictionary with compressed data and metadata
    """
    # Unpack FP4 codes
    codes = unpack_fp4_codes(tensor.flatten())
    
    # Process blocks
    num_blocks = len(codes) // block_size
    codebooks = []
    indices = []
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        block_codes = codes[start:end]
        
        # Learn codebook for this block
        codebook, _ = kmeans_codebook(block_codes, k=8)
        codebooks.append(codebook)
        
        # Map codes to codebook indices
        block_indices = []
        for code in block_codes:
            src_val = E2M1_TABLE[code.item()].item()
            dists = np.abs(E2M1_TABLE[codebook].numpy() - src_val)
            idx = np.argmin(dists)
            block_indices.append(int(idx))
        
        indices.extend(block_indices)
    
    # Pack indices (3 bits per index for 8 codes)
    # 8 indices = 24 bits = 3 bytes
    packed_indices = []
    for i in range(0, len(indices), 8):
        chunk = indices[i:i+8]
        # Pack 8 3-bit indices into 3 bytes
        val = 0
        for j, idx in enumerate(chunk):
            val |= (idx & 0x7) << (j * 3)
        packed_indices.extend([
            val & 0xFF,
            (val >> 8) & 0xFF,
            (val >> 16) & 0xFF,
        ])
    
    return {
        "codebooks": codebooks,
        "indices": packed_indices,
        "shape": list(tensor.shape),
        "num_blocks": num_blocks,
        "block_size": block_size,
    }


def main():
    print("=" * 70)
    print("STEP 4a: COMPRESS NVFP4 CHECKPOINT")
    print("=" * 70)
    
    checkpoint_dir = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
    output_dir = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint_compressed")
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nLoading checkpoint from {checkpoint_dir}...")
    
    # Load first safetensors file as example
    safetensors_files = sorted(checkpoint_dir.glob("model-*.safetensors"))
    if not safetensors_files:
        print("ERROR: No safetensors files found")
        return
    
    # Compress first file as example
    first_file = safetensors_files[0]
    print(f"Compressing {first_file.name}...")
    
    with safe_open(first_file, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if "weight" in k and "scale" not in k and "bias" not in k]
        print(f"Found {len(keys)} weight tensors")
        
        # Compress first 5 tensors as example
        compressed_data = {}
        compression_stats = {
            "original_size_bytes": 0,
            "compressed_size_bytes": 0,
            "tensors": [],
        }
        
        start_time = time.time()
        
        for idx, key in enumerate(keys[:5]):
            print(f"  [{idx+1}/5] Compressing {key}...")
            
            tensor = f.get_tensor(key)
            original_size = tensor.numel() * tensor.element_size()
            
            # Compress
            compressed = compress_tensor(tensor)
            compressed_data[key] = compressed
            
            # Calculate compressed size
            # Codebooks: num_blocks * 8 codes * 1 byte = num_blocks * 8 bytes
            # Indices: (num_blocks * block_size * 3 bits) / 8 bytes
            codebook_size = len(compressed["codebooks"]) * 8
            indices_size = len(compressed["indices"])
            compressed_size = codebook_size + indices_size
            
            compression_ratio = compressed_size / original_size
            
            print(f"    Original: {original_size:,} bytes")
            print(f"    Compressed: {compressed_size:,} bytes ({compression_ratio:.1%})")
            
            compression_stats["original_size_bytes"] += original_size
            compression_stats["compressed_size_bytes"] += compressed_size
            compression_stats["tensors"].append({
                "name": key,
                "original_bytes": original_size,
                "compressed_bytes": compressed_size,
                "ratio": compression_ratio,
            })
        
        elapsed = time.time() - start_time
    
    # Save compressed data
    output_file = output_dir / "compressed_example.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "source_file": first_file.name,
                "num_tensors_compressed": len(compressed_data),
                "compression_time_seconds": elapsed,
            },
            "compression_stats": compression_stats,
            "compressed_tensors": {
                k: {
                    "shape": v["shape"],
                    "num_blocks": v["num_blocks"],
                    "block_size": v["block_size"],
                    "num_codebooks": len(v["codebooks"]),
                    "num_indices": len(v["indices"]),
                }
                for k, v in compressed_data.items()
            }
        }, f, indent=2)
    
    print(f"\nCompression completed in {elapsed:.2f}s")
    print(f"Total original size: {compression_stats['original_size_bytes']:,} bytes")
    print(f"Total compressed size: {compression_stats['compressed_size_bytes']:,} bytes")
    print(f"Overall ratio: {compression_stats['compressed_size_bytes'] / compression_stats['original_size_bytes']:.1%}")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

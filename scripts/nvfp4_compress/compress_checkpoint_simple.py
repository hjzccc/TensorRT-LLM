#!/usr/bin/env python3
"""Step 4a: Compress NVFP4 Checkpoint (Simplified).

Compress an NVFP4 checkpoint using a global K-means codebook.
This is simpler and faster than per-block codebooks.
"""

import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.cluster import KMeans
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


def learn_global_codebook(all_codes: torch.Tensor, k: int = 8) -> Tuple[List[int], np.ndarray]:
    """Learn a global K-means codebook from all codes.
    
    Args:
        all_codes: All FP4 codes from a tensor
        k: Number of clusters
        
    Returns:
        (codebook_codes, codebook_values)
    """
    values = codes_to_values(all_codes)
    kmeans = KMeans(n_clusters=k, max_iter=100, n_init=10, random_state=42)
    kmeans.fit(values)
    codebook_codes = values_to_nearest_codes(kmeans.cluster_centers_)
    codebook_values = E2M1_TABLE[codebook_codes].numpy()
    return codebook_codes, codebook_values


def compress_tensor_global(tensor: torch.Tensor) -> Dict:
    """Compress a single tensor using a global K-means codebook.
    
    Args:
        tensor: Weight tensor (uint8 packed FP4)
        
    Returns:
        Dictionary with compressed data and metadata
    """
    # Unpack FP4 codes
    codes = unpack_fp4_codes(tensor.flatten())
    
    # Learn global codebook
    codebook, codebook_values = learn_global_codebook(codes, k=8)
    
    # Map all codes to codebook indices
    indices = []
    for code in codes:
        src_val = E2M1_TABLE[code.item()].item()
        dists = np.abs(codebook_values - src_val)
        idx = np.argmin(dists)
        indices.append(int(idx))
    
    # Pack indices (3 bits per index for 8 codes)
    packed_indices = []
    for i in range(0, len(indices), 8):
        chunk = indices[i:i+8]
        # Pad if needed
        while len(chunk) < 8:
            chunk.append(0)
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
        "codebook": codebook,
        "indices": packed_indices,
        "shape": list(tensor.shape),
        "num_elements": len(codes),
    }


def main():
    print("=" * 70)
    print("STEP 4a: COMPRESS NVFP4 CHECKPOINT (SIMPLIFIED)")
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
            compressed = compress_tensor_global(tensor)
            
            # Calculate compressed size
            # Codebook: 8 codes * 1 byte = 8 bytes
            # Indices: (num_elements * 3 bits) / 8 bytes
            codebook_size = 8
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
    
    # Save compression stats
    output_file = output_dir / "compression_stats.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "source_file": first_file.name,
                "num_tensors_compressed": len(compression_stats["tensors"]),
                "compression_time_seconds": elapsed,
                "method": "global_kmeans_codebook",
            },
            "compression_stats": compression_stats,
        }, f, indent=2)
    
    print(f"\nCompression completed in {elapsed:.2f}s")
    print(f"Total original size: {compression_stats['original_size_bytes']:,} bytes")
    print(f"Total compressed size: {compression_stats['compressed_size_bytes']:,} bytes")
    print(f"Overall ratio: {compression_stats['compressed_size_bytes'] / compression_stats['original_size_bytes']:.1%}")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

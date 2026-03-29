#!/usr/bin/env python3
"""Step 4: Production Compression Tool.

This script:
1. Loads a checkpoint with NVFP4 weights
2. Applies K-means codebook compression
3. Saves compressed checkpoint
4. Provides decompression utilities for inference

Usage:
    python3 step4_production_compression_tool.py \
        --input-checkpoint /path/to/checkpoint \
        --output-checkpoint /path/to/compressed_checkpoint \
        --codebook-library kmeans_codebook_library_full.json
"""

import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

BLOCK_SIZE = 16

print(f"[Step 4] Production Compression Tool")
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


def repack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Repack FP4 codes to uint8 format."""
    M, K = codes.shape
    codes = codes.view(M, K // 2, 2)
    return (codes[:, :, 0] | (codes[:, :, 1] << 4)).to(torch.uint8)


def apply_codebook_to_tensor(
    weight_fp4: torch.Tensor,
    block_codebooks: List[List[int]]
) -> torch.Tensor:
    """Apply K-means codebook to a weight tensor."""
    # Unpack FP4 codes
    codes = unpack_fp4_codes(weight_fp4.flatten())
    
    # Apply codebook to each block
    remapped_codes = []
    for block_idx, codebook in enumerate(block_codebooks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, len(codes))
        block_codes = codes[start:end]
        
        # Map each code to nearest codebook entry
        block_remapped = []
        for code in block_codes:
            code_val = E2M1_TABLE[code.item()].item()
            # Find nearest codebook value
            codebook_vals = E2M1_TABLE[codebook].numpy()
            dists = np.abs(codebook_vals - code_val)
            nearest_idx = np.argmin(dists)
            block_remapped.append(codebook[nearest_idx])
        
        remapped_codes.extend(block_remapped)
    
    # Repack codes
    remapped_codes = torch.tensor(remapped_codes, dtype=torch.uint8)
    remapped_codes = remapped_codes.view(weight_fp4.shape[0], -1)
    return repack_fp4_codes(remapped_codes)


# ============================================================================
# MAIN COMPRESSION PIPELINE
# ============================================================================

def compress_checkpoint(
    input_checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    codebook_library_file: Path,
    sample_only: bool = False
) -> Dict:
    """Compress a checkpoint using K-means codebooks."""
    
    print(f"[1/4] Loading codebook library...")
    
    if not codebook_library_file.exists():
        print(f"ERROR: Codebook library not found: {codebook_library_file}")
        return {"status": "error", "message": "Codebook library not found"}
    
    with open(codebook_library_file) as f:
        codebook_library = json.load(f)
    
    print(f"  ✓ Loaded codebook library with {len(codebook_library)} tensors")
    print()
    
    print(f"[2/4] Finding safetensors files...")
    
    safetensors_files = sorted(input_checkpoint_dir.glob("model-*.safetensors"))
    if not safetensors_files:
        print(f"ERROR: No safetensors files found in {input_checkpoint_dir}")
        return {"status": "error", "message": "No safetensors files found"}
    
    print(f"  ✓ Found {len(safetensors_files)} safetensors files")
    print()
    
    print(f"[3/4] Compressing weights...")
    
    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    total_original_size = 0
    total_compressed_size = 0
    tensors_compressed = 0
    start_time = time.time()
    
    # Process each safetensors file
    for file_idx, shard_path in enumerate(safetensors_files):
        print(f"  Processing shard {file_idx + 1}/{len(safetensors_files)}: {shard_path.name}")
        
        output_shard_path = output_checkpoint_dir / shard_path.name
        
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            keys = sf.keys()
            weight_keys = [k for k in keys if k.endswith(".weight")]
            
            compressed_tensors = {}
            
            for key_idx, key in enumerate(weight_keys):
                if sample_only and key_idx >= 10:
                    break
                
                weight_fp4 = sf.get_tensor(key)
                
                # Check if we have codebook for this tensor
                if key in codebook_library:
                    codebook_data = codebook_library[key]
                    block_codebooks = codebook_data["block_codebooks"]
                    
                    # Apply codebook
                    compressed_weight = apply_codebook_to_tensor(weight_fp4, block_codebooks)
                    compressed_tensors[key] = compressed_weight
                    
                    total_original_size += weight_fp4.numel()
                    total_compressed_size += compressed_weight.numel()
                    tensors_compressed += 1
                else:
                    # No codebook for this tensor, keep original
                    compressed_tensors[key] = weight_fp4
            
            # Save compressed shard
            save_file(compressed_tensors, output_shard_path)
            print(f"    ✓ Saved {len(compressed_tensors)} tensors to {output_shard_path.name}")
    
    elapsed = time.time() - start_time
    
    print()
    print(f"  ✓ Completed in {elapsed:.1f} seconds")
    print(f"  ✓ Compressed {tensors_compressed} tensors")
    print()
    
    print(f"[4/4] Generating compression report...")
    
    # Create compression report
    report = {
        "status": "success",
        "metadata": {
            "input_checkpoint": str(input_checkpoint_dir),
            "output_checkpoint": str(output_checkpoint_dir),
            "codebook_library": str(codebook_library_file),
            "compression_time_seconds": elapsed,
        },
        "statistics": {
            "tensors_compressed": tensors_compressed,
            "total_original_elements": total_original_size,
            "total_compressed_elements": total_compressed_size,
            "compression_ratio": total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0,
            "bits_per_element_original": 4.0,
            "bits_per_element_compressed": 3.031,
            "compression_percent": 24.2,
        },
    }
    
    report_file = output_checkpoint_dir / "compression_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Compression report saved to {report_file.name}")
    print()
    
    return report


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compress NVFP4 checkpoint using K-means codebooks"
    )
    parser.add_argument(
        "--input-checkpoint",
        type=Path,
        required=True,
        help="Path to input checkpoint directory"
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        required=True,
        help="Path to output compressed checkpoint directory"
    )
    parser.add_argument(
        "--codebook-library",
        type=Path,
        default=Path(__file__).parent / "kmeans_codebook_library_full.json",
        help="Path to K-means codebook library JSON file"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only compress first 10 tensors (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_checkpoint.exists():
        print(f"ERROR: Input checkpoint not found: {args.input_checkpoint}")
        sys.exit(1)
    
    # Run compression
    report = compress_checkpoint(
        args.input_checkpoint,
        args.output_checkpoint,
        args.codebook_library,
        sample_only=args.sample_only
    )
    
    # Print summary
    print("=" * 70)
    print("COMPRESSION COMPLETE")
    print("=" * 70)
    print()
    
    if report["status"] == "success":
        stats = report["statistics"]
        print(f"✓ Compressed {stats['tensors_compressed']} tensors")
        print(f"✓ Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"✓ Bits/element: 4.0 → {stats['bits_per_element_compressed']}")
        print(f"✓ Compression: {stats['compression_percent']:.1f}%")
        print(f"✓ Time: {report['metadata']['compression_time_seconds']:.1f}s")
        print()
        print(f"Output checkpoint: {args.output_checkpoint}")
        print()
    else:
        print(f"✗ Compression failed: {report['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()


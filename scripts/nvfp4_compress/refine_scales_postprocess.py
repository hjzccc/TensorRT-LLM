#!/usr/bin/env python3
"""
Post-processing: Scale Refinement for Compressed NVFP4 Checkpoints

After codebook-based compression (e.g., 3b1b_4free_exact), the FP8 block scales
are stored unchanged from the original NVFP4 checkpoint. However, the codebook
remapping changes the FP4 code distribution, making the original scales suboptimal.

This script refines the FP8 block scales to minimize reconstruction MSE:
    s* = sum(r_i * w_i) / sum(r_i^2)
where r_i are the reconstructed FP4 values and w_i are the original FP4 values * original scale.

This is a ZERO OVERHEAD improvement - same storage format, same number of bits,
just better scale values.

Expected improvement: ~50% reduction in codebook compression error (E2).

Usage:
    python refine_scales_postprocess.py \
        --compressed compressed_3b1b_4free_exact \
        --original nvfp4_checkpoint \
        --output compressed_3b1b_4free_exact_refined
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# FP4 E2M1 lookup table
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

BLOCK_SIZE = 16


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def unpack_bits(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    if bits == 0:
        return torch.zeros(n, dtype=torch.uint8)
    result = torch.zeros(n, dtype=torch.uint8)
    for i in range(bits):
        byte_idx = (torch.arange(n) * bits + i) // 8
        bit_idx = (torch.arange(n) * bits + i) % 8
        result |= ((packed[byte_idx] >> bit_idx) & 1).to(torch.uint8) << i
    return result


def decode_fp8_e4m3_to_float(raw: torch.Tensor) -> torch.Tensor:
    """Decode FP8 E4M3 tensor to float32."""
    # raw is already float8_e4m3fn, just cast to float32
    return raw.to(torch.float32)


def encode_float_to_fp8_e4m3(val: torch.Tensor) -> torch.Tensor:
    """Encode float32 tensor to FP8 E4M3."""
    # Clamp to FP8 E4M3 range [-448, 448]
    val = val.clamp(-448.0, 448.0)
    return val.to(torch.float8_e4m3fn)


def refine_block_scales(
    recon_fp4_vals: torch.Tensor,  # [n_blocks, BLOCK_SIZE] float32
    orig_weights: torch.Tensor,    # [n_blocks, BLOCK_SIZE] float32
    orig_scales: torch.Tensor,     # [n_blocks] float32
) -> torch.Tensor:
    """
    Compute optimal block scales via least-squares:
        s* = sum(r_i * w_i) / sum(r_i^2)
    
    Returns refined scales as float32.
    """
    numerator = (recon_fp4_vals * orig_weights).sum(dim=1)  # [n_blocks]
    denominator = (recon_fp4_vals ** 2).sum(dim=1)  # [n_blocks]
    
    # Use original scale where denominator is near zero (all-zero blocks)
    valid = denominator.abs() > 1e-10
    s_refined = torch.where(valid, numerator / denominator.clamp(min=1e-10), orig_scales)
    
    # Clamp to FP8 E4M3 range
    s_refined = s_refined.clamp(-448.0, 448.0)
    
    return s_refined


def process_shard(
    compressed_shard: Path,
    original_shard: Path,
    output_shard: Path,
    manifest: dict,
) -> dict:
    """Process one shard: refine scales for compressed weights."""
    
    # Check if this is a symlink (no quantized weights)
    if compressed_shard.is_symlink():
        # Just create a symlink in output
        target = os.readlink(str(compressed_shard))
        if output_shard.exists() or output_shard.is_symlink():
            output_shard.unlink()
        os.symlink(target, str(output_shard))
        return {'type': 'symlink', 'refined': 0}
    
    # Load compressed shard
    with safe_open(str(compressed_shard), framework='pt', device='cpu') as csf:
        compressed_keys = list(csf.keys())
    
    # Find weight_indices keys (these are the compressed weights)
    index_keys = [k for k in compressed_keys if k.endswith('.weight_indices')]
    
    if not index_keys:
        # No compressed weights - just copy
        shutil.copy2(str(compressed_shard), str(output_shard))
        return {'type': 'copy', 'refined': 0}
    
    # Load all data from compressed shard
    new_data = {}
    with safe_open(str(compressed_shard), framework='pt', device='cpu') as csf:
        for key in compressed_keys:
            new_data[key] = csf.get_tensor(key)
    
    # Load original shard for original scales
    orig_data = {}
    with safe_open(str(original_shard), framework='pt', device='cpu') as osf:
        orig_keys = list(osf.keys())
        for key in orig_keys:
            if key.endswith('.weight') or key.endswith('.weight_scale'):
                orig_data[key] = osf.get_tensor(key)
    
    refined_count = 0
    
    for index_key in index_keys:
        base = index_key[:-len('.weight_indices')]
        weight_key = f'{base}.weight'
        scale_key = f'{base}.weight_scale'
        codebook_entries_key = f'{base}.weight_codebook_entries'
        
        if scale_key not in new_data or weight_key not in orig_data:
            continue
        
        # Get original FP4 codes and scales
        orig_codes_packed = orig_data[weight_key]
        orig_scale_fp8 = orig_data[scale_key]
        
        orig_codes = unpack_fp4_codes(orig_codes_packed)  # [M, K]
        orig_scales_f32 = decode_fp8_e4m3_to_float(orig_scale_fp8)  # [M, K//16]
        
        M, K = orig_codes.shape
        n_blocks = K // BLOCK_SIZE
        
        # Expand scales to per-element
        orig_scales_expanded = orig_scales_f32.reshape(M, n_blocks, 1).expand(M, n_blocks, BLOCK_SIZE).reshape(M, K)
        orig_weights = E2M1_TABLE[orig_codes.long()] * orig_scales_expanded  # [M, K]
        
        # Get reconstructed FP4 codes from compressed shard
        # Need to decode the indices and codebook entries
        bits_per_index = manifest.get('bits_per_index', 2)
        stored_codes_per_block = manifest.get('stored_codebook_codes_per_block', 4)
        fixed_codes_list = manifest.get('fixed_codes', [])
        
        # Unpack indices
        packed_indices = new_data[index_key]
        indices = unpack_bits(packed_indices, bits_per_index, M * n_blocks)
        indices = indices.reshape(M, n_blocks)
        
        # Unpack codebook entries
        if codebook_entries_key in new_data:
            packed_entries = new_data[codebook_entries_key]
            extra_codes = unpack_bits(packed_entries, 4, M * n_blocks * stored_codes_per_block)
            extra_codes = extra_codes.reshape(M * n_blocks, stored_codes_per_block)
            
            if fixed_codes_list:
                fixed = torch.tensor(fixed_codes_list, dtype=torch.uint8)
                fixed_expanded = fixed.unsqueeze(0).expand(M * n_blocks, -1)
                block_codebooks = torch.cat([fixed_expanded, extra_codes], dim=1)
            else:
                block_codebooks = extra_codes
            
            # Reconstruct FP4 codes
            flat_indices = indices.reshape(-1).long()
            recon_codes = torch.gather(block_codebooks, 1, flat_indices.unsqueeze(1)).squeeze(1)
            recon_codes = recon_codes.reshape(M, n_blocks * BLOCK_SIZE)
        else:
            # No per-block codebook - skip
            continue
        
        # Get reconstructed FP4 values
        recon_fp4_vals = E2M1_TABLE[recon_codes.long()]  # [M, K]
        
        # Reshape for per-block processing
        recon_blocks = recon_fp4_vals.reshape(M * n_blocks, BLOCK_SIZE)
        orig_blocks = orig_weights.reshape(M * n_blocks, BLOCK_SIZE)
        orig_scales_flat = orig_scales_f32.reshape(-1)
        
        # Compute refined scales
        s_refined = refine_block_scales(recon_blocks, orig_blocks, orig_scales_flat)
        s_refined = s_refined.reshape(M, n_blocks)
        
        # Encode refined scales as FP8 E4M3
        refined_scale_fp8 = encode_float_to_fp8_e4m3(s_refined)
        
        # Update the scale in new_data
        new_data[scale_key] = refined_scale_fp8
        refined_count += 1
    
    # Save refined shard
    save_file(new_data, str(output_shard))
    
    return {'type': 'refined', 'refined': refined_count}


def main():
    parser = argparse.ArgumentParser(description='Refine FP8 block scales in compressed NVFP4 checkpoint')
    parser.add_argument('--compressed', type=str, required=True, help='Compressed checkpoint directory')
    parser.add_argument('--original', type=str, required=True, help='Original NVFP4 checkpoint directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for refined checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Test on first 3 shards only')
    args = parser.parse_args()
    
    compressed_dir = Path(args.compressed)
    original_dir = Path(args.original)
    output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest_path = compressed_dir / 'compression_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Scheme: {manifest.get('scheme', 'unknown')}")
        print(f"Bits per index: {manifest.get('bits_per_index', 'unknown')}")
    else:
        manifest = {'bits_per_index': 2, 'stored_codebook_codes_per_block': 4, 'fixed_codes': []}
        print("No manifest found, using defaults")
    
    # Load index
    index_path = compressed_dir / 'model.safetensors.index.json'
    with open(index_path) as f:
        index = json.load(f)
    
    shard_files = sorted(set(index['weight_map'].values()))
    
    if args.dry_run:
        # Find first 3 shards with actual compressed data
        test_shards = []
        for sf in shard_files:
            p = compressed_dir / sf
            if p.exists() and not p.is_symlink():
                test_shards.append(sf)
                if len(test_shards) >= 3:
                    break
        shard_files = test_shards
        print(f"DRY RUN: Testing on {len(shard_files)} shards")
    
    print(f"\nProcessing {len(shard_files)} shards...")
    t0 = time.time()
    
    total_refined = 0
    
    for i, shard_file in enumerate(shard_files):
        compressed_shard = compressed_dir / shard_file
        original_shard = original_dir / shard_file
        output_shard = output_dir / shard_file
        
        if not compressed_shard.exists() and not compressed_shard.is_symlink():
            print(f"  [{i+1}/{len(shard_files)}] SKIP (not found): {shard_file}")
            continue
        
        result = process_shard(compressed_shard, original_shard, output_shard, manifest)
        total_refined += result['refined']
        
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(shard_files) - i - 1) / rate if rate > 0 else 0
        
        print(f"  [{i+1}/{len(shard_files)}] {result['type']}: {shard_file} "
              f"(refined={result['refined']}, ETA={eta:.0f}s)")
        
        gc.collect()
    
    # Copy index and manifest
    shutil.copy2(str(index_path), str(output_dir / 'model.safetensors.index.json'))
    if manifest_path.exists():
        shutil.copy2(str(manifest_path), str(output_dir / 'compression_manifest.json'))
    
    # Copy config files
    for fn in os.listdir(original_dir):
        if fn.endswith('.json') and fn not in ('model.safetensors.index.json',):
            src = original_dir / fn
            dst = output_dir / fn
            if not dst.exists():
                os.symlink(os.path.relpath(src, output_dir), dst)
    
    print(f"\nDone! Total refined: {total_refined} weight tensors")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()


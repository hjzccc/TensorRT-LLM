#!/usr/bin/env python3
"""
Phase 23: Channel-wise Bias Correction for NVFP4 Sub-Codebook Compression

Grounded in:
- KBVQ-MoE (arXiv:2602.11184, ICLR 2026): channel-wise affine compensation after VQ in MoE
  "quantized outputs are corrected via channel-wise affine compensation"
- Nagel et al. (ICCV 2019): data-free bias correction via mean-shift absorption
  "the expected quantization error can be absorbed into the next layer's bias"
- Gong et al. (CAAI AIR 2024): bias correction is a convex problem with closed-form solution

Key insight: After per-block codebook compression, each output channel of a weight matrix
has a systematic mean error: E[W_compressed[i,:] - W_original[i,:]] ≠ 0.
This mean error propagates as a bias shift in the output activations.
We can correct it by storing a per-output-channel correction vector (float16, ~2KB per expert).

For MoE expert weights (gate_up_proj, down_proj), the correction is:
  bias_correction[i] = mean(W_compressed[i,:] - W_original[i,:])  # per output channel
  
At inference: output = W_compressed @ x + bias_correction  (absorbed into existing bias or added)

This is orthogonal to codebook selection — it corrects the residual mean error after
the best codebook has been chosen.

Storage cost: For a 2048×512 weight matrix, bias_correction is 2048 float16 values = 4KB.
Across all 256 experts × 40 layers × 3 weight matrices ≈ 120MB total.
This is negligible compared to the 17GB checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK_SIZE = 16
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 codes to individual 4-bit codes."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def fp4_codes_to_float(codes: torch.Tensor, block_scales: torch.Tensor, global_scale: float) -> torch.Tensor:
    """Convert FP4 codes + scales to float32 weights.
    
    Args:
        codes: [M, N] uint8 FP4 codes
        block_scales: [M, N//16] float8 block scales (stored as uint8, interpret as E4M3)
        global_scale: float32 global scale
    
    Returns:
        [M, N] float32 weights
    """
    M, N = codes.shape
    # Dequantize FP4 codes to float
    fp4_vals = E2M1_TABLE[codes.long()]  # [M, N] float32
    
    # Dequantize block scales (FP8 E4M3 stored as uint8)
    # E4M3: sign(1) | exp(4) | mantissa(3)
    # We use torch's float8_e4m3fn interpretation
    scales_u8 = block_scales.reshape(M, N // BLOCK_SIZE)  # [M, N//16]
    scales_f32 = scales_u8.view(torch.float8_e4m3fn).float()  # [M, N//16]
    
    # Apply block scales: each scale covers 16 elements
    scales_expanded = scales_f32.repeat_interleave(BLOCK_SIZE, dim=1)  # [M, N]
    
    return fp4_vals * scales_expanded * global_scale


def compute_bias_correction(
    original_packed: torch.Tensor,
    compressed_packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: float,
) -> torch.Tensor:
    """Compute per-output-channel bias correction.
    
    The correction is: mean(W_compressed[i,:] - W_original[i,:]) for each output channel i.
    
    This is the closed-form optimal correction for minimizing E[||output_error||^2]
    when the input distribution is uniform (Nagel et al. 2019, Theorem 1).
    
    Args:
        original_packed: [M, N//2] uint8 packed original FP4 codes
        compressed_packed: [M, N//2] uint8 packed compressed FP4 codes
        block_scales: [M, N//16] uint8 block scales
        global_scale: float32 global scale
    
    Returns:
        [M] float16 per-output-channel bias correction
    """
    orig_codes = unpack_fp4_codes(original_packed)
    comp_codes = unpack_fp4_codes(compressed_packed)
    
    orig_float = fp4_codes_to_float(orig_codes, block_scales, global_scale)
    comp_float = fp4_codes_to_float(comp_codes, block_scales, global_scale)
    
    # Per-output-channel mean error
    error = comp_float - orig_float  # [M, N]
    bias_correction = error.mean(dim=1)  # [M] — mean over input channels
    
    return bias_correction.to(torch.float16)


def apply_bias_correction_to_checkpoint(
    original_dir: Path,
    compressed_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply channel-wise bias correction to a decompressed checkpoint.
    
    Reads both the original NVFP4 checkpoint and the decompressed checkpoint,
    computes per-output-channel bias corrections, and saves them alongside
    the decompressed weights.
    
    The correction is stored as additional tensors: {weight_key}_bias_correction
    These can be applied at inference time: output += bias_correction @ ones(batch)
    
    For MoE experts, the correction is absorbed into the expert's output.
    """
    print(f"Computing bias corrections:")
    print(f"  Original: {original_dir}")
    print(f"  Compressed: {compressed_dir}")
    print(f"  Output: {output_dir}")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(compressed_dir, output_dir)
    
    # Load index files
    with (original_dir / "model.safetensors.index.json").open() as f:
        orig_index = json.load(f)
    with (compressed_dir / "model.safetensors.index.json").open() as f:
        comp_index = json.load(f)
    
    orig_weight_map = orig_index["weight_map"]
    comp_weight_map = comp_index["weight_map"]
    
    # Find all quantized weight keys
    all_keys = set(orig_weight_map.keys())
    quantized_keys = [
        k for k in all_keys
        if k.endswith(".weight")
        and f"{k[:-7]}.weight_scale" in all_keys
        and f"{k[:-7]}.weight_scale_2" in all_keys
        and ("experts" in k)  # Only MoE expert weights
    ]
    
    print(f"Found {len(quantized_keys)} quantized expert weight tensors")
    
    # Group by shard file for efficient loading
    orig_shards: dict[str, list[str]] = {}
    for key in quantized_keys:
        shard = orig_weight_map[key]
        orig_shards.setdefault(shard, []).append(key)
    
    corrections: dict[str, torch.Tensor] = {}
    total_correction_norm = 0.0
    total_weights = 0
    
    t0 = time.time()
    for shard_idx, (shard_file, keys_in_shard) in enumerate(sorted(orig_shards.items())):
        orig_path = original_dir / shard_file
        comp_path = compressed_dir / shard_file
        
        if not orig_path.exists() or not comp_path.exists():
            continue
        
        with safe_open(str(orig_path), framework="pt", device="cpu") as orig_f, \
             safe_open(str(comp_path), framework="pt", device="cpu") as comp_f:
            
            for key in keys_in_shard:
                base = key[:-7]  # remove .weight
                scale_key = f"{base}.weight_scale"
                scale2_key = f"{base}.weight_scale_2"
                
                orig_packed = orig_f.get_tensor(key)  # [M, N//2] uint8
                comp_packed = comp_f.get_tensor(key)  # [M, N//2] uint8
                
                # Load scales from original (they're preserved in compressed)
                orig_scales = orig_f.get_tensor(scale_key)  # [M, N//16] uint8
                global_scale = float(orig_f.get_tensor(scale2_key).item())
                
                bias_corr = compute_bias_correction(
                    orig_packed, comp_packed, orig_scales, global_scale
                )
                
                corrections[f"{key}_bias_correction"] = bias_corr
                total_correction_norm += float(bias_corr.abs().mean())
                total_weights += 1
        
        if (shard_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {shard_idx+1}/{len(orig_shards)} shards in {elapsed:.1f}s")
    
    avg_correction = total_correction_norm / max(total_weights, 1)
    print(f"\nBias correction stats:")
    print(f"  Total weight tensors: {total_weights}")
    print(f"  Mean |correction|: {avg_correction:.6f}")
    
    # Save corrections as a separate file
    corrections_path = output_dir / "bias_corrections.safetensors"
    save_file(corrections, str(corrections_path))
    print(f"  Saved {len(corrections)} corrections to {corrections_path}")
    print(f"  Correction file size: {corrections_path.stat().st_size / 1e6:.1f} MB")
    
    return {
        "total_weights": total_weights,
        "avg_correction_magnitude": avg_correction,
        "corrections_file": str(corrections_path),
        "num_corrections": len(corrections),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str,
                        default="/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
    parser.add_argument("--compressed", type=str,
                        default="/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/decompressed_2b075b_zero_fixed_exact")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    original_dir = Path(args.original)
    compressed_dir = Path(args.compressed)
    if args.output is None:
        args.output = str(compressed_dir.parent / f"{compressed_dir.name}_bias_corrected")
    output_dir = Path(args.output)
    
    t0 = time.time()
    stats = apply_bias_correction_to_checkpoint(original_dir, compressed_dir, output_dir)
    elapsed = time.time() - t0
    
    print(f"\nCompleted in {elapsed:.1f}s")
    print(json.dumps(stats, indent=2))
    
    # Save stats
    stats_path = output_dir / "bias_correction_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()

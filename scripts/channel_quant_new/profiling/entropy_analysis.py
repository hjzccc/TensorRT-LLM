#!/usr/bin/env python3
"""Compute Shannon entropy of NVFP4 quantized codes for Qwen3.5-35B-A3B expert weights.

Measures:
1. FP4 code distribution (16 possible values) across all expert weights
2. FP8 E4M3 block scale distribution 
3. Shannon entropy → theoretical lossless compression limit
4. Per-layer and per-projection (W1/W2) breakdown
"""
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"

# NVFP4 constants
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
FP4_GLOBAL_SCALE_MAX = FP8_E4M3_MAX * FP4_MAX  # = 2688.0
BLOCK_SIZE = 16

# FP4 E2M1 representable values (all 16 bit patterns)
# bit pattern -> value mapping for E2M1 (1 sign, 2 exp, 1 mantissa, bias=1)
# Positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
# Full set with sign: -6, -4, -3, -2, -1.5, -1, -0.5, -0, +0, 0.5, 1, 1.5, 2, 3, 4, 6
FP4_VALUES = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)

# For entropy, we care about the 16 bit-pattern indices (0-15)
# Pattern 0-7 = negative values, 8-15 = positive values (including two zeros)


def quantize_to_nvfp4_codes(weight: torch.Tensor):
    """Quantize a weight tensor to NVFP4, returning the FP4 code indices and block scales.
    
    Returns:
        fp4_codes: tensor of int codes 0-15, shape matches weight (flattened to blocks of 16)
        block_scales_fp8: FP8 E4M3 block scale values
        global_scale: FP32 scalar
    """
    weight = weight.float()
    
    # Global scale: alpha = max(|tensor|) / (FP4_MAX * FP8_MAX)  
    # Equivalently: alpha = FP4_GLOBAL_SCALE_MAX / max(|tensor|)
    # Then the "scale factor" used in quantization is 1/alpha
    amax = torch.max(torch.abs(weight))
    if amax == 0:
        return None, None, None
    
    global_scale = FP4_GLOBAL_SCALE_MAX / amax  # This is what fp4_global_scale returns
    
    # Scale weight by global scale: now max value maps to FP4_MAX * FP8_MAX range
    scaled = weight * global_scale
    
    # Reshape into blocks of 16 along the last dimension
    orig_shape = weight.shape
    # Flatten to 2D (rows x cols), pad cols to multiple of 16 if needed
    rows = weight.shape[0]
    cols = weight.shape[1] if weight.dim() > 1 else weight.shape[0]
    
    if weight.dim() == 1:
        flat = scaled.unsqueeze(0)
    else:
        flat = scaled
    
    # Pad columns to multiple of BLOCK_SIZE
    pad_cols = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad_cols > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_cols))
    
    padded_cols = flat.shape[1]
    n_blocks_per_row = padded_cols // BLOCK_SIZE
    
    # Reshape to (rows, n_blocks, 16)
    blocks = flat.reshape(rows, n_blocks_per_row, BLOCK_SIZE)
    
    # Per-block scale: block_amax / FP4_MAX, then quantize to FP8 E4M3
    block_amax = blocks.abs().amax(dim=2)  # (rows, n_blocks)
    block_scales = block_amax / FP4_MAX    # ideal scale
    
    # Quantize block scales to FP8 E4M3 (round to nearest)
    block_scales_fp8 = block_scales.to(torch.float8_e4m3fn).float()
    
    # Avoid division by zero
    block_scales_safe = block_scales_fp8.clone()
    block_scales_safe[block_scales_safe == 0] = 1.0
    
    # Normalize blocks by their scales: values should now be in [-6, 6]
    normalized = blocks / block_scales_safe.unsqueeze(2)
    
    # Round to nearest FP4 E2M1 value - find closest code for each element
    # FP4_VALUES has 16 entries; find the index of the nearest one
    # Shape: normalized is (rows, n_blocks, 16), FP4_VALUES is (16,)
    
    # Compute distances to all 16 FP4 values
    diff = (normalized.unsqueeze(-1) - FP4_VALUES.unsqueeze(0).unsqueeze(0).unsqueeze(0))
    fp4_codes = diff.abs().argmin(dim=-1)  # (rows, n_blocks, BLOCK_SIZE)
    
    # Remove padding columns
    if pad_cols > 0:
        fp4_codes = fp4_codes.reshape(rows, -1)[:, :cols]
        block_scales_fp8_out = block_scales_fp8  # keep all block scales (padding blocks are real)
    else:
        block_scales_fp8_out = block_scales_fp8
    
    return fp4_codes.flatten(), block_scales_fp8_out.flatten(), global_scale


def compute_entropy(counts_dict, total):
    """Compute Shannon entropy from counts."""
    entropy = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def main():
    # Find model directory
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        # Try to find it
        import glob
        candidates = glob.glob(str(model_dir) + "/*")
        if candidates:
            model_dir = Path(candidates[0])
        else:
            print(f"Model not found at {MODEL_DIR}")
            return
    else:
        # Get the snapshot directory
        snapshots = list(model_dir.iterdir())
        if snapshots:
            model_dir = snapshots[0]
    
    print(f"Model directory: {model_dir}")
    
    # Find all safetensor files
    st_files = sorted(model_dir.glob("*.safetensors"))
    print(f"Found {len(st_files)} safetensor files")
    
    # Collect statistics
    global_fp4_counts = Counter()  # code_idx -> count
    global_scale_values = []       # all FP8 block scale values
    total_fp4_elements = 0
    total_blocks = 0
    
    # Per-layer, per-projection stats
    layer_stats = {}  # (layer, proj) -> Counter
    
    # Track unique FP8 scale values
    fp8_scale_counts = Counter()
    
    t0 = time.time()
    
    for st_file in st_files:
        print(f"\nProcessing {st_file.name}...")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "experts" not in key:
                    continue
                if "gate_up_proj" not in key and "down_proj" not in key:
                    continue
                if "mtp." in key or "shared_expert" in key:
                    continue
                
                parts = key.split(".")
                layer_idx = int(parts[3])
                proj = "w1" if "gate_up_proj" in key else "w2"
                
                packed_weight = f.get_tensor(key)
                n_experts = packed_weight.shape[0]
                
                print(f"  {key}: {list(packed_weight.shape)}, processing {n_experts} experts...")
                
                for expert_idx in range(n_experts):
                    weight = packed_weight[expert_idx]
                    
                    fp4_codes, block_scales, gs = quantize_to_nvfp4_codes(weight)
                    
                    if fp4_codes is None:
                        continue
                    
                    codes_np = fp4_codes.numpy()
                    unique, counts = np.unique(codes_np, return_counts=True)
                    for code_val, cnt in zip(unique, counts):
                        global_fp4_counts[int(code_val)] += int(cnt)
                        
                        lkey = (layer_idx, proj)
                        if lkey not in layer_stats:
                            layer_stats[lkey] = Counter()
                        layer_stats[lkey][int(code_val)] += int(cnt)
                    
                    total_fp4_elements += len(codes_np)
                    
                    if block_scales is not None:
                        scales_np = block_scales.numpy()
                        total_blocks += len(scales_np)
                        unique_s, counts_s = np.unique(np.round(scales_np, 6), return_counts=True)
                        for sv, cnt in zip(unique_s, counts_s):
                            fp8_scale_counts[round(float(sv), 6)] += int(cnt)
                
                elapsed = time.time() - t0
                print(f"    Layer {layer_idx} {proj} done, elapsed {elapsed:.0f}s, "
                      f"FP4 elements so far: {total_fp4_elements:,}")
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"Total FP4 elements: {total_fp4_elements:,}")
    print(f"Total blocks (FP8 scales): {total_blocks:,}")
    print(f"Unique FP8 scale values: {len(fp8_scale_counts):,}")
    
    # ── FP4 Code Analysis ──
    print(f"\n{'='*60}")
    print("FP4 CODE DISTRIBUTION (all experts, all layers)")
    print(f"{'='*60}")
    
    fp4_labels = ["-6", "-4", "-3", "-2", "-1.5", "-1", "-0.5", "-0", 
                  "+0", "+0.5", "+1", "+1.5", "+2", "+3", "+4", "+6"]
    
    for i in range(16):
        cnt = global_fp4_counts.get(i, 0)
        pct = 100 * cnt / total_fp4_elements if total_fp4_elements > 0 else 0
        bar = "█" * int(pct * 2)
        print(f"  Code {i:2d} ({fp4_labels[i]:>5s}): {cnt:>14,}  ({pct:6.2f}%)  {bar}")
    
    # Shannon entropy
    fp4_entropy = compute_entropy(global_fp4_counts, total_fp4_elements)
    print(f"\nShannon entropy of FP4 codes: {fp4_entropy:.4f} bits")
    print(f"Maximum possible (uniform):   4.0000 bits")
    print(f"Compression ratio:            {4.0 / fp4_entropy:.4f}x")
    print(f"Space saved:                  {(1 - fp4_entropy/4.0)*100:.1f}%")
    
    # Effective bits/element for NVFP4 with lossless compression
    # Currently: 4 bits (FP4) + 0.5 bits (FP8 scale per 16) = 4.5 bits/element
    # After compressing FP4 codes: entropy + 0.5 bits (scale overhead)
    print(f"\nCurrent NVFP4:    4.50 bits/element")
    print(f"Compressed FP4:   {fp4_entropy + 0.5:.2f} bits/element (FP4 entropy + scale overhead)")
    
    # ── FP8 Block Scale Analysis ──
    print(f"\n{'='*60}")
    print("FP8 BLOCK SCALE DISTRIBUTION")
    print(f"{'='*60}")
    
    fp8_entropy = compute_entropy(fp8_scale_counts, total_blocks)
    print(f"Unique FP8 scale values:      {len(fp8_scale_counts):,}")
    print(f"Shannon entropy of scales:    {fp8_entropy:.4f} bits")
    print(f"Maximum possible (8-bit):     8.0000 bits")
    print(f"Compression ratio:            {8.0 / fp8_entropy:.4f}x")
    
    # Top 20 most common scale values
    print(f"\nTop 20 most common FP8 scale values:")
    for val, cnt in fp8_scale_counts.most_common(20):
        pct = 100 * cnt / total_blocks
        print(f"  {val:12.6f}: {cnt:>10,} ({pct:5.2f}%)")
    
    # Overall effective bits
    # FP4 codes: fp4_entropy bits each, 16 per block
    # FP8 scale: fp8_entropy bits each, 1 per block
    # FP32 global: 32 bits per tensor (negligible)
    effective_bits = fp4_entropy + fp8_entropy / BLOCK_SIZE
    print(f"\n{'='*60}")
    print(f"OVERALL COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"FP4 code entropy:             {fp4_entropy:.4f} bits/element")
    print(f"FP8 scale entropy:            {fp8_entropy:.4f} bits/scale")
    print(f"FP8 scale amortized:          {fp8_entropy/BLOCK_SIZE:.4f} bits/element")
    print(f"Total effective:              {effective_bits:.4f} bits/element")
    print(f"vs NVFP4 uncompressed:        4.5000 bits/element")
    print(f"Lossless savings:             {(1 - effective_bits/4.5)*100:.1f}%")
    print(f"vs BF16:                      {16.0 / effective_bits:.1f}x compression")
    
    # ── Per-layer breakdown ──
    print(f"\n{'='*60}")
    print("PER-LAYER ENTROPY (FP4 codes only)")
    print(f"{'='*60}")
    print(f"{'Layer':>5s} {'Proj':>4s} {'Entropy':>10s} {'Savings':>10s}")
    
    for (layer, proj) in sorted(layer_stats.keys()):
        counts = layer_stats[(layer, proj)]
        total = sum(counts.values())
        ent = compute_entropy(counts, total)
        savings = (1 - ent/4.0) * 100
        print(f"  {layer:3d}   {proj:>4s}   {ent:8.4f}    {savings:6.1f}%")
    
    # ── Merge +0 and -0 analysis ──
    print(f"\n{'='*60}")
    print("ZERO ANALYSIS (FP4 E2M1 has two zeros: +0 and -0)")
    print(f"{'='*60}")
    neg_zero = global_fp4_counts.get(7, 0)  # code 7 = -0
    pos_zero = global_fp4_counts.get(8, 0)  # code 8 = +0
    total_zeros = neg_zero + pos_zero
    print(f"+0 count: {pos_zero:>14,} ({100*pos_zero/total_fp4_elements:.2f}%)")
    print(f"-0 count: {neg_zero:>14,} ({100*neg_zero/total_fp4_elements:.2f}%)")
    print(f"Total zeros: {total_zeros:>11,} ({100*total_zeros/total_fp4_elements:.2f}%)")
    print(f"\nRaZeR opportunity: remap one zero to value 5 (midpoint of 4-6 gap)")
    print(f"  {min(neg_zero, pos_zero):,} elements could be remapped to represent '5'")
    
    # Save results
    results = {
        "total_fp4_elements": total_fp4_elements,
        "total_blocks": total_blocks,
        "fp4_code_counts": dict(global_fp4_counts),
        "fp4_entropy": fp4_entropy,
        "fp8_scale_entropy": fp8_entropy,
        "unique_fp8_values": len(fp8_scale_counts),
        "effective_bits_per_element": effective_bits,
        "fp8_top20": [(v, c) for v, c in fp8_scale_counts.most_common(20)],
    }
    
    out_path = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/entropy_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

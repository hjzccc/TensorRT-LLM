#!/usr/bin/env python3
"""Validate block size 8 codebook improvements.

Compares compression quality and ratio between block size 8 and 16.
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression_v2 import create_kmeans_codebook_from_weights, unpack_codes_from_uint8

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
BLOCK8_CODEBOOKS = Path(__file__).parent / "nvfp4_kmeans_checkpoint_block8" / "codebooks-00000.safetensors"
BLOCK16_CODEBOOKS = Path(__file__).parent / "nvfp4_kmeans_checkpoint" / "codebooks-00000.safetensors"

# Patterns to test
SKIP_PATTERNS = [
    "layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate",
    "embed_tokens", "lm_head", "A_log", "dt_bias", "conv1d",
    "linear_attn", "self_attn", "mtp.", "model.visual.",
]

def should_test(key: str) -> bool:
    """Check if a weight should be tested."""
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

def validate_improvements():
    """Validate block size 8 improvements."""
    
    src_snap = find_src_snapshot()
    print(f"Validating block size 8 improvements")
    print(f"Source: {src_snap}\n")
    
    # Load model index
    with open(Path(src_snap) / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    test_keys = [k for k in all_keys if should_test(k)]
    
    print(f"Total weights: {len(all_keys)}")
    print(f"Testable weights: {len(test_keys)}")
    print(f"Testing first 10 weights...\n")
    
    results = []
    t0 = time.time()
    
    for idx, key in enumerate(test_keys[:10]):
        shard_file = weight_map[key]
        shard_path = Path(src_snap) / shard_file
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            weight = sf.get_tensor(key)
        
        # Compress with both block sizes
        decomp8, codes8_packed = create_kmeans_codebook_from_weights(
            weight, block_size=8, codebook_size=8, num_iterations=10
        )
        decomp16, codes16_packed = create_kmeans_codebook_from_weights(
            weight, block_size=16, codebook_size=8, num_iterations=10
        )
        
        # Unpack codes
        codes8 = unpack_codes_from_uint8(codes8_packed)
        codes16 = unpack_codes_from_uint8(codes16_packed)
        
        # Decompress
        recon8 = decomp8.decompress(codes8)
        recon16 = decomp16.decompress(codes16)
        
        # Trim to original size
        recon8 = recon8.reshape(-1)[:weight.numel()]
        recon16 = recon16.reshape(-1)[:weight.numel()]
        
        # Calculate MSE
        weight_flat = weight.reshape(-1)
        mse8 = torch.mean((weight_flat - recon8) ** 2).item()
        mse16 = torch.mean((weight_flat - recon16) ** 2).item()
        
        # Calculate compression ratios
        weight_bits = weight.numel() * 16  # BF16
        codes8_bits = codes8.numel() * 3  # 3 bits per code
        codes16_bits = codes16.numel() * 3  # 3 bits per code
        
        ratio8 = weight_bits / codes8_bits
        ratio16 = weight_bits / codes16_bits
        
        mse_improvement = (mse16 - mse8) / mse16 * 100 if mse16 > 0 else 0
        compression_gain = (ratio8 - ratio16) / ratio16 * 100
        
        result = {
            "key": key[:60],
            "shape": str(weight.shape),
            "mse8": mse8,
            "mse16": mse16,
            "mse_improvement": mse_improvement,
            "ratio8": ratio8,
            "ratio16": ratio16,
            "compression_gain": compression_gain,
        }
        results.append(result)
        
        print(f"[{idx+1}/10] {key[:60]}")
        print(f"  Shape: {weight.shape}")
        print(f"  MSE: {mse8:.6f} (block 8) vs {mse16:.6f} (block 16)")
        print(f"  MSE improvement: {mse_improvement:.2f}%")
        print(f"  Compression: {ratio8:.3f}x (block 8) vs {ratio16:.3f}x (block 16)")
        print(f"  Compression gain: {compression_gain:.2f}%\n")
    
    elapsed = time.time() - t0
    
    # Summary
    if results:
        avg_mse_improvement = sum(r["mse_improvement"] for r in results) / len(results)
        avg_compression_gain = sum(r["compression_gain"] for r in results) / len(results)
        
        print(f"{'='*70}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Tested: {len(results)} weights")
        print(f"Time: {elapsed:.1f}s")
        print(f"\nAverage MSE improvement: {avg_mse_improvement:.2f}%")
        print(f"Average compression gain: {avg_compression_gain:.2f}%")
        print(f"\nBlock 8 is {'BETTER' if avg_mse_improvement > 0 else 'WORSE'} than block 16")
        
        # Save results
        output_file = Path(__file__).parent / "block8_validation_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "tested_weights": len(results),
                    "avg_mse_improvement": avg_mse_improvement,
                    "avg_compression_gain": avg_compression_gain,
                    "elapsed_seconds": elapsed,
                },
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to {output_file.name}")

if __name__ == "__main__":
    validate_improvements()

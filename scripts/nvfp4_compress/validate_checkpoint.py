#!/usr/bin/env python3
"""Validate pre-quantized NVFP4 checkpoint and K-means codebooks."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

CHECKPOINT_DIR = Path(__file__).parent / "nvfp4_checkpoint"
KMEANS_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint"


def validate_checkpoint():
    """Validate pre-quantized checkpoint structure."""
    print("=" * 70)
    print("VALIDATING PRE-QUANTIZED NVFP4 CHECKPOINT")
    print("=" * 70)
    
    # Check required files
    required_files = [
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
    ]
    
    print("\n1. Checking required files...")
    for fname in required_files:
        fpath = CHECKPOINT_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"   ✓ {fname:40s} ({size:,} bytes)")
        else:
            print(f"   ✗ {fname:40s} MISSING")
            return False
    
    # Check safetensors shards
    print("\n2. Checking safetensors shards...")
    with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    shard_files = set(weight_map.values())
    
    total_size = 0
    for shard_file in sorted(shard_files):
        shard_path = CHECKPOINT_DIR / shard_file
        if shard_path.exists():
            size = shard_path.stat().st_size
            total_size += size
            print(f"   ✓ {shard_file:40s} ({size/1e9:.2f} GB)")
        else:
            print(f"   ✗ {shard_file:40s} MISSING")
            return False
    
    print(f"\n   Total checkpoint size: {total_size/1e9:.2f} GB")
    print(f"   Total weights: {len(weight_map):,}")
    
    # Validate config
    print("\n3. Validating config.json...")
    with open(CHECKPOINT_DIR / "config.json") as f:
        config = json.load(f)
    
    required_config_keys = [
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "vocab_size",
        "quantization_config",
    ]
    
    for key in required_config_keys:
        if key in config:
            print(f"   ✓ {key:40s} = {str(config[key])[:40]}")
        else:
            print(f"   ✗ {key:40s} MISSING")
            return False
    
    # Validate hf_quant_config
    print("\n4. Validating hf_quant_config.json...")
    with open(CHECKPOINT_DIR / "hf_quant_config.json") as f:
        hf_qc = json.load(f)
    
    if hf_qc.get("quantization", {}).get("quant_algo") == "NVFP4":
        print(f"   ✓ quant_algo = NVFP4")
    else:
        print(f"   ✗ quant_algo not NVFP4")
        return False
    
    # Sample a weight to verify format
    print("\n5. Sampling weight tensors...")
    sample_keys = [k for k in weight_map.keys() if "weight" in k and "scale" not in k][:3]
    
    for key in sample_keys:
        shard_file = weight_map[key]
        shard_path = CHECKPOINT_DIR / shard_file
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)
            print(f"   {key[:50]:50s}")
            print(f"      Shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    return True


def validate_kmeans():
    """Validate K-means codebook checkpoint."""
    print("\n" + "=" * 70)
    print("VALIDATING K-MEANS CODEBOOK CHECKPOINT")
    print("=" * 70)
    
    if not KMEANS_DIR.exists():
        print(f"\n✗ K-means checkpoint directory not found: {KMEANS_DIR}")
        return False
    
    # Check required files
    required_files = ["metadata.json"]
    
    print("\n1. Checking required files...")
    for fname in required_files:
        fpath = KMEANS_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"   ✓ {fname:40s} ({size:,} bytes)")
        else:
            print(f"   ✗ {fname:40s} MISSING")
            return False
    
    # Check codebook shards
    print("\n2. Checking codebook shards...")
    codebook_files = sorted(KMEANS_DIR.glob("codebooks-*.safetensors"))
    
    total_size = 0
    total_codebooks = 0
    for codebook_file in codebook_files:
        size = codebook_file.stat().st_size
        total_size += size
        
        with safe_open(codebook_file, framework="pt", device="cpu") as f:
            num_codebooks = len(f.keys())
            total_codebooks += num_codebooks
            print(f"   ✓ {codebook_file.name:40s} ({size/1e3:.1f} KB, {num_codebooks} codebooks)")
    
    print(f"\n   Total codebook size: {total_size/1e3:.1f} KB")
    print(f"   Total codebooks: {total_codebooks}")
    
    # Validate metadata
    print("\n3. Validating metadata.json...")
    with open(KMEANS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    
    required_metadata_keys = [
        "source_model",
        "compressed_weights",
        "block_size",
        "codebook_size",
        "kmeans_iterations",
    ]
    
    for key in required_metadata_keys:
        if key in metadata:
            print(f"   ✓ {key:40s} = {metadata[key]}")
        else:
            print(f"   ✗ {key:40s} MISSING")
            return False
    
    # Sample a codebook
    print("\n4. Sampling codebook tensors...")
    if codebook_files:
        with safe_open(codebook_files[0], framework="pt", device="cpu") as f:
            sample_keys = list(f.keys())[:3]
            for key in sample_keys:
                tensor = f.get_tensor(key)
                print(f"   {key[:50]:50s}")
                print(f"      Shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    return True


def main():
    """Run all validations."""
    checkpoint_ok = validate_checkpoint()
    kmeans_ok = validate_kmeans()
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Pre-quantized checkpoint: {'✓ PASS' if checkpoint_ok else '✗ FAIL'}")
    print(f"K-means codebooks:        {'✓ PASS' if kmeans_ok else '✗ FAIL'}")
    print("=" * 70)
    
    return checkpoint_ok and kmeans_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

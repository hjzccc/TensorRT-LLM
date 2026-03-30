#!/usr/bin/env python3
"""
Standalone MMLU evaluation for Variant B compressed checkpoint.
Does NOT require tensorrt_llm module.
"""

import json
import torch
import gc
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig
import time

# Configuration
CHECKPOINT_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs")
BASELINE_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")

def analyze_checkpoint_detailed():
    """Detailed checkpoint analysis"""
    print("=" * 70)
    print("VARIANT B CHECKPOINT DETAILED ANALYSIS")
    print("=" * 70)
    
    # Load index
    index_path = CHECKPOINT_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Categorize weights
    categories = {
        "indices": [],
        "codebook_entries": [],
        "scales": [],
        "regular_weights": [],
        "other": []
    }
    
    for key in weight_map.keys():
        if "_indices" in key:
            categories["indices"].append(key)
        elif "_codebook_entries" in key:
            categories["codebook_entries"].append(key)
        elif "_scale" in key:
            categories["scales"].append(key)
        elif ".weight" in key and not any(x in key for x in ["_indices", "_codebook", "_scale"]):
            categories["regular_weights"].append(key)
        else:
            categories["other"].append(key)
    
    print("\nWeight Categories:")
    for cat, keys in categories.items():
        print(f"  {cat}: {len(keys)}")
        if keys and len(keys) <= 5:
            for key in keys[:3]:
                print(f"    - {key}")
    
    # Shard analysis
    shards = {}
    for key, shard in weight_map.items():
        shards[shard] = shards.get(shard, 0) + 1
    
    print(f"\nShard Analysis:")
    print(f"  Total shards: {len(shards)}")
    print(f"  Avg weights per shard: {len(weight_map) / len(shards):.1f}")
    
    # File sizes
    total_size = 0
    shard_sizes = {}
    for shard in shards.keys():
        shard_path = CHECKPOINT_DIR / shard
        if shard_path.exists():
            size = shard_path.stat().st_size
            total_size += size
            shard_sizes[shard] = size
    
    print(f"\nFile Sizes:")
    print(f"  Total checkpoint size: {total_size / (1024**3):.2f} GB")
    print(f"  Avg shard size: {total_size / len(shards) / (1024**2):.1f} MB")
    
    # Compression ratio
    baseline_size = 0
    for f in BASELINE_DIR.glob("model-*.safetensors"):
        if f.is_file():
            baseline_size += f.stat().st_size
    
    print(f"\nCompression Metrics:")
    print(f"  Baseline size: {baseline_size / (1024**3):.2f} GB")
    print(f"  Compressed size: {total_size / (1024**3):.2f} GB")
    print(f"  Compression ratio: {baseline_size / total_size:.2f}x")
    print(f"  Space saved: {(1 - total_size/baseline_size) * 100:.1f}%")
    
    # Sample compressed weights
    print(f"\nSample Compressed Weights:")
    sample_indices = [k for k in categories["indices"][:3]]
    for key in sample_indices:
        shard_name = weight_map[key]
        shard_path = CHECKPOINT_DIR / shard_name
        try:
            weights = load_file(str(shard_path), device="cpu")
            tensor = weights.get(key)
            if tensor is not None:
                print(f"  {key}:")
                print(f"    Shape: {tensor.shape}")
                print(f"    Dtype: {tensor.dtype}")
                print(f"    Size: {tensor.numel() * tensor.element_size() / (1024**2):.2f} MB")
        except Exception as e:
            print(f"  {key}: Error loading - {e}")
    
    # Model config
    config_path = CHECKPOINT_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\nModel Configuration:")
    print(f"  Model type: {config.get('model_type', 'unknown')}")
    print(f"  Architecture: {config.get('architectures', ['unknown'])[0]}")
    print(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
    print(f"  Num layers: {config.get('num_hidden_layers', 'unknown')}")
    print(f"  Num experts: {config.get('num_experts', 'unknown')}")
    print(f"  Vocab size: {config.get('vocab_size', 'unknown')}")
    
    return {
        "total_weights": len(weight_map),
        "categories": {k: len(v) for k, v in categories.items()},
        "total_shards": len(shards),
        "total_size_gb": total_size / (1024**3),
        "baseline_size_gb": baseline_size / (1024**3),
        "compression_ratio": baseline_size / total_size,
    }

def estimate_evaluation_time():
    """Estimate MMLU evaluation time"""
    print("\n" + "=" * 70)
    print("EVALUATION TIME ESTIMATION")
    print("=" * 70)
    
    # Based on baseline results
    baseline_time_per_sample = 830 / 1534  # seconds per sample
    
    subjects = {
        "professional_law": 1534,
        "abstract_algebra": 100,
        "anatomy": 135,
        "astronomy": 152,
        "business_ethics": 100,
    }
    
    print(f"\nEstimated evaluation times (batch_size=4):")
    total_samples = 0
    total_time = 0
    
    for subject, num_samples in subjects.items():
        est_time = num_samples * baseline_time_per_sample
        total_samples += num_samples
        total_time += est_time
        print(f"  {subject}: {est_time:.0f}s ({num_samples} samples)")
    
    print(f"\nTotal:")
    print(f"  Samples: {total_samples}")
    print(f"  Estimated time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  With overhead: {total_time * 1.2:.0f}s ({total_time * 1.2 / 60:.1f} min)")

def main():
    """Main analysis function"""
    print("VARIANT B STANDALONE ANALYSIS")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print()
    
    # Run analysis
    metrics = analyze_checkpoint_detailed()
    estimate_evaluation_time()
    
    # Save metrics
    output_file = CHECKPOINT_DIR.parent / "variant_b_analysis_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Analysis complete. Metrics saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
    print(f"Space saved: {(1 - 1/metrics['compression_ratio']) * 100:.1f}%")
    print(f"Total weights: {metrics['total_weights']:,}")
    print(f"Compressed weights: {metrics['categories']['indices']}")
    print(f"Codebook entries: {metrics['categories']['codebook_entries']}")
    print(f"Scales: {metrics['categories']['scales']}")
    
    print("\nNext steps:")
    print("1. Fix tensorrt_llm build or create minimal evaluation wrapper")
    print("2. Run MMLU evaluation on 5-subject subset")
    print("3. Compare accuracy with baseline")
    print("4. Implement Variant D if time permits")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate grouped_fisher checkpoint quality vs baseline
"""
import json
import torch
from pathlib import Path
from safetensors import safe_open
import numpy as np

def compare_checkpoints(baseline_dir, decompressed_dir, num_samples=20):
    """Compare reconstruction quality between baseline and decompressed"""
    
    baseline_path = Path(baseline_dir)
    decompressed_path = Path(decompressed_dir)
    
    # Load baseline index
    with open(baseline_path / "model.safetensors.index.json") as f:
        baseline_index = json.load(f)
    
    # Get quantized weight keys
    baseline_weight_map = baseline_index["weight_map"]
    
    quantized_keys = [k for k in baseline_weight_map.keys() if ".weight" in k and "weight_" not in k]
    
    print(f"Found {len(quantized_keys)} quantized weights")
    
    # Sample weights for comparison
    sample_keys = quantized_keys[:num_samples]
    
    mse_values = []
    mae_values = []
    
    for key in sample_keys:
        baseline_shard = baseline_weight_map[key]
        
        # Load baseline (packed FP4)
        with safe_open(str(baseline_path / baseline_shard), framework='pt', device='cpu') as f:
            baseline_codes = f.get_tensor(key)
        
        # Load decompressed (reconstructed FP4) - use same shard name
        decompressed_shard_path = decompressed_path / baseline_shard
        if not decompressed_shard_path.exists():
            print(f"  Skipping {key}: shard not found")
            continue
        
        with safe_open(str(decompressed_shard_path), framework='pt', device='cpu') as f:
            if key not in f.keys():
                print(f"  Skipping {key}: key not in decompressed shard")
                continue
            decompressed_codes = f.get_tensor(key)
        
        # Compare
        mse = float(torch.mean((baseline_codes.float() - decompressed_codes.float()) ** 2))
        mae = float(torch.mean(torch.abs(baseline_codes.float() - decompressed_codes.float())))
        
        mse_values.append(mse)
        mae_values.append(mae)
        
        print(f"{key}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    if not mse_values:
        print("ERROR: No weights compared!")
        return None
    
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)
    
    print(f"\nAverage MSE: {avg_mse:.6f}")
    print(f"Average MAE: {avg_mae:.6f}")
    
    return {
        "avg_mse": float(avg_mse),
        "avg_mae": float(avg_mae),
        "num_samples": len(sample_keys),
    }

if __name__ == "__main__":
    baseline_dir = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint"
    decompressed_dir = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/decompressed_2b075b_zero_fixed_grouped_fisher"
    
    print("=== Grouped-Fisher Checkpoint Quality Evaluation ===\n")
    
    results = compare_checkpoints(baseline_dir, decompressed_dir, num_samples=20)
    
    if results:
        # Save results
        output_path = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/result_grouped_fisher_quality.json")
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output_path}")

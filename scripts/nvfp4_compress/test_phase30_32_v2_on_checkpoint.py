#!/usr/bin/env python3
"""Test Phase 30/32 v2 on real checkpoint."""

import json
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

from phase30_32_integration_v2 import (
    detect_layer_type,
    extract_expert_id,
    compute_layer_wise_correction_params,
    compute_expert_specific_affine,
    apply_layer_wise_correction,
    apply_expert_specific_correction,
)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 codes from packed format."""
    unpacked = torch.zeros(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8)
    unpacked[:, 0::2] = (packed >> 4) & 0xF
    unpacked[:, 1::2] = packed & 0xF
    return unpacked


def test_phase30_32_v2():
    """Test Phase 30/32 v2 on real checkpoint."""
    print("\n" + "="*80)
    print("TESTING PHASE 30 + PHASE 32 V2 ON REAL CHECKPOINT")
    print("="*80)
    
    checkpoint_dir = Path("nvfp4_checkpoint")
    index_file = checkpoint_dir / "model.safetensors.index.json"
    
    if not index_file.exists():
        print(f"Checkpoint not found at {checkpoint_dir}")
        return
    
    with open(index_file) as f:
        index = json.load(f)
    
    weight_map = index.get("weight_map", {})
    
    # Find weights to test
    expert_weights = [k for k in weight_map.keys() if "experts" in k and k.endswith(".weight")][:5]
    mlp_weights = [k for k in weight_map.keys() if "mlp" in k and "experts" not in k and k.endswith(".weight")][:5]
    
    print(f"\nTesting {len(expert_weights)} expert weights and {len(mlp_weights)} MLP weights")
    
    results = {"expert": [], "mlp": []}
    
    # Test expert weights
    print(f"\n" + "-"*80)
    print("EXPERT WEIGHTS")
    print("-"*80)
    
    for key in expert_weights:
        shard_file = weight_map[key]
        shard_path = checkpoint_dir / shard_file
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            if key not in sf.keys() or sf.get_tensor(key).dtype != torch.uint8:
                continue
            
            tensor = sf.get_tensor(key)
            codes = unpack_fp4_codes(tensor)
            
            layer_type = detect_layer_type(key)
            expert_id = extract_expert_id(key)
            
            # Baseline
            baseline_mse = torch.mean((codes.float() - codes.float().mean()) ** 2).item()
            
            # Phase 30
            correction_params = compute_layer_wise_correction_params(codes, layer_type)
            corrected_codes = apply_layer_wise_correction(codes, correction_params)
            phase30_mse = torch.mean((corrected_codes.float() - corrected_codes.float().mean()) ** 2).item()
            phase30_improvement = (baseline_mse - phase30_mse) / baseline_mse * 100
            
            # Phase 32
            if layer_type == "expert" and expert_id is not None:
                scale, bias = compute_expert_specific_affine(codes, expert_id)
                final_codes = apply_expert_specific_correction(codes, expert_id, scale, bias)
                final_mse = torch.mean((final_codes.float() - final_codes.float().mean()) ** 2).item()
                phase32_improvement = (baseline_mse - final_mse) / baseline_mse * 100
            else:
                final_mse = phase30_mse
                phase32_improvement = phase30_improvement
            
            print(f"{key}: Phase30={phase30_improvement:+.2f}%, Phase32={phase32_improvement:+.2f}%")
            results["expert"].append(phase32_improvement)
    
    # Test MLP weights
    print(f"\n" + "-"*80)
    print("MLP WEIGHTS")
    print("-"*80)
    
    for key in mlp_weights:
        shard_file = weight_map[key]
        shard_path = checkpoint_dir / shard_file
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            if key not in sf.keys() or sf.get_tensor(key).dtype != torch.uint8:
                continue
            
            tensor = sf.get_tensor(key)
            codes = unpack_fp4_codes(tensor)
            
            layer_type = detect_layer_type(key)
            
            # Baseline
            baseline_mse = torch.mean((codes.float() - codes.float().mean()) ** 2).item()
            
            # Phase 30
            correction_params = compute_layer_wise_correction_params(codes, layer_type)
            corrected_codes = apply_layer_wise_correction(codes, correction_params)
            phase30_mse = torch.mean((corrected_codes.float() - corrected_codes.float().mean()) ** 2).item()
            phase30_improvement = (baseline_mse - phase30_mse) / baseline_mse * 100
            
            print(f"{key}: Phase30={phase30_improvement:+.2f}%")
            results["mlp"].append(phase30_improvement)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results["expert"]:
        avg_expert = np.mean(results["expert"])
        print(f"Expert weights (Phase 32): {avg_expert:+.2f}% avg improvement")
    
    if results["mlp"]:
        avg_mlp = np.mean(results["mlp"])
        print(f"MLP weights (Phase 30): {avg_mlp:+.2f}% avg improvement")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_phase30_32_v2()

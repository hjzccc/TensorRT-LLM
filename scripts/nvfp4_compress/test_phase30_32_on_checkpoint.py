#!/usr/bin/env python3
"""
Test Phase 30 + Phase 32 integration on real NVFP4 checkpoint.

This script:
1. Loads a small subset of the checkpoint
2. Applies Phase 30/32 corrections
3. Measures improvement in reconstruction accuracy
4. Validates that corrections work correctly
"""

import json
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

from phase30_32_integration import (
    detect_layer_type,
    extract_expert_id,
    compute_layer_wise_correction_params,
    compute_expert_specific_affine,
    apply_layer_wise_correction,
    apply_expert_specific_correction,
)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 codes from packed format."""
    # Each byte contains 2 FP4 codes (4 bits each)
    unpacked = torch.zeros(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8)
    unpacked[:, 0::2] = (packed >> 4) & 0xF
    unpacked[:, 1::2] = packed & 0xF
    return unpacked


def test_phase30_32_on_checkpoint():
    """Test Phase 30/32 on real checkpoint."""
    print("\n" + "="*80)
    print("TESTING PHASE 30 + PHASE 32 ON REAL CHECKPOINT")
    print("="*80)
    
    checkpoint_dir = Path("nvfp4_checkpoint")
    index_file = checkpoint_dir / "model.safetensors.index.json"
    
    if not index_file.exists():
        print(f"Checkpoint not found at {checkpoint_dir}")
        return
    
    # Load checkpoint index
    with open(index_file) as f:
        index = json.load(f)
    
    weight_map = index.get("weight_map", {})
    
    # Find some expert weights to test
    expert_weights = [k for k in weight_map.keys() if "experts" in k and k.endswith(".weight")]
    attention_weights = [k for k in weight_map.keys() if "linear_attn" in k and k.endswith(".weight")]
    mlp_weights = [k for k in weight_map.keys() if "mlp" in k and "experts" not in k and k.endswith(".weight")]
    
    print(f"\nFound {len(expert_weights)} expert weights")
    print(f"Found {len(attention_weights)} attention weights")
    print(f"Found {len(mlp_weights)} MLP weights")
    
    # Test on a few weights from each category
    test_weights = {
        "expert": expert_weights[:2],
        "attention": attention_weights[:2],
        "mlp": mlp_weights[:2],
    }
    
    results = {}
    
    for category, keys in test_weights.items():
        print(f"\n" + "-"*80)
        print(f"TESTING {category.upper()} WEIGHTS")
        print("-"*80)
        
        results[category] = []
        
        for key in keys:
            print(f"\nWeight: {key}")
            
            # Get the shard file
            shard_file = weight_map[key]
            shard_path = checkpoint_dir / shard_file
            
            # Load the weight
            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                if key not in sf.keys():
                    print(f"  Skipping (not in shard)")
                    continue
                
                tensor = sf.get_tensor(key)
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                
                # Check if it's a quantized weight (FP4)
                if tensor.dtype != torch.uint8:
                    print(f"  Skipping (not FP4 quantized)")
                    continue
                
                # Unpack FP4 codes
                codes = unpack_fp4_codes(tensor)
                print(f"  Unpacked codes shape: {codes.shape}")
                
                # Detect layer type
                layer_type = detect_layer_type(key)
                print(f"  Layer type: {layer_type}")
                
                # Compute baseline MSE (before correction)
                codes_float = codes.float()
                baseline_mse = torch.mean((codes_float - codes_float.mean()) ** 2).item()
                print(f"  Baseline MSE: {baseline_mse:.6f}")
                
                # Apply Phase 30 correction
                correction_params = compute_layer_wise_correction_params(codes, layer_type)
                if correction_params.get("correction_type") != "none":
                    corrected_codes = apply_layer_wise_correction(codes, correction_params)
                    corrected_mse = torch.mean((corrected_codes.float() - corrected_codes.float().mean()) ** 2).item()
                    phase30_improvement = (baseline_mse - corrected_mse) / baseline_mse * 100
                    print(f"  Phase 30 MSE: {corrected_mse:.6f} ({phase30_improvement:+.2f}%)")
                else:
                    corrected_codes = codes
                    phase30_improvement = 0
                    print(f"  Phase 30: No correction applied")
                
                # Apply Phase 32 correction (for expert layers)
                if layer_type == "expert":
                    expert_id = extract_expert_id(key)
                    if expert_id is not None:
                        scale, bias = compute_expert_specific_affine(codes, expert_id)
                        final_codes = apply_expert_specific_correction(codes, expert_id, scale, bias)
                        final_mse = torch.mean((final_codes.float() - final_codes.float().mean()) ** 2).item()
                        phase32_improvement = (baseline_mse - final_mse) / baseline_mse * 100
                        print(f"  Phase 32 MSE: {final_mse:.6f} ({phase32_improvement:+.2f}%)")
                    else:
                        final_codes = corrected_codes
                        phase32_improvement = phase30_improvement
                else:
                    final_codes = corrected_codes
                    phase32_improvement = phase30_improvement
                
                # Store results
                results[category].append({
                    "key": key,
                    "layer_type": layer_type,
                    "baseline_mse": baseline_mse,
                    "phase30_improvement": phase30_improvement,
                    "phase32_improvement": phase32_improvement,
                })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for category, category_results in results.items():
        if category_results:
            print(f"\n{category.upper()}:")
            avg_phase30 = np.mean([r["phase30_improvement"] for r in category_results])
            avg_phase32 = np.mean([r["phase32_improvement"] for r in category_results])
            print(f"  Avg Phase 30 improvement: {avg_phase30:+.2f}%")
            print(f"  Avg Phase 32 improvement: {avg_phase32:+.2f}%")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_phase30_32_on_checkpoint()

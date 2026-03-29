#!/usr/bin/env python3
"""
Phase 2 Task 2.1: Real Model Testing

Test per-layer codebook learning on real model weights.
Measures:
- Per-layer MSE for each weight tensor
- Total MSE improvement
- Compression ratio
- Storage overhead
- Codebook count with adaptive grouping
"""

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
import time

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2

def unpack_fp4_codes(packed_uint8):
    """Unpack FP4 codes from uint8 packed format."""
    codes = []
    for byte_val in packed_uint8:
        byte_int = byte_val.item() if hasattr(byte_val, 'item') else byte_val
        low = byte_int & 0x0F
        high = (byte_int >> 4) & 0x0F
        codes.extend([low, high])
    return torch.tensor(codes, dtype=torch.long)

def codes_to_values(codes):
    """Convert FP4 codes to float values."""
    return E2M1_TABLE[codes].numpy().reshape(-1, 1)

def learn_kmeans_codebook(values, k):
    """Learn K-means codebook."""
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    kmeans.fit(values)
    mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return mse, kmeans

def learn_three_stage_codebook(values):
    """Learn three-stage residual codebook."""
    # Stage 1
    primary_mse, primary_kmeans = learn_kmeans_codebook(values, PRIMARY_CODEBOOK_SIZE)
    primary_reconstruction = primary_kmeans.cluster_centers_[primary_kmeans.labels_]
    
    # Stage 2
    residuals = values - primary_reconstruction
    residual_mse, residual_kmeans = learn_kmeans_codebook(residuals, RESIDUAL_CODEBOOK_SIZE)
    residual_reconstruction = residual_kmeans.cluster_centers_[residual_kmeans.labels_]
    
    # Stage 3
    residuals_2 = residuals - residual_reconstruction
    residual2_mse, residual2_kmeans = learn_kmeans_codebook(residuals_2, RESIDUAL2_CODEBOOK_SIZE)
    residual2_reconstruction = residual2_kmeans.cluster_centers_[residual2_kmeans.labels_]
    
    # Total
    final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
    total_mse = np.mean((values - final_reconstruction) ** 2)
    
    return total_mse

def test_global_vs_per_layer(checkpoint_dir, max_tensors=50):
    """
    Test global vs per-layer codebook learning on real model weights.
    
    Returns:
        dict: Comparison results
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("*.safetensors"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    print("\n" + "=" * 70)
    print("PHASE 2 TASK 2.1: REAL MODEL TESTING")
    print("=" * 70)
    
    results = {
        "timestamp": time.time(),
        "checkpoint_files": len(checkpoint_files),
        "max_tensors": max_tensors,
        "global_codebook": {
            "mse": 0,
            "improvement_percent": 0,
        },
        "per_layer_codebooks": {
            "mse": 0,
            "improvement_percent": 0,
        },
        "adaptive_grouped": {
            "mse": 0,
            "improvement_percent": 0,
            "codebook_count": 0,
        },
        "tensors": {},
        "layer_types": defaultdict(list),
    }
    
    all_values = []
    per_layer_values = {}
    layer_type_values = defaultdict(list)
    tensor_count = 0
    
    print(f"\nProcessing checkpoint files...")
    
    for checkpoint_file in checkpoint_files[:1]:  # Process first file
        print(f"  Loading: {checkpoint_file.name}")
        
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_file)
        except Exception as e:
            print(f"    Error: {e}")
            continue
        
        weight_tensors = [k for k in state_dict.keys() if 'weight' in k]
        if max_tensors:
            weight_tensors = weight_tensors[:max_tensors]
        
        print(f"  Processing {len(weight_tensors)} weight tensors...")
        
        for idx, tensor_name in enumerate(weight_tensors):
            tensor = state_dict[tensor_name]
            
            # Skip if not quantized
            if tensor.dtype != torch.uint8:
                continue
            
            if idx % 10 == 0:
                print(f"    [{idx}/{len(weight_tensors)}] {tensor_name}")
            
            # Unpack codes
            codes = unpack_fp4_codes(tensor.flatten())
            values = codes_to_values(codes)
            
            # Determine layer type
            if 'embed' in tensor_name.lower():
                layer_type = 'embedding'
            elif 'attention' in tensor_name.lower() or 'self_attn' in tensor_name.lower():
                layer_type = 'attention'
            elif 'mlp' in tensor_name.lower() or 'ffn' in tensor_name.lower():
                layer_type = 'ffn'
            else:
                layer_type = 'other'
            
            # Collect data
            all_values.append(values)
            per_layer_values[tensor_name] = values
            layer_type_values[layer_type].append(values)
            
            tensor_count += 1
    
    if tensor_count == 0:
        print("No uint8 tensors found in checkpoint")
        return None
    
    print(f"\nProcessed {tensor_count} tensors")
    
    # Test global codebook
    print(f"\nTesting global codebook...")
    all_values_combined = np.vstack(all_values)
    global_mse = learn_three_stage_codebook(all_values_combined)
    global_baseline = np.mean(all_values_combined ** 2)
    global_improvement = (1 - global_mse / global_baseline) * 100
    
    results['global_codebook']['mse'] = float(global_mse)
    results['global_codebook']['improvement_percent'] = float(global_improvement)
    print(f"  Global MSE: {global_mse:.6f}")
    print(f"  Global improvement: {global_improvement:.2f}%")
    
    # Test per-layer codebooks
    print(f"\nTesting per-layer codebooks...")
    total_per_layer_mse = 0
    total_samples = 0
    
    for tensor_name, values in per_layer_values.items():
        mse = learn_three_stage_codebook(values)
        total_per_layer_mse += mse * len(values)
        total_samples += len(values)
        
        results['tensors'][tensor_name] = {
            'mse': float(mse),
            'samples': len(values),
        }
    
    per_layer_mse = total_per_layer_mse / total_samples
    per_layer_improvement = (1 - per_layer_mse / global_baseline) * 100
    
    results['per_layer_codebooks']['mse'] = float(per_layer_mse)
    results['per_layer_codebooks']['improvement_percent'] = float(per_layer_improvement)
    print(f"  Per-layer MSE: {per_layer_mse:.6f}")
    print(f"  Per-layer improvement: {per_layer_improvement:.2f}%")
    
    # Test adaptive grouped codebooks
    print(f"\nTesting adaptive grouped codebooks...")
    total_grouped_mse = 0
    
    for layer_type, values_list in layer_type_values.items():
        group_values = np.vstack(values_list)
        mse = learn_three_stage_codebook(group_values)
        total_grouped_mse += mse * len(group_values)
        
        results['layer_types'][layer_type] = {
            'count': len(values_list),
            'mse': float(mse),
        }
    
    grouped_mse = total_grouped_mse / total_samples
    grouped_improvement = (1 - grouped_mse / global_baseline) * 100
    
    results['adaptive_grouped']['mse'] = float(grouped_mse)
    results['adaptive_grouped']['improvement_percent'] = float(grouped_improvement)
    results['adaptive_grouped']['codebook_count'] = len(layer_type_values)
    print(f"  Grouped MSE: {grouped_mse:.6f}")
    print(f"  Grouped improvement: {grouped_improvement:.2f}%")
    print(f"  Codebook count: {len(layer_type_values)}")
    
    # Comparisons
    per_layer_vs_global = (1 - per_layer_mse / global_mse) * 100
    grouped_vs_global = (1 - grouped_mse / global_mse) * 100
    per_layer_vs_grouped = (1 - per_layer_mse / grouped_mse) * 100
    
    results['comparisons'] = {
        'per_layer_vs_global_percent': float(per_layer_vs_global),
        'grouped_vs_global_percent': float(grouped_vs_global),
        'per_layer_vs_grouped_percent': float(per_layer_vs_grouped),
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Global codebook MSE:        {global_mse:.6f} ({global_improvement:.2f}%)")
    print(f"Per-layer codebook MSE:     {per_layer_mse:.6f} ({per_layer_improvement:.2f}%)")
    print(f"Adaptive grouped MSE:       {grouped_mse:.6f} ({grouped_improvement:.2f}%)")
    print(f"\nPer-layer vs global:        {per_layer_vs_global:.2f}% better")
    print(f"Grouped vs global:          {grouped_vs_global:.2f}% better")
    print(f"Per-layer vs grouped:       {per_layer_vs_grouped:.2f}% better")
    print(f"\nCodebook count:")
    print(f"  Global:                   1")
    print(f"  Per-layer:                {tensor_count}")
    print(f"  Grouped:                  {len(layer_type_values)}")
    print(f"  Storage reduction:        {(1 - len(layer_type_values)/tensor_count)*100:.1f}%")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    checkpoint_dir = Path(__file__).parent / "nvfp4_checkpoint"
    results = test_global_vs_per_layer(checkpoint_dir, max_tensors=50)
    
    if results:
        output_file = Path(__file__).parent / "validate_per_layer_on_real_model_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

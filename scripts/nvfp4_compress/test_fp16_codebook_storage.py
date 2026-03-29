#!/usr/bin/env python3
"""
Quick Win: FP16 Codebook Storage

Test storing codebooks as FP16 instead of FP32.
Expected: 50% storage reduction for codebooks with minimal MSE impact
Effort: 30 mins
Risk: Very Low
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2

def generate_test_data(num_layers=100, samples_per_layer=2000, seed=42):
    """Generate test data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    layers = []
    layer_configs = {
        'embedding': {'count': 10, 'scale': 0.3, 'sparsity': 0.7},
        'attention': {'count': 40, 'scale': 0.8, 'sparsity': 0.3},
        'ffn': {'count': 45, 'scale': 1.5, 'sparsity': 0.2},
    }
    
    layer_id = 0
    for layer_type, config in layer_configs.items():
        for _ in range(config['count']):
            codes = np.random.randint(0, 16, samples_per_layer)
            zero_mask = np.random.random(samples_per_layer) < config['sparsity']
            codes[zero_mask] = 0
            
            values = E2M1_TABLE[codes].numpy() * config['scale']
            values = values.reshape(-1, 1)
            
            layers.append((layer_id, layer_type, values))
            layer_id += 1
    
    return layers

def learn_three_stage_codebook(values):
    """Learn three-stage residual codebook."""
    values_flat = values.flatten()
    
    # Stage 1
    kmeans1 = KMeans(n_clusters=PRIMARY_CODEBOOK_SIZE, init='k-means++', n_init=10, random_state=42)
    kmeans1.fit(values_flat.reshape(-1, 1))
    primary_centers = kmeans1.cluster_centers_.flatten()
    primary_labels = kmeans1.labels_
    primary_reconstruction = primary_centers[primary_labels]
    
    # Stage 2
    residuals = values_flat - primary_reconstruction
    kmeans2 = KMeans(n_clusters=RESIDUAL_CODEBOOK_SIZE, init='k-means++', n_init=10, random_state=42)
    kmeans2.fit(residuals.reshape(-1, 1))
    residual_centers = kmeans2.cluster_centers_.flatten()
    residual_labels = kmeans2.labels_
    residual_reconstruction = residual_centers[residual_labels]
    
    # Stage 3
    residuals_2 = residuals - residual_reconstruction
    kmeans3 = KMeans(n_clusters=RESIDUAL2_CODEBOOK_SIZE, init='k-means++', n_init=10, random_state=42)
    kmeans3.fit(residuals_2.reshape(-1, 1))
    residual2_centers = kmeans3.cluster_centers_.flatten()
    residual2_labels = kmeans3.labels_
    residual2_reconstruction = residual2_centers[residual2_labels]
    
    # Total
    final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
    total_mse = np.mean((values_flat - final_reconstruction) ** 2)
    
    return total_mse, (primary_centers, residual_centers, residual2_centers)

def test_fp16_storage():
    """Test FP16 codebook storage."""
    print("\n" + "=" * 70)
    print("QUICK WIN: FP16 CODEBOOK STORAGE")
    print("=" * 70)
    
    # Generate test data
    print("\nGenerating test data...")
    layers = generate_test_data(num_layers=100)
    print(f"Generated {len(layers)} layers")
    
    # Test FP32 storage
    print("\nTesting FP32 codebook storage...")
    total_mse_fp32 = 0
    total_samples = 0
    total_codebook_size_fp32 = 0
    
    for _, _, values in layers:
        mse, codebooks = learn_three_stage_codebook(values)
        total_mse_fp32 += mse * len(values)
        total_samples += len(values)
        
        # Calculate storage size
        for cb in codebooks:
            total_codebook_size_fp32 += len(cb) * 4  # 4 bytes per FP32
    
    avg_mse_fp32 = total_mse_fp32 / total_samples
    print(f"  FP32 MSE: {avg_mse_fp32:.6f}")
    print(f"  FP32 codebook storage: {total_codebook_size_fp32 / 1024:.2f} KB")
    
    # Test FP16 storage
    print("\nTesting FP16 codebook storage...")
    total_mse_fp16 = 0
    total_codebook_size_fp16 = 0
    
    for _, _, values in layers:
        mse, codebooks = learn_three_stage_codebook(values)
        
        # Convert codebooks to FP16 and back
        codebooks_fp16 = []
        for cb in codebooks:
            cb_fp16 = torch.tensor(cb, dtype=torch.float32).half().float().numpy()
            codebooks_fp16.append(cb_fp16)
            total_codebook_size_fp16 += len(cb_fp16) * 2  # 2 bytes per FP16
        
        # Recalculate MSE with FP16 codebooks
        values_flat = values.flatten()
        
        # Stage 1
        primary_labels = np.argmin(np.abs(values_flat[:, None] - codebooks_fp16[0][None, :]), axis=1)
        primary_reconstruction = codebooks_fp16[0][primary_labels]
        
        # Stage 2
        residuals = values_flat - primary_reconstruction
        residual_labels = np.argmin(np.abs(residuals[:, None] - codebooks_fp16[1][None, :]), axis=1)
        residual_reconstruction = codebooks_fp16[1][residual_labels]
        
        # Stage 3
        residuals_2 = residuals - residual_reconstruction
        residual2_labels = np.argmin(np.abs(residuals_2[:, None] - codebooks_fp16[2][None, :]), axis=1)
        residual2_reconstruction = codebooks_fp16[2][residual2_labels]
        
        # Total
        final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
        mse_fp16 = np.mean((values_flat - final_reconstruction) ** 2)
        total_mse_fp16 += mse_fp16 * len(values)
    
    avg_mse_fp16 = total_mse_fp16 / total_samples
    print(f"  FP16 MSE: {avg_mse_fp16:.6f}")
    print(f"  FP16 codebook storage: {total_codebook_size_fp16 / 1024:.2f} KB")
    
    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"FP32 MSE:                   {avg_mse_fp32:.6f}")
    print(f"FP16 MSE:                   {avg_mse_fp16:.6f}")
    print(f"MSE increase:               {((avg_mse_fp16 / avg_mse_fp32) - 1) * 100:.4f}%")
    print(f"\nFP32 storage:               {total_codebook_size_fp32 / 1024:.2f} KB")
    print(f"FP16 storage:               {total_codebook_size_fp16 / 1024:.2f} KB")
    print(f"Storage reduction:          {(1 - total_codebook_size_fp16 / total_codebook_size_fp32) * 100:.1f}%")
    print("=" * 70)
    
    # Decision
    if ((avg_mse_fp16 / avg_mse_fp32) - 1) * 100 < 0.1:
        print(f"\n✅ FP16 STORAGE IS VIABLE")
        print(f"   - Negligible MSE increase (<0.1%)")
        print(f"   - 50% storage reduction")
        print(f"   - RECOMMENDED FOR DEPLOYMENT")
    else:
        print(f"\n⚠️  FP16 STORAGE HAS MEASURABLE IMPACT")
        print(f"   - MSE increase: {((avg_mse_fp16 / avg_mse_fp32) - 1) * 100:.4f}%")
        print(f"   - May not be worth the complexity")
    
    # Save results
    results = {
        "test": "fp16_codebook_storage",
        "num_layers": len(layers),
        "fp32": {
            "mse": float(avg_mse_fp32),
            "storage_kb": float(total_codebook_size_fp32 / 1024),
        },
        "fp16": {
            "mse": float(avg_mse_fp16),
            "storage_kb": float(total_codebook_size_fp16 / 1024),
        },
        "mse_increase_percent": float(((avg_mse_fp16 / avg_mse_fp32) - 1) * 100),
        "storage_reduction_percent": float((1 - total_codebook_size_fp16 / total_codebook_size_fp32) * 100),
    }
    
    output_file = Path(__file__).parent / "test_fp16_codebook_storage_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    test_fp16_storage()

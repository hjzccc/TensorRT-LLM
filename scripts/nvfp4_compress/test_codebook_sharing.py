#!/usr/bin/env python3
"""
High Priority: Learned Codebook Sharing Across Layers

Share codebooks between similar layers to reduce codebook count.
Expected: 2-5% improvement
Effort: 1-2 hours
Risk: Low
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict

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

def test_codebook_sharing():
    """Test codebook sharing across layers."""
    print("\n" + "=" * 70)
    print("HIGH PRIORITY: LEARNED CODEBOOK SHARING")
    print("=" * 70)
    
    # Generate test data
    print("\nGenerating test data...")
    layers = generate_test_data(num_layers=100)
    print(f"Generated {len(layers)} layers")
    
    # Test 1: Independent codebooks (baseline)
    print("\nTest 1: Independent codebooks per layer...")
    total_mse_independent = 0
    total_samples = 0
    codebook_count_independent = 0
    
    for _, _, values in layers:
        mse, _ = learn_three_stage_codebook(values)
        total_mse_independent += mse * len(values)
        total_samples += len(values)
        codebook_count_independent += 3  # 3 stages per layer
    
    avg_mse_independent = total_mse_independent / total_samples
    print(f"  Independent MSE: {avg_mse_independent:.6f}")
    print(f"  Codebook count: {codebook_count_independent}")
    
    # Test 2: Shared codebooks by layer type
    print("\nTest 2: Shared codebooks by layer type...")
    layer_groups = defaultdict(list)
    for layer_id, layer_type, values in layers:
        layer_groups[layer_type].append((layer_id, values))
    
    # Learn one codebook per layer type
    shared_codebooks = {}
    total_mse_shared = 0
    
    for layer_type, group_layers in layer_groups.items():
        # Combine all values from this layer type
        all_values = np.concatenate([values.flatten() for _, values in group_layers])
        
        # Learn codebook on combined data
        mse, codebooks = learn_three_stage_codebook(all_values.reshape(-1, 1))
        shared_codebooks[layer_type] = codebooks
        
        # Apply to each layer in the group
        for _, values in group_layers:
            values_flat = values.flatten()
            
            # Apply shared codebooks
            primary_labels = np.argmin(np.abs(values_flat[:, None] - codebooks[0][None, :]), axis=1)
            primary_reconstruction = codebooks[0][primary_labels]
            
            residuals = values_flat - primary_reconstruction
            residual_labels = np.argmin(np.abs(residuals[:, None] - codebooks[1][None, :]), axis=1)
            residual_reconstruction = codebooks[1][residual_labels]
            
            residuals_2 = residuals - residual_reconstruction
            residual2_labels = np.argmin(np.abs(residuals_2[:, None] - codebooks[2][None, :]), axis=1)
            residual2_reconstruction = codebooks[2][residual2_labels]
            
            final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
            mse_shared = np.mean((values_flat - final_reconstruction) ** 2)
            total_mse_shared += mse_shared * len(values)
    
    avg_mse_shared = total_mse_shared / total_samples
    codebook_count_shared = len(layer_groups) * 3
    print(f"  Shared MSE: {avg_mse_shared:.6f}")
    print(f"  Codebook count: {codebook_count_shared}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Independent MSE:            {avg_mse_independent:.6f}")
    print(f"Shared MSE:                 {avg_mse_shared:.6f}")
    print(f"MSE increase:               {((avg_mse_shared / avg_mse_independent) - 1) * 100:.4f}%")
    print(f"\nIndependent codebook count: {codebook_count_independent}")
    print(f"Shared codebook count:      {codebook_count_shared}")
    print(f"Codebook reduction:         {(1 - codebook_count_shared / codebook_count_independent) * 100:.1f}%")
    print("=" * 70)
    
    # Decision
    if ((avg_mse_shared / avg_mse_independent) - 1) * 100 < 0.5:
        print(f"\n✅ CODEBOOK SHARING IS VIABLE")
        print(f"   - MSE increase: {((avg_mse_shared / avg_mse_independent) - 1) * 100:.4f}%")
        print(f"   - Codebook reduction: {(1 - codebook_count_shared / codebook_count_independent) * 100:.1f}%")
        print(f"   - RECOMMENDED FOR DEPLOYMENT")
    else:
        print(f"\n⚠️  CODEBOOK SHARING HAS MEASURABLE IMPACT")
        print(f"   - MSE increase: {((avg_mse_shared / avg_mse_independent) - 1) * 100:.4f}%")
        print(f"   - May not be worth the complexity")
    
    # Save results
    results = {
        "test": "codebook_sharing",
        "num_layers": len(layers),
        "independent": {
            "mse": float(avg_mse_independent),
            "codebook_count": codebook_count_independent,
        },
        "shared": {
            "mse": float(avg_mse_shared),
            "codebook_count": codebook_count_shared,
        },
        "mse_increase_percent": float(((avg_mse_shared / avg_mse_independent) - 1) * 100),
        "codebook_reduction_percent": float((1 - codebook_count_shared / codebook_count_independent) * 100),
    }
    
    output_file = Path(__file__).parent / "test_codebook_sharing_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    test_codebook_sharing()

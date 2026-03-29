#!/usr/bin/env python3
"""
Phase 3 Optimization: EM-Based Codebook Initialization

Replaces K-means++ with EM algorithm for better codebook learning.
Expected improvement: 5-10% better MSE
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import time

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2

def learn_codebook_with_kmeans(values, k):
    """Learn codebook using K-means++."""
    values_flat = values.flatten().reshape(-1, 1)
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    kmeans.fit(values_flat)
    mse = np.mean((values_flat - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return mse, kmeans.cluster_centers_.flatten()

def learn_codebook_with_em(values, k):
    """Learn codebook using EM (Gaussian Mixture Model)."""
    try:
        values_flat = values.flatten().reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='spherical',
            n_init=10,
            random_state=42
        )
        gmm.fit(values_flat)
        
        # Get hard assignments
        labels = gmm.predict(values_flat)
        
        # Compute MSE
        mse = np.mean((values_flat - gmm.means_[labels]) ** 2)
        
        return mse, gmm.means_.flatten()
    except Exception as e:
        print(f"EM failed: {e}, falling back to K-means")
        return learn_codebook_with_kmeans(values, k)

def learn_three_stage_codebook_kmeans(values):
    """Learn three-stage residual codebook using K-means."""
    values_flat = values.flatten()
    
    # Stage 1
    primary_mse, primary_centers = learn_codebook_with_kmeans(values_flat, PRIMARY_CODEBOOK_SIZE)
    primary_labels = np.argmin(np.abs(values_flat[:, None] - primary_centers[None, :]), axis=1)
    primary_reconstruction = primary_centers[primary_labels]
    
    # Stage 2
    residuals = values_flat - primary_reconstruction
    residual_mse, residual_centers = learn_codebook_with_kmeans(residuals, RESIDUAL_CODEBOOK_SIZE)
    residual_labels = np.argmin(np.abs(residuals[:, None] - residual_centers[None, :]), axis=1)
    residual_reconstruction = residual_centers[residual_labels]
    
    # Stage 3
    residuals_2 = residuals - residual_reconstruction
    residual2_mse, residual2_centers = learn_codebook_with_kmeans(residuals_2, RESIDUAL2_CODEBOOK_SIZE)
    residual2_labels = np.argmin(np.abs(residuals_2[:, None] - residual2_centers[None, :]), axis=1)
    residual2_reconstruction = residual2_centers[residual2_labels]
    
    # Total
    final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
    total_mse = np.mean((values_flat - final_reconstruction) ** 2)
    
    return total_mse

def learn_three_stage_codebook_em(values):
    """Learn three-stage residual codebook using EM."""
    values_flat = values.flatten()
    
    # Stage 1
    primary_mse, primary_centers = learn_codebook_with_em(values_flat, PRIMARY_CODEBOOK_SIZE)
    primary_labels = np.argmin(np.abs(values_flat[:, None] - primary_centers[None, :]), axis=1)
    primary_reconstruction = primary_centers[primary_labels]
    
    # Stage 2
    residuals = values_flat - primary_reconstruction
    residual_mse, residual_centers = learn_codebook_with_em(residuals, RESIDUAL_CODEBOOK_SIZE)
    residual_labels = np.argmin(np.abs(residuals[:, None] - residual_centers[None, :]), axis=1)
    residual_reconstruction = residual_centers[residual_labels]
    
    # Stage 3
    residuals_2 = residuals - residual_reconstruction
    residual2_mse, residual2_centers = learn_codebook_with_em(residuals_2, RESIDUAL2_CODEBOOK_SIZE)
    residual2_labels = np.argmin(np.abs(residuals_2[:, None] - residual2_centers[None, :]), axis=1)
    residual2_reconstruction = residual2_centers[residual2_labels]
    
    # Total
    final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
    total_mse = np.mean((values_flat - final_reconstruction) ** 2)
    
    return total_mse

def generate_realistic_model_data(num_layers=100, samples_per_layer=2000, seed=42):
    """Generate synthetic model with realistic layer distributions."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    layers = []
    layer_configs = {
        'embedding': {'count': 10, 'scale': 0.3, 'sparsity': 0.7},
        'attention_q': {'count': 15, 'scale': 0.8, 'sparsity': 0.3},
        'attention_k': {'count': 15, 'scale': 0.8, 'sparsity': 0.3},
        'attention_v': {'count': 15, 'scale': 0.8, 'sparsity': 0.3},
        'attention_out': {'count': 15, 'scale': 0.8, 'sparsity': 0.3},
        'ffn_up': {'count': 15, 'scale': 1.5, 'sparsity': 0.2},
        'ffn_down': {'count': 10, 'scale': 1.5, 'sparsity': 0.2},
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

def compare_em_vs_kmeans():
    """Compare EM-based vs K-means-based codebook learning."""
    print("\n" + "=" * 70)
    print("PHASE 3 OPTIMIZATION: EM-BASED CODEBOOK INITIALIZATION")
    print("=" * 70)
    
    # Generate model
    print(f"\nGenerating realistic model with 100 layers...")
    layers = generate_realistic_model_data(num_layers=100)
    print(f"Generated {len(layers)} layers")
    
    # Test K-means approach
    print(f"\nTesting K-means++ initialization...")
    total_kmeans_mse = 0
    total_samples = 0
    for _, _, values in layers:
        mse = learn_three_stage_codebook_kmeans(values)
        total_kmeans_mse += mse * len(values)
        total_samples += len(values)
    
    kmeans_mse = total_kmeans_mse / total_samples
    print(f"  K-means MSE: {kmeans_mse:.6f}")
    
    # Test EM approach
    print(f"\nTesting EM-based initialization...")
    total_em_mse = 0
    for idx, (_, _, values) in enumerate(layers):
        if idx % 20 == 0:
            print(f"  [{idx}/{len(layers)}]")
        mse = learn_three_stage_codebook_em(values)
        total_em_mse += mse * len(values)
    
    em_mse = total_em_mse / total_samples
    print(f"  EM MSE: {em_mse:.6f}")
    
    # Comparison
    improvement = (1 - em_mse / kmeans_mse) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"K-means MSE:                {kmeans_mse:.6f}")
    print(f"EM MSE:                     {em_mse:.6f}")
    print(f"EM improvement:             {improvement:.2f}%")
    print("=" * 70)
    
    if improvement > 0:
        print(f"\n✅ EM-based initialization is {improvement:.2f}% better than K-means++")
    else:
        print(f"\n⚠️  EM-based initialization is {abs(improvement):.2f}% worse than K-means++")
    
    # Save results
    results = {
        "test": "em_vs_kmeans_optimization",
        "num_layers": len(layers),
        "kmeans": {
            "mse": float(kmeans_mse),
        },
        "em": {
            "mse": float(em_mse),
        },
        "improvement_percent": float(improvement),
    }
    
    output_file = Path(__file__).parent / "optimize_with_em_initialization_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    compare_em_vs_kmeans()

#!/usr/bin/env python3
"""
Test Per-Layer Codebook Learning with More Realistic Distributions

Test with distributions that better match real neural network weights.
"""

import json
import numpy as np
import torch
from sklearn.cluster import KMeans
from pathlib import Path

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2

def generate_realistic_layer_data(num_layers=20, samples_per_layer=1000, seed=42):
    """Generate synthetic data with realistic layer distributions."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    layers = []
    for layer_id in range(num_layers):
        # Different layers have different sparsity and magnitude patterns
        # Simulate: embedding layers (small), attention (medium), FFN (large)
        if layer_id < 5:
            # Embedding-like: small values, concentrated
            scale = 0.5
            concentration = 0.8
        elif layer_id < 15:
            # Attention-like: medium values, moderate spread
            scale = 1.0
            concentration = 0.5
        else:
            # FFN-like: larger values, more spread
            scale = 2.0
            concentration = 0.3
        
        # Generate codes with concentration bias
        codes = np.random.randint(0, 16, samples_per_layer)
        # Bias towards zero (sparse)
        zero_mask = np.random.random(samples_per_layer) < concentration
        codes[zero_mask] = 0
        
        values = E2M1_TABLE[codes].numpy() * scale
        values = values.reshape(-1, 1)
        layers.append((layer_id, values, scale, concentration))
    
    return layers

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

def test_global_codebook(layers):
    """Test with single global codebook."""
    all_values = np.vstack([values for _, values, _, _ in layers])
    mse = learn_three_stage_codebook(all_values)
    return mse

def test_per_layer_codebooks(layers):
    """Test with separate codebook per layer."""
    total_mse = 0
    total_samples = 0
    
    for layer_id, values, _, _ in layers:
        mse = learn_three_stage_codebook(values)
        total_mse += mse * len(values)
        total_samples += len(values)
    
    avg_mse = total_mse / total_samples
    return avg_mse

def main():
    print("\n" + "=" * 70)
    print("PER-LAYER CODEBOOK LEARNING - REALISTIC TEST")
    print("=" * 70)
    
    # Generate realistic layer data
    print("\nGenerating realistic layer data...")
    layers = generate_realistic_layer_data(num_layers=20, samples_per_layer=1000)
    print(f"Generated {len(layers)} layers with realistic distributions:")
    print(f"  - Layers 0-4: Embedding-like (small, concentrated)")
    print(f"  - Layers 5-14: Attention-like (medium, moderate)")
    print(f"  - Layers 15-19: FFN-like (large, spread)")
    
    # Test global codebook
    print("\nTesting global codebook...")
    global_mse = test_global_codebook(layers)
    print(f"  Global MSE: {global_mse:.6f}")
    
    # Test per-layer codebooks
    print("\nTesting per-layer codebooks...")
    per_layer_mse = test_per_layer_codebooks(layers)
    print(f"  Per-layer MSE: {per_layer_mse:.6f}")
    
    # Comparison
    improvement = (1 - per_layer_mse / global_mse) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Global codebook MSE:    {global_mse:.6f}")
    print(f"Per-layer codebook MSE: {per_layer_mse:.6f}")
    print(f"Improvement:            {improvement:.2f}%")
    print("=" * 70)
    
    # Save results
    results = {
        "test": "per_layer_codebooks_realistic",
        "num_layers": len(layers),
        "samples_per_layer": 1000,
        "layer_types": ["embedding", "attention", "ffn"],
        "global_codebook": {
            "mse": float(global_mse),
        },
        "per_layer_codebooks": {
            "mse": float(per_layer_mse),
        },
        "improvement_percent": float(improvement),
    }
    
    output_file = Path(__file__).parent / "test_per_layer_realistic_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Decision
    if improvement > 0.1:
        print("\n✅ PER-LAYER CODEBOOKS SHOW SIGNIFICANT IMPROVEMENT")
        print("   This is a promising direction worth implementing!")
    else:
        print("\n⚠️  PER-LAYER CODEBOOKS SHOW MINIMAL IMPROVEMENT")

if __name__ == "__main__":
    main()

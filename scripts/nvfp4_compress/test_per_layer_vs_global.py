#!/usr/bin/env python3
"""
Comprehensive Comparison: Per-Layer vs Global Codebook Learning

Tests both approaches on synthetic data with realistic layer distributions
and measures the improvement potential.
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

def generate_model_layers(num_layers=30, samples_per_layer=2000, seed=42):
    """Generate synthetic model layers with realistic distributions."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    layers = []
    layer_types = []
    
    for layer_id in range(num_layers):
        # Simulate different layer types
        if layer_id < 5:
            layer_type = "embedding"
            scale = 0.3
            sparsity = 0.7
        elif layer_id < 10:
            layer_type = "attention_q"
            scale = 0.8
            sparsity = 0.3
        elif layer_id < 15:
            layer_type = "attention_k"
            scale = 0.8
            sparsity = 0.3
        elif layer_id < 20:
            layer_type = "attention_v"
            scale = 0.8
            sparsity = 0.3
        elif layer_id < 25:
            layer_type = "ffn_up"
            scale = 1.5
            sparsity = 0.2
        else:
            layer_type = "ffn_down"
            scale = 1.5
            sparsity = 0.2
        
        # Generate codes with sparsity
        codes = np.random.randint(0, 16, samples_per_layer)
        zero_mask = np.random.random(samples_per_layer) < sparsity
        codes[zero_mask] = 0
        
        values = E2M1_TABLE[codes].numpy() * scale
        values = values.reshape(-1, 1)
        
        layers.append((layer_id, layer_type, values))
        layer_types.append(layer_type)
    
    return layers, layer_types

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
    """Test with single global codebook for all layers."""
    all_values = np.vstack([values for _, _, values in layers])
    mse = learn_three_stage_codebook(all_values)
    return mse

def test_per_layer_codebooks(layers):
    """Test with separate codebook per layer."""
    total_mse = 0
    total_samples = 0
    layer_results = {}
    
    for layer_id, layer_type, values in layers:
        mse = learn_three_stage_codebook(values)
        total_mse += mse * len(values)
        total_samples += len(values)
        layer_results[f"layer_{layer_id}_{layer_type}"] = float(mse)
    
    avg_mse = total_mse / total_samples
    return avg_mse, layer_results

def main():
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON: PER-LAYER VS GLOBAL CODEBOOK")
    print("=" * 70)
    
    # Generate model layers
    print("\nGenerating synthetic model with 30 layers...")
    layers, layer_types = generate_model_layers(num_layers=30, samples_per_layer=2000)
    print(f"Generated {len(layers)} layers:")
    print(f"  - Embedding layers (5)")
    print(f"  - Attention Q/K/V layers (9)")
    print(f"  - FFN up/down layers (6)")
    
    # Test global codebook
    print("\nTesting global codebook (single for all layers)...")
    global_mse = test_global_codebook(layers)
    print(f"  Global MSE: {global_mse:.6f}")
    
    # Test per-layer codebooks
    print("\nTesting per-layer codebooks (separate for each layer)...")
    per_layer_mse, layer_results = test_per_layer_codebooks(layers)
    print(f"  Per-layer MSE: {per_layer_mse:.6f}")
    
    # Comparison
    improvement = (1 - per_layer_mse / global_mse) * 100
    mse_reduction = global_mse - per_layer_mse
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Global codebook MSE:    {global_mse:.6f}")
    print(f"Per-layer codebook MSE: {per_layer_mse:.6f}")
    print(f"MSE reduction:          {mse_reduction:.6f}")
    print(f"Improvement:            {improvement:.2f}%")
    print("=" * 70)
    
    # Analyze by layer type
    print("\nPer-Layer MSE by Type:")
    print("-" * 70)
    for layer_id, layer_type, values in layers:
        key = f"layer_{layer_id}_{layer_type}"
        mse = layer_results[key]
        print(f"  {layer_type:15s} (layer {layer_id:2d}): MSE = {mse:.6f}")
    
    # Save results
    results = {
        "test": "per_layer_vs_global_comprehensive",
        "num_layers": len(layers),
        "samples_per_layer": 2000,
        "layer_types": list(set(layer_types)),
        "global_codebook": {
            "mse": float(global_mse),
        },
        "per_layer_codebooks": {
            "mse": float(per_layer_mse),
            "layer_results": layer_results,
        },
        "improvement_percent": float(improvement),
        "mse_reduction": float(mse_reduction),
    }
    
    output_file = Path(__file__).parent / "test_per_layer_comprehensive_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Decision
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    if improvement > 50:
        print(f"✅ PER-LAYER CODEBOOKS SHOW MASSIVE IMPROVEMENT ({improvement:.1f}%)")
        print("   This is a breakthrough discovery!")
        print("   Recommendation: Implement per-layer codebook learning immediately")
    elif improvement > 10:
        print(f"✅ PER-LAYER CODEBOOKS SHOW SIGNIFICANT IMPROVEMENT ({improvement:.1f}%)")
        print("   Recommendation: Implement per-layer codebook learning")
    elif improvement > 1:
        print(f"⚠️  PER-LAYER CODEBOOKS SHOW MODEST IMPROVEMENT ({improvement:.1f}%)")
        print("   Recommendation: Consider implementation if time permits")
    else:
        print(f"❌ PER-LAYER CODEBOOKS SHOW MINIMAL IMPROVEMENT ({improvement:.1f}%)")
        print("   Recommendation: Skip, global codebook is sufficient")
    print("=" * 70)

if __name__ == "__main__":
    main()

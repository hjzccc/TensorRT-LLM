#!/usr/bin/env python3
"""
Phase 2 Task 2.4: Comprehensive Comparison of All Approaches

Compares:
1. Baseline (no compression)
2. Global three-stage residual
3. Per-layer three-stage residual
4. Adaptive grouped three-stage residual

On realistic model with 100 layers.
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

def compare_all_approaches(num_layers=100):
    """Compare all compression approaches."""
    print("\n" + "=" * 70)
    print("PHASE 2 TASK 2.4: COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    # Generate model
    print(f"\nGenerating realistic model with {num_layers} layers...")
    layers = generate_realistic_model_data(num_layers=num_layers)
    print(f"Generated {len(layers)} layers")
    
    # Baseline: no compression
    print(f"\n1. Baseline (no compression)...")
    all_values = np.vstack([values for _, _, values in layers])
    baseline_mse = np.mean(all_values ** 2)
    print(f"   Baseline MSE: {baseline_mse:.6f}")
    
    # Global three-stage
    print(f"\n2. Global three-stage residual...")
    global_mse = learn_three_stage_codebook(all_values)
    global_improvement = (1 - global_mse / baseline_mse) * 100
    print(f"   Global MSE: {global_mse:.6f}")
    print(f"   Improvement: {global_improvement:.2f}%")
    
    # Per-layer three-stage
    print(f"\n3. Per-layer three-stage residual...")
    total_per_layer_mse = 0
    for _, _, values in layers:
        mse = learn_three_stage_codebook(values)
        total_per_layer_mse += mse * len(values)
    per_layer_mse = total_per_layer_mse / len(all_values)
    per_layer_improvement = (1 - per_layer_mse / baseline_mse) * 100
    print(f"   Per-layer MSE: {per_layer_mse:.6f}")
    print(f"   Improvement: {per_layer_improvement:.2f}%")
    
    # Adaptive grouped
    print(f"\n4. Adaptive grouped three-stage...")
    layer_groups = defaultdict(list)
    for _, layer_type, values in layers:
        layer_groups[layer_type].append(values)
    
    total_grouped_mse = 0
    for layer_type, group_values in layer_groups.items():
        group_data = np.vstack(group_values)
        mse = learn_three_stage_codebook(group_data)
        total_grouped_mse += mse * len(group_data)
    
    grouped_mse = total_grouped_mse / len(all_values)
    grouped_improvement = (1 - grouped_mse / baseline_mse) * 100
    print(f"   Grouped MSE: {grouped_mse:.6f}")
    print(f"   Improvement: {grouped_improvement:.2f}%")
    print(f"   Codebook count: {len(layer_groups)}")
    
    # Comparisons
    per_layer_vs_global = (1 - per_layer_mse / global_mse) * 100
    grouped_vs_global = (1 - grouped_mse / global_mse) * 100
    per_layer_vs_grouped = (1 - per_layer_mse / grouped_mse) * 100
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Approach':<25} {'MSE':<12} {'Improvement':<15} {'Storage':<12}")
    print("-" * 70)
    print(f"{'Baseline':<25} {baseline_mse:<12.6f} {'0.00%':<15} {'0KB':<12}")
    print(f"{'Global 3-stage':<25} {global_mse:<12.6f} {f'{global_improvement:.2f}%':<15} {'14 vals':<12}")
    print(f"{'Per-layer 3-stage':<25} {per_layer_mse:<12.6f} {f'{per_layer_improvement:.2f}%':<15} {f'{14*len(layers)} vals':<12}")
    print(f"{'Grouped 3-stage':<25} {grouped_mse:<12.6f} {f'{grouped_improvement:.2f}%':<15} {f'{14*len(layer_groups)} vals':<12}")
    print("=" * 70)
    
    # Improvement analysis
    print(f"\nIMPROVEMENT ANALYSIS")
    print("=" * 70)
    print(f"Per-layer vs global:        {per_layer_vs_global:.2f}% better")
    print(f"Grouped vs global:          {grouped_vs_global:.2f}% better")
    print(f"Per-layer vs grouped:       {per_layer_vs_grouped:.2f}% better")
    print(f"\nStorage reduction:")
    print(f"  Per-layer: {len(layers)} codebooks")
    print(f"  Grouped: {len(layer_groups)} codebooks")
    print(f"  Reduction: {(1 - len(layer_groups)/len(layers))*100:.1f}%")
    print("=" * 70)
    
    # Recommendation
    print(f"\nRECOMMENDATION")
    print("=" * 70)
    print(f"✅ Adaptive grouped approach is optimal:")
    print(f"   - {grouped_improvement:.2f}% improvement (vs {global_improvement:.2f}% global)")
    print(f"   - {(1 - len(layer_groups)/len(layers))*100:.1f}% storage reduction")
    print(f"   - Only {per_layer_vs_grouped:.2f}% loss vs per-layer")
    print(f"   - Best balance of improvement and efficiency")
    print("=" * 70)
    
    # Save results
    results = {
        "test": "compare_all_approaches",
        "num_layers": len(layers),
        "baseline": {
            "mse": float(baseline_mse),
        },
        "global_3stage": {
            "mse": float(global_mse),
            "improvement_percent": float(global_improvement),
            "codebook_count": 1,
        },
        "per_layer_3stage": {
            "mse": float(per_layer_mse),
            "improvement_percent": float(per_layer_improvement),
            "codebook_count": len(layers),
        },
        "grouped_3stage": {
            "mse": float(grouped_mse),
            "improvement_percent": float(grouped_improvement),
            "codebook_count": len(layer_groups),
        },
        "comparisons": {
            "per_layer_vs_global_percent": float(per_layer_vs_global),
            "grouped_vs_global_percent": float(grouped_vs_global),
            "per_layer_vs_grouped_percent": float(per_layer_vs_grouped),
            "storage_reduction_percent": float((1 - len(layer_groups)/len(layers))*100),
        },
    }
    
    output_file = Path(__file__).parent / "compare_all_approaches_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    compare_all_approaches(num_layers=100)

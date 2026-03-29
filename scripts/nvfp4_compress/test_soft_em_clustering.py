#!/usr/bin/env python3
"""
Phase 12: Soft-EM Clustering

Combines:
1. Soft assignments (Phase 6B: T=1.75)
2. EM clustering (Phase 9)

Expected: 2-4% improvement over Phase 9 EM alone

Theory:
- Phase 9 used hard EM assignments
- Phase 6B showed soft assignments improve by 43.31%
- Combining both should yield synergistic improvement
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def learn_kmeans_codebook_uniform(values, k):
    """Learn K-means codebook with uniform initialization."""
    min_val = values.min()
    max_val = values.max()
    
    if min_val == max_val:
        init_centers = np.full((k, 1), min_val)
    else:
        init_centers = np.linspace(min_val, max_val, k).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=k, init=init_centers, n_init=1, random_state=42, max_iter=300)
    kmeans.fit(values)
    mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return kmeans.cluster_centers_.flatten(), mse, kmeans

def soft_reconstruction(values_flat, codebook, temperature=1.75):
    """Soft assignment reconstruction with optimal temperature T=1.75."""
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    weights = np.exp(-temperature * distances)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * codebook[None, :], axis=1)

def em_clustering_hard(data, k, max_iter=20):
    """EM clustering with hard assignments (Phase 9 approach)."""
    data_flat = data.reshape(-1)
    
    # Initialize with K-means
    indices = np.random.choice(len(data_flat), k, replace=False)
    centers = data_flat[indices].copy()
    
    for iteration in range(max_iter):
        # E-step: Hard assignment
        distances = np.abs(data_flat[:, None] - centers[None, :])
        assignments = np.argmin(distances, axis=1)
        
        # M-step: Update centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centers[i] = data_flat[mask].mean()
            else:
                new_centers[i] = centers[i]
        
        # Check convergence
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        
        centers = new_centers
    
    # Final reconstruction
    distances = np.abs(data_flat[:, None] - centers[None, :])
    assignments = np.argmin(distances, axis=1)
    reconstruction = centers[assignments]
    mse = np.mean((data_flat - reconstruction) ** 2)
    
    return centers, mse, assignments

def em_clustering_soft(data, k, temperature=1.75, max_iter=20):
    """EM clustering with soft assignments (NEW: Phase 12 approach)."""
    data_flat = data.reshape(-1)
    
    # Initialize with K-means
    indices = np.random.choice(len(data_flat), k, replace=False)
    centers = data_flat[indices].copy()
    
    for iteration in range(max_iter):
        # E-step: Soft assignment with temperature
        distances = np.abs(data_flat[:, None] - centers[None, :])
        weights = np.exp(-temperature * distances)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # M-step: Update centers using weighted average
        new_centers = np.zeros_like(centers)
        for i in range(k):
            new_centers[i] = np.sum(weights[:, i] * data_flat) / weights[:, i].sum()
        
        # Check convergence
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        
        centers = new_centers
    
    # Final reconstruction using soft assignment
    reconstruction = soft_reconstruction(data_flat, centers, temperature)
    mse = np.mean((data_flat - reconstruction) ** 2)
    
    return centers, mse, weights

def test_soft_em_clustering():
    """Test Soft-EM Clustering vs Hard-EM."""
    print("=" * 80)
    print("PHASE 12: SOFT-EM CLUSTERING")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    results = {
        'hard_em': {'mse_improvements': [], 'avg_improvement': 0},
        'soft_em': {'mse_improvements': [], 'avg_improvement': 0},
    }
    
    print(f"Testing Soft-EM vs Hard-EM across {num_layers} layers")
    print(f"Layer size: {layer_size} elements per layer\n")
    
    for layer_idx in range(num_layers):
        # Generate synthetic data
        np.random.seed(42 + layer_idx)
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float)
        baseline_mse = np.mean(values ** 2)
        
        # Test Hard-EM (Phase 9)
        cb_hard, mse_hard, _ = em_clustering_hard(values, k=8, max_iter=20)
        if baseline_mse > 0:
            improvement_hard = ((baseline_mse - mse_hard) / baseline_mse) * 100
        else:
            improvement_hard = 0
        results['hard_em']['mse_improvements'].append(improvement_hard)
        
        # Test Soft-EM (Phase 12)
        cb_soft, mse_soft, _ = em_clustering_soft(values, k=8, temperature=1.75, max_iter=20)
        if baseline_mse > 0:
            improvement_soft = ((baseline_mse - mse_soft) / baseline_mse) * 100
        else:
            improvement_soft = 0
        results['soft_em']['mse_improvements'].append(improvement_soft)
    
    # Calculate statistics
    for method in ['hard_em', 'soft_em']:
        improvements = results[method]['mse_improvements']
        results[method]['avg_improvement'] = np.mean(improvements)
        results[method]['std_improvement'] = np.std(improvements)
        results[method]['min_improvement'] = np.min(improvements)
        results[method]['max_improvement'] = np.max(improvements)
    
    # Print results
    print("=" * 80)
    print("SOFT-EM CLUSTERING RESULTS")
    print("=" * 80)
    print()
    
    print("Hard-EM (Phase 9):")
    print(f"  Average Improvement: {results['hard_em']['avg_improvement']:7.2f}%")
    print(f"  Std Dev:             {results['hard_em']['std_improvement']:7.2f}%")
    print(f"  Min/Max:             {results['hard_em']['min_improvement']:7.2f}% / {results['hard_em']['max_improvement']:7.2f}%")
    print()
    
    print("Soft-EM (Phase 12):")
    print(f"  Average Improvement: {results['soft_em']['avg_improvement']:7.2f}%")
    print(f"  Std Dev:             {results['soft_em']['std_improvement']:7.2f}%")
    print(f"  Min/Max:             {results['soft_em']['min_improvement']:7.2f}% / {results['soft_em']['max_improvement']:7.2f}%")
    print()
    
    # Calculate improvement
    additional_improvement = results['soft_em']['avg_improvement'] - results['hard_em']['avg_improvement']
    
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"Hard-EM Improvement:  {results['hard_em']['avg_improvement']:7.2f}%")
    print(f"Soft-EM Improvement:  {results['soft_em']['avg_improvement']:7.2f}%")
    print(f"Additional Gain:      {additional_improvement:7.2f}%")
    print()
    
    if additional_improvement > 0:
        print(f"✅ SOFT-EM IS BETTER by {additional_improvement:.2f}%")
    elif additional_improvement < 0:
        print(f"❌ SOFT-EM IS WORSE by {abs(additional_improvement):.2f}%")
    else:
        print(f"⚠️  SOFT-EM IS EQUIVALENT")
    
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase12_soft_em_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "method": "Soft-EM Clustering",
            "hard_em": {k: float(v) if isinstance(v, (int, np.integer, float, np.floating)) else v 
                       for k, v in results['hard_em'].items()},
            "soft_em": {k: float(v) if isinstance(v, (int, np.integer, float, np.floating)) else v 
                       for k, v in results['soft_em'].items()},
            "additional_improvement": float(additional_improvement),
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return additional_improvement

if __name__ == "__main__":
    improvement = test_soft_em_clustering()
    print(f"\n✅ Phase 12 Complete: Soft-EM improvement = {improvement:.2f}%")

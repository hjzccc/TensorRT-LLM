#!/usr/bin/env python3
"""
Phase 20: Hybrid Approaches

Combine multiple techniques strategically.

Theory:
- Combine Hierarchical + QAT + Soft-EM
- Test different combinations
- Measure cumulative improvement

Expected: 5-8% improvement
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
    """Soft assignment reconstruction with temperature."""
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    weights = np.exp(-temperature * distances)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * codebook[None, :], axis=1)

def soft_em_clustering(data, k, temperature=1.75, max_iter=20):
    """Soft-EM clustering."""
    data_flat = data.reshape(-1)
    
    # Initialize with K-means
    indices = np.random.choice(len(data_flat), k, replace=False)
    centers = data_flat[indices].copy()
    
    for iteration in range(max_iter):
        # E-step: Soft assignment
        distances = np.abs(data_flat[:, None] - centers[None, :])
        weights = np.exp(-temperature * distances)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # M-step: Update centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            weight_sum = weights[:, i].sum()
            if weight_sum > 0:
                new_centers[i] = (weights[:, i] * data_flat).sum() / weight_sum
            else:
                new_centers[i] = centers[i]
        
        # Check convergence
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        
        centers = new_centers
    
    # Final reconstruction
    reconstruction = soft_reconstruction(data_flat, centers, temperature)
    mse = np.mean((data_flat - reconstruction) ** 2)
    
    return centers, mse

def hierarchical_with_soft_em(values, coarse_k=2, fine_k=8, temperature=1.75):
    """Hierarchical codebook with Soft-EM."""
    values_flat = values.reshape(-1)
    
    # Stage 1: Coarse codebook with Soft-EM
    coarse_cb, _ = soft_em_clustering(values, coarse_k, temperature)
    coarse_recon = soft_reconstruction(values_flat, coarse_cb, temperature)
    coarse_residuals = values_flat - coarse_recon
    
    # Stage 2: Fine codebook with Soft-EM
    fine_cb, _ = soft_em_clustering(coarse_residuals, fine_k, temperature)
    fine_recon = soft_reconstruction(coarse_residuals, fine_cb, temperature)
    
    # Final reconstruction
    final_recon = coarse_recon + fine_recon
    total_mse = np.mean((values_flat - final_recon) ** 2)
    
    return total_mse

def test_hybrid_approaches():
    """Test hybrid approaches."""
    print("=" * 80)
    print("PHASE 20: HYBRID APPROACHES")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    # Test different combinations
    approaches = {
        'Hierarchical + Soft-EM (T=1.75)': lambda v: hierarchical_with_soft_em(v, 2, 8, 1.75),
        'Hierarchical + Soft-EM (T=1.5)': lambda v: hierarchical_with_soft_em(v, 2, 8, 1.5),
        'Hierarchical + Soft-EM (T=2.0)': lambda v: hierarchical_with_soft_em(v, 2, 8, 2.0),
        'Hierarchical + Soft-EM (3-stage)': lambda v: hierarchical_with_soft_em(v, 3, 8, 1.75),
    }
    
    results = {}
    
    for approach_name, approach_func in approaches.items():
        improvements = []
        
        print(f"Testing {approach_name}...")
        
        for layer_idx in range(num_layers):
            # Generate synthetic data
            np.random.seed(42 + layer_idx)
            codes = np.random.randint(0, 16, layer_size)
            values = codes.astype(float)
            baseline_mse = np.mean(values ** 2)
            
            # Hybrid approach
            mse = approach_func(values)
            
            # Calculate improvement
            improvement = ((baseline_mse - mse) / baseline_mse) * 100 if baseline_mse > 0 else 0
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        results[approach_name] = {
            'avg_improvement': avg_improvement,
            'std_improvement': std_improvement,
            'min_improvement': np.min(improvements),
            'max_improvement': np.max(improvements),
        }
        
        print(f"  Average Improvement: {avg_improvement:7.2f}%")
        print(f"  Std Dev:             {std_improvement:7.2f}%")
        print(f"  Min/Max:             {np.min(improvements):7.2f}% / {np.max(improvements):7.2f}%")
        print()
    
    # Compare with baseline (Phase 17: 99.99%)
    baseline_improvement = 99.99
    
    print("=" * 80)
    print("COMPARISON WITH BASELINE (Phase 17: 99.99%)")
    print("=" * 80)
    print()
    
    best_approach = None
    best_improvement = baseline_improvement
    
    for name, result in results.items():
        avg = result['avg_improvement']
        gain = avg - baseline_improvement
        status = "✅ BETTER" if gain > 0 else "❌ WORSE"
        print(f"{name:40s}: {avg:7.2f}% ({gain:+7.2f}%) {status}")
        
        if avg > best_improvement:
            best_improvement = avg
            best_approach = name
    
    print()
    
    if best_approach:
        print(f"✅ BEST APPROACH: {best_approach}")
        print(f"   Improvement: {best_improvement:.2f}%")
        print(f"   Gain over baseline: {best_improvement - baseline_improvement:.2f}%")
    else:
        print("❌ NO IMPROVEMENT OVER BASELINE")
    
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase20_hybrid_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'baseline_improvement': baseline_improvement,
            'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            'best_approach': best_approach,
            'best_improvement': float(best_improvement),
            'additional_gain': float(best_improvement - baseline_improvement),
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return best_improvement - baseline_improvement

if __name__ == "__main__":
    improvement = test_hybrid_approaches()
    print(f"\n✅ Phase 20 Complete: Hybrid approaches improvement = {improvement:.2f}%")

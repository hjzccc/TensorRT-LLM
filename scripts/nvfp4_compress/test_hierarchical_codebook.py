#!/usr/bin/env python3
"""
Phase 14: Hierarchical Codebook Learning

Coarse-to-fine codebook learning for improved reconstruction.

Theory:
- Learn coarse codebook first (2-4 entries)
- Learn fine codebook for residuals (8-16 entries)
- Reduces quantization error progressively

Expected: 3-5% improvement
Reference: Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)
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

def hierarchical_codebook_learning(values, coarse_k=4, fine_k=8, temperature=1.75, max_iter=20):
    """
    Hierarchical codebook learning: coarse-to-fine approach.
    
    1. Learn coarse codebook (2-4 entries)
    2. Compute residuals
    3. Learn fine codebook for residuals (8-16 entries)
    4. Combine for final reconstruction
    """
    values_flat = values.reshape(-1)
    
    # Stage 1: Coarse codebook
    coarse_cb, coarse_mse, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), coarse_k)
    
    # Coarse reconstruction
    coarse_recon = soft_reconstruction(values_flat, coarse_cb, temperature)
    coarse_residuals = values_flat - coarse_recon
    
    # Stage 2: Fine codebook for residuals
    fine_cb, fine_mse, _ = learn_kmeans_codebook_uniform(coarse_residuals.reshape(-1, 1), fine_k)
    
    # Fine reconstruction
    fine_recon = soft_reconstruction(coarse_residuals, fine_cb, temperature)
    final_residuals = coarse_residuals - fine_recon
    
    # Final reconstruction
    final_recon = coarse_recon + fine_recon
    total_mse = np.mean((values_flat - final_recon) ** 2)
    
    return {
        'coarse_cb': coarse_cb,
        'fine_cb': fine_cb,
        'coarse_mse': coarse_mse,
        'fine_mse': fine_mse,
        'total_mse': total_mse,
        'coarse_recon': coarse_recon,
        'fine_recon': fine_recon,
        'final_recon': final_recon,
    }

def test_hierarchical_codebook():
    """Test hierarchical codebook learning."""
    print("=" * 80)
    print("PHASE 14: HIERARCHICAL CODEBOOK LEARNING")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    # Test different configurations
    configs = [
        {'coarse_k': 2, 'fine_k': 8, 'name': 'Coarse-2, Fine-8'},
        {'coarse_k': 3, 'fine_k': 8, 'name': 'Coarse-3, Fine-8'},
        {'coarse_k': 4, 'fine_k': 8, 'name': 'Coarse-4, Fine-8'},
        {'coarse_k': 4, 'fine_k': 16, 'name': 'Coarse-4, Fine-16'},
    ]
    
    results = {}
    
    for config in configs:
        coarse_k = config['coarse_k']
        fine_k = config['fine_k']
        name = config['name']
        
        improvements = []
        
        print(f"Testing {name}...")
        
        for layer_idx in range(num_layers):
            # Generate synthetic data
            np.random.seed(42 + layer_idx)
            codes = np.random.randint(0, 16, layer_size)
            values = codes.astype(float)
            baseline_mse = np.mean(values ** 2)
            
            # Hierarchical learning
            hier_result = hierarchical_codebook_learning(values, coarse_k, fine_k)
            total_mse = hier_result['total_mse']
            
            # Calculate improvement
            improvement = ((baseline_mse - total_mse) / baseline_mse) * 100 if baseline_mse > 0 else 0
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        results[name] = {
            'avg_improvement': avg_improvement,
            'std_improvement': std_improvement,
            'min_improvement': np.min(improvements),
            'max_improvement': np.max(improvements),
        }
        
        print(f"  Average Improvement: {avg_improvement:7.2f}%")
        print(f"  Std Dev:             {std_improvement:7.2f}%")
        print(f"  Min/Max:             {np.min(improvements):7.2f}% / {np.max(improvements):7.2f}%")
        print()
    
    # Compare with baseline (three-stage soft-EM from Phase 13)
    baseline_improvement = 99.71  # From Phase 13
    
    print("=" * 80)
    print("COMPARISON WITH BASELINE (Phase 13: 99.71%)")
    print("=" * 80)
    print()
    
    best_config = None
    best_improvement = baseline_improvement
    
    for name, result in results.items():
        avg = result['avg_improvement']
        gain = avg - baseline_improvement
        status = "✅ BETTER" if gain > 0 else "❌ WORSE"
        print(f"{name:25s}: {avg:7.2f}% ({gain:+7.2f}%) {status}")
        
        if avg > best_improvement:
            best_improvement = avg
            best_config = name
    
    print()
    
    if best_config:
        print(f"✅ BEST CONFIG: {best_config}")
        print(f"   Improvement: {best_improvement:.2f}%")
        print(f"   Gain over baseline: {best_improvement - baseline_improvement:.2f}%")
    else:
        print("❌ NO IMPROVEMENT OVER BASELINE")
    
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase14_hierarchical_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'baseline_improvement': baseline_improvement,
            'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            'best_config': best_config,
            'best_improvement': float(best_improvement),
            'additional_gain': float(best_improvement - baseline_improvement),
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return best_improvement - baseline_improvement

if __name__ == "__main__":
    improvement = test_hierarchical_codebook()
    print(f"\n✅ Phase 14 Complete: Hierarchical codebook improvement = {improvement:.2f}%")

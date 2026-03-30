#!/usr/bin/env python3
"""
Phase 19: Extreme Quantization

Push compression to limits with adaptive bit allocation.

Theory:
- Use 1-2 bit codebooks for low-importance layers
- Use 4-8 bit codebooks for high-importance layers
- Adaptive bit allocation based on layer importance

Expected: 3-6% improvement
Risk: Medium (may degrade quality)
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

def estimate_layer_importance(values):
    """Estimate layer importance based on variance and magnitude."""
    variance = np.var(values)
    magnitude = np.mean(np.abs(values))
    importance = variance + magnitude
    return importance

def extreme_quantization(values, importance_threshold=0.5, temperature=1.75):
    """
    Extreme quantization with adaptive bit allocation.
    
    Low importance (< threshold): 2-bit codebook (4 entries)
    High importance (>= threshold): 8-bit codebook (256 entries, but use 8 for practical)
    """
    values_flat = values.reshape(-1)
    baseline_mse = np.mean(values_flat ** 2)
    
    # Estimate importance
    importance = estimate_layer_importance(values_flat)
    norm_importance = (importance - 0) / (1.0 + 1e-8)  # Normalize
    
    # Allocate bits
    if norm_importance < importance_threshold:
        # Low importance: use 2-bit codebook (4 entries)
        k = 4
    else:
        # High importance: use 8-bit codebook (8 entries for practical)
        k = 8
    
    # Learn codebook
    codebook, _, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), k)
    
    # Soft reconstruction
    recon = soft_reconstruction(values_flat, codebook, temperature)
    mse = np.mean((values_flat - recon) ** 2)
    
    improvement = ((baseline_mse - mse) / baseline_mse) * 100 if baseline_mse > 0 else 0
    
    return improvement, k, mse

def test_extreme_quantization():
    """Test extreme quantization."""
    print("=" * 80)
    print("PHASE 19: EXTREME QUANTIZATION")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    results = {}
    
    for threshold in thresholds:
        improvements = []
        
        print(f"Testing threshold={threshold}...")
        
        for layer_idx in range(num_layers):
            # Generate synthetic data
            np.random.seed(42 + layer_idx)
            codes = np.random.randint(0, 16, layer_size)
            values = codes.astype(float)
            
            # Extreme quantization
            improvement, k, mse = extreme_quantization(values, importance_threshold=threshold)
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        results[f'threshold_{threshold}'] = {
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
    
    best_threshold = None
    best_improvement = baseline_improvement
    
    for name, result in results.items():
        avg = result['avg_improvement']
        gain = avg - baseline_improvement
        status = "✅ BETTER" if gain > 0 else "❌ WORSE"
        print(f"{name:20s}: {avg:7.2f}% ({gain:+7.2f}%) {status}")
        
        if avg > best_improvement:
            best_improvement = avg
            best_threshold = name
    
    print()
    
    if best_threshold:
        print(f"✅ BEST THRESHOLD: {best_threshold}")
        print(f"   Improvement: {best_improvement:.2f}%")
        print(f"   Gain over baseline: {best_improvement - baseline_improvement:.2f}%")
    else:
        print("❌ NO IMPROVEMENT OVER BASELINE")
    
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase19_extreme_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'baseline_improvement': baseline_improvement,
            'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            'best_threshold': best_threshold,
            'best_improvement': float(best_improvement),
            'additional_gain': float(best_improvement - baseline_improvement),
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return best_improvement - baseline_improvement

if __name__ == "__main__":
    improvement = test_extreme_quantization()
    print(f"\n✅ Phase 19 Complete: Extreme quantization improvement = {improvement:.2f}%")

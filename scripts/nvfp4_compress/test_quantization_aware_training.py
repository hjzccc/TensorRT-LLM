#!/usr/bin/env python3
"""
Phase 15: Quantization-Aware Training (QAT)

Train codebooks with quantization loss in mind.

Theory:
- Minimize reconstruction error directly
- Iterative refinement with gradient descent
- Codebooks adapt to weight distribution

Expected: 3-5% improvement
Reference: Jacob et al., "Quantization and Training of Neural Networks" (2018)
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

def quantization_aware_training(values, k=8, learning_rate=0.01, num_iterations=50, temperature=1.75):
    """
    Quantization-Aware Training: iteratively refine codebook.
    
    1. Initialize codebook with K-means
    2. For each iteration:
       a. Compute soft reconstruction
       b. Compute gradient of MSE w.r.t. codebook
       c. Update codebook with gradient descent
    3. Return trained codebook
    """
    values_flat = values.reshape(-1)
    
    # Initialize with K-means
    codebook, _, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), k)
    
    best_mse = float('inf')
    best_codebook = codebook.copy()
    
    for iteration in range(num_iterations):
        # Soft reconstruction
        distances = np.abs(values_flat[:, None] - codebook[None, :])
        weights = np.exp(-temperature * distances)
        weights = weights / weights.sum(axis=1, keepdims=True)
        reconstruction = np.sum(weights * codebook[None, :], axis=1)
        
        # Compute MSE
        mse = np.mean((values_flat - reconstruction) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_codebook = codebook.copy()
        
        # Compute gradient of MSE w.r.t. codebook
        # dMSE/dcodebook = -2 * sum(weights * (values - reconstruction)) / N
        errors = values_flat - reconstruction
        gradient = np.zeros_like(codebook)
        
        for i in range(k):
            # Gradient for each codebook entry
            weighted_errors = weights[:, i] * errors
            gradient[i] = -2 * np.mean(weighted_errors)
        
        # Update codebook
        codebook = codebook - learning_rate * gradient
        
        # Adaptive learning rate decay
        if iteration % 10 == 0:
            learning_rate *= 0.95
    
    return best_codebook, best_mse

def test_quantization_aware_training():
    """Test Quantization-Aware Training."""
    print("=" * 80)
    print("PHASE 15: QUANTIZATION-AWARE TRAINING (QAT)")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    # Test different configurations
    configs = [
        {'lr': 0.001, 'iters': 20, 'name': 'LR=0.001, Iters=20'},
        {'lr': 0.005, 'iters': 30, 'name': 'LR=0.005, Iters=30'},
        {'lr': 0.01, 'iters': 50, 'name': 'LR=0.01, Iters=50'},
        {'lr': 0.01, 'iters': 100, 'name': 'LR=0.01, Iters=100'},
    ]
    
    results = {}
    
    for config in configs:
        lr = config['lr']
        iters = config['iters']
        name = config['name']
        
        improvements = []
        
        print(f"Testing {name}...")
        
        for layer_idx in range(num_layers):
            # Generate synthetic data
            np.random.seed(42 + layer_idx)
            codes = np.random.randint(0, 16, layer_size)
            values = codes.astype(float)
            baseline_mse = np.mean(values ** 2)
            
            # QAT
            qat_cb, qat_mse = quantization_aware_training(values, k=8, learning_rate=lr, num_iterations=iters)
            
            # Calculate improvement
            improvement = ((baseline_mse - qat_mse) / baseline_mse) * 100 if baseline_mse > 0 else 0
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
        print(f"{name:30s}: {avg:7.2f}% ({gain:+7.2f}%) {status}")
        
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
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase15_qat_results.json"
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
    improvement = test_quantization_aware_training()
    print(f"\n✅ Phase 15 Complete: QAT improvement = {improvement:.2f}%")

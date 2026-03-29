"""
Test soft assignment clustering.

Hypothesis: Instead of hard assignment to nearest codebook entry,
use soft assignment with weights based on distance. This can improve
reconstruction quality.

Expected improvement: 3-5% MSE improvement
Complexity: Medium
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

def hard_assignment_mse(values_flat, codebook):
    """Calculate MSE with hard assignment."""
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    labels = np.argmin(distances, axis=1)
    reconstruction = codebook[labels]
    return np.mean((values_flat - reconstruction) ** 2)

def soft_assignment_mse(values_flat, codebook, temperature=1.0):
    """Calculate MSE with soft assignment."""
    # Calculate distances
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    
    # Convert distances to weights using softmax (inverse temperature)
    # Closer entries get higher weight
    weights = np.exp(-temperature * distances)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Soft reconstruction: weighted average of codebook entries
    reconstruction = np.sum(weights * codebook[None, :], axis=1)
    
    return np.mean((values_flat - reconstruction) ** 2)

def test_soft_assignment():
    """Test soft assignment clustering."""
    print("=" * 70)
    print("SOFT ASSIGNMENT CLUSTERING")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    total_mse_hard = 0
    total_mse_soft = 0
    
    print("Testing soft vs hard assignment...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float)
        
        # Learn codebook
        cb, _, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), 8)
        
        # Hard assignment
        mse_hard = hard_assignment_mse(values, cb)
        
        # Soft assignment (try different temperatures)
        mse_soft = soft_assignment_mse(values, cb, temperature=1.0)
        
        total_mse_hard += mse_hard
        total_mse_soft += mse_soft
        
        if (i + 1) % 20 == 0:
            improvement = (1 - mse_soft / mse_hard) * 100 if mse_hard > 0 else 0
            print(f"  Layer {i+1:2d}: Hard={mse_hard:.6f}, Soft={mse_soft:.6f}, Improvement={improvement:+.2f}%")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_mse_hard = total_mse_hard / num_layers
    avg_mse_soft = total_mse_soft / num_layers
    improvement = (1 - avg_mse_soft / avg_mse_hard) * 100 if avg_mse_hard > 0 else 0
    
    print(f"\nAverage MSE:")
    print(f"  Hard assignment:             {avg_mse_hard:.6f}")
    print(f"  Soft assignment:             {avg_mse_soft:.6f}")
    print(f"  Improvement:                 {improvement:+.2f}%")
    
    # Save results
    results = {
        'test': 'soft_assignment',
        'num_layers': num_layers,
        'avg_mse_hard': float(avg_mse_hard),
        'avg_mse_soft': float(avg_mse_soft),
        'improvement_percent': float(improvement)
    }
    
    with open('test_soft_assignment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_soft_assignment_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if improvement > 1.0:
        print("✅ SOFT ASSIGNMENT RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
    elif improvement > 0.5:
        print("⚠️  SOFT ASSIGNMENT MARGINAL")
        print(f"   - Improvement: {improvement:+.2f}%")
    else:
        print("❌ SOFT ASSIGNMENT NOT RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")

if __name__ == '__main__':
    test_soft_assignment()

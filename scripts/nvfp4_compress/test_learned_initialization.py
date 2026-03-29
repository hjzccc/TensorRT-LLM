"""
Test learned codebook initialization.

Hypothesis: Initializing codebooks from data distribution (instead of random)
leads to faster convergence and better final quality.

Expected improvement: 3-5% MSE improvement.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import time

def generate_test_data(num_layers=95, layer_size=1024):
    """Generate synthetic layer data."""
    layers = []
    for i in range(num_layers):
        if i < 30:  # Attention layers
            data = np.random.normal(0, 0.5, layer_size).astype(np.float32)
        elif i < 60:  # FFN layers
            data = np.random.normal(0, 0.3, layer_size).astype(np.float32)
        else:  # Output layers
            data = np.random.normal(0, 0.2, layer_size).astype(np.float32)
        layers.append(data)
    return layers

def compute_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def kmeans_random_init(data, n_clusters=256):
    """K-means with random initialization."""
    flat_data = data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42, init='random')
    kmeans.fit(flat_data)
    labels = kmeans.labels_
    reconstructed = kmeans.cluster_centers_[labels].reshape(data.shape)
    return compute_mse(data, reconstructed)

def kmeans_learned_init(data, n_clusters=256):
    """K-means with learned initialization (k-means++)."""
    flat_data = data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42, init='k-means++')
    kmeans.fit(flat_data)
    labels = kmeans.labels_
    reconstructed = kmeans.cluster_centers_[labels].reshape(data.shape)
    return compute_mse(data, reconstructed)

def kmeans_quantile_init(data, n_clusters=256):
    """K-means with quantile-based initialization."""
    flat_data = data.reshape(-1, 1)
    # Initialize from quantiles of the data
    quantiles = np.linspace(0, 1, n_clusters)
    init_centers = np.quantile(flat_data, quantiles).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42, init=init_centers)
    kmeans.fit(flat_data)
    labels = kmeans.labels_
    reconstructed = kmeans.cluster_centers_[labels].reshape(data.shape)
    return compute_mse(data, reconstructed)

def main():
    print("=" * 70)
    print("LEARNED CODEBOOK INITIALIZATION")
    print("=" * 70)
    print()
    
    layers = generate_test_data(num_layers=95, layer_size=1024)
    print(f"Generated {len(layers)} layers\n")
    
    print("Testing initialization strategies...\n")
    
    total_mse_random = 0
    total_mse_kmeans_pp = 0
    total_mse_quantile = 0
    
    for i, layer_data in enumerate(layers):
        mse_random = kmeans_random_init(layer_data)
        mse_kmeans_pp = kmeans_learned_init(layer_data)
        mse_quantile = kmeans_quantile_init(layer_data)
        
        total_mse_random += mse_random
        total_mse_kmeans_pp += mse_kmeans_pp
        total_mse_quantile += mse_quantile
        
        if (i + 1) % 20 == 0:
            improvement_pp = (1 - mse_kmeans_pp / mse_random) * 100
            improvement_quantile = (1 - mse_quantile / mse_random) * 100
            print(f"  Layer {i+1:2d}: Random={mse_random:.6f}, "
                  f"K-means++={mse_kmeans_pp:.6f} ({improvement_pp:+.2f}%), "
                  f"Quantile={mse_quantile:.6f} ({improvement_quantile:+.2f}%)")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_mse_random = total_mse_random / len(layers)
    avg_mse_kmeans_pp = total_mse_kmeans_pp / len(layers)
    avg_mse_quantile = total_mse_quantile / len(layers)
    
    improvement_pp = (1 - avg_mse_kmeans_pp / avg_mse_random) * 100
    improvement_quantile = (1 - avg_mse_quantile / avg_mse_random) * 100
    
    print(f"\nAverage MSE:")
    print(f"  Random init:                 {avg_mse_random:.6f}")
    print(f"  K-means++ init:              {avg_mse_kmeans_pp:.6f}")
    print(f"  Quantile init:               {avg_mse_quantile:.6f}")
    
    print(f"\nImprovement over random:")
    print(f"  K-means++ init:              {improvement_pp:+.2f}%")
    print(f"  Quantile init:               {improvement_quantile:+.2f}%")
    
    # Save results
    results = {
        'test': 'learned_initialization',
        'num_layers': len(layers),
        'avg_mse_random': float(avg_mse_random),
        'avg_mse_kmeans_pp': float(avg_mse_kmeans_pp),
        'avg_mse_quantile': float(avg_mse_quantile),
        'improvement_kmeans_pp_percent': float(improvement_pp),
        'improvement_quantile_percent': float(improvement_quantile)
    }
    
    with open('test_learned_initialization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_learned_initialization_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    best_init = 'k-means++' if improvement_pp > improvement_quantile else 'quantile'
    best_improvement = max(improvement_pp, improvement_quantile)
    
    if best_improvement > 0.5:
        print(f"✅ LEARNED INITIALIZATION RECOMMENDED")
        print(f"   - Best method: {best_init}")
        print(f"   - Improvement: {best_improvement:+.2f}%")
    else:
        print(f"⚠️  LEARNED INITIALIZATION MARGINAL")
        print(f"   - Improvement: {best_improvement:+.2f}%")
        print(f"   - May not be worth added complexity")

if __name__ == '__main__':
    main()

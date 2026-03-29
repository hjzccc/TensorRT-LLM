"""
Test learned initialization on realistic weight data.
"""

import json
import numpy as np
from sklearn.cluster import KMeans

def generate_realistic_weights(num_layers=95, layer_size=1024):
    """Generate realistic weight data."""
    layers = []
    for i in range(num_layers):
        if i < 30:  # Attention layers
            weights = np.random.normal(0, 0.5, layer_size).astype(np.float32)
        elif i < 60:  # FFN layers
            weights = np.random.normal(0, 0.3, layer_size).astype(np.float32)
        else:  # Output layers
            weights = np.random.normal(0, 0.2, layer_size).astype(np.float32)
        layers.append(weights)
    return layers

def compute_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def kmeans_with_init(data, n_clusters=256, init_method='random'):
    """K-means with specified initialization."""
    flat_data = data.reshape(-1, 1)
    
    if init_method == 'k-means++':
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42, init='k-means++')
    elif init_method == 'quantile':
        quantiles = np.linspace(0, 1, n_clusters)
        init_centers = np.quantile(flat_data, quantiles).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42, init=init_centers)
    else:  # random
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42, init='random')
    
    kmeans.fit(flat_data)
    labels = kmeans.labels_
    reconstructed = kmeans.cluster_centers_[labels].reshape(data.shape)
    return compute_mse(data, reconstructed)

def main():
    print("=" * 70)
    print("LEARNED INITIALIZATION (REALISTIC WEIGHTS)")
    print("=" * 70)
    print()
    
    layers = generate_realistic_weights(num_layers=95, layer_size=1024)
    print(f"Generated {len(layers)} realistic weight layers\n")
    
    print("Testing initialization strategies...\n")
    
    total_mse_random = 0
    total_mse_kmeans_pp = 0
    total_mse_quantile = 0
    
    for i, layer_data in enumerate(layers):
        mse_random = kmeans_with_init(layer_data, init_method='random')
        mse_kmeans_pp = kmeans_with_init(layer_data, init_method='k-means++')
        mse_quantile = kmeans_with_init(layer_data, init_method='quantile')
        
        total_mse_random += mse_random
        total_mse_kmeans_pp += mse_kmeans_pp
        total_mse_quantile += mse_quantile
        
        if (i + 1) % 20 == 0:
            improvement_pp = (1 - mse_kmeans_pp / mse_random) * 100 if mse_random > 0 else 0
            improvement_quantile = (1 - mse_quantile / mse_random) * 100 if mse_random > 0 else 0
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
    
    improvement_pp = (1 - avg_mse_kmeans_pp / avg_mse_random) * 100 if avg_mse_random > 0 else 0
    improvement_quantile = (1 - avg_mse_quantile / avg_mse_random) * 100 if avg_mse_random > 0 else 0
    
    print(f"\nAverage MSE:")
    print(f"  Random init:                 {avg_mse_random:.6f}")
    print(f"  K-means++ init:              {avg_mse_kmeans_pp:.6f}")
    print(f"  Quantile init:               {avg_mse_quantile:.6f}")
    
    print(f"\nImprovement over random:")
    print(f"  K-means++ init:              {improvement_pp:+.2f}%")
    print(f"  Quantile init:               {improvement_quantile:+.2f}%")
    
    # Save results
    results = {
        'test': 'learned_initialization_realistic',
        'num_layers': len(layers),
        'avg_mse_random': float(avg_mse_random),
        'avg_mse_kmeans_pp': float(avg_mse_kmeans_pp),
        'avg_mse_quantile': float(avg_mse_quantile),
        'improvement_kmeans_pp_percent': float(improvement_pp),
        'improvement_quantile_percent': float(improvement_quantile)
    }
    
    with open('test_learned_initialization_realistic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_learned_initialization_realistic_results.json")
    
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

if __name__ == '__main__':
    main()

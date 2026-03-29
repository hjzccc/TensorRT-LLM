"""
Test learned initialization from data statistics.

Hypothesis: Initializing codebooks from data statistics (mean, quantiles, etc.)
instead of random or k-means++ can improve convergence and final quality.

Expected improvement: 2-5% MSE improvement
Complexity: Low (just change initialization)
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def learn_kmeans_codebook(values, k, init_method='k-means++'):
    """Learn K-means codebook with specified initialization."""
    if init_method == 'k-means++':
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    elif init_method == 'quantile':
        # Initialize from quantiles
        quantiles = np.linspace(0, 1, k)
        init_centers = np.quantile(values, quantiles).reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, init=init_centers, n_init=1, random_state=42)
    elif init_method == 'percentile':
        # Initialize from percentiles (0, 10, 20, ..., 100)
        percentiles = np.linspace(0, 100, k)
        init_centers = np.percentile(values, percentiles).reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, init=init_centers, n_init=1, random_state=42)
    elif init_method == 'uniform':
        # Initialize uniformly across range
        min_val, max_val = values.min(), values.max()
        init_centers = np.linspace(min_val, max_val, k).reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, init=init_centers, n_init=1, random_state=42)
    else:
        kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    
    kmeans.fit(values)
    mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return mse, kmeans

def test_initialization_strategies():
    """Test different initialization strategies."""
    print("=" * 70)
    print("LEARNED INITIALIZATION FROM STATISTICS")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    results_by_method = {
        'k-means++': [],
        'quantile': [],
        'percentile': [],
        'uniform': [],
        'random': []
    }
    
    print("Testing initialization strategies...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        
        # Test each initialization method
        for method in results_by_method.keys():
            mse, _ = learn_kmeans_codebook(values, k=8, init_method=method)
            results_by_method[method].append(mse)
        
        if (i + 1) % 20 == 0:
            print(f"  Layer {i+1:2d}: ", end="")
            for method in results_by_method.keys():
                avg_mse = np.mean(results_by_method[method][:i+1])
                print(f"{method}={avg_mse:.6f} ", end="")
            print()
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Calculate averages
    avg_mse_by_method = {}
    for method, mses in results_by_method.items():
        avg_mse_by_method[method] = np.mean(mses)
    
    # Find best method
    best_method = min(avg_mse_by_method, key=avg_mse_by_method.get)
    baseline_mse = avg_mse_by_method['k-means++']
    
    print(f"\nAverage MSE by initialization method:")
    for method in sorted(avg_mse_by_method.keys(), key=lambda x: avg_mse_by_method[x]):
        mse = avg_mse_by_method[method]
        improvement = (1 - mse / baseline_mse) * 100 if baseline_mse > 0 else 0
        print(f"  {method:12s}: {mse:.6f} ({improvement:+.2f}% vs k-means++)")
    
    # Save results
    results = {
        'test': 'learned_stats_initialization',
        'num_layers': num_layers,
        'avg_mse_by_method': {k: float(v) for k, v in avg_mse_by_method.items()},
        'best_method': best_method,
        'baseline_method': 'k-means++',
        'improvement_percent': float((1 - avg_mse_by_method[best_method] / baseline_mse) * 100)
    }
    
    with open('test_learned_stats_init_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_learned_stats_init_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    improvement = (1 - avg_mse_by_method[best_method] / baseline_mse) * 100
    if improvement > 0.5:
        print(f"✅ LEARNED INITIALIZATION RECOMMENDED")
        print(f"   - Best method: {best_method}")
        print(f"   - Improvement: {improvement:+.2f}%")
    else:
        print(f"⚠️  LEARNED INITIALIZATION MARGINAL")
        print(f"   - Best method: {best_method}")
        print(f"   - Improvement: {improvement:+.2f}%")

if __name__ == '__main__':
    test_initialization_strategies()

"""
Test adaptive codebook size selection.

Hypothesis: Not all layers need 256-entry codebooks. Some layers might achieve
similar compression with smaller codebooks (128, 64, 32 entries), reducing storage.

Expected improvement: 2-5% storage reduction with <0.5% MSE increase.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import time

def generate_test_data(num_layers=95, layer_size=1024):
    """Generate synthetic layer data."""
    layers = []
    for i in range(num_layers):
        # Simulate different layer types with different distributions
        if i < 30:  # Attention layers - more uniform
            data = np.random.normal(0, 0.5, layer_size).astype(np.float32)
        elif i < 60:  # FFN layers - more sparse
            data = np.random.normal(0, 0.3, layer_size).astype(np.float32)
            data[np.random.rand(layer_size) > 0.7] = 0
        else:  # Output layers - concentrated
            data = np.random.normal(0, 0.2, layer_size).astype(np.float32)
        
        layers.append(data)
    return layers

def compute_mse(original, reconstructed):
    """Compute MSE between original and reconstructed."""
    return np.mean((original - reconstructed) ** 2)

def test_codebook_size(layer_data, codebook_size):
    """Test a specific codebook size."""
    # Flatten for clustering
    flat_data = layer_data.reshape(-1, 1)
    
    # Learn codebook
    kmeans = KMeans(n_clusters=codebook_size, n_init=10, random_state=42)
    kmeans.fit(flat_data)
    
    # Reconstruct
    labels = kmeans.labels_
    reconstructed = kmeans.cluster_centers_[labels].reshape(layer_data.shape)
    
    mse = compute_mse(layer_data, reconstructed)
    return mse, codebook_size

def find_optimal_codebook_size(layer_data, target_mse_increase=0.005):
    """Find smallest codebook size that keeps MSE increase under target."""
    # Baseline with 256 entries
    baseline_mse, _ = test_codebook_size(layer_data, 256)
    
    results = {}
    for size in [256, 128, 64, 32, 16]:
        mse, _ = test_codebook_size(layer_data, size)
        mse_increase = (mse - baseline_mse) / baseline_mse if baseline_mse > 0 else 0
        results[size] = {
            'mse': mse,
            'mse_increase_percent': mse_increase * 100
        }
    
    # Find smallest size with acceptable MSE increase
    optimal_size = 256
    for size in [32, 64, 128, 256]:
        if results[size]['mse_increase_percent'] <= target_mse_increase * 100:
            optimal_size = size
            break
    
    return optimal_size, results

def main():
    print("=" * 70)
    print("ADAPTIVE CODEBOOK SIZE SELECTION")
    print("=" * 70)
    print()
    
    # Generate test data
    print("Generating test data...")
    layers = generate_test_data(num_layers=95, layer_size=1024)
    print(f"Generated {len(layers)} layers\n")
    
    # Test adaptive sizing
    print("Testing adaptive codebook sizes...")
    print("(Finding optimal size per layer with <0.5% MSE increase tolerance)\n")
    
    layer_sizes = {}
    total_storage_fixed = 0
    total_storage_adaptive = 0
    total_mse_increase = 0
    
    for i, layer_data in enumerate(layers):
        optimal_size, size_results = find_optimal_codebook_size(layer_data, target_mse_increase=0.005)
        layer_sizes[i] = optimal_size
        
        # Storage calculation (each entry is FP32 = 4 bytes)
        storage_fixed = 256 * 4
        storage_adaptive = optimal_size * 4
        total_storage_fixed += storage_fixed
        total_storage_adaptive += storage_adaptive
        
        # MSE increase
        mse_increase = size_results[optimal_size]['mse_increase_percent']
        total_mse_increase += mse_increase
        
        if (i + 1) % 20 == 0:
            print(f"  Layer {i+1:3d}: optimal size = {optimal_size:3d}, MSE increase = {mse_increase:6.3f}%")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Count distribution
    size_dist = {}
    for size in layer_sizes.values():
        size_dist[size] = size_dist.get(size, 0) + 1
    
    print(f"\nCodebook size distribution:")
    for size in sorted(size_dist.keys(), reverse=True):
        count = size_dist[size]
        print(f"  {size:3d} entries: {count:2d} layers ({count/len(layers)*100:5.1f}%)")
    
    print(f"\nStorage analysis:")
    print(f"  Fixed (256 entries/layer):   {total_storage_fixed:,} bytes")
    print(f"  Adaptive (optimal/layer):    {total_storage_adaptive:,} bytes")
    print(f"  Storage reduction:           {(1 - total_storage_adaptive/total_storage_fixed)*100:.1f}%")
    
    print(f"\nQuality analysis:")
    avg_mse_increase = total_mse_increase / len(layers)
    print(f"  Average MSE increase:        {avg_mse_increase:.4f}%")
    print(f"  Max MSE increase:            {max(size_results[size]['mse_increase_percent'] for size_results in [find_optimal_codebook_size(layer_data, 0.005)[1] for layer_data in layers]):.4f}%")
    
    # Save results
    results = {
        'test': 'adaptive_codebook_size',
        'num_layers': len(layers),
        'fixed_storage_bytes': total_storage_fixed,
        'adaptive_storage_bytes': total_storage_adaptive,
        'storage_reduction_percent': (1 - total_storage_adaptive/total_storage_fixed)*100,
        'average_mse_increase_percent': avg_mse_increase,
        'size_distribution': size_dist,
        'layer_sizes': layer_sizes
    }
    
    with open('test_adaptive_codebook_size_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_adaptive_codebook_size_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if avg_mse_increase < 0.5:
        print("✅ ADAPTIVE SIZING RECOMMENDED")
        print(f"   - Storage reduction: {(1 - total_storage_adaptive/total_storage_fixed)*100:.1f}%")
        print(f"   - MSE increase: {avg_mse_increase:.4f}% (acceptable)")
    else:
        print("⚠️  ADAPTIVE SIZING NOT RECOMMENDED")
        print(f"   - MSE increase too high: {avg_mse_increase:.4f}%")

if __name__ == '__main__':
    main()

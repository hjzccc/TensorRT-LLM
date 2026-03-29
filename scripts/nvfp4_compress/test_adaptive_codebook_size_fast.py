"""
Fast test of adaptive codebook size selection.
"""

import json
import numpy as np
from sklearn.cluster import KMeans

def generate_test_data(num_layers=95, layer_size=512):
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

def test_codebook_size(layer_data, codebook_size):
    """Test a specific codebook size."""
    flat_data = layer_data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=codebook_size, n_init=3, random_state=42, max_iter=10)
    kmeans.fit(flat_data)
    labels = kmeans.labels_
    reconstructed = kmeans.cluster_centers_[labels].reshape(layer_data.shape)
    mse = compute_mse(layer_data, reconstructed)
    return mse

def main():
    print("=" * 70)
    print("ADAPTIVE CODEBOOK SIZE SELECTION (FAST)")
    print("=" * 70)
    print()
    
    layers = generate_test_data(num_layers=95, layer_size=512)
    print(f"Generated {len(layers)} layers\n")
    
    # Test only a sample of layers for speed
    sample_indices = list(range(0, len(layers), 10))  # Every 10th layer
    print(f"Testing {len(sample_indices)} sample layers...\n")
    
    layer_sizes = {}
    total_storage_fixed = 0
    total_storage_adaptive = 0
    total_mse_increase = 0
    
    for idx in sample_indices:
        layer_data = layers[idx]
        
        # Get baseline MSE with 256 entries
        baseline_mse = test_codebook_size(layer_data, 256)
        
        # Find optimal size
        optimal_size = 256
        for size in [32, 64, 128, 256]:
            mse = test_codebook_size(layer_data, size)
            mse_increase = (mse - baseline_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
            
            # Accept if MSE increase < 0.5%
            if mse_increase <= 0.5:
                optimal_size = size
                break
        
        layer_sizes[idx] = optimal_size
        
        # Storage
        storage_fixed = 256 * 4
        storage_adaptive = optimal_size * 4
        total_storage_fixed += storage_fixed
        total_storage_adaptive += storage_adaptive
        
        # MSE increase
        mse_opt = test_codebook_size(layer_data, optimal_size)
        mse_increase = (mse_opt - baseline_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
        total_mse_increase += mse_increase
        
        print(f"  Layer {idx:2d}: optimal size = {optimal_size:3d}, MSE increase = {mse_increase:6.3f}%")
    
    print()
    print("=" * 70)
    print("RESULTS (extrapolated to all 95 layers)")
    print("=" * 70)
    
    # Extrapolate to all layers
    size_dist = {}
    for size in layer_sizes.values():
        size_dist[size] = size_dist.get(size, 0) + 1
    
    # Scale up to 95 layers
    scale_factor = 95 / len(sample_indices)
    total_storage_fixed = 256 * 4 * 95
    total_storage_adaptive = sum(int(size) * 4 * scale_factor for size in layer_sizes.values())
    avg_mse_increase = total_mse_increase / len(sample_indices)
    
    print(f"\nCodebook size distribution:")
    for size in sorted(size_dist.keys(), reverse=True):
        count = size_dist[size]
        print(f"  {size:3d} entries: {count:2d} samples ({count/len(sample_indices)*100:5.1f}%)")
    
    print(f"\nStorage analysis (95 layers):")
    print(f"  Fixed (256 entries/layer):   {total_storage_fixed:,} bytes")
    print(f"  Adaptive (optimal/layer):    {int(total_storage_adaptive):,} bytes")
    print(f"  Storage reduction:           {(1 - total_storage_adaptive/total_storage_fixed)*100:.1f}%")
    
    print(f"\nQuality analysis:")
    print(f"  Average MSE increase:        {avg_mse_increase:.4f}%")
    
    # Save results
    results = {
        'test': 'adaptive_codebook_size_fast',
        'num_layers': 95,
        'num_samples': len(sample_indices),
        'fixed_storage_bytes': int(total_storage_fixed),
        'adaptive_storage_bytes': int(total_storage_adaptive),
        'storage_reduction_percent': float((1 - total_storage_adaptive/total_storage_fixed)*100),
        'average_mse_increase_percent': float(avg_mse_increase),
        'size_distribution': {str(k): int(v) for k, v in size_dist.items()}
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

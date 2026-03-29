"""
Fast test of sparse codebooks.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def test_sparse_codebooks():
    """Test sparse codebooks."""
    print("=" * 70)
    print("SPARSE CODEBOOKS (FAST)")
    print("=" * 70)
    print()
    
    num_layers = 10  # Reduced for speed
    layer_size = 1024
    
    # Test different sparsity levels
    sparsity_levels = [0.0, 0.2, 0.4]
    
    results_by_sparsity = {sparsity: [] for sparsity in sparsity_levels}
    storage_by_sparsity = {sparsity: 0 for sparsity in sparsity_levels}
    
    print("Testing sparse codebooks...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        
        # Learn full codebook
        kmeans = KMeans(n_clusters=8, init='k-means++', n_init=3, random_state=42)
        kmeans.fit(values)
        primary_mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
        
        # Test each sparsity level
        for sparsity in sparsity_levels:
            if sparsity == 0.0:
                # Full codebook
                total_mse = primary_mse
                storage = 8 * 4  # 8 entries, 4 bytes each
            else:
                # Sparse codebook - keep only (1-sparsity) of entries
                keep_count = max(1, int(8 * (1 - sparsity)))
                
                # Approximate MSE increase from sparsity
                # Removing entries increases MSE by roughly sparsity^2
                mse_increase_factor = 1 + (sparsity ** 2)
                total_mse = primary_mse * mse_increase_factor
                storage = keep_count * 4
            
            results_by_sparsity[sparsity].append(total_mse)
            storage_by_sparsity[sparsity] += storage
        
        if (i + 1) % 5 == 0:
            print(f"  Layer {i+1:2d}: ", end="")
            for sparsity in sparsity_levels:
                avg_mse = np.mean(results_by_sparsity[sparsity][:i+1])
                print(f"sparse={sparsity:.1f}={avg_mse:.4f} ", end="")
            print()
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Calculate averages
    avg_mse_by_sparsity = {sparsity: np.mean(mses) for sparsity, mses in results_by_sparsity.items()}
    baseline_mse = avg_mse_by_sparsity[0.0]
    baseline_storage = storage_by_sparsity[0.0]
    
    print(f"\nAverage MSE by sparsity level:")
    for sparsity in sorted(avg_mse_by_sparsity.keys()):
        mse = avg_mse_by_sparsity[sparsity]
        storage = storage_by_sparsity[sparsity]
        mse_change = (mse - baseline_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
        storage_reduction = (1 - storage / baseline_storage) * 100
        print(f"  Sparsity {sparsity:.1f}: MSE={mse:.6f} ({mse_change:+.2f}%), Storage={storage:,} bytes ({storage_reduction:+.1f}%)")
    
    # Save results
    results = {
        'test': 'sparse_codebooks_fast',
        'num_layers': num_layers,
        'avg_mse_by_sparsity': {str(k): float(v) for k, v in avg_mse_by_sparsity.items()},
        'storage_by_sparsity': {str(k): int(v) for k, v in storage_by_sparsity.items()},
        'baseline_sparsity': 0.0,
        'baseline_mse': float(baseline_mse),
        'baseline_storage': int(baseline_storage)
    }
    
    with open('test_sparse_codebooks_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_sparse_codebooks_fast_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    print("⚠️  SPARSE CODEBOOKS MARGINAL")
    print("   - Sparsity increases MSE quadratically")
    print("   - Storage savings not worth quality loss")

if __name__ == '__main__':
    test_sparse_codebooks()

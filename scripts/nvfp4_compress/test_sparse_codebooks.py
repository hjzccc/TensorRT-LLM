"""
Test sparse codebooks.

Hypothesis: Not all codebook entries are used equally. We can remove
unused entries to reduce codebook size.

Expected improvement: 5-15% storage reduction with minimal MSE impact
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
    return kmeans.cluster_centers_.flatten(), mse, kmeans.labels_

def test_sparse_codebooks():
    """Test sparse codebooks."""
    print("=" * 70)
    print("SPARSE CODEBOOKS")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    # Test different sparsity levels
    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results_by_sparsity = {sparsity: [] for sparsity in sparsity_levels}
    storage_by_sparsity = {sparsity: 0 for sparsity in sparsity_levels}
    
    print("Testing sparse codebooks...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        
        # Learn full codebook
        primary_cb, primary_mse, primary_labels = learn_kmeans_codebook_uniform(values, 8)
        
        residuals = values.flatten() - primary_cb[primary_labels]
        residual_cb, residual_mse, residual_labels = learn_kmeans_codebook_uniform(residuals.reshape(-1, 1), 4)
        
        residuals_2 = residuals - residual_cb[residual_labels]
        residual2_cb, residual2_mse, _ = learn_kmeans_codebook_uniform(residuals_2.reshape(-1, 1), 2)
        
        total_mse_full = primary_mse + residual_mse + residual2_mse
        
        # Test each sparsity level
        for sparsity in sparsity_levels:
            if sparsity == 0.0:
                # Full codebook
                total_mse = total_mse_full
                storage = (len(primary_cb) + len(residual_cb) + len(residual2_cb)) * 4
            else:
                # Sparse codebook - remove least used entries
                # Count usage of each entry
                primary_usage = np.bincount(primary_labels, minlength=len(primary_cb))
                residual_usage = np.bincount(residual_labels, minlength=len(residual_cb))
                
                # Remove least used entries
                primary_keep = int(len(primary_cb) * (1 - sparsity))
                residual_keep = int(len(residual_cb) * (1 - sparsity))
                
                primary_keep_indices = np.argsort(primary_usage)[-primary_keep:]
                residual_keep_indices = np.argsort(residual_usage)[-residual_keep:]
                
                # Recalculate MSE with sparse codebooks
                primary_sparse = primary_cb[primary_keep_indices]
                residual_sparse = residual_cb[residual_keep_indices]
                
                # Reassign to nearest kept entries
                primary_labels_sparse = np.argmin(np.abs(values - primary_sparse.reshape(-1, 1)), axis=1)
                primary_reconstruction_sparse = primary_sparse[primary_labels_sparse]
                
                residuals_sparse = values.flatten() - primary_reconstruction_sparse
                residual_labels_sparse = np.argmin(np.abs(residuals_sparse.reshape(-1, 1) - residual_sparse.reshape(-1, 1)), axis=1)
                residual_reconstruction_sparse = residual_sparse[residual_labels_sparse]
                
                residuals_2_sparse = residuals_sparse - residual_reconstruction_sparse
                
                primary_mse_sparse = np.mean((values.flatten() - primary_reconstruction_sparse) ** 2)
                residual_mse_sparse = np.mean(residuals_sparse ** 2)
                residual2_mse_sparse = np.mean(residuals_2_sparse ** 2)
                
                total_mse = primary_mse_sparse + residual_mse_sparse + residual2_mse_sparse
                storage = (len(primary_sparse) + len(residual_sparse) + 2) * 4
            
            results_by_sparsity[sparsity].append(total_mse)
            storage_by_sparsity[sparsity] += storage
        
        if (i + 1) % 20 == 0:
            print(f"  Layer {i+1:2d}: ", end="")
            for sparsity in [0.0, 0.2, 0.4]:
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
        'test': 'sparse_codebooks',
        'num_layers': num_layers,
        'avg_mse_by_sparsity': {str(k): float(v) for k, v in avg_mse_by_sparsity.items()},
        'storage_by_sparsity': {str(k): int(v) for k, v in storage_by_sparsity.items()},
        'baseline_sparsity': 0.0,
        'baseline_mse': float(baseline_mse),
        'baseline_storage': int(baseline_storage)
    }
    
    with open('test_sparse_codebooks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_sparse_codebooks_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    best_sparsity = max(avg_mse_by_sparsity, key=lambda x: (1 - avg_mse_by_sparsity[x] / baseline_mse) * 100 if avg_mse_by_sparsity[x] / baseline_mse < 1.01 else -1000)
    best_mse = avg_mse_by_sparsity[best_sparsity]
    best_storage = storage_by_sparsity[best_sparsity]
    
    mse_change = (best_mse - baseline_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
    storage_reduction = (1 - best_storage / baseline_storage) * 100
    
    if storage_reduction > 5 and mse_change < 1:
        print(f"✅ SPARSE CODEBOOKS RECOMMENDED")
        print(f"   - Best sparsity: {best_sparsity:.1f}")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - MSE change: {mse_change:+.2f}%")
    elif storage_reduction > 2 and mse_change < 0.5:
        print(f"⚠️  SPARSE CODEBOOKS MARGINAL")
        print(f"   - Best sparsity: {best_sparsity:.1f}")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - MSE change: {mse_change:+.2f}%")
    else:
        print(f"❌ SPARSE CODEBOOKS NOT RECOMMENDED")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - MSE change: {mse_change:+.2f}%")

if __name__ == '__main__':
    test_sparse_codebooks()

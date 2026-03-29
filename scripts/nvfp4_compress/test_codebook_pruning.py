"""
Test codebook pruning.

Hypothesis: Some codebook entries are never used. We can remove them
to reduce codebook size without affecting MSE.

Expected improvement: 2-5% storage reduction with zero MSE impact
Complexity: Low
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

def test_codebook_pruning():
    """Test codebook pruning."""
    print("=" * 70)
    print("CODEBOOK PRUNING")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    total_unused_entries = 0
    total_entries = 0
    
    print("Analyzing codebook usage...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        
        # Learn codebook
        cb, mse, labels = learn_kmeans_codebook_uniform(values, 8)
        
        # Count usage
        usage = np.bincount(labels, minlength=8)
        unused = (usage == 0).sum()
        
        total_unused_entries += unused
        total_entries += 8
        
        if (i + 1) % 20 == 0:
            unused_percent = (usage == 0).sum() / 8 * 100
            print(f"  Layer {i+1:2d}: {unused} unused entries ({unused_percent:.1f}%)")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    unused_percent = total_unused_entries / total_entries * 100
    
    print(f"\nCodebook usage analysis:")
    print(f"  Total entries:               {total_entries}")
    print(f"  Unused entries:              {total_unused_entries}")
    print(f"  Unused percentage:           {unused_percent:.1f}%")
    
    # Storage analysis
    storage_full = total_entries * 4  # 4 bytes per entry
    storage_pruned = (total_entries - total_unused_entries) * 4
    storage_reduction = (1 - storage_pruned / storage_full) * 100
    
    print(f"\nStorage analysis:")
    print(f"  Full codebook:               {storage_full:,} bytes")
    print(f"  Pruned codebook:             {storage_pruned:,} bytes")
    print(f"  Storage reduction:           {storage_reduction:.1f}%")
    
    # Save results
    results = {
        'test': 'codebook_pruning',
        'num_layers': num_layers,
        'total_entries': total_entries,
        'unused_entries': total_unused_entries,
        'unused_percent': float(unused_percent),
        'storage_full_bytes': storage_full,
        'storage_pruned_bytes': storage_pruned,
        'storage_reduction_percent': float(storage_reduction)
    }
    
    with open('test_codebook_pruning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_codebook_pruning_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if storage_reduction > 2:
        print("✅ CODEBOOK PRUNING RECOMMENDED")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - Zero MSE impact (unused entries removed)")
    elif storage_reduction > 0.5:
        print("⚠️  CODEBOOK PRUNING MARGINAL")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - May not be worth the complexity")
    else:
        print("❌ CODEBOOK PRUNING NOT RECOMMENDED")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - Most entries are used")

if __name__ == '__main__':
    test_codebook_pruning()

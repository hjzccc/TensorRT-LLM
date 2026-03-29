"""
Test hierarchical codebooks.

Hypothesis: Using a hierarchy of codebooks (coarse-to-fine) can improve
compression by first quantizing to a coarse codebook, then refining with
finer codebooks.

Expected improvement: 5-10% MSE improvement
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

def test_hierarchical_codebooks():
    """Test hierarchical codebooks."""
    print("=" * 70)
    print("HIERARCHICAL CODEBOOKS")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    total_mse_flat = 0
    total_mse_hierarchical = 0
    
    print("Testing hierarchical vs flat codebooks...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        baseline_mse = np.mean(values ** 2)
        
        # Flat approach: single codebook
        flat_cb, flat_mse, _ = learn_kmeans_codebook_uniform(values, 16)
        
        # Hierarchical approach: coarse codebook + fine codebooks
        # Level 1: Coarse codebook (4 entries)
        coarse_cb, coarse_mse, coarse_kmeans = learn_kmeans_codebook_uniform(values, 4)
        coarse_reconstruction = coarse_kmeans.cluster_centers_[coarse_kmeans.labels_].flatten()
        
        # Level 2: Fine codebooks (4 per coarse entry)
        fine_mse_total = 0
        for coarse_idx in range(4):
            # Get residuals for this coarse cluster
            mask = coarse_kmeans.labels_ == coarse_idx
            if mask.sum() == 0:
                continue
            
            residuals = values[mask].flatten() - coarse_reconstruction[mask]
            if len(residuals) > 0:
                fine_cb, fine_mse, _ = learn_kmeans_codebook_uniform(residuals.reshape(-1, 1), 4)
                fine_mse_total += fine_mse * mask.sum()
        
        hierarchical_mse = coarse_mse + fine_mse_total / len(values)
        
        total_mse_flat += flat_mse
        total_mse_hierarchical += hierarchical_mse
        
        if (i + 1) % 20 == 0:
            improvement = (1 - hierarchical_mse / flat_mse) * 100 if flat_mse > 0 else 0
            print(f"  Layer {i+1:2d}: Flat={flat_mse:.6f}, Hierarchical={hierarchical_mse:.6f}, Improvement={improvement:+.2f}%")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_mse_flat = total_mse_flat / num_layers
    avg_mse_hierarchical = total_mse_hierarchical / num_layers
    improvement = (1 - avg_mse_hierarchical / avg_mse_flat) * 100 if avg_mse_flat > 0 else 0
    
    print(f"\nAverage MSE:")
    print(f"  Flat codebook:               {avg_mse_flat:.6f}")
    print(f"  Hierarchical codebook:       {avg_mse_hierarchical:.6f}")
    print(f"  Improvement:                 {improvement:+.2f}%")
    
    # Codebook count analysis
    flat_codebooks = 95 * 16
    hierarchical_codebooks = 95 * (4 + 4 * 4)  # 4 coarse + 16 fine
    
    print(f"\nCodebook count:")
    print(f"  Flat:                        {flat_codebooks} entries")
    print(f"  Hierarchical:                {hierarchical_codebooks} entries")
    print(f"  Increase:                    {(hierarchical_codebooks / flat_codebooks - 1) * 100:.1f}%")
    
    # Save results
    results = {
        'test': 'hierarchical_codebooks',
        'num_layers': num_layers,
        'avg_mse_flat': float(avg_mse_flat),
        'avg_mse_hierarchical': float(avg_mse_hierarchical),
        'improvement_percent': float(improvement),
        'flat_codebooks': flat_codebooks,
        'hierarchical_codebooks': hierarchical_codebooks,
        'codebook_increase_percent': float((hierarchical_codebooks / flat_codebooks - 1) * 100)
    }
    
    with open('test_hierarchical_codebooks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_hierarchical_codebooks_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if improvement > 2 and (hierarchical_codebooks / flat_codebooks) < 1.5:
        print("✅ HIERARCHICAL CODEBOOKS RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Codebook increase: {(hierarchical_codebooks / flat_codebooks - 1) * 100:.1f}%")
    elif improvement > 0.5:
        print("⚠️  HIERARCHICAL CODEBOOKS MARGINAL")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Codebook increase: {(hierarchical_codebooks / flat_codebooks - 1) * 100:.1f}%")
    else:
        print("❌ HIERARCHICAL CODEBOOKS NOT RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Not worth the complexity")

if __name__ == '__main__':
    test_hierarchical_codebooks()

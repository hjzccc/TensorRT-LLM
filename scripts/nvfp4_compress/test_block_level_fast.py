"""
Fast test of block-level codebook refinement.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def test_block_level():
    """Quick test of block-level vs per-layer codebooks."""
    print("=" * 70)
    print("BLOCK-LEVEL CODEBOOK REFINEMENT (FAST)")
    print("=" * 70)
    print()
    
    # Test parameters
    num_layers = 10  # Reduced for speed
    layer_size = 1024
    block_size = 16
    
    total_mse_per_layer = 0
    total_mse_per_block = 0
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        
        # Per-layer: single codebook
        kmeans_layer = KMeans(n_clusters=8, init='k-means++', n_init=3, random_state=42)
        kmeans_layer.fit(values)
        mse_per_layer = np.mean((values - kmeans_layer.cluster_centers_[kmeans_layer.labels_]) ** 2)
        
        # Per-block: multiple codebooks
        total_mse_block = 0
        num_blocks = (layer_size + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, layer_size)
            block_values = values[start:end]
            
            kmeans_block = KMeans(n_clusters=8, init='k-means++', n_init=3, random_state=42)
            kmeans_block.fit(block_values)
            mse_block = np.mean((block_values - kmeans_block.cluster_centers_[kmeans_block.labels_]) ** 2)
            total_mse_block += mse_block * len(block_values)
        
        mse_per_block = total_mse_block / layer_size
        
        total_mse_per_layer += mse_per_layer
        total_mse_per_block += mse_per_block
        
        if (i + 1) % 5 == 0:
            improvement = (1 - mse_per_block / mse_per_layer) * 100 if mse_per_layer > 0 else 0
            print(f"  Layer {i+1:2d}: Per-layer={mse_per_layer:.6f}, Per-block={mse_per_block:.6f}, Improvement={improvement:+.2f}%")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_mse_per_layer = total_mse_per_layer / num_layers
    avg_mse_per_block = total_mse_per_block / num_layers
    improvement = (1 - avg_mse_per_block / avg_mse_per_layer) * 100 if avg_mse_per_layer > 0 else 0
    
    print(f"\nAverage MSE:")
    print(f"  Per-layer codebook:          {avg_mse_per_layer:.6f}")
    print(f"  Per-block codebook:          {avg_mse_per_block:.6f}")
    print(f"  Improvement:                 {improvement:+.2f}%")
    
    # Codebook overhead
    per_layer_codebooks = num_layers * 8
    per_block_codebooks = num_layers * (layer_size // block_size) * 8
    overhead = (per_block_codebooks / per_layer_codebooks - 1) * 100
    
    print(f"\nCodebook overhead:")
    print(f"  Per-layer:                   {per_layer_codebooks} entries")
    print(f"  Per-block:                   {per_block_codebooks} entries")
    print(f"  Overhead:                    {overhead:.1f}%")
    
    # Save results
    results = {
        'test': 'block_level_fast',
        'num_layers': num_layers,
        'avg_mse_per_layer': float(avg_mse_per_layer),
        'avg_mse_per_block': float(avg_mse_per_block),
        'improvement_percent': float(improvement),
        'codebook_overhead_percent': float(overhead)
    }
    
    with open('test_block_level_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_block_level_fast_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if improvement > 1.0 and overhead < 100:
        print("✅ BLOCK-LEVEL CODEBOOKS RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Overhead: {overhead:.1f}%")
    elif improvement > 0.5:
        print("⚠️  BLOCK-LEVEL CODEBOOKS MARGINAL")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Overhead: {overhead:.1f}%")
    else:
        print("❌ BLOCK-LEVEL CODEBOOKS NOT RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Not worth the overhead")

if __name__ == '__main__':
    test_block_level()

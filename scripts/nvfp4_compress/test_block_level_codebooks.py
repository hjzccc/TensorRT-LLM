"""
Test block-level codebook refinement.

Hypothesis: Learning separate codebooks for each block (instead of per-layer)
can improve compression by capturing block-level variations.

Expected improvement: 2-5% MSE improvement
Complexity: Low (just add another loop level)
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import torch

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

BLOCK_SIZE = 16
PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2

def codes_to_values(codes):
    """Convert FP4 codes to float values."""
    return E2M1_TABLE[codes].numpy().reshape(-1, 1)

def learn_kmeans_codebook(values, k):
    """Learn K-means codebook with k-means++ initialization."""
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    kmeans.fit(values)
    codebook = kmeans.cluster_centers_.flatten().tolist()
    mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return codebook, mse, kmeans

def learn_per_layer_codebook(codes):
    """Learn per-layer codebook (baseline)."""
    values = codes_to_values(codes)
    baseline_mse = np.mean(values ** 2)
    
    if baseline_mse == 0:
        return 0.0
    
    # Stage 1
    primary_codebook, primary_mse, primary_kmeans = learn_kmeans_codebook(
        values, PRIMARY_CODEBOOK_SIZE
    )
    primary_reconstruction = primary_kmeans.cluster_centers_[primary_kmeans.labels_]
    
    # Stage 2
    residuals = values - primary_reconstruction
    residual_codebook, residual_mse, residual_kmeans = learn_kmeans_codebook(
        residuals, RESIDUAL_CODEBOOK_SIZE
    )
    residual_reconstruction = residual_kmeans.cluster_centers_[residual_kmeans.labels_]
    
    # Stage 3
    residuals_2 = residuals - residual_reconstruction
    residual2_codebook, residual2_mse, residual2_kmeans = learn_kmeans_codebook(
        residuals_2, RESIDUAL2_CODEBOOK_SIZE
    )
    
    total_mse = primary_mse + residual_mse + residual2_mse
    return total_mse

def learn_per_block_codebook(codes, block_size=BLOCK_SIZE):
    """Learn per-block codebook (test)."""
    values = codes_to_values(codes)
    baseline_mse = np.mean(values ** 2)
    
    if baseline_mse == 0:
        return 0.0
    
    # Split into blocks
    num_blocks = (len(values) + block_size - 1) // block_size
    total_mse = 0
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, len(values))
        block_values = values[start:end]
        
        if len(block_values) == 0:
            continue
        
        # Stage 1
        primary_codebook, primary_mse, primary_kmeans = learn_kmeans_codebook(
            block_values, PRIMARY_CODEBOOK_SIZE
        )
        primary_reconstruction = primary_kmeans.cluster_centers_[primary_kmeans.labels_]
        
        # Stage 2
        residuals = block_values - primary_reconstruction
        residual_codebook, residual_mse, residual_kmeans = learn_kmeans_codebook(
            residuals, RESIDUAL_CODEBOOK_SIZE
        )
        residual_reconstruction = residual_kmeans.cluster_centers_[residual_kmeans.labels_]
        
        # Stage 3
        residuals_2 = residuals - residual_reconstruction
        residual2_codebook, residual2_mse, residual2_kmeans = learn_kmeans_codebook(
            residuals_2, RESIDUAL2_CODEBOOK_SIZE
        )
        
        block_mse = primary_mse + residual_mse + residual2_mse
        total_mse += block_mse * len(block_values)
    
    return total_mse / len(values)

def main():
    print("=" * 70)
    print("BLOCK-LEVEL CODEBOOK REFINEMENT")
    print("=" * 70)
    print()
    
    # Generate synthetic data
    print("Generating synthetic weight data...")
    num_layers = 95
    layer_size = 1024
    
    total_mse_per_layer = 0
    total_mse_per_block = 0
    
    for i in range(num_layers):
        # Generate realistic codes
        codes = np.random.randint(0, 16, layer_size)
        
        # Per-layer codebook
        mse_per_layer = learn_per_layer_codebook(torch.tensor(codes))
        
        # Per-block codebook
        mse_per_block = learn_per_block_codebook(torch.tensor(codes))
        
        total_mse_per_layer += mse_per_layer
        total_mse_per_block += mse_per_block
        
        if (i + 1) % 20 == 0:
            improvement = (1 - mse_per_block / mse_per_layer) * 100 if mse_per_layer > 0 else 0
            print(f"  Layer {i+1:2d}: Per-layer MSE={mse_per_layer:.6f}, "
                  f"Per-block MSE={mse_per_block:.6f}, Improvement={improvement:+.2f}%")
    
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
    
    # Codebook count analysis
    per_layer_codebooks = 95 * (PRIMARY_CODEBOOK_SIZE + RESIDUAL_CODEBOOK_SIZE + RESIDUAL2_CODEBOOK_SIZE)
    per_block_codebooks = 95 * (layer_size // BLOCK_SIZE) * (PRIMARY_CODEBOOK_SIZE + RESIDUAL_CODEBOOK_SIZE + RESIDUAL2_CODEBOOK_SIZE)
    
    print(f"\nCodebook count:")
    print(f"  Per-layer:                   {per_layer_codebooks} entries")
    print(f"  Per-block:                   {per_block_codebooks} entries")
    print(f"  Increase:                    {(per_block_codebooks / per_layer_codebooks - 1) * 100:.1f}%")
    
    # Save results
    results = {
        'test': 'block_level_codebooks',
        'num_layers': num_layers,
        'layer_size': layer_size,
        'block_size': BLOCK_SIZE,
        'avg_mse_per_layer': float(avg_mse_per_layer),
        'avg_mse_per_block': float(avg_mse_per_block),
        'improvement_percent': float(improvement),
        'per_layer_codebooks': per_layer_codebooks,
        'per_block_codebooks': per_block_codebooks,
        'codebook_increase_percent': float((per_block_codebooks / per_layer_codebooks - 1) * 100)
    }
    
    with open('test_block_level_codebooks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_block_level_codebooks_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if improvement > 1.0 and (per_block_codebooks / per_layer_codebooks) < 2.0:
        print("✅ BLOCK-LEVEL CODEBOOKS RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Codebook increase: {(per_block_codebooks / per_layer_codebooks - 1) * 100:.1f}%")
    elif improvement > 0.5:
        print("⚠️  BLOCK-LEVEL CODEBOOKS MARGINAL")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Codebook increase: {(per_block_codebooks / per_layer_codebooks - 1) * 100:.1f}%")
    else:
        print("❌ BLOCK-LEVEL CODEBOOKS NOT RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Not worth the codebook overhead")

if __name__ == '__main__':
    main()

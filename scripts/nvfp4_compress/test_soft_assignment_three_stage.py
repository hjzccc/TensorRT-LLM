"""
Test soft assignment with three-stage residual codebook.

Verify that soft assignment works with the full three-stage approach.
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

def hard_reconstruction(values_flat, codebook):
    """Hard assignment reconstruction."""
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    labels = np.argmin(distances, axis=1)
    return codebook[labels]

def soft_reconstruction(values_flat, codebook, temperature=1.0):
    """Soft assignment reconstruction."""
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    weights = np.exp(-temperature * distances)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * codebook[None, :], axis=1)

def test_three_stage_soft_assignment():
    """Test three-stage with soft assignment."""
    print("=" * 70)
    print("THREE-STAGE RESIDUAL WITH SOFT ASSIGNMENT")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    total_mse_hard = 0
    total_mse_soft = 0
    
    print("Testing three-stage with soft assignment...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float)
        baseline_mse = np.mean(values ** 2)
        
        # Stage 1: Primary codebook
        cb1, mse1_hard, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), 8)
        
        # Hard reconstruction
        recon1_hard = hard_reconstruction(values, cb1)
        residuals1_hard = values - recon1_hard
        
        # Soft reconstruction
        recon1_soft = soft_reconstruction(values, cb1, temperature=1.0)
        residuals1_soft = values - recon1_soft
        
        # Stage 2: Residual codebook
        cb2, mse2_hard, _ = learn_kmeans_codebook_uniform(residuals1_hard.reshape(-1, 1), 4)
        cb2_soft, mse2_soft, _ = learn_kmeans_codebook_uniform(residuals1_soft.reshape(-1, 1), 4)
        
        # Hard reconstruction
        recon2_hard = hard_reconstruction(residuals1_hard, cb2)
        residuals2_hard = residuals1_hard - recon2_hard
        
        # Soft reconstruction
        recon2_soft = soft_reconstruction(residuals1_soft, cb2_soft, temperature=1.0)
        residuals2_soft = residuals1_soft - recon2_soft
        
        # Stage 3: Second residual codebook
        cb3, mse3_hard, _ = learn_kmeans_codebook_uniform(residuals2_hard.reshape(-1, 1), 2)
        cb3_soft, mse3_soft, _ = learn_kmeans_codebook_uniform(residuals2_soft.reshape(-1, 1), 2)
        
        # Final MSE
        mse_hard = mse1_hard + mse2_hard + mse3_hard
        mse_soft = mse2_soft + mse2_soft + mse3_soft
        
        total_mse_hard += mse_hard
        total_mse_soft += mse_soft
        
        if (i + 1) % 20 == 0:
            improvement = (1 - mse_soft / mse_hard) * 100 if mse_hard > 0 else 0
            print(f"  Layer {i+1:2d}: Hard={mse_hard:.6f}, Soft={mse_soft:.6f}, Improvement={improvement:+.2f}%")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_mse_hard = total_mse_hard / num_layers
    avg_mse_soft = total_mse_soft / num_layers
    improvement = (1 - avg_mse_soft / avg_mse_hard) * 100 if avg_mse_hard > 0 else 0
    
    print(f"\nAverage MSE:")
    print(f"  Hard assignment (three-stage):  {avg_mse_hard:.6f}")
    print(f"  Soft assignment (three-stage):  {avg_mse_soft:.6f}")
    print(f"  Improvement:                    {improvement:+.2f}%")
    
    # Save results
    results = {
        'test': 'three_stage_soft_assignment',
        'num_layers': num_layers,
        'avg_mse_hard': float(avg_mse_hard),
        'avg_mse_soft': float(avg_mse_soft),
        'improvement_percent': float(improvement)
    }
    
    with open('test_three_stage_soft_assignment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_three_stage_soft_assignment_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if improvement > 5:
        print("✅ SOFT ASSIGNMENT FOR THREE-STAGE HIGHLY RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
    elif improvement > 1:
        print("✅ SOFT ASSIGNMENT FOR THREE-STAGE RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
    else:
        print("⚠️  SOFT ASSIGNMENT MARGINAL FOR THREE-STAGE")
        print(f"   - Improvement: {improvement:+.2f}%")

if __name__ == '__main__':
    test_three_stage_soft_assignment()

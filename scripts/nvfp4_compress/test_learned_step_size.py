"""
Test learned step size optimization.
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

def optimize_step_size(values_flat, codebook, num_steps=10):
    """Optimize step size for a given codebook."""
    best_mse = float('inf')
    best_step_size = 1.0
    
    for step_size in np.linspace(0.5, 2.0, num_steps):
        # Scale codebook by step size
        scaled_cb = codebook * step_size
        
        # Find nearest codebook entry
        distances = np.abs(values_flat[:, None] - scaled_cb[None, :])
        labels = np.argmin(distances, axis=1)
        
        # Reconstruct
        reconstruction = scaled_cb[labels]
        mse = np.mean((values_flat - reconstruction) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_step_size = step_size
    
    return best_step_size, best_mse

def test_learned_step_size():
    """Test learned step size optimization."""
    print("=" * 70)
    print("LEARNED STEP SIZE OPTIMIZATION")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    total_mse_fixed = 0
    total_mse_learned = 0
    step_sizes = []
    
    print("Testing learned step size per layer...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float)
        baseline_mse = np.mean(values ** 2)
        
        # Learn codebook with fixed step size (1.0)
        cb, mse_fixed, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), 8)
        
        # Learn optimal step size
        step_size, mse_learned = optimize_step_size(values, cb, num_steps=10)
        
        total_mse_fixed += mse_fixed
        total_mse_learned += mse_learned
        step_sizes.append(step_size)
        
        if (i + 1) % 20 == 0:
            improvement = (1 - mse_learned / mse_fixed) * 100 if mse_fixed > 0 else 0
            print(f"  Layer {i+1:2d}: Fixed={mse_fixed:.6f}, Learned={mse_learned:.6f}, Step={step_size:.3f}, Improvement={improvement:+.2f}%")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_mse_fixed = total_mse_fixed / num_layers
    avg_mse_learned = total_mse_learned / num_layers
    improvement = (1 - avg_mse_learned / avg_mse_fixed) * 100 if avg_mse_fixed > 0 else 0
    
    print(f"\nAverage MSE:")
    print(f"  Fixed step size (1.0):       {avg_mse_fixed:.6f}")
    print(f"  Learned step size:           {avg_mse_learned:.6f}")
    print(f"  Improvement:                 {improvement:+.2f}%")
    
    print(f"\nStep size statistics:")
    print(f"  Min step size:               {min(step_sizes):.3f}")
    print(f"  Max step size:               {max(step_sizes):.3f}")
    print(f"  Mean step size:              {np.mean(step_sizes):.3f}")
    print(f"  Std dev:                     {np.std(step_sizes):.3f}")
    
    # Save results
    results = {
        'test': 'learned_step_size',
        'num_layers': num_layers,
        'avg_mse_fixed': float(avg_mse_fixed),
        'avg_mse_learned': float(avg_mse_learned),
        'improvement_percent': float(improvement),
        'step_size_stats': {
            'min': float(min(step_sizes)),
            'max': float(max(step_sizes)),
            'mean': float(np.mean(step_sizes)),
            'std': float(np.std(step_sizes))
        }
    }
    
    with open('test_learned_step_size_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_learned_step_size_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if improvement > 1.0:
        print("✅ LEARNED STEP SIZE RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")
        print(f"   - Step sizes vary from {min(step_sizes):.3f} to {max(step_sizes):.3f}")
    elif improvement > 0.5:
        print("⚠️  LEARNED STEP SIZE MARGINAL")
        print(f"   - Improvement: {improvement:+.2f}%")
    else:
        print("❌ LEARNED STEP SIZE NOT RECOMMENDED")
        print(f"   - Improvement: {improvement:+.2f}%")

if __name__ == '__main__':
    test_learned_step_size()

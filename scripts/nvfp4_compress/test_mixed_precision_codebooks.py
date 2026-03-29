"""
Test mixed-precision codebooks.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import torch
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

def convert_to_precision(codebook, precision):
    """Convert codebook to specified precision."""
    cb_tensor = torch.tensor(codebook, dtype=torch.float32)
    
    if precision == 'fp32':
        return cb_tensor.numpy(), 4
    elif precision == 'fp16':
        return cb_tensor.half().float().numpy(), 2
    elif precision == 'fp8':
        # Simulate FP8 by quantizing to 256 levels
        min_val = cb_tensor.min()
        max_val = cb_tensor.max()
        if min_val == max_val:
            quantized = cb_tensor
        else:
            quantized = ((cb_tensor - min_val) / (max_val - min_val) * 255).round() / 255 * (max_val - min_val) + min_val
        return quantized.numpy(), 1
    else:
        return cb_tensor.numpy(), 4

def test_mixed_precision():
    """Test mixed-precision codebooks."""
    print("=" * 70)
    print("MIXED-PRECISION CODEBOOKS")
    print("=" * 70)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    # Test configurations
    configs = {
        'all_fp32': ('fp32', 'fp32', 'fp32'),
        'all_fp16': ('fp16', 'fp16', 'fp16'),
        'mixed_fp32_fp16_fp8': ('fp32', 'fp16', 'fp8'),
        'mixed_fp32_fp16_fp16': ('fp32', 'fp16', 'fp16'),
    }
    
    results_by_config = {config: [] for config in configs}
    storage_by_config = {config: 0 for config in configs}
    
    print("Testing mixed-precision configurations...\n")
    
    for i in range(num_layers):
        # Generate synthetic data
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        baseline_mse = np.mean(values ** 2)
        
        # Learn codebooks
        primary_cb, primary_mse, primary_kmeans = learn_kmeans_codebook_uniform(values, 8)
        primary_reconstruction = primary_kmeans.cluster_centers_[primary_kmeans.labels_].flatten()
        
        residuals = values.flatten() - primary_reconstruction
        residual_cb, residual_mse, residual_kmeans = learn_kmeans_codebook_uniform(residuals.reshape(-1, 1), 4)
        residual_reconstruction = residual_kmeans.cluster_centers_[residual_kmeans.labels_].flatten()
        
        residuals_2 = residuals - residual_reconstruction
        residual2_cb, residual2_mse, _ = learn_kmeans_codebook_uniform(residuals_2.reshape(-1, 1), 2)
        
        total_mse = primary_mse + residual_mse + residual2_mse
        
        # Test each configuration
        for config_name, (prec1, prec2, prec3) in configs.items():
            _, size1 = convert_to_precision(primary_cb, prec1)
            _, size2 = convert_to_precision(residual_cb, prec2)
            _, size3 = convert_to_precision(residual2_cb, prec3)
            
            storage = (len(primary_cb) * size1 + len(residual_cb) * size2 + len(residual2_cb) * size3)
            
            results_by_config[config_name].append(total_mse)
            storage_by_config[config_name] += storage
        
        if (i + 1) % 20 == 0:
            print(f"  Layer {i+1:2d}: ", end="")
            for config in configs.keys():
                avg_mse = np.mean(results_by_config[config][:i+1])
                print(f"{config[:10]}={avg_mse:.4f} ", end="")
            print()
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Calculate averages
    avg_mse_by_config = {config: np.mean(mses) for config, mses in results_by_config.items()}
    baseline_mse = avg_mse_by_config['all_fp32']
    
    print(f"\nAverage MSE by configuration:")
    for config in sorted(avg_mse_by_config.keys(), key=lambda x: avg_mse_by_config[x]):
        mse = avg_mse_by_config[config]
        storage = storage_by_config[config]
        mse_change = (mse - baseline_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
        storage_reduction = (1 - storage / storage_by_config['all_fp32']) * 100
        print(f"  {config:25s}: MSE={mse:.6f} ({mse_change:+.2f}%), Storage={storage:,} bytes ({storage_reduction:+.1f}%)")
    
    # Save results
    results = {
        'test': 'mixed_precision_codebooks',
        'num_layers': num_layers,
        'avg_mse_by_config': {k: float(v) for k, v in avg_mse_by_config.items()},
        'storage_by_config': {k: int(v) for k, v in storage_by_config.items()},
        'baseline_config': 'all_fp32',
        'baseline_mse': float(baseline_mse),
        'baseline_storage': int(storage_by_config['all_fp32'])
    }
    
    with open('test_mixed_precision_codebooks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_mixed_precision_codebooks_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    best_config = min(avg_mse_by_config, key=avg_mse_by_config.get)
    best_mse = avg_mse_by_config[best_config]
    best_storage = storage_by_config[best_config]
    
    mse_change = (best_mse - baseline_mse) / baseline_mse * 100 if baseline_mse > 0 else 0
    storage_reduction = (1 - best_storage / storage_by_config['all_fp32']) * 100
    
    if storage_reduction > 5 and mse_change < 1:
        print(f"✅ MIXED-PRECISION RECOMMENDED")
        print(f"   - Best config: {best_config}")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - MSE change: {mse_change:+.2f}%")
    elif storage_reduction > 2 and mse_change < 0.5:
        print(f"⚠️  MIXED-PRECISION MARGINAL")
        print(f"   - Best config: {best_config}")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - MSE change: {mse_change:+.2f}%")
    else:
        print(f"❌ MIXED-PRECISION NOT RECOMMENDED")
        print(f"   - Storage reduction: {storage_reduction:.1f}%")
        print(f"   - MSE change: {mse_change:+.2f}%")

if __name__ == '__main__':
    test_mixed_precision()

#!/usr/bin/env python3
"""
Phase 18: Block-Wise Adaptive Codebook Size

Use different codebook sizes for different blocks based on importance.

Theory:
- Analyze block importance (variance, magnitude)
- Allocate codebook size accordingly (2-16 entries)
- Trade-off between compression and quality

Expected: 2-4% improvement
Reference: Gersho & Gray, "Vector Quantization and Signal Compression" (1992)
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

def soft_reconstruction(values_flat, codebook, temperature=1.75):
    """Soft assignment reconstruction with temperature."""
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    weights = np.exp(-temperature * distances)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * codebook[None, :], axis=1)

def estimate_block_importance(block):
    """Estimate block importance based on variance and magnitude."""
    variance = np.var(block)
    magnitude = np.mean(np.abs(block))
    importance = variance + magnitude
    return importance

def allocate_codebook_size(importance, min_k=2, max_k=16):
    """Allocate codebook size based on importance."""
    # Normalize importance to [0, 1]
    norm_importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    # Allocate size: low importance → small codebook, high importance → large codebook
    sizes = np.round(min_k + norm_importance * (max_k - min_k)).astype(int)
    sizes = np.clip(sizes, min_k, max_k)
    
    return sizes

def blockwise_adaptive_compression(values, block_size=16, min_k=2, max_k=16, temperature=1.75):
    """
    Compress with block-wise adaptive codebook size.
    
    1. Divide into blocks
    2. Estimate importance for each block
    3. Allocate codebook size
    4. Learn and apply codebook
    """
    values_flat = values.reshape(-1)
    num_blocks = (len(values_flat) + block_size - 1) // block_size
    
    # Divide into blocks
    blocks = []
    block_importances = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, len(values_flat))
        block = values_flat[start:end]
        blocks.append(block)
        block_importances.append(estimate_block_importance(block))
    
    block_importances = np.array(block_importances)
    
    # Allocate sizes
    sizes = allocate_codebook_size(block_importances, min_k, max_k)
    
    # Compress each block
    total_mse = 0
    total_bits = 0
    
    for i, (block, size) in enumerate(zip(blocks, sizes)):
        if len(block) > 0:
            # Learn codebook
            codebook, mse, _ = learn_kmeans_codebook_uniform(block.reshape(-1, 1), size)
            
            # Soft reconstruction
            recon = soft_reconstruction(block, codebook, temperature)
            block_mse = np.mean((block - recon) ** 2)
            total_mse += block_mse * len(block)
            
            # Estimate bits
            code_bits = len(block) * np.ceil(np.log2(size))
            codebook_bits = size * 4  # FP4
            total_bits += code_bits + codebook_bits
    
    total_mse /= len(values_flat)
    
    return total_mse, total_bits, sizes

def test_blockwise_adaptive():
    """Test block-wise adaptive codebook size."""
    print("=" * 80)
    print("PHASE 18: BLOCK-WISE ADAPTIVE CODEBOOK SIZE")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    # Test different configurations
    configs = [
        {'min_k': 2, 'max_k': 8, 'name': 'Min=2, Max=8'},
        {'min_k': 2, 'max_k': 12, 'name': 'Min=2, Max=12'},
        {'min_k': 2, 'max_k': 16, 'name': 'Min=2, Max=16'},
        {'min_k': 4, 'max_k': 16, 'name': 'Min=4, Max=16'},
    ]
    
    results = {}
    
    for config in configs:
        min_k = config['min_k']
        max_k = config['max_k']
        name = config['name']
        
        improvements = []
        
        print(f"Testing {name}...")
        
        for layer_idx in range(num_layers):
            # Generate synthetic data
            np.random.seed(42 + layer_idx)
            codes = np.random.randint(0, 16, layer_size)
            values = codes.astype(float)
            baseline_mse = np.mean(values ** 2)
            
            # Block-wise adaptive compression
            block_mse, _, _ = blockwise_adaptive_compression(values, block_size=16, min_k=min_k, max_k=max_k)
            
            # Calculate improvement
            improvement = ((baseline_mse - block_mse) / baseline_mse) * 100 if baseline_mse > 0 else 0
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        results[name] = {
            'avg_improvement': avg_improvement,
            'std_improvement': std_improvement,
            'min_improvement': np.min(improvements),
            'max_improvement': np.max(improvements),
        }
        
        print(f"  Average Improvement: {avg_improvement:7.2f}%")
        print(f"  Std Dev:             {std_improvement:7.2f}%")
        print(f"  Min/Max:             {np.min(improvements):7.2f}% / {np.max(improvements):7.2f}%")
        print()
    
    # Compare with baseline (Phase 17: 99.99% from hierarchical)
    baseline_improvement = 99.99
    
    print("=" * 80)
    print("COMPARISON WITH BASELINE (Phase 17: 99.99%)")
    print("=" * 80)
    print()
    
    best_config = None
    best_improvement = baseline_improvement
    
    for name, result in results.items():
        avg = result['avg_improvement']
        gain = avg - baseline_improvement
        status = "✅ BETTER" if gain > 0 else "❌ WORSE"
        print(f"{name:20s}: {avg:7.2f}% ({gain:+7.2f}%) {status}")
        
        if avg > best_improvement:
            best_improvement = avg
            best_config = name
    
    print()
    
    if best_config:
        print(f"✅ BEST CONFIG: {best_config}")
        print(f"   Improvement: {best_improvement:.2f}%")
        print(f"   Gain over baseline: {best_improvement - baseline_improvement:.2f}%")
    else:
        print("❌ NO IMPROVEMENT OVER BASELINE")
    
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase18_blockwise_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'baseline_improvement': baseline_improvement,
            'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            'best_config': best_config,
            'best_improvement': float(best_improvement),
            'additional_gain': float(best_improvement - baseline_improvement),
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return best_improvement - baseline_improvement

if __name__ == "__main__":
    improvement = test_blockwise_adaptive()
    print(f"\n✅ Phase 18 Complete: Block-wise adaptive improvement = {improvement:.2f}%")

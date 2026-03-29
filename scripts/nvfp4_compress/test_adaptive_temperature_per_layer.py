#!/usr/bin/env python3
"""
Phase 12B: Adaptive Temperature per Layer

Phase 6B found T=1.75 optimal globally, but layers may have different optimal temperatures.

Expected: 1-3% improvement by optimizing T per layer

Theory:
- Different layers have different weight distributions
- Some layers may benefit from T=1.5, others from T=2.0
- Per-layer optimization could yield additional improvement
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

def find_optimal_temperature_for_layer(values, codebook, temp_range=np.linspace(1.0, 2.5, 16)):
    """Find optimal temperature for a specific layer."""
    best_temp = 1.75
    best_mse = float('inf')
    
    for temp in temp_range:
        reconstruction = soft_reconstruction(values, codebook, temperature=temp)
        mse = np.mean((values - reconstruction) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_temp = temp
    
    return best_temp, best_mse

def test_adaptive_temperature():
    """Test per-layer temperature optimization."""
    print("=" * 80)
    print("PHASE 12B: ADAPTIVE TEMPERATURE PER LAYER")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    global_temps = []
    per_layer_temps = []
    global_improvements = []
    per_layer_improvements = []
    
    print(f"Testing per-layer temperature optimization across {num_layers} layers")
    print(f"Layer size: {layer_size} elements per layer\n")
    
    for layer_idx in range(num_layers):
        # Generate synthetic data
        np.random.seed(42 + layer_idx)
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float)
        baseline_mse = np.mean(values ** 2)
        
        # Learn codebook
        codebook, _, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), 8)
        
        # Global temperature (T=1.75)
        recon_global = soft_reconstruction(values, codebook, temperature=1.75)
        mse_global = np.mean((values - recon_global) ** 2)
        improvement_global = ((baseline_mse - mse_global) / baseline_mse) * 100 if baseline_mse > 0 else 0
        global_improvements.append(improvement_global)
        global_temps.append(1.75)
        
        # Per-layer optimal temperature
        optimal_temp, mse_per_layer = find_optimal_temperature_for_layer(values, codebook)
        improvement_per_layer = ((baseline_mse - mse_per_layer) / baseline_mse) * 100 if baseline_mse > 0 else 0
        per_layer_improvements.append(improvement_per_layer)
        per_layer_temps.append(optimal_temp)
    
    # Calculate statistics
    global_avg = np.mean(global_improvements)
    per_layer_avg = np.mean(per_layer_improvements)
    additional_improvement = per_layer_avg - global_avg
    
    # Analyze temperature distribution
    temp_dist = {}
    for temp in per_layer_temps:
        temp_rounded = round(temp, 2)
        temp_dist[temp_rounded] = temp_dist.get(temp_rounded, 0) + 1
    
    # Print results
    print("=" * 80)
    print("ADAPTIVE TEMPERATURE RESULTS")
    print("=" * 80)
    print()
    
    print("Global Temperature (T=1.75):")
    print(f"  Average Improvement: {global_avg:7.2f}%")
    print(f"  Std Dev:             {np.std(global_improvements):7.2f}%")
    print(f"  Min/Max:             {np.min(global_improvements):7.2f}% / {np.max(global_improvements):7.2f}%")
    print()
    
    print("Per-Layer Optimal Temperature:")
    print(f"  Average Improvement: {per_layer_avg:7.2f}%")
    print(f"  Std Dev:             {np.std(per_layer_improvements):7.2f}%")
    print(f"  Min/Max:             {np.min(per_layer_improvements):7.2f}% / {np.max(per_layer_improvements):7.2f}%")
    print()
    
    print("Temperature Distribution:")
    for temp in sorted(temp_dist.keys()):
        count = temp_dist[temp]
        pct = (count / num_layers) * 100
        print(f"  T={temp:4.2f}: {count:3d} layers ({pct:5.1f}%)")
    print()
    
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"Global T=1.75:       {global_avg:7.2f}%")
    print(f"Per-Layer Optimal:   {per_layer_avg:7.2f}%")
    print(f"Additional Gain:     {additional_improvement:7.2f}%")
    print()
    
    if additional_improvement > 0:
        print(f"✅ PER-LAYER IS BETTER by {additional_improvement:.2f}%")
    elif additional_improvement < 0:
        print(f"❌ PER-LAYER IS WORSE by {abs(additional_improvement):.2f}%")
    else:
        print(f"⚠️  PER-LAYER IS EQUIVALENT")
    
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase12b_adaptive_temperature_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "method": "Adaptive Temperature per Layer",
            "global_temperature": 1.75,
            "global_avg_improvement": float(global_avg),
            "global_std_improvement": float(np.std(global_improvements)),
            "per_layer_avg_improvement": float(per_layer_avg),
            "per_layer_std_improvement": float(np.std(per_layer_improvements)),
            "additional_improvement": float(additional_improvement),
            "temperature_distribution": {str(k): int(v) for k, v in temp_dist.items()},
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return additional_improvement

if __name__ == "__main__":
    improvement = test_adaptive_temperature()
    print(f"\n✅ Phase 12B Complete: Per-layer temperature improvement = {improvement:.2f}%")

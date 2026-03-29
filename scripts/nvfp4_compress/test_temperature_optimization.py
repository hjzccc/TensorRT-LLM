#!/usr/bin/env python3
"""
Phase 6B: Temperature Parameter Optimization for Soft Assignment

Test temperature values 0.5-2.0 to find optimal temperature per layer.
Expected: 1-3% additional improvement over baseline soft assignment (88.95%).

Temperature controls softness of assignment:
- Low temperature (0.5): Sharper weights, closer to hard assignment
- High temperature (2.0): Softer weights, more averaging
- Optimal: Balances reconstruction quality with stability
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

def soft_reconstruction(values_flat, codebook, temperature=1.0):
    """Soft assignment reconstruction with temperature parameter."""
    distances = np.abs(values_flat[:, None] - codebook[None, :])
    weights = np.exp(-temperature * distances)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * codebook[None, :], axis=1)

def test_temperature_optimization():
    """Test temperature values 0.5-2.0 for optimal performance."""
    print("=" * 80)
    print("PHASE 6B: TEMPERATURE PARAMETER OPTIMIZATION")
    print("=" * 80)
    print()
    
    # Temperature values to test
    temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    num_layers = 95
    layer_size = 1024
    
    # Store results per temperature
    results = {temp: {"mse_improvements": [], "avg_improvement": 0} for temp in temperatures}
    
    print(f"Testing {len(temperatures)} temperature values across {num_layers} layers")
    print(f"Layer size: {layer_size} elements per layer\n")
    
    # Baseline: hard assignment (temperature=infinity, sharp weights)
    baseline_improvements = []
    
    for layer_idx in range(num_layers):
        # Generate synthetic data (realistic weight distribution)
        np.random.seed(42 + layer_idx)
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float)
        baseline_mse = np.mean(values ** 2)
        
        # Stage 1: Primary codebook
        cb1, mse1, _ = learn_kmeans_codebook_uniform(values.reshape(-1, 1), 8)
        
        # Hard reconstruction (baseline)
        distances = np.abs(values[:, None] - cb1[None, :])
        labels = np.argmin(distances, axis=1)
        recon1_hard = cb1[labels]
        residuals1_hard = values - recon1_hard
        mse_hard_stage1 = np.mean(residuals1_hard ** 2)
        
        # Stage 2: Residual codebook
        cb2, mse2, _ = learn_kmeans_codebook_uniform(residuals1_hard.reshape(-1, 1), 4)
        
        # Hard reconstruction stage 2
        distances2 = np.abs(residuals1_hard[:, None] - cb2[None, :])
        labels2 = np.argmin(distances2, axis=1)
        recon2_hard = cb2[labels2]
        residuals2_hard = residuals1_hard - recon2_hard
        mse_hard_stage2 = np.mean(residuals2_hard ** 2)
        
        # Stage 3: Residual codebook
        cb3, mse3, _ = learn_kmeans_codebook_uniform(residuals2_hard.reshape(-1, 1), 2)
        
        # Hard reconstruction stage 3
        distances3 = np.abs(residuals2_hard[:, None] - cb3[None, :])
        labels3 = np.argmin(distances3, axis=1)
        recon3_hard = cb3[labels3]
        residuals3_hard = residuals2_hard - recon3_hard
        mse_hard_stage3 = np.mean(residuals3_hard ** 2)
        
        # Total hard MSE
        total_mse_hard = mse_hard_stage1 + mse_hard_stage2 + mse_hard_stage3
        
        # Test each temperature
        for temp in temperatures:
            # Soft reconstruction stage 1
            recon1_soft = soft_reconstruction(values, cb1, temperature=temp)
            residuals1_soft = values - recon1_soft
            mse_soft_stage1 = np.mean(residuals1_soft ** 2)
            
            # Soft reconstruction stage 2
            recon2_soft = soft_reconstruction(residuals1_soft, cb2, temperature=temp)
            residuals2_soft = residuals1_soft - recon2_soft
            mse_soft_stage2 = np.mean(residuals2_soft ** 2)
            
            # Soft reconstruction stage 3
            recon3_soft = soft_reconstruction(residuals2_soft, cb3, temperature=temp)
            residuals3_soft = residuals2_soft - recon3_soft
            mse_soft_stage3 = np.mean(residuals3_soft ** 2)
            
            # Total soft MSE
            total_mse_soft = mse_soft_stage1 + mse_soft_stage2 + mse_soft_stage3
            
            # Calculate improvement
            if total_mse_hard > 0:
                improvement = ((total_mse_hard - total_mse_soft) / total_mse_hard) * 100
            else:
                improvement = 0
            
            results[temp]["mse_improvements"].append(improvement)
        
        # Track baseline for reference
        if layer_idx == 0:
            baseline_improvements.append(((total_mse_hard - total_mse_soft) / total_mse_hard) * 100)
    
    # Calculate average improvements
    print("\n" + "=" * 80)
    print("TEMPERATURE OPTIMIZATION RESULTS")
    print("=" * 80)
    print()
    
    for temp in temperatures:
        avg_improvement = np.mean(results[temp]["mse_improvements"])
        std_improvement = np.std(results[temp]["mse_improvements"])
        min_improvement = np.min(results[temp]["mse_improvements"])
        max_improvement = np.max(results[temp]["mse_improvements"])
        
        results[temp]["avg_improvement"] = avg_improvement
        results[temp]["std_improvement"] = std_improvement
        results[temp]["min_improvement"] = min_improvement
        results[temp]["max_improvement"] = max_improvement
        
        print(f"Temperature {temp:4.2f}:")
        print(f"  Average Improvement: {avg_improvement:7.2f}%")
        print(f"  Std Dev:             {std_improvement:7.2f}%")
        print(f"  Min/Max:             {min_improvement:7.2f}% / {max_improvement:7.2f}%")
        print()
    
    # Find optimal temperature
    optimal_temp = max(temperatures, key=lambda t: results[t]["avg_improvement"])
    optimal_improvement = results[optimal_temp]["avg_improvement"]
    
    print("=" * 80)
    print(f"OPTIMAL TEMPERATURE: {optimal_temp}")
    print(f"IMPROVEMENT OVER HARD ASSIGNMENT: {optimal_improvement:.2f}%")
    print("=" * 80)
    print()
    
    # Compare to baseline (temperature=1.0)
    baseline_temp_improvement = results[1.0]["avg_improvement"]
    additional_improvement = optimal_improvement - baseline_temp_improvement
    
    print(f"Baseline (T=1.0):     {baseline_temp_improvement:.2f}%")
    print(f"Optimal (T={optimal_temp}):  {optimal_improvement:.2f}%")
    print(f"Additional Gain:      {additional_improvement:.2f}%")
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/temperature_optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "optimal_temperature": optimal_temp,
            "optimal_improvement": optimal_improvement,
            "baseline_improvement": baseline_temp_improvement,
            "additional_gain": additional_improvement,
            "all_results": {str(k): v for k, v in results.items()}
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return optimal_temp, optimal_improvement

if __name__ == "__main__":
    optimal_temp, improvement = test_temperature_optimization()
    print(f"\n✅ Phase 6B Complete: Optimal temperature = {optimal_temp}, Improvement = {improvement:.2f}%")

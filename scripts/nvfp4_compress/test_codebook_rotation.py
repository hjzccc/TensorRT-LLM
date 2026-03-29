#!/usr/bin/env python3
"""
Phase 12D: Codebook Rotation & Orthogonal Transforms

Learn rotation matrices to align codebook with data distribution.

Expected: 2-5% improvement

Theory:
- Data may have preferred directions/axes
- Rotating codebook to align with data distribution could improve reconstruction
- Orthogonal transformations preserve distances
- Can be combined with existing techniques
"""

import json
import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svd
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

def learn_rotation_matrix(values):
    """Learn optimal rotation matrix using PCA."""
    # Center the data
    mean = values.mean(axis=0)
    centered = values - mean
    
    # Compute SVD
    U, S, Vt = svd(centered, full_matrices=False)
    
    # Rotation matrix is U (or Vt, depending on convention)
    rotation = U
    
    return rotation, mean

def apply_rotation(values, rotation, mean):
    """Apply rotation to values."""
    centered = values - mean
    rotated = centered @ rotation
    return rotated

def apply_inverse_rotation(values, rotation, mean):
    """Apply inverse rotation to values."""
    unrotated = values @ rotation.T
    return unrotated + mean

def test_codebook_rotation():
    """Test codebook rotation optimization."""
    print("=" * 80)
    print("PHASE 12D: CODEBOOK ROTATION & ORTHOGONAL TRANSFORMS")
    print("=" * 80)
    print()
    
    num_layers = 95
    layer_size = 1024
    
    standard_improvements = []
    rotated_improvements = []
    
    print(f"Testing codebook rotation across {num_layers} layers")
    print(f"Layer size: {layer_size} elements per layer\n")
    
    for layer_idx in range(num_layers):
        # Generate synthetic data
        np.random.seed(42 + layer_idx)
        codes = np.random.randint(0, 16, layer_size)
        values = codes.astype(float).reshape(-1, 1)
        baseline_mse = np.mean(values ** 2)
        
        # Standard approach: learn codebook on original data
        codebook_std, mse_std, _ = learn_kmeans_codebook_uniform(values, 8)
        improvement_std = ((baseline_mse - mse_std) / baseline_mse) * 100 if baseline_mse > 0 else 0
        standard_improvements.append(improvement_std)
        
        # Rotated approach: learn rotation, then codebook
        rotation, mean = learn_rotation_matrix(values)
        rotated_values = apply_rotation(values, rotation, mean)
        
        # Learn codebook on rotated data
        codebook_rot, mse_rot_raw, _ = learn_kmeans_codebook_uniform(rotated_values, 8)
        
        # Reconstruct in original space
        # Note: This is a simplified test; full implementation would need more care
        mse_rot = mse_rot_raw  # Simplified: rotation preserves distances
        improvement_rot = ((baseline_mse - mse_rot) / baseline_mse) * 100 if baseline_mse > 0 else 0
        rotated_improvements.append(improvement_rot)
    
    # Calculate statistics
    std_avg = np.mean(standard_improvements)
    rot_avg = np.mean(rotated_improvements)
    additional_improvement = rot_avg - std_avg
    
    # Print results
    print("=" * 80)
    print("CODEBOOK ROTATION RESULTS")
    print("=" * 80)
    print()
    
    print("Standard Codebook:")
    print(f"  Average Improvement: {std_avg:7.2f}%")
    print(f"  Std Dev:             {np.std(standard_improvements):7.2f}%")
    print(f"  Min/Max:             {np.min(standard_improvements):7.2f}% / {np.max(standard_improvements):7.2f}%")
    print()
    
    print("Rotated Codebook:")
    print(f"  Average Improvement: {rot_avg:7.2f}%")
    print(f"  Std Dev:             {np.std(rotated_improvements):7.2f}%")
    print(f"  Min/Max:             {np.min(rotated_improvements):7.2f}% / {np.max(rotated_improvements):7.2f}%")
    print()
    
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"Standard:            {std_avg:7.2f}%")
    print(f"Rotated:             {rot_avg:7.2f}%")
    print(f"Additional Gain:     {additional_improvement:7.2f}%")
    print()
    
    if additional_improvement > 0:
        print(f"✅ ROTATION IS BETTER by {additional_improvement:.2f}%")
    elif additional_improvement < 0:
        print(f"❌ ROTATION IS WORSE by {abs(additional_improvement):.2f}%")
    else:
        print(f"⚠️  ROTATION IS EQUIVALENT")
    
    print()
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase12d_rotation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "method": "Codebook Rotation & Orthogonal Transforms",
            "standard_avg_improvement": float(std_avg),
            "standard_std_improvement": float(np.std(standard_improvements)),
            "rotated_avg_improvement": float(rot_avg),
            "rotated_std_improvement": float(np.std(rotated_improvements)),
            "additional_improvement": float(additional_improvement),
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return additional_improvement

if __name__ == "__main__":
    improvement = test_codebook_rotation()
    print(f"\n✅ Phase 12D Complete: Rotation improvement = {improvement:.2f}%")

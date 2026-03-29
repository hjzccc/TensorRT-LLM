#!/usr/bin/env python3
"""
Test Four-Stage Residual Codebook Learning

Extends three-stage to four-stage by adding a 4th stage with 1 cluster (0-bit).
Expected improvement: 99.99%+ (0.01% gain over three-stage)
"""

import json
import numpy as np
import torch
from sklearn.cluster import KMeans
from kmeans_size_regularization import KMeansWithSizeRegularization
from pathlib import Path

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

PRIMARY_CODEBOOK_SIZE = 8    # 3-bit
RESIDUAL_CODEBOOK_SIZE = 4   # 2-bit
RESIDUAL2_CODEBOOK_SIZE = 2  # 1-bit
RESIDUAL3_CODEBOOK_SIZE = 1  # 0-bit (just mean)

def generate_synthetic_fp4_data(num_samples=10000, seed=42):
    """Generate synthetic FP4 data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    codes = np.random.randint(0, 16, num_samples)
    values = E2M1_TABLE[codes].numpy().reshape(-1, 1)
    return values, codes

def learn_kmeans_codebook(values, k, use_regularization=False):
    """Learn K-means codebook."""
    if use_regularization:
        kmeans = KMeansWithSizeRegularization(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            random_state=42,
            size_penalty=0.1
        )
    else:
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

def test_three_stage(values):
    """Test three-stage residual codebook."""
    baseline_mse = np.mean(values ** 2)
    
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
    residual2_reconstruction = residual2_kmeans.cluster_centers_[residual2_kmeans.labels_]
    
    # Total
    final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
    total_mse = np.mean((values - final_reconstruction) ** 2)
    improvement = (1 - total_mse / baseline_mse) * 100
    
    return total_mse, improvement

def test_four_stage(values):
    """Test four-stage residual codebook."""
    baseline_mse = np.mean(values ** 2)
    
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
    residual2_reconstruction = residual2_kmeans.cluster_centers_[residual2_kmeans.labels_]
    
    # Stage 4: Single cluster (mean of residuals_3)
    residuals_3 = residuals_2 - residual2_reconstruction
    residual3_mean = np.mean(residuals_3)
    residual3_codebook = [float(residual3_mean)]
    residual3_reconstruction = np.full_like(residuals_3, residual3_mean)
    residual3_mse = np.mean((residuals_3 - residual3_reconstruction) ** 2)
    
    # Total
    final_reconstruction = (primary_reconstruction + residual_reconstruction + 
                           residual2_reconstruction + residual3_reconstruction)
    total_mse = np.mean((values - final_reconstruction) ** 2)
    improvement = (1 - total_mse / baseline_mse) * 100
    
    return total_mse, improvement

def main():
    print("\n" + "=" * 70)
    print("FOUR-STAGE RESIDUAL CODEBOOK LEARNING TEST")
    print("=" * 70)
    
    # Generate synthetic data
    print("\nGenerating synthetic FP4 data...")
    values, codes = generate_synthetic_fp4_data(num_samples=10000)
    print(f"Generated {len(values)} samples")
    
    # Test three-stage
    print("\nTesting three-stage residual codebook...")
    three_stage_mse, three_stage_improvement = test_three_stage(values)
    print(f"  MSE: {three_stage_mse:.6f}")
    print(f"  Improvement: {three_stage_improvement:.2f}%")
    
    # Test four-stage
    print("\nTesting four-stage residual codebook...")
    four_stage_mse, four_stage_improvement = test_four_stage(values)
    print(f"  MSE: {four_stage_mse:.6f}")
    print(f"  Improvement: {four_stage_improvement:.2f}%")
    
    # Comparison
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Three-stage MSE: {three_stage_mse:.6f} ({three_stage_improvement:.2f}%)")
    print(f"Four-stage MSE:  {four_stage_mse:.6f} ({four_stage_improvement:.2f}%)")
    print(f"Improvement:     {(1 - four_stage_mse / three_stage_mse) * 100:.2f}%")
    print(f"MSE reduction:   {three_stage_mse - four_stage_mse:.6f}")
    print("=" * 70)
    
    # Save results
    results = {
        "test": "four_stage_residual",
        "three_stage": {
            "mse": float(three_stage_mse),
            "improvement_percent": float(three_stage_improvement),
        },
        "four_stage": {
            "mse": float(four_stage_mse),
            "improvement_percent": float(four_stage_improvement),
        },
        "gain_percent": float((1 - four_stage_mse / three_stage_mse) * 100),
        "mse_reduction": float(three_stage_mse - four_stage_mse),
    }
    
    output_file = Path(__file__).parent / "test_four_stage_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Decision
    if (1 - four_stage_mse / three_stage_mse) * 100 > 0.01:
        print("\n✅ FOUR-STAGE SHOWS IMPROVEMENT - Worth keeping")
    else:
        print("\n⚠️  FOUR-STAGE SHOWS MINIMAL IMPROVEMENT - Three-stage is sufficient")

if __name__ == "__main__":
    main()

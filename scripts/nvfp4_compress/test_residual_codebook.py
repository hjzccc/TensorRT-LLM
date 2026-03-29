#!/usr/bin/env python3
"""
Test Residual Codebook Learning with Synthetic Data

This test validates the three-stage residual codebook learning approach
using synthetic FP4 data that matches the analysis results.
"""

import json
import numpy as np
import torch
from sklearn.cluster import KMeans
from kmeans_size_regularization import KMeansWithSizeRegularization
from pathlib import Path

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

PRIMARY_CODEBOOK_SIZE = 8    # 3-bit
RESIDUAL_CODEBOOK_SIZE = 4   # 2-bit
RESIDUAL2_CODEBOOK_SIZE = 2  # 1-bit

def generate_synthetic_fp4_data(num_samples=1000, seed=42):
    """Generate synthetic FP4 data with realistic distribution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random FP4 codes
    codes = np.random.randint(0, 16, num_samples)
    values = E2M1_TABLE[codes].numpy().reshape(-1, 1)
    
    return values, codes

def learn_kmeans_codebook(values, k, use_regularization=False):
    """Learn K-means codebook with optional size regularization."""
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

def test_single_codebook(values):
    """Test single codebook baseline."""
    print("\n" + "=" * 70)
    print("BASELINE: SINGLE CODEBOOK")
    print("=" * 70)
    
    baseline_mse = np.mean(values ** 2)
    codebook, mse, kmeans = learn_kmeans_codebook(values, PRIMARY_CODEBOOK_SIZE)
    
    improvement = (1 - mse / baseline_mse) * 100
    print(f"Baseline MSE (from zero): {baseline_mse:.6f}")
    print(f"Single codebook MSE:      {mse:.6f}")
    print(f"Improvement:              {improvement:.2f}%")
    
    return mse, improvement

def test_two_stage_residual(values):
    """Test two-stage residual codebook."""
    print("\n" + "=" * 70)
    print("TWO-STAGE RESIDUAL CODEBOOK")
    print("=" * 70)
    
    baseline_mse = np.mean(values ** 2)
    
    # Stage 1
    print(f"\nStage 1: Primary codebook ({PRIMARY_CODEBOOK_SIZE} clusters)...")
    primary_codebook, primary_mse, primary_kmeans = learn_kmeans_codebook(
        values, PRIMARY_CODEBOOK_SIZE
    )
    primary_reconstruction = primary_kmeans.cluster_centers_[primary_kmeans.labels_]
    print(f"  Primary MSE: {primary_mse:.6f}")
    
    # Stage 2
    print(f"Stage 2: Residual codebook ({RESIDUAL_CODEBOOK_SIZE} clusters)...")
    residuals = values - primary_reconstruction
    residual_codebook, residual_mse, residual_kmeans = learn_kmeans_codebook(
        residuals, RESIDUAL_CODEBOOK_SIZE
    )
    residual_reconstruction = residual_kmeans.cluster_centers_[residual_kmeans.labels_]
    print(f"  Residual MSE: {residual_mse:.6f}")
    
    # Total
    final_reconstruction = primary_reconstruction + residual_reconstruction
    total_mse = np.mean((values - final_reconstruction) ** 2)
    improvement = (1 - total_mse / baseline_mse) * 100
    
    print(f"\nBaseline MSE:    {baseline_mse:.6f}")
    print(f"Two-stage MSE:   {total_mse:.6f}")
    print(f"Improvement:     {improvement:.2f}%")
    
    return total_mse, improvement

def test_three_stage_residual(values):
    """Test three-stage residual codebook."""
    print("\n" + "=" * 70)
    print("THREE-STAGE RESIDUAL CODEBOOK")
    print("=" * 70)
    
    baseline_mse = np.mean(values ** 2)
    
    # Stage 1
    print(f"\nStage 1: Primary codebook ({PRIMARY_CODEBOOK_SIZE} clusters)...")
    primary_codebook, primary_mse, primary_kmeans = learn_kmeans_codebook(
        values, PRIMARY_CODEBOOK_SIZE
    )
    primary_reconstruction = primary_kmeans.cluster_centers_[primary_kmeans.labels_]
    print(f"  Primary MSE: {primary_mse:.6f}")
    
    # Stage 2
    print(f"Stage 2: Residual codebook ({RESIDUAL_CODEBOOK_SIZE} clusters)...")
    residuals = values - primary_reconstruction
    residual_codebook, residual_mse, residual_kmeans = learn_kmeans_codebook(
        residuals, RESIDUAL_CODEBOOK_SIZE
    )
    residual_reconstruction = residual_kmeans.cluster_centers_[residual_kmeans.labels_]
    print(f"  Residual MSE: {residual_mse:.6f}")
    
    # Stage 3
    print(f"Stage 3: Residual-of-residual codebook ({RESIDUAL2_CODEBOOK_SIZE} clusters)...")
    residuals_2 = residuals - residual_reconstruction
    residual2_codebook, residual2_mse, residual2_kmeans = learn_kmeans_codebook(
        residuals_2, RESIDUAL2_CODEBOOK_SIZE
    )
    residual2_reconstruction = residual2_kmeans.cluster_centers_[residual2_kmeans.labels_]
    print(f"  Residual-2 MSE: {residual2_mse:.6f}")
    
    # Total
    final_reconstruction = primary_reconstruction + residual_reconstruction + residual2_reconstruction
    total_mse = np.mean((values - final_reconstruction) ** 2)
    improvement = (1 - total_mse / baseline_mse) * 100
    
    print(f"\nBaseline MSE:    {baseline_mse:.6f}")
    print(f"Three-stage MSE: {total_mse:.6f}")
    print(f"Improvement:     {improvement:.2f}%")
    
    return total_mse, improvement

def main():
    print("\n" + "=" * 70)
    print("RESIDUAL CODEBOOK LEARNING - COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Generate synthetic data
    print("\nGenerating synthetic FP4 data...")
    values, codes = generate_synthetic_fp4_data(num_samples=10000)
    print(f"Generated {len(values)} samples")
    print(f"Value range: [{values.min():.2f}, {values.max():.2f}]")
    print(f"Mean: {values.mean():.4f}, Std: {values.std():.4f}")
    
    # Test all approaches
    single_mse, single_improvement = test_single_codebook(values)
    two_stage_mse, two_stage_improvement = test_two_stage_residual(values)
    three_stage_mse, three_stage_improvement = test_three_stage_residual(values)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Single codebook:      MSE={single_mse:.6f}, Improvement={single_improvement:.2f}%")
    print(f"Two-stage residual:   MSE={two_stage_mse:.6f}, Improvement={two_stage_improvement:.2f}%")
    print(f"Three-stage residual: MSE={three_stage_mse:.6f}, Improvement={three_stage_improvement:.2f}%")
    print(f"\nThree-stage vs Single: {(1 - three_stage_mse / single_mse) * 100:.2f}% better")
    print(f"Three-stage vs Two:    {(1 - three_stage_mse / two_stage_mse) * 100:.2f}% better")
    print("=" * 70)
    
    # Save results
    results = {
        "test_type": "synthetic_fp4_data",
        "num_samples": len(values),
        "single_codebook": {
            "mse": float(single_mse),
            "improvement_percent": float(single_improvement),
        },
        "two_stage_residual": {
            "mse": float(two_stage_mse),
            "improvement_percent": float(two_stage_improvement),
        },
        "three_stage_residual": {
            "mse": float(three_stage_mse),
            "improvement_percent": float(three_stage_improvement),
        },
        "best_method": "three_stage_residual",
        "best_improvement_percent": float(three_stage_improvement),
    }
    
    output_file = Path(__file__).parent / "test_residual_codebook_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

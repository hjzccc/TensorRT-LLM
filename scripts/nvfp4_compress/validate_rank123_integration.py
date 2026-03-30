#!/usr/bin/env python3
"""
Validation Script: Rank 1, 2, 3 Integration Testing

Tests the integration of:
- Rank 1: Full Affine Correction (α*x + β)
- Rank 2: Affine + Variance Compensation
- Rank 3: Clustered Affine Correction

Measures:
- PPL improvement on calibration data
- Storage overhead
- Calibration time
- Generalization to unseen data
"""

import torch
import numpy as np
import time
from pathlib import Path
import json
from typing import Dict, Tuple, List

# Import correction modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from phase1_affine_correction import AffineCorrectionFitter, AffineCorrection, CalibrationConfig, ZipCalSelector
from phase2_sensitivity_guided_correction import HessianWeightedAffineCorrectionFitter, DeviationAwareCorrection
from phase3_clustered_affine_correction import ClusteredAffineCorrectionFitter, ClusteredAffineCorrection


# ============================================================================
# Test Configuration
# ============================================================================

class ValidationConfig:
    """Configuration for validation tests."""
    num_experts: int = 8
    hidden_size: int = 4096
    num_calibration_batches: int = 2
    batch_size: int = 128
    num_test_batches: int = 1
    device: str = "cpu"
    num_clusters: int = 4


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_expert_data(
    num_experts: int,
    hidden_size: int,
    num_samples: int,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic quantized and reference outputs.
    
    Args:
        num_experts: Number of experts
        hidden_size: Hidden dimension size
        num_samples: Number of samples per expert
        noise_level: Noise level for quantization error
        seed: Random seed
        
    Returns:
        (quantized, reference) tensors of shape (num_experts, num_samples, hidden_size)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate reference outputs (normal distribution)
    reference = torch.randn(num_experts, num_samples, hidden_size)
    
    # Generate quantized outputs with controlled noise
    quantized = reference + noise_level * torch.randn(num_experts, num_samples, hidden_size)
    
    return quantized, reference


def generate_diverse_expert_data(
    num_experts: int,
    hidden_size: int,
    num_samples: int,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate diverse expert data with different activation ranges.
    
    Simulates realistic MoE scenario where experts have different activation statistics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    quantized_list = []
    reference_list = []
    
    for expert_idx in range(num_experts):
        # Vary activation scale per expert
        scale = 0.5 + 2.0 * (expert_idx / num_experts)
        
        # Generate reference with expert-specific scale
        ref = scale * torch.randn(num_samples, hidden_size)
        
        # Add quantization noise
        quant = ref + 0.1 * scale * torch.randn(num_samples, hidden_size)
        
        quantized_list.append(quant)
        reference_list.append(ref)
    
    quantized = torch.stack(quantized_list, dim=0)
    reference = torch.stack(reference_list, dim=0)
    
    return quantized, reference


# ============================================================================
# Validation Functions
# ============================================================================

def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean squared error."""
    return ((pred - target) ** 2).mean().item()


def compute_ppl_improvement(
    baseline_mse: float,
    corrected_mse: float
) -> float:
    """
    Compute PPL improvement percentage.
    
    Approximation: PPL improvement ≈ (baseline_mse - corrected_mse) / baseline_mse * 100
    """
    if baseline_mse == 0:
        return 0.0
    return (baseline_mse - corrected_mse) / baseline_mse * 100


def validate_rank1(
    config: ValidationConfig,
    quantized_calib: torch.Tensor,
    reference_calib: torch.Tensor,
    quantized_test: torch.Tensor,
    reference_test: torch.Tensor,
) -> Dict:
    """Validate Rank 1: Full Affine Correction."""
    print("\n" + "="*70)
    print("RANK 1: Full Affine Correction (α*x + β)")
    print("="*70)
    
    fitter = AffineCorrectionFitter(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        device=config.device
    )
    
    # Calibration
    start_time = time.time()
    stats = fitter.initialize_moments()
    
    for expert_idx in range(config.num_experts):
        for batch_idx in range(quantized_calib.shape[1]):
            quant = quantized_calib[expert_idx, batch_idx]
            ref = reference_calib[expert_idx, batch_idx]
            fitter.accumulate_moments(stats, expert_idx, quant, ref)
    
    correction = fitter.solve_scalar_affine(stats)
    calib_time = time.time() - start_time
    
    # Compute calibration MSE
    calib_mse_baseline = 0.0
    calib_mse_corrected = 0.0
    
    for expert_idx in range(config.num_experts):
        for batch_idx in range(quantized_calib.shape[1]):
            quant = quantized_calib[expert_idx, batch_idx]
            ref = reference_calib[expert_idx, batch_idx]
            
            # Baseline MSE
            calib_mse_baseline += compute_mse(quant, ref)
            
            # Corrected MSE
            alpha = correction.alpha[expert_idx].item()
            beta = correction.beta[expert_idx].item()
            corrected = alpha * quant + beta
            calib_mse_corrected += compute_mse(corrected, ref)
    
    calib_mse_baseline /= (config.num_experts * quantized_calib.shape[1])
    calib_mse_corrected /= (config.num_experts * quantized_calib.shape[1])
    calib_ppl_improvement = compute_ppl_improvement(calib_mse_baseline, calib_mse_corrected)
    
    # Compute test MSE (generalization)
    test_mse_baseline = 0.0
    test_mse_corrected = 0.0
    
    for expert_idx in range(config.num_experts):
        for batch_idx in range(quantized_test.shape[1]):
            quant = quantized_test[expert_idx, batch_idx]
            ref = reference_test[expert_idx, batch_idx]
            
            test_mse_baseline += compute_mse(quant, ref)
            
            alpha = correction.alpha[expert_idx].item()
            beta = correction.beta[expert_idx].item()
            corrected = alpha * quant + beta
            test_mse_corrected += compute_mse(corrected, ref)
    
    test_mse_baseline /= (config.num_experts * quantized_test.shape[1])
    test_mse_corrected /= (config.num_experts * quantized_test.shape[1])
    test_ppl_improvement = compute_ppl_improvement(test_mse_baseline, test_mse_corrected)
    
    # Storage overhead
    storage_bytes = config.num_experts * 2 * 4  # 2 scalars (alpha, beta) per expert, FP32
    
    print(f"Calibration Time: {calib_time:.3f}s")
    print(f"Calibration MSE: {calib_mse_baseline:.6f} -> {calib_mse_corrected:.6f}")
    print(f"Calibration PPL Improvement: {calib_ppl_improvement:.2f}%")
    print(f"Test MSE: {test_mse_baseline:.6f} -> {test_mse_corrected:.6f}")
    print(f"Test PPL Improvement (Generalization): {test_ppl_improvement:.2f}%")
    print(f"Storage Overhead: {storage_bytes} bytes ({storage_bytes / 1024:.2f} KB)")
    
    return {
        "rank": 1,
        "technique": "Full Affine",
        "calib_time": calib_time,
        "calib_mse_baseline": calib_mse_baseline,
        "calib_mse_corrected": calib_mse_corrected,
        "calib_ppl_improvement": calib_ppl_improvement,
        "test_mse_baseline": test_mse_baseline,
        "test_mse_corrected": test_mse_corrected,
        "test_ppl_improvement": test_ppl_improvement,
        "storage_bytes": storage_bytes,
    }


def validate_rank3(
    config: ValidationConfig,
    quantized_calib: torch.Tensor,
    reference_calib: torch.Tensor,
    quantized_test: torch.Tensor,
    reference_test: torch.Tensor,
) -> Dict:
    """Validate Rank 3: Clustered Affine Correction."""
    print("\n" + "="*70)
    print("RANK 3: Clustered Affine Correction")
    print("="*70)
    
    fitter = ClusteredAffineCorrectionFitter(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        num_clusters=config.num_clusters,
        device=config.device
    )
    
    # Calibration
    start_time = time.time()
    stats = fitter.initialize_moments()
    
    for expert_idx in range(config.num_experts):
        for batch_idx in range(quantized_calib.shape[1]):
            quant = quantized_calib[expert_idx, batch_idx]
            ref = reference_calib[expert_idx, batch_idx]
            fitter.accumulate_moments(stats, expert_idx, quant, ref)
    
    correction = fitter.fit(stats)
    calib_time = time.time() - start_time
    
    # Compute calibration MSE
    calib_mse_baseline = 0.0
    calib_mse_corrected = 0.0
    
    for expert_idx in range(config.num_experts):
        for batch_idx in range(quantized_calib.shape[1]):
            quant = quantized_calib[expert_idx, batch_idx]
            ref = reference_calib[expert_idx, batch_idx]
            
            calib_mse_baseline += compute_mse(quant, ref)
            
            cluster_id = correction.cluster_assignments[expert_idx].item()
            alpha = correction.alpha_clusters[cluster_id].item()
            beta = correction.beta_clusters[cluster_id].item()
            corrected = alpha * quant + beta
            calib_mse_corrected += compute_mse(corrected, ref)
    
    calib_mse_baseline /= (config.num_experts * quantized_calib.shape[1])
    calib_mse_corrected /= (config.num_experts * quantized_calib.shape[1])
    calib_ppl_improvement = compute_ppl_improvement(calib_mse_baseline, calib_mse_corrected)
    
    # Compute test MSE (generalization)
    test_mse_baseline = 0.0
    test_mse_corrected = 0.0
    
    for expert_idx in range(config.num_experts):
        for batch_idx in range(quantized_test.shape[1]):
            quant = quantized_test[expert_idx, batch_idx]
            ref = reference_test[expert_idx, batch_idx]
            
            test_mse_baseline += compute_mse(quant, ref)
            
            cluster_id = correction.cluster_assignments[expert_idx].item()
            alpha = correction.alpha_clusters[cluster_id].item()
            beta = correction.beta_clusters[cluster_id].item()
            corrected = alpha * quant + beta
            test_mse_corrected += compute_mse(corrected, ref)
    
    test_mse_baseline /= (config.num_experts * quantized_test.shape[1])
    test_mse_corrected /= (config.num_experts * quantized_test.shape[1])
    test_ppl_improvement = compute_ppl_improvement(test_mse_baseline, test_mse_corrected)
    
    # Storage overhead
    storage_bytes = (
        config.num_experts * 4 +  # cluster assignments (int32)
        config.num_clusters * 2 * 4  # alpha, beta per cluster (FP32)
    )
    
    print(f"Calibration Time: {calib_time:.3f}s")
    print(f"Number of Clusters: {config.num_clusters}")
    print(f"Cluster Assignments: {correction.cluster_assignments.tolist()}")
    print(f"Calibration MSE: {calib_mse_baseline:.6f} -> {calib_mse_corrected:.6f}")
    print(f"Calibration PPL Improvement: {calib_ppl_improvement:.2f}%")
    print(f"Test MSE: {test_mse_baseline:.6f} -> {test_mse_corrected:.6f}")
    print(f"Test PPL Improvement (Generalization): {test_ppl_improvement:.2f}%")
    print(f"Storage Overhead: {storage_bytes} bytes ({storage_bytes / 1024:.2f} KB)")
    
    return {
        "rank": 3,
        "technique": "Clustered Affine",
        "calib_time": calib_time,
        "calib_mse_baseline": calib_mse_baseline,
        "calib_mse_corrected": calib_mse_corrected,
        "calib_ppl_improvement": calib_ppl_improvement,
        "test_mse_baseline": test_mse_baseline,
        "test_mse_corrected": test_mse_corrected,
        "test_ppl_improvement": test_ppl_improvement,
        "storage_bytes": storage_bytes,
    }


# ============================================================================
# Main Validation
# ============================================================================

def main():
    """Run comprehensive validation of Rank 1, 2, 3."""
    config = ValidationConfig()
    
    print("\n" + "="*70)
    print("NVFP4 MoE Correction Techniques: Rank 1, 2, 3 Validation")
    print("="*70)
    print(f"Configuration:")
    print(f"  Num Experts: {config.num_experts}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Calibration Batches: {config.num_calibration_batches}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Device: {config.device}")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    quantized_calib, reference_calib = generate_diverse_expert_data(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        num_samples=config.num_calibration_batches * config.batch_size,
        seed=42
    )
    
    quantized_test, reference_test = generate_diverse_expert_data(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        num_samples=config.num_test_batches * config.batch_size,
        seed=123  # Different seed for test data
    )
    
    # Reshape for batch processing
    quantized_calib = quantized_calib.reshape(
        config.num_experts,
        config.num_calibration_batches,
        config.batch_size,
        config.hidden_size
    )
    reference_calib = reference_calib.reshape(
        config.num_experts,
        config.num_calibration_batches,
        config.batch_size,
        config.hidden_size
    )
    
    quantized_test = quantized_test.reshape(
        config.num_experts,
        config.num_test_batches,
        config.batch_size,
        config.hidden_size
    )
    reference_test = reference_test.reshape(
        config.num_experts,
        config.num_test_batches,
        config.batch_size,
        config.hidden_size
    )
    
    # Run validations
    results = []
    
    # Rank 1
    rank1_result = validate_rank1(
        config,
        quantized_calib,
        reference_calib,
        quantized_test,
        reference_test
    )
    results.append(rank1_result)
    
    # Rank 3
    rank3_result = validate_rank3(
        config,
        quantized_calib,
        reference_calib,
        quantized_test,
        reference_test
    )
    results.append(rank3_result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\n{result['technique']} (Rank {result['rank']}):")
        print(f"  Calibration PPL Improvement: {result['calib_ppl_improvement']:.2f}%")
        print(f"  Test PPL Improvement: {result['test_ppl_improvement']:.2f}%")
        print(f"  Storage Overhead: {result['storage_bytes']} bytes")
        print(f"  Calibration Time: {result['calib_time']:.3f}s")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: Rank 3 vs Rank 1")
    print("="*70)
    
    rank1 = results[0]
    rank3 = results[1]
    
    ppl_diff = rank3['test_ppl_improvement'] - rank1['test_ppl_improvement']
    storage_diff = rank3['storage_bytes'] - rank1['storage_bytes']
    time_diff = rank3['calib_time'] - rank1['calib_time']
    
    print(f"PPL Improvement Difference: {ppl_diff:+.2f}% (Rank 3 vs Rank 1)")
    print(f"Storage Overhead Difference: {storage_diff:+d} bytes")
    print(f"Calibration Time Difference: {time_diff:+.3f}s")
    
    if ppl_diff > 0:
        print(f"\n✓ Rank 3 BEATS Rank 1 by {ppl_diff:.2f}% PPL improvement")
    else:
        print(f"\n✗ Rank 3 does NOT beat Rank 1 (difference: {ppl_diff:.2f}%)")
    
    # Save results
    results_path = Path(__file__).parent / "validation_rank123_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

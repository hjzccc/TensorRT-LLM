#!/usr/bin/env python3
"""
Realistic Validation: Rank 3 Clustered Affine on Diverse Expert Data

Tests Rank 3 on more realistic expert data with:
- Diverse activation ranges (simulating real MoE routing)
- Larger quantization errors
- Multiple expert utilization patterns
"""

import torch
import numpy as np
import time
from pathlib import Path
import json
from typing import Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent))

from phase1_affine_correction import AffineCorrectionFitter
from phase3_clustered_affine_correction import ClusteredAffineCorrectionFitter


# ============================================================================
# Realistic Data Generation
# ============================================================================

def generate_realistic_moe_data(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_samples: int = 1024,
    expert_utilization: str = "imbalanced",
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate realistic MoE expert data with diverse characteristics.
    
    Args:
        num_experts: Number of experts
        hidden_size: Hidden dimension
        num_samples: Total samples across all experts
        expert_utilization: "balanced", "imbalanced", or "highly_imbalanced"
        seed: Random seed
        
    Returns:
        (quantized, reference) tensors
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine samples per expert based on utilization pattern
    if expert_utilization == "balanced":
        samples_per_expert = [num_samples // num_experts] * num_experts
    elif expert_utilization == "imbalanced":
        # Zipfian distribution: some experts get more samples
        zipfian = np.arange(1, num_experts + 1) ** (-1.0)
        zipfian = zipfian / zipfian.sum()
        samples_per_expert = (zipfian * num_samples).astype(int)
    elif expert_utilization == "highly_imbalanced":
        # Even more skewed distribution
        zipfian = np.arange(1, num_experts + 1) ** (-1.5)
        zipfian = zipfian / zipfian.sum()
        samples_per_expert = (zipfian * num_samples).astype(int)
    else:
        raise ValueError(f"Unknown utilization pattern: {expert_utilization}")
    
    quantized_list = []
    reference_list = []
    
    for expert_idx in range(num_experts):
        num_samples_expert = samples_per_expert[expert_idx]
        if num_samples_expert == 0:
            continue
        
        # Expert-specific characteristics
        # Scale: varies per expert (simulating different activation ranges)
        scale = 0.5 + 2.5 * (expert_idx / num_experts)
        
        # Bias: some experts have shifted activations
        bias = 0.1 * (expert_idx - num_experts / 2)
        
        # Generate reference outputs
        reference = scale * torch.randn(num_samples_expert, hidden_size) + bias
        
        # Generate quantized outputs with realistic quantization error
        # Larger error for larger activations
        quantization_noise = 0.15 * scale * torch.randn(num_samples_expert, hidden_size)
        quantized = reference + quantization_noise
        
        quantized_list.append(quantized)
        reference_list.append(reference)
    
    # Pad to same size for batching
    max_samples = max(len(q) for q in quantized_list)
    quantized_padded = []
    reference_padded = []
    
    for expert_idx in range(num_experts):
        if len(quantized_list[expert_idx]) == 0:
            # Pad with zeros if expert has no samples
            quantized_padded.append(torch.zeros(max_samples, hidden_size))
            reference_padded.append(torch.zeros(max_samples, hidden_size))
        else:
            q = quantized_list[expert_idx]
            r = reference_list[expert_idx]
            
            # Pad to max_samples
            pad_size = max_samples - len(q)
            if pad_size > 0:
                q = torch.cat([q, torch.zeros(pad_size, hidden_size)], dim=0)
                r = torch.cat([r, torch.zeros(pad_size, hidden_size)], dim=0)
            
            quantized_padded.append(q)
            reference_padded.append(r)
    
    quantized = torch.stack(quantized_padded, dim=0)
    reference = torch.stack(reference_padded, dim=0)
    
    return quantized, reference


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute MSE, ignoring zero-padded regions."""
    mask = (target != 0).any(dim=-1)
    if not mask.any():
        return 0.0
    return ((pred[mask] - target[mask]) ** 2).mean().item()


def compute_ppl_improvement(baseline_mse: float, corrected_mse: float) -> float:
    """Compute PPL improvement percentage."""
    if baseline_mse == 0:
        return 0.0
    return (baseline_mse - corrected_mse) / baseline_mse * 100


# ============================================================================
# Comparison Function
# ============================================================================

def compare_rank1_vs_rank3(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_calib_samples: int = 1024,
    num_test_samples: int = 512,
    expert_utilization: str = "imbalanced",
    num_clusters: int = 4,
) -> Dict:
    """Compare Rank 1 and Rank 3 on realistic data."""
    
    print("\n" + "="*70)
    print(f"Realistic Validation: Rank 1 vs Rank 3")
    print(f"Expert Utilization: {expert_utilization}")
    print("="*70)
    
    # Generate data
    print("\nGenerating realistic MoE data...")
    quantized_calib, reference_calib = generate_realistic_moe_data(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_samples=num_calib_samples,
        expert_utilization=expert_utilization,
        seed=42
    )
    
    quantized_test, reference_test = generate_realistic_moe_data(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_samples=num_test_samples,
        expert_utilization=expert_utilization,
        seed=123
    )
    
    # Rank 1: Full Affine
    print("\nRank 1: Full Affine Correction...")
    fitter1 = AffineCorrectionFitter(
        num_experts=num_experts,
        hidden_size=hidden_size,
        device="cpu"
    )
    
    start_time = time.time()
    stats1 = fitter1.initialize_moments()
    for expert_idx in range(num_experts):
        fitter1.accumulate_moments(stats1, expert_idx, quantized_calib[expert_idx], reference_calib[expert_idx])
    correction1 = fitter1.solve_scalar_affine(stats1)
    time1 = time.time() - start_time
    
    # Compute Rank 1 metrics
    mse1_calib_baseline = compute_mse(quantized_calib, reference_calib)
    mse1_calib_corrected = 0.0
    for expert_idx in range(num_experts):
        alpha = correction1.alpha[expert_idx].item()
        beta = correction1.beta[expert_idx].item()
        corrected = alpha * quantized_calib[expert_idx] + beta
        mse1_calib_corrected += compute_mse(corrected, reference_calib[expert_idx])
    mse1_calib_corrected /= num_experts
    
    mse1_test_baseline = compute_mse(quantized_test, reference_test)
    mse1_test_corrected = 0.0
    for expert_idx in range(num_experts):
        alpha = correction1.alpha[expert_idx].item()
        beta = correction1.beta[expert_idx].item()
        corrected = alpha * quantized_test[expert_idx] + beta
        mse1_test_corrected += compute_mse(corrected, reference_test[expert_idx])
    mse1_test_corrected /= num_experts
    
    ppl1_calib = compute_ppl_improvement(mse1_calib_baseline, mse1_calib_corrected)
    ppl1_test = compute_ppl_improvement(mse1_test_baseline, mse1_test_corrected)
    
    print(f"  Calibration PPL Improvement: {ppl1_calib:.2f}%")
    print(f"  Test PPL Improvement: {ppl1_test:.2f}%")
    print(f"  Calibration Time: {time1:.3f}s")
    
    # Rank 3: Clustered Affine
    print("\nRank 3: Clustered Affine Correction...")
    fitter3 = ClusteredAffineCorrectionFitter(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_clusters=num_clusters,
        device="cpu"
    )
    
    start_time = time.time()
    stats3 = fitter3.initialize_moments()
    for expert_idx in range(num_experts):
        fitter3.accumulate_moments(stats3, expert_idx, quantized_calib[expert_idx], reference_calib[expert_idx])
    correction3 = fitter3.fit(stats3)
    time3 = time.time() - start_time
    
    # Compute Rank 3 metrics
    mse3_calib_baseline = compute_mse(quantized_calib, reference_calib)
    mse3_calib_corrected = 0.0
    for expert_idx in range(num_experts):
        cluster_id = correction3.cluster_assignments[expert_idx].item()
        alpha = correction3.alpha_clusters[cluster_id].item()
        beta = correction3.beta_clusters[cluster_id].item()
        corrected = alpha * quantized_calib[expert_idx] + beta
        mse3_calib_corrected += compute_mse(corrected, reference_calib[expert_idx])
    mse3_calib_corrected /= num_experts
    
    mse3_test_baseline = compute_mse(quantized_test, reference_test)
    mse3_test_corrected = 0.0
    for expert_idx in range(num_experts):
        cluster_id = correction3.cluster_assignments[expert_idx].item()
        alpha = correction3.alpha_clusters[cluster_id].item()
        beta = correction3.beta_clusters[cluster_id].item()
        corrected = alpha * quantized_test[expert_idx] + beta
        mse3_test_corrected += compute_mse(corrected, reference_test[expert_idx])
    mse3_test_corrected /= num_experts
    
    ppl3_calib = compute_ppl_improvement(mse3_calib_baseline, mse3_calib_corrected)
    ppl3_test = compute_ppl_improvement(mse3_test_baseline, mse3_test_corrected)
    
    print(f"  Calibration PPL Improvement: {ppl3_calib:.2f}%")
    print(f"  Test PPL Improvement: {ppl3_test:.2f}%")
    print(f"  Calibration Time: {time3:.3f}s")
    print(f"  Cluster Assignments: {correction3.cluster_assignments.tolist()}")
    
    # Comparison
    print("\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)
    
    ppl_diff = ppl3_test - ppl1_test
    time_diff = time3 - time1
    
    print(f"Test PPL Improvement Difference: {ppl_diff:+.2f}%")
    print(f"Calibration Time Difference: {time_diff:+.3f}s")
    
    if ppl_diff > 0:
        print(f"✓ Rank 3 BEATS Rank 1 by {ppl_diff:.2f}%")
    else:
        print(f"✗ Rank 3 does NOT beat Rank 1 (difference: {ppl_diff:.2f}%)")
    
    return {
        "expert_utilization": expert_utilization,
        "rank1_ppl_test": ppl1_test,
        "rank3_ppl_test": ppl3_test,
        "ppl_difference": ppl_diff,
        "rank1_time": time1,
        "rank3_time": time3,
        "rank3_clusters": correction3.cluster_assignments.tolist(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run realistic validation with different utilization patterns."""
    
    print("\n" + "="*70)
    print("REALISTIC VALIDATION: Rank 3 Clustered Affine Correction")
    print("="*70)
    
    results = []
    
    # Test with different expert utilization patterns
    for utilization in ["balanced", "imbalanced", "highly_imbalanced"]:
        result = compare_rank1_vs_rank3(
            num_experts=8,
            hidden_size=4096,
            num_calib_samples=1024,
            num_test_samples=512,
            expert_utilization=utilization,
            num_clusters=4,
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY ACROSS UTILIZATION PATTERNS")
    print("="*70)
    
    for result in results:
        print(f"\n{result['expert_utilization'].upper()}:")
        print(f"  Rank 1 PPL Improvement: {result['rank1_ppl_test']:.2f}%")
        print(f"  Rank 3 PPL Improvement: {result['rank3_ppl_test']:.2f}%")
        print(f"  Difference: {result['ppl_difference']:+.2f}%")
    
    # Save results
    results_path = Path(__file__).parent / "validation_rank3_realistic_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

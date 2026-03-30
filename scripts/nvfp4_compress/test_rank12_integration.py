#!/usr/bin/env python3
"""
Test Rank 1 + Rank 2 Integration: Measure Cumulative Improvement

This test validates that Rank 1 (Full Affine) + Rank 2 (Variance Compensation)
work together and produce cumulative improvement over baseline.

Expected Results:
- Rank 1 alone: +10-15% PPL improvement
- Rank 2 alone: +5-10% PPL improvement  
- Rank 1 + Rank 2: +15-25% cumulative PPL improvement
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from phase1_affine_correction import (
    AffineCorrectionFitter,
    AffineCorrection,
    LayerFitMoments,
)
from phase2_sensitivity_guided_correction import (
    HessianWeightedAffineCorrectionFitter,
    DeviationAwareCorrectionFitter,
    HessianWeightedAffineCorrection,
    DeviationAwareCorrection,
)


def generate_synthetic_moe_data(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_batches: int = 2,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate synthetic MoE expert data with realistic quantization errors.
    
    Returns:
        (quantized_outputs, reference_outputs, expert_indices)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    quantized_outputs = []
    reference_outputs = []
    expert_indices = []
    
    for batch_idx in range(num_batches):
        # Generate reference activations (normal distribution)
        reference = torch.randn(batch_size, num_experts, hidden_size) * 2.0 + 0.5
        
        # Simulate quantization with realistic errors
        # Different experts have different error characteristics
        quantized = reference.clone()
        for expert_idx in range(num_experts):
            # Expert-specific error scale (some experts are harder to quantize)
            error_scale = 0.05 + 0.02 * (expert_idx % 3)
            error = torch.randn_like(reference[:, expert_idx, :]) * error_scale
            quantized[:, expert_idx, :] = reference[:, expert_idx, :] + error
        
        quantized_outputs.append(quantized)
        reference_outputs.append(reference)
        
        # Track which expert each sample routes to (for realistic routing)
        routing = torch.randint(0, num_experts, (batch_size,))
        expert_indices.append(routing)
    
    return quantized_outputs, reference_outputs, expert_indices


def apply_affine_correction_per_expert(
    quantized: torch.Tensor,  # (batch, num_experts, hidden_size)
    correction: AffineCorrection,
) -> torch.Tensor:
    """Apply affine correction per expert: y_corrected = alpha * y_quantized + beta"""
    batch_size, num_experts, hidden_size = quantized.shape
    corrected = quantized.clone()
    
    for expert_idx in range(num_experts):
        alpha = correction.alpha[expert_idx].item()
        beta = correction.beta[expert_idx].item()
        corrected[:, expert_idx, :] = quantized[:, expert_idx, :] * alpha + beta
    
    return corrected


def apply_hessian_weighted_affine_correction_per_expert(
    quantized: torch.Tensor,  # (batch, num_experts, hidden_size)
    correction: HessianWeightedAffineCorrection,
) -> torch.Tensor:
    """Apply Hessian-weighted affine correction per expert."""
    batch_size, num_experts, hidden_size = quantized.shape
    corrected = quantized.clone()
    
    for expert_idx in range(num_experts):
        alpha = correction.alpha[expert_idx].item()
        beta = correction.beta[expert_idx].item()
        corrected[:, expert_idx, :] = quantized[:, expert_idx, :] * alpha + beta
    
    return corrected


def test_rank1_only():
    """Test Rank 1 (Full Affine) alone."""
    print("\n" + "="*80)
    print("TEST: Rank 1 (Full Affine) Only")
    print("="*80)
    
    num_experts = 8
    hidden_size = 4096
    
    # Generate data
    quantized_list, reference_list, _ = generate_synthetic_moe_data(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_batches=2,
        batch_size=128,
    )
    
    # Compute baseline MSE
    baseline_mse = 0.0
    for q, r in zip(quantized_list, reference_list):
        baseline_mse += torch.mean((r - q) ** 2).item()
    baseline_mse /= len(quantized_list)
    print(f"Baseline MSE: {baseline_mse:.6f}")
    
    # Fit Rank 1 correction
    fitter = AffineCorrectionFitter(num_experts, hidden_size, device="cpu")
    stats = fitter.initialize_moments()
    
    for batch_idx, (quantized, reference) in enumerate(zip(quantized_list, reference_list)):
        for expert_idx in range(num_experts):
            q_expert = quantized[:, expert_idx, :]
            r_expert = reference[:, expert_idx, :]
            fitter.accumulate_moments(stats, expert_idx, q_expert, r_expert)
    
    correction = fitter.solve_scalar_affine(stats)
    print(f"Rank 1 Correction (scalar mode):")
    print(f"  Alpha shape: {correction.alpha.shape}")
    print(f"  Beta shape: {correction.beta.shape}")
    
    # Apply correction and measure improvement
    corrected_mse = 0.0
    for quantized, reference in zip(quantized_list, reference_list):
        corrected = apply_affine_correction_per_expert(quantized, correction)
        corrected_mse += torch.mean((reference - corrected) ** 2).item()
    corrected_mse /= len(quantized_list)
    
    improvement = (baseline_mse - corrected_mse) / baseline_mse * 100
    print(f"Corrected MSE: {corrected_mse:.6f}")
    print(f"Rank 1 Improvement: {improvement:.2f}%")
    
    return {
        "baseline_mse": baseline_mse,
        "rank1_mse": corrected_mse,
        "rank1_improvement": improvement,
    }


def test_rank2_only():
    """Test Rank 2 (Variance Compensation) alone."""
    print("\n" + "="*80)
    print("TEST: Rank 2 (Variance Compensation) Only")
    print("="*80)
    
    num_experts = 8
    hidden_size = 4096
    
    # Generate data
    quantized_list, reference_list, _ = generate_synthetic_moe_data(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_batches=2,
        batch_size=128,
    )
    
    # Compute baseline MSE
    baseline_mse = 0.0
    for q, r in zip(quantized_list, reference_list):
        baseline_mse += torch.mean((r - q) ** 2).item()
    baseline_mse /= len(quantized_list)
    print(f"Baseline MSE: {baseline_mse:.6f}")
    
    # Fit Rank 2 correction (Hessian-weighted affine)
    fitter = HessianWeightedAffineCorrectionFitter(num_experts, hidden_size, device="cpu")
    stats = fitter.initialize_moments()
    
    # Compute Fisher weights (uniform for this test)
    fisher_weights = fitter.compute_fisher_weights(num_experts=num_experts)
    
    for batch_idx, (quantized, reference) in enumerate(zip(quantized_list, reference_list)):
        for expert_idx in range(num_experts):
            q_expert = quantized[:, expert_idx, :]
            r_expert = reference[:, expert_idx, :]
            fitter.accumulate_weighted_moments(
                stats, expert_idx, q_expert, r_expert,
                fisher_weight=fisher_weights[expert_idx].item()
            )
    
    correction = fitter.solve_hessian_weighted_affine(stats)
    print(f"Rank 2 Correction (Hessian-weighted scalar mode):")
    print(f"  Alpha shape: {correction.alpha.shape}")
    print(f"  Beta shape: {correction.beta.shape}")
    
    # Apply correction and measure improvement
    corrected_mse = 0.0
    for quantized, reference in zip(quantized_list, reference_list):
        corrected = apply_hessian_weighted_affine_correction_per_expert(quantized, correction)
        corrected_mse += torch.mean((reference - corrected) ** 2).item()
    corrected_mse /= len(quantized_list)
    
    improvement = (baseline_mse - corrected_mse) / baseline_mse * 100
    print(f"Corrected MSE: {corrected_mse:.6f}")
    print(f"Rank 2 Improvement: {improvement:.2f}%")
    
    return {
        "baseline_mse": baseline_mse,
        "rank2_mse": corrected_mse,
        "rank2_improvement": improvement,
    }


def test_rank1_plus_rank2():
    """Test Rank 1 + Rank 2 combined (cumulative improvement)."""
    print("\n" + "="*80)
    print("TEST: Rank 1 + Rank 2 Combined (Cumulative)")
    print("="*80)
    
    num_experts = 8
    hidden_size = 4096
    
    # Generate data
    quantized_list, reference_list, _ = generate_synthetic_moe_data(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_batches=2,
        batch_size=128,
    )
    
    # Compute baseline MSE
    baseline_mse = 0.0
    for q, r in zip(quantized_list, reference_list):
        baseline_mse += torch.mean((r - q) ** 2).item()
    baseline_mse /= len(quantized_list)
    print(f"Baseline MSE: {baseline_mse:.6f}")
    
    # Step 1: Apply Rank 1 (Full Affine)
    print("\nStep 1: Applying Rank 1 (Full Affine)...")
    fitter1 = AffineCorrectionFitter(num_experts, hidden_size, device="cpu")
    stats1 = fitter1.initialize_moments()
    
    for quantized, reference in zip(quantized_list, reference_list):
        for expert_idx in range(num_experts):
            q_expert = quantized[:, expert_idx, :]
            r_expert = reference[:, expert_idx, :]
            fitter1.accumulate_moments(stats1, expert_idx, q_expert, r_expert)
    
    correction1 = fitter1.solve_scalar_affine(stats1)
    
    # Apply Rank 1 correction
    quantized_after_rank1 = []
    for quantized, reference in zip(quantized_list, reference_list):
        corrected = apply_affine_correction_per_expert(quantized, correction1)
        quantized_after_rank1.append(corrected)
    
    rank1_mse = 0.0
    for corrected, reference in zip(quantized_after_rank1, reference_list):
        rank1_mse += torch.mean((reference - corrected) ** 2).item()
    rank1_mse /= len(quantized_after_rank1)
    rank1_improvement = (baseline_mse - rank1_mse) / baseline_mse * 100
    print(f"After Rank 1: MSE = {rank1_mse:.6f}, Improvement = {rank1_improvement:.2f}%")
    
    # Step 2: Apply Rank 2 (Variance Compensation) on top of Rank 1
    print("\nStep 2: Applying Rank 2 (Variance Compensation) on top of Rank 1...")
    fitter2 = HessianWeightedAffineCorrectionFitter(num_experts, hidden_size, device="cpu")
    stats2 = fitter2.initialize_moments()
    
    fisher_weights = fitter2.compute_fisher_weights(num_experts=num_experts)
    
    for quantized_r1, reference in zip(quantized_after_rank1, reference_list):
        for expert_idx in range(num_experts):
            q_expert = quantized_r1[:, expert_idx, :]
            r_expert = reference[:, expert_idx, :]
            fitter2.accumulate_weighted_moments(
                stats2, expert_idx, q_expert, r_expert,
                fisher_weight=fisher_weights[expert_idx].item()
            )
    
    correction2 = fitter2.solve_hessian_weighted_affine(stats2)
    
    # Apply Rank 2 correction
    quantized_after_rank12 = []
    for quantized_r1 in quantized_after_rank1:
        corrected = apply_hessian_weighted_affine_correction_per_expert(quantized_r1, correction2)
        quantized_after_rank12.append(corrected)
    
    rank12_mse = 0.0
    for corrected, reference in zip(quantized_after_rank12, reference_list):
        rank12_mse += torch.mean((reference - corrected) ** 2).item()
    rank12_mse /= len(quantized_after_rank12)
    rank12_improvement = (baseline_mse - rank12_mse) / baseline_mse * 100
    print(f"After Rank 1+2: MSE = {rank12_mse:.6f}, Improvement = {rank12_improvement:.2f}%")
    
    # Compute cumulative improvement
    cumulative_improvement = rank12_improvement
    additional_improvement = rank12_improvement - rank1_improvement
    print(f"\nCumulative Improvement (Rank 1+2): {cumulative_improvement:.2f}%")
    print(f"Additional Improvement from Rank 2: {additional_improvement:.2f}%")
    
    return {
        "baseline_mse": baseline_mse,
        "rank1_mse": rank1_mse,
        "rank1_improvement": rank1_improvement,
        "rank12_mse": rank12_mse,
        "rank12_improvement": rank12_improvement,
        "additional_improvement_from_rank2": additional_improvement,
    }


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RANK 1 + RANK 2 INTEGRATION TEST")
    print("="*80)
    
    results = {}
    
    # Test Rank 1 only
    try:
        results["rank1_only"] = test_rank1_only()
    except Exception as e:
        print(f"ERROR in Rank 1 test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Rank 2 only
    try:
        results["rank2_only"] = test_rank2_only()
    except Exception as e:
        print(f"ERROR in Rank 2 test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Rank 1 + Rank 2
    try:
        results["rank1_plus_rank2"] = test_rank1_plus_rank2()
    except Exception as e:
        print(f"ERROR in Rank 1+2 test: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if "rank1_only" in results:
        r1 = results["rank1_only"]
        print(f"Rank 1 Improvement: {r1['rank1_improvement']:.2f}%")
    
    if "rank2_only" in results:
        r2 = results["rank2_only"]
        print(f"Rank 2 Improvement: {r2['rank2_improvement']:.2f}%")
    
    if "rank1_plus_rank2" in results:
        r12 = results["rank1_plus_rank2"]
        print(f"Rank 1+2 Cumulative Improvement: {r12['rank12_improvement']:.2f}%")
        print(f"Additional Improvement from Rank 2: {r12['additional_improvement_from_rank2']:.2f}%")
    
    # Save results
    output_file = Path(__file__).parent / "test_rank12_integration_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()

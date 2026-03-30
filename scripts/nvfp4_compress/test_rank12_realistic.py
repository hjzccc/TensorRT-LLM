#!/usr/bin/env python3
"""
Test Rank 1 + Rank 2 Integration with Realistic Quantization Errors

This test uses realistic quantization errors from actual FP4 quantization
to validate that Rank 1 + Rank 2 produce cumulative improvement.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

sys.path.insert(0, str(Path(__file__).parent))

from phase1_affine_correction import (
    AffineCorrectionFitter,
    AffineCorrection,
)
from phase2_sensitivity_guided_correction import (
    HessianWeightedAffineCorrectionFitter,
    HessianWeightedAffineCorrection,
)


def quantize_to_fp4(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Simulate FP4 quantization with realistic errors."""
    # FP4 has 4 bits: 1 sign + 3 exponent/mantissa
    # Quantization levels: [-6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6]
    fp4_levels = torch.tensor(
        [-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0],
        dtype=torch.float32
    )
    
    # Scale and quantize
    x_scaled = x / scale
    
    # Find nearest FP4 level
    distances = torch.abs(x_scaled.unsqueeze(-1) - fp4_levels.unsqueeze(0))
    indices = torch.argmin(distances, dim=-1)
    quantized = fp4_levels[indices] * scale
    
    return quantized


def generate_realistic_moe_data(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_batches: int = 2,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate realistic MoE data with FP4 quantization errors.
    
    Returns:
        (quantized_outputs, reference_outputs)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    quantized_outputs = []
    reference_outputs = []
    
    for batch_idx in range(num_batches):
        # Generate reference activations with realistic distribution
        # MoE experts typically have skewed activation distributions
        reference = torch.randn(batch_size, num_experts, hidden_size) * 2.0
        reference = torch.abs(reference)  # ReLU-like activations
        
        # Quantize with FP4
        quantized = reference.clone()
        for expert_idx in range(num_experts):
            # Different experts have different scales
            scale = reference[:, expert_idx, :].abs().max() / 6.0
            scale = max(scale.item(), 0.01)  # Avoid division by zero
            quantized[:, expert_idx, :] = quantize_to_fp4(
                reference[:, expert_idx, :],
                scale=scale
            )
        
        quantized_outputs.append(quantized)
        reference_outputs.append(reference)
    
    return quantized_outputs, reference_outputs


def apply_affine_correction_per_expert(
    quantized: torch.Tensor,
    correction: AffineCorrection,
) -> torch.Tensor:
    """Apply affine correction per expert."""
    batch_size, num_experts, hidden_size = quantized.shape
    corrected = quantized.clone()
    
    for expert_idx in range(num_experts):
        alpha = correction.alpha[expert_idx].item()
        beta = correction.beta[expert_idx].item()
        corrected[:, expert_idx, :] = quantized[:, expert_idx, :] * alpha + beta
    
    return corrected


def apply_hessian_weighted_affine_correction_per_expert(
    quantized: torch.Tensor,
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


def test_rank1_plus_rank2_realistic():
    """Test Rank 1 + Rank 2 with realistic FP4 quantization."""
    print("\n" + "="*80)
    print("TEST: Rank 1 + Rank 2 with Realistic FP4 Quantization")
    print("="*80)
    
    num_experts = 8
    hidden_size = 4096
    
    # Generate realistic data
    print("\nGenerating realistic FP4 quantization data...")
    quantized_list, reference_list = generate_realistic_moe_data(
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
    print(f"Baseline MSE (FP4 quantization error): {baseline_mse:.6f}")
    
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
    print(f"Rank 1 Alpha (sample): {correction1.alpha[:3]}")
    print(f"Rank 1 Beta (sample): {correction1.beta[:3]}")
    
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
    print(f"Rank 2 Alpha (sample): {correction2.alpha[:3]}")
    print(f"Rank 2 Beta (sample): {correction2.beta[:3]}")
    
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
    """Run test."""
    print("\n" + "="*80)
    print("RANK 1 + RANK 2 INTEGRATION TEST (REALISTIC FP4)")
    print("="*80)
    
    results = {}
    
    try:
        results["rank1_plus_rank2_realistic"] = test_rank1_plus_rank2_realistic()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if "rank1_plus_rank2_realistic" in results:
        r = results["rank1_plus_rank2_realistic"]
        print(f"Rank 1 Improvement: {r['rank1_improvement']:.2f}%")
        print(f"Rank 1+2 Cumulative Improvement: {r['rank12_improvement']:.2f}%")
        print(f"Additional Improvement from Rank 2: {r['additional_improvement_from_rank2']:.2f}%")
        
        # Verdict
        if r['rank12_improvement'] > r['rank1_improvement']:
            print("\n✓ VERDICT: Rank 2 provides additional improvement over Rank 1")
        else:
            print("\n✗ VERDICT: Rank 2 does NOT provide additional improvement")
    
    # Save results
    output_file = Path(__file__).parent / "test_rank12_realistic_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()

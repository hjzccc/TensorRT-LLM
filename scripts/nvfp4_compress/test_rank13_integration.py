#!/usr/bin/env python3
"""
Test Rank 1 vs Rank 3 Integration: Validate Clustered Affine Improvement

This test compares:
- Rank 1 (Full Affine): Per-expert affine correction
- Rank 3 (Clustered Affine): Shared affine parameters per cluster

Expected Results:
- Rank 1: 7-10% PPL improvement
- Rank 3: 3-7% additional improvement (better generalization)
- Rank 1+3: 10-17% cumulative improvement
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
from phase3_clustered_affine_correction import (
    ClusteredAffineCorrectionFitter,
    ClusteredAffineCorrection,
)


def quantize_to_fp4(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Simulate FP4 quantization with realistic errors."""
    fp4_levels = torch.tensor(
        [-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0],
        dtype=torch.float32
    )
    
    x_scaled = x / scale
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
    """Generate realistic MoE data with FP4 quantization errors."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    quantized_outputs = []
    reference_outputs = []
    
    for batch_idx in range(num_batches):
        reference = torch.randn(batch_size, num_experts, hidden_size) * 2.0
        reference = torch.abs(reference)
        
        quantized = reference.clone()
        for expert_idx in range(num_experts):
            scale = reference[:, expert_idx, :].abs().max() / 6.0
            scale = max(scale.item(), 0.01)
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


def apply_clustered_affine_correction(
    quantized: torch.Tensor,
    correction: ClusteredAffineCorrection,
) -> torch.Tensor:
    """Apply clustered affine correction."""
    batch_size, num_experts, hidden_size = quantized.shape
    corrected = quantized.clone()
    
    for expert_idx in range(num_experts):
        cluster_id = correction.cluster_assignments[expert_idx].item()
        alpha = correction.alpha_clusters[cluster_id].item()
        beta = correction.beta_clusters[cluster_id].item()
        corrected[:, expert_idx, :] = quantized[:, expert_idx, :] * alpha + beta
    
    return corrected


def test_rank1_vs_rank3():
    """Test Rank 1 vs Rank 3 on realistic FP4 quantization."""
    print("\n" + "="*80)
    print("TEST: Rank 1 vs Rank 3 (Clustered Affine)")
    print("="*80)
    
    num_experts = 8
    hidden_size = 4096
    num_clusters = 4
    
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
    
    # Test Rank 1 (Full Affine)
    print("\n" + "-"*80)
    print("Rank 1: Full Affine Correction (Per-Expert)")
    print("-"*80)
    
    fitter1 = AffineCorrectionFitter(num_experts, hidden_size, device="cpu")
    stats1 = fitter1.initialize_moments()
    
    for quantized, reference in zip(quantized_list, reference_list):
        for expert_idx in range(num_experts):
            q_expert = quantized[:, expert_idx, :]
            r_expert = reference[:, expert_idx, :]
            fitter1.accumulate_moments(stats1, expert_idx, q_expert, r_expert)
    
    correction1 = fitter1.solve_scalar_affine(stats1)
    print(f"Learned {num_experts} per-expert affine parameters")
    print(f"  Alpha (sample): {correction1.alpha[:3]}")
    print(f"  Beta (sample): {correction1.beta[:3]}")
    
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
    print(f"Rank 1 MSE: {rank1_mse:.6f}")
    print(f"Rank 1 Improvement: {rank1_improvement:.2f}%")
    
    # Test Rank 3 (Clustered Affine)
    print("\n" + "-"*80)
    print(f"Rank 3: Clustered Affine Correction ({num_clusters} clusters)")
    print("-"*80)
    
    fitter3 = ClusteredAffineCorrectionFitter(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_clusters=num_clusters,
        device="cpu"
    )
    stats3 = fitter3.initialize_moments()
    
    for quantized, reference in zip(quantized_list, reference_list):
        for expert_idx in range(num_experts):
            q_expert = quantized[:, expert_idx, :]
            r_expert = reference[:, expert_idx, :]
            fitter3.accumulate_moments(stats3, expert_idx, q_expert, r_expert)
    
    correction3 = fitter3.fit(stats3)
    print(f"Learned {num_clusters} shared affine parameters")
    print(f"Cluster assignments: {correction3.cluster_assignments}")
    print(f"  Alpha (per cluster): {correction3.alpha_clusters}")
    print(f"  Beta (per cluster): {correction3.beta_clusters}")
    
    # Apply Rank 3 correction
    quantized_after_rank3 = []
    for quantized, reference in zip(quantized_list, reference_list):
        corrected = apply_clustered_affine_correction(quantized, correction3)
        quantized_after_rank3.append(corrected)
    
    rank3_mse = 0.0
    for corrected, reference in zip(quantized_after_rank3, reference_list):
        rank3_mse += torch.mean((reference - corrected) ** 2).item()
    rank3_mse /= len(quantized_after_rank3)
    rank3_improvement = (baseline_mse - rank3_mse) / baseline_mse * 100
    print(f"Rank 3 MSE: {rank3_mse:.6f}")
    print(f"Rank 3 Improvement: {rank3_improvement:.2f}%")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON: Rank 1 vs Rank 3")
    print("="*80)
    print(f"Baseline MSE:        {baseline_mse:.6f}")
    print(f"Rank 1 MSE:          {rank1_mse:.6f} (Improvement: {rank1_improvement:.2f}%)")
    print(f"Rank 3 MSE:          {rank3_mse:.6f} (Improvement: {rank3_improvement:.2f}%)")
    print(f"Difference (R1 - R3): {rank1_mse - rank3_mse:.6f}")
    
    if rank3_improvement > rank1_improvement:
        additional = rank3_improvement - rank1_improvement
        print(f"\n✓ Rank 3 BEATS Rank 1 by {additional:.2f}%")
    elif rank3_improvement < rank1_improvement:
        difference = rank1_improvement - rank3_improvement
        print(f"\n✗ Rank 1 BEATS Rank 3 by {difference:.2f}%")
    else:
        print(f"\n= Rank 1 and Rank 3 are EQUIVALENT")
    
    return {
        "baseline_mse": baseline_mse,
        "rank1_mse": rank1_mse,
        "rank1_improvement": rank1_improvement,
        "rank3_mse": rank3_mse,
        "rank3_improvement": rank3_improvement,
        "rank3_beats_rank1": rank3_improvement > rank1_improvement,
        "difference": rank3_improvement - rank1_improvement,
    }


def main():
    """Run test."""
    print("\n" + "="*80)
    print("RANK 1 vs RANK 3 COMPARISON TEST")
    print("="*80)
    
    results = {}
    
    try:
        results["rank1_vs_rank3"] = test_rank1_vs_rank3()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if "rank1_vs_rank3" in results:
        r = results["rank1_vs_rank3"]
        print(f"Rank 1 Improvement: {r['rank1_improvement']:.2f}%")
        print(f"Rank 3 Improvement: {r['rank3_improvement']:.2f}%")
        print(f"Difference: {r['difference']:.2f}%")
        
        if r['rank3_beats_rank1']:
            print(f"\n✓ RANK 3 IS BETTER (by {r['difference']:.2f}%)")
        else:
            print(f"\n✗ RANK 1 IS BETTER (by {-r['difference']:.2f}%)")
    
    # Save results
    output_file = Path(__file__).parent / "test_rank13_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()

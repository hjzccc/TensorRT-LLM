#!/usr/bin/env python3
"""
Phase 31: Conditional Affine Correction

Idea: Instead of a single affine correction (α*x + β), use different
affine parameters for different ranges of activation magnitudes.

This accounts for the fact that quantization error may vary across
different activation ranges.

Expected improvement: +1-4% additional over Phase 25
"""

import torch
import json
import numpy as np
from typing import Tuple, List


def apply_phase25_bias_correction(quantized, reference):
    """Apply Phase 25: Bias-Only Correction"""
    corrected = quantized.clone()
    error = reference - quantized
    bias = error.mean(dim=0)
    corrected += bias
    return corrected


def apply_conditional_affine_correction(
    quantized: torch.Tensor,
    reference: torch.Tensor,
    num_ranges: int = 4,
) -> torch.Tensor:
    """
    Apply conditional affine correction.
    
    Divide the data into ranges based on activation magnitude and apply
    different affine corrections to each range.
    
    Args:
        quantized: Quantized data
        reference: Reference (original) data
        num_ranges: Number of activation ranges
    
    Returns:
        corrected_data: Data with conditional affine correction applied
    """
    corrected = quantized.clone()
    
    # Compute range boundaries based on quantized data magnitude
    magnitude = quantized.abs()
    range_boundaries = torch.quantile(
        magnitude,
        torch.linspace(0, 1, num_ranges + 1)
    )
    
    # Apply different affine corrections to each range
    for i in range(num_ranges):
        lower = range_boundaries[i]
        upper = range_boundaries[i + 1]
        
        # Find elements in this range
        mask = (magnitude >= lower) & (magnitude <= upper)
        
        if mask.sum() > 0:
            # Compute optimal affine parameters for this range
            q = quantized[mask]
            r = reference[mask]
            
            # Compute α and β
            q_mean = q.mean()
            r_mean = r.mean()
            
            q_centered = q - q_mean
            r_centered = r - r_mean
            
            cov = (q_centered * r_centered).mean()
            var = (q_centered ** 2).mean()
            
            alpha = cov / (var + 1e-8)
            beta = r_mean - alpha * q_mean
            
            # Apply correction
            corrected[mask] = alpha * corrected[mask] + beta
    
    return corrected


def apply_piecewise_linear_correction(
    quantized: torch.Tensor,
    reference: torch.Tensor,
    num_segments: int = 4,
) -> torch.Tensor:
    """
    Apply piecewise linear correction.
    
    Similar to conditional affine, but uses a continuous piecewise linear
    function instead of separate affine corrections per range.
    
    Args:
        quantized: Quantized data
        reference: Reference (original) data
        num_segments: Number of linear segments
    
    Returns:
        corrected_data: Data with piecewise linear correction applied
    """
    corrected = quantized.clone()
    
    # Compute segment boundaries
    magnitude = quantized.abs()
    segment_boundaries = torch.quantile(
        magnitude,
        torch.linspace(0, 1, num_segments + 1)
    )
    
    # For each segment, compute the optimal linear function
    slopes = []
    intercepts = []
    
    for i in range(num_segments):
        lower = segment_boundaries[i]
        upper = segment_boundaries[i + 1]
        
        # Find elements in this segment
        mask = (magnitude >= lower) & (magnitude <= upper)
        
        if mask.sum() > 0:
            q = quantized[mask]
            r = reference[mask]
            
            # Compute optimal linear function
            q_mean = q.mean()
            r_mean = r.mean()
            
            q_centered = q - q_mean
            r_centered = r - r_mean
            
            cov = (q_centered * r_centered).mean()
            var = (q_centered ** 2).mean()
            
            slope = cov / (var + 1e-8)
            intercept = r_mean - slope * q_mean
            
            slopes.append(slope)
            intercepts.append(intercept)
        else:
            slopes.append(1.0)
            intercepts.append(0.0)
    
    # Apply piecewise linear correction
    for i in range(num_segments):
        lower = segment_boundaries[i]
        upper = segment_boundaries[i + 1]
        
        mask = (magnitude >= lower) & (magnitude <= upper)
        
        if mask.sum() > 0:
            corrected[mask] = slopes[i] * corrected[mask] + intercepts[i]
    
    return corrected


def test_conditional_affine_correction():
    """Test conditional affine correction"""
    
    print("=" * 80)
    print("PHASE 31: CONDITIONAL AFFINE CORRECTION")
    print("=" * 80)
    
    # Configuration
    num_experts = 8
    hidden_size = 4096
    num_calibration_batches = 2
    batch_size = 128
    device = "cpu"
    
    # Generate realistic FP4 data with varying error patterns
    print("\nGenerating realistic FP4 quantization data...")
    torch.manual_seed(42)
    
    reference_data = torch.randn(
        num_experts, num_calibration_batches * batch_size, hidden_size,
        device=device
    )
    
    # Simulate FP4 quantization with magnitude-dependent error
    quantized_data = reference_data.clone()
    
    for expert_idx in range(num_experts):
        expert_data = reference_data[expert_idx]
        # Quantization error scales with activation magnitude
        noise_scale = 0.02 * expert_data.abs().mean()
        quantized_data[expert_idx] += torch.randn_like(expert_data) * noise_scale
    
    # Compute baseline MSE
    baseline_mse = torch.mean((reference_data - quantized_data) ** 2).item()
    print(f"Baseline MSE (FP4 quantization error): {baseline_mse:.6f}")
    
    # ========================================================================
    # PHASE 25: BASELINE (BIAS-ONLY)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 25: Bias-Only Correction (Baseline)")
    print("-" * 80)
    
    phase25_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        phase25_corrected[expert_idx] = apply_phase25_bias_correction(
            quantized_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1]
        )[0]
    
    phase25_mse = torch.mean((reference_data - phase25_corrected) ** 2).item()
    phase25_improvement = 100 * (baseline_mse - phase25_mse) / baseline_mse
    
    print(f"Phase 25 MSE: {phase25_mse:.6f}")
    print(f"Phase 25 Improvement: {phase25_improvement:.2f}%")
    
    # ========================================================================
    # PHASE 31A: CONDITIONAL AFFINE (4 RANGES)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 31A: Conditional Affine Correction (4 Ranges)")
    print("-" * 80)
    
    phase31a_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        phase31a_corrected[expert_idx] = apply_conditional_affine_correction(
            quantized_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1],
            num_ranges=4
        )[0]
    
    phase31a_mse = torch.mean((reference_data - phase31a_corrected) ** 2).item()
    phase31a_improvement = 100 * (baseline_mse - phase31a_mse) / baseline_mse
    phase31a_additional = phase31a_improvement - phase25_improvement
    
    print(f"Phase 31A MSE: {phase31a_mse:.6f}")
    print(f"Phase 31A Improvement: {phase31a_improvement:.2f}%")
    print(f"Additional vs Phase 25: {phase31a_additional:.2f}%")
    
    # ========================================================================
    # PHASE 31B: CONDITIONAL AFFINE (8 RANGES)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 31B: Conditional Affine Correction (8 Ranges)")
    print("-" * 80)
    
    phase31b_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        phase31b_corrected[expert_idx] = apply_conditional_affine_correction(
            quantized_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1],
            num_ranges=8
        )[0]
    
    phase31b_mse = torch.mean((reference_data - phase31b_corrected) ** 2).item()
    phase31b_improvement = 100 * (baseline_mse - phase31b_mse) / baseline_mse
    phase31b_additional = phase31b_improvement - phase25_improvement
    
    print(f"Phase 31B MSE: {phase31b_mse:.6f}")
    print(f"Phase 31B Improvement: {phase31b_improvement:.2f}%")
    print(f"Additional vs Phase 25: {phase31b_additional:.2f}%")
    
    # ========================================================================
    # PHASE 31C: PIECEWISE LINEAR (4 SEGMENTS)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 31C: Piecewise Linear Correction (4 Segments)")
    print("-" * 80)
    
    phase31c_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        phase31c_corrected[expert_idx] = apply_piecewise_linear_correction(
            quantized_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1],
            num_segments=4
        )[0]
    
    phase31c_mse = torch.mean((reference_data - phase31c_corrected) ** 2).item()
    phase31c_improvement = 100 * (baseline_mse - phase31c_mse) / baseline_mse
    phase31c_additional = phase31c_improvement - phase25_improvement
    
    print(f"Phase 31C MSE: {phase31c_mse:.6f}")
    print(f"Phase 31C Improvement: {phase31c_improvement:.2f}%")
    print(f"Additional vs Phase 25: {phase31c_additional:.2f}%")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONDITIONAL AFFINE CORRECTION ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nBaseline MSE:                           {baseline_mse:.6f}")
    print(f"\nPhase 25 (Baseline):                    {phase25_improvement:.2f}%")
    print(f"Phase 31A (Conditional, 4 ranges):      {phase31a_improvement:.2f}% (+{phase31a_additional:.2f}%)")
    print(f"Phase 31B (Conditional, 8 ranges):      {phase31b_improvement:.2f}% (+{phase31b_additional:.2f}%)")
    print(f"Phase 31C (Piecewise Linear, 4):        {phase31c_improvement:.2f}% (+{phase31c_additional:.2f}%)")
    
    # Determine best technique
    improvements = {
        "phase25": phase25_improvement,
        "phase31a": phase31a_improvement,
        "phase31b": phase31b_improvement,
        "phase31c": phase31c_improvement,
    }
    
    best_technique = max(improvements, key=improvements.get)
    best_improvement = improvements[best_technique]
    
    print(f"\n✓ BEST TECHNIQUE: {best_technique.upper()} ({best_improvement:.2f}%)")
    
    # Save results
    results_dict = {
        "baseline_mse": baseline_mse,
        "phase25_improvement": phase25_improvement,
        "phase31a_conditional_4": {
            "improvement": phase31a_improvement,
            "additional_vs_phase25": phase31a_additional,
        },
        "phase31b_conditional_8": {
            "improvement": phase31b_improvement,
            "additional_vs_phase25": phase31b_additional,
        },
        "phase31c_piecewise_linear_4": {
            "improvement": phase31c_improvement,
            "additional_vs_phase25": phase31c_additional,
        },
        "best_technique": best_technique,
        "best_improvement": best_improvement,
    }
    
    with open("test_phase31_conditional_affine_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: test_phase31_conditional_affine_results.json")
    print("=" * 80)
    
    return results_dict


if __name__ == "__main__":
    test_conditional_affine_correction()

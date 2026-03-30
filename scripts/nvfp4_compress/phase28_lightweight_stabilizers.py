#!/usr/bin/env python3
"""
Phase 28: Lightweight Stabilizers for NVFP4 Quantization

Implements three lightweight stabilizer techniques:
1. Outlier Clipping: Remove extreme values before correction
2. Quantile-Based Scaling: Per-expert scaling adjustment
3. Per-Block Variance Normalization: Normalize variance before correction

These are orthogonal to Phase 25 (Bias-Only) and can be combined.
Expected improvement: +1-5% additional over Phase 25.
"""

import torch
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass(frozen=True)
class StabilizerConfig:
    """Configuration for stabilizers"""
    outlier_percentile: float = 95.0  # Remove top 5% outliers
    quantile_scale_percentile: float = 90.0  # Use 90th percentile for scaling
    variance_norm_epsilon: float = 1e-8
    apply_clipping: bool = True
    apply_quantile_scaling: bool = True
    apply_variance_norm: bool = True


def apply_outlier_clipping(
    data: torch.Tensor,
    percentile: float = 95.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clip outliers based on percentile.
    
    Args:
        data: Input tensor of shape (batch_size, hidden_size)
        percentile: Percentile threshold (e.g., 95 means remove top 5%)
    
    Returns:
        clipped_data: Data with outliers clipped
        clip_mask: Boolean mask indicating clipped values
    """
    # Compute percentile threshold per element
    threshold = torch.quantile(data.abs(), percentile / 100.0)
    
    # Clip values
    clipped_data = torch.clamp(data, -threshold, threshold)
    
    # Track which values were clipped
    clip_mask = (data.abs() > threshold)
    
    return clipped_data, clip_mask


def apply_quantile_scaling(
    quantized: torch.Tensor,
    reference: torch.Tensor,
    percentile: float = 90.0
) -> torch.Tensor:
    """
    Apply per-expert scaling based on quantile of reference data.
    
    Idea: Scale quantized data to match the scale of reference data.
    This helps when quantization changes the scale of activations.
    
    Args:
        quantized: Quantized data
        reference: Reference (original) data
        percentile: Percentile to use for scaling (e.g., 90 means use 90th percentile)
    
    Returns:
        scaled_data: Quantized data scaled to match reference scale
    """
    # Compute quantile of absolute values
    q_quantile = torch.quantile(quantized.abs(), percentile / 100.0)
    r_quantile = torch.quantile(reference.abs(), percentile / 100.0)
    
    # Compute scale factor
    scale = r_quantile / (q_quantile + 1e-8)
    
    # Apply scaling
    scaled_data = quantized * scale
    
    return scaled_data


def apply_variance_normalization(
    data: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Normalize variance of data.
    
    Idea: Normalize variance to unit variance, which can help with
    correction stability.
    
    Args:
        data: Input tensor
        epsilon: Small constant for numerical stability
    
    Returns:
        normalized_data: Data with unit variance
    """
    mean = data.mean()
    std = data.std() + epsilon
    
    normalized_data = (data - mean) / std
    
    return normalized_data


def apply_phase25_bias_correction(quantized, reference):
    """Apply Phase 25: Bias-Only Correction"""
    corrected = quantized.clone()
    
    # Compute mean error
    error = reference - quantized
    bias = error.mean(dim=0)  # Mean across batch dimension
    corrected += bias
    
    return corrected


def test_stabilizers():
    """Test lightweight stabilizers"""
    
    print("=" * 80)
    print("PHASE 28: LIGHTWEIGHT STABILIZERS FOR NVFP4 QUANTIZATION")
    print("=" * 80)
    
    # Configuration
    num_experts = 8
    hidden_size = 4096
    num_calibration_batches = 2
    batch_size = 128
    device = "cpu"
    
    # Generate realistic FP4 data with outliers
    print("\nGenerating realistic FP4 quantization data with outliers...")
    torch.manual_seed(42)
    
    reference_data = torch.randn(
        num_experts, num_calibration_batches * batch_size, hidden_size,
        device=device
    )
    
    # Simulate FP4 quantization with realistic error patterns and outliers
    quantized_data = reference_data.clone()
    
    for expert_idx in range(num_experts):
        expert_data = reference_data[expert_idx]
        # Quantization error scales with activation magnitude
        noise_scale = 0.02 * expert_data.abs().mean()
        quantized_data[expert_idx] += torch.randn_like(expert_data) * noise_scale
        
        # Add some outliers (5% of values)
        outlier_mask = torch.rand_like(expert_data) < 0.05
        outlier_magnitude = 5.0 * expert_data.abs().mean()
        quantized_data[expert_idx][outlier_mask] += torch.randn_like(expert_data[outlier_mask]) * outlier_magnitude
    
    # Compute baseline MSE
    baseline_mse = torch.mean((reference_data - quantized_data) ** 2).item()
    print(f"Baseline MSE (FP4 quantization error): {baseline_mse:.6f}")
    
    # ========================================================================
    # PHASE 25: BASELINE (NO STABILIZERS)
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
    # STABILIZER 1: OUTLIER CLIPPING
    # ========================================================================
    print("\n" + "-" * 80)
    print("Stabilizer 1: Outlier Clipping")
    print("-" * 80)
    
    clipped_data = quantized_data.clone()
    for expert_idx in range(num_experts):
        clipped_data[expert_idx], _ = apply_outlier_clipping(
            quantized_data[expert_idx],
            percentile=95.0
        )
    
    # Apply Phase 25 on clipped data
    clipped_corrected = clipped_data.clone()
    for expert_idx in range(num_experts):
        clipped_corrected[expert_idx] = apply_phase25_bias_correction(
            clipped_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1]
        )[0]
    
    clipped_mse = torch.mean((reference_data - clipped_corrected) ** 2).item()
    clipped_improvement = 100 * (baseline_mse - clipped_mse) / baseline_mse
    clipped_vs_phase25 = 100 * (phase25_mse - clipped_mse) / phase25_mse if phase25_mse > 0 else 0
    
    print(f"Outlier Clipping + Phase 25 MSE: {clipped_mse:.6f}")
    print(f"Total Improvement: {clipped_improvement:.2f}%")
    print(f"Additional vs Phase 25: {clipped_vs_phase25:.2f}%")
    
    # ========================================================================
    # STABILIZER 2: QUANTILE-BASED SCALING
    # ========================================================================
    print("\n" + "-" * 80)
    print("Stabilizer 2: Quantile-Based Scaling")
    print("-" * 80)
    
    scaled_data = quantized_data.clone()
    for expert_idx in range(num_experts):
        scaled_data[expert_idx] = apply_quantile_scaling(
            quantized_data[expert_idx],
            reference_data[expert_idx],
            percentile=90.0
        )
    
    # Apply Phase 25 on scaled data
    scaled_corrected = scaled_data.clone()
    for expert_idx in range(num_experts):
        scaled_corrected[expert_idx] = apply_phase25_bias_correction(
            scaled_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1]
        )[0]
    
    scaled_mse = torch.mean((reference_data - scaled_corrected) ** 2).item()
    scaled_improvement = 100 * (baseline_mse - scaled_mse) / baseline_mse
    scaled_vs_phase25 = 100 * (phase25_mse - scaled_mse) / phase25_mse if phase25_mse > 0 else 0
    
    print(f"Quantile Scaling + Phase 25 MSE: {scaled_mse:.6f}")
    print(f"Total Improvement: {scaled_improvement:.2f}%")
    print(f"Additional vs Phase 25: {scaled_vs_phase25:.2f}%")
    
    # ========================================================================
    # STABILIZER 3: VARIANCE NORMALIZATION
    # ========================================================================
    print("\n" + "-" * 80)
    print("Stabilizer 3: Variance Normalization")
    print("-" * 80)
    
    # Note: Variance normalization changes scale, so we need to denormalize after correction
    var_norm_data = quantized_data.clone()
    var_norm_scales = []
    
    for expert_idx in range(num_experts):
        q = quantized_data[expert_idx]
        r = reference_data[expert_idx]
        
        # Normalize quantized data
        q_mean = q.mean()
        q_std = q.std() + 1e-8
        q_normalized = (q - q_mean) / q_std
        
        # Normalize reference data
        r_mean = r.mean()
        r_std = r.std() + 1e-8
        r_normalized = (r - r_mean) / r_std
        
        # Apply Phase 25 on normalized data
        error = r_normalized - q_normalized
        bias = error.mean(dim=0)
        q_corrected = q_normalized + bias
        
        # Denormalize
        q_corrected = q_corrected * r_std + r_mean
        
        var_norm_data[expert_idx] = q_corrected
        var_norm_scales.append((q_mean, q_std, r_mean, r_std))
    
    var_norm_mse = torch.mean((reference_data - var_norm_data) ** 2).item()
    var_norm_improvement = 100 * (baseline_mse - var_norm_mse) / baseline_mse
    var_norm_vs_phase25 = 100 * (phase25_mse - var_norm_mse) / phase25_mse if phase25_mse > 0 else 0
    
    print(f"Variance Normalization + Phase 25 MSE: {var_norm_mse:.6f}")
    print(f"Total Improvement: {var_norm_improvement:.2f}%")
    print(f"Additional vs Phase 25: {var_norm_vs_phase25:.2f}%")
    
    # ========================================================================
    # COMBINED: CLIPPING + SCALING + VARIANCE NORM
    # ========================================================================
    print("\n" + "-" * 80)
    print("Combined: Clipping + Scaling + Variance Normalization")
    print("-" * 80)
    
    combined_data = quantized_data.clone()
    
    for expert_idx in range(num_experts):
        q = quantized_data[expert_idx]
        r = reference_data[expert_idx]
        
        # Step 1: Outlier clipping
        q_clipped, _ = apply_outlier_clipping(q, percentile=95.0)
        
        # Step 2: Quantile scaling
        q_scaled = apply_quantile_scaling(q_clipped, r, percentile=90.0)
        
        # Step 3: Variance normalization + Phase 25
        q_mean = q_scaled.mean()
        q_std = q_scaled.std() + 1e-8
        q_normalized = (q_scaled - q_mean) / q_std
        
        r_mean = r.mean()
        r_std = r.std() + 1e-8
        r_normalized = (r - r_mean) / r_std
        
        error = r_normalized - q_normalized
        bias = error.mean(dim=0)
        q_corrected = q_normalized + bias
        
        # Denormalize
        q_corrected = q_corrected * r_std + r_mean
        
        combined_data[expert_idx] = q_corrected
    
    combined_mse = torch.mean((reference_data - combined_data) ** 2).item()
    combined_improvement = 100 * (baseline_mse - combined_mse) / baseline_mse
    combined_vs_phase25 = 100 * (phase25_mse - combined_mse) / phase25_mse if phase25_mse > 0 else 0
    
    print(f"Combined Stabilizers + Phase 25 MSE: {combined_mse:.6f}")
    print(f"Total Improvement: {combined_improvement:.2f}%")
    print(f"Additional vs Phase 25: {combined_vs_phase25:.2f}%")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("STABILIZER ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nBaseline MSE:                           {baseline_mse:.6f}")
    print(f"\nPhase 25 (Baseline):                    {phase25_improvement:.2f}%")
    print(f"  + Outlier Clipping:                   {clipped_improvement:.2f}% (+{clipped_vs_phase25:.2f}%)")
    print(f"  + Quantile Scaling:                   {scaled_improvement:.2f}% (+{scaled_vs_phase25:.2f}%)")
    print(f"  + Variance Normalization:             {var_norm_improvement:.2f}% (+{var_norm_vs_phase25:.2f}%)")
    print(f"  + Combined (All Three):               {combined_improvement:.2f}% (+{combined_vs_phase25:.2f}%)")
    
    # Determine best stabilizer
    improvements = {
        "phase25": phase25_improvement,
        "clipping": clipped_improvement,
        "scaling": scaled_improvement,
        "variance_norm": var_norm_improvement,
        "combined": combined_improvement,
    }
    
    best_technique = max(improvements, key=improvements.get)
    best_improvement = improvements[best_technique]
    
    print(f"\n✓ BEST TECHNIQUE: {best_technique.upper()} ({best_improvement:.2f}%)")
    
    # Save results
    results_dict = {
        "baseline_mse": baseline_mse,
        "phase25_improvement": phase25_improvement,
        "outlier_clipping": {
            "improvement": clipped_improvement,
            "additional_vs_phase25": clipped_vs_phase25,
        },
        "quantile_scaling": {
            "improvement": scaled_improvement,
            "additional_vs_phase25": scaled_vs_phase25,
        },
        "variance_normalization": {
            "improvement": var_norm_improvement,
            "additional_vs_phase25": var_norm_vs_phase25,
        },
        "combined": {
            "improvement": combined_improvement,
            "additional_vs_phase25": combined_vs_phase25,
        },
        "best_technique": best_technique,
        "best_improvement": best_improvement,
    }
    
    with open("test_phase28_stabilizers_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: test_phase28_stabilizers_results.json")
    print("=" * 80)
    
    return results_dict


if __name__ == "__main__":
    test_stabilizers()

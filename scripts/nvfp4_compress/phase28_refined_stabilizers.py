#!/usr/bin/env python3
"""
Phase 28 Refined: Lightweight Stabilizers - Focus on Outlier Clipping

Tests show outlier clipping is highly effective.
This refined version focuses on:
1. Outlier clipping alone
2. Outlier clipping + Phase 25
3. Different clipping thresholds
"""

import torch
import json
import numpy as np
from typing import Tuple


def apply_outlier_clipping(
    data: torch.Tensor,
    percentile: float = 95.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip outliers based on percentile"""
    threshold = torch.quantile(data.abs(), percentile / 100.0)
    clipped_data = torch.clamp(data, -threshold, threshold)
    clip_mask = (data.abs() > threshold)
    return clipped_data, clip_mask


def apply_phase25_bias_correction(quantized, reference):
    """Apply Phase 25: Bias-Only Correction"""
    corrected = quantized.clone()
    error = reference - quantized
    bias = error.mean(dim=0)
    corrected += bias
    return corrected


def test_refined_stabilizers():
    """Test refined stabilizers with focus on outlier clipping"""
    
    print("=" * 80)
    print("PHASE 28 REFINED: OUTLIER CLIPPING STABILIZER")
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
    # PHASE 25: BASELINE (NO CLIPPING)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 25: Bias-Only Correction (No Clipping)")
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
    # TEST DIFFERENT CLIPPING THRESHOLDS
    # ========================================================================
    print("\n" + "-" * 80)
    print("Testing Different Clipping Thresholds")
    print("-" * 80)
    
    clipping_results = {}
    
    for percentile in [90.0, 92.5, 95.0, 97.5, 99.0]:
        print(f"\nClipping Percentile: {percentile}%")
        
        # Apply clipping
        clipped_data = quantized_data.clone()
        total_clipped = 0
        
        for expert_idx in range(num_experts):
            clipped_data[expert_idx], clip_mask = apply_outlier_clipping(
                quantized_data[expert_idx],
                percentile=percentile
            )
            total_clipped += clip_mask.sum().item()
        
        # Apply Phase 25 on clipped data
        clipped_corrected = clipped_data.clone()
        for expert_idx in range(num_experts):
            clipped_corrected[expert_idx] = apply_phase25_bias_correction(
                clipped_data[expert_idx:expert_idx+1],
                reference_data[expert_idx:expert_idx+1]
            )[0]
        
        clipped_mse = torch.mean((reference_data - clipped_corrected) ** 2).item()
        clipped_improvement = 100 * (baseline_mse - clipped_mse) / baseline_mse
        additional_improvement = clipped_improvement - phase25_improvement
        
        total_elements = reference_data.numel()
        clipped_percentage = 100 * total_clipped / total_elements
        
        print(f"  MSE: {clipped_mse:.6f}")
        print(f"  Total Improvement: {clipped_improvement:.2f}%")
        print(f"  Additional vs Phase 25: {additional_improvement:.2f}%")
        print(f"  Elements Clipped: {clipped_percentage:.2f}%")
        
        clipping_results[percentile] = {
            "mse": clipped_mse,
            "improvement": clipped_improvement,
            "additional": additional_improvement,
            "clipped_percentage": clipped_percentage,
        }
    
    # ========================================================================
    # BEST CLIPPING THRESHOLD
    # ========================================================================
    print("\n" + "-" * 80)
    print("Best Clipping Threshold")
    print("-" * 80)
    
    best_percentile = max(clipping_results, key=lambda p: clipping_results[p]["improvement"])
    best_result = clipping_results[best_percentile]
    
    print(f"Best Percentile: {best_percentile}%")
    print(f"  MSE: {best_result['mse']:.6f}")
    print(f"  Total Improvement: {best_result['improvement']:.2f}%")
    print(f"  Additional vs Phase 25: {best_result['additional']:.2f}%")
    print(f"  Elements Clipped: {best_result['clipped_percentage']:.2f}%")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("STABILIZER ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nBaseline MSE:                           {baseline_mse:.6f}")
    print(f"Phase 25 (No Clipping):                 {phase25_improvement:.2f}%")
    print(f"Phase 25 + Outlier Clipping (Best):     {best_result['improvement']:.2f}%")
    print(f"  Additional Improvement:               {best_result['additional']:.2f}%")
    
    # Save results
    results_dict = {
        "baseline_mse": baseline_mse,
        "phase25_improvement": phase25_improvement,
        "clipping_thresholds": clipping_results,
        "best_percentile": best_percentile,
        "best_improvement": best_result["improvement"],
        "best_additional": best_result["additional"],
    }
    
    with open("test_phase28_refined_stabilizers_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: test_phase28_refined_stabilizers_results.json")
    print("=" * 80)
    
    return results_dict


if __name__ == "__main__":
    test_refined_stabilizers()

#!/usr/bin/env python3
"""
Phase 29: Per-Element Correction vs Per-Block Correction

Compares:
1. Per-Block Correction (Phase 25): One bias per block
2. Per-Element Correction: One bias per element
3. Per-Channel Correction: One bias per channel

Expected: Per-element > per-block > per-channel
Storage cost: Per-element > per-block > per-channel
"""

import torch
import json
import numpy as np
from typing import Tuple


def apply_per_block_correction(quantized, reference):
    """Apply per-block bias correction (Phase 25)"""
    corrected = quantized.clone()
    error = reference - quantized
    bias = error.mean(dim=0)  # Mean across batch dimension
    corrected += bias
    return corrected


def apply_per_element_correction(quantized, reference):
    """Apply per-element bias correction"""
    corrected = quantized.clone()
    error = reference - quantized
    bias = error  # No averaging - one bias per element
    corrected += bias
    return corrected


def apply_per_channel_correction(quantized, reference):
    """Apply per-channel bias correction"""
    corrected = quantized.clone()
    error = reference - quantized
    # Average across batch dimension only
    bias = error.mean(dim=0)  # Shape: (hidden_size,)
    corrected += bias
    return corrected


def test_per_element_correction():
    """Test per-element vs per-block vs per-channel correction"""
    
    print("=" * 80)
    print("PHASE 29: PER-ELEMENT CORRECTION ANALYSIS")
    print("=" * 80)
    
    # Configuration
    num_experts = 8
    hidden_size = 4096
    num_calibration_batches = 2
    batch_size = 128
    device = "cpu"
    
    # Generate realistic FP4 data
    print("\nGenerating realistic FP4 quantization data...")
    torch.manual_seed(42)
    
    reference_data = torch.randn(
        num_experts, num_calibration_batches * batch_size, hidden_size,
        device=device
    )
    
    # Simulate FP4 quantization
    quantized_data = reference_data.clone()
    
    for expert_idx in range(num_experts):
        expert_data = reference_data[expert_idx]
        noise_scale = 0.02 * expert_data.abs().mean()
        quantized_data[expert_idx] += torch.randn_like(expert_data) * noise_scale
    
    # Compute baseline MSE
    baseline_mse = torch.mean((reference_data - quantized_data) ** 2).item()
    print(f"Baseline MSE (FP4 quantization error): {baseline_mse:.6f}")
    
    # ========================================================================
    # PER-BLOCK CORRECTION (PHASE 25)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Per-Block Correction (Phase 25)")
    print("-" * 80)
    
    per_block_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        per_block_corrected[expert_idx] = apply_per_block_correction(
            quantized_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1]
        )[0]
    
    per_block_mse = torch.mean((reference_data - per_block_corrected) ** 2).item()
    per_block_improvement = 100 * (baseline_mse - per_block_mse) / baseline_mse
    
    # Storage cost: 1 bias per block (hidden_size values per expert)
    per_block_storage = num_experts * hidden_size * 4  # 4 bytes per float32
    
    print(f"Per-Block MSE: {per_block_mse:.6f}")
    print(f"Per-Block Improvement: {per_block_improvement:.2f}%")
    print(f"Storage Cost: {per_block_storage / 1024:.1f} KB ({per_block_storage} bytes)")
    
    # ========================================================================
    # PER-ELEMENT CORRECTION
    # ========================================================================
    print("\n" + "-" * 80)
    print("Per-Element Correction")
    print("-" * 80)
    
    per_element_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        per_element_corrected[expert_idx] = apply_per_element_correction(
            quantized_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1]
        )[0]
    
    per_element_mse = torch.mean((reference_data - per_element_corrected) ** 2).item()
    per_element_improvement = 100 * (baseline_mse - per_element_mse) / baseline_mse
    
    # Storage cost: 1 bias per element
    num_elements = num_experts * num_calibration_batches * batch_size * hidden_size
    per_element_storage = num_elements * 4  # 4 bytes per float32
    
    print(f"Per-Element MSE: {per_element_mse:.6f}")
    print(f"Per-Element Improvement: {per_element_improvement:.2f}%")
    print(f"Storage Cost: {per_element_storage / (1024**2):.1f} MB ({per_element_storage} bytes)")
    
    # ========================================================================
    # PER-CHANNEL CORRECTION
    # ========================================================================
    print("\n" + "-" * 80)
    print("Per-Channel Correction")
    print("-" * 80)
    
    per_channel_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        per_channel_corrected[expert_idx] = apply_per_channel_correction(
            quantized_data[expert_idx:expert_idx+1],
            reference_data[expert_idx:expert_idx+1]
        )[0]
    
    per_channel_mse = torch.mean((reference_data - per_channel_corrected) ** 2).item()
    per_channel_improvement = 100 * (baseline_mse - per_channel_mse) / baseline_mse
    
    # Storage cost: 1 bias per channel (hidden_size values per expert)
    per_channel_storage = num_experts * hidden_size * 4  # 4 bytes per float32
    
    print(f"Per-Channel MSE: {per_channel_mse:.6f}")
    print(f"Per-Channel Improvement: {per_channel_improvement:.2f}%")
    print(f"Storage Cost: {per_channel_storage / 1024:.1f} KB ({per_channel_storage} bytes)")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("CORRECTION GRANULARITY COMPARISON")
    print("=" * 80)
    
    print(f"\nBaseline MSE:                           {baseline_mse:.6f}")
    print(f"\nPer-Block (Phase 25):")
    print(f"  Improvement: {per_block_improvement:.2f}%")
    print(f"  Storage: {per_block_storage / 1024:.1f} KB")
    print(f"\nPer-Channel:")
    print(f"  Improvement: {per_channel_improvement:.2f}%")
    print(f"  Storage: {per_channel_storage / 1024:.1f} KB")
    print(f"\nPer-Element:")
    print(f"  Improvement: {per_element_improvement:.2f}%")
    print(f"  Storage: {per_element_storage / (1024**2):.1f} MB")
    
    # Calculate additional improvements
    per_channel_additional = per_channel_improvement - per_block_improvement
    per_element_additional = per_element_improvement - per_block_improvement
    
    print(f"\nAdditional Improvements vs Per-Block:")
    print(f"  Per-Channel: {per_channel_additional:.2f}%")
    print(f"  Per-Element: {per_element_additional:.2f}%")
    
    # Storage efficiency
    print(f"\nStorage Efficiency (Improvement per MB):")
    print(f"  Per-Block: {per_block_improvement / (per_block_storage / (1024**2)):.2f}% per MB")
    print(f"  Per-Channel: {per_channel_improvement / (per_channel_storage / (1024**2)):.2f}% per MB")
    print(f"  Per-Element: {per_element_improvement / (per_element_storage / (1024**2)):.2f}% per MB")
    
    # Determine best technique
    techniques = {
        "per_block": {
            "improvement": per_block_improvement,
            "storage": per_block_storage,
        },
        "per_channel": {
            "improvement": per_channel_improvement,
            "storage": per_channel_storage,
        },
        "per_element": {
            "improvement": per_element_improvement,
            "storage": per_element_storage,
        },
    }
    
    # Best by improvement
    best_improvement_technique = max(techniques, key=lambda t: techniques[t]["improvement"])
    
    # Best by efficiency (improvement per MB)
    best_efficiency_technique = max(
        techniques,
        key=lambda t: techniques[t]["improvement"] / (techniques[t]["storage"] / (1024**2))
    )
    
    print(f"\n✓ BEST BY IMPROVEMENT: {best_improvement_technique.upper()}")
    print(f"✓ BEST BY EFFICIENCY: {best_efficiency_technique.upper()}")
    
    # Save results
    results_dict = {
        "baseline_mse": baseline_mse,
        "per_block": {
            "improvement": per_block_improvement,
            "storage_bytes": per_block_storage,
            "storage_kb": per_block_storage / 1024,
        },
        "per_channel": {
            "improvement": per_channel_improvement,
            "storage_bytes": per_channel_storage,
            "storage_kb": per_channel_storage / 1024,
        },
        "per_element": {
            "improvement": per_element_improvement,
            "storage_bytes": per_element_storage,
            "storage_mb": per_element_storage / (1024**2),
        },
        "additional_improvements": {
            "per_channel_vs_per_block": per_channel_additional,
            "per_element_vs_per_block": per_element_additional,
        },
        "best_by_improvement": best_improvement_technique,
        "best_by_efficiency": best_efficiency_technique,
    }
    
    with open("test_phase29_per_element_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: test_phase29_per_element_results.json")
    print("=" * 80)
    
    return results_dict


if __name__ == "__main__":
    test_per_element_correction()

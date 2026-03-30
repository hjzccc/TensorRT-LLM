#!/usr/bin/env python3
"""
Phase 25 Integration Tests: Bias-Only Correction with Phase 1 & Phase 18B

Simple, self-contained test that doesn't depend on complex imports.
Tests whether Phase 25 (Bias-Only) is orthogonal to Phase 1 and Phase 18B.
"""

import torch
import json
import numpy as np


def generate_realistic_fp4_data(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_calibration_batches: int = 2,
    batch_size: int = 128,
    device: str = "cpu"
):
    """Generate realistic FP4 quantization data"""
    torch.manual_seed(42)
    
    # Generate reference (original) data
    reference_data = torch.randn(
        num_experts, num_calibration_batches * batch_size, hidden_size,
        device=device
    )
    
    # Simulate FP4 quantization with realistic error patterns
    quantized_data = reference_data.clone()
    
    # Add realistic quantization noise (per-expert variation)
    for expert_idx in range(num_experts):
        expert_data = reference_data[expert_idx]
        # Quantization error scales with activation magnitude
        noise_scale = 0.02 * expert_data.abs().mean()
        quantized_data[expert_idx] += torch.randn_like(expert_data) * noise_scale
    
    return reference_data, quantized_data


def apply_phase25_bias_correction(quantized, reference):
    """Apply Phase 25: Bias-Only Correction"""
    num_experts = quantized.shape[0]
    corrected = quantized.clone()
    
    for expert_idx in range(num_experts):
        # Compute mean error per expert
        error = reference[expert_idx] - quantized[expert_idx]
        bias = error.mean(dim=0)  # Mean across batch dimension
        corrected[expert_idx] += bias
    
    return corrected


def apply_phase1_affine_correction(quantized, reference):
    """Apply Phase 1: Full Affine Correction (α*x + β)"""
    num_experts = quantized.shape[0]
    corrected = quantized.clone()
    
    for expert_idx in range(num_experts):
        q = quantized[expert_idx]
        r = reference[expert_idx]
        
        # Compute optimal affine parameters: α, β
        # Minimize: ||α*q + β - r||^2
        # Solution: α = cov(q,r) / var(q), β = mean(r) - α*mean(q)
        
        q_mean = q.mean()
        r_mean = r.mean()
        
        q_centered = q - q_mean
        r_centered = r - r_mean
        
        cov = (q_centered * r_centered).mean()
        var = (q_centered ** 2).mean()
        
        alpha = cov / (var + 1e-8)
        beta = r_mean - alpha * q_mean
        
        corrected[expert_idx] = alpha * corrected[expert_idx] + beta
    
    return corrected


def apply_phase18b_fisher_correction(quantized, reference):
    """Apply Phase 18B: Block-Diagonal Fisher Codebook Selection (simplified)"""
    # For this test, we simulate Phase 18B as a small per-expert scaling
    # In reality, it would select better codebooks
    num_experts = quantized.shape[0]
    corrected = quantized.clone()
    
    for expert_idx in range(num_experts):
        q = quantized[expert_idx]
        r = reference[expert_idx]
        
        # Compute Fisher-weighted scaling
        # Simplified: scale by ratio of variances
        q_var = q.var()
        r_var = r.var()
        
        scale = torch.sqrt(r_var / (q_var + 1e-8))
        corrected[expert_idx] = corrected[expert_idx] * scale
    
    return corrected


def test_phase25_integration():
    """Test Phase 25 integration with Phase 1 and Phase 18B"""
    
    print("=" * 80)
    print("PHASE 25 INTEGRATION TESTS: Bias-Only with Phase 1 & Phase 18B")
    print("=" * 80)
    
    # Configuration
    num_experts = 8
    hidden_size = 4096
    num_calibration_batches = 2
    batch_size = 128
    device = "cpu"
    
    # Generate data
    print("\nGenerating realistic FP4 quantization data...")
    reference_data, quantized_data = generate_realistic_fp4_data(
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_calibration_batches=num_calibration_batches,
        batch_size=batch_size,
        device=device
    )
    
    # Compute baseline MSE
    baseline_mse = torch.mean((reference_data - quantized_data) ** 2).item()
    print(f"Baseline MSE (FP4 quantization error): {baseline_mse:.6f}")
    
    # ========================================================================
    # PHASE 25: BIAS-ONLY CORRECTION
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 25: Bias-Only Correction")
    print("-" * 80)
    
    phase25_corrected = apply_phase25_bias_correction(quantized_data, reference_data)
    phase25_mse = torch.mean((reference_data - phase25_corrected) ** 2).item()
    phase25_improvement = 100 * (baseline_mse - phase25_mse) / baseline_mse
    
    print(f"Phase 25 MSE: {phase25_mse:.6f}")
    print(f"Phase 25 Improvement: {phase25_improvement:.2f}%")
    
    # ========================================================================
    # PHASE 1: FULL AFFINE CORRECTION
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 1: Full Affine Correction")
    print("-" * 80)
    
    phase1_corrected = apply_phase1_affine_correction(quantized_data, reference_data)
    phase1_mse = torch.mean((reference_data - phase1_corrected) ** 2).item()
    phase1_improvement = 100 * (baseline_mse - phase1_mse) / baseline_mse
    
    print(f"Phase 1 MSE: {phase1_mse:.6f}")
    print(f"Phase 1 Improvement: {phase1_improvement:.2f}%")
    
    # ========================================================================
    # PHASE 25 + PHASE 1: COMBINED
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 25 + Phase 1: Combined Integration")
    print("-" * 80)
    
    # Apply Phase 1 first, then Phase 25
    phase25_phase1_corrected = apply_phase1_affine_correction(quantized_data, reference_data)
    phase25_phase1_corrected = apply_phase25_bias_correction(phase25_phase1_corrected, reference_data)
    
    phase25_phase1_mse = torch.mean((reference_data - phase25_phase1_corrected) ** 2).item()
    phase25_phase1_improvement = 100 * (baseline_mse - phase25_phase1_mse) / baseline_mse
    phase25_phase1_expected = phase25_improvement + phase1_improvement
    phase25_phase1_orthogonality = phase25_phase1_improvement / phase25_phase1_expected if phase25_phase1_expected > 0 else 0
    
    print(f"Phase 25 + Phase 1 MSE: {phase25_phase1_mse:.6f}")
    print(f"Phase 25 + Phase 1 Improvement: {phase25_phase1_improvement:.2f}%")
    print(f"Expected Additive: {phase25_phase1_expected:.2f}%")
    print(f"Orthogonality Ratio: {phase25_phase1_orthogonality:.4f}")
    
    if phase25_phase1_orthogonality > 0.95:
        print("✓ VERDICT: Phase 25 and Phase 1 are ORTHOGONAL")
    else:
        print("⚠ VERDICT: Phase 25 and Phase 1 have INTERACTION")
    
    # ========================================================================
    # PHASE 18B: BLOCK-DIAGONAL FISHER CODEBOOK SELECTION
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 18B: Block-Diagonal Fisher Codebook Selection")
    print("-" * 80)
    
    phase18b_corrected = apply_phase18b_fisher_correction(quantized_data, reference_data)
    phase18b_mse = torch.mean((reference_data - phase18b_corrected) ** 2).item()
    phase18b_improvement = 100 * (baseline_mse - phase18b_mse) / baseline_mse
    
    print(f"Phase 18B MSE: {phase18b_mse:.6f}")
    print(f"Phase 18B Improvement: {phase18b_improvement:.2f}%")
    
    # ========================================================================
    # PHASE 25 + PHASE 18B: COMBINED
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 25 + Phase 18B: Combined Integration")
    print("-" * 80)
    
    # Apply Phase 18B first, then Phase 25
    phase25_phase18b_corrected = apply_phase18b_fisher_correction(quantized_data, reference_data)
    phase25_phase18b_corrected = apply_phase25_bias_correction(phase25_phase18b_corrected, reference_data)
    
    phase25_phase18b_mse = torch.mean((reference_data - phase25_phase18b_corrected) ** 2).item()
    phase25_phase18b_improvement = 100 * (baseline_mse - phase25_phase18b_mse) / baseline_mse
    phase25_phase18b_expected = phase25_improvement + phase18b_improvement
    phase25_phase18b_orthogonality = phase25_phase18b_improvement / phase25_phase18b_expected if phase25_phase18b_expected > 0 else 0
    
    print(f"Phase 25 + Phase 18B MSE: {phase25_phase18b_mse:.6f}")
    print(f"Phase 25 + Phase 18B Improvement: {phase25_phase18b_improvement:.2f}%")
    print(f"Expected Additive: {phase25_phase18b_expected:.2f}%")
    print(f"Orthogonality Ratio: {phase25_phase18b_orthogonality:.4f}")
    
    if phase25_phase18b_orthogonality > 0.95:
        print("✓ VERDICT: Phase 25 and Phase 18B are ORTHOGONAL")
    else:
        print("⚠ VERDICT: Phase 25 and Phase 18B have INTERACTION")
    
    # ========================================================================
    # PHASE 1 + PHASE 18B: COMBINED (BASELINE)
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 1 + Phase 18B: Combined Integration (Baseline)")
    print("-" * 80)
    
    # Apply Phase 18B first, then Phase 1
    phase1_phase18b_corrected = apply_phase18b_fisher_correction(quantized_data, reference_data)
    phase1_phase18b_corrected = apply_phase1_affine_correction(phase1_phase18b_corrected, reference_data)
    
    phase1_phase18b_mse = torch.mean((reference_data - phase1_phase18b_corrected) ** 2).item()
    phase1_phase18b_improvement = 100 * (baseline_mse - phase1_phase18b_mse) / baseline_mse
    phase1_phase18b_expected = phase1_improvement + phase18b_improvement
    phase1_phase18b_orthogonality = phase1_phase18b_improvement / phase1_phase18b_expected if phase1_phase18b_expected > 0 else 0
    
    print(f"Phase 1 + Phase 18B MSE: {phase1_phase18b_mse:.6f}")
    print(f"Phase 1 + Phase 18B Improvement: {phase1_phase18b_improvement:.2f}%")
    print(f"Expected Additive: {phase1_phase18b_expected:.2f}%")
    print(f"Orthogonality Ratio: {phase1_phase18b_orthogonality:.4f}")
    
    if phase1_phase18b_orthogonality > 0.95:
        print("✓ VERDICT: Phase 1 and Phase 18B are ORTHOGONAL")
    else:
        print("⚠ VERDICT: Phase 1 and Phase 18B have INTERACTION")
    
    # ========================================================================
    # PHASE 25 + PHASE 1 + PHASE 18B: TRIPLE COMBINATION
    # ========================================================================
    print("\n" + "-" * 80)
    print("Phase 25 + Phase 1 + Phase 18B: Triple Combination")
    print("-" * 80)
    
    # Apply Phase 18B first, then Phase 1, then Phase 25
    phase25_phase1_phase18b_corrected = apply_phase18b_fisher_correction(quantized_data, reference_data)
    phase25_phase1_phase18b_corrected = apply_phase1_affine_correction(phase25_phase1_phase18b_corrected, reference_data)
    phase25_phase1_phase18b_corrected = apply_phase25_bias_correction(phase25_phase1_phase18b_corrected, reference_data)
    
    phase25_phase1_phase18b_mse = torch.mean((reference_data - phase25_phase1_phase18b_corrected) ** 2).item()
    phase25_phase1_phase18b_improvement = 100 * (baseline_mse - phase25_phase1_phase18b_mse) / baseline_mse
    phase25_phase1_phase18b_expected = phase25_improvement + phase1_improvement + phase18b_improvement
    phase25_phase1_phase18b_orthogonality = phase25_phase1_phase18b_improvement / phase25_phase1_phase18b_expected if phase25_phase1_phase18b_expected > 0 else 0
    
    print(f"Phase 25 + Phase 1 + Phase 18B MSE: {phase25_phase1_phase18b_mse:.6f}")
    print(f"Phase 25 + Phase 1 + Phase 18B Improvement: {phase25_phase1_phase18b_improvement:.2f}%")
    print(f"Expected Additive: {phase25_phase1_phase18b_expected:.2f}%")
    print(f"Orthogonality Ratio: {phase25_phase1_phase18b_orthogonality:.4f}")
    
    if phase25_phase1_phase18b_orthogonality > 0.95:
        print("✓ VERDICT: All three techniques are ORTHOGONAL")
    else:
        print("⚠ VERDICT: Techniques have INTERACTION")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("INTEGRATION ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nBaseline MSE:                           {baseline_mse:.6f}")
    print(f"\nPhase 25 Improvement:                   {phase25_improvement:.2f}%")
    print(f"Phase 1 Improvement:                    {phase1_improvement:.2f}%")
    print(f"Phase 18B Improvement:                  {phase18b_improvement:.2f}%")
    print(f"\nPhase 25 + Phase 1:")
    print(f"  Actual:   {phase25_phase1_improvement:.2f}%")
    print(f"  Expected: {phase25_phase1_expected:.2f}%")
    print(f"  Orthogonality: {phase25_phase1_orthogonality:.4f}")
    print(f"\nPhase 25 + Phase 18B:")
    print(f"  Actual:   {phase25_phase18b_improvement:.2f}%")
    print(f"  Expected: {phase25_phase18b_expected:.2f}%")
    print(f"  Orthogonality: {phase25_phase18b_orthogonality:.4f}")
    print(f"\nPhase 1 + Phase 18B:")
    print(f"  Actual:   {phase1_phase18b_improvement:.2f}%")
    print(f"  Expected: {phase1_phase18b_expected:.2f}%")
    print(f"  Orthogonality: {phase1_phase18b_orthogonality:.4f}")
    print(f"\nPhase 25 + Phase 1 + Phase 18B:")
    print(f"  Actual:   {phase25_phase1_phase18b_improvement:.2f}%")
    print(f"  Expected: {phase25_phase1_phase18b_expected:.2f}%")
    print(f"  Orthogonality: {phase25_phase1_phase18b_orthogonality:.4f}")
    
    # Save results
    results_dict = {
        "baseline_mse": baseline_mse,
        "phase25_improvement": phase25_improvement,
        "phase1_improvement": phase1_improvement,
        "phase18b_improvement": phase18b_improvement,
        "phase25_phase1": {
            "improvement": phase25_phase1_improvement,
            "expected": phase25_phase1_expected,
            "orthogonality": phase25_phase1_orthogonality,
        },
        "phase25_phase18b": {
            "improvement": phase25_phase18b_improvement,
            "expected": phase25_phase18b_expected,
            "orthogonality": phase25_phase18b_orthogonality,
        },
        "phase1_phase18b": {
            "improvement": phase1_phase18b_improvement,
            "expected": phase1_phase18b_expected,
            "orthogonality": phase1_phase18b_orthogonality,
        },
        "phase25_phase1_phase18b": {
            "improvement": phase25_phase1_phase18b_improvement,
            "expected": phase25_phase1_phase18b_expected,
            "orthogonality": phase25_phase1_phase18b_orthogonality,
        },
    }
    
    with open("test_phase25_simple_integration_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: test_phase25_simple_integration_results.json")
    print("=" * 80)
    
    return results_dict


if __name__ == "__main__":
    test_phase25_integration()

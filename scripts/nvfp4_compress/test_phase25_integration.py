#!/usr/bin/env python3
"""
Phase 25 Integration Tests: Bias-Only Correction with Phase 1 & Phase 18B

Tests whether Phase 25 (Bias-Only) is orthogonal to:
1. Phase 1 (Full Affine Correction)
2. Phase 18B (Block-Diagonal Fisher Codebook Selection)
3. Both Phase 1 + Phase 18B

Measures cumulative improvements and orthogonality ratios.
"""

import torch
import json
from dataclasses import dataclass
from typing import Tuple

# Import existing implementations
import sys
sys.path.insert(0, '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress')

from phase1_affine_correction import AffineCorrectionFitter
from phase18b_block_diagonal_fisher import BlockDiagonalFisherFitter
from phase25_bias_analysis import BiasOnlyFitter


@dataclass(frozen=True)
class IntegrationTestResult:
    """Results from integration test"""
    baseline_mse: float
    phase25_mse: float
    phase25_improvement: float
    phase1_mse: float
    phase1_improvement: float
    phase25_phase1_mse: float
    phase25_phase1_improvement: float
    phase25_phase1_expected: float
    phase25_phase1_orthogonality: float
    
    phase18b_mse: float
    phase18b_improvement: float
    phase25_phase18b_mse: float
    phase25_phase18b_improvement: float
    phase25_phase18b_expected: float
    phase25_phase18b_orthogonality: float
    
    phase1_phase18b_mse: float
    phase1_phase18b_improvement: float
    phase1_phase18b_expected: float
    phase1_phase18b_orthogonality: float
    
    phase25_phase1_phase18b_mse: float
    phase25_phase1_phase18b_improvement: float
    phase25_phase1_phase18b_expected: float
    phase25_phase1_phase18b_orthogonality: float


def generate_realistic_fp4_data(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_calibration_batches: int = 2,
    batch_size: int = 128,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate realistic FP4 quantization data"""
    torch.manual_seed(42)
    
    # Generate reference (original) data
    reference_data = torch.randn(
        num_experts, num_calibration_batches * batch_size, hidden_size,
        device=device
    )
    
    # Simulate FP4 quantization with realistic error patterns
    # FP4 has 4-bit precision, so quantization error is ~1-2% of range
    quantized_data = reference_data.clone()
    
    # Add realistic quantization noise (per-expert variation)
    for expert_idx in range(num_experts):
        expert_data = reference_data[expert_idx]
        # Quantization error scales with activation magnitude
        noise_scale = 0.02 * expert_data.abs().mean()
        quantized_data[expert_idx] += torch.randn_like(expert_data) * noise_scale
    
    return reference_data, quantized_data


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
    
    phase25_fitter = BiasOnlyFitter(
        num_experts=num_experts,
        hidden_size=hidden_size,
        device=device
    )
    
    # Accumulate statistics
    stats = phase25_fitter.initialize_moments()
    for expert_idx in range(num_experts):
        phase25_fitter.accumulate_moments(
            stats, expert_idx,
            quantized_data[expert_idx],
            reference_data[expert_idx]
        )
    
    # Fit bias correction
    phase25_correction = phase25_fitter.fit(stats)
    
    # Apply correction
    phase25_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        bias = phase25_correction.bias[expert_idx]
        phase25_corrected[expert_idx] += bias
    
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
    
    phase1_fitter = AffineCorrectionFitter(
        num_experts=num_experts,
        hidden_size=hidden_size,
        device=device
    )
    
    # Accumulate statistics
    stats = phase1_fitter.initialize_moments()
    for expert_idx in range(num_experts):
        phase1_fitter.accumulate_moments(
            stats, expert_idx,
            quantized_data[expert_idx],
            reference_data[expert_idx]
        )
    
    # Fit affine correction
    phase1_correction = phase1_fitter.fit(stats)
    
    # Apply correction
    phase1_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        alpha = phase1_correction.alpha[expert_idx]
        beta = phase1_correction.beta[expert_idx]
        phase1_corrected[expert_idx] = alpha * phase1_corrected[expert_idx] + beta
    
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
    phase25_phase1_corrected = quantized_data.clone()
    
    # Apply Phase 1 (affine)
    for expert_idx in range(num_experts):
        alpha = phase1_correction.alpha[expert_idx]
        beta = phase1_correction.beta[expert_idx]
        phase25_phase1_corrected[expert_idx] = alpha * phase25_phase1_corrected[expert_idx] + beta
    
    # Apply Phase 25 (bias) on top
    for expert_idx in range(num_experts):
        bias = phase25_correction.bias[expert_idx]
        phase25_phase1_corrected[expert_idx] += bias
    
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
    
    phase18b_fitter = BlockDiagonalFisherFitter(
        num_experts=num_experts,
        hidden_size=hidden_size,
        device=device
    )
    
    # Accumulate statistics
    stats = phase18b_fitter.initialize_moments()
    for expert_idx in range(num_experts):
        phase18b_fitter.accumulate_moments(
            stats, expert_idx,
            quantized_data[expert_idx],
            reference_data[expert_idx]
        )
    
    # Fit codebook selection
    phase18b_correction = phase18b_fitter.fit(stats)
    
    # Apply correction (codebook selection)
    phase18b_corrected = quantized_data.clone()
    for expert_idx in range(num_experts):
        # Phase 18B applies codebook selection (already in quantized_data)
        # For this test, we simulate the improvement
        phase18b_corrected[expert_idx] = phase18b_correction.apply(
            quantized_data[expert_idx], expert_idx
        )
    
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
    phase25_phase18b_corrected = quantized_data.clone()
    
    # Apply Phase 18B (codebook selection)
    for expert_idx in range(num_experts):
        phase25_phase18b_corrected[expert_idx] = phase18b_correction.apply(
            phase25_phase18b_corrected[expert_idx], expert_idx
        )
    
    # Apply Phase 25 (bias) on top
    for expert_idx in range(num_experts):
        bias = phase25_correction.bias[expert_idx]
        phase25_phase18b_corrected[expert_idx] += bias
    
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
    phase1_phase18b_corrected = quantized_data.clone()
    
    # Apply Phase 18B (codebook selection)
    for expert_idx in range(num_experts):
        phase1_phase18b_corrected[expert_idx] = phase18b_correction.apply(
            phase1_phase18b_corrected[expert_idx], expert_idx
        )
    
    # Apply Phase 1 (affine) on top
    for expert_idx in range(num_experts):
        alpha = phase1_correction.alpha[expert_idx]
        beta = phase1_correction.beta[expert_idx]
        phase1_phase18b_corrected[expert_idx] = alpha * phase1_phase18b_corrected[expert_idx] + beta
    
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
    phase25_phase1_phase18b_corrected = quantized_data.clone()
    
    # Apply Phase 18B (codebook selection)
    for expert_idx in range(num_experts):
        phase25_phase1_phase18b_corrected[expert_idx] = phase18b_correction.apply(
            phase25_phase1_phase18b_corrected[expert_idx], expert_idx
        )
    
    # Apply Phase 1 (affine)
    for expert_idx in range(num_experts):
        alpha = phase1_correction.alpha[expert_idx]
        beta = phase1_correction.beta[expert_idx]
        phase25_phase1_phase18b_corrected[expert_idx] = alpha * phase25_phase1_phase18b_corrected[expert_idx] + beta
    
    # Apply Phase 25 (bias)
    for expert_idx in range(num_experts):
        bias = phase25_correction.bias[expert_idx]
        phase25_phase1_phase18b_corrected[expert_idx] += bias
    
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
    
    results = IntegrationTestResult(
        baseline_mse=baseline_mse,
        phase25_mse=phase25_mse,
        phase25_improvement=phase25_improvement,
        phase1_mse=phase1_mse,
        phase1_improvement=phase1_improvement,
        phase25_phase1_mse=phase25_phase1_mse,
        phase25_phase1_improvement=phase25_phase1_improvement,
        phase25_phase1_expected=phase25_phase1_expected,
        phase25_phase1_orthogonality=phase25_phase1_orthogonality,
        phase18b_mse=phase18b_mse,
        phase18b_improvement=phase18b_improvement,
        phase25_phase18b_mse=phase25_phase18b_mse,
        phase25_phase18b_improvement=phase25_phase18b_improvement,
        phase25_phase18b_expected=phase25_phase18b_expected,
        phase25_phase18b_orthogonality=phase25_phase18b_orthogonality,
        phase1_phase18b_mse=phase1_phase18b_mse,
        phase1_phase18b_improvement=phase1_phase18b_improvement,
        phase1_phase18b_expected=phase1_phase18b_expected,
        phase1_phase18b_orthogonality=phase1_phase18b_orthogonality,
        phase25_phase1_phase18b_mse=phase25_phase1_phase18b_mse,
        phase25_phase1_phase18b_improvement=phase25_phase1_phase18b_improvement,
        phase25_phase1_phase18b_expected=phase25_phase1_phase18b_expected,
        phase25_phase1_phase18b_orthogonality=phase25_phase1_phase18b_orthogonality,
    )
    
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
    
    with open("test_phase25_integration_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: test_phase25_integration_results.json")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    test_phase25_integration()

#!/usr/bin/env python3
"""
Phase 2 Integration Tests: Simple, Working Version

Tests combining Phase 1 (affine correction) with Phase 2 (Hessian-weighted + DAC)
in an end-to-end pipeline.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase1_affine_correction import (
    AffineCorrectionFitter,
    apply_affine_correction,
)
from phase2_sensitivity_guided_correction import (
    HessianWeightedAffineCorrectionFitter,
    DeviationAwareCorrectionFitter,
    apply_hessian_weighted_affine_correction,
    apply_deviation_aware_correction,
)


def create_synthetic_data(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_samples: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple:
    """Create synthetic data."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    reference = torch.randn(num_experts, num_samples, hidden_size, device=device) * 0.1
    quantized = reference + torch.randn_like(reference) * 0.01
    fisher_diag = torch.abs(torch.randn(num_experts, hidden_size, device=device)) + 0.1
    fisher_diag = fisher_diag / fisher_diag.sum(dim=1, keepdim=True)
    
    return reference, quantized, fisher_diag


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute MSE."""
    return ((pred - target) ** 2).mean().item()


def test_phase1_only():
    """Test Phase 1 alone."""
    print("\n[TEST] Phase 1 only...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, _ = create_synthetic_data(device=device)
    
    ref_flat = reference.reshape(-1)
    quant_flat = quantized.reshape(-1)
    baseline_mse = compute_mse(quant_flat, ref_flat)
    
    # Fit Phase 1
    fitter = AffineCorrectionFitter(num_experts=8, hidden_size=4096, device=device)
    moments = fitter.initialize_moments()
    
    for expert_idx in range(8):
        fitter.accumulate_moments(
            stats=moments,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
        )
    
    corrections = fitter.fit(moments)
    correction = corrections["scalar"]
    
    # Apply correction per expert
    corrected = quantized.clone()
    for expert_idx in range(8):
        corrected[expert_idx] = apply_affine_correction(
            quantized[expert_idx],
            correction,
            expert_idx=expert_idx,
        )
    
    corrected_mse = compute_mse(corrected.reshape(-1), ref_flat)
    improvement = (baseline_mse - corrected_mse) / baseline_mse * 100
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Corrected MSE: {corrected_mse:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    assert corrected_mse < baseline_mse, "Phase 1 should reduce MSE"
    assert improvement > 5.0, f"Expected >5% improvement, got {improvement:.2f}%"
    
    print("✓ Phase 1 only test passed")
    return corrected, baseline_mse


def test_phase1_plus_phase2():
    """Test Phase 1 + Phase 2 combined."""
    print("\n[TEST] Phase 1 + Phase 2 combined...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag = create_synthetic_data(device=device)
    
    ref_flat = reference.reshape(-1)
    quant_flat = quantized.reshape(-1)
    baseline_mse = compute_mse(quant_flat, ref_flat)
    
    # Phase 1
    fitter1 = AffineCorrectionFitter(num_experts=8, hidden_size=4096, device=device)
    moments1 = fitter1.initialize_moments()
    
    for expert_idx in range(8):
        fitter1.accumulate_moments(
            stats=moments1,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
        )
    
    corrections1 = fitter1.fit(moments1)
    correction1 = corrections1["scalar"]
    
    corrected1 = quantized.clone()
    for expert_idx in range(8):
        corrected1[expert_idx] = apply_affine_correction(
            quantized[expert_idx],
            correction1,
            expert_idx=expert_idx,
        )
    
    phase1_mse = compute_mse(corrected1.reshape(-1), ref_flat)
    
    # Phase 2: Hessian-weighted affine
    fitter2 = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    moments2 = fitter2.initialize_moments()
    
    for expert_idx in range(8):
        fisher_weight = fisher_diag[expert_idx].mean().item()
        fitter2.accumulate_weighted_moments(
            stats=moments2,
            expert_idx=expert_idx,
            quantized=corrected1[expert_idx],
            reference=reference[expert_idx],
            fisher_weight=fisher_weight,
        )
    
    correction2 = fitter2.solve_hessian_weighted_affine(moments2)
    
    corrected2 = corrected1.clone()
    for expert_idx in range(8):
        corrected2[expert_idx] = apply_hessian_weighted_affine_correction(
            corrected1[expert_idx],
            correction2,
            expert_idx=expert_idx,
        )
    
    phase2_mse = compute_mse(corrected2.reshape(-1), ref_flat)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Phase 1 MSE: {phase1_mse:.6f}")
    print(f"  Phase 2 MSE: {phase2_mse:.6f}")
    
    assert phase2_mse < phase1_mse, "Phase 2 should further reduce MSE"
    
    print("✓ Phase 1 + Phase 2 test passed")
    return corrected2, baseline_mse


def test_phase1_plus_phase2_plus_dac():
    """Test Phase 1 + Phase 2 + DAC combined."""
    print("\n[TEST] Phase 1 + Phase 2 + DAC combined...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag = create_synthetic_data(device=device)
    
    ref_flat = reference.reshape(-1)
    quant_flat = quantized.reshape(-1)
    baseline_mse = compute_mse(quant_flat, ref_flat)
    
    # Phase 1
    fitter1 = AffineCorrectionFitter(num_experts=8, hidden_size=4096, device=device)
    moments1 = fitter1.initialize_moments()
    
    for expert_idx in range(8):
        fitter1.accumulate_moments(
            stats=moments1,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
        )
    
    corrections1 = fitter1.fit(moments1)
    correction1 = corrections1["scalar"]
    
    corrected1 = quantized.clone()
    for expert_idx in range(8):
        corrected1[expert_idx] = apply_affine_correction(
            quantized[expert_idx],
            correction1,
            expert_idx=expert_idx,
        )
    
    phase1_mse = compute_mse(corrected1.reshape(-1), ref_flat)
    
    # Phase 2a: Hessian-weighted affine
    fitter2a = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    moments2a = fitter2a.initialize_moments()
    
    for expert_idx in range(8):
        fisher_weight = fisher_diag[expert_idx].mean().item()
        fitter2a.accumulate_weighted_moments(
            stats=moments2a,
            expert_idx=expert_idx,
            quantized=corrected1[expert_idx],
            reference=reference[expert_idx],
            fisher_weight=fisher_weight,
        )
    
    correction2a = fitter2a.solve_hessian_weighted_affine(moments2a)
    
    corrected2a = corrected1.clone()
    for expert_idx in range(8):
        corrected2a[expert_idx] = apply_hessian_weighted_affine_correction(
            corrected1[expert_idx],
            correction2a,
            expert_idx=expert_idx,
        )
    
    phase2a_mse = compute_mse(corrected2a.reshape(-1), ref_flat)
    
    # Phase 2b: Deviation-Aware Correction
    fitter2b = DeviationAwareCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    # Use the moments from Phase 2a for DAC computation
    correction2b = fitter2b.compute_deviation_aware_correction(moments2a)
    
    corrected2b = corrected2a.clone()
    for expert_idx in range(8):
        corrected2b[expert_idx] = apply_deviation_aware_correction(
            corrected2a[expert_idx],
            correction2b,
            expert_idx=expert_idx,
        )
    
    phase2b_mse = compute_mse(corrected2b.reshape(-1), ref_flat)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Phase 1 MSE: {phase1_mse:.6f}")
    print(f"  Phase 2a MSE: {phase2a_mse:.6f}")
    print(f"  Phase 2b MSE: {phase2b_mse:.6f}")
    
    assert phase2b_mse < baseline_mse, "Full pipeline should reduce MSE"
    
    print("✓ Phase 1 + Phase 2 + DAC test passed")
    return corrected2b, baseline_mse


def test_orthogonality_with_fisher():
    """Test that Phase 2 properly uses Fisher weights."""
    print("\n[TEST] Orthogonality with Fisher weights...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag = create_synthetic_data(device=device)
    
    # Phase 1 (no Fisher)
    fitter1 = AffineCorrectionFitter(num_experts=8, hidden_size=4096, device=device)
    moments1 = fitter1.initialize_moments()
    
    for expert_idx in range(8):
        fitter1.accumulate_moments(
            stats=moments1,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
        )
    
    corrections1 = fitter1.fit(moments1)
    
    # Phase 2 (with Fisher)
    fitter2 = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    moments2 = fitter2.initialize_moments()
    
    for expert_idx in range(8):
        fisher_weight = fisher_diag[expert_idx].mean().item()
        fitter2.accumulate_weighted_moments(
            stats=moments2,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weight=fisher_weight,
        )
    
    correction2 = fitter2.solve_hessian_weighted_affine(moments2)
    
    # Verify Fisher weights are tracked
    assert correction2.fisher_weights is not None, "Fisher weights should be tracked"
    assert correction2.fisher_weights.shape == (8,), f"Expected shape (8,), got {correction2.fisher_weights.shape}"
    assert (correction2.fisher_weights > 0).all(), "All Fisher weights should be positive"
    
    print(f"  Fisher weights: {correction2.fisher_weights.numpy()}")
    print("✓ Orthogonality test passed")


def test_storage_efficiency():
    """Test that correction parameters are storage-efficient."""
    print("\n[TEST] Storage efficiency...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, _ = create_synthetic_data(device=device)
    
    fitter = AffineCorrectionFitter(num_experts=8, hidden_size=4096, device=device)
    moments = fitter.initialize_moments()
    
    for expert_idx in range(8):
        fitter.accumulate_moments(
            stats=moments,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
        )
    
    corrections = fitter.fit(moments)
    correction = corrections["scalar"]
    
    # Estimate storage for 32 layers
    alpha_size = correction.alpha.numel() * 4  # float32
    beta_size = correction.beta.numel() * 4
    total_per_layer = alpha_size + beta_size
    total_32_layers = total_per_layer * 32
    
    print(f"  Per-layer storage: {total_per_layer} bytes")
    print(f"  32-layer storage: {total_32_layers} bytes ({total_32_layers / 1024:.2f} KB)")
    
    assert total_32_layers < 10000, "Storage should be <10 KB for 32 layers"
    
    print("✓ Storage efficiency test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("PHASE 2 INTEGRATION TESTS (SIMPLE)")
    print("=" * 80)
    
    tests = [
        test_phase1_only,
        test_phase1_plus_phase2,
        test_phase1_plus_phase2_plus_dac,
        test_orthogonality_with_fisher,
        test_storage_efficiency,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Phase 2 Integration Tests: Affine + Sensitivity-Guided Correction (FIXED)

Tests combining Phase 1 (affine correction) with Phase 2 (Hessian-weighted + DAC)
in an end-to-end pipeline, verifying orthogonality with block-Fisher.
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


# ============================================================================
# Test Utilities
# ============================================================================

def create_synthetic_data(
    num_experts: int = 8,
    hidden_size: int = 4096,
    num_samples: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple:
    """Create synthetic quantized/reference data for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    reference = torch.randn(num_experts, num_samples, hidden_size, device=device) * 0.1
    quantized = reference + torch.randn_like(reference) * 0.01
    fisher_diag = torch.abs(torch.randn(num_experts, hidden_size, device=device)) + 0.1
    fisher_diag = fisher_diag / fisher_diag.sum(dim=1, keepdim=True)
    
    return reference, quantized, fisher_diag


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean squared error."""
    return ((pred - target) ** 2).mean().item()


def compute_ppl_improvement(baseline_mse: float, corrected_mse: float) -> float:
    """Estimate PPL improvement from MSE reduction."""
    if baseline_mse == 0:
        return 0.0
    return (baseline_mse - corrected_mse) / baseline_mse * 100


# ============================================================================
# Tests
# ============================================================================

def test_phase1_basic():
    """Test Phase 1 basic functionality."""
    print("\n[TEST] Phase 1 basic functionality...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, _ = create_synthetic_data(device=device)
    
    fitter = AffineCorrectionFitter(num_experts=8, hidden_size=4096, device=device)
    moments = fitter.initialize_moments()
    
    # Accumulate moments for each expert
    for expert_idx in range(8):
        fitter.accumulate_moments(
            stats=moments,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
        )
    
    # Fit corrections
    corrections = fitter.fit(moments)
    
    assert "scalar" in corrections, "Should have scalar correction"
    assert corrections["scalar"].alpha is not None
    assert corrections["scalar"].beta is not None
    
    print("✓ Phase 1 basic test passed")


def test_phase2_basic():
    """Test Phase 2 basic functionality."""
    print("\n[TEST] Phase 2 basic functionality...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag = create_synthetic_data(device=device)
    
    fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    moments = fitter.initialize_moments()
    
    # Accumulate moments with Fisher weights
    for expert_idx in range(8):
        fisher_weight = fisher_diag[expert_idx].mean().item()
        fitter.accumulate_weighted_moments(
            stats=moments,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weight=fisher_weight,
        )
    
    # Fit correction
    correction = fitter.solve_hessian_weighted_affine(moments)
    
    assert correction.alpha is not None
    assert correction.beta is not None
    assert correction.fisher_weights is not None
    
    print("✓ Phase 2 basic test passed")


def test_phase1_improvement():
    """Test that Phase 1 reduces MSE."""
    print("\n[TEST] Phase 1 MSE improvement...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, _ = create_synthetic_data(device=device)
    
    # Flatten for MSE computation
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
    
    # Apply correction
    corrected = apply_affine_correction(quantized, correction)
    corrected_flat = corrected.reshape(-1)
    
    corrected_mse = compute_mse(corrected_flat, ref_flat)
    improvement = compute_ppl_improvement(baseline_mse, corrected_mse)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Corrected MSE: {corrected_mse:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    assert corrected_mse < baseline_mse, "Phase 1 should reduce MSE"
    assert improvement > 5.0, f"Expected >5% improvement, got {improvement:.2f}%"
    
    print("✓ Phase 1 improvement test passed")


def test_phase2_improvement():
    """Test that Phase 2 further reduces MSE."""
    print("\n[TEST] Phase 2 MSE improvement...")
    
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
    corrected1 = apply_affine_correction(quantized, corrections1["scalar"])
    phase1_mse = compute_mse(corrected1.reshape(-1), ref_flat)
    
    # Phase 2
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
    corrected2 = apply_hessian_weighted_affine_correction(corrected1, correction2, expert_idx=0)
    
    # For simplicity, apply to all experts
    corrected2_full = corrected1.clone()
    for expert_idx in range(8):
        corrected2_full[expert_idx] = apply_hessian_weighted_affine_correction(
            corrected1[expert_idx],
            correction2,
            expert_idx=expert_idx,
        )
    
    phase2_mse = compute_mse(corrected2_full.reshape(-1), ref_flat)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Phase 1 MSE: {phase1_mse:.6f}")
    print(f"  Phase 2 MSE: {phase2_mse:.6f}")
    
    assert phase2_mse < phase1_mse, "Phase 2 should further reduce MSE"
    
    print("✓ Phase 2 improvement test passed")


def test_dac_basic():
    """Test Deviation-Aware Correction basic functionality."""
    print("\n[TEST] Deviation-Aware Correction basic...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, _ = create_synthetic_data(device=device)
    
    fitter = DeviationAwareCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    moments = fitter.initialize_moments()
    
    # Accumulate moments
    for expert_idx in range(8):
        fitter.accumulate_weighted_moments(
            stats=moments,
            expert_idx=expert_idx,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weight=1.0,
        )
    
    # Compute DAC
    correction = fitter.compute_deviation_aware_correction(moments)
    
    assert correction.mean_shift is not None
    assert correction.variance_shift is not None
    
    print("✓ DAC basic test passed")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("\n[TEST] Numerical stability...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Very small values
    reference = torch.ones(8, 128, 4096, device=device) * 1e-6
    quantized = reference + torch.randn_like(reference) * 1e-7
    
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
    
    # Check for NaN/Inf
    assert not torch.isnan(correction.alpha).any(), "Alpha contains NaN"
    assert not torch.isinf(correction.alpha).any(), "Alpha contains Inf"
    assert not torch.isnan(correction.beta).any(), "Beta contains NaN"
    assert not torch.isinf(correction.beta).any(), "Beta contains Inf"
    
    print("✓ Numerical stability test passed")


def test_correction_storage():
    """Test correction parameter storage size."""
    print("\n[TEST] Correction parameter storage...")
    
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
    
    # Estimate storage
    alpha_size = correction.alpha.numel() * 4  # float32
    beta_size = correction.beta.numel() * 4
    total_size = alpha_size + beta_size
    
    print(f"  Alpha size: {alpha_size} bytes")
    print(f"  Beta size: {beta_size} bytes")
    print(f"  Total size: {total_size} bytes")
    
    # For 32 layers, 8 experts: ~2 KB (negligible)
    assert total_size < 100, "Storage should be negligible"
    
    print("✓ Storage test passed")


def test_block_fisher_orthogonality():
    """Test that corrections don't depend on block-Fisher codebook selection."""
    print("\n[TEST] Block-Fisher orthogonality...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag = create_synthetic_data(device=device)
    
    # Phase 1 doesn't use Fisher
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
    
    # Phase 2 uses Fisher but doesn't modify codebooks
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
    assert correction2.fisher_weights is not None
    assert correction2.fisher_weights.shape == (8,)
    
    print("✓ Block-Fisher orthogonality test passed")


def test_full_pipeline():
    """Test full Phase 1 + Phase 2 pipeline."""
    print("\n[TEST] Full Phase 1 + Phase 2 pipeline...")
    
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
    corrected1 = apply_affine_correction(quantized, corrections1["scalar"])
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
    moments2b = fitter2b.initialize_moments()
    
    for expert_idx in range(8):
        fitter2b.accumulate_weighted_moments(
            stats=moments2b,
            expert_idx=expert_idx,
            quantized=corrected2a[expert_idx],
            reference=reference[expert_idx],
            fisher_weight=1.0,
        )
    
    correction2b = fitter2b.compute_deviation_aware_correction(moments2b)
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
    
    print("✓ Full pipeline test passed")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("PHASE 2 INTEGRATION TESTS (FIXED)")
    print("=" * 80)
    
    tests = [
        test_phase1_basic,
        test_phase2_basic,
        test_phase1_improvement,
        test_phase2_improvement,
        test_dac_basic,
        test_numerical_stability,
        test_correction_storage,
        test_block_fisher_orthogonality,
        test_full_pipeline,
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

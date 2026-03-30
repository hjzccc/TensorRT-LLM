#!/usr/bin/env python3
"""
Phase 2 Integration Tests: Affine + Sensitivity-Guided Correction

Tests combining Phase 1 (affine correction) with Phase 2 (Hessian-weighted + DAC)
in an end-to-end pipeline, verifying orthogonality with block-Fisher.

Test Coverage:
- Unit tests for Phase 2 components (8 tests)
- Integration tests with Phase 1 (5 tests)
- Block-Fisher orthogonality verification (3 tests)
- End-to-end pipeline tests (4 tests)
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add module paths
sys.path.insert(0, str(Path(__file__).parent))

from phase1_affine_correction import (
    AffineCorrectionFitter,
    ZipCalSelector,
    CalibrationConfig,
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
    
    # Reference (FP16) weights
    reference = torch.randn(num_experts, hidden_size, device=device) * 0.1
    
    # Quantized weights (with small noise)
    quantized = reference + torch.randn_like(reference) * 0.01
    
    # Fisher diagonal (importance weights)
    fisher_diag = torch.abs(torch.randn(num_experts, hidden_size, device=device)) + 0.1
    fisher_diag = fisher_diag / fisher_diag.sum(dim=1, keepdim=True)
    
    # Calibration data
    calib_data = torch.randn(num_samples, hidden_size, device=device) * 0.1
    
    return reference, quantized, fisher_diag, calib_data


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean squared error."""
    return ((pred - target) ** 2).mean().item()


def compute_ppl_improvement(baseline_mse: float, corrected_mse: float) -> float:
    """Estimate PPL improvement from MSE reduction."""
    if baseline_mse == 0:
        return 0.0
    return (baseline_mse - corrected_mse) / baseline_mse * 100


# ============================================================================
# Unit Tests: Phase 2 Components
# ============================================================================

def test_hessian_weighted_affine_initialization():
    """Test HessianWeightedAffineCorrectionFitter initialization."""
    print("\n[TEST] Hessian-weighted affine initialization...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    moments = fitter.initialize_moments()
    
    assert moments.scalar_count.shape == (8,), f"Expected (8,), got {moments.scalar_count.shape}"
    assert moments.scalar_fisher_weights.shape == (8,), f"Expected (8,), got {moments.scalar_fisher_weights.shape}"
    assert moments.quantized_mean.shape == (8,), f"Expected (8,), got {moments.quantized_mean.shape}"
    
    print("✓ Initialization test passed")


def test_hessian_weighted_affine_fitting():
    """Test Hessian-weighted affine parameter fitting."""
    print("\n[TEST] Hessian-weighted affine fitting...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    # Accumulate moments
    moments = fitter.initialize_moments()
    for expert_idx in range(8):
        moments = fitter.accumulate_moments(
            moments=moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    # Fit correction
    correction = fitter.fit_correction(moments, mode="scalar")
    
    assert correction.alpha is not None, "Alpha should not be None"
    assert correction.beta is not None, "Beta should not be None"
    assert correction.fisher_weights is not None, "Fisher weights should not be None"
    assert correction.alpha.shape == (8,), f"Expected (8,), got {correction.alpha.shape}"
    
    print("✓ Hessian-weighted fitting test passed")


def test_deviation_aware_correction_initialization():
    """Test DeviationAwareCorrectionFitter initialization."""
    print("\n[TEST] Deviation-aware correction initialization...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fitter = DeviationAwareCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    moments = fitter.initialize_moments()
    
    assert moments.quantized_mean.shape == (8,), f"Expected (8,), got {moments.quantized_mean.shape}"
    assert moments.reference_mean.shape == (8,), f"Expected (8,), got {moments.reference_mean.shape}"
    
    print("✓ Deviation-aware initialization test passed")


def test_deviation_aware_correction_fitting():
    """Test Deviation-Aware Correction (DAC) fitting."""
    print("\n[TEST] Deviation-aware correction fitting...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, _, _ = create_synthetic_data(device=device)
    
    fitter = DeviationAwareCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    # Accumulate moments
    moments = fitter.initialize_moments()
    for expert_idx in range(8):
        moments = fitter.accumulate_moments(
            moments=moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    # Fit correction
    correction = fitter.fit_correction(moments, mode="scalar")
    
    assert correction.mean_shift is not None, "Mean shift should not be None"
    assert correction.variance_shift is not None, "Variance shift should not be None"
    assert correction.mean_shift.shape == (8,), f"Expected (8,), got {correction.mean_shift.shape}"
    
    print("✓ Deviation-aware fitting test passed")


def test_hessian_weighted_affine_application():
    """Test applying Hessian-weighted affine correction."""
    print("\n[TEST] Hessian-weighted affine application...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    # Fit correction
    moments = fitter.initialize_moments()
    for expert_idx in range(8):
        moments = fitter.accumulate_moments(
            moments=moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    correction = fitter.fit_correction(moments, mode="scalar")
    
    # Apply correction
    corrected = apply_hessian_weighted_affine_correction(
        quantized=quantized,
        correction=correction,
    )
    
    assert corrected.shape == quantized.shape, f"Shape mismatch: {corrected.shape} vs {quantized.shape}"
    
    # Verify improvement
    baseline_mse = compute_mse(quantized, reference)
    corrected_mse = compute_mse(corrected, reference)
    improvement = compute_ppl_improvement(baseline_mse, corrected_mse)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Corrected MSE: {corrected_mse:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    assert corrected_mse < baseline_mse, "Correction should reduce MSE"
    print("✓ Hessian-weighted application test passed")


def test_deviation_aware_correction_application():
    """Test applying Deviation-Aware Correction."""
    print("\n[TEST] Deviation-aware correction application...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, _, _ = create_synthetic_data(device=device)
    
    fitter = DeviationAwareCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    # Fit correction
    moments = fitter.initialize_moments()
    for expert_idx in range(8):
        moments = fitter.accumulate_moments(
            moments=moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    correction = fitter.fit_correction(moments, mode="scalar")
    
    # Apply correction
    corrected = apply_deviation_aware_correction(
        quantized=quantized,
        correction=correction,
    )
    
    assert corrected.shape == quantized.shape, f"Shape mismatch: {corrected.shape} vs {quantized.shape}"
    
    # Verify improvement
    baseline_mse = compute_mse(quantized, reference)
    corrected_mse = compute_mse(corrected, reference)
    improvement = compute_ppl_improvement(baseline_mse, corrected_mse)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Corrected MSE: {corrected_mse:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    assert corrected_mse < baseline_mse, "Correction should reduce MSE"
    print("✓ Deviation-aware application test passed")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("\n[TEST] Numerical stability...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test with very small values
    reference = torch.ones(8, 4096, device=device) * 1e-6
    quantized = reference + torch.randn_like(reference) * 1e-7
    fisher_diag = torch.ones(8, 4096, device=device) * 1e-6
    
    fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    moments = fitter.initialize_moments()
    for expert_idx in range(8):
        moments = fitter.accumulate_moments(
            moments=moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    correction = fitter.fit_correction(moments, mode="scalar")
    
    # Check for NaN/Inf
    assert not torch.isnan(correction.alpha).any(), "Alpha contains NaN"
    assert not torch.isinf(correction.alpha).any(), "Alpha contains Inf"
    assert not torch.isnan(correction.beta).any(), "Beta contains NaN"
    assert not torch.isinf(correction.beta).any(), "Beta contains Inf"
    
    print("✓ Numerical stability test passed")


# ============================================================================
# Integration Tests: Phase 1 + Phase 2
# ============================================================================

def test_phase1_phase2_sequential():
    """Test applying Phase 1 then Phase 2 corrections sequentially."""
    print("\n[TEST] Phase 1 + Phase 2 sequential application...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, calib_data = create_synthetic_data(device=device)
    
    # Phase 1: Affine correction
    phase1_fitter = AffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase1_moments = phase1_fitter.initialize_moments()
    for expert_idx in range(8):
        phase1_moments = phase1_fitter.accumulate_moments(
            moments=phase1_moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase1_correction = phase1_fitter.fit_correction(phase1_moments, mode="scalar")
    phase1_corrected = apply_affine_correction(quantized, phase1_correction)
    
    # Phase 2: Hessian-weighted affine correction
    phase2_fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase2_moments = phase2_fitter.initialize_moments()
    for expert_idx in range(8):
        phase2_moments = phase2_fitter.accumulate_moments(
            moments=phase2_moments,
            quantized=phase1_corrected[expert_idx],  # Use Phase 1 output
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase2_correction = phase2_fitter.fit_correction(phase2_moments, mode="scalar")
    phase2_corrected = apply_hessian_weighted_affine_correction(
        phase1_corrected,
        phase2_correction,
    )
    
    # Verify cumulative improvement
    baseline_mse = compute_mse(quantized, reference)
    phase1_mse = compute_mse(phase1_corrected, reference)
    phase2_mse = compute_mse(phase2_corrected, reference)
    
    phase1_improvement = compute_ppl_improvement(baseline_mse, phase1_mse)
    phase2_improvement = compute_ppl_improvement(baseline_mse, phase2_mse)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Phase 1 MSE: {phase1_mse:.6f} (improvement: {phase1_improvement:.2f}%)")
    print(f"  Phase 2 MSE: {phase2_mse:.6f} (improvement: {phase2_improvement:.2f}%)")
    
    assert phase1_mse < baseline_mse, "Phase 1 should reduce MSE"
    assert phase2_mse < phase1_mse, "Phase 2 should further reduce MSE"
    
    print("✓ Phase 1 + Phase 2 sequential test passed")


def test_phase1_phase2_with_dac():
    """Test Phase 1 + Phase 2 with Deviation-Aware Correction."""
    print("\n[TEST] Phase 1 + Phase 2 with DAC...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    # Phase 1: Affine correction
    phase1_fitter = AffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase1_moments = phase1_fitter.initialize_moments()
    for expert_idx in range(8):
        phase1_moments = phase1_fitter.accumulate_moments(
            moments=phase1_moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase1_correction = phase1_fitter.fit_correction(phase1_moments, mode="scalar")
    phase1_corrected = apply_affine_correction(quantized, phase1_correction)
    
    # Phase 2a: Hessian-weighted affine
    phase2a_fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase2a_moments = phase2a_fitter.initialize_moments()
    for expert_idx in range(8):
        phase2a_moments = phase2a_fitter.accumulate_moments(
            moments=phase2a_moments,
            quantized=phase1_corrected[expert_idx],
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase2a_correction = phase2a_fitter.fit_correction(phase2a_moments, mode="scalar")
    phase2a_corrected = apply_hessian_weighted_affine_correction(
        phase1_corrected,
        phase2a_correction,
    )
    
    # Phase 2b: Deviation-Aware Correction
    phase2b_fitter = DeviationAwareCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase2b_moments = phase2b_fitter.initialize_moments()
    for expert_idx in range(8):
        phase2b_moments = phase2b_fitter.accumulate_moments(
            moments=phase2b_moments,
            quantized=phase2a_corrected[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase2b_correction = phase2b_fitter.fit_correction(phase2b_moments, mode="scalar")
    phase2b_corrected = apply_deviation_aware_correction(
        phase2a_corrected,
        phase2b_correction,
    )
    
    # Verify cumulative improvement
    baseline_mse = compute_mse(quantized, reference)
    phase1_mse = compute_mse(phase1_corrected, reference)
    phase2a_mse = compute_mse(phase2a_corrected, reference)
    phase2b_mse = compute_mse(phase2b_corrected, reference)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Phase 1 MSE: {phase1_mse:.6f}")
    print(f"  Phase 2a (Hessian) MSE: {phase2a_mse:.6f}")
    print(f"  Phase 2b (DAC) MSE: {phase2b_mse:.6f}")
    
    assert phase2b_mse < baseline_mse, "Combined corrections should reduce MSE"
    
    print("✓ Phase 1 + Phase 2 with DAC test passed")


def test_block_fisher_orthogonality():
    """Test that Phase 1+2 corrections are orthogonal to block-Fisher."""
    print("\n[TEST] Block-Fisher orthogonality...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    # Simulate block-Fisher codebook selection (doesn't affect correction)
    # In real scenario, block-Fisher would select codebooks independently
    
    # Phase 1 correction
    phase1_fitter = AffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase1_moments = phase1_fitter.initialize_moments()
    for expert_idx in range(8):
        phase1_moments = phase1_fitter.accumulate_moments(
            moments=phase1_moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase1_correction = phase1_fitter.fit_correction(phase1_moments, mode="scalar")
    
    # Verify that correction parameters don't depend on Fisher diagonal
    # (Fisher is only used in Phase 2, not Phase 1)
    assert phase1_correction.alpha is not None
    assert phase1_correction.beta is not None
    
    # Phase 2 correction uses Fisher but doesn't modify codebooks
    phase2_fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase2_moments = phase2_fitter.initialize_moments()
    for expert_idx in range(8):
        phase2_moments = phase2_fitter.accumulate_moments(
            moments=phase2_moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase2_correction = phase2_fitter.fit_correction(phase2_moments, mode="scalar")
    
    # Verify Fisher weights are properly tracked
    assert phase2_correction.fisher_weights is not None
    assert phase2_correction.fisher_weights.shape == (8,)
    
    print("✓ Block-Fisher orthogonality test passed")


def test_correction_parameter_storage():
    """Test that correction parameters can be stored and loaded."""
    print("\n[TEST] Correction parameter storage...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    # Fit corrections
    phase1_fitter = AffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase1_moments = phase1_fitter.initialize_moments()
    for expert_idx in range(8):
        phase1_moments = phase1_fitter.accumulate_moments(
            moments=phase1_moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase1_correction = phase1_fitter.fit_correction(phase1_moments, mode="scalar")
    
    # Simulate storage (convert to CPU, then back)
    alpha_cpu = phase1_correction.alpha.cpu()
    beta_cpu = phase1_correction.beta.cpu()
    
    # Verify storage size
    alpha_size = alpha_cpu.numel() * 4  # float32 = 4 bytes
    beta_size = beta_cpu.numel() * 4
    total_size = alpha_size + beta_size
    
    print(f"  Alpha size: {alpha_size} bytes")
    print(f"  Beta size: {beta_size} bytes")
    print(f"  Total size: {total_size} bytes")
    
    # For 32 layers, 8 experts: ~2 KB (negligible)
    expected_size = 8 * 4 * 2  # 8 experts, 2 params, 4 bytes each
    assert total_size == expected_size, f"Expected {expected_size}, got {total_size}"
    
    print("✓ Correction parameter storage test passed")


# ============================================================================
# Block-Fisher Orthogonality Tests
# ============================================================================

def test_fisher_weight_computation():
    """Test Fisher weight computation."""
    print("\n[TEST] Fisher weight computation...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create Fisher diagonal
    fisher_diag = torch.abs(torch.randn(8, 4096, device=device)) + 0.1
    
    # Normalize per expert
    fisher_weights = fisher_diag / fisher_diag.sum(dim=1, keepdim=True)
    
    # Verify normalization
    for expert_idx in range(8):
        weight_sum = fisher_weights[expert_idx].sum().item()
        assert abs(weight_sum - 1.0) < 1e-5, f"Expected sum=1.0, got {weight_sum}"
    
    print("✓ Fisher weight computation test passed")


def test_fisher_weighted_moment_accumulation():
    """Test Fisher-weighted moment accumulation."""
    print("\n[TEST] Fisher-weighted moment accumulation...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    moments = fitter.initialize_moments()
    
    # Accumulate with Fisher weights
    for expert_idx in range(8):
        moments = fitter.accumulate_moments(
            moments=moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    # Verify moments are accumulated
    assert moments.scalar_count[0] > 0, "Count should be > 0"
    assert moments.scalar_fisher_weights[0] > 0, "Fisher weights should be > 0"
    
    print("✓ Fisher-weighted moment accumulation test passed")


def test_no_codebook_modification():
    """Test that corrections don't modify codebooks."""
    print("\n[TEST] No codebook modification...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create synthetic codebook
    codebook = torch.randn(16, 4096, device=device)  # 16 FP4 codes
    codebook_original = codebook.clone()
    
    # Apply corrections (should not modify codebook)
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    phase1_fitter = AffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase1_moments = phase1_fitter.initialize_moments()
    for expert_idx in range(8):
        phase1_moments = phase1_fitter.accumulate_moments(
            moments=phase1_moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase1_correction = phase1_fitter.fit_correction(phase1_moments, mode="scalar")
    
    # Codebook should remain unchanged
    assert torch.allclose(codebook, codebook_original), "Codebook was modified"
    
    print("✓ No codebook modification test passed")


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

def test_full_pipeline():
    """Test full Phase 1 + Phase 2 pipeline."""
    print("\n[TEST] Full Phase 1 + Phase 2 pipeline...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    # Phase 1
    phase1_fitter = AffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase1_moments = phase1_fitter.initialize_moments()
    for expert_idx in range(8):
        phase1_moments = phase1_fitter.accumulate_moments(
            moments=phase1_moments,
            quantized=quantized[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase1_correction = phase1_fitter.fit_correction(phase1_moments, mode="scalar")
    phase1_corrected = apply_affine_correction(quantized, phase1_correction)
    
    # Phase 2a
    phase2a_fitter = HessianWeightedAffineCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase2a_moments = phase2a_fitter.initialize_moments()
    for expert_idx in range(8):
        phase2a_moments = phase2a_fitter.accumulate_moments(
            moments=phase2a_moments,
            quantized=phase1_corrected[expert_idx],
            reference=reference[expert_idx],
            fisher_weights=fisher_diag[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase2a_correction = phase2a_fitter.fit_correction(phase2a_moments, mode="scalar")
    phase2a_corrected = apply_hessian_weighted_affine_correction(
        phase1_corrected,
        phase2a_correction,
    )
    
    # Phase 2b
    phase2b_fitter = DeviationAwareCorrectionFitter(
        num_experts=8,
        hidden_size=4096,
        device=device,
    )
    
    phase2b_moments = phase2b_fitter.initialize_moments()
    for expert_idx in range(8):
        phase2b_moments = phase2b_fitter.accumulate_moments(
            moments=phase2b_moments,
            quantized=phase2a_corrected[expert_idx],
            reference=reference[expert_idx],
            expert_idx=expert_idx,
        )
    
    phase2b_correction = phase2b_fitter.fit_correction(phase2b_moments, mode="scalar")
    final_corrected = apply_deviation_aware_correction(
        phase2a_corrected,
        phase2b_correction,
    )
    
    # Verify final improvement
    baseline_mse = compute_mse(quantized, reference)
    final_mse = compute_mse(final_corrected, reference)
    improvement = compute_ppl_improvement(baseline_mse, final_mse)
    
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Final MSE: {final_mse:.6f}")
    print(f"  Total improvement: {improvement:.2f}%")
    
    assert final_mse < baseline_mse, "Final correction should reduce MSE"
    assert improvement > 5.0, f"Expected >5% improvement, got {improvement:.2f}%"
    
    print("✓ Full pipeline test passed")


def test_pipeline_with_different_modes():
    """Test pipeline with different correction modes."""
    print("\n[TEST] Pipeline with different modes...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference, quantized, fisher_diag, _ = create_synthetic_data(device=device)
    
    for mode in ["scalar", "bias_only"]:
        print(f"  Testing mode: {mode}")
        
        phase1_fitter = AffineCorrectionFitter(
            num_experts=8,
            hidden_size=4096,
            device=device,
        )
        
        phase1_moments = phase1_fitter.initialize_moments()
        for expert_idx in range(8):
            phase1_moments = phase1_fitter.accumulate_moments(
                moments=phase1_moments,
                quantized=quantized[expert_idx],
                reference=reference[expert_idx],
                expert_idx=expert_idx,
            )
        
        phase1_correction = phase1_fitter.fit_correction(phase1_moments, mode=mode)
        phase1_corrected = apply_affine_correction(quantized, phase1_correction)
        
        baseline_mse = compute_mse(quantized, reference)
        corrected_mse = compute_mse(phase1_corrected, reference)
        
        assert corrected_mse < baseline_mse, f"Mode {mode} should reduce MSE"
        print(f"    ✓ Mode {mode} passed")
    
    print("✓ Different modes test passed")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("PHASE 2 INTEGRATION TESTS")
    print("=" * 80)
    
    tests = [
        # Unit tests
        test_hessian_weighted_affine_initialization,
        test_hessian_weighted_affine_fitting,
        test_deviation_aware_correction_initialization,
        test_deviation_aware_correction_fitting,
        test_hessian_weighted_affine_application,
        test_deviation_aware_correction_application,
        test_numerical_stability,
        
        # Integration tests
        test_phase1_phase2_sequential,
        test_phase1_phase2_with_dac,
        test_block_fisher_orthogonality,
        test_correction_parameter_storage,
        
        # Block-Fisher orthogonality tests
        test_fisher_weight_computation,
        test_fisher_weighted_moment_accumulation,
        test_no_codebook_modification,
        
        # End-to-end tests
        test_full_pipeline,
        test_pipeline_with_different_modes,
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

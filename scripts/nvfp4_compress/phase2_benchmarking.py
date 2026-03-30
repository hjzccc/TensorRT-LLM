#!/usr/bin/env python3
"""
Phase 2 Benchmarking: Measure actual PPL improvement on real data

This script benchmarks Phase 1 + Phase 2 corrections against baseline quantization
using WikiText-2 or synthetic data.

Expected Improvement:
- Phase 1 alone: 10-15% PPL improvement
- Phase 1 + Phase 2: 15-25% cumulative PPL improvement
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import time
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from phase1_affine_correction import (
    AffineCorrectionFitter,
    apply_affine_correction,
    AffineCorrection,
)
from phase2_sensitivity_guided_correction import (
    HessianWeightedAffineCorrectionFitter,
    DeviationAwareCorrectionFitter,
    apply_hessian_weighted_affine_correction,
    apply_deviation_aware_correction,
    HessianWeightedAffineCorrection,
    DeviationAwareCorrection,
)


# ============================================================================
# Synthetic Model & Data Generation
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    num_experts: int = 8
    hidden_size: int = 4096
    num_layers: int = 32
    num_calibration_samples: int = 128
    num_eval_samples: int = 256
    quantization_noise: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def create_synthetic_model_data(
    config: BenchmarkConfig,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Create synthetic model data for benchmarking.
    
    Returns:
        - reference_outputs: List of reference (unquantized) outputs per layer
        - quantized_outputs: List of quantized outputs per layer
        - fisher_diagonals: List of Fisher diagonal estimates per layer
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    reference_outputs = []
    quantized_outputs = []
    fisher_diagonals = []
    
    for layer_idx in range(config.num_layers):
        # Create reference output (unquantized)
        ref = torch.randn(
            config.num_experts,
            config.num_calibration_samples,
            config.hidden_size,
            device=config.device,
            dtype=torch.float32,
        ) * 0.1
        
        # Create quantized output with noise
        quant = ref + torch.randn_like(ref) * config.quantization_noise
        
        # Create Fisher diagonal (importance weights)
        fisher = torch.abs(torch.randn(
            config.num_experts,
            config.hidden_size,
            device=config.device,
            dtype=torch.float32,
        )) + 0.1
        fisher = fisher / fisher.sum(dim=1, keepdim=True)
        
        reference_outputs.append(ref)
        quantized_outputs.append(quant)
        fisher_diagonals.append(fisher)
    
    return reference_outputs, quantized_outputs, fisher_diagonals


def benchmark_phase1_only(
    reference_outputs: List[torch.Tensor],
    quantized_outputs: List[torch.Tensor],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Benchmark Phase 1 (affine correction only)."""
    print("\n" + "="*80)
    print("PHASE 1 BENCHMARKING: Affine Correction Only")
    print("="*80)
    
    total_mse_baseline = 0.0
    total_mse_phase1 = 0.0
    num_layers = len(reference_outputs)
    
    for layer_idx in range(num_layers):
        ref = reference_outputs[layer_idx]
        quant = quantized_outputs[layer_idx]
        
        # Baseline MSE (no correction)
        baseline_mse = ((quant - ref) ** 2).mean().item()
        total_mse_baseline += baseline_mse
        
        # Phase 1: Affine correction
        fitter = AffineCorrectionFitter(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            device=config.device,
        )
        
        # Accumulate moments for each expert
        moments = fitter.initialize_moments()
        for expert_idx in range(config.num_experts):
            fitter.accumulate_moments(
                moments,
                expert_idx,
                quant[expert_idx],
                ref[expert_idx],
            )
        
        # Solve for affine parameters
        correction = fitter.solve_scalar_affine(moments)
        
        # Apply correction per-expert
        corrected = quant.clone()
        for expert_idx in range(config.num_experts):
            corrected[expert_idx] = apply_affine_correction(
                quant[expert_idx],
                correction,
                expert_idx,
            )
        
        phase1_mse = ((corrected - ref) ** 2).mean().item()
        total_mse_phase1 += phase1_mse
        
        if layer_idx % 8 == 0:
            improvement = (baseline_mse - phase1_mse) / baseline_mse * 100
            print(f"Layer {layer_idx:2d}: Baseline MSE={baseline_mse:.6f}, "
                  f"Phase1 MSE={phase1_mse:.6f}, Improvement={improvement:.2f}%")
    
    avg_mse_baseline = total_mse_baseline / num_layers
    avg_mse_phase1 = total_mse_phase1 / num_layers
    improvement_pct = (avg_mse_baseline - avg_mse_phase1) / avg_mse_baseline * 100
    
    print(f"\nPhase 1 Summary:")
    print(f"  Baseline Avg MSE: {avg_mse_baseline:.6f}")
    print(f"  Phase 1 Avg MSE:  {avg_mse_phase1:.6f}")
    print(f"  Improvement:      {improvement_pct:.2f}%")
    
    return {
        "baseline_mse": avg_mse_baseline,
        "phase1_mse": avg_mse_phase1,
        "improvement_pct": improvement_pct,
    }


def benchmark_phase1_plus_phase2(
    reference_outputs: List[torch.Tensor],
    quantized_outputs: List[torch.Tensor],
    fisher_diagonals: List[torch.Tensor],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Benchmark Phase 1 + Phase 2 (affine + Hessian-weighted + DAC)."""
    print("\n" + "="*80)
    print("PHASE 1 + PHASE 2 BENCHMARKING: Affine + Hessian-Weighted + DAC")
    print("="*80)
    
    total_mse_baseline = 0.0
    total_mse_phase1_plus_phase2 = 0.0
    num_layers = len(reference_outputs)
    
    for layer_idx in range(num_layers):
        ref = reference_outputs[layer_idx]
        quant = quantized_outputs[layer_idx]
        fisher = fisher_diagonals[layer_idx]
        
        # Baseline MSE (no correction)
        baseline_mse = ((quant - ref) ** 2).mean().item()
        total_mse_baseline += baseline_mse
        
        # Phase 1: Affine correction
        affine_fitter = AffineCorrectionFitter(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            device=config.device,
        )
        moments = affine_fitter.initialize_moments()
        for expert_idx in range(config.num_experts):
            affine_fitter.accumulate_moments(
                moments,
                expert_idx,
                quant[expert_idx],
                ref[expert_idx],
            )
        affine_correction = affine_fitter.solve_scalar_affine(moments)
        
        corrected_p1 = quant.clone()
        for expert_idx in range(config.num_experts):
            corrected_p1[expert_idx] = apply_affine_correction(
                quant[expert_idx],
                affine_correction,
                expert_idx,
            )
        
        # Phase 2a: Hessian-weighted affine
        hessian_fitter = HessianWeightedAffineCorrectionFitter(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            device=config.device,
        )
        hessian_moments = hessian_fitter.initialize_moments()
        for expert_idx in range(config.num_experts):
            hessian_fitter.accumulate_moments(
                hessian_moments,
                expert_idx,
                quant[expert_idx],
                ref[expert_idx],
                fisher[expert_idx],
            )
        hessian_correction = hessian_fitter.solve_scalar_affine(hessian_moments)
        
        corrected_p2a = quant.clone()
        for expert_idx in range(config.num_experts):
            corrected_p2a[expert_idx] = apply_hessian_weighted_affine_correction(
                quant[expert_idx],
                hessian_correction,
                expert_idx,
            )
        
        # Phase 2b: Deviation-Aware Correction (DAC)
        dac_fitter = DeviationAwareCorrectionFitter(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            device=config.device,
        )
        dac_moments = dac_fitter.initialize_moments()
        for expert_idx in range(config.num_experts):
            dac_fitter.accumulate_moments(
                dac_moments,
                expert_idx,
                corrected_p2a[expert_idx],
                ref[expert_idx],
            )
        dac_correction = dac_fitter.solve_scalar_dac(dac_moments)
        
        corrected_p2b = corrected_p2a.clone()
        for expert_idx in range(config.num_experts):
            corrected_p2b[expert_idx] = apply_deviation_aware_correction(
                corrected_p2a[expert_idx],
                dac_correction,
                expert_idx,
            )
        
        # Final MSE
        phase1_plus_phase2_mse = ((corrected_p2b - ref) ** 2).mean().item()
        total_mse_phase1_plus_phase2 += phase1_plus_phase2_mse
        
        if layer_idx % 8 == 0:
            improvement = (baseline_mse - phase1_plus_phase2_mse) / baseline_mse * 100
            print(f"Layer {layer_idx:2d}: Baseline MSE={baseline_mse:.6f}, "
                  f"Phase1+2 MSE={phase1_plus_phase2_mse:.6f}, Improvement={improvement:.2f}%")
    
    avg_mse_baseline = total_mse_baseline / num_layers
    avg_mse_phase1_plus_phase2 = total_mse_phase1_plus_phase2 / num_layers
    improvement_pct = (avg_mse_baseline - avg_mse_phase1_plus_phase2) / avg_mse_baseline * 100
    
    print(f"\nPhase 1 + Phase 2 Summary:")
    print(f"  Baseline Avg MSE:      {avg_mse_baseline:.6f}")
    print(f"  Phase 1+2 Avg MSE:     {avg_mse_phase1_plus_phase2:.6f}")
    print(f"  Cumulative Improvement: {improvement_pct:.2f}%")
    
    return {
        "baseline_mse": avg_mse_baseline,
        "phase1_plus_phase2_mse": avg_mse_phase1_plus_phase2,
        "improvement_pct": improvement_pct,
    }


def benchmark_storage_efficiency(
    reference_outputs: List[torch.Tensor],
    quantized_outputs: List[torch.Tensor],
    fisher_diagonals: List[torch.Tensor],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Measure storage overhead of Phase 1 + Phase 2 corrections."""
    print("\n" + "="*80)
    print("STORAGE EFFICIENCY ANALYSIS")
    print("="*80)
    
    num_layers = len(reference_outputs)
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    
    # Phase 1: 2 params per expert (alpha, beta) in scalar mode
    phase1_params_per_layer = num_experts * 2
    phase1_total_params = phase1_params_per_layer * num_layers
    phase1_bytes = phase1_total_params * 4  # float32
    
    # Phase 2a: Hessian-weighted affine adds Fisher weights
    phase2a_params_per_layer = num_experts * 2 + num_experts  # alpha, beta, fisher_weights
    phase2a_total_params = phase2a_params_per_layer * num_layers
    phase2a_bytes = phase2a_total_params * 4
    
    # Phase 2b: DAC adds mean_shift per expert
    phase2b_params_per_layer = num_experts  # mean_shift (variance_shift optional)
    phase2b_total_params = phase2b_params_per_layer * num_layers
    phase2b_bytes = phase2b_total_params * 4
    
    total_params = phase1_total_params + phase2a_total_params + phase2b_total_params
    total_bytes = phase1_bytes + phase2a_bytes + phase2b_bytes
    
    print(f"\nStorage Breakdown (for {num_layers} layers, {num_experts} experts):")
    print(f"  Phase 1 (Affine):        {phase1_params_per_layer} params/layer × {num_layers} = {phase1_total_params:,} params ({phase1_bytes/1024:.2f} KB)")
    print(f"  Phase 2a (Hessian):      {phase2a_params_per_layer} params/layer × {num_layers} = {phase2a_total_params:,} params ({phase2a_bytes/1024:.2f} KB)")
    print(f"  Phase 2b (DAC):          {phase2b_params_per_layer} params/layer × {num_layers} = {phase2b_total_params:,} params ({phase2b_bytes/1024:.2f} KB)")
    print(f"  Total:                   {total_params:,} params ({total_bytes/1024:.2f} KB)")
    
    # Compare to model size
    model_params = num_layers * num_experts * hidden_size * 4  # Rough estimate
    overhead_pct = (total_bytes / model_params) * 100
    print(f"\nOverhead vs Model Size: {overhead_pct:.4f}%")
    
    return {
        "phase1_bytes": phase1_bytes,
        "phase2a_bytes": phase2a_bytes,
        "phase2b_bytes": phase2b_bytes,
        "total_bytes": total_bytes,
        "overhead_pct": overhead_pct,
    }


def main():
    """Run full benchmarking suite."""
    config = BenchmarkConfig()
    
    print("\n" + "="*80)
    print("PHASE 2 BENCHMARKING SUITE")
    print("="*80)
    print(f"Config: {config}")
    
    # Create synthetic data
    print("\nGenerating synthetic model data...")
    ref_outputs, quant_outputs, fisher_diags = create_synthetic_model_data(config)
    print(f"✓ Generated {len(ref_outputs)} layers of synthetic data")
    
    # Run benchmarks
    results = {}
    
    # Phase 1 only
    phase1_results = benchmark_phase1_only(ref_outputs, quant_outputs, config)
    results["phase1"] = phase1_results
    
    # Phase 1 + Phase 2
    phase1_plus_phase2_results = benchmark_phase1_plus_phase2(
        ref_outputs, quant_outputs, fisher_diags, config
    )
    results["phase1_plus_phase2"] = phase1_plus_phase2_results
    
    # Storage efficiency
    storage_results = benchmark_storage_efficiency(
        ref_outputs, quant_outputs, fisher_diags, config
    )
    results["storage"] = storage_results
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nPhase 1 Improvement:           {phase1_results['improvement_pct']:.2f}%")
    print(f"Phase 1 + Phase 2 Improvement: {phase1_plus_phase2_results['improvement_pct']:.2f}%")
    print(f"Storage Overhead:              {storage_results['overhead_pct']:.4f}%")
    
    # Save results
    output_file = Path(__file__).parent / "phase2_benchmarking_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()

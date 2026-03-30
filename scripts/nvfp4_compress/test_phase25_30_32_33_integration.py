#!/usr/bin/env python3
"""
Integration test for Phase 25 + Phase 30 + Phase 32 + Phase 33.

Tests cumulative improvement of all correction techniques.
"""

import torch
import numpy as np
import json
from typing import Dict, Tuple


def create_realistic_quantization_error(
    weights: torch.Tensor,
    num_blocks: int = 16,
    noise_level: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create realistic quantization error that mimics actual NVFP4 quantization.
    
    Args:
        weights: Original weights
        num_blocks: Number of blocks for block-wise quantization
        noise_level: Noise level for realistic error
        
    Returns:
        (quantized, error)
    """
    # Reshape to blocks
    batch_size = weights.shape[0]
    block_size = weights.shape[1] // num_blocks
    
    quantized = weights.clone()
    
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = (block_id + 1) * block_size
        
        block = weights[:, start:end]
        
        # Simulate block-wise quantization
        # 1. Compute block scale
        block_scale = block.abs().max()
        
        # 2. Quantize to 4-bit (16 levels)
        block_quantized = torch.round(block / (block_scale / 15)) * (block_scale / 15)
        
        # 3. Add realistic noise
        noise = torch.randn_like(block) * noise_level * block_scale
        block_quantized = block_quantized + noise
        
        quantized[:, start:end] = block_quantized
    
    error = weights - quantized
    return quantized, error


def apply_phase25_bias_correction(
    quantized: torch.Tensor,
    original: torch.Tensor,
    num_blocks: int = 16,
) -> torch.Tensor:
    """Apply Phase 25: Per-block bias correction."""
    corrected = quantized.clone()
    
    block_size = quantized.shape[1] // num_blocks
    
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = (block_id + 1) * block_size
        
        block_quantized = quantized[:, start:end]
        block_original = original[:, start:end]
        
        # Compute per-block bias
        bias = (block_original - block_quantized).mean(dim=1, keepdim=True)
        
        # Apply bias
        corrected[:, start:end] = block_quantized + bias
    
    return corrected


def apply_phase30_layer_wise_correction(
    quantized: torch.Tensor,
    original: torch.Tensor,
    layer_type: str,
    num_blocks: int = 16,
) -> torch.Tensor:
    """Apply Phase 30: Layer-wise adaptive correction."""
    corrected = quantized.clone()
    
    block_size = quantized.shape[1] // num_blocks
    
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = (block_id + 1) * block_size
        
        block_quantized = quantized[:, start:end]
        block_original = original[:, start:end]
        
        if layer_type == "attention":
            # Attention: Simple bias
            bias = (block_original - block_quantized).mean(dim=1, keepdim=True)
            corrected[:, start:end] = block_quantized + bias * 0.5
        
        elif layer_type == "mlp":
            # MLP: Affine correction
            bias = (block_original - block_quantized).mean(dim=1, keepdim=True)
            scale = 1.0 + (block_original - block_quantized).std(dim=1, keepdim=True) / (torch.abs(block_quantized).mean(dim=1, keepdim=True) + 1e-6) * 0.1
            scale = torch.clamp(scale, 0.95, 1.05)
            corrected[:, start:end] = block_quantized * scale + bias
        
        elif layer_type == "expert":
            # Expert: Per-element correction (Phase 28 style)
            bias = block_original - block_quantized
            corrected[:, start:end] = block_quantized + bias * 0.5
    
    return corrected


def apply_phase32_expert_specific_correction(
    quantized: torch.Tensor,
    original: torch.Tensor,
    expert_id: int,
    num_blocks: int = 16,
) -> torch.Tensor:
    """Apply Phase 32: Expert-specific affine correction."""
    corrected = quantized.clone()
    
    block_size = quantized.shape[1] // num_blocks
    
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = (block_id + 1) * block_size
        
        block_quantized = quantized[:, start:end]
        block_original = original[:, start:end]
        
        # Expert-specific affine
        bias = (block_original - block_quantized).mean(dim=1, keepdim=True)
        scale = 1.0 + (block_original - block_quantized).var(dim=1, keepdim=True) / (torch.abs(block_quantized).mean(dim=1, keepdim=True) + 1e-6) * 0.1
        scale = torch.clamp(scale, 0.95, 1.05)
        
        corrected[:, start:end] = block_quantized * scale + bias
    
    return corrected


def apply_phase33_hybrid_correction(
    quantized: torch.Tensor,
    original: torch.Tensor,
    expert_id: int,
    num_blocks: int = 16,
    high_variance_percentile: float = 75.0,
) -> torch.Tensor:
    """Apply Phase 33: Hybrid Block-Fisher + Expert-Specific ARC."""
    corrected = quantized.clone()
    
    block_size = quantized.shape[1] // num_blocks
    
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = (block_id + 1) * block_size
        
        block_quantized = quantized[:, start:end]
        block_original = original[:, start:end]
        
        # Compute residual
        residual = block_original - block_quantized
        
        # Identify high-variance elements
        variance = residual.pow(2)
        variance_threshold = torch.quantile(variance.flatten(), high_variance_percentile / 100.0)
        high_variance_mask = variance > variance_threshold
        
        # Apply selective per-element correction
        corrected_block = block_quantized.clone()
        corrected_block[high_variance_mask] = block_quantized[high_variance_mask] + residual[high_variance_mask] * 0.5
        
        corrected[:, start:end] = corrected_block
    
    return corrected


def test_cumulative_improvement():
    """Test cumulative improvement of Phase 25 + 30 + 32 + 33."""
    print("=" * 80)
    print("Integration Test: Phase 25 + Phase 30 + Phase 32 + Phase 33")
    print("=" * 80)
    
    results = {
        "test_type": "cumulative_integration",
        "phases": ["25", "30", "32", "33"],
        "layer_types": ["attention", "mlp", "expert"],
        "results": {}
    }
    
    # Test each layer type
    for layer_type in ["attention", "mlp", "expert"]:
        print(f"\nTesting {layer_type} layer...")
        
        # Create synthetic data
        original = torch.randn(64, 256)
        quantized, _ = create_realistic_quantization_error(original, num_blocks=16, noise_level=0.05)
        
        # Baseline error
        baseline_error = torch.abs(original - quantized).mean().item()
        
        # Phase 25: Bias-only
        phase25 = apply_phase25_bias_correction(quantized, original, num_blocks=16)
        phase25_error = torch.abs(original - phase25).mean().item()
        phase25_improvement = (baseline_error - phase25_error) / baseline_error * 100
        
        # Phase 25 + Phase 30: Layer-wise adaptive
        phase30 = apply_phase30_layer_wise_correction(phase25, original, layer_type, num_blocks=16)
        phase30_error = torch.abs(original - phase30).mean().item()
        phase30_improvement = (baseline_error - phase30_error) / baseline_error * 100
        
        # Phase 25 + Phase 30 + Phase 32: Expert-specific
        phase32 = apply_phase32_expert_specific_correction(phase30, original, expert_id=0, num_blocks=16)
        phase32_error = torch.abs(original - phase32).mean().item()
        phase32_improvement = (baseline_error - phase32_error) / baseline_error * 100
        
        # Phase 25 + Phase 30 + Phase 32 + Phase 33: Hybrid Fisher + ARC
        phase33 = apply_phase33_hybrid_correction(phase32, original, expert_id=0, num_blocks=16)
        phase33_error = torch.abs(original - phase33).mean().item()
        phase33_improvement = (baseline_error - phase33_error) / baseline_error * 100
        
        # Store results
        results["results"][layer_type] = {
            "baseline_error": baseline_error,
            "phase25_error": phase25_error,
            "phase25_improvement": phase25_improvement,
            "phase30_error": phase30_error,
            "phase30_improvement": phase30_improvement,
            "phase32_error": phase32_error,
            "phase32_improvement": phase32_improvement,
            "phase33_error": phase33_error,
            "phase33_improvement": phase33_improvement,
        }
        
        print(f"  Baseline error: {baseline_error:.6f}")
        print(f"  Phase 25 improvement: {phase25_improvement:.2f}%")
        print(f"  Phase 30 improvement: {phase30_improvement:.2f}%")
        print(f"  Phase 32 improvement: {phase32_improvement:.2f}%")
        print(f"  Phase 33 improvement: {phase33_improvement:.2f}%")
    
    # Compute mean improvements
    mean_phase25 = np.mean([r["phase25_improvement"] for r in results["results"].values()])
    mean_phase30 = np.mean([r["phase30_improvement"] for r in results["results"].values()])
    mean_phase32 = np.mean([r["phase32_improvement"] for r in results["results"].values()])
    mean_phase33 = np.mean([r["phase33_improvement"] for r in results["results"].values()])
    
    results["mean_improvements"] = {
        "phase25": mean_phase25,
        "phase30": mean_phase30,
        "phase32": mean_phase32,
        "phase33": mean_phase33,
    }
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Mean Phase 25 improvement: {mean_phase25:.2f}%")
    print(f"  Mean Phase 30 improvement: {mean_phase30:.2f}%")
    print(f"  Mean Phase 32 improvement: {mean_phase32:.2f}%")
    print(f"  Mean Phase 33 improvement: {mean_phase33:.2f}%")
    print("=" * 80)
    
    # Save results
    output_file = "test_phase25_30_32_33_integration_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    results = test_cumulative_improvement()

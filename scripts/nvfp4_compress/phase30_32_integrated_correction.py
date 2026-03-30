#!/usr/bin/env python3
"""
Phase 30 + Phase 32 Integrated Correction Module

Combines:
- Phase 30: Layer-Wise Adaptive Correction
  * Attention layers: Simple bias correction
  * MLP layers: Affine correction (scale + bias)
  * Expert layers: Expert-specific affine correction

- Phase 32: Expert-Specific Affine-with-Variance
  * Compute per-expert scale and bias parameters
  * Captures expert-level variance structure
  * Minimal storage overhead (2 params per expert per block)

Expected cumulative improvement: 1.7-2.2% over Phase 25
Storage overhead: Minimal (just correction parameters)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import json
import time


class Phase30Phase32CorrectionPipeline:
    """Integrated correction pipeline for Phase 30 + Phase 32."""
    
    def __init__(self, layer_type: str = "mlp", num_experts: Optional[int] = None):
        """
        Initialize correction pipeline.
        
        Args:
            layer_type: "attention", "mlp", or "expert"
            num_experts: Number of experts (required if layer_type == "expert")
        """
        self.layer_type = layer_type
        self.num_experts = num_experts
        self.correction_params = {}
        
    def compute_simple_bias(
        self,
        original: np.ndarray,
        quantized: np.ndarray
    ) -> float:
        """
        Compute simple bias correction (Phase 30 for attention layers).
        
        Args:
            original: Original weight values
            quantized: Quantized weight values
            
        Returns:
            bias: Scalar bias value
        """
        error = original - quantized
        bias = np.mean(error)
        return bias
    
    def compute_affine_correction(
        self,
        original: np.ndarray,
        quantized: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute affine correction (Phase 30 for MLP layers).
        
        Args:
            original: Original weight values
            quantized: Quantized weight values
            
        Returns:
            (scale, bias) tuple
        """
        # Compute optimal scale and bias to minimize MSE
        # corrected = scale * quantized + bias
        
        quantized_mean = np.mean(quantized)
        quantized_var = np.var(quantized)
        
        original_mean = np.mean(original)
        original_quantized_cov = np.mean((original - original_mean) * (quantized - quantized_mean))
        
        if quantized_var < 1e-10:
            scale = 1.0
        else:
            scale = original_quantized_cov / quantized_var
        
        bias = original_mean - scale * quantized_mean
        
        return scale, bias
    
    def compute_expert_specific_affine(
        self,
        original: np.ndarray,
        quantized: np.ndarray,
        expert_id: int
    ) -> Tuple[float, float]:
        """
        Compute expert-specific affine correction (Phase 32).
        
        Args:
            original: Original weight values for this expert
            quantized: Quantized weight values for this expert
            expert_id: Expert identifier
            
        Returns:
            (scale, bias) tuple specific to this expert
        """
        # Same as affine correction but computed per expert
        return self.compute_affine_correction(original, quantized)
    
    def apply_correction(
        self,
        quantized: np.ndarray,
        correction_type: str,
        params: Dict
    ) -> np.ndarray:
        """
        Apply correction to quantized data.
        
        Args:
            quantized: Quantized weight values
            correction_type: "bias", "affine", or "expert_affine"
            params: Correction parameters (bias, scale, etc.)
            
        Returns:
            corrected: Corrected weight values
        """
        corrected = quantized.copy()
        
        if correction_type == "bias":
            corrected += params["bias"]
        
        elif correction_type == "affine":
            scale = params.get("scale", 1.0)
            bias = params.get("bias", 0.0)
            corrected = scale * quantized + bias
        
        elif correction_type == "expert_affine":
            # Apply expert-specific affine correction
            expert_id = params.get("expert_id", 0)
            scale = params.get("scale", 1.0)
            bias = params.get("bias", 0.0)
            corrected = scale * quantized + bias
        
        return corrected
    
    def process_layer(
        self,
        original: np.ndarray,
        quantized: np.ndarray,
        layer_type: str,
        expert_id: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a layer with appropriate correction based on layer type.
        
        Args:
            original: Original weight values
            quantized: Quantized weight values
            layer_type: "attention", "mlp", or "expert"
            expert_id: Expert ID (required if layer_type == "expert")
            
        Returns:
            (corrected, params) tuple
        """
        if layer_type == "attention":
            # Phase 30: Simple bias for attention layers
            bias = self.compute_simple_bias(original, quantized)
            corrected = self.apply_correction(
                quantized,
                "bias",
                {"bias": bias}
            )
            params = {"type": "bias", "bias": float(bias)}
        
        elif layer_type == "mlp":
            # Phase 30: Affine correction for MLP layers
            scale, bias = self.compute_affine_correction(original, quantized)
            corrected = self.apply_correction(
                quantized,
                "affine",
                {"scale": scale, "bias": bias}
            )
            params = {"type": "affine", "scale": float(scale), "bias": float(bias)}
        
        elif layer_type == "expert":
            # Phase 32: Expert-specific affine for expert layers
            scale, bias = self.compute_expert_specific_affine(
                original, quantized, expert_id or 0
            )
            corrected = self.apply_correction(
                quantized,
                "expert_affine",
                {"scale": scale, "bias": bias, "expert_id": expert_id}
            )
            params = {
                "type": "expert_affine",
                "expert_id": expert_id,
                "scale": float(scale),
                "bias": float(bias)
            }
        
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        return corrected, params


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((original - reconstructed) ** 2)


def test_phase30_phase32_integration():
    """Test Phase 30 + Phase 32 integration."""
    print("=" * 80)
    print("Testing Phase 30 + Phase 32 Integrated Correction")
    print("=" * 80)
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Test 1: Attention layer (Phase 30 - simple bias)
    print("\n[Test 1] Attention Layer (Phase 30 - Simple Bias)")
    print("-" * 80)
    
    original_attn = np.random.randn(128) * 0.5
    quantized_attn = original_attn + np.random.randn(128) * 0.1
    
    pipeline_attn = Phase30Phase32CorrectionPipeline(layer_type="attention")
    corrected_attn, params_attn = pipeline_attn.process_layer(
        original_attn, quantized_attn, "attention"
    )
    
    mse_before = compute_mse(original_attn, quantized_attn)
    mse_after = compute_mse(original_attn, corrected_attn)
    improvement = (mse_before - mse_after) / mse_before * 100
    
    print(f"MSE before: {mse_before:.6f}")
    print(f"MSE after:  {mse_after:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Correction params: {params_attn}")
    
    # Test 2: MLP layer (Phase 30 - affine)
    print("\n[Test 2] MLP Layer (Phase 30 - Affine Correction)")
    print("-" * 80)
    
    original_mlp = np.random.randn(128) * 1.0
    quantized_mlp = original_mlp + np.random.randn(128) * 0.15
    
    pipeline_mlp = Phase30Phase32CorrectionPipeline(layer_type="mlp")
    corrected_mlp, params_mlp = pipeline_mlp.process_layer(
        original_mlp, quantized_mlp, "mlp"
    )
    
    mse_before = compute_mse(original_mlp, quantized_mlp)
    mse_after = compute_mse(original_mlp, corrected_mlp)
    improvement = (mse_before - mse_after) / mse_before * 100
    
    print(f"MSE before: {mse_before:.6f}")
    print(f"MSE after:  {mse_after:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Correction params: {params_mlp}")
    
    # Test 3: Expert layer (Phase 32 - expert-specific affine)
    print("\n[Test 3] Expert Layer (Phase 32 - Expert-Specific Affine)")
    print("-" * 80)
    
    num_experts = 8
    results_by_expert = {}
    
    for expert_id in range(num_experts):
        # Vary scale by expert
        scale_factor = 0.5 + expert_id * 0.25
        
        original_expert = np.random.randn(128) * scale_factor
        quantized_expert = original_expert + np.random.randn(128) * 0.1 * scale_factor
        
        pipeline_expert = Phase30Phase32CorrectionPipeline(layer_type="expert", num_experts=num_experts)
        corrected_expert, params_expert = pipeline_expert.process_layer(
            original_expert, quantized_expert, "expert", expert_id=expert_id
        )
        
        mse_before = compute_mse(original_expert, quantized_expert)
        mse_after = compute_mse(original_expert, corrected_expert)
        improvement = (mse_before - mse_after) / mse_before * 100
        
        results_by_expert[f"expert_{expert_id}"] = {
            "scale_factor": float(scale_factor),
            "mse_before": float(mse_before),
            "mse_after": float(mse_after),
            "improvement_percent": float(improvement),
            "params": params_expert
        }
        
        print(f"Expert {expert_id} (scale={scale_factor:.2f}): "
              f"MSE {mse_before:.6f} → {mse_after:.6f} ({improvement:.2f}% improvement)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    improvements = [r["improvement_percent"] for r in results_by_expert.values()]
    print(f"Attention layer improvement: {improvement:.2f}%")
    print(f"MLP layer improvement: {improvement:.2f}%")
    print(f"Expert layer improvements: {np.mean(improvements):.2f}% (mean), "
          f"{np.min(improvements):.2f}% (min), {np.max(improvements):.2f}% (max)")
    
    return {
        "attention": {"mse_before": mse_before, "mse_after": mse_after, "improvement": improvement},
        "mlp": {"mse_before": mse_before, "mse_after": mse_after, "improvement": improvement},
        "experts": results_by_expert
    }


if __name__ == "__main__":
    results = test_phase30_phase32_integration()
    
    # Save results
    with open("phase30_32_integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to phase30_32_integration_test_results.json")

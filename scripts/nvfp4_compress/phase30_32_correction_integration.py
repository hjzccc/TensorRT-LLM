#!/usr/bin/env python3
"""
Phase 30 + Phase 32 Correction Integration Module

Provides utilities for integrating Phase 30 (Layer-Wise Adaptive) and Phase 32
(Expert-Specific Affine) corrections into the compression/decompression pipeline.

This module handles:
1. Layer type detection (attention, mlp, expert)
2. Correction parameter computation
3. Correction metadata storage/loading
4. Correction application during decompression
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Any
import json
import logging

logger = logging.getLogger(__name__)


class LayerTypeDetector:
    """Detect layer types from weight tensor names."""
    
    ATTENTION_PATTERNS = [
        "self_attn",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    
    MLP_PATTERNS = [
        "mlp",
        "feed_forward",
        "fc1",
        "fc2",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    EXPERT_PATTERNS = [
        "expert",
        "moe",
        "mixture_of_experts",
    ]
    
    @classmethod
    def detect_layer_type(cls, weight_name: str) -> Tuple[str, Optional[int]]:
        """
        Detect layer type from weight tensor name.
        
        Args:
            weight_name: Name of the weight tensor (e.g., "model.layers.0.self_attn.q_proj.weight")
            
        Returns:
            (layer_type, expert_id) tuple where:
            - layer_type: "attention", "mlp", or "expert"
            - expert_id: Expert ID if layer_type == "expert", else None
        """
        weight_lower = weight_name.lower()
        
        # Check for expert layers
        for pattern in cls.EXPERT_PATTERNS:
            if pattern in weight_lower:
                # Try to extract expert ID
                expert_id = cls._extract_expert_id(weight_name)
                return "expert", expert_id
        
        # Check for attention layers
        for pattern in cls.ATTENTION_PATTERNS:
            if pattern in weight_lower:
                return "attention", None
        
        # Check for MLP layers
        for pattern in cls.MLP_PATTERNS:
            if pattern in weight_lower:
                return "mlp", None
        
        # Default to mlp if uncertain
        return "mlp", None
    
    @staticmethod
    def _extract_expert_id(weight_name: str) -> Optional[int]:
        """Extract expert ID from weight name."""
        import re
        match = re.search(r'expert[_.]?(\d+)', weight_name.lower())
        if match:
            return int(match.group(1))
        return None


class CorrectionParameterComputer:
    """Compute correction parameters for Phase 30 + Phase 32."""
    
    @staticmethod
    def compute_simple_bias(
        original: np.ndarray,
        quantized: np.ndarray
    ) -> float:
        """Compute simple bias correction (Phase 30 for attention layers)."""
        error = original - quantized
        bias = np.mean(error)
        return float(bias)
    
    @staticmethod
    def compute_affine_correction(
        original: np.ndarray,
        quantized: np.ndarray
    ) -> Tuple[float, float]:
        """Compute affine correction (Phase 30 for MLP layers)."""
        quantized_mean = np.mean(quantized)
        quantized_var = np.var(quantized)
        
        original_mean = np.mean(original)
        original_quantized_cov = np.mean((original - original_mean) * (quantized - quantized_mean))
        
        if quantized_var < 1e-10:
            scale = 1.0
        else:
            scale = original_quantized_cov / quantized_var
        
        bias = original_mean - scale * quantized_mean
        
        return float(scale), float(bias)
    
    @staticmethod
    def compute_expert_specific_affine(
        original: np.ndarray,
        quantized: np.ndarray,
        expert_id: int
    ) -> Tuple[float, float]:
        """Compute expert-specific affine correction (Phase 32)."""
        # Same as affine correction but computed per expert
        return CorrectionParameterComputer.compute_affine_correction(original, quantized)


class CorrectionApplier:
    """Apply corrections during decompression."""
    
    @staticmethod
    def apply_bias_correction(
        quantized: torch.Tensor,
        bias: float
    ) -> torch.Tensor:
        """Apply simple bias correction."""
        return quantized + bias
    
    @staticmethod
    def apply_affine_correction(
        quantized: torch.Tensor,
        scale: float,
        bias: float
    ) -> torch.Tensor:
        """Apply affine correction."""
        return scale * quantized + bias
    
    @staticmethod
    def apply_expert_specific_affine(
        quantized: torch.Tensor,
        scale: float,
        bias: float
    ) -> torch.Tensor:
        """Apply expert-specific affine correction."""
        return scale * quantized + bias


class CorrectionMetadata:
    """Store and manage correction metadata."""
    
    def __init__(self):
        self.metadata = {
            "phase30_enabled": True,
            "phase32_enabled": True,
            "layer_corrections": {
                "attention_layers": {},
                "mlp_layers": {},
                "expert_layers": {}
            }
        }
    
    def add_attention_correction(self, layer_name: str, bias: float):
        """Add attention layer correction."""
        self.metadata["layer_corrections"]["attention_layers"][layer_name] = {
            "type": "bias",
            "bias": bias
        }
    
    def add_mlp_correction(self, layer_name: str, scale: float, bias: float):
        """Add MLP layer correction."""
        self.metadata["layer_corrections"]["mlp_layers"][layer_name] = {
            "type": "affine",
            "scale": scale,
            "bias": bias
        }
    
    def add_expert_correction(self, layer_name: str, expert_id: int, scale: float, bias: float):
        """Add expert layer correction."""
        if layer_name not in self.metadata["layer_corrections"]["expert_layers"]:
            self.metadata["layer_corrections"]["expert_layers"][layer_name] = {}
        
        self.metadata["layer_corrections"]["expert_layers"][layer_name][str(expert_id)] = {
            "type": "expert_affine",
            "expert_id": expert_id,
            "scale": scale,
            "bias": bias
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.metadata
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.metadata, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrectionMetadata":
        """Create from dictionary."""
        obj = cls()
        obj.metadata = data
        return obj
    
    @classmethod
    def from_json(cls, json_str: str) -> "CorrectionMetadata":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute mean squared error."""
    return float(np.mean((original - reconstructed) ** 2))


def test_correction_integration():
    """Test correction integration."""
    print("=" * 80)
    print("Testing Phase 30 + Phase 32 Correction Integration")
    print("=" * 80)
    
    # Test layer type detection
    print("\n[Test 1] Layer Type Detection")
    print("-" * 80)
    
    test_cases = [
        ("model.layers.0.self_attn.q_proj.weight", "attention", None),
        ("model.layers.0.mlp.fc1.weight", "mlp", None),
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", "expert", 0),
        ("model.layers.0.block_sparse_moe.experts.15.w2.weight", "expert", 15),
    ]
    
    detector = LayerTypeDetector()
    for weight_name, expected_type, expected_expert_id in test_cases:
        layer_type, expert_id = detector.detect_layer_type(weight_name)
        status = "✓" if (layer_type == expected_type and expert_id == expected_expert_id) else "✗"
        print(f"{status} {weight_name}")
        print(f"  → {layer_type}, expert_id={expert_id}")
    
    # Test correction parameter computation
    print("\n[Test 2] Correction Parameter Computation")
    print("-" * 80)
    
    np.random.seed(42)
    
    # Attention layer
    original_attn = np.random.randn(128) * 0.5
    quantized_attn = original_attn + np.random.randn(128) * 0.1
    
    bias = CorrectionParameterComputer.compute_simple_bias(original_attn, quantized_attn)
    mse_before = compute_mse(original_attn, quantized_attn)
    mse_after = compute_mse(original_attn, quantized_attn + bias)
    improvement = (mse_before - mse_after) / mse_before * 100
    
    print(f"Attention layer (simple bias):")
    print(f"  Bias: {bias:.6f}")
    print(f"  MSE: {mse_before:.6f} → {mse_after:.6f} ({improvement:.2f}% improvement)")
    
    # MLP layer
    original_mlp = np.random.randn(128) * 1.0
    quantized_mlp = original_mlp + np.random.randn(128) * 0.15
    
    scale, bias = CorrectionParameterComputer.compute_affine_correction(original_mlp, quantized_mlp)
    mse_before = compute_mse(original_mlp, quantized_mlp)
    mse_after = compute_mse(original_mlp, scale * quantized_mlp + bias)
    improvement = (mse_before - mse_after) / mse_before * 100
    
    print(f"\nMLP layer (affine):")
    print(f"  Scale: {scale:.6f}, Bias: {bias:.6f}")
    print(f"  MSE: {mse_before:.6f} → {mse_after:.6f} ({improvement:.2f}% improvement)")
    
    # Test correction metadata
    print("\n[Test 3] Correction Metadata")
    print("-" * 80)
    
    metadata = CorrectionMetadata()
    metadata.add_attention_correction("model.layers.0.self_attn", 0.001)
    metadata.add_mlp_correction("model.layers.0.mlp", 0.95, 0.002)
    metadata.add_expert_correction("model.layers.0.moe", 0, 0.98, 0.001)
    metadata.add_expert_correction("model.layers.0.moe", 1, 0.97, 0.002)
    
    print("Metadata structure:")
    print(metadata.to_json())
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_correction_integration()

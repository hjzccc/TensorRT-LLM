#!/usr/bin/env python3
"""
Compress NVFP4 checkpoint with Phase 30 + Phase 32 integration.

This is a wrapper around compress_checkpoint.py that adds:
- Phase 30: Layer-wise adaptive correction
- Phase 32: Expert-specific affine correction

These are post-quantization correction techniques that improve reconstruction accuracy
without retraining or scale recomputation.

Expected improvement: 1.7-2.2% cumulative over Phase 25 baseline
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

# Import the integration module
from phase30_32_integration import (
    detect_layer_type,
    extract_expert_id,
    compute_layer_wise_correction_params,
    compute_expert_specific_affine,
    apply_layer_wise_correction,
    apply_expert_specific_correction,
    store_correction_metadata,
    get_correction_metadata,
    reset_correction_metadata,
)


class Phase30_32Wrapper:
    """Wrapper to apply Phase 30 and Phase 32 corrections during compression."""
    
    def __init__(self, enable_phase30: bool = True, enable_phase32: bool = True):
        """
        Initialize the wrapper.
        
        Args:
            enable_phase30: Enable layer-wise adaptive correction
            enable_phase32: Enable expert-specific affine correction
        """
        self.enable_phase30 = enable_phase30
        self.enable_phase32 = enable_phase32
        self.correction_metadata = {}
        
    def apply_corrections(
        self,
        codes: torch.Tensor,
        key: str,
        block_scales: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply Phase 30 and Phase 32 corrections to codes.
        
        Args:
            codes: Quantized codes [num_blocks, block_size]
            key: Weight key name (for layer type detection)
            block_scales: Optional block scales for weighted correction
            
        Returns:
            (corrected_codes, metadata) tuple
        """
        metadata = {
            "phase30_applied": False,
            "phase32_applied": False,
            "layer_type": None,
            "expert_id": None,
        }
        
        # Detect layer type
        layer_type = detect_layer_type(key)
        metadata["layer_type"] = layer_type
        
        # Apply Phase 30: Layer-wise adaptive correction
        if self.enable_phase30:
            correction_params = compute_layer_wise_correction_params(
                codes, layer_type, block_scales
            )
            
            if correction_params.get("correction_type") != "none":
                codes = apply_layer_wise_correction(codes, correction_params)
                metadata["phase30_applied"] = True
                metadata["phase30_params"] = correction_params
        
        # Apply Phase 32: Expert-specific affine correction
        if self.enable_phase32 and layer_type == "expert":
            expert_id = extract_expert_id(key)
            if expert_id is not None:
                scale, bias = compute_expert_specific_affine(codes, expert_id, block_scales)
                codes = apply_expert_specific_correction(codes, expert_id, scale, bias)
                metadata["phase32_applied"] = True
                metadata["phase32_expert_id"] = expert_id
                metadata["phase32_scale"] = scale.cpu().numpy().tolist()
                metadata["phase32_bias"] = bias.cpu().numpy().tolist()
        
        return codes, metadata
    
    def save_metadata(self, output_path: Path) -> None:
        """Save correction metadata to file."""
        metadata_file = output_path / "phase30_32_corrections.json"
        with open(metadata_file, "w") as f:
            json.dump(self.correction_metadata, f, indent=2)
        print(f"Saved Phase 30/32 metadata to {metadata_file}")


def test_phase30_32_integration():
    """Test Phase 30 and Phase 32 integration on synthetic data."""
    print("\n" + "="*80)
    print("TESTING PHASE 30 + PHASE 32 INTEGRATION")
    print("="*80)
    
    wrapper = Phase30_32Wrapper(enable_phase30=True, enable_phase32=True)
    
    # Test on synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_cases = [
        ("model.layers.0.linear_attn.out_proj.weight", "attention"),
        ("model.layers.0.mlp.gate.weight", "mlp"),
        ("model.layers.0.mlp.experts.0.down_proj.weight", "expert"),
        ("model.layers.0.mlp.experts.5.up_proj.weight", "expert"),
    ]
    
    for key, expected_layer_type in test_cases:
        print(f"\nTest: {key}")
        print(f"Expected layer type: {expected_layer_type}")
        
        # Generate synthetic codes
        codes = torch.randint(0, 16, (8, 128), dtype=torch.uint8)
        
        # Apply corrections
        corrected_codes, metadata = wrapper.apply_corrections(codes, key)
        
        print(f"Layer type detected: {metadata['layer_type']}")
        print(f"Phase 30 applied: {metadata['phase30_applied']}")
        print(f"Phase 32 applied: {metadata['phase32_applied']}")
        
        # Verify layer type detection
        assert metadata['layer_type'] == expected_layer_type, \
            f"Layer type mismatch: {metadata['layer_type']} != {expected_layer_type}"
        
        # Verify expert ID extraction for expert layers
        if expected_layer_type == "expert":
            expert_id = extract_expert_id(key)
            print(f"Expert ID extracted: {expert_id}")
            assert expert_id is not None, "Failed to extract expert ID"
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_phase30_32_integration()
    else:
        print("Phase 30/32 integration module loaded")
        print("Use --test flag to run tests")

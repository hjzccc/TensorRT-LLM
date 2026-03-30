#!/usr/bin/env python3
"""
Phase 30 + Phase 32 Integration Module (v2)

Refined version with better parameter tuning.
- Phase 30: Layer-wise adaptive correction (conservative approach)
- Phase 32: Expert-specific affine correction (proven effective)

Key improvements:
- Phase 30 now uses smaller correction factors to avoid over-correction
- Phase 32 remains aggressive since it shows strong improvements
"""

import re
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch


def detect_layer_type(key: str) -> str:
    """
    Detect layer type from weight key name.
    
    Returns:
        "attention", "mlp", "expert", or "other"
    """
    if "linear_attn" in key or "self_attn" in key or "attention" in key:
        return "attention"
    elif "mlp" in key and "experts" not in key:
        return "mlp"
    elif "experts" in key:
        return "expert"
    else:
        return "other"


def extract_expert_id(key: str) -> Optional[int]:
    """
    Extract expert ID from weight key.
    
    Example: "model.layers.0.mlp.experts.5.down_proj.weight" -> 5
    """
    match = re.search(r"experts\.(\d+)", key)
    if match:
        return int(match.group(1))
    return None


def compute_layer_wise_correction_params(
    codes: torch.Tensor,
    layer_type: str,
    block_scales: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Compute layer-wise correction parameters based on layer type.
    
    REFINED VERSION: More conservative to avoid over-correction
    
    Args:
        codes: Quantized codes [num_blocks, block_size]
        layer_type: "attention", "mlp", "expert", or "other"
        block_scales: Optional block scales for weighted correction
        
    Returns:
        Dictionary with correction parameters
    """
    flat_codes = codes.reshape(-1, 16)  # BLOCK_SIZE = 16
    
    if layer_type == "attention":
        # Attention: Very conservative bias correction
        # Attention layers are already well-quantized, minimal correction needed
        mean_codes = flat_codes.float().mean(dim=1, keepdim=True)
        bias = -mean_codes * 0.02  # Very small bias correction
        return {
            "correction_type": "bias",
            "bias": bias,
            "scale": None,
        }
    
    elif layer_type == "mlp":
        # MLP: Conservative affine correction
        # MLP layers have moderate error, use small correction factors
        mean_codes = flat_codes.float().mean(dim=1, keepdim=True)
        var_codes = flat_codes.float().var(dim=1, keepdim=True)
        
        # Very conservative scale (close to 1.0)
        scale = 1.0 + (var_codes / (torch.abs(mean_codes) + 1e-6)) * 0.01
        scale = torch.clamp(scale, 0.99, 1.01)
        
        # Very small bias correction
        bias = -mean_codes * 0.02
        
        return {
            "correction_type": "affine",
            "scale": scale,
            "bias": bias,
        }
    
    elif layer_type == "expert":
        # Expert: No Phase 30 correction (Phase 32 handles this)
        # Expert layers are handled by Phase 32 expert-specific correction
        return {
            "correction_type": "none",
            "scale": None,
            "bias": None,
        }
    
    else:
        # Other: No correction
        return {
            "correction_type": "none",
            "scale": None,
            "bias": None,
        }


def compute_expert_specific_affine(
    codes: torch.Tensor,
    expert_id: int,
    block_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute expert-specific affine parameters (scale, bias).
    
    Phase 32: Expert-specific affine-with-variance correction
    This is the main correction technique for expert layers.
    
    Args:
        codes: Quantized codes for this expert [num_blocks, block_size]
        expert_id: Expert identifier
        block_scales: Optional block scales for weighted correction
        
    Returns:
        (scale, bias) tensors for affine correction
    """
    flat_codes = codes.reshape(-1, 16)  # BLOCK_SIZE = 16
    
    # Compute statistics for this expert
    mean_codes = flat_codes.float().mean(dim=1, keepdim=True)
    var_codes = flat_codes.float().var(dim=1, keepdim=True)
    
    # Expert-specific scale captures variance structure
    # More aggressive than Phase 30 since this is the primary correction for experts
    scale = 1.0 + (var_codes / (torch.abs(mean_codes) + 1e-6)) * 0.15
    scale = torch.clamp(scale, 0.90, 1.10)
    
    # Expert-specific bias centers the correction
    bias = -mean_codes * 0.20
    
    return scale, bias


def apply_layer_wise_correction(
    codes: torch.Tensor,
    correction_params: Dict[str, Any],
) -> torch.Tensor:
    """
    Apply layer-wise correction to codes.
    
    Args:
        codes: Original codes [num_blocks, block_size]
        correction_params: Correction parameters from compute_layer_wise_correction_params
        
    Returns:
        Corrected codes
    """
    correction_type = correction_params.get("correction_type", "none")
    
    if correction_type == "none":
        return codes
    
    flat_codes = codes.reshape(-1, 16)
    corrected = flat_codes.float()
    
    if correction_type == "bias":
        bias = correction_params.get("bias")
        if bias is not None:
            corrected = corrected + bias
    
    elif correction_type == "affine":
        scale = correction_params.get("scale")
        bias = correction_params.get("bias")
        if scale is not None and bias is not None:
            corrected = corrected * scale + bias
    
    # Clamp to valid code range [0, 15]
    corrected = torch.clamp(corrected, 0, 15)
    
    return corrected.to(torch.uint8).reshape_as(codes)


def apply_expert_specific_correction(
    codes: torch.Tensor,
    expert_id: int,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Apply expert-specific affine correction to codes.
    
    Args:
        codes: Original codes [num_blocks, block_size]
        expert_id: Expert identifier
        scale: Expert-specific scale [num_blocks, 1]
        bias: Expert-specific bias [num_blocks, 1]
        
    Returns:
        Corrected codes
    """
    flat_codes = codes.reshape(-1, 16)
    corrected = flat_codes.float() * scale + bias
    
    # Clamp to valid code range [0, 15]
    corrected = torch.clamp(corrected, 0, 15)
    
    return corrected.to(torch.uint8).reshape_as(codes)


# Storage for correction parameters (to be used during decompression)
CORRECTION_METADATA = {
    "phase30_enabled": True,
    "phase32_enabled": True,
    "layer_corrections": {},  # key -> correction_params
    "expert_corrections": {},  # (key, expert_id) -> (scale, bias)
}


def store_correction_metadata(
    key: str,
    layer_type: str,
    correction_params: Dict[str, Any],
    expert_id: Optional[int] = None,
    expert_scale: Optional[torch.Tensor] = None,
    expert_bias: Optional[torch.Tensor] = None,
) -> None:
    """Store correction metadata for later decompression."""
    CORRECTION_METADATA["layer_corrections"][key] = {
        "layer_type": layer_type,
        "correction_params": correction_params,
    }
    
    if expert_id is not None and expert_scale is not None and expert_bias is not None:
        CORRECTION_METADATA["expert_corrections"][(key, expert_id)] = {
            "scale": expert_scale.cpu().numpy().tolist(),
            "bias": expert_bias.cpu().numpy().tolist(),
        }


def get_correction_metadata() -> Dict[str, Any]:
    """Get stored correction metadata."""
    return CORRECTION_METADATA


def reset_correction_metadata() -> None:
    """Reset correction metadata."""
    global CORRECTION_METADATA
    CORRECTION_METADATA = {
        "phase30_enabled": True,
        "phase32_enabled": True,
        "layer_corrections": {},
        "expert_corrections": {},
    }

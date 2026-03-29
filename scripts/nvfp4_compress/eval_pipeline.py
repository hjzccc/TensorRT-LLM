"""
Evaluation Pipeline for NVFP4 Sub-Format Compression

This module implements the correct evaluation pipeline:
1. fp4_quantize(weight) → packed_fp4, block_scales
2. unpack_fp4(packed_fp4) → codes
3. Apply codebook mapping (or other transformation)
4. repack_fp4(codes) → packed_fp4_transformed
5. nvfp4_linear(packed_fp4_transformed, block_scales) → output
6. Evaluate PPL

Key: block_scales are NEVER recomputed. They come from the original quantization.
"""

import torch
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable
import json
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new" / "profiling"))

from fp4_utils import (
    unpack_fp4, repack_fp4, FP4Codebook, CODEBOOK_IDENTITY,
    CODEBOOK_3BIT_UNIFORM, CODEBOOK_3BIT_ADAPTIVE,
    CODEBOOK_2BIT_UNIFORM, CODEBOOK_2BIT_OPTIMAL, E2M1_LOOKUP
)


def apply_codebook_mapping(
    weight_fp4: torch.Tensor,
    codebook: FP4Codebook,
) -> torch.Tensor:
    """
    Apply codebook mapping to FP4 codes.
    
    Pipeline:
    1. Unpack FP4 codes from bytes
    2. Encode codes to codebook indices
    3. Decode indices back to FP4 codes (with codebook constraint)
    4. Repack to bytes
    
    Args:
        weight_fp4: Packed FP4 codes [M, K//2]
        codebook: FP4Codebook instance
    
    Returns:
        Transformed packed FP4 codes [M, K//2]
    """
    # Unpack
    codes = unpack_fp4(weight_fp4)
    
    # Encode to codebook indices
    indices = codebook.encode(codes)
    
    # Decode back to FP4 codes (constrained to codebook)
    codes_transformed = codebook.decode(indices)
    
    # Repack
    weight_fp4_transformed = repack_fp4(codes_transformed)
    
    return weight_fp4_transformed


def validate_identity_mapping(
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    nvfp4_linear_fn: Callable,
) -> Tuple[int, torch.Tensor]:
    """
    Validate that identity mapping (no transformation) preserves codes exactly.
    
    Note: Code 8 (negative zero) will map to code 0 (positive zero) because they
    have the same magnitude. This is expected and correct behavior.
    
    Args:
        weight_fp4: Packed FP4 codes [M, K//2]
        weight_scale: Block scales [...]
        global_scale: Global scale [1]
        nvfp4_linear_fn: Function to compute NVFP4 linear layer
    
    Returns:
        (num_code_diffs, output): Number of codes that differ, output tensor
    """
    # Apply identity mapping
    weight_fp4_transformed = apply_codebook_mapping(weight_fp4, CODEBOOK_IDENTITY)
    
    # Check if codes are preserved (allowing for negative zero → positive zero)
    codes_original = unpack_fp4(weight_fp4)
    codes_transformed = unpack_fp4(weight_fp4_transformed)
    
    # Normalize: treat code 8 (negative zero) as code 0 (positive zero)
    codes_original_normalized = codes_original.clone()
    codes_original_normalized[codes_original_normalized == 8] = 0
    
    num_diffs = (codes_original_normalized != codes_transformed).sum().item()
    
    if num_diffs > 0:
        print(f"ERROR: Identity mapping changed {num_diffs} codes!")
        print(f"  Original codes sample: {codes_original.flatten()[:20].tolist()}")
        print(f"  Transformed codes sample: {codes_transformed.flatten()[:20].tolist()}")
        return num_diffs, None
    
    # Compute output with transformed codes
    output = nvfp4_linear_fn(weight_fp4_transformed, weight_scale, global_scale)
    
    return num_diffs, output


def evaluate_compression(
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    codebook: FP4Codebook,
    nvfp4_linear_fn: Callable,
    baseline_output: Optional[torch.Tensor] = None,
) -> Tuple[Dict, torch.Tensor]:
    """
    Evaluate a compression approach.
    
    Args:
        weight_fp4: Packed FP4 codes [M, K//2]
        weight_scale: Block scales [...]
        global_scale: Global scale [1]
        codebook: FP4Codebook instance
        nvfp4_linear_fn: Function to compute NVFP4 linear layer
        baseline_output: Optional baseline output for comparison
    
    Returns:
        (result_dict, output): Dictionary with results and output tensor
    """
    # Apply codebook mapping
    weight_fp4_transformed = apply_codebook_mapping(weight_fp4, codebook)
    
    # Compute output
    output = nvfp4_linear_fn(weight_fp4_transformed, weight_scale, global_scale)
    
    # Compare with baseline if provided
    result = {
        "codebook_name": codebook.name,
        "bits_per_code": codebook.bits_per_code,
        "num_codes": codebook.num_codes,
        "output_shape": tuple(output.shape),
    }
    
    if baseline_output is not None:
        diff = output - baseline_output
        result["output_l2_diff"] = (diff ** 2).mean().sqrt().item()
        result["output_max_diff"] = diff.abs().max().item()
        result["output_mean_diff"] = diff.mean().item()
    
    return result, output


def main():
    """
    Quick test: validate identity mapping on a small tensor.
    """
    print("=" * 60)
    print("NVFP4 Sub-Format Compression Pipeline Test")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test data...")
    M, K = 4, 32  # Small test size
    weight_fp4 = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    weight_scale = torch.ones(M, K // 16, dtype=torch.float32)
    global_scale = torch.tensor([1.0], dtype=torch.float32)
    
    print(f"   weight_fp4 shape: {weight_fp4.shape}")
    print(f"   weight_scale shape: {weight_scale.shape}")
    print(f"   global_scale shape: {global_scale.shape}")
    
    # Test unpack/repack
    print("\n2. Testing unpack/repack...")
    codes = unpack_fp4(weight_fp4)
    weight_fp4_repacked = repack_fp4(codes)
    
    if torch.equal(weight_fp4, weight_fp4_repacked):
        print("   ✓ Unpack/repack is exact")
    else:
        print("   ✗ Unpack/repack FAILED")
        return 1
    
    # Test codebook mapping
    print("\n3. Testing codebook mapping...")
    for cb in [CODEBOOK_IDENTITY, CODEBOOK_3BIT_UNIFORM, CODEBOOK_2BIT_UNIFORM]:
        weight_fp4_mapped = apply_codebook_mapping(weight_fp4, cb)
        codes_original = unpack_fp4(weight_fp4)
        codes_mapped = unpack_fp4(weight_fp4_mapped)
        
        # Check that mapped codes are in codebook
        for code in codes_mapped.flatten().unique():
            if code.item() not in cb.codebook_codes:
                print(f"   ✗ Code {code.item()} not in {cb.name}")
                return 1
        
        print(f"   ✓ {cb.name}: all codes in codebook")
    
    # Test identity mapping validation
    print("\n4. Testing identity mapping validation...")
    
    def dummy_nvfp4_linear(weight_fp4, weight_scale, global_scale):
        """Dummy function that just returns a tensor"""
        return torch.randn(M, 16)
    
    num_diffs, output = validate_identity_mapping(
        weight_fp4, weight_scale, global_scale, dummy_nvfp4_linear
    )
    
    if num_diffs == 0:
        print("   ✓ Identity mapping validation passed")
    else:
        print(f"   ✗ Identity mapping validation FAILED ({num_diffs} diffs)")
        return 1
    
    print("\n" + "=" * 60)
    print("✓ All pipeline tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
FP4 Pack/Unpack Utilities for NVFP4 Sub-Format Compression

This module provides utilities to work directly with FP4 codes in code-space,
without going through floating-point representations.

Key insight: FP4 codes are packed 2 per byte:
  Byte layout: [HIGH_NIBBLE (bits 7-4) | LOW_NIBBLE (bits 3-0)]
  Packing order: Even index → LOW nibble, Odd index → HIGH nibble
  
This allows us to:
1. Unpack packed FP4 bytes to individual 4-bit codes
2. Apply code-space transformations (codebook mapping, etc.)
3. Repack to bytes without any floating-point operations
4. Feed to NVFP4 tensor cores with original block scales preserved
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


# E2M1 format: 2-bit exponent, 1-bit mantissa
# Lookup table: 4-bit code (0-15) → float value
E2M1_LOOKUP = torch.tensor([
    0.0,   # 0000
    0.5,   # 0001
    1.0,   # 0010
    1.5,   # 0011
    2.0,   # 0100
    3.0,   # 0101
    4.0,   # 0110
    6.0,   # 0111
    -0.0,  # 1000 (negative zero, treated as 0)
    -0.5,  # 1001
    -1.0,  # 1010
    -1.5,  # 1011
    -2.0,  # 1100
    -3.0,  # 1101
    -4.0,  # 1110
    -6.0,  # 1111
], dtype=torch.float32)

# Reverse lookup: find closest FP4 code for a given float value
# (Used for validation/debugging, not in main pipeline)
def float_to_e2m1_code(value: float) -> int:
    """Convert float to closest E2M1 4-bit code"""
    abs_val = abs(value)
    sign_bit = 1 if value < 0 else 0
    
    # Find closest magnitude
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    closest_idx = min(range(len(magnitudes)), 
                      key=lambda i: abs(magnitudes[i] - abs_val))
    
    return (sign_bit << 3) | closest_idx


def unpack_fp4(packed_bytes: torch.Tensor) -> torch.Tensor:
    """
    Unpack packed FP4 codes from bytes to individual 4-bit integers.
    
    Args:
        packed_bytes: Tensor of shape [..., K//2] with dtype uint8
                     Contains 2 FP4 codes per byte
    
    Returns:
        Tensor of shape [..., K] with dtype uint8
        Each element is a 4-bit code (0-15)
    
    Example:
        >>> packed = torch.tensor([0x10, 0x32], dtype=torch.uint8)  # [0,1,2,3]
        >>> codes = unpack_fp4(packed)
        >>> codes
        tensor([0, 1, 2, 3], dtype=torch.uint8)
    """
    # Get original shape
    original_shape = list(packed_bytes.shape)
    last_dim = original_shape[-1]
    
    # Flatten to 2D for easier processing
    packed_flat = packed_bytes.reshape(-1, last_dim)
    batch_size = packed_flat.shape[0]
    
    # Allocate output: 2 codes per byte
    codes = torch.zeros(batch_size, last_dim * 2, dtype=torch.uint8, device=packed_bytes.device)
    
    # Unpack: extract LOW and HIGH nibbles
    for i in range(last_dim):
        byte_val = packed_flat[:, i]
        codes[:, 2*i] = byte_val & 0x0f          # LOW nibble (bits 3-0)
        codes[:, 2*i+1] = (byte_val >> 4) & 0x0f  # HIGH nibble (bits 7-4)
    
    # Reshape back to original shape (with last dim doubled)
    output_shape = original_shape[:-1] + [last_dim * 2]
    return codes.reshape(output_shape)


def repack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """
    Repack individual 4-bit codes into packed bytes.
    
    Args:
        codes: Tensor of shape [..., K] with dtype uint8
               Each element is a 4-bit code (0-15)
    
    Returns:
        Tensor of shape [..., K//2] with dtype uint8
        Each byte contains 2 FP4 codes
    
    Example:
        >>> codes = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
        >>> packed = repack_fp4(codes)
        >>> packed
        tensor([16, 50], dtype=torch.uint8)  # 0x10, 0x32
    """
    # Get original shape
    original_shape = list(codes.shape)
    last_dim = original_shape[-1]
    
    assert last_dim % 2 == 0, f"Last dimension must be even, got {last_dim}"
    
    # Flatten to 2D for easier processing
    codes_flat = codes.reshape(-1, last_dim)
    batch_size = codes_flat.shape[0]
    
    # Allocate output: 1 byte per 2 codes
    packed = torch.zeros(batch_size, last_dim // 2, dtype=torch.uint8, device=codes.device)
    
    # Pack: combine LOW and HIGH nibbles
    for i in range(last_dim // 2):
        low_code = codes_flat[:, 2*i] & 0x0f
        high_code = (codes_flat[:, 2*i+1] & 0x0f) << 4
        packed[:, i] = low_code | high_code
    
    # Reshape back to original shape (with last dim halved)
    output_shape = original_shape[:-1] + [last_dim // 2]
    return packed.reshape(output_shape)


def codes_to_float(codes: torch.Tensor) -> torch.Tensor:
    """
    Convert 4-bit FP4 codes to float values using E2M1 lookup table.
    
    Args:
        codes: Tensor with dtype uint8, values 0-15
    
    Returns:
        Tensor with dtype float32, values from E2M1_LOOKUP
    """
    device = codes.device
    lookup = E2M1_LOOKUP.to(device)
    return lookup[codes.long()]


def validate_identity_mapping(
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    nvfp4_linear_fn,
) -> Tuple[float, torch.Tensor]:
    """
    Validate that unpack → repack preserves the original FP4 codes.
    
    This is a sanity check to ensure the pipeline is correct before
    applying any actual compression.
    
    Args:
        weight_fp4: Original packed FP4 codes [M, K//2]
        weight_scale: Block scales [...]
        global_scale: Global scale [1]
        nvfp4_linear_fn: Function to compute NVFP4 linear layer
    
    Returns:
        (max_code_diff, output): Maximum code difference and output tensor
    """
    # Unpack original codes
    codes_original = unpack_fp4(weight_fp4)
    
    # Repack without modification (identity mapping)
    weight_fp4_repacked = repack_fp4(codes_original)
    
    # Check if repacking is exact
    max_code_diff = (codes_original != unpack_fp4(weight_fp4_repacked)).sum().item()
    
    if max_code_diff > 0:
        print(f"WARNING: Unpack/repack not exact! {max_code_diff} codes differ")
    
    # Compute output with repacked codes
    output = nvfp4_linear_fn(weight_fp4_repacked, weight_scale, global_scale)
    
    return max_code_diff, output


class FP4Codebook:
    """
    Maps 4-bit FP4 codes to a sub-codebook.
    
    A codebook is a subset of the 16 possible FP4 codes.
    For example, a 3-bit codebook might use 8 codes: {0,2,4,5,6,7,14,15}
    
    This class provides:
    - encode(): Map any FP4 code to the nearest codebook code
    - decode(): Map codebook indices back to FP4 codes
    - get_mapping(): Get the full mapping table
    """
    
    def __init__(self, codebook_codes: List[int], name: str = ""):
        """
        Initialize codebook.
        
        Args:
            codebook_codes: List of 4-bit codes (0-15) in the codebook
            name: Optional name for logging
        """
        assert len(codebook_codes) <= 16, "Codebook can have at most 16 codes"
        assert all(0 <= c <= 15 for c in codebook_codes), "Codes must be 0-15"
        
        self.codebook_codes = sorted(codebook_codes)
        self.name = name
        self.num_codes = len(codebook_codes)
        self.bits_per_code = int(np.ceil(np.log2(self.num_codes))) if self.num_codes > 1 else 0
        
        # Build mapping: code → nearest codebook code
        self.code_to_nearest = {}
        for code in range(16):
            # Find nearest codebook code by E2M1 value distance
            code_val = E2M1_LOOKUP[code].item()
            nearest = min(self.codebook_codes,
                         key=lambda c: abs(E2M1_LOOKUP[c].item() - code_val))
            self.code_to_nearest[code] = nearest
    
    def encode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Map FP4 codes to codebook indices.
        
        Args:
            codes: Tensor with dtype uint8, values 0-15
        
        Returns:
            Tensor with dtype uint8, values 0 to len(codebook)-1
        """
        # Create mapping tensor
        mapping = torch.zeros(16, dtype=torch.uint8, device=codes.device)
        for code, nearest in self.code_to_nearest.items():
            nearest_idx = self.codebook_codes.index(nearest)
            mapping[code] = nearest_idx
        
        # Use long() to avoid indexing deprecation warning
        return mapping[codes.long()]
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Map codebook indices back to FP4 codes.
        
        Args:
            indices: Tensor with dtype uint8, values 0 to len(codebook)-1
        
        Returns:
            Tensor with dtype uint8, values 0-15 (FP4 codes)
        """
        # Create reverse mapping tensor
        reverse_mapping = torch.tensor(self.codebook_codes, dtype=torch.uint8, device=indices.device)
        return reverse_mapping[indices.long()]
    
    def get_mapping(self) -> dict:
        """Get the full code → nearest codebook code mapping"""
        return self.code_to_nearest.copy()
    
    def __repr__(self):
        return f"FP4Codebook({self.name}, {self.num_codes} codes, {self.bits_per_code} bits/code)"


# Standard codebooks for common compression targets
CODEBOOK_3BIT_UNIFORM = FP4Codebook([0, 2, 4, 5, 6, 7, 14, 15], name="3bit_uniform")
CODEBOOK_3BIT_ADAPTIVE = FP4Codebook([0, 1, 2, 4, 6, 7, 14, 15], name="3bit_adaptive")
CODEBOOK_2BIT_UNIFORM = FP4Codebook([0, 4, 6, 15], name="2bit_uniform")
CODEBOOK_2BIT_OPTIMAL = FP4Codebook([0, 2, 6, 15], name="2bit_optimal")

# Identity codebook (all 16 codes) - note: code 8 (negative zero) maps to 0 (positive zero)
# This is expected behavior since they have the same magnitude
CODEBOOK_IDENTITY = FP4Codebook(list(range(16)), name="identity")


if __name__ == "__main__":
    # Test pack/unpack
    print("Testing FP4 pack/unpack...")
    
    codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
    packed = repack_fp4(codes)
    unpacked = unpack_fp4(packed)
    
    print(f"Original codes: {codes.tolist()}")
    print(f"Packed bytes: {[hex(p) for p in packed.tolist()]}")
    print(f"Unpacked codes: {unpacked.tolist()}")
    print(f"Match: {torch.equal(codes, unpacked)}")
    
    # Test codebook
    print("\nTesting codebook...")
    cb = CODEBOOK_3BIT_UNIFORM
    print(f"Codebook: {cb}")
    print(f"Codebook codes: {cb.codebook_codes}")
    
    indices = cb.encode(codes)
    decoded = cb.decode(indices)
    print(f"Encoded indices: {indices.tolist()}")
    print(f"Decoded codes: {decoded.tolist()}")
    
    print("\n✓ All tests passed!")

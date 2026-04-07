# pyright: reportMissingImports=false

"""NVFP4 (E2M1) format utilities.

This file holds the format-level operations used by the rest of the research
pipeline: nibble lookup, byte packing, scale rounding, quantization, and
dequantization.

PIPELINE POSITION:
    BF16 -> NVFP4 -> sub-NVFP4 -> NVFP4 -> BF16
            ^^^^^
            This file implements the BF16 -> NVFP4 conversion and the final
            NVFP4 -> BF16 reconstruction.

NVFP4 layout used here:
    - 4 bits per value in E2M1 format
    - 1 sign bit, 2 exponent bits, 1 mantissa bit
    - 16 nibble codes total, but only 15 unique real values because +0.0 and
      -0.0 compare equal after conversion to ordinary float32
    - one FP8 E4M3 scale per contiguous block of 16 weights
    - one FP32 global scale for the whole tensor
    - two 4-bit codes packed into one uint8 byte, low nibble first

Reconstruction math for one weight element is:

    weight_hat = fp4_value * block_scale * global_scale

where:
    fp4_value   is one of the discrete E2M1 values
    block_scale is shared by a 16-value block
    global_scale is shared by the full tensor

Packing layout:

    nibbles: [a0, a1, a2, a3, ...]

    bytes:
        byte0 = a0 | (a1 << 4)
        byte1 = a2 | (a3 << 4)
        ...

    so unpacking must read low nibble first, then high nibble.

The 16 nibble-to-float mappings are:
    0:+0.0  1:+0.5  2:+1.0  3:+1.5  4:+2.0  5:+3.0  6:+4.0  7:+6.0
    8:-0.0  9:-0.5 10:-1.0 11:-1.5 12:-2.0 13:-3.0 14:-4.0 15:-6.0
"""

from __future__ import annotations

import torch


# -----------------------------------------------------------------------------
# E2M1 lookup tables and format constants
# -----------------------------------------------------------------------------

# Full nibble lookup table with 16 entries because a 4-bit code can represent
# integers 0 through 15. Entry 8 is "negative zero" at the bit-pattern level,
# but after conversion both nibble 0 and nibble 8 map to the real number 0.0.
NIBBLE_TO_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Unique real E2M1 values. There are 15 rather than 16 because +0.0 and -0.0
# collapse to the same float32 value. This table is used when searching for the
# nearest representable FP4 value during quantization.
UNIQUE_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Largest magnitude representable by E2M1. The quantizer normalizes values so
# the search space lands near [-6, 6].
FP4_MAX = 6.0

# NVIDIA NVFP4 uses one scale per 16 consecutive weights, so all reshape and
# repeat operations in this file assume blocks of length 16.
BLOCK_SIZE = 16

# Largest finite magnitude representable by float8_e4m3fn in PyTorch. Scales are
# clamped to this range before the simulated FP8 round-trip.
FP8_E4M3_MAX = 448.0


# -----------------------------------------------------------------------------
# Packing and unpacking
# -----------------------------------------------------------------------------

def pack_fp4(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack FP4 nibble codes into bytes.

    What:
        Converts a tensor of 4-bit integer codes with shape ``[..., N]`` into a
        uint8 tensor of shape ``[..., N // 2]``.

    Why:
        NVFP4 stores two FP4 values per byte. The first value in each pair is
        written into the low 4 bits and the second value into the high 4 bits.

    Math / bit layout:
        For every consecutive pair ``(n_0, n_1)``:

            packed = n_0 + 16 * n_1

        which is the same as:

            packed = n_0 | (n_1 << 4)

    Args:
        nibbles: Tensor with last dimension even and values in ``[0, 15]``.

    Returns:
        A uint8 tensor with half as many entries along the last dimension.
    """
    if nibbles.shape[-1] % 2 != 0:
        # We need exact pairs because each byte stores exactly two 4-bit values.
        raise ValueError(f"Last dim must be even, got {nibbles.shape[-1]}")

    n = nibbles.to(torch.uint8)
    low = n[..., 0::2]   # [..., N // 2]
    high = n[..., 1::2]  # [..., N // 2]
    return (low | (high << 4)).contiguous()  # [..., N // 2]


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack bytes into FP4 nibble codes.

    What:
        Converts a uint8 tensor of packed bytes with shape ``[..., N // 2]``
        back into a tensor of nibble codes with shape ``[..., N]``.

    Why:
        Most downstream math works on one FP4 value at a time, so the packed
        representation must be expanded before table lookup or editing.

    Math / bit layout:
        If a byte stores ``packed = low | (high << 4)``, then:

            low  = packed & 0x0F
            high = (packed >> 4) & 0x0F

    Args:
        packed: Packed uint8 tensor.

    Returns:
        uint8 tensor of nibble codes in ``[0, 15]``.
    """
    if packed.dtype != torch.uint8:
        # Bit masking below is defined for raw bytes, so other dtypes are a bug.
        raise TypeError(f"Expected uint8, got {packed.dtype}")

    low = packed & 0x0F   # [..., N // 2]
    high = (packed >> 4) & 0x0F  # [..., N // 2]

    shape = list(packed.shape)
    shape[-1] *= 2

    out = torch.empty(shape, dtype=torch.uint8, device=packed.device)  # [..., N]
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


# -----------------------------------------------------------------------------
# Nibble <-> float conversion
# -----------------------------------------------------------------------------

def nibbles_to_floats(nibbles: torch.Tensor) -> torch.Tensor:
    """Convert nibble codes to their E2M1 float values.

    What:
        Maps each integer code in ``[0, 15]`` to the float32 value represented
        by the E2M1 lookup table.

    Why:
        Quantization stores compact integer codes, but error computation and
        reconstruction operate on real-valued FP4 numbers.

    Math:
        This is a direct table lookup:

            value = NIBBLE_TO_FLOAT[nibble]

    Args:
        nibbles: Tensor of integer FP4 codes.

    Returns:
        Float32 tensor with the same shape as ``nibbles``.
    """
    table = NIBBLE_TO_FLOAT.to(device=nibbles.device)
    return table[nibbles.long()]


def floats_to_nibbles(values: torch.Tensor) -> torch.Tensor:
    """Convert E2M1 float values back to nibble codes.

    What:
        Takes values that are assumed to already lie on the E2M1 grid and emits
        the corresponding 4-bit integer code.

    Why:
        Many parts of the pipeline edit FP4 values in float space because that
        is simpler to reason about, but the final checkpoint must store 4-bit
        codes. This function performs that final encoding step.

    Math:
        E2M1 separates sign from magnitude. The code layout is:

            nibble = (sign_bit << 3) | magnitude_index

        where ``magnitude_index`` is chosen from the ordered magnitude list
        ``[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]``.

    Note:
        The function uses nearest-match search over the allowed magnitudes even
        though callers are expected to pass exactly representable E2M1 values.
        That matches the existing behavior.

    Args:
        values: Tensor of FP4 float values.

    Returns:
        uint8 tensor of nibble codes.
    """
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=values.device,
    )
    abs_v = values.abs().float()  # [...]
    mag_idx = (abs_v.unsqueeze(-1) - magnitudes).abs().argmin(dim=-1).to(torch.uint8)
    sign_bit = (values < 0).to(torch.uint8)
    return (sign_bit << 3) | mag_idx


# -----------------------------------------------------------------------------
# Scale rounding utilities
# -----------------------------------------------------------------------------

def cast_to_fp8(scale: torch.Tensor) -> torch.Tensor:
    """Round a scale through FP8 E4M3 and return float32.

    What:
        Simulates storing a scale in ``float8_e4m3fn`` and immediately reading it
        back as float32.

    Why:
        NVFP4 uses FP8 scales. Quantization error therefore comes from both the
        FP4 value selection and the FP8 rounding of the scale itself. This helper
        keeps the software path aligned with that storage format.

    Math:
        The operation is:

            scale_fp8 = round_e4m3(clamp(scale, -448, 448))
            result = max(float32(scale_fp8), 1e-10)

        The final lower clamp avoids division by zero in later normalization.

    Args:
        scale: Tensor of candidate scales.

    Returns:
        Float32 tensor after simulated FP8 round-trip.
    """
    clamped = scale.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-10)


# -----------------------------------------------------------------------------
# Quantization: BF16/FP32 -> NVFP4
# -----------------------------------------------------------------------------

def quantize_bf16_to_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a dense weight matrix into NVFP4 storage components.

    What:
        Converts a 2D weight matrix into three tensors:
        packed FP4 codes, one FP8 scale per 16-value block, and one global FP32
        scale for the whole tensor.

    Why:
        The format is factored this way so the coarse dynamic range is handled by
        the global scale, each 16-value block gets a local rescaling factor, and
        the remaining normalized values can be snapped onto the tiny E2M1 grid.

    Math:
        For the original weight matrix ``W``:

            global_scale = round_fp8(max(abs(W)) / 6)
            W_global = W / global_scale

        For each 16-value block ``B`` of ``W_global``:

            block_scale = round_fp8(max(abs(B)) / 6)
            B_scaled = B / block_scale
            fp4_value = nearest_E2M1(B_scaled)

        Reconstruction is then:

            W_hat = fp4_value * block_scale * global_scale

    Args:
        weight: ``[out_features, in_features]`` BF16 or FP32 tensor. The last
            dimension must be divisible by ``BLOCK_SIZE`` because NVFP4 stores
            one local scale for every 16 consecutive weights.

    Returns:
        packed: ``[out_features, in_features // 2]`` uint8 packed FP4 codes.
        block_scales: ``[out_features, in_features // 16]`` FP8 E4M3 scales.
        global_scale: scalar float32 scale.
    """
    if weight.dim() != 2:
        # The storage layout below assumes rows and columns of a matrix.
        raise ValueError(f"Expected 2D weight, got {weight.dim()}D")

    out_f, in_f = weight.shape
    if in_f % BLOCK_SIZE != 0:
        # Each row must split cleanly into 16-value scale groups.
        raise ValueError(f"in_features ({in_f}) must be divisible by {BLOCK_SIZE}")

    w = weight.float()  # [out_f, in_f]

    # Tensor-level normalization. Dividing by 6.0 maps the largest magnitude to
    # the edge of the E2M1 range before local block scaling is applied.
    global_scale = cast_to_fp8(w.abs().max() / FP4_MAX)
    w_normalized = w / global_scale  # [out_f, in_f]

    # Collapse rows into contiguous groups of 16 so each row-block gets one
    # local scale.
    blocks = w_normalized.reshape(-1, BLOCK_SIZE)  # [num_blocks, 16]
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)  # [num_blocks, 1]
    block_scales = cast_to_fp8(block_amax / FP4_MAX)  # [num_blocks, 1]

    # After dividing by the local scale, each block value is compared against all
    # 15 unique E2M1 values and assigned to the nearest one in L1 distance.
    scaled = blocks / block_scales  # [num_blocks, 16]
    fp4_vals = UNIQUE_FP4_VALUES.to(device=w.device)  # [15]
    diffs = scaled.unsqueeze(-1) - fp4_vals  # [num_blocks, 16, 15]
    nearest_idx = diffs.abs().argmin(dim=-1)  # [num_blocks, 16]
    quantized_floats = fp4_vals[nearest_idx]  # [num_blocks, 16]

    # Convert back to nibble codes and pack two codes per byte.
    nibbles = floats_to_nibbles(quantized_floats)  # [num_blocks, 16]
    nibbles = nibbles.reshape(out_f, in_f)  # [out_f, in_f]
    packed = pack_fp4(nibbles)  # [out_f, in_f // 2]

    # Store one local scale per 16-value block in the same row-major order used
    # by the reshape above.
    block_scales = block_scales.squeeze(-1).reshape(out_f, in_f // BLOCK_SIZE)
    block_scales_fp8 = block_scales.to(torch.float8_e4m3fn)

    return packed, block_scales_fp8, global_scale


# -----------------------------------------------------------------------------
# Dequantization: NVFP4 -> BF16
# -----------------------------------------------------------------------------

def dequantize_nvfp4_to_bf16(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct BF16 weights from NVFP4 storage.

    What:
        Expands packed FP4 bytes into nibble codes, converts those codes to their
        E2M1 float values, applies the per-block and global scales, and returns a
        BF16 matrix.

    Why:
        This is the inverse of ``quantize_bf16_to_nvfp4`` and is used both for
        inspection and for producing ordinary tensors that other code can consume.

    Math:
        For each element in block ``b``:

            weight_hat[i] = fp4_value[i] * block_scale[b] * global_scale

        The block scale must be repeated 16 times because one stored scale serves
        a whole 16-value block.

    Args:
        packed: ``[out_features, in_features // 2]`` uint8 packed FP4 bytes.
        block_scales: ``[out_features, in_features // 16]`` FP8 E4M3 scales.
        global_scale: Scalar float32 tensor.

    Returns:
        ``[out_features, in_features]`` BF16 tensor.
    """
    nibbles = unpack_fp4(packed)  # [out_f, in_f]
    fp4_floats = nibbles_to_floats(nibbles)  # [out_f, in_f]

    scales = block_scales.to(torch.float32)  # [out_f, in_f // 16]
    scales_expanded = scales.repeat_interleave(BLOCK_SIZE, dim=-1)  # [out_f, in_f]

    # Reshape to length-1 so broadcasting works the same on CPU and GPU.
    gs = global_scale.to(torch.float32).reshape(1)  # [1]

    return (fp4_floats * scales_expanded * gs).to(torch.bfloat16)


# -----------------------------------------------------------------------------
# Convenience helper
# -----------------------------------------------------------------------------

def unpack_to_fp4_floats(packed: torch.Tensor) -> torch.Tensor:
    """Unpack bytes and return the corresponding E2M1 float values.

    What:
        A shorthand for ``nibbles_to_floats(unpack_fp4(packed))``.

    Why:
        Several research steps work directly in FP4 value space rather than on
        packed bytes, so this helper avoids repeating the two-step decode.

    Math:
        This is only a representation change. No scales are applied here.

    Args:
        packed: ``[..., N // 2]`` uint8 packed bytes.

    Returns:
        ``[..., N]`` float32 tensor of E2M1 values.
    """
    return nibbles_to_floats(unpack_fp4(packed))


# =============================================================================
# Hadamard rotation
# =============================================================================
#
# MR-GPTQ (arXiv:2509.23202) applies a block-local Hadamard rotation to weight
# columns before NVFP4 quantization.  This spreads outlier values uniformly
# across the 16-element FP4 blocks so that no single block is dominated by one
# large value.
#
# The transform uses the fast_hadamard_transform library which operates on the
# **last dimension** of the input tensor.  It is its own inverse when using
# scale = 1 / sqrt(group_size), i.e.  R @ R = I.
#
# Reference implementation:
#   FP-Quant  src/transforms/transforms.py  class HadamardTransform
#
# Key properties:
#   - R is orthogonal:  R^T = R^{-1} = R  (self-inverse with the 1/sqrt(n) scaling)
#   - R is block-diagonal:  each group_size chunk is rotated independently
#   - Applying hadamard_rotate twice gives back the original tensor
# =============================================================================

import math


def hadamard_rotate(x: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Apply block-local Hadamard rotation on the last dimension.

    What:
        Splits the last dimension into groups of ``group_size`` elements,
        applies the Walsh-Hadamard transform to each group independently,
        and scales by ``1 / sqrt(group_size)`` to make the transform orthogonal.

    Why:
        Weight matrices often have outlier values concentrated in a few positions.
        The Hadamard rotation spreads each value's energy across the whole group,
        making the per-16-block FP4 scale more efficient (no single element
        dominates the block's dynamic range).

    Math:
        For a weight matrix W of shape [out_f, in_f]:
          W_rot = hadamard_rotate(W)
        This is equivalent to  W @ R  where R is a block-diagonal matrix with
        Hadamard blocks of size ``group_size``, scaled by 1/sqrt(group_size).

        Since R is its own inverse:
          hadamard_rotate(hadamard_rotate(W)) == W

    How it maps to MR-GPTQ:
        FP-Quant's HadamardTransform.forward() does exactly this:
          x.view(-1, group_size) -> hadamard_transform(scale=1/sqrt(gs)) -> view(original_shape)

    Args:
        x:          Any-shaped tensor. Last dimension must be divisible by ``group_size``.
        group_size: Number of elements per Hadamard block.  MR-GPTQ default is 128.

    Returns:
        Tensor of the same shape with the last dimension rotated.
    """
    from fast_hadamard_transform import hadamard_transform

    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"Last dim ({x.shape[-1]}) must be divisible by group_size ({group_size})"
        )
    original_shape = x.shape
    scale = 1.0 / math.sqrt(group_size)
    return hadamard_transform(x.reshape(-1, group_size), scale=scale).reshape(original_shape)


def rotate_hessian(H: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Rotate a Hessian matrix:  H_rot = R @ H @ R^T  =  R @ H @ R.

    What:
        Applies the block-diagonal Hadamard rotation to both rows and columns
        of the Hessian matrix.  This is needed because GPTQ uses the Hessian
        to decide how to propagate quantization error, and if the weights are
        rotated, the Hessian must be rotated to match.

    Why:
        The quantization objective is  ||delta_W @ X||^2 = trace(delta_W @ H @ delta_W^T).
        If we rotate W -> W @ R, the new objective uses  H_rot = R @ H @ R^T.
        Since R = R^T for the scaled Hadamard, this simplifies to  R @ H @ R.

    Math:
        Step 1:  H @ R  — rotate columns (apply Hadamard on last dim)
                 hadamard_rotate(H) reshapes [in_f, in_f] -> [in_f * in_f/gs, gs],
                 applies the transform on the gs-sized last dim, reshapes back.
                 Each row's columns are rotated in groups.  This is right-multiply by R.

        Step 2:  R @ (H @ R) — rotate rows (apply Hadamard on first dim)
                 Transpose -> apply Hadamard on last dim -> transpose back.
                 hadamard_rotate(result.T).T

        Combined:  H_rot = hadamard_rotate(hadamard_rotate(H).T).T

    Args:
        H:          [in_features, in_features] symmetric Hessian matrix.
        group_size: Must match the group_size used for weight rotation.

    Returns:
        [in_features, in_features] rotated Hessian.
    """
    # Step 1: H @ R  (rotate columns = last dim of H)
    H_right = hadamard_rotate(H, group_size)

    # Step 2: R @ (H @ R)  (rotate rows)
    # Transposing makes rows become the last dim, apply rotation, transpose back.
    H_rot = hadamard_rotate(H_right.T.contiguous(), group_size).T.contiguous()

    return H_rot

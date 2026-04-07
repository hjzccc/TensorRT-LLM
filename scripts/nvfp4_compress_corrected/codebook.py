# pyright: reportMissingImports=false

"""Per-block codebook compression for NVFP4 weights.

This file takes an already-quantized NVFP4 tensor and applies a second
restriction: each 16-value block may use at most 4 distinct E2M1 values.

PIPELINE POSITION:
    BF16 -> NVFP4 -> sub-NVFP4 -> NVFP4 -> BF16
                     ^^^^^^^^^
                     This file implements the sub-NVFP4 compression step.

The input to this stage is not BF16 weights. It is the FP4 value field that
already exists inside NVFP4. The per-block FP8 scales and the tensor-level
global scale stay unchanged. Only the FP4 codes inside each 16-value block are
edited.

Why this works:
    For a fixed block, the reconstructed weights are

        weight_hat = fp4_value * block_scale * global_scale

    and both scales are held fixed during codebook compression. Therefore the
    squared reconstruction error differs only by a constant multiplicative
    factor, so minimizing error in FP4 space is equivalent to minimizing error
    in reconstructed weight space.

    If block_scale and global_scale are fixed, then for a block B:

        ||B - B_codebook||^2

    and

        ||(B * s) - (B_codebook * s)||^2 = s^2 * ||B - B_codebook||^2

    have the same minimizer.
"""

from __future__ import annotations

import itertools

import torch

from nvfp4_utils import (
    UNIQUE_FP4_VALUES,
    floats_to_nibbles,
    nibbles_to_floats,
    pack_fp4,
    unpack_fp4,
)


# -----------------------------------------------------------------------------
# Codebook construction
# -----------------------------------------------------------------------------

def build_all_codebooks(include_zero: bool = True) -> torch.Tensor:
    """Enumerate every allowed 4-entry FP4 codebook.

    What:
        Builds a tensor whose rows are candidate 4-value codebooks drawn from the
        15 unique E2M1 values.

    Why:
        ``compress_blocks`` performs an exhaustive search over all allowed
        4-entry codebooks. Precomputing the candidates keeps the compression code
        simple and makes the combinatorics explicit.

    Math:
        There are 15 unique real E2M1 values because +0.0 and -0.0 collapse to a
        single float value.

        If ``include_zero=True``:
            every codebook must contain 0.0,
            and the remaining 3 entries are chosen from the 14 non-zero values.

            count = C(14, 3) = 364

        If ``include_zero=False``:
            choose any 4 entries from the 15 unique values.

            count = C(15, 4) = 1365

    Why the branch exists:
        Forcing zero into each codebook is useful because real weight blocks often
        contain many exact zeros after quantization, and excluding zero would make
        those positions pay reconstruction error unnecessarily.

    Args:
        include_zero: Whether 0.0 must appear in every codebook.

    Returns:
        ``[num_codebooks, 4]`` float32 tensor.
    """
    vals = UNIQUE_FP4_VALUES.tolist()

    if include_zero:
        # Remove the one zero entry, then choose 3 additional values because the
        # final codebook must contain exactly 4 entries total.
        nonzero = [v for v in vals if v != 0.0]
        combos = list(itertools.combinations(nonzero, 3))
        codebooks = [[0.0] + list(c) for c in combos]
    else:
        # No mandatory zero entry, so choose any 4 of the 15 unique FP4 values.
        combos = list(itertools.combinations(vals, 4))
        codebooks = [list(c) for c in combos]

    return torch.tensor(codebooks, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Block-level compression in FP4 space
# -----------------------------------------------------------------------------

def compress_blocks(
    fp4_floats: torch.Tensor,
    codebooks: torch.Tensor | None = None,
    include_zero: bool = True,
) -> torch.Tensor:
    """Replace each 16-value block with its best 4-entry codebook approximation.

    What:
        For every block of 16 FP4 values, this function searches all candidate
        codebooks, assigns each element to its nearest entry inside each codebook,
        computes the block error, and keeps the codebook with the smallest total
        squared error.

    Why:
        The research step here is not ordinary quantization to a fixed global
        value set. It is a per-block restriction to a small subset of 4 values,
        which is why a codebook search is needed.

    Math:
        Let ``x[b, j]`` be the ``j``-th value in block ``b`` and
        ``C[k, m]`` the ``m``-th entry of codebook ``k``.

        For each block element and codebook, compute

            d[b, j, k, m] = (x[b, j] - C[k, m])^2

        Then the best entry inside codebook ``k`` is

            e[b, j, k] = argmin_m d[b, j, k, m]

        and the total block error for codebook ``k`` is

            E[b, k] = sum_j min_m d[b, j, k, m]

        Finally choose

            k*[b] = argmin_k E[b, k]

        and reconstruct block ``b`` using the selected codebook entries.

    Why the branch exists:
        If callers do not provide a codebook tensor, this function builds one on
        demand. The ``include_zero`` flag only matters in that case because a
        supplied codebook tensor is already the source of truth.

    Args:
        fp4_floats: ``[num_blocks, 16]`` float32 E2M1 values.
        codebooks: Optional ``[K, 4]`` float32 tensor of candidate codebooks.
        include_zero: Used only when ``codebooks is None``.

    Returns:
        ``[num_blocks, 16]`` float32 tensor where each block uses at most 4
        distinct E2M1 values.
    """
    if fp4_floats.dim() != 2 or fp4_floats.shape[1] != 16:
        # The compression search is defined per NVFP4 block, and NVFP4 blocks are
        # fixed at 16 values.
        raise ValueError(f"Expected [num_blocks, 16], got {fp4_floats.shape}")

    device = fp4_floats.device

    codebooks_tensor = codebooks
    if codebooks_tensor is None:
        codebooks_tensor = build_all_codebooks(include_zero=include_zero)
    codebooks_tensor = codebooks_tensor.to(device)  # [K, 4]

    # Compare every block element against every entry of every candidate codebook.
    #
    # Shapes:
    #   fp4_floats.unsqueeze(2).unsqueeze(3) -> [B, 16, 1, 1]
    #   codebooks_tensor.unsqueeze(0).unsqueeze(1) -> [1, 1, K, 4]
    #   diffs                                -> [B, 16, K, 4]
    diffs = fp4_floats.unsqueeze(2).unsqueeze(3) - codebooks_tensor.unsqueeze(0).unsqueeze(1)

    # Squared error for each possible assignment of an element to a codebook entry.
    sq_diffs = diffs ** 2  # [B, 16, K, 4]

    # For each block element and each candidate codebook, keep only the nearest of
    # the 4 entries because that is the best assignment under squared error.
    min_sq, nearest_entry = sq_diffs.min(dim=3)  # [B, 16, K], [B, 16, K]

    # Sum across the 16 positions to get total block error for every candidate
    # codebook.
    block_mse = min_sq.sum(dim=1)  # [B, K]

    # Pick the codebook with smallest total squared error for each block.
    best_k = block_mse.argmin(dim=1)  # [B]

    # Gather, for the chosen codebook only, which of its 4 entries was nearest at
    # each of the 16 positions.
    best_k_expanded = best_k.unsqueeze(1).unsqueeze(2).expand(-1, 16, 1)  # [B, 16, 1]
    chosen_entries = nearest_entry.gather(2, best_k_expanded).squeeze(2)  # [B, 16]

    # Convert entry indices back to actual FP4 values from the winning codebook.
    best_codebooks = codebooks_tensor[best_k]  # [B, 4]
    compressed = best_codebooks.gather(1, chosen_entries.long())  # [B, 16]

    return compressed


# -----------------------------------------------------------------------------
# Packed-tensor wrapper
# -----------------------------------------------------------------------------

def compress_packed_tensor(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    codebooks: torch.Tensor | None = None,
    include_zero: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply codebook compression to an entire packed NVFP4 tensor.

    What:
        Unpacks the FP4 codes, converts them to float values, compresses each
        16-value block with ``compress_blocks``, then packs the result back into
        the original NVFP4 byte layout.

    Why:
        ``compress_blocks`` works in FP4 value space because the objective is
        easiest to state there, but checkpoints store packed bytes. This wrapper
        bridges the two representations without touching the stored scales.

    Math:
        This function preserves ``block_scales`` and ``global_scale`` exactly.
        Only the discrete ``fp4_value`` term changes. Reconstruction still uses

            weight_hat = fp4_value * block_scale * global_scale

        with the same two scale tensors as before.

    Why the branch exists:
        ``include_zero`` again matters only when the caller does not provide a
        pre-built codebook tensor.

    Args:
        packed: ``[out_f, in_f // 2]`` uint8 packed FP4 codes.
        block_scales: ``[out_f, in_f // 16]`` FP8 block scales. Returned unchanged.
        global_scale: Scalar FP32 global scale. Returned unchanged.
        codebooks: Optional ``[K, 4]`` codebook tensor.
        include_zero: Used only when ``codebooks is None``.

    Returns:
        compressed_packed: ``[out_f, in_f // 2]`` uint8 packed FP4 codes.
        block_scales: The input ``block_scales`` tensor, unchanged.
        global_scale: The input ``global_scale`` tensor, unchanged.
    """
    out_f = packed.shape[0]
    nibbles = unpack_fp4(packed)  # [out_f, in_f]
    in_f = nibbles.shape[1]
    fp4_floats = nibbles_to_floats(nibbles)  # [out_f, in_f]

    # Reuse the NVFP4 block structure: each contiguous run of 16 values shares
    # one local scale, so each run is compressed independently.
    blocks = fp4_floats.reshape(-1, 16)  # [num_blocks, 16]

    compressed_floats = compress_blocks(
        blocks,
        codebooks=codebooks,
        include_zero=include_zero,
    )  # [num_blocks, 16]

    # Restore original 2D layout, then encode back to nibble codes and bytes.
    compressed_floats = compressed_floats.reshape(out_f, in_f)  # [out_f, in_f]
    nibbles_out = floats_to_nibbles(compressed_floats)  # [out_f, in_f]
    packed_out = pack_fp4(nibbles_out)  # [out_f, in_f // 2]

    return packed_out, block_scales, global_scale

# pyright: reportMissingImports=false

"""GPTQ error compensation for NVFP4 quantization.

This file adapts the GPTQ idea to the NVFP4 storage scheme used in this
directory. It has two jobs:
    1. collect an activation Hessian for one linear layer
    2. quantize weights column-by-column while propagating quantization error

PIPELINE POSITION:
    BF16 -> NVFP4 -> sub-NVFP4 -> NVFP4 -> BF16
            ^^^^^
            This file is an alternative BF16 -> NVFP4 path that uses GPTQ
            instead of plain nearest-value quantization.

Reference:
    FP-Quant (IST-DASLab), arXiv:2509.23202.

High-level flow:

    calibration activations X
            |
            v
    H = X^T X / n
            |
            v
    invert / factorize H
            |
            v
    quantize one column at a time
            |
            v
    push each column's error into later columns

The Hadamard rotation described in MR-GPTQ is not applied here. This file keeps
the pipeline to the base GPTQ mechanics only.
"""

from __future__ import annotations

import torch

from nvfp4_utils import (
    BLOCK_SIZE,
    FP4_MAX,
    UNIQUE_FP4_VALUES,
    cast_to_fp8,
    floats_to_nibbles,
    pack_fp4,
)


# -----------------------------------------------------------------------------
# Hessian factor preparation
# -----------------------------------------------------------------------------

def _get_hessian_inverse(H: torch.Tensor, rel_damp: float = 0.01) -> torch.Tensor:
    """Regularize the Hessian inverse and return its upper Cholesky factor.

    What:
        Adds diagonal damping to the Hessian, inverts it, symmetrizes the result,
        and computes the upper-triangular Cholesky factor of the inverse.

    Why:
        GPTQ needs curvature information to decide how much the error from one
        quantized column should be pushed into later columns. The diagonal terms
        and cross-terms of the inverse Hessian control that propagation. The
        Cholesky factor is the form used by the existing update equations.

    Math:
        Let the empirical Hessian be ``H``. This function computes

            H_damped = H + lambda * I

        where

            lambda = rel_damp * mean(diag(H))

        Then it forms

            H_inv = inverse(H_damped)

        forces exact symmetry numerically with

            H_inv <- (H_inv + H_inv^T) / 2

        and returns ``R`` such that

            H_inv = R^T R

        because ``torch.linalg.cholesky(..., upper=True)`` returns that upper
        factor.

    Why the branch exists:
        Hessian inversion can fail if the calibration matrix is singular or too
        ill-conditioned. In that case this function falls back to the identity,
        which means "do not use curvature coupling; propagate using a neutral
        metric instead" while preserving the original code path.

    Args:
        H: ``[in_features, in_features]`` Hessian matrix.
        rel_damp: Relative diagonal damping strength.

    Returns:
        ``[in_features, in_features]`` float32 upper-triangular factor.
    """
    d = H.shape[0]

    # The damping scale comes from the average diagonal magnitude so it tracks the
    # overall Hessian scale instead of using a hard-coded absolute constant.
    damp = rel_damp * H.diag().mean()
    H = H.clone()
    H[range(d), range(d)] += damp

    try:
        H_inv = torch.linalg.inv(H.float())

        # Numerical inversion can introduce tiny asymmetry even when the true
        # inverse is symmetric. Cholesky expects a symmetric positive matrix.
        H_inv = (H_inv + H_inv.T) / 2
        H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
    except Exception:
        print("  WARNING: Hessian inversion failed, using identity")
        H_inv_cho = torch.eye(d, device=H.device, dtype=torch.float32)

    return H_inv_cho


# -----------------------------------------------------------------------------
# Single-column NVFP4 quantization
# -----------------------------------------------------------------------------

def _quantize_column_nvfp4(
    w_col: torch.Tensor,
    group_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one weight column using precomputed NVFP4 scales.

    What:
        Maps a column of floating-point weights to the nearest E2M1 values under
        the provided per-row block scale and shared tensor-level global scale.

    Why:
        GPTQ quantizes one column at a time. This helper isolates the actual NVFP4
        nearest-value step so the outer loop can focus on the Hessian-guided error
        updates.

    Math:
        For each row ``r`` in the column:

            effective_scale[r] = group_scale[r] * global_scale
            normalized[r] = w_col[r] / effective_scale[r]
            fp4_code[r] = nearest_E2M1(normalized[r])
            w_q[r] = fp4_code[r] * effective_scale[r]

        ``group_scale`` is per row because each output row has its own local scale
        for the 16-column group containing this column.

    Args:
        w_col: ``[out_features]`` current working column values.
        group_scale: ``[out_features]`` local scale for the 16-column group that
            contains this column.
        global_scale: Scalar tensor shared by the full weight matrix.

    Returns:
        w_q: ``[out_features]`` dequantized values after nearest-E2M1 rounding.
        fp4_codes: ``[out_features]`` float32 E2M1 values before scaling.
    """
    fp4_vals = UNIQUE_FP4_VALUES.to(device=w_col.device)  # [15]

    # The full NVFP4 scale for one element is the product of the tensor-level
    # global scale and the row-specific 16-column local scale.
    effective_scale = group_scale * global_scale  # [out_f]
    w_normalized = w_col / effective_scale.clamp(min=1e-10)  # [out_f]

    # Compare every row value against all 15 unique E2M1 values.
    diffs = w_normalized.unsqueeze(-1) - fp4_vals  # [out_f, 15]
    nearest_idx = diffs.abs().argmin(dim=-1)  # [out_f]
    fp4_codes = fp4_vals[nearest_idx]  # [out_f]

    # Return dequantized weights because GPTQ error propagation is performed in
    # the original weight domain, not in code space.
    w_q = fp4_codes * effective_scale  # [out_f]

    return w_q, fp4_codes


# -----------------------------------------------------------------------------
# Main GPTQ quantization routine
# -----------------------------------------------------------------------------

def gptq_quantize_to_nvfp4(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    block_size: int = 128,
    rel_damp: float = 0.01,
    activation_order: bool = False,
    hadamard_group_size: int | None = None,
    hadamard_mode: str = "undo",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Quantize a weight matrix to NVFP4 with GPTQ error compensation.

    Args:
        weight: [out_features, in_features] BF16 or FP32 tensor.
        hessian: [in_features, in_features] Hessian matrix (X^T X / n).
        block_size: GPTQ processing block width (default 128).
        rel_damp: Relative damping for Hessian regularization.
        activation_order: If True, process columns in descending order of
            H.diag() (most important first). MR-GPTQ Ingredient 2.
            Scales are computed BEFORE permutation (static reordering),
            so no runtime column shuffle is needed at inference.
        hadamard_group_size: If set (e.g. 128), apply block-local Hadamard
            rotation before GPTQ. None = no rotation.
        hadamard_mode: Only used when hadamard_group_size is set.
            "undo"    — Option A: dequant in rotated space, undo rotation,
                        re-quantize in original basis. Output is original-basis
                        NVFP4. TRT-LLM compatible. Costs one extra quant step.
            "rotated" — Option B (MR-GPTQ style): output stays in rotated basis.
                        Activations must be rotated at inference time.
                        No re-quantization error. NOT directly TRT-LLM compatible.

    Returns:
        packed:       [out_f, in_f // 2] uint8 packed FP4.
        block_scales: [out_f, in_f // 16] FP8 E4M3 per-block scales.
        global_scale: scalar float32.
        metadata:     dict with keys:
            "hadamard_group_size": int or None
            "hadamard_mode": "none" | "undo" | "rotated"
            "activation_order": bool
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got {weight.dim()}D")
    out_f, in_f = weight.shape
    if in_f % BLOCK_SIZE != 0:
        raise ValueError(f"in_features ({in_f}) must be divisible by {BLOCK_SIZE}")
    if hadamard_mode not in ("undo", "rotated"):
        raise ValueError(f"hadamard_mode must be 'undo' or 'rotated', got '{hadamard_mode}'")

    device = weight.device
    w = weight.float().clone()  # [out_f, in_f]
    H = hessian.to(device).float()  # [in_f, in_f]

    # --- Hadamard pre-rotation (optional) ---
    use_hadamard = hadamard_group_size is not None
    if use_hadamard:
        from nvfp4_utils import hadamard_rotate, rotate_hessian
        if in_f % hadamard_group_size != 0:
            raise ValueError(
                f"in_features ({in_f}) must be divisible by "
                f"hadamard_group_size ({hadamard_group_size})"
            )
        w = hadamard_rotate(w, hadamard_group_size)
        H = rotate_hessian(H, hadamard_group_size)

    # --- Activation reordering (optional) ---
    # Sort columns by H.diag() descending so important columns are quantized
    # first (when the weight matrix is least disturbed by error propagation).
    # Scales are computed BEFORE permutation — this is MR-GPTQ's "static"
    # reordering: the scale groups stay aligned with the original column layout,
    # so no runtime shuffle is needed.
    perm = None
    perm_inv = None
    perm_to_group = None
    if activation_order:
        perm = torch.argsort(H.diag(), descending=True)
        perm_inv = torch.argsort(perm)

    # --- Compute scales on unpermuted columns ---
    global_scale = cast_to_fp8(w.abs().max() / FP4_MAX)
    w_normalized = w / global_scale  # [out_f, in_f]
    num_groups = in_f // BLOCK_SIZE
    group_scales = torch.zeros(out_f, num_groups, device=device, dtype=torch.float32)
    for g in range(num_groups):
        g_start = g * BLOCK_SIZE
        g_end = g_start + BLOCK_SIZE
        group_amax = w_normalized[:, g_start:g_end].abs().amax(dim=-1)
        group_scales[:, g] = cast_to_fp8(group_amax / FP4_MAX)

    # --- Apply permutation AFTER scale computation ---
    if perm is not None:
        w = w[:, perm]
        H = H[perm][:, perm]
        perm_to_group = perm // BLOCK_SIZE

    H_inv_cho = _get_hessian_inverse(H, rel_damp)  # [in_f, in_f]

    fp4_all = torch.zeros(out_f, in_f, device=device, dtype=torch.float32)

    for c1 in range(0, in_f, block_size):
        c2 = min(c1 + block_size, in_f)
        ncols = c2 - c1

        w_blk = w[:, c1:c2].clone()
        errs = torch.zeros_like(w_blk)
        H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]

        for i in range(ncols):
            col_idx = c1 + i
            w_ci = w_blk[:, i]
            d = H_inv_cho_blk[i, i]

            # Look up the scale group for this column.
            # With activation_order, permuted column col_idx maps back to its
            # original NVFP4 group via perm_to_group.
            if perm is not None:
                g_idx = perm_to_group[col_idx]
            else:
                g_idx = col_idx // BLOCK_SIZE
            g_scale = group_scales[:, g_idx]

            w_q, fp4_code = _quantize_column_nvfp4(w_ci, g_scale, global_scale)
            fp4_all[:, col_idx] = fp4_code

            w[:, col_idx] = w_q
            err = (w_ci - w_q) / d
            w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
            errs[:, i] = err

        w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

    # --- Undo column permutation ---
    if perm is not None:
        fp4_all = fp4_all[:, perm_inv]

    # --- Hadamard post-processing ---
    metadata = {
        "hadamard_group_size": hadamard_group_size,
        "hadamard_mode": "none" if not use_hadamard else hadamard_mode,
        "activation_order": activation_order,
    }

    if use_hadamard and hadamard_mode == "undo":
        from nvfp4_utils import hadamard_rotate, quantize_bf16_to_nvfp4, dequantize_nvfp4_to_bf16
        # Dequant in rotated space -> undo rotation -> re-quantize in original basis
        nibbles_rot = floats_to_nibbles(fp4_all).reshape(out_f, in_f)
        packed_rot = pack_fp4(nibbles_rot)
        scales_rot_fp8 = group_scales.to(torch.float8_e4m3fn)
        w_dequant_rot = dequantize_nvfp4_to_bf16(packed_rot, scales_rot_fp8, global_scale)
        w_dequant_orig = hadamard_rotate(w_dequant_rot.float(), hadamard_group_size)
        packed, block_scales_fp8, global_scale = quantize_bf16_to_nvfp4(w_dequant_orig)
        return packed, block_scales_fp8, global_scale, metadata

    # "rotated" mode or no Hadamard: pack directly
    nibbles = floats_to_nibbles(fp4_all).reshape(out_f, in_f)
    packed = pack_fp4(nibbles)
    block_scales_fp8 = group_scales.to(torch.float8_e4m3fn)

    return packed, block_scales_fp8, global_scale, metadata


# -----------------------------------------------------------------------------
# Hessian collection from calibration data
# -----------------------------------------------------------------------------

def collect_hessian_for_linear(
    model: torch.nn.Module,
    layer_name: str,
    calibration_inputs: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Collect the empirical Hessian ``X^T X / n`` for one ``nn.Linear`` layer.

    What:
        Registers a forward hook on the requested linear layer, records the input
        activations seen during calibration runs, and accumulates their uncentered
        second-moment matrix.

    Why:
        GPTQ needs a curvature estimate for the layer's input dimensions. For a
        linear layer, a common approximation is the activation Gram matrix

            H = X^T X / n

        where each row of ``X`` is one observed input vector.

    Math:
        If calibration produces activation rows ``x_1, ..., x_n`` with dimension
        ``in_features``, then

            H = (1 / n) * sum_t x_t x_t^T

        which is the same as the batched matrix product ``X^T X / n``.

    Why the branches exist:
        - The layer lookup walks a dot-separated attribute path because callers
          refer to nested modules by their HF-style names.
        - If the captured activation is rank-3, it is reshaped from
          ``[batch, seq, in_features]`` to ``[batch * seq, in_features]`` because
          each token position contributes one sample to the Hessian estimate.
        - The final division is skipped when no samples were seen to avoid a
          divide-by-zero.

    Args:
        model: Full model containing the target linear layer.
        layer_name: Dot-separated path to that layer.
        calibration_inputs: List of model inputs used for Hessian estimation.
        device: Device on which to accumulate the Hessian.

    Returns:
        ``[in_features, in_features]`` float32 Hessian estimate.
    """
    parts = layer_name.split(".")
    layer = model
    for p in parts:
        layer = getattr(layer, p)

    in_features = layer.in_features
    H = torch.zeros(in_features, in_features, device=device, dtype=torch.float32)
    n_samples = 0

    def hook_fn(module, input, output):
        """Accumulate ``X^T X`` from the layer input activations.

        The hook receives the module input tuple. Only ``input[0]`` is used
        because ``nn.Linear`` consumes a single activation tensor there.
        """
        nonlocal H, n_samples
        x = input[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])  # [batch * seq, in_features]
        H.addmm_(x.T, x, alpha=1.0)  # [in_features, in_features]
        n_samples += x.shape[0]

    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for inp in calibration_inputs:
            model(inp.to(device))

    handle.remove()

    if n_samples > 0:
        H /= n_samples

    return H

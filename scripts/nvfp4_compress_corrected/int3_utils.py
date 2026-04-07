"""INT3 quantization helpers for BF16 -> INT3 -> BF16 experiments.

This module intentionally keeps the first INT3 implementation simple:

- quantized weights are stored as int8 tensors containing INT3 codes
- no 3-bit packing is performed in this first version
- evaluation always happens through BF16 dequantized checkpoints

That means this code measures quantization quality without introducing an extra
bit-packing format bug surface.
"""

from __future__ import annotations

import torch


INT3_SYM_QMIN = -3
INT3_SYM_QMAX = 3
INT3_ASYM_QMIN = 0
INT3_ASYM_QMAX = 7


def _validate_2d_grouped(weight: torch.Tensor, group_size: int) -> tuple[int, int, int]:
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {weight.dim()}D")
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"in_features ({in_f}) must be divisible by group_size ({group_size})"
        )
    num_groups = in_f // group_size
    return out_f, in_f, num_groups


def _reshape_groups(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Reshape [out_f, in_f] -> [out_f, num_groups, group_size]."""
    out_f, in_f, num_groups = _validate_2d_grouped(weight, group_size)
    return weight.reshape(out_f, num_groups, group_size)


def compute_symmetric_params(weight: torch.Tensor,
                             group_size: int = 128) -> torch.Tensor:
    """Compute per-group symmetric INT3 scales.

    The symmetric path uses signed integer codes in [-3, 3].
    One code remains unused, which is common when people want a truly symmetric
    dequantization around zero while still storing values inside an INT3 budget.
    """
    groups = _reshape_groups(weight.float(), group_size)  # [out_f, G, group]
    amax = groups.abs().amax(dim=-1)  # [out_f, G]
    scale = (amax / max(abs(INT3_SYM_QMIN), abs(INT3_SYM_QMAX))).clamp(min=1e-10)
    return scale


def compute_asymmetric_params(weight: torch.Tensor,
                              group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group asymmetric INT3 scale and zero-point.

    This follows the usual affine quantization pattern:

        q = clamp(round(w / scale) + zero, qmin, qmax)
        w_hat = (q - zero) * scale

    with qmin=0, qmax=7 for INT3.
    """
    groups = _reshape_groups(weight.float(), group_size)  # [out_f, G, group]
    w_min = groups.amin(dim=-1)  # [out_f, G]
    w_max = groups.amax(dim=-1)  # [out_f, G]
    scale = ((w_max - w_min) / (INT3_ASYM_QMAX - INT3_ASYM_QMIN)).clamp(min=1e-10)
    zero = torch.round(INT3_ASYM_QMIN - (w_min / scale)).clamp(INT3_ASYM_QMIN,
                                                                INT3_ASYM_QMAX)
    return scale, zero.to(torch.int8)


def quantize_symmetric_int3(weight: torch.Tensor,
                            scale: torch.Tensor,
                            group_size: int = 128) -> torch.Tensor:
    """Quantize BF16/FP32 weights to signed symmetric INT3 codes in int8 storage."""
    groups = _reshape_groups(weight.float(), group_size)  # [out_f, G, group]
    scale_3d = scale.unsqueeze(-1)  # [out_f, G, 1]
    q = torch.round(groups / scale_3d).clamp(INT3_SYM_QMIN, INT3_SYM_QMAX)
    return q.to(torch.int8).reshape(weight.shape)


def dequantize_symmetric_int3(qweight: torch.Tensor,
                              scale: torch.Tensor,
                              group_size: int = 128) -> torch.Tensor:
    """Dequantize signed symmetric INT3 codes back to BF16."""
    q = _reshape_groups(qweight.float(), group_size)  # [out_f, G, group]
    scale_3d = scale.unsqueeze(-1)  # [out_f, G, 1]
    return (q * scale_3d).reshape(qweight.shape).to(torch.bfloat16)


def quantize_asymmetric_int3(weight: torch.Tensor,
                             scale: torch.Tensor,
                             zero: torch.Tensor,
                             group_size: int = 128) -> torch.Tensor:
    """Quantize BF16/FP32 weights to affine INT3 codes in [0, 7], stored in int8."""
    groups = _reshape_groups(weight.float(), group_size)  # [out_f, G, group]
    scale_3d = scale.unsqueeze(-1)
    zero_3d = zero.float().unsqueeze(-1)
    q = torch.round(groups / scale_3d + zero_3d).clamp(INT3_ASYM_QMIN, INT3_ASYM_QMAX)
    return q.to(torch.int8).reshape(weight.shape)


def dequantize_asymmetric_int3(qweight: torch.Tensor,
                               scale: torch.Tensor,
                               zero: torch.Tensor,
                               group_size: int = 128) -> torch.Tensor:
    """Dequantize affine INT3 codes back to BF16."""
    q = _reshape_groups(qweight.float(), group_size)
    scale_3d = scale.unsqueeze(-1)
    zero_3d = zero.float().unsqueeze(-1)
    return ((q - zero_3d) * scale_3d).reshape(qweight.shape).to(torch.bfloat16)


def rtn_quantize_to_int3(weight: torch.Tensor,
                         scheme: str = "sym",
                         group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Round-to-nearest INT3 quantization.

    Returns:
        qweight: int8 tensor with INT3 codes
        scale:   [out_f, num_groups] float32
        zero:    None for symmetric, int8 [out_f, num_groups] for asymmetric
    """
    if scheme == "sym":
        scale = compute_symmetric_params(weight, group_size)
        qweight = quantize_symmetric_int3(weight, scale, group_size)
        return qweight, scale, None
    if scheme == "asym":
        scale, zero = compute_asymmetric_params(weight, group_size)
        qweight = quantize_asymmetric_int3(weight, scale, zero, group_size)
        return qweight, scale, zero
    raise ValueError(f"Unsupported scheme: {scheme}")


def dequantize_int3(qweight: torch.Tensor,
                    scale: torch.Tensor,
                    zero: torch.Tensor | None,
                    scheme: str,
                    group_size: int = 128) -> torch.Tensor:
    """Unified dequantization helper for symmetric / asymmetric INT3."""
    if scheme == "sym":
        return dequantize_symmetric_int3(qweight, scale, group_size)
    if scheme == "asym":
        if zero is None:
            raise ValueError("asym dequantization requires zero-point tensor")
        return dequantize_asymmetric_int3(qweight, scale, zero, group_size)
    raise ValueError(f"Unsupported scheme: {scheme}")


def _get_hessian_inverse(H: torch.Tensor, rel_damp: float = 0.01) -> torch.Tensor:
    """Same GPTQ Hessian preparation pattern used in the NVFP4 path."""
    d = H.shape[0]
    damp = rel_damp * H.diag().mean()
    H = H.clone()
    H[range(d), range(d)] += damp
    try:
        H_inv = torch.linalg.inv(H.float())
        H_inv = (H_inv + H_inv.T) / 2
        return torch.linalg.cholesky(H_inv, upper=True)
    except Exception:
        return torch.eye(d, device=H.device, dtype=torch.float32)


def _quantize_column_int3(w_col: torch.Tensor,
                          scale_col: torch.Tensor,
                          zero_col: torch.Tensor | None,
                          scheme: str) -> torch.Tensor:
    """Quantize one column and return dequantized values for GPTQ error updates."""
    if scheme == "sym":
        q = torch.round(w_col / scale_col.clamp(min=1e-10)).clamp(INT3_SYM_QMIN, INT3_SYM_QMAX)
        return (q * scale_col).float()
    if scheme == "asym":
        if zero_col is None:
            raise ValueError("asym scheme requires zero_col")
        q = torch.round(w_col / scale_col.clamp(min=1e-10) + zero_col.float()).clamp(
            INT3_ASYM_QMIN, INT3_ASYM_QMAX)
        return ((q - zero_col.float()) * scale_col).float()
    raise ValueError(f"Unsupported scheme: {scheme}")


def gptq_quantize_to_int3(weight: torch.Tensor,
                          hessian: torch.Tensor,
                          scheme: str = "sym",
                          group_size: int = 128,
                          block_size: int = 128,
                          rel_damp: float = 0.01,
                          activation_order: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
    """GPTQ INT3 quantization.

    The GPTQ loop is the same idea as the NVFP4 path: quantize one column,
    compute error, and push it into future columns using the inverse-Hessian
    Cholesky factor. The quantizer itself is INT3 instead of E2M1 FP4.
    """
    out_f, in_f, num_groups = _validate_2d_grouped(weight, group_size)
    device = weight.device

    w = weight.float().clone()  # [out_f, in_f]
    H = hessian.to(device).float()  # [in_f, in_f]

    if scheme == "sym":
        group_scale = compute_symmetric_params(w, group_size)  # [out_f, G]
        group_zero = None
    elif scheme == "asym":
        group_scale, group_zero = compute_asymmetric_params(w, group_size)
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    perm = None
    perm_inv = None
    perm_to_group = None
    if activation_order:
        perm = torch.argsort(H.diag(), descending=True)
        perm_inv = torch.argsort(perm)
        w = w[:, perm]
        H = H[perm][:, perm]
        perm_to_group = perm // group_size

    H_inv_cho = _get_hessian_inverse(H, rel_damp)

    dequant_w = torch.zeros_like(w)  # [out_f, in_f]
    for c1 in range(0, in_f, block_size):
        c2 = min(c1 + block_size, in_f)
        ncols = c2 - c1

        w_blk = w[:, c1:c2].clone()  # [out_f, ncols]
        errs = torch.zeros_like(w_blk)  # [out_f, ncols]
        H_blk = H_inv_cho[c1:c2, c1:c2]  # [ncols, ncols]

        for i in range(ncols):
            col_idx = c1 + i
            g_idx = perm_to_group[col_idx] if perm_to_group is not None else col_idx // group_size
            scale_col = group_scale[:, g_idx]  # [out_f]
            zero_col = None if group_zero is None else group_zero[:, g_idx]  # [out_f] or None

            w_ci = w_blk[:, i]  # [out_f]
            d = H_blk[i, i]
            w_q = _quantize_column_int3(w_ci, scale_col, zero_col, scheme)  # [out_f]
            dequant_w[:, col_idx] = w_q

            w[:, col_idx] = w_q
            err = (w_ci - w_q) / d
            w_blk[:, i:].addr_(err, H_blk[i, i:], alpha=-1)
            errs[:, i] = err

        w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

    if perm_inv is not None:
        dequant_w = dequant_w[:, perm_inv]

    # Final discrete qweight from the dequantized matrix, using the original group params.
    if scheme == "sym":
        qweight = quantize_symmetric_int3(dequant_w, group_scale, group_size)
        zero = None
    else:
        qweight = quantize_asymmetric_int3(dequant_w, group_scale, group_zero, group_size)
        zero = group_zero

    meta = {
        "scheme": scheme,
        "group_size": group_size,
        "activation_order": activation_order,
    }
    return qweight, group_scale, zero, meta

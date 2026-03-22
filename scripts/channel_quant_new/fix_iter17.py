#!/usr/bin/env python3
"""Fix iter17 split_linear_3tier to use row-moving instead of zero-padding."""
import re

path = "/workspace/channel_quant_new/exact_explore_iter17_fp8_centric_3tier.py"
with open(path, "r") as f:
    code = f.read()

old = '''def split_linear_3tier(x, w, tiers, device):
    """3-tier linear: BF16 + FP8 + NVFP4 split."""
    tiers_d = tiers.to(device)
    n_out = w.shape[0]
    out = torch.zeros(x.shape[0], n_out, dtype=x.dtype, device=device)
    
    for tier_val, linear_fn, gran in [
        (2, ee.bf16_linear, 1),
        (1, ee.fp8_linear, 16),
        (0, ee.nvfp4_linear, 32),
    ]:
        mask = tiers_d == tier_val
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        w_sub = w[idx]
        n = w_sub.shape[0]
        pad = (gran - n % gran) % gran if gran > 1 else 0
        if pad and tier_val < 2:
            w_sub = F.pad(w_sub, (0, 0, 0, pad))
        o = linear_fn(x, w_sub) if tier_val > 0 else ee.nvfp4_linear(x, w_sub)
        if pad and tier_val < 2:
            o = o[:, :n]
        out.index_copy_(1, idx, o)
    return out'''

new = '''def split_linear_3tier(x, w, tiers, device):
    """3-tier linear with row-moving alignment (no zero-padding)."""
    tiers_d = tiers.to(device)
    n_out = w.shape[0]
    out = torch.zeros(x.shape[0], n_out, dtype=x.dtype, device=device)

    bf16_rows = (tiers_d == 2).nonzero(as_tuple=True)[0]
    fp8_rows = (tiers_d == 1).nonzero(as_tuple=True)[0]
    nvfp4_rows = (tiers_d == 0).nonzero(as_tuple=True)[0]

    # Alignment: move excess FP8 rows to NVFP4
    if fp8_rows.numel() > 0 and fp8_rows.numel() % 16 != 0:
        excess = fp8_rows.numel() % 16
        moved = fp8_rows[-excess:]
        nvfp4_rows = torch.cat([nvfp4_rows, moved])
        fp8_rows = fp8_rows[:-excess]

    # Alignment: move excess NVFP4 rows to FP8
    if nvfp4_rows.numel() > 0 and nvfp4_rows.numel() % 32 != 0:
        deficit = nvfp4_rows.numel() % 32
        moved = nvfp4_rows[-deficit:]
        fp8_rows = torch.cat([fp8_rows, moved])
        nvfp4_rows = nvfp4_rows[:-deficit]

    # Re-check FP8 alignment
    if fp8_rows.numel() > 0 and fp8_rows.numel() % 16 != 0:
        excess = fp8_rows.numel() % 16
        moved = fp8_rows[-excess:]
        bf16_rows = torch.cat([bf16_rows, moved])
        fp8_rows = fp8_rows[:-excess]

    if bf16_rows.numel() > 0:
        out.index_copy_(1, bf16_rows, ee.bf16_linear(x, w[bf16_rows]))
    if fp8_rows.numel() > 0:
        out.index_copy_(1, fp8_rows, ee.fp8_linear(x, w[fp8_rows]))
    if nvfp4_rows.numel() > 0:
        out.index_copy_(1, nvfp4_rows, ee.nvfp4_linear(x, w[nvfp4_rows]))
    return out'''

if old in code:
    code = code.replace(old, new)
    with open(path, "w") as f:
        f.write(code)
    print("FIXED: split_linear_3tier now uses row-moving alignment")
else:
    print("ERROR: old function not found in code")

#!/usr/bin/env python3
"""Diagnose codebook mapping issues."""
import torch

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

CODEBOOKS = {
    'nvfp4_full': [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
    '3bit_uniform': [-6, -4, -2, 0, 2, 4, 6],
    '3bit_dense': [-6, -2, -1, 0, 1, 2, 6],
    '3bit_truncate': [-4, -2, -1, 0, 1, 2, 4],
}

def build_code_lut(sub_values):
    """Build 16-entry LUT: lut[old_code] = new_code."""
    cb = torch.tensor(sub_values, dtype=torch.float32)
    lut = torch.zeros(16, dtype=torch.uint8)
    
    for src_code in range(16):
        src_val = E2M1_TABLE[src_code]
        dists = (cb - src_val).abs()
        nearest_val = cb[dists.argmin()].item()
        
        best_code = None
        best_dist = float('inf')
        for dst_code in range(16):
            if abs(E2M1_TABLE[dst_code].item() - nearest_val) < 1e-6:
                d = abs(E2M1_TABLE[dst_code].item() - src_val)
                if d < best_dist or (d == best_dist and best_code is not None and dst_code < best_code):
                    best_dist = d
                    best_code = dst_code
        lut[src_code] = best_code
    
    return lut

print("Codebook Mapping Analysis")
print("="*70)

for cb_name, cb_vals in CODEBOOKS.items():
    print(f"\n{cb_name}:")
    print(f"  Values: {cb_vals}")
    
    lut = build_code_lut(cb_vals)
    
    print(f"  LUT mapping:")
    for src in range(16):
        src_val = E2M1_TABLE[src].item()
        dst = lut[src].item()
        dst_val = E2M1_TABLE[dst].item()
        error = abs(src_val - dst_val)
        print(f"    {src:2d} ({src_val:+6.1f}) -> {dst:2d} ({dst_val:+6.1f}) [error={error:.1f}]")
    
    # Check for NaN or Inf
    if torch.isnan(lut.float()).any() or torch.isinf(lut.float()).any():
        print(f"  ✗ WARNING: NaN or Inf in LUT!")
    else:
        print(f"  ✓ LUT is valid")


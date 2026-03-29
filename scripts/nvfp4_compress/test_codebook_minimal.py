#!/usr/bin/env python3
"""Minimal test of codebook compression on a single expert."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")

import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

BLOCK_SIZE = 16

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

def unpack_fp4_codes(packed):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)

def repack_fp4_codes(codes):
    M, K = codes.shape
    codes = codes.view(M, K // 2, 2)
    return (codes[:, :, 0] | (codes[:, :, 1] << 4)).to(torch.uint8)

def build_code_lut(sub_values):
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

def test_codebook(codebook_name, codebook_vals):
    print(f"\nTesting {codebook_name}...")
    
    # Create a random weight matrix
    weight_bf16 = torch.randn(256, 2048, dtype=torch.bfloat16, device='cuda')
    input_tensor = torch.randn(1, 2048, dtype=torch.bfloat16, device='cuda')
    
    # Quantize to NVFP4
    s_w = fp4_global_scale(weight_bf16).to(torch.float32)
    packed_orig, block_scales = torch.ops.trtllm.fp4_quantize(weight_bf16, s_w, BLOCK_SIZE, False)
    
    # Build LUT and remap codes
    lut = build_code_lut(codebook_vals).to('cuda')
    codes = unpack_fp4_codes(packed_orig)
    codes_mapped = lut[codes.long()]
    packed_new = repack_fp4_codes(codes_mapped)
    
    # Run NVFP4 linear with original and remapped codes
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    s_in = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in * s_w)).to(torch.float32)
    
    try:
        out_orig = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            input_2d, packed_orig, bias=None,
            input_scale=s_in, weight_scale=block_scales, alpha=alpha,
        )
        print(f"  Original output: shape={out_orig.shape}, dtype={out_orig.dtype}")
        print(f"    min={out_orig.min():.4f}, max={out_orig.max():.4f}, mean={out_orig.mean():.4f}")
        if torch.isnan(out_orig).any() or torch.isinf(out_orig).any():
            print(f"    ✗ Contains NaN/Inf!")
        else:
            print(f"    ✓ Valid")
    except Exception as e:
        print(f"  ✗ Error with original: {e}")
    
    try:
        out_mapped = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            input_2d, packed_new, bias=None,
            input_scale=s_in, weight_scale=block_scales, alpha=alpha,
        )
        print(f"  Mapped output: shape={out_mapped.shape}, dtype={out_mapped.dtype}")
        print(f"    min={out_mapped.min():.4f}, max={out_mapped.max():.4f}, mean={out_mapped.mean():.4f}")
        if torch.isnan(out_mapped).any() or torch.isinf(out_mapped).any():
            print(f"    ✗ Contains NaN/Inf!")
        else:
            print(f"    ✓ Valid")
    except Exception as e:
        print(f"  ✗ Error with mapped: {e}")

if __name__ == '__main__':
    test_codebook('nvfp4_full', [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])
    test_codebook('3bit_uniform', [-6, -4, -2, 0, 2, 4, 6])
    test_codebook('3bit_dense', [-6, -2, -1, 0, 1, 2, 6])


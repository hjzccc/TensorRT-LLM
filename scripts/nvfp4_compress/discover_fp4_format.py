#!/usr/bin/env python3
"""Discover the exact FP4 packed format used by torch.ops.trtllm.fp4_quantize.

Goal: determine nibble order, byte layout, and the 4-bit code ↔ FP4 value mapping.
"""
import torch
import numpy as np
import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

BLOCK_SIZE = 16

# FP4 E2M1 values (15 unique, but 16 codes including ±0)
FP4_VALUES = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]

def test_known_values():
    """Create weights with known FP4 values and see what codes fp4_quantize produces."""
    print("=" * 60)
    print("TEST 1: Single block with known values")
    print("=" * 60)

    # Create a 1×16 weight where each element is a known FP4 value
    # Use block_scale = 1.0 (amax = 6, scale = fp8(6/6) = fp8(1.0) = 1.0)
    # global_scale should be chosen so weight * global_scale = desired values
    
    # Let's make a weight where elements are FP4 values directly
    # We need weight * global_scale to give us the FP4 values after block normalization
    # If block contains values in [-6, 6], block_amax=6, block_scale=fp8(1.0)=1.0
    # normalized = weight * gs / block_scale = weight * gs
    # For normalized to be FP4 values, we need weight * gs = FP4 values
    
    # Simple approach: create weight = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0]
    # Then gs = fp4_global_scale(weight) will give us something, and fp4_quantize should round each to itself
    
    vals = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0]], dtype=torch.bfloat16)
    gs = fp4_global_scale(vals).to(torch.float32)
    print(f"Weight: {vals[0].tolist()}")
    print(f"Global scale: {gs.item()}")
    print(f"Weight * gs: {(vals.float() * gs)[0].tolist()}")
    
    packed, block_scales = torch.ops.trtllm.fp4_quantize(vals, gs, BLOCK_SIZE, False)
    print(f"\nPacked shape: {packed.shape}, dtype: {packed.dtype}")
    print(f"Block scales shape: {block_scales.shape}, dtype: {block_scales.dtype}")
    print(f"Block scale value: {block_scales.float()}")
    
    # Examine packed bytes
    packed_np = packed.cpu().numpy()
    print(f"\nPacked bytes (hex):")
    for i in range(packed_np.shape[1]):
        byte_val = packed_np[0, i]
        low = byte_val & 0x0F
        high = (byte_val >> 4) & 0x0F
        print(f"  Byte {i}: 0x{byte_val:02x} → low_nibble={low:04b}({low}) high_nibble={high:04b}({high})")

def test_sequential():
    """Create weight with values that make it easy to identify element-to-nibble mapping."""
    print("\n" + "=" * 60)
    print("TEST 2: Sequential distinct values")
    print("=" * 60)
    
    # Use just positive distinct values to make identification easier
    # 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, 0.5, 1, 1.5, 2, 3, 4, 6
    vals = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, 0.5, 1, 1.5, 2, 3, 4, 6]], dtype=torch.bfloat16)
    gs = fp4_global_scale(vals).to(torch.float32)
    packed, block_scales = torch.ops.trtllm.fp4_quantize(vals, gs, BLOCK_SIZE, False)
    
    packed_np = packed.cpu().numpy()
    print(f"Weight: {vals[0].tolist()}")
    print(f"Global scale: {gs.item()}")
    bs_val = block_scales.float().item()
    print(f"Block scale: {bs_val}")
    scaled = vals.float() * gs
    print(f"Weight * gs: {scaled[0].tolist()}")
    normalized = scaled / bs_val
    print(f"Normalized: {normalized[0].tolist()}")
    
    print(f"\nPacked bytes:")
    for i in range(packed_np.shape[1]):
        byte_val = packed_np[0, i]
        low = byte_val & 0x0F
        high = (byte_val >> 4) & 0x0F
        print(f"  Byte {i}: 0x{byte_val:02x} → low={low} high={high}")

def test_all_zero_and_max():
    """Test with all zeros and all 6s to identify code for 0 and 6."""
    print("\n" + "=" * 60)
    print("TEST 3: All same values")
    print("=" * 60)
    
    for val in [0, 0.5, 1, 1.5, 2, 3, 4, 6, -6]:
        w = torch.full((1, 16), val, dtype=torch.bfloat16)
        if val == 0:
            # Can't quantize all zeros meaningfully, add a tiny nonzero
            w[0, 0] = 6.0  # force nonzero scale
        gs = fp4_global_scale(w).to(torch.float32)
        packed, bs = torch.ops.trtllm.fp4_quantize(w, gs, BLOCK_SIZE, False)
        packed_np = packed.cpu().numpy()
        codes = []
        for i in range(packed_np.shape[1]):
            byte_val = packed_np[0, i]
            codes.append(byte_val & 0x0F)
            codes.append((byte_val >> 4) & 0x0F)
        print(f"  val={val:+5.1f} → codes: {codes}")

def test_single_nonzero():
    """Put a single nonzero value at each position to map position to nibble."""
    print("\n" + "=" * 60)
    print("TEST 4: Single nonzero at each position")
    print("=" * 60)
    
    for pos in range(16):
        w = torch.zeros(1, 16, dtype=torch.bfloat16)
        w[0, pos] = 6.0  # max positive value
        gs = fp4_global_scale(w).to(torch.float32)
        packed, bs = torch.ops.trtllm.fp4_quantize(w, gs, BLOCK_SIZE, False)
        packed_np = packed.cpu().numpy()
        codes = []
        for i in range(packed_np.shape[1]):
            byte_val = packed_np[0, i]
            codes.append(byte_val & 0x0F)
            codes.append((byte_val >> 4) & 0x0F)
        nonzero_positions = [i for i, c in enumerate(codes) if c != 0]
        print(f"  Weight pos {pos:2d} → nonzero nibble positions: {nonzero_positions}, codes: {codes}")

def test_roundtrip():
    """Verify that unpacking and repacking produces the same result."""
    print("\n" + "=" * 60)
    print("TEST 5: Round-trip verification")
    print("=" * 60)
    
    # Random-ish weight
    torch.manual_seed(42)
    w = torch.randn(4, 32, dtype=torch.bfloat16)
    gs = fp4_global_scale(w).to(torch.float32)
    packed, bs = torch.ops.trtllm.fp4_quantize(w, gs, BLOCK_SIZE, False)
    
    print(f"Weight shape: {w.shape}")
    print(f"Packed shape: {packed.shape}")
    print(f"Block scales shape: {bs.shape}")
    print(f"Global scale: {gs.item()}")
    
    # Try using the kernel
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale as fp4_gs
    inp = torch.randn(2, 32, dtype=torch.bfloat16, device='cuda')
    w_cuda = w.cuda()
    gs_cuda = fp4_gs(w_cuda).to(torch.float32)
    s_in = fp4_gs(inp.reshape(-1, inp.shape[-1])).to(torch.float32)
    packed_cuda, bs_cuda = torch.ops.trtllm.fp4_quantize(w_cuda, gs_cuda, BLOCK_SIZE, False)
    alpha = (1.0 / (s_in * gs_cuda)).to(torch.float32)
    
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        inp.reshape(-1, inp.shape[-1]), packed_cuda, bias=None,
        input_scale=s_in, weight_scale=bs_cuda, alpha=alpha,
    )
    print(f"Kernel output shape: {out.shape}")
    print(f"Kernel output sample: {out[0, :8].tolist()}")
    
    # Compare with BF16 reference
    ref = torch.nn.functional.linear(inp, w_cuda)
    print(f"BF16 ref sample:     {ref[0, :8].tolist()}")
    print(f"Max diff: {(out.float() - ref.float()).abs().max().item():.6f}")

def test_dequant_formula():
    """Figure out dequant by comparing kernel output to manual calculation."""
    print("\n" + "=" * 60)
    print("TEST 6: Dequantize formula verification")
    print("=" * 60)
    
    # Create a simple weight and input
    w = torch.tensor([[6, 4, 3, 2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -3, -4, -6, 0]], dtype=torch.bfloat16).cuda()
    inp = torch.ones(1, 16, dtype=torch.bfloat16, device='cuda')
    
    gs = fp4_global_scale(w).to(torch.float32)
    s_in = fp4_global_scale(inp).to(torch.float32)
    packed, bs = torch.ops.trtllm.fp4_quantize(w, gs, BLOCK_SIZE, False)
    alpha = (1.0 / (s_in * gs)).to(torch.float32)
    
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        inp, packed, bias=None, input_scale=s_in, weight_scale=bs, alpha=alpha,
    )
    
    # Manual: with input = [1,1,...,1], output = sum of dequantized weights
    # dequant = fp4_val * block_scale * alpha * input_scale = fp4_val * block_scale / gs
    # kernel output = sum(input[i] * s_in * dequant_weight[i] * ... )
    
    ref_bf16 = torch.nn.functional.linear(inp, w)
    print(f"BF16 sum of weight: {w.float().sum().item()}")
    print(f"BF16 linear output: {ref_bf16.item()}")
    print(f"NVFP4 linear output: {out.item()}")
    print(f"Block scale: {bs.float().item()}")
    print(f"Global scale: {gs.item()}")
    print(f"Input scale: {s_in.item()}")
    print(f"Alpha: {alpha.item()}")

if __name__ == '__main__':
    test_known_values()
    test_sequential()
    test_all_zero_and_max()
    test_single_nonzero()
    test_roundtrip()
    test_dequant_formula()

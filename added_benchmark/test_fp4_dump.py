#!/usr/bin/env python3
"""
Direct FP4 byte comparison: run fused vs unfused, skip GEMM2, compare fc1_result_ raw bytes.

Strategy: Run the full pipeline but use a specially constructed fc2 weight = identity-ish 
to isolate fc1_result_ contribution. Actually simpler: compare final outputs and work backward.

Better strategy: Use the runner's internal state. After GEMM1+activation, the fc1_result_ 
buffer has FP4 data. Both paths write to the same buffer. If we can read it...

Simplest strategy: Make fc2 weights = zero. Then output = 0 regardless of fc1_result_.
That doesn't help.

Actual simplest: Make fc2 = a single column that sums all elements. 
Then output = sum(dequantize(fp4_row)). If fp4 bytes differ, sums differ.

Let's do something even simpler: set inter_size=128 (minimum for NVFP4), 
so each expert has exactly one SF block, and compare outputs more carefully.

ACTUALLY: Let's just print the FC2 inputs from Python side.
We can't directly access fc1_result_, but we can use SKIP_ACTIVATION + FORCE_UNFUSED
to control what happens. 

NEW APPROACH: Run GEMM1 only (skip activation, skip GEMM2) and read back fc1_result_ 
as raw bytes. But we can't easily do this from Python.

BEST APPROACH: Use the debug printf to print ALL FP4 uint32 values for a given row.
"""
import torch
import sys
import os

torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")

DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768
N = INTER * 2
SWIGLU = 5

def quantize_for_trtllm(w_bf16, global_sf):
    rows, cols = w_bf16.shape
    if isinstance(global_sf, float):
        global_sf = torch.tensor(global_sf, dtype=torch.float32, device=w_bf16.device)
    packed_u8, sf_u8 = torch.ops.trtllm.fp4_quantize(w_bf16, global_sf, 16, False, False)
    w_int64 = packed_u8.view(torch.int64).reshape(rows, cols // 16)
    sf_2d = sf_u8.reshape(rows, cols // 16)
    sf_interleaved = torch.ops.trtllm.block_scale_interleave(sf_2d)
    sf_int32 = sf_interleaved.view(torch.int32).reshape(rows, cols // 16 // 4)
    return w_int64, sf_int32

def make_simple_test():
    """Create a minimal test: 1 token routed to 1 expert, identity-like fc2."""
    torch.manual_seed(42)
    num_tokens = 1
    
    # Route to expert 0 only  
    router = torch.zeros(num_tokens, NUM_EXPERTS, dtype=DTYPE, device=DEVICE)
    router[0, 0] = 1.0
    for j in range(1, TOP_K):
        router[0, j] = 0.5  # Route to experts 0..7
    scores = torch.softmax(router.float(), dim=-1)
    topk_scores, topk_idx = torch.topk(scores, TOP_K, dim=-1)
    topk_scores = (topk_scores / topk_scores.sum(dim=-1, keepdim=True)).float()
    
    input_tensor = torch.randn(num_tokens, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.1
    
    # Simple weights: small random
    fc1_weights_gate = torch.randn(NUM_EXPERTS, INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01
    fc1_weights_up = torch.randn(NUM_EXPERTS, INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01
    
    # FC2: Make it diagonal-ish to see fc1 output directly
    # fc2 shape: [NUM_EXPERTS, HIDDEN, INTER]
    # Each expert's fc2 is HIDDEN x INTER, maps INTER -> HIDDEN
    # Use identity-like: fc2[e, i, i] = 1 for i < min(HIDDEN, INTER)
    fc2_bf16 = torch.zeros(NUM_EXPERTS, HIDDEN, INTER, dtype=DTYPE, device=DEVICE)
    for i in range(min(HIDDEN, INTER)):
        fc2_bf16[:, i, i] = 1.0  # Identity mapping for first INTER dims
    
    return {
        'num_tokens': num_tokens,
        'input_tensor': input_tensor,
        'topk_idx': topk_idx,
        'topk_scores': topk_scores,
        'fc1_weights_gate': fc1_weights_gate,
        'fc1_weights_up': fc1_weights_up,
        'fc2_bf16': fc2_bf16,
    }

def run_with_mode(inputs, mode='fused'):
    fc1_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
    fc2_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
    fc1_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    fc2_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    if mode == 'fused':
        fc1_bf16 = torch.zeros(NUM_EXPERTS, N, HIDDEN, dtype=DTYPE, device=DEVICE)
        fc1_bf16[:, 0::2, :] = inputs['fc1_weights_up']
        fc1_bf16[:, 1::2, :] = inputs['fc1_weights_gate']
    else:
        fc1_bf16 = torch.cat([inputs['fc1_weights_up'], inputs['fc1_weights_gate']], dim=1)

    fc1_w_list, fc1_sf_list = [], []
    for e in range(NUM_EXPERTS):
        w, sf = quantize_for_trtllm(fc1_bf16[e], 1.0)
        fc1_w_list.append(w)
        fc1_sf_list.append(sf)
    fc1_w = torch.stack(fc1_w_list)
    fc1_sf = torch.stack(fc1_sf_list)

    fc2_w_list, fc2_sf_list = [], []
    for e in range(NUM_EXPERTS):
        w, sf = quantize_for_trtllm(inputs['fc2_bf16'][e], 1.0)
        fc2_w_list.append(w)
        fc2_sf_list.append(sf)
    fc2_w = torch.stack(fc2_w_list)
    fc2_sf = torch.stack(fc2_sf_list)

    quant_scales = [fc1_act_g, fc1_sf, fc1_g, fc2_act_g, fc2_sf, fc2_g]
    isf = torch.ones(inputs['num_tokens'], HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

    env_backup = os.environ.get("FORCE_UNFUSED_SWIGLU")
    if mode == 'unfused':
        os.environ["FORCE_UNFUSED_SWIGLU"] = "1"
    elif "FORCE_UNFUSED_SWIGLU" in os.environ:
        del os.environ["FORCE_UNFUSED_SWIGLU"]

    runner = torch.classes.trtllm.FusedMoeRunner(
        DTYPE, torch.int64, DTYPE, False, False, False, False, True)
    runner.set_dual_tile_profiles([0, 0, 0, 0], 32)
    output = runner.run_moe_dual_tile(
        inputs['input_tensor'], inputs['topk_idx'].int(), inputs['topk_scores'],
        fc1_w, None, fc2_w, None,
        quant_scales, isf, False, None, None, None,
        1, 0, 1, 0, False, SWIGLU, None, None, None)
    torch.cuda.synchronize()

    if env_backup is not None:
        os.environ["FORCE_UNFUSED_SWIGLU"] = env_backup
    elif "FORCE_UNFUSED_SWIGLU" in os.environ:
        del os.environ["FORCE_UNFUSED_SWIGLU"]

    return output

if __name__ == "__main__":
    inputs = make_simple_test()
    
    print("=" * 60)
    print("Running with identity FC2 to isolate FC1 output...")
    print("=" * 60)
    
    print("\n--- FUSED ---")
    output_fused = run_with_mode(inputs, 'fused')
    print(f"Shape: {output_fused.shape}")
    print(f"abs_mean: {output_fused.abs().mean():.6f}")
    print(f"[0,:20]: {output_fused[0,:20].tolist()}")
    
    print("\n--- UNFUSED ---")
    output_unfused = run_with_mode(inputs, 'unfused')
    print(f"Shape: {output_unfused.shape}")
    print(f"abs_mean: {output_unfused.abs().mean():.6f}")
    print(f"[0,:20]: {output_unfused[0,:20].tolist()}")
    
    print("\n--- COMPARISON ---")
    # With identity FC2, the first INTER=768 elements of output should 
    # directly reflect the dequantized FC1 SwiGLU output.
    # The rest should be zero (since FC2 is zero beyond INTER columns).
    fused_inter = output_fused[0, :INTER]
    unfused_inter = output_unfused[0, :INTER]
    
    ratio = fused_inter.abs().mean() / (unfused_inter.abs().mean() + 1e-8)
    print(f"Fused inter abs_mean: {fused_inter.abs().mean():.6f}")
    print(f"Unfused inter abs_mean: {unfused_inter.abs().mean():.6f}")
    print(f"Ratio: {ratio:.4f}")
    
    # Check element-wise
    diff = (fused_inter - unfused_inter).abs()
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    
    # Print first 20 elements side by side
    print("\nFirst 20 elements:")
    print(f"{'Index':>5} {'Fused':>12} {'Unfused':>12} {'Diff':>12}")
    for i in range(20):
        f, u = fused_inter[i].item(), unfused_inter[i].item()
        print(f"{i:5d} {f:12.6f} {u:12.6f} {abs(f-u):12.6f}")

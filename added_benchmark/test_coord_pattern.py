#!/usr/bin/env python3
"""
Coordinate mapping diagnostic: visitor writes (compact_col + 1) as float.
Expected: fc1_result_[row, col] = col + 1 for all active (row, col) pairs.
If all 32 expanded rows show the pattern 1,2,3,...,768 then coordinates are correct.
If only some rows show it, those are the only CTAs executing the visitor.
"""
import torch

torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")

torch.manual_seed(42)
DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768
N = INTER * 2  # 1536
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


num_tokens = 4
expanded_rows = num_tokens * TOP_K

# Router
router = torch.zeros(num_tokens, NUM_EXPERTS, dtype=DTYPE, device=DEVICE)
for i in range(num_tokens):
    for j in range(TOP_K):
        router[i, i * TOP_K + j] = 1.0 + 0.1 * j
scores = torch.softmax(router.float(), dim=-1)
topk_scores, topk_idx = torch.topk(scores, TOP_K, dim=-1)
topk_scores = (topk_scores / topk_scores.sum(dim=-1, keepdim=True)).float()

input_tensor = torch.randn(num_tokens, HIDDEN, dtype=DTYPE, device=DEVICE)

# Create weights (interleaved)
fc1_weights_gate = torch.randn(NUM_EXPERTS, INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01
fc1_weights_up = torch.randn(NUM_EXPERTS, INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01
fc2_bf16 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=DTYPE, device=DEVICE) * 0.01

fc1_iw_bf16 = torch.zeros(NUM_EXPERTS, N, HIDDEN, dtype=DTYPE, device=DEVICE)
fc1_iw_bf16[:, 0::2, :] = fc1_weights_gate
fc1_iw_bf16[:, 1::2, :] = fc1_weights_up

# FP4 quantize
fc1_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
fc2_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
fc1_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
fc2_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

fc1_iw_list, fc1_isf_list = [], []
for e in range(NUM_EXPERTS):
    w, sf = quantize_for_trtllm(fc1_iw_bf16[e], 1.0)
    fc1_iw_list.append(w)
    fc1_isf_list.append(sf)
fc1_iw = torch.stack(fc1_iw_list)
fc1_isf = torch.stack(fc1_isf_list)

fc2_w_list, fc2_sf_list = [], []
for e in range(NUM_EXPERTS):
    w, sf = quantize_for_trtllm(fc2_bf16[e], 1.0)
    fc2_w_list.append(w)
    fc2_sf_list.append(sf)
fc2_w = torch.stack(fc2_w_list)
fc2_sf = torch.stack(fc2_sf_list)

quant_scales_interleaved = [fc1_act_g, fc1_isf, fc1_g, fc2_act_g, fc2_sf, fc2_g]

isf = torch.ones(num_tokens, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

runner = torch.classes.trtllm.FusedMoeRunner(
    DTYPE, torch.int64, DTYPE, False, False, False, False, True)

runner.set_dual_tile_profiles([0, 0, 1, 3], 32)
output_fused = runner.run_moe_dual_tile(
    input_tensor, topk_idx.int(), topk_scores,
    fc1_iw, None, fc2_w, None,
    quant_scales_interleaved, isf, False, None, None, None,
    1, 0, 1, 0, False, SWIGLU, None, None, None)
torch.cuda.synchronize()

# The C++ code has a diagnostic that scans fc1_result_ and prints nonzero values.
# But we can also check the final output - if GEMM2 reads fc1_result_ correctly,
# and fc1_result_ has the pattern (compact_col+1), then GEMM2 output = weights * pattern.
# 
# More useful: let's look at the stderr diagnostic output for the pattern.
print(f"\nFinal output abs_mean: {output_fused.abs().mean():.6f}")
print(f"Final output [0, :10]: {output_fused[0, :10].tolist()}")
print()
print("Check stderr output for fc1_result_ scan results.")
print("Expected: all 32 rows should have values in columns 0..767")
print("         where value at col c = c+1 (as bf16)")

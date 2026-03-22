#!/usr/bin/env python3
"""
Diagnostic: force visitor to write gate_val only (no SwiGLU, no silu) to
isolate whether the raw accumulator values are correct.

If visitor writes gate_val and reference uses identity activation (not gated),
we can compare the first N/2 columns directly.
"""
import torch
import torch.nn.functional as F

torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")

torch.manual_seed(42)
DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768
N = INTER * 2  # 1536
SWIGLU = 5  # ActivationType enum value


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

# Create weights
fc1_weights_gate = torch.randn(NUM_EXPERTS, INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01
fc1_weights_up = torch.randn(NUM_EXPERTS, INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01
fc2_bf16 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=DTYPE, device=DEVICE) * 0.01

# Contiguous: [gate | up]
fc1_contig_bf16 = torch.cat([fc1_weights_gate, fc1_weights_up], dim=1)

# Interleaved: [g0, u0, g1, u1, ...]
fc1_iw_bf16 = torch.zeros(NUM_EXPERTS, N, HIDDEN, dtype=DTYPE, device=DEVICE)
fc1_iw_bf16[:, 0::2, :] = fc1_weights_gate
fc1_iw_bf16[:, 1::2, :] = fc1_weights_up

# FP4 quantize
fc1_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
fc2_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
fc1_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
fc2_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

fc1_cw_list, fc1_csf_list = [], []
for e in range(NUM_EXPERTS):
    w, sf = quantize_for_trtllm(fc1_contig_bf16[e], 1.0)
    fc1_cw_list.append(w)
    fc1_csf_list.append(sf)
fc1_cw = torch.stack(fc1_cw_list)
fc1_csf = torch.stack(fc1_csf_list)

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

quant_scales_contig = [fc1_act_g, fc1_csf, fc1_g, fc2_act_g, fc2_sf, fc2_g]
quant_scales_interleaved = [fc1_act_g, fc1_isf, fc1_g, fc2_act_g, fc2_sf, fc2_g]

isf = torch.ones(num_tokens, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

runner = torch.classes.trtllm.FusedMoeRunner(
    DTYPE, torch.int64, DTYPE, False, False, False, False, True)

# =========================================================================
# Compute expected SwiGLU result manually in Python
# =========================================================================
print("=" * 70)
print("Manual SwiGLU computation on raw bf16 weights")
print("(This is the ground truth before FP4 quantization)")
# For each expert that gets tokens, compute GEMM1 + SwiGLU manually
# Expert assignment from topk: token i -> experts [i*8, i*8+1, ..., i*8+7]
# Each token goes to 8 experts with equal weight (1/8)
for t in range(min(1, num_tokens)):  # just check token 0
    exp_id = topk_idx[t, 0].item()  # first expert for token 0
    gemm_contig = fc1_contig_bf16[exp_id].float() @ input_tensor[t].float()
    gate = gemm_contig[:INTER]
    up = gemm_contig[INTER:]
    silu_up = torch.sigmoid(up) * up
    swiglu_output = silu_up * gate
    print(f'  Token {t}, Expert {exp_id}:')
    print(f'    GEMM1 gate (first half) [:5]: {gate[:5].tolist()}')
    print(f'    GEMM1 up (second half) [:5]:  {up[:5].tolist()}')
    print(f'    silu(up) * gate [:5]:          {swiglu_output[:5].tolist()}')
    print(f'    SwiGLU abs_mean: {swiglu_output.abs().mean():.6f}')

    gemm_iw = fc1_iw_bf16[exp_id].float() @ input_tensor[t].float()
    gate_i = gemm_iw[0::2]  # even
    up_i = gemm_iw[1::2]    # odd
    silu_up_i = torch.sigmoid(up_i) * up_i
    swiglu_iw = silu_up_i * gate_i
    print(f'    Interleaved gate (even) [:5]:  {gate_i[:5].tolist()}')
    print(f'    Interleaved up (odd) [:5]:     {up_i[:5].tolist()}')
    print(f'    silu(up) * gate (iw) [:5]:     {swiglu_iw[:5].tolist()}')
    diff = (swiglu_output - swiglu_iw).abs().max().item()
    print(f'    Max diff contiguous vs interleaved: {diff:.8f}')

# =========================================================================
# Reference (single-tile, contiguous, unfused)
# =========================================================================
print("\n" + "=" * 70)
print("Reference: run_moe (contiguous, unfused)")
ref = runner.run_moe(
    input_tensor, topk_idx.int(), topk_scores,
    fc1_cw, None, fc2_w, None,
    quant_scales_contig, isf, False, None, None, None,
    1, 0, 1, 0, 1, 0, False, False, [1, 3], SWIGLU, None, None, None)
torch.cuda.synchronize()
print(f'  abs_mean={ref.abs().mean():.6f}')
print(f'  [0, :10]: {ref[0, :10].tolist()}')

# =========================================================================
# Fused (dual-tile, interleaved)
# =========================================================================
print("\n" + "=" * 70)
print("Fused: run_moe_dual_tile (interleaved, fused SwiGLU)")
runner.set_dual_tile_profiles([0, 0, 1, 3], 32)
output_fused = runner.run_moe_dual_tile(
    input_tensor, topk_idx.int(), topk_scores,
    fc1_iw, None, fc2_w, None,
    quant_scales_interleaved, isf, False, None, None, None,
    1, 0, 1, 0, False, SWIGLU, None, None, None)
torch.cuda.synchronize()
print(f'  abs_mean={output_fused.abs().mean():.6f}')
print(f'  [0, :10]: {output_fused[0, :10].tolist()}')

rel_err = (ref - output_fused).abs().mean().item() / (ref.abs().mean().item() + 1e-8)
print(f'\n  rel_err={rel_err:.6f}')

# =========================================================================
# Key insight: if fused output abs_mean is ~10x reference, it suggests
# the SwiGLU intermediate is being passed to GEMM2 raw (without proper
# scaling), OR the GEMM2 input is reading from the wrong buffer/offset.
# =========================================================================
print("\n" + "=" * 70)
print("ANALYSIS:")
print(f'  Reference final abs_mean: {ref.abs().mean():.6f}')
print(f'  Fused final abs_mean:     {output_fused.abs().mean():.6f}')
print(f'  Ratio:                    {output_fused.abs().mean() / ref.abs().mean():.2f}x')
print()
# The SwiGLU intermediate should have abs_mean ~0.07 (from manual calc)
# After GEMM2 (K=768 * weights~0.01) + routing (1/8), expect ~0.07 * 768 * 0.01 / 8 ~ 0.007
# Reference has 0.008 which matches.
# Fused has 0.077 which is 10x. This is roughly = SwiGLU intermediate * routing_scale
# = 0.07 * 1/8 * 8_experts ~= 0.07  (if GEMM2 somehow acts as identity??)
# OR: GEMM2 is reading wrong data / not reading from fc1_result_ at all.
# OR: fc1_result_ has overlapping data from both dual-tile GEMM1 groups.

# Let's check: what if the problem is that dual-tile GEMM1 writes to
# fc1_result_ (compact) but GEMM2 setup still reads from gemm1_output_buf?
# In the code, gemm2_input_final comes from applyPrequantScale(smoothed_act_, fc1_result_)
# which returns fc1_result_ (no prequant needed). So GEMM2 should read from fc1_result_.
#
# BUT: the GEMM2 stride setup via computeStridesTmaWarpSpecialized uses gemm2_in
# which is set to gemm2_input_buf. Let me check if gemm2_input_buf == fc1_result_.

if rel_err < 0.5:
    print(f'*** PASS ***')
else:
    print(f'*** FAIL ***')

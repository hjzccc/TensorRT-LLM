#!/usr/bin/env python3
"""
Diagnostic test for the FINALIZE epilogue hypothesis.

Oracle's hypothesis: runMoeDualTile forces FINALIZE fused epilogue while
runMoe with profile [0,0] may use unfused finalize. The bug might be in
the FINALIZE epilogue itself, not in dual-tile logic.

This test:
1. Lists all GEMM2 profiles and identifies which use FINALIZE
2. Compares runMoe(non-FINALIZE) vs runMoe(FINALIZE) vs runMoeDualTile([same,same,same,same])
3. Tests with k=1 (no reduction) vs k=2 (with reduction) to isolate scatter
4. If runMoe(FINALIZE) == runMoeDualTile, the bug is in FINALIZE, not dual-tile
"""

import torch
import sys

torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")

DEVICE = "cuda"
DTYPE = torch.bfloat16
SWIGLU = 5

def quantize_weight_fp4(w_bf16, global_sf):
    rows, cols = w_bf16.shape
    packed_u8, sf_u8 = torch.ops.trtllm.fp4_quantize(w_bf16, global_sf, 16, False, False)
    w_int64 = packed_u8.view(torch.int64).reshape(rows, cols // 16)
    sf_2d = sf_u8.reshape(rows, cols // 16)
    sf_interleaved = torch.ops.trtllm.block_scale_interleave(sf_2d)
    sf_int32 = sf_interleaved.view(torch.int32).reshape(rows, cols // 16 // 4)
    return w_int64, sf_int32

def prepare_weights(num_experts, hidden, inter):
    fc1_g = torch.ones(num_experts, dtype=torch.float32, device=DEVICE)
    fc2_g = torch.ones(num_experts, dtype=torch.float32, device=DEVICE)
    fc1_w_list, fc1_sf_list = [], []
    fc2_w_list, fc2_sf_list = [], []
    for e in range(num_experts):
        w1 = torch.randn(2 * inter, hidden, dtype=DTYPE, device=DEVICE) * 0.02
        w1q, sf1 = quantize_weight_fp4(w1, fc1_g[e])
        fc1_w_list.append(w1q); fc1_sf_list.append(sf1)
        w2 = torch.randn(hidden, inter, dtype=DTYPE, device=DEVICE) * 0.02
        w2q, sf2 = quantize_weight_fp4(w2, fc2_g[e])
        fc2_w_list.append(w2q); fc2_sf_list.append(sf2)
    fc1_w = torch.stack(fc1_w_list); fc1_sf = torch.stack(fc1_sf_list)
    fc2_w = torch.stack(fc2_w_list); fc2_sf = torch.stack(fc2_sf_list)
    quant_scales = [
        torch.tensor(1.0, dtype=torch.float32, device=DEVICE),
        fc1_sf, fc1_g,
        torch.tensor(1.0, dtype=torch.float32, device=DEVICE),
        fc2_sf, fc2_g,
    ]
    return fc1_w, fc2_w, quant_scales

def call_run_moe(runner, input_bf16, token_experts, token_scales,
                 fc1_w, fc2_w, quant_scales, input_sf, g1_profile, g2_profile):
    return runner.run_moe(
        input_bf16, token_experts, token_scales,
        fc1_w, None, fc2_w, None,
        quant_scales, input_sf, False,
        None, None, None,
        1, 0, 1, 0,    # tp_size, tp_rank, ep_size, ep_rank
        1, 0,           # cluster_size, cluster_rank
        False, False,   # enable_alltoall, min_latency_mode
        [g1_profile, g2_profile],
        SWIGLU,
        None, None, None,
    )

def call_run_moe_dual_tile(runner, input_bf16, token_experts, token_scales,
                           fc1_w, fc2_w, quant_scales, input_sf,
                           g1s, g2s, g1l, g2l, threshold):
    runner.set_dual_tile_profiles([g1s, g2s, g1l, g2l], threshold)
    return runner.run_moe_dual_tile(
        input_bf16, token_experts, token_scales,
        fc1_w, None, fc2_w, None,
        quant_scales, input_sf, False,
        None, None, None,
        1, 0, 1, 0,    # tp_size, tp_rank, ep_size, ep_rank
        False,          # enable_alltoall
        SWIGLU,
        None, None, None,
    )

def compare(label, ref, test):
    diff = (ref - test).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    ref_norm = ref.abs().mean().item()
    rel = mean_d / (ref_norm + 1e-8)
    status = "MATCH" if rel < 0.01 else ("CLOSE" if rel < 0.05 else "DIFFER")
    print(f"  {label:45s} | max={max_d:.6f} mean={mean_d:.6f} rel_err={rel:.6f} [{status}]")
    return rel

def main():
    # Use manageable size: 8 experts, H=2048, I=768 (Qwen3 dims)
    NUM_EXPERTS = 8
    HIDDEN = 2048
    INTER = 768

    print("=" * 80)
    print("FINALIZE EPILOGUE DIAGNOSTIC TEST")
    print(f"  {NUM_EXPERTS} experts, H={HIDDEN}, I={INTER}, NVFP4+Swiglu")
    print("=" * 80)

    runner = torch.classes.trtllm.FusedMoeRunner(
        DTYPE, torch.int64, DTYPE,
        False, False, False, False, True,
    )

    fc1_w, fc2_w, quant_scales = prepare_weights(NUM_EXPERTS, HIDDEN, INTER)
    torch.manual_seed(42)

    # ──────────────────────────────────────────────────────
    # TEST 1: Profile enumeration
    # For SM120 FP4 at H=2048, I=768:
    #   GEMM1: profiles 0,1 (non-FINALIZE only, since GEMM1 never uses FINALIZE)
    #   GEMM2: profiles 0,1,2,3 -- some may be FINALIZE, some not
    # The config generation duplicates non-FINALIZE configs with FINALIZE versions.
    # So GEMM2 indices are: [non-FINALIZE...][FINALIZE...]
    # With 2 base tiles (128x128x64B, 128x128x128B) at 2 swap_ab variants,
    # we might have: indices 0-N = non-FINALIZE, indices N+1-2N = FINALIZE
    # ──────────────────────────────────────────────────────
    print("\n--- TEST 1: Compare runMoe with different GEMM2 profiles ---")
    print("  Each profile may use FINALIZE or non-FINALIZE epilogue.\n")

    NUM_TOKENS = 32
    TOP_K = 2  # Start with k=2 for reduction
    input_bf16 = torch.randn(NUM_TOKENS, HIDDEN, dtype=DTYPE, device=DEVICE)
    gate = torch.randn(NUM_TOKENS, NUM_EXPERTS, device=DEVICE, dtype=DTYPE)
    topk_vals, topk_ids = torch.topk(gate, TOP_K, dim=-1)
    topk_weights = torch.softmax(topk_vals.float(), dim=-1)
    token_experts = topk_ids.int()
    token_scales = topk_weights.float()
    input_sf = torch.ones(NUM_TOKENS, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

    # Try all GEMM2 profile indices (0-3) with GEMM1=0
    results = {}
    for g2_idx in range(4):
        try:
            out = call_run_moe(runner, input_bf16, token_experts, token_scales,
                               fc1_w, fc2_w, quant_scales, input_sf, 0, g2_idx)
            results[g2_idx] = out.clone()
            norm = out.abs().mean().item()
            print(f"  runMoe(g1=0, g2={g2_idx}): norm={norm:.6f}")
        except Exception as e:
            print(f"  runMoe(g1=0, g2={g2_idx}): ERROR - {e}")

    # Compare all profiles against profile 0
    if 0 in results:
        print(f"\n  Comparing all profiles against profile g2=0:")
        for g2_idx in sorted(results.keys()):
            if g2_idx == 0:
                continue
            compare(f"g2={g2_idx} vs g2=0", results[0], results[g2_idx])

    # ──────────────────────────────────────────────────────
    # TEST 2: runMoe vs runMoeDualTile with all experts in one group (high threshold)
    # If dual-tile with high threshold matches runMoe for SOME profiles but not others,
    # the FINALIZE epilogue is the differentiator.
    # ──────────────────────────────────────────────────────
    print("\n--- TEST 2: runMoe vs runMoeDualTile (all-in-one-group, thr=99999) ---")
    print("  If dual-tile matches SOME profiles but not the one we used as reference,")
    print("  then the mismatch is because dual-tile uses FINALIZE.\n")

    dt_results = {}
    for g1_idx in range(2):
        for g2_idx in range(4):
            try:
                dual = call_run_moe_dual_tile(
                    runner, input_bf16, token_experts, token_scales,
                    fc1_w, fc2_w, quant_scales, input_sf,
                    g1_idx, g2_idx, g1_idx, g2_idx, 99999)
                dt_results[(g1_idx, g2_idx)] = dual.clone()
                norm = dual.abs().mean().item()
                print(f"  dual_tile(g1={g1_idx}, g2={g2_idx}, thr=99999): norm={norm:.6f}")
            except Exception as e:
                print(f"  dual_tile(g1={g1_idx}, g2={g2_idx}, thr=99999): ERROR - {e}")

    # Compare dual-tile against each runMoe profile
    print(f"\n  Cross-comparison (dual_tile vs runMoe):")
    for (g1_idx, g2_idx), dt_out in sorted(dt_results.items()):
        if g2_idx in results:
            compare(f"dual(g1={g1_idx},g2={g2_idx}) vs runMoe(g1=0,g2={g2_idx})",
                    results[g2_idx], dt_out)

    # KEY: compare dual-tile with EVERY runMoe profile to find which one matches
    print(f"\n  Which runMoe profile does dual_tile(0,0) match?")
    if (0,0) in dt_results:
        for g2_idx in sorted(results.keys()):
            compare(f"dual(0,0) vs runMoe(g1=0,g2={g2_idx})",
                    results[g2_idx], dt_results[(0,0)])

    # ──────────────────────────────────────────────────────
    # TEST 3: k=1 (no reduction) — isolates scatter/reduction path
    # ──────────────────────────────────────────────────────
    print("\n--- TEST 3: k=1 (no reduction) — isolate scatter ---")
    TOP_K_1 = 1
    gate1 = torch.randn(NUM_TOKENS, NUM_EXPERTS, device=DEVICE, dtype=DTYPE)
    topk_vals1, topk_ids1 = torch.topk(gate1, TOP_K_1, dim=-1)
    topk_weights1 = torch.ones(NUM_TOKENS, TOP_K_1, device=DEVICE, dtype=torch.float32)
    token_experts1 = topk_ids1.int()
    token_scales1 = topk_weights1

    # runMoe with k=1
    ref_k1 = {}
    for g2_idx in range(4):
        try:
            out = call_run_moe(runner, input_bf16, token_experts1, token_scales1,
                               fc1_w, fc2_w, quant_scales, input_sf, 0, g2_idx)
            ref_k1[g2_idx] = out.clone()
            norm = out.abs().mean().item()
            print(f"  runMoe(k=1, g2={g2_idx}): norm={norm:.6f}")
        except Exception as e:
            print(f"  runMoe(k=1, g2={g2_idx}): ERROR - {e}")

    # Compare k=1 profiles against each other
    if 0 in ref_k1:
        print(f"\n  k=1 profile comparison:")
        for g2_idx in sorted(ref_k1.keys()):
            if g2_idx == 0:
                continue
            compare(f"k=1: g2={g2_idx} vs g2=0", ref_k1[0], ref_k1[g2_idx])

    # dual-tile with k=1
    print(f"\n  k=1 dual-tile vs runMoe:")
    for g2_idx in range(4):
        try:
            dual_k1 = call_run_moe_dual_tile(
                runner, input_bf16, token_experts1, token_scales1,
                fc1_w, fc2_w, quant_scales, input_sf,
                0, g2_idx, 0, g2_idx, 99999)
            if g2_idx in ref_k1:
                compare(f"k=1: dual(0,{g2_idx}) vs runMoe(0,{g2_idx})",
                        ref_k1[g2_idx], dual_k1)
            # Also compare against all runMoe profiles
            for ref_g2 in sorted(ref_k1.keys()):
                if ref_g2 != g2_idx:
                    rel = compare(f"k=1: dual(0,{g2_idx}) vs runMoe(0,{ref_g2})",
                                  ref_k1[ref_g2], dual_k1)
        except Exception as e:
            print(f"  k=1: dual_tile(0,{g2_idx}): ERROR - {e}")

    # ──────────────────────────────────────────────────────
    # TEST 4: Dual-tile with actual split (low threshold) vs high threshold
    # ──────────────────────────────────────────────────────
    print("\n--- TEST 4: Dual-tile split vs no-split ---")
    print("  Comparing dual_tile at thr=1 (most split) vs thr=99999 (no split)\n")

    for thr in [1, 2, 3, 5, 10, 99999]:
        try:
            dual_out = call_run_moe_dual_tile(
                runner, input_bf16, token_experts, token_scales,
                fc1_w, fc2_w, quant_scales, input_sf,
                0, 0, 0, 0, thr)
            if (0,0) in dt_results:
                compare(f"dual(0,0,thr={thr}) vs dual(0,0,thr=99999)",
                        dt_results[(0,0)], dual_out)
        except Exception as e:
            print(f"  dual(0,0,thr={thr}): ERROR - {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print("""
If TEST 2 shows dual_tile matches a DIFFERENT runMoe profile than expected:
  → FINALIZE epilogue produces different results than unfused finalize
  → Bug is in FINALIZE, not in dual-tile logic

If TEST 3 shows k=1 eliminates the error:
  → Bug is in the reduction/scatter part of FINALIZE (source_token_index)
  → Not in GEMM computation

If TEST 4 shows thr=1 differs from thr=99999 even with same tile configs:
  → Bug is in how experts are split between groups
  → The intermediate buffer aliasing or doActivation issue
""")

if __name__ == "__main__":
    main()

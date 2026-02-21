#!/usr/bin/env python3
"""
Real-scale dual-tile MoE GEMM test for Qwen3-30B-A3B-NVFP4.

Model dimensions: 128 experts, top-8, H=2048, I=768, NVFP4, Swiglu
Tests correctness (dual-tile vs single-tile) and benchmarks performance.

Tile configs for SM120 FP4 grouped GEMM:
  0: CtaShape128x128x64B
  1: CtaShape128x128x128B
  2: CtaShape128x256x64B
  3: CtaShape256x128x64B

run_moe signature (26 args after self):
  input, token_selected_experts, token_final_scales?,
  fc1_weights, fc1_biases?, fc2_weights, fc2_biases?,
  quant_scales?, input_sf?, swizzled_input_sf,
  swiglu_alpha?, swiglu_beta?, swiglu_limit?,
  tp_size, tp_rank, ep_size, ep_rank,
  cluster_size, cluster_rank,
  enable_alltoall, min_latency_mode, profile_ids?,
  activation_type?, unpadded_hidden_size?, num_valid_tokens?, out_tensor?

run_moe_dual_tile signature (22 args after self):
  input, token_selected_experts, token_final_scales?,
  fc1_weights, fc1_biases?, fc2_weights, fc2_biases?,
  quant_scales?, input_sf?, swizzled_input_sf,
  swiglu_alpha?, swiglu_beta?, swiglu_limit?,
  tp_size, tp_rank, ep_size, ep_rank,
  enable_alltoall, activation_type?, unpadded_hidden_size?,
  num_valid_tokens?, out_tensor?
"""

import torch
import time

torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")

# ── Model constants ──────────────────────────────────────────────────
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768  # moe_intermediate_size
SWIGLU = 5
DEVICE = "cuda"
DTYPE = torch.bfloat16

# ── FP4 quantization helpers ────────────────────────────────────────
def quantize_weight_fp4(w_bf16, global_sf):
    rows, cols = w_bf16.shape
    packed_u8, sf_u8 = torch.ops.trtllm.fp4_quantize(w_bf16, global_sf, 16, False, False)
    w_int64 = packed_u8.view(torch.int64).reshape(rows, cols // 16)
    sf_2d = sf_u8.reshape(rows, cols // 16)
    sf_interleaved = torch.ops.trtllm.block_scale_interleave(sf_2d)
    sf_int32 = sf_interleaved.view(torch.int32).reshape(rows, cols // 16 // 4)
    return w_int64, sf_int32


def prepare_weights():
    print(f"Preparing weights: {NUM_EXPERTS} experts, H={HIDDEN}, I={INTER}")
    print(f"  FC1 per expert: ({2*INTER}, {HIDDEN}), FC2 per expert: ({HIDDEN}, {INTER})")

    fc1_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
    fc2_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)

    fc1_w_list, fc1_sf_list = [], []
    fc2_w_list, fc2_sf_list = [], []

    for e in range(NUM_EXPERTS):
        w1 = torch.randn(2 * INTER, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.5
        w1q, sf1 = quantize_weight_fp4(w1, fc1_g[e])
        fc1_w_list.append(w1q); fc1_sf_list.append(sf1)
        del w1

        w2 = torch.randn(HIDDEN, INTER, dtype=DTYPE, device=DEVICE) * 0.5
        w2q, sf2 = quantize_weight_fp4(w2, fc2_g[e])
        fc2_w_list.append(w2q); fc2_sf_list.append(sf2)
        del w2

    fc1_w = torch.stack(fc1_w_list); fc1_sf = torch.stack(fc1_sf_list)
    fc2_w = torch.stack(fc2_w_list); fc2_sf = torch.stack(fc2_sf_list)
    del fc1_w_list, fc1_sf_list, fc2_w_list, fc2_sf_list
    torch.cuda.empty_cache()

    print(f"  fc1_w: {fc1_w.shape}, fc1_sf: {fc1_sf.shape}")
    print(f"  fc2_w: {fc2_w.shape}, fc2_sf: {fc2_sf.shape}")

    quant_scales = [
        torch.tensor(1.0, dtype=torch.float32, device=DEVICE),  # fc1_act_global
        fc1_sf, fc1_g,
        torch.tensor(1.0, dtype=torch.float32, device=DEVICE),  # fc2_act_global
        fc2_sf, fc2_g,
    ]
    return fc1_w, fc2_w, quant_scales


def generate_routing(num_tokens):
    gate = torch.randn(num_tokens, NUM_EXPERTS, device=DEVICE, dtype=DTYPE)
    topk_vals, topk_ids = torch.topk(gate, TOP_K, dim=-1)
    topk_weights = torch.softmax(topk_vals.float(), dim=-1)
    # Both 2D: (num_tokens, top_k), token_scales must be float32
    return topk_ids.int(), topk_weights.float()


def call_run_moe(runner, input_bf16, token_experts, token_scales,
                 fc1_w, fc2_w, quant_scales, input_sf, g1_profile, g2_profile):
    return runner.run_moe(
        input_bf16,                    # 1
        token_experts,                 # 2: 2D int32
        token_scales,                  # 3: 2D float32
        fc1_w,                         # 4
        None,                          # 5: fc1_biases
        fc2_w,                         # 6
        None,                          # 7: fc2_biases
        quant_scales,                  # 8
        input_sf,                      # 9
        False,                         # 10: swizzled_input_sf
        None,                          # 11: swiglu_alpha
        None,                          # 12: swiglu_beta
        None,                          # 13: swiglu_limit
        1,                             # 14: tp_size
        0,                             # 15: tp_rank
        1,                             # 16: ep_size
        0,                             # 17: ep_rank
        1,                             # 18: cluster_size
        0,                             # 19: cluster_rank
        False,                         # 20: enable_alltoall
        False,                         # 21: min_latency_mode
        [g1_profile, g2_profile],      # 22: profile_ids [gemm1_idx, gemm2_idx]
        SWIGLU,                        # 23: activation_type
        None,                          # 24: unpadded_hidden_size
        None,                          # 25: num_valid_tokens
        None,                          # 26: out_tensor
    )


def call_run_moe_dual_tile(runner, input_bf16, token_experts, token_scales,
                           fc1_w, fc2_w, quant_scales, input_sf,
                           g1_small, g2_small, g1_large, g2_large, threshold):
    """run_moe_dual_tile with 22 args."""
    runner.set_dual_tile_profiles([g1_small, g2_small, g1_large, g2_large], threshold)
    return runner.run_moe_dual_tile(
        input_bf16,          # 1
        token_experts,       # 2
        token_scales,        # 3
        fc1_w,               # 4
        None,                # 5: fc1_biases
        fc2_w,               # 6
        None,                # 7: fc2_biases
        quant_scales,        # 8
        input_sf,            # 9
        False,               # 10: swizzled_input_sf
        None,                # 11: swiglu_alpha
        None,                # 12: swiglu_beta
        None,                # 13: swiglu_limit
        1,                   # 14: tp_size
        0,                   # 15: tp_rank
        1,                   # 16: ep_size
        0,                   # 17: ep_rank
        False,               # 18: enable_alltoall
        SWIGLU,              # 19: activation_type
        None,                # 20: unpadded_hidden_size
        None,                # 21: num_valid_tokens
        None,                # 22: out_tensor
    )


def run_correctness_test(runner, fc1_w, fc2_w, quant_scales, num_tokens, threshold,
                         g1_small, g2_small, g1_large, g2_large):
    input_bf16 = torch.randn(num_tokens, HIDDEN, dtype=DTYPE, device=DEVICE)
    token_experts, token_scales = generate_routing(num_tokens)
    input_sf = torch.ones(num_tokens, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

    ref = call_run_moe(runner, input_bf16, token_experts, token_scales,
                       fc1_w, fc2_w, quant_scales, input_sf, g1_large, g2_large)
    dual = call_run_moe_dual_tile(runner, input_bf16, token_experts, token_scales,
                                  fc1_w, fc2_w, quant_scales, input_sf,
                                  g1_small, g2_small, g1_large, g2_large, threshold)

    max_diff = (ref - dual).abs().max().item()
    mean_diff = (ref - dual).abs().mean().item()
    ref_norm = ref.abs().mean().item()
    rel_err = mean_diff / (ref_norm + 1e-8)
    return max_diff, mean_diff, rel_err, ref_norm


def run_benchmark(runner, fc1_w, fc2_w, quant_scales, num_tokens, threshold,
                  g1_small, g2_small, g1_large, g2_large,
                  warmup=5, repeats=20):
    input_bf16 = torch.randn(num_tokens, HIDDEN, dtype=DTYPE, device=DEVICE)
    token_experts, token_scales = generate_routing(num_tokens)
    input_sf = torch.ones(num_tokens, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

    # Single-tile warmup + bench
    for _ in range(warmup):
        call_run_moe(runner, input_bf16, token_experts, token_scales,
                     fc1_w, fc2_w, quant_scales, input_sf, g1_large, g2_large)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        call_run_moe(runner, input_bf16, token_experts, token_scales,
                     fc1_w, fc2_w, quant_scales, input_sf, g1_large, g2_large)
    torch.cuda.synchronize()
    single_ms = (time.perf_counter() - start) / repeats * 1000

    # Dual-tile warmup + bench
    for _ in range(warmup):
        call_run_moe_dual_tile(runner, input_bf16, token_experts, token_scales,
                               fc1_w, fc2_w, quant_scales, input_sf,
                               g1_small, g2_small, g1_large, g2_large, threshold)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        call_run_moe_dual_tile(runner, input_bf16, token_experts, token_scales,
                               fc1_w, fc2_w, quant_scales, input_sf,
                               g1_small, g2_small, g1_large, g2_large, threshold)
    torch.cuda.synchronize()
    dual_ms = (time.perf_counter() - start) / repeats * 1000

    return single_ms, dual_ms


def main():
    print("=" * 70)
    print("Real-Scale Dual-Tile MoE Test: Qwen3-30B-A3B-NVFP4 Dimensions")
    print(f"  {NUM_EXPERTS} experts, top-{TOP_K}, H={HIDDEN}, I={INTER}, NVFP4+Swiglu")
    print("=" * 70)

    runner = torch.classes.trtllm.FusedMoeRunner(
        DTYPE, torch.int64, DTYPE,
        False, False, False, False, True,
    )

    fc1_w, fc2_w, quant_scales = prepare_weights()
    mem_used = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory used: {mem_used:.2f} GB")

    # GEMM1: 2 profiles (0,1), GEMM2: 4 profiles (0-3) at Qwen3 dims
    # Format: (name, g1_small, g2_small, g1_large, g2_large)
    tile_configs = [
        ("small=0,0 large=1,3", 0, 0, 1, 3),
        ("small=0,0 large=1,2", 0, 0, 1, 2),
        ("small=0,1 large=1,3", 0, 1, 1, 3),
        ("small=0,0 large=0,3", 0, 0, 0, 3),
        ("small=0,0 large=1,1", 0, 0, 1, 1),
    ]

    # Profiled P70 thresholds from real Qwen3 routing data
    batch_thresholds = {64: 4, 128: 9, 256: 18}

    # ════════════════════════════════════════════════════════════
    # PHASE 1: Correctness
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: CORRECTNESS VERIFICATION")
    print("=" * 70)

    cfg_name, g1s, g2s, g1l, g2l = tile_configs[0]
    print(f"\nTile config: {cfg_name}")

    all_pass = True
    for num_tokens in [8, 32, 64, 128, 256]:
        threshold = batch_thresholds.get(num_tokens, max(1, num_tokens // 16))
        try:
            max_d, mean_d, rel_e, ref_n = run_correctness_test(
                runner, fc1_w, fc2_w, quant_scales,
                num_tokens, threshold, g1s, g2s, g1l, g2l)
            status = "PASS" if rel_e < 0.05 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  tokens={num_tokens:4d} thr={threshold:3d} | "
                  f"max_diff={max_d:.4f} mean_diff={mean_d:.6f} "
                  f"rel_err={rel_e:.6f} ref_norm={ref_n:.4f} [{status}]")
        except Exception as e:
            print(f"  tokens={num_tokens:4d} thr={threshold:3d} | ERROR: {e}")
            all_pass = False

    print(f"\nCross-tile correctness at batch=128, threshold=9:")
    for cfg_name, g1s, g2s, g1l, g2l in tile_configs:
        try:
            max_d, mean_d, rel_e, ref_n = run_correctness_test(
                runner, fc1_w, fc2_w, quant_scales,
                128, 9, g1s, g2s, g1l, g2l)
            status = "PASS" if rel_e < 0.05 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {cfg_name:30s} | max={max_d:.4f} rel_err={rel_e:.6f} [{status}]")
        except Exception as e:
            print(f"  {cfg_name:30s} | ERROR: {e}")
            all_pass = False

    print(f"\nOverall correctness: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ════════════════════════════════════════════════════════════
    # PHASE 2: Performance Benchmarking
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: PERFORMANCE BENCHMARKING")
    print("=" * 70)

    for cfg_name, g1s, g2s, g1l, g2l in tile_configs:
        print(f"\nTile config: {cfg_name}")
        print(f"  {'Batch':>6s} {'Thresh':>6s} | {'Single(ms)':>10s} {'Dual(ms)':>10s} {'Speedup':>8s}")
        print(f"  {'-'*6} {'-'*6} | {'-'*10} {'-'*10} {'-'*8}")

        for num_tokens in [64, 128, 256]:
            threshold = batch_thresholds[num_tokens]
            try:
                single_ms, dual_ms = run_benchmark(
                    runner, fc1_w, fc2_w, quant_scales,
                    num_tokens, threshold, g1s, g2s, g1l, g2l,
                    warmup=3, repeats=10)
                speedup = single_ms / dual_ms if dual_ms > 0 else 0
                print(f"  {num_tokens:6d} {threshold:6d} | {single_ms:10.3f} {dual_ms:10.3f} {speedup:7.2f}x")
            except Exception as e:
                print(f"  {num_tokens:6d} {threshold:6d} | ERROR: {e}")

    # ════════════════════════════════════════════════════════════
    # PHASE 3: Threshold sensitivity
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: THRESHOLD SENSITIVITY (batch=128)")
    print("=" * 70)

    best_cfg = tile_configs[0]
    _, g1s, g2s, g1l, g2l = best_cfg
    print(f"Tile config: {best_cfg[0]}")
    print(f"  {'Thresh':>6s} | {'Single(ms)':>10s} {'Dual(ms)':>10s} {'Speedup':>8s}")
    print(f"  {'-'*6} | {'-'*10} {'-'*10} {'-'*8}")

    for threshold in [1, 3, 5, 9, 15, 25, 50]:
        try:
            single_ms, dual_ms = run_benchmark(
                runner, fc1_w, fc2_w, quant_scales,
                128, threshold, g1s, g2s, g1l, g2l,
                warmup=3, repeats=10)
            speedup = single_ms / dual_ms if dual_ms > 0 else 0
            print(f"  {threshold:6d} | {single_ms:10.3f} {dual_ms:10.3f} {speedup:7.2f}x")
        except Exception as e:
            print(f"  {threshold:6d} | ERROR: {e}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

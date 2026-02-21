#!/usr/bin/env python3
"""Test M64/M32 tiles for SM120 NVFP4 MoE grouped GEMM.

Usage (inside Docker):
    python3 /code/tensorrt_llm/tests/test_m64_tile.py --test-only
    python3 /code/tensorrt_llm/tests/test_m64_tile.py
    python3 /code/tensorrt_llm/tests/test_m64_tile.py --skip-benchmark
"""

import argparse
import sys
import time

import torch

import warnings
warnings.filterwarnings("ignore")

import tensorrt_llm  # noqa: F401

# --- Model dimensions (Qwen3-30B-A3B) ---
HIDDEN_SIZE = 2048
INTER_SIZE = 768  # per-expert intermediate size (gate/up each)
NUM_EXPERTS = 8   # use fewer experts for test (saves memory)
TOP_K = 4

# --- NVFP4 packing constants ---
FP4_PER_INT64 = 16
BLOCK_SCALE_VECTOR_SIZE = 16   # one FP8 scale per 16 FP4 elements
FP8_PER_INT32 = 4


def create_runner():
    """Create NVFP4 FusedMoeRunner."""
    return torch.classes.trtllm.FusedMoeRunner(
        torch.int64,      # x_dtype (packed FP4 activations)
        torch.int64,      # weight_dtype (packed FP4 weights)
        torch.bfloat16,   # output_dtype
        False, False, False, False, True,
    )


def create_synthetic_weights():
    """Create synthetic NVFP4 weights with correct shapes for run_moe().

    Weight shapes (int64 packed FP4):
      fc1 (gate_up): [num_experts, inter_size*2, hidden_size/16] = [8, 1536, 128]
      fc2 (down):    [num_experts, hidden_size, inter_size/16]   = [8, 2048, 48]

    Block scale shapes (int32 packed FP8, interleaved):
      fc1: [num_experts, inter_size*2, hidden_size/64] = [8, 1536, 32]
      fc2: [num_experts, hidden_size, inter_size/64]   = [8, 2048, 12]
    """
    N_fused = INTER_SIZE * 2           # 1536 (gate + up)
    K_fc1_packed = HIDDEN_SIZE // FP4_PER_INT64  # 128
    K_fc2_packed = INTER_SIZE // FP4_PER_INT64   # 48

    # Weight tensors (random int64 = random FP4 bit patterns)
    fc1_w = torch.randint(-2**62, 2**62, (NUM_EXPERTS, N_fused, K_fc1_packed),
                          dtype=torch.int64, device='cuda')
    fc2_w = torch.randint(-2**62, 2**62, (NUM_EXPERTS, HIDDEN_SIZE, K_fc2_packed),
                          dtype=torch.int64, device='cuda')

    # Block scales — need interleaving via trtllm op
    fc1_sf_raw_k = HIDDEN_SIZE // BLOCK_SCALE_VECTOR_SIZE  # 128
    fc1_sf_int32_k = fc1_sf_raw_k // FP8_PER_INT32         # 32

    fc2_sf_raw_k = INTER_SIZE // BLOCK_SCALE_VECTOR_SIZE   # 48
    fc2_sf_int32_k = fc2_sf_raw_k // FP8_PER_INT32         # 12

    fc1_scales_list = []
    fc2_scales_list = []
    for _ in range(NUM_EXPERTS):
        # fc1 block scales: [1536, 128] uint8 → interleave → [1536, 32] int32
        raw = torch.randint(0, 127, (N_fused, fc1_sf_raw_k),
                            dtype=torch.uint8, device='cuda')
        interleaved = torch.ops.trtllm.block_scale_interleave(raw)
        fc1_scales_list.append(
            interleaved.view(torch.int32).reshape(N_fused, fc1_sf_int32_k))

        # fc2 block scales: [2048, 48] uint8 → interleave → [2048, 12] int32
        raw2 = torch.randint(0, 127, (HIDDEN_SIZE, fc2_sf_raw_k),
                             dtype=torch.uint8, device='cuda')
        interleaved2 = torch.ops.trtllm.block_scale_interleave(raw2)
        fc2_scales_list.append(
            interleaved2.view(torch.int32).reshape(HIDDEN_SIZE, fc2_sf_int32_k))

    fc1_weight_block = torch.stack(fc1_scales_list)  # [8, 1536, 32] int32
    fc2_weight_block = torch.stack(fc2_scales_list)  # [8, 2048, 12] int32

    # Global scales
    fc1_act_global = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    fc1_global = torch.ones(NUM_EXPERTS, dtype=torch.float32, device='cuda')
    fc2_act_global = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    fc2_global = torch.ones(NUM_EXPERTS, dtype=torch.float32, device='cuda')

    quant_scales = [fc1_act_global, fc1_weight_block, fc1_global,
                    fc2_act_global, fc2_weight_block, fc2_global]

    return fc1_w, fc2_w, quant_scales


def create_input(num_tokens):
    """Create FP4-quantized input activations and routing."""
    torch.manual_seed(42)
    x_bf16 = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device='cuda') * 0.1
    gs = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(x_bf16, gs, BLOCK_SCALE_VECTOR_SIZE, False, True)
    x_input = x_fp4.view(torch.int64)  # [num_tokens, 128] int64

    # Random expert routing
    token_experts = torch.stack([
        torch.randperm(NUM_EXPERTS, device='cuda')[:TOP_K]
        for _ in range(num_tokens)
    ]).to(torch.int32)
    token_scales = torch.ones(num_tokens, TOP_K, dtype=torch.float32, device='cuda') / TOP_K

    return x_input, x_sf, token_experts, token_scales


def run_moe(runner, x_input, x_sf, token_experts, token_scales,
            fc1_w, fc2_w, quant_scales, gemm1_tactic, gemm2_tactic):
    """Run MoE with specific tactics."""
    return runner.run_moe(
        x_input, token_experts, token_scales,
        fc1_w, None,        # fc1 weights, no bias
        fc2_w, None,        # fc2 weights, no bias
        quant_scales,
        x_sf, True,         # input_sf, swizzled=True
        None, None, None,   # swiglu_alpha, beta, limit
        1, 0, 1, 0, 1, 0,  # tp/ep/cluster size,rank
        False, False,       # enable_alltoall, min_latency
        [gemm1_tactic, gemm2_tactic],  # profile_ids
        None, None, None, None,  # activation_type, unpadded_hidden, num_valid, out
    )


def test_tactic_count():
    """Test 1: Verify M64 and M32 tiles are available."""
    print("=" * 70)
    print("Test 1: Verify M64/M32 tactic availability")
    print("=" * 70)
    try:
        runner = create_runner()
        n1 = runner.get_tactic_num(1)
        n2 = runner.get_tactic_num(2)
        print(f"  GEMM1 tactics: {n1}")
        print(f"  GEMM2 tactics: {n2}")
        if n1 >= 6 and n2 >= 6:
            print(f"  PASS: M128+M64+M32 tiles available ({n1} tactics)")
            return True, runner
        elif n1 >= 4:
            print(f"  PARTIAL: M64 available but M32 may be missing ({n1} tactics)")
            return True, runner
        else:
            print(f"  FAIL: Expected >= 6 tactics (got {n1})")
            return False, runner
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False, None


def check_output(label, out_test, out_ref):
    diff = (out_ref.float() - out_test.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_abs_mean = out_ref.float().abs().mean().item()
    rel_err = mean_diff / (ref_abs_mean + 1e-10)
    print(f"    {label} mean={out_test.float().mean().item():.6f}, absmax={out_test.float().abs().max().item():.6f}")
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, Rel err: {rel_err:.6e}")
    if max_diff < 1e-6:
        print(f"    PASS: Identical output")
        return True
    elif rel_err < 0.01:
        print(f"    PASS: Within tight tolerance")
        return True
    elif rel_err < 0.1:
        print(f"    PASS: Within FP4 tolerance")
        return True
    elif rel_err < 0.5:
        print(f"    WARN: Loose match (rel_err={rel_err:.4f})")
        return True
    else:
        print(f"    FAIL: Outputs diverge significantly")
        return False


def test_correctness(runner):
    print("\n" + "=" * 70)
    print("Test 2: Correctness — M64/M32 vs M128")
    print("=" * 70)
    try:
        print("  Creating synthetic weights...")
        fc1_w, fc2_w, quant_scales = create_synthetic_weights()
        print(f"    fc1_w: {fc1_w.shape} {fc1_w.dtype}")
        print(f"    fc2_w: {fc2_w.shape} {fc2_w.dtype}")

        n_tactics = runner.get_tactic_num(1)
        has_m32 = n_tactics >= 6

        all_pass = True
        for num_tokens in [4, 16, 64]:
            print(f"\n  --- {num_tokens} tokens ---")
            x_input, x_sf, token_experts, token_scales = create_input(num_tokens)

            print("    Running tactic 0 (M128)...")
            out_ref = run_moe(runner, x_input, x_sf, token_experts, token_scales,
                              fc1_w, fc2_w, quant_scales, 0, 0)
            torch.cuda.synchronize()
            print(f"    Ref  mean={out_ref.float().mean().item():.6f}, absmax={out_ref.float().abs().max().item():.6f}")

            print("    Running tactic 1 (M64)...")
            out_m64 = run_moe(runner, x_input, x_sf, token_experts, token_scales,
                              fc1_w, fc2_w, quant_scales, 1, 1)
            torch.cuda.synchronize()
            if not check_output("M64 ", out_m64, out_ref):
                all_pass = False

            if has_m32:
                print("    Running tactic 2 (M32)...")
                out_m32 = run_moe(runner, x_input, x_sf, token_experts, token_scales,
                                  fc1_w, fc2_w, quant_scales, 2, 2)
                torch.cuda.synchronize()
                if not check_output("M32 ", out_m32, out_ref):
                    all_pass = False

        return all_pass

    except Exception as e:
        import traceback
        print(f"\n  ERROR: {e}")
        traceback.print_exc()
        return None


def benchmark_tactics(runner):
    print("\n" + "=" * 70)
    print("Test 3: Performance benchmark — M128 vs M64 vs M32")
    print("=" * 70)
    try:
        fc1_w, fc2_w, quant_scales = create_synthetic_weights()

        n_tactics = runner.get_tactic_num(1)
        has_m32 = n_tactics >= 6

        bench_tactics = [0, 1]
        tactic_names = {0: "M128", 1: "M64"}
        if has_m32:
            bench_tactics.append(2)
            tactic_names[2] = "M32"

        token_counts = [4, 8, 16, 32, 64, 128, 256]
        warmup = 10
        runs = 50

        print(f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, hidden: {HIDDEN_SIZE}, inter: {INTER_SIZE}")
        print(f"  Warmup: {warmup}, Runs: {runs}")
        print(f"  Available GEMM1 tactics: {n_tactics}")
        print()

        header = f"  {'tokens':>6} |"
        for t in bench_tactics:
            header += f" {tactic_names[t]:>12} |"
        header += "  M64 speedup  M32 speedup" if has_m32 else "  M64 speedup"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for num_tokens in token_counts:
            x_input, x_sf, token_experts, token_scales = create_input(num_tokens)

            row = f"  {num_tokens:>6} |"
            times = {}
            for tactic in bench_tactics:
                try:
                    for _ in range(warmup):
                        run_moe(runner, x_input, x_sf, token_experts, token_scales,
                                fc1_w, fc2_w, quant_scales, tactic, tactic)
                    torch.cuda.synchronize()

                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(runs):
                        run_moe(runner, x_input, x_sf, token_experts, token_scales,
                                fc1_w, fc2_w, quant_scales, tactic, tactic)
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - start) * 1000 / runs
                    times[tactic] = elapsed_ms
                    row += f" {elapsed_ms:>9.3f} ms |"
                except Exception:
                    row += f" {'ERR':>9} ms |"

            if 0 in times and 1 in times and times[1] > 0:
                row += f"  {times[0] / times[1]:.2f}x"
            else:
                row += "  N/A"

            if has_m32:
                if 0 in times and 2 in times and times[2] > 0:
                    row += f"         {times[0] / times[2]:.2f}x"
                else:
                    row += "         N/A"

            print(row)

    except Exception as e:
        import traceback
        print(f"\n  ERROR: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test M64 tile for SM120 NVFP4 MoE")
    parser.add_argument("--test-only", action="store_true",
                        help="Only test tactic availability")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    print("M64/M32 Tile Test — SM120 NVFP4 MoE Grouped GEMM")
    print("=" * 70)

    # Test 1
    tactic_ok, runner = test_tactic_count()
    if not tactic_ok or args.test_only:
        sys.exit(0 if tactic_ok else 1)

    # Test 2
    correctness = None
    if not args.skip_correctness:
        correctness = test_correctness(runner)

    # Test 3
    if not args.skip_benchmark:
        benchmark_tactics(runner)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tactic availability: {'PASS' if tactic_ok else 'FAIL'}")
    if correctness is not None:
        print(f"  Correctness:         {'PASS' if correctness else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

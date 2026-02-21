"""
Verification test for dual-tile MoE GEMM implementation.

Run inside Docker container after building with the appropriate SM architecture:
    docker exec trtllm-build python3 /code/tensorrt_llm/tests/test_dual_tile_moe.py

Or with pytest:
    docker exec trtllm-build pytest /code/tensorrt_llm/tests/test_dual_tile_moe.py -v -s
"""
import sys

import torch
import torch.nn.functional as F

# Load the C++ extension directly to avoid full tensorrt_llm import issues
torch.ops.load_library(
    "/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so"
)

# C++ ActivationType enum values (from common.h)
ACTIVATION_SWIGLU = 5
ACTIVATION_RELU2 = 8


def ref_moe_swiglu(x, fc1_w, fc2_w, selected_experts, routing_weights,
                    num_experts):
    """Reference MoE with SwiGLU activation (pure PyTorch)."""
    results = torch.zeros_like(x)
    for eid in range(num_experts):
        mask = selected_experts == eid
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)
        w3, w1 = torch.chunk(fc1_w[eid], 2, dim=0)
        inp = x[batch_idx]
        inter = F.silu(inp @ w1.t()) * (inp @ w3.t())
        out = inter @ fc2_w[eid].t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * out
    return results


def compute_routing(router_logits, top_k):
    """Compute routing weights and selected experts from logits."""
    routing_weights = F.softmax(router_logits.float(), dim=1)
    routing_weights, selected_experts = torch.topk(
        routing_weights, top_k, dim=-1
    )
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.float(), selected_experts.int()


def create_runner(dtype=torch.bfloat16):
    """Create a fresh C++ FusedMoeRunner."""
    return torch.classes.trtllm.FusedMoeRunner(
        dtype, dtype, dtype,  # activation, weight, output
        False, False, False, False,  # quant flags
        True,  # use_fused_finalize
    )


def run_regular_moe(runner, x, selected_experts, routing_weights, fc1_w, fc2_w,
                    activation_type=ACTIVATION_SWIGLU):
    """Call the regular run_moe C++ method."""
    return runner.run_moe(
        x, selected_experts, routing_weights,
        fc1_w, None, fc2_w, None,
        [],             # quant_scales
        None, True,     # input_sf, swizzled_input_sf
        None, None, None,  # swiglu alpha/beta/limit
        1, 0, 1, 0, 1, 0,  # tp_size/rank, ep_size/rank, cluster_size/rank
        False, False,       # enable_alltoall, min_latency
        [0, 0],             # profile_ids [gemm1, gemm2]
        activation_type,
        None, None, None,   # unpadded_hidden, num_valid, out_tensor
    )


def run_dual_tile_moe(runner, x, selected_experts, routing_weights, fc1_w,
                      fc2_w, threshold, tactics=(0, 0, 0, 0),
                      activation_type=ACTIVATION_SWIGLU):
    """Call set_dual_tile_profiles + run_moe_dual_tile."""
    runner.set_dual_tile_profiles(list(tactics), threshold)
    return runner.run_moe_dual_tile(
        x, selected_experts, routing_weights,
        fc1_w, None, fc2_w, None,
        [],             # quant_scales
        None, True,     # input_sf, swizzled_input_sf
        None, None, None,  # swiglu alpha/beta/limit
        1, 0, 1, 0,        # tp_size/rank, ep_size/rank
        False,              # enable_alltoall
        activation_type,
        None, None, None,   # unpadded_hidden, num_valid, out_tensor
    )


def test_dual_tile_correctness():
    """Main test: dual-tile output should match regular MoE output."""
    torch.manual_seed(42)
    BATCH = 64
    HIDDEN = 256
    INTER = 512
    NUM_EXPERTS = 8
    TOP_K = 2
    DTYPE = torch.bfloat16
    ATOL = 0.15  # bf16 GEMM tolerance (multiple GEMMs accumulate error)

    print(f"Config: batch={BATCH}, hidden={HIDDEN}, inter={INTER}, "
          f"experts={NUM_EXPERTS}, top_k={TOP_K}, dtype={DTYPE}")

    # Generate test data
    x = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device="cuda") * 0.1
    router_logits = torch.randn(
        BATCH, NUM_EXPERTS, dtype=DTYPE, device="cuda"
    )
    fc1_w = torch.randn(
        NUM_EXPERTS, 2 * INTER, HIDDEN, dtype=DTYPE, device="cuda"
    ) * 0.1
    fc2_w = torch.randn(
        NUM_EXPERTS, HIDDEN, INTER, dtype=DTYPE, device="cuda"
    ) * 0.1

    routing_weights, selected_experts = compute_routing(router_logits, TOP_K)

    # Reference output
    ref = ref_moe_swiglu(
        x, fc1_w, fc2_w, selected_experts, routing_weights, NUM_EXPERTS
    )

    # --- Test 1: Regular run_moe vs reference ---
    runner1 = create_runner(DTYPE)
    num_tactics = runner1.get_tactic_num(1)
    print(f"Available tactics: {num_tactics}")

    torch.cuda.synchronize()
    regular_out = run_regular_moe(runner1, x, selected_experts,
                                  routing_weights, fc1_w, fc2_w)
    torch.cuda.synchronize()

    d1 = (ref - regular_out).abs()
    print(f"[{'PASS' if d1.max() < ATOL else 'FAIL'}] "
          f"run_moe vs ref:          max={d1.max():.6f}  mean={d1.mean():.6f}")
    assert d1.max() < ATOL, f"run_moe vs reference failed: max_diff={d1.max()}"

    # --- Test 2: Dual-tile threshold=1 (almost all experts "large") ---
    r2 = create_runner(DTYPE)
    torch.cuda.synchronize()
    out2 = run_dual_tile_moe(r2, x, selected_experts, routing_weights,
                             fc1_w, fc2_w, threshold=1)
    torch.cuda.synchronize()
    d2 = (ref - out2).abs()
    print(f"[{'PASS' if d2.max() < ATOL else 'FAIL'}] "
          f"dual_tile (th=1, large):  max={d2.max():.6f}  mean={d2.mean():.6f}")
    assert d2.max() < ATOL, f"dual-tile th=1 failed: max_diff={d2.max()}"

    # --- Test 3: Dual-tile threshold=10000 (all experts "small") ---
    r3 = create_runner(DTYPE)
    torch.cuda.synchronize()
    out3 = run_dual_tile_moe(r3, x, selected_experts, routing_weights,
                             fc1_w, fc2_w, threshold=10000)
    torch.cuda.synchronize()
    d3 = (ref - out3).abs()
    print(f"[{'PASS' if d3.max() < ATOL else 'FAIL'}] "
          f"dual_tile (th=10k, small): max={d3.max():.6f}  mean={d3.mean():.6f}")
    assert d3.max() < ATOL, f"dual-tile th=10000 failed: max_diff={d3.max()}"

    # --- Test 4: Dual-tile mixed threshold ---
    # avg tokens per expert = 64*2/8 = 16, threshold=10 splits experts
    r4 = create_runner(DTYPE)
    torch.cuda.synchronize()
    out4 = run_dual_tile_moe(r4, x, selected_experts, routing_weights,
                             fc1_w, fc2_w, threshold=10)
    torch.cuda.synchronize()
    d4 = (ref - out4).abs()
    print(f"[{'PASS' if d4.max() < ATOL else 'FAIL'}] "
          f"dual_tile (th=10, mixed):  max={d4.max():.6f}  mean={d4.mean():.6f}")
    assert d4.max() < ATOL, f"dual-tile th=10 failed: max_diff={d4.max()}"

    # --- Test 5: Different tactics for small vs large ---
    r5 = create_runner(DTYPE)
    max_tactic = min(num_tactics - 1, 1)
    torch.cuda.synchronize()
    out5 = run_dual_tile_moe(r5, x, selected_experts, routing_weights,
                             fc1_w, fc2_w, threshold=10,
                             tactics=(0, 0, max_tactic, max_tactic))
    torch.cuda.synchronize()
    d5 = (ref - out5).abs()
    print(f"[{'PASS' if d5.max() < ATOL else 'FAIL'}] "
          f"dual_tile (diff tactics):  max={d5.max():.6f}  mean={d5.mean():.6f}")
    assert d5.max() < ATOL, f"dual-tile diff tactics failed: max_diff={d5.max()}"

    # --- Test 6: Dual-tile vs regular run_moe (should be very close) ---
    d6 = (regular_out - out2).abs()
    print(f"[{'PASS' if d6.max() < ATOL else 'FAIL'}] "
          f"dual_tile vs run_moe:      max={d6.max():.6f}  mean={d6.mean():.6f}")

    # --- Test 7: Larger batch to stress the split ---
    torch.manual_seed(123)
    x_big = torch.randn(256, HIDDEN, dtype=DTYPE, device="cuda") * 0.1
    rl_big = torch.randn(256, NUM_EXPERTS, dtype=DTYPE, device="cuda")
    rw_big, se_big = compute_routing(rl_big, TOP_K)
    ref_big = ref_moe_swiglu(
        x_big, fc1_w, fc2_w, se_big, rw_big, NUM_EXPERTS
    )
    r7 = create_runner(DTYPE)
    torch.cuda.synchronize()
    out7 = run_dual_tile_moe(r7, x_big, se_big, rw_big, fc1_w, fc2_w,
                             threshold=32)
    torch.cuda.synchronize()
    d7 = (ref_big - out7).abs()
    print(f"[{'PASS' if d7.max() < ATOL else 'FAIL'}] "
          f"dual_tile (batch=256):     max={d7.max():.6f}  mean={d7.mean():.6f}")
    assert d7.max() < ATOL, f"dual-tile batch=256 failed: max_diff={d7.max()}"

    print("\n✓ All dual-tile MoE tests passed!")


def test_dual_tile_single_token():
    """Edge case: single token (all experts get 0 or 1 token)."""
    torch.manual_seed(99)
    HIDDEN = 256
    INTER = 512
    NUM_EXPERTS = 8
    DTYPE = torch.bfloat16

    x = torch.randn(1, HIDDEN, dtype=DTYPE, device="cuda") * 0.1
    router_logits = torch.randn(1, NUM_EXPERTS, dtype=DTYPE, device="cuda")
    fc1_w = torch.randn(
        NUM_EXPERTS, 2 * INTER, HIDDEN, dtype=DTYPE, device="cuda"
    ) * 0.1
    fc2_w = torch.randn(
        NUM_EXPERTS, HIDDEN, INTER, dtype=DTYPE, device="cuda"
    ) * 0.1

    routing_weights, selected_experts = compute_routing(router_logits, 2)
    ref = ref_moe_swiglu(
        x, fc1_w, fc2_w, selected_experts, routing_weights, NUM_EXPERTS
    )

    r = create_runner(DTYPE)
    torch.cuda.synchronize()
    out = run_dual_tile_moe(r, x, selected_experts, routing_weights,
                            fc1_w, fc2_w, threshold=1)
    torch.cuda.synchronize()
    diff = (ref - out).abs()
    print(f"[{'PASS' if diff.max() < 0.15 else 'FAIL'}] "
          f"single_token:              max={diff.max():.6f}")
    assert diff.max() < 0.15


def test_dual_tile_top_k_1():
    """Edge case: top_k=1 (no reduction needed)."""
    torch.manual_seed(77)
    BATCH = 32
    HIDDEN = 256
    INTER = 512
    NUM_EXPERTS = 8
    DTYPE = torch.bfloat16

    x = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device="cuda") * 0.1
    router_logits = torch.randn(
        BATCH, NUM_EXPERTS, dtype=DTYPE, device="cuda"
    )
    fc1_w = torch.randn(
        NUM_EXPERTS, 2 * INTER, HIDDEN, dtype=DTYPE, device="cuda"
    ) * 0.1
    fc2_w = torch.randn(
        NUM_EXPERTS, HIDDEN, INTER, dtype=DTYPE, device="cuda"
    ) * 0.1

    routing_weights, selected_experts = compute_routing(router_logits, 1)
    ref = ref_moe_swiglu(
        x, fc1_w, fc2_w, selected_experts, routing_weights, NUM_EXPERTS
    )

    r = create_runner(DTYPE)
    torch.cuda.synchronize()
    out = run_dual_tile_moe(r, x, selected_experts, routing_weights,
                            fc1_w, fc2_w, threshold=5)
    torch.cuda.synchronize()
    diff = (ref - out).abs()
    print(f"[{'PASS' if diff.max() < 0.15 else 'FAIL'}] "
          f"top_k=1:                   max={diff.max():.6f}")
    assert diff.max() < 0.15


if __name__ == "__main__":
    print("=" * 60)
    print("Dual-Tile MoE Verification Tests")
    print("=" * 60)

    test_dual_tile_correctness()
    print()
    test_dual_tile_single_token()
    test_dual_tile_top_k_1()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    sys.exit(0)

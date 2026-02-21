#!/usr/bin/env python3
"""
End-to-end test for dual-tile MoE pipeline integration.

Tests:
1. MoeDualTileConfig propagates through ModelConfig → ConfigurableMoE → CutlassFusedMoE
2. Config disabled by default
3. Various config values propagate correctly
4. NVFP4 dual-tile execution via fused_moe_dual_tile custom op (SM90+ TMA kernels)

Run inside Docker:
    docker exec -e LD_LIBRARY_PATH="/usr/local/tensorrt/lib" \
        -e PYTHONPATH="/code/tensorrt_llm" trtllm-build \
        python3 /code/tensorrt_llm/tests/test_dual_tile_e2e.py
"""
import sys
import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig, MoeDualTileConfig
from tensorrt_llm._torch.modules.fused_moe import (
    CutlassFusedMoE, RenormalizeMoeRoutingMethod, create_moe)
from tensorrt_llm.mapping import Mapping

DEVICE = "cuda"
DTYPE = torch.bfloat16


def get_cutlass_backend(moe):
    inner = moe
    while hasattr(inner, 'backend') and inner.backend is not None:
        inner = inner.backend
    return inner


def make_moe(num_experts, hidden_size, intermediate_size, top_k,
             dual_tile_cfg=None):
    routing = RenormalizeMoeRoutingMethod(top_k=top_k)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.dtype = DTYPE

    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(),
        moe_backend="CUTLASS",
        moe_dual_tile=dual_tile_cfg,
    )

    return create_moe(
        routing_method=routing,
        reduce_results=True,
        model_config=model_config,
    )


def test_config_propagation():
    print("TEST: config propagation...", end=" ", flush=True)
    cfg = MoeDualTileConfig(
        threshold=42,
        gemm1_small_tactic=0, gemm2_small_tactic=1,
        gemm1_large_tactic=1, gemm2_large_tactic=0,
    )
    moe = make_moe(8, 64, 32, 2, dual_tile_cfg=cfg)
    inner = get_cutlass_backend(moe)

    assert inner.use_dual_tile is True
    assert inner.dual_tile_threshold == 42
    assert inner.gemm1_small_tactic == 0
    assert inner.gemm2_small_tactic == 1
    assert inner.gemm1_large_tactic == 1
    assert inner.gemm2_large_tactic == 0
    print("PASS")


def test_disabled_by_default():
    print("TEST: disabled by default...", end=" ", flush=True)
    moe = make_moe(8, 64, 32, 2, dual_tile_cfg=None)
    inner = get_cutlass_backend(moe)
    assert inner.use_dual_tile is False
    print("PASS")


def test_config_values():
    print("TEST: various config values...", end=" ", flush=True)
    for threshold in [1, 7, 16, 128]:
        for tactic in [0, 1, 2]:
            cfg = MoeDualTileConfig(
                threshold=threshold,
                gemm1_small_tactic=tactic,
                gemm2_small_tactic=tactic,
                gemm1_large_tactic=tactic,
                gemm2_large_tactic=tactic,
            )
            moe = make_moe(8, 64, 32, 2, dual_tile_cfg=cfg)
            inner = get_cutlass_backend(moe)
            assert inner.dual_tile_threshold == threshold
            assert inner.gemm1_small_tactic == tactic
    print("PASS")


def test_nvfp4_dual_tile_execution():
    """NVFP4 dual-tile via C++ FusedMoeRunner directly.
    The C++ runner constructor requires (bf16, int64, bf16) type combo for NVFP4.
    Actual FP4-packed data is passed at call time."""
    print("TEST: NVFP4 dual-tile execution...", flush=True)
    N, H, I, K = 8, 2048, 768, 2
    SWIGLU = 5

    torch.manual_seed(0)
    bf16_w1 = torch.randn(N, I, H, dtype=DTYPE, device=DEVICE)
    bf16_w2 = torch.randn(N, H, I, dtype=DTYPE, device=DEVICE)
    bf16_w3 = torch.randn(N, I, H, dtype=DTYPE, device=DEVICE)

    def quantize_for_trtllm(w_bf16, gsf):
        rows, cols = w_bf16.shape
        packed_u8, sf_u8 = torch.ops.trtllm.fp4_quantize(
            w_bf16, gsf, 16, False, False)
        w_int64 = packed_u8.view(torch.int64).reshape(rows, cols // 16)
        sf_2d = sf_u8.reshape(rows, cols // 16)
        sf_interleaved = torch.ops.trtllm.block_scale_interleave(sf_2d)
        sf_int32 = sf_interleaved.view(torch.int32).reshape(
            rows, cols // 16 // 4)
        return w_int64, sf_int32

    fc1_weights = []
    fc2_weights = []
    fc1_sf_list = []
    fc2_sf_list = []
    global_sf = torch.tensor(1.0, device=DEVICE)

    for eid in range(N):
        w1_q, w1_sf = quantize_for_trtllm(bf16_w1[eid], global_sf)
        w3_q, w3_sf = quantize_for_trtllm(bf16_w3[eid], global_sf)
        fc1_q = torch.cat([w3_q, w1_q], dim=0)
        fc1_sf = torch.cat([w3_sf, w1_sf], dim=0)
        fc1_weights.append(fc1_q)
        fc1_sf_list.append(fc1_sf)

        w2_q, w2_sf = quantize_for_trtllm(bf16_w2[eid], global_sf)
        fc2_weights.append(w2_q)
        fc2_sf_list.append(w2_sf)

    fc1_w = torch.stack(fc1_weights)
    fc2_w = torch.stack(fc2_weights)
    fc1_sf = torch.stack(fc1_sf_list)
    fc2_sf = torch.stack(fc2_sf_list)
    fc1_g = torch.ones(N, dtype=torch.float32, device=DEVICE)
    fc2_g = torch.ones(N, dtype=torch.float32, device=DEVICE)
    fc1_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    fc2_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    quant_scales = [fc1_act_g, fc1_sf, fc1_g, fc2_act_g, fc2_sf, fc2_g]

    runner = torch.classes.trtllm.FusedMoeRunner(
        DTYPE, torch.int64, DTYPE, False, False, False, False, True)

    for num_tokens in [16, 32, 64]:
        torch.manual_seed(num_tokens)
        x = torch.randn(num_tokens, H, dtype=DTYPE, device=DEVICE)
        isf = torch.ones(num_tokens, H // 16, dtype=torch.float8_e4m3fn,
                         device=DEVICE)

        router_logits = torch.randn(num_tokens, N, dtype=DTYPE, device=DEVICE)
        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        ref = runner.run_moe(
            x, topk_indices.int(), topk_weights.float(),
            fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, 1, 0, False, False, [0, 0], SWIGLU, None, None, None)

        runner.set_dual_tile_profiles([0, 0, 0, 0], 8)
        dual = runner.run_moe_dual_tile(
            x, topk_indices.int(), topk_weights.float(),
            fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, False, SWIGLU, None, None, None)

        max_d = (ref - dual).abs().max().item()
        mean_d = (ref - dual).abs().mean().item()
        ref_n = ref.abs().mean().item()
        rel_e = mean_d / (ref_n + 1e-8)
        ok = rel_e < 0.01
        print(f"  tokens={num_tokens:4d}: max={max_d:.6f} rel_err={rel_e:.6f} "
              f"ref={ref_n:.4f} [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            return False

    print("  ALL PASS")
    return True


def main():
    print("=" * 60)
    print("Dual-Tile MoE E2E Integration Test")
    print("=" * 60)

    torch.cuda.set_device(0)
    all_pass = True

    with torch.device(DEVICE):
        test_config_propagation()
        test_disabled_by_default()
        test_config_values()

        if not test_nvfp4_dual_tile_execution():
            all_pass = False

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'=' * 60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

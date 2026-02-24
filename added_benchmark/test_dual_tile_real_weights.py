#!/usr/bin/env python3
"""
Dual-tile MoE test with REAL Qwen3-30B-A3B-NVFP4 checkpoint weights.
Loads actual NVFP4 expert weights from layer 0, converts to TRT-LLM format,
generates routing via real gate weights, then compares single-tile vs dual-tile.
"""
import json, time
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")

MODEL_PATH = '/root/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3'
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768
SWIGLU = 5
DEVICE = "cuda"
DTYPE = torch.bfloat16
TEST_LAYER = 0

FP4_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
                        device=DEVICE)


def dequant_nvfp4(w_u8, w_scale, w_scale2):
    rows, packed = w_u8.shape
    low = (w_u8.to(torch.int32) & 0x0F)
    high = ((w_u8.to(torch.int32) >> 4) & 0x0F)
    unpacked = torch.stack([low, high], dim=-1).reshape(rows, packed * 2).long()
    fp4_vals = FP4_LUT[unpacked]
    sf = w_scale.float().repeat_interleave(16, dim=1)
    return (fp4_vals * sf * w_scale2.float()).to(DTYPE)


def quantize_for_trtllm(w_bf16, global_sf):
    rows, cols = w_bf16.shape
    packed_u8, sf_u8 = torch.ops.trtllm.fp4_quantize(w_bf16, global_sf, 16, False, False)
    w_int64 = packed_u8.view(torch.int64).reshape(rows, cols // 16)
    sf_2d = sf_u8.reshape(rows, cols // 16)
    sf_interleaved = torch.ops.trtllm.block_scale_interleave(sf_2d)
    sf_int32 = sf_interleaved.view(torch.int32).reshape(rows, cols // 16 // 4)
    return w_int64, sf_int32


def load_real_weights():
    """Load actual NVFP4 expert weights from checkpoint, reformat for TRT-LLM."""
    print(f"Loading real weights from layer {TEST_LAYER}...")
    with open(f'{MODEL_PATH}/model.safetensors.index.json') as f:
        weight_map = json.load(f)['weight_map']

    handles = {}
    def get_tensor(key):
        fname = weight_map[key]
        if fname not in handles:
            handles[fname] = safe_open(f'{MODEL_PATH}/{fname}', framework='pt', device=DEVICE)
        return handles[fname].get_tensor(key)

    gate_w = get_tensor(f'model.layers.{TEST_LAYER}.mlp.gate.weight')
    print(f"  Gate: {gate_w.shape} {gate_w.dtype}")

    fc1_w_list, fc1_sf_list = [], []
    fc2_w_list, fc2_sf_list = [], []


    for eid in range(NUM_EXPERTS):
        prefix = f'model.layers.{TEST_LAYER}.mlp.experts.{eid}'

        gp_u8 = get_tensor(f'{prefix}.gate_proj.weight')
        gp_sc = get_tensor(f'{prefix}.gate_proj.weight_scale')
        gp_sc2 = get_tensor(f'{prefix}.gate_proj.weight_scale_2')

        gp_bf16 = dequant_nvfp4(gp_u8, gp_sc, gp_sc2)

        up_u8 = get_tensor(f'{prefix}.up_proj.weight')
        up_sc = get_tensor(f'{prefix}.up_proj.weight_scale')
        up_sc2 = get_tensor(f'{prefix}.up_proj.weight_scale_2')
        up_bf16 = dequant_nvfp4(up_u8, up_sc, up_sc2)

        dp_u8 = get_tensor(f'{prefix}.down_proj.weight')
        dp_sc = get_tensor(f'{prefix}.down_proj.weight_scale')
        dp_sc2 = get_tensor(f'{prefix}.down_proj.weight_scale_2')

        dp_bf16 = dequant_nvfp4(dp_u8, dp_sc, dp_sc2)

        fc1_bf16 = torch.cat([gp_bf16, up_bf16], dim=0)
        unit_sf = torch.tensor(1.0, device=DEVICE)

        fc1_q, fc1_s = quantize_for_trtllm(fc1_bf16, unit_sf)
        fc2_q, fc2_s = quantize_for_trtllm(dp_bf16, unit_sf)

        fc1_w_list.append(fc1_q)
        fc1_sf_list.append(fc1_s)
        fc2_w_list.append(fc2_q)
        fc2_sf_list.append(fc2_s)


        if eid % 32 == 31:
            print(f"  Loaded experts 0-{eid}")

    fc1_w = torch.stack(fc1_w_list)
    fc1_sf = torch.stack(fc1_sf_list)
    fc2_w = torch.stack(fc2_w_list)
    fc2_sf = torch.stack(fc2_sf_list)
    fc1_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
    fc2_g = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)

    # Use 1.0 for activation global scales — the per-token block scales (input_sf)
    # handle activation scaling. Checkpoint input_scale values (~0.001) cause underflow
    # when combined with re-quantized weights (also using global_sf=1.0).
    fc1_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    fc2_act_g = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    print(f"  fc1_w: {fc1_w.shape}, fc1_sf: {fc1_sf.shape}")
    print(f"  fc2_w: {fc2_w.shape}, fc2_sf: {fc2_sf.shape}")
    print(f"  fc1_act_global: 1.0, fc2_act_global: 1.0 (re-quantized)")

    quant_scales = [fc1_act_g, fc1_sf, fc1_g, fc2_act_g, fc2_sf, fc2_g]
    return fc1_w, fc2_w, quant_scales, gate_w


def generate_real_routing(gate_w, num_tokens):
    """Generate routing using real gate weights + random hidden states."""
    hidden = torch.randn(num_tokens, HIDDEN, device=DEVICE, dtype=DTYPE)
    logits = hidden.float() @ gate_w.float().T
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_idx = torch.topk(scores, TOP_K, dim=-1)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    return topk_idx.int(), topk_scores.float()


def call_run_moe(runner, inp, experts, scales, fc1_w, fc2_w, qs, isf, g1, g2):
    return runner.run_moe(
        inp, experts, scales, fc1_w, None, fc2_w, None,
        qs, isf, False, None, None, None,
        1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)


def call_dual_tile(runner, inp, experts, scales, fc1_w, fc2_w, qs, isf,
                   g1s, g2s, g1l, g2l, threshold):
    runner.set_dual_tile_profiles([g1s, g2s, g1l, g2l], threshold)
    return runner.run_moe_dual_tile(
        inp, experts, scales, fc1_w, None, fc2_w, None,
        qs, isf, False, None, None, None,
        1, 0, 1, 0, False, SWIGLU, None, None, None)


def main():
    print("=" * 70)
    print("Dual-Tile MoE Test with REAL Qwen3 NVFP4 Weights")
    print(f"  Layer {TEST_LAYER}: {NUM_EXPERTS} experts, top-{TOP_K}, H={HIDDEN}, I={INTER}")
    print("=" * 70)

    fc1_w, fc2_w, quant_scales, gate_w = load_real_weights()
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory: {mem_gb:.2f} GB")

    runner = torch.classes.trtllm.FusedMoeRunner(
        DTYPE, torch.int64, DTYPE, False, False, False, False, True)

    tile_configs = [
        ("small=[0,0] large=[1,3]", 0, 0, 1, 3),
        ("small=[0,0] large=[1,1]", 0, 0, 1, 1),
        ("small=[0,0] large=[0,3]", 0, 0, 0, 3),
        ("small=[0,1] large=[1,3]", 0, 1, 1, 3),
    ]

    thresholds = {64: 7, 128: 12, 256: 23}

    # ═══════════ PHASE 1: Correctness ═══════════
    print(f"\n{'='*70}")
    print("PHASE 1: CORRECTNESS (real weights, real gate routing)")
    print(f"{'='*70}")

    cfg_name, g1s, g2s, g1l, g2l = tile_configs[0]
    print(f"Tile config: {cfg_name}")

    all_pass = True
    for num_tokens in [32, 64, 128, 256]:
        thr = thresholds.get(num_tokens, max(1, num_tokens // 8))
        experts, scales = generate_real_routing(gate_w, num_tokens)
        inp = torch.randn(num_tokens, HIDDEN, dtype=DTYPE, device=DEVICE)
        isf = torch.ones(num_tokens, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

        ref = call_run_moe(runner, inp, experts, scales,
                           fc1_w, fc2_w, quant_scales, isf, g1l, g2l)
        dual = call_dual_tile(runner, inp, experts, scales,
                              fc1_w, fc2_w, quant_scales, isf,
                              g1s, g2s, g1l, g2l, thr)

        max_d = (ref - dual).abs().max().item()
        mean_d = (ref - dual).abs().mean().item()
        ref_n = ref.abs().mean().item()
        rel_e = mean_d / (ref_n + 1e-8)
        ok = rel_e < 0.05
        if not ok: all_pass = False

        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=DEVICE)
        counts.scatter_add_(0, experts.reshape(-1).long(),
                            torch.ones(num_tokens * TOP_K, dtype=torch.int32, device=DEVICE))
        active = int((counts > 0).sum())
        p70 = counts[counts > 0].float().quantile(0.7).int().item() if active > 0 else 0

        print(f"  tokens={num_tokens:4d} thr={thr:3d} active={active:3d} p70={p70:3d} | "
              f"max={max_d:.4f} rel_err={rel_e:.6f} ref={ref_n:.4f} [{'PASS' if ok else 'FAIL'}]")

    print(f"\nCross-tile configs (batch=128, thr=12):")
    for cfg_name, g1s, g2s, g1l, g2l in tile_configs:
        experts, scales = generate_real_routing(gate_w, 128)
        inp = torch.randn(128, HIDDEN, dtype=DTYPE, device=DEVICE)
        isf = torch.ones(128, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

        ref = call_run_moe(runner, inp, experts, scales,
                           fc1_w, fc2_w, quant_scales, isf, g1l, g2l)
        dual = call_dual_tile(runner, inp, experts, scales,
                              fc1_w, fc2_w, quant_scales, isf,
                              g1s, g2s, g1l, g2l, 12)

        rel_e = (ref - dual).abs().mean().item() / (ref.abs().mean().item() + 1e-8)
        ok = rel_e < 0.05
        if not ok: all_pass = False
        print(f"  {cfg_name:30s} | rel_err={rel_e:.6f} [{'PASS' if ok else 'FAIL'}]")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ═══════════ PHASE 2: Benchmark ═══════════
    print(f"\n{'='*70}")
    print("PHASE 2: PERFORMANCE (real weights)")
    print(f"{'='*70}")
    torch.cuda.cudart().cudaProfilerStart()
    for cfg_name, g1s, g2s, g1l, g2l in tile_configs[:2]:
        print(f"\n{cfg_name}")
        print(f"  {'Batch':>6} {'Thr':>4} | {'Single':>10} {'Dual':>10} {'Ratio':>8}")
        print(f"  {'-'*6} {'-'*4} | {'-'*10} {'-'*10} {'-'*8}")

        for nt in [64, 128, 256]:
            thr = thresholds[nt]
            experts, scales = generate_real_routing(gate_w, nt)
            inp = torch.randn(nt, HIDDEN, dtype=DTYPE, device=DEVICE)
            isf = torch.ones(nt, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

            for _ in range(5):
                call_run_moe(runner, inp, experts, scales,
                             fc1_w, fc2_w, quant_scales, isf, g1l, g2l)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.nvtx.range_push("Single-tile runs")
            for _ in range(20):
                call_run_moe(runner, inp, experts, scales,
                             fc1_w, fc2_w, quant_scales, isf, g1l, g2l)
            torch.cuda.synchronize()
            single_ms = (time.perf_counter() - t0) / 20 * 1000
            torch.cuda.nvtx.range_pop()

            for _ in range(5):
                call_dual_tile(runner, inp, experts, scales,
                               fc1_w, fc2_w, quant_scales, isf,
                               g1s, g2s, g1l, g2l, thr)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.nvtx.range_push("Dual-tile runs")
            for _ in range(20):
                call_dual_tile(runner, inp, experts, scales,
                               fc1_w, fc2_w, quant_scales, isf,
                               g1s, g2s, g1l, g2l, thr)
            torch.cuda.synchronize()
            dual_ms = (time.perf_counter() - t0) / 20 * 1000
            torch.cuda.nvtx.range_pop()

            ratio = single_ms / dual_ms if dual_ms > 0 else 0
            print(f"  {nt:6d} {thr:4d} | {single_ms:9.3f}ms {dual_ms:9.3f}ms {ratio:7.2f}x")
    
    torch.cuda.cudart().cudaProfilerStop()
    # ═══════════ PHASE 3: Threshold sensitivity ═══════════
    print(f"\n{'='*70}")
    print("PHASE 3: THRESHOLD SENSITIVITY (batch=128, real weights)")
    print(f"{'='*70}")

    _, g1s, g2s, g1l, g2l = tile_configs[0]
    experts, scales = generate_real_routing(gate_w, 128)
    inp = torch.randn(128, HIDDEN, dtype=DTYPE, device=DEVICE)
    isf = torch.ones(128, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

    for _ in range(5):
        call_run_moe(runner, inp, experts, scales,
                     fc1_w, fc2_w, quant_scales, isf, g1l, g2l)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        call_run_moe(runner, inp, experts, scales,
                     fc1_w, fc2_w, quant_scales, isf, g1l, g2l)
    torch.cuda.synchronize()
    single_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"Single-tile baseline: {single_ms:.3f}ms")

    print(f"  {'Thr':>4} | {'Dual':>10} {'Ratio':>8}")
    for thr in [1, 3, 5, 7, 12, 20, 40, 80]:
        for _ in range(3):
            call_dual_tile(runner, inp, experts, scales,
                           fc1_w, fc2_w, quant_scales, isf,
                           g1s, g2s, g1l, g2l, thr)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            call_dual_tile(runner, inp, experts, scales,
                           fc1_w, fc2_w, quant_scales, isf,
                           g1s, g2s, g1l, g2l, thr)
        torch.cuda.synchronize()
        dual_ms = (time.perf_counter() - t0) / 20 * 1000
        ratio = single_ms / dual_ms
        print(f"  {thr:4d} | {dual_ms:9.3f}ms {ratio:7.2f}x")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Broad dual-tile sweep with many configs, thresholds, and batch sizes."""
import torch, time
import torch.nn.functional as F
torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")
from test_dual_tile_real_weights import (
    load_real_weights, generate_real_routing,
    NUM_EXPERTS, TOP_K, HIDDEN, SWIGLU, DEVICE, DTYPE
)

fc1_w, fc2_w, quant_scales, gate_w = load_real_weights()
runner = torch.classes.trtllm.FusedMoeRunner(DTYPE, torch.int64, DTYPE, False, False, False, False, True)

def bench_single(inp, eidx, sc, isf, g1, g2, iters=200):
    for _ in range(5):
        runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def bench_dual(inp, eidx, sc, isf, g1s, g2s, g1l, g2l, thr, iters=200):
    runner.set_dual_tile_profiles([g1s, g2s, g1l, g2l], thr)
    for _ in range(5):
        runner.run_moe_dual_tile(inp, eidx, sc, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, False, SWIGLU, None, None, None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run_moe_dual_tile(inp, eidx, sc, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, False, SWIGLU, None, None, None)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

# Tactic indices (FAST_BUILD, SM120, GROUPED_GEMM, NVFP4):
# GEMM1 (6):  [0]=M128  [1]=M64  [2]=M32  [3]=M128-swap  [4]=M64-swap  [5]=M32-swap
# GEMM2 (12): [0]=M128  [1]=M64  [2]=M32  [3]=M128+FIN  [4]=M64+FIN  [5]=M32+FIN
#              [6..11] = same with swap_ab
# Dual: (name, g1_small, g2_small, g1_large, g2_large)
dual_cfgs = [
    ("M32+M64",              2, 5, 1, 4),
    ("M32+M128",             2, 5, 0, 3),
    ("M64+M128",             1, 4, 0, 3),
    ("M32+M64 g2=M128",     2, 3, 1, 3),
    ("M32+M128 g2=M128",    2, 3, 0, 3),
    ("M64+M128 g2=M128",    1, 3, 0, 3),
]

for batch in [256, 512, 1024, 2048, 4096]:
    eidx, sc = generate_real_routing(gate_w, batch)
    inp = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE)
    isf = torch.ones(batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)
    avg_m = batch * TOP_K // NUM_EXPERTS

    s32 = bench_single(inp, eidx, sc, isf, 2, 5)
    s64 = bench_single(inp, eidx, sc, isf, 1, 4)
    s128 = bench_single(inp, eidx, sc, isf, 0, 3)
    best_s = min(s32, s64, s128)
    best_name = "M32" if best_s == s32 else ("M64" if best_s == s64 else "M128")

    print(f"\n===== batch={batch}, avg M/expert={avg_m} =====")
    print(f"Single-tile: M32={s32:.3f}ms  M64={s64:.3f}ms  M128={s128:.3f}ms  best={best_name} {best_s:.3f}ms")
    hdr = f"{'Dual config':>25} | Thr  |    Time   | vs best"
    print(hdr)
    print("-" * len(hdr))

    torch.cuda.cudart().cudaProfilerStart()
    for name, g1s, g2s, g1l, g2l in dual_cfgs:
        for thr in [16, 32, 64, 128, 256]:
            d = bench_dual(inp, eidx, sc, isf, g1s, g2s, g1l, g2l, thr)
            diff_pct = (d - best_s) / best_s * 100
            marker = " <<<" if diff_pct < -0.5 else ""
            print(f"{name:>25} | {thr:4d} | {d:7.3f}ms | {diff_pct:+6.1f}%{marker}")
    torch.cuda.cudart().cudaProfilerStop()

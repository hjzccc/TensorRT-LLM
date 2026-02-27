#!/usr/bin/env python3
"""Targeted nsys profile: batch=1024 comparing M128 single vs M64 single vs dual-tile configs."""
import torch
torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")
import sys; sys.path.insert(0, "/code/tensorrt_llm/added_benchmark")
from test_dual_tile_real_weights import (
    load_real_weights, generate_real_routing,
    NUM_EXPERTS, TOP_K, HIDDEN, SWIGLU, DEVICE, DTYPE
)

fc1_w, fc2_w, quant_scales, gate_w = load_real_weights()
runner = torch.classes.trtllm.FusedMoeRunner(DTYPE, torch.int64, DTYPE, False, False, False, False, True)
ITERS = 50

batch = 1024
eidx, sc = generate_real_routing(gate_w, batch)
inp = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE)
isf = torch.ones(batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

def run_single(g1, g2, iters=ITERS):
    for _ in range(iters):
        runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)

def run_dual(g1s, g2s, g1l, g2l, thr, iters=ITERS):
    runner.set_dual_tile_profiles([g1s, g2s, g1l, g2l], thr)
    for _ in range(iters):
        runner.run_moe_dual_tile(inp, eidx, sc, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, False, SWIGLU, None, None, None)

# Warmup (outside profiler range)
run_single(3, 3, 10)
run_single(1, 1, 10)
runner.set_dual_tile_profiles([1, 1, 3, 3], 16)
run_dual(1, 1, 3, 3, 16, 10)
torch.cuda.synchronize()

# === Profiled region ===
torch.cuda.cudart().cudaProfilerStart()

# Section 1: M128 single-tile (batch=1024)
run_single(3, 3)
torch.cuda.synchronize()

# Section 2: M64 single-tile (batch=1024)
run_single(1, 1)
torch.cuda.synchronize()

# Section 3: M64+M128 dual thr=16 (batch=1024)  -- best dual at -1.3%
run_dual(1, 1, 3, 3, 16)
torch.cuda.synchronize()

# Section 4: M64+M128 dual thr=256 (batch=1024) -- best dual at -2.0%
run_dual(1, 1, 3, 3, 256)
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStop()
print("nsys profiling done")

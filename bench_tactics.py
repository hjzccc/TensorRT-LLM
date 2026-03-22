#!/usr/bin/env python3
"""Benchmark different tactic combinations for M=32 MoE."""
import torch, time, sys
torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")
from test_dual_tile_real_weights import load_real_weights, NUM_EXPERTS, TOP_K, HIDDEN, SWIGLU, DEVICE, DTYPE

fc1_w, fc2_w, quant_scales, _gate_w = load_real_weights()
runner = torch.classes.trtllm.FusedMoeRunner(DTYPE, torch.int64, DTYPE, False, False, False, False, True)

batch = 512
expert_ids = torch.arange(NUM_EXPERTS, device=DEVICE, dtype=torch.int32)
all_slots = expert_ids.repeat(batch * TOP_K // NUM_EXPERTS)
all_slots = all_slots[torch.randperm(len(all_slots), device=DEVICE)]
eidx = all_slots.reshape(batch, TOP_K)
sc = torch.ones(batch, TOP_K, dtype=torch.float32, device=DEVICE) / TOP_K
inp = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE)
isf = torch.ones(batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

NUM_BASE = 4  # M128, M64, M32, M256

configs = [
    (2, "M32-Pingpong (current)"),
    (1, "M64-Cooperative"),
    (0, "M128-Cooperative"),
]

for tactic, name in configs:
    g1 = tactic
    g2 = tactic + NUM_BASE if tactic != 3 else 0 + NUM_BASE
    
    # Warmup
    for _ in range(20):
        runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None, quant_scales, isf,
                       False, None, None, None, 1, 0, 1, 0, 1, 0, False, False,
                       [g1, g2], SWIGLU, None, None, None)
    torch.cuda.synchronize()
    
    # Benchmark
    N = 100
    start = time.perf_counter()
    for _ in range(N):
        runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None, quant_scales, isf,
                       False, None, None, None, 1, 0, 1, 0, 1, 0, False, False,
                       [g1, g2], SWIGLU, None, None, None)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / N * 1e6
    print("%s: %.1f us/iter (g1=%d, g2=%d)" % (name, elapsed, g1, g2))

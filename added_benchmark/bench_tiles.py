#!/usr/bin/env python3
"""Benchmark all tile configs at different batch sizes."""
import torch, time
import torch.nn.functional as F
torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")

from test_dual_tile_real_weights import (
    load_real_weights, generate_real_routing,
    NUM_EXPERTS, TOP_K, HIDDEN, INTER, SWIGLU, DEVICE, DTYPE
)

fc1_w, fc2_w, quant_scales, gate_w = load_real_weights()

runner = torch.classes.trtllm.FusedMoeRunner(
    DTYPE, torch.int64, DTYPE, False, False, False, False, True)

# Config indices: 0=M32, 1=M64, 3=M128
test_configs = [
    ("M32/M32",   0, 0),
    ("M32/M128",  0, 3),
    ("M64/M64",   1, 1),
    ("M128/M128", 3, 3),
]

header_names = [n for n, _, _ in test_configs]
print(f"{'Batch':>5} | {'PerExp':>6} | " + " | ".join(f"{n:>9}" for n in header_names) + " | Best")
print("-" * (5 + 3 + 6 + 3 + (9 + 3) * len(test_configs) + 6))

for batch in [16, 32, 64, 128, 256, 512, 1024]:
    experts_idx, scales = generate_real_routing(gate_w, batch)
    inp = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE)
    isf = torch.ones(batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)
    per_exp = batch * TOP_K // NUM_EXPERTS

    results = {}
    for name, g1, g2 in test_configs:
        iters = 200
        # warmup
        for _ in range(5):
            runner.run_moe(inp, experts_idx, scales, fc1_w, None, fc2_w, None,
                quant_scales, isf, False, None, None, None,
                1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            runner.run_moe(inp, experts_idx, scales, fc1_w, None, fc2_w, None,
                quant_scales, isf, False, None, None, None,
                1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)
        torch.cuda.synchronize()
        results[name] = (time.perf_counter() - t0) / iters * 1000

    best = min(results, key=results.get)
    row = f"{batch:5d} | {per_exp:6d} | "
    row += " | ".join(f"{results[n]:8.3f}ms" for n in header_names)
    row += f" | {best}"
    print(row)

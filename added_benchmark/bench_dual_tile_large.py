#!/usr/bin/env python3
"""
Benchmark dual-tile vs single-tile at large batch sizes with proper thresholds.

Threshold = max tokens per expert below which we use the small tile.
For Qwen3-30B-A3B: 128 experts, top_k=8, per_expert_avg = batch * 8 / 128 = batch / 16.

Config index mapping: 0=M32, 1=M64, 3=M128
"""
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


def bench_single(runner, inp, experts_idx, scales, isf, g1, g2, iters=200):
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
    return (time.perf_counter() - t0) / iters * 1000


def bench_dual(runner, inp, experts_idx, scales, isf, g1s, g2s, g1l, g2l, thr, iters=200):
    runner.set_dual_tile_profiles([g1s, g2s, g1l, g2l], thr)
    for _ in range(5):
        runner.run_moe_dual_tile(
            inp, experts_idx, scales, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, False, SWIGLU, None, None, None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.run_moe_dual_tile(
            inp, experts_idx, scales, fc1_w, None, fc2_w, None,
            quant_scales, isf, False, None, None, None,
            1, 0, 1, 0, False, SWIGLU, None, None, None)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


batches = [256, 512, 1024, 2048, 4096]

single_configs = [
    ("M32",  0, 0),
    ("M64",  1, 1),
    ("M128", 3, 3),
]

dual_configs = [
    ("M32+M64",  0, 0, 1, 1),
    ("M32+M128", 0, 0, 3, 3),
    ("M64+M128", 1, 1, 3, 3),
]

thresholds = [16, 32, 64, 128]

print("=" * 90)
print("PART 1: Single-tile baselines")
print("=" * 90)
hdr = f"{'Batch':>5} | {'AvgM':>4} |"
for name, _, _ in single_configs:
    hdr += f" {name:>9} |"
hdr += " Best"
print(hdr)
print("-" * len(hdr))

best_single = {}
for batch in batches:
    experts_idx, scales = generate_real_routing(gate_w, batch)
    inp = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE)
    isf = torch.ones(batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)
    avg_m = batch * TOP_K // NUM_EXPERTS

    results = {}
    for name, g1, g2 in single_configs:
        results[name] = bench_single(runner, inp, experts_idx, scales, isf, g1, g2)

    best_name = min(results, key=results.get)
    best_single[batch] = (best_name, results[best_name])

    row = f"{batch:5d} | {avg_m:4d} |"
    for name, _, _ in single_configs:
        marker = " *" if name == best_name else "  "
        row += f" {results[name]:7.3f}ms{marker}|"
    row += f" {best_name}"
    print(row)

print()
print("=" * 90)
print("PART 2: Dual-tile vs best single-tile")
print("=" * 90)

for batch in batches:
    experts_idx, scales = generate_real_routing(gate_w, batch)
    inp = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE)
    isf = torch.ones(batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)
    avg_m = batch * TOP_K // NUM_EXPERTS
    best_name, best_ms = best_single[batch]

    print(f"\n--- batch={batch}, avg per-expert M={avg_m}, best single={best_name} ({best_ms:.3f}ms) ---")
    print(f"{'Config':>12} | {'Thr':>4} |  {'Dual ms':>8} | {'Ratio':>6} | Note")
    print("-" * 60)

    for dual_name, g1s, g2s, g1l, g2l in dual_configs:
        for thr in thresholds:
            dual_ms = bench_dual(runner, inp, experts_idx, scales, isf, g1s, g2s, g1l, g2l, thr)
            ratio = best_ms / dual_ms
            note = "FASTER" if ratio < 0.99 else ("SLOWER" if ratio > 1.01 else "~same")
            print(f"{dual_name:>12} | {thr:4d} | {dual_ms:7.3f}ms | {ratio:5.2f}x | {note}")

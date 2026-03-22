#!/usr/bin/env python3
"""Benchmark ALL tactic combinations for M=32 MoE on SM120."""
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

# Query actual tactic counts
n_g1 = runner.get_tactic_num(1)
n_g2 = runner.get_tactic_num(2)
print(f"Number of GEMM1 tactics: {n_g1}")
print(f"Number of GEMM2 tactics: {n_g2}")

# SM120 FP4 tile configs (ordered as in get_candidate_configs_sm120):
# 0: 128x128x128B
# 1: 128x128x64B
# 2: 128x256x64B
# 3: 256x128x64B
# 4: 64x128x64B
# 5: 32x128x64B
tile_names = [
    "128x128x128B",
    "128x128x64B",
    "128x256x64B",
    "256x128x64B",
    "64x128x64B",
    "32x128x64B",
]

def run_bench(g1, g2, name, N=100):
    """Run benchmark with given tactic indices."""
    try:
        # Warmup
        for _ in range(20):
            runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None, quant_scales, isf,
                           False, None, None, None, 1, 0, 1, 0, 1, 0, False, False,
                           [g1, g2], SWIGLU, None, None, None)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(N):
            runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None, quant_scales, isf,
                           False, None, None, None, 1, 0, 1, 0, 1, 0, False, False,
                           [g1, g2], SWIGLU, None, None, None)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / N * 1e6
        print(f"  {name}: {elapsed:.1f} us/iter (g1={g1}, g2={g2})")
        return elapsed
    except Exception as e:
        print(f"  {name}: FAILED - {e} (g1={g1}, g2={g2})")
        return None

print(f"\n=== Testing all GEMM1 tactics (with best GEMM2=-1) ===")
results_g1 = {}
for g1 in range(n_g1):
    label = tile_names[g1] if g1 < len(tile_names) else f"tactic_{g1}"
    t = run_bench(g1, -1, f"G1={label}")
    if t:
        results_g1[g1] = t

best_g1 = min(results_g1, key=results_g1.get)
print(f"\nBest GEMM1: tactic {best_g1} ({tile_names[best_g1] if best_g1 < len(tile_names) else '?'}) = {results_g1[best_g1]:.1f} us")

print(f"\n=== Testing all GEMM2 tactics (with best GEMM1={best_g1}) ===")
results_g2 = {}
for g2 in range(n_g2):
    label = tile_names[g2] if g2 < len(tile_names) else f"tactic_{g2}"
    t = run_bench(best_g1, g2, f"G2={label}")
    if t:
        results_g2[g2] = t

best_g2 = min(results_g2, key=results_g2.get)
print(f"\nBest GEMM2: tactic {best_g2} ({tile_names[best_g2] if best_g2 < len(tile_names) else '?'}) = {results_g2[best_g2]:.1f} us")

print(f"\n=== Final best combination ===")
run_bench(best_g1, best_g2, f"G1={tile_names[best_g1]}, G2={tile_names[best_g2]}", N=200)

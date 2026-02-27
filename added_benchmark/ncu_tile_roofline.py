#!/usr/bin/env python3
"""Minimal script for ncu profiling of M32 vs M64 vs M128 vs M256 single-tile MoE GEMM.
Run with ncu to get roofline/SOL analysis per tile size.
Usage:
  ncu --set full --kernel-name-base demangled \
      --kernel-name regex:'GemmUniversal' \
      --launch-skip 5 --launch-count 3 \
      python3 ncu_tile_roofline.py <tactic_id>

  tactic_id: 0=M128, 1=M64, 2=M32, 3=M256
"""
import sys
import torch
torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")
from test_dual_tile_real_weights import (
    load_real_weights,
    NUM_EXPERTS, TOP_K, HIDDEN, SWIGLU, DEVICE, DTYPE
)

tactic = int(sys.argv[1]) if len(sys.argv) > 1 else 0
names = {0: "M128", 1: "M64", 2: "M32", 3: "M256"}
# GEMM1 tactic = tactic, GEMM2 tactic = tactic + NUM_BASE_TACTICS (FIN variant)
# Exception: M256+FIN exceeds SMEM for GEMM2, so fall back to M128+FIN for GEMM2
NUM_BASE_TACTICS = 4  # M128, M64, M32, M256
g1 = tactic
g2 = tactic + NUM_BASE_TACTICS if tactic != 3 else 0 + NUM_BASE_TACTICS  # M256 GEMM2 → M128+FIN

batch = 512  # avg M/expert = 32 — small enough that tile choice matters
print(f"Profiling tactic {tactic} ({names[tactic]}), batch={batch}, "
      f"g1={g1}, g2={g2}")

# Equal routing: each token gets experts assigned round-robin so every expert
# receives exactly (batch * TOP_K / NUM_EXPERTS) tokens.
assert (batch * TOP_K) % NUM_EXPERTS == 0, (
    f"batch*TOP_K ({batch*TOP_K}) must be divisible by NUM_EXPERTS ({NUM_EXPERTS})"
)
expert_ids = torch.arange(NUM_EXPERTS, device=DEVICE, dtype=torch.int32)
# Repeat to fill [batch, TOP_K]: assign expert round-robin across all slots
all_slots = expert_ids.repeat(batch * TOP_K // NUM_EXPERTS)  # length = batch * TOP_K
all_slots = all_slots[torch.randperm(len(all_slots), device=DEVICE)]  # shuffle
eidx = all_slots.reshape(batch, TOP_K)
sc = torch.ones(batch, TOP_K, dtype=torch.float32, device=DEVICE) / TOP_K

# Verify uniform distribution
counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=DEVICE)
counts.scatter_add_(0, eidx.reshape(-1).long(),
                     torch.ones(batch * TOP_K, dtype=torch.int32, device=DEVICE))
assert counts.min() == counts.max(), f"Non-uniform routing: min={counts.min()}, max={counts.max()}"
tokens_per_expert = counts[0].item()
print(f"Equal routing: {tokens_per_expert} tokens/expert (total slots={batch * TOP_K})")

fc1_w, fc2_w, quant_scales, _gate_w = load_real_weights()
runner = torch.classes.trtllm.FusedMoeRunner(
    DTYPE, torch.int64, DTYPE, False, False, False, False, True)

inp = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEVICE)
isf = torch.ones(batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

# Warmup (skipped by ncu with --launch-skip)
for _ in range(10):
    runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None,
        quant_scales, isf, False, None, None, None,
        1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)
    torch.cuda.synchronize()

print("Warmup done, starting profiled iterations...")
for i in range(5):
    runner.run_moe(inp, eidx, sc, fc1_w, None, fc2_w, None,
        quant_scales, isf, False, None, None, None,
        1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None)
    torch.cuda.synchronize()
print("Done.")

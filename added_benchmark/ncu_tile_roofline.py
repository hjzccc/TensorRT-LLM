#!/usr/bin/env python3
"""Deterministic single-tile profiler driver for fused vs unfused SwiGLU runs.

This script keeps the logical workload fixed across profiling runs and only changes
the FC1 physical layout to match the runtime fusion gate:
- fused gate on  -> interleaved FC1 [up_i chunk, gate_i chunk]
- fused gate off -> contiguous FC1 [up | gate]

Usage:
  ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 ncu --set full --kernel-name-base demangled \
      --kernel-name regex:'GemmUniversal' --launch-skip 10 --launch-count 5 \
      python3 added_benchmark/ncu_tile_roofline.py 0

  FORCE_UNFUSED_SWIGLU=1 nsys profile --trace=cuda,nvtx \
      --output=.sisyphus/evidence/task-7-single-tile-unfused \
      python3 added_benchmark/ncu_tile_roofline.py 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

try:
    from .test_single_tile_swiglu_fusion import (
        DEFAULT_ACTIVATION,
        DEFAULT_QUANTIZATION,
        NUM_EXPERTS,
        TACTIC_NAMES,
        build_single_tile_case,
        call_run_moe,
        compute_fusion_gate_report,
        create_runner,
        load_trtllm_library,
        make_deterministic_inputs,
        resolve_fc1_layout,
        validate_fc1_layout_contract,
    )
except ImportError:
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from added_benchmark.test_single_tile_swiglu_fusion import (
        DEFAULT_ACTIVATION,
        DEFAULT_QUANTIZATION,
        NUM_EXPERTS,
        TACTIC_NAMES,
        build_single_tile_case,
        call_run_moe,
        compute_fusion_gate_report,
        create_runner,
        load_trtllm_library,
        make_deterministic_inputs,
        resolve_fc1_layout,
        validate_fc1_layout_contract,
    )


DEFAULT_BATCH = 512
DEFAULT_SEED = 1234
DEFAULT_WARMUP_ITERS = 10
DEFAULT_PROFILE_ITERS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic single-tile MoE profiler driver")
    parser.add_argument("tactic", nargs="?", type=int, default=0, choices=sorted(TACTIC_NAMES))
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--profile-iters", type=int, default=DEFAULT_PROFILE_ITERS)
    parser.add_argument("--layout", choices=("auto", "fused", "unfused"), default="auto")
    args = parser.parse_args()

    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")
    if args.profile_iters <= 0:
        parser.error("--profile-iters must be positive")
    return args


def main() -> None:
    args = parse_args()
    load_trtllm_library()

    shared_inputs = make_deterministic_inputs(seed=args.seed, num_tokens=args.batch)
    validate_fc1_layout_contract(shared_inputs)

    gate_report = compute_fusion_gate_report(
        args.tactic,
        activation=DEFAULT_ACTIVATION,
        quantization=DEFAULT_QUANTIZATION,
        use_prequant_scale=False,
    )
    layout = args.layout
    if layout == "auto":
        layout = resolve_fc1_layout(
            args.tactic,
            activation=DEFAULT_ACTIVATION,
            quantization=DEFAULT_QUANTIZATION,
            use_prequant_scale=False,
        )

    case = build_single_tile_case(shared_inputs, args.tactic, layout)
    runner = create_runner()

    routing_counts = torch.bincount(shared_inputs["topk_idx"].reshape(-1).to(torch.int64), minlength=NUM_EXPERTS)
    print(
        f"Profiling tactic {args.tactic} ({TACTIC_NAMES[args.tactic]}), batch={args.batch}, layout={layout}, "
        + f"profile_ids={case['profile_ids']}"
    )
    print(f"FUSION_GATE={1 if gate_report['enabled'] else 0}")
    print(f"FUSION_REASON={gate_report['reason']}")
    print(
        "Routing distribution: "
        + f"min={int(routing_counts.min().item())} max={int(routing_counts.max().item())} "
        + f"total_slots={int(routing_counts.sum().item())}"
    )

    for _ in range(args.warmup_iters):
        call_run_moe(runner, shared_inputs, case)
    torch.cuda.synchronize()

    print("Warmup done, starting profiled iterations...")
    torch.cuda.nvtx.range_push(f"single_tile_swiglu_{layout}_tactic_{args.tactic}")
    for _ in range(args.profile_iters):
        call_run_moe(runner, shared_inputs, case)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("Done.")


if __name__ == "__main__":
    main()

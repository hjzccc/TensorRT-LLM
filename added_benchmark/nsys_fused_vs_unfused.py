#!/usr/bin/env python3
"""Single-process nsys capture comparing fused vs unfused for M128/M64/M32.

Because the C++ fusion gate caches env vars on first read, we cannot toggle
mid-process. Instead we run all sections in one process but set the env vars
BEFORE importing/loading the library.

Usage:
  ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 nsys profile --trace=cuda,nvtx \
      -o evidence_file python3 nsys_fused_vs_unfused.py --mode fused
  nsys profile --trace=cuda,nvtx \
      -o evidence_file python3 nsys_fused_vs_unfused.py --mode unfused
"""
import argparse
import os
import sys

sys.path.insert(0, "/code/tensorrt_llm/added_benchmark")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fused", "unfused"], required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    # Set env BEFORE any C++ code is loaded
    if args.mode == "fused":
        os.environ["ENABLE_SINGLE_TILE_SWIGLU_FUSION"] = "1"
        os.environ.pop("FORCE_UNFUSED_SWIGLU", None)
    else:
        os.environ.pop("ENABLE_SINGLE_TILE_SWIGLU_FUSION", None)
        os.environ.pop("FORCE_UNFUSED_SWIGLU", None)

    import torch
    torch.ops.load_library("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so")
    from test_dual_tile_real_weights import (
        load_real_weights, generate_real_routing,
        NUM_EXPERTS, TOP_K, HIDDEN, SWIGLU, DEVICE, DTYPE,
    )

    fc1_w, fc2_w, quant_scales, gate_w = load_real_weights()
    runner = torch.classes.trtllm.FusedMoeRunner(
        DTYPE, torch.int64, DTYPE, False, False, False, False, True
    )

    eidx, sc = generate_real_routing(gate_w, args.batch)
    inp = torch.randn(args.batch, HIDDEN, dtype=DTYPE, device=DEVICE)
    isf = torch.ones(args.batch, HIDDEN // 16, dtype=torch.float8_e4m3fn, device=DEVICE)

    def run(g1, g2, iters):
        for _ in range(iters):
            runner.run_moe(
                inp, eidx, sc, fc1_w, None, fc2_w, None,
                quant_scales, isf, False, None, None, None,
                1, 0, 1, 0, 1, 0, False, False, [g1, g2], SWIGLU, None, None, None,
            )

    # Tactic mapping (current build):
    #   GEMM1: 0=M128, 1=M64, 2=M32
    #   GEMM2: 4=M128, 5=M64, 6=M32
    tiles = [
        ("M128", 0, 4),
        ("M64",  1, 5),
        ("M32",  2, 6),
    ]

    for name, g1, g2 in tiles:
        run(g1, g2, 10)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for name, g1, g2 in tiles:
        label = f"{args.mode}_{name}_b{args.batch}"
        torch.cuda.nvtx.range_push(label)
        run(g1, g2, args.iters)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"nsys {args.mode} profiling done")


if __name__ == "__main__":
    main()

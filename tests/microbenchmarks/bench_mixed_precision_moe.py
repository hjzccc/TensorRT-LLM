# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark: Fused Mixed-Precision MoE vs. Separate-Call HeterMoE.

NEW: Compares the fused mixed-precision MoE kernel (single C++ call for
bf16 + fp4 groups) against the HeterCutlassFusedMoE baseline (separate
fused_moe() calls per precision group).

The benchmark sweeps:
  - seq_len: [32, 64, 128, 256, 512]
  - num_high_precision_experts: [E//8, E//4, E//2]

For each configuration, it reports:
  - HeterMoE latency (ms)
  - FusedMixedPrecMoE latency (ms)
  - Speedup (ratio)

All timing is:
  - L2 cache aware (flush before each timed run)
  - CUDA graph captured for accurate timing
  - Median of multiple runs to reduce variance

Usage:
    python tests/microbenchmarks/bench_mixed_precision_moe.py

Requirements:
    - GPU with SM >= 100 (Blackwell) for NVFP4 support
    - The fused mixed-precision MoE kernel and HeterCutlassFusedMoE both available
"""

import copy
import csv
import sys
import time
from pathlib import Path

# Add tests/unittest to sys.path for shared utils
_UNITTEST_DIR = str(Path(__file__).resolve().parents[1] / "unittest")
if _UNITTEST_DIR not in sys.path:
    sys.path.insert(0, _UNITTEST_DIR)

import torch

from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
WARMUP_ITERS = 10
BENCH_ITERS = 50
NUM_TIMING_RUNS = 5

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
SEQ_LENS = [32, 64, 128, 256, 512]
HIGH_PREC_FRACTIONS = [1 / 8, 1 / 4, 1 / 2]

CSV_PATH = "bench_mixed_precision_moe.csv"


def flush_l2_cache(size_mb: int = 40):
    """Flush GPU L2 cache by writing a large buffer."""
    size_bytes = size_mb * 1024 * 1024
    flush_buf = torch.empty(size_bytes // 4, dtype=torch.float32, device="cuda")
    flush_buf.fill_(1.0)
    torch.cuda.synchronize()
    del flush_buf


def time_forward_cuda_graph(forward_fn, warmup_iters=WARMUP_ITERS, bench_iters=BENCH_ITERS):
    """Time a forward function using CUDA graph capture and replay.

    Returns median per-iteration time in milliseconds.
    """
    # Warmup
    for _ in range(warmup_iters):
        forward_fn()
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(bench_iters):
            forward_fn()
    torch.cuda.synchronize()

    # Graph warmup
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(NUM_TIMING_RUNS):
        flush_l2_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / bench_iters
        times.append(elapsed_ms)

    return sorted(times)[len(times) // 2]  # median


def time_forward_eager(forward_fn, warmup_iters=WARMUP_ITERS, bench_iters=BENCH_ITERS):
    """Fallback: time without CUDA graphs."""
    for _ in range(warmup_iters):
        forward_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(NUM_TIMING_RUNS):
        flush_l2_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(bench_iters):
            forward_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / bench_iters
        times.append(elapsed_ms)

    return sorted(times)[len(times) // 2]


def create_heter_backend(
    routing_method, mapping, num_experts, hidden_size, intermediate_size, dtype,
    bf16_weights, fp4_weights, bf16_ratio,
):
    """Create HeterCutlassFusedMoE backend with mixed bf16/fp4 groups."""
    from _torch.modules.moe.heter_moe_utils import (
        create_backend,
        create_model_config,
        mixed_heter_config,
    )
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import (
        HeterCutlassFusedMoE,
    )

    heter_config = mixed_heter_config(bf16_ratio=bf16_ratio)
    model_config = create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="HETER",
        mapping=mapping,
        heter_config=heter_config,
    )
    backend = create_backend(
        moe_cls=HeterCutlassFusedMoE,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        model_config=model_config,
    )
    # HeterCutlassFusedMoE expects per-group weights
    # For the benchmark, we load bf16 weights for the bf16 group
    # and use the same bf16 weights (it handles quantization internally)
    backend.load_weights([copy.deepcopy(bf16_weights)])
    backend.post_load_weights()
    backend.cuda()
    return backend


def create_fused_mixed_prec_module(
    num_experts, hidden_size, intermediate_size, top_k,
    num_high_precision, bf16_weights, fp4_weights,
):
    """Create FusedMixedPrecisionMoE module."""
    from _torch.modules.moe.mixed_precision_moe_utils import (
        create_fused_mixed_precision_module,
    )
    return create_fused_mixed_precision_module(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_high_precision_experts=num_high_precision,
        bf16_weights=bf16_weights,
        fp4_weights=fp4_weights,
    )


def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.")
        return

    sm = torch.cuda.get_device_capability()
    if sm[0] < 10:
        print(f"WARNING: SM {sm[0]}.{sm[1]} < 10.0. FP4 not supported. "
              "Benchmark will only test HeterMoE with bf16 groups.")

    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name} (SM {sm[0]}.{sm[1]})")
    print(f"Config: E={NUM_EXPERTS}, top_k={TOP_K}, hidden={HIDDEN_SIZE}, "
          f"inter={INTERMEDIATE_SIZE}")
    print(f"Warmup={WARMUP_ITERS}, Bench={BENCH_ITERS}, Runs={NUM_TIMING_RUNS}")
    print("=" * 80)

    mapping = Mapping()
    mapping.rank = mpi_rank()
    routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)

    results = []
    header = ["seq_len", "num_high_prec", "bf16_ratio",
              "heter_ms", "fused_ms", "speedup"]

    for seq_len in SEQ_LENS:
        with torch.device(f"cuda:{mapping.rank}"):
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            x = torch.randn((seq_len, HIDDEN_SIZE), dtype=DTYPE, device="cuda")
            router_logits = torch.randn(
                (seq_len, NUM_EXPERTS), dtype=DTYPE, device="cuda"
            )
            all_rank_num_tokens = [seq_len] * mapping.world_size

            # Create shared weights
            from _torch.modules.moe.mixed_precision_moe_utils import (
                create_unquantized_weights,
                quantize_bf16_to_nvfp4,
            )

            bf16_weights = create_unquantized_weights(
                num_experts=NUM_EXPERTS,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
                dtype=DTYPE,
            )
            fp4_weights = quantize_bf16_to_nvfp4(
                bf16_weights, NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, DTYPE, x
            )

            for frac in HIGH_PREC_FRACTIONS:
                num_high_prec = max(1, int(NUM_EXPERTS * frac))
                bf16_ratio = frac

                print(f"\nseq_len={seq_len}, num_high_prec={num_high_prec} "
                      f"({bf16_ratio*100:.0f}% bf16)")

                # --- HeterMoE (separate calls) ---
                try:
                    heter_backend = create_heter_backend(
                        routing_method=routing_method,
                        mapping=mapping,
                        num_experts=NUM_EXPERTS,
                        hidden_size=HIDDEN_SIZE,
                        intermediate_size=INTERMEDIATE_SIZE,
                        dtype=DTYPE,
                        bf16_weights=bf16_weights,
                        fp4_weights=fp4_weights,
                        bf16_ratio=bf16_ratio,
                    )

                    def heter_fn():
                        with torch.inference_mode():
                            return heter_backend.forward(
                                x, router_logits,
                                all_rank_num_tokens=all_rank_num_tokens,
                            )

                    try:
                        heter_ms = time_forward_cuda_graph(heter_fn)
                    except RuntimeError:
                        heter_ms = time_forward_eager(heter_fn)
                    print(f"  HeterMoE:     {heter_ms:.3f} ms")
                except Exception as e:
                    print(f"  HeterMoE:     FAILED ({e})")
                    heter_ms = float("nan")

                # --- Fused Mixed-Precision MoE (single call) ---
                try:
                    fused_module = create_fused_mixed_prec_module(
                        num_experts=NUM_EXPERTS,
                        hidden_size=HIDDEN_SIZE,
                        intermediate_size=INTERMEDIATE_SIZE,
                        top_k=TOP_K,
                        num_high_precision=num_high_prec,
                        bf16_weights=bf16_weights,
                        fp4_weights=fp4_weights,
                    )

                    def fused_fn():
                        with torch.inference_mode():
                            return fused_module.forward(
                                x, router_logits,
                                routing_method=routing_method,
                            )

                    try:
                        fused_ms = time_forward_cuda_graph(fused_fn)
                    except RuntimeError:
                        fused_ms = time_forward_eager(fused_fn)
                    print(f"  FusedMixPrec:  {fused_ms:.3f} ms")
                except Exception as e:
                    print(f"  FusedMixPrec:  FAILED ({e})")
                    fused_ms = float("nan")

                # Speedup
                if heter_ms > 0 and fused_ms > 0:
                    speedup = heter_ms / fused_ms
                    print(f"  Speedup:       {speedup:.2f}x")
                else:
                    speedup = float("nan")

                results.append([
                    seq_len, num_high_prec, f"{bf16_ratio:.2f}",
                    f"{heter_ms:.3f}", f"{fused_ms:.3f}", f"{speedup:.2f}",
                ])

                # Cleanup to free GPU memory
                del heter_backend, fused_module
                torch.cuda.empty_cache()

    # --- Print results table ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    col_widths = [10, 14, 12, 12, 12, 10]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*header))
    print("-" * sum(col_widths))
    for row in results:
        print(fmt.format(*row))

    # --- Save CSV ---
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"\nResults saved to {CSV_PATH}")


if __name__ == "__main__":
    main()

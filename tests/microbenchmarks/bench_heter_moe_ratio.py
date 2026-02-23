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
"""Heter MoE BF16/NVFP4 ratio sweep benchmark.

Sweeps BF16/NVFP4 expert ratio for HeterCutlassFusedMoE and prints a
results table.  Each row is a bf16_ratio, columns are different seq_lens
(batch sizes).  Output is also saved as CSV for easy plotting.

Usage:
    python tests/microbenchmarks/bench_heter_moe_ratio.py

Edit the "Sweep configuration" globals below to control the parameter grid.
"""

import copy
import csv
import os

os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"

import torch

from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import HeterCutlassFusedMoE
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# ---------------------------------------------------------------------------
# Model configuration  (edit these)
# ---------------------------------------------------------------------------
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Benchmark configuration  (edit these)
# ---------------------------------------------------------------------------
WARMUP_ITERS = 10
BENCH_ITERS = 20

# torch.compile + CUDA graph flags
ENABLE_TORCH_COMPILE = True
ENABLE_CUDA_GRAPHS = True

# ---------------------------------------------------------------------------
# Sweep configuration  (edit these)
#   BF16_RATIOS: fraction of experts kept at BF16 (0.0 = all NVFP4, 1.0 = all BF16)
#   SEQ_LENS:    number of tokens per forward pass (one line per value in the plot)
# ---------------------------------------------------------------------------
BF16_RATIOS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
SEQ_LENS = [1, 4, 16, 64, 256, 512]

# Output CSV path (relative to cwd)
CSV_PATH = "bench_heter_moe_ratio.csv"


# ---------------------------------------------------------------------------
# Helpers  (self-contained copies from test_heter_moe.py)
# ---------------------------------------------------------------------------


def _create_model_config(
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    moe_backend,
    mapping,
    quant_config=None,
    heter_config=None,
):
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    kwargs = {
        "pretrained_config": pretrained_config,
        "mapping": mapping,
        "moe_backend": moe_backend,
    }
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    model_config = ModelConfig(**kwargs)

    if heter_config is not None:
        model_config.extra_attrs["heter_moe_config"] = heter_config

    return model_config


def _create_backend(
    moe_cls,
    routing_method,
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    model_config,
):
    return create_moe_backend(
        moe_cls=moe_cls,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
    )


def _create_unquantized_weights(
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    kaiming_fan_out=True,
):
    weights = {}
    for eid in range(num_experts):
        w1 = torch.randn(
            (intermediate_size, hidden_size), dtype=dtype, device="cuda")
        w2 = torch.randn(
            (hidden_size, intermediate_size), dtype=dtype, device="cuda")
        w3 = torch.randn(
            (intermediate_size, hidden_size), dtype=dtype, device="cuda")
        weights[f"{eid}.w1.weight"] = w1
        weights[f"{eid}.w2.weight"] = w2
        weights[f"{eid}.w3.weight"] = w3

    if kaiming_fan_out:
        for key, val in weights.items():
            if isinstance(val, torch.Tensor) and val.ndim == 2:
                fan_out = val.shape[0]
                weights[key] = val * (2.0 / fan_out) ** 0.5

    return weights


def _quantize_bf16_to_nvfp4(
    bf16_weights,
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    x,
    scaling_vector_size=16,
):
    x_sf_global = (448 * 6) / x.abs().max().float()
    weights = {}
    for eid in range(num_experts):
        w1 = bf16_weights[f"{eid}.w1.weight"]
        w2 = bf16_weights[f"{eid}.w2.weight"]
        w3 = bf16_weights[f"{eid}.w3.weight"]

        w1_sf = (448 * 6) / w1.abs().max().float()
        w2_sf = (448 * 6) / w2.abs().max().float()
        w3_sf = (448 * 6) / w3.abs().max().float()
        w3_w1_sf = min(w1_sf, w3_sf)

        w1_q, w1_sb = torch.ops.trtllm.fp4_quantize(
            w1, w3_w1_sf, scaling_vector_size, False, False)
        w1_sb = w1_sb.view(intermediate_size, -1)

        w2_q, w2_sb = torch.ops.trtllm.fp4_quantize(
            w2, w2_sf, scaling_vector_size, False, False)
        w2_sb = w2_sb.view(hidden_size, -1)

        w3_q, w3_sb = torch.ops.trtllm.fp4_quantize(
            w3, w3_w1_sf, scaling_vector_size, False, False)
        w3_sb = w3_sb.view(intermediate_size, -1)

        weights[f"{eid}.w1.weight"] = w1_q
        weights[f"{eid}.w2.weight"] = w2_q
        weights[f"{eid}.w3.weight"] = w3_q
        weights[f"{eid}.w1.weight_scale"] = w1_sb.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{eid}.w2.weight_scale"] = w2_sb.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{eid}.w3.weight_scale"] = w3_sb.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{eid}.w1.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{eid}.w2.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{eid}.w3.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{eid}.w1.weight_scale_2"] = 1.0 / w3_w1_sf
        weights[f"{eid}.w2.weight_scale_2"] = 1.0 / w2_sf
        weights[f"{eid}.w3.weight_scale_2"] = 1.0 / w3_w1_sf
    return weights


def _create_cutlass_backend(
    routing_method,
    mapping,
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    weights,
    quant_config=None,
):
    model_config = _create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="CUTLASS",
        mapping=mapping,
        quant_config=quant_config,
    )
    backend = _create_backend(
        moe_cls=CutlassFusedMoE,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        model_config=model_config,
    )
    backend.load_weights([copy.deepcopy(weights)])
    backend.post_load_weights()
    backend.cuda()
    return backend


# ---------------------------------------------------------------------------
# Heter config factories
# ---------------------------------------------------------------------------


def _all_bf16_heter_config():
    return {
        "groups": [{
            "name": "all_bf16",
            "quant_algo": None,
            "size_ratio": 1.0,
            "checkpoint": None,
        }],
    }


def _all_nvfp4_heter_config():
    return {
        "groups": [{
            "name": "all_nvfp4",
            "quant_algo": QuantAlgo.NVFP4,
            "size_ratio": 1.0,
            "checkpoint": None,
        }],
    }


def _mixed_heter_config(bf16_ratio):
    return {
        "groups": [
            {
                "name": "cold_nvfp4",
                "quant_algo": QuantAlgo.NVFP4,
                "size_ratio": round(1.0 - bf16_ratio, 4),
                "checkpoint": None,
            },
            {
                "name": "hot_bf16",
                "quant_algo": None,
                "size_ratio": bf16_ratio,
                "checkpoint": None,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------


def _benchmark(backend, x, router_logits, all_rank_num_tokens,
               warmup_iters=WARMUP_ITERS, bench_iters=BENCH_ITERS,
               timed_replays=5):
    """Compile, capture CUDA graph, warm up, and time.

    1. Optional torch.compile wrapping.
    2. Pre-capture warmup: ``warmup_iters`` eager forward passes
       (triggers compilation on first call).
    3. Capture a CUDA graph containing ``bench_iters`` forward passes.
    4. Graph warmup: replay the captured graph ``warmup_iters`` times.
    5. Time ``timed_replays`` individual graph replays, return the
       minimum per-iteration time in milliseconds.

    When CUDA graphs are disabled, falls back to timing ``bench_iters``
    individual eager calls and returning the median.
    """
    forward_fn = backend.forward
    if ENABLE_TORCH_COMPILE:
        from tensorrt_llm._torch.compilation.backend import Backend as TrtBackend
        forward_fn = torch.compile(forward_fn, backend=TrtBackend())

    def fn():
        with torch.inference_mode():
            return forward_fn(
                x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
            )

    # --- Pre-capture warmup (also triggers torch.compile) ---
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    if ENABLE_CUDA_GRAPHS:
        # Capture bench_iters forward passes in a single graph.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(bench_iters):
                fn()
        torch.cuda.synchronize()

        # Warm up the captured graph.
        for _ in range(warmup_iters):
            graph.replay()
        torch.cuda.synchronize()

        # Time multiple replays, take the minimum per-iteration time.
        best_ms = float("inf")
        for _ in range(timed_replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            best_ms = min(best_ms, start.elapsed_time(end) / bench_iters)

        return best_ms

    # --- Eager fallback: time individual calls, return median ---
    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)
    ]
    for i in range(bench_iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def bench_one(seq_len, bf16_ratio, bf16_weights, nvfp4_cutlass, mapping):
    """Benchmark one (seq_len, bf16_ratio) configuration.

    Creates a HeterCutlassFusedMoE backend, warms up with torch.compile +
    optional CUDA graph, times it, cleans up, and returns the median
    runtime in milliseconds.
    """
    routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
    all_rank_num_tokens = [seq_len] * mapping.world_size

    x = torch.randn((seq_len, HIDDEN_SIZE), dtype=DTYPE, device="cuda")
    router_logits = torch.randn(
        (seq_len, NUM_EXPERTS), dtype=DTYPE, device="cuda")

    # --- Select heter config based on ratio ---
    if bf16_ratio <= 0.0:
        heter_config = _all_nvfp4_heter_config()
    elif bf16_ratio >= 1.0:
        heter_config = _all_bf16_heter_config()
    else:
        heter_config = _mixed_heter_config(bf16_ratio=bf16_ratio)

    model_config = _create_model_config(
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=DTYPE,
        moe_backend="HETER",
        mapping=mapping,
        heter_config=heter_config,
    )
    backend = _create_backend(
        moe_cls=HeterCutlassFusedMoE,
        routing_method=routing_method,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=DTYPE,
        model_config=model_config,
    )
    backend.load_weights([copy.deepcopy(bf16_weights)])
    backend.post_load_weights()
    backend.cuda()

    # Register NVFP4 group weights when the config has an NVFP4 group.
    if bf16_ratio < 1.0:
        backend.register_group_weights(
            group_idx=0,
            w3_w1_weight=nvfp4_cutlass.w3_w1_weight.data,
            w2_weight=nvfp4_cutlass.w2_weight.data,
            quant_scales=nvfp4_cutlass.quant_scales,
            weight_dtype=nvfp4_cutlass.w3_w1_weight.dtype,
            fc31_input_scale=nvfp4_cutlass.fc31_input_scale,
            scaling_vector_size=getattr(
                nvfp4_cutlass, "scaling_vector_size", 16),
        )

    # --- Compile + CUDA-graph + time ---
    time_ms = _benchmark(backend, x, router_logits, all_rank_num_tokens)

    # Cleanup to free GPU memory between runs.
    del backend, x, router_logits
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    return time_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    mapping = Mapping()
    mapping.rank = mpi_rank()

    print(f"Model : {NUM_EXPERTS} experts, top_k={TOP_K}, "
          f"hidden={HIDDEN_SIZE}, inter={INTERMEDIATE_SIZE}, dtype={DTYPE}")
    print(f"Bench : compile={ENABLE_TORCH_COMPILE}, "
          f"cuda_graphs={ENABLE_CUDA_GRAPHS}, "
          f"warmup={WARMUP_ITERS}, iters={BENCH_ITERS}")
    print(f"Sweep : {len(BF16_RATIOS)} ratios × {len(SEQ_LENS)} seq_lens "
          f"= {len(BF16_RATIOS) * len(SEQ_LENS)} runs")
    print()

    with torch.device(f"cuda:{mapping.rank}"):
        # ------------------------------------------------------------------
        # One-time setup: create BF16 weights and NVFP4 reference backend
        # ------------------------------------------------------------------
        print("Creating BF16 weights …")
        bf16_weights = _create_unquantized_weights(
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=DTYPE,
        )

        print("Quantizing to NVFP4 …")
        x_dummy = torch.randn(
            (max(SEQ_LENS), HIDDEN_SIZE), dtype=DTYPE, device="cuda")
        nvfp4_weights = _quantize_bf16_to_nvfp4(
            bf16_weights=bf16_weights,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=DTYPE,
            x=x_dummy,
        )
        del x_dummy

        print("Creating NVFP4 reference backend …")
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        nvfp4_cutlass = _create_cutlass_backend(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=DTYPE,
            weights=nvfp4_weights,
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        )

        # ------------------------------------------------------------------
        # Sweep
        # ------------------------------------------------------------------
        results = {}  # results[seq_len][bf16_ratio] = time_ms
        total = len(SEQ_LENS) * len(BF16_RATIOS)
        done = 0

        print(f"\nRunning {total} benchmarks …\n")

        for seq_len in SEQ_LENS:
            results[seq_len] = {}
            for ratio in BF16_RATIOS:
                done += 1
                time_ms = bench_one(
                    seq_len, ratio, bf16_weights, nvfp4_cutlass, mapping)
                results[seq_len][ratio] = time_ms
                print(f"  [{done:>3}/{total}]  seq_len={seq_len:>4}  "
                      f"bf16_ratio={ratio:.3f}  →  {time_ms:.3f} ms")

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("RESULTS  (median ms per forward pass)")
    print("=" * 80)

    col_w = 12
    header = f"{'bf16_ratio':>{col_w}}"
    for s in SEQ_LENS:
        header += f"{'seq=' + str(s):>{col_w}}"
    print(header)
    print("-" * len(header))

    for ratio in BF16_RATIOS:
        row = f"{ratio:>{col_w}.3f}"
        for seq_len in SEQ_LENS:
            row += f"{results[seq_len][ratio]:>{col_w}.3f}"
        print(row)

    # ------------------------------------------------------------------
    # Save CSV  (for plotting: x = bf16_ratio, y = time, lines = seq_len)
    # ------------------------------------------------------------------
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bf16_ratio"] + [f"seq_{s}" for s in SEQ_LENS])
        for ratio in BF16_RATIOS:
            writer.writerow(
                [ratio] + [results[s][ratio] for s in SEQ_LENS])

    print(f"\nCSV saved to {CSV_PATH}")


if __name__ == "__main__":
    main()

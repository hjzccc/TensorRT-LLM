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
"""Benchmark fused MoE kernel under skewed vs even per-expert token distributions.
Usage:
    python test_fused_moe_distribution_benchmark.py
"""

import copy
import os

os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"
import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
ENABLE_TORCH_COMPILE = True
ENABLE_CUDA_GRAPHS = True
_WARMUP_ITERS = 10
_BENCH_ITERS = 100
# ---------------------------------------------------------------------------
# MoE configuration
# ---------------------------------------------------------------------------
NUM_EXPERTS = 31
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1536
DTYPE = torch.bfloat16
TOP_K = 1
SEQ_LEN = 6144


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def make_skewed_distribution(shuffle=False):
    """Heavy-hitter distribution, optionally shuffled across experts."""
    head = [4000, 1200, 500, 200, 100, 20, 4]
    tail_count = NUM_EXPERTS - len(head)
    remaining = SEQ_LEN - sum(head)
    base, extra = divmod(remaining, tail_count)
    tail = [base + 1] * extra + [base] * (tail_count - extra)
    dist = head + tail
    assert len(dist) == NUM_EXPERTS and sum(dist) == SEQ_LEN
    if shuffle:
        import random
        random.seed(123)
        random.shuffle(dist)
    return dist


def make_even_distribution():
    base, extra = divmod(SEQ_LEN, NUM_EXPERTS)
    dist = [base + 1] * extra + [base] * (NUM_EXPERTS - extra)
    assert len(dist) == NUM_EXPERTS and sum(dist) == SEQ_LEN
    return dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_unquantized_weights(num_experts, hidden_size, intermediate_size,
                               dtype):
    weights = {}
    for i in range(num_experts):
        weights[f"{i}.w1.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=dtype, device="cuda")
        weights[f"{i}.w2.weight"] = torch.randn(
            hidden_size, intermediate_size, dtype=dtype, device="cuda")
        weights[f"{i}.w3.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=dtype, device="cuda")
    for key, val in weights.items():
        if isinstance(val, torch.Tensor) and val.ndim == 2:
            fan_out = val.shape[0]
            weights[key] = val * (2.0 / fan_out) ** 0.5
    return weights


def create_cutlass_backend(routing_method, mapping, num_experts, hidden_size,
                           intermediate_size, dtype, weights, seq_len):
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.dtype = dtype
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        moe_backend="CUTLASS",
        max_num_tokens=seq_len,
    )

    backend = CutlassFusedMoE(
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
    )
    backend.load_weights([copy.deepcopy(weights)])
    backend.post_load_weights()
    backend.cuda()
    return backend


def distribution_to_router_logits(distribution, num_experts, dtype):
    """Craft router_logits so argmax routing matches *distribution* exactly."""
    total_tokens = sum(distribution)
    logits = torch.full(
        (total_tokens, num_experts), -100.0, dtype=dtype, device="cuda",
    )
    offset = 0
    for expert_id, count in enumerate(distribution):
        logits[offset:offset + count, expert_id] = 100.0
        offset += count
    return logits


def prepare_runner(backend, x, router_logits, all_rank_num_tokens,
                   warmup_iters=_WARMUP_ITERS):
    forward_fn = backend.forward

    if ENABLE_TORCH_COMPILE:
        from tensorrt_llm._torch.compilation.backend import Backend as TrtBackend
        forward_fn = torch.compile(forward_fn, backend=TrtBackend())

    def fn():
        with torch.inference_mode():
            return forward_fn(
                x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
            )

    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    if ENABLE_CUDA_GRAPHS:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        return graph.replay

    return fn


def benchmark_timed(runner, bench_iters=_BENCH_ITERS):
    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)
    ]

    for i in range(bench_iters):
        start_events[i].record()
        runner()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    with torch.device("cuda"):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        mapping = Mapping()
        mapping.rank = mpi_rank()
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        skewed_sorted = make_skewed_distribution(shuffle=False)
        skewed_shuffled = make_skewed_distribution(shuffle=True)
        even_dist = make_even_distribution()

        x = torch.randn(
            (SEQ_LEN, HIDDEN_SIZE), dtype=DTYPE, device="cuda",
        )
        all_rank_num_tokens = [SEQ_LEN] * mapping.world_size

        sorted_logits = distribution_to_router_logits(
            skewed_sorted, NUM_EXPERTS, DTYPE,
        )
        shuffled_logits = distribution_to_router_logits(
            skewed_shuffled, NUM_EXPERTS, DTYPE,
        )
        even_logits = distribution_to_router_logits(
            even_dist, NUM_EXPERTS, DTYPE,
        )

        print(f"[config] experts={NUM_EXPERTS}, tokens={SEQ_LEN}, "
              f"hidden={HIDDEN_SIZE}, inter={INTERMEDIATE_SIZE}, "
              f"top_k={TOP_K}")
        print(f"[config] Skewed-sorted:   {skewed_sorted}")
        print(f"[config] Skewed-shuffled: {skewed_shuffled}")
        print(f"[config] Even:            {even_dist}")
        weights = create_unquantized_weights(
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=DTYPE,
        )

        def make_backend():
            return create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=NUM_EXPERTS,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
                dtype=DTYPE,
                weights=weights,
                seq_len=SEQ_LEN,
            )

        backend_sorted = make_backend()
        backend_shuffled = make_backend()
        backend_even = make_backend()

        sorted_runner = prepare_runner(
            backend_sorted, x, sorted_logits, all_rank_num_tokens,
        )
        shuffled_runner = prepare_runner(
            backend_shuffled, x, shuffled_logits, all_rank_num_tokens,
        )
        even_runner = prepare_runner(
            backend_even, x, even_logits, all_rank_num_tokens,
        )

        sorted_ms = benchmark_timed(sorted_runner)
        shuffled_ms = benchmark_timed(shuffled_runner)
        even_ms = benchmark_timed(even_runner)
        print(f"[runtime] Skewed-sorted   = {sorted_ms:.3f} ms")
        print(f"[runtime] Skewed-shuffled = {shuffled_ms:.3f} ms")
        print(f"[runtime] Even            = {even_ms:.3f} ms")
        print(f"[runtime] ratio(sorted/even)   = {sorted_ms / even_ms:.3f}x")
        print(f"[runtime] ratio(shuffled/even)  = {shuffled_ms / even_ms:.3f}x")
        print(f"[runtime] ratio(sorted/shuffled)= {sorted_ms / shuffled_ms:.3f}x")


if __name__ == "__main__":
    main()

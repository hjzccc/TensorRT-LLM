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

import copy

import pytest
import torch
from _torch.modules.moe.heter_moe_utils import (
    NVFP4_UNAVAILABLE_REASON,
    create_backend,
    create_cutlass_backend,
    create_model_config,
    create_unquantized_weights,
    mixed_heter_config,
    nvfp4_supported,
    quantize_bf16_to_nvfp4,
)

from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import HeterCutlassFusedMoE
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

ENABLE_TORCH_COMPILE = True
ENABLE_CUDA_GRAPHS = True

_WARMUP_ITERS = 10
_BENCH_ITERS = 1


def _prepare_single_runner(
    backend,
    x,
    router_logits,
    all_rank_num_tokens,
    warmup_iters=_WARMUP_ITERS,
):
    forward_fn = backend.forward

    if ENABLE_TORCH_COMPILE:
        from tensorrt_llm._torch.compilation.backend import Backend as TrtBackend
        forward_fn = torch.compile(forward_fn, backend=TrtBackend())

    def fn():
        with torch.inference_mode():
            return forward_fn(
                x,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
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


def _benchmark_timed(runner, bench_iters=_BENCH_ITERS):
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


def _create_mixed_heter_backend(
    *,
    mapping,
    routing_method,
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    x,
    bf16_weights,
):
    nvfp4_weights = quantize_bf16_to_nvfp4(
        bf16_weights=bf16_weights,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        x=x,
    )
    nvfp4_backend = create_cutlass_backend(
        routing_method=routing_method,
        mapping=mapping,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        weights=nvfp4_weights,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
    )

    heter_model_config = create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="HETER",
        mapping=mapping,
        heter_config=mixed_heter_config(bf16_ratio=0.5),
    )
    heter_backend = create_backend(
        moe_cls=HeterCutlassFusedMoE,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        model_config=heter_model_config,
    )
    heter_backend.load_weights([copy.deepcopy(bf16_weights)])
    heter_backend.post_load_weights()
    heter_backend.cuda()
    heter_backend.register_group_weights(
        group_idx=0,
        w3_w1_weight=nvfp4_backend.w3_w1_weight.data,
        w2_weight=nvfp4_backend.w2_weight.data,
        quant_scales=nvfp4_backend.quant_scales,
        weight_dtype=nvfp4_backend.w3_w1_weight.dtype,
        fc31_input_scale=nvfp4_backend.fc31_input_scale,
        scaling_vector_size=getattr(nvfp4_backend, "scaling_vector_size", 16),
    )
    return heter_backend


@pytest.mark.skipif(not nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
def test_heter_stream_overlap_not_slower_than_sequential():
    dtype = torch.bfloat16
    seq_len = 128
    top_k = 8
    num_experts = 128
    hidden_size = 2048
    intermediate_size = 768

    mapping = Mapping()
    mapping.rank = mpi_rank()
    all_rank_num_tokens = [seq_len] * mapping.world_size

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn(
            (seq_len, num_experts), dtype=dtype, device="cuda"
        )
        bf16_weights = create_unquantized_weights(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )
        heter_backend = _create_mixed_heter_backend(
            mapping=mapping,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            x=x,
            bf16_weights=bf16_weights,
        )

        with with_multi_stream(False):
            baseline_runner = _prepare_single_runner(
                heter_backend,
                x,
                router_logits,
                all_rank_num_tokens,
            )
            baseline_ms = _benchmark_timed(baseline_runner)

        with with_multi_stream(True):
            overlap_runner = _prepare_single_runner(
                heter_backend,
                x,
                router_logits,
                all_rank_num_tokens,
            )
            overlap_ms = _benchmark_timed(overlap_runner)

        speedup = baseline_ms / overlap_ms
        print(
            f"[stream overlap] sequential={baseline_ms:.3f}ms, "
            f"overlap={overlap_ms:.3f}ms, speedup={speedup:.3f}x"
        )

        assert overlap_ms <= baseline_ms * 1.05, (
            f"Overlap path is unexpectedly slower: "
            f"overlap={overlap_ms:.3f}ms, baseline={baseline_ms:.3f}ms"
        )

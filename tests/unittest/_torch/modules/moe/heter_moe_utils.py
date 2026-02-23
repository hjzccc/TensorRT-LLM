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
"""Shared helpers for heterogeneous MoE tests and benchmarks.

This module is imported by:
  - tests/unittest/_torch/modules/moe/test_heter_moe_correctness.py
  - tests/unittest/_torch/modules/moe/test_heter_moe_config_policy.py
  - tests/unittest/_torch/modules/moe/test_heter_moe_benchmark.py
  - tests/microbenchmarks/bench_heter_moe_ratio.py
"""

import copy
import os

import torch
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import HeterCutlassFusedMoE
from tensorrt_llm.models.modeling_utils import QuantAlgo

# Bypass ConfigurableMoE wrapper so create_moe falls through to
# create_moe_backend.  Set once at import time — every consumer of this
# module needs it.
os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"


# ---------------------------------------------------------------------------
# Hardware capability checks
# ---------------------------------------------------------------------------


def nvfp4_supported(dtype: torch.dtype = torch.bfloat16) -> bool:
    can_impl, _ = CutlassFusedMoE.can_implement(
        quant_algo=QuantAlgo.NVFP4,
        dtype_activation=dtype,
    )
    return can_impl


NVFP4_UNAVAILABLE_REASON = "NVFP4 is not implementable on current test hardware"


# ---------------------------------------------------------------------------
# Heter config factories
# ---------------------------------------------------------------------------


def single_group_config():
    return {
        "groups": [
            {
                "name": "all_bf16",
                "quant_algo": None,
                "size_ratio": 1.0,
                "checkpoint": None,
            },
        ],
    }


def two_group_config():
    return {
        "groups": [
            {
                "name": "hot_bf16",
                "quant_algo": None,
                "size_ratio": 0.5,
                "checkpoint": None,
            },
            {
                "name": "cold_nvfp4",
                "quant_algo": QuantAlgo.NVFP4,
                "size_ratio": 0.5,
                "checkpoint": None,
            },
        ],
    }


def two_bf16_group_config():
    """Two BF16-only groups — for testing multi-group dispatch without NVFP4."""
    return {
        "groups": [
            {
                "name": "group_a_bf16",
                "quant_algo": None,
                "size_ratio": 0.5,
                "checkpoint": None,
            },
            {
                "name": "group_b_bf16",
                "quant_algo": None,
                "size_ratio": 0.5,
                "checkpoint": None,
            },
        ],
    }


def all_bf16_heter_config():
    """Single-group heter config: all experts BF16 (fast-path)."""
    return {
        "groups": [
            {
                "name": "all_bf16",
                "quant_algo": None,
                "size_ratio": 1.0,
                "checkpoint": None,
            },
        ],
    }


def all_nvfp4_heter_config():
    """Single-group heter config: all experts NVFP4."""
    return {
        "groups": [
            {
                "name": "all_nvfp4",
                "quant_algo": QuantAlgo.NVFP4,
                "size_ratio": 1.0,
                "checkpoint": None,
            },
        ],
    }


def mixed_heter_config(bf16_ratio=0.5):
    """Two-group heter config: *bf16_ratio* BF16, remainder NVFP4."""
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
# Backend / weight helpers
# ---------------------------------------------------------------------------


def create_model_config(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    moe_backend: str,
    mapping,
    quant_config=None,
    heter_config=None,
) -> ModelConfig:
    """Create a ModelConfig for testing, optionally with heter_moe_config."""
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    model_config_kwargs = {
        "pretrained_config": pretrained_config,
        "mapping": mapping,
        "moe_backend": moe_backend,
    }
    if quant_config is not None:
        model_config_kwargs["quant_config"] = quant_config

    model_config = ModelConfig(**model_config_kwargs)

    if heter_config is not None:
        model_config.extra_attrs["heter_moe_config"] = heter_config

    return model_config


def create_backend(
    moe_cls,
    routing_method,
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    model_config,
):
    """Instantiate a MoE backend via create_moe_backend."""
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


def run_forward(backend, x, router_logits, all_rank_num_tokens):
    """Run forward pass on a MoE backend."""
    with torch.inference_mode():
        return backend.forward(
            x,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
        )


def create_unquantized_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    kaiming_fan_out: bool = True,
):
    quantize_util_cls, quant_config, _ = get_test_quant_params(
        quant_algo=None, x=None)
    quantize_util_kwargs = {
        "num_experts": num_experts,
        "dtype": dtype,
        "intermediate_size": intermediate_size,
        "hidden_size": hidden_size,
        "quant_config": quant_config,
    }
    quantize_util = quantize_util_cls(**quantize_util_kwargs)
    weights = quantize_util.create_weights()

    # Apply Kaiming fan-out scaling to keep activations in a sane range.
    # Without this, two back-to-back matmuls with a multiplicative gate
    # (SiLU(x@W1) * x@W3) @ W2 cause variance to explode as O(dim^2).
    #
    # Weight shapes (Linear convention: [out_features, in_features]):
    #   W1 [intermediate, hidden] → fan_out = intermediate
    #   W3 [intermediate, hidden] → fan_out = intermediate
    #   W2 [hidden, intermediate] → fan_out = hidden
    if kaiming_fan_out:
        for key, val in weights.items():
            if isinstance(val, torch.Tensor) and val.ndim == 2:
                fan_out = val.shape[0]
                weights[key] = val * (2.0 / fan_out) ** 0.5

    return weights


def create_cutlass_and_heter_backends(
    *,
    routing_method,
    mapping,
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    weights,
    heter_config,
):
    cutlass_config = create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="CUTLASS",
        mapping=mapping,
    )
    cutlass_backend = create_backend(
        moe_cls=CutlassFusedMoE,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        model_config=cutlass_config,
    )
    cutlass_backend.load_weights([copy.deepcopy(weights)])
    cutlass_backend.post_load_weights()
    cutlass_backend.cuda()

    heter_model_config = create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="HETER",
        mapping=mapping,
        heter_config=heter_config,
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
    heter_backend.load_weights([copy.deepcopy(weights)])
    heter_backend.post_load_weights()
    heter_backend.cuda()

    return cutlass_backend, heter_backend


def quantize_bf16_to_nvfp4(
    bf16_weights: dict,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    x: torch.Tensor,
    scaling_vector_size: int = 16,
):
    """Quantize existing BF16 MoE weights to NVFP4 format.

    This compresses *bf16_weights* rather than generating random quantized
    weights, ensuring BF16-vs-NVFP4 comparisons measure only quantization
    error, not unrelated weight differences.

    Args:
        bf16_weights: Weight dict from :func:`create_unquantized_weights`.
        x: Input tensor used to derive the activation scale.
    """
    x_sf_global = (448 * 6) / x.abs().max().float()
    weights = {}
    for expert_id in range(num_experts):
        w1 = bf16_weights[f"{expert_id}.w1.weight"]
        w2 = bf16_weights[f"{expert_id}.w2.weight"]
        w3 = bf16_weights[f"{expert_id}.w3.weight"]

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

        weights[f"{expert_id}.w1.weight"] = w1_q
        weights[f"{expert_id}.w2.weight"] = w2_q
        weights[f"{expert_id}.w3.weight"] = w3_q
        weights[f"{expert_id}.w1.weight_scale"] = w1_sb.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w2.weight_scale"] = w2_sb.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w3.weight_scale"] = w3_sb.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w1.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{expert_id}.w2.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{expert_id}.w3.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_sf
        weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf
        weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_sf
    return weights


def create_cutlass_backend(
    routing_method,
    mapping,
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    weights,
    quant_config=None,
):
    """Create a :class:`CutlassFusedMoE` backend, load *weights*, and move
    to CUDA."""
    model_config = create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="CUTLASS",
        mapping=mapping,
        quant_config=quant_config,
    )
    backend = create_backend(
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

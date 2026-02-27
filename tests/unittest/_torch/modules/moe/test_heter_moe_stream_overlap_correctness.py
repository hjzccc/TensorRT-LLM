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
def test_stream_overlap_matches_sequential():
    dtype = torch.bfloat16
    seq_len = 128
    top_k = 8
    num_experts = 128
    hidden_size = 512
    intermediate_size = 512

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

        with torch.inference_mode():
            with with_multi_stream(False):
                baseline_out = heter_backend.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                )
            with with_multi_stream(True):
                overlap_out = heter_backend.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                )

        torch.testing.assert_close(overlap_out, baseline_out, rtol=0, atol=0)


@pytest.mark.skipif(not nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
def test_stream_overlap_matches_cutlass_baseline():
    dtype = torch.bfloat16
    seq_len = 128
    top_k = 8
    num_experts = 128
    hidden_size = 512
    intermediate_size = 512

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

        cutlass_backend = create_cutlass_backend(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weights=bf16_weights,
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

        with torch.inference_mode():
            with with_multi_stream(True):
                overlap_out = heter_backend.forward(
                    x,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                )
            cutlass_out = cutlass_backend.forward(
                x,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )

        torch.testing.assert_close(overlap_out, cutlass_out, rtol=0.016, atol=0.016)

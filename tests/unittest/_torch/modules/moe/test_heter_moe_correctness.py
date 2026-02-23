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
"""Forward-equivalence tests for HeterCutlassFusedMoE vs CutlassFusedMoE."""

import pytest
import torch
from _torch.modules.moe.heter_moe_utils import (
    create_cutlass_and_heter_backends,
    create_unquantized_weights,
    run_forward,
    single_group_config,
    two_bf16_group_config,
)

from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping


# ---------------------------------------------------------------------------
# Tests: forward equivalence vs CutlassFusedMoE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda val: f"dtype={val}")
def test_heter_single_group_matches_cutlass(dtype):
    seq_len = 4
    top_k = 2
    num_experts = 8
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
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")
        weights = create_unquantized_weights(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

        cutlass_backend, heter_backend = create_cutlass_and_heter_backends(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weights=weights,
            heter_config=single_group_config(),
        )

        cutlass_out = run_forward(cutlass_backend, x, router_logits, all_rank_num_tokens)
        heter_out = run_forward(heter_backend, x, router_logits, all_rank_num_tokens)

        print(cutlass_out)
        print(heter_out)

        torch.testing.assert_close(heter_out, cutlass_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda val: f"dtype={val}")
def test_heter_two_groups_matches_cutlass(dtype):
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
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")
        weights = create_unquantized_weights(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

        cutlass_backend, heter_backend = create_cutlass_and_heter_backends(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weights=weights,
            heter_config=two_bf16_group_config(),
        )

        cutlass_out = run_forward(cutlass_backend, x, router_logits, all_rank_num_tokens)
        heter_out = run_forward(heter_backend, x, router_logits, all_rank_num_tokens)


        # Two-group dispatch splits experts across groups and combines partial
        # sums, changing BF16 accumulation order vs monolithic Cutlass.
        # BF16 has ~7-bit mantissa (1 ULP ≈ 0.008 near 1.0); with top_k=8
        # across 2 groups the max observed diff is ~0.016 (2 ULP).  Use
        # tolerances that accommodate this expected numerical noise.
        torch.testing.assert_close(heter_out, cutlass_out, rtol=0.016, atol=0.016)


# ---------------------------------------------------------------------------
# Test: run_moe level equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda val: f"dtype={val}")
def test_heter_run_moe_single_group(dtype):
    seq_len = 4
    top_k = 2
    num_experts = 8
    hidden_size = 512
    intermediate_size = 512

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")
        weights = create_unquantized_weights(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

        cutlass_backend, heter_backend = create_cutlass_and_heter_backends(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weights=weights,
            heter_config=single_group_config(),
        )

        token_selected_experts, token_final_scales = routing_method.apply(router_logits)

        with torch.inference_mode():
            cutlass_out = cutlass_backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=dtype,
            )
            heter_out = heter_backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=dtype,
            )

        torch.testing.assert_close(heter_out, cutlass_out, rtol=1e-5, atol=1e-5)

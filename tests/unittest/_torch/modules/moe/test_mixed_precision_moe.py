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
"""Correctness tests for MixedPrecisionMoERunner forward equivalence.

Tests the MixedPrecisionMoERunner (wrapping the C++ MixPrecisionMoEFCWrapper)
against pure bf16 and NVFP4 CutlassFusedMoE references. Verifies:
  1. All-bf16 case: runner delegates to bf16 CutlassMoeFCRunner,
     output ≈ pure bf16 CutlassFusedMoE reference.
  2. Half-bf16/half-fp4: full mixed-precision pipeline (WIP in C++ kernel).
  3. All-fp4 case: runner delegates to fp4 CutlassMoeFCRunner,
     output ≈ pure NVFP4 CutlassFusedMoE reference.

Test parameters use a scaled-down config for memory-limited GPUs:
  E=16, top_k=4, hidden_size=512, inter_size=512
"""

import pytest
import torch
from _torch.modules.moe.mixed_precision_moe_utils import (
    MIXED_PREC_UNAVAILABLE_REASON,
    SMALL_HIDDEN_SIZE,
    SMALL_INTER_SIZE,
    SMALL_NUM_EXPERTS,
    SMALL_TOP_K,
    create_bf16_reference_backend,
    create_mixed_precision_runner,
    create_nvfp4_reference_backend,
    create_unquantized_weights,
    mixed_precision_supported,
    quantize_bf16_to_nvfp4,
    run_mixed_precision_forward,
    run_reference_forward,
)

from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping


# ---------------------------------------------------------------------------
# Pytest skip decorator for hardware requirements
# ---------------------------------------------------------------------------
requires_mixed_prec = pytest.mark.skipif(
    not mixed_precision_supported(),
    reason=MIXED_PREC_UNAVAILABLE_REASON,
)


# ===========================================================================
# Forward Equivalence Tests
# ===========================================================================


@requires_mixed_prec
class TestMixedPrecisionForwardEquivalence:
    """Test MixedPrecisionMoERunner output matches CutlassFusedMoE reference.

    The C++ MixPrecisionMoEFCWrapper currently supports two fast-paths:
      - all-bf16:  delegates entirely to bf16 CutlassMoeFCRunner
      - all-fp4:   delegates entirely to fp4 CutlassMoeFCRunner
    The full mixed-precision pipeline (mixed sort/expand/gemm) is WIP.

    These tests verify the wrapper's delegation path produces identical output
    to the standalone CutlassFusedMoE backends.
    """

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda val: f"dtype={val}")
    def test_all_bf16_matches_reference(self, dtype):
        """When all experts are bf16, runner delegates to bf16 runner → matches pure bf16.

        In C++: all_bf16 = (num_high_precision_experts_ == num_experts_per_node)
        → bf16_runner_.runMoe() is called with the standard bf16 pipeline.
        Output should match a standalone bf16 CutlassFusedMoE within bf16 precision.
        """
        seq_len = 8
        top_k = SMALL_TOP_K
        num_experts = SMALL_NUM_EXPERTS
        hidden_size = SMALL_HIDDEN_SIZE
        intermediate_size = SMALL_INTER_SIZE
        num_high_precision = num_experts  # ALL bf16

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

            # Create bf16 weights
            bf16_weights = create_unquantized_weights(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
            )

            # Quantize to fp4 (needed to configure the runner, even though unused)
            fp4_weights = quantize_bf16_to_nvfp4(
                bf16_weights, num_experts, hidden_size, intermediate_size, dtype, x
            )

            # Create bf16 reference backend (CutlassFusedMoE, pure bf16)
            ref_backend = create_bf16_reference_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                weights=bf16_weights,
            )

            # Create MixedPrecisionMoERunner with all experts as bf16
            runner, bf16_fc1, bf16_fc2 = create_mixed_precision_runner(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
                num_high_precision_experts=num_high_precision,
                bf16_weights=bf16_weights,
                fp4_weights=fp4_weights,
            )

            # Run reference (CutlassFusedMoE routes internally from logits)
            ref_out = run_reference_forward(
                ref_backend, x, router_logits, all_rank_num_tokens
            )
            torch.cuda.synchronize()

            # Run MixedPrecisionMoERunner (routing applied in Python, then run_moe)
            runner_out = run_mixed_precision_forward(
                runner, bf16_fc1, bf16_fc2, x, router_logits, routing_method
            )
            torch.cuda.synchronize()

            # When all experts are bf16, outputs should be very close.
            # BF16 accumulation order may differ, so allow ~2 ULP tolerance.
            torch.testing.assert_close(runner_out, ref_out, rtol=0.016, atol=0.016)

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda val: f"dtype={val}")
    @pytest.mark.skip(
        reason="Full mixed-precision pipeline (mixed sort/expand/gemm) "
               "not yet implemented in C++ MixPrecisionMoEFCWrapper"
    )
    def test_half_bf16_half_fp4_bounded_error(self, dtype):
        """When half experts are fp4, output difference is bounded by quant error.

        NOTE: This test is skipped because the C++ kernel currently throws
        'Full mixed-precision MoE pipeline not yet implemented' for the
        mixed case (0 < num_high_precision < num_experts).
        """
        seq_len = 32
        top_k = SMALL_TOP_K
        num_experts = SMALL_NUM_EXPERTS
        hidden_size = SMALL_HIDDEN_SIZE
        intermediate_size = SMALL_INTER_SIZE
        num_high_precision = num_experts // 2

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

            fp4_weights = quantize_bf16_to_nvfp4(
                bf16_weights, num_experts, hidden_size, intermediate_size, dtype, x
            )

            ref_backend = create_bf16_reference_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                weights=bf16_weights,
            )

            runner, bf16_fc1, bf16_fc2 = create_mixed_precision_runner(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
                num_high_precision_experts=num_high_precision,
                bf16_weights=bf16_weights,
                fp4_weights=fp4_weights,
            )

            ref_out = run_reference_forward(
                ref_backend, x, router_logits, all_rank_num_tokens
            )
            runner_out = run_mixed_precision_forward(
                runner, bf16_fc1, bf16_fc2, x, router_logits, routing_method
            )

            # FP4 quantization introduces significant per-element error.
            # We use relaxed tolerances and check that relative error is bounded.
            abs_diff = (runner_out.float() - ref_out.float()).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            ref_norm = ref_out.float().norm().item()

            # Relative error should be bounded
            if ref_norm > 1e-6:
                rel_error = abs_diff.norm().item() / ref_norm
                assert rel_error < 0.5, (
                    f"Relative error {rel_error:.4f} too large "
                    f"(max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f})"
                )

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda val: f"dtype={val}")
    def test_all_fp4_matches_nvfp4_reference(self, dtype):
        """When all experts are fp4, runner delegates to fp4 runner → matches pure NVFP4.

        In C++: all_fp4 = (num_high_precision_experts_ == 0)
        → fp4_runner_.runMoe() is called with fp4 weights and quant params.
        Output should match a standalone NVFP4 CutlassFusedMoE.
        """
        seq_len = 8
        top_k = 2
        num_experts = 8
        hidden_size = SMALL_HIDDEN_SIZE
        intermediate_size = SMALL_INTER_SIZE
        num_high_precision = 0  # ALL fp4

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

            # Create bf16 weights (needed for runner construction, though unused)
            bf16_weights = create_unquantized_weights(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
            )

            # Quantize to fp4
            fp4_weights = quantize_bf16_to_nvfp4(
                bf16_weights, num_experts, hidden_size, intermediate_size, dtype, x
            )

            # Create pure NVFP4 CutlassFusedMoE reference backend
            ref_backend = create_nvfp4_reference_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                fp4_weights=fp4_weights,
            )

            # Create MixedPrecisionMoERunner with all experts as fp4
            runner, bf16_fc1, bf16_fc2 = create_mixed_precision_runner(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
                num_high_precision_experts=num_high_precision,
                bf16_weights=bf16_weights,
                fp4_weights=fp4_weights,
            )

            # Run reference (NVFP4 CutlassFusedMoE)
            ref_out = run_reference_forward(
                ref_backend, x, router_logits, all_rank_num_tokens
            )
            torch.cuda.synchronize()

            # Run MixedPrecisionMoERunner (all fp4 path)
            runner_out = run_mixed_precision_forward(
                runner, bf16_fc1, bf16_fc2, x, router_logits, routing_method
            )
            torch.cuda.synchronize()

            # Basic sanity checks
            assert runner_out.shape == (seq_len, hidden_size)
            assert runner_out.dtype == torch.bfloat16
            assert not torch.isnan(runner_out).any(), "Runner output contains NaN"
            assert not torch.isinf(runner_out).any(), "Runner output contains Inf"

            # Both use the same fp4 weights, so outputs should be close.
            # Allow some tolerance for accumulation order differences.
            torch.testing.assert_close(runner_out, ref_out, rtol=0.016, atol=0.016)

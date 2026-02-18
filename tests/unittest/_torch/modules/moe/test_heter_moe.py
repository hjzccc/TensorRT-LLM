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
"""Unit tests for HeterCutlassFusedMoE (heterogeneous precision MoE backend)."""

import copy
import os

import pytest
import torch
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import HeterCutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.policy.dispatch_plan import DispatchPlan
from tensorrt_llm._torch.modules.fused_moe.policy.strategies import (
    BaseDispatchPolicy,
    RandomDispatchPolicy,
)
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

# Bypass ConfigurableMoE wrapper so create_moe falls through to create_moe_backend.
os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"


def _nvfp4_supported(dtype: torch.dtype = torch.bfloat16) -> bool:
    can_impl, _ = CutlassFusedMoE.can_implement(
        quant_algo=QuantAlgo.NVFP4,
        dtype_activation=dtype,
    )
    return can_impl


NVFP4_UNAVAILABLE_REASON = "NVFP4 is not implementable on current test hardware"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_group_config():
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


def _two_group_config():
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


def _create_model_config(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    moe_backend: str,
    mapping: Mapping,
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


def _create_backend(
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


def _run_forward(backend, x, router_logits, all_rank_num_tokens):
    """Run forward pass on a MoE backend."""
    with torch.inference_mode():
        return backend.forward(
            x,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
        )


def _create_unquantized_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
):
    quantize_util_cls, quant_config, _ = get_test_quant_params(quant_algo=None, x=None)
    quantize_util_kwargs = {
        "num_experts": num_experts,
        "dtype": dtype,
        "intermediate_size": intermediate_size,
        "hidden_size": hidden_size,
        "quant_config": quant_config,
    }
    quantize_util = quantize_util_cls(**quantize_util_kwargs)
    return quantize_util.create_weights()


def _create_cutlass_and_heter_backends(
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
    cutlass_config = _create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="CUTLASS",
        mapping=mapping,
    )
    cutlass_backend = _create_backend(
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

    heter_model_config = _create_model_config(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        moe_backend="HETER",
        mapping=mapping,
        heter_config=heter_config,
    )
    heter_backend = _create_backend(
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
        weights = _create_unquantized_weights(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

        cutlass_backend, heter_backend = _create_cutlass_and_heter_backends(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weights=weights,
            heter_config=_single_group_config(),
        )

        cutlass_out = _run_forward(cutlass_backend, x, router_logits, all_rank_num_tokens)
        heter_out = _run_forward(heter_backend, x, router_logits, all_rank_num_tokens)

        torch.testing.assert_close(heter_out, cutlass_out, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda val: f"dtype={val}")
def test_heter_two_groups_matches_cutlass(dtype):
    seq_len = 8
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
        weights = _create_unquantized_weights(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

        cutlass_backend, heter_backend = _create_cutlass_and_heter_backends(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weights=weights,
            heter_config=_two_group_config(),
        )

        cutlass_out = _run_forward(cutlass_backend, x, router_logits, all_rank_num_tokens)
        heter_out = _run_forward(heter_backend, x, router_logits, all_rank_num_tokens)

        torch.testing.assert_close(heter_out, cutlass_out, rtol=1e-5, atol=1e-5)


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
        weights = _create_unquantized_weights(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
        )

        cutlass_backend, heter_backend = _create_cutlass_and_heter_backends(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            weights=weights,
            heter_config=_single_group_config(),
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


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestHeterConfigValidation:
    """Tests for heter_config validation in HeterCutlassFusedMoE.__init__."""

    NUM_EXPERTS = 8
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 512
    DTYPE = torch.bfloat16

    def _try_create(self, heter_config):
        mapping = Mapping()
        mapping.rank = mpi_rank()
        routing_method = RenormalizeMoeRoutingMethod(top_k=2)

        model_config = _create_model_config(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            moe_backend="HETER",
            mapping=mapping,
            heter_config=heter_config,
        )
        with torch.device(f"cuda:{mapping.rank}"):
            return _create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=model_config,
            )

    def test_missing_heter_config(self):
        with pytest.raises(ValueError, match="heter_moe_config"):
            self._try_create(heter_config=None)

    def test_empty_groups(self):
        with pytest.raises(ValueError, match="non-empty list"):
            self._try_create({"groups": []})

    def test_missing_size_ratio(self):
        with pytest.raises(ValueError, match="size_ratio is required"):
            self._try_create({
                "groups": [
                    {
                        "name": "bad_group",
                        "quant_algo": None,
                        "checkpoint": None,
                    },
                ],
            })

    @pytest.mark.parametrize("bad_ratio", [0.0, -0.1, 1.1])
    def test_invalid_size_ratio(self, bad_ratio):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
            self._try_create({
                "groups": [
                    {
                        "name": "bad_ratio",
                        "quant_algo": None,
                        "size_ratio": bad_ratio,
                        "checkpoint": None,
                    },
                ],
            })

    def test_size_ratio_sum_mismatch(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            self._try_create({
                "groups": [
                    {
                        "name": "g0",
                        "quant_algo": None,
                        "size_ratio": 0.4,
                        "checkpoint": None,
                    },
                    {
                        "name": "g1",
                        "quant_algo": None,
                        "size_ratio": 0.4,
                        "checkpoint": None,
                    },
                ],
            })

    def test_invalid_quant_algo(self):
        with pytest.raises(ValueError, match="Unsupported quant_algo value"):
            self._try_create({
                "groups": [
                    {
                        "name": "bad_quant",
                        "quant_algo": "NOT_A_REAL_ALGO",
                        "size_ratio": 1.0,
                        "checkpoint": None,
                    },
                ],
            })

    def test_checkpoint_not_exists(self, tmp_path):
        missing_ckpt = tmp_path / "not_found"
        with pytest.raises(ValueError, match="does not exist"):
            self._try_create({
                "groups": [
                    {
                        "name": "all_bf16",
                        "quant_algo": None,
                        "size_ratio": 1.0,
                        "checkpoint": str(missing_ckpt),
                    },
                ],
            })

    def test_valid_single_group(self):
        backend = self._try_create(_single_group_config())
        assert isinstance(backend, HeterCutlassFusedMoE)

    def test_valid_two_groups(self):
        if not _nvfp4_supported(self.DTYPE):
            pytest.skip(NVFP4_UNAVAILABLE_REASON)
        backend = self._try_create(_two_group_config())
        assert isinstance(backend, HeterCutlassFusedMoE)


# ---------------------------------------------------------------------------
# Tests: dispatch policy
# ---------------------------------------------------------------------------


class TestDispatchPolicy:
    NUM_EXPERTS = 8
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 512
    DTYPE = torch.bfloat16
    SEQ_LEN = 8
    TOP_K = 2

    def _create_backend_with_weights(self, heter_config):
        mapping = Mapping()
        mapping.rank = mpi_rank()
        routing_method = RenormalizeMoeRoutingMethod(top_k=self.TOP_K)
        weights = _create_unquantized_weights(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
        )

        model_config = _create_model_config(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            moe_backend="HETER",
            mapping=mapping,
            heter_config=heter_config,
        )

        with torch.device(f"cuda:{mapping.rank}"):
            backend = _create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=model_config,
            )
            backend.load_weights([copy.deepcopy(weights)])
            backend.post_load_weights()
            backend.cuda()
            x = torch.randn((self.SEQ_LEN, self.HIDDEN_SIZE), dtype=self.DTYPE, device="cuda")
            router_logits = torch.randn(
                (self.SEQ_LEN, self.NUM_EXPERTS),
                dtype=self.DTYPE,
                device="cuda",
            )
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
        return backend, x, token_selected_experts, token_final_scales

    def test_random_policy_deterministic(self):
        backend, x, token_selected_experts, token_final_scales = (
            self._create_backend_with_weights(_single_group_config())
        )

        assert backend._current_expert_ids
        assert backend._current_expert_ids[0]

        with torch.inference_mode():
            backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=self.DTYPE,
            )
            first_key = backend._current_assignment_key
            first_remap_id = id(backend._remap_tables)

            backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=self.DTYPE,
            )

        assert backend._current_assignment_key == first_key
        assert id(backend._remap_tables) == first_remap_id

    @pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
    def test_set_dispatch_policy(self):
        backend, x, token_selected_experts, token_final_scales = (
            self._create_backend_with_weights(_two_group_config())
        )

        before_key = backend._current_assignment_key
        before_assignment = copy.deepcopy(backend._current_expert_ids)

        backend.set_dispatch_policy(RandomDispatchPolicy(seed=99))
        assert backend._current_assignment_key is None

        with torch.inference_mode():
            backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=self.DTYPE,
            )

        assert backend._current_assignment_key is not None
        assert backend._current_assignment_key != before_key
        assert backend._current_expert_ids != before_assignment

    def test_dispatch_plan_validate_coverage(self):
        DispatchPlan(group_assignments=[[0, 1], [2, 3]]).validate(num_experts=4)

        with pytest.raises(ValueError, match="coverage mismatch"):
            DispatchPlan(group_assignments=[[0, 1], [2]]).validate(num_experts=4)

        with pytest.raises(ValueError, match="duplicate expert IDs"):
            DispatchPlan(group_assignments=[[0, 1], [1, 2, 3]]).validate(
                num_experts=4
            )

    @pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
    def test_custom_policy(self):

        class AllExpertsInGroup0Policy(BaseDispatchPolicy):

            def assign(
                self,
                num_experts,
                group_size_ratios,
                token_selected_experts=None,
                token_final_scales=None,
            ):
                return DispatchPlan(
                    group_assignments=[list(range(num_experts))]
                    + [[] for _ in range(len(group_size_ratios) - 1)]
                )

        backend, x, token_selected_experts, token_final_scales = (
            self._create_backend_with_weights(_two_group_config())
        )
        backend.set_dispatch_policy(AllExpertsInGroup0Policy())

        with torch.inference_mode():
            backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=self.DTYPE,
            )

        assert backend._current_expert_ids[0] == list(range(self.NUM_EXPERTS))
        for group_ids in backend._current_expert_ids[1:]:
            assert group_ids == []

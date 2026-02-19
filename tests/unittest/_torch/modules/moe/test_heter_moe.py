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
from _torch.modules.moe.quantize_utils import (
    NVFP4QuantizeUtil,
    get_test_quant_params,
)
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import HeterCutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.policy.dispatch_plan import DispatchPlan
from tensorrt_llm._torch.modules.fused_moe.policy.strategies import (
    BaseDispatchPolicy,
    ConfidenceThresholdPolicy,
    ExpertLoadPolicy,
    RandomDispatchPolicy,
)
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# Bypass ConfigurableMoE wrapper so create_moe falls through to create_moe_backend.
os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"

# ---------------------------------------------------------------------------
# Global flags for torch.compile and CUDA graph testing.
# These are OFF by default.  Set to True to exercise those code paths.
# NOTE: Current correctness tests run in eager mode.  The runtime benchmark
# tests below use these flags to optionally wrap forward with torch.compile
# and/or measure via CUDA graph replay.
# ---------------------------------------------------------------------------
ENABLE_TORCH_COMPILE = False
ENABLE_CUDA_GRAPHS = False

# Benchmark parameters
_WARMUP_ITERS = 10
_BENCH_ITERS = 50


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


def _create_nvfp4_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    x: torch.Tensor,
):
    """Create NVFP4-quantized weights for *num_experts* experts.

    Args:
        x: Input tensor used to derive the NVFP4 activation scale
           (``x_sf_global``).
    """
    x_sf_global = (448 * 6) / x.abs().max().float()
    nvfp4_util = NVFP4QuantizeUtil(
        num_experts=num_experts,
        dtype=dtype,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
    )
    return nvfp4_util.create_weights(x_sf_global=x_sf_global)


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
    """Create a :class:`CutlassFusedMoE` backend, load *weights*, and move
    to CUDA."""
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
# Benchmark utilities
# ---------------------------------------------------------------------------


def _make_benchmark_fn(backend, x, router_logits, all_rank_num_tokens):
    """Create a callable that runs one forward pass on *backend*.

    When ``ENABLE_TORCH_COMPILE`` is ``True`` the forward is wrapped with
    ``torch.compile(mode="reduce-overhead")``.
    """
    forward_fn = backend.forward

    if ENABLE_TORCH_COMPILE:
        forward_fn = torch.compile(forward_fn, mode="reduce-overhead")

    def fn():
        with torch.inference_mode():
            return forward_fn(
                x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
            )

    return fn


def _benchmark_forward(fn, warmup_iters=_WARMUP_ITERS, bench_iters=_BENCH_ITERS):
    """Benchmark *fn* using CUDA events.  Returns the **median** time in ms.

    * Warmup for *warmup_iters* iterations first.
    * If ``ENABLE_CUDA_GRAPHS`` is ``True`` (and ``ENABLE_TORCH_COMPILE``
      is ``False``), a CUDA graph is captured after warmup and replayed
      for timing.  When ``ENABLE_TORCH_COMPILE`` is ``True`` the compiler
      manages graphs internally, so manual capture is skipped.
    """
    torch.cuda.synchronize()

    # --- Warmup ---
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # --- Optionally capture a CUDA graph ---
    use_manual_graph = ENABLE_CUDA_GRAPHS and not ENABLE_TORCH_COMPILE
    if use_manual_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        runner = graph.replay
    else:
        runner = fn

    # --- Timed iterations ---
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
    return times[len(times) // 2]  # median


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
        return backend, x, router_logits, token_selected_experts, token_final_scales

    def test_random_policy_deterministic(self):
        backend, x, router_logits, token_selected_experts, token_final_scales = (
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
        backend, x, router_logits, token_selected_experts, token_final_scales = (
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
                router_logits=None,
            ):
                return DispatchPlan(
                    group_assignments=[list(range(num_experts))]
                    + [[] for _ in range(len(group_size_ratios) - 1)]
                )

        backend, x, router_logits, token_selected_experts, token_final_scales = (
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

    @pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
    def test_confidence_threshold_policy(self):
        """ConfidenceThresholdPolicy assigns experts by mean routing weight."""
        backend, x, router_logits, token_selected_experts, token_final_scales = (
            self._create_backend_with_weights(_two_group_config())
        )
        backend.set_dispatch_policy(
            ConfidenceThresholdPolicy(confidence_threshold=0.5)
        )

        with torch.inference_mode():
            backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=self.DTYPE,
            )

        # Assignment should cover all experts.
        all_assigned = []
        for group_ids in backend._current_expert_ids:
            all_assigned.extend(group_ids)
        assert sorted(all_assigned) == list(range(self.NUM_EXPERTS))

    @pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
    def test_expert_load_policy(self):
        """ExpertLoadPolicy assigns experts by activation frequency."""
        backend, x, router_logits, token_selected_experts, token_final_scales = (
            self._create_backend_with_weights(_two_group_config())
        )
        backend.set_dispatch_policy(ExpertLoadPolicy())

        with torch.inference_mode():
            backend.run_moe(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                output_dtype=self.DTYPE,
            )

        all_assigned = []
        for group_ids in backend._current_expert_ids:
            all_assigned.extend(group_ids)
        assert sorted(all_assigned) == list(range(self.NUM_EXPERTS))

    def test_confidence_policy_fallback_without_signals(self):
        """ConfidenceThresholdPolicy falls back to random when no signals."""
        policy = ConfidenceThresholdPolicy(fallback_seed=42)
        plan = policy.assign(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            token_selected_experts=None,
            token_final_scales=None,
        )
        plan.validate(num_experts=8)
        assert len(plan.group_assignments) == 2
        total = sum(len(g) for g in plan.group_assignments)
        assert total == 8

    def test_expert_load_policy_fallback_without_signals(self):
        """ExpertLoadPolicy falls back to random when no signals."""
        policy = ExpertLoadPolicy(fallback_seed=42)
        plan = policy.assign(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            token_selected_experts=None,
            token_final_scales=None,
        )
        plan.validate(num_experts=8)
        assert len(plan.group_assignments) == 2

    def test_confidence_policy_ranks_by_weight(self):
        """ConfidenceThresholdPolicy puts high-weight experts in last group."""
        policy = ConfidenceThresholdPolicy()
        # Simulate 4 experts.  Experts 2 and 3 receive much higher weights.
        token_selected_experts = torch.tensor(
            [[0, 2], [1, 3], [2, 3], [0, 2]], dtype=torch.int32
        )
        token_final_scales = torch.tensor(
            [[0.1, 0.9], [0.1, 0.9], [0.8, 0.9], [0.1, 0.8]],
            dtype=torch.float32,
        )
        plan = policy.assign(
            num_experts=4,
            group_size_ratios=[0.5, 0.5],
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
        )
        plan.validate(num_experts=4)
        # Last group (high-precision) should get the highest-weight experts.
        # Experts 2 and 3 have much higher mean weights.
        assert 2 in plan.group_assignments[-1]
        assert 3 in plan.group_assignments[-1]

    def test_expert_load_policy_ranks_by_frequency(self):
        """ExpertLoadPolicy puts frequently-activated experts in last group."""
        policy = ExpertLoadPolicy()
        # Expert 0 selected 4 times, expert 1 selected 3 times,
        # expert 2 once, expert 3 once.
        token_selected_experts = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 2], [3, 2]],
            dtype=torch.int32,
        )
        token_final_scales = torch.ones_like(
            token_selected_experts, dtype=torch.float32,
        )
        plan = policy.assign(
            num_experts=4,
            group_size_ratios=[0.5, 0.5],
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
        )
        plan.validate(num_experts=4)
        # Last group (high-precision) should contain most-active experts.
        assert 0 in plan.group_assignments[-1]
        assert 1 in plan.group_assignments[-1]


# ---------------------------------------------------------------------------
# Tests: Phase 2 runtime benchmarks (mixed-precision dispatch)
# ---------------------------------------------------------------------------


def _all_bf16_heter_config():
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


def _all_nvfp4_heter_config():
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


def _mixed_heter_config(bf16_ratio=0.5):
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


class TestPhase2RuntimeBenchmark:
    """Phase 2 runtime benchmarks: verify NVFP4 speedup and mixed ordering.

    These tests measure wall-clock runtime using CUDA events.  They compare
    three configurations:

    1. **All BF16**  — CutlassFusedMoE with unquantized weights (baseline).
    2. **All NVFP4** — CutlassFusedMoE with NVFP4 weights (should be fastest).
    3. **Mixed**     — HeterCutlassFusedMoE with BF16 + NVFP4 groups (in
       between).  Requires phase 2 per-group precision dispatch.

    The global flags ``ENABLE_TORCH_COMPILE`` and ``ENABLE_CUDA_GRAPHS``
    control whether the benchmark exercises those code paths.

    Use larger model dimensions so that weight memory bandwidth dominates
    kernel launch overhead and the speedup is measurable.
    """

    NUM_EXPERTS = 8
    HIDDEN_SIZE = 4096
    INTERMEDIATE_SIZE = 4096
    DTYPE = torch.bfloat16
    SEQ_LEN = 64
    TOP_K = 2

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _setup_common(self):
        """Return mapping, routing_method, inputs, and BF16 weights."""
        mapping = Mapping()
        mapping.rank = mpi_rank()

        routing_method = RenormalizeMoeRoutingMethod(top_k=self.TOP_K)

        x = torch.randn(
            (self.SEQ_LEN, self.HIDDEN_SIZE),
            dtype=self.DTYPE, device="cuda",
        )
        router_logits = torch.randn(
            (self.SEQ_LEN, self.NUM_EXPERTS),
            dtype=self.DTYPE, device="cuda",
        )
        all_rank_num_tokens = [self.SEQ_LEN] * mapping.world_size

        bf16_weights = _create_unquantized_weights(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
        )

        return mapping, routing_method, x, router_logits, all_rank_num_tokens, bf16_weights

    # ------------------------------------------------------------------
    # test: all-NVFP4 should be faster than all-BF16
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
    def test_all_nvfp4_faster_than_all_bf16(self):
        """All-NVFP4 CutlassFusedMoE is faster than all-BF16.

        Also logs the output deviation (max / mean absolute difference)
        between the two precision modes for reference.
        """
        with torch.device("cuda"):
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            mapping, routing_method, x, router_logits, all_rank_num_tokens, bf16_weights = (
                self._setup_common()
            )

            # --- BF16 baseline ---
            bf16_backend = _create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=bf16_weights,
            )

            # --- NVFP4 ---
            nvfp4_weights = _create_nvfp4_weights(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                x=x,
            )
            nvfp4_backend = _create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=nvfp4_weights,
                quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            )

            # --- Log output deviation for reference ---
            with torch.inference_mode():
                bf16_out = bf16_backend.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )
                nvfp4_out = nvfp4_backend.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )

            diff = (bf16_out.float() - nvfp4_out.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(
                f"\n[Phase2 output deviation] BF16 vs NVFP4: "
                f"max={max_diff:.6f}, mean={mean_diff:.6f}"
            )

            # --- Benchmark ---
            bf16_fn = _make_benchmark_fn(
                bf16_backend, x, router_logits, all_rank_num_tokens,
            )
            nvfp4_fn = _make_benchmark_fn(
                nvfp4_backend, x, router_logits, all_rank_num_tokens,
            )

            bf16_ms = _benchmark_forward(bf16_fn)
            nvfp4_ms = _benchmark_forward(nvfp4_fn)
            print(
                f"[Phase2 runtime] BF16={bf16_ms:.3f}ms, "
                f"NVFP4={nvfp4_ms:.3f}ms, "
                f"speedup={bf16_ms / nvfp4_ms:.2f}x"
            )

            assert nvfp4_ms < bf16_ms, (
                f"NVFP4 ({nvfp4_ms:.3f}ms) should be faster than "
                f"BF16 ({bf16_ms:.3f}ms)"
            )

    # ------------------------------------------------------------------
    # test: mixed heter runtime is between all-BF16 and all-NVFP4
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
    @pytest.mark.skip(
        reason="Requires phase 2 per-group precision dispatch in "
               "HeterCutlassFusedMoE._build_group_caches / run_moe"
    )
    def test_heter_mixed_runtime_between_extremes(self):
        """Mixed heter (BF16 + NVFP4) runtime is between the two extremes.

        **Preconditions** (phase 2):

        * ``HeterCutlassFusedMoE._build_group_caches()`` selects from the
          correct weight set (BF16 or NVFP4) based on each group's
          ``quant_algo``.
        * ``run_moe()`` quantises input per-group and passes per-group
          quant flags to ``fused_moe()``.
        * The heter backend must be loaded with *dual* weight sets (one
          per precision) via a phase-2 weight loading API.

        **Test outline**:

        1. Create all-BF16 CutlassFusedMoE → ``bf16_ms``
        2. Create all-NVFP4 CutlassFusedMoE → ``nvfp4_ms``
        3. Create HeterCutlassFusedMoE with 50/50 BF16+NVFP4 → ``mixed_ms``
        4. Assert ``nvfp4_ms <= mixed_ms <= bf16_ms``
        """
        with torch.device("cuda"):
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            mapping, routing_method, x, router_logits, all_rank_num_tokens, bf16_weights = (
                self._setup_common()
            )

            # --- BF16 baseline ---
            bf16_backend = _create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=bf16_weights,
            )

            # --- NVFP4 baseline ---
            nvfp4_weights = _create_nvfp4_weights(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                x=x,
            )
            nvfp4_backend = _create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=nvfp4_weights,
                quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            )

            # --- Mixed heter backend ---
            # NOTE: Phase 2 weight loading TBD — this section must be
            # updated once the dual-weight-set API is available.
            heter_model_config = _create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=_mixed_heter_config(bf16_ratio=0.5),
            )
            heter_backend = _create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=heter_model_config,
            )
            # Phase 2 TODO: load dual weight sets (BF16 + NVFP4) into
            # heter_backend.  The exact API will be determined by the
            # phase 2 implementation.  For now we load BF16 weights only
            # (phase 1 behavior — both groups share the same weights).
            heter_backend.load_weights([copy.deepcopy(bf16_weights)])
            heter_backend.post_load_weights()
            heter_backend.cuda()

            # --- Benchmark all three ---
            bf16_fn = _make_benchmark_fn(
                bf16_backend, x, router_logits, all_rank_num_tokens,
            )
            nvfp4_fn = _make_benchmark_fn(
                nvfp4_backend, x, router_logits, all_rank_num_tokens,
            )
            mixed_fn = _make_benchmark_fn(
                heter_backend, x, router_logits, all_rank_num_tokens,
            )

            bf16_ms = _benchmark_forward(bf16_fn)
            nvfp4_ms = _benchmark_forward(nvfp4_fn)
            mixed_ms = _benchmark_forward(mixed_fn)

            print(
                f"\n[Phase2 runtime] BF16={bf16_ms:.3f}ms, "
                f"NVFP4={nvfp4_ms:.3f}ms, "
                f"Mixed={mixed_ms:.3f}ms"
            )

            # Output deviation for reference
            with torch.inference_mode():
                bf16_out = bf16_backend.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )
                mixed_out = heter_backend.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )
            diff = (bf16_out.float() - mixed_out.float()).abs()
            print(
                f"[Phase2 output deviation] BF16 vs Mixed: "
                f"max={diff.max().item():.6f}, "
                f"mean={diff.mean().item():.6f}"
            )

            assert nvfp4_ms <= mixed_ms, (
                f"Mixed ({mixed_ms:.3f}ms) should not be faster than "
                f"all-NVFP4 ({nvfp4_ms:.3f}ms)"
            )
            assert mixed_ms <= bf16_ms, (
                f"Mixed ({mixed_ms:.3f}ms) should not be slower than "
                f"all-BF16 ({bf16_ms:.3f}ms)"
            )

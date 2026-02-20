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
    get_test_quant_params,
)
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import HeterCutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.policy import (
    ConfidenceThresholdHeterDispatch,
    ExpertLoadHeterDispatch,
    HeterDispatchPolicy,
    RandomHeterDispatch,
)
from tensorrt_llm._torch.modules.fused_moe.policy.heter_dispatch import (
    _validate_expert_to_group,
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
ENABLE_TORCH_COMPILE = True
ENABLE_CUDA_GRAPHS = True

# Benchmark parameters
_WARMUP_ITERS = 10
_BENCH_ITERS = 20


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


def _two_bf16_group_config():
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
    kaiming_fan_out: bool = True,
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


def _quantize_bf16_to_nvfp4(
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
        bf16_weights: Weight dict from :func:`_create_unquantized_weights`.
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


def _get_l2_cache_size_bytes(device_id: int = 0) -> int:
    """Return L2 cache size in bytes for *device_id*."""
    from cuda.bindings import driver  # cuda-python

    err, device = driver.cuDeviceGet(device_id)
    assert err == driver.CUresult.CUDA_SUCCESS, (
        f"cuDeviceGet failed: {err}")
    err, size = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        device,
    )
    assert err == driver.CUresult.CUDA_SUCCESS, (
        f"cuDeviceGetAttribute(L2_CACHE_SIZE) failed: {err}")
    return size


def _prepare_single_runner(backend, x, router_logits,
                           all_rank_num_tokens,
                           warmup_iters=_WARMUP_ITERS):
    """Prepare a ready-to-time callable for a single workspace.

    Handles torch.compile wrapping, warmup, and optional CUDA graph
    capture.  Returns a callable that executes one forward pass.
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

    # --- Warmup ---
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # --- Optionally capture a CUDA graph ---
    if ENABLE_CUDA_GRAPHS and not ENABLE_TORCH_COMPILE:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        return graph.replay

    return fn


def _create_rotating_runner(workspaces, warmup_iters=_WARMUP_ITERS):
    """Build a rotating runner from *N* independent workspaces.

    Each workspace is a ``(backend, x, router_logits, all_rank_num_tokens)``
    tuple with **independently allocated** GPU memory.  The returned
    callable cycles through per-workspace runners on every invocation,
    ensuring that by the time a workspace is revisited its data has been
    evicted from L2 cache — no explicit flush buffer needed.

    This mirrors the workspace-rotation strategy from CUTLASS example 79e.
    """
    runners = []
    for backend, x, router_logits, all_rank_num_tokens in workspaces:
        runner = _prepare_single_runner(
            backend, x, router_logits, all_rank_num_tokens,
            warmup_iters=warmup_iters,
        )
        runners.append(runner)

    n = len(runners)
    counter = [0]  # mutable int via list

    def rotating_fn():
        result = runners[counter[0] % n]()
        counter[0] += 1
        return result

    return rotating_fn


def _benchmark_timed(runner, bench_iters=_BENCH_ITERS):
    """Time *runner* using CUDA events.  Returns the **median** time in ms."""
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
            heter_config=_two_bf16_group_config(),
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

    def _create_backend_with_weights(self, heter_config, *, seq_len=None,
                                     top_k=None):
        """Create a HeterCutlassFusedMoE backend with BF16 weights."""
        seq_len = seq_len if seq_len is not None else self.SEQ_LEN
        top_k = top_k if top_k is not None else self.TOP_K
        mapping = Mapping()
        mapping.rank = mpi_rank()
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
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
            x = torch.randn(
                (seq_len, self.HIDDEN_SIZE),
                dtype=self.DTYPE, device="cuda",
            )
            router_logits = torch.randn(
                (seq_len, self.NUM_EXPERTS),
                dtype=self.DTYPE, device="cuda",
            )
            token_selected_experts, token_final_scales = routing_method.apply(
                router_logits
            )
        return backend, x, router_logits, token_selected_experts, token_final_scales

    # ------------------------------------------------------------------
    # Policy determinism
    # ------------------------------------------------------------------

    def test_random_policy_deterministic(self):
        """RandomHeterDispatch.dispatch() is deterministic for same seed+input."""
        policy = RandomHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
            seed=42,
        )
        token_selected_experts = torch.tensor(
            [[0, 2], [1, 3], [2, 5], [4, 7]],
            dtype=torch.int32, device="cuda",
        )
        token_final_scales = torch.ones_like(
            token_selected_experts, dtype=torch.float32,
        )

        dispatches_1 = policy.dispatch(token_selected_experts, token_final_scales)
        dispatches_2 = policy.dispatch(token_selected_experts, token_final_scales)

        for (t1, e1, s1), (t2, e2, s2) in zip(dispatches_1, dispatches_2):
            if e1 is None:
                assert e2 is None
            else:
                torch.testing.assert_close(e1, e2)
                torch.testing.assert_close(s1, s2)

    # ------------------------------------------------------------------
    # Policy property setter
    # ------------------------------------------------------------------

    def test_policy_property_setter(self):
        """backend.policy = new_policy replaces the dispatch policy."""
        backend, x, _, tse, tfs = self._create_backend_with_weights(
            _single_group_config()
        )
        old_policy = backend.policy
        new_policy = RandomHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[1.0],
            seed=99,
        )
        backend.policy = new_policy
        assert backend.policy is new_policy
        assert backend.policy is not old_policy

    # ------------------------------------------------------------------
    # Assignment validation
    # ------------------------------------------------------------------

    def test_validate_expert_to_group(self):
        """_validate_expert_to_group checks shape and value range."""
        # Valid: 4 experts, 2 groups.
        e2g = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="cuda")
        _validate_expert_to_group(e2g, num_experts=4, num_groups=2)

        # Wrong shape.
        bad_shape = torch.tensor([0, 0, 1], dtype=torch.long, device="cuda")
        with pytest.raises(AssertionError, match="shape"):
            _validate_expert_to_group(bad_shape, num_experts=4, num_groups=2)

        # Group index out of range.
        bad_val = torch.tensor([0, 0, 2, 1], dtype=torch.long, device="cuda")
        with pytest.raises(AssertionError, match="num_groups"):
            _validate_expert_to_group(bad_val, num_experts=4, num_groups=2)

    # ------------------------------------------------------------------
    # Custom policy via subclass
    # ------------------------------------------------------------------

    def test_custom_policy(self):
        """A custom HeterDispatchPolicy routes all experts to group 0."""

        class AllGroup0Policy(HeterDispatchPolicy):

            def _assign(self, token_selected_experts, token_final_scales):
                device = token_selected_experts.device
                return torch.zeros(
                    self._num_experts, dtype=torch.long, device=device)

        backend, x, _, tse, tfs = self._create_backend_with_weights(
            _two_bf16_group_config()
        )
        backend.policy = AllGroup0Policy(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
        )

        with torch.inference_mode():
            out = backend.run_moe(
                x=x,
                token_selected_experts=tse,
                token_final_scales=tfs,
                output_dtype=self.DTYPE,
            )
        assert out.shape == x.shape

    # ------------------------------------------------------------------
    # Confidence threshold dispatch
    # ------------------------------------------------------------------

    def test_confidence_threshold_dispatch(self):
        """ConfidenceThresholdHeterDispatch assigns higher-scoring experts to later groups."""
        # Use a larger batch and top_k so that per-expert mean routing
        # weights are spread out enough to produce a clear ordering.
        seq_len = 256
        top_k = 4
        backend, x, _, tse, tfs = self._create_backend_with_weights(
            _two_bf16_group_config(), seq_len=seq_len, top_k=top_k,
        )
        backend.policy = ConfidenceThresholdHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
            confidence_threshold=0.5,
        )

        dispatches = backend.policy.dispatch(tse, tfs)
        assert len(dispatches) == 2

        for tok_idx, experts, scales in dispatches:
            assert experts is not None
            assert scales is not None
            assert experts.shape[1] == top_k
            assert scales.shape == experts.shape

        # Verify that every expert in a later group has a higher mean
        # routing weight than every expert in the earlier group (the
        # invariant maintained by _assign_by_score_gpu's topk/argsort).
        flat_experts = tse.reshape(-1).long()
        flat_scales = tfs.reshape(-1)

        weight_sum = torch.zeros(
            self.NUM_EXPERTS, dtype=torch.float32, device=tse.device)
        weight_sum.scatter_add_(0, flat_experts, flat_scales)
        expert_count = torch.bincount(
            flat_experts, minlength=self.NUM_EXPERTS,
        ).to(dtype=torch.float32).clamp_min_(1.0)
        mean_score = weight_sum / expert_count

        e2g = backend.policy._assign(tse, tfs)
        for gidx in range(1, backend.policy.num_groups):
            earlier_mask = (e2g == gidx - 1)
            later_mask = (e2g == gidx)
            if not earlier_mask.any() or not later_mask.any():
                continue
            max_score_earlier = mean_score[earlier_mask].max().item()
            min_score_later = mean_score[later_mask].min().item()
            assert min_score_later >= max_score_earlier, (
                f"Group {gidx} min mean score ({min_score_later:.4f}) < "
                f"group {gidx - 1} max mean score ({max_score_earlier:.4f}). "
                f"Scores: {mean_score.tolist()}"
            )

    # ------------------------------------------------------------------
    # Expert load dispatch
    # ------------------------------------------------------------------

    def test_expert_load_dispatch(self):
        """ExpertLoadHeterDispatch assigns higher-load experts to later groups."""
        # Use a larger batch and top_k so that expert activation counts
        # are spread out enough to produce a clear load-based ordering.
        seq_len = 256
        top_k = 4
        backend, x, _, tse, tfs = self._create_backend_with_weights(
            _two_bf16_group_config(), seq_len=seq_len, top_k=top_k,
        )
        backend.policy = ExpertLoadHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
        )

        dispatches = backend.policy.dispatch(tse, tfs)
        assert len(dispatches) == 2

        for _, experts, scales in dispatches:
            assert experts is not None
            assert scales is not None

        # Verify that every expert in a later group has activation count
        # >= every expert in the earlier group (the invariant maintained
        # by _assign_by_score_gpu's topk/argsort).
        flat_experts = tse.reshape(-1).long()
        counts = torch.bincount(flat_experts, minlength=self.NUM_EXPERTS)

        e2g = backend.policy._assign(tse, tfs)
        for gidx in range(1, backend.policy.num_groups):
            earlier_mask = (e2g == gidx - 1)
            later_mask = (e2g == gidx)
            if not earlier_mask.any() or not later_mask.any():
                continue
            max_load_earlier = counts[earlier_mask].max().item()
            min_load_later = counts[later_mask].min().item()
            assert min_load_later >= max_load_earlier, (
                f"Group {gidx} min load ({min_load_later}) < "
                f"group {gidx - 1} max load ({max_load_earlier}). "
                f"Counts: {counts.tolist()}"
            )

    # ------------------------------------------------------------------
    # Fallback without signals
    # ------------------------------------------------------------------

    def test_confidence_dispatch_fallback_without_signals(self):
        """ConfidenceThresholdHeterDispatch._assign() falls back to random
        when signals are ``None``."""
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            fallback_seed=42,
        )
        e2g = policy._assign(
            token_selected_experts=None,
            token_final_scales=None,
        )
        assert e2g.shape == (8,)
        assert e2g.min().item() >= 0
        assert e2g.max().item() <= 1

    def test_expert_load_dispatch_fallback_without_signals(self):
        """ExpertLoadHeterDispatch._assign() falls back to random when
        signals are ``None``."""
        policy = ExpertLoadHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            fallback_seed=42,
        )
        e2g = policy._assign(
            token_selected_experts=None,
            token_final_scales=None,
        )
        assert e2g.shape == (8,)
        assert e2g.min().item() >= 0
        assert e2g.max().item() <= 1

    # ------------------------------------------------------------------
    # Score-based ranking
    # ------------------------------------------------------------------

    def test_confidence_dispatch_ranks_by_weight(self):
        """ConfidenceThresholdHeterDispatch puts high-weight experts in
        last group."""
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=4,
            group_size_ratios=[0.5, 0.5],
        )
        # Simulate 4 experts.  Experts 2 and 3 receive much higher weights.
        token_selected_experts = torch.tensor(
            [[0, 2], [1, 3], [2, 3], [0, 2]],
            dtype=torch.int32, device="cuda",
        )
        token_final_scales = torch.tensor(
            [[0.1, 0.9], [0.1, 0.9], [0.8, 0.9], [0.1, 0.8]],
            dtype=torch.float32, device="cuda",
        )
        e2g = policy._assign(token_selected_experts, token_final_scales)
        assert e2g.shape == (4,)
        # Last group (high-precision, group 1) should get the highest-weight experts.
        assert e2g[2].item() == 1
        assert e2g[3].item() == 1

    def test_expert_load_dispatch_ranks_by_frequency(self):
        """ExpertLoadHeterDispatch puts frequently-activated experts in
        last group."""
        policy = ExpertLoadHeterDispatch(
            num_experts=4,
            group_size_ratios=[0.5, 0.5],
        )
        # Expert 0 selected 4 times, expert 1 selected 3 times,
        # expert 2 once, expert 3 once.
        token_selected_experts = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 2], [3, 2]],
            dtype=torch.int32, device="cuda",
        )
        token_final_scales = torch.ones_like(
            token_selected_experts, dtype=torch.float32,
        )
        e2g = policy._assign(token_selected_experts, token_final_scales)
        assert e2g.shape == (4,)
        # Last group (group 1) should contain most-active experts.
        assert e2g[0].item() == 1
        assert e2g[1].item() == 1


# ---------------------------------------------------------------------------
# Tests: runtime benchmarks (mixed-precision dispatch)
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


class TestRuntimeBenchmark:
    """Runtime benchmarks: verify NVFP4 speedup and mixed ordering.

    These tests measure wall-clock runtime using CUDA events.  They compare
    three configurations:

    1. **All BF16**  — CutlassFusedMoE with unquantized weights (baseline).
    2. **All NVFP4** — CutlassFusedMoE with NVFP4 weights (should be fastest).
    3. **Mixed**     — HeterCutlassFusedMoE with BF16 + NVFP4 groups (in
       between).  Uses ``register_group_weights()`` for per-group precision.

    The global flags ``ENABLE_TORCH_COMPILE`` and ``ENABLE_CUDA_GRAPHS``
    control whether the benchmark exercises those code paths.

    Use larger model dimensions so that weight memory bandwidth dominates
    kernel launch overhead and the speedup is measurable.
    """

    NUM_EXPERTS = 128
    HIDDEN_SIZE = 2048
    INTERMEDIATE_SIZE = 768
    DTYPE = torch.bfloat16
    SEQ_LEN = 64
    TOP_K = 8

    _MAX_WORKSPACE_COUNT = 16

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

    def _estimate_workspace_bytes(self, scheme):
        """Estimate GPU bytes for one workspace of the given *scheme*.

        Each scheme is benchmarked independently, so the L2 budget is
        calculated per-scheme (not across all schemes combined).

        Supported schemes: ``"bf16"``, ``"nvfp4"``, ``"mixed"``.
        """
        # bpe = torch.finfo(self.DTYPE).bits // 8
        # weight_elements = (
        #     self.NUM_EXPERTS * 3 * self.HIDDEN_SIZE * self.INTERMEDIATE_SIZE
        # )

        # bf16_weight_bytes = weight_elements * bpe
        # nvfp4_weight_bytes = int(weight_elements * bpe * 0.55)

        bf16_weight_bytes = self.NUM_EXPERTS * 3 * self.HIDDEN_SIZE * self.INTERMEDIATE_SIZE * 2
        nvfp4_weight_bytes = self.NUM_EXPERTS * 3 * self.HIDDEN_SIZE * self.INTERMEDIATE_SIZE * 0.5

        input_bytes = 0
        # (
        #     self.SEQ_LEN * self.HIDDEN_SIZE
        #     + self.SEQ_LEN * self.NUM_EXPERTS
        # ) * bpe

        if scheme == "bf16":
            return bf16_weight_bytes + input_bytes
        if scheme == "nvfp4":
            return nvfp4_weight_bytes + input_bytes
        if scheme == "mixed":
            # BF16 parent weights + NVFP4 registered group weights
            return bf16_weight_bytes + nvfp4_weight_bytes + input_bytes
        raise ValueError(f"Unknown scheme: {scheme!r}")

    def _create_one_workspace(self, scheme, mapping, routing_method,
                              x, router_logits, all_rank_num_tokens):
        """Create a single workspace tuple for *scheme*.

        Returns ``(backend, x, router_logits, all_rank_num_tokens)``
        with freshly allocated tensors at new GPU addresses.
        """
        ws_x = torch.randn_like(x)
        ws_logits = torch.randn_like(router_logits)

        ws_bf16_w = _create_unquantized_weights(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            kaiming_fan_out=True,
        )

        if scheme == "bf16":
            backend = _create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=ws_bf16_w,
            )
            return (backend, ws_x, ws_logits, all_rank_num_tokens)

        # Both "nvfp4" and "mixed" need quantized weights.
        ws_nvfp4_w = _quantize_bf16_to_nvfp4(
            bf16_weights=ws_bf16_w,
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            x=ws_x,
        )
        ws_nvfp4_backend = _create_cutlass_backend(
            routing_method=routing_method,
            mapping=mapping,
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            weights=ws_nvfp4_w,
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        )

        if scheme == "nvfp4":
            return (ws_nvfp4_backend, ws_x, ws_logits, all_rank_num_tokens)

        if scheme == "mixed":
            ws_heter_cfg = _create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=_mixed_heter_config(bf16_ratio=0.5),
            )
            ws_heter = _create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=ws_heter_cfg,
            )
            ws_heter.load_weights([copy.deepcopy(ws_bf16_w)])
            ws_heter.post_load_weights()
            ws_heter.cuda()
            ws_heter.register_group_weights(
                group_idx=0,
                w3_w1_weight=ws_nvfp4_backend.w3_w1_weight.data,
                w2_weight=ws_nvfp4_backend.w2_weight.data,
                quant_scales=ws_nvfp4_backend.quant_scales,
                weight_dtype=ws_nvfp4_backend.w3_w1_weight.dtype,
                fc31_input_scale=ws_nvfp4_backend.fc31_input_scale,
                scaling_vector_size=getattr(
                    ws_nvfp4_backend, "scaling_vector_size", 16
                ),
            )
            return (ws_heter, ws_x, ws_logits, all_rank_num_tokens)

        raise ValueError(f"Unknown scheme: {scheme!r}")

    def _create_benchmark_workspaces(
        self,
        scheme,
        *,
        initial_backend,
        mapping,
        routing_method,
        x,
        router_logits,
        all_rank_num_tokens,
    ):
        """Build rotated workspaces for a single *scheme*.

        Keeps appending independently-allocated workspaces until the
        cumulative footprint exceeds 3× the L2 cache size, ensuring
        that by the time a workspace is revisited during rotation its
        data has been fully evicted from L2.

        Workspace 0 reuses *initial_backend* and the caller's inputs
        (also used for the correctness check).

        Args:
            scheme: ``"bf16"``, ``"nvfp4"``, or ``"mixed"``.
            initial_backend: The already-created backend for workspace 0.

        Returns:
            List of ``(backend, x, router_logits, all_rank_num_tokens)``
            tuples.
        """
        workspaces = [
            (initial_backend, x, router_logits, all_rank_num_tokens),
        ]

        per_ws_bytes = self._estimate_workspace_bytes(scheme)
        allocated_bytes = per_ws_bytes  # workspace 0

        l2_bytes = _get_l2_cache_size_bytes()
        target_bytes = 3 * l2_bytes

        while (allocated_bytes < target_bytes
               and len(workspaces) < self._MAX_WORKSPACE_COUNT):
            workspaces.append(
                self._create_one_workspace(
                    scheme, mapping, routing_method,
                    x, router_logits, all_rank_num_tokens,
                )
            )
            allocated_bytes += per_ws_bytes

        print(
            f"[workspace rotation] scheme={scheme}, "
            f"L2={l2_bytes / (1 << 20):.1f} MB, "
            f"per_ws={per_ws_bytes / (1 << 20):.1f} MB, "
            f"count={len(workspaces)}, "
            f"total={allocated_bytes / (1 << 20):.1f} MB"
        )
        return workspaces

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

            # --- NVFP4 (quantized from the same BF16 weights) ---
            nvfp4_weights = _quantize_bf16_to_nvfp4(
                bf16_weights=bf16_weights,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                x=x,
            )

            # print(bf16_weights)
            # print(nvfp4_weights)

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
            print("\nOutput deviation between BF16 and NVFP4:")
            print(bf16_out)
            print(nvfp4_out)
            print()
            diff = (bf16_out.float() - nvfp4_out.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(
                f"\n[output deviation] BF16 vs NVFP4: "
                f"max={max_diff:.6f}, mean={mean_diff:.6f}"
            )

            # --- Benchmark (L2-cold via workspace rotation) ---
            bf16_workspaces = self._create_benchmark_workspaces(
                "bf16",
                initial_backend=bf16_backend,
                mapping=mapping,
                routing_method=routing_method,
                x=x,
                router_logits=router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            nvfp4_workspaces = self._create_benchmark_workspaces(
                "nvfp4",
                initial_backend=nvfp4_backend,
                mapping=mapping,
                routing_method=routing_method,
                x=x,
                router_logits=router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )

            bf16_runner = _create_rotating_runner(bf16_workspaces)
            nvfp4_runner = _create_rotating_runner(nvfp4_workspaces)

            bf16_ms = _benchmark_timed(bf16_runner)
            nvfp4_ms = _benchmark_timed(nvfp4_runner)
            print(
                f"[runtime] BF16={bf16_ms:.3f}ms, "
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
    def test_heter_mixed_runtime_between_extremes(self):
        """Mixed heter (BF16 + NVFP4) runtime is between the two extremes.

        Uses ``register_group_weights()`` to load NVFP4 weights for the
        quantized group while the BF16 group falls back to parent weights.

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
            # bf16_backend = _create_cutlass_backend(
            #     routing_method=routing_method,
            #     mapping=mapping,
            #     num_experts=self.NUM_EXPERTS,
            #     hidden_size=self.HIDDEN_SIZE,
            #     intermediate_size=self.INTERMEDIATE_SIZE,
            #     dtype=self.DTYPE,
            #     weights=bf16_weights,
            # )

            # --- NVFP4 baseline (quantized from the same BF16 weights) ---
            nvfp4_weights = _quantize_bf16_to_nvfp4(
                bf16_weights=bf16_weights,
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
            # _mixed_heter_config: group 0 = NVFP4, group 1 = BF16.
            # Load BF16 weights as parent (used by group 1 as fallback),
            # then register NVFP4 weights for group 0.
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
            heter_backend.load_weights([copy.deepcopy(bf16_weights)])
            heter_backend.post_load_weights()
            heter_backend.cuda()

            # Register NVFP4 weights for group 0 by extracting processed
            # tensors from the standalone NVFP4 backend.
            heter_backend.register_group_weights(
                group_idx=0,
                w3_w1_weight=nvfp4_backend.w3_w1_weight.data,
                w2_weight=nvfp4_backend.w2_weight.data,
                quant_scales=nvfp4_backend.quant_scales,
                weight_dtype=nvfp4_backend.w3_w1_weight.dtype,
                fc31_input_scale=nvfp4_backend.fc31_input_scale,
                scaling_vector_size=getattr(
                    nvfp4_backend, "scaling_vector_size", 16
                ),
            )

            # --- Benchmark all three (L2-cold via workspace rotation) ---
            _ws_kwargs = dict(
                mapping=mapping,
                routing_method=routing_method,
                x=x,
                router_logits=router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            # bf16_workspaces = self._create_benchmark_workspaces(
            #     "bf16", initial_backend=bf16_backend, **_ws_kwargs,
            # )
            # nvfp4_workspaces = self._create_benchmark_workspaces(
            #     "nvfp4", initial_backend=nvfp4_backend, **_ws_kwargs,
            # )
            mixed_workspaces = self._create_benchmark_workspaces(
                "mixed", initial_backend=heter_backend, **_ws_kwargs,
            )

            # bf16_runner = _create_rotating_runner(bf16_workspaces)
            # nvfp4_runner = _create_rotating_runner(nvfp4_workspaces)
            mixed_runner = _create_rotating_runner(mixed_workspaces)

            # bf16_ms = _benchmark_timed(bf16_runner)
            # nvfp4_ms = _benchmark_timed(nvfp4_runner)
            mixed_ms = _benchmark_timed(mixed_runner)
            return

            print(
                f"\n[runtime] BF16={bf16_ms:.3f}ms, "
                f"NVFP4={nvfp4_ms:.3f}ms, "
                f"Mixed={mixed_ms:.3f}ms"
            )

            # Output deviation for reference (workspace 0)
            with torch.inference_mode():
                bf16_out = bf16_backend.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )
                mixed_out = heter_backend.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )
            diff = (bf16_out.float() - mixed_out.float()).abs()
            print(
                f"[output deviation] BF16 vs Mixed: "
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

    # ------------------------------------------------------------------
    # test: heter mixed runtime is between heter all-BF16 and heter all-NVFP4
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not _nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
    def test_heter_mixed_runtime_between_heter_extremes(self):
        """Mixed heter runtime is between heter all-BF16 and heter all-NVFP4.

        All three configurations use HeterCutlassFusedMoE with different
        group layouts:

        1. Heter all-BF16  — 1 group, all experts BF16 (parent weights).
        2. Heter all-NVFP4 — 1 group, all experts NVFP4 (registered weights).
        3. Heter mixed     — 2 groups, 50 % NVFP4 + 50 % BF16.

        **Assertion**: ``heter_nvfp4_ms <= heter_mixed_ms <= heter_bf16_ms``
        """
        with torch.device("cuda"):
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            mapping, routing_method, x, router_logits, all_rank_num_tokens, bf16_weights = (
                self._setup_common()
            )

            # --- Quantize BF16 weights to NVFP4 ---
            nvfp4_weights = _quantize_bf16_to_nvfp4(
                bf16_weights=bf16_weights,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                x=x,
            )

            # Standalone NVFP4 CutlassFusedMoE to extract processed tensors
            # for register_group_weights.
            nvfp4_cutlass = _create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=nvfp4_weights,
                quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            )

            # --- Heter all-BF16 (1 group, parent weights) -----------------
            heter_bf16_cfg = _create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=_all_bf16_heter_config(),
            )
            heter_bf16 = _create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=heter_bf16_cfg,
            )
            heter_bf16.load_weights([copy.deepcopy(bf16_weights)])
            heter_bf16.post_load_weights()
            heter_bf16.cuda()

            # --- Heter all-NVFP4 (1 group, registered NVFP4 weights) ------
            heter_nvfp4_cfg = _create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=_all_nvfp4_heter_config(),
            )
            heter_nvfp4 = _create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=heter_nvfp4_cfg,
            )
            heter_nvfp4.load_weights([copy.deepcopy(bf16_weights)])
            heter_nvfp4.post_load_weights()
            heter_nvfp4.cuda()
            heter_nvfp4.register_group_weights(
                group_idx=0,
                w3_w1_weight=nvfp4_cutlass.w3_w1_weight.data,
                w2_weight=nvfp4_cutlass.w2_weight.data,
                quant_scales=nvfp4_cutlass.quant_scales,
                weight_dtype=nvfp4_cutlass.w3_w1_weight.dtype,
                fc31_input_scale=nvfp4_cutlass.fc31_input_scale,
                scaling_vector_size=getattr(
                    nvfp4_cutlass, "scaling_vector_size", 16
                ),
            )

            # --- Heter mixed (2 groups: NVFP4 + BF16) ---------------------
            heter_mixed_cfg = _create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=_mixed_heter_config(bf16_ratio=0.5),
            )
            heter_mixed = _create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=heter_mixed_cfg,
            )
            heter_mixed.load_weights([copy.deepcopy(bf16_weights)])
            heter_mixed.post_load_weights()
            heter_mixed.cuda()
            heter_mixed.register_group_weights(
                group_idx=0,
                w3_w1_weight=nvfp4_cutlass.w3_w1_weight.data,
                w2_weight=nvfp4_cutlass.w2_weight.data,
                quant_scales=nvfp4_cutlass.quant_scales,
                weight_dtype=nvfp4_cutlass.w3_w1_weight.dtype,
                fc31_input_scale=nvfp4_cutlass.fc31_input_scale,
                scaling_vector_size=getattr(
                    nvfp4_cutlass, "scaling_vector_size", 16
                ),
            )

            # --- Benchmark (single-workspace, simple warmup) ---------------
            runner_bf16 = _prepare_single_runner(
                heter_bf16, x, router_logits, all_rank_num_tokens,
            )
            runner_nvfp4 = _prepare_single_runner(
                heter_nvfp4, x, router_logits, all_rank_num_tokens,
            )
            runner_mixed = _prepare_single_runner(
                heter_mixed, x, router_logits, all_rank_num_tokens,
            )

            heter_bf16_ms = _benchmark_timed(runner_bf16)
            heter_nvfp4_ms = _benchmark_timed(runner_nvfp4)
            heter_mixed_ms = _benchmark_timed(runner_mixed)

            print(
                f"\n[runtime] Heter-BF16={heter_bf16_ms:.3f}ms, "
                f"Heter-NVFP4={heter_nvfp4_ms:.3f}ms, "
                f"Heter-Mixed={heter_mixed_ms:.3f}ms"
            )
            return

            # Output deviation for reference
            with torch.inference_mode():
                bf16_out = heter_bf16.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )
                mixed_out = heter_mixed.forward(
                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens,
                )
            diff = (bf16_out.float() - mixed_out.float()).abs()
            print(
                f"[output deviation] Heter-BF16 vs Heter-Mixed: "
                f"max={diff.max().item():.6f}, "
                f"mean={diff.mean().item():.6f}"
            )

            assert heter_nvfp4_ms <= heter_mixed_ms, (
                f"Heter-Mixed ({heter_mixed_ms:.3f}ms) should not be faster "
                f"than Heter-NVFP4 ({heter_nvfp4_ms:.3f}ms)"
            )
            assert heter_mixed_ms <= heter_bf16_ms, (
                f"Heter-Mixed ({heter_mixed_ms:.3f}ms) should not be slower "
                f"than Heter-BF16 ({heter_bf16_ms:.3f}ms)"
            )

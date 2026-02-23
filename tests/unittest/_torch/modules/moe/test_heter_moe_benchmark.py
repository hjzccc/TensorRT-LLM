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
"""Runtime benchmark tests for HeterCutlassFusedMoE (mixed-precision dispatch).

These tests measure wall-clock runtime using CUDA events.  They compare
BF16, NVFP4, and mixed configurations to verify expected performance
ordering.

The global flags ``ENABLE_TORCH_COMPILE`` and ``ENABLE_CUDA_GRAPHS``
control whether the benchmark exercises torch.compile / CUDA graph paths.
"""

import copy

import pytest
import torch
from _torch.modules.moe.heter_moe_utils import (
    NVFP4_UNAVAILABLE_REASON,
    all_bf16_heter_config,
    all_nvfp4_heter_config,
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
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# ---------------------------------------------------------------------------
# Global flags for torch.compile and CUDA graph testing.
# These are OFF by default.  Set to True to exercise those code paths.
# NOTE: Current correctness tests run in eager mode.  The runtime benchmark
# tests below use these flags to optionally wrap forward with torch.compile
# and/or measure via CUDA graph replay.
# ---------------------------------------------------------------------------
ENABLE_TORCH_COMPILE = True
ENABLE_CUDA_GRAPHS = True
# ENABLE_TORCH_COMPILE = False
# ENABLE_CUDA_GRAPHS = False

# Benchmark parameters
_WARMUP_ITERS = 10
_BENCH_ITERS = 1


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
    if ENABLE_CUDA_GRAPHS:
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
# Tests: runtime benchmarks (mixed-precision dispatch)
# ---------------------------------------------------------------------------


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
    SEQ_LEN = 128
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

        bf16_weights = create_unquantized_weights(
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
        bf16_weight_bytes = self.NUM_EXPERTS * 3 * self.HIDDEN_SIZE * self.INTERMEDIATE_SIZE * 2
        nvfp4_weight_bytes = self.NUM_EXPERTS * 3 * self.HIDDEN_SIZE * self.INTERMEDIATE_SIZE * 0.5

        input_bytes = 0

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

        ws_bf16_w = create_unquantized_weights(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            kaiming_fan_out=True,
        )

        if scheme == "bf16":
            backend = create_cutlass_backend(
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
        ws_nvfp4_w = quantize_bf16_to_nvfp4(
            bf16_weights=ws_bf16_w,
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            x=ws_x,
        )
        ws_nvfp4_backend = create_cutlass_backend(
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
            ws_heter_cfg = create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=mixed_heter_config(bf16_ratio=0.5),
            )
            ws_heter = create_backend(
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

    @pytest.mark.skipif(not nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
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
            bf16_backend = create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=bf16_weights,
            )

            # --- NVFP4 (quantized from the same BF16 weights) ---
            nvfp4_weights = quantize_bf16_to_nvfp4(
                bf16_weights=bf16_weights,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                x=x,
            )

            nvfp4_backend = create_cutlass_backend(
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

    @pytest.mark.skipif(not nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
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
            bf16_backend = create_cutlass_backend(
                routing_method=routing_method,
                mapping=mapping,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                weights=bf16_weights,
            )
            # --- NVFP4 baseline (quantized from the same BF16 weights) ---
            nvfp4_weights = quantize_bf16_to_nvfp4(
                bf16_weights=bf16_weights,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                x=x,
            )
            nvfp4_backend = create_cutlass_backend(
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
            # mixed_heter_config: group 0 = NVFP4, group 1 = BF16.
            # Load BF16 weights as parent (used by group 1 as fallback),
            # then register NVFP4 weights for group 0.
            heter_model_config = create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=mixed_heter_config(bf16_ratio=0.5),
            )
            heter_backend = create_backend(
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
            bf16_workspaces = self._create_benchmark_workspaces(
                "bf16", initial_backend=bf16_backend, **_ws_kwargs,
            )
            nvfp4_workspaces = self._create_benchmark_workspaces(
                "nvfp4", initial_backend=nvfp4_backend, **_ws_kwargs,
            )
            mixed_workspaces = self._create_benchmark_workspaces(
                "mixed", initial_backend=heter_backend, **_ws_kwargs,
            )

            bf16_runner = _create_rotating_runner(bf16_workspaces)
            nvfp4_runner = _create_rotating_runner(nvfp4_workspaces)
            mixed_runner = _create_rotating_runner(mixed_workspaces)
            bf16_ms = _benchmark_timed(bf16_runner)
            nvfp4_ms = _benchmark_timed(nvfp4_runner)
            mixed_ms = _benchmark_timed(mixed_runner)


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

    @pytest.mark.skipif(not nvfp4_supported(), reason=NVFP4_UNAVAILABLE_REASON)
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
            nvfp4_weights = quantize_bf16_to_nvfp4(
                bf16_weights=bf16_weights,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                x=x,
            )

            # Standalone NVFP4 CutlassFusedMoE to extract processed tensors
            # for register_group_weights.
            nvfp4_cutlass = create_cutlass_backend(
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
            heter_bf16_cfg = create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=all_bf16_heter_config(),
            )
            heter_bf16 = create_backend(
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
            heter_nvfp4_cfg = create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=all_nvfp4_heter_config(),
            )
            heter_nvfp4 = create_backend(
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
            heter_mixed_cfg = create_model_config(
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                moe_backend="HETER",
                mapping=mapping,
                heter_config=mixed_heter_config(bf16_ratio=0.2),
            )
            heter_mixed = create_backend(
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

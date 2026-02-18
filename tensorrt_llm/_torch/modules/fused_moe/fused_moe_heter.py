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
"""Heterogeneous Precision MoE backend.

Stores **N full sets of weights** (one per precision group) for all
experts and dispatches each precision group through a separate
``torch.ops.trtllm.fused_moe()`` call with the appropriate weights and
quantization flags, then sums the outputs.

Expert-to-group assignment is determined by a pluggable **dispatch
policy** (see :mod:`~.policy`).  The policy is invoked every
``run_moe()`` call with router signals, and the assignment is cached —
weight subset caches are only rebuilt when the assignment changes.

Usage::

    from tensorrt_llm.llmapi import LLM
    from tensorrt_llm.llmapi.llm_args import MoeConfig

    llm = LLM(
        model=bf16_model_dir,
        moe_config=MoeConfig(
            backend="HETER",
            heter_config={
                "groups": [
                    {
                        "name": "cold",
                        "quant_algo": "NVFP4",
                        "size_ratio": 0.80,
                        "checkpoint": "/path/to/nvfp4/model",
                    },
                    {
                        "name": "hot",
                        "quant_algo": null,
                        "size_ratio": 0.20,
                        "checkpoint": "/path/to/bf16/model",
                    },
                ],
            },
        ),
    )
"""

import math
import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...model_config import ModelConfig
from ...utils import Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .policy.dispatch_plan import DispatchPlan
from .policy.strategies import BaseDispatchPolicy, RandomDispatchPolicy
from .routing import BaseMoeRoutingMethod

# Quantization algorithms supported by the HETER backend.
# None means unquantized (BF16/FP16).
_SUPPORTED_QUANT_ALGOS = frozenset({None, QuantAlgo.NVFP4})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_quant_algo(raw: Any) -> Optional[QuantAlgo]:
    """Convert a user-provided quant_algo value to ``QuantAlgo`` or ``None``.

    Accepts ``None``, ``"BF16"``, ``"FP16"``, ``"NONE"`` (all → ``None``),
    ``"NVFP4"`` (→ ``QuantAlgo.NVFP4``), or an existing ``QuantAlgo`` member.

    Raises:
        ValueError: If *raw* cannot be resolved.
    """
    if raw is None:
        return None
    if isinstance(raw, QuantAlgo):
        return raw
    if isinstance(raw, str):
        upper = raw.upper()
        if upper in ("BF16", "FP16", "NONE"):
            return None
        try:
            return QuantAlgo(upper)
        except (ValueError, KeyError):
            pass
    raise ValueError(
        f"Unsupported quant_algo value: {raw!r}.  "
        f"Supported: None / 'BF16' / 'NVFP4'"
    )


def _subset_quant_scales(
    quant_scales: Optional[NamedTuple],
    expert_ids: List[int],
) -> Optional[NamedTuple]:
    """Subset a quant-scales NamedTuple along the expert (dim-0) axis.

    Every field is expected to have shape ``[num_experts, ...]``.
    Returns a new NamedTuple of the same type containing only the rows
    for *expert_ids*, or ``None`` if *quant_scales* is ``None``.
    """
    if quant_scales is None:
        return None

    idx = torch.tensor(expert_ids, dtype=torch.long,
                        device=quant_scales[0].device)
    subset_fields: Dict[str, torch.Tensor] = {}
    for name in quant_scales._fields:
        tensor = getattr(quant_scales, name)
        if tensor.dim() >= 1 and tensor.shape[0] > 1:
            subset_fields[name] = tensor[idx].contiguous()
        else:
            subset_fields[name] = tensor
    return type(quant_scales)(**subset_fields)


# ------------------------------------------------------------------
# Group descriptor (produced by config validation)
# ------------------------------------------------------------------


class _GroupDescriptor:
    """Internal metadata for one precision group."""

    __slots__ = ("name", "quant_algo", "size_ratio", "checkpoint")

    def __init__(
        self,
        name: str,
        quant_algo: Optional[QuantAlgo],
        size_ratio: float,
        checkpoint: Optional[str],
    ):
        self.name = name
        self.quant_algo = quant_algo
        self.size_ratio = size_ratio
        self.checkpoint = checkpoint

    @property
    def quant_label(self) -> str:
        return self.quant_algo.value if self.quant_algo else "BF16"


# ==================================================================
# HeterCutlassFusedMoE
# ==================================================================


class HeterCutlassFusedMoE(CutlassFusedMoE):
    """CutlassFusedMoE with heterogeneous-precision expert dispatch.

    N full weight sets are stored for every expert (one per precision
    group).  At runtime a pluggable **dispatch policy** determines
    which experts use which precision group's weights.  The policy is
    invoked every ``run_moe()`` call with router signals; the resulting
    assignment is cached and weight subset caches are only rebuilt when
    the assignment changes.

    Config schema (``heter_config``)::

        {
            "groups": [
                {
                    "name": "cold",
                    "quant_algo": "NVFP4",
                    "size_ratio": 0.80,
                    "checkpoint": "/path/to/nvfp4/checkpoint",
                },
                {
                    "name": "hot",
                    "quant_algo": null,
                    "size_ratio": 0.20,
                    "checkpoint": "/path/to/bf16/checkpoint",
                },
            ]
        }

    Per-group fields:
        name (str): Human-readable label for logging.
        quant_algo (str | None): ``"NVFP4"`` or ``null`` (BF16).
        size_ratio (float): Target fraction of experts for this group.
            All ratios must sum to 1.0.
        checkpoint (str | None): Path to the weight checkpoint for this
            precision.  Verified for existence at init time.
    """

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        model_config: ModelConfig = ModelConfig(),
        **kwargs: Any,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            model_config=model_config,
            **kwargs,
        )

        # --- Parse & validate config ---
        heter_config = self._extract_heter_config(model_config)
        dtype_act = getattr(
            model_config.pretrained_config, "torch_dtype", torch.bfloat16,
        )
        self._group_descs: List[_GroupDescriptor] = (
            self._validate_heter_config(heter_config, num_experts, dtype_act)
        )

        # --- Dispatch policy (default: random with fixed seed) ---
        self._dispatch_policy: BaseDispatchPolicy = (
            RandomDispatchPolicy(seed=42)
        )

        # --- Current assignment (populated by _recompute_dispatch) ---
        self._current_expert_ids: List[List[int]] = (
            [[] for _ in self._group_descs]
        )
        # Cache key for O(1) change detection
        self._current_assignment_key: Optional[tuple] = None

        # Router logits stashed by forward_chunk() for run_moe() to consume.
        # This is set immediately before super().forward_chunk() which
        # synchronously calls self.run_moe(), so the lifecycle is a single
        # call-stack frame.  Safe under torch.compile (moe_custom_op body
        # executes eagerly) and CUDA graphs (assignment is captured once;
        # the stashed tensor shares memory with the graph input).
        self._pending_router_logits: Optional[torch.Tensor] = None

        # Caches (built in post_load_weights / _build_group_caches)
        self._remap_tables: Optional[List[torch.Tensor]] = None
        self._group_w3_w1: Optional[List[torch.Tensor]] = None
        self._group_w2: Optional[List[torch.Tensor]] = None
        self._group_w3_w1_bias: Optional[List[Optional[torch.Tensor]]] = None
        self._group_w2_bias: Optional[List[Optional[torch.Tensor]]] = None
        self._group_quant_scales: Optional[List[Any]] = None

        self._log_config_summary()

    # ==============================================================
    # Config extraction & validation
    # ==============================================================

    @staticmethod
    def _extract_heter_config(model_config: ModelConfig) -> Dict[str, Any]:
        extra_attrs = getattr(model_config, "extra_attrs", None) or {}
        heter_config = extra_attrs.get("heter_moe_config")
        if heter_config is None:
            raise ValueError(
                "HeterCutlassFusedMoE requires 'heter_moe_config' in "
                "model_config.extra_attrs.  Set MoeConfig(backend='HETER', "
                "heter_config={...}) when constructing the LLM instance."
            )
        return heter_config

    @staticmethod
    def _validate_heter_config(
        heter_config: Dict[str, Any],
        num_experts: int,
        dtype_activation: torch.dtype,
    ) -> List[_GroupDescriptor]:
        """Validate ``heter_config`` and return group descriptors.

        Checks:
        1. ``groups`` is a non-empty list of dicts.
        2. Each group has a valid ``quant_algo`` supported by
           ``CutlassFusedMoE.can_implement()``.
        3. ``size_ratio`` values are in (0, 1] and sum to 1.0.
        4. ``checkpoint`` path exists on disk (if provided).
        """
        groups_raw = heter_config.get("groups")
        if not groups_raw or not isinstance(groups_raw, list):
            raise ValueError(
                "heter_config['groups'] must be a non-empty list of dicts.  "
                f"Got: {groups_raw!r}"
            )

        descs: List[_GroupDescriptor] = []
        total_ratio = 0.0

        for i, grp in enumerate(groups_raw):
            if not isinstance(grp, dict):
                raise ValueError(
                    f"heter_config['groups'][{i}] must be a dict, "
                    f"got {type(grp).__name__}"
                )
            name = grp.get("name", f"group_{i}")

            # -- quant_algo --
            raw_quant = grp.get("quant_algo")
            quant_algo = _resolve_quant_algo(raw_quant)
            if quant_algo not in _SUPPORTED_QUANT_ALGOS:
                raise ValueError(
                    f"Group '{name}': quant_algo={raw_quant!r} resolved to "
                    f"{quant_algo!r} which is not supported.  "
                    f"Supported: {_SUPPORTED_QUANT_ALGOS}"
                )
            can_impl, reason = CutlassFusedMoE.can_implement(
                quant_algo=quant_algo,
                dtype_activation=dtype_activation,
            )
            if not can_impl:
                raise ValueError(
                    f"Group '{name}': quant_algo={quant_algo!r} is not "
                    f"implementable on this hardware — {reason}"
                )

            # -- size_ratio --
            size_ratio = grp.get("size_ratio")
            if size_ratio is None:
                raise ValueError(
                    f"Group '{name}': 'size_ratio' is required."
                )
            size_ratio = float(size_ratio)
            if not (0.0 < size_ratio <= 1.0):
                raise ValueError(
                    f"Group '{name}': size_ratio must be in (0, 1].  "
                    f"Got: {size_ratio}"
                )
            total_ratio += size_ratio

            # -- checkpoint --
            checkpoint = grp.get("checkpoint")
            if checkpoint is not None:
                checkpoint = str(checkpoint)
                if not os.path.exists(checkpoint):
                    raise ValueError(
                        f"Group '{name}': checkpoint path does not exist: "
                        f"{checkpoint!r}"
                    )

            descs.append(_GroupDescriptor(
                name=name,
                quant_algo=quant_algo,
                size_ratio=size_ratio,
                checkpoint=checkpoint,
            ))

        if not math.isclose(total_ratio, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"size_ratio values must sum to 1.0 — got {total_ratio:.4f} "
                f"(groups: {[d.name for d in descs]})"
            )

        return descs

    # ==============================================================
    # Dispatch policy management
    # ==============================================================

    def set_dispatch_policy(self, policy: BaseDispatchPolicy) -> None:
        """Replace the current dispatch policy.

        Forces a recompute on the next ``run_moe()`` call.

        Args:
            policy: A :class:`BaseDispatchPolicy` implementation.
        """
        self._dispatch_policy = policy
        self._current_assignment_key = None  # Force recompute

    def _recompute_dispatch(
        self,
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
    ) -> bool:
        """Run the dispatch policy and rebuild caches if assignment changed.

        Args:
            token_selected_experts: Expert IDs selected by the router.
                Shape ``[num_tokens, top_k]``.
            token_final_scales: Routing weights per token.
                Shape ``[num_tokens, top_k]``.
            router_logits: Raw logits from the router gate.
                Shape ``[num_tokens, num_experts]``.  ``None`` when
                ``run_moe()`` is called directly (without
                ``forward_chunk``).

        Returns:
            ``True`` if caches were rebuilt, ``False`` if unchanged.
        """
        group_ratios = [d.size_ratio for d in self._group_descs]
        plan = self._dispatch_policy.assign(
            num_experts=self.num_experts,
            group_size_ratios=group_ratios,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            router_logits=router_logits,
        )
        # Validation is debug-only — pure Python overhead (set ops)
        # that should not be on the critical path in production.
        # Runs with `python` but optimised away with `python -O`.
        if __debug__:
            plan.validate(self.num_experts)

        key = tuple(tuple(ids) for ids in plan.group_assignments)
        if key == self._current_assignment_key:
            return False

        self._current_expert_ids = plan.group_assignments
        self._current_assignment_key = key
        self._build_group_caches()

        logger.info(
            f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: "
            f"dispatch updated — "
            + ", ".join(
                f"{d.name}/{d.quant_label}"
                f"({len(self._current_expert_ids[i])})"
                for i, d in enumerate(self._group_descs)
            )
        )
        return True

    # ==============================================================
    # Logging
    # ==============================================================

    def _log_config_summary(self) -> None:
        """Log a summary of the validated heterogeneous MoE configuration."""
        lines = [
            f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: "
            f"configuration validated — "
            f"{len(self._group_descs)} precision groups, "
            f"policy={type(self._dispatch_policy).__name__}",
        ]
        for i, desc in enumerate(self._group_descs):
            ckpt_str = (
                f"checkpoint={desc.checkpoint!r}"
                if desc.checkpoint else "checkpoint=<not set>"
            )
            lines.append(
                f"  [{i}] '{desc.name}': quant={desc.quant_label}, "
                f"size_ratio={desc.size_ratio:.0%}, {ckpt_str}"
            )

        logger.info("\n".join(lines))

    # ==============================================================
    # can_implement — verify each precision in the supported set
    # ==============================================================

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Check that every precision in ``_SUPPORTED_QUANT_ALGOS`` is
        implementable by CutlassFusedMoE on the current hardware."""
        for algo in _SUPPORTED_QUANT_ALGOS:
            can, reason = CutlassFusedMoE.can_implement(
                quant_algo=algo,
                dtype_activation=dtype_activation,
                gptoss_style=gptoss_style,
            )
            if not can:
                return (
                    False,
                    f"HeterCutlassFusedMoE requires {algo!r} support — "
                    f"{reason}",
                )
        return True, None

    # ==============================================================
    # post_load_weights / _build_group_caches
    # ==============================================================

    def post_load_weights(self) -> None:
        """Parent post-processing, then build initial caches via policy."""
        super().post_load_weights()
        # Compute initial assignment (no signals available yet)
        self._recompute_dispatch()

    def _build_group_caches(self) -> None:
        """(Re)build remap tables and weight/scale subsets for the current
        dispatch assignment.

        TODO(phase2): when dual weight sets are loaded, select from the
        appropriate weight set based on each group's ``quant_algo``.
        Currently all groups subset from the single parent weight set.
        """
        device = self.w3_w1_weight.device
        weight_dtype = self.w3_w1_weight.dtype

        remap_tables: List[torch.Tensor] = []
        group_w3_w1: List[torch.Tensor] = []
        group_w2: List[torch.Tensor] = []
        group_w3_w1_bias: List[Optional[torch.Tensor]] = []
        group_w2_bias: List[Optional[torch.Tensor]] = []
        group_quant_scales: List[Any] = []

        for group_idx, expert_ids in enumerate(self._current_expert_ids):
            num_group = len(expert_ids)
            desc = self._group_descs[group_idx]

            if num_group == 0:
                # Empty group — placeholder entries so indices stay aligned
                remap_tables.append(
                    torch.full((self.num_experts,), 0,
                               device=device, dtype=torch.int32)
                )
                group_w3_w1.append(torch.empty(0, device=device))
                group_w2.append(torch.empty(0, device=device))
                group_w3_w1_bias.append(None)
                group_w2_bias.append(None)
                group_quant_scales.append(None)
                continue

            # -- Remap table: global expert ID → local index --
            remap = torch.full(
                (self.num_experts,), num_group,
                device=device, dtype=torch.int32,
            )
            ids_tensor = torch.tensor(
                expert_ids, device=device, dtype=torch.long,
            )
            remap[ids_tensor] = torch.arange(
                num_group, device=device, dtype=torch.int32,
            )
            remap_tables.append(remap)

            # -- Subset weights --
            group_w3_w1.append(
                self.w3_w1_weight[ids_tensor].contiguous().view(weight_dtype)
            )
            group_w2.append(
                self.w2_weight[ids_tensor].contiguous().view(weight_dtype)
            )

            # -- Subset biases --
            if self.w3_w1_bias is not None:
                group_w3_w1_bias.append(
                    self.w3_w1_bias[ids_tensor].contiguous()
                )
            else:
                group_w3_w1_bias.append(None)
            if self.w2_bias is not None:
                group_w2_bias.append(
                    self.w2_bias[ids_tensor].contiguous()
                )
            else:
                group_w2_bias.append(None)

            # -- Subset quant scales --
            group_quant_scales.append(
                _subset_quant_scales(self.quant_scales, expert_ids)
            )

            logger.debug(
                f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: "
                f"cache built for '{desc.name}' "
                f"({desc.quant_label}, {num_group} experts)"
            )

        self._remap_tables = remap_tables
        self._group_w3_w1 = group_w3_w1
        self._group_w2 = group_w2
        self._group_w3_w1_bias = group_w3_w1_bias
        self._group_w2_bias = group_w2_bias
        self._group_quant_scales = group_quant_scales

    # ==============================================================
    # forward_chunk — capture router_logits for dispatch policy
    # ==============================================================

    def forward_chunk(
            self,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        """Override parent to capture ``router_logits`` for the dispatch
        policy.

        HETER MoE always receives **BF16 input** from attention.
        Per-group input quantization (e.g. FP4) is handled inside
        ``run_moe()`` on a per-group basis — never before this point.

        The stashed ``router_logits`` tensor is consumed by
        :meth:`run_moe` (our override) within the same synchronous call
        stack, then cleared in a ``finally`` block.

        **torch.compile safety**:

        ``MoE.forward()`` routes through ``moe_custom_op``
        (``@torch.library.custom_op``) when ``is_torch_compiling()`` is
        ``True``.  Dynamo treats custom ops as opaque — it uses
        ``register_fake`` for shape inference but never traces the real
        body.  At execution time the body runs as **plain eager Python**.
        Therefore:

        * ``self._pending_router_logits = router_logits`` — module
          attribute write inside eager code, invisible to Dynamo.
        * ``self._pending_router_logits`` read in ``run_moe()`` — same
          eager scope.
        * ``mutates_args=()`` on the custom op constrains tensor argument
          mutations only.  We never mutate ``router_logits`` itself.

        **CUDA graph safety**:

        During graph **capture** the full Python call stack executes
        normally, including stash / dispatch-policy / clear.  During
        **replay** only the recorded CUDA ops execute — no Python code
        re-runs.  Consequences:

        * The dispatch assignment from the capture batch is **baked**.
          Signal-based policies (e.g. ``ConfidenceThresholdPolicy``) will
          not re-evaluate until the graph is invalidated and recaptured.
        * ``_pending_router_logits`` remains ``None`` after capture
          (cleared by ``finally``).  This is harmless because ``run_moe``
          does not re-execute on replay.

        **Multi-chunk / aux-stream**:

        HETER MoE targets the **decoding** path where batch sizes are
        well below ``moe_max_num_tokens`` (always single chunk).  The
        parent alternates even/odd chunks between main and aux CUDA
        streams, but Python code runs sequentially on one thread — no
        race condition on ``_pending_router_logits``.
        """
        assert not isinstance(x, Fp4QuantizedTensor), (
            "HeterCutlassFusedMoE expects BF16 input from attention. "
            "Per-group quantization is handled inside run_moe()."
        )
        self._pending_router_logits = router_logits
        try:
            return super().forward_chunk(
                x,
                router_logits,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                repeating_info=repeating_info,
            )
        finally:
            self._pending_router_logits = None

    # ==============================================================
    # run_moe — per-group dispatch
    # ==============================================================

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        is_sf_swizzled: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: Optional[bool] = None,
    ) -> torch.Tensor:
        """Per-group dispatch: call fused_moe once per active group, sum.

        1. Call the dispatch policy with router signals to (re)compute
           expert-to-group assignment.  Weight caches are only rebuilt
           when the assignment changes (tuple-key comparison).
        2. **Fast path**: when all experts are in a single group,
           delegate to ``super().run_moe()`` with zero overhead.
        3. For each active group: remap, zero scales, call fused_moe,
           accumulate output.

        Router logits are obtained from :meth:`forward_chunk` (stashed
        in ``_pending_router_logits``) or ``None`` when ``run_moe`` is
        called directly (e.g. in tests).

        TODO(phase2): the quantized group paths must additionally
        quantise the input and pass per-group quant flags.  Currently
        all groups share the parent's quant flags (same-precision
        dispatch).
        """
        assert self._remap_tables is not None, (
            "post_load_weights() must be called before run_moe()"
        )

        # Retrieve router_logits stashed by forward_chunk(), if available.
        router_logits = self._pending_router_logits  # may be None

        # --- (Re)compute dispatch assignment via policy ---
        self._recompute_dispatch(
            token_selected_experts, token_final_scales, router_logits,
        )

        # --- Fast path: single active group covering all experts ---
        active = [
            i for i, ids in enumerate(self._current_expert_ids) if ids
        ]
        if len(active) == 1:
            gidx = active[0]
            if len(self._current_expert_ids[gidx]) == self.num_experts:
                return super().run_moe(
                    x=x,
                    token_selected_experts=token_selected_experts,
                    token_final_scales=token_final_scales,
                    x_sf=x_sf,
                    is_sf_swizzled=is_sf_swizzled,
                    output_dtype=output_dtype,
                    tuner_num_tokens=tuner_num_tokens,
                    tuner_top_k=tuner_top_k,
                    moe_output=moe_output,
                    enable_alltoall=enable_alltoall,
                )

        if enable_alltoall is None:
            enable_alltoall = self.enable_alltoall

        weight_dtype = self.w3_w1_weight.dtype
        if self.has_any_quant:
            if self.has_w4afp8:
                weight_dtype = torch.quint4x2
            elif self.has_w4a16_mxfp4:
                weight_dtype = torch.uint8

        accumulated: Optional[torch.Tensor] = None

        for group_idx in active:
            num_group = len(self._current_expert_ids[group_idx])
            desc = self._group_descs[group_idx]

            # -- Remap expert IDs to group-local indices --
            remap = self._remap_tables[group_idx]
            local_experts = remap[
                token_selected_experts.long()
            ].to(torch.int32)

            # -- Zero scales for non-group experts --
            valid_mask = local_experts < num_group
            group_scales = torch.where(
                valid_mask,
                token_final_scales,
                torch.zeros_like(token_final_scales),
            )

            # -- Per-group quant flags --
            # TODO(phase2): branch on desc.quant_algo to select the
            # correct weight set and quant flags per group.  For now
            # all groups use the parent's single-precision flags.
            group_x = x
            group_x_sf = x_sf
            group_is_sf_swizzled = is_sf_swizzled
            group_weight_dtype = weight_dtype

            group_result = torch.ops.trtllm.fused_moe(
                group_x,
                local_experts,
                group_scales,
                self._group_w3_w1[group_idx].view(group_weight_dtype),
                self._group_w3_w1_bias[group_idx],
                self._group_w2[group_idx].view(group_weight_dtype),
                self._group_w2_bias[group_idx],
                output_dtype,
                quant_scales=self._group_quant_scales[group_idx],
                input_sf=group_x_sf,
                swizzled_input_sf=group_is_sf_swizzled,
                swiglu_alpha=self.swiglu_alpha,
                swiglu_beta=self.swiglu_beta,
                swiglu_limit=self.swiglu_limit,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                cluster_size=self.cluster_size,
                cluster_rank=self.cluster_rank,
                enable_alltoall=enable_alltoall,
                use_deepseek_fp8_block_scale=self.has_deepseek_fp8_block_scales,
                use_w4_group_scaling=self.has_w4afp8 or self.has_w4a16_mxfp4,
                use_int8_woq_per_channel=self.has_int8_woq_per_channel,
                use_mxfp8_act_scaling=self.has_w4a8_mxfp4_mxfp8,
                min_latency_mode=False,
                use_fused_finalize=self.use_fused_finalize,
                tune_max_num_tokens=self.tune_max_num_tokens,
                tuner_num_tokens=tuner_num_tokens,
                tuner_top_k=tuner_top_k,
                activation_type=self.activation_type,
                unpadded_hidden_size=self.unpadded_hidden_size,
                out_tensor=None,
            )[0]

            if accumulated is None:
                accumulated = group_result
            else:
                accumulated = accumulated + group_result

        if moe_output is not None:
            moe_output.copy_(accumulated)
            return moe_output
        return accumulated

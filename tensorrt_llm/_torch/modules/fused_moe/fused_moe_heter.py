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

Stores **N full sets of weights** (one per precision group) covering
**all** experts and dispatches each precision group through a separate
``torch.ops.trtllm.fused_moe()`` call with the appropriate weights and
quantization flags, then sums the outputs.

Expert-to-group assignment and per-group token dispatch are handled by
a pluggable :class:`~.policy.heter_dispatch.HeterDispatchPolicy`.  The
policy's :meth:`dispatch` method transforms standard *N*-expert routing
into per-group dispatches — conceptually expanding the expert space to
*N × G* where *G* is the number of precision groups.  Each group's
call only includes tokens with ≥1 expert in that group; the CUTLASS
kernel's built-in sparsity skips experts with zero routed tokens.

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
                "policy": "expert_load",
            },
        ),
    )

Supported dispatch policies (``"policy"`` key):

* ``"random"`` (default) — deterministic random assignment; ignores
  runtime signals.
* ``"confidence_threshold"`` — assigns by per-expert mean routing
  weight; high-weight experts go to the last (high-precision) group.
* ``"expert_load"`` — assigns by expert activation frequency; hot
  experts go to the last group.

The ``"policy"`` value can be a string (policy name with defaults) or
a dict ``{"type": "<name>", ...}`` with extra constructor kwargs::

    "policy": {
        "type": "confidence_threshold",
        "confidence_threshold": 0.7,
        "fallback_seed": 123,
    }
"""

import math
import os
import glob
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...model_config import ModelConfig
from ...utils import Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .quantization import NVFP4CutlassFusedMoEMethod, UnquantizedFusedMoEMethod
from .policy import HeterDispatchPolicy, resolve_dispatch_policy
from .routing import BaseMoeRoutingMethod

# Quantization algorithms supported by the HETER backend.
# None means unquantized (BF16/FP16).
_SUPPORTED_QUANT_ALGOS = frozenset({None, QuantAlgo.NVFP4})

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _QuantizedInput(NamedTuple):
    """Result of per-group input quantisation."""
    x: torch.Tensor
    x_sf: Optional[torch.Tensor]
    is_sf_swizzled: bool


def _quantize_input_for_group(
    x: torch.Tensor,
    quant_algo: Optional[QuantAlgo],
    weight_set: Optional['_GroupWeightSet'],
) -> _QuantizedInput:
    """Quantise *x* according to a precision group's ``quant_algo``.

    This is the single dispatch point for per-group input quantisation
    inside :meth:`HeterCutlassFusedMoE.run_moe`.  Adding support for a
    new quantisation algorithm only requires a new ``elif`` branch here
    (plus registering the algo in ``_SUPPORTED_QUANT_ALGOS``).

    Args:
        x: BF16 activations from attention.
        quant_algo: The group's quantisation algorithm (``None`` for
            BF16 / unquantised).
        weight_set: The group's :class:`_GroupWeightSet`.  Required for
            quantised groups (carries input scales, block size, etc.).

    Returns:
        A :class:`_QuantizedInput` triple ``(x, x_sf, is_sf_swizzled)``.
    """
    if quant_algo is None:
        # BF16 / unquantised — pass through unchanged.
        return _QuantizedInput(x=x, x_sf=None, is_sf_swizzled=True)

    if quant_algo == QuantAlgo.NVFP4:
        assert weight_set is not None, (
            "NVFP4 group requires a registered weight set "
            "with fc31_input_scale"
        )
        qx, qx_sf = torch.ops.trtllm.fp4_quantize(
            x,
            weight_set.fc31_input_scale,
            weight_set.scaling_vector_size,
            False,   # input sf is not swizzled
            True,    # output sf swizzled for kernel
        )
        return _QuantizedInput(x=qx, x_sf=qx_sf, is_sf_swizzled=True)

    raise ValueError(
        f"_quantize_input_for_group: unsupported quant_algo={quant_algo!r}. "
        f"Supported: {_SUPPORTED_QUANT_ALGOS}"
    )


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
        return str(self.quant_algo.value) if self.quant_algo else "BF16"


class _GroupWeightSet:
    """Full weight set for one precision group covering all experts.

    Each group stores its own copy of model weights in the appropriate
    dtype/format for its quantization algorithm.  The BF16 group may
    reference the parent module's weights directly.
    """

    __slots__ = (
        "w3_w1_weight", "w2_weight", "w3_w1_bias", "w2_bias",
        "quant_scales", "weight_dtype", "quant_algo",
        "fc31_input_scale", "scaling_vector_size",
    )

    def __init__(
        self,
        w3_w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        quant_scales: Any,
        weight_dtype: torch.dtype,
        quant_algo: Optional[QuantAlgo] = None,
        w3_w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
        fc31_input_scale: Optional[torch.Tensor] = None,
        scaling_vector_size: int = 16,
    ):
        self.w3_w1_weight = w3_w1_weight
        self.w2_weight = w2_weight
        self.w3_w1_bias = w3_w1_bias
        self.w2_bias = w2_bias
        self.quant_scales = quant_scales
        self.weight_dtype = weight_dtype
        self.quant_algo = quant_algo
        self.fc31_input_scale = fc31_input_scale
        self.scaling_vector_size = scaling_vector_size


class _ModuleProxy(torch.nn.Module):

    def __init__(self, module: 'HeterCutlassFusedMoE'):
        super().__init__()
        self.num_experts = module.num_experts
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.intermediate_size_per_partition = module.intermediate_size_per_partition
        self.intermediate_size_expand_ratio = module.intermediate_size_expand_ratio
        self.expert_size_per_partition = module.expert_size_per_partition
        self.tp_size = module.tp_size
        self.tp_rank = module.tp_rank
        self.ep_size = module.ep_size
        self.ep_rank = module.ep_rank
        self.bias = module.bias
        self.weight_loading_mode = module.weight_loading_mode
        self.initial_local_expert_ids = module.initial_local_expert_ids
        self.dtype = module.dtype
        self.scaling_vector_size = getattr(module, 'scaling_vector_size', 16)
        self.layer_load_balancer = None
        self.rebuild_tensor_metadata = {}

    @property
    def expand_intermediate_size_per_partition(self) -> int:
        return (
            self.intermediate_size_per_partition
            * self.intermediate_size_expand_ratio
        )

    def _add_raw_shared_weights_for_unmap(self, weight_tensors):
        pass


# ==================================================================
# HeterCutlassFusedMoE
# ==================================================================


class HeterCutlassFusedMoE(CutlassFusedMoE):
    """CutlassFusedMoE with heterogeneous-precision expert dispatch.

    N full weight sets are stored covering **all** experts (one set per
    precision group).  At runtime a pluggable **dispatch policy**
    determines which experts belong to which precision group.  For each
    group, ``run_moe()`` passes the **full** weight tensor to the
    CUTLASS kernel with router scales zeroed for non-group experts.
    The kernel skips experts with zero tokens automatically (built-in
    sparsity), so no weight subsetting or remapping is needed.

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

    Optional top-level fields:
        weight_key_prefix (str | None): Key prefix pattern for extracting
            MoE expert weights from checkpoint files.  Use ``{layer_idx}``
            as a placeholder for the transformer layer index.  Example:
            ``"model.layers.{layer_idx}.mlp.experts"``.  If ``None``,
            checkpoint weights are assumed to be layer-relative (no prefix
            stripping).
        policy (str | dict | None): Dispatch policy selection.  A string
            selects a built-in policy with default kwargs; a dict with a
            ``"type"`` key passes extra kwargs to the constructor.
            Supported values: ``"random"`` (default),
            ``"confidence_threshold"``, ``"expert_load"``.
    """

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        model_config: ModelConfig[Any] = ModelConfig(),
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
        self._weight_key_prefix: Optional[str] = heter_config.get(
            "weight_key_prefix"
        )

        # --- Dispatch policy ---
        self._policy: HeterDispatchPolicy = resolve_dispatch_policy(
            heter_config,
            num_experts,
            [d.size_ratio for d in self._group_descs],
        )

        # # Router logits stashed by forward_chunk() for run_moe() to
        # # pass to the dispatch policy.  Currently unused by all
        # # dispatch policies.
        # self._pending_router_logits: Optional[torch.Tensor] = None

        # Per-group full weight sets (populated by register_group_weights).
        # ``None`` means "fall back to the parent module's weights".
        self._heter_weight_sets: List[Optional[_GroupWeightSet]] = [
            None for _ in self._group_descs
        ]

        self._log_config_summary()

    # ==============================================================
    # Config extraction & validation
    # ==============================================================

    @staticmethod
    def _extract_heter_config(model_config: ModelConfig[Any]) -> Dict[str, Any]:
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

        weight_key_prefix = heter_config.get("weight_key_prefix")
        if weight_key_prefix is not None and not isinstance(weight_key_prefix,
                                                             str):
            raise ValueError(
                "heter_config['weight_key_prefix'] must be a string or None.  "
                f"Got: {weight_key_prefix!r}"
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
    # Dispatch policy
    # ==============================================================

    @property
    def policy(self) -> HeterDispatchPolicy:
        """The current heterogeneous dispatch policy."""
        return self._policy

    @policy.setter
    def policy(self, new_policy: HeterDispatchPolicy) -> None:
        """Replace the dispatch policy."""
        self._policy = new_policy

    def register_group_weights(
        self,
        group_idx: int,
        w3_w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        quant_scales: Any = tuple(),
        weight_dtype: Optional[torch.dtype] = None,
        w3_w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
        fc31_input_scale: Optional[torch.Tensor] = None,
        scaling_vector_size: int = 16,
    ) -> None:
        """Register a full weight set for one precision group.

        This must be called (once per group) before ``post_load_weights``.
        Groups that are not registered fall back to the parent module's
        weights at runtime (valid for BF16 groups whose weights match
        the parent).

        Args:
            group_idx: Index into ``_group_descs``.
            w3_w1_weight: Fused gate/up-proj weight for **all** experts.
                Shape ``[num_experts, intermediate, hidden]`` (or packed
                equivalent for quantized formats).
            w2_weight: Down-proj weight for all experts.
            quant_scales: Quantization scales NamedTuple (e.g.
                ``FusedMoEQuantScalesNVFP4``) or ``tuple()`` for
                unquantized weights.
            weight_dtype: Storage dtype for ``.view()`` in fused_moe.
                Defaults to ``w3_w1_weight.dtype``.
            w3_w1_bias: Optional bias for gate/up-proj.
            w2_bias: Optional bias for down-proj.
            fc31_input_scale: NVFP4 input activation scale (scalar).
                Required when ``quant_algo == QuantAlgo.NVFP4``.
            scaling_vector_size: NVFP4 block size for input
                quantization.  Default 16.
        """
        if group_idx < 0 or group_idx >= len(self._group_descs):
            raise IndexError(
                f"group_idx={group_idx} out of range "
                f"[0, {len(self._group_descs)})"
            )
        desc = self._group_descs[group_idx]
        if weight_dtype is None:
            weight_dtype = w3_w1_weight.dtype

        if desc.quant_algo == QuantAlgo.NVFP4 and fc31_input_scale is None:
            raise ValueError(
                f"Group '{desc.name}' uses NVFP4 but no "
                f"fc31_input_scale was provided."
            )

        self._heter_weight_sets[group_idx] = _GroupWeightSet(
            w3_w1_weight=w3_w1_weight,
            w2_weight=w2_weight,
            quant_scales=quant_scales,
            weight_dtype=weight_dtype,
            quant_algo=desc.quant_algo,
            w3_w1_bias=w3_w1_bias,
            w2_bias=w2_bias,
            fc31_input_scale=fc31_input_scale,
            scaling_vector_size=scaling_vector_size,
        )
        logger.info(
            f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: "
            f"registered weight set for group '{desc.name}' "
            f"({desc.quant_label}, dtype={weight_dtype})"
        )

    # ==============================================================
    # Logging
    # ==============================================================

    def _log_config_summary(self) -> None:
        """Log a summary of the validated heterogeneous MoE configuration."""
        lines = [
            f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: "
            f"configuration validated — "
            f"{len(self._group_descs)} precision groups, "
            f"policy={type(self._policy).__name__}",
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

    @staticmethod
    def _get_group_quant_method(quant_algo: Optional[QuantAlgo]):
        if quant_algo is None:
            return UnquantizedFusedMoEMethod()
        if quant_algo == QuantAlgo.NVFP4:
            return NVFP4CutlassFusedMoEMethod()
        raise ValueError(f"Unsupported quant_algo for group loading: {quant_algo!r}")

    @staticmethod
    def _load_checkpoint_weights(checkpoint_path: str) -> Dict[str, Any]:
        weight_files = glob.glob(f"{checkpoint_path}/*.safetensors")
        filtered_weight_files = [
            x for x in weight_files if "consolidated" not in os.path.split(x)[1]
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        if not weight_files:
            raise RuntimeError(
                "No .safetensors files found in checkpoint directory: "
                f"{checkpoint_path}"
            )

        import importlib

        st = importlib.import_module("safetensors.torch")

        merged: Dict[str, Any] = {}
        for file in weight_files:
            merged.update(st.load_file(file))
        return merged

    @staticmethod
    def _extract_moe_layer_weights(
        all_weights: Dict[str, Any],
        weight_key_prefix: str,
    ) -> Dict[str, Any]:
        full_prefix = f"{weight_key_prefix}."
        return {
            k[len(full_prefix):]: v
            for k, v in all_weights.items()
            if k.startswith(full_prefix)
        }

    def _load_group_from_checkpoint(
        self,
        group_idx: int,
        layer_weights: Dict[str, Any],
    ) -> None:
        desc = self._group_descs[group_idx]
        proxy = _ModuleProxy(self)
        quant_method = self._get_group_quant_method(desc.quant_algo)
        quant_method.create_weights(proxy)
        quant_method.load_weights(proxy, layer_weights, self.weight_loading_mode)
        quant_method.post_load_weights(proxy)

        fc31_input_scale = getattr(proxy, 'fc31_input_scale', None)
        if fc31_input_scale is not None:
            fc31_input_scale = fc31_input_scale.data
        scaling_vector_size = getattr(proxy, 'scaling_vector_size', 16)

        quant_scales = getattr(proxy, 'quant_scales', ())
        self.register_group_weights(
            group_idx,
            w3_w1_weight=proxy.w3_w1_weight.data,
            w2_weight=proxy.w2_weight.data,
            quant_scales=quant_scales,
            weight_dtype=proxy.w3_w1_weight.dtype,
            w3_w1_bias=proxy.w3_w1_bias.data if proxy.w3_w1_bias is not None else None,
            w2_bias=proxy.w2_bias.data if proxy.w2_bias is not None else None,
            fc31_input_scale=fc31_input_scale,
            scaling_vector_size=scaling_vector_size,
        )

        logger.info(
            f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: loaded group "
            f"'{desc.name}' weights from checkpoint"
        )

    def _load_all_group_checkpoints(
        self,
        weight_key_prefix: Optional[str],
    ) -> None:
        for group_idx, desc in enumerate(self._group_descs):
            if desc.checkpoint is None:
                continue

            logger.info(
                f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: loading "
                f"checkpoint for group '{desc.name}' from {desc.checkpoint}"
            )

            all_weights = self._load_checkpoint_weights(desc.checkpoint)
            if weight_key_prefix is not None:
                prefix = weight_key_prefix.format(layer_idx=self.layer_idx)
                layer_weights = self._extract_moe_layer_weights(
                    all_weights, prefix)
            else:
                layer_weights = all_weights

            self._load_group_from_checkpoint(group_idx, layer_weights)
            del all_weights

    def load_weights(self,
                     weights: List[Dict[Any, Any]],
                     allow_partial_loading: bool = False):
        """Load parent weights normally, then load per-group checkpoint weights."""
        super().load_weights(weights, allow_partial_loading=allow_partial_loading)
        if any(desc.checkpoint is not None for desc in self._group_descs):
            self._load_all_group_checkpoints(self._weight_key_prefix)

    # ==============================================================
    # post_load_weights
    # ==============================================================

    def post_load_weights(self) -> None:
        """Parent post-processing."""
        super().post_load_weights()

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
            repeating_info: tuple[bool, bool] = (True, True),
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
        # router_logits not stashed — no dispatch policy currently uses them.
        return super().forward_chunk(
            x,
            router_logits,
            output_dtype=output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            repeating_info=repeating_info,
        )

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
        """Per-group dispatch: call fused_moe once per group, sum outputs.

        The dispatch policy transforms standard *N*-expert routing into
        per-group dispatches — conceptually *N × G* virtual experts.
        Each group's call only includes tokens with ≥1 expert in that
        group; the kernel skips experts with zero routed tokens.

        See :class:`~.policy.heter_dispatch.HeterDispatchPolicy` for
        the dispatch transform details.
        """
        # --- Dispatch: split routing by group via policy ---
        dispatches = self._policy.dispatch(
            token_selected_experts,
            token_final_scales,
        )

        # --- Fast path: single active group with parent weights ---
        active = [
            (i, d) for i, d in enumerate(dispatches) if d[0] is not None
        ]
        if len(active) == 1:
            gidx, (tok_idx, _, _) = active[0]
            assert tok_idx is not None  # guaranteed by filter above
            if (tok_idx.numel() == x.shape[0]
                    and self._heter_weight_sets[gidx] is None):
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

        out_dtype = output_dtype if output_dtype is not None else x.dtype
        accumulated = torch.zeros(
            x.shape[0], x.shape[1], dtype=out_dtype, device=x.device,
        )

        for group_idx, (tok_idx, grp_experts, grp_scales) in active:

            # print("### [group]", group_idx, tok_idx.shape, tok_idx)

            desc = self._group_descs[group_idx]

            # -- Resolve weight source --
            ws = self._heter_weight_sets[group_idx]
            if ws is not None:
                src_w3_w1 = ws.w3_w1_weight
                src_w2 = ws.w2_weight
                src_w3_w1_bias = ws.w3_w1_bias
                src_w2_bias = ws.w2_bias
                src_quant_scales = ws.quant_scales
                src_weight_dtype = ws.weight_dtype
            else:
                src_w3_w1 = self.w3_w1_weight
                src_w2 = self.w2_weight
                src_w3_w1_bias = self.w3_w1_bias
                src_w2_bias = self.w2_bias
                src_quant_scales = self.quant_scales
                src_weight_dtype = self.w3_w1_weight.dtype

            # -- Per-group input quantisation --
            qi = _quantize_input_for_group(
                x[tok_idx], desc.quant_algo, ws,
            )

            group_result = torch.ops.trtllm.fused_moe(
                qi.x,
                grp_experts,
                grp_scales,
                src_w3_w1.view(src_weight_dtype),
                src_w3_w1_bias,
                src_w2.view(src_weight_dtype),
                src_w2_bias,
                output_dtype,
                quant_scales=src_quant_scales,
                input_sf=qi.x_sf,
                swizzled_input_sf=qi.is_sf_swizzled,
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
                use_deepseek_fp8_block_scale=False,
                use_w4_group_scaling=False,
                use_int8_woq_per_channel=False,
                use_mxfp8_act_scaling=False,
                min_latency_mode=False,
                use_fused_finalize=self.use_fused_finalize,
                tune_max_num_tokens=self.tune_max_num_tokens,
                tuner_num_tokens=tuner_num_tokens,
                tuner_top_k=tuner_top_k,
                activation_type=self.activation_type,
                unpadded_hidden_size=self.unpadded_hidden_size,
                out_tensor=None,
            )[0]

            accumulated[tok_idx] += group_result

            # break

        if moe_output is not None:
            moe_output.copy_(accumulated)
            return moe_output
        return accumulated

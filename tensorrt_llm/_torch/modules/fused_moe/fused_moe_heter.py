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

Stores N full weight sets (one per precision group) covering all experts.
Each group is dispatched through a separate ``fused_moe()`` call; outputs
are summed.

Config schema (``heter_config``)::

    {
        "groups": [
            {"name": "cold", "quant_algo": "NVFP4", "size_ratio": 0.80,
             "checkpoint": "/path/to/nvfp4/model"},
            {"name": "hot", "quant_algo": null, "size_ratio": 0.20,
             "checkpoint": "/path/to/bf16/model"},
        ],
        "policy": "expert_load",
        "weight_key_prefix": "model.layers.{layer_idx}.mlp.experts",
    }

Supported policies: ``"random"``, ``"confidence_threshold"``, ``"expert_load"``.
Pass a dict ``{"type": "<name>", ...}`` for extra constructor kwargs.
"""

import math
import os
import glob
from typing import Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...model_config import ModelConfig
from ...utils import Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .quantization import NVFP4CutlassFusedMoEMethod, UnquantizedFusedMoEMethod
from .policy import HeterDispatchPolicy, resolve_dispatch_policy
from .routing import BaseMoeRoutingMethod

_SUPPORTED_QUANT_ALGOS = frozenset({None, QuantAlgo.NVFP4})


def _resolve_quant_algo(raw: Any) -> Optional[QuantAlgo]:
    """Convert user-provided quant_algo to ``QuantAlgo`` or ``None``.

    Accepts ``None``, ``"BF16"``, ``"FP16"``, ``"NONE"`` (→ ``None``),
    ``"NVFP4"`` (→ ``QuantAlgo.NVFP4``), or a ``QuantAlgo`` member.
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


class _GroupDescriptor:
    """Internal metadata for one precision group."""

    __slots__ = ("name", "quant_algo", "size_ratio", "checkpoint")

    def __init__(self, name: str, quant_algo: Optional[QuantAlgo],
                 size_ratio: float, checkpoint: Optional[str]):
        self.name = name
        self.quant_algo = quant_algo
        self.size_ratio = size_ratio
        self.checkpoint = checkpoint

    @property
    def quant_label(self) -> str:
        return str(self.quant_algo.value) if self.quant_algo else "BF16"


class _GroupWeightSet:
    """Full weight set for one precision group covering all experts."""

    __slots__ = (
        "w3_w1_weight", "w2_weight", "w3_w1_bias", "w2_bias",
        "quant_scales", "weight_dtype", "quant_algo",
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
    ):
        self.w3_w1_weight = w3_w1_weight
        self.w2_weight = w2_weight
        self.w3_w1_bias = w3_w1_bias
        self.w2_bias = w2_bias
        self.quant_scales = quant_scales
        self.weight_dtype = weight_dtype
        self.quant_algo = quant_algo


class _ModuleProxy(torch.nn.Module):
    """Lightweight proxy that mimics module attributes for quant methods."""

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


class HeterCutlassFusedMoE(CutlassFusedMoE):
    """CutlassFusedMoE with heterogeneous-precision expert dispatch.

    Stores N full weight sets (one per precision group).  A pluggable
    dispatch policy assigns experts to groups at runtime.  Each group's
    ``run_moe()`` call uses the full weight tensor with non-group expert
    slots sentinel-masked (zero scale); the kernel skips them automatically.

    Config schema (``heter_config``)::

        {
            "groups": [
                {"name": "cold", "quant_algo": "NVFP4", "size_ratio": 0.80,
                 "checkpoint": "/path/to/nvfp4/checkpoint"},
                {"name": "hot", "quant_algo": null, "size_ratio": 0.20,
                 "checkpoint": "/path/to/bf16/checkpoint"},
            ],
            "weight_key_prefix": "model.layers.{layer_idx}.mlp.experts",
            "policy": "expert_load",
        }

    Per-group fields:
        name: Human-readable label.
        quant_algo: ``"NVFP4"`` or ``null`` (BF16).
        size_ratio: Fraction of experts (must sum to 1.0).
        checkpoint: Path to weights for this precision.

    Optional fields:
        weight_key_prefix: Key prefix with ``{layer_idx}`` placeholder.
        policy: ``"random"`` | ``"confidence_threshold"`` | ``"expert_load"``
            or dict ``{"type": "<name>", ...}`` with extra kwargs.
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
        self._policy: HeterDispatchPolicy = resolve_dispatch_policy(
            heter_config,
            num_experts,
            [d.size_ratio for d in self._group_descs],
        )

        # Per-group weight sets (populated by register_group_weights).
        self._heter_weight_sets: List[Optional[_GroupWeightSet]] = [
            None for _ in self._group_descs
        ]

        self._log_config_summary()

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
        """Validate ``heter_config`` and return group descriptors."""
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

            size_ratio = grp.get("size_ratio")
            if size_ratio is None:
                raise ValueError(
                    f"Group '{name}': size_ratio is required."
                )
            size_ratio = float(size_ratio)
            if not (0.0 < size_ratio <= 1.0):
                raise ValueError(
                    f"Group '{name}': size_ratio must be in (0, 1].  "
                    f"Got: {size_ratio}"
                )
            total_ratio += size_ratio

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

    @property
    def policy(self) -> HeterDispatchPolicy:
        return self._policy

    @policy.setter
    def policy(self, new_policy: HeterDispatchPolicy) -> None:
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
    ) -> None:
        """Register a full weight set for one precision group.

        Must be called once per group before ``post_load_weights``.
        Groups not registered fall back to the parent module's weights.
        """
        if group_idx < 0 or group_idx >= len(self._group_descs):
            raise IndexError(
                f"group_idx={group_idx} out of range "
                f"[0, {len(self._group_descs)})"
            )
        desc = self._group_descs[group_idx]
        if weight_dtype is None:
            weight_dtype = w3_w1_weight.dtype

        self._heter_weight_sets[group_idx] = _GroupWeightSet(
            w3_w1_weight=w3_w1_weight,
            w2_weight=w2_weight,
            quant_scales=quant_scales,
            weight_dtype=weight_dtype,
            quant_algo=desc.quant_algo,
            w3_w1_bias=w3_w1_bias,
            w2_bias=w2_bias,
        )
        logger.info(
            f"HeterCutlassFusedMoE layer_idx={self.layer_idx}: "
            f"registered weight set for group '{desc.name}' "
            f"({desc.quant_label}, dtype={weight_dtype})"
        )

    def _log_config_summary(self) -> None:
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

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Check all precisions in ``_SUPPORTED_QUANT_ALGOS`` are implementable."""
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
        raise ValueError(f"Unsupported quant_algo: {quant_algo!r}")

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

        self.register_group_weights(
            group_idx,
            w3_w1_weight=proxy.w3_w1_weight.data,
            w2_weight=proxy.w2_weight.data,
            quant_scales=getattr(proxy, 'quant_scales', ()),
            weight_dtype=proxy.w3_w1_weight.dtype,
            w3_w1_bias=proxy.w3_w1_bias.data if proxy.w3_w1_bias is not None else None,
            w2_bias=proxy.w2_bias.data if proxy.w2_bias is not None else None,
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
                full_prefix = f"{weight_key_prefix.format(layer_idx=self.layer_idx)}."
                layer_weights = {
                    k[len(full_prefix):]: v
                    for k, v in all_weights.items()
                    if k.startswith(full_prefix)
                }
            else:
                layer_weights = all_weights

            self._load_group_from_checkpoint(group_idx, layer_weights)
            del all_weights

    def load_weights(self, weights: List[Dict[Any, Any]],
                     allow_partial_loading: bool = False):
        """Load per-group checkpoint weights (skips parent weight loading)."""
        self._load_all_group_checkpoints(self._weight_key_prefix)

    def post_load_weights(self) -> None:
        super().post_load_weights()

    def forward_chunk(
            self,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple[bool, bool] = (True, True),
    ) -> torch.Tensor:
        """Verify BF16 input, then delegate to parent.

        Per-group quantisation is handled by the CUTLASS kernel inside
        ``run_moe()``, not here.
        """
        assert not isinstance(x, Fp4QuantizedTensor), (
            "HeterCutlassFusedMoE expects BF16 input from attention. "
            "Per-group quantization is handled inside run_moe()."
        )
        return super().forward_chunk(
            x,
            router_logits,
            output_dtype=output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            repeating_info=repeating_info,
        )

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

        The dispatch policy sentinel-masks non-group expert slots (zero
        scale) so the kernel skips them automatically.
        """
        dispatches = self._policy.dispatch(
            token_selected_experts,
            token_final_scales,
        )

        if enable_alltoall is None:
            enable_alltoall = self.enable_alltoall

        out_dtype = output_dtype if output_dtype is not None else x.dtype
        accumulated = torch.zeros(
            x.shape[0], x.shape[1], dtype=out_dtype, device=x.device,
        )

        for group_idx, (grp_experts, grp_scales) in enumerate(dispatches):
            ws = self._heter_weight_sets[group_idx]
            assert ws is not None, (
                f"Group {group_idx} has no registered weight set. "
                f"All groups must be populated via checkpoint or "
                f"register_group_weights() before forward."
            )
            src_w3_w1 = ws.w3_w1_weight
            src_w2 = ws.w2_weight
            src_w3_w1_bias = ws.w3_w1_bias
            src_w2_bias = ws.w2_bias
            src_quant_scales = ws.quant_scales
            src_weight_dtype = ws.weight_dtype

            # No Python-side pre-quantisation: the CUTLASS kernel fuses
            # bf16→fp4 quantisation into expandInputRowsKernel.
            group_result = torch.ops.trtllm.fused_moe(
                x,
                grp_experts,
                grp_scales,
                src_w3_w1.view(src_weight_dtype),
                src_w3_w1_bias,
                src_w2.view(src_weight_dtype),
                src_w2_bias,
                output_dtype,
                quant_scales=src_quant_scales,
                input_sf=None,
                swizzled_input_sf=True,
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

            accumulated += group_result

        if moe_output is not None:
            moe_output.copy_(accumulated)
            return moe_output
        return accumulated

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
"""Shared helpers for fused mixed-precision MoE tests and benchmarks.

NEW: Utility module for testing FusedMixedPrecisionMoE against a reference
implementation (CutlassFusedMoE with pure bf16).

This module is imported by:
  - tests/unittest/_torch/modules/moe/test_mixed_precision_moe.py
"""

import copy
from typing import Dict, Optional, Tuple

import torch
from _torch.modules.moe.quantize_utils import get_test_quant_params

from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import (
    CutlassFusedMoE,
)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_mixed_precision import (
    FusedMixedPrecisionMoE,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


# ---------------------------------------------------------------------------
# Hardware capability checks
# ---------------------------------------------------------------------------


def mixed_precision_supported(dtype: torch.dtype = torch.bfloat16) -> bool:
    """Check if the current GPU supports mixed-precision MoE (requires SM >= 100 for FP4)."""
    if not torch.cuda.is_available():
        return False
    sm = torch.cuda.get_device_capability()
    # SM >= 10.0 (Blackwell) for FP4 support
    return sm[0] >= 10


MIXED_PREC_UNAVAILABLE_REASON = (
    "Mixed-precision MoE requires SM >= 10.0 (Blackwell) for NVFP4 support"
)


# ---------------------------------------------------------------------------
# DeepSeek-V3-like configuration constants
# ---------------------------------------------------------------------------

# Full DeepSeek-V3 MoE layer parameters
DEEPSEEK_V3_NUM_EXPERTS = 256
DEEPSEEK_V3_TOP_K = 8
DEEPSEEK_V3_HIDDEN_SIZE = 7168
DEEPSEEK_V3_INTER_SIZE = 2048

# Scaled-down config for memory-limited GPUs (e.g., RTX 5070 with 12GB)
SMALL_NUM_EXPERTS = 16
SMALL_TOP_K = 4
SMALL_HIDDEN_SIZE = 512
SMALL_INTER_SIZE = 512


# ---------------------------------------------------------------------------
# Weight creation helpers
# ---------------------------------------------------------------------------


def create_unquantized_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    kaiming_fan_out: bool = True,
) -> Dict[str, torch.Tensor]:
    """Create unquantized bf16 weights for all experts.

    Returns a weight dict compatible with CutlassFusedMoE.load_weights().
    Applies Kaiming fan-out scaling to keep activations in a sane range.
    """
    quantize_util_cls, quant_config, _ = get_test_quant_params(
        quant_algo=None, x=None
    )
    quantize_util_kwargs = {
        "num_experts": num_experts,
        "dtype": dtype,
        "intermediate_size": intermediate_size,
        "hidden_size": hidden_size,
        "quant_config": quant_config,
    }
    quantize_util = quantize_util_cls(**quantize_util_kwargs)
    weights = quantize_util.create_weights()

    if kaiming_fan_out:
        for key, val in weights.items():
            if isinstance(val, torch.Tensor) and val.ndim == 2:
                fan_out = val.shape[0]
                weights[key] = val * (2.0 / fan_out) ** 0.5

    return weights


def quantize_bf16_to_nvfp4(
    bf16_weights: Dict[str, torch.Tensor],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    x: torch.Tensor,
    scaling_vector_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Quantize bf16 MoE weights to NVFP4 format.

    This reuses the quantize_bf16_to_nvfp4 pattern from heter_moe_utils.
    Returns a weight dict with fp4-quantized weights and scaling factors.
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
            w1, w3_w1_sf, scaling_vector_size, False, False
        )
        w1_sb = w1_sb.view(intermediate_size, -1)

        w2_q, w2_sb = torch.ops.trtllm.fp4_quantize(
            w2, w2_sf, scaling_vector_size, False, False
        )
        w2_sb = w2_sb.view(hidden_size, -1)

        w3_q, w3_sb = torch.ops.trtllm.fp4_quantize(
            w3, w3_w1_sf, scaling_vector_size, False, False
        )
        w3_sb = w3_sb.view(intermediate_size, -1)

        weights[f"{expert_id}.w1.weight"] = w1_q
        weights[f"{expert_id}.w2.weight"] = w2_q
        weights[f"{expert_id}.w3.weight"] = w3_q
        weights[f"{expert_id}.w1.weight_scale"] = w1_sb.view(
            torch.float8_e4m3fn
        ).cuda()
        weights[f"{expert_id}.w2.weight_scale"] = w2_sb.view(
            torch.float8_e4m3fn
        ).cuda()
        weights[f"{expert_id}.w3.weight_scale"] = w3_sb.view(
            torch.float8_e4m3fn
        ).cuda()
        weights[f"{expert_id}.w1.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{expert_id}.w2.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{expert_id}.w3.input_scale"] = 1.0 / x_sf_global.cuda()
        weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_sf
        weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf
        weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_sf
    return weights


# ---------------------------------------------------------------------------
# Weight packing helpers for FusedMixedPrecisionMoE
# ---------------------------------------------------------------------------


def pack_bf16_weights_for_fused_module(
    bf16_weights: Dict[str, torch.Tensor],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack per-expert bf16 weights into fused [E, inter*2, hidden] and [E, hidden, inter] tensors.

    The w3_w1 tensor fuses w3 (gate) and w1 (up) weights along the out_features dim.

    Returns:
        (w3_w1_weight_bf16, w2_weight_bf16)
    """
    w3_w1_list = []
    w2_list = []
    for expert_id in range(num_experts):
        w1 = bf16_weights[f"{expert_id}.w1.weight"]  # [inter, hidden]
        w2 = bf16_weights[f"{expert_id}.w2.weight"]  # [hidden, inter]
        w3 = bf16_weights[f"{expert_id}.w3.weight"]  # [inter, hidden]
        # Fuse w3 (gate) on top, w1 (up) on bottom: [inter*2, hidden]
        w3_w1 = torch.cat([w3, w1], dim=0)
        w3_w1_list.append(w3_w1)
        w2_list.append(w2)

    w3_w1_weight = torch.stack(w3_w1_list, dim=0)  # [E, inter*2, hidden]
    w2_weight = torch.stack(w2_list, dim=0)  # [E, hidden, inter]
    return w3_w1_weight, w2_weight


def _interleave_block_scales(block_scales_3d: torch.Tensor) -> torch.Tensor:
    """Apply block_scale_interleave per-expert and pack as int32.

    The C++ MoE kernel expects block scales in interleaved int32 format.
    This replicates the conversion done by
    NVFP4CutlassFusedMoEMethod.load_expert_w*_weight_scale_nvfp4.

    Args:
        block_scales_3d: [E, rows, cols] float8_e4m3fn from fp4_quantize

    Returns:
        [E, rows, cols // 4] int32 interleaved block scales
    """
    results = []
    for i in range(block_scales_3d.shape[0]):
        expert_scales = block_scales_3d[i]  # [rows, cols] float8_e4m3fn
        orig_shape = expert_scales.shape
        interleaved = torch.ops.trtllm.block_scale_interleave(
            expert_scales.contiguous().view(torch.uint8))
        results.append(interleaved.view(torch.int32).reshape(
            orig_shape[0], -1))
    return torch.stack(results, dim=0)

def pack_fp4_weights_for_fused_module(
    fp4_weights: Dict[str, torch.Tensor],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Pack per-expert NVFP4 weights into fused tensors for FusedMixedPrecisionMoE.

    Returns:
        (w3_w1_weight_fp4, w2_weight_fp4, fp4_quant_scales)

    The fp4_quant_scales is a list of 6 tensors:
        [fc1_act_global, fc1_weight_block, fc1_global,
         fc2_act_global, fc2_weight_block, fc2_global]
    """
    fp4_pack_factor = 16  # 16 nvfp4 values per int64

    w3_w1_list = []
    w2_list = []

    # For scales, we stack per-expert scales
    fc1_weight_block_scales = []
    fc2_weight_block_scales = []
    fc1_act_global_scales = []
    fc2_act_global_scales = []
    fc1_global_scales = []
    fc2_global_scales = []

    for expert_id in range(num_experts):
        w1_q = fp4_weights[f"{expert_id}.w1.weight"]  # packed fp4
        w2_q = fp4_weights[f"{expert_id}.w2.weight"]
        w3_q = fp4_weights[f"{expert_id}.w3.weight"]

        # Fuse w3+w1 quantized (they share the same global scale)
        w3_w1_q = torch.cat([w3_q, w1_q], dim=0)
        w3_w1_list.append(w3_w1_q)
        w2_list.append(w2_q)

        # Block scales (per-expert)
        w1_bs = fp4_weights[f"{expert_id}.w1.weight_scale"]
        w2_bs = fp4_weights[f"{expert_id}.w2.weight_scale"]
        w3_bs = fp4_weights[f"{expert_id}.w3.weight_scale"]
        fc1_weight_block_scales.append(torch.cat([w3_bs, w1_bs], dim=0))
        fc2_weight_block_scales.append(w2_bs)

        # Global scales
        fc1_act_global_scales.append(fp4_weights[f"{expert_id}.w1.input_scale"])
        fc2_act_global_scales.append(fp4_weights[f"{expert_id}.w2.input_scale"])
        fc1_global_scales.append(fp4_weights[f"{expert_id}.w1.weight_scale_2"])
        fc2_global_scales.append(fp4_weights[f"{expert_id}.w2.weight_scale_2"])

    # Reinterpret uint8 (2 fp4/byte) as int64 (16 fp4/int64) for C++ kernel
    w3_w1_weight_fp4 = torch.stack(w3_w1_list, dim=0).view(torch.int64)
    w2_weight_fp4 = torch.stack(w2_list, dim=0).view(torch.int64)

    # Stack block scales: [E, rows, cols] in float8_e4m3fn
    # Then apply block_scale_interleave + view as int32 to match C++ kernel format.
    # Production code does this in NVFP4CutlassFusedMoEMethod.load_expert_w*_weight_scale_nvfp4.
    fc1_weight_block_raw = torch.stack(fc1_weight_block_scales, dim=0)
    fc2_weight_block_raw = torch.stack(fc2_weight_block_scales, dim=0)
    fc1_weight_block = _interleave_block_scales(fc1_weight_block_raw)
    fc2_weight_block = _interleave_block_scales(fc2_weight_block_raw)

    # Global scales: [E] tensor
    def _to_tensor(vals):
        if isinstance(vals[0], torch.Tensor):
            return torch.stack([v.squeeze() for v in vals])
        return torch.tensor(vals, dtype=torch.float32, device="cuda")

    fc1_act_global = _to_tensor(fc1_act_global_scales)
    fc2_act_global = _to_tensor(fc2_act_global_scales)
    fc1_global = _to_tensor(fc1_global_scales)
    fc2_global = _to_tensor(fc2_global_scales)

    fp4_quant_scales = [
        fc1_act_global,
        fc1_weight_block,
        fc1_global,
        fc2_act_global,
        fc2_weight_block,
        fc2_global,
    ]

    return w3_w1_weight_fp4, w2_weight_fp4, fp4_quant_scales


# ---------------------------------------------------------------------------
# Reference backend creation (pure bf16 CutlassFusedMoE)
# ---------------------------------------------------------------------------


def create_bf16_reference_backend(
    routing_method: RenormalizeMoeRoutingMethod,
    mapping: Mapping,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    weights: Dict[str, torch.Tensor],
) -> CutlassFusedMoE:
    """Create a pure bf16 CutlassFusedMoE backend as reference.

    This serves as ground truth: all experts run in bf16.
    """
    import os

    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe.create_moe import (
        create_moe_backend,
    )

    os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        moe_backend="CUTLASS",
    )

    backend = create_moe_backend(
        moe_cls=CutlassFusedMoE,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
    )
    backend.load_weights([copy.deepcopy(weights)])
    backend.post_load_weights()
    backend.cuda()
    return backend


def create_nvfp4_reference_backend(
    routing_method: RenormalizeMoeRoutingMethod,
    mapping: Mapping,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    fp4_weights: Dict[str, torch.Tensor],
) -> CutlassFusedMoE:
    """Create a pure NVFP4 CutlassFusedMoE backend as reference.

    This serves as ground truth for the all-fp4 case: all experts run in NVFP4
    through the standard CUTLASS NVFP4 path.
    """
    import os

    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe.create_moe import (
        create_moe_backend,
    )

    os.environ["ENABLE_CONFIGURABLE_MOE"] = "0"

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="CUTLASS",
    )

    backend = create_moe_backend(
        moe_cls=CutlassFusedMoE,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
    )
    backend.load_weights([copy.deepcopy(fp4_weights)])
    backend.post_load_weights()
    backend.cuda()
    return backend


# ---------------------------------------------------------------------------
# FusedMixedPrecisionMoE creation
# ---------------------------------------------------------------------------


def create_fused_mixed_precision_module(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_high_precision_experts: int,
    bf16_weights: Dict[str, torch.Tensor],
    fp4_weights: Dict[str, torch.Tensor],
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> FusedMixedPrecisionMoE:
    """Create a FusedMixedPrecisionMoE module with loaded bf16 + fp4 weights."""
    module = FusedMixedPrecisionMoE(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_high_precision_experts=num_high_precision_experts,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
    )

    # Pack and load bf16 weights
    w3_w1_bf16, w2_bf16 = pack_bf16_weights_for_fused_module(
        bf16_weights, num_experts, hidden_size, intermediate_size
    )
    module.load_bf16_weights(w3_w1_bf16, w2_bf16)

    # Pack and load fp4 weights
    w3_w1_fp4, w2_fp4, fp4_scales = pack_fp4_weights_for_fused_module(
        fp4_weights, num_experts, hidden_size, intermediate_size
    )
    module.load_fp4_weights(w3_w1_fp4, w2_fp4, fp4_scales)

    module.cuda()
    return module


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------


def run_reference_forward(
    backend: CutlassFusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: list,
) -> torch.Tensor:
    """Run forward pass on the bf16 reference CutlassFusedMoE backend."""
    with torch.inference_mode():
        return backend.forward(
            x,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
        )


def run_mixed_precision_forward(
    module: FusedMixedPrecisionMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    routing_method: Optional[RenormalizeMoeRoutingMethod] = None,
) -> torch.Tensor:
    """Run forward pass on the FusedMixedPrecisionMoE module."""
    with torch.inference_mode():
        return module.forward(
            x,
            router_logits,
            routing_method=routing_method,
        )


# ---------------------------------------------------------------------------
# L2 cache flush utility
# ---------------------------------------------------------------------------


def flush_l2_cache(size_mb: int = 40):
    """Flush GPU L2 cache by writing a large buffer.

    This is critical for accurate benchmarking to avoid cache effects
    from previous runs.
    """
    size_bytes = size_mb * 1024 * 1024
    flush_buf = torch.empty(size_bytes // 4, dtype=torch.float32, device="cuda")
    flush_buf.fill_(1.0)
    torch.cuda.synchronize()
    del flush_buf


# ---------------------------------------------------------------------------
# CUDA graph wrapper
# ---------------------------------------------------------------------------


def run_with_cuda_graph(
    fn,
    *args,
    warmup_iters: int = 3,
    graph_iters: int = 10,
) -> torch.Tensor:
    """Execute a function using CUDA graphs for reproducible benchmarking.

    Captures the function into a CUDA graph after warmup, then replays it.
    This verifies that the kernel is CUDA graph compatible (no per-call mallocs,
    no host-device sync, all buffers preallocated).

    Args:
        fn: Callable that returns a tensor.
        *args: Arguments passed to fn.
        warmup_iters: Number of warmup iterations before capture.
        graph_iters: Number of graph replay iterations.

    Returns:
        The output tensor from the last graph replay.
    """
    # Warmup
    for _ in range(warmup_iters):
        output = fn(*args)
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(*args)
    torch.cuda.synchronize()

    # Replay
    for _ in range(graph_iters):
        graph.replay()
    torch.cuda.synchronize()

    return output

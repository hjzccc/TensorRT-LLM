# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
NEW: Fused Mixed-Precision MoE Module.

Handles bf16 (hot experts) and nvfp4 (cold experts) in a single C++ kernel call,
replacing the separate per-group fused_moe() calls used by HeterCutlassFusedMoE.

Key benefits over HeterCutlassFusedMoE:
- One routing/sorting pass (vs. N separate calls)
- Fused activation across both precision groups
- Minimized quantization overhead (co-located with data movement)
- Full CUDA graph, PDL, and torch.compile compatibility

Usage:
    module = FusedMixedPrecisionMoE(
        num_experts=64,
        hidden_size=7168,
        intermediate_size=2048,
        top_k=8,
        num_high_precision_experts=12,
        tp_size=1, tp_rank=0,
        ep_size=1, ep_rank=0,
    )
    # Load bf16 weights
    module.load_bf16_weights(fc1_bf16, fc2_bf16)
    # Load fp4 weights + quant scales
    module.load_fp4_weights(fc1_fp4, fc2_fp4, fp4_quant_scales)
    # Forward
    output = module(hidden_states, router_logits)
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from ...utils import ActivationType
from .ops.moe_op_cutlass import CutlassMixedPrecisionMoEOp
from .routing import BaseMoeRoutingMethod


class FusedMixedPrecisionMoE(nn.Module):
    """Fused mixed-precision MoE: bf16 (hot) + nvfp4 (cold) experts in one kernel.

    This module stores dual weight sets and delegates computation to the
    CutlassMixedPrecisionMoEOp which calls torch.ops.trtllm.fused_moe_mixed_precision.

    Attributes:
        num_experts: Total number of experts (all must have both bf16 and fp4 weights).
        hidden_size: Hidden dimension.
        intermediate_size: Intermediate (feedforward) dimension per expert.
        top_k: Number of experts selected per token.
        num_high_precision_experts: How many top-loaded experts get bf16 precision.
        tp_size, tp_rank: Tensor parallelism config.
        ep_size, ep_rank: Expert parallelism config.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        num_high_precision_experts: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        use_fused_finalize: bool = True,
        activation_type: ActivationType = ActivationType.Swiglu,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.num_high_precision_experts = num_high_precision_experts
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.use_fused_finalize = use_fused_finalize
        self.activation_type = int(activation_type)
        self.dtype = dtype

        # Number of experts on this rank (for EP)
        self.num_experts_on_rank = num_experts // ep_size

        # Intermediate size per partition (for TP)
        self.intermediate_size_per_partition = intermediate_size // tp_size

        # Activation params (can be set after construction)
        self.swiglu_alpha = None
        self.swiglu_beta = None
        self.swiglu_limit = None

        # Op instance
        self._op = CutlassMixedPrecisionMoEOp()

        # Weight buffers (registered as buffers for proper device placement)
        # bf16 group weights
        self.register_buffer(
            'w3_w1_weight_bf16',
            torch.empty(
                self.num_experts_on_rank,
                self.intermediate_size_per_partition * 2,  # gated (w3+w1 fused)
                self.hidden_size,
                dtype=torch.bfloat16,
            ),
        )
        self.register_buffer(
            'w2_weight_bf16',
            torch.empty(
                self.num_experts_on_rank,
                self.hidden_size,
                self.intermediate_size_per_partition,
                dtype=torch.bfloat16,
            ),
        )

        # fp4 group weights (packed as int64: 16 fp4 elements per int64)
        fp4_pack_factor = 16  # 16 nvfp4 elements packed into 1 int64
        self.register_buffer(
            'w3_w1_weight_fp4',
            torch.empty(
                self.num_experts_on_rank,
                self.intermediate_size_per_partition * 2,
                self.hidden_size // fp4_pack_factor,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            'w2_weight_fp4',
            torch.empty(
                self.num_experts_on_rank,
                self.hidden_size,
                self.intermediate_size_per_partition // fp4_pack_factor,
                dtype=torch.int64,
            ),
        )

        # fp4 quant scales (6 tensors for NVFP4)
        # These will be set via load_fp4_quant_scales()
        self._fp4_quant_scales: Optional[List[torch.Tensor]] = None

        # Optional biases
        self.w3_w1_bias: Optional[torch.Tensor] = None
        self.w2_bias: Optional[torch.Tensor] = None

    def load_bf16_weights(
        self,
        w3_w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w3_w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
    ):
        """Load bf16 group weights.

        Args:
            w3_w1_weight: [num_experts, inter_size*2, hidden_size] bf16
            w2_weight: [num_experts, hidden_size, inter_size] bf16
        """
        assert w3_w1_weight.dtype == torch.bfloat16
        assert w2_weight.dtype == torch.bfloat16
        self.w3_w1_weight_bf16.copy_(w3_w1_weight)
        self.w2_weight_bf16.copy_(w2_weight)
        if w3_w1_bias is not None:
            self.w3_w1_bias = w3_w1_bias
        if w2_bias is not None:
            self.w2_bias = w2_bias

    def load_fp4_weights(
        self,
        w3_w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        fp4_quant_scales: List[torch.Tensor],
    ):
        """Load fp4 group weights and quant scales.

        Args:
            w3_w1_weight: [num_experts, inter_size*2, hidden_size/16] int64 packed nvfp4
            w2_weight: [num_experts, hidden_size, inter_size/16] int64 packed nvfp4
            fp4_quant_scales: 6 tensors for NVFP4:
                [fc1_act_global, fc1_weight_block, fc1_global,
                 fc2_act_global, fc2_weight_block, fc2_global]
        """
        assert w3_w1_weight.dtype == torch.int64
        assert w2_weight.dtype == torch.int64
        assert len(fp4_quant_scales) == 6
        self.w3_w1_weight_fp4.copy_(w3_w1_weight)
        self.w2_weight_fp4.copy_(w2_weight)
        self._fp4_quant_scales = fp4_quant_scales

    @property
    def fp4_quant_scales(self) -> List[torch.Tensor]:
        """Get fp4 quant scales, ensuring they are set."""
        assert self._fp4_quant_scales is not None, (
            "fp4 quant scales not loaded. Call load_fp4_weights() first."
        )
        return self._fp4_quant_scales

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        routing_method: Optional[BaseMoeRoutingMethod] = None,
        out_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: route tokens, then run fused mixed-precision MoE.

        Args:
            hidden_states: [num_tokens, hidden_size] bf16 input activations
            router_logits: [num_tokens, num_experts] float32 router logits
            routing_method: Optional routing method (topk selection + scale).
                If None, uses default top-k routing.
            out_tensor: Optional pre-allocated output tensor.

        Returns:
            [num_tokens, hidden_size] bf16 output
        """
        # --- Routing ---
        if routing_method is not None:
            token_selected_experts, token_final_scales = routing_method(
                router_logits
            )
        else:
            # Default top-k routing
            topk_weights, topk_ids = torch.topk(
                router_logits.float().softmax(dim=-1),
                self.top_k,
                dim=-1,
            )
            token_selected_experts = topk_ids.to(torch.int32)
            token_final_scales = topk_weights.to(torch.float32)

        # --- Call fused mixed-precision MoE ---
        # Create a lightweight proxy to pass module attributes to the op
        proxy = _ModuleProxy(self)

        output = self._op.compute_mixed_precision_moe(
            module=proxy,
            x=hidden_states,
            token_selected_slots=token_selected_experts,
            token_final_scales=token_final_scales,
            fc1_expert_weights_bf16=self.w3_w1_weight_bf16,
            fc2_expert_weights_bf16=self.w2_weight_bf16,
            fc1_expert_weights_fp4=self.w3_w1_weight_fp4,
            fc2_expert_weights_fp4=self.w2_weight_fp4,
            fc1_expert_biases=self.w3_w1_bias,
            fc2_expert_biases=self.w2_bias,
            fp4_quant_scales=self.fp4_quant_scales,
            num_high_precision_experts=self.num_high_precision_experts,
            use_fused_finalize=self.use_fused_finalize,
            unpadded_hidden_size=self.hidden_size,
            out_tensor=out_tensor,
        )

        return output


class _ModuleProxy:
    """Lightweight proxy to pass module attributes to CutlassMixedPrecisionMoEOp.

    The op expects a module-like object with tp_size, tp_rank, ep_size, ep_rank,
    swiglu_alpha, swiglu_beta, swiglu_limit, and activation_type attributes.
    """

    def __init__(self, module: FusedMixedPrecisionMoE):
        self.tp_size = module.tp_size
        self.tp_rank = module.tp_rank
        self.ep_size = module.ep_size
        self.ep_rank = module.ep_rank
        self.swiglu_alpha = module.swiglu_alpha
        self.swiglu_beta = module.swiglu_beta
        self.swiglu_limit = module.swiglu_limit
        self.activation_type = module.activation_type

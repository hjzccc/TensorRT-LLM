# Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin Python wrapper around the C++ MixedPrecisionMoeRunner torch class.

This wrapper exposes the MixPrecisionMoEFCWrapper (bf16 + NVFP4 mixed-precision
MoE) to Python via torch custom classes. It always uses bf16 backbone.

Usage:
    runner = MixedPrecisionMoERunner()

    # Set per-expert precision: 1 = bf16, 0 = fp4
    assignment = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32,
                              device='cuda')
    runner.set_expert_precision_assignment(assignment, num_bf16_experts=2)

    # Set FP4 weights and NVFP4 quant scales (6 tensors)
    runner.set_fp4_weights(fp4_fc1, fp4_fc2, fp4_quant_scales)

    # Optionally set GEMM tactics (-1 = default first tactic)
    runner.set_tactic(bf16_gemm1_id=-1, bf16_gemm2_id=-1,
                      fp4_gemm1_id=-1, fp4_gemm2_id=-1)

    # Run MoE
    output = runner.run_moe(
        input=x,
        token_selected_experts=experts,
        token_final_scales=scales,
        fc1_expert_weights=bf16_fc1,  # bf16 weights
        fc1_expert_biases=None,
        fc2_expert_weights=bf16_fc2,  # bf16 weights
        fc2_expert_biases=None,
    )
"""

from typing import List, Optional

import torch


class MixedPrecisionMoERunner:
    """Python wrapper for the C++ MixedPrecisionMoeRunner torch class.

    Wraps MixPrecisionMoEFCWrapper which orchestrates bf16 and NVFP4 runners
    for mixed-precision MoE inference. The bf16 weights are passed through
    run_moe(); FP4 weights and quant params are set separately.

    Bare minimum: supports all-bf16 or all-fp4 expert assignment.
    Full mixed-precision pipeline (mixed sort/expand/gemm) is WIP.
    """

    def __init__(self):
        self._runner = torch.classes.trtllm.MixedPrecisionMoeRunner()

    def set_expert_precision_assignment(self, assignment: torch.Tensor,
                                        num_bf16_experts: int):
        """Set per-expert precision assignment.

        Args:
            assignment: 1D int32 CUDA tensor of shape [num_experts_on_rank].
                        assignment[i] = 1 means expert i uses bf16,
                        assignment[i] = 0 means expert i uses fp4.
            num_bf16_experts: Number of experts using bf16 precision.
        """
        self._runner.set_expert_precision_assignment(assignment,
                                                     num_bf16_experts)

    def set_fp4_weights(self, fp4_fc1_expert_weights: torch.Tensor,
                        fp4_fc2_expert_weights: torch.Tensor,
                        fp4_quant_scales: List[torch.Tensor]):
        """Set FP4 expert weights and NVFP4 quantization parameters.

        Args:
            fp4_fc1_expert_weights: 3D CUDA tensor (Long/INT64, packed FP4).
            fp4_fc2_expert_weights: 3D CUDA tensor (Long/INT64, packed FP4).
            fp4_quant_scales: List of 6 CUDA tensors for NVFP4 quant params:
                [0] fc1_act_global    (Float, scalar or 1D)
                [1] fc1_weight_block  (Int, 3D)
                [2] fc1_global        (Float, 1D)
                [3] fc2_act_global    (Float, scalar or 1D)
                [4] fc2_weight_block  (Int, 3D)
                [5] fc2_global        (Float, 1D)
        """
        self._runner.set_fp4_weights(fp4_fc1_expert_weights,
                                     fp4_fc2_expert_weights, fp4_quant_scales)

    def set_tactic(self,
                   bf16_gemm1_id: int = -1,
                   bf16_gemm2_id: int = -1,
                   fp4_gemm1_id: int = -1,
                   fp4_gemm2_id: int = -1):
        """Set GEMM tactics for both runners.

        Args:
            bf16_gemm1_id: Tactic index for bf16 GEMM1 (-1 = default).
            bf16_gemm2_id: Tactic index for bf16 GEMM2 (-1 = default).
            fp4_gemm1_id: Tactic index for fp4 GEMM1 (-1 = default).
            fp4_gemm2_id: Tactic index for fp4 GEMM2 (-1 = default).
        """
        self._runner.set_tactic(bf16_gemm1_id, bf16_gemm2_id, fp4_gemm1_id,
                                fp4_gemm2_id)

    def get_bf16_tactic_num(self, gemm_idx: int) -> int:
        """Get number of available tactics for bf16 runner.

        Args:
            gemm_idx: 1 for GEMM1, 2 for GEMM2.
        """
        return self._runner.get_bf16_tactic_num(gemm_idx)

    def get_fp4_tactic_num(self, gemm_idx: int) -> int:
        """Get number of available tactics for fp4 runner.

        Args:
            gemm_idx: 1 for GEMM1, 2 for GEMM2.
        """
        return self._runner.get_fp4_tactic_num(gemm_idx)

    def run_moe(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor] = None,
        fc1_expert_weights: Optional[torch.Tensor] = None,
        fc1_expert_biases: Optional[torch.Tensor] = None,
        fc2_expert_weights: Optional[torch.Tensor] = None,
        fc2_expert_biases: Optional[torch.Tensor] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        enable_alltoall: bool = False,
        activation_type: Optional[int] = None,
        unpadded_hidden_size: Optional[int] = None,
        num_valid_tokens: Optional[int] = None,
        out_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run mixed-precision MoE forward pass.

        bf16 weights are passed here directly. FP4 weights must be set via
        set_fp4_weights() before calling this method.

        Args:
            input: 2D bf16 CUDA tensor [num_tokens, hidden_size].
            token_selected_experts: 2D int32 [num_tokens, experts_per_token].
            token_final_scales: Optional 2D float32 router scales.
            fc1_expert_weights: 3D bf16 [num_experts, inter_size*2, hidden_size].
            fc1_expert_biases: Optional 2D bf16 biases.
            fc2_expert_weights: 3D bf16 [num_experts, hidden_size, inter_size].
            fc2_expert_biases: Optional 2D bf16 biases.
            tp_size: Tensor parallel size.
            tp_rank: Tensor parallel rank.
            ep_size: Expert parallel size.
            ep_rank: Expert parallel rank.
            enable_alltoall: Enable alltoall communication.
            activation_type: Activation type enum value (default: Swiglu).
            unpadded_hidden_size: Unpadded hidden size for the output.
            num_valid_tokens: Number of valid tokens (for padding).
            out_tensor: Pre-allocated output tensor.

        Returns:
            Output tensor of shape [num_tokens, hidden_size] in bf16.
        """
        return self._runner.run_moe(input, token_selected_experts,
                                    token_final_scales, fc1_expert_weights,
                                    fc1_expert_biases, fc2_expert_weights,
                                    fc2_expert_biases, tp_size, tp_rank,
                                    ep_size, ep_rank, enable_alltoall,
                                    activation_type, unpadded_hidden_size,
                                    num_valid_tokens, out_tensor)

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
"""Dispatch policies for heterogeneous precision MoE.

All policies inherit from :class:`BaseDispatchPolicy` and implement
:meth:`assign`, which determines the expert-to-group mapping based on
group size constraints and optional runtime signals.

Currently implemented:
    :class:`RandomDispatchPolicy` — assigns experts randomly respecting
    ``size_ratio``.

Planned (stubs):
    :class:`ConfidenceThresholdPolicy` — uses router confidence scores.
    :class:`ExpertLoadPolicy` — uses cumulative expert activation counts.
"""

import abc
import random
from typing import List, Optional

import torch

from .dispatch_plan import DispatchPlan


class BaseDispatchPolicy(abc.ABC):
    """Abstract base for expert dispatch policies.

    Subclasses implement :meth:`assign` to produce a
    :class:`DispatchPlan` that maps experts to precision groups.  The
    policy receives group size constraints (``size_ratio`` values from
    ``heter_config``) and optional runtime signals (router outputs).
    """

    @abc.abstractmethod
    def assign(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
    ) -> DispatchPlan:
        """Determine expert-to-group assignment.

        Args:
            num_experts: Total number of experts in the MoE layer.
            group_size_ratios: Target fraction of experts for each
                group.  Sums to 1.0.
                ``len(group_size_ratios)`` = number of groups.
            token_selected_experts: Router expert selections per token.
                Shape ``[num_tokens, top_k]``.  Optional — not all
                policies use runtime signals.
            token_final_scales: Routing weights per token.
                Shape ``[num_tokens, top_k]``.  Optional.

        Returns:
            A :class:`DispatchPlan` whose ``group_assignments`` has the
            same length as *group_size_ratios*.
        """
        ...


class RandomDispatchPolicy(BaseDispatchPolicy):
    """Randomly assign experts to groups respecting ``size_ratio``.

    The assignment is deterministic when *seed* is set (produces the
    same :class:`DispatchPlan` every call).  Ignores runtime signals.

    Args:
        seed: Random seed for reproducibility.  If ``None``,
            assignment is random each call.
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    def assign(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
    ) -> DispatchPlan:
        rng = random.Random(self._seed)
        all_ids = list(range(num_experts))
        rng.shuffle(all_ids)

        assignments: List[List[int]] = []
        offset = 0
        for i, ratio in enumerate(group_size_ratios):
            if i == len(group_size_ratios) - 1:
                # Last group gets all remaining experts
                count = num_experts - offset
            else:
                count = round(ratio * num_experts)
            assignments.append(sorted(all_ids[offset:offset + count]))
            offset += count

        return DispatchPlan(group_assignments=assignments)


class ConfidenceThresholdPolicy(BaseDispatchPolicy):
    """Assign precision based on router confidence scores.

    Experts receiving high-confidence routing weights are assigned to
    the high-precision group (accuracy matters).  Low-confidence
    selections go to the low-precision group (less impact on quality,
    benefit from speed).

    Not yet implemented — raises :class:`NotImplementedError`.

    Args:
        confidence_threshold: Experts with mean routing weight above
            this threshold are assigned to the high-precision group.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self._confidence_threshold = confidence_threshold

    def assign(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
    ) -> DispatchPlan:
        raise NotImplementedError(
            "ConfidenceThresholdPolicy requires router signals.  "
            "Not yet implemented."
        )


class ExpertLoadPolicy(BaseDispatchPolicy):
    """Assign precision based on cumulative expert activation frequency.

    Hot experts (frequently activated) are assigned to the
    high-precision group.  Cold experts (rarely activated) go to the
    low-precision group.

    Not yet implemented — raises :class:`NotImplementedError`.

    Args:
        hot_ratio: Fraction of experts to keep at high precision
            based on activation frequency.
    """

    def __init__(self, hot_ratio: float = 0.2):
        self._hot_ratio = hot_ratio

    def assign(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
    ) -> DispatchPlan:
        raise NotImplementedError(
            "ExpertLoadPolicy requires accumulated expert load counts.  "
            "Not yet implemented."
        )

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

**CUDA graph / torch.compile notes**:

Policy ``assign()`` runs inside ``moe_custom_op`` (eager, not traced
by Dynamo).  Under CUDA graph capture the Python logic executes
normally; during replay it does **not** re-execute — the dispatch
assignment is baked.

Working tensors in signal-based policies are **pre-allocated** on
first call and reused via in-place ops (``zero_()``, ``div_()``,
``scatter_add_``).  This avoids per-call ``torch.zeros`` / ``torch.ones``
allocations that would otherwise be recorded as unnecessary CUDA ops
in a captured graph.
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

    Signals available:
        ``token_selected_experts``
            Expert IDs selected by the router per token.
            Shape ``[num_tokens, top_k]``, dtype ``int32``.

        ``token_final_scales``
            Routing weights (post-softmax / post-sigmoid) per token.
            Shape ``[num_tokens, top_k]``, dtype ``float32``.

        ``router_logits``
            Raw logits from the router gate **before** top-k / softmax.
            Shape ``[num_tokens, num_experts]``.  Available when the
            call originates from ``forward_chunk()``; ``None`` when
            ``run_moe()`` is called directly (e.g. tests).
    """

    @abc.abstractmethod
    def assign(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
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
            router_logits: Raw router logits (pre-softmax / pre-topk).
                Shape ``[num_tokens, num_experts]``.  Optional — only
                available when called from ``forward_chunk()``.

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
        router_logits: Optional[torch.Tensor] = None,
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


# ------------------------------------------------------------------
# Score-based helper used by signal-driven policies
# ------------------------------------------------------------------


def _compute_group_sizes(
    num_experts: int,
    group_size_ratios: List[float],
) -> List[int]:
    """Convert fractional ratios to concrete group sizes.

    Last group absorbs rounding remainder.
    """
    sizes: List[int] = []
    offset = 0
    for i, ratio in enumerate(group_size_ratios):
        if i == len(group_size_ratios) - 1:
            sizes.append(num_experts - offset)
        else:
            count = round(ratio * num_experts)
            sizes.append(count)
            offset += count
    return sizes


def _assign_by_score(
    scores: torch.Tensor,
    num_experts: int,
    group_size_ratios: List[float],
) -> DispatchPlan:
    """Split experts into groups by descending score.

    The **last** group gets the highest-scoring experts (number
    determined by last group's ``size_ratio``), the second-to-last
    gets the next batch, and so on.  This means the first group
    (conventionally low-precision) gets the lowest-scoring experts.

    ``torch.argsort`` is the only allocation (``num_experts``-sized,
    tiny).  The ``.tolist()`` sync is acceptable because this only
    runs during CUDA graph capture, not replay.
    """
    sorted_ids = torch.argsort(scores, descending=True).tolist()
    group_sizes = _compute_group_sizes(num_experts, group_size_ratios)

    # The last group takes the top-scoring experts.
    assignments: List[List[int]] = []
    cursor = 0
    for size in reversed(group_sizes):
        chunk = sorted(sorted_ids[cursor:cursor + size])
        assignments.insert(0, chunk)
        cursor += size

    return DispatchPlan(group_assignments=assignments)


# ------------------------------------------------------------------
# Signal-based policies
# ------------------------------------------------------------------


class ConfidenceThresholdPolicy(BaseDispatchPolicy):
    """Assign precision based on per-expert mean routing weight.

    Experts whose mean ``token_final_scales`` (routing weight) across
    all tokens is **high** are considered important for accuracy and
    placed in the **last** group (conventionally the high-precision /
    BF16 group).  The remaining experts go to the **first** group
    (low-precision / NVFP4).

    The ``size_ratio`` constraints are respected: experts are ranked
    by mean routing weight and the group sizes determined by the
    ratios decide the split point.

    When signals are unavailable (``token_final_scales is None``),
    falls back to :class:`RandomDispatchPolicy` with *fallback_seed*.

    Working tensors are **pre-allocated** on first call and reused
    via in-place ops to avoid per-call allocations.

    Args:
        confidence_threshold: Reserved for future per-token gating
            (not used in the current per-expert ranking logic).
        fallback_seed: Seed for fallback random assignment.  Default 42.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        fallback_seed: int = 42,
    ):
        self._confidence_threshold = confidence_threshold
        self._fallback = RandomDispatchPolicy(seed=fallback_seed)
        # Lazily initialized working buffer (num_experts-sized, float32).
        # Reused across assign() calls via zero_() + scatter_add_.
        self._expert_weight_sum: Optional[torch.Tensor] = None

    def _ensure_buffers(
        self, num_experts: int, device: torch.device,
    ) -> None:
        """Allocate working tensors on first call; no-op afterwards."""
        if (self._expert_weight_sum is not None
                and self._expert_weight_sum.shape[0] == num_experts
                and self._expert_weight_sum.device == device):
            return
        self._expert_weight_sum = torch.empty(
            num_experts, device=device, dtype=torch.float32,
        )

    def assign(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
    ) -> DispatchPlan:
        # Fall back when signals are absent (e.g. post_load_weights).
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback.assign(
                num_experts, group_size_ratios,
            )

        self._ensure_buffers(num_experts, token_final_scales.device)

        flat_experts = token_selected_experts.reshape(-1).long()
        flat_scales = token_final_scales.reshape(-1)

        # Accumulate per-expert weight sum (in-place, pre-allocated).
        self._expert_weight_sum.zero_()
        self._expert_weight_sum.scatter_add_(0, flat_experts, flat_scales)

        # Count activations per expert — single fused op, replaces
        # scatter_add_ + torch.ones_like (which allocated a full
        # num_tokens*top_k ones tensor each call).
        expert_count = torch.bincount(
            flat_experts, minlength=num_experts,
        ).to(dtype=torch.float32)

        # Safe mean: clamp count to 1 so inactive experts (count=0)
        # get score 0 (0 / 1 = 0).  Replaces torch.where + zeros_like.
        expert_count.clamp_min_(1.0)
        self._expert_weight_sum.div_(expert_count)

        return _assign_by_score(
            self._expert_weight_sum, num_experts, group_size_ratios,
        )


class ExpertLoadPolicy(BaseDispatchPolicy):
    """Assign precision based on expert activation frequency.

    "Hot" experts (frequently selected by the router) are assigned to
    the **last** group (high-precision); "cold" experts go to the
    **first** group (low-precision).

    Frequency is computed per ``assign()`` call from
    ``token_selected_experts``.  For a smoothed view across multiple
    calls, wrap this policy with an EMA tracker (future work).

    When signals are unavailable, falls back to
    :class:`RandomDispatchPolicy`.

    Args:
        fallback_seed: Seed for fallback random assignment.  Default 42.
    """

    def __init__(self, fallback_seed: int = 42):
        self._fallback = RandomDispatchPolicy(seed=fallback_seed)

    def assign(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        token_selected_experts: Optional[torch.Tensor] = None,
        token_final_scales: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
    ) -> DispatchPlan:
        if token_selected_experts is None:
            return self._fallback.assign(
                num_experts, group_size_ratios,
            )

        # Single fused op — replaces torch.zeros + scatter_add_ + ones.
        flat_experts = token_selected_experts.reshape(-1).long()
        counts = torch.bincount(
            flat_experts, minlength=num_experts,
        ).to(dtype=torch.float32)

        return _assign_by_score(
            counts, num_experts, group_size_ratios,
        )

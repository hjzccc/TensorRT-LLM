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
"""Heterogeneous dispatch policies for mixed-precision MoE.

A :class:`HeterDispatchPolicy` partitions experts into precision groups
and transforms standard MoE routing into per-group dispatches.

Each group's dispatch is a plain ``(token_indices, experts, scales)``
tuple containing only the tokens that have at least one expert in that
group, with scales zeroed for non-group expert slots.

The policy knows nothing about quantization algorithms or weight
storage — that mapping lives in
:class:`~...fused_moe_heter.HeterCutlassFusedMoE`.

Usage::

    policy = RandomHeterDispatch(
        num_experts=128,
        group_size_ratios=[0.8, 0.2],
        seed=42,
    )

    dispatches = policy.dispatch(
        token_selected_experts,   # [num_tokens, top_k]
        token_final_scales,       # [num_tokens, top_k]
    )
    for group_idx, (tok_idx, experts, scales) in enumerate(dispatches):
        if tok_idx is None:
            continue
        result = fused_moe(x[tok_idx], experts, scales, ...)
        accumulated[tok_idx] += result
"""

import abc
import random
from collections import Counter
from typing import List, Optional, Tuple

import torch

# Type alias for a per-group dispatch result.
# (token_indices, experts, scales) or (None, None, None) when empty.
GroupDispatchTuple = Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
                          Optional[torch.Tensor]]


# ------------------------------------------------------------------
# Helpers
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
) -> List[List[int]]:
    """Split experts into groups by descending score.

    The **last** group gets the highest-scoring experts (number
    determined by last group's ``size_ratio``), the second-to-last
    gets the next batch, and so on.  This means the first group
    (conventionally low-precision) gets the lowest-scoring experts.

    The ``.tolist()`` sync is acceptable because this only runs during
    CUDA graph capture, not replay.
    """
    sorted_ids = torch.argsort(scores, descending=True).tolist()
    group_sizes = _compute_group_sizes(num_experts, group_size_ratios)

    assignments: List[List[int]] = []
    cursor = 0
    for size in reversed(group_sizes):
        chunk = sorted(sorted_ids[cursor:cursor + size])
        assignments.insert(0, chunk)
        cursor += size

    return assignments


def _validate_assignment(
    group_assignments: List[List[int]],
    num_experts: int,
) -> None:
    """Check that every expert is assigned to exactly one group."""
    all_ids: List[int] = []
    for ids in group_assignments:
        all_ids.extend(ids)

    all_set = set(all_ids)
    expected = set(range(num_experts))

    if all_set != expected:
        missing = sorted(expected - all_set)
        extra = sorted(all_set - expected)
        raise ValueError(
            f"Assignment coverage mismatch (num_experts={num_experts}).  "
            f"Missing: {missing}, Extra: {extra}")

    if len(all_ids) != len(all_set):
        dupes = sorted(
            eid for eid, cnt in Counter(all_ids).items() if cnt > 1)
        raise ValueError(f"Assignment has duplicate expert IDs: {dupes}")


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------


class HeterDispatchPolicy(abc.ABC):
    """Base class for heterogeneous MoE dispatch policies.

    Subclasses implement :meth:`_assign` — the expert-to-group
    assignment strategy.  The base class provides :meth:`dispatch`
    which calls ``_assign``, builds boolean masks, and splits routing
    into per-group ``(token_indices, experts, scales)`` tuples.

    Args:
        num_experts: Total number of experts in the MoE layer.
        group_size_ratios: Target fraction of experts for each group.
            Must sum to 1.0.  ``len(group_size_ratios)`` determines
            the number of groups.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
    ):
        self._num_experts = num_experts
        self._group_size_ratios = group_size_ratios

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_groups(self) -> int:
        return len(self._group_size_ratios)

    @property
    def group_size_ratios(self) -> List[float]:
        return self._group_size_ratios

    # ----------------------------------------------------------
    # Abstract: assignment strategy (subclasses implement this)
    # ----------------------------------------------------------

    @abc.abstractmethod
    def _assign(
        self,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
        router_logits: Optional[torch.Tensor],
    ) -> List[List[int]]:
        """Return ``group_assignments[i]`` = sorted list of expert IDs
        for group *i*.  Union must equal ``range(num_experts)``."""
        ...

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> List[GroupDispatchTuple]:
        """Transform N-expert routing into per-group dispatches.

        Returns a list of length ``num_groups``.  Each element is
        ``(token_indices, experts, scales)`` or
        ``(None, None, None)`` when no tokens are routed to that group.
        """
        assignment = self._assign(
            token_selected_experts,
            token_final_scales,
            router_logits,
        )
        if __debug__:
            _validate_assignment(assignment, self._num_experts)

        return self._dispatch_by_assignment(
            assignment, token_selected_experts, token_final_scales)

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------

    def _dispatch_by_assignment(
        self,
        group_assignments: List[List[int]],
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
    ) -> List[GroupDispatchTuple]:
        device = token_selected_experts.device
        results: List[GroupDispatchTuple] = []

        for expert_ids in group_assignments:
            mask = torch.zeros(
                self._num_experts, dtype=torch.bool, device=device)
            if expert_ids:
                ids = torch.tensor(expert_ids, dtype=torch.long, device=device)
                mask[ids] = True

            valid = mask[token_selected_experts.long()]
            token_in_group = valid.any(dim=1)
            indices = token_in_group.nonzero(as_tuple=False).squeeze(1)

            if indices.numel() == 0:
                results.append((None, None, None))
                continue

            sub_experts = token_selected_experts[indices]
            sub_scales = token_final_scales[indices]
            masked_scales = torch.where(
                valid[indices], sub_scales, sub_scales.new_zeros(1))

            results.append((indices, sub_experts, masked_scales))

        return results


# ------------------------------------------------------------------
# Concrete policies
# ------------------------------------------------------------------


class RandomHeterDispatch(HeterDispatchPolicy):
    """Random expert-to-group assignment.  Ignores runtime signals.

    Deterministic when *seed* is set.

    Args:
        num_experts: Total number of experts.
        group_size_ratios: Target fraction per group.
        seed: Random seed for reproducibility.  Default 42.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        seed: int = 42,
    ):
        super().__init__(num_experts, group_size_ratios)
        self._seed = seed

    def _assign(self, token_selected_experts, token_final_scales,
                router_logits):
        rng = random.Random(self._seed)
        all_ids = list(range(self._num_experts))
        rng.shuffle(all_ids)

        assignments: List[List[int]] = []
        offset = 0
        for i, ratio in enumerate(self._group_size_ratios):
            if i == len(self._group_size_ratios) - 1:
                count = self._num_experts - offset
            else:
                count = round(ratio * self._num_experts)
            assignments.append(sorted(all_ids[offset:offset + count]))
            offset += count

        return assignments


class ConfidenceThresholdHeterDispatch(HeterDispatchPolicy):
    """Assign by per-expert mean routing weight.

    High-weight experts go to the **last** group (high-precision);
    low-weight experts go to the **first** group (low-precision).
    Falls back to random when signals are unavailable.

    Args:
        num_experts: Total number of experts.
        group_size_ratios: Target fraction per group.
        confidence_threshold: Reserved for future per-token gating.
        fallback_seed: Seed for fallback random assignment.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        confidence_threshold: float = 0.5,
        fallback_seed: int = 42,
    ):
        super().__init__(num_experts, group_size_ratios)
        self._confidence_threshold = confidence_threshold
        self._fallback = RandomHeterDispatch(
            num_experts,
            group_size_ratios,
            seed=fallback_seed,
        )
        self._expert_weight_sum: Optional[torch.Tensor] = None

    def _ensure_buffers(self, num_experts: int,
                        device: torch.device) -> None:
        if (self._expert_weight_sum is not None
                and self._expert_weight_sum.shape[0] == num_experts
                and self._expert_weight_sum.device == device):
            return
        self._expert_weight_sum = torch.empty(
            num_experts,
            device=device,
            dtype=torch.float32,
        )

    def _assign(self, token_selected_experts, token_final_scales,
                router_logits):
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback._assign(
                token_selected_experts, token_final_scales, router_logits)

        self._ensure_buffers(self._num_experts,
                             token_final_scales.device)
        buf = self._expert_weight_sum
        assert buf is not None

        flat_experts = token_selected_experts.reshape(-1).long()
        flat_scales = token_final_scales.reshape(-1)

        buf.zero_()
        buf.scatter_add_(0, flat_experts, flat_scales)

        expert_count = torch.bincount(
            flat_experts,
            minlength=self._num_experts,
        ).to(dtype=torch.float32)
        expert_count.clamp_min_(1.0)
        buf.div_(expert_count)

        return _assign_by_score(
            buf,
            self._num_experts,
            self._group_size_ratios,
        )


class ExpertLoadHeterDispatch(HeterDispatchPolicy):
    """Assign by expert activation frequency.

    Frequently activated ("hot") experts go to the **last** group
    (high-precision); rarely activated ("cold") experts go to the
    **first** group (low-precision).
    Falls back to random when signals are unavailable.

    Args:
        num_experts: Total number of experts.
        group_size_ratios: Target fraction per group.
        fallback_seed: Seed for fallback random assignment.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        fallback_seed: int = 42,
    ):
        super().__init__(num_experts, group_size_ratios)
        self._fallback = RandomHeterDispatch(
            num_experts,
            group_size_ratios,
            seed=fallback_seed,
        )

    def _assign(self, token_selected_experts, token_final_scales,
                router_logits):
        if token_selected_experts is None:
            return self._fallback._assign(
                token_selected_experts, token_final_scales, router_logits)

        flat_experts = token_selected_experts.reshape(-1).long()
        counts = torch.bincount(
            flat_experts,
            minlength=self._num_experts,
        ).to(dtype=torch.float32)

        return _assign_by_score(
            counts,
            self._num_experts,
            self._group_size_ratios,
        )

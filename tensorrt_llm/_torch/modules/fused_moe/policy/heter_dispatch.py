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

Each group's dispatch is an ``(experts, scales)`` tuple with shape
``[N, K]`` — all tokens participate in every group.  Non-group expert
slots are filled with a sentinel expert ID (``num_experts``) and zero
scale so the CUTLASS kernel skips them.

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
    for group_idx, (experts, scales) in enumerate(dispatches):
        result = fused_moe(x, experts, scales, ...)
        accumulated += result

Implementation notes — torch.compile & CUDA graph safety:

* ``_assign()`` returns ``expert_to_group: torch.Tensor`` on GPU.
  No ``.tolist()`` sync, no ``List[List[int]]`` round-trip.
* ``dispatch()`` uses ``torch.where()`` for sentinel masking — all
  ops have fixed shapes, no ``nonzero()``, no dynamic outputs.
* 2-group fast path uses ``torch.topk()`` (O(E) partial sort).
* N-group fallback uses ``torch.argsort() + scatter_()`` on GPU.
"""

import abc
import random
from typing import List, Optional, Tuple

import torch

# Type alias for a per-group dispatch result: (experts, scales).
# Both tensors have shape [N, K].  Non-group expert slots use a sentinel
# expert ID (num_experts) and zero scale so the kernel skips them.
GroupDispatchTuple = Tuple[torch.Tensor, torch.Tensor]


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


def _assign_by_score_gpu(
    scores: torch.Tensor,
    num_experts: int,
    group_size_ratios: List[float],
) -> torch.Tensor:
    """GPU-only: split experts into groups by descending score.

    Returns ``expert_to_group`` tensor of shape ``[num_experts]`` on the
    same device as *scores*.  **Zero GPU-CPU synchronisation.**

    The **last** group gets the highest-scoring experts (high-precision);
    the **first** group gets the lowest-scoring experts (low-precision).

    For G=2 uses ``torch.topk`` (O(E) partial sort, 2 kernels).
    For G>2 uses ``torch.argsort + scatter_`` (O(E log E), 3 kernels).
    """
    num_groups = len(group_size_ratios)
    device = scores.device

    if num_groups == 2:
        # Fast path: single topk for the high-precision (last) group.
        k_high = round(num_experts * group_size_ratios[1])
        expert_to_group = torch.zeros(
            num_experts, dtype=torch.long, device=device)
        _, top_indices = torch.topk(scores, k_high)
        expert_to_group[top_indices] = 1
        return expert_to_group
    else:
        # General N-group path: argsort on GPU + scatter.
        group_sizes = _compute_group_sizes(num_experts, group_size_ratios)
        sorted_ids = torch.argsort(scores, descending=True)

        # Build position-to-group mapping.  Last group gets top positions.
        # This list is O(num_experts) ints built on CPU (no GPU sync).
        group_labels_list: List[int] = []
        for rev_idx, size in enumerate(reversed(group_sizes)):
            original_gidx = num_groups - 1 - rev_idx
            group_labels_list.extend([original_gidx] * size)

        group_labels = torch.tensor(
            group_labels_list, dtype=torch.long, device=device)

        expert_to_group = torch.empty(
            num_experts, dtype=torch.long, device=device)
        expert_to_group.scatter_(0, sorted_ids, group_labels)
        return expert_to_group


def _validate_expert_to_group(
    expert_to_group: torch.Tensor,
    num_experts: int,
    num_groups: int,
) -> None:
    """Debug-only validation for expert_to_group tensor.

    Checks shape and value range.  Uses ``.item()`` calls that sync
    with GPU — only runs under ``__debug__`` (``python -O`` disables).
    """
    assert expert_to_group.shape == (num_experts,), (
        f"expert_to_group shape {expert_to_group.shape} != ({num_experts},)")
    assert expert_to_group.min().item() >= 0, "expert_to_group has negative values"
    assert expert_to_group.max().item() < num_groups, (
        f"expert_to_group max {expert_to_group.max().item()} >= num_groups {num_groups}")


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------


class HeterDispatchPolicy(abc.ABC):
    """Base class for heterogeneous MoE dispatch policies.

    Subclasses implement :meth:`_assign` — the expert-to-group
    assignment strategy.  The base class provides :meth:`dispatch`
    which calls ``_assign``, then uses ``torch.where`` to build
    per-group ``(experts, scales)`` tuples with sentinel masking.

    All operations use fixed-shape tensors on GPU — no ``nonzero()``,
    no ``.tolist()``, fully compatible with ``torch.compile`` and
    CUDA graphs.

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
    ) -> torch.Tensor:
        """Return ``expert_to_group`` tensor of shape ``[num_experts]``
        on the same device as input tensors.  ``expert_to_group[e]``
        is the group index for expert *e*."""
        ...

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
    ) -> List[GroupDispatchTuple]:
        """Transform N-expert routing into per-group dispatches.

        Returns a list of length ``num_groups``.  Each element is
        ``(experts, scales)`` with shape ``[N, K]``.  Non-group expert
        slots are sentinel-masked (expert ID = ``num_experts``, scale = 0).

        Uses ``torch.where`` for sentinel masking — all ops have
        fixed shapes, fully torch.compile and CUDA graph safe.
        """
        expert_to_group = self._assign(
            token_selected_experts,
            token_final_scales,
        )
        if __debug__:
            _validate_expert_to_group(
                expert_to_group, self._num_experts, self.num_groups)

        return self._dispatch_from_expert_to_group(
            expert_to_group, token_selected_experts, token_final_scales)

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------

    def _dispatch_from_expert_to_group(
        self,
        expert_to_group: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
    ) -> List[GroupDispatchTuple]:
        """Build per-group dispatch tuples using torch.where.

        All N tokens are sent to every group.  Non-group expert slots
        get sentinel expert ID (``num_experts``) and zero scale.  The
        CUTLASS kernel skips sentinel slots (no GEMM, no weight loads).

        No ``nonzero()``, no ``clone()``, no dynamic shapes.
        """
        num_groups = self.num_groups

        # [N, K] — which group each expert slot belongs to.
        slot_groups = expert_to_group[token_selected_experts.long()]

        results: List[GroupDispatchTuple] = []
        for gidx in range(num_groups):
            in_group = (slot_groups == gidx)  # [N, K], bool

            # torch.where: fixed shape [N, K], no dynamic output.
            experts_g = torch.where(
                in_group, token_selected_experts, self._num_experts)
            scales_g = torch.where(
                in_group, token_final_scales, 0.0)

            results.append((experts_g, scales_g))

        return results


# ------------------------------------------------------------------
# Concrete policies
# ------------------------------------------------------------------


class RandomHeterDispatch(HeterDispatchPolicy):
    """Random expert-to-group assignment.  Ignores runtime signals.

    Deterministic when *seed* is set.  The assignment is computed once
    and cached — subsequent calls return the same GPU tensor.

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
        self._cached_e2g: Optional[torch.Tensor] = None

    def _assign(self, token_selected_experts, token_final_scales):
        # Determine target device.
        if token_selected_experts is not None:
            device = token_selected_experts.device
        else:
            device = torch.device('cuda:0')

        # Return cached tensor if device matches.
        if (self._cached_e2g is not None
                and self._cached_e2g.device == device):
            return self._cached_e2g

        # Build assignment on CPU (O(num_experts), runs once).
        rng = random.Random(self._seed)
        all_ids = list(range(self._num_experts))
        rng.shuffle(all_ids)

        group_sizes = _compute_group_sizes(
            self._num_experts, self._group_size_ratios)
        e2g_list = [0] * self._num_experts
        offset = 0
        for gidx, size in enumerate(group_sizes):
            for j in range(offset, offset + size):
                e2g_list[all_ids[j]] = gidx
            offset += size

        self._cached_e2g = torch.tensor(
            e2g_list, dtype=torch.long, device=device)
        return self._cached_e2g


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

    def _assign(self, token_selected_experts, token_final_scales):
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback._assign(
                token_selected_experts, token_final_scales)

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

        return _assign_by_score_gpu(
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

    Uses ``torch.topk`` for 2-group assignment (O(E) partial sort,
    2 kernel launches, zero GPU-CPU sync).

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

    def _assign(self, token_selected_experts, token_final_scales):
        if token_selected_experts is None:
            return self._fallback._assign(
                token_selected_experts, token_final_scales)

        flat_experts = token_selected_experts.reshape(-1).long()
        counts = torch.bincount(
            flat_experts,
            minlength=self._num_experts,
        ).to(dtype=torch.float32)

        return _assign_by_score_gpu(
            counts,
            self._num_experts,
            self._group_size_ratios,
        )

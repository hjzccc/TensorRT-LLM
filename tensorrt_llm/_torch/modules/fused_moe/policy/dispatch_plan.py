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
"""Dispatch plan for heterogeneous precision MoE.

A :class:`DispatchPlan` maps each expert to a precision group.  It is
produced by a dispatch policy (see ``strategies.py``) and consumed by
:class:`~tensorrt_llm._torch.modules.fused_moe.fused_moe_heter.HeterCutlassFusedMoE`
to split execution into per-group GEMM calls.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass
class DispatchPlan:
    """Maps experts to precision groups.

    Attributes:
        group_assignments: ``group_assignments[i]`` is a sorted list
            of expert IDs assigned to group *i*.  The union of all
            groups must equal ``range(num_experts)`` exactly — every
            expert appears in exactly one group.
    """

    group_assignments: List[List[int]]

    @property
    def num_groups(self) -> int:
        return len(self.group_assignments)

    def validate(self, num_experts: int) -> None:
        """Check that every expert is assigned to exactly one group.

        Raises:
            ValueError: On coverage mismatch or duplicate assignments.
        """
        all_ids: List[int] = []
        for ids in self.group_assignments:
            all_ids.extend(ids)

        all_set = set(all_ids)
        expected = set(range(num_experts))

        if all_set != expected:
            missing = sorted(expected - all_set)
            extra = sorted(all_set - expected)
            raise ValueError(
                f"DispatchPlan coverage mismatch "
                f"(num_experts={num_experts}).  "
                f"Missing: {missing}, Extra: {extra}"
            )

        if len(all_ids) != len(all_set):
            dupes = sorted(
                eid for eid, cnt in Counter(all_ids).items() if cnt > 1
            )
            raise ValueError(
                f"DispatchPlan has duplicate expert IDs: {dupes}"
            )

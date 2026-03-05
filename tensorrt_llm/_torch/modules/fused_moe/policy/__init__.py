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

from typing import Any, Dict, List

from .heter_dispatch import (
    ConfidenceThresholdHeterDispatch,
    ExpertLoadHeterDispatch,
    GroupDispatchTuple,
    HeterDispatchPolicy,
    RandomHeterDispatch,
    _validate_expert_to_group,
)

POLICY_REGISTRY: Dict[str, type] = {
    "random": RandomHeterDispatch,
    "confidence_threshold": ConfidenceThresholdHeterDispatch,
    "expert_load": ExpertLoadHeterDispatch,
}


def resolve_dispatch_policy(
    heter_config: Dict[str, Any],
    num_experts: int,
    group_size_ratios: List[float],
) -> HeterDispatchPolicy:
    """Instantiate a dispatch policy from ``heter_config["policy"]``.

    Accepts a string (policy name) or dict ``{"type": "<name>", ...}``
    with extra constructor kwargs.  Defaults to ``"expert_load"``.
    """
    policy_cfg = heter_config.get("policy", "expert_load")

    if isinstance(policy_cfg, str):
        policy_type = policy_cfg
        policy_kwargs: Dict[str, Any] = {}
    elif isinstance(policy_cfg, dict):
        policy_type = policy_cfg.get("type", "expert_load")
        policy_kwargs = {
            k: v for k, v in policy_cfg.items() if k != "type"
        }
    else:
        raise ValueError(
            f"heter_config['policy'] must be a string or dict, "
            f"got {type(policy_cfg).__name__}: {policy_cfg!r}"
        )

    policy_cls = POLICY_REGISTRY.get(policy_type)
    if policy_cls is None:
        raise ValueError(
            f"Unknown dispatch policy type: {policy_type!r}.  "
            f"Available: {sorted(POLICY_REGISTRY.keys())}"
        )

    return policy_cls(
        num_experts=num_experts,
        group_size_ratios=group_size_ratios,
        **policy_kwargs,
    )


__all__ = [
    "ConfidenceThresholdHeterDispatch",
    "ExpertLoadHeterDispatch",
    "GroupDispatchTuple",
    "HeterDispatchPolicy",
    "POLICY_REGISTRY",
    "RandomHeterDispatch",
    "resolve_dispatch_policy",
]

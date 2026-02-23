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
"""Config validation and dispatch-policy tests for HeterCutlassFusedMoE."""

import copy

import pytest
import torch
from _torch.modules.moe.heter_moe_utils import (
    NVFP4_UNAVAILABLE_REASON,
    create_backend,
    create_model_config,
    create_unquantized_weights,
    nvfp4_supported,
    single_group_config,
    two_bf16_group_config,
    two_group_config,
)

from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.fused_moe_heter import HeterCutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.policy import (
    ConfidenceThresholdHeterDispatch,
    ExpertLoadHeterDispatch,
    HeterDispatchPolicy,
    RandomHeterDispatch,
)
from tensorrt_llm._torch.modules.fused_moe.policy.heter_dispatch import (
    _validate_expert_to_group,
)
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestHeterConfigValidation:
    """Tests for heter_config validation in HeterCutlassFusedMoE.__init__."""

    NUM_EXPERTS = 8
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 512
    DTYPE = torch.bfloat16

    def _try_create(self, heter_config):
        mapping = Mapping()
        mapping.rank = mpi_rank()
        routing_method = RenormalizeMoeRoutingMethod(top_k=2)

        model_config = create_model_config(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            moe_backend="HETER",
            mapping=mapping,
            heter_config=heter_config,
        )
        with torch.device(f"cuda:{mapping.rank}"):
            return create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=model_config,
            )

    def test_missing_heter_config(self):
        with pytest.raises(ValueError, match="heter_moe_config"):
            self._try_create(heter_config=None)

    def test_empty_groups(self):
        with pytest.raises(ValueError, match="non-empty list"):
            self._try_create({"groups": []})

    def test_missing_size_ratio(self):
        with pytest.raises(ValueError, match="size_ratio is required"):
            self._try_create({
                "groups": [
                    {
                        "name": "bad_group",
                        "quant_algo": None,
                        "checkpoint": None,
                    },
                ],
            })

    @pytest.mark.parametrize("bad_ratio", [0.0, -0.1, 1.1])
    def test_invalid_size_ratio(self, bad_ratio):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
            self._try_create({
                "groups": [
                    {
                        "name": "bad_ratio",
                        "quant_algo": None,
                        "size_ratio": bad_ratio,
                        "checkpoint": None,
                    },
                ],
            })

    def test_size_ratio_sum_mismatch(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            self._try_create({
                "groups": [
                    {
                        "name": "g0",
                        "quant_algo": None,
                        "size_ratio": 0.4,
                        "checkpoint": None,
                    },
                    {
                        "name": "g1",
                        "quant_algo": None,
                        "size_ratio": 0.4,
                        "checkpoint": None,
                    },
                ],
            })

    def test_invalid_quant_algo(self):
        with pytest.raises(ValueError, match="Unsupported quant_algo value"):
            self._try_create({
                "groups": [
                    {
                        "name": "bad_quant",
                        "quant_algo": "NOT_A_REAL_ALGO",
                        "size_ratio": 1.0,
                        "checkpoint": None,
                    },
                ],
            })

    def test_checkpoint_not_exists(self, tmp_path):
        missing_ckpt = tmp_path / "not_found"
        with pytest.raises(ValueError, match="does not exist"):
            self._try_create({
                "groups": [
                    {
                        "name": "all_bf16",
                        "quant_algo": None,
                        "size_ratio": 1.0,
                        "checkpoint": str(missing_ckpt),
                    },
                ],
            })

    def test_valid_single_group(self):
        backend = self._try_create(single_group_config())
        assert isinstance(backend, HeterCutlassFusedMoE)

    def test_valid_two_groups(self):
        if not nvfp4_supported(self.DTYPE):
            pytest.skip(NVFP4_UNAVAILABLE_REASON)
        backend = self._try_create(two_group_config())
        assert isinstance(backend, HeterCutlassFusedMoE)


# ---------------------------------------------------------------------------
# Tests: dispatch policy
# ---------------------------------------------------------------------------


class TestDispatchPolicy:
    NUM_EXPERTS = 8
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 512
    DTYPE = torch.bfloat16
    SEQ_LEN = 8
    TOP_K = 2

    def _create_backend_with_weights(self, heter_config, *, seq_len=None,
                                     top_k=None):
        """Create a HeterCutlassFusedMoE backend with BF16 weights."""
        seq_len = seq_len if seq_len is not None else self.SEQ_LEN
        top_k = top_k if top_k is not None else self.TOP_K
        mapping = Mapping()
        mapping.rank = mpi_rank()
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        weights = create_unquantized_weights(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
        )

        model_config = create_model_config(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            dtype=self.DTYPE,
            moe_backend="HETER",
            mapping=mapping,
            heter_config=heter_config,
        )

        with torch.device(f"cuda:{mapping.rank}"):
            backend = create_backend(
                moe_cls=HeterCutlassFusedMoE,
                routing_method=routing_method,
                num_experts=self.NUM_EXPERTS,
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                dtype=self.DTYPE,
                model_config=model_config,
            )
            backend.load_weights([copy.deepcopy(weights)])
            backend.post_load_weights()
            backend.cuda()
            x = torch.randn(
                (seq_len, self.HIDDEN_SIZE),
                dtype=self.DTYPE, device="cuda",
            )
            router_logits = torch.randn(
                (seq_len, self.NUM_EXPERTS),
                dtype=self.DTYPE, device="cuda",
            )
            token_selected_experts, token_final_scales = routing_method.apply(
                router_logits
            )
        return backend, x, router_logits, token_selected_experts, token_final_scales

    # ------------------------------------------------------------------
    # Policy determinism
    # ------------------------------------------------------------------

    def test_random_policy_deterministic(self):
        """RandomHeterDispatch.dispatch() is deterministic for same seed+input."""
        policy = RandomHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
            seed=42,
        )
        token_selected_experts = torch.tensor(
            [[0, 2], [1, 3], [2, 5], [4, 7]],
            dtype=torch.int32, device="cuda",
        )
        token_final_scales = torch.ones_like(
            token_selected_experts, dtype=torch.float32,
        )

        dispatches_1 = policy.dispatch(token_selected_experts, token_final_scales)
        dispatches_2 = policy.dispatch(token_selected_experts, token_final_scales)

        for (e1, s1), (e2, s2) in zip(dispatches_1, dispatches_2):
            torch.testing.assert_close(e1, e2)
            torch.testing.assert_close(s1, s2)

    # ------------------------------------------------------------------
    # Policy property setter
    # ------------------------------------------------------------------

    def test_policy_property_setter(self):
        """backend.policy = new_policy replaces the dispatch policy."""
        backend, x, _, tse, tfs = self._create_backend_with_weights(
            single_group_config()
        )
        old_policy = backend.policy
        new_policy = RandomHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[1.0],
            seed=99,
        )
        backend.policy = new_policy
        assert backend.policy is new_policy
        assert backend.policy is not old_policy

    # ------------------------------------------------------------------
    # Assignment validation
    # ------------------------------------------------------------------

    def test_validate_expert_to_group(self):
        """_validate_expert_to_group checks shape and value range."""
        # Valid: 4 experts, 2 groups.
        e2g = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="cuda")
        _validate_expert_to_group(e2g, num_experts=4, num_groups=2)

        # Wrong shape.
        bad_shape = torch.tensor([0, 0, 1], dtype=torch.long, device="cuda")
        with pytest.raises(AssertionError, match="shape"):
            _validate_expert_to_group(bad_shape, num_experts=4, num_groups=2)

        # Group index out of range.
        bad_val = torch.tensor([0, 0, 2, 1], dtype=torch.long, device="cuda")
        with pytest.raises(AssertionError, match="num_groups"):
            _validate_expert_to_group(bad_val, num_experts=4, num_groups=2)

    # ------------------------------------------------------------------
    # Custom policy via subclass
    # ------------------------------------------------------------------

    def test_custom_policy(self):
        """A custom HeterDispatchPolicy routes all experts to group 0."""

        class AllGroup0Policy(HeterDispatchPolicy):

            def _assign(self, token_selected_experts, token_final_scales):
                device = token_selected_experts.device
                return torch.zeros(
                    self._num_experts, dtype=torch.long, device=device)

        backend, x, _, tse, tfs = self._create_backend_with_weights(
            two_bf16_group_config()
        )
        backend.policy = AllGroup0Policy(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
        )

        with torch.inference_mode():
            out = backend.run_moe(
                x=x,
                token_selected_experts=tse,
                token_final_scales=tfs,
                output_dtype=self.DTYPE,
            )
        assert out.shape == x.shape

    # ------------------------------------------------------------------
    # Confidence threshold dispatch
    # ------------------------------------------------------------------

    def test_confidence_threshold_dispatch(self):
        """ConfidenceThresholdHeterDispatch assigns higher-scoring experts to later groups."""
        # Use a larger batch and top_k so that per-expert mean routing
        # weights are spread out enough to produce a clear ordering.
        seq_len = 256
        top_k = 4
        backend, x, _, tse, tfs = self._create_backend_with_weights(
            two_bf16_group_config(), seq_len=seq_len, top_k=top_k,
        )
        backend.policy = ConfidenceThresholdHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
            confidence_threshold=0.5,
        )

        dispatches = backend.policy.dispatch(tse, tfs)
        assert len(dispatches) == 2

        for experts, scales in dispatches:
            assert experts is not None
            assert scales is not None
            assert experts.shape[1] == top_k
            assert scales.shape == experts.shape

        # Verify that every expert in a later group has a higher mean
        # routing weight than every expert in the earlier group (the
        # invariant maintained by _assign_by_score_gpu's topk/argsort).
        flat_experts = tse.reshape(-1).long()
        flat_scales = tfs.reshape(-1)

        weight_sum = torch.zeros(
            self.NUM_EXPERTS, dtype=torch.float32, device=tse.device)
        weight_sum.scatter_add_(0, flat_experts, flat_scales)
        expert_count = torch.bincount(
            flat_experts, minlength=self.NUM_EXPERTS,
        ).to(dtype=torch.float32).clamp_min_(1.0)
        mean_score = weight_sum / expert_count

        e2g = backend.policy._assign(tse, tfs)
        for gidx in range(1, backend.policy.num_groups):
            earlier_mask = (e2g == gidx - 1)
            later_mask = (e2g == gidx)
            if not earlier_mask.any() or not later_mask.any():
                continue
            max_score_earlier = mean_score[earlier_mask].max().item()
            min_score_later = mean_score[later_mask].min().item()
            assert min_score_later >= max_score_earlier, (
                f"Group {gidx} min mean score ({min_score_later:.4f}) < "
                f"group {gidx - 1} max mean score ({max_score_earlier:.4f}). "
                f"Scores: {mean_score.tolist()}"
            )

    # ------------------------------------------------------------------
    # Expert load dispatch
    # ------------------------------------------------------------------

    def test_expert_load_dispatch(self):
        """ExpertLoadHeterDispatch assigns higher-load experts to later groups."""
        # Use a larger batch and top_k so that expert activation counts
        # are spread out enough to produce a clear load-based ordering.
        seq_len = 256
        top_k = 4
        backend, x, _, tse, tfs = self._create_backend_with_weights(
            two_bf16_group_config(), seq_len=seq_len, top_k=top_k,
        )
        backend.policy = ExpertLoadHeterDispatch(
            num_experts=self.NUM_EXPERTS,
            group_size_ratios=[0.5, 0.5],
        )

        dispatches = backend.policy.dispatch(tse, tfs)
        assert len(dispatches) == 2

        for experts, scales in dispatches:
            assert experts is not None
            assert scales is not None

        # Verify that every expert in a later group has activation count
        # >= every expert in the earlier group (the invariant maintained
        # by _assign_by_score_gpu's topk/argsort).
        flat_experts = tse.reshape(-1).long()
        counts = torch.bincount(flat_experts, minlength=self.NUM_EXPERTS)

        e2g = backend.policy._assign(tse, tfs)
        for gidx in range(1, backend.policy.num_groups):
            earlier_mask = (e2g == gidx - 1)
            later_mask = (e2g == gidx)
            if not earlier_mask.any() or not later_mask.any():
                continue
            max_load_earlier = counts[earlier_mask].max().item()
            min_load_later = counts[later_mask].min().item()
            assert min_load_later >= max_load_earlier, (
                f"Group {gidx} min load ({min_load_later}) < "
                f"group {gidx - 1} max load ({max_load_earlier}). "
                f"Counts: {counts.tolist()}"
            )

    # ------------------------------------------------------------------
    # Fallback without signals
    # ------------------------------------------------------------------

    def test_confidence_dispatch_fallback_without_signals(self):
        """ConfidenceThresholdHeterDispatch._assign() falls back to random
        when signals are ``None``."""
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            fallback_seed=42,
        )
        e2g = policy._assign(
            token_selected_experts=None,
            token_final_scales=None,
        )
        assert e2g.shape == (8,)
        assert e2g.min().item() >= 0
        assert e2g.max().item() <= 1

    def test_expert_load_dispatch_fallback_without_signals(self):
        """ExpertLoadHeterDispatch._assign() falls back to random when
        signals are ``None``."""
        policy = ExpertLoadHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            fallback_seed=42,
        )
        e2g = policy._assign(
            token_selected_experts=None,
            token_final_scales=None,
        )
        assert e2g.shape == (8,)
        assert e2g.min().item() >= 0
        assert e2g.max().item() <= 1

    # ------------------------------------------------------------------
    # Score-based ranking
    # ------------------------------------------------------------------

    def test_confidence_dispatch_ranks_by_weight(self):
        """ConfidenceThresholdHeterDispatch puts high-weight experts in
        last group."""
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=4,
            group_size_ratios=[0.5, 0.5],
        )
        # Simulate 4 experts.  Experts 2 and 3 receive much higher weights.
        token_selected_experts = torch.tensor(
            [[0, 2], [1, 3], [2, 3], [0, 2]],
            dtype=torch.int32, device="cuda",
        )
        token_final_scales = torch.tensor(
            [[0.1, 0.9], [0.1, 0.9], [0.8, 0.9], [0.1, 0.8]],
            dtype=torch.float32, device="cuda",
        )
        e2g = policy._assign(token_selected_experts, token_final_scales)
        assert e2g.shape == (4,)
        # Last group (high-precision, group 1) should get the highest-weight experts.
        assert e2g[2].item() == 1
        assert e2g[3].item() == 1

    def test_expert_load_dispatch_ranks_by_frequency(self):
        """ExpertLoadHeterDispatch puts frequently-activated experts in
        last group."""
        policy = ExpertLoadHeterDispatch(
            num_experts=4,
            group_size_ratios=[0.5, 0.5],
        )
        # Expert 0 selected 4 times, expert 1 selected 3 times,
        # expert 2 once, expert 3 once.
        token_selected_experts = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 2], [3, 2]],
            dtype=torch.int32, device="cuda",
        )
        token_final_scales = torch.ones_like(
            token_selected_experts, dtype=torch.float32,
        )
        e2g = policy._assign(token_selected_experts, token_final_scales)
        assert e2g.shape == (4,)
        # Last group (group 1) should contain most-active experts.
        assert e2g[0].item() == 1
        assert e2g[1].item() == 1

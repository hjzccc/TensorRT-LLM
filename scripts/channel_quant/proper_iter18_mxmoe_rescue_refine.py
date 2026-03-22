#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import LayerMetricBundle, fp8_weights_from_masks, resolve_non_expert_bytes
from proper_eval import CALIBRATION_SAMPLES, SEQLEN, atomic_json_dump, dtype_from_name, load_gptq_standard_data
from proper_iter01 import TOPUP_FRACTION as MXMOE_TOPUP_FRACTION
from proper_iter01 import build_mxmoe_topup_masks, load_cache
from proper_iter10_novel_perchannel import load_metric_cache
from proper_iter16_counterfactual_rescue import (
    advance_bundle_pointer,
    build_plan_from_masks,
    build_router_orders,
    clone_masks,
    evaluate_and_record_plan,
    load_json,
    load_reference_rows,
    resolve_requested_plans,
)
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter18_mxmoe_rescue_refine.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
SECTION_MARKER = "## [21] Iteration 18 - MxMoE Rescue Refine"
BASE_NAME = "mxmoe_block_plus_channel_topup"
W1_BUNDLE_PAIRS = 16
DEFAULT_W2_BUNDLE_CHANNELS = 64
SMALL_W2_BUNDLE_CHANNELS = 32

ACTION_PRIORITY = {
    "w2_bundle_rescue": 0,
    "w2_small_bundle_rescue": 1,
    "w2_projection_rescue": 2,
    "w1_bundle_rescue": 3,
}


@dataclass(frozen=True)
class RescueConfig:
    name: str
    description: str
    extra_budget_gb: float
    rescue_policy: str
    w2_bundle_channels: int
    allow_w1_bundle: bool
    allow_w2_projection: bool


@dataclass(order=True)
class RescueCandidate:
    sort_key: tuple[float, int, int, int, int, str]
    version: int
    action_kind: str
    projection: str
    layer_idx: int
    expert_idx: int
    gain_proxy: float
    bytes_added: int
    fp8_weights_added: int
    selected_indices: tuple[int, ...] = ()
    next_pointer: int = 0
    bundle_size: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 18 rescue-refine configs.",
    )
    return parser.parse_args()


def mxmoe_rescue_configs() -> list[RescueConfig]:
    return [
        RescueConfig(
            name="cfk_mxmoe_plus_0p10gb",
            description="Refine the MxMoE block-plus-topup base with mixed rescue actions under a 0.10 GB extra expert-memory budget.",
            extra_budget_gb=0.10,
            rescue_policy="mixed",
            w2_bundle_channels=DEFAULT_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=True,
            allow_w2_projection=True,
        ),
        RescueConfig(
            name="cfk_mxmoe_plus_0p15gb",
            description="Refine the MxMoE block-plus-topup base with mixed rescue actions under a 0.15 GB extra expert-memory budget.",
            extra_budget_gb=0.15,
            rescue_policy="mixed",
            w2_bundle_channels=DEFAULT_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=True,
            allow_w2_projection=True,
        ),
        RescueConfig(
            name="cfk_mxmoe_plus_0p20gb",
            description="Refine the MxMoE block-plus-topup base with mixed rescue actions under a 0.20 GB extra expert-memory budget.",
            extra_budget_gb=0.20,
            rescue_policy="mixed",
            w2_bundle_channels=DEFAULT_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=True,
            allow_w2_projection=True,
        ),
        RescueConfig(
            name="cfk_mxmoe_plus_0p30gb",
            description="Refine the MxMoE block-plus-topup base with mixed rescue actions under a 0.30 GB extra expert-memory budget.",
            extra_budget_gb=0.30,
            rescue_policy="mixed",
            w2_bundle_channels=DEFAULT_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=True,
            allow_w2_projection=True,
        ),
        RescueConfig(
            name="cfk_mxmoe_w2only_plus_0p10gb",
            description="Refine the MxMoE block-plus-topup base with W2-only rescue actions under a 0.10 GB extra expert-memory budget.",
            extra_budget_gb=0.10,
            rescue_policy="w2_only",
            w2_bundle_channels=DEFAULT_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=False,
            allow_w2_projection=True,
        ),
        RescueConfig(
            name="cfk_mxmoe_w2only_plus_0p20gb",
            description="Refine the MxMoE block-plus-topup base with W2-only rescue actions under a 0.20 GB extra expert-memory budget.",
            extra_budget_gb=0.20,
            rescue_policy="w2_only",
            w2_bundle_channels=DEFAULT_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=False,
            allow_w2_projection=True,
        ),
        RescueConfig(
            name="cfk_mxmoe_w2only_plus_0p30gb",
            description="Refine the MxMoE block-plus-topup base with W2-only rescue actions under a 0.30 GB extra expert-memory budget.",
            extra_budget_gb=0.30,
            rescue_policy="w2_only",
            w2_bundle_channels=DEFAULT_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=False,
            allow_w2_projection=True,
        ),
        RescueConfig(
            name="cfk_mxmoe_smallbundles_plus_0p20gb",
            description="Refine the MxMoE block-plus-topup base with mixed rescue actions, but use 32-channel W2 bundles for finer granularity under a 0.20 GB extra expert-memory budget.",
            extra_budget_gb=0.20,
            rescue_policy="mixed_small_w2_bundles",
            w2_bundle_channels=SMALL_W2_BUNDLE_CHANNELS,
            allow_w1_bundle=True,
            allow_w2_projection=True,
        ),
    ]


def per_unit_fp8_weights(config: Any, projection: str) -> int:
    if projection == "w1":
        return 2 * int(config.hidden_size)
    return int(config.moe_intermediate_size)


def scaled_delta_share(delta_value: float, added_units: int, total_units: int) -> float:
    if added_units <= 0 or total_units <= 0:
        return 0.0
    return max(float(delta_value), 0.0) * (float(added_units) / float(total_units))


def build_bundle_candidate(
    config: Any,
    action_kind: str,
    projection: str,
    layer_idx: int,
    expert_idx: int,
    mask: torch.Tensor,
    score_tensor: torch.Tensor,
    order: torch.Tensor,
    pointer: int,
    bundle_size: int,
    projection_delta: float,
    version: int,
) -> RescueCandidate | None:
    ptr = advance_bundle_pointer(mask, order, pointer)
    order_cpu = order.detach().cpu()
    score_cpu = score_tensor.detach().cpu().to(torch.float32)
    chosen: list[int] = []
    while ptr < int(order_cpu.numel()) and len(chosen) < bundle_size:
        index = int(order_cpu[ptr].item())
        if not bool(mask[index]):
            chosen.append(index)
        ptr += 1
    if not chosen:
        return None
    router_gain = float(sum(float(score_cpu[index].item()) for index in chosen))
    gain_proxy = router_gain + scaled_delta_share(projection_delta, len(chosen), int(mask.numel()))
    if gain_proxy <= 0.0:
        return None
    fp8_weights_added = len(chosen) * per_unit_fp8_weights(config, projection)
    bytes_added = fp8_weights_added // 2
    if bytes_added <= 0:
        return None
    ratio = gain_proxy / float(bytes_added)
    return RescueCandidate(
        sort_key=(-ratio, ACTION_PRIORITY[action_kind], bytes_added, layer_idx, expert_idx, action_kind),
        version=version,
        action_kind=action_kind,
        projection=projection,
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        gain_proxy=float(gain_proxy),
        bytes_added=int(bytes_added),
        fp8_weights_added=int(fp8_weights_added),
        selected_indices=tuple(chosen),
        next_pointer=int(ptr),
        bundle_size=int(len(chosen)),
    )


def build_w2_projection_candidate(
    config: Any,
    layer_idx: int,
    expert_idx: int,
    mask: torch.Tensor,
    score_tensor: torch.Tensor,
    projection_delta: float,
    version: int,
) -> RescueCandidate | None:
    remaining_mask = ~mask.detach().cpu()
    remaining = int(remaining_mask.sum().item())
    if remaining <= 0:
        return None
    score_cpu = score_tensor.detach().cpu().to(torch.float32)
    router_gain = float(score_cpu[remaining_mask].sum().item())
    gain_proxy = router_gain + scaled_delta_share(projection_delta, remaining, int(mask.numel()))
    if gain_proxy <= 0.0:
        return None
    fp8_weights_added = remaining * per_unit_fp8_weights(config, "w2")
    bytes_added = fp8_weights_added // 2
    if bytes_added <= 0:
        return None
    ratio = gain_proxy / float(bytes_added)
    return RescueCandidate(
        sort_key=(-ratio, ACTION_PRIORITY["w2_projection_rescue"], bytes_added, layer_idx, expert_idx, "w2_projection_rescue"),
        version=version,
        action_kind="w2_projection_rescue",
        projection="w2",
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        gain_proxy=float(gain_proxy),
        bytes_added=int(bytes_added),
        fp8_weights_added=int(fp8_weights_added),
        bundle_size=int(remaining),
    )


def push_candidates_for_projection(
    heap: list[RescueCandidate],
    calibration: Any,
    router_affinity_cache: dict[int, LayerMetricBundle],
    router_orders: dict[str, dict[int, dict[int, torch.Tensor]]],
    router_pointers: dict[str, dict[int, dict[int, int]]],
    config: Any,
    rescue_config: RescueConfig,
    allowed_action_kinds: set[str],
    projection: str,
    layer_idx: int,
    expert_idx: int,
    mask: torch.Tensor,
    version: int,
) -> None:
    if projection == "w1":
        if not rescue_config.allow_w1_bundle or "w1_bundle_rescue" not in allowed_action_kinds:
            return
        delta_value = float(calibration.mxmoe_w1_deltas[layer_idx][expert_idx].item())
        candidate = build_bundle_candidate(
            config,
            "w1_bundle_rescue",
            "w1",
            layer_idx,
            expert_idx,
            mask,
            router_affinity_cache[layer_idx].w1_pair_scores[expert_idx],
            router_orders["w1"][layer_idx][expert_idx],
            router_pointers["w1"][layer_idx][expert_idx],
            W1_BUNDLE_PAIRS,
            delta_value,
            version,
        )
        if candidate is not None:
            heapq.heappush(heap, candidate)
        return

    delta_value = float(calibration.mxmoe_w2_deltas[layer_idx][expert_idx].item())
    bundle_kind = "w2_small_bundle_rescue" if rescue_config.w2_bundle_channels == SMALL_W2_BUNDLE_CHANNELS else "w2_bundle_rescue"
    if bundle_kind in allowed_action_kinds:
        bundle_candidate = build_bundle_candidate(
            config,
            bundle_kind,
            "w2",
            layer_idx,
            expert_idx,
            mask,
            router_affinity_cache[layer_idx].w2_channel_scores[expert_idx],
            router_orders["w2"][layer_idx][expert_idx],
            router_pointers["w2"][layer_idx][expert_idx],
            rescue_config.w2_bundle_channels,
            delta_value,
            version,
        )
        if bundle_candidate is not None:
            heapq.heappush(heap, bundle_candidate)
    if rescue_config.allow_w2_projection and "w2_projection_rescue" in allowed_action_kinds:
        projection_candidate = build_w2_projection_candidate(
            config,
            layer_idx,
            expert_idx,
            mask,
            router_affinity_cache[layer_idx].w2_channel_scores[expert_idx],
            delta_value,
            version,
        )
        if projection_candidate is not None:
            heapq.heappush(heap, projection_candidate)


def rescue_stage_sequence(rescue_config: RescueConfig) -> list[set[str]]:
    w2_bundle_kind = "w2_small_bundle_rescue" if rescue_config.w2_bundle_channels == SMALL_W2_BUNDLE_CHANNELS else "w2_bundle_rescue"
    stages: list[set[str]] = [{w2_bundle_kind}]
    if rescue_config.allow_w2_projection:
        stages.append({"w2_projection_rescue"})
    if rescue_config.allow_w1_bundle:
        stages.append({"w1_bundle_rescue"})
    return stages


def summarize_actions(selected_actions: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = [
        "w2_bundle_rescue",
        "w2_small_bundle_rescue",
        "w2_projection_rescue",
        "w1_bundle_rescue",
    ]
    counts = {kind: 0 for kind in kinds}
    bytes_by_type = {kind: 0 for kind in kinds}
    fp8_weights_by_type = {kind: 0 for kind in kinds}
    for action in selected_actions:
        kind = str(action["action_kind"])
        counts[kind] += 1
        bytes_by_type[kind] += int(action["bytes_added"])
        fp8_weights_by_type[kind] += int(action["fp8_weights_added"])
    w2_action_count = counts["w2_bundle_rescue"] + counts["w2_small_bundle_rescue"] + counts["w2_projection_rescue"]
    w2_action_bytes = bytes_by_type["w2_bundle_rescue"] + bytes_by_type["w2_small_bundle_rescue"] + bytes_by_type["w2_projection_rescue"]
    return {
        "selected_action_count": int(len(selected_actions)),
        "selected_action_type_counts": counts,
        "selected_action_bytes": bytes_by_type,
        "selected_action_fp8_weights": fp8_weights_by_type,
        "w2_action_count": int(w2_action_count),
        "w1_action_count": int(counts["w1_bundle_rescue"]),
        "w2_action_bytes": int(w2_action_bytes),
        "w1_action_bytes": int(bytes_by_type["w1_bundle_rescue"]),
    }


def apply_rescue_knapsack(
    calibration: Any,
    router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    rescue_config: RescueConfig,
    base_w1_masks: dict[int, dict[int, torch.Tensor]],
    base_w2_masks: dict[int, dict[int, torch.Tensor]],
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_masks = clone_masks(base_w1_masks)
    w2_masks = clone_masks(base_w2_masks)
    router_orders = {
        "w1": build_router_orders(router_affinity_cache, "w1"),
        "w2": build_router_orders(router_affinity_cache, "w2"),
    }
    router_pointers = {
        "w1": {layer_idx: {expert_idx: 0 for expert_idx in expert_map} for layer_idx, expert_map in router_orders["w1"].items()},
        "w2": {layer_idx: {expert_idx: 0 for expert_idx in expert_map} for layer_idx, expert_map in router_orders["w2"].items()},
    }
    versions = {
        "w1": {layer_idx: {expert_idx: 0 for expert_idx in expert_map} for layer_idx, expert_map in router_orders["w1"].items()},
        "w2": {layer_idx: {expert_idx: 0 for expert_idx in expert_map} for layer_idx, expert_map in router_orders["w2"].items()},
    }
    selected_actions: list[dict[str, Any]] = []
    extra_budget_bytes = int(round(rescue_config.extra_budget_gb * 1e9))
    remaining_budget_bytes = int(extra_budget_bytes)
    stage_order = rescue_stage_sequence(rescue_config)

    for stage_idx, allowed_action_kinds in enumerate(stage_order, start=1):
        heap: list[RescueCandidate] = []
        for layer_idx in range(config.num_hidden_layers):
            routing_counts = calibration.routing_counts[layer_idx]
            for expert_idx in range(config.num_experts):
                if int(routing_counts[expert_idx].item()) <= 0:
                    continue
                push_candidates_for_projection(
                    heap,
                    calibration,
                    router_affinity_cache,
                    router_orders,
                    router_pointers,
                    config,
                    rescue_config,
                    allowed_action_kinds,
                    "w2",
                    layer_idx,
                    expert_idx,
                    w2_masks[layer_idx][expert_idx],
                    versions["w2"][layer_idx][expert_idx],
                )
                if rescue_config.allow_w1_bundle:
                    push_candidates_for_projection(
                        heap,
                        calibration,
                        router_affinity_cache,
                        router_orders,
                        router_pointers,
                        config,
                        rescue_config,
                        allowed_action_kinds,
                        "w1",
                        layer_idx,
                        expert_idx,
                        w1_masks[layer_idx][expert_idx],
                        versions["w1"][layer_idx][expert_idx],
                    )

        while heap and remaining_budget_bytes > 0:
            candidate = heapq.heappop(heap)
            current_version = versions[candidate.projection][candidate.layer_idx][candidate.expert_idx]
            if candidate.version != current_version:
                continue
            if candidate.bytes_added > remaining_budget_bytes:
                continue

            mask_map = w1_masks if candidate.projection == "w1" else w2_masks
            current_mask = mask_map[candidate.layer_idx][candidate.expert_idx]
            selected_before = int(current_mask.sum().item())

            if candidate.action_kind == "w2_projection_rescue":
                current_mask.fill_(True)
                new_units_added = int(current_mask.numel()) - selected_before
            else:
                if not candidate.selected_indices:
                    continue
                index_tensor = torch.as_tensor(candidate.selected_indices, dtype=torch.long)
                if bool(current_mask[index_tensor].all()):
                    continue
                current_mask[index_tensor] = True
                router_pointers[candidate.projection][candidate.layer_idx][candidate.expert_idx] = int(candidate.next_pointer)
                new_units_added = len(candidate.selected_indices)

            selected_after = int(current_mask.sum().item())
            if selected_after <= selected_before:
                continue

            remaining_budget_bytes -= int(candidate.bytes_added)
            action_record: dict[str, Any] = {
                "step": int(len(selected_actions) + 1),
                "stage_idx": int(stage_idx),
                "action_kind": candidate.action_kind,
                "projection": candidate.projection,
                "layer_idx": int(candidate.layer_idx),
                "expert_idx": int(candidate.expert_idx),
                "gain_proxy": round(float(candidate.gain_proxy), 6),
                "bytes_added": int(candidate.bytes_added),
                "fp8_weights_added": int(candidate.fp8_weights_added),
                "selected_before": int(selected_before),
                "selected_after": int(selected_after),
                "new_units_added": int(new_units_added),
                "bundle_size": int(candidate.bundle_size),
                "remaining_budget_bytes": int(remaining_budget_bytes),
                "gain_per_byte": round(float(candidate.gain_proxy) / float(max(candidate.bytes_added, 1)), 12),
            }
            if candidate.selected_indices:
                action_record["selected_indices"] = [int(index) for index in candidate.selected_indices]
            selected_actions.append(action_record)

            versions[candidate.projection][candidate.layer_idx][candidate.expert_idx] += 1
            new_version = versions[candidate.projection][candidate.layer_idx][candidate.expert_idx]
            push_candidates_for_projection(
                heap,
                calibration,
                router_affinity_cache,
                router_orders,
                router_pointers,
                config,
                rescue_config,
                allowed_action_kinds,
                candidate.projection,
                candidate.layer_idx,
                candidate.expert_idx,
                current_mask,
                new_version,
            )

    realized_fp8_weights = fp8_weights_from_masks(config, w1_masks, w2_masks)
    base_fp8_weights = fp8_weights_from_masks(config, base_w1_masks, base_w2_masks)
    added_fp8_weights = int(realized_fp8_weights - base_fp8_weights)
    realized_extra_bytes = added_fp8_weights // 2
    return w1_masks, w2_masks, {
        "base_name": BASE_NAME,
        "rescue_policy": rescue_config.rescue_policy,
        "w2_bundle_channels": int(rescue_config.w2_bundle_channels),
        "w1_bundle_pairs": int(W1_BUNDLE_PAIRS),
        "allow_w1_bundle": bool(rescue_config.allow_w1_bundle),
        "allow_w2_projection": bool(rescue_config.allow_w2_projection),
        "rescue_stage_order": [sorted(stage) for stage in stage_order],
        "extra_budget_gb": float(rescue_config.extra_budget_gb),
        "target_extra_budget_bytes": int(extra_budget_bytes),
        "target_extra_fp8_weights": int(extra_budget_bytes * 2),
        "realized_extra_bytes": int(realized_extra_bytes),
        "realized_extra_gb": round(float(realized_extra_bytes) / 1e9, 6),
        "realized_extra_fp8_weights": int(added_fp8_weights),
        "unused_budget_bytes": int(max(remaining_budget_bytes, 0)),
        "unused_budget_gb": round(float(max(remaining_budget_bytes, 0)) / 1e9, 6),
        "selected_actions": selected_actions,
        **summarize_actions(selected_actions),
    }


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_ref = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 184, flush=True)
    print("proper_iter18_mxmoe_rescue_refine | MxMoE base + refined W2-biased rescue | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 184, flush=True)
    print(
        f"{'Config':<34} {'PPL':>10} {'dBest':>10} {'Memory GB':>12} {'Extra GB':>10} {'Policy':<24} {'W2 acts':>8} {'W1 acts':>8}",
        flush=True,
    )
    print("-" * 184, flush=True)
    for name, row in ordered:
        d_best = "-" if best_ref is None else f"{float(row['ppl']) - best_ref:+.4f}"
        print(
            f"{name:<34} {float(row['ppl']):>10.4f} {d_best:>10} {float(row['memory_gb']):>12.3f} {float(row['realized_extra_gb']):>10.3f} {str(row['rescue_policy']):<24} {int(row['w2_action_count']):>8} {int(row['w1_action_count']):>8}",
            flush=True,
        )


def build_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    rows = payload["results"]
    ordered = sorted(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_name, best_row = ordered[0]
    previous_best = min((row["ppl"] for row in references.values()), default=None)
    table_lines = [
        "| Config | Policy | PPL | Delta vs prev best | Memory (GB) | Extra GB | W2 actions | W1 actions |",
        "|--------|--------|-----|--------------------|-------------|----------|------------|------------|",
    ]
    for name, row in ordered:
        delta_prev = "-" if previous_best is None else f"{float(row['ppl']) - previous_best:+.4f}"
        table_lines.append(
            f"| {name} | {row['rescue_policy']} | {float(row['ppl']):.4f} | {delta_prev} | {float(row['memory_gb']):.3f} | {float(row['realized_extra_gb']):.3f} | {int(row['w2_action_count'])} | {int(row['w1_action_count'])} |"
        )

    if previous_best is None:
        insight = f"Best refined rescue config is `{best_name}` at PPL {float(best_row['ppl']):.4f}."
    else:
        delta = float(best_row["ppl"]) - previous_best
        verb = "beats" if delta < 0.0 else "still trails"
        insight = (
            f"Best refined rescue config is `{best_name}` at PPL {float(best_row['ppl']):.4f}; it {verb} the prior best by {abs(delta):.4f} PPL. "
            f"That run used `{best_row['rescue_policy']}` rescue with {int(best_row['w2_action_count'])} W2 actions and {int(best_row['w1_action_count'])} W1 actions for {float(best_row['realized_extra_gb']):.3f} GB of extra expert memory."
        )

    return "\n".join([
        SECTION_MARKER,
        "**Approach**: Reused the Iteration 16 counterfactual rescue framework, but narrowed it to the strongest MxMoE block-plus-topup base and replaced the coarse action pool with a W2-biased one: 64-channel W2 bundles, optional 32-channel W2 bundles, 16-pair W1 bundles, and full W2 projection rescues. Mixed policies now run in explicit stages so W2 bundles are exhausted first, then W2 projection rescues, and only then W1 bundles if budget remains. Within each stage, actions are ranked by proxy gain per added byte, where bundle gains combine next-unused router-affinity scores with a proportional share of the cached MxMoE block delta, and full W2 projection rescues use the remaining W2 router-affinity mass plus the remaining W2 block delta share.",
        f"**Eval**: Full WikiText-2 test ({payload['metadata']['evaluation']['total_tokens']} tokens, {payload['metadata']['evaluation']['nsamples']} chunks of {SEQLEN}), BF16 `F.linear`, FP32 loss, `loss.float() * seqlen`.",
        "**Result**:",
        "\n".join(table_lines),
        f"**Insight**: {insight}",
        "**Next**: If the best refined rescue still misses the joint frontier, keep the same MxMoE seed and test residual-style rescue only on the remaining W2 channels in the best 0.10-0.20 GB window.",
    ])


def upsert_exploration_section(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SECTION_MARKER in existing:
        prefix = existing.split(SECTION_MARKER, 1)[0].rstrip()
        updated = prefix + "\n\n" + content.strip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + content.strip() + "\n" if existing.strip() else content.strip() + "\n"
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    requested_plans = resolve_requested_plans(args.plans)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    _tokenizer, _calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    references = load_reference_rows(args.output_json)
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP32 logits and loss",
            "protocol": {
                "reference": "/home/jerry/Documents/fork_new/MC-MoE/eval_ppl_utils.py",
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "metric_cache_path": str(args.metric_cache_path),
            "references": references,
            "experiment": "Iteration 18 refined MxMoE rescue with W2-biased action sets and finer extra-budget steps",
            "base_name": BASE_NAME,
            "candidate_actions": {
                "w2_bundle_rescue": {"channels_per_action": DEFAULT_W2_BUNDLE_CHANNELS},
                "w2_small_bundle_rescue": {"channels_per_action": SMALL_W2_BUNDLE_CHANNELS},
                "w1_bundle_rescue": {"pairs_per_action": W1_BUNDLE_PAIRS},
                "w2_projection_rescue": "full remaining W2 block to FP8",
            },
        },
        "results": {},
    }
    if args.output_json.exists():
        try:
            existing = load_json(args.output_json)
            if isinstance(existing, dict):
                merged_metadata = dict(payload["metadata"])
                merged_metadata.update(existing.get("metadata", {}))
                payload.update(existing)
                payload["metadata"] = merged_metadata
                payload.setdefault("results", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise FileNotFoundError(f"Calibration cache missing or incompatible: {args.cache_path}")

    router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = load_metric_cache(
        args.metric_cache_path,
        args.model_id,
    )
    if router_affinity_cache is None or metric_cache_meta is None:
        raise FileNotFoundError(f"Router-affinity metric cache missing or incompatible: {args.metric_cache_path}")

    payload["metadata"]["metric_cache"] = metric_cache_meta
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    base_w1, base_w2, base_meta = build_mxmoe_topup_masks(calibration, text_config, total_expert_elems, MXMOE_TOPUP_FRACTION)
    base_fp8_weights = fp8_weights_from_masks(text_config, base_w1, base_w2)
    base_memory_gb = round(float(non_expert_bytes + int((0.5 * total_expert_elems) + (0.5 * base_fp8_weights))) / 1e9, 3)

    configs = mxmoe_rescue_configs()
    if requested_plans is not None:
        available_names = {config.name for config in configs}
        missing = sorted(requested_plans.difference(available_names))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
        configs = [config for config in configs if config.name in requested_plans]
    payload["metadata"]["requested_plan_order"] = [config.name for config in configs]
    payload["metadata"]["mxmoe_base_builder_meta"] = base_meta
    atomic_json_dump(args.output_json, payload)

    for config_row in configs:
        rescued_w1, rescued_w2, rescue_meta = apply_rescue_knapsack(
            calibration,
            router_affinity_cache,
            text_config,
            config_row,
            base_w1,
            base_w2,
        )
        plan = build_plan_from_masks(
            config_row.name,
            config_row.description,
            text_config,
            non_expert_bytes,
            total_expert_elems,
            rescued_w1,
            rescued_w2,
        )
        evaluate_and_record_plan(
            payload,
            plan,
            {
                "base_name": BASE_NAME,
                "base_fp8_weights": int(base_fp8_weights),
                "base_memory_gb": round(float(references.get(BASE_NAME, {}).get("memory_gb", base_memory_gb)), 3),
                "channel_metric_requested": "router_affinity_weighted_qerror",
                "channel_metric_effective": "router_affinity_weighted_qerror",
                "channel_metric_fallback_used": False,
                "projection_gain_proxy": "remaining_router_affinity_mass_plus_remaining_mxmoe_w2_delta_share",
                "bundle_gain_proxy": "selected_router_affinity_mass_plus_proportional_mxmoe_delta_share",
                "base_builder_meta": base_meta,
                **rescue_meta,
            },
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )

    payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    payload["metadata"]["cache_version_expectation"] = {
        "seqlen": SEQLEN,
        "calibration_samples": CALIBRATION_SAMPLES,
    }
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload["results"], references)
    upsert_exploration_section(args.exploration_md, build_exploration_section(payload, references))
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

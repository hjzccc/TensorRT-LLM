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
from proper_eval import CALIBRATION_SAMPLES, SEQLEN, EvalPlan, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import TOPUP_FRACTION as MXMOE_TOPUP_FRACTION
from proper_iter01 import build_mxmoe_topup_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter10_novel_perchannel import load_metric_cache
from proper_iter14 import build_union_base_router_affinity_topup_masks
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter16_counterfactual_rescue.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
SECTION_MARKER = "## [21] Counterfactual Rescue Knapsack"
BEST_REFERENCE_NAME = "joint_w1w2_with_topup"
W2_BUNDLE_CHANNELS = 64
W1_BUNDLE_PAIRS = 16
UNION_TOPUP_FRACTION = 0.05


@dataclass(frozen=True)
class RescueConfig:
    name: str
    description: str
    base_name: str
    extra_budget_gb: float


@dataclass(order=True)
class RescueCandidate:
    sort_key: tuple[float, int, int, int, str, str]
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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 16 rescue configs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def clone_masks(source: dict[int, dict[int, torch.Tensor]]) -> dict[int, dict[int, torch.Tensor]]:
    return {
        int(layer_idx): {int(expert_idx): mask.detach().cpu().clone() for expert_idx, mask in expert_masks.items()}
        for layer_idx, expert_masks in source.items()
    }


def build_plan_from_masks(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_channel_masks: dict[int, dict[int, torch.Tensor]],
) -> EvalPlan:
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvalPlan(
        name=name,
        description=description,
        mode="per_channel",
        memory_gb=round(float(non_expert_bytes + int((0.5 * total_expert_elems) + (0.5 * fp8_weights))) / 1e9, 3),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def load_reference_rows(exclude_path: Path | None = None) -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    exclude_resolved = exclude_path.resolve() if exclude_path is not None else None
    for path in sorted(RESULTS_DIR.glob("proper*.json")):
        if exclude_resolved is not None and path.resolve() == exclude_resolved:
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        for name, row in payload.get("results", {}).items():
            if isinstance(row, dict) and "ppl" in row and "memory_gb" in row:
                references[str(name)] = {
                    "ppl": float(row["ppl"]),
                    "memory_gb": float(row["memory_gb"]),
                }
    return references


def build_router_orders(metric_cache: dict[int, LayerMetricBundle], projection: str) -> dict[int, dict[int, torch.Tensor]]:
    orders: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        orders[layer_idx] = {}
        scores = bundle.w1_pair_scores if projection == "w1" else bundle.w2_channel_scores
        for expert_idx in range(int(bundle.routing_counts.numel())):
            orders[layer_idx][expert_idx] = torch.argsort(scores[expert_idx].detach().cpu().to(torch.float32), descending=True)
    return orders


def base_action_configs() -> list[RescueConfig]:
    return [
        RescueConfig(
            name="cfk_joint_plus_0p25gb",
            description="Start from Iteration 7 joint W1/W2-with-topup masks, then greedily add rescue actions up to an extra 0.25 GB expert-memory budget.",
            base_name="joint_w1w2_with_topup",
            extra_budget_gb=0.25,
        ),
        RescueConfig(
            name="cfk_joint_plus_0p50gb",
            description="Start from Iteration 7 joint W1/W2-with-topup masks, then greedily add rescue actions up to an extra 0.50 GB expert-memory budget.",
            base_name="joint_w1w2_with_topup",
            extra_budget_gb=0.50,
        ),
        RescueConfig(
            name="cfk_union_plus_0p25gb",
            description="Start from the Iteration 14 union base with router-affinity topups, then greedily add rescue actions up to an extra 0.25 GB expert-memory budget.",
            base_name="output_perturbation_mxmoe_topup_router_affinity_5pct",
            extra_budget_gb=0.25,
        ),
        RescueConfig(
            name="cfk_union_plus_0p50gb",
            description="Start from the Iteration 14 union base with router-affinity topups, then greedily add rescue actions up to an extra 0.50 GB expert-memory budget.",
            base_name="output_perturbation_mxmoe_topup_router_affinity_5pct",
            extra_budget_gb=0.50,
        ),
        RescueConfig(
            name="cfk_mxmoe_plus_0p25gb",
            description="Start from Iteration 1 MxMoE block-plus-channel-topup masks, then greedily add rescue actions up to an extra 0.25 GB expert-memory budget.",
            base_name="mxmoe_block_plus_channel_topup",
            extra_budget_gb=0.25,
        ),
        RescueConfig(
            name="cfk_mxmoe_plus_0p50gb",
            description="Start from Iteration 1 MxMoE block-plus-channel-topup masks, then greedily add rescue actions up to an extra 0.50 GB expert-memory budget.",
            base_name="mxmoe_block_plus_channel_topup",
            extra_budget_gb=0.50,
        ),
    ]


def build_base_masks(
    calibration: Any,
    router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    total_expert_elems: int,
) -> dict[str, tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]]:
    joint_w1, joint_w2, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    union_w1, union_w2, union_meta = build_union_base_router_affinity_topup_masks(
        calibration,
        router_affinity_cache,
        config,
        total_expert_elems,
        UNION_TOPUP_FRACTION,
    )
    mxmoe_w1, mxmoe_w2, mxmoe_meta = build_mxmoe_topup_masks(
        calibration,
        config,
        total_expert_elems,
        MXMOE_TOPUP_FRACTION,
    )
    return {
        "joint_w1w2_with_topup": (joint_w1, joint_w2, joint_meta),
        "output_perturbation_mxmoe_topup_router_affinity_5pct": (union_w1, union_w2, union_meta),
        "mxmoe_block_plus_channel_topup": (mxmoe_w1, mxmoe_w2, mxmoe_meta),
    }


def advance_bundle_pointer(mask: torch.Tensor, order: torch.Tensor, pointer: int) -> int:
    ptr = int(pointer)
    order_cpu = order.detach().cpu()
    while ptr < int(order_cpu.numel()) and bool(mask[int(order_cpu[ptr].item())]):
        ptr += 1
    return ptr


def per_unit_fp8_weights(config: Any, projection: str) -> int:
    if projection == "w1":
        return 2 * int(config.hidden_size)
    return int(config.moe_intermediate_size)


def build_projection_candidate(
    config: Any,
    projection: str,
    layer_idx: int,
    expert_idx: int,
    mask: torch.Tensor,
    gain_proxy: float,
    version: int,
) -> RescueCandidate | None:
    selected = int(mask.sum().item())
    remaining = int(mask.numel()) - selected
    if remaining <= 0 or gain_proxy <= 0.0:
        return None
    fp8_weights_added = remaining * per_unit_fp8_weights(config, projection)
    bytes_added = fp8_weights_added // 2
    if bytes_added <= 0:
        return None
    ratio = gain_proxy / float(bytes_added)
    return RescueCandidate(
        sort_key=(-ratio, bytes_added, layer_idx, expert_idx, projection, "projection"),
        version=version,
        action_kind="projection_rescue",
        projection=projection,
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        gain_proxy=float(gain_proxy),
        bytes_added=int(bytes_added),
        fp8_weights_added=int(fp8_weights_added),
    )


def build_bundle_candidate(
    config: Any,
    projection: str,
    layer_idx: int,
    expert_idx: int,
    mask: torch.Tensor,
    score_tensor: torch.Tensor,
    order: torch.Tensor,
    pointer: int,
    version: int,
) -> RescueCandidate | None:
    bundle_size = W1_BUNDLE_PAIRS if projection == "w1" else W2_BUNDLE_CHANNELS
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
    gain_proxy = float(sum(float(score_cpu[index].item()) for index in chosen))
    if gain_proxy <= 0.0:
        return None
    fp8_weights_added = len(chosen) * per_unit_fp8_weights(config, projection)
    bytes_added = fp8_weights_added // 2
    if bytes_added <= 0:
        return None
    ratio = gain_proxy / float(bytes_added)
    return RescueCandidate(
        sort_key=(-ratio, bytes_added, layer_idx, expert_idx, projection, "bundle"),
        version=version,
        action_kind="channel_bundle_rescue",
        projection=projection,
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        gain_proxy=float(gain_proxy),
        bytes_added=int(bytes_added),
        fp8_weights_added=int(fp8_weights_added),
        selected_indices=tuple(chosen),
        next_pointer=int(ptr),
    )


def push_projection_candidates(
    heap: list[RescueCandidate],
    calibration: Any,
    config: Any,
    projection: str,
    layer_idx: int,
    expert_idx: int,
    mask: torch.Tensor,
    version: int,
) -> None:
    gain_map = calibration.mxmoe_w1_deltas if projection == "w1" else calibration.mxmoe_w2_deltas
    candidate = build_projection_candidate(
        config,
        projection,
        layer_idx,
        expert_idx,
        mask,
        float(gain_map[layer_idx][expert_idx].item()),
        version,
    )
    if candidate is not None:
        heapq.heappush(heap, candidate)


def push_bundle_candidates(
    heap: list[RescueCandidate],
    router_affinity_cache: dict[int, LayerMetricBundle],
    router_orders: dict[str, dict[int, dict[int, torch.Tensor]]],
    router_pointers: dict[str, dict[int, dict[int, int]]],
    config: Any,
    projection: str,
    layer_idx: int,
    expert_idx: int,
    mask: torch.Tensor,
    version: int,
) -> None:
    bundle = router_affinity_cache[layer_idx]
    score_tensor = bundle.w1_pair_scores[expert_idx] if projection == "w1" else bundle.w2_channel_scores[expert_idx]
    candidate = build_bundle_candidate(
        config,
        projection,
        layer_idx,
        expert_idx,
        mask,
        score_tensor,
        router_orders[projection][layer_idx][expert_idx],
        router_pointers[projection][layer_idx][expert_idx],
        version,
    )
    if candidate is not None:
        heapq.heappush(heap, candidate)


def summarize_actions(selected_actions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "projection_rescue_w1": 0,
        "projection_rescue_w2": 0,
        "channel_bundle_rescue_w1": 0,
        "channel_bundle_rescue_w2": 0,
    }
    bytes_by_type = {key: 0 for key in counts}
    fp8_weights_by_type = {key: 0 for key in counts}
    for action in selected_actions:
        key = f"{action['action_kind']}_{action['projection']}"
        counts[key] += 1
        bytes_by_type[key] += int(action["bytes_added"])
        fp8_weights_by_type[key] += int(action["fp8_weights_added"])
    return {
        "selected_action_count": int(len(selected_actions)),
        "selected_action_type_counts": counts,
        "selected_action_bytes": bytes_by_type,
        "selected_action_fp8_weights": fp8_weights_by_type,
    }


def apply_rescue_knapsack(
    calibration: Any,
    router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    base_w1_masks: dict[int, dict[int, torch.Tensor]],
    base_w2_masks: dict[int, dict[int, torch.Tensor]],
    extra_budget_gb: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_masks = clone_masks(base_w1_masks)
    w2_masks = clone_masks(base_w2_masks)
    router_orders = {
        "w1": build_router_orders(router_affinity_cache, "w1"),
        "w2": build_router_orders(router_affinity_cache, "w2"),
    }
    router_pointers = {
        "w1": {
            layer_idx: {expert_idx: 0 for expert_idx in expert_map}
            for layer_idx, expert_map in router_orders["w1"].items()
        },
        "w2": {
            layer_idx: {expert_idx: 0 for expert_idx in expert_map}
            for layer_idx, expert_map in router_orders["w2"].items()
        },
    }
    versions = {
        "w1": {
            layer_idx: {expert_idx: 0 for expert_idx in expert_map}
            for layer_idx, expert_map in router_orders["w1"].items()
        },
        "w2": {
            layer_idx: {expert_idx: 0 for expert_idx in expert_map}
            for layer_idx, expert_map in router_orders["w2"].items()
        },
    }
    heap: list[RescueCandidate] = []
    selected_actions: list[dict[str, Any]] = []
    extra_budget_bytes = int(round(extra_budget_gb * 1e9))
    remaining_budget_bytes = int(extra_budget_bytes)

    for layer_idx in range(config.num_hidden_layers):
        routing_counts = calibration.routing_counts[layer_idx]
        for expert_idx in range(config.num_experts):
            if int(routing_counts[expert_idx].item()) <= 0:
                continue
            push_projection_candidates(heap, calibration, config, "w1", layer_idx, expert_idx, w1_masks[layer_idx][expert_idx], versions["w1"][layer_idx][expert_idx])
            push_projection_candidates(heap, calibration, config, "w2", layer_idx, expert_idx, w2_masks[layer_idx][expert_idx], versions["w2"][layer_idx][expert_idx])
            push_bundle_candidates(heap, router_affinity_cache, router_orders, router_pointers, config, "w1", layer_idx, expert_idx, w1_masks[layer_idx][expert_idx], versions["w1"][layer_idx][expert_idx])
            push_bundle_candidates(heap, router_affinity_cache, router_orders, router_pointers, config, "w2", layer_idx, expert_idx, w2_masks[layer_idx][expert_idx], versions["w2"][layer_idx][expert_idx])

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

        if candidate.action_kind == "projection_rescue":
            current_mask.fill_(True)
            bundle_count = int(current_mask.numel()) - selected_before
        else:
            if not candidate.selected_indices:
                continue
            index_tensor = torch.as_tensor(candidate.selected_indices, dtype=torch.long)
            if bool(current_mask[index_tensor].all()):
                continue
            current_mask[index_tensor] = True
            router_pointers[candidate.projection][candidate.layer_idx][candidate.expert_idx] = int(candidate.next_pointer)
            bundle_count = len(candidate.selected_indices)

        selected_after = int(current_mask.sum().item())
        if selected_after <= selected_before:
            continue

        remaining_budget_bytes -= int(candidate.bytes_added)
        action_record: dict[str, Any] = {
            "step": int(len(selected_actions) + 1),
            "action_kind": candidate.action_kind,
            "projection": candidate.projection,
            "layer_idx": int(candidate.layer_idx),
            "expert_idx": int(candidate.expert_idx),
            "gain_proxy": round(float(candidate.gain_proxy), 6),
            "bytes_added": int(candidate.bytes_added),
            "fp8_weights_added": int(candidate.fp8_weights_added),
            "selected_before": int(selected_before),
            "selected_after": int(selected_after),
            "new_units_added": int(bundle_count),
            "remaining_budget_bytes": int(remaining_budget_bytes),
            "gain_per_byte": round(float(candidate.gain_proxy) / float(max(candidate.bytes_added, 1)), 12),
        }
        if candidate.action_kind == "channel_bundle_rescue":
            action_record["selected_indices"] = [int(index) for index in candidate.selected_indices]
            action_record["bundle_size"] = int(len(candidate.selected_indices))
        selected_actions.append(action_record)

        versions[candidate.projection][candidate.layer_idx][candidate.expert_idx] += 1
        new_version = versions[candidate.projection][candidate.layer_idx][candidate.expert_idx]
        push_projection_candidates(
            heap,
            calibration,
            config,
            candidate.projection,
            candidate.layer_idx,
            candidate.expert_idx,
            current_mask,
            new_version,
        )
        push_bundle_candidates(
            heap,
            router_affinity_cache,
            router_orders,
            router_pointers,
            config,
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
        "extra_budget_gb": float(extra_budget_gb),
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


def evaluate_and_record_plan(
    payload: dict[str, Any],
    plan: EvalPlan,
    extras: dict[str, Any],
    output_json: Path,
    config: Any,
    total_expert_elems: int,
    test_ids: torch.Tensor,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    result_rows = payload.setdefault("results", {})
    if plan.name in result_rows:
        print(f"[skip] {plan.name} already present", flush=True)
        return result_rows[plan.name]

    print(f"\n=== Eval: {plan.name} ===", flush=True)
    start_time = time.time()
    ppl, nll, nsamples = evaluate_plan(plan, test_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - start_time
    row = {
        "description": plan.description,
        "mode": plan.mode,
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "memory_gb": round(plan.memory_gb, 3),
        "fp8_weights": int(plan.fp8_weights),
        "fp8_fraction": round(float(plan.fp8_weights) / float(max(total_expert_elems, 1)), 6),
        "w1_pair_fraction": round(total_pair_fraction(config, plan.w1_pair_masks or {}), 6),
        "w2_channel_fraction": round(total_channel_fraction(config, plan.w2_channel_masks or {}), 6),
        "eval_chunks": int(nsamples),
        "seqlen": SEQLEN,
        "time_s": round(elapsed, 1),
        **extras,
    }
    result_rows[plan.name] = row
    atomic_json_dump(output_json, payload)
    print(
        f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_ref = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 180, flush=True)
    print("proper_iter16_counterfactual_rescue | greedy marginal rescue over strong bases | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 180, flush=True)
    print(
        f"{'Config':<28} {'PPL':>10} {'dBest':>10} {'Memory GB':>12} {'Base':<46} {'Extra GB':>10} {'Actions':>10}",
        flush=True,
    )
    print("-" * 180, flush=True)
    for name, row in ordered:
        d_best = "-" if best_ref is None else f"{float(row['ppl']) - best_ref:+.4f}"
        print(
            f"{name:<28} {float(row['ppl']):>10.4f} {d_best:>10} {float(row['memory_gb']):>12.3f} {str(row['base_name']):<46} {float(row['realized_extra_gb']):>10.3f} {int(row['selected_action_count']):>10}",
            flush=True,
        )


def build_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    rows = payload["results"]
    ordered = sorted(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_name, best_row = ordered[0]
    previous_best = min((row["ppl"] for row in references.values()), default=None)
    table_lines = [
        "| Config | Base | PPL | Delta vs prev best | Memory (GB) | Extra GB | Actions |",
        "|--------|------|-----|--------------------|-------------|----------|---------|",
    ]
    for name, row in ordered:
        delta_prev = "-" if previous_best is None else f"{float(row['ppl']) - previous_best:+.4f}"
        table_lines.append(
            f"| {name} | {row['base_name']} | {float(row['ppl']):.4f} | {delta_prev} | {float(row['memory_gb']):.3f} | {float(row['realized_extra_gb']):.3f} | {int(row['selected_action_count'])} |"
        )

    if previous_best is None:
        insight = f"Best rescue config is `{best_name}` at PPL {float(best_row['ppl']):.4f}."
    else:
        delta = float(best_row["ppl"]) - previous_best
        verb = "beats" if delta < 0.0 else "still trails"
        insight = (
            f"Best rescue config is `{best_name}` at PPL {float(best_row['ppl']):.4f}; it {verb} the prior best by {abs(delta):.4f} PPL. "
            f"The greedy rescue path selected {int(best_row['selected_action_count'])} actions for {float(best_row['realized_extra_gb']):.3f} GB of extra expert memory."
        )

    return "\n".join([
        SECTION_MARKER,
        "**Approach**: Reused the strongest existing base masks from Iterations 1, 7, and 14, then ran a counterfactual rescue knapsack on top of each base. Candidate actions were full W1/W2 projection rescues scored by MxMoE output-perturbation deltas and fixed-size W1/W2 channel bundles scored by the next-unused router-affinity cache values. The greedy selection used proxy-gain per added byte and stopped when the extra expert-memory budget was exhausted.",
        f"**Eval**: Full WikiText-2 test ({payload['metadata']['evaluation']['total_tokens']} tokens, {payload['metadata']['evaluation']['nsamples']} chunks of {SEQLEN}), BF16 `F.linear`, FP32 loss, `loss.float() * seqlen`.",
        "**Result**:",
        "\n".join(table_lines),
        f"**Insight**: {insight}",
        "**Next**: If the best rescue still misses the frontier, try the same greedy rescue restricted to W2-only actions or let projection rescues use a residual gain proxy that discounts channels already promoted by the base mask.",
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
            "experiment": "Iteration 16 counterfactual rescue knapsack over the strongest existing per-channel bases",
            "candidate_actions": {
                "projection_rescue": ["w2_full_block", "w1_full_block"],
                "channel_bundle_rescue": {
                    "w2_channels_per_action": W2_BUNDLE_CHANNELS,
                    "w1_pairs_per_action": W1_BUNDLE_PAIRS,
                },
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

    base_masks = build_base_masks(calibration, router_affinity_cache, text_config, total_expert_elems)
    configs = base_action_configs()
    if requested_plans is not None:
        available_names = {config.name for config in configs}
        missing = sorted(requested_plans.difference(available_names))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
        configs = [config for config in configs if config.name in requested_plans]
    payload["metadata"]["requested_plan_order"] = [config.name for config in configs]
    atomic_json_dump(args.output_json, payload)

    for config_row in configs:
        base_w1, base_w2, base_meta = base_masks[config_row.base_name]
        base_fp8_weights = fp8_weights_from_masks(text_config, base_w1, base_w2)
        base_memory_gb = round(float(non_expert_bytes + int((0.5 * total_expert_elems) + (0.5 * base_fp8_weights))) / 1e9, 3)
        rescued_w1, rescued_w2, rescue_meta = apply_rescue_knapsack(
            calibration,
            router_affinity_cache,
            text_config,
            base_w1,
            base_w2,
            config_row.extra_budget_gb,
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
                "base_name": config_row.base_name,
                "base_fp8_weights": int(base_fp8_weights),
                "base_memory_gb": round(float(references.get(config_row.base_name, {}).get("memory_gb", base_memory_gb)), 3),
                "channel_metric_requested": "router_affinity_weighted_qerror_for_bundle_rescues",
                "channel_metric_effective": "router_affinity_weighted_qerror_for_bundle_rescues",
                "channel_metric_fallback_used": False,
                "projection_gain_proxy": "mxmoe_output_perturbation_delta",
                "bundle_gain_proxy": "sum_next_unused_router_affinity_scores",
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

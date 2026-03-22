#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportReturnType=false, reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SPIKE1_PATH = RESULTS_DIR / "spike1_sensitivity.json"
SPIKE2_PATH = RESULTS_DIR / "spike2_metrics.json"
OUTPUT_JSON_PATH = RESULTS_DIR / "spike3_strategies.json"
OUTPUT_PLOT_PATH = RESULTS_DIR / "spike3_strategies.png"

NUM_RANDOM_SEEDS = 10
DEFAULT_BUDGETS = (0.10, 0.25, 0.50, 0.75)
DEFAULT_STRATEGIES = (
    "uniform_sort",
    "global_greedy",
    "freq_weighted",
    "hierarchical",
    "random",
)
@dataclass(frozen=True)
class GroundTruthExpert:
    layer_idx: int
    expert_id: int
    role: str
    token_count: int
    w1: np.ndarray
    w2: np.ndarray


@dataclass(frozen=True)
class CostAggregate:
    raw_mean: float
    normalized_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike1", type=Path, default=SPIKE1_PATH, help="Spike 1 ground-truth JSON path")
    parser.add_argument("--spike2", type=Path, default=SPIKE2_PATH, help="Spike 2 proxy-metric JSON path")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON_PATH, help="Output JSON path")
    parser.add_argument("--output-plot", type=Path, default=OUTPUT_PLOT_PATH, help="Output plot path")
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="FP8 budget fractions",
    )
    parser.add_argument("--random-seeds", type=int, default=NUM_RANDOM_SEEDS, help="Random seeds for random baseline")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(output_path)


def load_ground_truth(spike1_payload: dict[str, Any]) -> list[GroundTruthExpert]:
    experts: list[GroundTruthExpert] = []
    for layer in spike1_payload["layers"]:
        layer_idx = int(layer["layer_idx"])
        for expert in layer["experts"]:
            experts.append(
                GroundTruthExpert(
                    layer_idx=layer_idx,
                    expert_id=int(expert["expert_id"]),
                    role=str(expert["role"]),
                    token_count=int(expert["token_count"]),
                    w1=np.asarray(expert["w1_pair_sensitivity"], dtype=np.float64),
                    w2=np.asarray(expert["w2_channel_sensitivity"], dtype=np.float64),
                )
            )
    return experts


def load_proxy_scores(spike2_payload: dict[str, Any]) -> dict[str, dict[int, dict[int, np.ndarray]]]:
    layers = [int(layer) for layer in spike2_payload["_metadata"]["layers"]]
    num_experts = int(spike2_payload["_metadata"]["num_experts"])
    proxy_scores: dict[str, dict[int, dict[int, np.ndarray]]] = {"w1": {}, "w2": {}}
    for matrix_name in ("w1", "w2"):
        for layer_idx in layers:
            expert_map: dict[int, np.ndarray] = {}
            for expert_id in range(num_experts):
                key = f"layer_{layer_idx}_expert_{expert_id}_{matrix_name}"
                if key not in spike2_payload:
                    raise KeyError(f"Missing proxy metrics for {key}")
                expert_map[expert_id] = np.asarray(spike2_payload[key]["weight_l1"], dtype=np.float64)
            proxy_scores[matrix_name][layer_idx] = expert_map
    return proxy_scores


def build_ground_truth_lookup(
    experts: list[GroundTruthExpert],
) -> dict[str, dict[int, dict[int, np.ndarray]]]:
    lookup: dict[str, dict[int, dict[int, np.ndarray]]] = {"w1": {}, "w2": {}}
    for expert in experts:
        lookup["w1"].setdefault(expert.layer_idx, {})[expert.expert_id] = expert.w1
        lookup["w2"].setdefault(expert.layer_idx, {})[expert.expert_id] = expert.w2
    return lookup


def build_routing_weights(experts: list[GroundTruthExpert]) -> dict[int, dict[int, float]]:
    weights: dict[int, dict[int, float]] = {}
    for expert in experts:
        weights.setdefault(expert.layer_idx, {})[expert.expert_id] = float(expert.token_count)
    return weights


def build_layer_sensitivity_weights(experts: list[GroundTruthExpert]) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, list[float]]] = {"w1": {}, "w2": {}}
    for expert in experts:
        grouped["w1"].setdefault(expert.layer_idx, []).append(float(np.mean(expert.w1)))
        grouped["w2"].setdefault(expert.layer_idx, []).append(float(np.mean(expert.w2)))
    return {
        matrix_name: {
            layer_idx: float(np.mean(values, dtype=np.float64))
            for layer_idx, values in matrix_group.items()
        }
        for matrix_name, matrix_group in grouped.items()
    }


def build_expert_sensitivity_weights(experts: list[GroundTruthExpert]) -> dict[str, dict[int, dict[int, float]]]:
    weights: dict[str, dict[int, dict[int, float]]] = {"w1": {}, "w2": {}}
    for expert in experts:
        weights["w1"].setdefault(expert.layer_idx, {})[expert.expert_id] = float(np.mean(expert.w1))
        weights["w2"].setdefault(expert.layer_idx, {})[expert.expert_id] = float(np.mean(expert.w2))
    return weights


def expand_layer_defaults(
    base_weights: dict[int, dict[int, float]],
    proxy_scores: dict[int, dict[int, np.ndarray]],
) -> dict[int, dict[int, float]]:
    expanded: dict[int, dict[int, float]] = {}
    for layer_idx, layer_scores in proxy_scores.items():
        layer_known = base_weights.get(layer_idx, {})
        default_weight = float(np.mean(list(layer_known.values()), dtype=np.float64)) if layer_known else 1.0
        expanded[layer_idx] = {
            expert_id: float(layer_known.get(expert_id, default_weight))
            for expert_id in layer_scores
        }
    return expanded


def expand_layer_defaults(
    base_weights: dict[int, dict[int, float]],
    proxy_scores: dict[int, dict[int, np.ndarray]],
) -> dict[int, dict[int, float]]:
    expanded: dict[int, dict[int, float]] = {}
    for layer_idx, layer_scores in proxy_scores.items():
        layer_known = base_weights.get(layer_idx, {})
        default_weight = float(np.mean(list(layer_known.values()), dtype=np.float64)) if layer_known else 1.0
        expanded[layer_idx] = {
            expert_id: float(layer_known.get(expert_id, default_weight))
            for expert_id in layer_scores
        }
    return expanded


def budget_to_channels(budget: float, total_channels: int) -> int:
    return int(round(budget * total_channels))


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    channel_count = int(scores.shape[0])
    if k <= 0:
        return np.zeros(channel_count, dtype=bool)
    if k >= channel_count:
        return np.ones(channel_count, dtype=bool)
    threshold_idx = channel_count - k
    top_indices = np.argpartition(scores, threshold_idx)[threshold_idx:]
    mask = np.zeros(channel_count, dtype=bool)
    mask[top_indices] = True
    return mask


def allocate_weighted_counts(total: int, capacities: list[int], weights: list[float]) -> list[int]:
    if len(capacities) != len(weights):
        raise ValueError("Capacities and weights must have the same length")

    allocations = [0] * len(capacities)
    remaining_capacity = list(capacities)
    remaining_total = min(total, sum(capacities))
    active = {idx for idx, capacity in enumerate(capacities) if capacity > 0}

    while active and remaining_total > 0:
        weight_sum = sum(weights[idx] for idx in active)
        if weight_sum <= 0.0:
            ordered = sorted(active)
            for idx in ordered:
                if remaining_total <= 0:
                    break
                equal_share = int(np.ceil(remaining_total / len(ordered)))
                give = min(equal_share, remaining_capacity[idx])
                allocations[idx] += give
                remaining_capacity[idx] -= give
                remaining_total -= give
            active = {idx for idx in active if remaining_capacity[idx] > 0}
            continue

        quotas = {idx: remaining_total * weights[idx] / weight_sum for idx in active}
        saturated = [idx for idx in active if quotas[idx] >= remaining_capacity[idx] - 1e-12]
        if saturated:
            for idx in saturated:
                give = remaining_capacity[idx]
                allocations[idx] += give
                remaining_total -= give
                remaining_capacity[idx] = 0
            active = {idx for idx in active if remaining_capacity[idx] > 0}
            continue

        floors = {idx: min(int(np.floor(quotas[idx])), remaining_capacity[idx]) for idx in active}
        floor_total = sum(floors.values())
        if floor_total == 0:
            ordered = sorted(active, key=lambda idx: (quotas[idx], weights[idx], -idx), reverse=True)
            for idx in ordered:
                if remaining_total <= 0:
                    break
                if remaining_capacity[idx] <= 0:
                    continue
                allocations[idx] += 1
                remaining_capacity[idx] -= 1
                remaining_total -= 1
            active = {idx for idx in active if remaining_capacity[idx] > 0}
            continue

        for idx, give in floors.items():
            allocations[idx] += give
            remaining_capacity[idx] -= give
        remaining_total -= floor_total
        if remaining_total <= 0:
            break

        ordered = sorted(active, key=lambda idx: (quotas[idx] - floors[idx], weights[idx], -idx), reverse=True)
        for idx in ordered:
            if remaining_total <= 0:
                break
            if remaining_capacity[idx] <= 0:
                continue
            allocations[idx] += 1
            remaining_capacity[idx] -= 1
            remaining_total -= 1
        active = {idx for idx in active if remaining_capacity[idx] > 0}

    return allocations


def uniform_sort_masks(layer_scores: dict[int, np.ndarray], budget: float) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for expert_id, scores in layer_scores.items():
        masks[expert_id] = topk_mask(scores, budget_to_channels(budget, int(scores.shape[0])))
    return masks


def global_greedy_masks(layer_scores: dict[int, np.ndarray], budget: float) -> dict[int, np.ndarray]:
    expert_ids = sorted(layer_scores)
    stacked = np.stack([layer_scores[expert_id] for expert_id in expert_ids], axis=0)
    total_fp8 = budget_to_channels(budget, int(stacked.size))
    flat_mask = topk_mask(stacked.reshape(-1), total_fp8).reshape(stacked.shape)
    return {expert_id: flat_mask[row_idx].copy() for row_idx, expert_id in enumerate(expert_ids)}


def weighted_budget_masks(
    layer_scores: dict[int, np.ndarray],
    budget: float,
    expert_weights: dict[int, float],
) -> dict[int, np.ndarray]:
    expert_ids = sorted(layer_scores)
    per_expert_channels = int(next(iter(layer_scores.values())).shape[0])
    total_fp8 = budget_to_channels(budget, per_expert_channels * len(expert_ids))
    capacities = [per_expert_channels] * len(expert_ids)
    weights = [float(expert_weights.get(expert_id, 1.0)) for expert_id in expert_ids]
    allocations = allocate_weighted_counts(total_fp8, capacities, weights)
    return {
        expert_id: topk_mask(layer_scores[expert_id], allocations[row_idx])
        for row_idx, expert_id in enumerate(expert_ids)
    }


def hierarchical_masks(
    all_layer_scores: dict[int, dict[int, np.ndarray]],
    budget: float,
    layer_weights: dict[int, float],
    expert_weights_by_layer: dict[int, dict[int, float]],
) -> dict[int, dict[int, np.ndarray]]:
    layer_ids = sorted(all_layer_scores)
    layer_capacities = [sum(scores.shape[0] for scores in all_layer_scores[layer_idx].values()) for layer_idx in layer_ids]
    total_fp8 = budget_to_channels(budget, sum(layer_capacities))
    layer_allocations = allocate_weighted_counts(
        total_fp8,
        layer_capacities,
        [float(layer_weights.get(layer_idx, 1.0)) for layer_idx in layer_ids],
    )

    layer_masks: dict[int, dict[int, np.ndarray]] = {}
    for allocation, layer_idx in zip(layer_allocations, layer_ids, strict=True):
        layer_scores = all_layer_scores[layer_idx]
        expert_ids = sorted(layer_scores)
        per_expert_channels = int(next(iter(layer_scores.values())).shape[0])
        capacities = [per_expert_channels] * len(expert_ids)
        expert_weights = [
            float(expert_weights_by_layer.get(layer_idx, {}).get(expert_id, 1.0))
            for expert_id in expert_ids
        ]
        expert_allocations = allocate_weighted_counts(allocation, capacities, expert_weights)
        layer_masks[layer_idx] = {
            expert_id: topk_mask(layer_scores[expert_id], expert_allocations[row_idx])
            for row_idx, expert_id in enumerate(expert_ids)
        }
    return layer_masks


def random_masks(layer_scores: dict[int, np.ndarray], budget: float, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    masks: dict[int, np.ndarray] = {}
    for expert_id, scores in layer_scores.items():
        k = budget_to_channels(budget, int(scores.shape[0]))
        mask = np.zeros(int(scores.shape[0]), dtype=bool)
        if k > 0:
            chosen = rng.choice(int(scores.shape[0]), size=k, replace=False)
            mask[chosen] = True
        masks[expert_id] = mask
    return masks


def proxy_cost(layer_scores: dict[int, np.ndarray], fp8_masks: dict[int, np.ndarray]) -> float:
    total = 0.0
    for expert_id, scores in layer_scores.items():
        total += float(np.sum(scores[~fp8_masks[expert_id]], dtype=np.float64))
    return total


def ground_truth_cost(layer_ground_truth: dict[int, np.ndarray], fp8_masks: dict[int, np.ndarray]) -> float:
    total = 0.0
    for expert_id, gt_scores in layer_ground_truth.items():
        total += float(np.sum(gt_scores[~fp8_masks[expert_id]], dtype=np.float64))
    return total


def evaluate_layered_strategy(
    layer_masks: dict[int, dict[int, np.ndarray]],
    proxy_scores: dict[int, dict[int, np.ndarray]],
    gt_scores: dict[int, dict[int, np.ndarray]],
    proxy_baselines: dict[int, float],
    gt_baselines: dict[int, float],
) -> tuple[CostAggregate, CostAggregate]:
    proxy_raw: list[float] = []
    proxy_norm: list[float] = []
    gt_raw: list[float] = []
    gt_norm: list[float] = []

    for layer_idx, masks in layer_masks.items():
        current_proxy_cost = proxy_cost(proxy_scores[layer_idx], masks)
        proxy_raw.append(current_proxy_cost)
        proxy_norm.append(current_proxy_cost / proxy_baselines[layer_idx])

        current_gt_cost = ground_truth_cost(gt_scores[layer_idx], masks)
        gt_raw.append(current_gt_cost)
        gt_norm.append(current_gt_cost / gt_baselines[layer_idx])

    return (
        CostAggregate(
            raw_mean=float(np.mean(proxy_raw, dtype=np.float64)),
            normalized_mean=float(np.mean(proxy_norm, dtype=np.float64)),
        ),
        CostAggregate(
            raw_mean=float(np.mean(gt_raw, dtype=np.float64)),
            normalized_mean=float(np.mean(gt_norm, dtype=np.float64)),
        ),
    )


def evaluate_random_strategy(
    budget: float,
    proxy_scores: dict[int, dict[int, np.ndarray]],
    gt_scores: dict[int, dict[int, np.ndarray]],
    proxy_baselines: dict[int, float],
    gt_baselines: dict[int, float],
    random_seeds: int,
) -> tuple[CostAggregate, CostAggregate]:
    proxy_raw: list[float] = []
    proxy_norm: list[float] = []
    gt_raw: list[float] = []
    gt_norm: list[float] = []

    for seed in range(random_seeds):
        layer_masks = {
            layer_idx: random_masks(layer_scores, budget, seed + layer_idx * 1000)
            for layer_idx, layer_scores in proxy_scores.items()
        }
        proxy_agg, gt_agg = evaluate_layered_strategy(layer_masks, proxy_scores, gt_scores, proxy_baselines, gt_baselines)
        proxy_raw.append(proxy_agg.raw_mean)
        proxy_norm.append(proxy_agg.normalized_mean)
        gt_raw.append(gt_agg.raw_mean)
        gt_norm.append(gt_agg.normalized_mean)

    return (
        CostAggregate(
            raw_mean=float(np.mean(proxy_raw, dtype=np.float64)),
            normalized_mean=float(np.mean(proxy_norm, dtype=np.float64)),
        ),
        CostAggregate(
            raw_mean=float(np.mean(gt_raw, dtype=np.float64)),
            normalized_mean=float(np.mean(gt_norm, dtype=np.float64)),
        ),
    )


def nested_budget_dict() -> dict[str, dict[str, float]]:
    return {strategy: {} for strategy in DEFAULT_STRATEGIES}


def format_budget_key(budget: float) -> str:
    return f"{budget:.2f}".rstrip("0").rstrip(".")


def collect_results(
    budgets: list[float],
    proxy_scores: dict[str, dict[int, dict[int, np.ndarray]]],
    gt_scores: dict[str, dict[int, dict[int, np.ndarray]]],
    routing_weights: dict[int, dict[int, float]],
    layer_sensitivity_weights: dict[str, dict[int, float]],
    expert_sensitivity_weights: dict[str, dict[int, dict[int, float]]],
    random_seeds: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "strategies": list(DEFAULT_STRATEGIES),
        "budgets": budgets,
        "w1_proxy_cost": nested_budget_dict(),
        "w1_proxy_cost_raw_mean": nested_budget_dict(),
        "w1_gt_cost": nested_budget_dict(),
        "w1_gt_cost_raw_mean": nested_budget_dict(),
        "w2_proxy_cost": nested_budget_dict(),
        "w2_proxy_cost_raw_mean": nested_budget_dict(),
        "w2_gt_cost": nested_budget_dict(),
        "w2_gt_cost_raw_mean": nested_budget_dict(),
    }

    for matrix_name in ("w1", "w2"):
        matrix_proxy_scores = proxy_scores[matrix_name]
        matrix_gt_scores = gt_scores[matrix_name]
        proxy_baselines = {
            layer_idx: proxy_cost(layer_scores, {expert_id: np.zeros(scores.shape[0], dtype=bool) for expert_id, scores in layer_scores.items()})
            for layer_idx, layer_scores in matrix_proxy_scores.items()
        }
        gt_baselines = {
            layer_idx: ground_truth_cost(layer_gt_scores, {expert_id: np.zeros(scores.shape[0], dtype=bool) for expert_id, scores in matrix_proxy_scores[layer_idx].items()})
            for layer_idx, layer_gt_scores in matrix_gt_scores.items()
        }

        for budget in budgets:
            budget_key = format_budget_key(budget)

            uniform_masks_by_layer = {
                layer_idx: uniform_sort_masks(layer_scores, budget)
                for layer_idx, layer_scores in matrix_proxy_scores.items()
            }
            proxy_agg, gt_agg = evaluate_layered_strategy(
                uniform_masks_by_layer,
                matrix_proxy_scores,
                matrix_gt_scores,
                proxy_baselines,
                gt_baselines,
            )
            results[f"{matrix_name}_proxy_cost"]["uniform_sort"][budget_key] = proxy_agg.normalized_mean
            results[f"{matrix_name}_proxy_cost_raw_mean"]["uniform_sort"][budget_key] = proxy_agg.raw_mean
            results[f"{matrix_name}_gt_cost"]["uniform_sort"][budget_key] = gt_agg.normalized_mean
            results[f"{matrix_name}_gt_cost_raw_mean"]["uniform_sort"][budget_key] = gt_agg.raw_mean

            greedy_masks_by_layer = {
                layer_idx: global_greedy_masks(layer_scores, budget)
                for layer_idx, layer_scores in matrix_proxy_scores.items()
            }
            proxy_agg, gt_agg = evaluate_layered_strategy(
                greedy_masks_by_layer,
                matrix_proxy_scores,
                matrix_gt_scores,
                proxy_baselines,
                gt_baselines,
            )
            results[f"{matrix_name}_proxy_cost"]["global_greedy"][budget_key] = proxy_agg.normalized_mean
            results[f"{matrix_name}_proxy_cost_raw_mean"]["global_greedy"][budget_key] = proxy_agg.raw_mean
            results[f"{matrix_name}_gt_cost"]["global_greedy"][budget_key] = gt_agg.normalized_mean
            results[f"{matrix_name}_gt_cost_raw_mean"]["global_greedy"][budget_key] = gt_agg.raw_mean

            freq_masks_by_layer = {
                layer_idx: weighted_budget_masks(layer_scores, budget, routing_weights.get(layer_idx, {}))
                for layer_idx, layer_scores in matrix_proxy_scores.items()
            }
            proxy_agg, gt_agg = evaluate_layered_strategy(
                freq_masks_by_layer,
                matrix_proxy_scores,
                matrix_gt_scores,
                proxy_baselines,
                gt_baselines,
            )
            results[f"{matrix_name}_proxy_cost"]["freq_weighted"][budget_key] = proxy_agg.normalized_mean
            results[f"{matrix_name}_proxy_cost_raw_mean"]["freq_weighted"][budget_key] = proxy_agg.raw_mean
            results[f"{matrix_name}_gt_cost"]["freq_weighted"][budget_key] = gt_agg.normalized_mean
            results[f"{matrix_name}_gt_cost_raw_mean"]["freq_weighted"][budget_key] = gt_agg.raw_mean

            hierarchical_masks_by_layer = hierarchical_masks(
                matrix_proxy_scores,
                budget,
                layer_sensitivity_weights[matrix_name],
                expert_sensitivity_weights[matrix_name],
            )
            proxy_agg, gt_agg = evaluate_layered_strategy(
                hierarchical_masks_by_layer,
                matrix_proxy_scores,
                matrix_gt_scores,
                proxy_baselines,
                gt_baselines,
            )
            results[f"{matrix_name}_proxy_cost"]["hierarchical"][budget_key] = proxy_agg.normalized_mean
            results[f"{matrix_name}_proxy_cost_raw_mean"]["hierarchical"][budget_key] = proxy_agg.raw_mean
            results[f"{matrix_name}_gt_cost"]["hierarchical"][budget_key] = gt_agg.normalized_mean
            results[f"{matrix_name}_gt_cost_raw_mean"]["hierarchical"][budget_key] = gt_agg.raw_mean

            proxy_agg, gt_agg = evaluate_random_strategy(
                budget,
                matrix_proxy_scores,
                matrix_gt_scores,
                proxy_baselines,
                gt_baselines,
                random_seeds,
            )
            results[f"{matrix_name}_proxy_cost"]["random"][budget_key] = proxy_agg.normalized_mean
            results[f"{matrix_name}_proxy_cost_raw_mean"]["random"][budget_key] = proxy_agg.raw_mean
            results[f"{matrix_name}_gt_cost"]["random"][budget_key] = gt_agg.normalized_mean
            results[f"{matrix_name}_gt_cost_raw_mean"]["random"][budget_key] = gt_agg.raw_mean

    return results


def mean_over_budgets(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values()), dtype=np.float64))


def best_strategy(cost_table: dict[str, dict[str, float]]) -> str:
    return min(DEFAULT_STRATEGIES, key=lambda strategy: (mean_over_budgets(cost_table[strategy]), strategy))


def strategy_label(strategy: str) -> str:
    return {
        "uniform_sort": "Uniform sort",
        "global_greedy": "Global greedy",
        "freq_weighted": "Freq weighted",
        "hierarchical": "Hierarchical",
        "random": "Random",
    }[strategy]


def build_analysis(results: dict[str, Any]) -> str:
    w2_proxy = results["w2_proxy_cost"]
    w2_gt = results["w2_gt_cost"]
    budget_keys = [format_budget_key(float(budget)) for budget in results["budgets"]]
    proxy_best = best_strategy(w2_proxy)
    gt_best = best_strategy(w2_gt)

    proxy_wins = [
        min(DEFAULT_STRATEGIES, key=lambda strategy: (w2_proxy[strategy][budget_key], strategy))
        for budget_key in budget_keys
    ]
    gt_wins = [
        min(DEFAULT_STRATEGIES, key=lambda strategy: (w2_gt[strategy][budget_key], strategy))
        for budget_key in budget_keys
    ]

    uniform_vs_greedy = mean_over_budgets(w2_gt["uniform_sort"]) - mean_over_budgets(w2_gt["global_greedy"])
    freq_vs_uniform = mean_over_budgets(w2_gt["uniform_sort"]) - mean_over_budgets(w2_gt["freq_weighted"])
    hier_vs_greedy = mean_over_budgets(w2_gt["hierarchical"]) - mean_over_budgets(w2_gt["global_greedy"])
    random_gap = mean_over_budgets(w2_gt["random"]) - mean_over_budgets(w2_gt[gt_best])
    proxy_gt_match = proxy_wins == gt_wins

    greedy_phrase = (
        f"Global greedy improves over uniform sort by {uniform_vs_greedy:.4f} normalized GT cost on average"
        if uniform_vs_greedy >= 0.0
        else f"Global greedy trails uniform sort by {abs(uniform_vs_greedy):.4f} normalized GT cost on average"
    )
    freq_phrase = (
        f"frequency weighting improves over uniform sort by {freq_vs_uniform:.4f}"
        if freq_vs_uniform >= 0.0
        else f"frequency weighting trails uniform sort by {abs(freq_vs_uniform):.4f}"
    )
    hier_phrase = (
        f"hierarchical improves over global greedy by {abs(hier_vs_greedy):.4f}"
        if hier_vs_greedy <= 0.0
        else f"hierarchical trails global greedy by {hier_vs_greedy:.4f}"
    )

    return (
        f"On W2, proxy favors {strategy_label(proxy_best).lower()} while ground truth favors "
        f"{strategy_label(gt_best).lower()}. {greedy_phrase}; {freq_phrase}; {hier_phrase}. "
        f"Random remains {random_gap:.4f} behind the best GT strategy. "
        f"Proxy/GT budget-wise winners match={proxy_gt_match}."
    )


def save_plot(results: dict[str, Any], output_path: Path) -> None:
    budgets = [float(budget) for budget in results["budgets"]]
    budget_labels = [f"{int(round(budget * 100))}%" for budget in budgets]
    colors = {
        "uniform_sort": "#1f77b4",
        "global_greedy": "#d62728",
        "freq_weighted": "#2ca02c",
        "hierarchical": "#ff7f0e",
        "random": "#7f7f7f",
    }

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4.5), sharex=True, sharey=True)
    panels = [
        (axes[0], "w2_proxy_cost", "W2 proxy cost (weight_l1)"),
        (axes[1], "w2_gt_cost", "W2 ground-truth cost"),
    ]

    for axis, key, title in panels:
        for strategy in DEFAULT_STRATEGIES:
            values = [results[key][strategy][format_budget_key(budget)] for budget in budgets]
            axis.plot(
                budget_labels,
                values,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=colors[strategy],
                label=strategy_label(strategy),
            )
        axis.set_title(title)
        axis.set_xlabel("FP8 budget")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Normalized sensitivity cost")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_table(results: dict[str, Any]) -> None:
    print("W2 assignment strategies (mean across layers; normalized cost, raw mean in parentheses)")
    for budget in results["budgets"]:
        budget_key = format_budget_key(float(budget))
        print(f"\nBudget {int(round(float(budget) * 100)):>2d}%")
        print(f"{'strategy':<16} {'proxy':>22} {'ground_truth':>22}")
        for strategy in DEFAULT_STRATEGIES:
            proxy_norm = results["w2_proxy_cost"][strategy][budget_key]
            proxy_raw = results["w2_proxy_cost_raw_mean"][strategy][budget_key]
            gt_norm = results["w2_gt_cost"][strategy][budget_key]
            gt_raw = results["w2_gt_cost_raw_mean"][strategy][budget_key]
            print(
                f"{strategy:<16} {proxy_norm:>8.4f} ({proxy_raw:>10.2f}) "
                f"{gt_norm:>8.4f} ({gt_raw:>10.4f})"
            )

    print("\nW1 proxy-only view (mean across layers; normalized cost)")
    header = " ".join(f"{int(round(float(budget) * 100)):>7d}%" for budget in results["budgets"])
    print(f"{'strategy':<16} {header}")
    for strategy in DEFAULT_STRATEGIES:
        values = [results["w1_proxy_cost"][strategy][format_budget_key(float(budget))] for budget in results["budgets"]]
        formatted = " ".join(f"{value:>8.4f}" for value in values)
        print(f"{strategy:<16} {formatted}")


def main() -> None:
    args = parse_args()
    spike1_payload = load_json(args.spike1)
    spike2_payload = load_json(args.spike2)

    budgets = [float(budget) for budget in args.budgets]
    if any(budget <= 0.0 or budget >= 1.0 for budget in budgets):
        raise ValueError(f"Budgets must be in (0, 1), got {budgets}")

    ground_truth_experts = load_ground_truth(spike1_payload)
    proxy_scores = load_proxy_scores(spike2_payload)
    gt_scores = build_ground_truth_lookup(ground_truth_experts)
    routing_weights = build_routing_weights(ground_truth_experts)
    layer_sensitivity_weights = build_layer_sensitivity_weights(ground_truth_experts)
    expert_sensitivity_weights = build_expert_sensitivity_weights(ground_truth_experts)
    routing_weights_full = expand_layer_defaults(routing_weights, proxy_scores["w2"])
    expert_sensitivity_weights_full = {
        matrix_name: expand_layer_defaults(expert_sensitivity_weights[matrix_name], proxy_scores[matrix_name])
        for matrix_name in ("w1", "w2")
    }

    results = collect_results(
        budgets=budgets,
        proxy_scores=proxy_scores,
        gt_scores=gt_scores,
        routing_weights=routing_weights_full,
        layer_sensitivity_weights=layer_sensitivity_weights,
        expert_sensitivity_weights=expert_sensitivity_weights_full,
        random_seeds=int(args.random_seeds),
    )
    results["metadata"] = {
        "spike1_path": str(args.spike1),
        "spike2_path": str(args.spike2),
        "layers": sorted(proxy_scores["w2"].keys()),
        "random_seeds": int(args.random_seeds),
        "proxy_metric": "weight_l1",
    }
    results["best_strategy_w2"] = best_strategy(results["w2_gt_cost"])
    results["best_strategy_w2_proxy"] = best_strategy(results["w2_proxy_cost"])
    results["analysis"] = build_analysis(results)

    atomic_json_dump(results, args.output_json)
    save_plot(results, args.output_plot)
    print_table(results)
    print(f"\nBest W2 strategy by GT cost: {results['best_strategy_w2']}")
    print(results["analysis"])


if __name__ == "__main__":
    main()

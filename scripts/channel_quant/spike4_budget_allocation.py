#!/usr/bin/env python3
# pyright: basic, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SPIKE2_PATH = RESULTS_DIR / "spike2_metrics.json"
SPIKE1_PATH = RESULTS_DIR / "spike1_sensitivity.json"
OUTPUT_JSON_PATH = RESULTS_DIR / "spike4_budget.json"
OUTPUT_LAYER_DISTRIBUTION_PATH = RESULTS_DIR / "spike4_per_layer_distribution.png"
OUTPUT_EXPERT_RANKING_PATH = RESULTS_DIR / "spike4_expert_ranking.png"
OUTPUT_CROSS_LAYER_SHAPE_PATH = RESULTS_DIR / "spike4_cross_layer_shape.png"
OUTPUT_BUDGET_COMPARISON_PATH = RESULTS_DIR / "spike4_budget_comparison.png"
OUTPUT_CHANNEL_CONCENTRATION_PATH = RESULTS_DIR / "spike4_channel_concentration.png"

LAYER_IDS = (5, 20, 35)
NUM_EXPERTS = 256
W1_CHANNELS = 512
W2_CHANNELS = 2048
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 512
W1_COST_WEIGHTS = HIDDEN_SIZE * 2
W2_COST_WEIGHTS = MOE_INTERMEDIATE_SIZE
BUDGET_LEVELS = (0.10, 0.25, 0.50, 0.75)


@dataclass(frozen=True)
class LayerData:
    layer_idx: int
    w1: np.ndarray
    w2: np.ndarray

    @property
    def expert_w1_mean(self) -> np.ndarray:
        return self.w1.mean(axis=1)

    @property
    def expert_w2_mean(self) -> np.ndarray:
        return self.w2.mean(axis=1)

    @property
    def expert_total_score(self) -> np.ndarray:
        return self.w1.sum(axis=1) + self.w2.sum(axis=1)

    @property
    def layer_total_score(self) -> float:
        return float(self.w1.sum() + self.w2.sum())

    @property
    def layer_total_cost(self) -> int:
        return int(self.w1.size * W1_COST_WEIGHTS + self.w2.size * W2_COST_WEIGHTS)


@dataclass(frozen=True)
class BudgetRun:
    name: str
    total_sensitivity: float
    remaining_fraction: float
    promoted_sensitivity: float
    actual_budget_fraction: float
    layer_budgets: list[float]
    layer_budget_shares: list[float]
    layer_remaining_sensitivity: list[float]
    layer_promoted_sensitivity: list[float]
    promoted_units: dict[str, int]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def layer_key(layer_idx: int, expert_idx: int, projection: str) -> str:
    return f"layer_{layer_idx}_expert_{expert_idx}_{projection}"


def load_layer_data(payload: dict[str, Any]) -> dict[int, LayerData]:
    layers: dict[int, LayerData] = {}
    for layer_idx in LAYER_IDS:
        w1 = np.zeros((NUM_EXPERTS, W1_CHANNELS), dtype=np.float64)
        w2 = np.zeros((NUM_EXPERTS, W2_CHANNELS), dtype=np.float64)
        for expert_idx in range(NUM_EXPERTS):
            w1_metrics = payload[layer_key(layer_idx, expert_idx, "w1")]
            w2_metrics = payload[layer_key(layer_idx, expert_idx, "w2")]
            w1_values = np.asarray(w1_metrics["weight_l1"], dtype=np.float64)
            w2_values = np.asarray(w2_metrics["weight_l1"], dtype=np.float64)
            if w1_values.shape != (W1_CHANNELS,):
                raise ValueError(f"Expected W1 length {W1_CHANNELS}, found {w1_values.shape} for layer {layer_idx} expert {expert_idx}")
            if w2_values.shape != (W2_CHANNELS,):
                raise ValueError(f"Expected W2 length {W2_CHANNELS}, found {w2_values.shape} for layer {layer_idx} expert {expert_idx}")
            w1[expert_idx] = w1_values
            w2[expert_idx] = w2_values
        layers[layer_idx] = LayerData(layer_idx=layer_idx, w1=w1, w2=w2)
    return layers


def cumulative_share_desc(values: np.ndarray) -> np.ndarray:
    ranked = np.sort(values)[::-1]
    total = ranked.sum()
    if total <= 0:
        return np.zeros_like(ranked)
    return np.cumsum(ranked) / total


def experts_needed_for_share(values: np.ndarray, share: float) -> int:
    cumulative = cumulative_share_desc(values)
    if cumulative.size == 0:
        return 0
    return int(np.searchsorted(cumulative, share, side="left") + 1)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_values.size
    x = np.linspace(0.0, 1.0, n + 1)
    total = sorted_values.sum()
    if total <= 0:
        return x, np.linspace(0.0, 1.0, n + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(sorted_values) / total])
    return x, cumulative


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_values.size
    total = sorted_values.sum()
    if total <= 0 or n == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - (n + 1.0) / n)


def top_fraction_capture(values: np.ndarray, fraction: float) -> float:
    ranked = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = ranked.sum()
    if total <= 0:
        return 0.0
    take = max(1, int(np.ceil(ranked.size * fraction)))
    return float(ranked[:take].sum() / total)


def make_units(layers: dict[int, LayerData]) -> dict[int, dict[str, np.ndarray]]:
    units: dict[int, dict[str, np.ndarray]] = {}
    for layer_idx, layer in layers.items():
        scores = np.concatenate([layer.w1.reshape(-1), layer.w2.reshape(-1)])
        costs = np.concatenate(
            [
                np.full(layer.w1.size, W1_COST_WEIGHTS, dtype=np.int64),
                np.full(layer.w2.size, W2_COST_WEIGHTS, dtype=np.int64),
            ]
        )
        kinds = np.concatenate(
            [
                np.full(layer.w1.size, 1, dtype=np.int8),
                np.full(layer.w2.size, 2, dtype=np.int8),
            ]
        )
        units[layer_idx] = {
            "scores": scores,
            "costs": costs,
            "kinds": kinds,
            "order": np.argsort(scores)[::-1],
        }
    return units


def capped_proportional_costs(total_budget: int, layer_costs: dict[int, int], weights: dict[int, float]) -> dict[int, float]:
    budgets = {layer_idx: 0.0 for layer_idx in layer_costs}
    remaining = float(total_budget)
    active = set(layer_costs)
    while active and remaining > 1e-9:
        total_weight = sum(max(weights[layer_idx], 0.0) for layer_idx in active)
        if total_weight <= 0.0:
            even_share = remaining / len(active)
            share_map = {layer_idx: even_share for layer_idx in active}
        else:
            share_map = {layer_idx: remaining * weights[layer_idx] / total_weight for layer_idx in active}

        saturated: set[int] = set()
        spent_this_round = 0.0
        for layer_idx in active:
            room = layer_costs[layer_idx] - budgets[layer_idx]
            addition = min(room, share_map[layer_idx])
            budgets[layer_idx] += addition
            spent_this_round += addition
            if layer_costs[layer_idx] - budgets[layer_idx] <= 1e-9:
                saturated.add(layer_idx)
        if spent_this_round <= 1e-9:
            break
        remaining -= spent_this_round
        active -= saturated
    return budgets


def allocate_independent(
    units: dict[int, dict[str, np.ndarray]],
    layer_caps: dict[int, float],
    layer_total_scores: dict[int, float],
    total_cost: int,
    total_score: float,
    name: str,
) -> BudgetRun:
    layer_spend = {layer_idx: 0 for layer_idx in layer_caps}
    layer_promoted = {layer_idx: 0.0 for layer_idx in layer_caps}
    promoted_w1 = 0
    promoted_w2 = 0

    for layer_idx in layer_caps:
        layer_units = units[layer_idx]
        scores = layer_units["scores"]
        costs = layer_units["costs"]
        kinds = layer_units["kinds"]
        for unit_idx in layer_units["order"]:
            cost = int(costs[unit_idx])
            if layer_spend[layer_idx] + cost > layer_caps[layer_idx] + 1e-9:
                continue
            layer_spend[layer_idx] += cost
            layer_promoted[layer_idx] += float(scores[unit_idx])
            if kinds[unit_idx] == 1:
                promoted_w1 += 1
            else:
                promoted_w2 += 1

    promoted_sensitivity = float(sum(layer_promoted.values()))
    total_spend = sum(layer_spend.values())
    layer_cost_map = {layer_idx: int(units[layer_idx]["costs"].sum()) for layer_idx in layer_caps}
    layer_budgets = [layer_spend[layer_idx] / layer_cost_map[layer_idx] for layer_idx in LAYER_IDS]
    actual_budget_fraction = total_spend / total_cost
    layer_budget_shares = [layer_spend[layer_idx] / total_spend if total_spend else 0.0 for layer_idx in LAYER_IDS]
    layer_remaining = [layer_total_scores[layer_idx] - layer_promoted[layer_idx] for layer_idx in LAYER_IDS]
    layer_promoted_values = [layer_promoted[layer_idx] for layer_idx in LAYER_IDS]
    total_remaining = total_score - promoted_sensitivity
    return BudgetRun(
        name=name,
        total_sensitivity=float(total_remaining),
        remaining_fraction=float(total_remaining / total_score),
        promoted_sensitivity=promoted_sensitivity,
        actual_budget_fraction=float(actual_budget_fraction),
        layer_budgets=[float(value) for value in layer_budgets],
        layer_budget_shares=[float(value) for value in layer_budget_shares],
        layer_remaining_sensitivity=[float(value) for value in layer_remaining],
        layer_promoted_sensitivity=[float(value) for value in layer_promoted_values],
        promoted_units={"w1": promoted_w1, "w2": promoted_w2},
    )


def allocate_global(
    units: dict[int, dict[str, np.ndarray]],
    global_budget: int,
    layer_total_scores: dict[int, float],
    total_cost: int,
    total_score: float,
    name: str,
) -> BudgetRun:
    layer_spend = {layer_idx: 0 for layer_idx in units}
    layer_promoted = {layer_idx: 0.0 for layer_idx in units}
    promoted_w1 = 0
    promoted_w2 = 0
    total_spend = 0

    merged: list[tuple[float, int, int, int]] = []
    for layer_idx, layer_units in units.items():
        scores = layer_units["scores"]
        costs = layer_units["costs"]
        kinds = layer_units["kinds"]
        for unit_idx in layer_units["order"]:
            merged.append((float(scores[unit_idx]), int(costs[unit_idx]), int(kinds[unit_idx]), layer_idx))
    merged.sort(key=lambda item: item[0], reverse=True)

    for score, cost, kind, layer_idx in merged:
        if total_spend + cost > global_budget:
            continue
        total_spend += cost
        layer_spend[layer_idx] += cost
        layer_promoted[layer_idx] += score
        if kind == 1:
            promoted_w1 += 1
        else:
            promoted_w2 += 1

    layer_cost_map = {layer_idx: int(units[layer_idx]["costs"].sum()) for layer_idx in units}
    layer_budgets = [layer_spend[layer_idx] / layer_cost_map[layer_idx] for layer_idx in LAYER_IDS]
    layer_budget_shares = [layer_spend[layer_idx] / total_spend if total_spend else 0.0 for layer_idx in LAYER_IDS]
    layer_remaining = [layer_total_scores[layer_idx] - layer_promoted[layer_idx] for layer_idx in LAYER_IDS]
    layer_promoted_values = [layer_promoted[layer_idx] for layer_idx in LAYER_IDS]
    promoted_sensitivity = float(sum(layer_promoted.values()))
    total_remaining = total_score - promoted_sensitivity
    return BudgetRun(
        name=name,
        total_sensitivity=float(total_remaining),
        remaining_fraction=float(total_remaining / total_score),
        promoted_sensitivity=promoted_sensitivity,
        actual_budget_fraction=float(total_spend / total_cost),
        layer_budgets=[float(value) for value in layer_budgets],
        layer_budget_shares=[float(value) for value in layer_budget_shares],
        layer_remaining_sensitivity=[float(value) for value in layer_remaining],
        layer_promoted_sensitivity=[float(value) for value in layer_promoted_values],
        promoted_units={"w1": promoted_w1, "w2": promoted_w2},
    )


def run_budget_allocations(layers: dict[int, LayerData]) -> dict[str, dict[str, BudgetRun]]:
    units = make_units(layers)
    layer_total_scores = {layer_idx: layer.layer_total_score for layer_idx, layer in layers.items()}
    layer_total_costs = {layer_idx: layer.layer_total_cost for layer_idx, layer in layers.items()}
    total_score = float(sum(layer_total_scores.values()))
    total_cost = int(sum(layer_total_costs.values()))
    sensitivity_weights = {layer_idx: layer_total_scores[layer_idx] for layer_idx in layers}
    results: dict[str, dict[str, BudgetRun]] = {}

    for budget in BUDGET_LEVELS:
        total_budget_cost = int(round(total_cost * budget))
        uniform_caps = {layer_idx: layer_total_costs[layer_idx] * budget for layer_idx in layers}
        proportional_caps = capped_proportional_costs(total_budget_cost, layer_total_costs, sensitivity_weights)
        uniform_run = allocate_independent(units, uniform_caps, layer_total_scores, total_cost, total_score, "uniform")
        proportional_run = allocate_independent(units, proportional_caps, layer_total_scores, total_cost, total_score, "proportional")
        marginal_run = allocate_global(units, total_budget_cost, layer_total_scores, total_cost, total_score, "marginal_benefit")

        results[f"{budget:.2f}"] = {
            "uniform": uniform_run,
            "proportional": proportional_run,
            "marginal_benefit": marginal_run,
        }

    return results


def analyze_expert_ranking(layer: LayerData) -> dict[str, Any]:
    scores = layer.expert_total_score
    order = np.argsort(scores)[::-1]
    ranked_scores = scores[order]
    cumulative = np.cumsum(ranked_scores) / ranked_scores.sum()
    top10 = [
        {"expert_id": int(expert_id), "score": float(scores[expert_id])}
        for expert_id in order[:10]
    ]
    bottom10 = [
        {"expert_id": int(expert_id), "score": float(scores[expert_id])}
        for expert_id in order[-10:][::-1]
    ]
    return {
        "top10": top10,
        "bottom10": bottom10,
        "top_10pct_experts_capture_pct": float(cumulative[max(0, int(np.ceil(NUM_EXPERTS * 0.10)) - 1)]),
        "top_25pct_experts_capture_pct": float(cumulative[max(0, int(np.ceil(NUM_EXPERTS * 0.25)) - 1)]),
        "experts_for_50pct": experts_needed_for_share(scores, 0.50),
        "experts_for_80pct": experts_needed_for_share(scores, 0.80),
        "experts_for_90pct": experts_needed_for_share(scores, 0.90),
    }


def compute_concentration_stats(layers: dict[int, LayerData]) -> dict[str, Any]:
    overall: dict[str, dict[str, list[float]]] = {
        "w1": {"gini": [], "top10": [], "top25": [], "top50": []},
        "w2": {"gini": [], "top10": [], "top25": [], "top50": []},
    }
    by_layer: dict[str, Any] = {}

    for layer_idx, layer in layers.items():
        by_layer[str(layer_idx)] = {}
        for matrix_name, matrix in (("w1", layer.w1), ("w2", layer.w2)):
            ginis: list[float] = []
            top10: list[float] = []
            top25: list[float] = []
            top50: list[float] = []
            for expert_values in matrix:
                ginis.append(gini(expert_values))
                top10.append(top_fraction_capture(expert_values, 0.10))
                top25.append(top_fraction_capture(expert_values, 0.25))
                top50.append(top_fraction_capture(expert_values, 0.50))
            overall[matrix_name]["gini"].extend(ginis)
            overall[matrix_name]["top10"].extend(top10)
            overall[matrix_name]["top25"].extend(top25)
            overall[matrix_name]["top50"].extend(top50)
            by_layer[str(layer_idx)][matrix_name] = {
                "mean_gini": float(np.mean(ginis)),
                "top10_captures_pct": float(np.mean(top10)),
                "top25_captures_pct": float(np.mean(top25)),
                "top50_captures_pct": float(np.mean(top50)),
            }

    return {
        "overall": {
            matrix_name: {
                "mean_gini": float(np.mean(values["gini"])),
                "top10_captures_pct": float(np.mean(values["top10"])),
                "top25_captures_pct": float(np.mean(values["top25"])),
                "top50_captures_pct": float(np.mean(values["top50"])),
            }
            for matrix_name, values in overall.items()
        },
        "by_layer": by_layer,
    }


def plot_per_layer_distribution(layers: dict[int, LayerData]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    colors = sns.color_palette("colorblind", 2)
    for axis, layer_idx in zip(axes, LAYER_IDS):
        layer = layers[layer_idx]
        axis.hist(layer.expert_w1_mean, bins=28, alpha=0.65, label="W1 mean", color=colors[0])
        axis.hist(layer.expert_w2_mean, bins=28, alpha=0.65, label="W2 mean", color=colors[1])
        axis.set_title(f"Layer {layer_idx}")
        axis.set_xlabel("Per-expert mean weight_l1")
        axis.set_ylabel("Experts")
        axis.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT_LAYER_DISTRIBUTION_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_expert_ranking(layers: dict[int, LayerData]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    palette = sns.color_palette("colorblind", len(LAYER_IDS))
    for axis, layer_idx, color in zip(axes, LAYER_IDS, palette):
        layer = layers[layer_idx]
        scores = np.sort(layer.expert_total_score)[::-1]
        cumulative = np.cumsum(scores) / scores.sum()
        x = np.arange(1, scores.size + 1)
        axis.plot(x, scores, color=color, linewidth=2)
        axis.axvline(experts_needed_for_share(layer.expert_total_score, 0.50), color=color, linestyle="--", alpha=0.7)
        axis2 = axis.twinx()
        axis2.plot(x, cumulative, color="#333333", linewidth=1.2, alpha=0.7)
        axis2.set_ylim(0.0, 1.02)
        axis2.set_ylabel("Cumulative sensitivity share")
        axis.set_title(f"Layer {layer_idx}")
        axis.set_xlabel("Experts ranked by total weight_l1")
        axis.set_ylabel("Total expert sensitivity")
    fig.tight_layout()
    fig.savefig(OUTPUT_EXPERT_RANKING_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cross_layer_shape(layers: dict[int, LayerData]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("colorblind", len(LAYER_IDS))
    for layer_idx, color in zip(LAYER_IDS, palette):
        ranked = np.sort(layers[layer_idx].expert_total_score)[::-1]
        normalized = minmax_normalize(ranked)
        x = np.linspace(0.0, 1.0, ranked.size)
        ax.plot(x, normalized, label=f"Layer {layer_idx}", color=color, linewidth=2)
    ax.set_xlabel("Normalized expert rank")
    ax.set_ylabel("Min-max normalized expert sensitivity")
    ax.set_title("Cross-layer expert sensitivity shape")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT_CROSS_LAYER_SHAPE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_budget_comparison(budget_runs: dict[str, dict[str, BudgetRun]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    budgets = np.asarray([float(key) for key in budget_runs], dtype=np.float64)
    strategy_order = ["uniform", "proportional", "marginal_benefit"]
    colors = sns.color_palette("colorblind", len(strategy_order))

    for strategy, color in zip(strategy_order, colors):
        remaining = np.asarray([budget_runs[key][strategy].remaining_fraction for key in budget_runs], dtype=np.float64)
        improvement = np.asarray(
            [
                0.0
                if strategy == "uniform"
                else (budget_runs[key]["uniform"].total_sensitivity - budget_runs[key][strategy].total_sensitivity)
                / budget_runs[key]["uniform"].total_sensitivity
                for key in budget_runs
            ],
            dtype=np.float64,
        )
        axes[0].plot(budgets * 100.0, remaining * 100.0, marker="o", linewidth=2, label=strategy.replace("_", " "), color=color)
        axes[1].plot(budgets * 100.0, improvement * 100.0, marker="o", linewidth=2, label=strategy.replace("_", " "), color=color)

    axes[0].set_xlabel("FP8 budget (% of total weights)")
    axes[0].set_ylabel("Sensitivity left in FP4 (%)")
    axes[0].set_title("Remaining sensitivity")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("FP8 budget (% of total weights)")
    axes[1].set_ylabel("Improvement over uniform (%)")
    axes[1].set_title("Allocation gain vs uniform")
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT_BUDGET_COMPARISON_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_channel_concentration(layers: dict[int, LayerData], concentration_stats: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    palette = sns.color_palette("colorblind", len(LAYER_IDS))
    for axis, matrix_name, channel_count in zip(axes, ("w1", "w2"), (W1_CHANNELS, W2_CHANNELS)):
        x = np.linspace(0.0, 1.0, channel_count + 1)
        axis.plot(x, x, color="#777777", linestyle="--", linewidth=1, label="Uniform")
        for layer_idx, color in zip(LAYER_IDS, palette):
            matrix = layers[layer_idx].w1 if matrix_name == "w1" else layers[layer_idx].w2
            curves = np.stack([lorenz_curve(expert_values)[1] for expert_values in matrix], axis=0)
            mean_curve = curves.mean(axis=0)
            mean_gini = concentration_stats["by_layer"][str(layer_idx)][matrix_name]["mean_gini"]
            top10 = concentration_stats["by_layer"][str(layer_idx)][matrix_name]["top10_captures_pct"]
            axis.plot(
                x,
                mean_curve,
                color=color,
                linewidth=2,
                label=f"L{layer_idx} gini={mean_gini:.2f} top10={top10:.2f}",
            )
        axis.set_title(f"{matrix_name.upper()} channel concentration")
        axis.set_xlabel("Fraction of channels")
        axis.set_ylabel("Cumulative sensitivity")
        axis.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT_CHANNEL_CONCENTRATION_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_results_payload(
    layers: dict[int, LayerData],
    spike1_payload: dict[str, Any],
    budget_runs: dict[str, dict[str, BudgetRun]],
    concentration_stats: dict[str, Any],
) -> dict[str, Any]:
    per_layer_stats: dict[str, Any] = {}
    expert_ranking: dict[str, Any] = {}
    for layer_idx in LAYER_IDS:
        layer = layers[layer_idx]
        per_layer_stats[str(layer_idx)] = {
            "num_experts": NUM_EXPERTS,
            "layer_total_sensitivity": layer.layer_total_score,
            "w1_mean": float(np.mean(layer.expert_w1_mean)),
            "w1_median": float(np.median(layer.expert_w1_mean)),
            "w1_std": float(np.std(layer.expert_w1_mean)),
            "w1_min": float(np.min(layer.expert_w1_mean)),
            "w1_max": float(np.max(layer.expert_w1_mean)),
            "w2_mean": float(np.mean(layer.expert_w2_mean)),
            "w2_median": float(np.median(layer.expert_w2_mean)),
            "w2_std": float(np.std(layer.expert_w2_mean)),
            "w2_min": float(np.min(layer.expert_w2_mean)),
            "w2_max": float(np.max(layer.expert_w2_mean)),
            "expert_total_mean": float(np.mean(layer.expert_total_score)),
            "expert_total_std": float(np.std(layer.expert_total_score)),
        }
        expert_ranking[str(layer_idx)] = analyze_expert_ranking(layer)

    budget_payload: dict[str, Any] = {}
    for budget_key, run_map in budget_runs.items():
        uniform_remaining = run_map["uniform"].total_sensitivity
        budget_payload[budget_key] = {}
        for strategy, run in run_map.items():
            budget_payload[budget_key][strategy] = {
                "total_sensitivity": run.total_sensitivity,
                "remaining_fraction": run.remaining_fraction,
                "promoted_sensitivity": run.promoted_sensitivity,
                "actual_budget_fraction": run.actual_budget_fraction,
                "layer_budgets": run.layer_budgets,
                "layer_budget_shares": run.layer_budget_shares,
                "layer_remaining_sensitivity": run.layer_remaining_sensitivity,
                "layer_promoted_sensitivity": run.layer_promoted_sensitivity,
                "promoted_units": run.promoted_units,
                "improvement_vs_uniform_pct": 0.0 if strategy == "uniform" else float((uniform_remaining - run.total_sensitivity) / uniform_remaining),
            }

    optimal_ratios = budget_runs["0.25"]["marginal_benefit"].layer_budgets
    return {
        "metadata": {
            "source_metric": "weight_l1",
            "layers": list(LAYER_IDS),
            "num_experts": NUM_EXPERTS,
            "w1_channels": W1_CHANNELS,
            "w2_channels": W2_CHANNELS,
            "w1_channel_cost_weights": W1_COST_WEIGHTS,
            "w2_channel_cost_weights": W2_COST_WEIGHTS,
            "budget_levels": list(BUDGET_LEVELS),
            "ground_truth_reference_experts": [
                {
                    "layer": int(layer["layer_idx"]),
                    "expert_id": int(expert["expert_id"]),
                    "role": str(expert["role"]),
                }
                for layer in spike1_payload["layers"]
                for expert in layer["experts"]
            ],
            "budget_interpretation": "Budget is a fraction of total promoted weights. Each promoted W1 pair costs 4096 weights, each promoted W2 channel costs 512 weights. Channels are greedily selected by descending weight_l1 within each allocation policy.",
        },
        "per_layer_stats": per_layer_stats,
        "expert_ranking": expert_ranking,
        "budget_comparison": budget_payload,
        "channel_concentration": {
            "mean_gini_w1": concentration_stats["overall"]["w1"]["mean_gini"],
            "mean_gini_w2": concentration_stats["overall"]["w2"]["mean_gini"],
            "top10_captures_pct_w1": concentration_stats["overall"]["w1"]["top10_captures_pct"],
            "top25_captures_pct_w1": concentration_stats["overall"]["w1"]["top25_captures_pct"],
            "top50_captures_pct_w1": concentration_stats["overall"]["w1"]["top50_captures_pct"],
            "top10_captures_pct_w2": concentration_stats["overall"]["w2"]["top10_captures_pct"],
            "top25_captures_pct_w2": concentration_stats["overall"]["w2"]["top25_captures_pct"],
            "top50_captures_pct_w2": concentration_stats["overall"]["w2"]["top50_captures_pct"],
            "by_layer": concentration_stats["by_layer"],
        },
        "optimal_layer_budget_ratios": [float(value) for value in optimal_ratios],
    }


def main() -> None:
    sns.set_theme(style="ticks", context="paper", palette="colorblind")
    spike2_payload = load_json(SPIKE2_PATH)
    spike1_payload = load_json(SPIKE1_PATH)
    layers = load_layer_data(spike2_payload)
    budget_runs = run_budget_allocations(layers)
    concentration_stats = compute_concentration_stats(layers)

    plot_per_layer_distribution(layers)
    plot_expert_ranking(layers)
    plot_cross_layer_shape(layers)
    plot_budget_comparison(budget_runs)
    plot_channel_concentration(layers, concentration_stats)

    results_payload = build_results_payload(layers, spike1_payload, budget_runs, concentration_stats)
    write_json(OUTPUT_JSON_PATH, results_payload)
    print(f"Saved JSON to {OUTPUT_JSON_PATH}")
    print(f"Saved plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

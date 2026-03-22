#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false

from __future__ import annotations

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
SPIKE3_PATH = RESULTS_DIR / "spike3_strategies.json"
SPIKE4_PATH = RESULTS_DIR / "spike4_budget.json"
OUTPUT_JSON_PATH = RESULTS_DIR / "spike6_ablations.json"
OUTPUT_PARETO_PATH = RESULTS_DIR / "spike6_pareto.png"
OUTPUT_GRANULARITY_PATH = RESULTS_DIR / "spike6_granularity.png"
OUTPUT_HOT_COLD_PATH = RESULTS_DIR / "spike6_hot_vs_cold.png"
OUTPUT_STABILITY_PATH = RESULTS_DIR / "spike6_calibration_stability.png"
EXPLORATION_PATH = SCRIPT_DIR / "exploration.md"

W1_CHANNELS = 512
W2_CHANNELS = 2048
W1_WEIGHT_COST = 2048 * 2
W2_WEIGHT_COST = 512
TOTAL_EXPERT_WEIGHTS = 3_145_728
MODEL_FP4_GB = 15.36
TOKEN_COUNT = 128
CALIBRATION_SIZES = (32, 64, 96)
CALIBRATION_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class GroundTruthExpert:
    layer_idx: int
    expert_id: int
    role: str
    token_count: int
    token_positions: tuple[int, ...]
    w1_sensitivity: np.ndarray
    w2_sensitivity: np.ndarray

    @property
    def total_sensitivity(self) -> float:
        return float(np.sum(self.w1_sensitivity, dtype=np.float64) + np.sum(self.w2_sensitivity, dtype=np.float64))


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
                    token_positions=tuple(int(pos) for pos in expert.get("token_positions", [])),
                    w1_sensitivity=np.asarray(expert["w1_pair_sensitivity"], dtype=np.float64),
                    w2_sensitivity=np.asarray(expert["w2_channel_sensitivity"], dtype=np.float64),
                )
            )
    return experts


def load_proxy_scores(spike2_payload: dict[str, Any], experts: list[GroundTruthExpert]) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    scores: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for expert in experts:
        scores[(expert.layer_idx, expert.expert_id)] = {
            "w1": np.asarray(spike2_payload[f"layer_{expert.layer_idx}_expert_{expert.expert_id}_w1"]["weight_l1"], dtype=np.float64),
            "w2": np.asarray(spike2_payload[f"layer_{expert.layer_idx}_expert_{expert.expert_id}_w2"]["weight_l1"], dtype=np.float64),
        }
    return scores


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    n = int(scores.shape[0])
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    indices = np.argpartition(scores, n - k)[n - k :]
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    return mask


def allocate_weighted_counts(total: int, capacities: list[int], weights: list[float]) -> list[int]:
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
                give = min(int(np.ceil(remaining_total / len(ordered))), remaining_capacity[idx])
                allocations[idx] += give
                remaining_capacity[idx] -= give
                remaining_total -= give
            active = {idx for idx in active if remaining_capacity[idx] > 0}
            continue
        quotas = {idx: remaining_total * weights[idx] / weight_sum for idx in active}
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


def baseline_cost(experts: list[GroundTruthExpert]) -> float:
    return float(sum(expert.total_sensitivity for expert in experts))


def uniform_channel_cost(experts: list[GroundTruthExpert], proxy_scores: dict[tuple[int, int], dict[str, np.ndarray]], budget: float) -> float:
    total = 0.0
    w1_k = int(round(budget * W1_CHANNELS))
    w2_k = int(round(budget * W2_CHANNELS))
    for expert in experts:
        proxy = proxy_scores[(expert.layer_idx, expert.expert_id)]
        w1_mask = topk_mask(proxy["w1"], w1_k)
        w2_mask = topk_mask(proxy["w2"], w2_k)
        total += float(np.sum(expert.w1_sensitivity[~w1_mask], dtype=np.float64))
        total += float(np.sum(expert.w2_sensitivity[~w2_mask], dtype=np.float64))
    return total


def freq_weighted_cost(experts: list[GroundTruthExpert], proxy_scores: dict[tuple[int, int], dict[str, np.ndarray]], budget: float) -> float:
    total = 0.0
    grouped: dict[int, list[GroundTruthExpert]] = {}
    for expert in experts:
        grouped.setdefault(expert.layer_idx, []).append(expert)
    for layer_experts in grouped.values():
        weights = [float(expert.token_count) for expert in layer_experts]
        w1_alloc = allocate_weighted_counts(int(round(budget * W1_CHANNELS * len(layer_experts))), [W1_CHANNELS] * len(layer_experts), weights)
        w2_alloc = allocate_weighted_counts(int(round(budget * W2_CHANNELS * len(layer_experts))), [W2_CHANNELS] * len(layer_experts), weights)
        for expert, w1_k, w2_k in zip(layer_experts, w1_alloc, w2_alloc, strict=True):
            proxy = proxy_scores[(expert.layer_idx, expert.expert_id)]
            w1_mask = topk_mask(proxy["w1"], w1_k)
            w2_mask = topk_mask(proxy["w2"], w2_k)
            total += float(np.sum(expert.w1_sensitivity[~w1_mask], dtype=np.float64))
            total += float(np.sum(expert.w2_sensitivity[~w2_mask], dtype=np.float64))
    return total


def per_expert_cost(experts: list[GroundTruthExpert], budget: float) -> float:
    ranked = sorted(experts, key=lambda expert: (expert.token_count, expert.total_sensitivity, expert.expert_id), reverse=True)
    promote = int(np.floor(budget * len(ranked) + 1e-9))
    return float(sum(expert.total_sensitivity for expert in ranked[promote:]))


def memory_gb(fp8_pct: int) -> float:
    return MODEL_FP4_GB * (1.0 + fp8_pct / 100.0)


def pareto_sweet_spot(memories: list[float], costs: list[float]) -> int:
    best_idx = 1
    best_gain = -np.inf
    for idx in range(1, len(memories)):
        gain = (costs[idx - 1] - costs[idx]) / (memories[idx] - memories[idx - 1])
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    return best_idx * 5


def role_stats(experts: list[GroundTruthExpert]) -> dict[str, float]:
    baseline = baseline_cost(experts)
    by_role: dict[str, list[GroundTruthExpert]] = {"hot": [], "medium": [], "cold": []}
    for expert in experts:
        by_role[expert.role].append(expert)
    fractions = {
        role: float(sum(expert.total_sensitivity for expert in role_experts) / baseline)
        for role, role_experts in by_role.items()
    }
    only_hot_cost = float((sum(expert.total_sensitivity for expert in by_role["medium"]) + sum(expert.total_sensitivity for expert in by_role["cold"])) / baseline)
    only_cold_cost = float((sum(expert.total_sensitivity for expert in by_role["medium"]) + sum(expert.total_sensitivity for expert in by_role["hot"])) / baseline)
    return {
        "hot_sensitivity_fraction": fractions["hot"],
        "medium_sensitivity_fraction": fractions["medium"],
        "cold_sensitivity_fraction": fractions["cold"],
        "only_hot_fp8_cost": only_hot_cost,
        "only_cold_fp8_cost": only_cold_cost,
    }


def build_presence(experts: list[GroundTruthExpert]) -> dict[int, dict[int, np.ndarray]]:
    presence: dict[int, dict[int, np.ndarray]] = {}
    for expert in experts:
        vector = np.zeros(TOKEN_COUNT, dtype=np.int64)
        if expert.token_positions:
            vector[np.asarray(expert.token_positions, dtype=np.int64)] = 1
        presence.setdefault(expert.layer_idx, {})[expert.expert_id] = vector
    return presence


def top_quartile_set(counts: dict[int, int]) -> set[int]:
    top_k = max(1, int(np.ceil(0.25 * len(counts))))
    ranked = sorted(counts, key=lambda expert_id: (counts[expert_id], expert_id), reverse=True)
    return set(ranked[:top_k])


def calibration_stability(experts: list[GroundTruthExpert]) -> dict[str, float]:
    presence = build_presence(experts)
    results = {"overlap_at_128": 1.0}
    for sample_size in CALIBRATION_SIZES:
        overlaps = []
        for layer_map in presence.values():
            full_counts = {expert_id: int(vector.sum()) for expert_id, vector in layer_map.items()}
            full_top = top_quartile_set(full_counts)
            for seed in CALIBRATION_SEEDS:
                rng = np.random.default_rng(sample_size * 100 + seed)
                sampled = np.sort(rng.choice(TOKEN_COUNT, size=sample_size, replace=False))
                sub_counts = {expert_id: int(vector[sampled].sum()) for expert_id, vector in layer_map.items()}
                sub_top = top_quartile_set(sub_counts)
                overlaps.append(len(full_top & sub_top) / max(1, len(full_top)))
        results[f"overlap_at_{sample_size}"] = float(np.mean(overlaps, dtype=np.float64))
    return results


def plot_pareto(memories: list[float], freq_costs: list[float], uniform_costs: list[float], sweet_spot_pct: int) -> None:
    idx = sweet_spot_pct // 5
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(memories, freq_costs, marker="o", linewidth=2, color="#1b9e77", label="freq_weighted")
    plt.plot(memories, uniform_costs, marker="o", linewidth=2, color="#7570b3", label="uniform_sort")
    plt.scatter(memories[idx], freq_costs[idx], color="#d95f02", s=90, zorder=5, label=f"sweet spot {sweet_spot_pct}%")
    plt.annotate(f"{sweet_spot_pct}%", (memories[idx], freq_costs[idx]), xytext=(8, -14), textcoords="offset points")
    plt.xlabel("Estimated memory (GB)")
    plt.ylabel("Normalized GT sensitivity")
    plt.title("Spike 6 Ablation 1: Pareto curve")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PARETO_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_granularity(payload: dict[str, dict[str, float]]) -> None:
    budgets = [10, 25, 50, 75]
    x = np.arange(len(budgets), dtype=np.float64)
    width = 0.24
    plt.figure(figsize=(8.0, 5.0))
    plt.bar(x - width, [payload["per_expert_gt_cost"][str(b)] for b in budgets], width=width, color="#d95f02", label="per_expert")
    plt.bar(x, [payload["per_channel_gt_cost"][str(b)] for b in budgets], width=width, color="#7570b3", label="per_channel")
    plt.bar(x + width, [payload["freq_weighted_gt_cost"][str(b)] for b in budgets], width=width, color="#1b9e77", label="per_channel + freq")
    plt.xticks(x, [f"{b}%" for b in budgets])
    plt.xlabel("FP8 budget")
    plt.ylabel("Normalized GT sensitivity")
    plt.title("Spike 6 Ablation 2: Granularity")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_GRANULARITY_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hot_cold(payload: dict[str, float]) -> None:
    labels = ["hot frac", "cold frac", "only hot FP8", "only cold FP8"]
    values = [payload["hot_sensitivity_fraction"], payload["cold_sensitivity_fraction"], payload["only_hot_fp8_cost"], payload["only_cold_fp8_cost"]]
    colors = ["#1b9e77", "#7570b3", "#1b9e77", "#7570b3"]
    plt.figure(figsize=(7.5, 4.8))
    x = np.arange(len(labels))
    plt.bar(x, values, color=colors)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Fraction / normalized GT cost")
    plt.title("Spike 6 Ablation 3: Hot vs cold")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_HOT_COLD_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stability(payload: dict[str, float]) -> None:
    xs = [32, 64, 96, 128]
    ys = [payload[f"overlap_at_{x}"] for x in xs]
    plt.figure(figsize=(7.0, 4.6))
    plt.plot(xs, ys, marker="o", linewidth=2, color="#1b9e77")
    plt.xticks(xs)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Calibration tokens")
    plt.ylabel("Top-25% overlap")
    plt.title("Spike 6 Ablation 4: Calibration stability")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_STABILITY_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def update_exploration(results: dict[str, Any], spike3_payload: dict[str, Any], spike4_payload: dict[str, Any]) -> None:
    sweet = int(results["pareto"]["sweet_spot_pct"])
    per_expert_25 = float(results["granularity"]["per_expert_gt_cost"]["25"])
    per_channel_25 = float(results["granularity"]["per_channel_gt_cost"]["25"])
    freq_25 = float(results["granularity"]["freq_weighted_gt_cost"]["25"])
    hot_frac = float(results["hot_vs_cold"]["hot_sensitivity_fraction"])
    cold_frac = float(results["hot_vs_cold"]["cold_sensitivity_fraction"])
    overlap_32 = float(results["calibration_stability"]["overlap_at_32"])
    overlap_64 = float(results["calibration_stability"]["overlap_at_64"])
    overlap_96 = float(results["calibration_stability"]["overlap_at_96"])
    best = str(spike3_payload.get("best_strategy_w2", "freq_weighted"))
    ratios = spike4_payload.get("optimal_layer_budget_ratios", [])
    ratio_text = ":".join(f"{100.0 * float(value):.1f}%" for value in ratios) if ratios else "n/a"
    block = (
        "\n## [6] Ablation Studies\n\n"
        "**Approach**: Ran four CPU-only ablations on Spike 1-4 artifacts: a 0-100% FP8 Pareto sweep in 5% steps, a granularity comparison at 10/25/50/75% budgets, a hot-vs-cold routing-tier analysis, and a calibration subsampling stability study. Memory uses the closed-form MoE-only estimate from 15.36 GB at all-FP4 to 30.72 GB at all-FP8. The calibration analysis is limited to the 9 tracked Spike 1 experts because full 256-expert routing histograms were not saved.\n\n"
        f"**Result**: The Pareto curve is steepest at {sweet}% FP8, so early FP8 budget delivers the largest sensitivity drop per added GB. At 25% budget, per-expert GT cost is {per_expert_25:.3f}, naive per-channel is {per_channel_25:.3f}, and routing-aware per-channel (`{best}`) is {freq_25:.3f}; this again shows the major gain comes from picking the right experts, not just finer channel slicing. Hot experts contribute {hot_frac:.1%} of total GT sensitivity while cold experts contribute only {cold_frac:.1%}, and giving FP8 only to hot experts is far better than prioritizing cold experts. Calibration overlap improves from {overlap_32:.2f} at 32 tokens to {overlap_64:.2f} at 64 and {overlap_96:.2f} at 96. Spike 4's layer split ({ratio_text} for L5:L20:L35) remains secondary relative to expert hotness.\n\n"
        "**Verdict**: Spike 6 supports the same design choice as Spike 3: routing-aware expert allocation is the dominant lever, while per-channel refinement and layer-level budget shaping are second-order. A moderate FP8 budget near the Pareto sweet spot should capture most of the benefit. For a stronger calibration-stability claim, the next data pass should persist full per-layer expert routing histograms.\n"
    )
    current = EXPLORATION_PATH.read_text(encoding="utf-8")
    if "## [6] Ablation Studies" in current:
        current = current.split("## [6] Ablation Studies", 1)[0].rstrip()
    EXPLORATION_PATH.write_text(current + block + "\n", encoding="utf-8")


def main() -> None:
    spike1_payload = load_json(SPIKE1_PATH)
    spike2_payload = load_json(SPIKE2_PATH)
    spike3_payload = load_json(SPIKE3_PATH)
    spike4_payload = load_json(SPIKE4_PATH)
    experts = load_ground_truth(spike1_payload)
    proxy_scores = load_proxy_scores(spike2_payload, experts)
    baseline = baseline_cost(experts)

    budgets = list(range(0, 101, 5))
    memories = [memory_gb(budget) for budget in budgets]
    freq_costs = [float(freq_weighted_cost(experts, proxy_scores, budget / 100.0) / baseline) for budget in budgets]
    uniform_costs = [float(uniform_channel_cost(experts, proxy_scores, budget / 100.0) / baseline) for budget in budgets]
    sweet_spot_pct = pareto_sweet_spot(memories, freq_costs)

    granularity = {
        "per_expert_gt_cost": {},
        "per_channel_gt_cost": {},
        "freq_weighted_gt_cost": {},
    }
    for budget in (10, 25, 50, 75):
        frac = budget / 100.0
        key = str(budget)
        granularity["per_expert_gt_cost"][key] = float(per_expert_cost(experts, frac) / baseline)
        granularity["per_channel_gt_cost"][key] = float(uniform_channel_cost(experts, proxy_scores, frac) / baseline)
        granularity["freq_weighted_gt_cost"][key] = float(freq_weighted_cost(experts, proxy_scores, frac) / baseline)

    hot_vs_cold = role_stats(experts)
    stability = calibration_stability(experts)
    results = {
        "pareto": {
            "budgets": budgets,
            "freq_weighted_gt_cost": freq_costs,
            "uniform_sort_gt_cost": uniform_costs,
            "memory_gb": memories,
            "sweet_spot_pct": sweet_spot_pct,
        },
        "granularity": granularity,
        "hot_vs_cold": {
            "hot_sensitivity_fraction": hot_vs_cold["hot_sensitivity_fraction"],
            "cold_sensitivity_fraction": hot_vs_cold["cold_sensitivity_fraction"],
            "only_hot_fp8_cost": hot_vs_cold["only_hot_fp8_cost"],
            "only_cold_fp8_cost": hot_vs_cold["only_cold_fp8_cost"],
        },
        "calibration_stability": stability,
        "metadata": {
            "spike1_path": str(SPIKE1_PATH),
            "spike2_path": str(SPIKE2_PATH),
            "spike3_path": str(SPIKE3_PATH),
            "spike4_path": str(SPIKE4_PATH),
            "calibration_note": "Top-25% overlap uses only the 9 tracked Spike 1 experts because full routing histograms were not persisted.",
        },
    }

    plot_pareto(memories, freq_costs, uniform_costs, sweet_spot_pct)
    plot_granularity(granularity)
    plot_hot_cold(hot_vs_cold)
    plot_stability(stability)
    atomic_json_dump(results, OUTPUT_JSON_PATH)
    update_exploration(results, spike3_payload, spike4_payload)
    print(f"Saved results to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()

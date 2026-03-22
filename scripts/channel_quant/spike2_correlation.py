#!/usr/bin/env python3
# pyright: basic

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import matplotlib
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SPIKE1_PATH = RESULTS_DIR / "spike1_sensitivity.json"
SPIKE2_PATH = RESULTS_DIR / "spike2_metrics.json"
OUTPUT_JSON_PATH = RESULTS_DIR / "spike2_correlation.json"
OUTPUT_HEATMAP_PATH = RESULTS_DIR / "spike2_correlation_heatmap.png"
ROLE_SUFFIX = {"cold": "cold", "medium": "med", "hot": "hot"}


@dataclass(frozen=True)
class ExpertRecord:
    label: str
    layer_idx: int
    expert_id: int
    role: str
    w1_ground_truth: np.ndarray
    w2_ground_truth: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_expert_label(layer_idx: int, expert_id: int, role: str) -> str:
    return f"L{layer_idx}_E{expert_id}_{ROLE_SUFFIX.get(role, role)}"


def load_ground_truth(spike1_payload: dict[str, Any]) -> list[ExpertRecord]:
    experts: list[ExpertRecord] = []
    for layer in spike1_payload["layers"]:
        layer_idx = int(layer["layer_idx"])
        for expert in layer["experts"]:
            expert_id = int(expert["expert_id"])
            role = str(expert["role"])
            experts.append(
                ExpertRecord(
                    label=make_expert_label(layer_idx, expert_id, role),
                    layer_idx=layer_idx,
                    expert_id=expert_id,
                    role=role,
                    w1_ground_truth=np.asarray(expert["w1_pair_sensitivity"], dtype=np.float64),
                    w2_ground_truth=np.asarray(expert["w2_channel_sensitivity"], dtype=np.float64),
                )
            )
    return experts


def proxy_key(layer_idx: int, expert_id: int, matrix_name: str) -> str:
    return f"layer_{layer_idx}_expert_{expert_id}_{matrix_name}"


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Mismatched vector lengths: {left.shape} vs {right.shape}")
    correlation = cast(float, spearmanr(left, right)[0])
    if correlation is None or np.isnan(correlation):
        return 0.0
    return float(correlation)


def compute_correlations(
    experts: list[ExpertRecord],
    spike2_payload: dict[str, Any],
    metric_order: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], list[str]]:
    w1_correlations = {metric: {} for metric in metric_order}
    w2_correlations = {metric: {} for metric in metric_order}
    expert_labels: list[str] = []

    for expert in experts:
        expert_labels.append(expert.label)
        w1_key = proxy_key(expert.layer_idx, expert.expert_id, "w1")
        w2_key = proxy_key(expert.layer_idx, expert.expert_id, "w2")
        if w1_key not in spike2_payload:
            raise KeyError(f"Missing proxy metrics for {w1_key}")
        if w2_key not in spike2_payload:
            raise KeyError(f"Missing proxy metrics for {w2_key}")

        w1_metrics = spike2_payload[w1_key]
        w2_metrics = spike2_payload[w2_key]
        for metric in metric_order:
            if metric not in w1_metrics or metric not in w2_metrics:
                raise KeyError(f"Missing metric '{metric}' for expert {expert.label}")
            w1_proxy = np.asarray(w1_metrics[metric], dtype=np.float64)
            w2_proxy = np.asarray(w2_metrics[metric], dtype=np.float64)
            w1_correlations[metric][expert.label] = safe_spearman(expert.w1_ground_truth, w1_proxy)
            w2_correlations[metric][expert.label] = safe_spearman(expert.w2_ground_truth, w2_proxy)

    return w1_correlations, w2_correlations, expert_labels


def mean_by_metric(correlations: dict[str, dict[str, float]], metric_order: list[str]) -> dict[str, float]:
    return {
        metric: float(np.mean(list(correlations[metric].values()), dtype=np.float64))
        for metric in metric_order
    }


def overall_mean_by_metric(
    w1_correlations: dict[str, dict[str, float]],
    w2_correlations: dict[str, dict[str, float]],
    metric_order: list[str],
) -> dict[str, float]:
    combined: dict[str, float] = {}
    for metric in metric_order:
        values = list(w1_correlations[metric].values()) + list(w2_correlations[metric].values())
        combined[metric] = float(np.mean(values, dtype=np.float64))
    return combined


def ordered_metric_ranking(mean_values: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(mean_values.items(), key=lambda item: item[1], reverse=True)


def best_metric(mean_values: dict[str, float]) -> str:
    return ordered_metric_ranking(mean_values)[0][0]


def correlation_matrix(
    correlations: dict[str, dict[str, float]],
    expert_order: list[str],
    metric_order: list[str],
) -> np.ndarray:
    return np.asarray(
        [[correlations[metric][expert] for metric in metric_order] for expert in expert_order],
        dtype=np.float64,
    )


def save_heatmap(
    w1_correlations: dict[str, dict[str, float]],
    w2_correlations: dict[str, dict[str, float]],
    expert_order: list[str],
    metric_order: list[str],
    output_path: Path,
) -> None:
    sns.set_theme(style="white", context="paper")
    w1_matrix = correlation_matrix(w1_correlations, expert_order, metric_order)
    w2_matrix = correlation_matrix(w2_correlations, expert_order, metric_order)

    fig, axes = plt.subplots(ncols=2, figsize=(18, 6), sharey=True)
    common = {
        "xticklabels": metric_order,
        "yticklabels": expert_order,
        "annot": True,
        "fmt": ".2f",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    sns.heatmap(w1_matrix, ax=axes[0], cbar=False, **common)
    sns.heatmap(w2_matrix, ax=axes[1], cbar=True, cbar_kws={"label": "Spearman rho"}, **common)

    axes[0].set_title("W1 proxy vs. ground truth")
    axes[1].set_title("W2 proxy vs. ground truth")
    axes[0].set_xlabel("Proxy metric")
    axes[1].set_xlabel("Proxy metric")
    axes[0].set_ylabel("Expert")
    axes[1].set_ylabel("")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_results_json(
    w1_correlations: dict[str, dict[str, float]],
    w2_correlations: dict[str, dict[str, float]],
    w1_mean: dict[str, float],
    w2_mean: dict[str, float],
    overall_mean: dict[str, float],
    output_path: Path,
) -> None:
    payload = {
        "w1_correlations": w1_correlations,
        "w2_correlations": w2_correlations,
        "w1_mean_by_metric": w1_mean,
        "w2_mean_by_metric": w2_mean,
        "overall_mean_by_metric": overall_mean,
        "best_proxy_w1": best_metric(w1_mean),
        "best_proxy_w2": best_metric(w2_mean),
        "best_proxy_overall": best_metric(overall_mean),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def print_ranking(overall_mean: dict[str, float], w1_mean: dict[str, float], w2_mean: dict[str, float]) -> None:
    print("Proxy metrics ranked by mean Spearman correlation")
    for rank, (metric, overall_value) in enumerate(ordered_metric_ranking(overall_mean), start=1):
        print(
            f"{rank:2d}. {metric:<20} overall={overall_value:.4f} "
            f"w1={w1_mean[metric]:.4f} w2={w2_mean[metric]:.4f}"
        )


def main() -> None:
    spike1_payload = load_json(SPIKE1_PATH)
    spike2_payload = load_json(SPIKE2_PATH)
    metric_order = list(spike2_payload["_metadata"]["metric_order"])
    if len(metric_order) != 12:
        raise ValueError(f"Expected 12 proxy metrics, found {len(metric_order)}")

    experts = load_ground_truth(spike1_payload)
    w1_correlations, w2_correlations, expert_order = compute_correlations(experts, spike2_payload, metric_order)
    w1_mean = mean_by_metric(w1_correlations, metric_order)
    w2_mean = mean_by_metric(w2_correlations, metric_order)
    overall_mean = overall_mean_by_metric(w1_correlations, w2_correlations, metric_order)

    save_results_json(w1_correlations, w2_correlations, w1_mean, w2_mean, overall_mean, OUTPUT_JSON_PATH)
    save_heatmap(w1_correlations, w2_correlations, expert_order, metric_order, OUTPUT_HEATMAP_PATH)
    print_ranking(overall_mean, w1_mean, w2_mean)


if __name__ == "__main__":
    main()

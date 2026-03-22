#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from baselines_comparison import (
    DEFAULT_MINI_BATCH_TOKENS,
    EvaluationPlan,
    LayerMetricBundle,
    allocate_weighted_counts,
    estimate_mixed_memory_gb,
    evaluate_plan,
    fp8_weights_from_masks,
    load_prefix_dataset_tokens,
    quantize_linear_weight,
    resolve_non_expert_bytes,
    run_calibration_pass,
    topk_mask_from_scores,
)
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config, move_tensor, release_tensors


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_ITER01_JSON = RESULTS_DIR / "exploration_iter01.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "exploration_iter02.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
TARGET_FP8_FRACTION = 0.25
PER_PROJECTION_W1_FRACTION = 0.10
PER_PROJECTION_W2_FRACTION = 0.40
MXMOE_TARGET_PPL = 5.337
SECTION_MARKER = "## [11] Assignment Strategy Comparison"
EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--iter01-json", type=Path, default=DEFAULT_ITER01_JSON)
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--mini-batch-tokens", type=int, default=DEFAULT_MINI_BATCH_TOKENS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def vector_kurtosis(values: torch.Tensor, dim: int) -> torch.Tensor:
    values_f = values.float()
    if values_f.shape[dim] <= 1:
        shape = list(values_f.shape)
        del shape[dim]
        return torch.zeros(shape, dtype=torch.float32, device=values_f.device)
    mean = values_f.mean(dim=dim, keepdim=True)
    centered = values_f - mean
    var = centered.square().mean(dim=dim)
    central4 = centered.pow(4).mean(dim=dim)
    kurt = central4 / (var.square() + EPS)
    kurt = torch.where(torch.isfinite(kurt), kurt, torch.zeros_like(kurt))
    return kurt.to(torch.float32)


def metric_cache_from_iter01_payload(payload: dict[str, Any]) -> dict[str, dict[int, LayerMetricBundle]] | None:
    cache = payload.get("channel_score_cache")
    if not isinstance(cache, dict):
        return None
    result: dict[str, dict[int, LayerMetricBundle]] = {}
    for metric_name, metric_layers in cache.items():
        if not isinstance(metric_layers, dict):
            return None
        result[metric_name] = {}
        for layer_key, layer_payload in metric_layers.items():
            if not isinstance(layer_payload, dict):
                return None
            result[metric_name][int(layer_key)] = LayerMetricBundle(
                routing_counts=torch.tensor(layer_payload["routing_counts"], dtype=torch.int64),
                w1_pair_scores=torch.tensor(layer_payload["w1_pair_scores"], dtype=torch.float32),
                w2_channel_scores=torch.tensor(layer_payload["w2_channel_scores"], dtype=torch.float32),
            )
    return result


def compute_metric_caches(
    store: WeightStore,
    config: Any,
    captures: dict[int, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[int, LayerMetricBundle]]:
    print("\n=== Recomputing activation_kurtosis + hessian_diag ===", flush=True)
    metric_caches: dict[str, dict[int, LayerMetricBundle]] = {
        "activation_kurtosis": {},
        "hessian_diag": {},
    }

    for layer_idx in range(config.num_hidden_layers):
        capture = captures[layer_idx]
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        print(f"[metrics] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
        raw = store.load_tensors([gate_key, down_key])
        gate_up_proj = move_tensor(raw[gate_key], device, dtype)
        down_proj = move_tensor(raw[down_key], device, dtype)
        del raw

        w1_act_kurt = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
        w2_act_kurt = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
        w1_hessian = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
        w2_hessian = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()

        for expert_pos, expert_idx in enumerate(active_experts, start=1):
            if expert_pos == 1 or expert_pos % 16 == 0 or expert_pos == len(active_experts):
                print(f"  [metrics] expert {expert_pos}/{len(active_experts)} (E{expert_idx})", flush=True)
            token_idx, _route_pos = torch.where(capture.selected_experts == expert_idx)
            if int(token_idx.numel()) == 0:
                continue
            x_in = capture.mlp_input[token_idx].to(device=device, dtype=dtype)
            x_in_f = x_in.float()
            input_kurt = vector_kurtosis(x_in_f, dim=0)
            input_second = x_in_f.square().mean(dim=0)

            gate_up_weight = gate_up_proj[expert_idx]
            gate_up_fp4 = quantize_linear_weight(gate_up_weight, "fp4")
            gate_weight = gate_up_weight[: config.moe_intermediate_size].float()
            up_weight = gate_up_weight[config.moe_intermediate_size :].float()
            gate_diff = gate_weight - gate_up_fp4[: config.moe_intermediate_size].float()
            up_diff = up_weight - gate_up_fp4[config.moe_intermediate_size :].float()
            pair_diff_abs = gate_diff.abs() + up_diff.abs()
            pair_diff_sq = gate_diff.square() + up_diff.square()
            w1_act_kurt[expert_idx] = torch.matmul(pair_diff_abs, input_kurt).cpu()
            w1_hessian[expert_idx] = torch.matmul(pair_diff_sq, input_second).cpu()

            gate_up = F.linear(x_in, gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = (F.silu(gate.float()) * up.float()).to(torch.float32)
            inter_kurt = vector_kurtosis(intermediate, dim=0)
            inter_second = intermediate.square().mean(dim=0)

            down_weight = down_proj[expert_idx]
            down_fp4 = quantize_linear_weight(down_weight, "fp4")
            down_diff = down_weight.float() - down_fp4.float()
            w2_act_kurt[expert_idx] = torch.matmul(down_diff.abs(), inter_kurt).cpu()
            w2_hessian[expert_idx] = torch.matmul(down_diff.square(), inter_second).cpu()

            del x_in, x_in_f, input_kurt, input_second, gate_up_weight, gate_up_fp4, gate_weight, up_weight
            del gate_diff, up_diff, pair_diff_abs, pair_diff_sq, gate_up, gate, up, intermediate, inter_kurt, inter_second
            del down_weight, down_fp4, down_diff
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        release_tensors({"gate_up_proj": gate_up_proj, "down_proj": down_proj})
        metric_caches["activation_kurtosis"][layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_act_kurt,
            w2_channel_scores=w2_act_kurt,
        )
        metric_caches["hessian_diag"][layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_hessian,
            w2_channel_scores=w2_hessian,
        )
    return metric_caches


def make_combined_metric_cache(
    activation_cache: dict[int, LayerMetricBundle],
    hessian_cache: dict[int, LayerMetricBundle],
) -> dict[int, LayerMetricBundle]:
    combined: dict[int, LayerMetricBundle] = {}
    for layer_idx in sorted(activation_cache):
        act_bundle = activation_cache[layer_idx]
        hess_bundle = hessian_cache[layer_idx]
        combined[layer_idx] = LayerMetricBundle(
            routing_counts=act_bundle.routing_counts.clone(),
            w1_pair_scores=(act_bundle.w1_pair_scores * hess_bundle.w1_pair_scores).to(torch.float32),
            w2_channel_scores=(act_bundle.w2_channel_scores * hess_bundle.w2_channel_scores).to(torch.float32),
        )
    return combined


def build_empty_masks(config: Any) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx in range(config.num_hidden_layers):
        w1_pair_masks[layer_idx] = {
            expert_idx: torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
            for expert_idx in range(config.num_experts)
        }
        w2_channel_masks[layer_idx] = {
            expert_idx: torch.zeros(config.hidden_size, dtype=torch.bool)
            for expert_idx in range(config.num_experts)
        }
    return w1_pair_masks, w2_channel_masks


def build_topk_two_level_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_fraction: float,
    w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        weights = [float(value) for value in bundle.routing_counts.tolist()]
        total_w1 = int(round(w1_fraction * config.moe_intermediate_size * config.num_experts))
        total_w2 = int(round(w2_fraction * config.hidden_size * config.num_experts))
        w1_alloc = allocate_weighted_counts(total_w1, [config.moe_intermediate_size] * config.num_experts, weights)
        w2_alloc = allocate_weighted_counts(total_w2, [config.hidden_size] * config.num_experts, weights)
        w1_pair_masks[layer_idx] = {
            expert_idx: topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], int(w1_alloc[expert_idx]))
            for expert_idx in range(config.num_experts)
        }
        w2_channel_masks[layer_idx] = {
            expert_idx: topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], int(w2_alloc[expert_idx]))
            for expert_idx in range(config.num_experts)
        }
    return w1_pair_masks, w2_channel_masks


def active_flat_scores(metric_cache: dict[int, LayerMetricBundle]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for bundle in metric_cache.values():
        active_experts = torch.nonzero(bundle.routing_counts > 0, as_tuple=False).flatten().tolist()
        for expert_idx in active_experts:
            pieces.append(bundle.w1_pair_scores[expert_idx].reshape(-1).to(torch.float32))
            pieces.append(bundle.w2_channel_scores[expert_idx].reshape(-1).to(torch.float32))
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def cluster_high_mask(scores: torch.Tensor, max_iters: int = 32) -> torch.Tensor:
    values = scores.to(torch.float32)
    if int(values.numel()) == 0:
        return torch.zeros(0, dtype=torch.bool)
    low = float(values.min().item())
    high = float(values.max().item())
    if not math.isfinite(low) or not math.isfinite(high) or abs(high - low) <= EPS:
        return torch.zeros(values.numel(), dtype=torch.bool)
    center_low = low
    center_high = high
    mask = torch.zeros(values.numel(), dtype=torch.bool)
    for _ in range(max_iters):
        assign_high = (values - center_high).abs() <= (values - center_low).abs()
        if not bool(assign_high.any()) or bool(assign_high.all()):
            return topk_mask_from_scores(values, values.numel() // 2)
        new_low = float(values[~assign_high].mean().item())
        new_high = float(values[assign_high].mean().item())
        mask = assign_high if new_high >= new_low else ~assign_high
        if abs(new_low - center_low) <= 1e-6 and abs(new_high - center_high) <= 1e-6:
            break
        center_low, center_high = new_low, new_high
    if not bool(mask.any()) or bool(mask.all()):
        return topk_mask_from_scores(values, values.numel() // 2)
    return mask.cpu()


def weak_gap_mask(scores: torch.Tensor, max_fraction: float = 0.5) -> torch.Tensor:
    values = scores.to(torch.float32)
    count = int(values.numel())
    if count <= 0:
        return torch.zeros(0, dtype=torch.bool)
    if count == 1:
        return torch.ones(1, dtype=torch.bool)
    sorted_values, sorted_indices = torch.sort(values, descending=True)
    diffs = sorted_values[:-1] - sorted_values[1:]
    best_idx = int(torch.argmax(diffs).item())
    keep = best_idx + 1
    keep_cap = max(1, int(math.floor(max_fraction * count)))
    keep = min(max(1, keep), keep_cap)
    mask = torch.zeros(count, dtype=torch.bool)
    mask[sorted_indices[:keep]] = True
    return mask


def build_per_expert_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    selector: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    for layer_idx, bundle in metric_cache.items():
        active_experts = torch.nonzero(bundle.routing_counts > 0, as_tuple=False).flatten().tolist()
        for expert_idx in active_experts:
            w1_pair_masks[layer_idx][expert_idx] = selector(bundle.w1_pair_scores[expert_idx])
            w2_channel_masks[layer_idx][expert_idx] = selector(bundle.w2_channel_scores[expert_idx])
    return w1_pair_masks, w2_channel_masks


def build_threshold_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], float]:
    all_scores = active_flat_scores(metric_cache)
    threshold = float((all_scores.mean() + all_scores.std(unbiased=False)).item()) if int(all_scores.numel()) > 0 else 0.0
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    for layer_idx, bundle in metric_cache.items():
        active_experts = torch.nonzero(bundle.routing_counts > 0, as_tuple=False).flatten().tolist()
        for expert_idx in active_experts:
            w1_pair_masks[layer_idx][expert_idx] = (bundle.w1_pair_scores[expert_idx] > threshold).cpu()
            w2_channel_masks[layer_idx][expert_idx] = (bundle.w2_channel_scores[expert_idx] > threshold).cpu()
    return w1_pair_masks, w2_channel_masks, threshold


def total_pair_fraction(config: Any, w1_pair_masks: dict[int, dict[int, torch.Tensor]]) -> float:
    total_pairs = config.num_hidden_layers * config.num_experts * config.moe_intermediate_size
    selected_pairs = sum(int(mask.sum().item()) for layer_masks in w1_pair_masks.values() for mask in layer_masks.values())
    return float(selected_pairs) / float(total_pairs)


def total_channel_fraction(config: Any, w2_channel_masks: dict[int, dict[int, torch.Tensor]]) -> float:
    total_channels = config.num_hidden_layers * config.num_experts * config.hidden_size
    selected_channels = sum(int(mask.sum().item()) for layer_masks in w2_channel_masks.values() for mask in layer_masks.values())
    return float(selected_channels) / float(total_channels)


def build_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_channel_masks: dict[int, dict[int, torch.Tensor]],
) -> EvaluationPlan:
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvaluationPlan(
        name=name,
        description=description,
        mode="mixed_channel",
        memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def build_strategy_plans(
    activation_cache: dict[int, LayerMetricBundle],
    hessian_cache: dict[int, LayerMetricBundle],
    combined_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[list[EvaluationPlan], dict[str, dict[str, Any]]]:
    strategy_meta: dict[str, dict[str, Any]] = {}
    plans: list[EvaluationPlan] = []

    def register(plan: EvaluationPlan, metric_name: str, budget_fixed: bool, extras: dict[str, Any] | None = None) -> None:
        strategy_meta[plan.name] = {
            "metric": metric_name,
            "budget_fixed": budget_fixed,
        }
        if extras:
            strategy_meta[plan.name].update(extras)
        plans.append(plan)

    w1_masks, w2_masks = build_topk_two_level_masks(activation_cache, config, TARGET_FP8_FRACTION, TARGET_FP8_FRACTION)
    register(
        build_plan(
            "sort_and_split",
            "Two-level routing-aware top-k split using activation_kurtosis at 25%/25%.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "activation_kurtosis",
        True,
    )

    w1_masks, w2_masks = build_topk_two_level_masks(hessian_cache, config, TARGET_FP8_FRACTION, TARGET_FP8_FRACTION)
    register(
        build_plan(
            "sort_and_split_hessian",
            "Two-level routing-aware top-k split using hessian_diag at 25%/25%.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "hessian_diag",
        True,
    )

    w1_masks, w2_masks, threshold = build_threshold_masks(activation_cache, config)
    register(
        build_plan(
            "threshold_auto",
            "Global mean+std threshold over activation_kurtosis across all active W1/W2 channels.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "activation_kurtosis",
        False,
        {"threshold": threshold},
    )

    w1_masks, w2_masks = build_per_expert_masks(activation_cache, config, cluster_high_mask)
    register(
        build_plan(
            "kmeans_2cluster",
            "Per-expert 2-cluster split on activation_kurtosis; high cluster promoted to FP8.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "activation_kurtosis",
        False,
    )

    w1_masks, w2_masks = build_per_expert_masks(activation_cache, config, weak_gap_mask)
    register(
        build_plan(
            "weak_column_gap",
            "Per-expert largest-gap split on activation_kurtosis with a 50% cap.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "activation_kurtosis",
        False,
    )

    w1_masks, w2_masks = build_topk_two_level_masks(activation_cache, config, PER_PROJECTION_W1_FRACTION, PER_PROJECTION_W2_FRACTION)
    register(
        build_plan(
            "per_projection_split",
            "Two-level routing-aware split with W1=10% FP8 and W2=40% FP8 using activation_kurtosis.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "activation_kurtosis",
        True,
        {"w1_fraction_target": PER_PROJECTION_W1_FRACTION, "w2_fraction_target": PER_PROJECTION_W2_FRACTION},
    )

    w1_masks, w2_masks = build_topk_two_level_masks(activation_cache, config, TARGET_FP8_FRACTION, TARGET_FP8_FRACTION)
    register(
        build_plan(
            "submodular_greedy",
            "Routing-aware greedy promotion by activation_kurtosis until the 25% budget is filled.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "activation_kurtosis",
        True,
        {"equivalent_to": "sort_and_split"},
    )

    w1_masks, w2_masks = build_topk_two_level_masks(combined_cache, config, TARGET_FP8_FRACTION, TARGET_FP8_FRACTION)
    register(
        build_plan(
            "combined_metric",
            "Two-level routing-aware top-k split using hessian_diag * activation_kurtosis.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        "hessian_diag*activation_kurtosis",
        True,
    )

    return plans, strategy_meta


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: float(item[1]["ppl"]))
    print("\n" + "=" * 108, flush=True)
    print("Iteration 2 assignment strategies (sorted by PPL)", flush=True)
    print("=" * 108, flush=True)
    print(
        f"{'Strategy':<24} {'PPL':>9} {'NLL':>10} {'Memory GB':>11} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 108, flush=True)
    for name, row in ordered:
        print(
            f"{name:<24} {float(row['ppl']):>9.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>11.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_exploration_section(payload: dict[str, Any]) -> str:
    strategies = payload["strategies"]
    ordered = sorted(strategies.items(), key=lambda item: float(item[1]["ppl"]))
    best_name, best_row = ordered[0]
    gap = float(best_row["ppl"]) - MXMOE_TARGET_PPL
    table_lines = [
        "| Strategy | Metric | PPL | Memory (GB) | FP8 fraction |",
        "|----------|--------|-----|-------------|--------------|",
    ]
    for name, row in ordered:
        table_lines.append(
            f"| {name} | {row['metric']} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} | {float(row['fp8_fraction']):.4f} |"
        )
    if gap <= 0.0:
        verdict = f"closes the gap and beats the 5.337 MxMoE target by {abs(gap):.4f} PPL"
    else:
        verdict = f"does not close the gap; the remaining margin to 5.337 is {gap:.4f} PPL"
    return "\n".join([
        SECTION_MARKER,
        "**Approach**: Reused the streamed 40-layer perplexity loop and real mixed-precision MoE forward from `baselines_comparison.py`, then compared eight assignment rules at the nominal 25% FP8 operating point. `activation_kurtosis` was the Iteration 1 winner, `hessian_diag` was the highest-Spearman alternative, and strategies with adaptive thresholds/clusters/gaps were allowed to spend data-driven budgets.",
        f"**Result**: The best strategy in this sweep is `{best_name}` at PPL {float(best_row['ppl']):.4f}, {float(best_row['memory_gb']):.3f} GB, and FP8 fraction {float(best_row['fp8_fraction']):.4f}; it {verdict}. Thresholded and clustered strategies spend materially different budgets, so memory-normalized comparisons matter as much as raw PPL.",
        "\n".join(table_lines),
        "**Insight**: The experiment separates two effects that were entangled in Iteration 1: which sensitivity signal ranks channels best, and whether fixed top-k splitting is actually the right assignment rule once routing-aware expert allocation is already in place. The output JSON records both the realized FP8 fraction and the per-projection fractions so later sweeps can compare iso-memory and non-iso-memory variants cleanly.",
        "",
    ])


def upsert_exploration_section(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SECTION_MARKER in existing:
        prefix = existing.split(SECTION_MARKER, 1)[0].rstrip()
        updated = prefix + "\n\n" + section
    else:
        updated = existing.rstrip() + "\n\n" + section if existing.strip() else section
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print("=== Iteration 2: Assignment Strategy Comparison ===", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Device: {device} | dtype: {dtype}", flush=True)

    iter01_payload = load_json(args.iter01_json)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    calib_ids, calib_info = load_prefix_dataset_tokens(tokenizer, "train", args.calibration_tokens)
    eval_ids, eval_info = load_prefix_dataset_tokens(tokenizer, "test", args.eval_tokens)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    cached_metrics = metric_cache_from_iter01_payload(iter01_payload)
    metric_source = "iter01_json"
    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    captures: dict[int, Any] | None = None
    if cached_metrics is None:
        metric_source = "recomputed_from_calibration"
        captures = run_calibration_pass(store, text_config, calib_ids, args.mini_batch_tokens, device, dtype)
        metric_caches = compute_metric_caches(store, text_config, captures, device, dtype)
    else:
        metric_caches = cached_metrics

    activation_cache = metric_caches["activation_kurtosis"]
    hessian_cache = metric_caches["hessian_diag"]
    combined_cache = make_combined_metric_cache(activation_cache, hessian_cache)
    del captures

    plans, strategy_meta = build_strategy_plans(
        activation_cache,
        hessian_cache,
        combined_cache,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": str(device),
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "calibration_tokens": int(calib_ids.shape[1]),
            "eval_tokens": int(eval_ids.shape[1]),
            "mini_batch_tokens": int(args.mini_batch_tokens),
            "target_fp8_fraction": TARGET_FP8_FRACTION,
            "per_projection_split": {
                "w1_fraction": PER_PROJECTION_W1_FRACTION,
                "w2_fraction": PER_PROJECTION_W2_FRACTION,
            },
            "iter01_score_source": metric_source,
            "iter01_json": str(args.iter01_json),
            "mxmoe_target_ppl": MXMOE_TARGET_PPL,
            "notes": {
                "threshold_auto": "Global threshold is mean + std over active W1/W2 activation_kurtosis scores.",
                "kmeans_2cluster": "Manual 1D 2-means fallback; high cluster promoted per expert/projection.",
                "weak_column_gap": "Largest descending-score gap per expert/projection, capped at 50% promotion.",
                "submodular_greedy": "Equivalent to sort-and-split under uniform item size within each projection.",
            },
        },
        "strategies": {},
        "best_strategy": None,
        "best_ppl": None,
        "runtime_seconds": None,
    }
    atomic_json_dump(args.output_json, payload)

    for idx, plan in enumerate(plans, start=1):
        print(f"\n=== Eval {idx}/{len(plans)}: {plan.name} ===", flush=True)
        t0 = time.time()
        ppl, nll = evaluate_plan(plan, eval_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - t0
        w1_pair_fraction = total_pair_fraction(text_config, plan.w1_pair_masks or {})
        w2_channel_fraction = total_channel_fraction(text_config, plan.w2_channel_masks or {})
        fp8_fraction = float(plan.fp8_weights) / float(total_expert_elems)
        row = {
            "description": plan.description,
            "metric": strategy_meta[plan.name]["metric"],
            "budget_fixed": bool(strategy_meta[plan.name]["budget_fixed"]),
            "ppl": round(ppl, 6),
            "nll": round(nll, 6),
            "memory_gb": plan.memory_gb,
            "fp8_weights": int(plan.fp8_weights),
            "fp8_fraction": round(fp8_fraction, 6),
            "w1_pair_fraction": round(w1_pair_fraction, 6),
            "w2_channel_fraction": round(w2_channel_fraction, 6),
            "time_s": round(elapsed, 1),
        }
        for key, value in strategy_meta[plan.name].items():
            if key in {"metric", "budget_fixed"}:
                continue
            row[key] = value
        payload["strategies"][plan.name] = row
        atomic_json_dump(args.output_json, payload)
        print(
            f"  -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={fp8_fraction:.4f} | time={elapsed:.1f}s",
            flush=True,
        )

    if payload["strategies"]:
        best_name, best_row = min(payload["strategies"].items(), key=lambda item: float(item[1]["ppl"]))
        payload["best_strategy"] = best_name
        payload["best_ppl"] = float(best_row["ppl"])
    payload["runtime_seconds"] = round(time.time() - start_time, 3)
    atomic_json_dump(args.output_json, payload)

    print_results_table(payload["strategies"])
    upsert_exploration_section(args.exploration_md, build_exploration_section(payload))
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

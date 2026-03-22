#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines_comparison import (
    LayerMetricBundle,
    allocate_weighted_counts,
    estimate_mixed_memory_gb,
    expert_full_weight_count,
    fp8_weights_from_projection_promotions,
    quantize_linear_weight,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
    topk_mask_from_scores,
)
from proper_eval import (
    CALIBRATION_SAMPLES,
    SEQLEN,
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    build_two_level_masks,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    layer_type_at,
    load_gptq_standard_data,
)
from proper_iter01 import build_empty_masks, build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter07.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter07_scalebits_cache.pt"
EPS = 1e-10
METRIC_CACHE_VERSION = 1
SCALEBITS_GRADIENT_CHUNKS = 16
CASCADE_TOTAL_FRACTION = 0.25
MXMOE_20PCT_TOPUP_FRACTION = 0.08
JOINT_MEDIUM_TOPUP_FRACTION = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 7 plans.",
    )
    parser.add_argument("--gradient-calibration-chunks", type=int, default=SCALEBITS_GRADIENT_CHUNKS)
    parser.add_argument("--disable-gradient-term", action="store_true")
    parser.add_argument("--force-recompute-metric", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def clone_metric_cache(source: dict[int, LayerMetricBundle]) -> dict[int, LayerMetricBundle]:
    cloned: dict[int, LayerMetricBundle] = {}
    for layer_idx, bundle in source.items():
        cloned[int(layer_idx)] = LayerMetricBundle(
            routing_counts=bundle.routing_counts.detach().cpu().clone(),
            w1_pair_scores=bundle.w1_pair_scores.detach().cpu().clone(),
            w2_channel_scores=bundle.w2_channel_scores.detach().cpu().clone(),
        )
    return cloned


def metric_cache_is_compatible(
    payload: dict[str, Any],
    model_id: str,
    use_gradient_term: bool,
    gradient_chunks: int,
) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
        and bool(payload.get("gradient_term_used", False)) == bool(use_gradient_term)
        and int(payload.get("gradient_calibration_chunks", -1)) == int(gradient_chunks)
    )


def load_metric_cache(
    path: Path,
    model_id: str,
    use_gradient_term: bool,
    gradient_chunks: int,
) -> tuple[dict[int, torch.Tensor] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id, use_gradient_term, gradient_chunks):
        return None, None
    raw_scores = payload.get("w2_scores")
    if not isinstance(raw_scores, dict):
        return None, None
    restored = {int(layer_idx): tensor.detach().cpu().clone().to(torch.float32) for layer_idx, tensor in raw_scores.items()}
    metadata = payload.get("metadata")
    return restored, dict(metadata) if isinstance(metadata, dict) else {}


def save_metric_cache(
    path: Path,
    model_id: str,
    use_gradient_term: bool,
    gradient_chunks: int,
    w2_scores: dict[int, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "calibration_samples": CALIBRATION_SAMPLES,
            "gradient_term_used": bool(use_gradient_term),
            "gradient_calibration_chunks": int(gradient_chunks),
            "w2_scores": {int(layer_idx): scores.detach().cpu().clone() for layer_idx, scores in w2_scores.items()},
            "metadata": metadata,
        },
        path,
    )


def tensor_gini(values: torch.Tensor) -> float:
    flat = values.detach().cpu().to(torch.float64).flatten()
    if flat.numel() == 0:
        return 0.0
    flat = torch.clamp(flat, min=0.0)
    total = float(flat.sum().item())
    if total <= 0.0:
        return 0.0
    sorted_values = torch.sort(flat).values
    count = int(sorted_values.numel())
    indices = torch.arange(1, count + 1, dtype=torch.float64)
    gini = (2.0 * torch.sum(indices * sorted_values) / (count * total)) - ((count + 1) / count)
    return float(gini.item())


def summarize_w2_scores(w2_scores: dict[int, torch.Tensor], routing_counts: dict[int, torch.Tensor]) -> dict[str, Any]:
    ginis: list[float] = []
    layer_means: dict[str, float] = {}
    total_active = 0
    for layer_idx, scores in w2_scores.items():
        layer_counts = routing_counts[layer_idx]
        layer_ginis: list[float] = []
        for expert_idx in range(int(layer_counts.numel())):
            if int(layer_counts[expert_idx].item()) <= 0:
                continue
            gini_value = tensor_gini(scores[expert_idx])
            ginis.append(gini_value)
            layer_ginis.append(gini_value)
            total_active += 1
        if layer_ginis:
            layer_means[str(layer_idx)] = round(sum(layer_ginis) / len(layer_ginis), 6)
    if not ginis:
        return {
            "active_experts": 0,
            "mean_gini": 0.0,
            "median_gini": 0.0,
            "min_gini": 0.0,
            "max_gini": 0.0,
            "experts_in_040_060_band": 0,
            "layer_mean_gini": layer_means,
        }
    ordered = sorted(ginis)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
    in_band = sum(1 for value in ordered if 0.40 <= value <= 0.60)
    return {
        "active_experts": int(total_active),
        "mean_gini": round(sum(ordered) / len(ordered), 6),
        "median_gini": round(float(median), 6),
        "min_gini": round(min(ordered), 6),
        "max_gini": round(max(ordered), 6),
        "experts_in_040_060_band": int(in_band),
        "layer_mean_gini": layer_means,
    }


def build_w2_metric_cache(
    activation_cache: dict[int, LayerMetricBundle],
    w2_scores: dict[int, torch.Tensor],
    metric_name: str,
) -> tuple[dict[int, LayerMetricBundle], dict[str, Any]]:
    metric_cache = clone_metric_cache(activation_cache)
    total_active_experts = 0
    backed_active_experts = 0
    for layer_idx, _bundle in metric_cache.items():
        active_mask = activation_cache[layer_idx].routing_counts > 0
        total_active_experts += int(active_mask.sum().item())
        if layer_idx not in w2_scores:
            continue
        layer_scores = w2_scores[layer_idx].detach().cpu().to(torch.float32)
        metric_cache[layer_idx].w2_channel_scores = layer_scores
        backed_active_experts += int(active_mask.sum().item())
    return metric_cache, {
        "channel_metric_requested": metric_name,
        "channel_metric_effective": metric_name,
        "channel_metric_fallback_used": False,
        "scalebits_backed_active_experts": int(backed_active_experts),
        "total_active_experts": int(total_active_experts),
    }


def build_mxmoe_projection_promotions_fraction(
    config: Any,
    total_expert_elems: int,
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
    fraction: float,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    target_fp8_weights = int(round(fraction * total_expert_elems))
    w1_cost = 2 * config.moe_intermediate_size * config.hidden_size
    w2_cost = config.hidden_size * config.moe_intermediate_size
    items: list[tuple[float, int, int, int, str]] = []
    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            items.append((float(w1_deltas[layer_idx][expert_idx].item()) / float(w1_cost + EPS), w1_cost, layer_idx, expert_idx, "w1"))
            items.append((float(w2_deltas[layer_idx][expert_idx].item()) / float(w2_cost + EPS), w2_cost, layer_idx, expert_idx, "w2"))
    items.sort(key=lambda item: (item[0], -item[1], -item[2], -item[3], item[4]), reverse=True)

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    fp8_weights = 0
    for ratio, cost, layer_idx, expert_idx, projection in items:
        if ratio <= 0.0:
            continue
        if fp8_weights + cost > target_fp8_weights:
            continue
        if projection == "w1":
            w1_projection_fp8[layer_idx][expert_idx] = True
        else:
            w2_projection_fp8[layer_idx][expert_idx] = True
        fp8_weights += cost
        if fp8_weights >= target_fp8_weights:
            break
    return w1_projection_fp8, w2_projection_fp8


def build_per_block_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_proj: dict[int, torch.Tensor],
    w2_proj: dict[int, torch.Tensor],
) -> EvalPlan:
    fp8_weights = fp8_weights_from_projection_promotions(config, w1_proj, w2_proj)
    return EvalPlan(
        name=name,
        description=description,
        mode="per_block",
        memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_projection_fp8=w1_proj,
        w2_projection_fp8=w2_proj,
    )


def build_dynamic_oracle_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    fraction: float,
) -> EvalPlan:
    promoted_per_layer = max(0, min(config.num_experts, int(round(fraction * config.num_experts))))
    fp8_weights = promoted_per_layer * config.num_hidden_layers * expert_full_weight_count(config)
    return EvalPlan(
        name=name,
        description=description,
        mode="dynamic_oracle",
        memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
    )


def build_mxmoe_topup_masks_for_w2_metric(
    metric_cache: dict[int, LayerMetricBundle],
    calibration: CalibrationArtifacts,
    config: Any,
    total_expert_elems: int,
    base_fraction: float,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        base_fraction,
    )
    topup_w1 = int(round(topup_fraction * config.moe_intermediate_size))
    topup_w2 = int(round(topup_fraction * config.hidden_size))
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)

    for layer_idx in range(config.num_hidden_layers):
        activation_bundle = calibration.activation_cache[layer_idx]
        metric_bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            else:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(activation_bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(metric_bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "mxmoe_per_block_plus_channel_topup",
        "mxmoe_base_fraction": float(base_fraction),
        "topup_fraction": float(topup_fraction),
        "w1_topup_fraction": float(topup_fraction),
        "w2_topup_fraction": float(topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def resolve_joint_precision_tiers(
    calibration: CalibrationArtifacts,
    config: Any,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    high_mask = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    medium_mask = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    active_scores: list[torch.Tensor] = []

    for layer_idx in range(config.num_hidden_layers):
        combined_scores = (calibration.mxmoe_w1_deltas[layer_idx] + calibration.mxmoe_w2_deltas[layer_idx]).detach().cpu().to(torch.float32)
        active_mask = calibration.routing_counts[layer_idx] > 0
        if bool(active_mask.any()):
            active_scores.append(combined_scores[active_mask])

    if active_scores:
        flat_active = torch.cat(active_scores, dim=0)
        p50 = float(torch.quantile(flat_active, 0.50).item())
        p75 = float(torch.quantile(flat_active, 0.75).item())
    else:
        p50 = 0.0
        p75 = 0.0

    high_count = 0
    medium_count = 0
    low_count = 0
    for layer_idx in range(config.num_hidden_layers):
        combined_scores = (calibration.mxmoe_w1_deltas[layer_idx] + calibration.mxmoe_w2_deltas[layer_idx]).detach().cpu().to(torch.float32)
        active_mask = calibration.routing_counts[layer_idx] > 0
        for expert_idx in range(config.num_experts):
            if not bool(active_mask[expert_idx]):
                low_count += 1
                continue
            score = float(combined_scores[expert_idx].item())
            if score >= p75:
                high_mask[layer_idx][expert_idx] = True
                high_count += 1
            elif score >= p50:
                medium_mask[layer_idx][expert_idx] = True
                medium_count += 1
            else:
                low_count += 1

    return high_mask, medium_mask, {
        "budget_source": "global_perturbation_percentile_buckets_with_medium_topup",
        "joint_precision_threshold_p50": float(p50),
        "joint_precision_threshold_p75": float(p75),
        "joint_precision_high_experts": int(high_count),
        "joint_precision_medium_experts": int(medium_count),
        "joint_precision_low_experts": int(low_count),
    }


def build_joint_with_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    high_mask, medium_mask, tier_meta = resolve_joint_precision_tiers(calibration, config)
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    medium_w1_pairs = int(round(topup_fraction * config.moe_intermediate_size))
    medium_w2_channels = int(round(topup_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        activation_bundle = calibration.activation_cache[layer_idx]
        metric_bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(high_mask[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            elif bool(medium_mask[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(activation_bundle.w1_pair_scores[expert_idx], medium_w1_pairs)
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(metric_bundle.w2_channel_scores[expert_idx], medium_w2_channels)

    return w1_pair_masks, w2_channel_masks, {
        **tier_meta,
        "medium_tier_w1_fraction": float(topup_fraction),
        "medium_tier_w2_fraction": float(topup_fraction),
        "medium_tier_w1_pairs": int(medium_w1_pairs),
        "medium_tier_w2_channels": int(medium_w2_channels),
    }


def build_layer_weight_schedule(num_layers: int, reverse: bool) -> list[float]:
    if reverse:
        return [float(layer_idx + 1) for layer_idx in range(num_layers)]
    return [float(num_layers - layer_idx) for layer_idx in range(num_layers)]


def build_layer_budget_projection_promotions(
    config: Any,
    total_expert_elems: int,
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
    total_fraction: float,
    layer_weights: list[float],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    if len(layer_weights) != config.num_hidden_layers:
        raise ValueError("layer_weights must match num_hidden_layers")

    target_fp8_weights = int(round(total_fraction * total_expert_elems))
    per_layer_capacity = [config.num_experts * expert_full_weight_count(config) for _ in range(config.num_hidden_layers)]
    per_layer_targets = allocate_weighted_counts(target_fp8_weights, per_layer_capacity, [max(0.0, weight) for weight in layer_weights])
    w1_cost = 2 * config.moe_intermediate_size * config.hidden_size
    w2_cost = config.hidden_size * config.moe_intermediate_size

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    realized_total = 0
    realized_by_layer: dict[str, int] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_budget = int(per_layer_targets[layer_idx])
        items: list[tuple[float, int, int, str]] = []
        for expert_idx in range(config.num_experts):
            items.append((float(w1_deltas[layer_idx][expert_idx].item()) / float(w1_cost + EPS), w1_cost, expert_idx, "w1"))
            items.append((float(w2_deltas[layer_idx][expert_idx].item()) / float(w2_cost + EPS), w2_cost, expert_idx, "w2"))
        items.sort(key=lambda item: (item[0], -item[1], -item[2], item[3]), reverse=True)

        layer_realized = 0
        for ratio, cost, expert_idx, projection in items:
            if ratio <= 0.0:
                continue
            if layer_realized + cost > layer_budget:
                continue
            if projection == "w1":
                w1_projection_fp8[layer_idx][expert_idx] = True
            else:
                w2_projection_fp8[layer_idx][expert_idx] = True
            layer_realized += cost
            if layer_realized >= layer_budget:
                break
        realized_total += layer_realized
        realized_by_layer[str(layer_idx)] = int(layer_realized)

    return w1_projection_fp8, w2_projection_fp8, {
        "budget_source": "layer_weighted_mxmoe_projection_budget",
        "mxmoe_fraction_target": float(total_fraction),
        "target_fp8_weights": int(target_fp8_weights),
        "realized_fp8_weights": int(realized_total),
        "layer_weights": {str(layer_idx): float(layer_weights[layer_idx]) for layer_idx in range(config.num_hidden_layers)},
        "layer_target_fp8_weights": {str(layer_idx): int(per_layer_targets[layer_idx]) for layer_idx in range(config.num_hidden_layers)},
        "layer_realized_fp8_weights": realized_by_layer,
    }


def prepare_routing(mlp_input: torch.Tensor, gate_weight: torch.Tensor, config: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = mlp_input.shape
    flat = mlp_input.view(batch_size * sequence_length, hidden_dim)
    router_logits = F.linear(flat, gate_weight).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(mlp_input.dtype)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    return flat, routing_weights, selected_experts, expert_counts


def add_shared_expert(flat: torch.Tensor, base_output: torch.Tensor, moe_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    return base_output + (shared_out * shared_gate_value)


def forward_bf16_moe_with_fixed_routing(
    flat: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_counts: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    hidden_dim = int(flat.shape[-1])
    final_hidden_states = torch.zeros((int(flat.shape[0]), hidden_dim), dtype=flat.dtype, device=flat.device)
    gate_up_proj = moe_tensors["experts.gate_up_proj"]
    down_proj = moe_tensors["experts.down_proj"]
    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        weights = routing_weights[token_idx, route_pos].unsqueeze(-1)
        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        out = F.linear(hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (weights * out).to(flat.dtype))
    return add_shared_expert(flat, final_hidden_states, moe_tensors)


def forward_quantized_moe_with_fixed_routing(
    flat: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_counts: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    retain_output_grad: bool,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    hidden_dim = int(flat.shape[-1])
    final_hidden_states = torch.zeros((int(flat.shape[0]), hidden_dim), dtype=flat.dtype, device=flat.device)
    hidden_by_expert: dict[int, torch.Tensor] = {}
    out_by_expert: dict[int, torch.Tensor] = {}
    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        weights = routing_weights[token_idx, route_pos].unsqueeze(-1)
        gate_up = F.linear(current_state, fp4_gate_up[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        out = F.linear(hidden, fp4_down[expert_idx])
        if retain_output_grad:
            out.retain_grad()
        hidden_by_expert[int(expert_idx)] = hidden
        out_by_expert[int(expert_idx)] = out
        final_hidden_states.index_add_(0, token_idx, (weights * out).to(flat.dtype))
    return add_shared_expert(flat, final_hidden_states, moe_tensors), hidden_by_expert, out_by_expert


def compute_scalebits_w2_scores(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    routing_counts: dict[int, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    use_gradient_term: bool,
    gradient_chunks: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    metric_name = "scalebits_gxdelta_local_mse" if use_gradient_term else "act_weighted_qerror_local_mse"
    active_chunks = calib_chunks[:gradient_chunks].contiguous()
    print(f"\n=== Iteration 7 metric pass: {metric_name} on {active_chunks.shape[0]} calibration chunks ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, active_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)
    w2_scores: dict[int, torch.Tensor] = {}
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[metric] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        down_delta_abs = {
            expert_idx: (down_proj[expert_idx].float() - fp4_down[expert_idx].float()).abs().cpu()
            for expert_idx in range(config.num_experts)
            if int(routing_counts[layer_idx][expert_idx].item()) > 0
        }
        layer_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float64)

        for chunk_idx in range(inps.shape[0]):
            with torch.no_grad():
                hidden_states = inps[chunk_idx].unsqueeze(0)
                residual = hidden_states
                hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
                if layer_type == "full_attention":
                    attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                    mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
                else:
                    attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                    mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
                hidden_after_attn = residual + mixed
                mlp_input = rms_norm_qwen3_next(hidden_after_attn, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
                batch_size, sequence_length, hidden_dim = mlp_input.shape
                flat_bf16, routing_weights, selected_experts, expert_counts = prepare_routing(mlp_input, moe_tensors["gate.weight"], config)
                bf16_moe_out_flat = forward_bf16_moe_with_fixed_routing(flat_bf16, routing_weights, selected_experts, expert_counts, moe_tensors)
                bf16_moe_out = bf16_moe_out_flat.view(batch_size, sequence_length, hidden_dim)
                outs[chunk_idx] = hidden_after_attn + bf16_moe_out

            mlp_input_quant = mlp_input.detach().requires_grad_(use_gradient_term)
            flat_quant = mlp_input_quant.view(batch_size * sequence_length, hidden_dim)
            quant_moe_out_flat, hidden_by_expert, out_by_expert = forward_quantized_moe_with_fixed_routing(
                flat_quant,
                routing_weights,
                selected_experts,
                expert_counts,
                moe_tensors,
                fp4_gate_up,
                fp4_down,
                use_gradient_term,
            )
            quant_moe_out = quant_moe_out_flat.view(batch_size, sequence_length, hidden_dim)
            loss = F.mse_loss(quant_moe_out.float(), bf16_moe_out.float(), reduction="mean")
            if use_gradient_term:
                loss.backward()

            for expert_idx, hidden in hidden_by_expert.items():
                x_abs = hidden.detach().abs().to(torch.float32)
                delta_abs = down_delta_abs[expert_idx].to(torch.float32)
                if use_gradient_term:
                    grad = out_by_expert[expert_idx].grad
                    if grad is None:
                        raise RuntimeError(f"Missing expert output gradient for layer {layer_idx}, expert {expert_idx}")
                    g_abs = grad.detach().abs().to(torch.float32)
                    cross = torch.matmul(g_abs.transpose(0, 1), x_abs)
                    score = torch.sum(cross * delta_abs.to(device=cross.device), dim=1)
                else:
                    score = torch.matmul(delta_abs, x_abs.sum(dim=0).cpu()).to(torch.float32)
                layer_scores[expert_idx].add_(score.detach().cpu().to(torch.float64))

            del hidden_states, residual, hidden_norm, mixed, hidden_after_attn, mlp_input, mlp_input_quant
            del flat_bf16, flat_quant, routing_weights, selected_experts, expert_counts
            del bf16_moe_out_flat, bf16_moe_out, quant_moe_out_flat, quant_moe_out, hidden_by_expert, out_by_expert, loss
            if chunk_idx == 0 or (chunk_idx + 1) % 8 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[metric] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)
            gc.collect()
            if torch.cuda.is_available() and ((chunk_idx + 1) % 8 == 0 or chunk_idx + 1 == inps.shape[0]):
                torch.cuda.empty_cache()

        w2_scores[layer_idx] = layer_scores.to(torch.float32)
        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_name": metric_name,
        "gradient_term_used": bool(use_gradient_term),
        "gradient_calibration_chunks": int(active_chunks.shape[0]),
        "available_calibration_chunks": int(calib_chunks.shape[0]),
        "loss_proxy": "layer_local_mse_to_bf16_moe_output",
        "draft_quant_model": "fp4_gate_up_and_down",
        "runtime_seconds": round(time.time() - start_time, 3),
        "gini_summary": summarize_w2_scores(w2_scores, routing_counts),
    }
    return w2_scores, metadata


def resolve_scalebits_scores(
    metric_cache_path: Path,
    force_recompute: bool,
    model_id: str,
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    calibration: CalibrationArtifacts,
    device: torch.device,
    dtype: torch.dtype,
    use_gradient_term: bool,
    gradient_chunks: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if not force_recompute:
        cached_scores, cached_meta = load_metric_cache(metric_cache_path, model_id, use_gradient_term, gradient_chunks)
        if cached_scores is not None and cached_meta is not None:
            print(f"[metric-cache] loaded {metric_cache_path}", flush=True)
            return cached_scores, cached_meta

    try:
        scores, metadata = compute_scalebits_w2_scores(
            store,
            config,
            calib_chunks,
            calibration.routing_counts,
            device,
            dtype,
            use_gradient_term,
            gradient_chunks,
        )
        save_metric_cache(metric_cache_path, model_id, use_gradient_term, gradient_chunks, scores, metadata)
        print(f"[metric-cache] saved {metric_cache_path}", flush=True)
        return scores, metadata
    except RuntimeError as exc:
        if (not use_gradient_term) or ("out of memory" not in str(exc).lower() and "cuda" not in str(exc).lower()):
            raise
        print(f"[metric-cache] gradient metric failed ({exc}); retrying without gradient term", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scores, metadata = compute_scalebits_w2_scores(
            store,
            config,
            calib_chunks,
            calibration.routing_counts,
            device,
            dtype,
            False,
            gradient_chunks,
        )
        metadata = {
            **metadata,
            "fallback_from_gradient_term": True,
            "fallback_reason": str(exc),
        }
        save_metric_cache(metric_cache_path, model_id, False, gradient_chunks, scores, metadata)
        print(f"[metric-cache] saved fallback metric cache {metric_cache_path}", flush=True)
        return scores, metadata


def select_top_experts_by_counts(counts: torch.Tensor, fraction: float) -> set[int]:
    target = max(0, min(int(counts.numel()), int(round(float(fraction) * float(counts.numel())))))
    if target <= 0:
        return set()
    ranked = sorted(range(int(counts.numel())), key=lambda expert_idx: (-int(counts[expert_idx].item()), expert_idx))
    return set(ranked[:target])


def dynamic_oracle_moe_forward_eval(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    fp8_gate_up_cache: dict[int, torch.Tensor],
    fp8_down_cache: dict[int, torch.Tensor],
    config: Any,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, set[int]]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    promoted = select_top_experts_by_counts(expert_counts, fraction)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        weights = routing_weights[token_idx, route_pos].unsqueeze(-1)
        if expert_idx in promoted:
            quant_gate_up = fp8_gate_up_cache.get(expert_idx)
            if quant_gate_up is None:
                quant_gate_up = quantize_linear_weight(gate_up_proj[expert_idx], "fp8")
                fp8_gate_up_cache[expert_idx] = quant_gate_up
            quant_down = fp8_down_cache.get(expert_idx)
            if quant_down is None:
                quant_down = quantize_linear_weight(down_proj[expert_idx], "fp8")
                fp8_down_cache[expert_idx] = quant_down
        else:
            quant_gate_up = fp4_gate_up[expert_idx]
            quant_down = fp4_down[expert_idx]
        gate_up = F.linear(current_state, quant_gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, quant_down)
        final_hidden_states.index_add_(0, token_idx, (weights * current_hidden).to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), expert_counts, promoted


@torch.inference_mode()
def evaluate_dynamic_oracle_plan(
    plan: EvalPlan,
    test_ids: torch.Tensor,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    fraction: float,
) -> tuple[float, float, int, dict[str, Any]]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    nsamples = int(test_ids.numel() // SEQLEN)
    eval_chunks = test_ids[:, : nsamples * SEQLEN].view(nsamples, SEQLEN)
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

    root_tensors = store.load_tensors([norm_key, lm_head_key])
    final_norm = root_tensors[norm_key].to(device=device, dtype=dtype)
    lm_head = root_tensors[lm_head_key].to(device=device, dtype=dtype)
    del root_tensors

    inps = embed_chunks(store, embed_key, eval_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    active_totals = 0
    promoted_totals = 0
    chunk_decisions = 0

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[{plan.name}] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp8_gate_up_cache: dict[int, torch.Tensor] = {}
        fp8_down_cache: dict[int, torch.Tensor] = {}

        for chunk_idx in range(nsamples):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_out, expert_counts, promoted = dynamic_oracle_moe_forward_eval(
                mlp_input,
                moe_tensors,
                gate_up_proj,
                down_proj,
                fp4_gate_up,
                fp4_down,
                fp8_gate_up_cache,
                fp8_down_cache,
                config,
                fraction,
            )
            outs[chunk_idx] = residual + moe_out
            active_totals += int((expert_counts > 0).sum().item())
            promoted_totals += len(promoted)
            chunk_decisions += 1
            print(f"[{plan.name}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{nsamples}", flush=True)

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors({f"fp8_gate_up_{expert_idx}": tensor for expert_idx, tensor in fp8_gate_up_cache.items()})
        release_tensors({f"fp8_down_{expert_idx}": tensor for expert_idx, tensor in fp8_down_cache.items()})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids_device = test_ids.to(device=device, non_blocking=True)
    loss_fct = nn.CrossEntropyLoss()
    nlls: list[torch.Tensor] = []
    for chunk_idx in range(nsamples):
        hidden_states = inps[chunk_idx].unsqueeze(0)
        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
        logits = F.linear(hidden_states.float(), lm_head.float())
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = test_ids_device[:, chunk_idx * SEQLEN : (chunk_idx + 1) * SEQLEN][:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(loss.float() * SEQLEN)
        print(f"[{plan.name}] logits chunk {chunk_idx + 1}/{nsamples}", flush=True)

    mean_nll = torch.stack(nlls).sum() / (nsamples * SEQLEN)
    ppl = torch.exp(mean_nll)

    release_tensors({
        "inps": inps,
        "outs": outs,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "causal_mask": causal_mask,
        "test_ids_device": test_ids_device,
    })
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta = {
        "dynamic_expert_fraction": float(fraction),
        "dynamic_average_active_experts": round(float(active_totals) / float(max(chunk_decisions, 1)), 6),
        "dynamic_average_promoted_experts": round(float(promoted_totals) / float(max(chunk_decisions, 1)), 6),
        "dynamic_chunk_decisions": int(chunk_decisions),
    }
    return float(ppl.item()), float(mean_nll.item()), nsamples, meta


def load_reference_rows() -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in (
        RESULTS_DIR / "proper_eval.json",
        RESULTS_DIR / "proper_iter01.json",
        RESULTS_DIR / "proper_iter02.json",
        RESULTS_DIR / "proper_iter03.json",
        RESULTS_DIR / "proper_iter04.json",
        RESULTS_DIR / "proper_iter05.json",
        RESULTS_DIR / "proper_iter06.json",
    ):
        if not path.exists():
            continue
        payload = load_json(path)
        for name, row in payload.get("results", {}).items():
            if isinstance(row, dict) and "ppl" in row and "memory_gb" in row:
                references[str(name)] = {
                    "ppl": float(row["ppl"]),
                    "memory_gb": float(row["memory_gb"]),
                }
    return references


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
    eval_meta: dict[str, Any] = {}
    if plan.mode == "dynamic_oracle":
        ppl, nll, nsamples, eval_meta = evaluate_dynamic_oracle_plan(
            plan,
            test_ids,
            config,
            weight_map,
            snapshot_dir,
            device,
            dtype,
            float(extras["dynamic_expert_fraction"]),
        )
    else:
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
        **eval_meta,
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
    reference_best = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 160, flush=True)
    print("proper_iter07 | reduced-memory ScaleBITS + dynamic oracle + cascade-aware budgets | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 160, flush=True)
    print(
        f"{'Config':<38} {'PPL':>10} {'dPrevBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 160, flush=True)
    for name, row in ordered:
        d_prev = "-" if reference_best is None else f"{float(row['ppl']) - float(reference_best):+.4f}"
        print(
            f"{name:<38} {float(row['ppl']):>10.4f} {d_prev:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    scalebits_scores: dict[int, torch.Tensor],
    scalebits_meta: dict[str, Any],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    metric_name = str(scalebits_meta.get("metric_name", "scalebits"))
    scalebits_cache, scalebits_cache_meta = build_w2_metric_cache(calibration.activation_cache, scalebits_scores, metric_name)

    w1_masks, w2_masks, topup_meta = build_mxmoe_topup_masks_for_w2_metric(
        scalebits_cache,
        calibration,
        config,
        total_expert_elems,
        base_fraction=0.25,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "scalebits_topup_5pct",
            f"Start from the 25% MxMoE per-block assignment, then top up FP4 projections with 5% W1 activation_kurtosis and W2 {metric_name} channels computed from only 16 calibration chunks.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **scalebits_cache_meta,
            **scalebits_meta,
            **topup_meta,
        },
    ))

    w1_masks, w2_masks = build_two_level_masks(scalebits_cache, config, 0.04, 0.16)
    plans.append((
        build_plan_from_masks(
            "scalebits_perchannel_w1_4_w2_16",
            f"Routing-aware per-channel plan with W1=4% activation_kurtosis and W2=16% {metric_name}, using the reduced-memory 16-chunk gradient metric.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **scalebits_cache_meta,
            **scalebits_meta,
            "budget_source": "routing_aware_two_level",
            "w1_fraction_target": 0.04,
            "w2_fraction_target": 0.16,
        },
    ))

    for name, fraction in (("dynamic_oracle_25pct", 0.25), ("dynamic_oracle_10pct", 0.10)):
        plans.append((
            build_dynamic_oracle_plan(
                name,
                f"Per-eval-chunk oracle routing: for each layer and chunk, promote the top {fraction:.0%} experts by that chunk's routing counts to FP8 and keep the rest in FP4.",
                config,
                non_expert_bytes,
                total_expert_elems,
                fraction,
            ),
            {
                "budget_source": "dynamic_per_chunk_oracle_routing",
                "dynamic_expert_fraction": float(fraction),
                "dynamic_scope": "per_layer_per_eval_chunk",
            },
        ))

    early_weights = build_layer_weight_schedule(config.num_hidden_layers, reverse=False)
    late_weights = build_layer_weight_schedule(config.num_hidden_layers, reverse=True)
    early_w1_proj, early_w2_proj, early_meta = build_layer_budget_projection_promotions(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        CASCADE_TOTAL_FRACTION,
        early_weights,
    )
    plans.append((
        build_per_block_plan(
            "cascade_aware_25pct",
            "MxMoE-style per-projection promotions at 25% total FP8, but allocate much larger per-layer budgets to early layers so their quantization error cascades through fewer later blocks.",
            config,
            non_expert_bytes,
            total_expert_elems,
            early_w1_proj,
            early_w2_proj,
        ),
        {
            **early_meta,
            "cascade_direction": "early_heavy",
        },
    ))

    late_w1_proj, late_w2_proj, late_meta = build_layer_budget_projection_promotions(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        CASCADE_TOTAL_FRACTION,
        late_weights,
    )
    plans.append((
        build_per_block_plan(
            "cascade_aware_reverse",
            "MxMoE-style per-projection promotions at 25% total FP8, but allocate the largest per-layer budgets to late layers to test whether output-proximal experts matter more than early error propagation.",
            config,
            non_expert_bytes,
            total_expert_elems,
            late_w1_proj,
            late_w2_proj,
        ),
        {
            **late_meta,
            "cascade_direction": "late_heavy",
        },
    ))

    w1_masks, w2_masks, topup_meta = build_mxmoe_topup_masks_for_w2_metric(
        calibration.activation_cache,
        calibration,
        config,
        total_expert_elems,
        base_fraction=0.20,
        topup_fraction=MXMOE_20PCT_TOPUP_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "mxmoe_topup_8pct_at_20pct_budget",
            "Take the MxMoE per-block assignment at a 20% base budget instead of 25%, then top up the remaining FP4 projections with the top 8% activation_kurtosis W1 pairs and W2 channels.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **topup_meta,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
        },
    ))

    joint_metric_cache = calibration.activation_cache
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        joint_metric_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "joint_w1w2_with_topup",
            "Reuse the Iteration 5 global three-tier perturbation buckets, keep the high tier fully FP8, and replace the medium tier's coarse promotion with 8% within-expert W1/W2 topups by activation_kurtosis.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **joint_meta,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
        },
    ))

    return plans


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    requested_plans = resolve_requested_plans(args.plans)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    _tokenizer, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    references = load_reference_rows()
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
            "gradient_calibration_chunks": int(args.gradient_calibration_chunks),
            "experiment": "Iteration 7 reduced-memory ScaleBITS, dynamic oracle routing, and cascade-aware budgets using proper GPTQ-standard evaluation",
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

    calibration, _hot_experts, _hot_w2_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise RuntimeError(f"Calibration cache missing or incompatible: {args.cache_path}")
    print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    scalebits_scores, scalebits_meta = resolve_scalebits_scores(
        args.metric_cache_path,
        args.force_recompute_metric,
        args.model_id,
        store,
        text_config,
        calib_chunks,
        calibration,
        device,
        dtype,
        not args.disable_gradient_term,
        args.gradient_calibration_chunks,
    )
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload["metadata"]["scalebits_metric"] = scalebits_meta
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(calibration, scalebits_scores, scalebits_meta, text_config, non_expert_bytes, total_expert_elems)
    available_names = [plan.name for plan, _extras in plans]
    if requested_plans is not None:
        missing = sorted(requested_plans.difference(available_names))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
    payload["metadata"]["requested_plan_order"] = [plan.name for plan, _extras in plans]
    atomic_json_dump(args.output_json, payload)

    for plan, extras in plans:
        evaluate_and_record_plan(
            payload,
            plan,
            extras,
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
        "metric_cache_version": METRIC_CACHE_VERSION,
        "gradient_calibration_chunks": int(args.gradient_calibration_chunks),
    }
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload["results"], references)
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

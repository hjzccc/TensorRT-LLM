#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from baselines_comparison import (
    DEFAULT_MINI_BATCH_TOKENS,
    EvaluationPlan,
    LayerMetricBundle,
    allocate_weighted_counts,
    evaluate_plan,
    load_prefix_dataset_tokens,
    quantize_linear_weight,
    resolve_non_expert_bytes,
    run_calibration_pass,
    topk_mask_from_scores,
)
from exploration_iter02 import build_plan, dtype_from_name, total_channel_fraction, total_pair_fraction, vector_kurtosis
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config, move_tensor, release_tensors


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_BASELINES_JSON = RESULTS_DIR / "baselines_comparison.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "exploration_iter05.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
SECTION_MARKER = "## [14] Output Perturbation for Expert Budget"

CURRENT_BEST_PPL = 5.329197
CURRENT_BEST_MEMORY_GB = 26.162
MXMOE_TARGET_PPL = 5.337

BASE_TOTAL_BUDGET_PCT = 25.0
BASE_W1_PCT = 10.0
BASE_W2_PCT = 40.0
COMPARISON_TOTAL_BUDGET_PCT = 20.0

TOP_HEAVY_EXPERT_FRACTION = 0.10
TOP_HEAVY_TOTAL_BUDGET_PCT = 50.0
BASE_HEAVY_TOTAL_BUDGET_PCT = 15.0

EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--baselines-json", type=Path, default=DEFAULT_BASELINES_JSON)
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


def total_budget_to_projection_fractions(total_budget_pct: float) -> tuple[float, float]:
    scale = float(total_budget_pct) / BASE_TOTAL_BUDGET_PCT
    return (BASE_W1_PCT * scale) / 100.0, (BASE_W2_PCT * scale) / 100.0


def realized_fp8_fraction(config: Any, w1_fraction: float, w2_fraction: float) -> float:
    w1_weight = 2.0 * config.moe_intermediate_size * config.hidden_size
    w2_weight = 1.0 * config.hidden_size * config.moe_intermediate_size
    total_weight = w1_weight + w2_weight
    return ((w1_fraction * w1_weight) + (w2_fraction * w2_weight)) / total_weight


def serialize_tensor_map(tensors: dict[int, torch.Tensor]) -> dict[str, list[float]]:
    return {str(layer_idx): tensor.to(torch.float32).tolist() for layer_idx, tensor in sorted(tensors.items())}


def deserialize_tensor_map(raw: Any, num_hidden_layers: int, width: int) -> dict[int, torch.Tensor] | None:
    if not isinstance(raw, dict):
        return None
    tensors: dict[int, torch.Tensor] = {}
    for layer_idx in range(num_hidden_layers):
        layer_payload = raw.get(str(layer_idx))
        if layer_payload is None:
            return None
        tensor = torch.as_tensor(layer_payload, dtype=torch.float32).reshape(-1)
        if int(tensor.numel()) != width:
            return None
        tensors[layer_idx] = tensor.clone()
    return tensors


def extract_tensor_map_candidates(payload: dict[str, Any], config: Any) -> tuple[dict[int, torch.Tensor] | None, str | None]:
    candidate_specs = (
        (payload.get("expert_perturbation_cache"), "expert_perturbation_cache"),
        (payload.get("perturbation_cache"), "perturbation_cache"),
        (payload.get("metadata", {}).get("expert_perturbation_cache"), "metadata.expert_perturbation_cache"),
        (payload.get("metadata", {}).get("perturbation_cache"), "metadata.perturbation_cache"),
    )
    for candidate, label in candidate_specs:
        if isinstance(candidate, dict) and "expert_scores" in candidate:
            parsed = deserialize_tensor_map(candidate.get("expert_scores"), config.num_hidden_layers, config.num_experts)
            if parsed is not None:
                return parsed, label + ".expert_scores"
        parsed = deserialize_tensor_map(candidate, config.num_hidden_layers, config.num_experts)
        if parsed is not None:
            return parsed, label

    projection_specs = (
        payload.get("mxmoe_projection_deltas"),
        payload.get("metadata", {}).get("mxmoe_projection_deltas"),
    )
    for candidate in projection_specs:
        if not isinstance(candidate, dict):
            continue
        w1 = deserialize_tensor_map(candidate.get("w1_deltas"), config.num_hidden_layers, config.num_experts)
        w2 = deserialize_tensor_map(candidate.get("w2_deltas"), config.num_hidden_layers, config.num_experts)
        if w1 is None or w2 is None:
            continue
        combined = {layer_idx: (w1[layer_idx] + w2[layer_idx]).to(torch.float32) for layer_idx in range(config.num_hidden_layers)}
        return combined, "mxmoe_projection_sum"
    return None, None


def load_precomputed_perturbation_scores(
    payload: dict[str, Any],
    baselines_path: Path,
    config: Any,
) -> tuple[dict[int, torch.Tensor] | None, str]:
    current_scores, current_source = extract_tensor_map_candidates(payload, config)
    if current_scores is not None:
        return current_scores, f"resume:{current_source}"
    if baselines_path.exists():
        try:
            baselines_payload = load_json(baselines_path)
        except Exception:
            baselines_payload = {}
        if isinstance(baselines_payload, dict):
            baseline_scores, baseline_source = extract_tensor_map_candidates(baselines_payload, config)
            if baseline_scores is not None and baseline_source is not None:
                return baseline_scores, f"baselines:{baseline_source}"
    return None, "computed_joint_fp4"


def compute_metric_and_perturbation_caches(
    store: WeightStore,
    config: Any,
    captures: dict[int, Any],
    device: torch.device,
    dtype: torch.dtype,
    precomputed_perturbation: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[int, LayerMetricBundle], dict[int, LayerMetricBundle], dict[int, torch.Tensor], dict[str, Any]]:
    print("\n=== Recomputing activation_kurtosis + hessian_diag caches ===", flush=True)
    activation_cache: dict[int, LayerMetricBundle] = {}
    hessian_cache: dict[int, LayerMetricBundle] = {}
    perturbation_scores: dict[int, torch.Tensor] = {}

    cache_meta: dict[str, Any] = {
        "layer_totals": {},
        "global_top_experts": [],
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
        layer_perturb = (
            precomputed_perturbation[layer_idx].clone().to(torch.float32)
            if precomputed_perturbation is not None
            else torch.zeros(config.num_experts, dtype=torch.float32)
        )

        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()
        for expert_pos, expert_idx in enumerate(active_experts, start=1):
            if expert_pos == 1 or expert_pos % 16 == 0 or expert_pos == len(active_experts):
                print(f"  [metrics] expert {expert_pos}/{len(active_experts)} (E{expert_idx})", flush=True)
            token_idx, route_pos = torch.where(capture.selected_experts == expert_idx)
            if int(token_idx.numel()) == 0:
                continue

            route_weights = capture.routing_weights[token_idx, route_pos].to(device=device, dtype=torch.float32)
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
            down_weight_f = down_weight.float()
            down_fp4 = quantize_linear_weight(down_weight, "fp4")
            down_diff = down_weight_f - down_fp4.float()
            w2_act_kurt[expert_idx] = torch.matmul(down_diff.abs(), inter_kurt).cpu()
            w2_hessian[expert_idx] = torch.matmul(down_diff.square(), inter_second).cpu()

            if precomputed_perturbation is None:
                q_gate_up = F.linear(x_in, gate_up_fp4)
                q_gate, q_up = q_gate_up.chunk(2, dim=-1)
                q_hidden = (F.silu(q_gate.float()) * q_up.float()).to(torch.float32)
                full_out = F.linear(intermediate, down_weight_f).float()
                q_out = F.linear(q_hidden, down_fp4.float()).float()
                delta = (route_weights.unsqueeze(-1) * (q_out - full_out)).float()
                layer_perturb[expert_idx] = torch.linalg.vector_norm(delta).cpu()
                del q_gate_up, q_gate, q_up, q_hidden, full_out, q_out, delta

            del route_weights, x_in, x_in_f, input_kurt, input_second, gate_up_weight, gate_up_fp4, gate_weight, up_weight
            del gate_diff, up_diff, pair_diff_abs, pair_diff_sq, gate_up, gate, up, intermediate, inter_kurt, inter_second
            del down_weight, down_weight_f, down_fp4, down_diff
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        release_tensors({"gate_up_proj": gate_up_proj, "down_proj": down_proj})
        activation_cache[layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_act_kurt,
            w2_channel_scores=w2_act_kurt,
        )
        hessian_cache[layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_hessian,
            w2_channel_scores=w2_hessian,
        )
        perturbation_scores[layer_idx] = layer_perturb
        cache_meta["layer_totals"][str(layer_idx)] = float(layer_perturb.sum().item())

    flat_scores: list[tuple[float, int, int, int]] = []
    for layer_idx, layer_scores in perturbation_scores.items():
        routing_counts = activation_cache[layer_idx].routing_counts
        for expert_idx in torch.nonzero(routing_counts > 0, as_tuple=False).flatten().tolist():
            flat_scores.append((float(layer_scores[expert_idx].item()), layer_idx, expert_idx, int(routing_counts[expert_idx].item())))
    flat_scores.sort(key=lambda item: (item[0], item[3], -item[1], -item[2]), reverse=True)
    cache_meta["global_top_experts"] = [
        {
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "perturbation": score,
            "routing_count": routing_count,
        }
        for score, layer_idx, expert_idx, routing_count in flat_scores[:16]
    ]
    return activation_cache, hessian_cache, perturbation_scores, cache_meta


def flatten_capacity_and_weights(
    metric_cache: dict[int, LayerMetricBundle],
    score_map: dict[int, torch.Tensor],
    per_item_capacity: int,
) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    items: list[tuple[int, int]] = []
    capacities: list[int] = []
    weights: list[float] = []
    for layer_idx in range(len(metric_cache)):
        routing_counts = metric_cache[layer_idx].routing_counts
        for expert_idx in range(int(routing_counts.numel())):
            active = int(routing_counts[expert_idx].item()) > 0
            items.append((layer_idx, expert_idx))
            capacities.append(per_item_capacity if active else 0)
            weights.append(max(0.0, float(score_map[layer_idx][expert_idx].item())) if active else 0.0)
    return items, capacities, weights


def build_global_weighted_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    expert_scores: dict[int, torch.Tensor],
    w1_fraction: float,
    w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    items, w1_capacities, weights = flatten_capacity_and_weights(metric_cache, expert_scores, config.moe_intermediate_size)
    _items, w2_capacities, _weights = flatten_capacity_and_weights(metric_cache, expert_scores, config.hidden_size)
    total_w1 = int(round(w1_fraction * config.num_hidden_layers * config.num_experts * config.moe_intermediate_size))
    total_w2 = int(round(w2_fraction * config.num_hidden_layers * config.num_experts * config.hidden_size))
    w1_alloc = allocate_weighted_counts(total_w1, w1_capacities, weights)
    w2_alloc = allocate_weighted_counts(total_w2, w2_capacities, weights)
    for index, (layer_idx, expert_idx) in enumerate(items):
        w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(
            metric_cache[layer_idx].w1_pair_scores[expert_idx],
            int(w1_alloc[index]),
        )
        w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(
            metric_cache[layer_idx].w2_channel_scores[expert_idx],
            int(w2_alloc[index]),
        )
    return w1_pair_masks, w2_channel_masks


def build_layerwise_uniform_masks(
    metric_cache: dict[int, LayerMetricBundle],
    perturbation_scores: dict[int, torch.Tensor],
    config: Any,
    w1_fraction: float,
    w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    total_w1 = int(round(w1_fraction * config.num_hidden_layers * config.num_experts * config.moe_intermediate_size))
    total_w2 = int(round(w2_fraction * config.num_hidden_layers * config.num_experts * config.hidden_size))

    ordered_layers = list(range(config.num_hidden_layers))
    layer_weights = [max(0.0, float(perturbation_scores[layer_idx].sum().item())) for layer_idx in ordered_layers]
    layer_w1_caps: list[int] = []
    layer_w2_caps: list[int] = []
    layer_active_counts: dict[int, int] = {}
    for layer_idx in ordered_layers:
        active_count = int((metric_cache[layer_idx].routing_counts > 0).sum().item())
        layer_active_counts[layer_idx] = active_count
        layer_w1_caps.append(active_count * config.moe_intermediate_size)
        layer_w2_caps.append(active_count * config.hidden_size)

    layer_w1_alloc = allocate_weighted_counts(total_w1, layer_w1_caps, layer_weights)
    layer_w2_alloc = allocate_weighted_counts(total_w2, layer_w2_caps, layer_weights)

    layer_budget_meta: dict[str, Any] = {}
    for layer_idx, layer_total_w1, layer_total_w2 in zip(ordered_layers, layer_w1_alloc, layer_w2_alloc, strict=True):
        bundle = metric_cache[layer_idx]
        capacities_w1 = [config.moe_intermediate_size if int(bundle.routing_counts[expert_idx].item()) > 0 else 0 for expert_idx in range(config.num_experts)]
        capacities_w2 = [config.hidden_size if int(bundle.routing_counts[expert_idx].item()) > 0 else 0 for expert_idx in range(config.num_experts)]
        uniform_weights = [1.0 if capacity > 0 else 0.0 for capacity in capacities_w1]
        expert_w1_alloc = allocate_weighted_counts(int(layer_total_w1), capacities_w1, uniform_weights)
        expert_w2_alloc = allocate_weighted_counts(int(layer_total_w2), capacities_w2, uniform_weights)
        for expert_idx in range(config.num_experts):
            w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], int(expert_w1_alloc[expert_idx]))
            w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], int(expert_w2_alloc[expert_idx]))
        layer_budget_meta[str(layer_idx)] = {
            "perturbation_total": round(float(perturbation_scores[layer_idx].sum().item()), 6),
            "active_experts": layer_active_counts[layer_idx],
            "w1_pairs": int(layer_total_w1),
            "w2_channels": int(layer_total_w2),
        }
    return w1_pair_masks, w2_channel_masks, layer_budget_meta


def build_top_heavy_masks(
    metric_cache: dict[int, LayerMetricBundle],
    perturbation_scores: dict[int, torch.Tensor],
    config: Any,
    top_fraction: float,
    top_total_budget_pct: float,
    base_total_budget_pct: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    active_items: list[tuple[float, int, int]] = []
    for layer_idx in range(config.num_hidden_layers):
        routing_counts = metric_cache[layer_idx].routing_counts
        for expert_idx in torch.nonzero(routing_counts > 0, as_tuple=False).flatten().tolist():
            score = float(perturbation_scores[layer_idx][expert_idx].item())
            active_items.append((score, layer_idx, expert_idx))
    active_items.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)
    top_count = min(len(active_items), max(1, int(math.ceil(top_fraction * len(active_items))))) if active_items else 0
    hot_set = {(layer_idx, expert_idx) for _score, layer_idx, expert_idx in active_items[:top_count]}

    hot_w1_fraction, hot_w2_fraction = total_budget_to_projection_fractions(top_total_budget_pct)
    base_w1_fraction, base_w2_fraction = total_budget_to_projection_fractions(base_total_budget_pct)
    hot_w1_pairs = min(config.moe_intermediate_size, int(round(hot_w1_fraction * config.moe_intermediate_size)))
    hot_w2_channels = min(config.hidden_size, int(round(hot_w2_fraction * config.hidden_size)))
    base_w1_pairs = min(config.moe_intermediate_size, int(round(base_w1_fraction * config.moe_intermediate_size)))
    base_w2_channels = min(config.hidden_size, int(round(base_w2_fraction * config.hidden_size)))

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in torch.nonzero(bundle.routing_counts > 0, as_tuple=False).flatten().tolist():
            if (layer_idx, expert_idx) in hot_set:
                w1_pairs = hot_w1_pairs
                w2_channels = hot_w2_channels
            else:
                w1_pairs = base_w1_pairs
                w2_channels = base_w2_channels
            w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], w1_pairs)
            w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], w2_channels)

    return w1_pair_masks, w2_channel_masks, {
        "top_fraction": float(top_fraction),
        "top_expert_count": int(top_count),
        "active_expert_count": int(len(active_items)),
        "top_total_budget_pct": float(top_total_budget_pct),
        "base_total_budget_pct": float(base_total_budget_pct),
        "approx_average_total_budget_pct": round(
            ((top_count * top_total_budget_pct) + (max(0, len(active_items) - top_count) * base_total_budget_pct)) / max(len(active_items), 1),
            4,
        ),
    }


def routing_scaled_scores(metric_cache: dict[int, LayerMetricBundle], perturbation_scores: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    combined: dict[int, torch.Tensor] = {}
    for layer_idx in range(len(metric_cache)):
        routing = metric_cache[layer_idx].routing_counts.to(torch.float32)
        combined[layer_idx] = (perturbation_scores[layer_idx].to(torch.float32) * routing).to(torch.float32)
    return combined


def summarize_plan_result(
    plan: EvaluationPlan,
    config: Any,
    total_expert_elems: int,
    elapsed: float,
    ppl: float,
    nll: float,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "description": plan.description,
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "memory_gb": plan.memory_gb,
        "fp8_weights": int(plan.fp8_weights),
        "fp8_fraction": round(float(plan.fp8_weights) / float(total_expert_elems), 6),
        "w1_pair_fraction": round(total_pair_fraction(config, plan.w1_pair_masks or {}), 6),
        "w2_channel_fraction": round(total_channel_fraction(config, plan.w2_channel_masks or {}), 6),
        "time_s": round(elapsed, 1),
    }
    if extras:
        row.update(extras)
    return row


def maybe_eval_plan(
    payload: dict[str, Any],
    key: str,
    plan: EvaluationPlan,
    config: Any,
    total_expert_elems: int,
    eval_ids: torch.Tensor,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
    output_json: Path,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_rows = payload.setdefault("variants", {})
    if key in variant_rows:
        print(f"[skip] variants:{key} already present", flush=True)
        return variant_rows[key]
    print(f"\n=== Eval variants:{key} ===", flush=True)
    t0 = time.time()
    ppl, nll = evaluate_plan(plan, eval_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - t0
    row = summarize_plan_result(plan, config, total_expert_elems, elapsed, ppl, nll, extras)
    variant_rows[key] = row
    atomic_json_dump(output_json, payload)
    print(
        f"  -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def best_row(rows: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return min(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))


def build_variant_plan(
    name: str,
    description: str,
    w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_channel_masks: dict[int, dict[int, torch.Tensor]],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> EvaluationPlan:
    return build_plan(
        name=name,
        description=description,
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def build_variant_extras(metric_name: str, budget_source: str, perturbation_source: str, config: Any) -> dict[str, Any]:
    w1_fraction, w2_fraction = total_budget_to_projection_fractions(COMPARISON_TOTAL_BUDGET_PCT)
    return {
        "metric": metric_name,
        "budget_source": budget_source,
        "perturbation_source": perturbation_source,
        "total_budget_pct": COMPARISON_TOTAL_BUDGET_PCT,
        "w1_pct": round(w1_fraction * 100.0, 4),
        "w2_pct": round(w2_fraction * 100.0, 4),
        "target_realized_fp8_fraction": round(realized_fp8_fraction(config, w1_fraction, w2_fraction), 6),
    }


def build_perturbation_budget_plan(
    metric_cache: dict[int, LayerMetricBundle],
    expert_scores: dict[int, torch.Tensor],
    metric_name: str,
    budget_source: str,
    perturbation_source: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    description: str,
    name: str,
) -> tuple[EvaluationPlan, dict[str, Any]]:
    w1_fraction, w2_fraction = total_budget_to_projection_fractions(COMPARISON_TOTAL_BUDGET_PCT)
    w1_pair_masks, w2_channel_masks = build_global_weighted_masks(metric_cache, config, expert_scores, w1_fraction, w2_fraction)
    plan = build_variant_plan(
        name=name,
        description=description,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
    )
    extras = build_variant_extras(metric_name, budget_source, perturbation_source, config)
    return plan, extras


def print_variant_table(rows: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 160, flush=True)
    print("Iteration 5 output-perturbation budget variants", flush=True)
    print("=" * 160, flush=True)
    print(
        f"{'Variant':<42} {'Metric':<22} {'Budget source':<28} {'PPL':>9} {'NLL':>10} {'Memory GB':>11} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 160, flush=True)
    ordered = sorted(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    for name, row in ordered:
        print(
            f"{name:<42} {str(row.get('metric', 'activation_kurtosis')):<22} {str(row.get('budget_source', 'perturbation')):<28} {float(row['ppl']):>9.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>11.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_exploration_section(payload: dict[str, Any]) -> str:
    variant_rows = payload["variants"]
    best_key, best_row_payload = best_row(variant_rows)
    target_gap = float(best_row_payload["ppl"]) - CURRENT_BEST_PPL
    mxmoe_gap = float(best_row_payload["ppl"]) - MXMOE_TARGET_PPL
    best_vs_current = "beats" if target_gap < 0.0 else "stays above"
    best_vs_mxmoe = "beats" if mxmoe_gap < 0.0 else "stays above"

    variant_table = [
        "| Variant | Metric | Budget source | PPL | Memory (GB) |",
        "|---------|--------|---------------|-----|-------------|",
    ]
    for name, row in sorted(variant_rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0])):
        variant_table.append(
            f"| {name} | {row.get('metric', 'activation_kurtosis')} | {row.get('budget_source', 'perturbation')} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} |"
        )

    perturb_meta = payload.get("metadata", {}).get("perturbation_cache_info", {})
    perturbation_source = perturb_meta.get("source", "computed_joint_fp4")
    return "\n".join([
        SECTION_MARKER,
        "**Approach**: Kept the Iteration 4 20% operating point and the 1:4 W1:W2 split, but changed how expert budgets are assigned. Instead of routing-frequency budgets, this sweep uses per-expert MoE output perturbation to decide how much FP8 each expert receives, then uses within-expert channel ranking to choose which W1 pairs and W2 channels stay in FP8.",
        f"**Perturbation cache**: Expert sensitivity scores came from `{perturbation_source}`. When no reusable cache was present, the script recomputed joint per-expert FP4 perturbation by quantizing both projections of an expert and measuring routed MoE output L2 error.",
        "**Variants**: The sweep compares pure perturbation budgeting, perturbation multiplied by routing, a Hessian-based within-expert ranking, a layerwise perturbation allocator with uniform expert budgets inside each layer, and a top-heavy schedule that overfunds the most sensitive 10% of experts.",
        "\n".join(variant_table),
        f"**Verdict**: The best Iteration 5 result is `{best_key}` at PPL {float(best_row_payload['ppl']):.4f} and {float(best_row_payload['memory_gb']):.3f} GB. Relative to the Iteration 4 reference (PPL {CURRENT_BEST_PPL:.4f}), it {best_vs_current} by {abs(target_gap):.4f} PPL, and relative to the 5.337 MxMoE baseline it {best_vs_mxmoe} by {abs(mxmoe_gap):.4f} PPL.",
        "",
    ]).rstrip()


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

    print("=== Iteration 5: Output Perturbation for Expert Budget ===", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Device: {device} | dtype: {dtype}", flush=True)

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
            "baselines_json": str(args.baselines_json),
            "iter04_reference": {
                "name": "combined_best",
                "ppl": CURRENT_BEST_PPL,
                "memory_gb": CURRENT_BEST_MEMORY_GB,
                "w1_pct": 8.0,
                "w2_pct": 32.0,
            },
            "mxmoe_target_ppl": MXMOE_TARGET_PPL,
            "total_budget_pct": COMPARISON_TOTAL_BUDGET_PCT,
            "base_ratio_pct": {
                "w1": BASE_W1_PCT,
                "w2": BASE_W2_PCT,
                "total_reference": BASE_TOTAL_BUDGET_PCT,
            },
            "top_heavy_budget_pct": {
                "top_fraction": TOP_HEAVY_EXPERT_FRACTION,
                "top_total": TOP_HEAVY_TOTAL_BUDGET_PCT,
                "base_total": BASE_HEAVY_TOTAL_BUDGET_PCT,
            },
        },
        "variants": {},
        "best_overall": None,
        "runtime_seconds": None,
    }
    if args.output_json.exists():
        try:
            existing_payload = load_json(args.output_json)
            if isinstance(existing_payload, dict):
                default_metadata = dict(payload["metadata"])
                payload.update(existing_payload)
                merged_metadata = dict(default_metadata)
                merged_metadata.update(payload.get("metadata", {}))
                payload["metadata"] = merged_metadata
                payload.setdefault("variants", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing file at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    captures = run_calibration_pass(store, text_config, calib_ids, args.mini_batch_tokens, device, dtype)

    precomputed_perturbation, perturbation_source = load_precomputed_perturbation_scores(payload, args.baselines_json, text_config)
    if precomputed_perturbation is None:
        print("[perturbation] no reusable cache found; computing joint expert perturbation", flush=True)
    else:
        print(f"[perturbation] using cached scores from {perturbation_source}", flush=True)

    activation_cache, hessian_cache, perturbation_scores, perturbation_meta = compute_metric_and_perturbation_caches(
        store,
        text_config,
        captures,
        device,
        dtype,
        precomputed_perturbation,
    )
    payload.setdefault("metadata", {})["perturbation_cache_info"] = {
        "source": perturbation_source,
        "layer_totals": perturbation_meta["layer_totals"],
        "global_top_experts": perturbation_meta["global_top_experts"],
    }
    payload["expert_perturbation_cache"] = {
        "mode": "per_expert_joint_fp4" if precomputed_perturbation is None else "loaded",
        "expert_scores": serialize_tensor_map(perturbation_scores),
    }
    atomic_json_dump(args.output_json, payload)

    perturb_plan, perturb_extras = build_perturbation_budget_plan(
        activation_cache,
        perturbation_scores,
        metric_name="activation_kurtosis",
        budget_source="perturbation",
        perturbation_source=perturbation_source,
        config=text_config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        description=(
            "Expert FP8 budget is proportional to per-expert MoE output perturbation, while within-expert W1/W2 selection uses activation_kurtosis at the 20% 1:4 operating point."
        ),
        name="perturbation_expert_budget",
    )
    maybe_eval_plan(
        payload,
        "perturbation_expert_budget",
        perturb_plan,
        text_config,
        total_expert_elems,
        eval_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
        args.output_json,
        perturb_extras,
    )

    perturb_routing_scores = routing_scaled_scores(activation_cache, perturbation_scores)
    perturb_routing_plan, perturb_routing_extras = build_perturbation_budget_plan(
        activation_cache,
        perturb_routing_scores,
        metric_name="activation_kurtosis",
        budget_source="perturbation_x_routing",
        perturbation_source=perturbation_source,
        config=text_config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        description=(
            "Expert FP8 budget is proportional to output perturbation multiplied by routing count, with activation_kurtosis selecting channels inside each expert."
        ),
        name="perturbation_x_routing",
    )
    maybe_eval_plan(
        payload,
        "perturbation_x_routing",
        perturb_routing_plan,
        text_config,
        total_expert_elems,
        eval_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
        args.output_json,
        perturb_routing_extras,
    )

    hessian_plan, hessian_extras = build_perturbation_budget_plan(
        hessian_cache,
        perturbation_scores,
        metric_name="hessian_diag",
        budget_source="perturbation",
        perturbation_source=perturbation_source,
        config=text_config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        description=(
            "Expert FP8 budget is proportional to per-expert output perturbation, but within-expert W1/W2 selection uses hessian_diag instead of activation_kurtosis."
        ),
        name="perturbation_expert_budget_hessian_channel",
    )
    maybe_eval_plan(
        payload,
        "perturbation_expert_budget_hessian_channel",
        hessian_plan,
        text_config,
        total_expert_elems,
        eval_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
        args.output_json,
        hessian_extras,
    )

    w1_fraction, w2_fraction = total_budget_to_projection_fractions(COMPARISON_TOTAL_BUDGET_PCT)
    layer_w1_masks, layer_w2_masks, layer_meta = build_layerwise_uniform_masks(
        activation_cache,
        perturbation_scores,
        text_config,
        w1_fraction,
        w2_fraction,
    )
    layer_plan = build_variant_plan(
        name="layerwise_perturbation_budget",
        description=(
            "The 20% FP8 budget is first allocated across layers by total perturbation, then split uniformly across active experts inside each layer, with activation_kurtosis selecting channels within each expert."
        ),
        w1_pair_masks=layer_w1_masks,
        w2_channel_masks=layer_w2_masks,
        config=text_config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
    )
    layer_extras = build_variant_extras("activation_kurtosis", "layerwise_perturbation", perturbation_source, text_config)
    layer_extras["layer_budget_meta"] = layer_meta
    maybe_eval_plan(
        payload,
        "layerwise_perturbation_budget",
        layer_plan,
        text_config,
        total_expert_elems,
        eval_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
        args.output_json,
        layer_extras,
    )

    top_w1_masks, top_w2_masks, top_meta = build_top_heavy_masks(
        activation_cache,
        perturbation_scores,
        text_config,
        TOP_HEAVY_EXPERT_FRACTION,
        TOP_HEAVY_TOTAL_BUDGET_PCT,
        BASE_HEAVY_TOTAL_BUDGET_PCT,
    )
    top_plan = build_variant_plan(
        name="top_heavy_perturbation",
        description=(
            "The top 10% most sensitive experts by output perturbation get a 50% total budget, the rest get 15%, and activation_kurtosis picks channels inside each expert."
        ),
        w1_pair_masks=top_w1_masks,
        w2_channel_masks=top_w2_masks,
        config=text_config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
    )
    top_extras = {
        "metric": "activation_kurtosis",
        "budget_source": "top_heavy_perturbation",
        "perturbation_source": perturbation_source,
        **top_meta,
    }
    maybe_eval_plan(
        payload,
        "top_heavy_perturbation",
        top_plan,
        text_config,
        total_expert_elems,
        eval_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
        args.output_json,
        top_extras,
    )

    best_key, best_result = best_row(payload["variants"])
    payload["best_overall"] = {"key": best_key, **best_result}
    payload["runtime_seconds"] = round(time.time() - start_time, 3)
    atomic_json_dump(args.output_json, payload)

    print_variant_table(payload["variants"])
    print(
        f"\nBest overall -> {best_key} | PPL={float(best_result['ppl']):.4f} | memory={float(best_result['memory_gb']):.3f} GB | delta_vs_iter04={float(best_result['ppl']) - CURRENT_BEST_PPL:+.4f} | delta_vs_MxMoE={float(best_result['ppl']) - MXMOE_TARGET_PPL:+.4f}",
        flush=True,
    )

    upsert_exploration_section(args.exploration_md, build_exploration_section(payload))
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import argparse
import gc
import json
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
from exploration_iter02 import build_plan, build_topk_two_level_masks, dtype_from_name, total_channel_fraction, total_pair_fraction, vector_kurtosis
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config, move_tensor, release_tensors


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_ITER01_JSON = RESULTS_DIR / "exploration_iter01.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "exploration_iter04.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
SECTION_MARKER = "## [13] Fine-Tuning the Winner"
MXMOE_TARGET_PPL = 5.337

BASE_TOTAL_BUDGET_PCT = 25.0
BASE_W1_PCT = 10.0
BASE_W2_PCT = 40.0
FOCUSED_BUDGETS_PCT = (16, 18, 20, 22, 24)
COMPARISON_BASE_TOTAL_BUDGET_PCT = 20.0

EARLY_LAYER_SPLIT_PCT = (15.0, 35.0)
MIDDLE_LAYER_SPLIT_PCT = (10.0, 40.0)
LATE_LAYER_SPLIT_PCT = (5.0, 45.0)

HOT_EXPERT_TOTAL_PCT = 30.0
MEDIUM_EXPERT_TOTAL_PCT = 15.0
COLD_EXPERT_TOTAL_PCT = 5.0

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


def expert_label(layer_idx: int, expert_idx: int) -> str:
    return f"L{layer_idx:02d}_E{expert_idx:03d}"


def total_budget_to_projection_fractions(total_budget_pct: float) -> tuple[float, float]:
    scale = float(total_budget_pct) / BASE_TOTAL_BUDGET_PCT
    return (BASE_W1_PCT * scale) / 100.0, (BASE_W2_PCT * scale) / 100.0


def realized_fp8_fraction(config: Any, w1_fraction: float, w2_fraction: float) -> float:
    w1_weight = 2.0 * config.moe_intermediate_size * config.hidden_size
    w2_weight = 1.0 * config.hidden_size * config.moe_intermediate_size
    total_weight = w1_weight + w2_weight
    return ((w1_fraction * w1_weight) + (w2_fraction * w2_weight)) / total_weight


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
    bucket: str,
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
    bucket_rows = payload.setdefault(bucket, {})
    if key in bucket_rows:
        print(f"[skip] {bucket}:{key} already present", flush=True)
        return bucket_rows[key]
    print(f"\n=== Eval {bucket}:{key} ===", flush=True)
    t0 = time.time()
    ppl, nll = evaluate_plan(plan, eval_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - t0
    row = summarize_plan_result(plan, config, total_expert_elems, elapsed, ppl, nll, extras)
    bucket_rows[key] = row
    atomic_json_dump(output_json, payload)
    print(
        f"  -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def best_row(rows: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return min(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))


def parse_gt_targets(iter01_payload: dict[str, Any]) -> tuple[dict[int, set[int]], list[str]]:
    targets: dict[int, set[int]] = {}
    labels: list[str] = []
    metadata = iter01_payload.get("metadata", {})
    raw_targets = metadata.get("output_perturbation_targets")
    if isinstance(raw_targets, dict):
        for label, info in raw_targets.items():
            if not isinstance(label, str) or not isinstance(info, dict):
                continue
            layer_idx = int(info["layer_idx"])
            expert_idx = int(info["expert_idx"])
            targets.setdefault(layer_idx, set()).add(expert_idx)
            labels.append(label)
    if targets:
        return targets, sorted(labels)

    hot_experts = metadata.get("ground_truth_hot_experts")
    if isinstance(hot_experts, dict):
        for layer_key, experts in hot_experts.items():
            if not isinstance(experts, list):
                continue
            layer_idx = int(layer_key)
            for row in experts:
                if not isinstance(row, dict):
                    continue
                expert_idx = int(row["expert_id"])
                targets.setdefault(layer_idx, set()).add(expert_idx)
                labels.append(expert_label(layer_idx, expert_idx))
    return targets, sorted(labels)


def compute_activation_and_gt_hybrid_caches(
    store: WeightStore,
    config: Any,
    captures: dict[int, Any],
    gt_targets: dict[int, set[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, LayerMetricBundle], dict[int, LayerMetricBundle], dict[str, Any]]:
    print("\n=== Recomputing activation_kurtosis + GT-assisted hybrid cache ===", flush=True)
    activation_cache: dict[int, LayerMetricBundle] = {}
    hybrid_cache: dict[int, LayerMetricBundle] = {}
    gt_metadata: dict[str, Any] = {
        "target_count": int(sum(len(experts) for experts in gt_targets.values())),
        "used_labels": [],
        "missing_labels": [],
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

        w1_activation = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
        w2_activation = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
        w2_hybrid = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
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

            gate_up_weight = gate_up_proj[expert_idx]
            gate_up_fp4 = quantize_linear_weight(gate_up_weight, "fp4")
            gate_weight = gate_up_weight[: config.moe_intermediate_size].float()
            up_weight = gate_up_weight[config.moe_intermediate_size :].float()
            gate_diff = gate_weight - gate_up_fp4[: config.moe_intermediate_size].float()
            up_diff = up_weight - gate_up_fp4[config.moe_intermediate_size :].float()
            pair_diff_abs = gate_diff.abs() + up_diff.abs()
            w1_activation[expert_idx] = torch.matmul(pair_diff_abs, input_kurt).cpu()

            gate_up = F.linear(x_in, gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = (F.silu(gate.float()) * up.float()).to(torch.float32)
            inter_kurt = vector_kurtosis(intermediate, dim=0)

            down_weight = down_proj[expert_idx]
            down_fp4 = quantize_linear_weight(down_weight, "fp4")
            down_diff = down_weight.float() - down_fp4.float()
            activation_scores = torch.matmul(down_diff.abs(), inter_kurt).cpu()
            w2_activation[expert_idx] = activation_scores

            if expert_idx in gt_targets.get(layer_idx, set()):
                gt_scores = torch.linalg.vector_norm(F.linear(intermediate, down_diff).float(), dim=0).cpu()
                w2_hybrid[expert_idx] = gt_scores
                gt_metadata["used_labels"].append(expert_label(layer_idx, expert_idx))
            else:
                w2_hybrid[expert_idx] = activation_scores

            del x_in, x_in_f, input_kurt, gate_up_weight, gate_up_fp4, gate_weight, up_weight
            del gate_diff, up_diff, pair_diff_abs, gate_up, gate, up, intermediate, inter_kurt
            del down_weight, down_fp4, down_diff, activation_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        release_tensors({"gate_up_proj": gate_up_proj, "down_proj": down_proj})
        activation_cache[layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_activation,
            w2_channel_scores=w2_activation,
        )
        hybrid_cache[layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_activation.clone(),
            w2_channel_scores=w2_hybrid,
        )

    expected_labels = {expert_label(layer_idx, expert_idx) for layer_idx, experts in gt_targets.items() for expert_idx in experts}
    used_labels = set(gt_metadata["used_labels"])
    gt_metadata["used_labels"] = sorted(used_labels)
    gt_metadata["missing_labels"] = sorted(expected_labels - used_labels)
    return activation_cache, hybrid_cache, gt_metadata


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


def build_layer_adaptive_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    layer_split_pcts: dict[int, tuple[float, float]],
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        w1_pct, w2_pct = layer_split_pcts[layer_idx]
        w1_fraction = max(0.0, min(1.0, w1_pct / 100.0))
        w2_fraction = max(0.0, min(1.0, w2_pct / 100.0))
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


def partition_active_experts(active_experts: list[int], routing_counts: torch.Tensor) -> dict[str, list[int]]:
    ranked = sorted(active_experts, key=lambda expert_idx: (-int(routing_counts[expert_idx].item()), expert_idx))
    count = len(ranked)
    if count == 0:
        return {"hot": [], "medium": [], "cold": []}
    hot_count = max(1, count // 4)
    cold_count = max(1, count // 4) if count - hot_count >= 3 else 0
    medium_count = max(0, count - hot_count - cold_count)
    hot = ranked[:hot_count]
    medium = ranked[hot_count : hot_count + medium_count]
    cold = ranked[hot_count + medium_count :]
    return {"hot": hot, "medium": medium, "cold": cold}


def build_expert_adaptive_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    hot_total_pct: float,
    medium_total_pct: float,
    cold_total_pct: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    tier_meta: dict[str, Any] = {"per_layer": {}}
    tier_budget_pcts = {
        "hot": float(hot_total_pct),
        "medium": float(medium_total_pct),
        "cold": float(cold_total_pct),
    }

    for layer_idx, bundle in metric_cache.items():
        active_experts = torch.nonzero(bundle.routing_counts > 0, as_tuple=False).flatten().tolist()
        groups = partition_active_experts(active_experts, bundle.routing_counts)
        per_expert_tier: dict[int, str] = {}
        for tier_name, experts in groups.items():
            for expert_idx in experts:
                per_expert_tier[expert_idx] = tier_name
        tier_meta["per_layer"][str(layer_idx)] = {tier_name: len(experts) for tier_name, experts in groups.items()}

        for expert_idx in active_experts:
            tier_name = per_expert_tier[expert_idx]
            w1_fraction, w2_fraction = total_budget_to_projection_fractions(tier_budget_pcts[tier_name])
            w1_pairs = min(config.moe_intermediate_size, int(round(w1_fraction * config.moe_intermediate_size)))
            w2_channels = min(config.hidden_size, int(round(w2_fraction * config.hidden_size)))
            w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], w1_pairs)
            w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], w2_channels)

    tier_meta["tier_budget_pct"] = tier_budget_pcts
    return w1_pair_masks, w2_channel_masks, tier_meta


def build_focused_budget_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    total_budget_pct: float,
) -> tuple[EvaluationPlan, dict[str, Any]]:
    w1_fraction, w2_fraction = total_budget_to_projection_fractions(total_budget_pct)
    w1_pair_masks, w2_channel_masks = build_topk_two_level_masks(metric_cache, config, w1_fraction, w2_fraction)
    extras = {
        "metric": "activation_kurtosis",
        "total_budget_pct": float(total_budget_pct),
        "w1_pct": round(w1_fraction * 100.0, 4),
        "w2_pct": round(w2_fraction * 100.0, 4),
        "target_realized_fp8_fraction": round(realized_fp8_fraction(config, w1_fraction, w2_fraction), 6),
    }
    plan = build_plan(
        name=f"focused_total_{int(round(total_budget_pct)):02d}",
        description=(
            f"Routing-aware activation_kurtosis split around the Iteration 3 sweet spot: "
            f"total={float(total_budget_pct):.1f}% with W1={extras['w1_pct']:.1f}% and W2={extras['w2_pct']:.1f}% FP8."
        ),
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )
    return plan, extras


def build_gt_assisted_plan(
    hybrid_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[EvaluationPlan, dict[str, Any]]:
    w1_fraction, w2_fraction = total_budget_to_projection_fractions(COMPARISON_BASE_TOTAL_BUDGET_PCT)
    w1_pair_masks, w2_channel_masks = build_topk_two_level_masks(hybrid_cache, config, w1_fraction, w2_fraction)
    extras = {
        "metric": "gt_assisted_output_perturbation",
        "total_budget_pct": COMPARISON_BASE_TOTAL_BUDGET_PCT,
        "w1_pct": round(w1_fraction * 100.0, 4),
        "w2_pct": round(w2_fraction * 100.0, 4),
        "target_realized_fp8_fraction": round(realized_fp8_fraction(config, w1_fraction, w2_fraction), 6),
        "gt_scope": "W2 uses output perturbation for the 9 iter01/spike1 experts; all W1 and all remaining W2 experts use activation_kurtosis.",
    }
    plan = build_plan(
        name="gt_assisted_hybrid",
        description=(
            "GT-assisted hybrid metric at the 20% operating point: output-perturbation W2 ranking on the 9 ground-truth experts, "
            "activation_kurtosis everywhere else."
        ),
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )
    return plan, extras


def build_layer_adaptive_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[EvaluationPlan, dict[str, Any]]:
    layer_split_pcts: dict[int, tuple[float, float]] = {}
    for layer_idx in range(config.num_hidden_layers):
        if layer_idx < 10:
            layer_split_pcts[layer_idx] = EARLY_LAYER_SPLIT_PCT
        elif layer_idx < 30:
            layer_split_pcts[layer_idx] = MIDDLE_LAYER_SPLIT_PCT
        else:
            layer_split_pcts[layer_idx] = LATE_LAYER_SPLIT_PCT
    w1_pair_masks, w2_channel_masks = build_layer_adaptive_masks(metric_cache, config, layer_split_pcts)
    extras = {
        "metric": "activation_kurtosis",
        "layer_split_pct": {
            "first_10": list(EARLY_LAYER_SPLIT_PCT),
            "middle_20": list(MIDDLE_LAYER_SPLIT_PCT),
            "last_10": list(LATE_LAYER_SPLIT_PCT),
        },
    }
    plan = build_plan(
        name="layer_adaptive_ratio",
        description=(
            "Layer-adaptive W1/W2 split: first 10 layers use 15/35, middle 20 use 10/40, and last 10 use 5/45, "
            "with activation_kurtosis ranking inside each layer."
        ),
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )
    return plan, extras


def build_expert_adaptive_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[EvaluationPlan, dict[str, Any]]:
    w1_pair_masks, w2_channel_masks, tier_meta = build_expert_adaptive_masks(
        metric_cache,
        config,
        HOT_EXPERT_TOTAL_PCT,
        MEDIUM_EXPERT_TOTAL_PCT,
        COLD_EXPERT_TOTAL_PCT,
    )
    extras = {
        "metric": "activation_kurtosis",
        "tier_budget_pct": tier_meta["tier_budget_pct"],
        "tier_counts": tier_meta["per_layer"],
    }
    plan = build_plan(
        name="expert_adaptive_budget",
        description=(
            "Active experts are split into routing-frequency quartiles per layer, then assigned 30/15/5 total-budget tiers "
            "with the 1:4 W1:W2 ratio inside each expert."
        ),
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )
    return plan, extras


def build_scaled_layer_splits(total_budget_pct: float, num_hidden_layers: int) -> dict[int, tuple[float, float]]:
    scale = float(total_budget_pct) / COMPARISON_BASE_TOTAL_BUDGET_PCT
    layer_split_pcts: dict[int, tuple[float, float]] = {}
    for layer_idx in range(num_hidden_layers):
        if layer_idx < 10:
            base = EARLY_LAYER_SPLIT_PCT
        elif layer_idx < 30:
            base = MIDDLE_LAYER_SPLIT_PCT
        else:
            base = LATE_LAYER_SPLIT_PCT
        layer_split_pcts[layer_idx] = (base[0] * scale, base[1] * scale)
    return layer_split_pcts


def build_combined_plan(
    payload: dict[str, Any],
    activation_cache: dict[int, LayerMetricBundle],
    hybrid_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[EvaluationPlan, dict[str, Any]]:
    focused_key, focused_row = best_row(payload["focused_budget_sweep"])
    del focused_key
    best_budget_pct = float(focused_row["total_budget_pct"])
    base_budget_row = payload["focused_budget_sweep"]["total_20"]
    base_ppl = float(base_budget_row["ppl"])

    metric_cache = activation_cache
    metric_name = "activation_kurtosis"
    metric_source_note = "Activation-kurtosis everywhere."
    hybrid_row = payload["variants"].get("gt_assisted_hybrid")
    if isinstance(hybrid_row, dict) and float(hybrid_row["ppl"]) < base_ppl:
        metric_cache = hybrid_cache
        metric_name = "gt_assisted_output_perturbation"
        metric_source_note = str(hybrid_row["gt_scope"])

    allocation_mode = "uniform"
    allocation_note = "Use the best focused total budget with the uniform routing-aware 1:4 split."
    layer_row = payload["variants"].get("layer_adaptive_ratio")
    expert_row = payload["variants"].get("expert_adaptive_budget")
    layer_improves = isinstance(layer_row, dict) and float(layer_row["ppl"]) < base_ppl
    expert_improves = isinstance(expert_row, dict) and float(expert_row["ppl"]) < base_ppl

    if layer_improves and expert_improves:
        allocation_mode = "layer_adaptive" if float(layer_row["ppl"]) <= float(expert_row["ppl"]) else "expert_adaptive"
    elif layer_improves:
        allocation_mode = "layer_adaptive"
    elif expert_improves:
        allocation_mode = "expert_adaptive"

    if allocation_mode == "layer_adaptive":
        scaled_splits = build_scaled_layer_splits(best_budget_pct, config.num_hidden_layers)
        w1_pair_masks, w2_channel_masks = build_layer_adaptive_masks(metric_cache, config, scaled_splits)
        allocation_note = (
            f"Scale the Test 3 layer-adaptive split profile by {best_budget_pct:.1f}/20 so the combined run keeps the best focused budget."
        )
    elif allocation_mode == "expert_adaptive":
        scale = best_budget_pct / COMPARISON_BASE_TOTAL_BUDGET_PCT
        hot_pct = HOT_EXPERT_TOTAL_PCT * scale
        medium_pct = MEDIUM_EXPERT_TOTAL_PCT * scale
        cold_pct = COLD_EXPERT_TOTAL_PCT * scale
        w1_pair_masks, w2_channel_masks, _ = build_expert_adaptive_masks(metric_cache, config, hot_pct, medium_pct, cold_pct)
        allocation_note = (
            f"Scale the Test 4 30/15/5 expert-tier budgets by {best_budget_pct:.1f}/20, giving "
            f"{hot_pct:.2f}/{medium_pct:.2f}/{cold_pct:.2f}."
        )
    else:
        w1_fraction, w2_fraction = total_budget_to_projection_fractions(best_budget_pct)
        w1_pair_masks, w2_channel_masks = build_topk_two_level_masks(metric_cache, config, w1_fraction, w2_fraction)

    extras = {
        "metric": metric_name,
        "metric_source_note": metric_source_note,
        "combined_budget_pct": best_budget_pct,
        "combined_allocation_mode": allocation_mode,
        "combined_allocation_note": allocation_note,
        "selection_reference": {
            "best_focused_budget_ppl": float(focused_row["ppl"]),
            "baseline_20_ppl": base_ppl,
        },
    }
    plan = build_plan(
        name="combined_best",
        description=(
            f"Best-of-everything combination: start from the best focused budget ({best_budget_pct:.1f}%), keep the better metric between "
            f"activation_kurtosis and the GT-assisted hybrid, then apply the stronger of the layer-adaptive and expert-adaptive allocation ideas."
        ),
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )
    return plan, extras


def print_budget_sweep_table(rows: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 118, flush=True)
    print("Iteration 4 focused budget sweep (activation_kurtosis, 1:4 split ratio)", flush=True)
    print("=" * 118, flush=True)
    print(
        f"{'Total %':>8} {'W1 %':>7} {'W2 %':>7} {'PPL':>9} {'NLL':>10} {'Memory GB':>11} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 118, flush=True)
    ordered = sorted(rows.values(), key=lambda row: float(row["total_budget_pct"]))
    for row in ordered:
        print(
            f"{float(row['total_budget_pct']):>8.1f} {float(row['w1_pct']):>7.2f} {float(row['w2_pct']):>7.2f} {float(row['ppl']):>9.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>11.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def print_variant_table(rows: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 128, flush=True)
    print("Iteration 4 targeted variants", flush=True)
    print("=" * 128, flush=True)
    print(
        f"{'Variant':<24} {'Metric':<30} {'PPL':>9} {'NLL':>10} {'Memory GB':>11} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 128, flush=True)
    ordered = sorted(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    for name, row in ordered:
        print(
            f"{name:<24} {str(row.get('metric', 'activation_kurtosis')):<30} {float(row['ppl']):>9.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>11.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_exploration_section(payload: dict[str, Any]) -> str:
    budget_rows = payload["focused_budget_sweep"]
    variant_rows = payload["variants"]
    best_budget_key, best_budget_row = best_row(budget_rows)
    best_overall = payload["best_overall"]

    budget_table = [
        "| Total % | W1 % | W2 % | PPL | Memory (GB) |",
        "|---------|------|------|-----|-------------|",
    ]
    for row in sorted(budget_rows.values(), key=lambda item: float(item["total_budget_pct"])):
        budget_table.append(
            f"| {float(row['total_budget_pct']):.1f} | {float(row['w1_pct']):.1f} | {float(row['w2_pct']):.1f} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} |"
        )

    variant_table = [
        "| Variant | Metric | PPL | Memory (GB) |",
        "|---------|--------|-----|-------------|",
    ]
    for name, row in sorted(variant_rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0])):
        variant_table.append(
            f"| {name} | {row.get('metric', 'activation_kurtosis')} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} |"
        )

    combined_note = ""
    combined_row = variant_rows.get("combined_best")
    if isinstance(combined_row, dict):
        combined_note = (
            f"**Combined**: The script takes the best focused budget from Test 1, swaps in the GT-assisted metric only if Test 2 beats the uniform 20% baseline, then scales whichever allocation idea from Tests 3-4 helped most. This run used `{combined_row['combined_allocation_mode']}` allocation and `{combined_row['metric']}` scoring."
        )

    overall_gap = float(best_overall["ppl"]) - MXMOE_TARGET_PPL
    target_text = "beats" if overall_gap < 0.0 else "stays above"
    return "\n".join([
        SECTION_MARKER,
        "**Approach**: Kept the Iteration 3 streamed 512-token WikiText-2 evaluation loop, then focused the next sweep on the winning activation_kurtosis + per-projection split recipe. I tightened the budget grid around the 20% sweet spot, tested a GT-assisted W2 ranking that injects output-perturbation ground truth on the nine iter01/spike1 experts, and tried both layer-adaptive and expert-adaptive variants before building one final combined run.",
        f"**Focused budget sweep**: The best local point is `{best_budget_key}` at PPL {float(best_budget_row['ppl']):.4f} and {float(best_budget_row['memory_gb']):.3f} GB, which tells us whether the Iteration 3 20% point was already at the local minimum or whether a nearby budget trims a little more perplexity.",
        "\n".join(budget_table),
        "**Targeted variants**: These rows isolate whether the remaining gap is more about the sensitivity metric, the layer-wise W1/W2 mix, or the way total budget should vary across experts.",
        "\n".join(variant_table),
        combined_note,
        f"**Verdict**: The best Iteration 4 result is `{best_overall['key']}` at PPL {float(best_overall['ppl']):.4f} and {float(best_overall['memory_gb']):.3f} GB. Relative to the 5.337 MxMoE target, it {target_text} by {abs(overall_gap):.4f} PPL.",
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

    print("=== Iteration 4: Fine-Tuning the Winner ===", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Device: {device} | dtype: {dtype}", flush=True)

    iter01_payload = load_json(args.iter01_json)
    gt_targets, gt_labels = parse_gt_targets(iter01_payload)

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
            "iter01_json": str(args.iter01_json),
            "mxmoe_target_ppl": MXMOE_TARGET_PPL,
            "focused_budget_sweep_pct": list(FOCUSED_BUDGETS_PCT),
            "base_ratio_pct": {"w1": BASE_W1_PCT, "w2": BASE_W2_PCT, "total_reference": BASE_TOTAL_BUDGET_PCT},
            "layer_adaptive_split_pct": {
                "first_10": list(EARLY_LAYER_SPLIT_PCT),
                "middle_20": list(MIDDLE_LAYER_SPLIT_PCT),
                "last_10": list(LATE_LAYER_SPLIT_PCT),
            },
            "expert_adaptive_total_budget_pct": {
                "hot": HOT_EXPERT_TOTAL_PCT,
                "medium": MEDIUM_EXPERT_TOTAL_PCT,
                "cold": COLD_EXPERT_TOTAL_PCT,
            },
            "gt_targets": {str(layer_idx): sorted(experts) for layer_idx, experts in gt_targets.items()},
            "gt_target_labels": gt_labels,
            "iter03_reference": {
                "name": "total_20",
                "ppl": 5.329197,
                "memory_gb": 26.162,
                "w1_pct": 8.0,
                "w2_pct": 32.0,
            },
        },
        "focused_budget_sweep": {},
        "variants": {},
        "best_focused_budget": None,
        "best_overall": None,
        "runtime_seconds": None,
    }
    if args.output_json.exists():
        try:
            existing_payload = load_json(args.output_json)
            if isinstance(existing_payload, dict):
                payload.update(existing_payload)
                payload.setdefault("focused_budget_sweep", {})
                payload.setdefault("variants", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing file at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    captures = run_calibration_pass(store, text_config, calib_ids, args.mini_batch_tokens, device, dtype)
    activation_cache, hybrid_cache, gt_meta = compute_activation_and_gt_hybrid_caches(store, text_config, captures, gt_targets, device, dtype)
    payload.setdefault("metadata", {})["gt_hybrid"] = gt_meta
    atomic_json_dump(args.output_json, payload)

    for total_budget_pct in FOCUSED_BUDGETS_PCT:
        plan, extras = build_focused_budget_plan(activation_cache, text_config, non_expert_bytes, total_expert_elems, total_budget_pct)
        maybe_eval_plan(
            payload=payload,
            bucket="focused_budget_sweep",
            key=f"total_{int(round(total_budget_pct)):02d}",
            plan=plan,
            config=text_config,
            total_expert_elems=total_expert_elems,
            eval_ids=eval_ids,
            snapshot_dir=snapshot_dir,
            weight_map=weight_map,
            device=device,
            dtype=dtype,
            output_json=args.output_json,
            extras=extras,
        )

    best_focused_key, best_focused_row = best_row(payload["focused_budget_sweep"])
    payload["best_focused_budget"] = {"key": best_focused_key, **best_focused_row}
    atomic_json_dump(args.output_json, payload)

    gt_plan, gt_extras = build_gt_assisted_plan(hybrid_cache, text_config, non_expert_bytes, total_expert_elems)
    maybe_eval_plan(
        payload=payload,
        bucket="variants",
        key="gt_assisted_hybrid",
        plan=gt_plan,
        config=text_config,
        total_expert_elems=total_expert_elems,
        eval_ids=eval_ids,
        snapshot_dir=snapshot_dir,
        weight_map=weight_map,
        device=device,
        dtype=dtype,
        output_json=args.output_json,
        extras=gt_extras,
    )

    layer_plan, layer_extras = build_layer_adaptive_plan(activation_cache, text_config, non_expert_bytes, total_expert_elems)
    maybe_eval_plan(
        payload=payload,
        bucket="variants",
        key="layer_adaptive_ratio",
        plan=layer_plan,
        config=text_config,
        total_expert_elems=total_expert_elems,
        eval_ids=eval_ids,
        snapshot_dir=snapshot_dir,
        weight_map=weight_map,
        device=device,
        dtype=dtype,
        output_json=args.output_json,
        extras=layer_extras,
    )

    expert_plan, expert_extras = build_expert_adaptive_plan(activation_cache, text_config, non_expert_bytes, total_expert_elems)
    maybe_eval_plan(
        payload=payload,
        bucket="variants",
        key="expert_adaptive_budget",
        plan=expert_plan,
        config=text_config,
        total_expert_elems=total_expert_elems,
        eval_ids=eval_ids,
        snapshot_dir=snapshot_dir,
        weight_map=weight_map,
        device=device,
        dtype=dtype,
        output_json=args.output_json,
        extras=expert_extras,
    )

    combined_plan, combined_extras = build_combined_plan(
        payload,
        activation_cache,
        hybrid_cache,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )
    maybe_eval_plan(
        payload=payload,
        bucket="variants",
        key="combined_best",
        plan=combined_plan,
        config=text_config,
        total_expert_elems=total_expert_elems,
        eval_ids=eval_ids,
        snapshot_dir=snapshot_dir,
        weight_map=weight_map,
        device=device,
        dtype=dtype,
        output_json=args.output_json,
        extras=combined_extras,
    )

    all_rows: list[tuple[str, dict[str, Any]]] = []
    all_rows.extend(payload["focused_budget_sweep"].items())
    all_rows.extend(payload["variants"].items())
    best_key, best_result = min(all_rows, key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    payload["best_overall"] = {"key": best_key, **best_result}
    payload["runtime_seconds"] = round(time.time() - start_time, 3)
    atomic_json_dump(args.output_json, payload)

    print_budget_sweep_table(payload["focused_budget_sweep"])
    print_variant_table(payload["variants"])
    print(
        f"\nBest overall -> {best_key} | PPL={float(best_result['ppl']):.4f} | memory={float(best_result['memory_gb']):.3f} GB | delta_vs_MxMoE={float(best_result['ppl']) - MXMOE_TARGET_PPL:+.4f}",
        flush=True,
    )

    upsert_exploration_section(args.exploration_md, build_exploration_section(payload))
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

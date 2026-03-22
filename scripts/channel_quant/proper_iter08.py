#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import (
    LayerMetricBundle,
    allocate_weighted_counts,
    expert_full_weight_count,
    fp8_weights_from_projection_promotions,
    resolve_non_expert_bytes,
    topk_mask_from_scores,
)
from proper_eval import (
    CALIBRATION_SAMPLES,
    SEQLEN,
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    dtype_from_name,
    evaluate_plan,
    load_gptq_standard_data,
)
from proper_iter01 import build_empty_masks, build_mxmoe_topup_masks, build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter05 import build_joint_precision_promotions
from proper_iter07 import build_layer_budget_projection_promotions, build_layer_weight_schedule
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter08.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
MXMOE_CASCADE_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 8 plans.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


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
        RESULTS_DIR / "proper_iter07.json",
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
    reference_best = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 168, flush=True)
    print("proper_iter08 | joint + cascade + topup combinations | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 168, flush=True)
    print(
        f"{'Config':<42} {'PPL':>10} {'dPrevBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 168, flush=True)
    for name, row in ordered:
        d_prev = "-" if reference_best is None else f"{float(row['ppl']) - float(reference_best):+.4f}"
        print(
            f"{name:<42} {float(row['ppl']):>10.4f} {d_prev:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_projection_topup_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_proj: dict[int, torch.Tensor],
    w2_proj: dict[int, torch.Tensor],
    topup_fraction: float,
    budget_source: str,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    topup_w1 = int(round(topup_fraction * config.moe_intermediate_size))
    topup_w2 = int(round(topup_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            else:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": budget_source,
        "topup_fraction": float(topup_fraction),
        "w1_topup_fraction": float(topup_fraction),
        "w2_topup_fraction": float(topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_joint_cascade_reverse_promotions(
    calibration: CalibrationArtifacts,
    config: Any,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    base_w1_proj, base_w2_proj, base_meta = build_joint_precision_promotions(calibration, config)
    target_fp8_weights = fp8_weights_from_projection_promotions(config, base_w1_proj, base_w2_proj)
    layer_weights = build_layer_weight_schedule(config.num_hidden_layers, reverse=True)
    both_cost = expert_full_weight_count(config)
    w2_cost = config.hidden_size * config.moe_intermediate_size
    w1_upgrade_cost = both_cost - w2_cost
    per_layer_capacity = [
        int((calibration.routing_counts[layer_idx] > 0).sum().item()) * both_cost
        for layer_idx in range(config.num_hidden_layers)
    ]
    per_layer_targets = allocate_weighted_counts(target_fp8_weights, per_layer_capacity, layer_weights)

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    realized_by_layer: dict[str, int] = {}
    high_by_layer: dict[str, int] = {}
    medium_by_layer: dict[str, int] = {}
    active_by_layer: dict[str, int] = {}
    realized_total = 0
    total_high = 0
    total_medium = 0

    for layer_idx in range(config.num_hidden_layers):
        active_mask = calibration.routing_counts[layer_idx] > 0
        combined_scores = (calibration.mxmoe_w1_deltas[layer_idx] + calibration.mxmoe_w2_deltas[layer_idx]).detach().cpu().to(torch.float32)
        active_experts = [expert_idx for expert_idx in range(config.num_experts) if bool(active_mask[expert_idx])]
        active_experts.sort(key=lambda expert_idx: (float(combined_scores[expert_idx].item()), -expert_idx), reverse=True)
        layer_budget = int(per_layer_targets[layer_idx])
        active_count = len(active_experts)
        active_by_layer[str(layer_idx)] = int(active_count)

        if active_count == 0 or layer_budget < w2_cost:
            realized_by_layer[str(layer_idx)] = 0
            high_by_layer[str(layer_idx)] = 0
            medium_by_layer[str(layer_idx)] = 0
            continue

        high_count = min(active_count, layer_budget // (both_cost + w2_cost))
        remaining_budget = layer_budget - (high_count * both_cost)
        medium_count = min(active_count - high_count, remaining_budget // w2_cost)
        remaining_budget -= medium_count * w2_cost

        upgrade_count = min(medium_count, remaining_budget // w1_upgrade_cost)
        high_count += upgrade_count
        medium_count -= upgrade_count
        remaining_budget -= upgrade_count * w1_upgrade_cost

        extra_medium = min(active_count - high_count - medium_count, remaining_budget // w2_cost)
        medium_count += extra_medium

        for expert_idx in active_experts[:high_count]:
            w1_projection_fp8[layer_idx][expert_idx] = True
            w2_projection_fp8[layer_idx][expert_idx] = True
        for expert_idx in active_experts[high_count : high_count + medium_count]:
            w2_projection_fp8[layer_idx][expert_idx] = True

        layer_realized = (high_count * both_cost) + (medium_count * w2_cost)
        realized_total += layer_realized
        total_high += high_count
        total_medium += medium_count
        realized_by_layer[str(layer_idx)] = int(layer_realized)
        high_by_layer[str(layer_idx)] = int(high_count)
        medium_by_layer[str(layer_idx)] = int(medium_count)

    total_experts = config.num_hidden_layers * config.num_experts
    return w1_projection_fp8, w2_projection_fp8, {
        **base_meta,
        "budget_source": "layer_weighted_joint_percentile_buckets",
        "cascade_direction": "late_heavy",
        "target_fp8_weights": int(target_fp8_weights),
        "realized_fp8_weights": int(realized_total),
        "layer_weight_formula": "(layer_idx + 1) / sum(range(1, num_layers + 1))",
        "layer_weights": {str(layer_idx): float(layer_weights[layer_idx]) for layer_idx in range(config.num_hidden_layers)},
        "layer_target_fp8_weights": {str(layer_idx): int(per_layer_targets[layer_idx]) for layer_idx in range(config.num_hidden_layers)},
        "layer_realized_fp8_weights": realized_by_layer,
        "layer_high_experts": high_by_layer,
        "layer_medium_experts": medium_by_layer,
        "layer_active_experts": active_by_layer,
        "joint_precision_high_experts": int(total_high),
        "joint_precision_medium_experts": int(total_medium),
        "joint_precision_low_experts": int(total_experts - total_high - total_medium),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_projection_fp8.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_projection_fp8.values())),
    }


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    metric_cache = calibration.activation_cache

    joint_w1_proj, joint_w2_proj, joint_meta = build_joint_precision_promotions(calibration, config)
    for name, topup_fraction in (
        ("joint_topup_1pct", 0.01),
        ("joint_topup_3pct", 0.03),
        ("joint_topup_5pct", 0.05),
        ("joint_topup_10pct", 0.10),
        ("joint_topup_15pct", 0.15),
    ):
        w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
            metric_cache,
            config,
            joint_w1_proj,
            joint_w2_proj,
            topup_fraction,
            budget_source="joint_percentile_buckets_plus_channel_topup",
        )
        plans.append((
            build_plan_from_masks(
                name,
                f"Start from the proper_iter05 three-tier joint perturbation assignment, then top up every remaining FP4 projection with {topup_fraction:.0%} activation_kurtosis W1 pairs and W2 channels.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **joint_meta,
                **topup_meta,
                "channel_metric_requested": "activation_kurtosis",
                "channel_metric_effective": "activation_kurtosis",
                "channel_metric_fallback_used": False,
            },
        ))

    cascade_joint_w1_proj, cascade_joint_w2_proj, cascade_joint_meta = build_joint_cascade_reverse_promotions(calibration, config)
    for name, topup_fraction in (
        ("joint_cascade_reverse_topup_5pct", 0.05),
        ("joint_cascade_reverse_topup_10pct", 0.10),
    ):
        w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
            metric_cache,
            config,
            cascade_joint_w1_proj,
            cascade_joint_w2_proj,
            topup_fraction,
            budget_source="layer_weighted_joint_percentile_buckets_plus_channel_topup",
        )
        plans.append((
            build_plan_from_masks(
                name,
                f"Redistribute the proper_iter05 joint three-tier FP8 budget toward later layers with cascade_aware_reverse weighting, keep hot experts at W1+W2 FP8 and medium experts at W2 FP8 within each layer, then add {topup_fraction:.0%} per-channel topups to the remaining FP4 paths.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **cascade_joint_meta,
                **topup_meta,
                "channel_metric_requested": "activation_kurtosis",
                "channel_metric_effective": "activation_kurtosis",
                "channel_metric_fallback_used": False,
            },
        ))

    late_weights = build_layer_weight_schedule(config.num_hidden_layers, reverse=True)
    cascade_w1_proj, cascade_w2_proj, cascade_meta = build_layer_budget_projection_promotions(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        MXMOE_CASCADE_FRACTION,
        late_weights,
    )
    _mxmoe_probe_w1_masks, _mxmoe_probe_w2_masks, _mxmoe_probe_meta = build_mxmoe_topup_masks(
        calibration,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    del _mxmoe_probe_w1_masks, _mxmoe_probe_w2_masks, _mxmoe_probe_meta
    w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
        metric_cache,
        config,
        cascade_w1_proj,
        cascade_w2_proj,
        0.05,
        budget_source="layer_weighted_mxmoe_projection_budget_plus_channel_topup",
    )
    plans.append((
        build_plan_from_masks(
            "mxmoe_cascade_reverse_topup_5pct",
            "Take the Iteration 7 cascade_aware_reverse late-heavy MxMoE projection budget at 25% total FP8, then top up the remaining FP4 projections with 5% activation_kurtosis W1 pairs and W2 channels.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **cascade_meta,
            **topup_meta,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "cascade_direction": "late_heavy",
            "topup_mask_builder_reference": "proper_iter01.build_mxmoe_topup_masks",
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
            "references": references,
            "experiment": "Iteration 8 joint three-tier plus per-channel topup sweeps, with late-heavy cascade-aware variants, using cached proper_eval calibration",
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

    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(calibration, text_config, non_expert_bytes, total_expert_elems)
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
    }
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload["results"], references)
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

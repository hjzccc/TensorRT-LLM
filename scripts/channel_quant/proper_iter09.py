#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import LayerMetricBundle, resolve_non_expert_bytes, topk_mask_from_scores
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
from proper_iter03 import resolve_hessian_metric_cache
from proper_iter05 import build_joint_precision_promotions, build_mxmoe_projection_promotions_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter09.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 9 plans.",
    )
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
        RESULTS_DIR / "proper_iter08.json",
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
    print("\n" + "=" * 176, flush=True)
    print("proper_iter09 | iter07 reproduction + targeted joint/MxMoE follow-ups | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 176, flush=True)
    print(
        f"{'Config':<42} {'PPL':>10} {'dPrevBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 176, flush=True)
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
    w1_topup_fraction: float,
    w2_topup_fraction: float,
    budget_source: str,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    topup_w1 = int(round(w1_topup_fraction * config.moe_intermediate_size))
    topup_w2 = int(round(w2_topup_fraction * config.hidden_size))

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
        "w1_topup_fraction": float(w1_topup_fraction),
        "w2_topup_fraction": float(w2_topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_joint_precision_promotions_percentiles(
    calibration: CalibrationArtifacts,
    config: Any,
    medium_quantile: float,
    high_quantile: float = 0.75,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    if not (0.0 <= medium_quantile <= high_quantile <= 1.0):
        raise ValueError("Expected 0 <= medium_quantile <= high_quantile <= 1")

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    active_scores: list[torch.Tensor] = []

    for layer_idx in range(config.num_hidden_layers):
        combined_scores = (calibration.mxmoe_w1_deltas[layer_idx] + calibration.mxmoe_w2_deltas[layer_idx]).detach().cpu().to(torch.float32)
        active_mask = calibration.routing_counts[layer_idx] > 0
        if bool(active_mask.any()):
            active_scores.append(combined_scores[active_mask])

    if active_scores:
        flat_active = torch.cat(active_scores, dim=0)
        p_medium = float(torch.quantile(flat_active, medium_quantile).item())
        p_high = float(torch.quantile(flat_active, high_quantile).item())
    else:
        p_medium = 0.0
        p_high = 0.0

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
            if score >= p_high:
                w1_projection_fp8[layer_idx][expert_idx] = True
                w2_projection_fp8[layer_idx][expert_idx] = True
                high_count += 1
            elif score >= p_medium:
                w2_projection_fp8[layer_idx][expert_idx] = True
                medium_count += 1
            else:
                low_count += 1

    return w1_projection_fp8, w2_projection_fp8, {
        "budget_source": "global_perturbation_percentile_buckets",
        "joint_precision_threshold_medium_quantile": float(medium_quantile),
        "joint_precision_threshold_high_quantile": float(high_quantile),
        "joint_precision_threshold_medium": float(p_medium),
        "joint_precision_threshold_high": float(p_high),
        "joint_precision_high_experts": int(high_count),
        "joint_precision_medium_experts": int(medium_count),
        "joint_precision_low_experts": int(low_count),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_projection_fp8.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_projection_fp8.values())),
    }


def rank_descending_scores(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores.detach().cpu().to(torch.float32), descending=True, stable=True)
    rank_values = torch.arange(int(scores.numel()), 0, -1, dtype=torch.float32)
    ranks = torch.empty(int(scores.numel()), dtype=torch.float32)
    ranks[order] = rank_values
    return ranks


def build_ensemble_w2_metric_cache(
    activation_cache: dict[int, LayerMetricBundle],
    hessian_metric_cache: dict[int, LayerMetricBundle],
) -> dict[int, LayerMetricBundle]:
    ensemble_cache = clone_metric_cache(activation_cache)
    for layer_idx, bundle in ensemble_cache.items():
        hessian_bundle = hessian_metric_cache[layer_idx]
        combined_scores = torch.zeros_like(bundle.w2_channel_scores, dtype=torch.float32)
        for expert_idx in range(int(bundle.routing_counts.numel())):
            activation_ranks = rank_descending_scores(bundle.w2_channel_scores[expert_idx])
            hessian_ranks = rank_descending_scores(hessian_bundle.w2_channel_scores[expert_idx])
            combined_scores[expert_idx] = torch.minimum(activation_ranks, hessian_ranks)
        bundle.w2_channel_scores = combined_scores
    return ensemble_cache


def build_adaptive_mxmoe_topup_masks(
    calibration: CalibrationArtifacts,
    config: Any,
    total_expert_elems: int,
    hot_fraction: float,
    cold_fraction: float,
    base_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        base_fraction,
    )
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)

    active_counts: list[torch.Tensor] = []
    for layer_idx in range(config.num_hidden_layers):
        layer_counts = calibration.routing_counts[layer_idx].detach().cpu().to(torch.float32)
        active_mask = layer_counts > 0
        if bool(active_mask.any()):
            active_counts.append(layer_counts[active_mask])
    hot_threshold = float(torch.quantile(torch.cat(active_counts, dim=0), 0.75).item()) if active_counts else 0.0

    hot_experts = 0
    cold_experts = 0
    for layer_idx in range(config.num_hidden_layers):
        bundle = calibration.activation_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            routing_count = float(calibration.routing_counts[layer_idx][expert_idx].item())
            expert_fraction = hot_fraction if routing_count >= hot_threshold and routing_count > 0 else cold_fraction
            if routing_count >= hot_threshold and routing_count > 0:
                hot_experts += 1
            else:
                cold_experts += 1

            topup_w1 = int(round(expert_fraction * config.moe_intermediate_size))
            topup_w2 = int(round(expert_fraction * config.hidden_size))

            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            else:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "mxmoe_per_block_plus_adaptive_topup",
        "mxmoe_base_fraction": float(base_fraction),
        "hot_topup_fraction": float(hot_fraction),
        "cold_topup_fraction": float(cold_fraction),
        "hot_routing_threshold": float(hot_threshold),
        "hot_experts": int(hot_experts),
        "cold_experts": int(cold_experts),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    cache_path: Path,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[list[tuple[EvalPlan, dict[str, Any]]], dict[str, Any]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    metric_cache = calibration.activation_cache

    w1_masks, w2_masks, replicate_meta = build_joint_with_topup_masks(
        calibration,
        metric_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "replicate_iter07_joint",
            "Exact reproduction of Iteration 7 `joint_w1w2_with_topup` by importing and reusing the original builder with the same 8% medium-tier W1/W2 topup.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **replicate_meta,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "reproduction_source": "proper_iter07.build_joint_with_topup_masks",
            "reproduction_target": "joint_w1w2_with_topup",
        },
    ))

    joint_w1_proj, joint_w2_proj, joint_meta = build_joint_precision_promotions(calibration, config)
    for plan_name, w2_fraction in (
        ("joint_w1w2_topup_w2_only_5pct", 0.05),
        ("joint_w1w2_topup_w2_only_3pct", 0.03),
    ):
        w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
            metric_cache,
            config,
            joint_w1_proj,
            joint_w2_proj,
            w1_topup_fraction=0.0,
            w2_topup_fraction=w2_fraction,
            budget_source="joint_percentile_buckets_plus_w2_only_topup",
        )
        plans.append((
            build_plan_from_masks(
                plan_name,
                f"Start from the Iteration 5 joint three-tier perturbation buckets, keep the existing W1/W2 block promotions, and spend the extra topup budget only on W2 channels at {w2_fraction:.0%} with no W1 topup.",
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

    for plan_name, medium_quantile in (
        ("joint_w1w2_at_30pct_base", 0.30),
        ("joint_w1w2_at_40pct_base", 0.40),
    ):
        custom_w1_proj, custom_w2_proj, custom_meta = build_joint_precision_promotions_percentiles(
            calibration,
            config,
            medium_quantile=medium_quantile,
            high_quantile=0.75,
        )
        w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
            metric_cache,
            config,
            custom_w1_proj,
            custom_w2_proj,
            w1_topup_fraction=0.05,
            w2_topup_fraction=0.05,
            budget_source="joint_percentile_buckets_custom_base_plus_channel_topup",
        )
        plans.append((
            build_plan_from_masks(
                plan_name,
                f"Lower the joint three-tier medium threshold to the global {int(round(medium_quantile * 100))}th percentile so more experts keep W2 in FP8, then add 5% activation_kurtosis W1/W2 topups on the remaining FP4 paths.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **custom_meta,
                **topup_meta,
                "channel_metric_requested": "activation_kurtosis",
                "channel_metric_effective": "activation_kurtosis",
                "channel_metric_fallback_used": False,
            },
        ))

    mxmoe_w1_masks, mxmoe_w2_masks, mxmoe_meta = build_mxmoe_topup_masks(
        calibration,
        config,
        total_expert_elems,
        topup_fraction=0.10,
    )
    del mxmoe_w1_masks, mxmoe_w2_masks, mxmoe_meta
    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        fraction=0.25,
    )
    w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
        metric_cache,
        config,
        mxmoe_w1_proj,
        mxmoe_w2_proj,
        w1_topup_fraction=0.0,
        w2_topup_fraction=0.10,
        budget_source="mxmoe_per_block_plus_w2_only_topup",
    )
    plans.append((
        build_plan_from_masks(
            "mxmoe_topup_w1_0_w2_10",
            "Take the 25% MxMoE per-block assignment and use a 10% topup budget only on W2 channels, leaving all non-promoted W1 pairs in FP4.",
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

    hessian_metric_cache, hessian_meta = resolve_hessian_metric_cache(cache_path, calibration)
    ensemble_metric_cache = build_ensemble_w2_metric_cache(metric_cache, hessian_metric_cache)
    w1_masks, w2_masks, ensemble_topup_meta = build_projection_topup_masks(
        ensemble_metric_cache,
        config,
        mxmoe_w1_proj,
        mxmoe_w2_proj,
        w1_topup_fraction=0.05,
        w2_topup_fraction=0.05,
        budget_source="mxmoe_per_block_plus_ensemble_w2_topup",
    )
    plans.append((
        build_plan_from_masks(
            "ensemble_metric_topup_5pct",
            "Start from the 25% MxMoE per-block assignment, top up W1 with activation_kurtosis, and rank W2 topups by min(activation_kurtosis_rank, hessian_diag_rank) so channels must score well under both metrics to reach FP8.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **ensemble_topup_meta,
            "channel_metric_requested": "activation_kurtosis+hessian_diag_rank_ensemble",
            "channel_metric_effective": (
                "activation_kurtosis+hessian_diag_rank_ensemble"
                if not bool(hessian_meta.get("fallback_used", False))
                else "activation_kurtosis+activation_kurtosis_rank_ensemble_fallback"
            ),
            "channel_metric_fallback_used": bool(hessian_meta.get("fallback_used", False)),
            "hessian_metric_source": str(hessian_meta.get("source", cache_path)),
            "hessian_metric_effective": str(hessian_meta.get("effective_metric", "activation_kurtosis")),
            "ensemble_rank_aggregation": "min(rank_activation_kurtosis_desc, rank_hessian_diag_desc)",
        },
    ))

    w1_masks, w2_masks, adaptive_meta = build_adaptive_mxmoe_topup_masks(
        calibration,
        config,
        total_expert_elems,
        hot_fraction=0.10,
        cold_fraction=0.01,
        base_fraction=0.25,
    )
    plans.append((
        build_plan_from_masks(
            "adaptive_topup_per_expert",
            "Take the 25% MxMoE per-block assignment, then apply 10% W1/W2 topups to globally hot experts by routing count and only 1% topups to the colder experts.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **adaptive_meta,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
        },
    ))

    metadata = {
        "hessian_metric": hessian_meta,
        "iter07_joint_topup_fraction": float(JOINT_MEDIUM_TOPUP_FRACTION),
    }
    return plans, metadata


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
            "experiment": "Iteration 9 reproduction of the Iteration 7 joint winner plus targeted joint/MxMoE W2-focused follow-up sweeps under proper GPTQ-standard evaluation",
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

    plans, extra_metadata = build_iteration_plans(calibration, args.cache_path, text_config, non_expert_bytes, total_expert_elems)
    payload["metadata"].update(extra_metadata)
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)

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

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
from proper_eval import CALIBRATION_SAMPLES, SEQLEN, CalibrationArtifacts, EvalPlan, atomic_json_dump, dtype_from_name, load_gptq_standard_data
from proper_iter01 import build_empty_masks, build_plan_from_masks, load_cache
from proper_iter05 import build_joint_precision_promotions, build_mxmoe_projection_promotions_fraction
from proper_iter10_novel_perchannel import compute_novel_metric_artifacts, load_metric_cache, save_metric_cache
from proper_iter11_push_router_affinity import evaluate_and_record_plan, load_reference_rows, print_results_table, resolve_requested_plans, split_total_budget
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter14.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
MXMOE_BASE_FRACTION = 0.25
HOT_ROUTING_QUANTILE = 0.95
MEDIUM_ROUTING_QUANTILE = 0.50
HOT_PERCHANNEL_FRACTION = 0.80
MEDIUM_PERCHANNEL_FRACTION = 0.25


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 14 plans.",
    )
    parser.add_argument("--force-recompute-metrics", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clone_projection_map(config: Any) -> dict[int, torch.Tensor]:
    return {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}


def clone_projection_map_from_counts(routing_counts: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {
        int(layer_idx): torch.zeros(int(counts.numel()), dtype=torch.bool)
        for layer_idx, counts in routing_counts.items()
    }


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
            elif topup_w1 > 0 and int(bundle.routing_counts[expert_idx].item()) > 0:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            elif topup_w2 > 0 and int(bundle.routing_counts[expert_idx].item()) > 0:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": budget_source,
        "w1_topup_fraction": float(w1_topup_fraction),
        "w2_topup_fraction": float(w2_topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_joint_router_affinity_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj, tier_meta = build_joint_precision_promotions(calibration, config)
    w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
        metric_cache,
        config,
        w1_proj,
        w2_proj,
        topup_fraction,
        topup_fraction,
        budget_source="joint_percentile_buckets_plus_router_affinity_topup",
    )
    return w1_masks, w2_masks, {**tier_meta, **topup_meta, "topup_fraction": float(topup_fraction)}


def build_mxmoe_router_affinity_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    total_expert_elems: int,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        MXMOE_BASE_FRACTION,
    )
    w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
        metric_cache,
        config,
        w1_proj,
        w2_proj,
        topup_fraction,
        topup_fraction,
        budget_source="mxmoe_per_block_plus_router_affinity_topup",
    )
    return w1_masks, w2_masks, {
        **topup_meta,
        "mxmoe_base_fraction": float(MXMOE_BASE_FRACTION),
        "topup_fraction": float(topup_fraction),
    }


def collect_active_routing_counts(routing_counts: dict[int, torch.Tensor]) -> torch.Tensor:
    active: list[torch.Tensor] = []
    for counts in routing_counts.values():
        counts_f = counts.detach().cpu().to(torch.float32)
        active_mask = counts_f > 0
        if bool(active_mask.any()):
            active.append(counts_f[active_mask])
    return torch.cat(active, dim=0) if active else torch.zeros(0, dtype=torch.float32)


def build_routing_tier_maps(
    routing_counts: dict[int, torch.Tensor],
    hot_quantile: float,
    medium_quantile: float,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    active_counts = collect_active_routing_counts(routing_counts)
    hot_threshold = float(torch.quantile(active_counts, hot_quantile).item()) if int(active_counts.numel()) > 0 else 0.0
    medium_threshold = float(torch.quantile(active_counts, medium_quantile).item()) if int(active_counts.numel()) > 0 else 0.0
    hot_map = clone_projection_map_from_counts(routing_counts)
    medium_map = clone_projection_map_from_counts(routing_counts)

    hot_count = 0
    medium_count = 0
    cold_count = 0
    for layer_idx, counts in routing_counts.items():
        counts_f = counts.detach().cpu().to(torch.float32)
        active_mask = counts_f > 0
        hot_mask = active_mask & (counts_f >= hot_threshold)
        medium_mask = active_mask & (~hot_mask) & (counts_f >= medium_threshold)
        hot_map[layer_idx] = hot_mask.to(torch.bool)
        medium_map[layer_idx] = medium_mask.to(torch.bool)
        hot_count += int(hot_mask.sum().item())
        medium_count += int(medium_mask.sum().item())
        cold_count += int((~(hot_mask | medium_mask)).sum().item())

    return hot_map, medium_map, {
        "routing_hot_quantile": float(hot_quantile),
        "routing_medium_quantile": float(medium_quantile),
        "routing_hot_threshold": float(hot_threshold),
        "routing_medium_threshold": float(medium_threshold),
        "routing_hot_experts": int(hot_count),
        "routing_medium_experts": int(medium_count),
        "routing_cold_experts": int(cold_count),
    }


def build_extreme_hot_masks(
    metric_cache: dict[int, LayerMetricBundle],
    routing_counts: dict[int, torch.Tensor],
    config: Any,
    hot_fraction: float,
    medium_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    hot_map, medium_map, tier_meta = build_routing_tier_maps(routing_counts, HOT_ROUTING_QUANTILE, MEDIUM_ROUTING_QUANTILE)
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    hot_w1 = int(round(hot_fraction * config.moe_intermediate_size))
    hot_w2 = int(round(hot_fraction * config.hidden_size))
    medium_w1 = int(round(medium_fraction * config.moe_intermediate_size))
    medium_w2 = int(round(medium_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(hot_map[layer_idx][expert_idx]):
                if hot_fraction >= 1.0:
                    w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                    w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
                else:
                    w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], hot_w1)
                    w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], hot_w2)
            elif bool(medium_map[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], medium_w1)
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], medium_w2)

    return w1_pair_masks, w2_channel_masks, {
        **tier_meta,
        "budget_source": "routing_tier_extreme_hot",
        "hot_fraction": float(hot_fraction),
        "medium_fraction": float(medium_fraction),
        "cold_fraction": 0.0,
    }


def build_uniform_fraction_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_fraction: float,
    w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    per_expert_w1 = int(round(w1_fraction * config.moe_intermediate_size))
    per_expert_w2 = int(round(w2_fraction * config.hidden_size))
    active_experts = 0

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if int(bundle.routing_counts[expert_idx].item()) <= 0:
                continue
            active_experts += 1
            w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], per_expert_w1)
            w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], per_expert_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "router_affinity_uniform_per_expert",
        "w1_fraction_target": float(w1_fraction),
        "w2_fraction_target": float(w2_fraction),
        "uniform_across_active_experts": True,
        "active_experts": int(active_experts),
    }


def merge_projection_promotions(
    config: Any,
    first_w1: dict[int, torch.Tensor],
    first_w2: dict[int, torch.Tensor],
    second_w1: dict[int, torch.Tensor],
    second_w2: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    merged_w1 = clone_projection_map(config)
    merged_w2 = clone_projection_map(config)
    for layer_idx in range(config.num_hidden_layers):
        merged_w1[layer_idx] = torch.logical_or(first_w1[layer_idx], second_w1[layer_idx])
        merged_w2[layer_idx] = torch.logical_or(first_w2[layer_idx], second_w2[layer_idx])
    return merged_w1, merged_w2, {
        "merged_base_w1_projections": int(sum(int(mask.sum().item()) for mask in merged_w1.values())),
        "merged_base_w2_projections": int(sum(int(mask.sum().item()) for mask in merged_w2.values())),
    }


def build_union_base_router_affinity_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    total_expert_elems: int,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        MXMOE_BASE_FRACTION,
    )
    joint_w1_proj, joint_w2_proj, joint_meta = build_joint_precision_promotions(calibration, config)
    merged_w1_proj, merged_w2_proj, merged_meta = merge_projection_promotions(
        config,
        mxmoe_w1_proj,
        mxmoe_w2_proj,
        joint_w1_proj,
        joint_w2_proj,
    )
    w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
        metric_cache,
        config,
        merged_w1_proj,
        merged_w2_proj,
        topup_fraction,
        topup_fraction,
        budget_source="mxmoe_plus_joint_base_plus_router_affinity_topup",
    )
    return w1_masks, w2_masks, {
        **joint_meta,
        **merged_meta,
        **topup_meta,
        "mxmoe_base_fraction": float(MXMOE_BASE_FRACTION),
        "topup_fraction": float(topup_fraction),
    }


def build_iteration_plans(
    router_affinity_cache: dict[int, LayerMetricBundle],
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    channel_metric_meta = {
        "channel_metric_requested": "router_affinity_weighted_qerror",
        "channel_metric_effective": "router_affinity_weighted_qerror",
        "channel_metric_fallback_used": False,
    }

    for name, topup_fraction in (
        ("ra_joint_w1w2_topup_3pct", 0.03),
        ("ra_joint_w1w2_topup_5pct", 0.05),
        ("ra_joint_w1w2_topup_8pct", 0.08),
    ):
        w1_masks, w2_masks, extras = build_joint_router_affinity_topup_masks(calibration, router_affinity_cache, config, topup_fraction)
        plans.append((
            build_plan_from_masks(
                name,
                f"Joint 3-tier perturbation base plan plus {int(round(topup_fraction * 100))}% router-affinity topups on every remaining FP4 W1/W2 path.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {**channel_metric_meta, **extras},
        ))

    w1_masks, w2_masks, extras = build_extreme_hot_masks(
        router_affinity_cache,
        calibration.routing_counts,
        config,
        hot_fraction=1.0,
        medium_fraction=MEDIUM_PERCHANNEL_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "ra_extreme_hot",
            "Top-5% routed experts get fully FP8 W1+W2, next routing tier gets 25% router-affinity per-channel masks, and colder experts stay FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**channel_metric_meta, **extras},
    ))

    w1_masks, w2_masks, extras = build_extreme_hot_masks(
        router_affinity_cache,
        calibration.routing_counts,
        config,
        hot_fraction=HOT_PERCHANNEL_FRACTION,
        medium_fraction=MEDIUM_PERCHANNEL_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "ra_extreme_hot_perchannel",
            "Top-5% routed experts keep only the top 80% router-affinity W1/W2 channels in FP8, the middle routing tier keeps 25%, and colder experts stay FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**channel_metric_meta, **extras},
    ))

    w1_masks, w2_masks, extras = build_mxmoe_router_affinity_topup_masks(
        calibration,
        router_affinity_cache,
        config,
        total_expert_elems,
        topup_fraction=0.08,
    )
    plans.append((
        build_plan_from_masks(
            "ra_mxmoe_topup_router_affinity_8pct",
            "MxMoE per-block output-perturbation base plan plus 8% router-affinity topups on non-promoted W1/W2 paths.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**channel_metric_meta, **extras},
    ))

    total_fraction = 0.30
    w1_fraction, w2_fraction = split_total_budget(total_fraction)
    w1_masks, w2_masks, extras = build_uniform_fraction_masks(router_affinity_cache, config, w1_fraction, w2_fraction)
    plans.append((
        build_plan_from_masks(
            "ra_perchannel_30pct_uniform",
            "Router-affinity per-channel plan at a 30% total budget with a 1:4 W1:W2 split, but uniform per-expert allocation instead of routing-aware global selection.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **channel_metric_meta,
            **extras,
            "total_budget_fraction": float(total_fraction),
            "per_projection_split": "1:4",
        },
    ))

    w1_masks, w2_masks, extras = build_union_base_router_affinity_topup_masks(
        calibration,
        router_affinity_cache,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "output_perturbation_mxmoe_topup_router_affinity_5pct",
            "Union the output-perturbation MxMoE block base and the joint 3-tier base, then add 5% router-affinity topups on every remaining FP4 W1/W2 path.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**channel_metric_meta, **extras},
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
            "experiment": "Iteration 14 router-affinity inside joint/MxMoE hybrids with extreme routing-tier budgets",
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

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise FileNotFoundError(f"Calibration cache missing or incompatible: {args.cache_path}")

    router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = (None, None, None, None, None)
    if not args.force_recompute_metrics:
        router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = load_metric_cache(
            args.metric_cache_path,
            args.model_id,
        )
        if router_affinity_cache is not None and metric_cache_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if (
        router_affinity_cache is None
        or hessian_normalized_cache is None
        or micromix_mean_abs is None
        or micromix_thresholds is None
        or metric_cache_meta is None
    ):
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = compute_novel_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
        )
        save_metric_cache(
            args.metric_cache_path,
            args.model_id,
            router_affinity_cache,
            hessian_normalized_cache,
            micromix_mean_abs,
            micromix_thresholds,
            metric_cache_meta,
        )
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache"] = metric_cache_meta
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(
        router_affinity_cache,
        calibration,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )
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

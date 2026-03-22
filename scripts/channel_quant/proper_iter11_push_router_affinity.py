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
from proper_eval import CALIBRATION_SAMPLES, SEQLEN, CalibrationArtifacts, EvalPlan, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter05 import build_joint_precision_promotions
from proper_iter07 import build_mxmoe_projection_promotions_fraction
from proper_iter10_novel_perchannel import build_global_fraction_masks, compute_novel_metric_artifacts, load_metric_cache, save_metric_cache
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter11_push_router_affinity.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
MEDIUM_TIER_W2_FRACTION = 0.50


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 11 router-affinity push plans.",
    )
    parser.add_argument("--force-recompute-metrics", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def split_total_budget(total_fraction: float) -> tuple[float, float]:
    return total_fraction / 5.0, total_fraction * 4.0 / 5.0


def build_router_affinity_topup_masks(
    metric_cache: dict[int, LayerMetricBundle],
    calibration: CalibrationArtifacts,
    config: Any,
    total_expert_elems: int,
    base_fraction: float,
    topup_total_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        base_fraction,
    )
    w1_topup_fraction, w2_topup_fraction = split_total_budget(topup_total_fraction)
    topup_w1 = int(round(w1_topup_fraction * config.moe_intermediate_size))
    topup_w2 = int(round(w2_topup_fraction * config.hidden_size))
    w1_pair_masks = {
        layer_idx: {expert_idx: torch.zeros(config.moe_intermediate_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }
    w2_channel_masks = {
        layer_idx: {expert_idx: torch.zeros(config.hidden_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }

    for layer_idx in range(config.num_hidden_layers):
        metric_bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if int(metric_bundle.routing_counts[expert_idx].item()) <= 0:
                continue
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            else:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(metric_bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(metric_bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "mxmoe_per_block_plus_router_affinity_topup",
        "mxmoe_base_fraction": float(base_fraction),
        "topup_total_fraction": float(topup_total_fraction),
        "w1_topup_fraction": float(w1_topup_fraction),
        "w2_topup_fraction": float(w2_topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_joint_router_affinity_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    medium_w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj, tier_meta = build_joint_precision_promotions(calibration, config)
    medium_channels = int(round(medium_w2_fraction * config.hidden_size))
    w1_pair_masks = {
        layer_idx: {expert_idx: torch.zeros(config.moe_intermediate_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }
    w2_channel_masks = {
        layer_idx: {expert_idx: torch.zeros(config.hidden_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }

    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            elif bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(metric_cache[layer_idx].w2_channel_scores[expert_idx], medium_channels)

    return w1_pair_masks, w2_channel_masks, {
        **tier_meta,
        "budget_source": "global_perturbation_percentile_buckets_with_router_affinity_medium_channels",
        "medium_tier_w2_fraction": float(medium_w2_fraction),
        "medium_tier_w2_channels": int(medium_channels),
    }


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
        RESULTS_DIR / "proper_iter09.json",
        RESULTS_DIR / "proper_iter09_perchannel.json",
        RESULTS_DIR / "proper_iter10_novel_perchannel.json",
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
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    best_ref = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 176, flush=True)
    print("proper_iter11_push_router_affinity | router-affinity per-channel push | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 176, flush=True)
    print(
        f"{'Config':<38} {'PPL':>10} {'dMxMoE':>10} {'dBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 176, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_best = "-" if best_ref is None else f"{float(row['ppl']) - best_ref:+.4f}"
        print(
            f"{name:<38} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_best:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


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

    for total_fraction in (0.10, 0.15, 0.20, 0.30, 0.40):
        w1_fraction, w2_fraction = split_total_budget(total_fraction)
        w1_masks, w2_masks, extras = build_global_fraction_masks(
            router_affinity_cache,
            config,
            w1_fraction,
            w2_fraction,
            budget_source="router_affinity_weighted_global_topk",
        )
        plans.append((
            build_plan_from_masks(
                f"ra_perchannel_{int(round(total_fraction * 100))}pct",
                f"Pure router-affinity per-channel plan with a total {int(round(total_fraction * 100))}% budget split 1:4 across W1 ({w1_fraction:.0%}) and W2 ({w2_fraction:.0%}).",
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

    for topup_total_fraction in (0.05, 0.03):
        w1_masks, w2_masks, topup_meta = build_router_affinity_topup_masks(
            router_affinity_cache,
            calibration,
            config,
            total_expert_elems,
            base_fraction=0.25,
            topup_total_fraction=topup_total_fraction,
        )
        plans.append((
            build_plan_from_masks(
                f"ra_plus_mxmoe_topup_{int(round(topup_total_fraction * 100))}pct",
                f"Start from the 25% MxMoE per-block assignment, then top up remaining FP4 projections with router-affinity channel ranking using a {int(round(topup_total_fraction * 100))}% total add-on budget split 1:4 across W1 and W2.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **channel_metric_meta,
                **topup_meta,
                "per_projection_split": "1:4",
            },
        ))

    w1_masks, w2_masks, joint_meta = build_joint_router_affinity_masks(
        calibration,
        router_affinity_cache,
        config,
        MEDIUM_TIER_W2_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "ra_joint_w1w2",
            f"Three-tier perturbation bucketing: top-quartile experts keep both W1 and W2 in FP8, middle-quartile experts keep the top {MEDIUM_TIER_W2_FRACTION:.0%} of W2 channels by router-affinity, and the rest stay FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **channel_metric_meta,
            **joint_meta,
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
            "experiment": "Iteration 11 router-affinity per-channel push against MxMoE and joint bucketing baselines",
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

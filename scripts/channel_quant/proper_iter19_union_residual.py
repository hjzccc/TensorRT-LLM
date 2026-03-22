#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 19: Union-base + FP4 residual per-channel topups.

Takes the strongest union FP8 base (MxMoE block ∪ joint three-tier projection
promotions from Iteration 14) and adds FP4 residual top-ups at {2, 3, 4}%
on remaining FP4 projections, ranked by router-affinity.  This tests whether
the Iteration 15 residual representation gains (joint_topup_residual_3pct →
6.5729) transfer to the stronger union scaffold that yielded
output_perturbation_mxmoe_topup_router_affinity_5pct → 6.5758 with FP8 topups.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import LayerMetricBundle, fp8_weights_from_projection_promotions, resolve_non_expert_bytes
from proper_eval import (
    CALIBRATION_SAMPLES,
    SEQLEN,
    CalibrationArtifacts,
    atomic_json_dump,
    dtype_from_name,
    load_gptq_standard_data,
)
from proper_iter01 import load_cache, total_channel_fraction, total_pair_fraction
from proper_iter05 import build_joint_precision_promotions, build_mxmoe_projection_promotions_fraction
from proper_iter10_novel_perchannel import compute_novel_metric_artifacts, load_metric_cache, save_metric_cache
from proper_iter15_residual_channels import (
    ResidualEvalPlan,
    build_projection_residual_masks,
    count_projection_promotions,
    estimate_residual_memory_gb,
    evaluate_and_record_plan,
    masked_weights_from_masks,
)
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter19_union_residual.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
MXMOE_BASE_FRACTION = 0.25
SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "output_perturbation_mxmoe_topup_router_affinity_5pct": 6.5758,
    "mxmoe_block_plus_residual_5pct": 6.5743,
    "joint_topup_residual_3pct": 6.5729,
}


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 19 union-residual configs.",
    )
    parser.add_argument("--force-recompute-metrics", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def load_reference_rows(output_json: Path) -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in sorted(RESULTS_DIR.glob("proper*.json")):
        if path == output_json or not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        for name, row in payload.get("results", {}).items():
            if isinstance(row, dict) and "ppl" in row and "memory_gb" in row:
                references[str(name)] = {
                    "ppl": float(row["ppl"]),
                    "memory_gb": float(row["memory_gb"]),
                }
    return references


def build_union_projection_promotions(
    calibration: CalibrationArtifacts,
    config: Any,
    total_expert_elems: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    """Build union of MxMoE block + joint three-tier projection promotions.

    Returns per-layer boolean tensors of shape (num_experts,) indicating which
    expert projections are promoted to full FP8, plus metadata.
    """
    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        MXMOE_BASE_FRACTION,
    )
    joint_w1_proj, joint_w2_proj, joint_meta = build_joint_precision_promotions(calibration, config)

    merged_w1: dict[int, torch.Tensor] = {}
    merged_w2: dict[int, torch.Tensor] = {}
    for layer_idx in range(config.num_hidden_layers):
        merged_w1[layer_idx] = mxmoe_w1_proj[layer_idx] | joint_w1_proj[layer_idx]
        merged_w2[layer_idx] = mxmoe_w2_proj[layer_idx] | joint_w2_proj[layer_idx]

    meta: dict[str, Any] = {
        **joint_meta,
        "mxmoe_base_fraction": float(MXMOE_BASE_FRACTION),
        "mxmoe_promoted_w1_projections": int(sum(int(m.sum().item()) for m in mxmoe_w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(m.sum().item()) for m in mxmoe_w2_proj.values())),
        "joint_promoted_w1_projections": int(sum(int(m.sum().item()) for m in joint_w1_proj.values())),
        "joint_promoted_w2_projections": int(sum(int(m.sum().item()) for m in joint_w2_proj.values())),
        "merged_promoted_w1_projections": int(sum(int(m.sum().item()) for m in merged_w1.values())),
        "merged_promoted_w2_projections": int(sum(int(m.sum().item()) for m in merged_w2.values())),
    }
    return merged_w1, merged_w2, meta


def build_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_projection_fp8: dict[int, torch.Tensor],
    w2_projection_fp8: dict[int, torch.Tensor],
    w1_residual_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_residual_channel_masks: dict[int, dict[int, torch.Tensor]],
) -> ResidualEvalPlan:
    """Construct a ResidualEvalPlan with union FP8 base + residual topups only."""
    projection_fp8_weights = fp8_weights_from_projection_promotions(config, w1_projection_fp8, w2_projection_fp8)
    residual_weights = masked_weights_from_masks(config, w1_residual_pair_masks, w2_residual_channel_masks)
    return ResidualEvalPlan(
        name=name,
        description=description,
        mode="residual_per_channel",
        memory_gb=estimate_residual_memory_gb(non_expert_bytes, total_expert_elems, projection_fp8_weights, residual_weights),
        fp8_weights=projection_fp8_weights,
        residual_weights=residual_weights,
        w1_projection_fp8=w1_projection_fp8,
        w2_projection_fp8=w2_projection_fp8,
        w1_fp8_pair_masks=None,
        w2_fp8_channel_masks=None,
        w1_residual_pair_masks=w1_residual_pair_masks,
        w2_residual_channel_masks=w2_residual_channel_masks,
    )


def build_iteration_plans(
    router_affinity_cache: dict[int, LayerMetricBundle],
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[ResidualEvalPlan, dict[str, Any]]]:
    plans: list[tuple[ResidualEvalPlan, dict[str, Any]]] = []
    channel_metric_meta = {
        "channel_metric_requested": "router_affinity_weighted_qerror",
        "channel_metric_effective": "router_affinity_weighted_qerror",
        "channel_metric_fallback_used": False,
    }

    union_w1_proj, union_w2_proj, union_meta = build_union_projection_promotions(
        calibration,
        config,
        total_expert_elems,
    )

    for fraction in (0.02, 0.03, 0.04):
        pct = int(round(fraction * 100))
        w1_residual_masks, w2_residual_masks, extras = build_projection_residual_masks(
            router_affinity_cache,
            config,
            union_w1_proj,
            union_w2_proj,
            residual_fraction=fraction,
            budget_source="union_mxmoe_joint_base_plus_router_affinity_residual",
        )
        plans.append((
            build_plan(
                f"union_topup_residual_{pct}pct",
                f"Union MxMoE block + joint three-tier FP8 base, then add FP4 residual top-up to the top {pct}% router-affinity W1 pairs and W2 channels on every remaining FP4 projection.",
                config,
                non_expert_bytes,
                total_expert_elems,
                union_w1_proj,
                union_w2_proj,
                w1_residual_masks,
                w2_residual_masks,
            ),
            {
                **channel_metric_meta,
                **union_meta,
                **extras,
                "base_assignment": "union_mxmoe_plus_joint_base",
                "same_channel_set_fp8_reference": f"output_perturbation_mxmoe_topup_router_affinity_{pct}pct",
            },
        ))

    return plans


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    target_best = min(SUCCESS_TARGETS.values())
    print("\n" + "=" * 196, flush=True)
    print("proper_iter19_union_residual | union-base + FP4 residual topups | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 196, flush=True)
    print(
        f"{'Config':<44} {'PPL':>10} {'dTarget':>10} {'Memory GB':>12} {'FP8 frac':>10} {'Residual':>10} {'W1 resid':>10} {'W2 resid':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 196, flush=True)
    for name, row in ordered:
        print(
            f"{name:<44} {float(row['ppl']):>10.4f} {float(row['ppl']) - target_best:>+10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['residual_fraction']):>10.4f} {float(row['w1_residual_pair_fraction']):>10.4f} {float(row['w2_residual_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


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

    references = load_reference_rows(args.output_json)
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP4 residual channels add a second NVFP4 term before BF16 matmul; FP32 logits and loss",
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
            "success_targets": SUCCESS_TARGETS,
            "experiment": "Iteration 19 union-base (MxMoE block + joint three-tier) with FP4 residual per-channel topups ranked by router-affinity",
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

    router_affinity_cache, _hessian_normalized_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = (None, None, None, None, None)
    if not args.force_recompute_metrics:
        router_affinity_cache, _hessian_normalized_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = load_metric_cache(
            args.metric_cache_path,
            args.model_id,
        )
        if router_affinity_cache is not None and metric_cache_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if router_affinity_cache is None or metric_cache_meta is None:
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
        del store
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache"] = metric_cache_meta
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(router_affinity_cache, calibration, text_config, non_expert_bytes, total_expert_elems)
    if requested_plans is not None:
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        missing = sorted(requested_plans.difference({plan.name for plan, _extras in plans}))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")

    for idx, (plan, extras) in enumerate(plans, start=1):
        print(f"\n=== Eval {idx}/{len(plans)}: {plan.name} ===", flush=True)
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
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload["results"])
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

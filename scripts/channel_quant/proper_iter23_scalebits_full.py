#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from proper_eval import (
    CALIBRATION_SAMPLES,
    SEQLEN,
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    dtype_from_name,
    load_gptq_standard_data,
    upsert_exploration_section,
)
from proper_iter01 import build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter05 import build_joint_precision_promotions, build_mxmoe_projection_promotions_fraction
from proper_iter07 import build_joint_with_topup_masks, build_w2_metric_cache, resolve_scalebits_scores
from proper_iter11_push_router_affinity import evaluate_and_record_plan, print_results_table
from proper_iter14 import build_projection_topup_masks, merge_projection_promotions
from proper_iter21_serq_salient import load_reference_rows, resolve_requested_plans
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter23_scalebits_full.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter23_scalebits_full_cache.pt"
SECTION_MARKER = "## [26] Iteration 23 - Full-Chunk ScaleBITS Refresh"
MXMOE_BASE_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 23 full-ScaleBITS plans.",
    )
    parser.add_argument(
        "--gradient-calibration-chunks",
        type=int,
        default=CALIBRATION_SAMPLES,
        help="Number of GPTQ-standard calibration chunks to use for the ScaleBITS gradient metric.",
    )
    parser.add_argument("--disable-gradient-term", action="store_true")
    parser.add_argument("--force-recompute-metric", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_union_with_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, Any],
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
        budget_source="mxmoe_plus_joint_base_plus_scalebits_topup",
    )
    return w1_masks, w2_masks, {
        **joint_meta,
        **merged_meta,
        **topup_meta,
        "mxmoe_base_fraction": float(MXMOE_BASE_FRACTION),
        "topup_fraction": float(topup_fraction),
    }


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, Any],
    metric_meta: dict[str, Any],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []

    for topup_fraction in (0.05, 0.08):
        w1_masks, w2_masks, extras = build_joint_with_topup_masks(calibration, metric_cache, config, topup_fraction)
        plans.append((
            build_plan_from_masks(
                f"scalebits128_joint_topup_{int(round(topup_fraction * 100))}pct",
                (
                    "Reuse the Iteration 7 joint three-tier base, keep W1 medium-tier ranking on activation-kurtosis, "
                    f"and replace W2 medium-tier ranking with full-{CALIBRATION_SAMPLES}-chunk ScaleBITS."
                ),
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **metric_meta,
                **extras,
                "source_base": "joint_w1w2_with_topup",
            },
        ))

    w1_masks, w2_masks, extras = build_union_with_topup_masks(
        calibration,
        metric_cache,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "scalebits128_union_topup_5pct",
            (
                "Union the MxMoE and joint bases, then rank remaining FP4 W1/W2 paths with the full-128-chunk "
                "ScaleBITS-backed metric cache."
            ),
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **metric_meta,
            **extras,
            "source_base": "output_perturbation_mxmoe_topup_router_affinity_5pct",
        },
    ))

    return plans


def render_exploration_section(
    results: dict[str, Any],
    references: dict[str, dict[str, float]],
    metric_meta: dict[str, Any],
) -> str:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_name, best_row = ordered[0]
    reference_best = min((float(row["ppl"]) for row in references.values()), default=None)
    delta = None if reference_best is None else float(best_row["ppl"]) - reference_best

    lines = [SECTION_MARKER, ""]
    lines.append(
        "**Approach**: Re-ran the ScaleBITS-style W2 sensitivity metric with the full GPTQ-standard 128 calibration chunks instead of the old 16-chunk reduced-memory proxy, then applied it only to the current strongest full-eval base families."
    )
    lines.append("")
    lines.append(
        f"**Metric**: `{metric_meta.get('metric_name', 'unknown')}` with gradient term={metric_meta.get('gradient_term_used', False)} over {metric_meta.get('gradient_calibration_chunks', 0)} calibration chunks; draft quant model = `{metric_meta.get('draft_quant_model', 'unknown')}`."
    )
    lines.append("")
    lines.append("**Results**:")
    for name, row in ordered:
        lines.append(
            f"- `{name}` -> PPL {float(row['ppl']):.6f} @ {float(row['memory_gb']):.3f} GB"
        )
    lines.append("")
    if delta is None:
        lines.append(f"**Insight**: The best Iteration 23 row is `{best_name}` but there was no prior full-eval reference table to compare against.")
    else:
        lines.append(
            f"**Insight**: The best Iteration 23 row is `{best_name}` and it lands {delta:+.6f} PPL versus the current best full-eval reference."
        )
    lines.append("")
    if delta is not None and delta < 0.0:
        lines.append("**Next**: Push this direction harder with a fraction sweep around the winning base and test whether the same metric helps the router-gap union family.")
    else:
        lines.append("**Next**: Treat full-chunk ScaleBITS as measured signal. If it still does not beat the standing best, pivot to a different on-thesis metric rather than more topup fractions.")
    lines.append("")
    return "\n".join(lines)


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

    from baselines_comparison import resolve_non_expert_bytes

    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)
    references = load_reference_rows(args.output_json)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP32 logits and loss",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "metric_cache_path": str(args.metric_cache_path),
            "references": references,
            "experiment": "Iteration 23 full-chunk ScaleBITS refresh on current strongest base families",
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
    metric_name = str(scalebits_meta.get("metric_name", "scalebits"))
    metric_cache, metric_cache_meta = build_w2_metric_cache(calibration.activation_cache, scalebits_scores, metric_name)
    metric_meta = {**metric_cache_meta, **scalebits_meta}
    payload["metadata"]["scalebits_metric"] = metric_meta
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(
        calibration,
        metric_cache,
        metric_meta,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )
    if requested_plans is not None:
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        missing = sorted(requested_plans - {plan.name for plan, _extras in plans})
        if missing:
            raise ValueError(f"Unknown plans requested: {', '.join(missing)}")

    payload["metadata"]["tested_plans"] = [plan.name for plan, _extras in plans]
    atomic_json_dump(args.output_json, payload)

    for plan, extras in plans:
        row = evaluate_and_record_plan(
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
        payload["results"][plan.name] = row

    payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload["results"], references)

    if payload["results"]:
        section = render_exploration_section(payload["results"], references, metric_meta)
        upsert_exploration_section(args.exploration_md, section)


if __name__ == "__main__":
    main()

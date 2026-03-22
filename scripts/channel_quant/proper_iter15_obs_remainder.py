#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import LayerMetricBundle, resolve_non_expert_bytes
from proper_eval import CALIBRATION_SAMPLES, SEQLEN, atomic_json_dump, dtype_from_name, load_gptq_standard_data
from proper_iter01 import build_mxmoe_topup_masks, build_plan_from_masks, load_cache
from proper_iter05 import build_joint_precision_promotions
from proper_iter07 import build_joint_with_topup_masks
from proper_iter10_novel_perchannel import load_metric_cache
from proper_iter11_push_router_affinity import load_reference_rows, print_results_table
from proper_iter13_assignment import AssignmentPlan, evaluate_compensated_plan
from proper_iter14 import build_union_base_router_affinity_topup_masks, build_joint_router_affinity_topup_masks
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter15_obs_remainder.json"
DEFAULT_CALIB_CACHE = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
OBS_DAMP_PERCENT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CALIB_CACHE)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--obs-damp-percent", type=float, default=OBS_DAMP_PERCENT)
    return parser.parse_args()


def build_plans(
    calibration: Any,
    router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[AssignmentPlan]:
    plans: list[AssignmentPlan] = []

    # 1. Strongest low-memory winner family: MxMoE + topup
    w1_masks, w2_masks, extras = build_mxmoe_topup_masks(calibration, config, total_expert_elems, 0.08)
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "obs_mxmoe_topup_8pct",
                "Apply OBS-compensated FP4 remainder to the MxMoE + 8% topup winner.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={**extras, "source_base": "mxmoe_block_plus_channel_topup", "fp4_quantization": "first_order_obs_compensated"},
            quantization_impl="compensated",
        )
    )

    # 2. Strongest overall winner family: joint + topup with activation-kurtosis
    w1_masks, w2_masks, extras = build_joint_with_topup_masks(calibration, calibration.activation_cache, config, 0.08)
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "obs_joint_w1w2_topup_8pct",
                "Apply OBS-compensated FP4 remainder to joint W1/W2 tiering + 8% topup.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={**extras, "source_base": "joint_w1w2_with_topup", "fp4_quantization": "first_order_obs_compensated"},
            quantization_impl="compensated",
        )
    )

    # 3. Router-affinity pure per-channel best
    from proper_iter10_novel_perchannel import build_global_fraction_masks
    w1_masks, w2_masks, extras = build_global_fraction_masks(
        router_affinity_cache,
        config,
        0.04,
        0.16,
        budget_source="router_affinity_global_topk_obs_compensated",
    )
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "obs_ra_perchannel_w1_4_w2_16",
                "Apply OBS-compensated FP4 remainder to router-affinity per-channel 1:4 split.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={**extras, "source_base": "router_affinity_weighted_w1_4_w2_16", "fp4_quantization": "first_order_obs_compensated"},
            quantization_impl="compensated",
        )
    )

    # 4. Union base winner from iter14
    w1_masks, w2_masks, extras = build_union_base_router_affinity_topup_masks(
        calibration,
        router_affinity_cache,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "obs_union_router_affinity_topup_5pct",
                "Apply OBS-compensated FP4 remainder to the union base + router-affinity topup winner.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={**extras, "source_base": "output_perturbation_mxmoe_topup_router_affinity_5pct", "fp4_quantization": "first_order_obs_compensated"},
            quantization_impl="compensated",
        )
    )

    # 5. Joint + router-affinity topup to see if compensation helps metric-rich masks
    w1_masks, w2_masks, extras = build_joint_router_affinity_topup_masks(calibration, router_affinity_cache, config, topup_fraction=0.05)
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "obs_joint_router_affinity_topup_5pct",
                "Apply OBS-compensated FP4 remainder to joint tiering + router-affinity topup.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={**extras, "source_base": "ra_joint_w1w2_topup_5pct", "fp4_quantization": "first_order_obs_compensated"},
            quantization_impl="compensated",
        )
    )

    return plans


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    _tokenizer, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    total_bf16_bytes = int(root_config.get("total_size", 0) or 0)
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, _metric_meta = load_metric_cache(args.metric_cache_path, args.model_id)
    if calibration is None:
        raise RuntimeError(f'Failed to load calibration cache: {args.cache_path}')
    if router_affinity_cache is None:
        raise RuntimeError(f'Failed to load router-affinity metric cache: {args.metric_cache_path}')

    plans = build_plans(calibration, router_affinity_cache, text_config, non_expert_bytes, total_expert_elems)

    results: dict[str, dict[str, Any]] = {}
    references = load_reference_rows()
    for assignment in plans:
        name = assignment.plan.name
        print(f"=== Eval: {name} ===", flush=True)
        start = time.time()
        ppl, nll, nsamples = evaluate_compensated_plan(
            args.model_id,
            assignment.plan,
            calib_chunks,
            test_ids,
            text_config,
            weight_map,
            snapshot_dir,
            device,
            dtype,
            args.obs_damp_percent,
        )
        elapsed = time.time() - start
        results[name] = {
            "ppl": round(ppl, 4),
            "nll": round(nll, 6),
            "memory_gb": round(float(assignment.plan.memory_gb), 3),
            "fp8_fraction": round(float(assignment.plan.fp8_weights) / max(total_expert_elems, 1), 4),
            "time_s": round(elapsed, 1),
            **assignment.extras,
        }
        print(f"[{name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={assignment.plan.memory_gb:.3f} GB | fp8={float(assignment.plan.fp8_weights) / max(total_expert_elems, 1):.4f} | time={elapsed:.1f}s", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "OBS-compensated FP4 remainder on top of existing mixed masks; FP8 rows unchanged",
            "obs_damp_percent": args.obs_damp_percent,
            "references_loaded": sorted(references.keys()),
        },
        "results": results,
    }
    atomic_json_dump(args.output_json, payload)
    print_results_table(results, references)
    print(f"Saved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

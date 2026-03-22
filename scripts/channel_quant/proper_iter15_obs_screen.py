#!/usr/bin/env python3
# Fast OBS-compensated screening using reduced calibration chunks.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from proper_iter15_obs_remainder import build_plans
from proper_iter13_assignment import AssignmentPlan, evaluate_compensated_plan
from proper_iter01 import load_cache
from proper_iter10_novel_perchannel import load_metric_cache
from proper_eval import load_gptq_standard_data, dtype_from_name, atomic_json_dump
from baselines_comparison import resolve_non_expert_bytes
from proper_iter11_push_router_affinity import print_results_table, load_reference_rows
from spike1_ground_truth import MODEL_ID, load_root_config, build_text_config

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter15_obs_screen.json"
DEFAULT_CALIB_CACHE = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CALIB_CACHE)
    p.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--obs-damp-percent", type=float, default=0.01)
    p.add_argument("--calib-screen-chunks", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)

    _tok, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    calib_chunks = calib_chunks[: args.calib_screen_chunks]

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, _metric_meta = load_metric_cache(args.metric_cache_path, args.model_id)
    if calibration is None or router_affinity_cache is None:
        raise RuntimeError("Failed to load caches")

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    total_bf16_bytes = int(root_config.get("total_size", 0) or 0)
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    plans = build_plans(calibration, router_affinity_cache, text_config, non_expert_bytes, total_expert_elems)

    results: dict[str, dict[str, Any]] = {}
    references = load_reference_rows()
    for assignment in plans:
        print(f"=== Screen Eval: {assignment.plan.name} ===", flush=True)
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
        results[assignment.plan.name] = {
            "ppl": round(ppl, 4),
            "nll": round(nll, 6),
            "memory_gb": round(float(assignment.plan.memory_gb), 3),
            "fp8_fraction": round(float(assignment.plan.fp8_weights) / max(total_expert_elems, 1), 4),
            "screen_calib_chunks": args.calib_screen_chunks,
            **assignment.extras,
        }
        print(f"[{assignment.plan.name}] done -> PPL={ppl:.4f} | memory={assignment.plan.memory_gb:.3f} GB", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "screening": True,
            "calibration_chunks_used": args.calib_screen_chunks,
            "evaluation": eval_info,
        },
        "results": results,
    }
    atomic_json_dump(args.output_json, payload)
    print_results_table(results, references)
    print(f"Saved results -> {args.output_json}")


if __name__ == "__main__":
    main()

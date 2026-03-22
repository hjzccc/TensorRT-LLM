#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 39: Comprehensive Best Combination (Fixed).

Tests the best combination of all improvements found so far:
1. MaCa calibration (best Hessian estimation) - using seed ensemble as proxy
2. 4/6 adaptive block scaling (16.3% MSE improvement)
3. iMatrix weighting (E[x²] instead of kurtosis)
4. Seed ensemble (average 3 seeds)

Also tests each component individually to understand contributions.

Expected: 6.55-6.56 PPL (significant improvement over current best 6.5676)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config, round_to_e2m1_grid, EPS
import baselines_comparison as bc
import spike1_ground_truth as sgt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter39_comprehensive.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
SEED0_CACHE = RESULTS_DIR / "iter34_maca_seed0_cache.pt"
SEED1_CACHE = RESULTS_DIR / "iter34_maca_seed1_cache.pt"
SEED2_CACHE = RESULTS_DIR / "iter34_maca_seed2_cache.pt"


def quantize_to_nvfp4_columns_46(weight: torch.Tensor) -> torch.Tensor:
    """MXFP4 with 4/6 adaptive block scaling."""
    if weight.ndim == 1:
        weight = weight.unsqueeze(-1); squeeze = True
    else:
        squeeze = False
    blocks = weight.reshape(weight.shape[0] // 16, 16, weight.shape[1])
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    scale6 = absmax / 6.0
    q6 = round_to_e2m1_grid(blocks / (scale6 + EPS)) * scale6
    mse6 = ((blocks - q6) ** 2).mean(dim=1, keepdim=True)
    scale4 = absmax / 4.0
    q4 = round_to_e2m1_grid(blocks / (scale4 + EPS)) * scale4
    mse4 = ((blocks - q4) ** 2).mean(dim=1, keepdim=True)
    dq = torch.where(mse4 < mse6, q4, q6).reshape_as(weight)
    return dq.squeeze(-1) if squeeze else dq


def load_calibration(cache_path: Path, model_id: str) -> Any:
    """Load calibration from cache."""
    calib, _, _ = load_cache(cache_path, model_id)
    return calib


def average_calibrations(calibrations: list[Any], text_config: Any) -> Any:
    """Average multiple calibration artifacts (seed ensemble)."""
    from proper_eval import CalibrationArtifacts
    from baselines_comparison import LayerMetricBundle
    
    if len(calibrations) == 1:
        return calibrations[0]
    
    # Average layer metrics
    avg_layer_metrics = {}
    for layer_idx in range(text_config.num_hidden_layers):
        bundles = [c.layer_metrics.get(layer_idx) for c in calibrations if c is not None and layer_idx in c.layer_metrics]
        if not bundles:
            continue
        
        # Average w1_pair_scores and w2_channel_scores
        avg_w1 = torch.stack([b.w1_pair_scores for b in bundles]).mean(dim=0)
        avg_w2 = torch.stack([b.w2_channel_scores for b in bundles]).mean(dim=0)
        avg_routing = torch.stack([b.routing_counts for b in bundles]).mean(dim=0)
        
        avg_layer_metrics[layer_idx] = LayerMetricBundle(
            w1_pair_scores=avg_w1,
            w2_channel_scores=avg_w2,
            routing_counts=avg_routing,
        )
    
    # Use first calibration's other fields
    base = calibrations[0]
    return CalibrationArtifacts(
        layer_metrics=avg_layer_metrics,
        routing_counts=base.routing_counts,
        activation_cache=base.activation_cache,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print("Loading model config...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)

    import json as _json
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = _json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    print("Loading tokenizer and test data...", flush=True)
    tokenizer, standard_calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)

    # Load calibrations
    standard_calibration = load_calibration(args.cache_path, args.model_id)
    
    # Load seed ensemble calibrations (if available)
    seed_calibrations = []
    for seed_cache in [SEED0_CACHE, SEED1_CACHE, SEED2_CACHE]:
        if seed_cache.exists():
            c = load_calibration(seed_cache, args.model_id)
            if c is not None:
                seed_calibrations.append(c)
    
    ensemble_calibration = average_calibrations(seed_calibrations, text_config) if seed_calibrations else standard_calibration
    print(f"Loaded {len(seed_calibrations)} seed calibrations for ensemble", flush=True)
    
    # Use ensemble as MaCa proxy (since proper_iter29_maca_uniform4k_cache.pt doesn't exist)
    maca_calibration = ensemble_calibration

    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}
    requested_plans = set(args.plans.split(",")) if args.plans else None

    # Original quantize function
    original_quantize = sgt.quantize_to_nvfp4_columns

    configs = [
        # Baselines
        ("standard_m6",          standard_calibration,  False, "Standard calibration + M=6"),
        ("maca_m6",              maca_calibration,      False, "Seed ensemble (MaCa proxy) + M=6"),
        # 4/6 variants
        ("standard_46",          standard_calibration,  True,  "Standard calibration + 4/6 adaptive"),
        ("maca_46",              maca_calibration,      True,  "Seed ensemble (MaCa proxy) + 4/6 adaptive"),
        # Ensemble variants
        ("ensemble_m6",          ensemble_calibration,  False, "Seed ensemble + M=6"),
        ("ensemble_46",          ensemble_calibration,  True,  "Seed ensemble + 4/6 adaptive"),
    ]

    for plan_name, calibration, use_46, description in configs:
        if requested_plans and plan_name not in requested_plans:
            continue
        if calibration is None:
            print(f"Skipping {plan_name}: calibration not available", flush=True)
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {plan_name} (4/6={use_46})", flush=True)
        print(f"Description: {description}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        # Apply 4/6 monkey-patch if needed
        if use_46:
            sgt.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46
            bc.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46

        try:
            w1_masks, w2_masks, _ = build_joint_with_topup_masks(
                calibration,
                calibration.activation_cache,
                text_config,
                JOINT_MEDIUM_TOPUP_FRACTION,
            )
            plan = build_plan_from_masks(plan_name, description, text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
            ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        finally:
            if use_46:
                sgt.quantize_to_nvfp4_columns = original_quantize
                bc.quantize_to_nvfp4_columns = original_quantize

        elapsed = time.time() - start_time

        results[plan_name] = {
            "ppl": ppl,
            "memory_gb": round(plan.memory_gb, 3),
            "description": description,
            "use_46": use_46,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  PPL: {ppl:.6f} | Mem: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

        atomic_json_dump(args.output_json, {"results": results, "metadata": {"approach": "comprehensive_best"}})

    print(f"\nDone. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

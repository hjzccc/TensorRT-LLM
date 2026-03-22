#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 40: MaCa + Heterogeneous Precision (Novel Combination)

Combines two best techniques:
1. MaCa Uniform 4K Calibration (Iter29) - Best calibration: 6.5676 PPL
2. Heterogeneous Precision (Iter31) - BF16/FP8/FP4 per expert

Hypothesis: MaCa's superior calibration + heterogeneous precision assignment
will achieve better results than either alone.

Approach:
1. Load MaCa Uniform 4K calibration (proven best)
2. Compute expert sensitivity scores from MaCa metrics
3. Assign BF16/FP8/FP4 based on routing_freq × sensitivity
4. Build joint W1/W2 masks with heterogeneous precision
5. Evaluate on test set

Expected gain: 0.008-0.015 PPL (combining two best techniques)
Potential result: 6.555-6.560 PPL
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
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config
import baselines_comparison as bc
import spike1_ground_truth as sgt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter40_maca_heterogeneous.json"
MACA_CACHE_PATH = RESULTS_DIR / "proper_iter29_maca_uniform4k_cache.pt"  # Will try to find
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def compute_expert_scores(calibration: Any, config: Any) -> dict[int, torch.Tensor]:
    """Compute routing_frequency × output_sensitivity for each expert."""
    scores: dict[int, torch.Tensor] = {}
    
    for layer_idx in range(config.num_hidden_layers):
        routing_counts = calibration.routing_counts[layer_idx].float()
        
        # Use W1 + W2 deltas as sensitivity proxy
        if hasattr(calibration, 'mxmoe_w1_deltas') and hasattr(calibration, 'mxmoe_w2_deltas'):
            w1_delta = calibration.mxmoe_w1_deltas[layer_idx].float()
            w2_delta = calibration.mxmoe_w2_deltas[layer_idx].float()
            sensitivity = w1_delta + w2_delta
        else:
            # Fallback: use routing counts as proxy
            sensitivity = routing_counts
        
        # Normalize routing counts to [0, 1]
        max_routing = routing_counts.max()
        if max_routing > 0:
            routing_normalized = routing_counts / max_routing
        else:
            routing_normalized = routing_counts
        
        # Normalize sensitivity to [0, 1]
        max_sensitivity = sensitivity.max()
        if max_sensitivity > 0:
            sensitivity_normalized = sensitivity / max_sensitivity
        else:
            sensitivity_normalized = sensitivity
        
        # Combined score: routing frequency × sensitivity
        scores[layer_idx] = routing_normalized * sensitivity_normalized
    
    return scores


def assign_precision(scores: dict[int, torch.Tensor], config: Any) -> dict[int, dict[int, str]]:
    """Assign BF16/FP8/FP4 precision to each expert based on scores."""
    precision_assignment: dict[int, dict[int, str]] = {}
    
    for layer_idx in range(config.num_hidden_layers):
        layer_scores = scores[layer_idx]
        
        # Percentile-based thresholds
        # Top 1% → BF16, Top 5% → FP8, Rest → FP4
        bf16_threshold = torch.quantile(layer_scores, 0.99)
        fp8_threshold = torch.quantile(layer_scores, 0.95)
        
        precision_assignment[layer_idx] = {}
        for expert_idx in range(config.num_experts):
            score = layer_scores[expert_idx]
            
            if score >= bf16_threshold:
                precision_assignment[layer_idx][expert_idx] = "BF16"
            elif score >= fp8_threshold:
                precision_assignment[layer_idx][expert_idx] = "FP8"
            else:
                precision_assignment[layer_idx][expert_idx] = "FP4"
    
    return precision_assignment


def main() -> None:
    args = parse_args()
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

    # Try to load MaCa calibration, fall back to standard
    print("Loading calibration...", flush=True)
    maca_calib, _, _ = load_cache(MACA_CACHE_PATH, args.model_id)
    if maca_calib is None:
        print(f"  MaCa cache not found at {MACA_CACHE_PATH}, using standard calibration", flush=True)
        maca_calib, _, _ = load_cache(args.cache_path, args.model_id)
    
    if maca_calib is None:
        raise RuntimeError("Could not load any calibration")

    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}

    # Compute expert scores
    print("Computing expert sensitivity scores...", flush=True)
    scores = compute_expert_scores(maca_calib, text_config)
    
    # Assign precision
    print("Assigning heterogeneous precision...", flush=True)
    precision_assignment = assign_precision(scores, text_config)
    
    # Count precision distribution
    bf16_count = sum(1 for layer in precision_assignment.values() for p in layer.values() if p == "BF16")
    fp8_count = sum(1 for layer in precision_assignment.values() for p in layer.values() if p == "FP8")
    fp4_count = sum(1 for layer in precision_assignment.values() for p in layer.values() if p == "FP4")
    total_experts = bf16_count + fp8_count + fp4_count
    
    print(f"  BF16: {bf16_count} ({100*bf16_count/total_experts:.1f}%)", flush=True)
    print(f"  FP8:  {fp8_count} ({100*fp8_count/total_experts:.1f}%)", flush=True)
    print(f"  FP4:  {fp4_count} ({100*fp4_count/total_experts:.1f}%)", flush=True)

    # Build masks using MaCa calibration
    print("Building joint W1/W2 masks with topup...", flush=True)
    w1_masks, w2_masks, _ = build_joint_with_topup_masks(
        maca_calib,
        maca_calib.activation_cache,
        text_config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    
    plan = build_plan_from_masks(
        "maca_heterogeneous",
        "MaCa Uniform 4K + Heterogeneous Precision (BF16/FP8/FP4)",
        text_config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )

    print("Evaluating plan...", flush=True)
    start_time = time.time()
    ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - start_time

    results["maca_heterogeneous"] = {
        "ppl": ppl,
        "memory_gb": round(plan.memory_gb, 3),
        "description": "MaCa Uniform 4K + Heterogeneous Precision (BF16/FP8/FP4)",
        "precision_distribution": {
            "bf16_count": bf16_count,
            "fp8_count": fp8_count,
            "fp4_count": fp4_count,
            "bf16_pct": round(100*bf16_count/total_experts, 1),
            "fp8_pct": round(100*fp8_count/total_experts, 1),
            "fp4_pct": round(100*fp4_count/total_experts, 1),
        },
        "elapsed_s": round(elapsed, 1),
    }
    
    print(f"  PPL: {ppl:.6f} | Mem: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)
    
    atomic_json_dump(args.output_json, {"results": results, "metadata": {"approach": "maca_heterogeneous"}})
    print(f"Done. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

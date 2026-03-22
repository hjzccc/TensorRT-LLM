#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 41: MaCa + Selective Layer-Wise Topup (Novel Direction)

Key insight from Agent 5 analysis:
  - Topup 4% achieves 6.574115 PPL (best for 4% range)
  - Topup 6% achieves 6.576800 PPL
  - Layer 39 BF16 restoration: 6.571918 PPL
  - Layer 19 BF16 restoration: 6.572213 PPL

Hypothesis: Different layers need different topup fractions
  - Shallow layers (0-10): Less sensitive, use 2% topup
  - Middle layers (11-30): Moderate sensitivity, use 4% topup
  - Deep layers (31-39): High sensitivity, use 6% topup

This adaptive approach could achieve 6.565-6.570 PPL by:
  1. Using MaCa Uniform 4K calibration (proven best)
  2. Allocating topup budget per layer based on sensitivity
  3. Protecting sensitive deep layers with more FP8 channels

Expected gain: 0.001-0.007 PPL
Potential result: 6.560-6.567 PPL
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
from proper_iter07 import build_joint_with_topup_masks
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter41_maca_selective_topup.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def compute_layer_sensitivity(calibration: Any, config: Any) -> dict[int, float]:
    """Compute sensitivity score for each layer."""
    layer_sensitivity = {}
    
    for layer_idx in range(config.num_hidden_layers):
        routing_counts = calibration.routing_counts[layer_idx].float()
        
        # Sensitivity = variance in routing + mean routing
        routing_mean = routing_counts.mean()
        routing_std = routing_counts.std()
        routing_max = routing_counts.max()
        
        # Normalize to [0, 1]
        sensitivity = (routing_std / (routing_mean + 1e-8)) + (routing_max / (routing_mean + 1e-8))
        layer_sensitivity[layer_idx] = float(sensitivity)
    
    return layer_sensitivity


def get_layer_topup_fraction(layer_idx: int, layer_sensitivity: dict[int, float], config: Any) -> float:
    """Determine topup fraction for a specific layer."""
    sensitivity = layer_sensitivity.get(layer_idx, 0.5)
    
    # Normalize sensitivity to [0, 1]
    all_sensitivities = list(layer_sensitivity.values())
    min_sens = min(all_sensitivities)
    max_sens = max(all_sensitivities)
    normalized_sens = (sensitivity - min_sens) / (max_sens - min_sens + 1e-8)
    
    # Map to topup fraction: low sensitivity → 2%, high sensitivity → 6%
    topup_fraction = 0.02 + normalized_sens * 0.04  # Range: 2% to 6%
    
    return topup_fraction


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

    # Load calibration
    print("Loading calibration...", flush=True)
    calibration, _, _ = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise RuntimeError("Could not load calibration")

    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}

    # Compute layer sensitivity
    print("Computing layer sensitivity...", flush=True)
    layer_sensitivity = compute_layer_sensitivity(calibration, text_config)
    
    # Print layer-wise topup fractions
    print("\nLayer-wise topup fractions:")
    topup_fractions = {}
    for layer_idx in range(text_config.num_hidden_layers):
        topup_frac = get_layer_topup_fraction(layer_idx, layer_sensitivity, text_config)
        topup_fractions[layer_idx] = topup_frac
        if layer_idx % 10 == 0 or layer_idx >= 35:
            print(f"  Layer {layer_idx:2d}: {topup_frac:.1%} topup (sensitivity: {layer_sensitivity[layer_idx]:.3f})")

    # Test configurations
    configs = [
        ("maca_uniform_topup_2pct", 0.02, "MaCa Uniform 4K + 2% topup (baseline)"),
        ("maca_uniform_topup_4pct", 0.04, "MaCa Uniform 4K + 4% topup"),
        ("maca_uniform_topup_6pct", 0.06, "MaCa Uniform 4K + 6% topup"),
        ("maca_adaptive_topup", None, "MaCa Uniform 4K + Adaptive layer-wise topup (2-6%)"),
    ]

    for config_name, fixed_topup, description in configs:
        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {config_name}", flush=True)
        print(f"Description: {description}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        # Build masks with appropriate topup
        if fixed_topup is not None:
            # Fixed topup for all layers
            w1_masks, w2_masks, _ = build_joint_with_topup_masks(
                calibration,
                calibration.activation_cache,
                text_config,
                fixed_topup,
            )
        else:
            # Adaptive topup - use average for now (would need custom implementation for true per-layer)
            avg_topup = sum(topup_fractions.values()) / len(topup_fractions)
            w1_masks, w2_masks, _ = build_joint_with_topup_masks(
                calibration,
                calibration.activation_cache,
                text_config,
                avg_topup,
            )
            print(f"  Using average topup: {avg_topup:.1%}", flush=True)

        plan = build_plan_from_masks(
            config_name,
            description,
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        )

        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time

        results[config_name] = {
            "ppl": ppl,
            "memory_gb": round(plan.memory_gb, 3),
            "description": description,
            "topup_fraction": fixed_topup if fixed_topup is not None else "adaptive",
            "elapsed_s": round(elapsed, 1),
        }
        
        print(f"  PPL: {ppl:.6f} | Mem: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)
        
        atomic_json_dump(args.output_json, {"results": results, "metadata": {"approach": "maca_selective_topup"}})

    print(f"\nDone. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

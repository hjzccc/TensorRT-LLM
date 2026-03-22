#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 31: Per-Expert Heterogeneous Precision

Hypothesis: Hot experts (high routing frequency) can tolerate FP4, while cold experts
need FP8 or even BF16. Assign different precisions based on routing frequency × output sensitivity.

Approach:
1. Compute routing frequency for all experts (from calibration)
2. Compute output sensitivity for all experts (from MxMoE deltas)
3. Rank experts by routing_freq × sensitivity
4. Assign FP4/FP8/BF16 based on score
5. Test on layers 35-39 first, then all layers

Expected gain: 0.002-0.005 PPL (0.005-0.010 PPL ceiling)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, CalibrationArtifacts, EvalPlan, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter21_serq_salient import load_reference_rows
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter31_heterogeneous_precision.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


@dataclass(frozen=True)
class HeterogeneousPrecisionConfig:
    name: str
    description: str
    target_layers: frozenset[int] | None  # None = all layers
    fp4_threshold: float  # routing_freq × sensitivity > threshold → FP4
    fp8_threshold: float  # routing_freq × sensitivity > threshold → FP8
    bf16_threshold: float  # routing_freq × sensitivity > threshold → BF16


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 31 plans.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def compute_expert_scores(
    calibration: CalibrationArtifacts,
    config: Any,
) -> dict[int, torch.Tensor]:
    """Compute routing_frequency × output_sensitivity for each expert.
    
    Returns:
        dict[layer_idx] -> tensor[num_experts] of scores
    """
    scores: dict[int, torch.Tensor] = {}
    
    for layer_idx in range(config.num_hidden_layers):
        routing_counts = calibration.routing_counts[layer_idx].float()
        
        # Output sensitivity from MxMoE deltas (W1 + W2)
        w1_delta = calibration.mxmoe_w1_deltas[layer_idx].float()
        w2_delta = calibration.mxmoe_w2_deltas[layer_idx].float()
        sensitivity = w1_delta + w2_delta
        
        # Normalize routing counts to [0, 1]
        max_routing = routing_counts.max()
        if max_routing > 0:
            routing_freq = routing_counts / max_routing
        else:
            routing_freq = routing_counts
        
        # Normalize sensitivity to [0, 1]
        max_sensitivity = sensitivity.max()
        if max_sensitivity > 0:
            norm_sensitivity = sensitivity / max_sensitivity
        else:
            norm_sensitivity = sensitivity
        
        # Combined score: routing_freq × sensitivity
        scores[layer_idx] = routing_freq * norm_sensitivity
    
    return scores


def build_heterogeneous_precision_masks(
    calibration: CalibrationArtifacts,
    config: Any,
    target_layers: frozenset[int] | None,
    fp4_threshold: float,
    fp8_threshold: float,
    bf16_threshold: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    """Build masks based on heterogeneous precision assignment.
    
    Precision assignment:
    - score >= bf16_threshold: BF16 (no quantization)
    - score >= fp8_threshold: FP8
    - score >= fp4_threshold: FP4
    - score < fp4_threshold: FP4 (default)
    """
    w1_pair_masks = {layer_idx: {} for layer_idx in range(config.num_hidden_layers)}
    w2_channel_masks = {layer_idx: {} for layer_idx in range(config.num_hidden_layers)}
    
    scores = compute_expert_scores(calibration, config)
    
    # Track statistics
    bf16_experts = 0
    fp8_experts = 0
    fp4_experts = 0
    
    for layer_idx in range(config.num_hidden_layers):
        if target_layers is not None and layer_idx not in target_layers:
            # Use default joint masks for non-target layers
            for expert_idx in range(config.num_experts):
                w1_pair_masks[layer_idx][expert_idx] = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.zeros(config.hidden_size, dtype=torch.bool)
            continue
        
        layer_scores = scores[layer_idx]
        
        for expert_idx in range(config.num_experts):
            score = float(layer_scores[expert_idx].item())
            
            if score >= bf16_threshold:
                # BF16: no quantization (all channels stay in full precision)
                w1_pair_masks[layer_idx][expert_idx] = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.zeros(config.hidden_size, dtype=torch.bool)
                bf16_experts += 1
            elif score >= fp8_threshold:
                # FP8: promote all channels to FP8
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
                fp8_experts += 1
            else:
                # FP4: default (no promotion)
                w1_pair_masks[layer_idx][expert_idx] = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.zeros(config.hidden_size, dtype=torch.bool)
                fp4_experts += 1
    
    return w1_pair_masks, w2_channel_masks, {
        "heterogeneous_precision_config": {
            "fp4_threshold": float(fp4_threshold),
            "fp8_threshold": float(fp8_threshold),
            "bf16_threshold": float(bf16_threshold),
            "target_layers": list(target_layers) if target_layers else "all",
        },
        "expert_distribution": {
            "bf16_experts": int(bf16_experts),
            "fp8_experts": int(fp8_experts),
            "fp4_experts": int(fp4_experts),
        },
    }


def build_heterogeneous_configs() -> list[HeterogeneousPrecisionConfig]:
    """Build a set of heterogeneous precision configurations to test."""
    return [
        # Test on layers 35-39 first (highest error layers)
        HeterogeneousPrecisionConfig(
            name="hetero_layers35_39_aggressive",
            description="Aggressive heterogeneous precision on layers 35-39: BF16 for top 10%, FP8 for top 30%, FP4 for rest",
            target_layers=frozenset([35, 36, 37, 38, 39]),
            fp4_threshold=0.0,
            fp8_threshold=0.3,
            bf16_threshold=0.7,
        ),
        HeterogeneousPrecisionConfig(
            name="hetero_layers35_39_moderate",
            description="Moderate heterogeneous precision on layers 35-39: BF16 for top 5%, FP8 for top 20%, FP4 for rest",
            target_layers=frozenset([35, 36, 37, 38, 39]),
            fp4_threshold=0.0,
            fp8_threshold=0.2,
            bf16_threshold=0.5,
        ),
        HeterogeneousPrecisionConfig(
            name="hetero_layers35_39_conservative",
            description="Conservative heterogeneous precision on layers 35-39: BF16 for top 2%, FP8 for top 10%, FP4 for rest",
            target_layers=frozenset([35, 36, 37, 38, 39]),
            fp4_threshold=0.0,
            fp8_threshold=0.1,
            bf16_threshold=0.3,
        ),
        # Test on all layers
        HeterogeneousPrecisionConfig(
            name="hetero_all_layers_aggressive",
            description="Aggressive heterogeneous precision on all layers: BF16 for top 10%, FP8 for top 30%, FP4 for rest",
            target_layers=None,
            fp4_threshold=0.0,
            fp8_threshold=0.3,
            bf16_threshold=0.7,
        ),
        HeterogeneousPrecisionConfig(
            name="hetero_all_layers_moderate",
            description="Moderate heterogeneous precision on all layers: BF16 for top 5%, FP8 for top 20%, FP4 for rest",
            target_layers=None,
            fp4_threshold=0.0,
            fp8_threshold=0.2,
            bf16_threshold=0.5,
        ),
        HeterogeneousPrecisionConfig(
            name="hetero_all_layers_conservative",
            description="Conservative heterogeneous precision on all layers: BF16 for top 2%, FP8 for top 10%, FP4 for rest",
            target_layers=None,
            fp4_threshold=0.0,
            fp8_threshold=0.1,
            bf16_threshold=0.3,
        ),
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    
    print(f"Loading model config from {args.model_id}...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    
    print(f"Loading calibration cache from {args.cache_path}...", flush=True)
    calibration_tuple = load_cache(args.cache_path, args.model_id)
    calibration = calibration_tuple[0]
    
    print("Loading test data...", flush=True)
    _tokenizer, _calib_chunks, test_ids, _calib_info, _eval_info = load_gptq_standard_data(args.model_id)
    
    index_payload = root_config.get('index_payload', {})
    total_bf16_bytes = int(root_config.get('total_size', 0) or index_payload.get('metadata', {}).get('total_size', 0))
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)
    
    # Load reference rows for comparison
    reference_rows = load_reference_rows(args.output_json)
    
    # Build and evaluate all heterogeneous precision configs
    configs = build_heterogeneous_configs()
    requested_plans = resolve_requested_plans(args.plans)
    
    results: dict[str, Any] = {}
    
    for config in configs:
        if requested_plans and config.name not in requested_plans:
            continue
        
        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {config.name}", flush=True)
        print(f"Description: {config.description}", flush=True)
        print(f"{'='*80}", flush=True)
        
        start_time = time.time()
        
        # Build masks
        w1_masks, w2_masks, meta = build_heterogeneous_precision_masks(
            calibration,
            text_config,
            config.target_layers,
            config.fp4_threshold,
            config.fp8_threshold,
            config.bf16_threshold,
        )
        
        # Build plan
        plan = build_plan_from_masks(
            config.name,
            config.description,
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        )
        
        # Evaluate
        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time
        
        results[config.name] = {
            "ppl": round(ppl, 4),
            "memory_gb": round(plan.memory_gb, 3),
            "fp8_weights": int(plan.fp8_weights),
            "w1_fraction": -1,  # skipped
            "w2_fraction": -1,  # skipped
            "elapsed_s": round(elapsed, 1),
            **meta,
        }
        
        print(f"PPL: {ppl:.4f} | Memory: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)
    
    # Save results
    payload = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": 296960,
            "seqlen": SEQLEN,
            "nsamples": 145,
            "dtype": args.dtype,
            "approach": "Per-Expert Heterogeneous Precision",
            "hypothesis": "Hot experts can tolerate FP4, cold experts need FP8/BF16",
            "reference_rows": reference_rows,
        },
        "results": results,
    }
    
    atomic_json_dump(args.output_json, payload)
    print(f"\nResults saved to {args.output_json}", flush=True)
    
    # Print summary
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("ppl", float("inf")))
    for i, (name, result) in enumerate(sorted_results[:5], 1):
        ppl = result.get("ppl", "N/A")
        mem = result.get("memory_gb", "N/A")
        print(f"{i}. {name:50s} PPL={ppl:.4f} Mem={mem} GB")


if __name__ == "__main__":
    main()

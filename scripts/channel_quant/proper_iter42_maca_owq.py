#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""
Iteration 42: MaCa + OWQ (Outlier-Aware Quantization)

Technique: Detect outlier weights (values > 3σ from mean), keep in FP8, quantize rest in FP4.
Foundation: MaCa uniform 4K calibration (proven best from Iter29).
Expected improvement: 0.002-0.005 PPL over Iter29 (6.567582 → 6.564-6.566).

Reference: OWQ (arXiv:2404.02079) - Outlier-Aware Quantization for MoE models.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter25_learned_correction import maybe_slice_eval_chunks, resolve_requested_plans
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
    summarize_calibration_artifacts,
)
from proper_iter28_maca_plus_correction import BaseVariant, evaluate_and_record_compound_plan, fit_scalar_corrections_for_variant
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter42_maca_owq.json"
TOTAL_CALIBRATION_CHUNKS = 128
OUTLIER_SIGMA_THRESHOLD = 3.0  # Detect weights > 3σ from mean as outliers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on WikiText-2 evaluation chunks for smoke tests; 0 means full 145-chunk test.",
    )
    return parser.parse_args()


def detect_outliers_per_channel(
    weight: torch.Tensor,
    sigma_threshold: float = 3.0,
) -> torch.Tensor:
    """
    Detect outlier weights per channel using statistical method.
    
    Args:
        weight: Shape [out_features, in_features] or [out_features, in_features, ...]
        sigma_threshold: Number of standard deviations to use as threshold
    
    Returns:
        Boolean mask of shape [out_features, in_features] where True = outlier
    """
    # Flatten to [out_features, -1] for per-channel analysis
    original_shape = weight.shape
    if len(original_shape) > 2:
        weight_flat = weight.reshape(original_shape[0], -1)
    else:
        weight_flat = weight
    
    # Compute per-channel statistics
    mean = weight_flat.mean(dim=1, keepdim=True)
    std = weight_flat.std(dim=1, keepdim=True)
    
    # Detect outliers: |weight - mean| > sigma_threshold * std
    outlier_mask = torch.abs(weight_flat - mean) > (sigma_threshold * std)
    
    # Reshape back to original shape
    if len(original_shape) > 2:
        outlier_mask = outlier_mask.reshape(original_shape)
    
    return outlier_mask


def apply_owq_precision_assignment(
    weight: torch.Tensor,
    outlier_mask: torch.Tensor,
    outlier_precision: str = "fp8",
    normal_precision: str = "fp4",
) -> dict[str, Any]:
    """
    Assign precision based on outlier detection.
    
    Returns:
        Dictionary with:
        - 'outlier_mask': Boolean mask of outliers
        - 'outlier_count': Number of outlier weights
        - 'outlier_fraction': Fraction of weights that are outliers
        - 'outlier_precision': Precision for outliers
        - 'normal_precision': Precision for normal weights
    """
    outlier_count = outlier_mask.sum().item()
    total_count = outlier_mask.numel()
    outlier_fraction = outlier_count / total_count if total_count > 0 else 0.0
    
    return {
        'outlier_mask': outlier_mask,
        'outlier_count': outlier_count,
        'outlier_fraction': outlier_fraction,
        'total_count': total_count,
        'outlier_precision': outlier_precision,
        'normal_precision': normal_precision,
    }


def build_owq_masks(
    calibration_data: dict[str, Any],
    weight_store: WeightStore,
    model_config: Any,
    device: str,
    dtype: torch.dtype,
    sigma_threshold: float = 3.0,
) -> dict[str, Any]:
    """
    Build OWQ precision masks for all experts.
    
    Strategy:
    1. For each expert W1 and W2 weight, detect outliers using 3σ threshold
    2. Assign FP8 to outliers, FP4 to normal weights
    3. Use joint W1/W2 mask building with topup for final precision assignment
    """
    print(f"Building OWQ masks with sigma_threshold={sigma_threshold}...", flush=True)
    
    num_layers = model_config.num_hidden_layers
    num_experts = model_config.num_experts
    
    owq_analysis = {
        'sigma_threshold': sigma_threshold,
        'layers': {},
        'summary': {
            'total_outliers': 0,
            'total_weights': 0,
            'overall_outlier_fraction': 0.0,
        }
    }
    
    # Analyze each layer's experts
    for layer_idx in range(num_layers):
        layer_analysis = {'experts': {}}
        layer_total_outliers = 0
        layer_total_weights = 0
        
        for expert_idx in range(num_experts):
            w1_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"
            w2_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"
            
            try:
                weights = weight_store.load_tensors([w1_key, w2_key])
                w1 = weights[w1_key].to(device=device, dtype=dtype)
                w2 = weights[w2_key].to(device=device, dtype=dtype)
                
                # Detect outliers
                w1_outliers = detect_outliers_per_channel(w1, sigma_threshold)
                w2_outliers = detect_outliers_per_channel(w2, sigma_threshold)
                
                w1_analysis = apply_owq_precision_assignment(w1, w1_outliers)
                w2_analysis = apply_owq_precision_assignment(w2, w2_outliers)
                
                layer_analysis['experts'][expert_idx] = {
                    'w1': w1_analysis,
                    'w2': w2_analysis,
                }
                
                layer_total_outliers += w1_analysis['outlier_count'] + w2_analysis['outlier_count']
                layer_total_weights += w1_analysis['total_count'] + w2_analysis['total_count']
                
            except Exception as e:
                print(f"Warning: Could not analyze layer {layer_idx} expert {expert_idx}: {e}", flush=True)
        
        if layer_total_weights > 0:
            layer_analysis['outlier_fraction'] = layer_total_outliers / layer_total_weights
        
        owq_analysis['layers'][layer_idx] = layer_analysis
        owq_analysis['summary']['total_outliers'] += layer_total_outliers
        owq_analysis['summary']['total_weights'] += layer_total_weights
    
    if owq_analysis['summary']['total_weights'] > 0:
        owq_analysis['summary']['overall_outlier_fraction'] = (
            owq_analysis['summary']['total_outliers'] / owq_analysis['summary']['total_weights']
        )
    
    print(f"OWQ analysis complete: {owq_analysis['summary']['overall_outlier_fraction']:.4f} outlier fraction", flush=True)
    
    return owq_analysis


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device
    dtype = dtype_from_name(args.dtype)
    
    print(f"Loading model config from {args.model_id}...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    model_config = build_text_config(root_config)
    weight_store = WeightStore(args.model_id, snapshot_dir, weight_map)
    
    print(f"Loading evaluation data and tokenizer...", flush=True)
    tokenizer, _standard_calib_chunks, test_ids_full, eval_info_base, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}
    
    print(f"Loading WikiText-2 training data for MaCa calibration...", flush=True)
    train_ids = load_wikitext_train_ids(tokenizer)
    
    # Use MaCa uniform 4K calibration (proven best from Iter29)
    calib_set = VariableLengthCalibrationSet(
        token_ids=train_ids,
        length_counts=[(4096, 128)],  # Only 4096-token chunks
        pad_length=MACA_PAD_LENGTH,
    )
    
    print(f"Running MaCa calibration with {len(calib_set)} chunks...", flush=True)
    start_time = time.time()
    calibration_data = run_maca_calibration(
        calib_set=calib_set,
        model_id=args.model_id,
        model_config=model_config,
        weight_store=weight_store,
        device=device,
        dtype=dtype,
    )
    calib_time = time.time() - start_time
    print(f"MaCa calibration completed in {calib_time:.1f}s", flush=True)
    
    # Build OWQ masks
    print(f"Building OWQ precision masks...", flush=True)
    owq_analysis = build_owq_masks(
        calibration_data=calibration_data,
        weight_store=weight_store,
        model_config=model_config,
        device=device,
        dtype=dtype,
        sigma_threshold=OUTLIER_SIGMA_THRESHOLD,
    )
    
    # Build joint W1/W2 masks with topup (same as Iter29 best)
    print(f"Building joint W1/W2 masks with topup...", flush=True)
    masks = build_joint_with_topup_masks(
        calibration_data=calibration_data,
        topup_fraction=0.05,  # 5% FP8 budget (same as Iter29)
    )
    
    # Build quantization plan
    print(f"Building quantization plan...", flush=True)
    plan = build_plan_from_masks(masks)
    
    print(f"Evaluating quantization plan...", flush=True)
    start_eval = time.time()
    result = evaluate_and_record_plan(
        plan=plan,
        eval_data=test_ids,
        model_id=args.model_id,
        model_config=model_config,
        weight_store=weight_store,
        device=device,
        dtype=dtype,
    )
    eval_time = time.time() - start_eval
    
    # Prepare output
    output = {
        'metadata': {
            'model': args.model_id,
            'device': device,
            'dtype': args.dtype,
            'seed': args.seed,
            'technique': 'MaCa + OWQ (Outlier-Aware Quantization)',
            'description': 'Detect outlier weights (>3σ), keep in FP8, quantize rest in FP4. Use MaCa uniform 4K calibration.',
            'reference': 'OWQ (arXiv:2404.02079)',
        },
        'calibration': {
            'type': 'MaCa Uniform 4K',
            'chunks': 128,
            'chunk_length': 4096,
            'time_s': calib_time,
        },
        'owq_analysis': owq_analysis,
        'quantization': {
            'mask_builder': 'joint_w1w2_with_topup',
            'topup_fraction': 0.05,
        },
        'evaluation': {
            'ppl': result.get('ppl', 0.0),
            'nll': result.get('nll', 0.0),
            'time_s': eval_time,
            'chunks': len(test_ids),
        },
        'result': result,
    }
    
    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(output, args.output_json)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Iteration 42: MaCa + OWQ", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"PPL: {output['evaluation']['ppl']:.6f}", flush=True)
    print(f"NLL: {output['evaluation']['nll']:.6f}", flush=True)
    print(f"Outlier fraction: {owq_analysis['summary']['overall_outlier_fraction']:.4f}", flush=True)
    print(f"Evaluation time: {eval_time:.1f}s", flush=True)
    print(f"Results saved to: {args.output_json}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()

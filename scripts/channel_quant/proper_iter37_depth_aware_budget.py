#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 37: Depth-Aware Budget Allocation (Correct Implementation).

Key finding from iter33:
  top3_layers_fp8 (ALL channels FP8 for layers 37-39) = 6.5775 PPL — WORSE than:
  - MaCa calibration: 6.5676 PPL
  - BF16 restoration: 6.5712 PPL
  - Standard joint: 6.5725 PPL

The problem: overriding ALL channels to FP8 is wrong. The correct approach is to
INCREASE the FP8 budget for sensitive layers while keeping channel selection optimal.

Key insight from SliderQuant (ICLR 2026):
  "Shallow/deep layers are usually more sensitive to quantization than intermediate layers."

Correct depth-aware approach:
  - Sensitive layers (37-39): higher FP8 fraction (e.g., 30-50% of channels)
  - Other layers: standard FP8 fraction (e.g., 5% of channels)
  - Channel selection: always use MaCa calibration (optimal sensitivity scores)

This is different from iter33 which used ALL-FP8 for sensitive layers.

Configs:
  1. maca_depth_top3_30pct:  MaCa + layers 37-39 at 30% FP8, others at 5%
  2. maca_depth_top3_50pct:  MaCa + layers 37-39 at 50% FP8, others at 5%
  3. maca_depth_top5_30pct:  MaCa + layers 35-39 at 30% FP8, others at 5%
  4. maca_depth_top3_20pct:  MaCa + layers 37-39 at 20% FP8, others at 5%
  5. maca_uniform_10pct:     MaCa + all layers at 10% FP8 (budget-matched baseline)
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
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter10_novel_perchannel import build_global_fraction_masks
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
)
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter37_depth_aware_budget.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
MACA_CACHE_PATH = RESULTS_DIR / "proper_iter37_maca_cache.pt"

# Sensitivity ranking from iter31_heterogeneous_analysis.json
SENSITIVITY_RANKED_LAYERS = [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 25, 23, 26, 19, 22, 24, 21, 20, 18, 16, 15, 13, 11, 17, 12, 14, 10, 9, 8, 7, 5, 2, 6, 4, 3, 1, 0]


@dataclass(frozen=True)
class DepthBudgetConfig:
    name: str
    description: str
    sensitive_layers: frozenset[int]
    sensitive_w1_fraction: float  # FP8 fraction for sensitive layers
    sensitive_w2_fraction: float
    base_w1_fraction: float       # FP8 fraction for other layers
    base_w2_fraction: float
    use_maca: bool = True


def build_depth_budget_masks(
    calibration: Any,
    text_config: Any,
    config: DepthBudgetConfig,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build masks with depth-aware budget allocation.
    
    For sensitive layers: use higher FP8 fraction
    For other layers: use base FP8 fraction
    Channel selection: always use calibration sensitivity scores
    """
    from proper_iter10_novel_perchannel import build_global_fraction_masks
    
    metric_cache = calibration.activation_cache
    
    # Build per-layer fraction overrides
    w1_layer_fractions = {}
    w2_layer_fractions = {}
    for layer_idx in range(text_config.num_hidden_layers):
        if layer_idx in config.sensitive_layers:
            w1_layer_fractions[layer_idx] = config.sensitive_w1_fraction
            w2_layer_fractions[layer_idx] = config.sensitive_w2_fraction
        else:
            w1_layer_fractions[layer_idx] = config.base_w1_fraction
            w2_layer_fractions[layer_idx] = config.base_w2_fraction
    
    # Build masks using per-layer fractions
    w1_masks = {}
    w2_masks = {}
    
    for layer_idx in range(text_config.num_hidden_layers):
        if layer_idx not in metric_cache:
            continue
        
        bundle = metric_cache[layer_idx]
        w1_frac = w1_layer_fractions.get(layer_idx, config.base_w1_fraction)
        w2_frac = w2_layer_fractions.get(layer_idx, config.base_w2_fraction)
        
        w1_masks[layer_idx] = {}
        w2_masks[layer_idx] = {}
        
        for expert_idx in range(text_config.num_experts):
            # W1: select top-k pairs by sensitivity score
            if hasattr(bundle, 'w1_pair_scores') and expert_idx < bundle.w1_pair_scores.shape[0]:
                scores = bundle.w1_pair_scores[expert_idx]  # [num_pairs]
                k = max(1, int(w1_frac * len(scores)))
                topk_indices = torch.topk(scores, k).indices
                mask = torch.zeros(len(scores), dtype=torch.bool)
                mask[topk_indices] = True
                w1_masks[layer_idx][expert_idx] = mask
            else:
                w1_masks[layer_idx][expert_idx] = torch.zeros(text_config.moe_intermediate_size, dtype=torch.bool)
            
            # W2: select top-k channels by sensitivity score
            if hasattr(bundle, 'w2_channel_scores') and expert_idx < bundle.w2_channel_scores.shape[0]:
                scores = bundle.w2_channel_scores[expert_idx]  # [num_channels]
                k = max(1, int(w2_frac * len(scores)))
                topk_indices = torch.topk(scores, k).indices
                mask = torch.zeros(len(scores), dtype=torch.bool)
                mask[topk_indices] = True
                w2_masks[layer_idx][expert_idx] = mask
            else:
                w2_masks[layer_idx][expert_idx] = torch.zeros(text_config.hidden_size, dtype=torch.bool)
    
    return w1_masks, w2_masks


def build_maca_calibration(model_id: str, seed: int, device: torch.device, dtype: torch.dtype) -> Any:
    """Build MaCa uniform_4k calibration."""
    if MACA_CACHE_PATH.exists():
        print(f"Loading MaCa cache from {MACA_CACHE_PATH}...", flush=True)
        from proper_iter01 import load_cache
        calib, _, _ = load_cache(MACA_CACHE_PATH, model_id)
        if calib is not None:
            return calib
    
    print("Building MaCa uniform_4k calibration (128 × 4096-token chunks)...", flush=True)
    train_ids = load_wikitext_train_ids(model_id)
    length_counts = ((4096, 128),)
    calib_set = VariableLengthCalibrationSet.build(
        train_ids=train_ids,
        length_counts=length_counts,
        total_chunks=128,
        seed=seed,
    )
    snapshot_dir, root_config, weight_map = load_root_config(model_id)
    text_config = build_text_config(root_config)
    store = WeightStore(model_id, snapshot_dir, weight_map)
    calibration = run_maca_calibration(calib_set, text_config, store, device, dtype, MACA_PAD_LENGTH)
    torch.save({"calibration": calibration, "model_id": model_id}, MACA_CACHE_PATH)
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="")
    parser.add_argument("--seed", type=int, default=0)
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

    # Load MaCa calibration (best calibration)
    maca_calibration = build_maca_calibration(args.model_id, args.seed, device, dtype)
    
    # Also load standard calibration for comparison
    standard_calibration, _, _ = load_cache(args.cache_path, args.model_id)

    top3 = frozenset(SENSITIVITY_RANKED_LAYERS[:3])   # layers 37, 38, 39
    top5 = frozenset(SENSITIVITY_RANKED_LAYERS[:5])   # layers 35-39

    configs = [
        # Depth-aware budget with MaCa calibration
        DepthBudgetConfig(
            name="maca_depth_top3_30pct",
            description="MaCa + layers 37-39 at 30% FP8, others at 5%",
            sensitive_layers=top3,
            sensitive_w1_fraction=0.30, sensitive_w2_fraction=0.30,
            base_w1_fraction=0.05, base_w2_fraction=0.05,
        ),
        DepthBudgetConfig(
            name="maca_depth_top3_50pct",
            description="MaCa + layers 37-39 at 50% FP8, others at 5%",
            sensitive_layers=top3,
            sensitive_w1_fraction=0.50, sensitive_w2_fraction=0.50,
            base_w1_fraction=0.05, base_w2_fraction=0.05,
        ),
        DepthBudgetConfig(
            name="maca_depth_top5_30pct",
            description="MaCa + layers 35-39 at 30% FP8, others at 5%",
            sensitive_layers=top5,
            sensitive_w1_fraction=0.30, sensitive_w2_fraction=0.30,
            base_w1_fraction=0.05, base_w2_fraction=0.05,
        ),
        DepthBudgetConfig(
            name="maca_depth_top3_20pct",
            description="MaCa + layers 37-39 at 20% FP8, others at 5%",
            sensitive_layers=top3,
            sensitive_w1_fraction=0.20, sensitive_w2_fraction=0.20,
            base_w1_fraction=0.05, base_w2_fraction=0.05,
        ),
        # Budget-matched baselines
        DepthBudgetConfig(
            name="maca_uniform_10pct",
            description="MaCa + all layers at 10% FP8 (budget-matched)",
            sensitive_layers=frozenset(),
            sensitive_w1_fraction=0.10, sensitive_w2_fraction=0.10,
            base_w1_fraction=0.10, base_w2_fraction=0.10,
        ),
        DepthBudgetConfig(
            name="maca_uniform_5pct",
            description="MaCa + all layers at 5% FP8 (standard baseline)",
            sensitive_layers=frozenset(),
            sensitive_w1_fraction=0.05, sensitive_w2_fraction=0.05,
            base_w1_fraction=0.05, base_w2_fraction=0.05,
        ),
    ]

    requested_plans = set(args.plans.split(",")) if args.plans else None
    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}

    for config in configs:
        if requested_plans and config.name not in requested_plans:
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {config.name}", flush=True)
        print(f"Description: {config.description}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()
        
        calibration = maca_calibration
        w1_masks, w2_masks = build_depth_budget_masks(calibration, text_config, config)
        plan = build_plan_from_masks(config.name, config.description, text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time

        results[config.name] = {
            "ppl": ppl,
            "memory_gb": round(plan.memory_gb, 3),
            "description": config.description,
            "sensitive_layers": sorted(config.sensitive_layers),
            "sensitive_w1_fraction": config.sensitive_w1_fraction,
            "sensitive_w2_fraction": config.sensitive_w2_fraction,
            "base_w1_fraction": config.base_w1_fraction,
            "base_w2_fraction": config.base_w2_fraction,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  PPL: {ppl:.6f} | Mem: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

        atomic_json_dump({"results": results, "metadata": {"approach": "depth_aware_budget"}}, args.output_json)

    print(f"\nDone. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

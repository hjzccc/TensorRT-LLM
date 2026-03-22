#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 36: SINQ-Inspired Channel Sensitivity Metric.

Key insight from SINQ (arXiv:2509.22944, Huawei, Sep 2025):
  "Column-wise standard deviations of LLM weight matrices are predictive of
   input activation magnitudes the matrix received during training."
  
  SINQ uses Sinkhorn-Knopp iterations to find dual-axis (row + column) scales
  that normalize per-row and per-column variances simultaneously.

Our current metric uses kurtosis for channel selection:
  w2_scores[expert, channel] = sum_k(|W2_diff[channel, k]| * kurtosis[k])

SINQ-inspired metric: use column-wise weight std as proxy for activation magnitude:
  sinq_score[expert, channel] = 1 / std(W2[:, channel])  (higher = more sensitive)
  
This is calibration-FREE — no activation data needed. It recovers activation-awareness
from the weight matrix structure alone.

Also test: combined metric = sinq_score * kurtosis (captures both structure and outliers)

Expected gain: 0.001-0.003 PPL (better channel importance weighting without calibration)

Run:
  python proper_iter36_sinq_channel_metric.py
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    atomic_json_dump,
    dtype_from_name,
    evaluate_plan,
    load_gptq_standard_data,
)
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter36_sinq_channel_metric.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


def compute_sinq_channel_scores(
    gate_up_proj: torch.Tensor,  # [num_experts, 2*inter_size, hidden_size]
    down_proj: torch.Tensor,     # [num_experts, hidden_size, inter_size]
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SINQ-inspired channel sensitivity scores.
    
    For W1 (gate_up_proj): score = 1 / col_std(W1)
      - Columns with low std have high activation magnitudes (more sensitive)
    For W2 (down_proj): score = 1 / col_std(W2)
      - Columns with low std have high activation magnitudes (more sensitive)
    
    Returns:
      w1_scores: [num_experts, hidden_size] — per-input-channel scores for W1
      w2_scores: [num_experts, inter_size] — per-input-channel scores for W2
    """
    num_experts = gate_up_proj.shape[0]
    hidden_size = gate_up_proj.shape[2]
    inter_size = down_proj.shape[2]
    
    w1_scores = torch.zeros(num_experts, hidden_size, device=gate_up_proj.device)
    w2_scores = torch.zeros(num_experts, inter_size, device=down_proj.device)
    
    for expert_idx in range(num_experts):
        # W1: [2*inter_size, hidden_size] — columns are input channels
        w1 = gate_up_proj[expert_idx].float()  # [2*inter_size, hidden_size]
        col_std_w1 = w1.std(dim=0).clamp(min=1e-8)  # [hidden_size]
        w1_scores[expert_idx] = 1.0 / col_std_w1  # Higher = more sensitive
        
        # W2: [hidden_size, inter_size] — columns are input channels (intermediate)
        w2 = down_proj[expert_idx].float()  # [hidden_size, inter_size]
        col_std_w2 = w2.std(dim=0).clamp(min=1e-8)  # [inter_size]
        w2_scores[expert_idx] = 1.0 / col_std_w2  # Higher = more sensitive
    
    return w1_scores, w2_scores


def compute_combined_channel_scores(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    calibration: CalibrationArtifacts,
    config: Any,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute combined SINQ + kurtosis channel scores.
    
    combined = sinq_score * kurtosis_score
    """
    sinq_w1, sinq_w2 = compute_sinq_channel_scores(gate_up_proj, down_proj, config)
    
    # Get kurtosis scores from calibration
    if hasattr(calibration, 'layer_metrics') and layer_idx in calibration.layer_metrics:
        bundle = calibration.layer_metrics[layer_idx]
        kurtosis_w1 = bundle.w1_pair_scores.mean(dim=-1)  # [num_experts, hidden_size]
        kurtosis_w2 = bundle.w2_channel_scores  # [num_experts, inter_size]
        
        # Normalize both to [0, 1]
        sinq_w1_norm = (sinq_w1 - sinq_w1.min()) / (sinq_w1.max() - sinq_w1.min() + 1e-8)
        sinq_w2_norm = (sinq_w2 - sinq_w2.min()) / (sinq_w2.max() - sinq_w2.min() + 1e-8)
        kurtosis_w1_norm = (kurtosis_w1 - kurtosis_w1.min()) / (kurtosis_w1.max() - kurtosis_w1.min() + 1e-8)
        kurtosis_w2_norm = (kurtosis_w2 - kurtosis_w2.min()) / (kurtosis_w2.max() - kurtosis_w2.min() + 1e-8)
        
        combined_w1 = sinq_w1_norm * kurtosis_w1_norm
        combined_w2 = sinq_w2_norm * kurtosis_w2_norm
        return combined_w1, combined_w2
    
    return sinq_w1, sinq_w2


def build_sinq_calibration(
    model_id: str,
    snapshot_dir: Path,
    weight_map: dict,
    text_config: Any,
    device: torch.device,
    dtype: torch.dtype,
    metric: str = "sinq",  # "sinq", "combined"
    base_calibration: CalibrationArtifacts | None = None,
) -> CalibrationArtifacts:
    """Build calibration artifacts using SINQ-inspired channel scores."""
    from proper_eval import CalibrationArtifacts, ExpertMomentState, init_expert_moment_state
    
    store = WeightStore(model_id, snapshot_dir, weight_map)
    
    # Build a calibration artifact with SINQ-based scores
    layer_metrics = {}
    
    for layer_idx in range(text_config.num_hidden_layers):
        layer_type = text_config.layer_types[layer_idx]
        if layer_type != "linear_attention":
            continue
        
        # Load weights
        from spike1_ground_truth import layer_keys, shorten_layer_tensors
        raw = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw, device, dtype)
        del raw
        
        gate_up_proj = tensors.get("experts.gate_up_proj")
        down_proj = tensors.get("experts.down_proj")
        
        if gate_up_proj is None or down_proj is None:
            continue
        
        if metric == "sinq":
            w1_scores, w2_scores = compute_sinq_channel_scores(gate_up_proj, down_proj, text_config)
        elif metric == "combined" and base_calibration is not None:
            w1_scores, w2_scores = compute_combined_channel_scores(
                gate_up_proj, down_proj, base_calibration, text_config, layer_idx
            )
        else:
            w1_scores, w2_scores = compute_sinq_channel_scores(gate_up_proj, down_proj, text_config)
        
        # Build LayerMetricBundle with SINQ scores
        # w1_pair_scores: [num_experts, num_pairs] where num_pairs = hidden_size
        # w2_channel_scores: [num_experts, inter_size]
        bundle = LayerMetricBundle(
            w1_pair_scores=w1_scores.unsqueeze(-1).expand(-1, -1, 1),  # [num_experts, hidden_size, 1]
            w2_channel_scores=w2_scores,
            routing_counts=torch.ones(text_config.num_experts),
        )
        layer_metrics[layer_idx] = bundle
        
        del tensors, gate_up_proj, down_proj
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"  [sinq] layer {layer_idx+1}/{text_config.num_hidden_layers}", flush=True)
    
    # Create a CalibrationArtifacts with SINQ scores
    # Use base_calibration's other fields if available
    if base_calibration is not None:
        sinq_calib = CalibrationArtifacts(
            layer_metrics=layer_metrics,
            routing_counts=base_calibration.routing_counts,
            activation_cache=base_calibration.activation_cache,
        )
    else:
        sinq_calib = CalibrationArtifacts(
            layer_metrics=layer_metrics,
            routing_counts={},
            activation_cache={},
        )
    
    return sinq_calib


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

    # Load base calibration (for combined metric)
    base_calibration, _, _ = load_cache(args.cache_path, args.model_id)

    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}

    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    configs_to_eval = [
        ("sinq_joint_topup", "sinq", "SINQ-inspired channel scores (1/col_std) + joint W1/W2 topup"),
        ("combined_joint_topup", "combined", "Combined SINQ + kurtosis scores + joint W1/W2 topup"),
        ("baseline_kurtosis", None, "Baseline: kurtosis scores (existing best)"),
    ]

    requested_plans = set(args.plans.split(",")) if args.plans else None

    for plan_name, metric, description in configs_to_eval:
        if requested_plans and plan_name not in requested_plans:
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {plan_name} ({description})", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        if metric is None:
            # Use base calibration (kurtosis baseline)
            calibration = base_calibration
        else:
            # Build SINQ-based calibration
            calibration = build_sinq_calibration(
                args.model_id, snapshot_dir, weight_map, text_config,
                device, dtype, metric=metric, base_calibration=base_calibration,
            )

        w1_masks, w2_masks = build_joint_with_topup_masks(calibration, text_config, JOINT_MEDIUM_TOPUP_FRACTION)
        plan = build_plan_from_masks(plan_name, description, text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time

        results[plan_name] = {
            "ppl": ppl,
            "memory_gb": plan.memory_gb,
            "description": description,
            "metric": metric,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  PPL: {ppl:.6f} | Mem: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

        atomic_json_dump({"results": results, "metadata": {"metric": "sinq_inspired"}}, args.output_json)

    print(f"\nDone. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

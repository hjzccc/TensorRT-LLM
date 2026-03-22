#!/usr/bin/env python3
"""Iteration 31: Heterogeneous Precision Analysis

Quick analysis to understand expert-level precision requirements without full evaluation.
Computes routing frequency × sensitivity scores and analyzes distribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from proper_iter01 import load_cache
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter31_heterogeneous_analysis.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


def analyze_expert_scores(cache_path: Path, model_id: str) -> dict[str, Any]:
    """Analyze routing frequency × sensitivity scores for all experts."""
    
    print("Loading model config...")
    snapshot_dir, root_config, weight_map = load_root_config(model_id)
    text_config = build_text_config(root_config)
    
    print(f"Loading calibration cache from {cache_path}...")
    calibration = load_cache(cache_path, model_id)
    
    results = {
        "metadata": {
            "model": model_id,
            "num_layers": text_config.num_hidden_layers,
            "num_experts": text_config.num_experts,
        },
        "per_layer_analysis": {},
        "global_statistics": {},
    }
    
    all_scores = []
    
    for layer_idx in range(text_config.num_hidden_layers):
        routing_counts = calibration.routing_counts[layer_idx].float()
        w1_delta = calibration.mxmoe_w1_deltas[layer_idx].float()
        w2_delta = calibration.mxmoe_w2_deltas[layer_idx].float()
        sensitivity = w1_delta + w2_delta
        
        # Normalize
        max_routing = routing_counts.max()
        routing_freq = routing_counts / max_routing if max_routing > 0 else routing_counts
        
        max_sensitivity = sensitivity.max()
        norm_sensitivity = sensitivity / max_sensitivity if max_sensitivity > 0 else sensitivity
        
        # Combined score
        scores = routing_freq * norm_sensitivity
        all_scores.extend(scores.tolist())
        
        # Per-layer statistics
        layer_stats = {
            "mean_score": float(scores.mean().item()),
            "std_score": float(scores.std().item()),
            "min_score": float(scores.min().item()),
            "max_score": float(scores.max().item()),
            "median_score": float(scores.median().item()),
            "top_10_pct_threshold": float(torch.quantile(scores, 0.9).item()),
            "top_5_pct_threshold": float(torch.quantile(scores, 0.95).item()),
            "top_2_pct_threshold": float(torch.quantile(scores, 0.98).item()),
            "top_1_pct_threshold": float(torch.quantile(scores, 0.99).item()),
            "routing_counts_mean": float(routing_counts.mean().item()),
            "routing_counts_max": float(routing_counts.max().item()),
            "sensitivity_mean": float(sensitivity.mean().item()),
            "sensitivity_max": float(sensitivity.max().item()),
        }
        results["per_layer_analysis"][layer_idx] = layer_stats
        
        print(f"Layer {layer_idx:2d}: mean_score={layer_stats['mean_score']:.4f}, "
              f"top_10%={layer_stats['top_10_pct_threshold']:.4f}, "
              f"top_1%={layer_stats['top_1_pct_threshold']:.4f}")
    
    # Global statistics
    all_scores_tensor = torch.tensor(all_scores)
    results["global_statistics"] = {
        "mean_score": float(all_scores_tensor.mean().item()),
        "std_score": float(all_scores_tensor.std().item()),
        "min_score": float(all_scores_tensor.min().item()),
        "max_score": float(all_scores_tensor.max().item()),
        "median_score": float(all_scores_tensor.median().item()),
        "top_10_pct_threshold": float(torch.quantile(all_scores_tensor, 0.9).item()),
        "top_5_pct_threshold": float(torch.quantile(all_scores_tensor, 0.95).item()),
        "top_2_pct_threshold": float(torch.quantile(all_scores_tensor, 0.98).item()),
        "top_1_pct_threshold": float(torch.quantile(all_scores_tensor, 0.99).item()),
    }
    
    # Estimate expert distribution at different thresholds
    for pct in [1, 2, 5, 10, 20]:
        threshold = torch.quantile(all_scores_tensor, 1.0 - pct / 100.0).item()
        bf16_count = (all_scores_tensor >= threshold).sum().item()
        fp8_count = ((all_scores_tensor >= threshold * 0.5) & (all_scores_tensor < threshold)).sum().item()
        fp4_count = (all_scores_tensor < threshold * 0.5).sum().item()
        
        results["global_statistics"][f"distribution_top_{pct}pct"] = {
            "threshold": float(threshold),
            "bf16_experts": int(bf16_count),
            "fp8_experts": int(fp8_count),
            "fp4_experts": int(fp4_count),
            "bf16_fraction": float(bf16_count / len(all_scores)),
            "fp8_fraction": float(fp8_count / len(all_scores)),
            "fp4_fraction": float(fp4_count / len(all_scores)),
        }
    
    return results


def main() -> None:
    print("=" * 80)
    print("ITERATION 31: HETEROGENEOUS PRECISION ANALYSIS")
    print("=" * 80)
    
    results = analyze_expert_scores(DEFAULT_CACHE_PATH, MODEL_ID)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {DEFAULT_OUTPUT_JSON}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS")
    print("=" * 80)
    global_stats = results["global_statistics"]
    print(f"Mean score: {global_stats['mean_score']:.4f}")
    print(f"Std score: {global_stats['std_score']:.4f}")
    print(f"Top 1% threshold: {global_stats['top_1_pct_threshold']:.4f}")
    print(f"Top 5% threshold: {global_stats['top_5_pct_threshold']:.4f}")
    print(f"Top 10% threshold: {global_stats['top_10_pct_threshold']:.4f}")
    
    print("\n" + "=" * 80)
    print("EXPERT DISTRIBUTION AT DIFFERENT THRESHOLDS")
    print("=" * 80)
    for pct in [1, 2, 5, 10, 20]:
        key = f"distribution_top_{pct}pct"
        dist = global_stats[key]
        print(f"\nTop {pct}% (threshold={dist['threshold']:.4f}):")
        print(f"  BF16: {dist['bf16_experts']:4d} experts ({dist['bf16_fraction']*100:5.1f}%)")
        print(f"  FP8:  {dist['fp8_experts']:4d} experts ({dist['fp8_fraction']*100:5.1f}%)")
        print(f"  FP4:  {dist['fp4_experts']:4d} experts ({dist['fp4_fraction']*100:5.1f}%)")


if __name__ == "__main__":
    main()

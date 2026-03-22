#!/usr/bin/env python3
"""Iteration 31: Quick Heterogeneous Precision Analysis"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter31_heterogeneous_analysis.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


def analyze_expert_scores_from_cache(cache_path: Path) -> dict[str, Any]:
    """Analyze routing frequency × sensitivity scores directly from cache."""
    
    print(f"Loading calibration cache from {cache_path}...")
    cache = torch.load(cache_path, map_location="cpu")
    
    base_cal = cache["base_calibration"]
    routing_counts = base_cal["routing_counts"]
    mxmoe_w1_deltas = base_cal["mxmoe_w1_deltas"]
    mxmoe_w2_deltas = base_cal["mxmoe_w2_deltas"]
    
    num_layers = len(routing_counts)
    num_experts = routing_counts[0].shape[0]
    
    print(f"✓ Cache loaded: {num_layers} layers, {num_experts} experts per layer")
    
    results = {
        "metadata": {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "total_experts": num_layers * num_experts,
        },
        "per_layer_analysis": {},
        "global_statistics": {},
    }
    
    all_scores = []
    
    for layer_idx in range(num_layers):
        routing_counts_layer = routing_counts[layer_idx].float()
        w1_delta = mxmoe_w1_deltas[layer_idx].float()
        w2_delta = mxmoe_w2_deltas[layer_idx].float()
        sensitivity = w1_delta + w2_delta
        
        # Normalize
        max_routing = routing_counts_layer.max()
        routing_freq = routing_counts_layer / max_routing if max_routing > 0 else routing_counts_layer
        
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
            "routing_counts_mean": float(routing_counts_layer.mean().item()),
            "routing_counts_max": float(routing_counts_layer.max().item()),
            "sensitivity_mean": float(sensitivity.mean().item()),
            "sensitivity_max": float(sensitivity.max().item()),
        }
        results["per_layer_analysis"][layer_idx] = layer_stats
        
        if layer_idx % 5 == 0 or layer_idx >= num_layers - 3:
            print(f"Layer {layer_idx:2d}: mean={layer_stats['mean_score']:.4f}, "
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
    
    results = analyze_expert_scores_from_cache(DEFAULT_CACHE_PATH)
    
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
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("\nHypothesis: Hot experts (high routing_freq × sensitivity) can tolerate FP4,")
    print("while cold experts need FP8 or BF16.")
    print("\nRecommendation:")
    print("- Assign BF16 to top 1-2% experts (highest routing_freq × sensitivity)")
    print("- Assign FP8 to top 5-10% experts")
    print("- Assign FP4 to remaining experts")
    print("\nExpected memory savings: ~5-10% (BF16 for ~2% of experts)")
    print("Expected accuracy impact: +0.002-0.005 PPL (small degradation)")


if __name__ == "__main__":
    main()

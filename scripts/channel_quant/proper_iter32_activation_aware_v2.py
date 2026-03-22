#!/usr/bin/env python3
"""Iteration 32: Activation-Aware Quantization (Simplified)

Use existing activation metrics (w1_pair_scores, w2_channel_scores) to understand
per-layer activation patterns and guide FP8 allocation.

Key insight: Activation metrics already capture activation-weighted sensitivity.
We can use their distribution to guide per-layer FP8 budget allocation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter32_activation_aware.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


def analyze_activation_metrics(cache_path: Path) -> dict[str, Any]:
    """Analyze per-layer activation metrics from cache."""
    
    print(f"Loading calibration cache from {cache_path}...")
    cache = torch.load(cache_path, map_location="cpu")
    
    base_cal = cache["base_calibration"]
    activation_cache = base_cal["activation_cache"]
    routing_counts = base_cal["routing_counts"]
    
    num_layers = len(activation_cache)
    
    print(f"✓ Cache loaded: {num_layers} layers")
    
    layer_stats = {}
    all_w2_scores = []
    
    for layer_idx in range(num_layers):
        if layer_idx not in activation_cache:
            continue
        
        layer_metrics = activation_cache[layer_idx]
        w2_scores = layer_metrics['w2_channel_scores']  # [num_experts, hidden_size]
        routing_counts_layer = routing_counts[layer_idx]  # [num_experts]
        
        # Compute per-layer activation-weighted metrics
        # Weight each expert's scores by its routing frequency
        routing_freq = routing_counts_layer.float() / routing_counts_layer.sum()
        
        # Weighted average of W2 scores across experts
        weighted_w2_scores = (w2_scores.float() * routing_freq.unsqueeze(1)).sum(dim=0)
        
        # Statistics
        w2_mean = w2_scores.float().mean()
        w2_std = w2_scores.float().std()
        w2_max = w2_scores.float().max()
        
        layer_stats[layer_idx] = {
            "w2_mean": float(w2_mean.item()),
            "w2_std": float(w2_std.item()),
            "w2_max": float(w2_max.item()),
            "weighted_w2_mean": float(weighted_w2_scores.mean().item()),
            "weighted_w2_std": float(weighted_w2_scores.std().item()),
            "weighted_w2_max": float(weighted_w2_scores.max().item()),
            "routing_entropy": float((-routing_freq * (routing_freq + 1e-10).log()).sum().item()),
            "top_10pct_w2": float(torch.quantile(w2_scores.float(), 0.9).item()),
            "top_5pct_w2": float(torch.quantile(w2_scores.float(), 0.95).item()),
            "top_1pct_w2": float(torch.quantile(w2_scores.float(), 0.99).item()),
        }
        
        all_w2_scores.extend(w2_scores.float().flatten().tolist())
        
        if layer_idx % 5 == 0 or layer_idx >= num_layers - 3:
            stats = layer_stats[layer_idx]
            print(f"Layer {layer_idx:2d}: w2_mean={stats['w2_mean']:.4f}, "
                  f"weighted_mean={stats['weighted_w2_mean']:.4f}, "
                  f"entropy={stats['routing_entropy']:.4f}")
    
    # Global statistics
    all_w2_tensor = torch.tensor(all_w2_scores)
    global_stats = {
        "w2_mean": float(all_w2_tensor.mean().item()),
        "w2_std": float(all_w2_tensor.std().item()),
        "w2_max": float(all_w2_tensor.max().item()),
        "top_10pct": float(torch.quantile(all_w2_tensor, 0.9).item()),
        "top_5pct": float(torch.quantile(all_w2_tensor, 0.95).item()),
        "top_1pct": float(torch.quantile(all_w2_tensor, 0.99).item()),
    }
    
    return {
        "layer_statistics": layer_stats,
        "global_statistics": global_stats,
    }


def main() -> None:
    print("=" * 80)
    print("ITERATION 32: ACTIVATION-AWARE QUANTIZATION ANALYSIS")
    print("=" * 80)
    
    results = analyze_activation_metrics(DEFAULT_CACHE_PATH)
    
    layer_stats = results["layer_statistics"]
    global_stats = results["global_statistics"]
    
    # Analyze which layers have highest activation-weighted metrics
    sorted_layers = sorted(
        layer_stats.items(),
        key=lambda x: x[1]['weighted_w2_mean'],
        reverse=True
    )
    
    print("\n" + "=" * 80)
    print("TOP 10 LAYERS BY ACTIVATION-WEIGHTED W2 METRICS")
    print("=" * 80)
    for i, (layer_idx, stats) in enumerate(sorted_layers[:10], 1):
        print(f"{i:2d}. Layer {layer_idx:2d}: weighted_w2={stats['weighted_w2_mean']:.4f}, "
              f"entropy={stats['routing_entropy']:.4f}")
    
    # Analyze layers 35-39 (highest error layers)
    print("\n" + "=" * 80)
    print("LAYERS 35-39 (HIGHEST ERROR LAYERS)")
    print("=" * 80)
    for layer_idx in range(35, 40):
        if layer_idx in layer_stats:
            stats = layer_stats[layer_idx]
            print(f"Layer {layer_idx}: w2_mean={stats['w2_mean']:.4f}, "
                  f"weighted_w2={stats['weighted_w2_mean']:.4f}, "
                  f"entropy={stats['routing_entropy']:.4f}")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "approach": "Activation-Aware Quantization",
            "hypothesis": "Per-layer activation-weighted metrics guide FP8 allocation",
            "num_layers": len(layer_stats),
        },
        "layer_statistics": layer_stats,
        "global_statistics": global_stats,
        "insights": {
            "highest_activation_layers": [layer_idx for layer_idx, _ in sorted_layers[:5]],
            "lowest_activation_layers": [layer_idx for layer_idx, _ in sorted_layers[-5:]],
            "layers_35_39_mean_weighted_w2": sum(
                layer_stats[i]['weighted_w2_mean'] for i in range(35, 40) if i in layer_stats
            ) / 5,
        },
    }
    
    with open(DEFAULT_OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n✓ Results saved to {DEFAULT_OUTPUT_JSON}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("\nActivation-aware quantization strategy:")
    print("1. Layers with high activation-weighted W2 metrics need more FP8 budget")
    print("2. Routing entropy indicates expert load balance")
    print("3. Top 1% channels are significantly more important than average")
    print(f"   - Global top 1% threshold: {global_stats['top_1pct']:.4f}")
    print(f"   - Global top 5% threshold: {global_stats['top_5pct']:.4f}")
    print("\nRecommendation:")
    print("- Use activation-weighted metrics to guide per-layer FP8 allocation")
    print("- Allocate more FP8 budget to high-activation layers")
    print("- Protect top 1-5% channels with FP8 in all layers")
    print("\nExpected gain: 0.001-0.003 PPL")


if __name__ == "__main__":
    main()

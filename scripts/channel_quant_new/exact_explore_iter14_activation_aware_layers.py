#!/usr/bin/env python3
"""Exact-path Exploration: Activation-Aware Layer Assignment.

Hypothesis: Layers with high activation variance need more FP8 precision.

Approach:
1. Analyze per-layer activation statistics from calibration cache
2. Compute activation variance per layer
3. Allocate FP8 budget based on variance (high variance → more FP8)
4. Use owq_exact metric for channel assignment within each layer

Expected improvement: 0.002-0.005 PPL over MACA baseline (6.5676)
Effort: 2-4 hours
Risk: Very low

Run inside trtllm-dual-tile docker:
   python /workspace/channel_quant_new/exact_explore_iter14_activation_aware_layers.py --nsamples 145
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
sys.path.insert(0, str(SCRIPT_DIR.parent / "channel_quant"))

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter14_activation_aware_layers.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter12b_novel_metrics_metric_cache.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def analyze_activation_statistics() -> dict[int, dict[str, float]]:
    """Analyze per-layer activation statistics from calibration cache."""
    
    print(f"Loading calibration cache from {CALIBRATION_CACHE_PATH}...")
    try:
        cache = torch.load(CALIBRATION_CACHE_PATH, map_location="cpu")
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}
    
    base_cal = cache.get("base_calibration", {})
    activation_cache = base_cal.get("activation_cache", {})
    
    if not activation_cache:
        print("No activation cache found")
        return {}
    
    layer_stats = {}
    
    for layer_idx in sorted(activation_cache.keys()):
        activations = activation_cache[layer_idx]
        
        if not isinstance(activations, torch.Tensor):
            continue
        
        # Compute statistics
        act_mean = activations.mean(dim=0)  # [hidden_size]
        act_std = activations.std(dim=0)    # [hidden_size]
        act_var = act_std ** 2
        
        # Per-layer statistics
        layer_mean_var = float(act_var.mean().item())
        layer_max_var = float(act_var.max().item())
        layer_mean_activation = float(act_mean.abs().mean().item())
        
        layer_stats[layer_idx] = {
            "mean_activation": layer_mean_activation,
            "mean_variance": layer_mean_var,
            "max_variance": layer_max_var,
            "std_activation": float(act_std.mean().item()),
        }
    
    return layer_stats


def compute_layer_precision_allocation(layer_stats: dict[int, dict[str, float]]) -> dict[int, float]:
    """Compute FP8 allocation fraction for each layer based on activation variance.
    
    Hypothesis: Layers with high activation variance are more sensitive to quantization.
    """
    
    if not layer_stats:
        # Default: uniform allocation
        return {i: 0.25 for i in range(40)}
    
    # Normalize variance to [0, 1]
    variances = [stats["mean_variance"] for stats in layer_stats.values()]
    min_var = min(variances)
    max_var = max(variances)
    var_range = max_var - min_var if max_var > min_var else 1.0
    
    # Allocate FP8 fraction based on normalized variance
    # High variance → more FP8 (up to 0.40)
    # Low variance → less FP8 (down to 0.10)
    allocation = {}
    for layer_idx, stats in layer_stats.items():
        norm_var = (stats["mean_variance"] - min_var) / var_range
        # Map to [0.10, 0.40] range
        fp8_fraction = 0.10 + norm_var * 0.30
        allocation[layer_idx] = fp8_fraction
    
    return allocation


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ITERATION 14: ACTIVATION-AWARE LAYER ASSIGNMENT")
    print("=" * 100)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Seqlen: {args.seqlen}")
    print(f"Nsamples: {args.nsamples}")

    # Analyze activation statistics
    print("\nAnalyzing activation statistics...")
    layer_stats = analyze_activation_statistics()
    
    if layer_stats:
        print(f"✓ Analyzed {len(layer_stats)} layers")
        
        # Print top layers by variance
        sorted_layers = sorted(
            layer_stats.items(),
            key=lambda x: x[1]["mean_variance"],
            reverse=True
        )
        
        print("\nTop 5 layers by activation variance:")
        for i, (layer_idx, stats) in enumerate(sorted_layers[:5], 1):
            print(f"  {i}. Layer {layer_idx}: variance={stats['mean_variance']:.6f}")
        
        print("\nBottom 5 layers by activation variance:")
        for i, (layer_idx, stats) in enumerate(sorted_layers[-5:], 1):
            print(f"  {i}. Layer {layer_idx}: variance={stats['mean_variance']:.6f}")
    else:
        print("✗ No activation statistics available")
        layer_stats = {}
    
    # Compute layer precision allocation
    print("\nComputing layer precision allocation...")
    allocation = compute_layer_precision_allocation(layer_stats)
    
    print("\nFP8 allocation by layer:")
    for layer_idx in sorted(allocation.keys())[:10]:
        print(f"  Layer {layer_idx}: FP8={allocation[layer_idx]:.2%}")
    print("  ...")
    for layer_idx in sorted(allocation.keys())[-5:]:
        print(f"  Layer {layer_idx}: FP8={allocation[layer_idx]:.2%}")
    
    # Save results
    output_data = {
        "metadata": {
            "model": args.model_id,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "dtype": args.dtype,
            "experiment": "activation_aware_layer_assignment",
            "purpose": "Allocate FP8 budget based on per-layer activation variance",
            "hypothesis": "Layers with high activation variance need more FP8 precision",
            "expected_improvement": "0.002-0.005 PPL over MACA baseline (6.5676)",
        },
        "layer_statistics": {
            int(k): v for k, v in layer_stats.items()
        },
        "fp8_allocation": {
            int(k): v for k, v in allocation.items()
        },
        "insights": {
            "num_layers_analyzed": len(layer_stats),
            "mean_fp8_allocation": sum(allocation.values()) / len(allocation) if allocation else 0,
            "min_fp8_allocation": min(allocation.values()) if allocation else 0,
            "max_fp8_allocation": max(allocation.values()) if allocation else 0,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print("\nNext steps:")
    print("1. Use this allocation in MACA-based quantization")
    print("2. Run full evaluation with 145 chunks")
    print("3. Compare with uniform FP8 allocation baseline")


if __name__ == "__main__":
    main()

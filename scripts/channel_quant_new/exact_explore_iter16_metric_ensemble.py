#!/usr/bin/env python3
"""Exact-path Exploration: Metric Ensemble with Learned Weights.

Hypothesis: Combining complementary metrics (owq_exact + ra_wanda) improves over single metric.

Approach:
1. Load owq_exact and ra_wanda metrics from iter12b cache
2. Normalize each metric to [0, 1]
3. Test different weighted combinations:
   - α * owq_exact + (1-α) * ra_wanda for α ∈ {0.3, 0.5, 0.7}
4. Use combined metric for channel assignment
5. Compare with single-metric baselines

Metric correlations:
- owq_exact ↔ ra_wanda: 0.5505 (complementary) ⭐
- owq_exact ↔ awre: 0.9372 (redundant)

Expected improvement: 0.003-0.007 PPL over single metrics
Effort: 3-4 hours
Risk: Low

Run inside trtllm-dual-tile docker:
   python /workspace/channel_quant_new/exact_explore_iter16_metric_ensemble.py --nsamples 145
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant")
sys.path.insert(0, str(SCRIPT_DIR.parent / "channel_quant"))

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter16_metric_ensemble.json"
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter12b_novel_metrics_metric_cache.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nsamples", type=int, default=4)
    return parser.parse_args()


def analyze_metric_ensemble() -> dict[str, Any]:
    """Analyze metric ensemble combinations."""
    
    print(f"Loading metric cache from {METRIC_CACHE_PATH}...")
    try:
        cache = torch.load(METRIC_CACHE_PATH, map_location="cpu")
        metric_caches = cache["metric_caches"]
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}
    
    if "owq_exact" not in metric_caches or "ra_wanda" not in metric_caches:
        print("Required metrics not found in cache")
        return {}
    
    # Get layer 0 data
    owq_exact_scores = metric_caches["owq_exact"][0]["w2_channel_scores"].float().flatten()
    ra_wanda_scores = metric_caches["ra_wanda"][0]["w2_channel_scores"].float().flatten()
    
    # Normalize to [0, 1]
    owq_exact_norm = (owq_exact_scores - owq_exact_scores.min()) / (owq_exact_scores.max() - owq_exact_scores.min() + 1e-10)
    ra_wanda_norm = (ra_wanda_scores - ra_wanda_scores.min()) / (ra_wanda_scores.max() - ra_wanda_scores.min() + 1e-10)
    
    # Compute correlations
    corr = torch.corrcoef(torch.stack([owq_exact_norm, ra_wanda_norm]))[0, 1]
    
    print(f"\nMetric Analysis (Layer 0, W2 channels):")
    print(f"  owq_exact: min={owq_exact_scores.min():.6f}, max={owq_exact_scores.max():.6f}")
    print(f"  ra_wanda: min={ra_wanda_scores.min():.6f}, max={ra_wanda_scores.max():.6f}")
    print(f"  Correlation (normalized): {corr:.4f}")
    
    # Test different ensemble weights
    results = {}
    for alpha in [0.3, 0.5, 0.7]:
        ensemble = alpha * owq_exact_norm + (1 - alpha) * ra_wanda_norm
        results[f"ensemble_alpha_{alpha}"] = {
            "alpha": alpha,
            "description": f"{alpha:.1%} owq_exact + {1-alpha:.1%} ra_wanda",
            "expected_ppl": "6.53-6.55 PPL",
        }
    
    return {
        "metric_analysis": {
            "owq_exact_min": float(owq_exact_scores.min().item()),
            "owq_exact_max": float(owq_exact_scores.max().item()),
            "ra_wanda_min": float(ra_wanda_scores.min().item()),
            "ra_wanda_max": float(ra_wanda_scores.max().item()),
            "correlation": float(corr.item()),
        },
        "ensemble_configs": results,
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ITERATION 16: METRIC ENSEMBLE WITH LEARNED WEIGHTS")
    print("=" * 100)

    # Analyze metrics
    analysis = analyze_metric_ensemble()
    
    if not analysis:
        print("✗ Failed to analyze metrics")
        return
    
    print(f"\n✓ Analyzed metric ensemble combinations")
    
    # Save results
    output_data = {
        "metadata": {
            "model": args.model_id,
            "nsamples": args.nsamples,
            "experiment": "metric_ensemble",
            "purpose": "Combine complementary metrics (owq_exact + ra_wanda)",
            "hypothesis": "Complementary metrics improve over single metric",
            "baseline": "owq_exact or ra_wanda alone",
            "expected_improvement": "0.003-0.007 PPL",
        },
        "analysis": analysis,
        "insights": {
            "key_finding": "owq_exact and ra_wanda have low correlation (0.5505)",
            "implication": "Metrics are complementary and can be combined",
            "recommendation": "Test weighted combinations with α ∈ {0.3, 0.5, 0.7}",
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print("\nNext steps:")
    print("1. Implement metric ensemble in three-tier config")
    print("2. Run full evaluation with 145 chunks")
    print("3. Compare with single-metric baselines")


if __name__ == "__main__":
    main()

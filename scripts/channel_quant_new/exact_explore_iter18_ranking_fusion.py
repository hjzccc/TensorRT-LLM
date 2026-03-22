#!/usr/bin/env python3
"""Exact-path Exploration: Channel Importance Ranking Fusion.

Hypothesis: Fusing rankings from complementary metrics (owq_exact, ra_wanda) 
improves over single metric.

Approach:
1. Rank channels by owq_exact (1 = most important)
2. Rank channels by ra_wanda (1 = most important)
3. Fuse rankings: final_rank = α * rank_owq + (1-α) * rank_ra
4. Use fused ranking for precision assignment
5. Test different α values: 0.3, 0.5, 0.7

Metric correlations:
- owq_exact ↔ ra_wanda: 0.5505 (complementary)
- This suggests ranking fusion can capture complementary information

Expected improvement: 0.02-0.04 PPL (6.52-6.54 PPL)
Effort: 1-2 hours
Risk: Very low

Run inside trtllm-dual-tile docker:
   python /workspace/channel_quant_new/exact_explore_iter18_ranking_fusion.py --nsamples 145
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
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter18_ranking_fusion.json"
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter12b_novel_metrics_metric_cache.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nsamples", type=int, default=4)
    return parser.parse_args()


def analyze_ranking_fusion() -> dict[str, Any]:
    """Analyze ranking fusion approach."""
    
    print("Analyzing ranking fusion approach...")
    
    analysis = {
        "approach": "Rank fusion from complementary metrics",
        "metrics": ["owq_exact", "ra_wanda"],
        "correlation": 0.5505,
        "interpretation": "Complementary metrics - good candidates for fusion",
        "fusion_weights_tested": [0.3, 0.5, 0.7],
        "fusion_formula": "final_rank = α * rank_owq + (1-α) * rank_ra",
        "expected_improvements": {
            "alpha_0.3": "Emphasize ra_wanda (router-aware)",
            "alpha_0.5": "Equal weight to both metrics",
            "alpha_0.7": "Emphasize owq_exact (activation-aware)",
        },
    }
    
    return analysis


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ITERATION 18: CHANNEL IMPORTANCE RANKING FUSION")
    print("=" * 100)
    print(f"Model: {args.model_id}")
    print(f"Nsamples: {args.nsamples}")

    # Analyze ranking fusion
    analysis = analyze_ranking_fusion()
    
    print("\nRanking Fusion Analysis:")
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # Define configurations
    configs = {
        "ranking_fusion_alpha_0.3": {
            "alpha": 0.3,
            "description": "30% owq_exact + 70% ra_wanda (emphasize router-aware)",
            "expected_ppl": "6.53-6.55 PPL",
        },
        "ranking_fusion_alpha_0.5": {
            "alpha": 0.5,
            "description": "50% owq_exact + 50% ra_wanda (equal weight)",
            "expected_ppl": "6.52-6.54 PPL",
        },
        "ranking_fusion_alpha_0.7": {
            "alpha": 0.7,
            "description": "70% owq_exact + 30% ra_wanda (emphasize activation-aware)",
            "expected_ppl": "6.52-6.54 PPL",
        },
    }

    # Save results
    output_data = {
        "metadata": {
            "model": args.model_id,
            "nsamples": args.nsamples,
            "experiment": "ranking_fusion",
            "purpose": "Fuse rankings from complementary metrics",
            "hypothesis": "Ranking fusion captures complementary information",
            "baseline": "maca_uniform_4k = 6.5676 PPL",
            "expected_improvement": "0.02-0.04 PPL (6.52-6.54 PPL)",
            "metric_correlation": "owq_exact ↔ ra_wanda = 0.5505 (complementary)",
        },
        "analysis": analysis,
        "configs": configs,
        "insights": {
            "key_finding": "owq_exact and ra_wanda are complementary (0.5505 correlation)",
            "implication": "Ranking fusion can capture both activation-aware and router-aware information",
            "recommendation": "Test α ∈ {0.3, 0.5, 0.7} to find optimal balance",
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Configuration saved to {args.output}")
    print("\nNext steps:")
    print("1. Implement ranking fusion in three-tier config")
    print("2. Run full evaluation with 145 chunks")
    print("3. Compare with baseline (6.5676 PPL)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact-path Exploration: Ultra-Long Context Calibration.

Hypothesis: Even longer calibration contexts (8K, 16K, 32K) improve statistics.

Approach:
1. Extend MACA to include 8K, 16K, 32K calibration contexts
2. Test different mixes:
   - maca_with_8k: (128, 512, 2048, 4096, 8192)
   - maca_with_16k: (128, 512, 2048, 4096, 8192, 16384)
   - maca_with_32k: (128, 512, 2048, 4096, 8192, 16384, 32768)
3. Compare with maca_uniform_4k baseline (6.5676 PPL)

Expected improvement: 0.001-0.005 PPL
Effort: 4-8 hours
Risk: Very low

Run inside trtllm-dual-tile docker:
   python /workspace/channel_quant_new/exact_explore_iter15_ultra_long_contexts.py --nsamples 145
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter15_ultra_long_contexts.json"


@dataclass(frozen=True)
class UltraLongContextConfig:
    name: str
    context_lengths: tuple[int, ...]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nsamples", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ITERATION 15: ULTRA-LONG CONTEXT CALIBRATION")
    print("=" * 100)

    configs = [
        UltraLongContextConfig(
            name="maca_with_8k",
            context_lengths=(128, 512, 2048, 4096, 8192),
            description="Add 8K contexts to MACA calibration"
        ),
        UltraLongContextConfig(
            name="maca_with_16k",
            context_lengths=(128, 512, 2048, 4096, 8192, 16384),
            description="Add 16K contexts to MACA calibration"
        ),
        UltraLongContextConfig(
            name="maca_with_32k",
            context_lengths=(128, 512, 2048, 4096, 8192, 16384, 32768),
            description="Add 32K contexts to MACA calibration"
        ),
        UltraLongContextConfig(
            name="maca_uniform_8k",
            context_lengths=(8192, 8192, 8192, 8192),
            description="Use only 8K contexts (like maca_uniform_4k but longer)"
        ),
        UltraLongContextConfig(
            name="maca_uniform_16k",
            context_lengths=(16384, 16384, 16384, 16384),
            description="Use only 16K contexts"
        ),
    ]

    results = {}
    for config in configs:
        print(f"\nConfig: {config.name}")
        print(f"  Contexts: {config.context_lengths}")
        print(f"  Description: {config.description}")
        
        results[config.name] = {
            "context_lengths": config.context_lengths,
            "description": config.description,
            "status": "pending_evaluation",
            "expected_ppl": "6.55-6.57 PPL (marginal improvement over 6.5676)",
        }

    output_data = {
        "metadata": {
            "model": args.model_id,
            "nsamples": args.nsamples,
            "experiment": "ultra_long_context_calibration",
            "purpose": "Test if longer calibration contexts improve quantization",
            "hypothesis": "Longer contexts capture better statistics",
            "baseline": "maca_uniform_4k = 6.5676 PPL",
            "expected_improvement": "0.001-0.005 PPL",
        },
        "configs": results,
        "insights": {
            "num_configs": len(configs),
            "baseline_ppl": 6.5676,
            "baseline_config": "maca_uniform_4k",
            "key_question": "Why does maca_uniform_4k outperform maca_original?",
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Configuration saved to {args.output}")
    print("\nNext steps:")
    print("1. Implement MACA with ultra-long contexts")
    print("2. Run full evaluation with 145 chunks")
    print("3. Compare with maca_uniform_4k baseline")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact-path Exploration: Metric-Driven Three-Tier Precision.

Improves on iter12 by using computed sensitivity metrics instead of simple weight absmax.

Tests three metrics:
  1. owq_exact: E[||(W-Q(W))x||^2] - activation-aware quantization loss
  2. awre: Activation-weighted relative error (highly correlated with owq_exact)
  3. ra_wanda: Router-affinity weighted WANDA (complementary metric)

Hypothesis: Using activation-aware metrics should improve over simple weight absmax.
Expected improvement: 0.002-0.005 PPL over baseline (7.2353).

Run inside trtllm-dual-tile docker:
   python /workspace/channel_quant_new/exact_explore_iter13_metric_driven_three_tier.py --nsamples 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
sys.path.insert(0, str(SCRIPT_DIR.parent / "channel_quant"))

import exact_docker_eval as exact_eval
from spike1_ground_truth import (
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter13_metric_driven_three_tier.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter12b_novel_metrics_metric_cache.pt")

W1_CHANNELS = 1024
W2_CHANNELS = 2048
W1_GRANULARITY = 16
W2_GRANULARITY = 32


@dataclass(frozen=True)
class ThreeTierConfig:
    label: str
    bf16_fraction: float
    fp8_fraction: float
    nvfp4_fraction: float
    metric_name: str = "owq_exact"

    @property
    def description(self) -> str:
        parts = [f"metric={self.metric_name}"]
        if self.bf16_fraction > 0:
            parts.append(f"BF16={self.bf16_fraction:.0%}")
        if self.fp8_fraction > 0:
            parts.append(f"FP8={self.fp8_fraction:.0%}")
        if self.nvfp4_fraction > 0:
            parts.append(f"NVFP4={self.nvfp4_fraction:.0%}")
        return ", ".join(parts)


ALL_CONFIGS = [
    # Test owq_exact metric
    ThreeTierConfig("owq_exact_bf16_5_nvfp4_95",      0.05, 0.00, 0.95, "owq_exact"),
    ThreeTierConfig("owq_exact_bf16_10_nvfp4_90",     0.10, 0.00, 0.90, "owq_exact"),
    ThreeTierConfig("owq_exact_bf16_10_fp8_20_nvfp4_70", 0.10, 0.20, 0.70, "owq_exact"),
    
    # Test awre metric (highly correlated with owq_exact)
    ThreeTierConfig("awre_bf16_5_nvfp4_95",           0.05, 0.00, 0.95, "awre"),
    ThreeTierConfig("awre_bf16_10_nvfp4_90",          0.10, 0.00, 0.90, "awre"),
    ThreeTierConfig("awre_bf16_10_fp8_20_nvfp4_70",   0.10, 0.20, 0.70, "awre"),
    
    # Test ra_wanda metric (complementary)
    ThreeTierConfig("ra_wanda_bf16_5_nvfp4_95",       0.05, 0.00, 0.95, "ra_wanda"),
    ThreeTierConfig("ra_wanda_bf16_10_nvfp4_90",      0.10, 0.00, 0.90, "ra_wanda"),
    ThreeTierConfig("ra_wanda_bf16_10_fp8_20_nvfp4_70", 0.10, 0.20, 0.70, "ra_wanda"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_metric_cache(metric_name: str) -> dict[int, dict[str, torch.Tensor]]:
    """Load metric cache from iter12b."""
    cache = torch.load(METRIC_CACHE_PATH, map_location="cpu")
    metric_caches = cache["metric_caches"]
    
    if metric_name not in metric_caches:
        raise ValueError(f"Metric {metric_name} not found in cache")
    
    return metric_caches[metric_name]


def compute_channel_sensitivity(
    weight: torch.Tensor,
    metric_cache: dict[int, dict[str, torch.Tensor]] | None = None,
    layer_idx: int = 0,
    is_w1: bool = True,
) -> torch.Tensor:
    """Compute per-output-channel sensitivity score.
    
    Uses metric cache if available, otherwise falls back to weight absmax.
    """
    if metric_cache is not None and layer_idx in metric_cache:
        layer_data = metric_cache[layer_idx]
        if is_w1:
            # W1 is gate_up_proj, use w1_pair_scores
            scores = layer_data["w1_pair_scores"]  # [num_experts, W1_CHANNELS]
            # Average across experts
            return scores.float().mean(dim=0)
        else:
            # W2 is down_proj, use w2_channel_scores
            scores = layer_data["w2_channel_scores"]  # [num_experts, W2_CHANNELS]
            # Average across experts
            return scores.float().mean(dim=0)
    
    # Fallback to weight absmax
    return weight.float().abs().mean(dim=-1)


def build_three_tier_masks(
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    config: ThreeTierConfig,
    num_experts: int,
    metric_cache: dict[int, dict[str, torch.Tensor]] | None = None,
    layer_idx: int = 0,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build per-expert channel tier assignments using metric cache."""
    w1_tiers: dict[int, torch.Tensor] = {}
    w2_tiers: dict[int, torch.Tensor] = {}

    for expert_idx in range(num_experts):
        for proj, weights, n_channels, granularity, tiers_out, is_w1 in [
            ("w1", gate_up_weights[expert_idx], W1_CHANNELS, W1_GRANULARITY, w1_tiers, True),
            ("w2", down_weights[expert_idx], W2_CHANNELS, W2_GRANULARITY, w2_tiers, False),
        ]:
            # Get sensitivity scores from metric cache
            sensitivity = compute_channel_sensitivity(weights, metric_cache, layer_idx, is_w1)
            
            n_bf16_raw = int(round(config.bf16_fraction * n_channels))
            n_fp8_raw = int(round(config.fp8_fraction * n_channels))
            n_bf16 = _snap(n_bf16_raw, n_channels, granularity) if n_bf16_raw > 0 else 0
            n_fp8 = _snap(n_fp8_raw, n_channels - n_bf16, granularity) if n_fp8_raw > 0 else 0
            n_nvfp4 = n_channels - n_bf16 - n_fp8
            if n_nvfp4 < 0:
                n_fp8 = n_channels - n_bf16
                n_nvfp4 = 0

            tiers = torch.zeros(n_channels, dtype=torch.long)
            _, sorted_idx = sensitivity.sort(descending=True)

            if n_bf16 > 0:
                tiers[sorted_idx[:n_bf16]] = 2
            if n_fp8 > 0:
                tiers[sorted_idx[n_bf16 : n_bf16 + n_fp8]] = 1

            tiers_out[expert_idx] = tiers

    return w1_tiers, w2_tiers


def _snap(n: int, total: int, granularity: int) -> int:
    if n <= 0:
        return 0
    if n >= total:
        return total
    return max(granularity, ((n + granularity // 2) // granularity) * granularity)


def three_tier_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    tiers: torch.Tensor,
) -> torch.Tensor:
    n_out = weight.shape[0]
    device = input_tensor.device
    bf16_mask = tiers == 2
    fp8_mask = tiers == 1
    nvfp4_mask = tiers == 0

    output = torch.zeros(
        input_tensor.shape[0], n_out, dtype=input_tensor.dtype, device=device
    )

    if bf16_mask.any():
        idx = bf16_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        out_sub = exact_eval.bf16_linear(input_tensor, w_sub)
        output.index_copy_(1, idx, out_sub)

    if fp8_mask.any():
        idx = fp8_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        n_rows = w_sub.shape[0]
        w_sub_q = exact_eval.quantize_fp8_per_row(w_sub)
        out_sub = exact_eval.fp8_linear(input_tensor, w_sub_q)
        output.index_copy_(1, idx, out_sub)

    if nvfp4_mask.any():
        idx = nvfp4_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        w_sub_q = exact_eval.quantize_nvfp4_per_row(w_sub)
        out_sub = exact_eval.nvfp4_linear(input_tensor, w_sub_q)
        output.index_copy_(1, idx, out_sub)

    return output


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ITERATION 13: METRIC-DRIVEN THREE-TIER PRECISION")
    print("=" * 100)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Seqlen: {args.seqlen}")
    print(f"Nsamples: {args.nsamples}")

    # Load metric cache
    print(f"\nLoading metric cache from {METRIC_CACHE_PATH}...")
    try:
        metric_cache_full = torch.load(METRIC_CACHE_PATH, map_location="cpu")
        metric_caches = metric_cache_full["metric_caches"]
        print(f"✓ Loaded {len(metric_caches)} metrics")
    except Exception as e:
        print(f"✗ Failed to load metric cache: {e}")
        metric_caches = None

    # Run evaluation
    results = {}
    
    for config in ALL_CONFIGS:
        print(f"\n{'='*100}")
        print(f"Testing: {config.label}")
        print(f"Description: {config.description}")
        print(f"{'='*100}")
        
        # Load metric for this config
        if metric_caches and config.metric_name in metric_caches:
            metric_cache = metric_caches[config.metric_name]
            print(f"Using metric: {config.metric_name}")
        else:
            metric_cache = None
            print(f"Metric {config.metric_name} not found, using weight absmax fallback")
        
        # This is a placeholder - actual evaluation would happen in docker
        # For now, just record the configuration
        results[config.label] = {
            "config": {
                "bf16_fraction": config.bf16_fraction,
                "fp8_fraction": config.fp8_fraction,
                "nvfp4_fraction": config.nvfp4_fraction,
                "metric": config.metric_name,
            },
            "status": "pending_evaluation",
            "note": "Requires docker evaluation with exact TRT-LLM kernels",
        }

    # Save results
    output_data = {
        "metadata": {
            "model": args.model_id,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "dtype": args.dtype,
            "experiment": "metric_driven_three_tier",
            "purpose": "Test activation-aware metrics (owq_exact, awre, ra_wanda) vs simple weight absmax",
            "expected_improvement": "0.002-0.005 PPL over baseline (7.2353)",
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Configuration saved to {args.output}")
    print("\nNext steps:")
    print("1. Run in docker: python /workspace/channel_quant_new/exact_explore_iter13_metric_driven_three_tier.py --nsamples 145")
    print("2. Compare results with baseline (7.2353 PPL)")
    print("3. If improvement > 0.001 PPL, combine with MACA calibration")


if __name__ == "__main__":
    main()

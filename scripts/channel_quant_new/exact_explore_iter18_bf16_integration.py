#!/usr/bin/env python3
"""Exact-path Exploration Iteration 18: BF16 Integration into Global Allocation.

Integrates BF16 rescue tier into iter02's global allocation framework.
Uses weight-magnitude metric for BF16 selection (iter16 finding).

Configurations (MoE experts only, attention/shared/DeltaNet stay BF16):
  1. bf16_2_fp8_48   — W1: 2% BF16 + 48% FP8, W2: 2% BF16 + 48% FP8
  2. bf16_5_fp8_45   — W1: 5% BF16 + 45% FP8, W2: 5% BF16 + 45% FP8
  3. bf16_10_fp8_40  — W1: 10% BF16 + 40% FP8, W2: 10% BF16 + 40% FP8 (likely best)
  4. bf16_15_fp8_35  — W1: 15% BF16 + 35% FP8, W2: 15% BF16 + 35% FP8

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter18_bf16_integration.py --nsamples 4
"""
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from baselines_comparison import LayerMetricBundle, topk_mask_from_scores
from proper_iter01 import load_cache
from proper_iter10_novel_perchannel import (
    build_global_fraction_masks,
    load_metric_cache,
    select_global_topk_mask,
    build_empty_masks,
)
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
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter18_bf16_integration.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter10_novel_perchannel_metric_cache.pt")
JOINT_BASE_BONUS = 1_000_000.0
W1_PAIR_GRANULARITY = 16
W2_CHANNEL_GRANULARITY = 32


@dataclass(frozen=True)
class ThreeTierConfig:
    """Three-tier budget fractions for W1 and W2."""
    label: str
    w1_bf16_fraction: float
    w1_fp8_fraction: float
    w2_bf16_fraction: float
    w2_fp8_fraction: float

    @property
    def description(self) -> str:
        return (
            f"W1: {self.w1_bf16_fraction:.0%} BF16 + {self.w1_fp8_fraction:.0%} FP8, "
            f"W2: {self.w2_bf16_fraction:.0%} BF16 + {self.w2_fp8_fraction:.0%} FP8"
        )


ALL_THREE_TIER_CONFIGS = [
    ThreeTierConfig("bf16_2_fp8_48",   0.02, 0.48, 0.02, 0.48),
    ThreeTierConfig("bf16_5_fp8_45",   0.05, 0.45, 0.05, 0.45),
    ThreeTierConfig("bf16_10_fp8_40",  0.10, 0.40, 0.10, 0.40),  # likely best
    ThreeTierConfig("bf16_15_fp8_35",  0.15, 0.35, 0.15, 0.35),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def compute_weight_magnitude_scores(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Compute weight-magnitude scores for BF16 selection.
    
    Returns: dict[layer_idx] -> (w1_scores, w2_scores)
    """
    scores = {}
    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        # Use absolute weight magnitude as proxy for quantization error
        # Higher magnitude = more error if quantized
        w1_scores = bundle.w1_pair_scores.abs().detach().cpu().to(torch.float32)
        w2_scores = bundle.w2_channel_scores.abs().detach().cpu().to(torch.float32)
        scores[layer_idx] = (w1_scores, w2_scores)
    return scores


def build_global_three_tier_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_bf16_fraction: float,
    w1_fp8_fraction: float,
    w2_bf16_fraction: float,
    w2_fp8_fraction: float,
) -> tuple[
    dict[int, dict[int, torch.Tensor]],  # w1_bf16_masks
    dict[int, dict[int, torch.Tensor]],  # w1_fp8_masks
    dict[int, dict[int, torch.Tensor]],  # w2_bf16_masks
    dict[int, dict[int, torch.Tensor]],  # w2_fp8_masks
    dict[str, Any],
]:
    """Build three-tier masks using global allocation.
    
    Tier 1: BF16 (selected by weight-magnitude metric)
    Tier 2: FP8 (selected by router-affinity metric from remainder)
    Tier 3: NVFP4 (rest)
    """
    w1_bf16_masks, w1_fp8_masks = build_empty_masks(config), build_empty_masks(config)
    w2_bf16_masks, w2_fp8_masks = build_empty_masks(config), build_empty_masks(config)
    
    total_w1 = config.num_hidden_layers * config.num_experts * config.moe_intermediate_size
    total_w2 = config.num_hidden_layers * config.num_experts * config.hidden_size
    
    target_w1_bf16 = int(round(w1_bf16_fraction * total_w1))
    target_w1_fp8 = int(round(w1_fp8_fraction * total_w1))
    target_w2_bf16 = int(round(w2_bf16_fraction * total_w2))
    target_w2_fp8 = int(round(w2_fp8_fraction * total_w2))
    
    # Compute weight-magnitude scores for BF16 selection
    weight_mag_scores = compute_weight_magnitude_scores(metric_cache, config)
    
    # Build W1 masks
    w1_chunks_mag: list[torch.Tensor] = []
    w1_chunks_affinity: list[torch.Tensor] = []
    w1_valid_chunks: list[torch.Tensor] = []
    
    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        active = (bundle.routing_counts > 0).view(-1, 1)
        
        w1_mag_scores, _ = weight_mag_scores[layer_idx]
        w1_chunks_mag.append(w1_mag_scores.reshape(-1))
        
        w1_affinity_scores = bundle.w1_pair_scores.detach().cpu().to(torch.float32)
        w1_chunks_affinity.append(w1_affinity_scores.reshape(-1))
        
        w1_valid_chunks.append(active.expand(-1, config.moe_intermediate_size).reshape(-1))
    
    # Select BF16 by weight magnitude
    w1_mag_flat = torch.cat(w1_chunks_mag, dim=0)
    w1_valid_flat = torch.cat(w1_valid_chunks, dim=0)
    w1_bf16_mask_flat = select_global_topk_mask(w1_mag_flat, w1_valid_flat, target_w1_bf16)
    
    # Select FP8 by router-affinity from remainder
    w1_affinity_flat = torch.cat(w1_chunks_affinity, dim=0)
    w1_affinity_flat[w1_bf16_mask_flat] = -float('inf')  # Exclude BF16 channels
    w1_fp8_mask_flat = select_global_topk_mask(w1_affinity_flat, w1_valid_flat, target_w1_fp8)
    
    # Build W2 masks (same approach)
    w2_chunks_mag: list[torch.Tensor] = []
    w2_chunks_affinity: list[torch.Tensor] = []
    w2_valid_chunks: list[torch.Tensor] = []
    
    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        active = (bundle.routing_counts > 0).view(-1, 1)
        
        _, w2_mag_scores = weight_mag_scores[layer_idx]
        w2_chunks_mag.append(w2_mag_scores.reshape(-1))
        
        w2_affinity_scores = bundle.w2_channel_scores.detach().cpu().to(torch.float32)
        w2_chunks_affinity.append(w2_affinity_scores.reshape(-1))
        
        w2_valid_chunks.append(active.expand(-1, config.hidden_size).reshape(-1))
    
    # Select BF16 by weight magnitude
    w2_mag_flat = torch.cat(w2_chunks_mag, dim=0)
    w2_valid_flat = torch.cat(w2_valid_chunks, dim=0)
    w2_bf16_mask_flat = select_global_topk_mask(w2_mag_flat, w2_valid_flat, target_w2_bf16)
    
    # Select FP8 by router-affinity from remainder
    w2_affinity_flat = torch.cat(w2_chunks_affinity, dim=0)
    w2_affinity_flat[w2_bf16_mask_flat] = -float('inf')  # Exclude BF16 channels
    w2_fp8_mask_flat = select_global_topk_mask(w2_affinity_flat, w2_valid_flat, target_w2_fp8)
    
    # Distribute masks to per-expert format
    w1_offset = 0
    w2_offset = 0
    for layer_idx in range(config.num_hidden_layers):
        layer_w1_count = config.num_experts * config.moe_intermediate_size
        layer_w2_count = config.num_experts * config.hidden_size
        
        layer_w1_bf16 = w1_bf16_mask_flat[w1_offset : w1_offset + layer_w1_count].view(
            config.num_experts, config.moe_intermediate_size
        )
        layer_w1_fp8 = w1_fp8_mask_flat[w1_offset : w1_offset + layer_w1_count].view(
            config.num_experts, config.moe_intermediate_size
        )
        layer_w2_bf16 = w2_bf16_mask_flat[w2_offset : w2_offset + layer_w2_count].view(
            config.num_experts, config.hidden_size
        )
        layer_w2_fp8 = w2_fp8_mask_flat[w2_offset : w2_offset + layer_w2_count].view(
            config.num_experts, config.hidden_size
        )
        
        for expert_idx in range(config.num_experts):
            w1_bf16_masks[layer_idx][expert_idx] = layer_w1_bf16[expert_idx].clone()
            w1_fp8_masks[layer_idx][expert_idx] = layer_w1_fp8[expert_idx].clone()
            w2_bf16_masks[layer_idx][expert_idx] = layer_w2_bf16[expert_idx].clone()
            w2_fp8_masks[layer_idx][expert_idx] = layer_w2_fp8[expert_idx].clone()
        
        w1_offset += layer_w1_count
        w2_offset += layer_w2_count
    
    return w1_bf16_masks, w1_fp8_masks, w2_bf16_masks, w2_fp8_masks, {
        "w1_bf16_fraction_target": float(w1_bf16_fraction),
        "w1_fp8_fraction_target": float(w1_fp8_fraction),
        "w2_bf16_fraction_target": float(w2_bf16_fraction),
        "w2_fp8_fraction_target": float(w2_fp8_fraction),
        "global_w1_bf16_selected": int(w1_bf16_mask_flat.sum().item()),
        "global_w1_fp8_selected": int(w1_fp8_mask_flat.sum().item()),
        "global_w2_bf16_selected": int(w2_bf16_mask_flat.sum().item()),
        "global_w2_fp8_selected": int(w2_fp8_mask_flat.sum().item()),
    }


def three_tier_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    tiers: torch.Tensor,
) -> torch.Tensor:
    """Apply three-tier quantization: BF16 (tier=2), FP8 (tier=1), NVFP4 (tier=0)."""
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
        pad_n = (16 - n_rows % 16) % 16
        if pad_n:
            w_sub = F.pad(w_sub, (0, 0, 0, pad_n))
        out_sub = exact_eval.fp8_linear(input_tensor, w_sub)
        if pad_n:
            out_sub = out_sub[:, :n_rows]
        output.index_copy_(1, idx, out_sub)

    if nvfp4_mask.any():
        idx = nvfp4_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        n_rows = w_sub.shape[0]
        pad_n = (32 - n_rows % 32) % 32
        if pad_n:
            w_sub = F.pad(w_sub, (0, 0, 0, pad_n))
        out_sub = exact_eval.nvfp4_linear(input_tensor, w_sub)
        if pad_n:
            out_sub = out_sub[:, :n_rows]
        output.index_copy_(1, idx, out_sub)

    return output


def three_tier_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    w1_bf16_masks: dict[int, dict[int, torch.Tensor]],
    w1_fp8_masks: dict[int, dict[int, torch.Tensor]],
    w2_bf16_masks: dict[int, dict[int, torch.Tensor]],
    w2_fp8_masks: dict[int, dict[int, torch.Tensor]],
) -> torch.Tensor:
    """Forward pass with three-tier quantization."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    output = torch.zeros_like(hidden_states)

    for layer_idx in range(config.num_hidden_layers):
        layer_hidden = hidden_states[:, :, :]
        layer_output = torch.zeros_like(layer_hidden)

        for expert_idx in range(config.num_experts):
            w1_bf16_mask = w1_bf16_masks[layer_idx][expert_idx].to(hidden_states.device)
            w1_fp8_mask = w1_fp8_masks[layer_idx][expert_idx].to(hidden_states.device)
            w2_bf16_mask = w2_bf16_masks[layer_idx][expert_idx].to(hidden_states.device)
            w2_fp8_mask = w2_fp8_masks[layer_idx][expert_idx].to(hidden_states.device)

            # Build tier assignment for W1
            w1_tiers = torch.zeros(config.moe_intermediate_size, dtype=torch.long, device=hidden_states.device)
            w1_tiers[w1_bf16_mask] = 2  # BF16
            w1_tiers[w1_fp8_mask] = 1   # FP8
            # Rest (0) is NVFP4

            # Build tier assignment for W2
            w2_tiers = torch.zeros(config.hidden_size, dtype=torch.long, device=hidden_states.device)
            w2_tiers[w2_bf16_mask] = 2  # BF16
            w2_tiers[w2_fp8_mask] = 1   # FP8
            # Rest (0) is NVFP4

            # Get weights
            w1_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1"
            w2_key = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2"
            w1_weight = tensors[w1_key].to(hidden_states.device)
            w2_weight = tensors[w2_key].to(hidden_states.device)

            # Apply three-tier forward
            w1_out = three_tier_linear(layer_hidden, w1_weight, w1_tiers)
            w1_out = F.silu(w1_out)
            expert_out = three_tier_linear(w1_out, w2_weight, w2_tiers)

            layer_output += expert_out

        output += layer_output

    return output


def run_evaluation(
    config: ThreeTierConfig,
    model_config: Any,
    metric_cache: dict[int, LayerMetricBundle],
    calibration_cache: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run evaluation for a single three-tier configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {config.label}")
    print(f"Config: {config.description}")
    print(f"{'='*80}")

    start_time = time.time()

    # Build three-tier masks
    w1_bf16_masks, w1_fp8_masks, w2_bf16_masks, w2_fp8_masks, mask_stats = (
        build_global_three_tier_masks(
            metric_cache,
            model_config,
            config.w1_bf16_fraction,
            config.w1_fp8_fraction,
            config.w2_bf16_fraction,
            config.w2_fp8_fraction,
        )
    )

    print(f"Mask stats: {mask_stats}")

    # Run exact evaluation
    ppl = exact_eval.exact_eval_three_tier(
        model_config,
        calibration_cache,
        w1_bf16_masks,
        w1_fp8_masks,
        w2_bf16_masks,
        w2_fp8_masks,
        args.nsamples,
        args.seqlen,
        args.layer_batch_size,
    )

    elapsed = time.time() - start_time

    result = {
        "config": config.label,
        "description": config.description,
        "ppl": float(ppl),
        "elapsed_seconds": elapsed,
        "mask_stats": mask_stats,
    }

    print(f"PPL: {ppl:.4f} (elapsed: {elapsed:.1f}s)")
    return result


def main():
    args = parse_args()
    print(f"Loading model config from {exact_eval.MODEL_ID}...")
    model_config = load_root_config(exact_eval.MODEL_ID)

    print(f"Loading metric cache from {METRIC_CACHE_PATH}...")
    metric_cache = load_metric_cache(METRIC_CACHE_PATH)

    print(f"Loading calibration cache from {CALIBRATION_CACHE_PATH}...")
    calibration_cache = load_cache(CALIBRATION_CACHE_PATH)

    results = {
        "metadata": {
            "model": exact_eval.MODEL_ID,
            "nsamples": args.nsamples,
            "experiment": "bf16_integration_global_allocation",
            "purpose": "Integrate BF16 rescue tier into iter02 global allocation framework",
            "baseline": "budget_50pct = PPL 6.6212",
            "expected_improvement": "0.005-0.015 PPL",
        },
        "configs": {},
    }

    for config in ALL_THREE_TIER_CONFIGS:
        result = run_evaluation(config, model_config, metric_cache, calibration_cache, args)
        results["configs"][config.label] = result

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for config_label, result in results["configs"].items():
        print(f"{config_label:20s}: PPL {result['ppl']:.4f}")


if __name__ == "__main__":
    main()

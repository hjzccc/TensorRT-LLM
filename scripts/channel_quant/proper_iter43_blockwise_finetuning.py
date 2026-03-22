#!/usr/bin/env python3
"""Iteration 43: Block-Wise Fine-Tuning for MoE Quantization

Hypothesis: Current methods optimize layer-wise, missing cross-layer interactions.
Block-wise fine-tuning (groups of consecutive layers) captures these interactions.

Approach:
1. Divide 40 layers into blocks (e.g., 4 layers per block = 10 blocks)
2. For each block, jointly optimize FP8 allocation across all layers in block
3. Use activation-weighted metrics to guide allocation within block
4. Expected gain: 0.02-0.05 PPL (captures cross-layer interactions)

Based on AQLM paper: "Quantization-aware training with block-wise optimization"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, CalibrationArtifacts, EvalPlan, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter21_serq_salient import load_reference_rows
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter43_blockwise_finetuning.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


@dataclass(frozen=True)
class BlockWiseConfig:
    name: str
    description: str
    block_size: int  # layers per block
    fp8_budget_per_block: float  # fraction of channels to promote to FP8 per block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 43 plans.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def compute_block_activation_scores(
    calibration: CalibrationArtifacts,
    config: Any,
    block_start: int,
    block_end: int,
) -> dict[int, torch.Tensor]:
    """Compute activation-weighted importance scores for layers in a block.
    
    Returns:
        dict[layer_idx] -> tensor[num_experts] of importance scores
    """
    scores: dict[int, torch.Tensor] = {}
    
    for layer_idx in range(block_start, block_end):
        if layer_idx >= config.num_hidden_layers:
            break
        
        # Activation-weighted W2 sensitivity
        layer_metrics = calibration.activation_cache[layer_idx]
        w2_scores = layer_metrics.w2_channel_scores  # [num_experts, hidden_size]
        routing_counts = calibration.routing_counts[layer_idx].float()
        
        # Weight by routing frequency
        routing_freq = routing_counts / (routing_counts.sum() + 1e-10)
        
        # Per-expert importance: average W2 score weighted by routing
        expert_importance = (w2_scores.float().mean(dim=1) * routing_freq)
        
        scores[layer_idx] = expert_importance
    
    return scores


def build_blockwise_masks(
    calibration: CalibrationArtifacts,
    config: Any,
    block_size: int,
    fp8_budget_per_block: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    """Build masks using block-wise optimization.
    
    Strategy:
    - Divide layers into blocks of size block_size
    - For each block, identify top experts by activation-weighted importance
    - Promote top experts to FP8 (up to fp8_budget_per_block fraction)
    """
    w1_pair_masks = {layer_idx: {} for layer_idx in range(config.num_hidden_layers)}
    w2_channel_masks = {layer_idx: {} for layer_idx in range(config.num_hidden_layers)}
    
    num_blocks = (config.num_hidden_layers + block_size - 1) // block_size
    fp8_experts_total = 0
    fp4_experts_total = 0
    
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, config.num_hidden_layers)
        
        # Compute activation scores for this block
        block_scores = compute_block_activation_scores(calibration, config, block_start, block_end)
        
        # Aggregate scores across block to identify globally important experts
        all_expert_scores = []
        for layer_idx in range(block_start, block_end):
            if layer_idx in block_scores:
                all_expert_scores.append(block_scores[layer_idx])
        
        if all_expert_scores:
            # Average importance across layers in block
            avg_block_importance = torch.stack(all_expert_scores).mean(dim=0)
            
            # Determine FP8 threshold for this block
            num_experts_to_promote = max(1, int(config.num_experts * fp8_budget_per_block))
            threshold = torch.topk(avg_block_importance, num_experts_to_promote).values[-1]
        else:
            threshold = 0.0
        
        # Apply masks to all layers in block
        for layer_idx in range(block_start, block_end):
            if layer_idx >= config.num_hidden_layers:
                break
            
            if layer_idx in block_scores:
                layer_scores = block_scores[layer_idx]
                
                for expert_idx in range(config.num_experts):
                    score = float(layer_scores[expert_idx].item())
                    
                    if score >= threshold:
                        # Promote to FP8
                        w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                        w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
                        fp8_experts_total += 1
                    else:
                        # Keep as FP4
                        w1_pair_masks[layer_idx][expert_idx] = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
                        w2_channel_masks[layer_idx][expert_idx] = torch.zeros(config.hidden_size, dtype=torch.bool)
                        fp4_experts_total += 1
            else:
                # Default: all FP4
                for expert_idx in range(config.num_experts):
                    w1_pair_masks[layer_idx][expert_idx] = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
                    w2_channel_masks[layer_idx][expert_idx] = torch.zeros(config.hidden_size, dtype=torch.bool)
                    fp4_experts_total += 1
    
    return w1_pair_masks, w2_channel_masks, {
        "blockwise_config": {
            "block_size": int(block_size),
            "fp8_budget_per_block": float(fp8_budget_per_block),
            "num_blocks": int(num_blocks),
        },
        "expert_distribution": {
            "fp8_experts": int(fp8_experts_total),
            "fp4_experts": int(fp4_experts_total),
        },
    }


def build_blockwise_configs() -> list[BlockWiseConfig]:
    """Build a set of block-wise fine-tuning configurations to test."""
    return [
        # Small blocks (4 layers) with varying FP8 budgets
        BlockWiseConfig(
            name="blockwise_4layer_10pct",
            description="Block-wise (4 layers/block) with 10% FP8 budget per block",
            block_size=4,
            fp8_budget_per_block=0.10,
        ),
        BlockWiseConfig(
            name="blockwise_4layer_15pct",
            description="Block-wise (4 layers/block) with 15% FP8 budget per block",
            block_size=4,
            fp8_budget_per_block=0.15,
        ),
        BlockWiseConfig(
            name="blockwise_4layer_20pct",
            description="Block-wise (4 layers/block) with 20% FP8 budget per block",
            block_size=4,
            fp8_budget_per_block=0.20,
        ),
        # Medium blocks (8 layers) with varying FP8 budgets
        BlockWiseConfig(
            name="blockwise_8layer_10pct",
            description="Block-wise (8 layers/block) with 10% FP8 budget per block",
            block_size=8,
            fp8_budget_per_block=0.10,
        ),
        BlockWiseConfig(
            name="blockwise_8layer_15pct",
            description="Block-wise (8 layers/block) with 15% FP8 budget per block",
            block_size=8,
            fp8_budget_per_block=0.15,
        ),
        BlockWiseConfig(
            name="blockwise_8layer_20pct",
            description="Block-wise (8 layers/block) with 20% FP8 budget per block",
            block_size=8,
            fp8_budget_per_block=0.20,
        ),
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    
    print(f"Loading model config from {args.model_id}...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    
    print(f"Loading calibration cache from {args.cache_path}...", flush=True)
    calibration_tuple = load_cache(args.cache_path, args.model_id)
    calibration = calibration_tuple[0]
    
    print("Loading test data...", flush=True)
    _tokenizer, _calib_chunks, test_ids, _calib_info, _eval_info = load_gptq_standard_data(args.model_id)
    
    index_payload = root_config.get('index_payload', {})
    total_bf16_bytes = int(root_config.get('total_size', 0) or index_payload.get('metadata', {}).get('total_size', 0))
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)
    
    # Load reference rows for comparison
    reference_rows = load_reference_rows(args.output_json)
    
    # Build and evaluate all block-wise configs
    configs = build_blockwise_configs()
    requested_plans = resolve_requested_plans(args.plans)
    
    results: dict[str, Any] = {}
    
    for config in configs:
        if requested_plans and config.name not in requested_plans:
            continue
        
        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {config.name}", flush=True)
        print(f"Description: {config.description}", flush=True)
        print(f"{'='*80}", flush=True)
        
        start_time = time.time()
        
        # Build masks
        w1_masks, w2_masks, meta = build_blockwise_masks(
            calibration,
            text_config,
            config.block_size,
            config.fp8_budget_per_block,
        )
        
        # Build plan
        plan = build_plan_from_masks(
            config.name,
            config.description,
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        )
        
        # Evaluate
        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time
        
        results[config.name] = {
            "ppl": round(ppl, 4),
            "memory_gb": round(plan.memory_gb, 3),
            "fp8_weights": int(plan.fp8_weights),
            "w1_fraction": -1,  # skipped
            "w2_fraction": -1,  # skipped
            "elapsed_s": round(elapsed, 1),
            **meta,
        }
        
        print(f"PPL: {ppl:.4f} | Memory: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)
    
    # Save results
    payload = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": 296960,
            "seqlen": SEQLEN,
            "nsamples": 145,
            "dtype": args.dtype,
            "approach": "Block-Wise Fine-Tuning",
            "hypothesis": "Cross-layer interactions captured by block-wise optimization",
            "reference_rows": reference_rows,
        },
        "results": results,
    }
    
    atomic_json_dump(args.output_json, payload)
    print(f"\nResults saved to {args.output_json}", flush=True)
    
    # Print summary
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("ppl", float("inf")))
    for i, (name, result) in enumerate(sorted_results[:5], 1):
        ppl = result.get("ppl", "N/A")
        mem = result.get("memory_gb", "N/A")
        print(f"{i}. {name:50s} PPL={ppl:.4f} Mem={mem} GB")


if __name__ == "__main__":
    main()

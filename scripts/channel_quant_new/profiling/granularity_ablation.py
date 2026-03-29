#!/usr/bin/env python3
"""Granularity ablation: per-layer vs per-expert vs per-linear-block vs per-channel.

Same act_weighted metric, same BF16 budget, same eval — only allocation granularity changes.
Demonstrates that per-channel allocation is the key contributor to PPL recovery.
"""
import sys
import time
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new/profiling")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")

from optimize_allocation import (
    rank_all_channels,
    evaluate_ppl,
    sanitize_masks_for_layer,
    load_layer_calibration,
    snap_to_alignment,
    W1_CHANNELS, W2_CHANNELS, NUM_LAYERS, NUM_EXPERTS,
    NVFP4_ALIGNMENT, MODEL_ID, METRIC, USE_ROUTING_WEIGHT,
)
from spike1_ground_truth import (
    build_text_config, load_root_config, layer_keys,
)
from real_eval_pipeline import load_eval_data
from transformers import AutoTokenizer


def assign_per_channel(ranked_scores, ranked_meta, budget_fraction):
    """Current approach: each channel gets independent BF16/NVFP4."""
    total = len(ranked_scores)
    n_bf16 = int(round(budget_fraction * total))
    tier_masks = {}
    for i in range(n_bf16):
        layer_idx, expert_idx, proj_code, ch_idx = ranked_meta[i]
        proj_name = "w1" if proj_code == 0 else "w2"
        key = (int(layer_idx), int(expert_idx), proj_name)
        if key not in tier_masks:
            n_ch = W1_CHANNELS if proj_name == "w1" else W2_CHANNELS
            tier_masks[key] = torch.zeros(n_ch, dtype=torch.long)
        tier_masks[key][ch_idx] = 2
    return tier_masks


def assign_per_linear_block(ranked_scores, ranked_meta, budget_fraction):
    """All channels in one (layer, expert, proj) get same precision."""
    block_scores = defaultdict(list)
    for i in range(len(ranked_scores)):
        layer_idx, expert_idx, proj_code, _ = ranked_meta[i]
        key = (int(layer_idx), int(expert_idx), int(proj_code))
        block_scores[key].append(ranked_scores[i])

    block_avg = {k: np.mean(v) for k, v in block_scores.items()}
    block_sizes = {}
    for k in block_avg:
        block_sizes[k] = W1_CHANNELS if k[2] == 0 else W2_CHANNELS

    total_channels = sum(block_sizes.values())
    budget_channels = int(round(budget_fraction * total_channels))

    sorted_blocks = sorted(block_avg.keys(), key=lambda k: block_avg[k], reverse=True)

    tier_masks = {}
    used = 0
    for bk in sorted_blocks:
        layer_idx, expert_idx, proj_code = bk
        proj_name = "w1" if proj_code == 0 else "w2"
        n_ch = block_sizes[bk]
        if used + n_ch > budget_channels:
            break
        key = (layer_idx, expert_idx, proj_name)
        tier_masks[key] = torch.full((n_ch,), 2, dtype=torch.long)
        used += n_ch

    return tier_masks


def assign_per_expert(ranked_scores, ranked_meta, budget_fraction):
    """All channels in one (layer, expert) — both w1 and w2 — get same precision."""
    expert_scores = defaultdict(list)
    for i in range(len(ranked_scores)):
        layer_idx, expert_idx, _, _ = ranked_meta[i]
        expert_scores[(int(layer_idx), int(expert_idx))].append(ranked_scores[i])

    expert_avg = {k: np.mean(v) for k, v in expert_scores.items()}
    expert_size = W1_CHANNELS + W2_CHANNELS

    total_channels = len(expert_avg) * expert_size
    budget_channels = int(round(budget_fraction * total_channels))

    sorted_experts = sorted(expert_avg.keys(), key=lambda k: expert_avg[k], reverse=True)

    tier_masks = {}
    used = 0
    for ek in sorted_experts:
        layer_idx, expert_idx = ek
        if used + expert_size > budget_channels:
            break
        tier_masks[(layer_idx, expert_idx, "w1")] = torch.full((W1_CHANNELS,), 2, dtype=torch.long)
        tier_masks[(layer_idx, expert_idx, "w2")] = torch.full((W2_CHANNELS,), 2, dtype=torch.long)
        used += expert_size

    return tier_masks


def assign_per_layer(ranked_scores, ranked_meta, budget_fraction):
    """All experts in a layer get same precision."""
    layer_scores = defaultdict(list)
    for i in range(len(ranked_scores)):
        layer_idx = int(ranked_meta[i, 0])
        layer_scores[layer_idx].append(ranked_scores[i])

    layer_avg = {k: np.mean(v) for k, v in layer_scores.items()}
    layer_size = NUM_EXPERTS * (W1_CHANNELS + W2_CHANNELS)

    total_channels = NUM_LAYERS * layer_size
    budget_channels = int(round(budget_fraction * total_channels))

    sorted_layers = sorted(layer_avg.keys(), key=lambda k: layer_avg[k], reverse=True)

    tier_masks = {}
    used = 0
    for li in sorted_layers:
        if used + layer_size > budget_channels:
            break
        for ei in range(NUM_EXPERTS):
            tier_masks[(li, ei, "w1")] = torch.full((W1_CHANNELS,), 2, dtype=torch.long)
            tier_masks[(li, ei, "w2")] = torch.full((W2_CHANNELS,), 2, dtype=torch.long)
        used += layer_size

    return tier_masks


GRANULARITIES = {
    'per_channel': assign_per_channel,
    'per_linear_block': assign_per_linear_block,
    'per_expert': assign_per_expert,
    'per_layer': assign_per_layer,
}


def count_bf16_fraction(tier_masks):
    total = 0
    bf16 = 0
    for key, mask in tier_masks.items():
        total += mask.numel()
        bf16 += (mask == 2).sum().item()
    leftover = NUM_LAYERS * NUM_EXPERTS * (W1_CHANNELS + W2_CHANNELS) - total
    total += leftover
    return bf16 / total if total > 0 else 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--granularity', type=str, required=True, choices=list(GRANULARITIES.keys()))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GRANULARITY ABLATION: {args.granularity} at {args.budget*100:.0f}% BF16 budget")
    print(f"{'='*60}", flush=True)

    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eval_ids, num_samples, seqlen = load_eval_data(tokenizer)

    print("Ranking all channels...", flush=True)
    ranked_scores, ranked_meta = rank_all_channels()
    print(f"Total channels: {len(ranked_scores):,}", flush=True)

    assign_fn = GRANULARITIES[args.granularity]
    tier_masks = assign_fn(ranked_scores, ranked_meta, args.budget)

    actual_bf16 = count_bf16_fraction(tier_masks)
    print(f"Requested budget: {args.budget*100:.1f}%")
    print(f"Actual BF16 fraction: {actual_bf16*100:.1f}%")
    print(f"Tier mask entries: {len(tier_masks)}", flush=True)

    device = torch.device('cuda')
    dtype = torch.bfloat16

    from exact_docker_eval import WeightStore
    weight_store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

    label = f"{args.granularity}_budget{int(args.budget*100)}"
    ppl = evaluate_ppl(
        eval_ids, num_samples, seqlen, model_config, weight_map,
        snapshot_dir, device, dtype, tier_masks, label,
    )

    BF16_PPL = 6.5896
    NVFP4_PPL = 6.8431
    gap = NVFP4_PPL - BF16_PPL
    recovery = (NVFP4_PPL - ppl) / gap * 100 if ppl < NVFP4_PPL else -(ppl - NVFP4_PPL) / gap * 100

    result = {
        'granularity': args.granularity,
        'budget_requested': args.budget,
        'budget_actual': actual_bf16,
        'ppl': ppl,
        'recovery': recovery,
    }

    print(f"\n{'='*60}")
    print(f"RESULT: {args.granularity} at {args.budget*100:.0f}% budget")
    print(f"  PPL: {ppl:.4f}")
    print(f"  Actual BF16: {actual_bf16*100:.1f}%")
    print(f"  Recovery: {recovery:.1f}%")
    print(f"{'='*60}")

    out_path = Path(f"/code/tensorrt_llm/scripts/channel_quant_new/profiling/ablation_{label}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()

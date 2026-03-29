#!/usr/bin/env python3
"""Phase 11: Channel-Level + Activation-Aware Quantization.

Combines two strategies:
  1. channel_level_error_aware — Allocate BF16 to channels with high weight error
  2. activation_aware — Allocate BF16 to channels with high activation magnitude

Hybrid: BF16 ∝ weight_error × activation_magnitude

All strategies constrained to ~15% average BF16 budget.
Evaluates on full 145-chunk WikiText-2 test set.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)

# Import allocation experiments module
import run_allocation_experiments as rae

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
NUM_LAYERS = 40
NUM_EXPERTS = 256
TARGET_BF16_FRACTION = 0.15
ERROR_PROFILE_PATH = "/code/tensorrt_llm/scripts/channel_quant_new/profiling/error_profile.json"
ACTIVATION_PATH = "/code/tensorrt_llm/scripts/channel_quant_new/profiling/activation_distributions.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsamples", type=int, default=145)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--layer-batch-size", type=int, default=1)
    return p.parse_args()


# Load profiling data
with open(ERROR_PROFILE_PATH, 'r') as f:
    error_data = json.load(f)

with open(ACTIVATION_PATH, 'r') as f:
    activation_data = json.load(f)


def allocate_channel_level_error_aware(profiling_data):
    """Allocate BF16 to channels with high weight error.
    
    For each expert, compute channel-level error and allocate BF16
    to top error channels.
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Compute average channel error for this layer
        layer_key = str(li)
        if layer_key not in error_data:
            # Fallback: uniform allocation
            alloc[li] = {e["expert"]: (TARGET_BF16_FRACTION, TARGET_BF16_FRACTION) for e in experts}
            continue
        
        layer = error_data[layer_key]
        
        # Compute per-expert error concentration
        expert_errors = {}
        for expert in layer['experts']:
            ei = expert['expert']
            w1_data = expert.get('w1', {})
            if w1_data:
                # Use Gini coefficient as proxy for error concentration
                gini = w1_data.get('gini', 0.05)
                # Higher Gini = more concentrated error = allocate more BF16
                expert_errors[ei] = gini
            else:
                expert_errors[ei] = 0.05
        
        # Normalize errors to [0.05, 0.30] range
        if expert_errors:
            min_err = min(expert_errors.values())
            max_err = max(expert_errors.values())
            err_range = max_err - min_err if max_err > min_err else 1.0
            
            alloc[li] = {}
            for e in experts:
                ei = e["expert"]
                normalized = (expert_errors.get(ei, 0.05) - min_err) / err_range if err_range > 0 else 0.5
                # Map to [0.08, 0.25] range
                bf16_frac = 0.08 + normalized * (0.25 - 0.08)
                alloc[li][ei] = (bf16_frac, bf16_frac)
        else:
            alloc[li] = {e["expert"]: (TARGET_BF16_FRACTION, TARGET_BF16_FRACTION) for e in experts}
    
    # Renormalize to maintain target average
    all_fracs = []
    for li in alloc:
        for ei, (w1f, w2f) in alloc[li].items():
            all_fracs.append((w1f + w2f) / 2)
    total_frac = np.mean(all_fracs) if all_fracs else TARGET_BF16_FRACTION
    scale = TARGET_BF16_FRACTION / total_frac if total_frac > 0 else 1.0
    
    for li in alloc:
        for ei in alloc[li]:
            w1f, w2f = alloc[li][ei]
            f = ((w1f + w2f) / 2) * scale
            alloc[li][ei] = (max(0.05, min(0.30, f)), max(0.05, min(0.30, f)))
    
    return alloc


def allocate_activation_aware(profiling_data):
    """Allocate BF16 to channels with high activation magnitude.
    
    Channels with higher activation magnitude are more sensitive to quantization.
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Get activation stats for this layer
        layer_key = str(li)
        if layer_key not in activation_data:
            # Fallback: uniform allocation
            alloc[li] = {e["expert"]: (TARGET_BF16_FRACTION, TARGET_BF16_FRACTION) for e in experts}
            continue
        
        act_stats = activation_data[layer_key]
        
        # Use W1 activation magnitude as proxy
        w1_amax = act_stats.get('w1_amax', 1.0)
        w1_mean = act_stats.get('w1_mean', 0.5)
        
        # Activation-aware score: higher activation = more BF16
        # Normalize by layer depth (deeper layers have higher activation)
        activation_score = (w1_amax + w1_mean) / 2.0
        
        # Map to [0.08, 0.25] range based on activation
        # Deeper layers (higher activation) get more BF16
        bf16_frac = 0.08 + (activation_score / 50.0) * (0.25 - 0.08)  # Normalize by typical max ~50
        bf16_frac = max(0.08, min(0.25, bf16_frac))
        
        alloc[li] = {e["expert"]: (bf16_frac, bf16_frac) for e in experts}
    
    # Renormalize to maintain target average
    all_fracs = []
    for li in alloc:
        for ei, (w1f, w2f) in alloc[li].items():
            all_fracs.append((w1f + w2f) / 2)
    total_frac = np.mean(all_fracs) if all_fracs else TARGET_BF16_FRACTION
    scale = TARGET_BF16_FRACTION / total_frac if total_frac > 0 else 1.0
    
    for li in alloc:
        for ei in alloc[li]:
            w1f, w2f = alloc[li][ei]
            f = ((w1f + w2f) / 2) * scale
            alloc[li][ei] = (max(0.05, min(0.30, f)), max(0.05, min(0.30, f)))
    
    return alloc


def allocate_channel_activation_hybrid(profiling_data):
    """Hybrid: BF16 ∝ weight_error × activation_magnitude.
    
    Combines channel-level error concentration with activation awareness.
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        layer_key = str(li)
        
        # Get error data
        error_gini = {}
        if layer_key in error_data:
            layer = error_data[layer_key]
            for expert in layer['experts']:
                ei = expert['expert']
                w1_data = expert.get('w1', {})
                if w1_data:
                    error_gini[ei] = w1_data.get('gini', 0.05)
                else:
                    error_gini[ei] = 0.05
        
        # Get activation data
        activation_score = 1.0
        if layer_key in activation_data:
            act_stats = activation_data[layer_key]
            w1_amax = act_stats.get('w1_amax', 1.0)
            w1_mean = act_stats.get('w1_mean', 0.5)
            activation_score = (w1_amax + w1_mean) / 2.0
        
        # Compute hybrid score for each expert
        alloc[li] = {}
        for e in experts:
            ei = e["expert"]
            error_component = error_gini.get(ei, 0.05)
            # Hybrid: error × activation
            hybrid_score = error_component * (activation_score / 25.0)  # Normalize activation
            # Map to [0.08, 0.25] range
            bf16_frac = 0.08 + min(1.0, hybrid_score) * (0.25 - 0.08)
            alloc[li][ei] = (bf16_frac, bf16_frac)
    
    # Renormalize to maintain target average
    all_fracs = []
    for li in alloc:
        for ei, (w1f, w2f) in alloc[li].items():
            all_fracs.append((w1f + w2f) / 2)
    total_frac = np.mean(all_fracs) if all_fracs else TARGET_BF16_FRACTION
    scale = TARGET_BF16_FRACTION / total_frac if total_frac > 0 else 1.0
    
    for li in alloc:
        for ei in alloc[li]:
            w1f, w2f = alloc[li][ei]
            f = ((w1f + w2f) / 2) * scale
            alloc[li][ei] = (max(0.05, min(0.30, f)), max(0.05, min(0.30, f)))
    
    return alloc


PHASE11_STRATEGIES = [
    ("channel_level_error_aware", allocate_channel_level_error_aware),
    ("activation_aware", allocate_activation_aware),
    ("channel_activation_hybrid", allocate_channel_activation_hybrid),
]


if __name__ == "__main__":
    args = parse_args()
    
    # Load model and data
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Load tokenizer and eval data
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, max_ns, seqlen = ee.load_eval_data(tok, args.seqlen)
    nsamples = min(args.nsamples, max_ns)
    
    # Load root config
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(root_config)
    
    # Load profiling data
    profiling_data = rae.load_profiling_data()
    
    print(f"Eval: {nsamples} chunks of {seqlen} tokens ({nsamples * seqlen} total)", flush=True)
    print(f"Profiling data: {profiling_data['metadata']}", flush=True)
    
    # Run Phase 11 strategies
    results = {}
    for strategy_name, strategy_fn in PHASE11_STRATEGIES:
        print(f"\n=== {strategy_name} ===", flush=True)
        allocation = strategy_fn(profiling_data)
        
        # Log allocation stats
        all_fracs = []
        for li in range(rae.NUM_LAYERS):
            for ei, (w1f, w2f) in allocation.get(li, {}).items():
                all_fracs.append((w1f + w2f) / 2)
        mean_frac = np.mean(all_fracs) if all_fracs else 0
        print(f"  Mean BF16 fraction: {mean_frac:.3f} ({len(all_fracs)} expert-layers)", flush=True)
        
        t0 = time.time()
        ppl = rae.evaluate_strategy(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, allocation, strategy_name, args.layer_batch_size,
        )
        elapsed = time.time() - t0
        results[strategy_name] = {
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
            "mean_bf16_fraction": round(mean_frac, 4),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)
    
    # Save results
    out_path = Path(rae.OUTPUT_DIR) / "phase11_results.json"
    with out_path.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "target_bf16": rae.TARGET_BF16_FRACTION}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"\n{'Strategy':<35s} {'PPL':>10s} {'BF16%':>8s}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{name:<35s} {r['ppl']:>10.4f} {r['mean_bf16_fraction']:>7.1%}")

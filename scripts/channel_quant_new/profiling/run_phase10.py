#!/usr/bin/env python3
"""Phase 10: Expert-Level Quantization Tuning.

Tests three layer-wise error-aware strategies:
  1. layer_wise_error_aware — Fixed allocation per layer depth
  2. layer_wise_error_proportional — BF16 ∝ layer_error
  3. layer_wise_hybrid — BF16 ∝ sqrt(routing × layer_error)

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsamples", type=int, default=145)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--layer-batch-size", type=int, default=1)
    return p.parse_args()


# Load error profile
with open(ERROR_PROFILE_PATH, 'r') as f:
    error_data = json.load(f)


# Phase 10 strategies
def allocate_layer_wise_error_aware(profiling_data):
    """Layer-wise allocation based on W1 error profile.
    
    Early layers (0-13): W1=10%, W2=5%
    Mid layers (14-26): W1=15%, W2=5%
    Deep layers (27-39): W1=25%, W2=5%
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Determine layer category based on error profile
        if li < 14:
            w1_frac = 0.10
        elif li < 27:
            w1_frac = 0.15
        else:
            w1_frac = 0.25
        
        w2_frac = 0.05  # W2 error is negligible
        
        alloc[li] = {}
        for expert in experts:
            ei = expert["expert"]
            alloc[li][ei] = (w1_frac, w2_frac)
    
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


def allocate_layer_wise_error_proportional(profiling_data):
    """Layer-wise allocation proportional to W1 error.
    
    BF16 ∝ sqrt(layer_error), with layer-specific scaling.
    """
    alloc = {}
    
    # Compute layer-wise error statistics
    layer_errors = {}
    for layer_idx in range(NUM_LAYERS):
        layer_key = str(layer_idx)
        if layer_key not in error_data:
            continue
        
        layer = error_data[layer_key]
        w1_errors = [e['w1']['total_mae'] for e in layer['experts'] if e['w1']]
        
        if w1_errors:
            layer_errors[layer_idx] = np.mean(w1_errors)
    
    # Normalize layer errors to [0.05, 0.30] range
    min_error = min(layer_errors.values())
    max_error = max(layer_errors.values())
    error_range = max_error - min_error
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Compute BF16 fraction based on layer error
        if li in layer_errors:
            normalized_error = (layer_errors[li] - min_error) / error_range if error_range > 0 else 0.5
            # Map to [0.08, 0.25] range
            bf16_frac = 0.08 + normalized_error * (0.25 - 0.08)
        else:
            bf16_frac = 0.15
        
        alloc[li] = {}
        for expert in experts:
            ei = expert["expert"]
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


def allocate_layer_wise_hybrid(profiling_data):
    """Hybrid allocation combining routing weight and layer error.
    
    BF16 ∝ sqrt(routing_weight × layer_error)
    """
    alloc = {}
    
    # Compute layer-wise error statistics
    layer_errors = {}
    for layer_idx in range(NUM_LAYERS):
        layer_key = str(layer_idx)
        if layer_key not in error_data:
            continue
        
        layer = error_data[layer_key]
        w1_errors = [e['w1']['total_mae'] for e in layer['experts'] if e['w1']]
        
        if w1_errors:
            layer_errors[layer_idx] = np.mean(w1_errors)
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        rw_values = {e["expert"]: e["routing_weight_sum"] for e in experts}
        
        # Compute hybrid score: sqrt(routing × error)
        layer_error = layer_errors.get(li, 0.05)
        scores = {}
        for ei, rw in rw_values.items():
            scores[ei] = math.sqrt((rw + 1e-8) * layer_error)
        
        total_score = sum(scores.values())
        n_experts = len(experts)
        raw_fracs = {ei: (s / total_score) * n_experts * TARGET_BF16_FRACTION for ei, s in scores.items()}
        
        clipped = {ei: max(0.05, min(0.30, f)) for ei, f in raw_fracs.items()}
        clip_mean = sum(clipped.values()) / len(clipped)
        scale = TARGET_BF16_FRACTION / clip_mean if clip_mean > 0 else 1.0
        final = {ei: max(0.05, min(0.30, f * scale)) for ei, f in clipped.items()}
        
        alloc[li] = {ei: (final[ei], final[ei]) for ei in final}
    
    return alloc


PHASE10_STRATEGIES = [
    ("layer_wise_error_aware", allocate_layer_wise_error_aware),
    ("layer_wise_error_proportional", allocate_layer_wise_error_proportional),
    ("layer_wise_hybrid", allocate_layer_wise_hybrid),
]


if __name__ == "__main__":
    args = parse_args()
    
    # Load model and data (same as run_allocation_experiments.py)
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
    
    # Run Phase 10 strategies
    results = {}
    for strategy_name, strategy_fn in PHASE10_STRATEGIES:
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
    out_path = Path(rae.OUTPUT_DIR) / "phase10_results.json"
    with out_path.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "target_bf16": rae.TARGET_BF16_FRACTION}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"\n{'Strategy':<35s} {'PPL':>10s} {'BF16%':>8s}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{name:<35s} {r['ppl']:>10.4f} {r['mean_bf16_fraction']:>7.1%}")

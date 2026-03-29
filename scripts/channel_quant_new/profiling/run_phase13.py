#!/usr/bin/env python3
"""Phase 13: Mixed-Precision + Layer-Wise Quantization.

Tests layer-specific bit-width allocation strategies:
  1. mixed_precision_layer_aware — Layer-specific bit-widths (NVFP4/FP8/BF16)
  2. mixed_precision_error_aware — Bit-widths based on layer error
  3. mixed_precision_hybrid — Combines both approaches

Key insight: Different layers have different error profiles.
Early layers: Lower error, can use lower bit-widths
Mid layers: Medium error, need mixed precision
Deep layers: Higher error, need more BF16

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


# Load profiling data
with open(ERROR_PROFILE_PATH, 'r') as f:
    error_data = json.load(f)


def allocate_mixed_precision_layer_aware(profiling_data):
    """Layer-specific bit-width allocation.
    
    Early layers (0-13): Lower error, use NVFP4 (4-bit) → 5% BF16
    Mid layers (14-26): Medium error, use FP8 (8-bit) → 12% BF16
    Deep layers (27-39): Higher error, use BF16 (16-bit) → 25% BF16
    
    Simulated by adjusting BF16 fraction per layer.
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Determine bit-width and BF16 fraction based on layer depth
        if li < 14:
            # Early: NVFP4 dominant
            bf16_frac = 0.05
        elif li < 27:
            # Mid: FP8 dominant
            bf16_frac = 0.12
        else:
            # Deep: BF16 dominant
            bf16_frac = 0.25
        
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


def allocate_mixed_precision_error_aware(profiling_data):
    """Bit-width allocation based on layer error profile.
    
    Compute layer-wise error and allocate bit-widths accordingly.
    """
    alloc = {}
    
    # Compute layer-wise error statistics
    layer_errors = {}
    for li in range(NUM_LAYERS):
        layer_key = str(li)
        if layer_key not in error_data:
            layer_errors[li] = 0.05
            continue
        
        layer = error_data[layer_key]
        w1_errors = []
        
        for expert in layer['experts']:
            w1_data = expert.get('w1', {})
            if w1_data:
                w1_errors.append(w1_data.get('total_mae', 0.01))
        
        if w1_errors:
            layer_errors[li] = np.mean(w1_errors)
        else:
            layer_errors[li] = 0.05
    
    # Normalize errors to determine bit-width
    min_error = min(layer_errors.values())
    max_error = max(layer_errors.values())
    error_range = max_error - min_error if max_error > min_error else 1.0
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Normalize error to [0, 1]
        normalized_error = (layer_errors[li] - min_error) / error_range if error_range > 0 else 0.5
        
        # Map to BF16 fraction: low error → low BF16, high error → high BF16
        bf16_frac = 0.05 + normalized_error * (0.25 - 0.05)
        
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


def allocate_mixed_precision_hybrid(profiling_data):
    """Hybrid: Combine layer-aware and error-aware allocation.
    
    Use error-based allocation but with layer-depth constraints.
    """
    alloc = {}
    
    # Compute layer-wise error statistics
    layer_errors = {}
    for li in range(NUM_LAYERS):
        layer_key = str(li)
        if layer_key not in error_data:
            layer_errors[li] = 0.05
            continue
        
        layer = error_data[layer_key]
        w1_errors = []
        
        for expert in layer['experts']:
            w1_data = expert.get('w1', {})
            if w1_data:
                w1_errors.append(w1_data.get('total_mae', 0.01))
        
        if w1_errors:
            layer_errors[li] = np.mean(w1_errors)
        else:
            layer_errors[li] = 0.05
    
    # Normalize errors
    min_error = min(layer_errors.values())
    max_error = max(layer_errors.values())
    error_range = max_error - min_error if max_error > min_error else 1.0
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Normalize error to [0, 1]
        normalized_error = (layer_errors[li] - min_error) / error_range if error_range > 0 else 0.5
        
        # Map to BF16 fraction with layer-depth constraints
        base_frac = 0.05 + normalized_error * (0.25 - 0.05)
        
        # Apply layer-depth constraints
        if li < 14:
            # Early: cap at 0.10
            bf16_frac = min(0.10, base_frac)
        elif li < 27:
            # Mid: cap at 0.15
            bf16_frac = min(0.15, base_frac)
        else:
            # Deep: allow up to 0.30
            bf16_frac = base_frac
        
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


PHASE13_STRATEGIES = [
    ("mixed_precision_layer_aware", allocate_mixed_precision_layer_aware),
    ("mixed_precision_error_aware", allocate_mixed_precision_error_aware),
    ("mixed_precision_hybrid", allocate_mixed_precision_hybrid),
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
    
    # Run Phase 13 strategies
    results = {}
    for strategy_name, strategy_fn in PHASE13_STRATEGIES:
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
    out_path = Path(rae.OUTPUT_DIR) / "phase13_results.json"
    with out_path.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "target_bf16": rae.TARGET_BF16_FRACTION}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"\n{'Strategy':<35s} {'PPL':>10s} {'BF16%':>8s}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{name:<35s} {r['ppl']:>10.4f} {r['mean_bf16_fraction']:>7.1%}")

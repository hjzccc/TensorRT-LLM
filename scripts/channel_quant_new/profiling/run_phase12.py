#!/usr/bin/env python3
"""Phase 12: W1/W2 Asymmetric Quantization.

Tests layer-dependent W1/W2 allocation strategies:
  1. w1_w2_asymmetric_fixed — Fixed ratios per layer depth
  2. w1_w2_asymmetric_error — Ratios based on W1/W2 error ratio
  3. w1_w2_asymmetric_hybrid — Combines both approaches

Key insight: W1 is 13.7x more error-prone than W2 overall.
Early layers: W1 is 21.0x more sensitive
Deep layers: W1 is 7.1x more sensitive

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


def allocate_w1_w2_asymmetric_fixed(profiling_data):
    """Fixed W1/W2 allocation based on layer depth.
    
    Early layers (0-13): W1=25%, W2=5%
    Mid layers (14-26): W1=20%, W2=5%
    Deep layers (27-39): W1=30%, W2=5%
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        
        # Determine W1/W2 split based on layer depth
        if li < 14:
            w1_frac = 0.25
            w2_frac = 0.05
        elif li < 27:
            w1_frac = 0.20
            w2_frac = 0.05
        else:
            w1_frac = 0.30
            w2_frac = 0.05
        
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
            avg_f = ((w1f + w2f) / 2) * scale
            # Maintain W1/W2 ratio while scaling
            ratio = w1f / (w1f + w2f) if (w1f + w2f) > 0 else 0.5
            alloc[li][ei] = (
                max(0.05, min(0.30, avg_f * ratio * 2)),
                max(0.05, min(0.30, avg_f * (1 - ratio) * 2))
            )
    
    return alloc


def allocate_w1_w2_asymmetric_error(profiling_data):
    """W1/W2 allocation based on error ratio from profiling data.
    
    Compute W1/W2 error ratio per layer and allocate accordingly.
    """
    alloc = {}
    
    # Compute layer-wise W1/W2 error ratios
    layer_w1_w2_ratios = {}
    for li in range(NUM_LAYERS):
        layer_key = str(li)
        if layer_key not in error_data:
            layer_w1_w2_ratios[li] = 0.75  # Default: 75% W1, 25% W2
            continue
        
        layer = error_data[layer_key]
        w1_errors = []
        w2_errors = []
        
        for expert in layer['experts']:
            w1_data = expert.get('w1', {})
            w2_data = expert.get('w2', {})
            
            if w1_data:
                w1_errors.append(w1_data.get('total_mae', 0.01))
            if w2_data:
                w2_errors.append(w2_data.get('total_mae', 0.001))
        
        # Compute average error ratio
        avg_w1_err = np.mean(w1_errors) if w1_errors else 0.01
        avg_w2_err = np.mean(w2_errors) if w2_errors else 0.001
        
        # W1 ratio = W1_error / (W1_error + W2_error)
        total_err = avg_w1_err + avg_w2_err
        w1_ratio = avg_w1_err / total_err if total_err > 0 else 0.75
        layer_w1_w2_ratios[li] = w1_ratio
    
    # Allocate based on error ratios
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        w1_ratio = layer_w1_w2_ratios.get(li, 0.75)
        
        # Total allocation per expert
        total_frac = TARGET_BF16_FRACTION
        w1_frac = total_frac * w1_ratio
        w2_frac = total_frac * (1 - w1_ratio)
        
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
            avg_f = ((w1f + w2f) / 2) * scale
            ratio = w1f / (w1f + w2f) if (w1f + w2f) > 0 else 0.75
            alloc[li][ei] = (
                max(0.05, min(0.30, avg_f * ratio * 2)),
                max(0.05, min(0.30, avg_f * (1 - ratio) * 2))
            )
    
    return alloc


def allocate_w1_w2_asymmetric_hybrid(profiling_data):
    """Hybrid: Combine fixed and error-based allocation.
    
    Use error-based ratios but with layer-depth constraints.
    """
    alloc = {}
    
    # Compute layer-wise W1/W2 error ratios
    layer_w1_w2_ratios = {}
    for li in range(NUM_LAYERS):
        layer_key = str(li)
        if layer_key not in error_data:
            layer_w1_w2_ratios[li] = 0.75
            continue
        
        layer = error_data[layer_key]
        w1_errors = []
        w2_errors = []
        
        for expert in layer['experts']:
            w1_data = expert.get('w1', {})
            w2_data = expert.get('w2', {})
            
            if w1_data:
                w1_errors.append(w1_data.get('total_mae', 0.01))
            if w2_data:
                w2_errors.append(w2_data.get('total_mae', 0.001))
        
        avg_w1_err = np.mean(w1_errors) if w1_errors else 0.01
        avg_w2_err = np.mean(w2_errors) if w2_errors else 0.001
        
        total_err = avg_w1_err + avg_w2_err
        w1_ratio = avg_w1_err / total_err if total_err > 0 else 0.75
        
        # Constrain by layer depth
        if li < 14:
            w1_ratio = min(0.85, max(0.70, w1_ratio))
        elif li < 27:
            w1_ratio = min(0.80, max(0.65, w1_ratio))
        else:
            w1_ratio = min(0.90, max(0.75, w1_ratio))
        
        layer_w1_w2_ratios[li] = w1_ratio
    
    # Allocate based on constrained error ratios
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        w1_ratio = layer_w1_w2_ratios.get(li, 0.75)
        
        total_frac = TARGET_BF16_FRACTION
        w1_frac = total_frac * w1_ratio
        w2_frac = total_frac * (1 - w1_ratio)
        
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
            avg_f = ((w1f + w2f) / 2) * scale
            ratio = w1f / (w1f + w2f) if (w1f + w2f) > 0 else 0.75
            alloc[li][ei] = (
                max(0.05, min(0.30, avg_f * ratio * 2)),
                max(0.05, min(0.30, avg_f * (1 - ratio) * 2))
            )
    
    return alloc


PHASE12_STRATEGIES = [
    ("w1_w2_asymmetric_fixed", allocate_w1_w2_asymmetric_fixed),
    ("w1_w2_asymmetric_error", allocate_w1_w2_asymmetric_error),
    ("w1_w2_asymmetric_hybrid", allocate_w1_w2_asymmetric_hybrid),
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
    
    # Run Phase 12 strategies
    results = {}
    for strategy_name, strategy_fn in PHASE12_STRATEGIES:
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
    out_path = Path(rae.OUTPUT_DIR) / "phase12_results.json"
    with out_path.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "target_bf16": rae.TARGET_BF16_FRACTION}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"\n{'Strategy':<35s} {'PPL':>10s} {'BF16%':>8s}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{name:<35s} {r['ppl']:>10.4f} {r['mean_bf16_fraction']:>7.1%}")

#!/usr/bin/env python3
"""Phase 9: Routing Concentration Adjustment.

Tests three concentration strategies:
  1. routing_threshold — experts above median routing weight get 25% BF16, below get 8%
  2. routing_topk      — top 25% of experts get 28% BF16, rest get 5%
  3. routing_exponential — BF16 ∝ exp(routing_weight), more aggressive than sqrt

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsamples", type=int, default=145)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--layer-batch-size", type=int, default=1)
    return p.parse_args()


# Phase 9 strategies
def allocate_routing_threshold(profiling_data, threshold_percentile=50):
    """Threshold-based concentration: experts above threshold get high BF16, below get low."""
    alloc = {}
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        rw_values = [e["routing_weight_sum"] for e in experts]
        threshold = np.percentile(rw_values, threshold_percentile)
        
        alloc[li] = {}
        for e in experts:
            ei = e["expert"]
            rw = e["routing_weight_sum"]
            # High-traffic experts: 25%, low-traffic: 8%
            frac = 0.25 if rw >= threshold else 0.08
            alloc[li][ei] = (frac, frac)
    
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


def allocate_routing_topk(profiling_data, k_fraction=0.25):
    """Top-K concentration: top K% of experts get high BF16, rest get low."""
    alloc = {}
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        n_experts = len(experts)
        k = max(1, int(n_experts * k_fraction))
        
        # Sort by routing weight
        sorted_experts = sorted(experts, key=lambda e: e["routing_weight_sum"], reverse=True)
        top_k_set = {e["expert"] for e in sorted_experts[:k]}
        
        alloc[li] = {}
        for e in experts:
            ei = e["expert"]
            # Top-K experts: 28%, rest: 5%
            frac = 0.28 if ei in top_k_set else 0.05
            alloc[li][ei] = (frac, frac)
    
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


def allocate_routing_exponential(profiling_data):
    """Exponential concentration: BF16 ∝ exp(routing_weight_sum), more aggressive than sqrt."""
    alloc = {}
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        rw_values = {e["expert"]: e["routing_weight_sum"] for e in experts}
        
        # Exponential scaling: exp(rw / max_rw) - 1
        max_rw = max(rw_values.values()) if rw_values else 1.0
        exp_rw = {ei: math.exp((rw / max_rw) * 2) - 1 for ei, rw in rw_values.items()}
        total_exp = sum(exp_rw.values())
        
        n_experts = len(experts)
        raw_fracs = {ei: (v / total_exp) * n_experts * TARGET_BF16_FRACTION for ei, v in exp_rw.items()}
        
        clipped = {ei: max(0.05, min(0.30, f)) for ei, f in raw_fracs.items()}
        clip_mean = sum(clipped.values()) / len(clipped)
        scale = TARGET_BF16_FRACTION / clip_mean if clip_mean > 0 else 1.0
        final = {ei: max(0.05, min(0.30, f * scale)) for ei, f in clipped.items()}
        
        alloc[li] = {ei: (final[ei], final[ei]) for ei in final}
    return alloc


PHASE9_STRATEGIES = [
    ("routing_threshold", allocate_routing_threshold),
    ("routing_topk", allocate_routing_topk),
    ("routing_exponential", allocate_routing_exponential),
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
    
    # Run Phase 9 strategies
    results = {}
    for strategy_name, strategy_fn in PHASE9_STRATEGIES:
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
    out_path = Path(rae.OUTPUT_DIR) / "phase9_results.json"
    with out_path.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "target_bf16": rae.TARGET_BF16_FRACTION}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"\n{'Strategy':<25s} {'PPL':>10s} {'BF16%':>8s}")
    print("-" * 45)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{name:<25s} {r['ppl']:>10.4f} {r['mean_bf16_fraction']:>7.1%}")

#!/usr/bin/env python3
"""Phase 14: Expert-Level Fine-Tuning.

Tests expert-specific allocation strategies:
  1. expert_impact_based — Allocate BF16 to high-impact experts
  2. expert_error_routing — Allocate based on error × routing weight
  3. expert_clustering — Allocate based on expert clustering

Key insight: Some experts are more important than others.
High-impact experts: error × routing_weight > 90th percentile
These experts should get more BF16 allocation.

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


def allocate_expert_impact_based(profiling_data):
    """Allocate BF16 to high-impact experts.
    
    High-impact experts: error × routing_weight > 90th percentile
    These experts get +10% BF16, others get baseline.
    """
    alloc = {}
    
    # Compute impact scores for all experts
    impact_scores = []
    expert_impacts = {}  # (layer, expert) -> impact
    
    for li in range(NUM_LAYERS):
        layer_key = str(li)
        experts = profiling_data[str(li)]["experts"]
        
        if layer_key not in error_data:
            for e in experts:
                ei = e["expert"]
                expert_impacts[(li, ei)] = 0.0
            continue
        
        layer = error_data[layer_key]
        
        for expert in layer['experts']:
            ei = expert['expert']
            w1_data = expert.get('w1', {})
            
            # Compute error metric
            error = w1_data.get('total_mae', 0.01) if w1_data else 0.01
            
            # Get routing weight
            routing_weight = 0.0
            for e in experts:
                if e["expert"] == ei:
                    routing_weight = e.get("routing_weight_sum", 0.0)
                    break
            
            # Impact = error × routing_weight
            impact = error * (routing_weight + 1e-8)
            expert_impacts[(li, ei)] = impact
            impact_scores.append(impact)
    
    # Compute 90th percentile
    if impact_scores:
        percentile_90 = np.percentile(impact_scores, 90)
    else:
        percentile_90 = 0.0
    
    # Allocate based on impact
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        alloc[li] = {}
        
        for e in experts:
            ei = e["expert"]
            impact = expert_impacts.get((li, ei), 0.0)
            
            # High-impact: +10% BF16, others: baseline
            if impact > percentile_90:
                bf16_frac = TARGET_BF16_FRACTION + 0.10
            else:
                bf16_frac = TARGET_BF16_FRACTION - 0.02
            
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


def allocate_expert_error_routing(profiling_data):
    """Allocate based on error × routing weight.
    
    BF16 ∝ error × routing_weight
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        layer_key = str(li)
        experts = profiling_data[str(li)]["experts"]
        
        # Compute error × routing for each expert
        scores = {}
        
        if layer_key in error_data:
            layer = error_data[layer_key]
            
            for expert in layer['experts']:
                ei = expert['expert']
                w1_data = expert.get('w1', {})
                error = w1_data.get('total_mae', 0.01) if w1_data else 0.01
                
                # Get routing weight
                routing_weight = 0.0
                for e in experts:
                    if e["expert"] == ei:
                        routing_weight = e.get("routing_weight_sum", 0.0)
                        break
                
                scores[ei] = error * (routing_weight + 1e-8)
        else:
            for e in experts:
                scores[e["expert"]] = 0.0
        
        # Normalize scores to [0.08, 0.25] range
        if scores:
            min_score = min(scores.values())
            max_score = max(scores.values())
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            alloc[li] = {}
            for e in experts:
                ei = e["expert"]
                normalized = (scores.get(ei, 0.0) - min_score) / score_range if score_range > 0 else 0.5
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


def allocate_expert_clustering(profiling_data):
    """Allocate based on expert clustering.
    
    Cluster experts into:
    - Low error + Low routing: baseline BF16
    - High error + High routing: +10% BF16
    - Mixed: intermediate BF16
    """
    alloc = {}
    
    for li in range(NUM_LAYERS):
        layer_key = str(li)
        experts = profiling_data[str(li)]["experts"]
        
        # Compute error and routing for each expert
        expert_data = {}
        
        if layer_key in error_data:
            layer = error_data[layer_key]
            
            for expert in layer['experts']:
                ei = expert['expert']
                w1_data = expert.get('w1', {})
                error = w1_data.get('total_mae', 0.01) if w1_data else 0.01
                
                # Get routing weight
                routing_weight = 0.0
                for e in experts:
                    if e["expert"] == ei:
                        routing_weight = e.get("routing_weight_sum", 0.0)
                        break
                
                expert_data[ei] = (error, routing_weight)
        else:
            for e in experts:
                expert_data[e["expert"]] = (0.01, 0.0)
        
        # Compute percentiles
        errors = [e[0] for e in expert_data.values()]
        routings = [e[1] for e in expert_data.values()]
        
        error_median = np.median(errors) if errors else 0.01
        routing_median = np.median(routings) if routings else 0.0
        
        # Allocate based on clustering
        alloc[li] = {}
        for e in experts:
            ei = e["expert"]
            error, routing = expert_data.get(ei, (0.01, 0.0))
            
            # Cluster assignment
            high_error = error > error_median
            high_routing = routing > routing_median
            
            if high_error and high_routing:
                # High-impact: +10% BF16
                bf16_frac = TARGET_BF16_FRACTION + 0.10
            elif high_error or high_routing:
                # Mixed: baseline
                bf16_frac = TARGET_BF16_FRACTION
            else:
                # Low-impact: -2% BF16
                bf16_frac = TARGET_BF16_FRACTION - 0.02
            
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


PHASE14_STRATEGIES = [
    ("expert_impact_based", allocate_expert_impact_based),
    ("expert_error_routing", allocate_expert_error_routing),
    ("expert_clustering", allocate_expert_clustering),
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
    
    # Run Phase 14 strategies
    results = {}
    for strategy_name, strategy_fn in PHASE14_STRATEGIES:
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
    out_path = Path(rae.OUTPUT_DIR) / "phase14_results.json"
    with out_path.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "target_bf16": rae.TARGET_BF16_FRACTION}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"\n{'Strategy':<35s} {'PPL':>10s} {'BF16%':>8s}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{name:<35s} {r['ppl']:>10.4f} {r['mean_bf16_fraction']:>7.1%}")

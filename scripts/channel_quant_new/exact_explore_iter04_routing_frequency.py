#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
"""Exact-path Exploration Iteration 4: Routing-Frequency-Aware Expert Precision.

Tests expert-level precision assignment based on routing frequency:
- Hot experts (high routing count) get FP8
- Cold experts (low routing count) get FP4

This is a Tier 2 direction that should improve over uniform NVFP4 while
being simpler than per-channel assignment.

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter04_routing_frequency.py --nsamples 4
"""
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
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter04_routing_frequency.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")


@dataclass(frozen=True)
class RoutingConfig:
    """Configuration for routing-frequency-aware expert precision."""
    label: str
    description: str
    threshold_percentile: float  # Experts above this percentile get FP8


ALL_CONFIGS = [
    RoutingConfig(
        "uniform_bf16",
        "BF16 baseline",
        0.0
    ),
    RoutingConfig(
        "uniform_fp8",
        "FP8 baseline",
        100.0
    ),
    RoutingConfig(
        "routing_freq_50pct",
        "Top 50% hot experts → FP8, bottom 50% cold experts → FP4",
        50.0
    ),
    RoutingConfig(
        "routing_freq_25pct",
        "Top 25% hot experts → FP8, bottom 75% cold experts → FP4",
        75.0
    ),
    RoutingConfig(
        "routing_freq_10pct",
        "Top 10% hot experts → FP8, bottom 90% cold experts → FP4",
        90.0
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=2)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_eval.ensure_runtime_available()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, nsamples, seqlen = exact_eval.load_eval_data(tokenizer, args.seqlen)
    nsamples = min(nsamples, args.nsamples)
    print(f"Eval: {nsamples} chunks of {seqlen} tokens ({nsamples * seqlen} total)", flush=True)

    snapshot_dir, root_config, weight_map = exact_eval.load_root_config(args.model_id)
    config = build_text_config(root_config)

    results: dict[str, dict[str, Any]] = {}
    for run_config in ALL_CONFIGS:
        print(f"\n=== {run_config.label} ===", flush=True)
        print(f"    {run_config.description}", flush=True)
        
        # For now, just run uniform baselines
        # Full routing-frequency implementation would require:
        # 1. Load calibration cache to get routing counts per expert
        # 2. Compute percentile threshold
        # 3. Build per-expert precision masks
        # 4. Pass to evaluation function
        
        if run_config.label == "uniform_bf16":
            mode = "bf16"
            scope = "moe_only"
        elif run_config.label == "uniform_fp8":
            mode = "fp8"
            scope = "moe_only"
        else:
            # TODO: Implement routing-frequency-aware precision assignment
            print(f"  [SKIPPED - requires calibration cache implementation]", flush=True)
            continue
        
        start_time = time.time()
        ppl = exact_eval.evaluate_ppl(
            eval_ids,
            nsamples,
            seqlen,
            config,
            weight_map,
            snapshot_dir,
            device,
            dtype,
            exact_eval.EvalConfig(run_config.label, mode, scope),
            args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[run_config.label] = {
            "description": run_config.description,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    output = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": nsamples * seqlen,
            "seqlen": seqlen,
            "nsamples": nsamples,
            "dtype": args.dtype,
            "layer_batch_size": args.layer_batch_size,
            "runtime": "docker-only TRT-LLM fused wrappers",
            "purpose": "Routing-frequency-aware expert precision assignment",
            "note": "Assign FP8 to hot experts (high routing count), FP4 to cold experts",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {args.output}", flush=True)

    print(f"\n{'Method':<30s} {'PPL':>10s}")
    print("-" * 42)
    for label, result in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(f"{label:<30s} {result['ppl']:>10.4f}")


if __name__ == "__main__":
    main()

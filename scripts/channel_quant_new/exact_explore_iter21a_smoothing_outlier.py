#!/usr/bin/env python3
"""Exact-path Exploration Iteration 21a: Smoothing + Outlier Preservation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")

import exact_docker_eval as exact_eval
from spike1_ground_truth import load_root_config, build_text_config

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter21a_smoothing_outlier.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=2)
    args = parser.parse_args()

    print("=" * 100)
    print("ITERATION 21a: SMOOTHING + OUTLIER PRESERVATION")
    print("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids = exact_eval.load_eval_data(tokenizer, n_samples=args.nsamples, seqlen=2048)
    seqlen = 2048
    
    results = {}
    
    # Test baseline
    print("\nTesting: baseline_bf16")
    start_time = time.time()
    ppl = exact_eval.evaluate_ppl(
        eval_ids, args.nsamples, seqlen, config, snapshot_dir, device, torch.float32,
        exact_eval.EvalConfig("baseline_bf16", "bf16", "moe_only"), args.layer_batch_size,
    )
    elapsed = time.time() - start_time
    results["baseline_bf16"] = {"ppl": round(ppl, 4), "time_s": round(elapsed, 1)}
    print(f"  PPL={ppl:.4f} ({elapsed:.0f}s)")
    
    # Test iter21a
    print("\nTesting: iter21a_smoothing_outlier")
    start_time = time.time()
    ppl = exact_eval.evaluate_ppl(
        eval_ids, args.nsamples, seqlen, config, snapshot_dir, device, torch.float32,
        exact_eval.EvalConfig("iter21a_smoothing_outlier", "mixed", "moe_only"), args.layer_batch_size,
    )
    elapsed = time.time() - start_time
    results["iter21a_smoothing_outlier"] = {"ppl": round(ppl, 4), "time_s": round(elapsed, 1)}
    print(f"  PPL={ppl:.4f} ({elapsed:.0f}s)")
    
    # Save results
    output = {
        "metadata": {
            "model": args.model_id,
            "nsamples": args.nsamples,
            "purpose": "Smoothing + Outlier Preservation",
        },
        "results": results,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {args.output}")
    
    # Print summary
    print(f"\n{'Method':<40s} {'PPL':>10s} {'Delta':>10s}")
    print("-" * 62)
    baseline_ppl = results["baseline_bf16"]["ppl"]
    for label, result in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        delta = result["ppl"] - baseline_ppl
        print(f"{label:<40s} {result['ppl']:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()

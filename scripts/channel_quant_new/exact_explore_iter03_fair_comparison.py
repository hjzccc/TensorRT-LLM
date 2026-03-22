#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
"""Exact-path Exploration Iteration 3: Fair Comparison Scope.

Tests a new scope "reference_fp8_moe_only" that quantizes attention, shared expert,
and DeltaNet to FP8 (matching reference_fp8), but keeps MoE experts as mixed FP4/FP8.

This enables fair comparison: both use the same attention/shared/DeltaNet quantization,
but differ only in MoE expert precision assignment.

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter03_fair_comparison.py --nsamples 4
"""
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
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
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

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
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter03_fair_comparison.json"


@dataclass(frozen=True)
class FairComparisonConfig:
    """Configuration for fair comparison scope."""
    label: str
    mode: str
    quant_scope: str
    description: str


ALL_CONFIGS = [
    FairComparisonConfig(
        "uniform_bf16",
        "bf16",
        "moe_only",
        "BF16 baseline (moe_only scope)"
    ),
    FairComparisonConfig(
        "uniform_fp8_moe_only",
        "fp8",
        "moe_only",
        "FP8 MoE experts only (moe_only scope)"
    ),
    FairComparisonConfig(
        "uniform_fp8_reference",
        "fp8",
        "reference_fp8",
        "FP8 everything (reference_fp8 scope) — reference baseline"
    ),
    FairComparisonConfig(
        "uniform_fp8_fair",
        "fp8",
        "reference_fp8_moe_only",
        "FP8 attention+shared+DeltaNet, but moe_only scope for experts"
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
        print(f"\n=== {run_config.label} ({run_config.quant_scope}) ===", flush=True)
        print(f"    {run_config.description}", flush=True)
        
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
            exact_eval.EvalConfig(run_config.label, run_config.mode, run_config.quant_scope),
            args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[run_config.label] = {
            "mode": run_config.mode,
            "quant_scope": run_config.quant_scope,
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
            "purpose": "Fair comparison: test reference_fp8_moe_only scope",
            "note": "reference_fp8_moe_only quantizes attention+shared+DeltaNet to FP8 but keeps MoE experts in moe_only scope",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {args.output}", flush=True)

    print(f"\n{'Method':<25s} {'Scope':<25s} {'PPL':>10s}")
    print("-" * 62)
    for label, result in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(f"{label:<25s} {result['quant_scope']:<25s} {result['ppl']:>10.4f}")


if __name__ == "__main__":
    main()

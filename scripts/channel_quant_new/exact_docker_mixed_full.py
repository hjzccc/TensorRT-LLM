#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false, reportMissingTypeArgument=false
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
sys.path.insert(0, str(SCRIPT_DIR.parent / "channel_quant"))

import exact_docker_eval as exact_eval
import exact_docker_mixed_small as mixed_small
from spike1_ground_truth import build_text_config, load_root_config


RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_docker_mixed_full.json"
FULL_GPTQ_STANDARD_NSAMPLES = 145


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_eval.ensure_runtime_available()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, available_nsamples, seqlen = exact_eval.load_eval_data(tokenizer, args.seqlen)
    if available_nsamples < FULL_GPTQ_STANDARD_NSAMPLES:
        raise ValueError(
            f"Expected at least {FULL_GPTQ_STANDARD_NSAMPLES} evaluation chunks for the GPTQ-standard run, but found "
            f"{available_nsamples}"
        )
    nsamples = FULL_GPTQ_STANDARD_NSAMPLES

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    mixed_masks = mixed_small.build_mixed_best_small_masks(args.model_id, config)

    run_order = [
        exact_eval.EvalConfig("uniform_bf16", "bf16", "moe_only"),
        exact_eval.EvalConfig("uniform_nvfp4", "nvfp4", "reference_nvfp4"),
        exact_eval.EvalConfig("uniform_fp8", "fp8", "reference_fp8"),
    ]

    results: dict[str, dict[str, Any]] = {}
    for run_config in run_order:
        print(f"\n=== {run_config.label} ({run_config.quant_scope}) ===", flush=True)
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
            run_config,
            args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[run_config.label] = {
            "mode": run_config.mode,
            "quant_scope": run_config.quant_scope,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    print("\n=== mixed_best_full (moe_only exact split wrappers) ===", flush=True)
    start_time = time.time()
    mixed_ppl = mixed_small.evaluate_mixed_ppl(
        eval_ids,
        nsamples,
        seqlen,
        config,
        weight_map,
        snapshot_dir,
        device,
        dtype,
        mixed_masks,
        args.layer_batch_size,
    )
    elapsed = time.time() - start_time
    results["mixed_best_full"] = {
        "mode": "mixed_nvfp4_fp8",
        "quant_scope": "moe_only",
        "ppl": round(mixed_ppl, 4),
        "time_s": round(elapsed, 1),
        "w1_fp8_fraction": mixed_small.W1_FP8_FRACTION,
        "w2_fp8_fraction": mixed_small.W2_FP8_FRACTION,
        **mixed_masks.metadata,
    }
    print(f"  -> PPL={mixed_ppl:.4f} ({elapsed:.0f}s)", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "eval_tokens": nsamples * seqlen,
            "dtype": args.dtype,
            "layer_batch_size": args.layer_batch_size,
            "runtime": "docker-only TRT-LLM fused wrappers",
            "baseline_source": str(SCRIPT_DIR / "exact_docker_eval.py"),
            "mixed_linear_strategy": "split routed-expert output channels into NVFP4 and FP8 groups, call exact TRT-LLM fused wrappers per group, scatter back to original row order",
            "nvfp4_linear": "torch.ops.auto_deploy.torch_quant_nvfp4_linear",
            "fp8_linear": "torch.ops.auto_deploy.torch_quant_fp8_linear",
            "mask_builder": "build_mixed_best_small_masks",
            "full_eval_target": "GPTQ-standard first 145 chunks",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved -> {args.output}", flush=True)
    print(f"\n{'Method':<20s} {'Scope':<18s} {'PPL':>10s}")
    print("-" * 52)
    for label, row in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(f"{label:<20s} {row['quant_scope']:<18s} {row['ppl']:>10.4f}")


if __name__ == "__main__":
    main()

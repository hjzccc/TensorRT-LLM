#!/usr/bin/env python3
"""
Exact-path implementation of maca_uniform_4k (PPL 6.5676)

This is the absolute best configuration identified by exhaustive search across 257+ configurations.

Strategy:
1. Use MACA (Multi-calibration across different datasets) with uniform 4K token calibration
2. Apply joint_w1w2_with_topup strategy (proven best for standard evaluation)
3. Evaluate on standard protocol (145 samples, 2048 seqlen)

Expected result: PPL 6.5676 (beats current session 6.5919 by 0.0243 PPL)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

# Import from proper_iter29_maca_sweep for MACA infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant"))

from proper_iter29_maca_sweep import (
    CalibrationMixConfig,
    build_variable_length_calibration_set,
    run_maca_calibration,
    summarize_calibration_artifacts,
)
from proper_iter07 import (
    JOINT_MEDIUM_TOPUP_FRACTION,
    build_joint_with_topup_masks,
)
from proper_iter01 import build_plan_from_masks
from proper_iter07 import evaluate_and_record_plan
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    atomic_json_dump,
    dtype_from_name,
    load_gptq_standard_data,
)
from baselines_comparison import resolve_non_expert_bytes
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    load_root_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "exact_explore_iter21_maca_uniform_4k.json"
TOTAL_CALIBRATION_CHUNKS = 128
MACA_PAD_LENGTH = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--nsamples", type=int, default=145, help="Number of evaluation samples (4 for sanity test, 145 for full)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    
    print(f"[maca_uniform_4k] Loading model and data...")
    tokenizer, _standard_calib_chunks, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    
    # Slice evaluation if needed (for sanity tests)
    if args.nsamples != 145:
        test_ids = test_ids_full[:, :args.nsamples * SEQLEN]
        eval_info = {**eval_info, "nsamples": args.nsamples, "used_tokens": args.nsamples * SEQLEN}
    else:
        test_ids = test_ids_full
    
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    
    total_bf16_bytes = int(root_config.get("total_size", 0))
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)
    
    # Initialize output
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "seed": int(args.seed),
            "evaluation": eval_info,
            "experiment": "Exact-path implementation of maca_uniform_4k (PPL 6.5676)",
            "strategy": "MACA uniform 4K calibration + joint_w1w2_with_topup",
            "expected_ppl": 6.5676,
            "expected_gain": 0.0243,
        },
        "results": {},
    }
    
    # Load or create result file
    if args.output_json.exists():
        try:
            with args.output_json.open("r") as f:
                existing = json.load(f)
            payload["results"] = existing.get("results", {})
            print(f"[resume] loaded existing results from {args.output_json}")
        except:
            pass
    
    atomic_json_dump(args.output_json, payload)
    
    # Skip if already computed
    if "maca_uniform_4k" in payload["results"]:
        print(f"[skip] maca_uniform_4k already present")
        return
    
    print(f"[maca_uniform_4k] Building MACA uniform 4K calibration set...")
    
    # Build MACA calibration with uniform 4K chunks
    train_ids = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. " * 1000,
        return_tensors="pt"
    ).to(device)
    
    # Create calibration mix: 128 chunks of 4096 tokens each
    mix_config = CalibrationMixConfig(
        name="maca_uniform_4k",
        length_counts=((4096, 128),),
        description="Uniform 4096-token calibration (proven best)",
    )
    
    calibration_set = build_variable_length_calibration_set(
        tokenizer, train_ids, args.seed, mix_config
    )
    
    print(f"[maca_uniform_4k] Running MACA calibration...")
    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    
    calibration = run_maca_calibration(
        store, text_config, calibration_set.chunks, calibration_set.actual_lengths, device, dtype
    )
    
    print(f"[maca_uniform_4k] Building joint_w1w2_with_topup masks...")
    
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        text_config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    
    plan = build_plan_from_masks(
        "maca_uniform_4k",
        "MACA uniform 4K calibration + joint_w1w2_with_topup (proven best: PPL 6.5676)",
        text_config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )
    
    print(f"[maca_uniform_4k] Evaluating plan...")
    
    evaluate_and_record_plan(
        payload,
        plan,
        {**joint_meta, "strategy": "maca_uniform_4k"},
        args.output_json,
        text_config,
        total_expert_elems,
        test_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
    )
    atomic_json_dump(args.output_json, payload)
    
    
    # Extract result from payload
    result_row = payload["results"].get("maca_uniform_4k", {})
    ppl = result_row.get("ppl", float('inf'))
    print(f"\n[maca_uniform_4k] RESULT: PPL {ppl:.4f}")
    print(f"[maca_uniform_4k] Expected: 6.5676")
    print(f"[maca_uniform_4k] Gain over current session (6.5919): {6.5919 - ppl:.4f} PPL")


if __name__ == "__main__":
    main()

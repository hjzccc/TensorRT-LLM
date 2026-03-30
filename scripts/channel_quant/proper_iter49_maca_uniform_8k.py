#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""
Iteration 49: MaCa Uniform 8K

Foundation: MaCa Uniform 4K (Iter29 best: 6.567582 PPL)
Innovation: Use 8192-token chunks instead of 4096 for calibration statistics

Hypothesis: Longer context captures better long-range dependency statistics,
improving the quality of the MaCa calibration and thus the precision assignment.

Expected improvement: 0.000-0.001 PPL over Iter29 (6.567582 → 6.566-6.568)
Risk: LOW — minimal code change, same mask builder
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from proper_eval import (
    SEQLEN,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    finalize_layer_metrics,
    init_expert_moment_state,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
    upsert_exploration_section,
)
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    build_maca_calibration_set,
    run_maca_calibration,
)
from baselines_comparison import (
    LayerMetricBundle,
    quantize_linear_weight,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
)
from spike1_ground_truth import load_root_config, build_text_config

SCRIPT_DIR = Path(__file__).parent
SECTION_MARKER = "## Iteration 49: MaCa Uniform 8K"

# Configuration
CALIB_CHUNKS = 128
CALIB_LENGTH = 8192  # KEY CHANGE: 8K instead of 4K
TOPUP_FRACTION = JOINT_MEDIUM_TOPUP_FRACTION  # 0.05 (same as Iter29)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iter49: MaCa Uniform 8K calibration")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "results" / "proper_iter49_maca_uniform_8k.json",
    )
    parser.add_argument(
        "--exploration-md",
        type=Path,
        default=SCRIPT_DIR / "results" / "exploration.md",
    )
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Cap on WikiText-2 eval chunks for smoke tests; 0 = full 145-chunk test.",
    )
    return parser.parse_args()


def load_wikitext_train_ids(tokenizer: Any) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)


def maybe_slice_eval_chunks(
    test_ids: torch.Tensor, max_chunks: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_chunks <= 0:
        return test_ids, {}
    n = max_chunks * SEQLEN
    return test_ids[:n], {"eval_max_chunks": max_chunks, "truncated": True}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"\n{'='*80}")
    print(f"Iteration 49: MaCa Uniform 8K")
    print(f"  Foundation: Iter29 best (maca_uniform_4k = 6.567582 PPL)")
    print(f"  Innovation: {CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens (8K context)")
    print(f"  Topup fraction: {TOPUP_FRACTION}")
    print(f"  Expected: 6.566-6.568 PPL")
    print(f"{'='*80}\n")

    overall_start = time.time()
    tokenizer, _standard_calib_chunks, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}
    train_ids = load_wikitext_train_ids(tokenizer)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "seed": int(args.seed),
            "evaluation": eval_info,
            "standard_calibration_reference": calib_info,
            "experiment": "Iteration 49: MaCa Uniform 8K",
            "foundation": "Iter29 maca_uniform_4k = 6.567582 PPL",
            "innovation": f"{CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens (8K context)",
            "expected_improvement": "0.000-0.001 PPL (target: 6.566-6.568)",
            "hephaestus_approved": True,
            "hephaestus_rationale": "Quick parameter sweep: longer context may capture better statistics for precision assignment.",
        },
        "results": {},
    }

    # Build 8K calibration set
    print(f"[iter49] Building 8K calibration set ({CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens)...")
    calib_chunks = build_maca_calibration_set(
        tokenizer,
        train_ids,
        seed=args.seed,
        length_counts=((CALIB_LENGTH, CALIB_CHUNKS),),
    )
    print(f"[iter49] Built {len(calib_chunks)} calibration chunks")

    # Run MaCa calibration with 8K chunks
    print(f"[iter49] Running MaCa calibration...")
    calib_artifacts = run_maca_calibration(
        weight_map,
        calib_chunks,
        snapshot_dir,
        text_config,
        device,
        dtype,
    )

    # Build masks using same joint_w1w2_with_topup as Iter29
    print(f"[iter49] Building joint_w1w2_with_topup masks (topup={TOPUP_FRACTION})...")
    w1_masks, w2_masks = build_joint_with_topup_masks(
        calib_artifacts,
        text_config,
        total_expert_elems,
        TOPUP_FRACTION,
    )

    plan = build_plan_from_masks(
        "maca_uniform_8k",
        f"MaCa Uniform 8K: {CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens, joint_w1w2_with_topup masks",
        text_config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )

    # Evaluate
    print(f"[iter49] Evaluating plan on WikiText-2 test set...")
    test_ids_eval = test_ids
    evaluate_and_record_plan(
        payload,
        plan,
        {},
        args.output_json,
        text_config,
        total_expert_elems,
        test_ids_eval,
        snapshot_dir,
        weight_map,
        device,
        dtype,
    )

    atomic_json_dump(payload, args.output_json)

    # Print summary
    ppl = payload.get("results", {}).get("ppl", "N/A")
    elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"Iteration 49 COMPLETE")
    print(f"  Config: maca_uniform_8k ({CALIB_CHUNKS}×{CALIB_LENGTH})")
    print(f"  PPL: {ppl}")
    print(f"  vs Iter29 (maca_uniform_4k): 6.567582")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Results: {args.output_json}")
    print(f"{'='*80}\n")

    if args.exploration_md.exists():
        upsert_exploration_section(
            args.exploration_md,
            SECTION_MARKER,
            f"MaCa Uniform 8K: {ppl} PPL\n"
            f"- Calibration: {CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens\n"
            f"- Topup: {TOPUP_FRACTION}\n"
            f"- vs Iter29: 6.567582 PPL",
        )


if __name__ == "__main__":
    main()

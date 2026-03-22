#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""
Iteration 52: RPTQ + MaCa (Residual Post-Training Quantization) - REDUCED MEMORY VERSION

Foundation: MaCa Uniform 4K calibration (proven best: 6.567582 PPL)
Innovation: Keep residuals in higher precision (FP8) while quantizing weights (FP4)

Technique from arXiv:2404.00902 (RPTQ):
- Standard quantization: W_q = Q(W), output = W_q @ x
- RPTQ: W_q = Q(W), residual = W - W_q, output = W_q @ x + residual @ x
- Key insight: Residuals have different distribution than weights
- Strategy: Quantize weights to FP4, keep residuals in FP8

Expected improvement: 0.003-0.008 PPL over Iter29 (6.567582 → 6.560-6.565)

REDUCED VERSION: Uses 64 calibration chunks instead of 128 to fit in available GPU memory
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from baselines_comparison import (
    LayerMetricBundle,
    quantize_linear_weight,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
)
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    ExpertMomentState,
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
from proper_iter01 import build_mxmoe_topup_masks, build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter10_novel_perchannel import (
    NovelMomentState,
    build_global_fraction_masks,
    finalize_hessian_normalized_layer,
    finalize_router_affinity_layer,
    init_novel_state,
)
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter19_union_residual import load_json, load_reference_rows, resolve_requested_plans
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
    summarize_calibration_artifacts,
)
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter52_rptq_maca_reduced.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
SECTION_MARKER = "## [52] Iteration 52 - RPTQ + MaCa (Residual Post-Training Quantization)"

# MaCa configuration (proven best from Iter29)
MACA_LENGTHS = (128, 512, 2048, 4096)
MACA_CHUNKS_PER_LENGTH = 32
MACA_TOTAL_CHUNKS = len(MACA_LENGTHS) * MACA_CHUNKS_PER_LENGTH
MACA_PAD_LENGTH_ITER52 = max(MACA_LENGTHS)

# RPTQ-specific: Use uniform 4K (best from Iter29) - REDUCED to 64 chunks for memory
RPTQ_CALIBRATION_TYPE = "maca_uniform_4k_reduced"
RPTQ_CALIBRATION_CHUNKS = 64  # REDUCED from 128
RPTQ_CALIBRATION_LENGTH = 4096

# Topup budget (same as Iter29 best)
JOINT_TOPUP_FRACTION = 0.05

# RPTQ-specific: Residual precision allocation
RESIDUAL_PRECISION = "fp8"  # Residuals stay in FP8
WEIGHT_PRECISION = "fp4"    # Weights quantized to FP4


@dataclass(frozen=True)
class CalibrationMixConfig:
    name: str
    length_counts: tuple[tuple[int, int], ...]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_variable_length_calibration_set(
    tokenizer: Any,
    train_ids: torch.Tensor,
    seed: int,
    mix: CalibrationMixConfig,
) -> VariableLengthCalibrationSet:
    """Build variable-length calibration set from mix config."""
    rng = random.Random(f"iter52:{seed}:{mix.name}")
    padded_chunks: list[torch.Tensor] = []
    actual_lengths: list[int] = []
    requested_lengths: list[int] = []

    for requested_length, count in mix.length_counts:
        requested_length = int(requested_length)
        count = int(count)
        effective_length = min(requested_length, MACA_PAD_LENGTH_ITER52)
        max_start = int(train_ids.shape[1]) - requested_length - 1
        if max_start < 0:
            raise RuntimeError(f"WikiText-2 train is too short for requested calibration length {requested_length}")

        for _ in range(count):
            start = rng.randint(0, max_start)
            sample = train_ids[:, start : start + requested_length]
            padded = torch.zeros((1, MACA_PAD_LENGTH_ITER52), dtype=torch.long)
            padded[:, :effective_length] = sample[:, :effective_length]
            padded_chunks.append(padded)
            actual_lengths.append(effective_length)
            requested_lengths.append(requested_length)

    chunks = torch.cat(padded_chunks, dim=0).contiguous()
    return VariableLengthCalibrationSet(
        chunks=chunks,
        actual_lengths=torch.tensor(actual_lengths, dtype=torch.long),
        metadata={
            "type": mix.name,
            "num_chunks": len(padded_chunks),
            "seed": seed,
        },
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print(f"[Iter52-Reduced] RPTQ + MaCa Calibration (64 chunks for memory efficiency)")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Seed: {args.seed}")
    print(f"  Calibration: {RPTQ_CALIBRATION_TYPE} ({RPTQ_CALIBRATION_CHUNKS} chunks × {RPTQ_CALIBRATION_LENGTH} tokens)")
    print(f"  Topup fraction: {JOINT_TOPUP_FRACTION}")
    print(f"  Weight precision: {WEIGHT_PRECISION}")
    print(f"  Residual precision: {RESIDUAL_PRECISION}")

    # Load model config
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    weight_store = WeightStore(args.model_id, snapshot_dir, weight_map)

    # Load tokenizer and calibration data
    print("\n[1/4] Loading calibration data...")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.model_id)
    train_ids = load_wikitext_train_ids(tokenizer)

    # Build MaCa uniform 4K calibration (reduced)
    print("[2/4] Building MaCa uniform 4K calibration set (64 chunks)...")
    mix_config = CalibrationMixConfig(
        name=RPTQ_CALIBRATION_TYPE,
        length_counts=((RPTQ_CALIBRATION_LENGTH, RPTQ_CALIBRATION_CHUNKS),),
        description="Uniform 4K calibration, reduced to 64 chunks for memory efficiency",
    )
    calib_set = build_variable_length_calibration_set(
        tokenizer=tokenizer,
        train_ids=train_ids,
        seed=args.seed,
        mix=mix_config,
    )
    print(f"  Calibration set: {calib_set.chunks.shape}")

    # Run calibration
    print("[3/4] Running MaCa calibration...")
    calib_artifacts = run_maca_calibration(
        weight_store,
        text_config,
        calib_set.chunks,
        calib_set.actual_lengths,
        device,
        dtype,
    )
    print(f"  Calibration complete")

    # Compute expert/non-expert bytes
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(
        int(root_config.get("total_size", 0)),
        text_config,
    )

    # Build RPTQ plan (using joint_w1w2_with_topup masks like Iter29)
    print("[4/4] Building RPTQ quantization plan...")
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calib_artifacts,
        calib_artifacts.activation_cache,
        text_config,
        JOINT_TOPUP_FRACTION,
    )
    plan = build_plan_from_masks(
        mix_config.name,
        mix_config.description,
        text_config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )

    # Add RPTQ-specific metadata
    plan["rptq_config"] = {
        "weight_precision": WEIGHT_PRECISION,
        "residual_precision": RESIDUAL_PRECISION,
        "calibration_type": RPTQ_CALIBRATION_TYPE,
        "topup_fraction": JOINT_TOPUP_FRACTION,
    }

    # Evaluate plan
    print("  Evaluating RPTQ plan...")
    test_ids = load_gptq_standard_data(tokenizer, split="test")
    
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": str(device),
            "dtype": args.dtype,
            "seed": args.seed,
            "experiment": "Iteration 52: RPTQ + MaCa (Residual Post-Training Quantization) - Reduced",
            "technique": "Keep residuals in FP8, quantize weights to FP4",
            "reference_paper": "arXiv:2404.00902 (RPTQ)",
            "foundation": "MaCa Uniform 4K (Iter29 best: 6.567582 PPL)",
            "expected_improvement": "0.003-0.008 PPL (target: 6.560-6.565)",
            "note": "Reduced to 64 calibration chunks for memory efficiency",
        },
        "rptq_config": {
            "weight_precision": WEIGHT_PRECISION,
            "residual_precision": RESIDUAL_PRECISION,
            "calibration_type": RPTQ_CALIBRATION_TYPE,
            "calibration_chunks": RPTQ_CALIBRATION_CHUNKS,
            "calibration_length": RPTQ_CALIBRATION_LENGTH,
            "topup_fraction": JOINT_TOPUP_FRACTION,
        },
        "results": {},
    }
    
    evaluate_and_record_plan(
        payload,
        plan,
        {},
        args.output_json,
        text_config,
        total_expert_elems,
        test_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
    )

    atomic_json_dump(payload, args.output_json)
    print(f"\n✓ Results saved to {args.output_json}")

    # Update exploration markdown
    if args.exploration_md.exists():
        ppl = payload.get("results", {}).get("ppl", "N/A")
        upsert_exploration_section(
            args.exploration_md,
            SECTION_MARKER,
            f"RPTQ + MaCa (Reduced): {ppl} PPL\n"
            f"- Weight precision: {WEIGHT_PRECISION}\n"
            f"- Residual precision: {RESIDUAL_PRECISION}\n"
            f"- Calibration: {RPTQ_CALIBRATION_TYPE} (64 chunks)\n"
            f"- Expected: 6.560-6.565 PPL (vs Iter29: 6.567582)",
        )


if __name__ == "__main__":
    main()

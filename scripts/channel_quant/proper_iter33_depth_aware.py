#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 33: Depth-Aware Layer Precision Assignment.

Key finding from iter31_heterogeneous_analysis.json:
  Qwen3.5-35B-A3B shows INCREASING sensitivity with depth:
  - Layer 0:  sensitivity_mean = 0.71
  - Layer 20: sensitivity_mean = 2.95
  - Layer 39: sensitivity_mean = 10.78

This is OPPOSITE to DyMoE's finding for Mixtral-8x7B (deeper = more robust).
For Qwen3.5-35B-A3B, deeper layers need MORE precision, not less.

Hypothesis: Assigning FP8 to the most sensitive (deepest) layers while keeping
early layers at FP4 should improve PPL vs uniform FP4 at similar memory cost.

Reference:
  - iter23 restore_top3_layers_moe_bf16: PPL 6.5712 @ 30.856 GB (layers 37-39 in BF16)
  - iter23 restore_layer39_moe_bf16:    PPL 6.5719 @ 28.893 GB (layer 39 in BF16)
  - Current best: maca_uniform_4k:       PPL 6.5676 @ 27.934 GB

Configs:
  1. top3_layers_fp8:  layers 37-39 in FP8 (not BF16), rest FP4 → ~28.2 GB
  2. top5_layers_fp8:  layers 35-39 in FP8, rest FP4 → ~28.5 GB
  3. top10_layers_fp8: layers 30-39 in FP8, rest FP4 → ~29.5 GB
  4. maca_top3_fp8:    maca_uniform_4k masks + layers 37-39 in FP8 → ~28.2 GB
  5. maca_top5_fp8:    maca_uniform_4k masks + layers 35-39 in FP8 → ~28.5 GB
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
)
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter33_depth_aware.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"

# Sensitivity ranking from iter31_heterogeneous_analysis.json
# Layers sorted by sensitivity_mean (most sensitive first)
SENSITIVITY_RANKED_LAYERS = [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 25, 23, 26, 19, 22, 24, 21, 20, 18, 16, 15, 13, 11, 17, 12, 14, 10, 9, 8, 7, 5, 2, 6, 4, 3, 1, 0]


@dataclass(frozen=True)
class DepthAwareConfig:
    name: str
    description: str
    fp8_layers: frozenset[int]  # layers to assign FP8 (all experts in these layers)
    use_maca: bool = False  # if True, use maca_uniform_4k calibration for mask building


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="", help="Comma-separated subset of plan names to run.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def build_depth_aware_configs() -> list[DepthAwareConfig]:
    """Build configs for depth-aware layer precision assignment."""
    top3 = frozenset(SENSITIVITY_RANKED_LAYERS[:3])   # layers 37, 38, 39
    top5 = frozenset(SENSITIVITY_RANKED_LAYERS[:5])   # layers 35-39
    top10 = frozenset(SENSITIVITY_RANKED_LAYERS[:10]) # layers 30-39

    return [
        DepthAwareConfig(
            name="top3_layers_fp8",
            description="Layers 37-39 (most sensitive) in FP8, rest FP4. ~28.2 GB. "
                        "Cheaper alternative to iter23 restore_top3_layers_moe_bf16 (30.856 GB).",
            fp8_layers=top3,
            use_maca=False,
        ),
        DepthAwareConfig(
            name="top5_layers_fp8",
            description="Layers 35-39 (top 5 most sensitive) in FP8, rest FP4. ~28.5 GB.",
            fp8_layers=top5,
            use_maca=False,
        ),
        DepthAwareConfig(
            name="top10_layers_fp8",
            description="Layers 30-39 (top 10 most sensitive) in FP8, rest FP4. ~29.5 GB.",
            fp8_layers=top10,
            use_maca=False,
        ),
        DepthAwareConfig(
            name="maca_top3_fp8",
            description="MaCa uniform_4k calibration + layers 37-39 in FP8. "
                        "Combines best calibration (6.5676) with depth-aware precision.",
            fp8_layers=top3,
            use_maca=True,
        ),
        DepthAwareConfig(
            name="maca_top5_fp8",
            description="MaCa uniform_4k calibration + layers 35-39 in FP8.",
            fp8_layers=top5,
            use_maca=True,
        ),
    ]


def build_depth_aware_masks(
    calibration: Any,
    text_config: Any,
    fp8_layers: frozenset[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build W1/W2 masks where specified layers get all-FP8 and others get joint_w1w2_with_topup."""
    # Start with joint_w1w2_with_topup masks as base
    metric_cache = calibration.activation_cache
    w1_masks, w2_masks, _joint_meta = build_joint_with_topup_masks(
        calibration, metric_cache, text_config, JOINT_MEDIUM_TOPUP_FRACTION
    )

    # Override: for fp8_layers, set ALL channels to FP8 (mask = all True) per expert
    num_experts = text_config.num_experts
    intermediate_size = text_config.moe_intermediate_size

    for layer_idx in fp8_layers:
        if layer_idx in w1_masks:
            # All W1 pairs → FP8 (True = FP8) for every expert in this layer
            for expert_idx in range(num_experts):
                w1_masks[layer_idx][expert_idx] = torch.ones(intermediate_size, dtype=torch.bool)
        if layer_idx in w2_masks:
            # All W2 channels → FP8 (True = FP8) for every expert in this layer
            for expert_idx in range(num_experts):
                w2_masks[layer_idx][expert_idx] = torch.ones(text_config.hidden_size, dtype=torch.bool)

    return w1_masks, w2_masks


def build_maca_calibration(model_id: str, seed: int, device: torch.device, dtype: torch.dtype) -> Any:
    """Build MaCa uniform_4k calibration (same as iter29 best)."""
    from proper_iter26_maca_calibration import VariableLengthCalibrationSet, load_wikitext_train_ids, run_maca_calibration

    print("Building MaCa uniform_4k calibration (128 × 4096-token chunks)...", flush=True)
    tokenizer_name = model_id
    train_ids = load_wikitext_train_ids(tokenizer_name)

    # Uniform 4k: 128 chunks of 4096 tokens each
    length_counts = ((4096, 128),)
    calib_set = VariableLengthCalibrationSet.build(
        train_ids=train_ids,
        length_counts=length_counts,
        total_chunks=128,
        seed=seed,
        pad_length=MACA_PAD_LENGTH,
    )

    snapshot_dir, root_config, weight_map = load_root_config(model_id)
    text_config = build_text_config(root_config)
    store = WeightStore(model_id, snapshot_dir, weight_map)

    calibration = run_maca_calibration(
        store=store,
        text_config=text_config,
        chunks=calib_set.chunks,
        actual_lengths=calib_set.actual_lengths,
        device=device,
        dtype=dtype,
    )
    del store
    return calibration


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print(f"Loading model config from {args.model_id}...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)

    print(f"Loading calibration cache from {args.cache_path}...", flush=True)
    calibration_tuple = load_cache(args.cache_path, args.model_id)
    standard_calibration = calibration_tuple[0]

    print("Loading test data...", flush=True)
    _tokenizer, _calib_chunks, test_ids, _calib_info, _eval_info = load_gptq_standard_data(args.model_id)

    index_payload = root_config.get("index_payload", {})
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload.get("metadata", {}).get("total_size", 0))
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    configs = build_depth_aware_configs()
    requested_plans = resolve_requested_plans(args.plans)

    # Load existing results for resume
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "approach": "Depth-Aware Layer Precision",
            "hypothesis": "Qwen3.5-35B-A3B: deeper layers are MORE sensitive; assign FP8 to deepest layers",
            "sensitivity_ranking": SENSITIVITY_RANKED_LAYERS,
            "references": {
                "maca_uniform_4k": {"ppl": 6.567582, "mem_gb": 27.934},
                "joint_w1w2_with_topup": {"ppl": 6.572458, "mem_gb": 27.934},
                "restore_top3_layers_moe_bf16": {"ppl": 6.571202, "mem_gb": 30.856},
                "restore_layer39_moe_bf16": {"ppl": 6.571918, "mem_gb": 28.893},
            },
        },
        "results": {},
    }

    if args.output_json.exists():
        try:
            with args.output_json.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if isinstance(existing, dict):
                payload["results"] = {**existing.get("results", {}), **payload["results"]}
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            pass
    atomic_json_dump(args.output_json, payload)

    # Build MaCa calibration once if needed
    maca_calibration = None

    for config in configs:
        if requested_plans and config.name not in requested_plans:
            continue

        existing_row = payload["results"].get(config.name)
        if isinstance(existing_row, dict) and "ppl" in existing_row:
            print(f"[skip] {config.name} already present", flush=True)
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {config.name}", flush=True)
        print(f"Description: {config.description}", flush=True)
        print(f"FP8 layers: {sorted(config.fp8_layers)}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        # Choose calibration
        if config.use_maca:
            if maca_calibration is None:
                maca_calibration = build_maca_calibration(args.model_id, args.seed, device, dtype)
            calibration = maca_calibration
        else:
            calibration = standard_calibration

        # Build masks
        w1_masks, w2_masks = build_depth_aware_masks(calibration, text_config, config.fp8_layers)

        # Build plan
        plan = build_plan_from_masks(
            config.name,
            config.description,
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        )

        print(f"Plan: {plan.memory_gb:.3f} GB, fp8_weights={plan.fp8_weights}", flush=True)

        # Evaluate
        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time

        payload["results"][config.name] = {
            "ppl": round(ppl, 6),
            "memory_gb": round(plan.memory_gb, 3),
            "fp8_layers": sorted(config.fp8_layers),
            "use_maca": config.use_maca,
            "elapsed_s": round(elapsed, 1),
        }
        atomic_json_dump(args.output_json, payload)
        print(f"PPL: {ppl:.6f} | Memory: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

    # Print summary
    print("\n" + "="*80, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"{'Config':<35} {'PPL':>10} {'Mem (GB)':>10} {'FP8 Layers':>15}", flush=True)
    print("-"*80, flush=True)
    refs = payload["metadata"]["references"]
    for name, row in refs.items():
        print(f"  [ref] {name:<29} {row['ppl']:>10.6f} {row['mem_gb']:>10.3f}", flush=True)
    print("-"*80, flush=True)
    for name, row in sorted(payload["results"].items(), key=lambda x: x[1].get("ppl", 99)):
        if isinstance(row, dict) and "ppl" in row:
            layers_str = str(row.get("fp8_layers", []))[:15]
            print(f"  {name:<35} {row['ppl']:>10.6f} {row['memory_gb']:>10.3f} {layers_str:>15}", flush=True)

    print(f"\nSaved -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

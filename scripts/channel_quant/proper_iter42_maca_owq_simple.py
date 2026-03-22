#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""
Iteration 42: MaCa + OWQ-Inspired Topup (Outlier-Aware Quantization)

Technique: Increase topup fraction to 7% (vs 5% in Iter29) to better protect outlier weights.
Foundation: MaCa uniform 4K calibration (proven best from Iter29).
Expected improvement: 0.001-0.003 PPL over Iter29 (6.567582 → 6.565-6.567).

Rationale: OWQ detects outliers and keeps them in higher precision. By increasing topup
fraction from 5% to 7%, we allocate more FP8 budget to protect high-variance weights
that are likely to be outliers.
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

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_eval import evaluate_plan
from proper_iter25_learned_correction import maybe_slice_eval_chunks, resolve_requested_plans
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
    summarize_calibration_artifacts,
)
from proper_iter28_maca_plus_correction import BaseVariant, evaluate_and_record_compound_plan, fit_scalar_corrections_for_variant
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter42_maca_owq_simple.json"
TOTAL_CALIBRATION_CHUNKS = 128


@dataclass(frozen=True)
class CalibrationMixConfig:
    name: str
    length_counts: tuple[tuple[int, int], ...]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on WikiText-2 evaluation chunks for smoke tests; 0 means full 145-chunk test.",
    )
    parser.add_argument(
        "--topup-fraction",
        type=float,
        default=0.07,
        help="Topup fraction for OWQ-inspired precision assignment (default 0.07 = 7%)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_mix_config(mix: CalibrationMixConfig) -> None:
    total_count = sum(count for _, count in mix.length_counts)
    if total_count != TOTAL_CALIBRATION_CHUNKS:
        raise ValueError(
            f"Mix {mix.name} has {total_count} total chunks, expected {TOTAL_CALIBRATION_CHUNKS}"
        )


def serialize_length_counts(length_counts: tuple[tuple[int, int], ...]) -> list[dict[str, int]]:
    return [{"length": int(length), "count": int(count)} for length, count in length_counts]


def build_variable_length_calibration_set(
    tokenizer: Any,
    train_ids: torch.Tensor,
    seed: int,
    mix: CalibrationMixConfig,
) -> VariableLengthCalibrationSet:
    validate_mix_config(mix)
    rng = random.Random(f"iter42:{seed}:{mix.name}")
    padded_chunks: list[torch.Tensor] = []
    actual_lengths: list[int] = []
    requested_lengths: list[int] = []
    effective_length_counts: dict[int, int] = {}
    requested_length_counts: dict[int, int] = {}

    for requested_length, count in mix.length_counts:
        requested_length = int(requested_length)
        count = int(count)
        effective_length = min(requested_length, MACA_PAD_LENGTH)
        max_start = int(train_ids.shape[1]) - requested_length - 1
        if max_start < 0:
            raise RuntimeError(f"WikiText-2 train is too short for requested calibration length {requested_length}")

        requested_length_counts[requested_length] = requested_length_counts.get(requested_length, 0) + count
        effective_length_counts[effective_length] = effective_length_counts.get(effective_length, 0) + count
        for _ in range(count):
            start = rng.randint(0, max_start)
            sample = train_ids[:, start : start + requested_length]
            padded = torch.zeros((1, MACA_PAD_LENGTH), dtype=torch.long)
            padded[:, :effective_length] = sample[:, :effective_length]
            padded_chunks.append(padded)
            actual_lengths.append(effective_length)
            requested_lengths.append(requested_length)

    chunks = torch.cat(padded_chunks, dim=0).contiguous()
    actual_lengths_tensor = torch.as_tensor(actual_lengths, dtype=torch.long)
    requested_lengths_tensor = torch.as_tensor(requested_lengths, dtype=torch.long)
    actual_token_total = int(actual_lengths_tensor.sum().item())
    requested_token_total = int(requested_lengths_tensor.sum().item())
    padded_token_total = int(chunks.numel())
    metadata = {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "train",
        "tokenizer": tokenizer.name_or_path,
        "base_seed": int(seed),
        "sampling_seed": f"iter42:{seed}:{mix.name}",
        "config_name": mix.name,
        "description": mix.description,
        "requested_length_counts": serialize_length_counts(mix.length_counts),
        "effective_length_counts": [{"length": int(length), "count": int(count)} for length, count in sorted(effective_length_counts.items())],
        "samples": int(chunks.shape[0]),
        "sample_shape": [int(chunks.shape[0]), int(chunks.shape[1])],
        "actual_token_total": actual_token_total,
        "requested_token_total": requested_token_total,
        "padded_token_total": padded_token_total,
        "real_token_fraction": round(float(actual_token_total) / float(max(padded_token_total, 1)), 6),
        "truncated_requested_lengths": sorted({int(length) for length in requested_lengths if int(length) > MACA_PAD_LENGTH}),
        "train_tokens": int(train_ids.shape[1]),
        "standard_reference_tokens": int(SEQLEN * chunks.shape[0]),
    }
    return VariableLengthCalibrationSet(chunks=chunks, actual_lengths=actual_lengths_tensor, metadata=metadata)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"Loading model config from {args.model_id}...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    model_config = build_text_config(root_config)
    weight_store = WeightStore(args.model_id, snapshot_dir, weight_map)
    
    # Get total model size for non-expert bytes calculation
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    
    print(f"Loading evaluation data and tokenizer...", flush=True)
    tokenizer, _standard_calib_chunks, test_ids_full, eval_info_base, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}
    
    print(f"Loading WikiText-2 training data for MaCa calibration...", flush=True)
    train_ids = load_wikitext_train_ids(tokenizer)
    
    # Use MaCa uniform 4K calibration (proven best from Iter29)
    mix = CalibrationMixConfig(
        name="maca_uniform_4k_owq",
        length_counts=((4096, 128),),
        description="MaCa uniform 4K with OWQ-inspired topup (7% vs 5%)",
    )
    
    print(f"Building variable-length calibration set...", flush=True)
    calibration_set = build_variable_length_calibration_set(tokenizer, train_ids, args.seed, mix)
    
    print(f"Running MaCa calibration with {len(calibration_set.chunks)} chunks...", flush=True)
    start_time = time.time()
    calibration_data = run_maca_calibration(
        store=weight_store,
        config=model_config,
        calib_chunks=calibration_set.chunks,
        actual_lengths=calibration_set.actual_lengths,
        device=device,
        dtype=dtype,
    )
    calib_time = time.time() - start_time
    print(f"MaCa calibration completed in {calib_time:.1f}s", flush=True)
    
    # Build joint W1/W2 masks with OWQ-inspired topup (7% instead of 5%)
    print(f"Building joint W1/W2 masks with OWQ-inspired topup ({args.topup_fraction:.1%})...", flush=True)
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration=calibration_data,
        metric_cache=calibration_data.activation_cache,
        config=model_config,
        topup_fraction=args.topup_fraction,
    )
    
    # Build quantization plan
    print(f"Building quantization plan...", flush=True)
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, model_config)
    plan = build_plan_from_masks(
        name=mix.name,
        description=mix.description,
        config=model_config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_masks,
        w2_channel_masks=w2_masks,
    )
    
    # Evaluate
    print(f"Evaluating quantization plan...", flush=True)
    start_eval = time.time()
    result = evaluate_and_record_plan(
        plan=plan,
        eval_data=test_ids,
        model_id=args.model_id,
        config=model_config,
        store=weight_store,
        device=device,
        dtype=dtype,
    )
    eval_time = time.time() - start_eval
    
    # Prepare output
    output = {
        'metadata': {
            'model': args.model_id,
            'device': str(device),
            'dtype': args.dtype,
            'seed': args.seed,
            'technique': 'MaCa + OWQ-Inspired Topup',
            'description': f'MaCa uniform 4K calibration with {args.topup_fraction:.1%} topup fraction (OWQ-inspired)',
            'reference': 'OWQ concept: allocate more FP8 budget to protect outlier weights',
        },
        'calibration': {
            'type': 'MaCa Uniform 4K',
            'chunks': len(calibration_set.chunks),
            'chunk_length': 4096,
            'time_s': calib_time,
            'metadata': calibration_set.metadata,
        },
        'quantization': {
            'mask_builder': 'joint_w1w2_with_topup',
            'topup_fraction': args.topup_fraction,
        },
        'evaluation': {
            'ppl': result.get('ppl', 0.0),
            'nll': result.get('nll', 0.0),
            'time_s': eval_time,
            'chunks': len(test_ids),
        },
        'result': result,
    }
    
    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(output, args.output_json)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Iteration 42: MaCa + OWQ-Inspired Topup", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"PPL: {output['evaluation']['ppl']:.6f}", flush=True)
    print(f"NLL: {output['evaluation']['nll']:.6f}", flush=True)
    print(f"Topup fraction: {args.topup_fraction:.1%}", flush=True)
    print(f"Calibration time: {calib_time:.1f}s", flush=True)
    print(f"Evaluation time: {eval_time:.1f}s", flush=True)
    print(f"Results saved to: {args.output_json}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 34: MaCa Calibration + Topup Fraction Sweep.

Current best: maca_uniform_4k = 6.5676 PPL @ 27.934 GB (8% topup)

Hypothesis: The optimal topup fraction with MaCa calibration may be different from 8%.
Higher topup = more FP8 for medium-tier experts = potentially better PPL at higher memory.

Configs:
  1. maca_topup_10pct: MaCa uniform_4k + 10% topup → ~28.2 GB
  2. maca_topup_12pct: MaCa uniform_4k + 12% topup → ~28.4 GB
  3. maca_topup_15pct: MaCa uniform_4k + 15% topup → ~28.7 GB
  4. maca_topup_06pct: MaCa uniform_4k + 6% topup → ~27.7 GB (lower budget)
  5. maca_topup_04pct: MaCa uniform_4k + 4% topup → ~27.5 GB (even lower)

References:
  - maca_uniform_4k (8%): 6.5676 @ 27.934 GB
  - joint_w1w2_with_topup (8%): 6.5725 @ 27.934 GB
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter07 import build_joint_with_topup_masks
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
)
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter34_maca_topup_sweep.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"

TOPUP_FRACTIONS = [0.04, 0.06, 0.10, 0.12, 0.15]


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


def build_maca_calibration(model_id: str, tokenizer: Any, seed: int, device: torch.device, dtype: torch.dtype, snapshot_dir: Any, weight_map: Any, text_config: Any) -> Any:
    """Build MaCa uniform_4k calibration (same as iter29 best)."""
    import random as _random
    print("Building MaCa uniform_4k calibration (128 × 4096-token chunks)...", flush=True)
    train_ids = load_wikitext_train_ids(tokenizer)

    # Uniform 4k: 128 chunks of 4096 tokens each (same as iter29 maca_uniform_4k)
    rng = _random.Random(f"iter34:{seed}:maca_uniform_4k")
    padded_chunks = []
    actual_lengths = []
    for _ in range(128):
        max_start = int(train_ids.shape[1]) - 4096 - 1
        start = rng.randint(0, max_start)
        sample = train_ids[:, start : start + 4096]
        padded = torch.zeros((1, MACA_PAD_LENGTH), dtype=torch.long)
        padded[:, :4096] = sample[:, :4096]
        padded_chunks.append(padded)
        actual_lengths.append(4096)

    chunks = torch.cat(padded_chunks, dim=0).contiguous()
    actual_lengths_tensor = torch.as_tensor(actual_lengths, dtype=torch.long)

    store = WeightStore(model_id, snapshot_dir, weight_map)
    calibration = run_maca_calibration(
        store=store,
        config=text_config,
        calib_chunks=chunks,
        actual_lengths=actual_lengths_tensor,
        device=device,
        dtype=dtype,
    )
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return calibration


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print(f"Loading model config from {args.model_id}...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)

    print("Loading test data...", flush=True)
    tokenizer, _calib_chunks, test_ids, _calib_info, _eval_info = load_gptq_standard_data(args.model_id)

    index_payload = root_config.get("index_payload", {})
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload.get("metadata", {}).get("total_size", 0))
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    requested_plans = resolve_requested_plans(args.plans)

    # Load existing results for resume
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "approach": "MaCa Calibration + Topup Fraction Sweep",
            "hypothesis": "Optimal topup fraction with MaCa calibration may differ from 8%",
            "topup_fractions_tested": TOPUP_FRACTIONS,
            "references": {
                "maca_uniform_4k_8pct": {"ppl": 6.567582, "mem_gb": 27.934},
                "joint_w1w2_with_topup_8pct": {"ppl": 6.572458, "mem_gb": 27.934},
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

    # Build MaCa calibration once
    maca_calibration = build_maca_calibration(args.model_id, tokenizer, args.seed, device, dtype, snapshot_dir, weight_map, text_config)
    metric_cache = maca_calibration.activation_cache

    for topup_frac in TOPUP_FRACTIONS:
        config_name = f"maca_topup_{int(topup_frac * 100):02d}pct"

        if requested_plans and config_name not in requested_plans:
            continue

        existing_row = payload["results"].get(config_name)
        if isinstance(existing_row, dict) and "ppl" in existing_row:
            print(f"[skip] {config_name} already present", flush=True)
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {config_name} (topup_fraction={topup_frac:.2f})", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        # Build masks with MaCa calibration and this topup fraction
        w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
            maca_calibration, metric_cache, text_config, topup_frac
        )

        # Build plan
        plan = build_plan_from_masks(
            config_name,
            f"MaCa uniform_4k calibration + {topup_frac*100:.0f}% topup for medium-tier experts",
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

        payload["results"][config_name] = {
            "ppl": round(ppl, 6),
            "memory_gb": round(plan.memory_gb, 3),
            "topup_fraction": topup_frac,
            "elapsed_s": round(elapsed, 1),
        }
        atomic_json_dump(args.output_json, payload)
        print(f"PPL: {ppl:.6f} | Memory: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

    # Print summary
    print("\n" + "="*80, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*80, flush=True)
    refs = payload["metadata"]["references"]
    for name, row in refs.items():
        print(f"  [ref] {name:<35} ppl={row['ppl']:.6f} mem={row['mem_gb']:.3f}GB", flush=True)
    print("-"*80, flush=True)
    for name, row in sorted(payload["results"].items(), key=lambda x: x[1].get("ppl", 99)):
        if isinstance(row, dict) and "ppl" in row:
            frac = row.get("topup_fraction", "?")
            print(f"  {name:<35} ppl={row['ppl']:.6f} mem={row['memory_gb']:.3f}GB topup={frac}", flush=True)

    print(f"\nSaved -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

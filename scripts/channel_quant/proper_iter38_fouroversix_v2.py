#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 38 v2: Four Over Six (4/6) Adaptive Block Scaling for MXFP4.

From "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling"
(arXiv:2512.02010, MIT + NVIDIA, Dec 2025)

KEY INSIGHT: MXFP4 error comes from rounding near-maximal values.
FP4 E2M1 grid: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
With M=6 (standard): values in [4.0, 6.0] round to 4.0 or 6.0 — large gap.
With M=4: values in [2.67, 4.0] round to 3.0 or 4.0 — smaller gap.
4/6 adaptively chooses M=4 or M=6 per block to minimize MSE.

This is a DROP-IN improvement to MXFP4 quantization with NO memory overhead.

Configs:
  1. maca_fouroversix:     MaCa calibration + 4/6 adaptive block scaling
  2. standard_fouroversix: Standard calibration + 4/6 adaptive block scaling
  3. maca_baseline:        MaCa calibration + standard M=6 (reference = 6.5676)
  4. standard_baseline:    Standard calibration + standard M=6 (reference = 6.5725)

Expected gain: 0.001-0.005 PPL from 4/6 adaptive scaling alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from spike1_ground_truth import MODEL_ID, EPS, E2M1_GRID, build_text_config, load_root_config, round_to_e2m1_grid


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter38_fouroversix_v2.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
MACA_CACHE_PATH = RESULTS_DIR / "iter34_maca_seed0_cache.pt"


def quantize_to_nvfp4_columns_46(weight: torch.Tensor) -> torch.Tensor:
    """MXFP4 quantization with 4/6 adaptive block scaling.
    
    For each block of 16 elements, choose M=4 or M=6 based on MSE.
    Standard: scale = absmax / 6.0
    4/6 variant: scale = absmax / 4.0 for blocks where this gives lower MSE.
    
    The E2M1 grid is [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0].
    With M=6: max representable = 6.0, gap between 4.0 and 6.0 is large.
    With M=4: max representable = 4.0, values in [2.67, 4.0] are better covered.
    """
    if weight.ndim == 1:
        weight = weight.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    if weight.shape[0] % 16 != 0:
        raise ValueError(f"Expected K dimension multiple of 16, got {tuple(weight.shape)}")
    
    blocks = weight.reshape(weight.shape[0] // 16, 16, weight.shape[1])
    absmax = blocks.abs().amax(dim=1, keepdim=True)  # [num_blocks, 1, K]
    
    # Standard M=6 quantization
    scale6 = absmax / 6.0
    normalized6 = blocks / (scale6 + EPS)
    quantized6 = round_to_e2m1_grid(normalized6)
    dequantized6 = quantized6 * scale6
    mse6 = ((blocks - dequantized6) ** 2).mean(dim=1, keepdim=True)  # [num_blocks, 1, K]
    
    # M=4 quantization
    scale4 = absmax / 4.0
    normalized4 = blocks / (scale4 + EPS)
    quantized4 = round_to_e2m1_grid(normalized4)
    dequantized4 = quantized4 * scale4
    mse4 = ((blocks - dequantized4) ** 2).mean(dim=1, keepdim=True)  # [num_blocks, 1, K]
    
    # Choose M=4 where it gives lower MSE
    use_m4 = mse4 < mse6  # [num_blocks, 1, K]
    
    # Select best quantization per block
    dequantized = torch.where(use_m4, dequantized4, dequantized6)
    dequantized = dequantized.reshape_as(weight)
    
    return dequantized.squeeze(-1) if squeeze else dequantized


def load_maca_calibration(model_id: str) -> Any:
    """Load MaCa calibration from cache."""
    if MACA_CACHE_PATH.exists():
        print(f"Loading MaCa calibration from {MACA_CACHE_PATH}...", flush=True)
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        from proper_eval import CalibrationArtifacts
        from baselines_comparison import LayerMetricBundle
        cal = torch.load(MACA_CACHE_PATH, map_location='cpu', weights_only=False)
        if isinstance(cal, dict):
            cal = cal.get('calibration', cal)
        print(f"  Loaded: {len(cal.routing_counts)} layers, {len(cal.activation_cache)} activation layers", flush=True)
        return cal
    else:
        print(f"WARNING: MaCa cache not found at {MACA_CACHE_PATH}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print("Loading model config...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)

    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    print("Loading tokenizer and test data...", flush=True)
    tokenizer, standard_calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)

    # Load calibrations
    standard_calibration, _, _ = load_cache(args.cache_path, args.model_id)
    maca_calibration = load_maca_calibration(args.model_id)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Load existing results for resume
    payload: dict[str, Any] = {
        "metadata": {
            "approach": "fouroversix_mxfp4",
            "paper": "arXiv:2512.02010 (MIT + NVIDIA, Dec 2025)",
            "references": {
                "maca_uniform_4k": {"ppl": 6.567582, "mem_gb": 27.934},
                "joint_w1w2_with_topup": {"ppl": 6.572458, "mem_gb": 27.934},
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

    requested_plans = set(args.plans.split(",")) if args.plans else None

    configs = [
        ("maca_baseline",        maca_calibration,     False, "MaCa calibration + standard M=6 (reference = 6.5676)"),
        ("maca_fouroversix",     maca_calibration,     True,  "MaCa calibration + 4/6 adaptive block scaling"),
        ("standard_fouroversix", standard_calibration, True,  "Standard calibration + 4/6 adaptive block scaling"),
        ("standard_baseline",    standard_calibration, False, "Standard calibration + standard M=6 (reference = 6.5725)"),
    ]

    for plan_name, calibration, use_46, description in configs:
        if requested_plans and plan_name not in requested_plans:
            continue
        if calibration is None:
            print(f"[skip] {plan_name}: calibration not available", flush=True)
            continue

        existing_row = payload["results"].get(plan_name)
        if isinstance(existing_row, dict) and "ppl" in existing_row:
            print(f"[skip] {plan_name} already present", flush=True)
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {plan_name} (4/6={use_46})", flush=True)
        print(f"Description: {description}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        # Monkey-patch quantize_to_nvfp4_columns if using 4/6
        import spike1_ground_truth as sgt
        import baselines_comparison as bc
        original_fn = sgt.quantize_to_nvfp4_columns
        if use_46:
            sgt.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46
            bc.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46

        try:
            metric_cache = calibration.activation_cache
            w1_masks, w2_masks, _meta = build_joint_with_topup_masks(
                calibration, metric_cache, text_config, JOINT_MEDIUM_TOPUP_FRACTION
            )
            plan = build_plan_from_masks(
                plan_name, description, text_config,
                non_expert_bytes, total_expert_elems, w1_masks, w2_masks
            )
            print(f"Plan: {plan.memory_gb:.3f} GB, fp8_weights={plan.fp8_weights}", flush=True)
            ppl, _nll, _nsamples = evaluate_plan(
                plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype
            )
        finally:
            sgt.quantize_to_nvfp4_columns = original_fn
            bc.quantize_to_nvfp4_columns = original_fn

        elapsed = time.time() - start_time

        payload["results"][plan_name] = {
            "ppl": round(ppl, 6),
            "memory_gb": round(plan.memory_gb, 3),
            "description": description,
            "use_46": use_46,
            "elapsed_s": round(elapsed, 1),
        }
        atomic_json_dump(args.output_json, payload)
        print(f"  PPL: {ppl:.6f} | Mem: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

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
            print(f"  {name:<35} ppl={row['ppl']:.6f} mem={row['memory_gb']:.3f}GB 4/6={row['use_46']}", flush=True)
    print(f"\nSaved -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

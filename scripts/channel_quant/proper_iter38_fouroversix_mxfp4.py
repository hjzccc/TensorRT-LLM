#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 38: Four Over Six (4/6) Adaptive Block Scaling for MXFP4.

From "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling"
(arXiv:2512.02010, MIT + NVIDIA, Dec 2025)

KEY INSIGHT: MXFP4 error comes from rounding near-maximal values (around 5).
FP4 has no representable values between 66.6% and 100% of block maximum.
By adaptively choosing scale=4 vs scale=6 per block based on MSE, we reduce this error.

Our current MXFP4 uses scale = absmax / 6.0 (M=6).
4/6 variant: for each block, choose M=4 or M=6 based on which gives lower MSE.

This is a DROP-IN improvement to MXFP4 quantization with no memory overhead.

Configs:
  1. maca_fouroversix:     MaCa calibration + 4/6 adaptive block scaling
  2. standard_fouroversix: Standard calibration + 4/6 adaptive block scaling
  3. maca_baseline:        MaCa calibration + standard M=6 (reference)
  4. standard_baseline:    Standard calibration + standard M=6 (reference)
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
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
)
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config, quantize_to_nvfp4_columns, round_to_e2m1_grid, EPS


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter38_fouroversix_mxfp4.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
MACA_CACHE_PATH = RESULTS_DIR / "proper_iter29_maca_uniform4k_cache.pt"


def quantize_to_nvfp4_columns_46(weight: torch.Tensor) -> torch.Tensor:
    """MXFP4 quantization with 4/6 adaptive block scaling.
    
    For each block of 16 elements, choose M=4 or M=6 based on MSE.
    Standard: scale = absmax / 6.0
    4/6 variant: scale = absmax / 4.0 for blocks where this gives lower MSE
    """
    if weight.ndim == 1:
        weight = weight.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    if weight.shape[0] % 16 != 0:
        raise ValueError(f"Expected K dimension multiple of 16, got {tuple(weight.shape)}")
    
    blocks = weight.reshape(weight.shape[0] // 16, 16, weight.shape[1])
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    
    # Standard M=6 quantization
    scale6 = absmax / 6.0
    normalized6 = blocks / (scale6 + EPS)
    quantized6 = round_to_e2m1_grid(normalized6)
    dequantized6 = quantized6 * scale6
    mse6 = ((blocks - dequantized6) ** 2).mean(dim=1, keepdim=True)
    
    # M=4 quantization
    scale4 = absmax / 4.0
    normalized4 = blocks / (scale4 + EPS)
    quantized4 = round_to_e2m1_grid(normalized4)
    dequantized4 = quantized4 * scale4
    mse4 = ((blocks - dequantized4) ** 2).mean(dim=1, keepdim=True)
    
    # Choose M=4 where it gives lower MSE
    use_m4 = mse4 < mse6  # [num_blocks, 1, K]
    
    # Select best quantization per block
    dequantized = torch.where(use_m4, dequantized4, dequantized6)
    dequantized = dequantized.reshape_as(weight)
    
    return dequantized.squeeze(-1) if squeeze else dequantized


def build_maca_calibration(model_id: str, seed: int, device: torch.device, dtype: torch.dtype) -> Any:
    """Build MaCa uniform_4k calibration."""
    if MACA_CACHE_PATH.exists():
        print(f"Loading MaCa cache from {MACA_CACHE_PATH}...", flush=True)
        calib, _, _ = load_cache(MACA_CACHE_PATH, model_id)
        if calib is not None:
            return calib
    
    print("Building MaCa uniform_4k calibration (128 × 4096-token chunks)...", flush=True)
    train_ids = load_wikitext_train_ids(model_id)
    length_counts = ((4096, 128),)
    calib_set = VariableLengthCalibrationSet.build(
        train_ids=train_ids,
        length_counts=length_counts,
        total_chunks=128,
        seed=seed,
    )
    snapshot_dir, root_config, weight_map = load_root_config(model_id)
    text_config = build_text_config(root_config)
    store = WeightStore(model_id, snapshot_dir, weight_map)
    calibration = run_maca_calibration(calib_set, text_config, store, device, dtype, MACA_PAD_LENGTH)
    torch.save({"calibration": calibration, "model_id": model_id}, MACA_CACHE_PATH)
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print("Loading model config...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)

    import json as _json
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = _json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    print("Loading tokenizer and test data...", flush=True)
    tokenizer, standard_calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)

    # Load calibrations
    standard_calibration, _, _ = load_cache(args.cache_path, args.model_id)
    maca_calibration = build_maca_calibration(args.model_id, args.seed, device, dtype)

    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {}

    requested_plans = set(args.plans.split(",")) if args.plans else None

    configs = [
        ("maca_baseline",        maca_calibration,     False, "MaCa calibration + standard M=6 (reference)"),
        ("maca_fouroversix",     maca_calibration,     True,  "MaCa calibration + 4/6 adaptive block scaling"),
        ("standard_baseline",    standard_calibration, False, "Standard calibration + standard M=6 (reference)"),
        ("standard_fouroversix", standard_calibration, True,  "Standard calibration + 4/6 adaptive block scaling"),
    ]

    for plan_name, calibration, use_46, description in configs:
        if requested_plans and plan_name not in requested_plans:
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {plan_name} (4/6={use_46})", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()

        # Monkey-patch quantize_to_nvfp4_columns if using 4/6
        import spike1_ground_truth as sgt
        import baselines_comparison as bc
        if use_46:
            original_fn = sgt.quantize_to_nvfp4_columns
            sgt.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46
            bc.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46
        
        try:
            w1_masks, w2_masks = build_joint_with_topup_masks(calibration, text_config, JOINT_MEDIUM_TOPUP_FRACTION)
            plan = build_plan_from_masks(plan_name, description, text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
            ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        finally:
            if use_46:
                sgt.quantize_to_nvfp4_columns = original_fn
                bc.quantize_to_nvfp4_columns = original_fn

        elapsed = time.time() - start_time

        results[plan_name] = {
            "ppl": ppl,
            "memory_gb": round(plan.memory_gb, 3),
            "description": description,
            "use_46": use_46,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  PPL: {ppl:.6f} | Mem: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

        atomic_json_dump({"results": results, "metadata": {"approach": "fouroversix_mxfp4"}}, args.output_json)

    print(f"\nDone. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

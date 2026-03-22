#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 34: MaCa Seed Ensemble Calibration

Key finding: iter29 best (seed=0, 128×4096) = PPL 6.5676
             iter31 control (same config, different run) = PPL 6.5701
             Variance = 0.0025 PPL — calibration seed matters significantly.

Hypothesis: Averaging Hessians across multiple calibration seeds reduces variance
and produces a more representative sensitivity estimate.

Approach:
1. Run calibration with 3 different seeds (0, 1, 2) using 128×4096 chunks each
2. Average the sensitivity scores (w1_pair_scores, w2_channel_scores) across seeds
3. Build masks from the averaged scores
4. Compare against single-seed best

Also test: seed selection — run 3 seeds, pick the one that gives best masks
(evaluated on a small proxy metric, not full PPL).

Expected gain: 0.001-0.003 PPL (reduced variance in sensitivity estimation)
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    atomic_json_dump,
    build_position_context,
    collect_layer_calibration,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    finalize_layer_metrics,
    init_expert_moment_state,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
)
from proper_iter01 import build_plan_from_masks, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter21_serq_salient import load_reference_rows
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter34_maca_seed_ensemble.json"

CHUNK_LENGTH = 4096
CHUNKS_PER_SEED = 128
SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="")
    return parser.parse_args()


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def load_wikitext_train_ids(tokenizer: Any) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])  # type: ignore[index]
    return tokenizer(text, return_tensors="pt").input_ids


def build_maca_chunks(tokenizer: Any, n: int, chunk_length: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    train_ids = load_wikitext_train_ids(tokenizer)
    max_start = int(train_ids.shape[1]) - chunk_length - 1
    chunks = []
    for _ in range(n):
        start = rng.randint(0, max_start)
        chunks.append(train_ids[:, start : start + chunk_length])
    return torch.cat(chunks, dim=0).contiguous()


def run_maca_calibration_seed(
    store: WeightStore,
    config: Any,
    chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> CalibrationArtifacts:
    n = chunks.shape[0]
    print(f"\n=== MaCa Calibration (seed={seed}, {n}×{chunks.shape[1]}-token) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[seed{seed}] layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[e], "fp4") for e in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[e], "fp4") for e in range(config.num_experts)], dim=0)
        state = init_expert_moment_state(config, device)

        for chunk_idx in range(inps.shape[0]):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = collect_layer_calibration(mlp_input, moe_tensors, fp4_gate_up, fp4_down, config, state)
            outs[chunk_idx] = residual + moe_out
            print(f"[seed{seed}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        bundle, layer_w1, layer_w2, layer_mc = finalize_layer_metrics(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        routing_counts[layer_idx] = bundle.routing_counts
        activation_cache[layer_idx] = bundle
        mxmoe_w1_deltas[layer_idx] = layer_w1
        mxmoe_w2_deltas[layer_idx] = layer_w2
        mc_moe_scores[layer_idx] = layer_mc

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return CalibrationArtifacts(
        routing_counts=routing_counts,
        activation_cache=activation_cache,
        mxmoe_w1_deltas=mxmoe_w1_deltas,
        mxmoe_w2_deltas=mxmoe_w2_deltas,
        mc_moe_scores=mc_moe_scores,
    )


def average_calibrations(calibrations: list[CalibrationArtifacts]) -> CalibrationArtifacts:
    """Average sensitivity scores across multiple calibration runs."""
    n = len(calibrations)
    assert n > 0

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in calibrations[0].routing_counts.keys():
        # Average routing counts
        routing_counts[layer_idx] = torch.stack([c.routing_counts[layer_idx] for c in calibrations]).float().mean(dim=0).long()

        # Average w1_pair_scores and w2_channel_scores
        w1_scores = torch.stack([c.activation_cache[layer_idx].w1_pair_scores for c in calibrations]).mean(dim=0)
        w2_scores = torch.stack([c.activation_cache[layer_idx].w2_channel_scores for c in calibrations]).mean(dim=0)
        activation_cache[layer_idx] = LayerMetricBundle(
            routing_counts=routing_counts[layer_idx],
            w1_pair_scores=w1_scores,
            w2_channel_scores=w2_scores,
        )

        # Average MxMoE deltas
        mxmoe_w1_deltas[layer_idx] = torch.stack([c.mxmoe_w1_deltas[layer_idx] for c in calibrations]).mean(dim=0)
        mxmoe_w2_deltas[layer_idx] = torch.stack([c.mxmoe_w2_deltas[layer_idx] for c in calibrations]).mean(dim=0)
        mc_moe_scores[layer_idx] = torch.stack([c.mc_moe_scores[layer_idx] for c in calibrations]).mean(dim=0)

    return CalibrationArtifacts(
        routing_counts=routing_counts,
        activation_cache=activation_cache,
        mxmoe_w1_deltas=mxmoe_w1_deltas,
        mxmoe_w2_deltas=mxmoe_w2_deltas,
        mc_moe_scores=mc_moe_scores,
    )


def main() -> None:
    args = parse_args()
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

    reference_rows = load_reference_rows(args.output_json)
    requested_plans = resolve_requested_plans(args.plans)
    results: dict[str, Any] = {}

    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    # =========================================================================
    # Run calibration for each seed
    # =========================================================================
    calibrations = []
    for seed in SEEDS:
        cache_path = RESULTS_DIR / f"iter34_maca_seed{seed}_cache.pt"
        if cache_path.exists():
            print(f"Loading cached calibration for seed={seed}", flush=True)
            calib = torch.load(cache_path, weights_only=False, map_location='cpu')
            # Move to device
            calibrations.append(calib)
        else:
            chunks = build_maca_chunks(tokenizer, CHUNKS_PER_SEED, CHUNK_LENGTH, seed)
            calib = run_maca_calibration_seed(store, text_config, chunks, device, dtype, seed)
            torch.save(calib, cache_path)
            print(f"Saved calibration for seed={seed} to {cache_path}", flush=True)
            calibrations.append(calib)

    # =========================================================================
    # Build averaged calibration
    # =========================================================================
    print("\n=== Building averaged calibration ===", flush=True)
    avg_calibration = average_calibrations(calibrations)

    # =========================================================================
    # Evaluate configs
    # =========================================================================
    configs_to_eval = [
        ("maca_ensemble_joint_topup", avg_calibration, "MaCa 3-seed ensemble (avg) + joint W1/W2 topup"),
        ("maca_seed0_joint_topup", calibrations[0], "MaCa seed=0 (128×4096) + joint W1/W2 topup (reference)"),
        ("maca_seed1_joint_topup", calibrations[1], "MaCa seed=1 (128×4096) + joint W1/W2 topup"),
        ("maca_seed2_joint_topup", calibrations[2], "MaCa seed=2 (128×4096) + joint W1/W2 topup"),
    ]

    for plan_name, calibration, description in configs_to_eval:
        if requested_plans and plan_name not in requested_plans:
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {plan_name}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()
        w1_masks, w2_masks = build_joint_with_topup_masks(calibration, text_config, JOINT_MEDIUM_TOPUP_FRACTION)
        plan = build_plan_from_masks(plan_name, description, text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time

        results[plan_name] = {
            "ppl": round(ppl, 4),
            "memory_gb": round(plan.memory_gb, 3),
            "fp8_weights": int(plan.fp8_weights),
            "w1_fraction": round(total_pair_fraction(w1_masks, text_config), 4),
            "w2_fraction": round(total_channel_fraction(w2_masks, text_config), 4),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"PPL: {ppl:.4f} | Memory: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": 296960,
            "seqlen": SEQLEN,
            "nsamples": 145,
            "dtype": args.dtype,
            "approach": "MaCa Seed Ensemble Calibration",
            "hypothesis": "Averaging Hessians across multiple calibration seeds reduces variance",
            "seeds": SEEDS,
            "chunks_per_seed": CHUNKS_PER_SEED,
            "chunk_length": CHUNK_LENGTH,
            "reference_rows": reference_rows,
        },
        "results": results,
    }
    atomic_json_dump(args.output_json, payload)
    print(f"\nResults saved to {args.output_json}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    for i, (name, result) in enumerate(sorted(results.items(), key=lambda x: x[1].get("ppl", float("inf")))[:5], 1):
        print(f"{i}. {name:50s} PPL={result.get('ppl','N/A')} Mem={result.get('memory_gb','N/A')} GB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 35: iMatrix-Weighted Calibration

Key insight from llm-compressor RFC (Mar 2026):
  "iMatrix is additive. It improves every method it's composed with and never degrades quality."
  "iMatrix closes 92% of the MSE gap at W4."

The iMatrix technique weights quantization error by E[x²] per input channel:
  err_imatrix = sum(E[x²] * |Q(w) - w|^p)
  err_standard = sum(|Q(w) - w|^p)

Our current metric uses kurtosis as the weighting:
  w2_scores[expert, channel] = sum_k(|W2_diff[channel, k]| * kurtosis[k])

iMatrix would use E[x²] instead:
  w2_scores_imatrix[expert, channel] = sum_k(|W2_diff[channel, k]| * E[x²][k])

Where E[x²] = inter_sum2 / count (mean squared intermediate activation).

This is different from kurtosis because:
- Kurtosis measures tail-heaviness (outlier sensitivity)
- E[x²] measures overall activation magnitude (importance)
- Channels with high E[x²] carry more signal and need more careful quantization

Also test: combined metric = E[x²] * kurtosis (captures both magnitude and outlier sensitivity)

Expected gain: 0.001-0.003 PPL (better channel importance weighting)
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
    ExpertMomentState,
    atomic_json_dump,
    build_position_context,
    collect_layer_calibration,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    finalize_layer_metrics,
    init_expert_moment_state,
    kurtosis_from_moments,
    layer_type_at,
    load_gptq_standard_data,
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter35_imatrix_calibration.json"

CHUNK_LENGTH = 4096
NUM_CHUNKS = 128
SEED = 0


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


def finalize_layer_metrics_imatrix(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: ExpertMomentState,
    config: Any,
    metric: str = "imatrix",  # "imatrix", "kurtosis", "combined"
) -> tuple[LayerMetricBundle, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Modified finalize_layer_metrics with iMatrix weighting."""
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=gate_up_proj.device)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=gate_up_proj.device)
    mc_scores = torch.zeros(config.num_experts, dtype=torch.float32, device=gate_up_proj.device)

    for expert_idx in range(config.num_experts):
        count = int(state.counts[expert_idx].item())
        count_f = float(max(count, 1))

        # Compute kurtosis (existing metric)
        input_kurt = kurtosis_from_moments(
            state.input_sum1[expert_idx],
            state.input_sum2[expert_idx],
            state.input_sum3[expert_idx],
            state.input_sum4[expert_idx],
            count,
        )
        inter_kurt = kurtosis_from_moments(
            state.inter_sum1[expert_idx],
            state.inter_sum2[expert_idx],
            state.inter_sum3[expert_idx],
            state.inter_sum4[expert_idx],
            count,
        )

        # Compute E[x²] (iMatrix metric)
        input_mean_sq = (state.input_sum2[expert_idx] / count_f).clamp(min=0.0)  # E[x²] for W1 inputs
        inter_mean_sq = (state.inter_sum2[expert_idx] / count_f).clamp(min=0.0)  # E[x²] for W2 inputs

        # Select weighting
        if metric == "imatrix":
            input_weight = input_mean_sq
            inter_weight = inter_mean_sq
        elif metric == "kurtosis":
            input_weight = input_kurt
            inter_weight = inter_kurt
        elif metric == "combined":
            # Multiply kurtosis by E[x²]: captures both magnitude and outlier sensitivity
            input_weight = input_kurt * input_mean_sq
            inter_weight = inter_kurt * inter_mean_sq
        elif metric == "sqrt_imatrix":
            # E[|x|] ≈ sqrt(E[x²]) for zero-mean distributions
            input_weight = input_mean_sq.sqrt()
            inter_weight = inter_mean_sq.sqrt()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        gate_up_weight = gate_up_proj[expert_idx].float()
        gate_up_q = fp4_gate_up[expert_idx].float()
        gate_diff = gate_up_weight - gate_up_q
        pair_diff_abs = gate_diff[: config.moe_intermediate_size].abs() + gate_diff[config.moe_intermediate_size :].abs()
        w1_scores[expert_idx] = torch.matmul(pair_diff_abs, input_weight)

        down_weight = down_proj[expert_idx].float()
        down_q = fp4_down[expert_idx].float()
        down_diff = down_weight - down_q
        w2_scores[expert_idx] = torch.matmul(down_diff.abs(), inter_weight)

        weight_norm_sq = gate_up_weight.square().sum() + down_weight.square().sum()
        quant_loss_sq = (gate_up_weight - gate_up_q).square().sum() + (down_weight - down_q).square().sum()
        actnum = float(count)
        if actnum > 0.0:
            weight_norm = torch.sqrt(weight_norm_sq)
            quant_loss = torch.sqrt(quant_loss_sq)
            mc_scores[expert_idx] = float((actnum ** 1.0) * (float(weight_norm.item()) ** 1.5) * (float(quant_loss.item()) ** 2.0))

    return (
        LayerMetricBundle(
            routing_counts=state.counts.detach().cpu().to(torch.int64),
            w1_pair_scores=w1_scores.detach().cpu(),
            w2_channel_scores=w2_scores.detach().cpu(),
        ),
        torch.sqrt(state.mxmoe_w1_sq).detach().cpu() if hasattr(state, 'mxmoe_w1_sq') else torch.zeros(config.num_experts),
        torch.sqrt(state.mxmoe_w2_sq).detach().cpu() if hasattr(state, 'mxmoe_w2_sq') else torch.zeros(config.num_experts),
        mc_scores.detach().cpu(),
    )


def run_imatrix_calibration(
    store: WeightStore,
    config: Any,
    chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    metric: str = "imatrix",
) -> CalibrationArtifacts:
    """Run calibration with iMatrix-weighted sensitivity scores."""
    n = chunks.shape[0]
    print(f"\n=== iMatrix Calibration (metric={metric}, {n}×{chunks.shape[1]}-token) ===", flush=True)
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
        print(f"[{metric}] layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
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
            print(f"[{metric}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        bundle, layer_w1, layer_w2, layer_mc = finalize_layer_metrics_imatrix(
            gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config, metric=metric
        )
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

    # Build MaCa chunks (same as iter29 best)
    maca_chunks = build_maca_chunks(tokenizer, NUM_CHUNKS, CHUNK_LENGTH, SEED)

    # Run calibrations with different metrics
    metrics_to_test = ["imatrix", "combined", "kurtosis"]  # kurtosis = existing baseline

    calibrations: dict[str, CalibrationArtifacts] = {}
    for metric in metrics_to_test:
        cache_path = RESULTS_DIR / f"iter35_{metric}_cache.pt"
        if cache_path.exists():
            print(f"Loading cached calibration for metric={metric}", flush=True)
            calibrations[metric] = torch.load(cache_path, weights_only=False, map_location='cpu')
        else:
            calib = run_imatrix_calibration(store, text_config, maca_chunks, device, dtype, metric=metric)
            torch.save(calib, cache_path)
            calibrations[metric] = calib

    # Evaluate
    for metric, calibration in calibrations.items():
        plan_name = f"maca_{metric}_joint_topup"
        description = f"MaCa 128×4096 + {metric} weighting + joint W1/W2 topup"

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
            "metric": metric,
        }
        print(f"PPL: {ppl:.4f} | Memory: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": 296960,
            "seqlen": SEQLEN,
            "nsamples": 145,
            "dtype": args.dtype,
            "approach": "iMatrix-Weighted Calibration",
            "hypothesis": "E[x²] weighting better captures channel importance than kurtosis",
            "metrics_tested": metrics_to_test,
            "chunk_length": CHUNK_LENGTH,
            "num_chunks": NUM_CHUNKS,
            "seed": SEED,
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

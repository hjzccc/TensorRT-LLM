#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines_comparison import resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import SEQLEN, EvalPlan, atomic_json_dump, build_position_context, dtype_from_name, embed_chunks, layer_type_at, load_gptq_standard_data, upsert_exploration_section
from proper_iter01 import build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter21_serq_salient import load_reference_rows
from proper_iter23_subsystem_sweep import build_base_plan
from proper_iter25_learned_correction import (
    CORRECTION_PARAM_BYTES,
    AffineCorrection,
    apply_affine_correction,
    estimate_correction_memory_gb,
    evaluate_plan_with_corrections,
    filter_top_layers,
    forward_attention_to_mlp_input,
    load_json,
    maybe_slice_eval_chunks,
    prepare_quantized_layer_tensors,
    resolve_requested_plans,
    should_log_chunk,
)
from proper_iter26_maca_calibration import MACA_CHUNKS_PER_LENGTH, MACA_LENGTHS, VariableLengthCalibrationSet, build_maca_calibration_set, run_maca_calibration
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, layer_keys, load_root_config, release_tensors, shorten_layer_tensors


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter28_maca_plus_correction.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
SECTION_MARKER = "## [31] Iteration 28 - Compound MaCa + Top-10 Layer Correction"

PLAN_MACA_TOP10 = "maca_joint_topup_plus_correction_top10"
PLAN_MACA_ALL = "maca_joint_topup_plus_correction_all"
PLAN_MACA_NONE = "maca_joint_topup_no_correction"
PLAN_STANDARD_TOP10 = "standard_plus_correction_top10"

FIT_STANDARD = "standard_joint"
FIT_MACA = "maca_joint"


@dataclass(frozen=True)
class BaseVariant:
    key: str
    plan: EvalPlan
    extras: dict[str, Any]
    calibration_set: VariableLengthCalibrationSet


@dataclass
class ScalarFitMoments:
    routed_pairs: torch.Tensor
    scalar_count: torch.Tensor
    scalar_x_sum: torch.Tensor
    scalar_y_sum: torch.Tensor
    scalar_x2_sum: torch.Tensor
    scalar_xy_sum: torch.Tensor
    layer_sq_error: torch.Tensor
    layer_elem_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 28 configs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--retune-chunks",
        type=int,
        default=128,
        help="Number of standard 2048-token calibration chunks to use for the standard correction fit.",
    )
    parser.add_argument(
        "--maca-chunks-per-length",
        type=int,
        default=MACA_CHUNKS_PER_LENGTH,
        help="Number of MaCa chunks to keep per calibration length; useful for smoke runs.",
    )
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on evaluation chunks for smoke tests; 0 means full WikiText-2 test.",
    )
    return parser.parse_args()


def init_scalar_fit_moments(config: Any, device: torch.device) -> ScalarFitMoments:
    num_experts = int(config.num_experts)
    return ScalarFitMoments(
        routed_pairs=torch.zeros(num_experts, dtype=torch.int64, device=device),
        scalar_count=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_x_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_y_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_x2_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_xy_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        layer_sq_error=torch.zeros((), dtype=torch.float64, device=device),
        layer_elem_count=0,
    )


def update_scalar_fit_moments(
    stats: ScalarFitMoments,
    expert_idx: int,
    quant_out: torch.Tensor,
    teacher_out: torch.Tensor,
) -> None:
    if quant_out.numel() == 0:
        return
    stats.routed_pairs[expert_idx] += int(quant_out.shape[0])

    quant_f = quant_out.to(device=stats.scalar_x_sum.device, dtype=torch.float64)
    teacher_f = teacher_out.to(device=stats.scalar_y_sum.device, dtype=torch.float64)
    scalar_count = float(quant_out.numel())
    stats.scalar_count[expert_idx] += scalar_count
    stats.scalar_x_sum[expert_idx] += quant_f.sum()
    stats.scalar_y_sum[expert_idx] += teacher_f.sum()
    stats.scalar_x2_sum[expert_idx] += quant_f.square().sum()
    stats.scalar_xy_sum[expert_idx] += (quant_f * teacher_f).sum()


def solve_scalar_affine_from_moments(stats: ScalarFitMoments) -> AffineCorrection:
    count = stats.scalar_count.clamp(min=1.0)
    mean_x = stats.scalar_x_sum / count
    mean_y = stats.scalar_y_sum / count
    var_x = stats.scalar_x2_sum - (stats.scalar_x_sum.square() / count)
    cov_xy = stats.scalar_xy_sum - ((stats.scalar_x_sum * stats.scalar_y_sum) / count)

    alpha = torch.ones_like(mean_x, dtype=torch.float32)
    valid = stats.scalar_count > 0
    stable = torch.logical_and(valid, var_x.abs() > 1e-12)
    alpha[stable] = (cov_xy[stable] / var_x[stable]).to(torch.float32)
    alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))

    beta = torch.zeros_like(mean_x, dtype=torch.float32)
    beta[valid] = (mean_y[valid] - alpha[valid].to(torch.float64) * mean_x[valid]).to(torch.float32)
    beta = torch.where(torch.isfinite(beta), beta, torch.zeros_like(beta))
    return AffineCorrection(alpha=alpha.detach().cpu(), beta=beta.detach().cpu(), mode="scalar")


def summarize_scalar_fit(correction: AffineCorrection, stats: ScalarFitMoments) -> dict[str, Any]:
    active_experts = int((stats.routed_pairs > 0).sum().item())
    routed_pairs = int(stats.routed_pairs.sum().item())
    layer_mse = 0.0 if stats.layer_elem_count <= 0 else float((stats.layer_sq_error / float(stats.layer_elem_count)).item())
    alpha_delta = None if correction.alpha is None else (correction.alpha.to(torch.float32) - 1.0).abs()
    beta_abs = None if correction.beta is None else correction.beta.to(torch.float32).abs()
    return {
        "active_experts": active_experts,
        "routed_pairs": routed_pairs,
        "teacher_quant_mse": round(layer_mse, 8),
        "alpha_abs_delta_mean": 0.0 if alpha_delta is None else round(float(alpha_delta.mean().item()), 8),
        "alpha_abs_delta_max": 0.0 if alpha_delta is None else round(float(alpha_delta.max().item()), 8),
        "beta_abs_mean": 0.0 if beta_abs is None else round(float(beta_abs.mean().item()), 8),
        "beta_abs_max": 0.0 if beta_abs is None else round(float(beta_abs.max().item()), 8),
    }


def slice_maca_set(full_set: VariableLengthCalibrationSet, chunks_per_length: int) -> VariableLengthCalibrationSet:
    if chunks_per_length <= 0:
        raise ValueError("--maca-chunks-per-length must be positive")
    if chunks_per_length >= MACA_CHUNKS_PER_LENGTH:
        return full_set

    keep_indices: list[int] = []
    for length_idx, _length in enumerate(MACA_LENGTHS):
        start = length_idx * MACA_CHUNKS_PER_LENGTH
        stop = start + chunks_per_length
        keep_indices.extend(range(start, stop))
    index = torch.as_tensor(keep_indices, dtype=torch.long)
    chunks = full_set.chunks.index_select(0, index).contiguous()
    actual_lengths = full_set.actual_lengths.index_select(0, index).contiguous()
    metadata = dict(full_set.metadata)
    metadata["chunks_per_length"] = int(chunks_per_length)
    metadata["samples"] = int(chunks.shape[0])
    metadata["sample_shape"] = [int(chunks.shape[0]), int(chunks.shape[1])]
    metadata["actual_token_total"] = int(actual_lengths.sum().item())
    metadata["padded_token_total"] = int(chunks.numel())
    metadata["real_token_fraction"] = round(float(metadata["actual_token_total"]) / float(max(metadata["padded_token_total"], 1)), 6)
    metadata["standard_reference_tokens"] = int(SEQLEN * chunks.shape[0])
    return VariableLengthCalibrationSet(chunks=chunks, actual_lengths=actual_lengths, metadata=metadata)


def build_standard_calibration_set(calib_chunks: torch.Tensor, retune_chunks: int, calib_info: dict[str, Any]) -> VariableLengthCalibrationSet:
    used = calib_chunks[: max(1, min(int(retune_chunks), int(calib_chunks.shape[0])))].contiguous()
    actual_lengths = torch.full((used.shape[0],), int(used.shape[1]), dtype=torch.long)
    metadata = dict(calib_info)
    metadata["used_samples"] = int(used.shape[0])
    metadata["fit_actual_lengths"] = [int(used.shape[1])]
    return VariableLengthCalibrationSet(chunks=used, actual_lengths=actual_lengths, metadata=metadata)


def build_maca_joint_plan(
    calibration: Any,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[EvalPlan, dict[str, Any]]:
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    plan = build_plan_from_masks(
        "maca_joint_w1w2_topup",
        "Rebuild the joint_w1w2_with_topup mask from MaCa multi-scale calibration and keep GPTQ-standard evaluation unchanged.",
        config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )
    return plan, {
        **joint_meta,
        "calibration_variant": "maca_multiscale",
        "channel_metric_requested": "activation_kurtosis",
        "channel_metric_effective": "activation_kurtosis",
        "channel_metric_fallback_used": False,
        "source_base": "maca_joint_w1w2_topup",
    }


def moe_forward_teacher_quant_valid_tokens(
    hidden_states: torch.Tensor,
    teacher_tensors: dict[str, torch.Tensor],
    quant_tensors: dict[str, torch.Tensor],
    config: Any,
    actual_length: int,
    stats: ScalarFitMoments | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actual_length <= 0:
        zeros = torch.zeros_like(hidden_states)
        return zeros, zeros

    valid_hidden_states = hidden_states[:, :actual_length, :]
    batch_size, sequence_length, hidden_dim = valid_hidden_states.shape
    flat = valid_hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, teacher_tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1).indices
    selected_weights = routing_probs.gather(1, selected_experts)
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    selected_weights = selected_weights.to(valid_hidden_states.dtype)

    teacher_final = torch.zeros((flat.shape[0], hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)
    quant_final = torch.zeros((flat.shape[0], hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]

        teacher_gate_up = F.linear(current_state, teacher_tensors["experts.gate_up_proj"][expert_idx])
        teacher_gate, teacher_up = teacher_gate_up.chunk(2, dim=-1)
        teacher_out = F.linear(F.silu(teacher_gate) * teacher_up, teacher_tensors["experts.down_proj"][expert_idx])

        quant_gate_up = F.linear(current_state, quant_tensors["experts.gate_up_proj"][expert_idx])
        quant_gate, quant_up = quant_gate_up.chunk(2, dim=-1)
        quant_out = F.linear(F.silu(quant_gate) * quant_up, quant_tensors["experts.down_proj"][expert_idx])

        if stats is not None:
            update_scalar_fit_moments(stats, expert_idx, quant_out, teacher_out)

        routed_weight = selected_weights[token_idx, route_pos].unsqueeze(-1)
        teacher_final.index_add_(0, token_idx, (routed_weight * teacher_out).to(valid_hidden_states.dtype))
        quant_final.index_add_(0, token_idx, (routed_weight * quant_out).to(valid_hidden_states.dtype))

    shared_gate = F.linear(flat, teacher_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, teacher_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, teacher_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, teacher_tensors["shared_expert_gate.weight"]))
    shared_weighted = shared_out * shared_gate_value
    teacher_final = teacher_final + shared_weighted
    quant_final = quant_final + shared_weighted

    teacher_view = teacher_final.view(batch_size, sequence_length, hidden_dim)
    quant_view = quant_final.view(batch_size, sequence_length, hidden_dim)
    if stats is not None:
        diff = (teacher_view.float() - quant_view.float()).to(device=stats.layer_sq_error.device, dtype=torch.float64)
        stats.layer_sq_error += diff.square().sum()
        stats.layer_elem_count += int(diff.numel())

    teacher_padded = torch.zeros_like(hidden_states)
    quant_padded = torch.zeros_like(hidden_states)
    teacher_padded[:, :actual_length, :] = teacher_view
    quant_padded[:, :actual_length, :] = quant_view
    return teacher_padded, quant_padded


def moe_forward_with_correction_valid_tokens(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    correction: AffineCorrection | None,
    actual_length: int,
) -> torch.Tensor:
    if actual_length <= 0:
        return torch.zeros_like(hidden_states)

    valid_hidden_states = hidden_states[:, :actual_length, :]
    batch_size, sequence_length, hidden_dim = valid_hidden_states.shape
    flat = valid_hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    routing_weights = routing_weights.to(valid_hidden_states.dtype)
    final_hidden_states = torch.zeros((flat.shape[0], hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, tensors["experts.down_proj"][expert_idx])
        current_hidden = apply_affine_correction(current_hidden, correction, expert_idx)
        final_hidden_states.index_add_(
            0,
            token_idx,
            (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden).to(valid_hidden_states.dtype),
        )

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    valid_output = final_hidden_states + shared_out * shared_gate_value

    padded_output = torch.zeros_like(hidden_states)
    padded_output[:, :actual_length, :] = valid_output.view(batch_size, sequence_length, hidden_dim)
    return padded_output


def fit_scalar_corrections_for_variant(
    model_id: str,
    variant: BaseVariant,
    config: Any,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, AffineCorrection], list[int], dict[str, Any]]:
    store = WeightStore(model_id, snapshot_dir, weight_map)
    embed_key, _norm_key, _lm_head_key = resolve_terminal_keys(weight_map)
    calib_chunks = variant.calibration_set.chunks
    actual_lengths = variant.calibration_set.actual_lengths

    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    corrections: dict[int, AffineCorrection] = {}
    layer_rows: list[dict[str, Any]] = []
    fitting_layers: dict[str, Any] = {}
    cpu_device = torch.device("cpu")

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"\n=== Fit {variant.key} layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type}) ===", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        teacher_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        quantized_tensors: dict[str, torch.Tensor] | None = None
        quant_moe_tensors: dict[str, torch.Tensor] | None = None
        tensors_cpu: dict[str, torch.Tensor] | None = None
        quantized_tensors_cpu: dict[str, torch.Tensor] | None = None
        teacher_moe_tensors_cpu: dict[str, torch.Tensor] | None = None
        quant_moe_tensors_cpu: dict[str, torch.Tensor] | None = None
        fit_backend = "gpu"
        stats_device = device

        try:
            quantized_tensors = prepare_quantized_layer_tensors(variant.plan, layer_idx, tensors, config)
            quant_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in quantized_tensors.items() if k.startswith("mlp.")}
        except torch.OutOfMemoryError:
            fit_backend = "cpu_quant_compare"
            stats_device = cpu_device
            print(
                f"[fit:{variant.key}] layer {layer_idx + 1}/{config.num_hidden_layers} hit CUDA OOM while materializing quantized tensors; falling back to CPU comparison tensors for fitting.",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            quantized_tensors = None
            quant_moe_tensors = None
            tensors_cpu = shorten_layer_tensors(layer_idx, raw_tensors, cpu_device, dtype)
            quantized_tensors_cpu = prepare_quantized_layer_tensors(variant.plan, layer_idx, tensors_cpu, config)
            teacher_moe_tensors_cpu = {k.replace("mlp.", "", 1): v for k, v in tensors_cpu.items() if k.startswith("mlp.")}
            quant_moe_tensors_cpu = {k.replace("mlp.", "", 1): v for k, v in quantized_tensors_cpu.items() if k.startswith("mlp.")}
        del raw_tensors

        stats = init_scalar_fit_moments(config, stats_device)
        for chunk_idx in range(int(inps.shape[0])):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual, mlp_input = forward_attention_to_mlp_input(
                hidden_states,
                tensors,
                layer_type,
                config,
                position_embeddings,
                causal_mask,
            )
            actual_length = int(actual_lengths[chunk_idx].item())

            if quant_moe_tensors is not None:
                teacher_moe, _quant_moe = moe_forward_teacher_quant_valid_tokens(
                    mlp_input,
                    teacher_moe_tensors,
                    quant_moe_tensors,
                    config,
                    actual_length,
                    stats,
                )
            else:
                teacher_moe = moe_forward_with_correction_valid_tokens(
                    mlp_input,
                    teacher_moe_tensors,
                    config,
                    correction=None,
                    actual_length=actual_length,
                )
                if teacher_moe_tensors_cpu is None or quant_moe_tensors_cpu is None:
                    raise RuntimeError("CPU fitting fallback is missing MoE tensors")
                mlp_input_cpu = mlp_input.to(device=cpu_device, dtype=dtype, non_blocking=False)
                _teacher_cpu, _quant_cpu = moe_forward_teacher_quant_valid_tokens(
                    mlp_input_cpu,
                    teacher_moe_tensors_cpu,
                    quant_moe_tensors_cpu,
                    config,
                    actual_length,
                    stats,
                )
                del mlp_input_cpu, _teacher_cpu, _quant_cpu

            outs[chunk_idx] = residual + teacher_moe
            if should_log_chunk(chunk_idx, int(inps.shape[0])):
                print(
                    f"[fit:{variant.key}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{int(inps.shape[0])} ({fit_backend}) | actual_len={actual_length}",
                    flush=True,
                )

        correction = solve_scalar_affine_from_moments(stats)
        corrections[layer_idx] = correction
        summary = summarize_scalar_fit(correction, stats)
        layer_rows.append({"layer": int(layer_idx), "teacher_quant_mse": summary["teacher_quant_mse"]})
        fitting_layers[str(layer_idx)] = {
            "layer": int(layer_idx),
            "teacher_quant_mse": summary["teacher_quant_mse"],
            "active_experts": summary["active_experts"],
            "routed_pairs": summary["routed_pairs"],
            "scalar": summary,
            "fit_backend": fit_backend,
        }
        print(
            f"[fit:{variant.key}] layer {layer_idx + 1}/{config.num_hidden_layers} -> mse={summary['teacher_quant_mse']:.8f} | active={summary['active_experts']} | beta_max={summary['beta_abs_max']:.6f}",
            flush=True,
        )

        release_tensors(tensors)
        if quantized_tensors is not None:
            release_tensors(quantized_tensors)
        if tensors_cpu is not None:
            release_tensors(tensors_cpu)
        if quantized_tensors_cpu is not None:
            release_tensors(quantized_tensors_cpu)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ranked_layers = sorted(layer_rows, key=lambda row: (float(row["teacher_quant_mse"]), -int(row["layer"])), reverse=True)
    top10_layers = [int(row["layer"]) for row in ranked_layers[: min(10, len(ranked_layers))]]
    summary_payload = {
        "variant": variant.key,
        "base_plan": variant.plan.name,
        "calibration": variant.calibration_set.metadata,
        "top10_layers_by_local_mse": top10_layers,
        "layer_teacher_quant_mse": ranked_layers,
        "layers": fitting_layers,
    }
    return corrections, top10_layers, summary_payload


def build_eval_plan(base_plan: EvalPlan, eval_name: str, correction_layers: list[int], config: Any) -> EvalPlan:
    parameter_count = parameter_count_for_layers(config, correction_layers)
    description = (
        f"Apply scalar expert-output corrections on layers {correction_layers} on top of `{base_plan.name}`."
        if correction_layers
        else f"Evaluate `{base_plan.name}` without post-expert correction."
    )
    return EvalPlan(
        name=eval_name,
        description=description,
        mode=base_plan.mode,
        memory_gb=estimate_correction_memory_gb(base_plan.memory_gb, parameter_count),
        fp8_weights=base_plan.fp8_weights,
        w1_pair_masks=base_plan.w1_pair_masks,
        w2_channel_masks=base_plan.w2_channel_masks,
    )


def parameter_count_for_layers(config: Any, correction_layers: list[int]) -> int:
    return 2 * int(len(correction_layers)) * int(config.num_experts)


def evaluate_and_record_compound_plan(
    model_id: str,
    payload: dict[str, Any],
    eval_name: str,
    base_variant: BaseVariant,
    corrections: dict[int, AffineCorrection],
    extras: dict[str, Any],
    output_json: Path,
    config: Any,
    total_expert_elems: int,
    test_ids: torch.Tensor,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    result_rows = payload.setdefault("results", {})
    if eval_name in result_rows:
        print(f"[skip] {eval_name} already present", flush=True)
        return result_rows[eval_name]

    correction_layers = list(extras.get("correction_layers", []))
    plan = build_eval_plan(base_variant.plan, eval_name, correction_layers, config)
    print(f"\n=== Eval: {eval_name} ===", flush=True)
    start_time = time.time()
    ppl, nll, nsamples = evaluate_plan_with_corrections(
        model_id,
        plan,
        corrections,
        test_ids,
        config,
        weight_map,
        snapshot_dir,
        device,
        dtype,
    )
    elapsed = time.time() - start_time
    parameter_count = parameter_count_for_layers(config, correction_layers)
    row = {
        "description": plan.description,
        "mode": plan.mode,
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "memory_gb": round(plan.memory_gb, 3),
        "fp8_weights": int(plan.fp8_weights),
        "fp8_fraction": round(float(plan.fp8_weights) / float(max(total_expert_elems, 1)), 6),
        "w1_pair_fraction": round(total_pair_fraction(config, plan.w1_pair_masks or {}), 6),
        "w2_channel_fraction": round(total_channel_fraction(config, plan.w2_channel_masks or {}), 6),
        "eval_chunks": int(nsamples),
        "seqlen": SEQLEN,
        "time_s": round(elapsed, 1),
        "correction_parameter_count": int(parameter_count),
        **extras,
    }
    result_rows[eval_name] = row
    atomic_json_dump(output_json, payload)
    print(
        f"[{eval_name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | params={parameter_count} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def update_summary(payload: dict[str, Any], base_name: str) -> None:
    results = payload.get("results", {})
    if base_name not in results:
        return
    base_ppl = float(results[base_name]["ppl"])
    ranked: list[dict[str, Any]] = []
    for name, row in results.items():
        gain = round(base_ppl - float(row["ppl"]), 6)
        row["ppl_gain_vs_base"] = gain
        row["ppl_delta_vs_base"] = round(float(row["ppl"]) - base_ppl, 6)
        ranked.append(
            {
                "name": name,
                "ppl": float(row["ppl"]),
                "memory_gb": float(row["memory_gb"]),
                "ppl_gain_vs_base": gain,
                "correction_scope": row.get("correction_scope"),
                "base_variant": row.get("base_variant"),
            }
        )
    ranked.sort(key=lambda item: (float(item["ppl"]), float(item["memory_gb"]), item["name"]))
    payload["summary"] = {
        "base_plan": base_name,
        "base_ppl": base_ppl,
        "best_plan": ranked[0] if ranked else None,
        "ranked_results": ranked,
    }


def render_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    results = payload.get("results", {})
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    lines = [SECTION_MARKER, ""]
    lines.append(
        "**Approach**: Combined Iteration 26 MaCa multi-scale calibration with Iteration 25 scalar post-expert correction. The MaCa branch rebuilds `joint_w1w2_with_topup` from multi-scale sensitivity metrics, then fits closed-form scalar `alpha/beta` repairs on the quantized expert outputs before routing-weight mixing."
    )
    lines.append("")
    maca_top10 = payload.get("fitting", {}).get(FIT_MACA, {}).get("top10_layers_by_local_mse", [])
    standard_top10 = payload.get("fitting", {}).get(FIT_STANDARD, {}).get("top10_layers_by_local_mse", [])
    lines.append(f"**MaCa top-10 layers by local MSE**: {', '.join(str(layer) for layer in maca_top10) if maca_top10 else 'n/a'}")
    lines.append(f"**Standard top-10 layers by local MSE**: {', '.join(str(layer) for layer in standard_top10) if standard_top10 else 'n/a'}")
    lines.append("")
    lines.append("**Results**:")
    for name, row in ordered:
        lines.append(
            f"- `{name}` -> PPL {float(row['ppl']):.6f} @ {float(row['memory_gb']):.3f} GB, gain vs MaCa no-correction {float(row.get('ppl_gain_vs_base', 0.0)):+.6f}"
        )
    lines.append("")
    maca_no = results.get(PLAN_MACA_NONE)
    maca_top10_row = results.get(PLAN_MACA_TOP10)
    standard_top10_row = results.get(PLAN_STANDARD_TOP10)
    if maca_no is not None and maca_top10_row is not None:
        delta = float(maca_no["ppl"]) - float(maca_top10_row["ppl"])
        lines.append(
            f"**Insight**: MaCa + top-10 scalar correction moves from {float(maca_no['ppl']):.6f} to {float(maca_top10_row['ppl']):.6f}, a {delta:+.6f} PPL change relative to MaCa alone."
        )
    else:
        lines.append("**Insight**: Need both the MaCa no-correction row and the MaCa top-10 correction row to measure whether the gains stack.")
    if standard_top10_row is not None:
        ref = references.get("correction_scalar_top10layers", references.get("joint_w1w2_with_topup"))
        if ref is not None and "ppl" in ref:
            lines.append(
                f"**Control check**: `standard_plus_correction_top10` lands at {float(standard_top10_row['ppl']):.6f}, versus reference {float(ref['ppl']):.6f}."
            )
    lines.append("**Next**: If the compound row improves on MaCa alone, the next sanity check is a second seed for the MaCa calibration set rather than another mask family.")
    return "\n".join(lines)


def print_results_table(payload: dict[str, Any]) -> None:
    results = payload.get("results", {})
    ordered_names = [row["name"] for row in payload.get("summary", {}).get("ranked_results", []) if row["name"] in results]
    print("\n" + "=" * 170, flush=True)
    print("proper_iter28_maca_plus_correction | MaCa joint mask + scalar expert-output correction | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 170, flush=True)
    print(
        f"{'Config':<40} {'PPL':>10} {'Gain':>10} {'Delta':>10} {'Memory GB':>12} {'Params':>12} {'Layers':>8} {'Base':>16}",
        flush=True,
    )
    print("-" * 170, flush=True)
    for name in ordered_names:
        row = results[name]
        correction_layers = row.get("correction_layers")
        layer_text = str(len(correction_layers)) if isinstance(correction_layers, list) else "0"
        print(
            f"{name:<40} {float(row['ppl']):>10.4f} {float(row.get('ppl_gain_vs_base', 0.0)):>+10.4f} {float(row.get('ppl_delta_vs_base', 0.0)):>+10.4f} {float(row['memory_gb']):>12.3f} {int(row.get('correction_parameter_count', 0)):>12d} {layer_text:>8} {str(row.get('base_variant', 'n/a')):>16}",
            flush=True,
        )
    print("-" * 170, flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    requested_plans = resolve_requested_plans(args.plans)
    all_plan_names = {PLAN_MACA_TOP10, PLAN_MACA_ALL, PLAN_MACA_NONE, PLAN_STANDARD_TOP10}
    if requested_plans is not None:
        missing = sorted(requested_plans - all_plan_names)
        if missing:
            raise ValueError(f"Unknown plans requested: {', '.join(missing)}")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    tokenizer, calib_chunks_full, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    standard_fit_set = build_standard_calibration_set(calib_chunks_full, int(args.retune_chunks), calib_info)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}

    full_maca_set = build_maca_calibration_set(tokenizer, args.seed)
    maca_fit_set = slice_maca_set(full_maca_set, int(args.maca_chunks_per_length))

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise RuntimeError(f"Missing compatible calibration cache at {args.cache_path}")

    standard_plan, standard_extras = build_base_plan(calibration, text_config, non_expert_bytes, total_expert_elems)
    references = load_reference_rows(args.output_json)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "standard_calibration": standard_fit_set.metadata,
            "maca_calibration": maca_fit_set.metadata,
            "evaluation": eval_info,
            "quantization": "simulated quantization: routed MoE experts use either the standard joint_w1w2_with_topup mask or the Iteration 26 MaCa-rebuilt joint mask; scalar alpha/beta corrections repair expert outputs only and never modify stored weight tensors.",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "experiment": "Iteration 28 compound MaCa joint mask plus scalar learned correction",
            "requested_plans": sorted(requested_plans) if requested_plans is not None else sorted(all_plan_names),
            "references": references,
            "correction_param_bytes": CORRECTION_PARAM_BYTES,
        },
        "fitting": {},
        "results": {},
    }
    if args.output_json.exists():
        try:
            existing = load_json(args.output_json)
            if isinstance(existing, dict):
                merged_metadata = dict(payload["metadata"])
                merged_metadata.update(existing.get("metadata", {}))
                payload.update(existing)
                payload["metadata"] = merged_metadata
                payload.setdefault("fitting", {})
                payload.setdefault("results", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    need_maca_plan = requested_plans is None or bool({PLAN_MACA_TOP10, PLAN_MACA_ALL, PLAN_MACA_NONE} & requested_plans)
    need_standard_fit = requested_plans is None or PLAN_STANDARD_TOP10 in requested_plans
    need_maca_fit = requested_plans is None or bool({PLAN_MACA_TOP10, PLAN_MACA_ALL} & requested_plans)

    base_variants: dict[str, BaseVariant] = {
        FIT_STANDARD: BaseVariant(
            key=FIT_STANDARD,
            plan=standard_plan,
            extras={
                **standard_extras,
                "calibration_variant": "standard_fixed_2048",
                "source_base": standard_plan.name,
            },
            calibration_set=standard_fit_set,
        )
    }

    if need_maca_plan:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        maca_calibration = run_maca_calibration(store, text_config, maca_fit_set.chunks, maca_fit_set.actual_lengths, device, dtype)
        maca_plan, maca_extras = build_maca_joint_plan(maca_calibration, text_config, non_expert_bytes, total_expert_elems)
        payload["metadata"]["maca_calibration_summary"] = {
            "active_experts_per_layer": {str(layer_idx): int((counts > 0).sum().item()) for layer_idx, counts in maca_calibration.routing_counts.items()}
        }
        atomic_json_dump(args.output_json, payload)
        base_variants[FIT_MACA] = BaseVariant(
            key=FIT_MACA,
            plan=maca_plan,
            extras=maca_extras,
            calibration_set=maca_fit_set,
        )
        del store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fitted_corrections: dict[str, dict[int, AffineCorrection]] = {}
    top_layers_by_variant: dict[str, list[int]] = {}
    for variant_key, should_fit in ((FIT_STANDARD, need_standard_fit), (FIT_MACA, need_maca_fit)):
        if not should_fit or variant_key not in base_variants:
            continue
        corrections, top10_layers, fit_payload = fit_scalar_corrections_for_variant(
            args.model_id,
            base_variants[variant_key],
            text_config,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )
        fitted_corrections[variant_key] = corrections
        top_layers_by_variant[variant_key] = top10_layers
        payload["fitting"][variant_key] = fit_payload
        atomic_json_dump(args.output_json, payload)

    evaluation_specs: list[tuple[str, str, dict[int, AffineCorrection], dict[str, Any]]] = []
    if FIT_MACA in base_variants:
        maca_top10 = top_layers_by_variant.get(FIT_MACA, [])
        maca_all_layers = list(range(text_config.num_hidden_layers))
        evaluation_specs.extend([
            (
                PLAN_MACA_TOP10,
                FIT_MACA,
                filter_top_layers(fitted_corrections.get(FIT_MACA, {}), maca_top10),
                {
                    **base_variants[FIT_MACA].extras,
                    "base_variant": FIT_MACA,
                    "correction_type": "scalar_affine",
                    "correction_scope": "top10_local_mse_layers",
                    "correction_layers": maca_top10,
                    "fit_variant": FIT_MACA,
                },
            ),
            (
                PLAN_MACA_ALL,
                FIT_MACA,
                fitted_corrections.get(FIT_MACA, {}),
                {
                    **base_variants[FIT_MACA].extras,
                    "base_variant": FIT_MACA,
                    "correction_type": "scalar_affine",
                    "correction_scope": "all_layers",
                    "correction_layers": maca_all_layers,
                    "fit_variant": FIT_MACA,
                },
            ),
            (
                PLAN_MACA_NONE,
                FIT_MACA,
                {},
                {
                    **base_variants[FIT_MACA].extras,
                    "base_variant": FIT_MACA,
                    "correction_type": "none",
                    "correction_scope": "none",
                    "correction_layers": [],
                    "fit_variant": None,
                },
            ),
        ])

    if FIT_STANDARD in base_variants:
        standard_top10 = top_layers_by_variant.get(FIT_STANDARD, [])
        evaluation_specs.append(
            (
                PLAN_STANDARD_TOP10,
                FIT_STANDARD,
                filter_top_layers(fitted_corrections.get(FIT_STANDARD, {}), standard_top10),
                {
                    **base_variants[FIT_STANDARD].extras,
                    "base_variant": FIT_STANDARD,
                    "correction_type": "scalar_affine",
                    "correction_scope": "top10_local_mse_layers",
                    "correction_layers": standard_top10,
                    "fit_variant": FIT_STANDARD,
                },
            )
        )

    for eval_name, variant_key, corrections, extras in evaluation_specs:
        if requested_plans is not None and eval_name not in requested_plans:
            continue
        evaluate_and_record_compound_plan(
            args.model_id,
            payload,
            eval_name,
            base_variants[variant_key],
            corrections,
            extras,
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )

    update_summary(payload, PLAN_MACA_NONE)
    payload["metadata"]["elapsed_s"] = round(time.time() - overall_start, 1)
    atomic_json_dump(args.output_json, payload)

    section = render_exploration_section(payload, references)
    upsert_exploration_section(args.exploration_md, section)
    print(f"[exploration] updated {args.exploration_md}", flush=True)
    print_results_table(payload)


if __name__ == "__main__":
    main()

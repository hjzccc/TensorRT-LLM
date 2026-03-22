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
from proper_eval import (
    LOGITS_CHUNK_TOKENS,
    SEQLEN,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    layer_type_at,
    load_gptq_standard_data,
    prepare_layer_tensors_for_plan,
    upsert_exploration_section,
)
from proper_iter01 import load_cache, total_channel_fraction, total_pair_fraction
from proper_iter21_serq_salient import load_reference_rows
from proper_iter23_subsystem_sweep import build_base_plan
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter25_learned_correction.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
SECTION_MARKER = "## [28] Iteration 25 - Learned Expert Output Correction"
CORRECTION_PARAM_BYTES = 2
AFFINE_EPS = 1e-12


@dataclass(frozen=True)
class CorrectionConfig:
    name: str
    mode: str
    top_layers: int | None = None


@dataclass
class AffineCorrection:
    alpha: torch.Tensor | None
    beta: torch.Tensor | None
    mode: str


@dataclass
class LayerFitMoments:
    routed_pairs: torch.Tensor
    scalar_count: torch.Tensor
    scalar_x_sum: torch.Tensor
    scalar_y_sum: torch.Tensor
    scalar_x2_sum: torch.Tensor
    scalar_xy_sum: torch.Tensor
    perchannel_count: torch.Tensor
    perchannel_x_sum: torch.Tensor
    perchannel_y_sum: torch.Tensor
    perchannel_x2_sum: torch.Tensor
    perchannel_xy_sum: torch.Tensor
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
        help=(
            "Comma-separated subset of plan names to evaluate; default runs baseline plus all learned-correction configs."
        ),
    )
    parser.add_argument(
        "--retune-chunks",
        type=int,
        default=128,
        help="Number of GPTQ-standard calibration chunks to use for target collection and closed-form fitting.",
    )
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on evaluation chunks for smoke tests; 0 means full WikiText-2 test.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def should_log_chunk(chunk_idx: int, total_chunks: int) -> bool:
    return chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == total_chunks


def maybe_slice_eval_chunks(test_ids: torch.Tensor, eval_max_chunks: int) -> tuple[torch.Tensor, dict[str, Any]]:
    total_chunks = int(test_ids.numel() // SEQLEN)
    if eval_max_chunks <= 0 or eval_max_chunks >= total_chunks:
        return test_ids, {
            "nsamples": total_chunks,
            "used_tokens": int(total_chunks * SEQLEN),
            "truncated": False,
        }
    used_tokens = int(eval_max_chunks * SEQLEN)
    return test_ids[:, :used_tokens].contiguous(), {
        "nsamples": int(eval_max_chunks),
        "used_tokens": used_tokens,
        "truncated": True,
    }


def build_correction_configs() -> list[CorrectionConfig]:
    return [
        CorrectionConfig("correction_scalar", "scalar"),
        CorrectionConfig("correction_perchannel", "perchannel"),
        CorrectionConfig("correction_scalar_top10layers", "scalar", top_layers=10),
        CorrectionConfig("correction_bias_only", "bias_only"),
    ]


def prepare_quantized_layer_tensors(
    plan: EvalPlan,
    layer_idx: int,
    tensors: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    quantized = dict(tensors)
    prepare_layer_tensors_for_plan(plan, layer_idx, quantized, config)
    return quantized


def forward_attention_to_mlp_input(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    layer_type: str,
    config: Any,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    causal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = hidden_states
    hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
    if layer_type == "full_attention":
        attn_tensors = {k.replace("self_attn.", "", 1): v for k, v in tensors.items() if k.startswith("self_attn.")}
        mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
    else:
        attn_tensors = {k.replace("linear_attn.", "", 1): v for k, v in tensors.items() if k.startswith("linear_attn.")}
        mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
    post_attention = residual + mixed
    mlp_input = rms_norm_qwen3_next(post_attention, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
    return post_attention, mlp_input


def init_layer_fit_moments(config: Any, device: torch.device) -> LayerFitMoments:
    num_experts = int(config.num_experts)
    hidden_size = int(config.hidden_size)
    return LayerFitMoments(
        routed_pairs=torch.zeros(num_experts, dtype=torch.int64, device=device),
        scalar_count=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_x_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_y_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_x2_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_xy_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        perchannel_count=torch.zeros(num_experts, dtype=torch.float64, device=device),
        perchannel_x_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        perchannel_y_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        perchannel_x2_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        perchannel_xy_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        layer_sq_error=torch.zeros((), dtype=torch.float64, device=device),
        layer_elem_count=0,
    )


def update_layer_fit_moments(
    stats: LayerFitMoments,
    expert_idx: int,
    quant_out: torch.Tensor,
    teacher_out: torch.Tensor,
) -> None:
    if quant_out.numel() == 0:
        return
    stats.routed_pairs[expert_idx] += int(quant_out.shape[0])

    quant_f = quant_out.to(torch.float64)
    teacher_f = teacher_out.to(torch.float64)
    scalar_count = float(quant_out.numel())
    stats.scalar_count[expert_idx] += scalar_count
    stats.scalar_x_sum[expert_idx] += quant_f.sum()
    stats.scalar_y_sum[expert_idx] += teacher_f.sum()
    stats.scalar_x2_sum[expert_idx] += quant_f.square().sum()
    stats.scalar_xy_sum[expert_idx] += (quant_f * teacher_f).sum()

    token_count = float(quant_out.shape[0])
    stats.perchannel_count[expert_idx] += token_count
    stats.perchannel_x_sum[expert_idx].add_(quant_f.sum(dim=0))
    stats.perchannel_y_sum[expert_idx].add_(teacher_f.sum(dim=0))
    stats.perchannel_x2_sum[expert_idx].add_(quant_f.square().sum(dim=0))
    stats.perchannel_xy_sum[expert_idx].add_((quant_f * teacher_f).sum(dim=0))


def apply_affine_correction(
    expert_output: torch.Tensor,
    correction: AffineCorrection | None,
    expert_idx: int,
) -> torch.Tensor:
    if correction is None:
        return expert_output

    corrected = expert_output
    if correction.mode == "scalar":
        if correction.alpha is not None:
            corrected = corrected * correction.alpha[expert_idx].to(device=corrected.device, dtype=corrected.dtype)
        if correction.beta is not None:
            corrected = corrected + correction.beta[expert_idx].to(device=corrected.device, dtype=corrected.dtype)
        return corrected

    if correction.mode == "bias_only":
        if correction.beta is None:
            return corrected
        return corrected + correction.beta[expert_idx].to(device=corrected.device, dtype=corrected.dtype)

    if correction.mode == "perchannel":
        if correction.alpha is not None:
            corrected = corrected * correction.alpha[expert_idx].to(device=corrected.device, dtype=corrected.dtype)
        if correction.beta is not None:
            corrected = corrected + correction.beta[expert_idx].to(device=corrected.device, dtype=corrected.dtype)
        return corrected

    raise ValueError(f"Unsupported correction mode: {correction.mode}")


def solve_scalar_affine(stats: LayerFitMoments) -> AffineCorrection:
    count = stats.scalar_count.clamp(min=1.0)
    mean_x = stats.scalar_x_sum / count
    mean_y = stats.scalar_y_sum / count
    var_x = stats.scalar_x2_sum - (stats.scalar_x_sum.square() / count)
    cov_xy = stats.scalar_xy_sum - ((stats.scalar_x_sum * stats.scalar_y_sum) / count)

    alpha = torch.ones_like(mean_x, dtype=torch.float32)
    valid = stats.scalar_count > 0
    stable = torch.logical_and(valid, var_x.abs() > AFFINE_EPS)
    alpha[stable] = (cov_xy[stable] / var_x[stable]).to(torch.float32)
    alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))

    beta = torch.zeros_like(mean_x, dtype=torch.float32)
    beta[valid] = (mean_y[valid] - alpha[valid].to(torch.float64) * mean_x[valid]).to(torch.float32)
    beta = torch.where(torch.isfinite(beta), beta, torch.zeros_like(beta))
    return AffineCorrection(alpha=alpha.detach().cpu(), beta=beta.detach().cpu(), mode="scalar")


def solve_bias_only(stats: LayerFitMoments) -> AffineCorrection:
    count = stats.scalar_count.clamp(min=1.0)
    beta = torch.zeros_like(stats.scalar_x_sum, dtype=torch.float32)
    valid = stats.scalar_count > 0
    beta[valid] = ((stats.scalar_y_sum[valid] - stats.scalar_x_sum[valid]) / count[valid]).to(torch.float32)
    beta = torch.where(torch.isfinite(beta), beta, torch.zeros_like(beta))
    return AffineCorrection(alpha=None, beta=beta.detach().cpu(), mode="bias_only")


def solve_perchannel_affine(stats: LayerFitMoments) -> AffineCorrection:
    count = stats.perchannel_count.unsqueeze(1).clamp(min=1.0)
    mean_x = stats.perchannel_x_sum / count
    mean_y = stats.perchannel_y_sum / count
    var_x = stats.perchannel_x2_sum - (stats.perchannel_x_sum.square() / count)
    cov_xy = stats.perchannel_xy_sum - ((stats.perchannel_x_sum * stats.perchannel_y_sum) / count)

    alpha = torch.ones_like(mean_x, dtype=torch.float32)
    valid = (stats.perchannel_count.unsqueeze(1) > 0).expand_as(mean_x)
    stable = torch.logical_and(valid, var_x.abs() > AFFINE_EPS)
    alpha[stable] = (cov_xy[stable] / var_x[stable]).to(torch.float32)
    alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))

    beta = torch.zeros_like(mean_x, dtype=torch.float32)
    beta[valid] = (mean_y[valid] - alpha[valid].to(torch.float64) * mean_x[valid]).to(torch.float32)
    beta = torch.where(torch.isfinite(beta), beta, torch.zeros_like(beta))
    return AffineCorrection(alpha=alpha.detach().cpu(), beta=beta.detach().cpu(), mode="perchannel")


def summarize_correction(correction: AffineCorrection, stats: LayerFitMoments) -> dict[str, Any]:
    active_experts = int((stats.routed_pairs > 0).sum().item())
    routed_pairs = int(stats.routed_pairs.sum().item())
    layer_mse = 0.0 if stats.layer_elem_count <= 0 else float((stats.layer_sq_error / float(stats.layer_elem_count)).item())

    if correction.mode == "perchannel":
        alpha_delta = None if correction.alpha is None else (correction.alpha.to(torch.float32) - 1.0).abs()
        beta_abs = None if correction.beta is None else correction.beta.to(torch.float32).abs()
    elif correction.mode == "bias_only":
        alpha_delta = None
        beta_abs = None if correction.beta is None else correction.beta.to(torch.float32).abs()
    else:
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


def learn_layer_corrections(stats: LayerFitMoments) -> dict[str, AffineCorrection]:
    return {
        "scalar": solve_scalar_affine(stats),
        "perchannel": solve_perchannel_affine(stats),
        "bias_only": solve_bias_only(stats),
    }


def moe_forward_teacher_quant(
    hidden_states: torch.Tensor,
    teacher_tensors: dict[str, torch.Tensor],
    quant_tensors: dict[str, torch.Tensor],
    config: Any,
    stats: LayerFitMoments | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, teacher_tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1).indices
    selected_weights = routing_probs.gather(1, selected_experts)
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    selected_weights = selected_weights.to(hidden_states.dtype)

    teacher_final = torch.zeros((flat.shape[0], hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    quant_final = torch.zeros((flat.shape[0], hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
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
            update_layer_fit_moments(stats, expert_idx, quant_out, teacher_out)

        routed_weight = selected_weights[token_idx, route_pos].unsqueeze(-1)
        teacher_final.index_add_(0, token_idx, (routed_weight * teacher_out).to(hidden_states.dtype))
        quant_final.index_add_(0, token_idx, (routed_weight * quant_out).to(hidden_states.dtype))

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
        diff = (teacher_view.float() - quant_view.float()).to(torch.float64)
        stats.layer_sq_error += diff.square().sum()
        stats.layer_elem_count += int(diff.numel())
    return teacher_view, quant_view


def moe_forward_with_correction(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    correction: AffineCorrection | None,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((flat.shape[0], hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
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
            (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden).to(hidden_states.dtype),
        )

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def correction_parameter_count(config: Any, correction_name: str) -> int:
    layers = int(config.num_hidden_layers)
    experts = int(config.num_experts)
    hidden_size = int(config.hidden_size)
    if correction_name == "baseline_no_correction":
        return 0
    if correction_name == "correction_scalar":
        return 2 * layers * experts
    if correction_name == "correction_perchannel":
        return 2 * layers * experts * hidden_size
    if correction_name == "correction_scalar_top10layers":
        return 2 * min(10, layers) * experts
    if correction_name == "correction_bias_only":
        return layers * experts
    raise ValueError(f"Unknown correction config: {correction_name}")


def estimate_correction_memory_gb(base_memory_gb: float, parameter_count: int) -> float:
    return round(float(base_memory_gb) + (float(parameter_count * CORRECTION_PARAM_BYTES) / 1e9), 3)


def filter_top_layers(
    corrections: dict[int, AffineCorrection],
    top_layers: list[int],
) -> dict[int, AffineCorrection]:
    selected = set(int(layer_idx) for layer_idx in top_layers)
    return {int(layer_idx): correction for layer_idx, correction in corrections.items() if int(layer_idx) in selected}


@torch.inference_mode()
def evaluate_plan_with_corrections(
    model_id: str,
    plan: EvalPlan,
    corrections: dict[int, AffineCorrection],
    test_ids: torch.Tensor,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    nsamples = int(test_ids.numel() // SEQLEN)
    eval_chunks = test_ids[:, : nsamples * SEQLEN].view(nsamples, SEQLEN)
    store = WeightStore(model_id, snapshot_dir, weight_map)

    root_tensors = store.load_tensors([norm_key, lm_head_key])
    final_norm = root_tensors[norm_key].to(device=device, dtype=dtype, non_blocking=True)
    lm_head = root_tensors[lm_head_key].to(device=device, dtype=dtype, non_blocking=True)
    del root_tensors

    inps = embed_chunks(store, embed_key, eval_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[{plan.name}] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors
        tensors = prepare_quantized_layer_tensors(plan, layer_idx, tensors, config)
        correction = corrections.get(layer_idx)
        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}

        for chunk_idx in range(nsamples):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual, mlp_input = forward_attention_to_mlp_input(
                hidden_states,
                tensors,
                layer_type,
                config,
                position_embeddings,
                causal_mask,
            )
            moe_out = moe_forward_with_correction(mlp_input, moe_tensors, config, correction)
            outs[chunk_idx] = residual + moe_out
            if should_log_chunk(chunk_idx, nsamples):
                print(
                    f"[{plan.name}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{nsamples}",
                    flush=True,
                )

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids_device = test_ids.to(device=device, non_blocking=True)
    nlls: list[torch.Tensor] = []
    for chunk_idx in range(nsamples):
        hidden_states = inps[chunk_idx].unsqueeze(0)
        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
        shift_labels = test_ids_device[:, chunk_idx * SEQLEN : (chunk_idx + 1) * SEQLEN][:, 1:]
        shift_tokens = int(shift_labels.numel())
        chunk_loss_sum = torch.zeros((), dtype=torch.float64, device=hidden_states.device)
        for start in range(0, shift_tokens, LOGITS_CHUNK_TOKENS):
            end = min(start + LOGITS_CHUNK_TOKENS, shift_tokens)
            logits = F.linear(hidden_states[:, start:end, :].float(), lm_head.float())
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), shift_labels[:, start:end].reshape(-1), reduction="sum")
            chunk_loss_sum = chunk_loss_sum + loss.to(torch.float64)
            del logits, loss
        nlls.append((chunk_loss_sum / float(max(shift_tokens, 1))).to(torch.float32) * SEQLEN)
        if should_log_chunk(chunk_idx, nsamples):
            print(f"[{plan.name}] logits chunk {chunk_idx + 1}/{nsamples}", flush=True)

    mean_nll = torch.stack(nlls).sum() / (nsamples * SEQLEN)
    ppl = torch.exp(mean_nll)
    release_tensors(
        {
            "inps": inps,
            "outs": outs,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "causal_mask": causal_mask,
            "test_ids_device": test_ids_device,
        }
    )
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(ppl.item()), float(mean_nll.item()), nsamples


def evaluate_and_record_plan(
    model_id: str,
    payload: dict[str, Any],
    plan: EvalPlan,
    correction_name: str,
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
    if correction_name in result_rows:
        print(f"[skip] {correction_name} already present", flush=True)
        return result_rows[correction_name]

    print(f"\n=== Eval: {correction_name} ===", flush=True)
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
    parameter_count = correction_parameter_count(config, correction_name)
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
    result_rows[correction_name] = row
    atomic_json_dump(output_json, payload)
    print(
        f"[{correction_name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | params={parameter_count} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def update_summary(payload: dict[str, Any], base_plan_name: str) -> None:
    results = payload.get("results", {})
    if base_plan_name not in results:
        return
    base_ppl = float(results[base_plan_name]["ppl"])
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
                "correction_type": row.get("correction_type"),
                "correction_layers": row.get("correction_layers"),
                "correction_parameter_count": row.get("correction_parameter_count"),
            }
        )
    ranked.sort(key=lambda item: (float(item["ppl"]), float(item["memory_gb"]), item["name"]))
    payload["summary"] = {
        "base_plan": base_plan_name,
        "base_ppl": base_ppl,
        "best_plan": ranked[0] if ranked else None,
        "ranked_results": ranked,
    }


def render_exploration_section(
    payload: dict[str, Any],
    references: dict[str, dict[str, float]],
    top_layers: list[int],
) -> str:
    results = payload.get("results", {})
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_name, best_row = ordered[0]
    baseline_row = results.get("baseline_no_correction")
    baseline_ppl = None if baseline_row is None else float(baseline_row["ppl"])
    reference_ppl = references.get("joint_w1w2_with_topup", references.get("baseline_no_correction", {})).get("ppl")

    lines = [SECTION_MARKER, ""]
    lines.append(
        "**Approach**: Reused the deployed `joint_w1w2_with_topup` routed-expert masks, froze every quantized weight, and learned tiny post-expert affine repairs from calibration data so each quantized expert better matches its BF16 teacher output before routing-weight mixing."
    )
    lines.append("")
    lines.append(
        "**Fitting note**: Scalar, per-channel, and bias-only repairs use closed-form least squares on routed token/expert pairs. The `correction_scalar_top10layers` row reuses the scalar fits, but only activates them on the 10 layers with the largest local teacher-vs-quant MoE MSE."
    )
    lines.append("")
    lines.append(f"**Top-10 layers by local teacher-vs-quant MoE MSE**: {', '.join(str(layer_idx) for layer_idx in top_layers)}")
    lines.append("")
    lines.append("**Results**:")
    for name, row in ordered:
        lines.append(
            f"- `{name}` -> PPL {float(row['ppl']):.6f} @ {float(row['memory_gb']):.3f} GB, gain vs baseline {float(row.get('ppl_gain_vs_base', 0.0)):+.6f}"
        )
    lines.append("")
    if baseline_ppl is None:
        lines.append(
            f"**Insight**: `{best_name}` is the best Iteration 25 row, but the baseline row is missing so the gain cannot be grounded against the no-correction control."
        )
    else:
        lines.append(
            f"**Insight**: `{best_name}` is the best Iteration 25 row at {float(best_row['ppl']):.6f}, which is {baseline_ppl - float(best_row['ppl']):+.6f} PPL versus the frozen-mask baseline."
        )
    lines.append("")
    if reference_ppl is None:
        lines.append("**Next**: If learned output repair helps at all, the remaining question is whether the gain survives a second calibration seed rather than more static bit-allocation sweeps.")
    elif float(best_row["ppl"]) < float(reference_ppl):
        lines.append("**Next**: Learned function repair beats the standing static-mask reference, so the next experiment should stress-test stability across seeds and calibration subsets rather than search new masks.")
    else:
        lines.append("**Next**: If even learned post-expert repair stays flat, the floor likely comes from deeper quantization mismatch than tiny affine output fixes can absorb.")
    return "\n".join(lines)


def build_evaluation_plan(
    base_plan: EvalPlan,
    correction_name: str,
    config: Any,
) -> EvalPlan:
    parameter_count = correction_parameter_count(config, correction_name)
    description = (
        "Current best joint_w1w2_with_topup plan without learned output correction."
        if correction_name == "baseline_no_correction"
        else f"Apply `{correction_name}` as a post-expert affine repair on top of `{base_plan.name}` without modifying any quantized weight tensor."
    )
    return EvalPlan(
        name=correction_name,
        description=description,
        mode=base_plan.mode,
        memory_gb=estimate_correction_memory_gb(base_plan.memory_gb, parameter_count),
        fp8_weights=base_plan.fp8_weights,
        w1_pair_masks=base_plan.w1_pair_masks,
        w2_channel_masks=base_plan.w2_channel_masks,
    )


def print_results_table(payload: dict[str, Any], base_plan_name: str) -> None:
    results = payload.get("results", {})
    if base_plan_name not in results:
        return
    ordered_names = [row["name"] for row in payload.get("summary", {}).get("ranked_results", []) if row["name"] in results]
    print("\n" + "=" * 156, flush=True)
    print(
        "proper_iter25_learned_correction | expert-output affine repair on joint_w1w2_with_topup | full WikiText-2 | GPTQ-standard eval",
        flush=True,
    )
    print("=" * 156, flush=True)
    print(
        f"{'Config':<34} {'PPL':>10} {'Gain':>10} {'Delta':>10} {'Memory GB':>12} {'Params':>12} {'Layers':>8} {'Type':>18}",
        flush=True,
    )
    print("-" * 156, flush=True)
    for name in ordered_names:
        row = results[name]
        correction_layers = row.get("correction_layers")
        if isinstance(correction_layers, list):
            layer_text = str(len(correction_layers))
        else:
            layer_text = "0"
        print(
            f"{name:<34} {float(row['ppl']):>10.4f} {float(row.get('ppl_gain_vs_base', 0.0)):>+10.4f} {float(row.get('ppl_delta_vs_base', 0.0)):>+10.4f} {float(row['memory_gb']):>12.3f} {int(row.get('correction_parameter_count', 0)):>12d} {layer_text:>8} {str(row.get('correction_type', 'none')):>18}",
            flush=True,
        )
    print("-" * 156, flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    requested_plans = resolve_requested_plans(args.plans)
    correction_configs = build_correction_configs()
    all_plan_names = ["baseline_no_correction", *[cfg.name for cfg in correction_configs]]
    if requested_plans is not None:
        missing = sorted(requested_plans - set(all_plan_names))
        if missing:
            raise ValueError(f"Unknown plans requested: {', '.join(missing)}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    _tokenizer, calib_chunks, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    calib_chunks = calib_chunks[: max(1, min(int(args.retune_chunks), int(calib_chunks.shape[0])))]
    calib_info = dict(calib_info)
    calib_info["used_samples"] = int(calib_chunks.shape[0])
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}

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

    base_plan, base_extras = build_base_plan(calibration, text_config, non_expert_bytes, total_expert_elems)
    references = load_reference_rows(args.output_json)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: routed MoE experts use the existing joint_w1w2_with_topup masks; learned affine corrections repair expert outputs only and do not modify any stored weight tensor.",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "references": references,
            "experiment": "Iteration 25 learned expert-output affine correction on top of joint_w1w2_with_topup",
            "base_plan": base_plan.name,
            "base_plan_reference": references.get(base_plan.name),
            "correction_storage_dtype": "bfloat16-equivalent accounting",
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

    active_nonbaseline = [cfg for cfg in correction_configs if requested_plans is None or cfg.name in requested_plans]
    learned_corrections: dict[str, dict[int, AffineCorrection]] = {
        "correction_scalar": {},
        "correction_perchannel": {},
        "correction_bias_only": {},
    }
    layer_mse_rows: list[dict[str, Any]] = []

    if active_nonbaseline:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        embed_key, _norm_key, _lm_head_key = resolve_terminal_keys(weight_map)
        inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
        outs = torch.zeros_like(inps)
        causal_mask, position_embeddings = build_position_context(text_config, inps[0:1], device)

        cpu_device = torch.device("cpu")
        for layer_idx in range(text_config.num_hidden_layers):
            layer_type = layer_type_at(text_config, layer_idx)
            print(f"\n=== Fit layer {layer_idx + 1}/{text_config.num_hidden_layers} ({layer_type}) ===", flush=True)
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
            try:
                quantized_tensors = prepare_quantized_layer_tensors(base_plan, layer_idx, tensors, text_config)
                quant_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in quantized_tensors.items() if k.startswith("mlp.")}
            except torch.OutOfMemoryError:
                fit_backend = "cpu_quant_compare"
                print(
                    f"[fit] layer {layer_idx + 1}/{text_config.num_hidden_layers} hit CUDA OOM while materializing quantized expert tensors; falling back to CPU comparison tensors for fitting.",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                quantized_tensors = None
                quant_moe_tensors = None
                tensors_cpu = shorten_layer_tensors(layer_idx, raw_tensors, cpu_device, dtype)
                quantized_tensors_cpu = prepare_quantized_layer_tensors(base_plan, layer_idx, tensors_cpu, text_config)
                teacher_moe_tensors_cpu = {k.replace("mlp.", "", 1): v for k, v in tensors_cpu.items() if k.startswith("mlp.")}
                quant_moe_tensors_cpu = {
                    k.replace("mlp.", "", 1): v for k, v in quantized_tensors_cpu.items() if k.startswith("mlp.")
                }
            del raw_tensors
            stats = init_layer_fit_moments(text_config, device)

            for chunk_idx in range(int(inps.shape[0])):
                hidden_states = inps[chunk_idx].unsqueeze(0)
                residual, mlp_input = forward_attention_to_mlp_input(
                    hidden_states,
                    tensors,
                    layer_type,
                    text_config,
                    position_embeddings,
                    causal_mask,
                )
                if quant_moe_tensors is not None:
                    teacher_moe, _quant_moe = moe_forward_teacher_quant(
                        mlp_input,
                        teacher_moe_tensors,
                        quant_moe_tensors,
                        text_config,
                        stats,
                    )
                else:
                    teacher_moe = moe_forward_with_correction(mlp_input, teacher_moe_tensors, text_config, correction=None)
                    if teacher_moe_tensors_cpu is None or quant_moe_tensors_cpu is None:
                        raise RuntimeError("CPU fitting fallback is missing MoE tensors")
                    mlp_input_cpu = mlp_input.to(device=cpu_device, dtype=dtype, non_blocking=False)
                    _teacher_cpu, _quant_cpu = moe_forward_teacher_quant(
                        mlp_input_cpu,
                        teacher_moe_tensors_cpu,
                        quant_moe_tensors_cpu,
                        text_config,
                        stats,
                    )
                    del mlp_input_cpu, _teacher_cpu, _quant_cpu
                outs[chunk_idx] = residual + teacher_moe
                if should_log_chunk(chunk_idx, int(inps.shape[0])):
                    print(
                        f"[fit] layer {layer_idx + 1}/{text_config.num_hidden_layers} chunk {chunk_idx + 1}/{int(inps.shape[0])} ({fit_backend})",
                        flush=True,
                    )

            learned = learn_layer_corrections(stats)
            learned_corrections["correction_scalar"][layer_idx] = learned["scalar"]
            learned_corrections["correction_perchannel"][layer_idx] = learned["perchannel"]
            learned_corrections["correction_bias_only"][layer_idx] = learned["bias_only"]

            scalar_summary = summarize_correction(learned["scalar"], stats)
            perchannel_summary = summarize_correction(learned["perchannel"], stats)
            bias_summary = summarize_correction(learned["bias_only"], stats)
            layer_mse_rows.append({"layer": int(layer_idx), "teacher_quant_mse": scalar_summary["teacher_quant_mse"]})
            payload["fitting"][str(layer_idx)] = {
                "layer": int(layer_idx),
                "teacher_quant_mse": scalar_summary["teacher_quant_mse"],
                "active_experts": scalar_summary["active_experts"],
                "routed_pairs": scalar_summary["routed_pairs"],
                "scalar": scalar_summary,
                "perchannel": perchannel_summary,
                "bias_only": bias_summary,
            }
            atomic_json_dump(args.output_json, payload)
            print(
                f"[fit] layer {layer_idx + 1}/{text_config.num_hidden_layers} -> mse={scalar_summary['teacher_quant_mse']:.8f} | active={scalar_summary['active_experts']} | scalar beta_max={scalar_summary['beta_abs_max']:.6f}",
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
    else:
        layer_mse_rows = [
            {
                "layer": int(layer_idx),
                "teacher_quant_mse": 0.0,
            }
            for layer_idx in range(text_config.num_hidden_layers)
        ]

    ranked_layers = sorted(layer_mse_rows, key=lambda row: (float(row["teacher_quant_mse"]), -int(row["layer"])), reverse=True)
    top10_layers = [int(row["layer"]) for row in ranked_layers[: min(10, len(ranked_layers))]]
    payload.setdefault("metadata", {})["top10_layers_by_local_mse"] = top10_layers
    payload.setdefault("metadata", {})["layer_teacher_quant_mse"] = ranked_layers
    learned_corrections["correction_scalar_top10layers"] = filter_top_layers(learned_corrections["correction_scalar"], top10_layers)
    atomic_json_dump(args.output_json, payload)

    evaluation_rows: list[tuple[str, dict[int, AffineCorrection], dict[str, Any]]] = [
        (
            "baseline_no_correction",
            {},
            {
                **base_extras,
                "correction_type": "none",
                "correction_scope": "none",
                "correction_layers": [],
            },
        ),
        (
            "correction_scalar",
            learned_corrections["correction_scalar"],
            {
                **base_extras,
                "correction_type": "scalar_affine",
                "correction_scope": "all_layers",
                "correction_layers": list(range(text_config.num_hidden_layers)),
            },
        ),
        (
            "correction_perchannel",
            learned_corrections["correction_perchannel"],
            {
                **base_extras,
                "correction_type": "perchannel_affine",
                "correction_scope": "all_layers",
                "correction_layers": list(range(text_config.num_hidden_layers)),
            },
        ),
        (
            "correction_scalar_top10layers",
            learned_corrections["correction_scalar_top10layers"],
            {
                **base_extras,
                "correction_type": "scalar_affine",
                "correction_scope": "top10_local_mse_layers",
                "correction_layers": top10_layers,
            },
        ),
        (
            "correction_bias_only",
            learned_corrections["correction_bias_only"],
            {
                **base_extras,
                "correction_type": "bias_only",
                "correction_scope": "all_layers",
                "correction_layers": list(range(text_config.num_hidden_layers)),
            },
        ),
    ]

    for correction_name, corrections, extras in evaluation_rows:
        if requested_plans is not None and correction_name not in requested_plans:
            continue
        plan = build_evaluation_plan(base_plan, correction_name, text_config)
        evaluate_and_record_plan(
            args.model_id,
            payload,
            plan,
            correction_name,
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

    update_summary(payload, "baseline_no_correction")
    payload.setdefault("metadata", {})["elapsed_s"] = round(time.time() - overall_start, 1)
    atomic_json_dump(args.output_json, payload)

    if requested_plans is None or any(name in set(all_plan_names) for name in requested_plans):
        section = render_exploration_section(payload, references, top10_layers)
        upsert_exploration_section(args.exploration_md, section)
        print(f"[exploration] updated {args.exploration_md}", flush=True)

    print_results_table(payload, "baseline_no_correction")


if __name__ == "__main__":
    main()

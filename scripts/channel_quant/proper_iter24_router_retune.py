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
    CALIBRATION_SAMPLES,
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter24_router_retune.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
SECTION_MARKER = "## [27] Iteration 24 - Quantization-Aware Router Retuning"
TRAIN_TEMPERATURE = 1.0
KL_TRAIN_TOKENS = 128


@dataclass(frozen=True)
class RetuneConfig:
    name: str
    objective: str
    steps: int
    lr: float
    batch_chunks: int
    shortlist_size: int
    patience: int


@dataclass
class LayerTargets:
    residuals: list[torch.Tensor]
    mlp_inputs: list[torch.Tensor]
    teacher_outputs: list[torch.Tensor]
    teacher_router_logits: list[torch.Tensor]


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
        help="Comma-separated subset of plan names to evaluate; default runs baseline plus all Iteration 24 retune configs.",
    )
    parser.add_argument(
        "--retune-chunks",
        type=int,
        default=CALIBRATION_SAMPLES,
        help="Number of GPTQ-standard calibration chunks to use for target collection and bias learning.",
    )
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on evaluation chunks for smoke tests; 0 means full WikiText-2 test.",
    )
    parser.add_argument("--mse-batch-chunks", type=int, default=4)
    parser.add_argument("--kl-batch-chunks", type=int, default=1)
    parser.add_argument("--shortlist-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def should_log_chunk(chunk_idx: int, total_chunks: int) -> bool:
    return chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == total_chunks


def build_retune_configs(args: argparse.Namespace) -> list[RetuneConfig]:
    shortlist_size = max(4, int(args.shortlist_size))
    mse_batch = max(1, int(args.mse_batch_chunks))
    kl_batch = max(1, int(args.kl_batch_chunks))
    lr = float(args.lr)
    return [
        RetuneConfig("router_retune_mse_10steps", "mse", 10, lr, mse_batch, shortlist_size, 4),
        RetuneConfig("router_retune_mse_50steps", "mse", 50, lr, mse_batch, shortlist_size, 8),
        RetuneConfig("router_retune_kl_10steps", "kl", 10, lr, kl_batch, shortlist_size, 4),
        RetuneConfig("router_retune_mse_100steps", "mse", 100, lr, mse_batch, shortlist_size, 12),
    ]


def summarize_biases(router_biases: dict[int, torch.Tensor], config: Any) -> dict[str, Any]:
    if not router_biases:
        return {
            "layers": int(config.num_hidden_layers),
            "experts_per_layer": int(config.num_experts),
            "bias_l2_mean": 0.0,
            "bias_l2_max": 0.0,
            "bias_abs_mean": 0.0,
            "bias_abs_max": 0.0,
            "nonzero_fraction": 0.0,
        }

    stacked = torch.stack([router_biases[layer_idx].detach().cpu().to(torch.float32) for layer_idx in sorted(router_biases)], dim=0)
    l2 = torch.linalg.vector_norm(stacked, dim=1)
    abs_bias = stacked.abs()
    return {
        "layers": int(stacked.shape[0]),
        "experts_per_layer": int(stacked.shape[1]),
        "bias_l2_mean": round(float(l2.mean().item()), 8),
        "bias_l2_max": round(float(l2.max().item()), 8),
        "bias_abs_mean": round(float(abs_bias.mean().item()), 8),
        "bias_abs_max": round(float(abs_bias.max().item()), 8),
        "nonzero_fraction": round(float((abs_bias > 1e-9).to(torch.float32).mean().item()), 8),
    }


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


def moe_forward_hard(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    router_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
    if router_bias is not None:
        router_logits = router_logits + router_bias.to(device=router_logits.device).view(1, -1)
    routing_probs = torch.softmax(router_logits, dim=1)
    selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1).indices
    selected_weights = routing_probs.gather(1, selected_experts)
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    selected_weights = selected_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((flat.shape[0], hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, moe_tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, moe_tensors["experts.down_proj"][expert_idx])
        weighted = selected_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden
        final_hidden_states.index_add_(0, token_idx, weighted.to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits, selected_experts


def moe_forward_shortlist_surrogate(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    router_bias: torch.Tensor,
    shortlist_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, moe_tensors["gate.weight"]).float() + router_bias.view(1, -1)
    shortlist = min(max(int(shortlist_size), config.num_experts_per_tok), int(config.num_experts))
    shortlist_indices = torch.topk(router_logits, k=shortlist, dim=-1).indices
    shortlist_logits = router_logits.gather(1, shortlist_indices)
    shortlist_weights = torch.softmax(shortlist_logits / TRAIN_TEMPERATURE, dim=-1).to(hidden_states.dtype)

    final_hidden_states = torch.zeros((flat.shape[0], hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    expert_counts = torch.bincount(shortlist_indices.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, shortlist_pos = torch.where(shortlist_indices == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, moe_tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, moe_tensors["experts.down_proj"][expert_idx])
        weighted = shortlist_weights[token_idx, shortlist_pos].unsqueeze(-1) * current_hidden
        final_hidden_states.index_add_(0, token_idx, weighted.to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits


def compute_shift_kl(
    teacher_hidden: torch.Tensor,
    quant_hidden: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    teacher_hidden = rms_norm_qwen3_next(teacher_hidden, final_norm, config.rms_norm_eps)
    quant_hidden = rms_norm_qwen3_next(quant_hidden, final_norm, config.rms_norm_eps)
    total_kl = torch.zeros((), dtype=torch.float32, device=teacher_hidden.device)
    total_tokens = teacher_hidden.shape[1]
    chunks = 0
    for start in range(0, total_tokens, LOGITS_CHUNK_TOKENS):
        end = min(start + LOGITS_CHUNK_TOKENS, total_tokens)
        teacher_logits = F.linear(teacher_hidden[:, start:end, :].float(), lm_head.float())
        quant_logits = F.linear(quant_hidden[:, start:end, :].float(), lm_head.float())
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        quant_log_probs = torch.log_softmax(quant_logits, dim=-1)
        total_kl = total_kl + F.kl_div(quant_log_probs, teacher_probs, reduction="batchmean")
        chunks += 1
        del teacher_logits, quant_logits, teacher_probs, quant_log_probs
    return total_kl / float(max(chunks, 1))


def run_suffix_hidden_states(
    store: WeightStore,
    base_plan: EvalPlan,
    config: Any,
    start_layer_idx: int,
    hidden_states: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    causal_mask: torch.Tensor,
    quantized: bool,
    router_biases: dict[int, torch.Tensor] | None = None,
    cpu_quantize: bool = False,
) -> torch.Tensor:
    current_hidden = hidden_states
    for layer_idx in range(start_layer_idx, config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        if quantized and cpu_quantize:
            cpu_device = torch.device("cpu")
            tensors_cpu = shorten_layer_tensors(layer_idx, raw_tensors, cpu_device, dtype)
            del raw_tensors
            tensors_cpu = prepare_quantized_layer_tensors(base_plan, layer_idx, tensors_cpu, config)
            tensors = {
                name: tensor.to(device=device, dtype=dtype, non_blocking=True)
                for name, tensor in tensors_cpu.items()
            }
            release_tensors(tensors_cpu)
        else:
            tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
            del raw_tensors
            if quantized:
                tensors = prepare_quantized_layer_tensors(base_plan, layer_idx, tensors, config)
        post_attention, mlp_input = forward_attention_to_mlp_input(current_hidden, tensors, layer_type, config, position_embeddings, causal_mask)
        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        router_bias = None if router_biases is None else router_biases.get(layer_idx)
        moe_out, _router_logits, _selected = moe_forward_hard(mlp_input, moe_tensors, config, router_bias)
        current_hidden = post_attention + moe_out
        release_tensors(tensors)
    return current_hidden


def collect_layer_targets(
    inps: torch.Tensor,
    layer_idx: int,
    tensors: dict[str, torch.Tensor],
    config: Any,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    causal_mask: torch.Tensor,
) -> tuple[LayerTargets, torch.Tensor]:
    chunk_count = int(inps.shape[0])
    outs = torch.zeros_like(inps)
    residuals: list[torch.Tensor] = []
    mlp_inputs: list[torch.Tensor] = []
    teacher_outputs: list[torch.Tensor] = []
    teacher_router_logits: list[torch.Tensor] = []
    layer_type = layer_type_at(config, layer_idx)
    moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}

    for chunk_idx in range(chunk_count):
        hidden_states = inps[chunk_idx].unsqueeze(0)
        residual, mlp_input = forward_attention_to_mlp_input(hidden_states, tensors, layer_type, config, position_embeddings, causal_mask)
        teacher_out, router_logits, _selected = moe_forward_hard(mlp_input, moe_tensors, config, router_bias=None)
        outs[chunk_idx] = residual + teacher_out
        residuals.append(residual.squeeze(0).detach().cpu().to(torch.bfloat16).contiguous())
        mlp_inputs.append(mlp_input.squeeze(0).detach().cpu().to(torch.bfloat16).contiguous())
        teacher_outputs.append(teacher_out.squeeze(0).detach().cpu().to(torch.bfloat16).contiguous())
        teacher_router_logits.append(router_logits.detach().cpu().to(torch.float32).contiguous())
        if should_log_chunk(chunk_idx, chunk_count):
            print(
                f"[collect] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{chunk_count}",
                flush=True,
            )

    return LayerTargets(residuals, mlp_inputs, teacher_outputs, teacher_router_logits), outs


def compute_route_change_fraction(
    layer_targets: LayerTargets,
    quant_moe_tensors: dict[str, torch.Tensor],
    config: Any,
    router_bias: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    max_chunks: int = 4,
) -> float:
    compared = min(max_chunks, len(layer_targets.mlp_inputs))
    changed = 0.0
    total = 0.0
    for chunk_idx in range(compared):
        mlp_input = layer_targets.mlp_inputs[chunk_idx].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
        _base_out, _base_logits, base_selected = moe_forward_hard(mlp_input, quant_moe_tensors, config, router_bias=None)
        _new_out, _new_logits, new_selected = moe_forward_hard(mlp_input, quant_moe_tensors, config, router_bias=router_bias)
        total += float(base_selected.numel())
        changed += float((base_selected != new_selected).to(torch.float32).sum().item())
        del mlp_input, base_selected, new_selected
    return 0.0 if total <= 0.0 else float(changed / total)


def optimize_bias_mse(
    retune_config: RetuneConfig,
    layer_targets: LayerTargets,
    quant_moe_tensors: dict[str, torch.Tensor],
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    bias = torch.zeros(config.num_experts, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=retune_config.lr)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(1000 + retune_config.steps)
    best_loss = float("inf")
    best_bias = bias.detach().clone()
    no_improve = 0
    history: list[float] = []
    chunk_count = len(layer_targets.mlp_inputs)

    for _step_idx in range(retune_config.steps):
        perm = torch.randperm(chunk_count, generator=rng)
        batch_indices = perm[: min(retune_config.batch_chunks, chunk_count)].tolist()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), dtype=torch.float32, device=device)
        for chunk_idx in batch_indices:
            mlp_input = layer_targets.mlp_inputs[chunk_idx].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
            teacher_out = layer_targets.teacher_outputs[chunk_idx].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
            quant_out, _router_logits = moe_forward_shortlist_surrogate(
                mlp_input,
                quant_moe_tensors,
                config,
                bias,
                retune_config.shortlist_size,
            )
            loss_accum = loss_accum + F.mse_loss(quant_out.float(), teacher_out.float())
            del mlp_input, teacher_out, quant_out
        loss = loss_accum / float(max(len(batch_indices), 1))
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        history.append(loss_value)
        if loss_value + 1e-9 < best_loss:
            best_loss = loss_value
            best_bias = bias.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= retune_config.patience:
                break

    final_bias = best_bias.detach().cpu().to(torch.float32)
    route_change = compute_route_change_fraction(
        layer_targets,
        quant_moe_tensors,
        config,
        best_bias.detach(),
        device,
        dtype,
    )
    return final_bias, {
        "objective": retune_config.objective,
        "steps_requested": int(retune_config.steps),
        "steps_ran": int(len(history)),
        "best_loss": round(best_loss, 8),
        "final_loss": round(history[-1], 8) if history else None,
        "batch_chunks": int(retune_config.batch_chunks),
        "shortlist_size": int(retune_config.shortlist_size),
        "lr": float(retune_config.lr),
        "bias_l2": round(float(torch.linalg.vector_norm(final_bias).item()), 8),
        "bias_abs_max": round(float(final_bias.abs().max().item()), 8),
        "route_change_fraction_preview": round(route_change, 8),
    }


def optimize_bias_kl(
    retune_config: RetuneConfig,
    layer_idx: int,
    layer_targets: LayerTargets,
    quant_moe_tensors: dict[str, torch.Tensor],
    store: WeightStore,
    base_plan: EvalPlan,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    _position_embeddings: tuple[torch.Tensor, torch.Tensor],
    _causal_mask: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    bias = torch.zeros(config.num_experts, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=retune_config.lr)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(2000 + layer_idx)
    best_loss = float("inf")
    best_bias = bias.detach().clone()
    no_improve = 0
    history: list[float] = []
    chunk_count = len(layer_targets.mlp_inputs)
    train_seq_len = min(KL_TRAIN_TOKENS, int(layer_targets.residuals[0].shape[0]))
    context_probe = layer_targets.residuals[0][:train_seq_len, :].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    kl_causal_mask, kl_position_embeddings = build_position_context(config, context_probe, device)
    del context_probe

    for _step_idx in range(retune_config.steps):
        perm = torch.randperm(chunk_count, generator=rng)
        batch_indices = perm[: min(retune_config.batch_chunks, chunk_count)].tolist()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), dtype=torch.float32, device=device)
        for chunk_idx in batch_indices:
            residual = layer_targets.residuals[chunk_idx][:train_seq_len, :].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
            mlp_input = layer_targets.mlp_inputs[chunk_idx][:train_seq_len, :].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
            teacher_out = layer_targets.teacher_outputs[chunk_idx][:train_seq_len, :].unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)

            with torch.no_grad():
                teacher_hidden = residual + teacher_out
                teacher_suffix = run_suffix_hidden_states(
                    store,
                    base_plan,
                    config,
                    layer_idx + 1,
                    teacher_hidden,
                    device,
                    dtype,
                    kl_position_embeddings,
                    kl_causal_mask,
                    quantized=False,
                )

            quant_out, _router_logits = moe_forward_shortlist_surrogate(
                mlp_input,
                quant_moe_tensors,
                config,
                bias,
                retune_config.shortlist_size,
            )
            quant_hidden = residual + quant_out
            quant_suffix = run_suffix_hidden_states(
                store,
                base_plan,
                config,
                layer_idx + 1,
                quant_hidden,
                device,
                dtype,
                kl_position_embeddings,
                kl_causal_mask,
                quantized=True,
                cpu_quantize=True,
            )
            loss_accum = loss_accum + compute_shift_kl(teacher_suffix, quant_suffix, final_norm, lm_head, config)
            del residual, mlp_input, teacher_out, teacher_hidden, teacher_suffix, quant_out, quant_hidden, quant_suffix

        loss = loss_accum / float(max(len(batch_indices), 1))
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        history.append(loss_value)
        if loss_value + 1e-9 < best_loss:
            best_loss = loss_value
            best_bias = bias.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= retune_config.patience:
                break

    final_bias = best_bias.detach().cpu().to(torch.float32)
    route_change = compute_route_change_fraction(
        layer_targets,
        quant_moe_tensors,
        config,
        best_bias.detach(),
        device,
        dtype,
    )
    return final_bias, {
        "objective": retune_config.objective,
        "steps_requested": int(retune_config.steps),
        "steps_ran": int(len(history)),
        "best_loss": round(best_loss, 8),
        "final_loss": round(history[-1], 8) if history else None,
        "batch_chunks": int(retune_config.batch_chunks),
        "train_tokens": int(train_seq_len),
        "shortlist_size": int(retune_config.shortlist_size),
        "lr": float(retune_config.lr),
        "bias_l2": round(float(torch.linalg.vector_norm(final_bias).item()), 8),
        "bias_abs_max": round(float(final_bias.abs().max().item()), 8),
        "route_change_fraction_preview": round(route_change, 8),
    }


def learn_router_biases(
    retune_config: RetuneConfig,
    layer_idx: int,
    layer_targets: LayerTargets,
    quant_moe_tensors: dict[str, torch.Tensor],
    store: WeightStore,
    base_plan: EvalPlan,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    causal_mask: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if retune_config.objective == "mse":
        return optimize_bias_mse(retune_config, layer_targets, quant_moe_tensors, config, device, dtype)
    if retune_config.objective == "kl":
        return optimize_bias_kl(
            retune_config,
            layer_idx,
            layer_targets,
            quant_moe_tensors,
            store,
            base_plan,
            config,
            device,
            dtype,
            position_embeddings,
            causal_mask,
            final_norm,
            lm_head,
        )
    raise ValueError(f"Unsupported objective: {retune_config.objective}")


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


@torch.inference_mode()
def evaluate_plan_with_router_bias(
    plan: EvalPlan,
    router_biases: dict[int, torch.Tensor],
    test_ids: torch.Tensor,
    eval_info_override: dict[str, Any],
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    nsamples = int(test_ids.numel() // SEQLEN)
    eval_chunks = test_ids[:, : nsamples * SEQLEN].view(nsamples, SEQLEN)
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

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
        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        router_bias = router_biases.get(layer_idx)

        for chunk_idx in range(nsamples):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual, mlp_input = forward_attention_to_mlp_input(hidden_states, tensors, layer_type, config, position_embeddings, causal_mask)
            moe_out, _router_logits, _selected = moe_forward_hard(mlp_input, moe_tensors, config, router_bias)
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
    del store, eval_info_override
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(ppl.item()), float(mean_nll.item()), nsamples


def evaluate_and_record_plan(
    payload: dict[str, Any],
    plan: EvalPlan,
    router_biases: dict[int, torch.Tensor],
    extras: dict[str, Any],
    output_json: Path,
    config: Any,
    total_expert_elems: int,
    test_ids: torch.Tensor,
    eval_info_override: dict[str, Any],
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    result_rows = payload.setdefault("results", {})
    if plan.name in result_rows:
        print(f"[skip] {plan.name} already present", flush=True)
        return result_rows[plan.name]

    print(f"\n=== Eval: {plan.name} ===", flush=True)
    start_time = time.time()
    ppl, nll, nsamples = evaluate_plan_with_router_bias(
        plan,
        router_biases,
        test_ids,
        eval_info_override,
        config,
        weight_map,
        snapshot_dir,
        device,
        dtype,
    )
    elapsed = time.time() - start_time
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
        **extras,
    }
    result_rows[plan.name] = row
    atomic_json_dump(output_json, payload)
    print(
        f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def update_summary(payload: dict[str, Any], base_plan_name: str) -> None:
    results = payload.get("results", {})
    if base_plan_name not in results:
        return
    base_ppl = float(results[base_plan_name]["ppl"])
    ranked = []
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
                "objective": row.get("retune_objective"),
                "bias_abs_max": row.get("bias_abs_max"),
            }
        )
    ranked.sort(key=lambda item: (float(item["ppl"]), float(item["memory_gb"]), item["name"]))
    payload["summary"] = {
        "base_plan": base_plan_name,
        "base_ppl": base_ppl,
        "best_plan": ranked[0] if ranked else None,
        "ranked_results": ranked,
    }


def render_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    results = payload.get("results", {})
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_name, best_row = ordered[0]
    baseline_row = results.get("baseline_no_retune")
    baseline_ppl = None if baseline_row is None else float(baseline_row["ppl"])
    best_ref = references.get("joint_w1w2_with_topup", references.get("baseline_no_retune", {})).get("ppl")

    lines = [SECTION_MARKER, ""]
    lines.append(
        "**Approach**: Reused the current `joint_w1w2_with_topup` expert masks, collected BF16 teacher MoE outputs on the full 128x2048 calibration set, then learned one 256-way router-logit bias vector per MoE layer so routing better matches the quantized expert landscape."
    )
    lines.append("")
    lines.append(
        "**Training note**: Bias learning keeps all weights frozen. MSE configs optimize local MoE output reconstruction; the KL config backpropagates through the downstream quantized suffix. Training uses a shortlist soft-routing surrogate, but evaluation uses hard top-k routing with the learned biases."
    )
    lines.append("")
    lines.append("**Results**:")
    for name, row in ordered:
        lines.append(
            f"- `{name}` -> PPL {float(row['ppl']):.6f} @ {float(row['memory_gb']):.3f} GB, gain vs baseline {float(row.get('ppl_gain_vs_base', 0.0)):+.6f}"
        )
    lines.append("")
    if baseline_ppl is None:
        lines.append(f"**Insight**: `{best_name}` is the best Iteration 24 row, but the baseline row is missing so the gain cannot be grounded against the no-retune control.")
    else:
        lines.append(
            f"**Insight**: `{best_name}` is the best Iteration 24 row at {float(best_row['ppl']):.6f}, which is {baseline_ppl - float(best_row['ppl']):+.6f} PPL versus the no-retune control."
        )
    lines.append("")
    if best_ref is None:
        lines.append("**Next**: If router retuning moves the ranking at all, repeat with a stricter routing objective or larger shortlist before spending more budget on static mask search.")
    elif float(best_row["ppl"]) < float(best_ref):
        lines.append("**Next**: Router bias helped enough to beat the standing joint-topup reference, so the next iteration should test stronger router-side objectives instead of more expert-mask sweeps.")
    else:
        lines.append("**Next**: If the gains stay flat versus the standing reference, treat the remaining floor as quantization-noise dominated rather than routing-limited.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    requested_plans = resolve_requested_plans(args.plans)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    retune_configs = build_retune_configs(args)
    all_plan_names = ["baseline_no_retune", *[cfg.name for cfg in retune_configs]]
    if requested_plans is not None:
        missing = sorted(requested_plans - set(all_plan_names))
        if missing:
            raise ValueError(f"Unknown plans requested: {', '.join(missing)}")
    active_retune_configs = retune_configs if requested_plans is None else [cfg for cfg in retune_configs if cfg.name in requested_plans]

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
            "quantization": "simulated quantization: routed MoE experts use the existing joint_w1w2_with_topup masks; router bias learning adds only per-expert logit offsets and does not modify any weight tensor.",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "references": references,
            "experiment": "Iteration 24 quantization-aware router retuning on top of joint_w1w2_with_topup",
            "base_plan": base_plan.name,
            "base_plan_reference": references.get(base_plan.name),
            "router_retune": {
                "retune_chunks": int(calib_chunks.shape[0]),
                "mse_batch_chunks": int(args.mse_batch_chunks),
                "kl_batch_chunks": int(args.kl_batch_chunks),
                "shortlist_size": int(args.shortlist_size),
                "lr": float(args.lr),
                "teacher_router_logits": "recorded per layer/chunk during BF16 target collection",
                "note": "proper_eval keeps gate.weight in BF16, so the learned correction targets the deployed quantized-expert landscape without quantizing router weights.",
            },
        },
        "retuning": {},
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
                payload.setdefault("retuning", {})
                payload.setdefault("results", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    root_tensors = store.load_tensors([norm_key, lm_head_key])
    final_norm = root_tensors[norm_key].to(device=device, dtype=dtype, non_blocking=True)
    lm_head = root_tensors[lm_head_key].to(device=device, dtype=dtype, non_blocking=True)
    del root_tensors

    router_biases_by_plan: dict[str, dict[int, torch.Tensor]] = {cfg.name: {} for cfg in active_retune_configs}
    layer_summaries_by_plan: dict[str, list[dict[str, Any]]] = {cfg.name: [] for cfg in active_retune_configs}

    if active_retune_configs:
        inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
        causal_mask, position_embeddings = build_position_context(text_config, inps[0:1], device)

        for layer_idx in range(text_config.num_hidden_layers):
            layer_type = layer_type_at(text_config, layer_idx)
            print(f"\n=== Retune layer {layer_idx + 1}/{text_config.num_hidden_layers} ({layer_type}) ===", flush=True)
            raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
            tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
            del raw_tensors
            quantized_tensors = prepare_quantized_layer_tensors(base_plan, layer_idx, tensors, text_config)
            quant_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in quantized_tensors.items() if k.startswith("mlp.")}
            layer_targets, outs = collect_layer_targets(inps, layer_idx, tensors, text_config, position_embeddings, causal_mask)

            teacher_router_stats = torch.stack(
                [logits.max(dim=1).values.mean() - logits.mean(dim=1) for logits in layer_targets.teacher_router_logits],
                dim=0,
            )
            payload["retuning"].setdefault("teacher_router_stats", {})[str(layer_idx)] = {
                "mean_max_minus_mean": round(float(teacher_router_stats.mean().item()), 8),
                "std_max_minus_mean": round(float(teacher_router_stats.std(unbiased=False).item()), 8),
            }

            for retune_config in active_retune_configs:
                start = time.time()
                bias, layer_summary = learn_router_biases(
                    retune_config,
                    layer_idx,
                    layer_targets,
                    quant_moe_tensors,
                    store,
                    base_plan,
                    text_config,
                    device,
                    dtype,
                    position_embeddings,
                    causal_mask,
                    final_norm,
                    lm_head,
                )
                elapsed = time.time() - start
                router_biases_by_plan[retune_config.name][layer_idx] = bias
                summary_row = {
                    "layer": int(layer_idx),
                    "time_s": round(elapsed, 2),
                    **layer_summary,
                }
                layer_summaries_by_plan[retune_config.name].append(summary_row)
                print(
                    f"[{retune_config.name}] layer {layer_idx + 1}/{text_config.num_hidden_layers} -> best_loss={summary_row['best_loss']:.8f} | bias_l2={summary_row['bias_l2']:.6f} | route_change={summary_row['route_change_fraction_preview']:.4f} | time={elapsed:.1f}s",
                    flush=True,
                )

            payload["retuning"].setdefault("layers_completed", 0)
            payload["retuning"]["layers_completed"] = int(layer_idx + 1)
            for retune_config in active_retune_configs:
                payload["retuning"][retune_config.name] = {
                    "objective": retune_config.objective,
                    "steps": int(retune_config.steps),
                    "batch_chunks": int(retune_config.batch_chunks),
                    "shortlist_size": int(retune_config.shortlist_size),
                    "lr": float(retune_config.lr),
                    "layer_summaries": layer_summaries_by_plan[retune_config.name],
                    "bias_summary": summarize_biases(router_biases_by_plan[retune_config.name], text_config),
                }
            atomic_json_dump(args.output_json, payload)

            release_tensors(tensors)
            release_tensors(quantized_tensors)
            inps = outs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        release_tensors({"inps": inps, "final_norm": final_norm, "lm_head": lm_head, "causal_mask": causal_mask})
    else:
        release_tensors({"final_norm": final_norm, "lm_head": lm_head})

    baseline_biases = {layer_idx: torch.zeros(text_config.num_experts, dtype=torch.float32) for layer_idx in range(text_config.num_hidden_layers)}
    evaluation_plans: list[tuple[str, dict[int, torch.Tensor], dict[str, Any]]] = [
        (
            "baseline_no_retune",
            baseline_biases,
            {
                **base_extras,
                "retune_objective": "none",
                "retune_steps": 0,
                **summarize_biases({}, text_config),
            },
        )
    ]
    for retune_config in active_retune_configs:
        evaluation_plans.append(
            (
                retune_config.name,
                router_biases_by_plan[retune_config.name],
                {
                    **base_extras,
                    "retune_objective": retune_config.objective,
                    "retune_steps": int(retune_config.steps),
                    **payload["retuning"][retune_config.name]["bias_summary"],
                },
            )
        )

    for plan_name, router_biases, extras in evaluation_plans:
        if requested_plans is not None and plan_name not in requested_plans:
            continue
        eval_plan = EvalPlan(
            name=plan_name,
            description=(
                "Current best joint_w1w2_with_topup plan without router retuning."
                if plan_name == "baseline_no_retune"
                else f"Apply learned per-expert router-logit biases to `{base_plan.name}` using the {extras['retune_objective']} objective."
            ),
            mode=base_plan.mode,
            memory_gb=base_plan.memory_gb,
            fp8_weights=base_plan.fp8_weights,
            w1_pair_masks=base_plan.w1_pair_masks,
            w2_channel_masks=base_plan.w2_channel_masks,
        )
        evaluate_and_record_plan(
            payload,
            eval_plan,
            router_biases,
            extras,
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            eval_info,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )

    update_summary(payload, "baseline_no_retune")
    payload.setdefault("metadata", {})["elapsed_s"] = round(time.time() - overall_start, 1)
    atomic_json_dump(args.output_json, payload)

    if requested_plans is None or any(name in {"baseline_no_retune", *[cfg.name for cfg in retune_configs]} for name in requested_plans):
        section = render_exploration_section(payload, references)
        upsert_exploration_section(args.exploration_md, section)
        print(f"[exploration] updated {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

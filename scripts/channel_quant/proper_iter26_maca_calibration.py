#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

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
import torch.nn.functional as F
from datasets import load_dataset

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    ExpertMomentState,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    finalize_layer_metrics,
    init_expert_moment_state,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
    upsert_exploration_section,
)
from proper_iter01 import build_mxmoe_topup_masks, build_plan_from_masks
from proper_iter07 import build_joint_with_topup_masks
from proper_iter10_novel_perchannel import (
    NovelMomentState,
    build_global_fraction_masks,
    finalize_hessian_normalized_layer,
    finalize_router_affinity_layer,
    init_novel_state,
)
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter19_union_residual import load_json, load_reference_rows, resolve_requested_plans
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter26_maca_calibration.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
SECTION_MARKER = "## [29] Iteration 26 - MaCa Multi-Scale Calibration"

MACA_LENGTHS = (128, 512, 2048, 4096)
MACA_CHUNKS_PER_LENGTH = 32
MACA_TOTAL_CHUNKS = len(MACA_LENGTHS) * MACA_CHUNKS_PER_LENGTH
MACA_PAD_LENGTH = max(MACA_LENGTHS)
JOINT_TOPUP_FRACTION = 0.08
MXMOE_TOPUP_FRACTION = 0.05
PERCHANNEL_W1_FRACTION = 0.04
PERCHANNEL_W2_FRACTION = 0.16

SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "mxmoe_block_plus_channel_topup": 6.5763,
    "router_affinity_weighted_w1_4_w2_16": 6.5796,
}


@dataclass(frozen=True)
class VariableLengthCalibrationSet:
    chunks: torch.Tensor
    actual_lengths: torch.Tensor
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 26 plans.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def should_log_chunk(chunk_idx: int, total_chunks: int) -> bool:
    return chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == total_chunks


def load_wikitext_train_ids(tokenizer: Any) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_text = "\n\n".join(dataset["text"])
    encoded = tokenizer(train_text, return_tensors="pt")
    return torch.as_tensor(encoded.input_ids, dtype=torch.long)


def build_maca_calibration_set(tokenizer: Any, seed: int) -> VariableLengthCalibrationSet:
    rng = random.Random(seed)
    train_ids = load_wikitext_train_ids(tokenizer)
    padded_chunks: list[torch.Tensor] = []
    actual_lengths: list[int] = []

    for length in MACA_LENGTHS:
        max_start = int(train_ids.shape[1]) - int(length) - 1
        if max_start < 0:
            raise RuntimeError(f"WikiText-2 train is too short for MaCa length {length}")
        for _ in range(MACA_CHUNKS_PER_LENGTH):
            start = rng.randint(0, max_start)
            sample = train_ids[:, start : start + length]
            padded = torch.zeros((1, MACA_PAD_LENGTH), dtype=torch.long)
            padded[:, :length] = sample
            padded_chunks.append(padded)
            actual_lengths.append(int(length))

    chunks = torch.cat(padded_chunks, dim=0).contiguous()
    actual_lengths_tensor = torch.as_tensor(actual_lengths, dtype=torch.long)
    actual_token_total = int(actual_lengths_tensor.sum().item())
    padded_token_total = int(chunks.numel())
    metadata = {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "train",
        "tokenizer": tokenizer.name_or_path,
        "random_seed": int(seed),
        "lengths": [int(length) for length in MACA_LENGTHS],
        "chunks_per_length": int(MACA_CHUNKS_PER_LENGTH),
        "samples": int(chunks.shape[0]),
        "sample_shape": [int(chunks.shape[0]), int(chunks.shape[1])],
        "padding_value": 0,
        "actual_token_total": int(actual_token_total),
        "padded_token_total": int(padded_token_total),
        "real_token_fraction": round(float(actual_token_total) / float(max(padded_token_total, 1)), 6),
        "train_tokens": int(train_ids.shape[1]),
        "standard_reference_tokens": int(SEQLEN * chunks.shape[0]),
    }
    return VariableLengthCalibrationSet(chunks=chunks, actual_lengths=actual_lengths_tensor, metadata=metadata)


def collect_layer_calibration_valid_tokens(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    config: Any,
    state: ExpertMomentState,
    actual_length: int,
) -> torch.Tensor:
    if actual_length <= 0:
        return torch.zeros_like(hidden_states)

    valid_hidden_states = hidden_states[:, :actual_length, :]
    batch_size, sequence_length, hidden_dim = valid_hidden_states.shape
    flat = valid_hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(valid_hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)

    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    state.counts.add_(expert_counts)

    active_experts = torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist()
    for expert_idx in active_experts:
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        weights = routing_weights[token_idx, route_pos].unsqueeze(-1).to(torch.float32)

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        full_out = F.linear(full_hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (weights * full_out.float()).to(valid_hidden_states.dtype))

        current_state_f = current_state.float()
        full_hidden_f = full_hidden.float()
        state.input_sum1[expert_idx].add_(current_state_f.sum(dim=0))
        squared = current_state_f.square()
        state.input_sum2[expert_idx].add_(squared.sum(dim=0))
        cubed = squared * current_state_f
        state.input_sum3[expert_idx].add_(cubed.sum(dim=0))
        state.input_sum4[expert_idx].add_((cubed * current_state_f).sum(dim=0))
        state.inter_sum1[expert_idx].add_(full_hidden_f.sum(dim=0))
        inter_squared = full_hidden_f.square()
        state.inter_sum2[expert_idx].add_(inter_squared.sum(dim=0))
        inter_cubed = inter_squared * full_hidden_f
        state.inter_sum3[expert_idx].add_(inter_cubed.sum(dim=0))
        state.inter_sum4[expert_idx].add_((inter_cubed * full_hidden_f).sum(dim=0))

        q_gate_up = F.linear(current_state, fp4_gate_up[expert_idx])
        q_gate, q_up = q_gate_up.chunk(2, dim=-1)
        q_hidden_w1 = F.silu(q_gate) * q_up
        q_out_w1 = F.linear(q_hidden_w1, down_proj[expert_idx]).float()
        q_out_w2 = F.linear(full_hidden, fp4_down[expert_idx]).float()
        full_out_f = full_out.float()
        state.mxmoe_w1_sq[expert_idx] += torch.sum((weights * (q_out_w1 - full_out_f)).square())
        state.mxmoe_w2_sq[expert_idx] += torch.sum((weights * (q_out_w2 - full_out_f)).square())

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    valid_output = final_hidden_states + (shared_out * shared_gate_value)

    padded_output = torch.zeros_like(hidden_states)
    padded_output[:, :actual_length, :] = valid_output.view(batch_size, sequence_length, hidden_dim)
    return padded_output


@torch.inference_mode()
def run_maca_calibration(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    actual_lengths: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> CalibrationArtifacts:
    print("\n=== Calibration (MaCa multi-scale, padded-to-4096 / valid-token stats) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[maca-calib] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_expert_moment_state(config, device)

        for chunk_idx in range(inps.shape[0]):
            actual_length = int(actual_lengths[chunk_idx].item())
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", "", 1): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", "", 1): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = collect_layer_calibration_valid_tokens(mlp_input, moe_tensors, fp4_gate_up, fp4_down, config, state, actual_length)
            outs[chunk_idx] = residual + moe_out
            if should_log_chunk(chunk_idx, int(inps.shape[0])):
                print(
                    f"[maca-calib] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]} | actual_len={actual_length}",
                    flush=True,
                )

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


def collect_layer_novel_metrics_valid_tokens(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: NovelMomentState,
    actual_length: int,
) -> torch.Tensor:
    if actual_length <= 0:
        return torch.zeros_like(hidden_states)

    valid_hidden_states = hidden_states[:, :actual_length, :]
    batch_size, sequence_length, hidden_dim = valid_hidden_states.shape
    flat = valid_hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(valid_hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)

    gate_up_proj = moe_tensors["experts.gate_up_proj"]
    down_proj = moe_tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    state.counts.add_(expert_counts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        route_prob = routing_probs[token_idx, expert_idx].unsqueeze(-1).to(torch.float32)
        route_weight = routing_weights[token_idx, route_pos].unsqueeze(-1)

        current_sq = current_state.float().square()
        state.input_sq_sum[expert_idx].add_(current_sq.sum(dim=0))
        state.router_input_sq_sum[expert_idx].add_((route_prob * current_sq).sum(dim=0))

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        full_hidden_sq = full_hidden.float().square()
        state.inter_sq_sum[expert_idx].add_(full_hidden_sq.sum(dim=0))
        state.router_inter_sq_sum[expert_idx].add_((route_prob * full_hidden_sq).sum(dim=0))

        full_out = F.linear(full_hidden, down_proj[expert_idx])
        state.w2_output_abs_sum[expert_idx].add_(full_out.float().abs().sum(dim=0))
        final_hidden_states.index_add_(0, token_idx, (route_weight * full_out.float()).to(valid_hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    valid_output = final_hidden_states + (shared_out * shared_gate_value)

    padded_output = torch.zeros_like(hidden_states)
    padded_output[:, :actual_length, :] = valid_output.view(batch_size, sequence_length, hidden_dim)
    return padded_output


@torch.inference_mode()
def compute_maca_novel_metric_artifacts(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    actual_lengths: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    dict[int, LayerMetricBundle],
    dict[int, LayerMetricBundle],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[str, Any],
]:
    print("\n=== Iteration 26 calibration: MaCa router-affinity / normalized Hessian / MicroMix ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    router_affinity_cache: dict[int, LayerMetricBundle] = {}
    hessian_normalized_cache: dict[int, LayerMetricBundle] = {}
    micromix_mean_abs: dict[int, torch.Tensor] = {}
    micromix_thresholds: dict[int, torch.Tensor] = {}
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[maca-metric] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_novel_state(config, device)

        for chunk_idx in range(inps.shape[0]):
            actual_length = int(actual_lengths[chunk_idx].item())
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", "", 1): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", "", 1): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_out = collect_layer_novel_metrics_valid_tokens(mlp_input, moe_tensors, config, state, actual_length)
            outs[chunk_idx] = residual + moe_out

            if should_log_chunk(chunk_idx, int(inps.shape[0])):
                print(
                    f"[maca-metric] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]} | actual_len={actual_length}",
                    flush=True,
                )

        router_affinity_cache[layer_idx] = finalize_router_affinity_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        hessian_normalized_cache[layer_idx] = finalize_hessian_normalized_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)

        counts_f = state.counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        mean_abs = state.w2_output_abs_sum / counts_f
        active_mask = (state.counts > 0).unsqueeze(-1)
        micromix_mean_abs[layer_idx] = torch.where(active_mask, mean_abs, torch.zeros_like(mean_abs)).detach().cpu().to(torch.float32)
        micromix_thresholds[layer_idx] = (down_proj.float().abs().amax(dim=(1, 2)).unsqueeze(-1) / 12.0).detach().cpu().to(torch.float32)

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_name": "iter26_maca_multiscale_artifacts",
        "runtime_seconds": round(time.time() - start_time, 3),
        "calibration_chunks": int(calib_chunks.shape[0]),
        "calibration_max_length": int(calib_chunks.shape[1]),
        "actual_token_total": int(actual_lengths.sum().item()),
        "actual_lengths": [int(length) for length in MACA_LENGTHS],
        "chunks_per_length": int(MACA_CHUNKS_PER_LENGTH),
        "micromix_threshold_formula": "weight_absmax / (2 * 6.0)",
    }
    return router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metadata


def summarize_calibration_artifacts(calibration: CalibrationArtifacts) -> dict[str, Any]:
    return {
        "active_experts_per_layer": {str(layer_idx): int((counts > 0).sum().item()) for layer_idx, counts in calibration.routing_counts.items()},
        "routing_top5_per_layer": {
            str(layer_idx): [
                {"expert_idx": int(expert_idx), "count": int(count)}
                for count, expert_idx in zip(
                    torch.topk(counts, k=min(5, counts.numel())).values.tolist(),
                    torch.topk(counts, k=min(5, counts.numel())).indices.tolist(),
                )
            ]
            for layer_idx, counts in calibration.routing_counts.items()
        },
    }


def build_iteration_plans(
    standard_calibration: CalibrationArtifacts,
    maca_calibration: CalibrationArtifacts,
    maca_router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[Any, dict[str, Any]]]:
    plans: list[tuple[Any, dict[str, Any]]] = []

    maca_joint_w1_masks, maca_joint_w2_masks, maca_joint_meta = build_joint_with_topup_masks(
        maca_calibration,
        maca_calibration.activation_cache,
        config,
        JOINT_TOPUP_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "maca_joint_w1w2_topup",
            "Rebuild the joint three-tier plus 8% within-expert topup plan from MaCa multi-scale calibration, while keeping GPTQ-standard evaluation unchanged.",
            config,
            non_expert_bytes,
            total_expert_elems,
            maca_joint_w1_masks,
            maca_joint_w2_masks,
        ),
        {
            **maca_joint_meta,
            "calibration_variant": "maca_multiscale",
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "source_base": "joint_w1w2_with_topup",
        },
    ))

    maca_mxmoe_w1_masks, maca_mxmoe_w2_masks, maca_mxmoe_meta = build_mxmoe_topup_masks(
        maca_calibration,
        config,
        total_expert_elems,
        MXMOE_TOPUP_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "maca_mxmoe_topup_5pct",
            "Rebuild the MxMoE per-block base and add 5% activation-ranked W1/W2 topups using MaCa multi-scale calibration statistics.",
            config,
            non_expert_bytes,
            total_expert_elems,
            maca_mxmoe_w1_masks,
            maca_mxmoe_w2_masks,
        ),
        {
            **maca_mxmoe_meta,
            "calibration_variant": "maca_multiscale",
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "source_base": "mxmoe_block_plus_channel_topup",
        },
    ))

    maca_ra_w1_masks, maca_ra_w2_masks, maca_ra_meta = build_global_fraction_masks(
        maca_router_affinity_cache,
        config,
        PERCHANNEL_W1_FRACTION,
        PERCHANNEL_W2_FRACTION,
        budget_source="router_affinity_weighted_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "maca_perchannel_ra_20pct",
            "Rebuild the 20% router-affinity per-channel plan from MaCa multi-scale calibration statistics with the standard 4% W1 / 16% W2 split.",
            config,
            non_expert_bytes,
            total_expert_elems,
            maca_ra_w1_masks,
            maca_ra_w2_masks,
        ),
        {
            **maca_ra_meta,
            "calibration_variant": "maca_multiscale",
            "channel_metric_requested": "router_affinity_weighted_qerror",
            "channel_metric_effective": "router_affinity_weighted_qerror",
            "channel_metric_fallback_used": False,
            "source_base": "router_affinity_weighted_w1_4_w2_16",
        },
    ))

    standard_joint_w1_masks, standard_joint_w2_masks, standard_joint_meta = build_joint_with_topup_masks(
        standard_calibration,
        standard_calibration.activation_cache,
        config,
        JOINT_TOPUP_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "standard_joint_w1w2_topup",
            "Rebuild the current best joint three-tier plus 8% within-expert topup plan from the standard 128x2048 GPTQ calibration set.",
            config,
            non_expert_bytes,
            total_expert_elems,
            standard_joint_w1_masks,
            standard_joint_w2_masks,
        ),
        {
            **standard_joint_meta,
            "calibration_variant": "standard_fixed_2048",
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "source_base": "joint_w1w2_with_topup",
        },
    ))
    return plans


def format_result_table(results: dict[str, Any]) -> str:
    lines = [
        "| Config | PPL | Memory (GB) | FP8 frac | W1 frac | W2 frac |",
        "|--------|-----|-------------|----------|---------|---------|",
    ]
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    for name, row in ordered:
        lines.append(
            f"| {name} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} | {float(row['fp8_fraction']):.4f} | {float(row['w1_pair_fraction']):.4f} | {float(row['w2_channel_fraction']):.4f} |"
        )
    return "\n".join(lines)


def build_insight_text(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    standard_row = results.get("standard_joint_w1w2_topup")
    maca_rows = {name: row for name, row in results.items() if name.startswith("maca_")}
    if standard_row is None or not maca_rows:
        return "Iteration 26 needs both the standard control and at least one MaCa row to compare calibration effects cleanly."

    best_maca_name, best_maca_row = min(maca_rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    delta_vs_standard = float(best_maca_row["ppl"]) - float(standard_row["ppl"])
    if delta_vs_standard < 0.0:
        maca_text = f"`{best_maca_name}` is the best MaCa row at {float(best_maca_row['ppl']):.4f}, beating `standard_joint_w1w2_topup` by {abs(delta_vs_standard):.4f} PPL."
    elif delta_vs_standard > 0.0:
        maca_text = f"`{best_maca_name}` is the best MaCa row at {float(best_maca_row['ppl']):.4f}, trailing `standard_joint_w1w2_topup` by {delta_vs_standard:.4f} PPL."
    else:
        maca_text = f"`{best_maca_name}` exactly matches `standard_joint_w1w2_topup` at {float(best_maca_row['ppl']):.4f}."

    historical_row = references.get("joint_w1w2_with_topup")
    if historical_row is None:
        return maca_text
    control_drift = float(standard_row["ppl"]) - float(historical_row["ppl"])
    return f"{maca_text} The rebuilt standard control differs from the historical `joint_w1w2_with_topup` reference by {control_drift:+.4f} PPL."


def render_exploration_section(
    results: dict[str, Any],
    maca_set: VariableLengthCalibrationSet,
    references: dict[str, dict[str, float]],
) -> str:
    table = format_result_table(results)
    metadata = maca_set.metadata
    return (
        f"{SECTION_MARKER}\n"
        f"**Approach**: Replaced the fixed 128x2048 calibration set with a MaCa-style mix of 32x128, 32x512, 32x2048, and 32x4096 WikiText-2 train chunks. All calibration forwards run on a padded 4096-token workspace, but routing counts, activation_kurtosis, MxMoE perturbation deltas, router-affinity scores, and MicroMix statistics only accumulate over each chunk's valid prefix so the GPTQ-standard WikiText-2 evaluation path stays unchanged at seqlen=2048.\n"
        f"**Calibration**: {metadata['samples']} chunks, actual tokens {metadata['actual_token_total']}, padded workspace tokens {metadata['padded_token_total']}, real-token fraction {metadata['real_token_fraction']:.4f}, lengths {metadata['lengths']}.\n"
        f"**Result**:\n{table}\n"
        f"**Insight**: {build_insight_text(results, references)}\n"
        f"**Next**: If mixed lengths still hug the standard control, test MaCa again with sliding-window 4096 calibration so long-context statistics are preserved without padding-heavy waste."
    )


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    print("\n" + "=" * 182, flush=True)
    print("proper_iter26_maca_calibration | multi-scale calibration rebuilds | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 182, flush=True)
    print(
        f"{'Config':<32} {'PPL':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 182, flush=True)
    for name, row in ordered:
        print(
            f"{name:<32} {float(row['ppl']):>10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    requested_plans = resolve_requested_plans(args.plans)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    tokenizer, standard_calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    maca_set = build_maca_calibration_set(tokenizer, args.seed)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    output_payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "standard_calibration": calib_info,
            "maca_calibration": maca_set.metadata,
            "evaluation": eval_info,
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP32 logits and loss",
            "protocol": {
                "reference": "/home/jerry/Documents/fork_new/MC-MoE/eval_ppl_utils.py",
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "experiment": "Iteration 26 MaCa multi-scale calibration rebuilds",
            "success_targets": SUCCESS_TARGETS,
        },
        "results": {},
    }
    if args.output_json.exists():
        try:
            existing = load_json(args.output_json)
            if isinstance(existing, dict):
                merged_metadata = dict(existing.get("metadata", {}))
                merged_metadata.update(output_payload["metadata"])
                output_payload.update(existing)
                output_payload["metadata"] = merged_metadata
                output_payload.setdefault("results", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, output_payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    standard_calibration = run_calibration(store, text_config, standard_calib_chunks, device, dtype)
    maca_calibration = run_maca_calibration(store, text_config, maca_set.chunks, maca_set.actual_lengths, device, dtype)
    maca_router_affinity_cache, _maca_hessian_cache, _maca_micromix_mean_abs, _maca_micromix_thresholds, maca_metric_meta = compute_maca_novel_metric_artifacts(
        store,
        text_config,
        maca_set.chunks,
        maca_set.actual_lengths,
        device,
        dtype,
    )

    output_payload["metadata"]["standard_calibration_summary"] = summarize_calibration_artifacts(standard_calibration)
    output_payload["metadata"]["maca_calibration_summary"] = summarize_calibration_artifacts(maca_calibration)
    output_payload["metadata"]["maca_metric_cache"] = maca_metric_meta
    atomic_json_dump(args.output_json, output_payload)

    plans = build_iteration_plans(
        standard_calibration,
        maca_calibration,
        maca_router_affinity_cache,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )
    if requested_plans is not None:
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        missing = sorted(requested_plans.difference({plan.name for plan, _extras in plans}))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")

    for idx, (plan, extras) in enumerate(plans, start=1):
        print(f"\n=== Eval {idx}/{len(plans)}: {plan.name} ===", flush=True)
        evaluate_and_record_plan(
            output_payload,
            plan,
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

    references = load_reference_rows(args.output_json)
    section = render_exploration_section(output_payload["results"], maca_set, references)
    upsert_exploration_section(args.exploration_md, section)
    output_payload["metadata"]["references"] = references
    output_payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, output_payload)
    print_results_table(output_payload["results"])
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

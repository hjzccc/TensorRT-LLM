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

from baselines_comparison import LayerMetricBundle, resolve_non_expert_bytes, resolve_terminal_keys, topk_mask_from_scores
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    layer_type_at,
    load_gptq_standard_data,
    upsert_exploration_section,
)
from proper_iter01 import build_empty_masks, build_plan_from_masks, build_weighted_masks_from_fractions, load_cache, quantize_linear_weight
from proper_iter05 import build_joint_precision_promotions, build_mxmoe_projection_promotions_fraction
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter20_router_preserve.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter20_router_preserve_metric_cache.pt"
SECTION_MARKER = "## [23] Iteration 20 - Router-Rank Preservation"
METRIC_CACHE_VERSION = 1
MXMOE_BASE_FRACTION = 0.25
GAP_EPS = 1e-4
FRAGILE_QUANTILE = 0.20

SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "output_perturbation_mxmoe_topup_router_affinity_5pct": 6.5758,
    "mxmoe_block_plus_channel_topup": 6.5763,
    "router_affinity_weighted_w1_4_w2_16": 6.5796,
}


@dataclass
class GapAwareState:
    counts: torch.Tensor
    gap_weighted_counts: torch.Tensor
    router_input_sq_sum: torch.Tensor
    router_inter_sq_sum: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 20 router-gap plans.",
    )
    parser.add_argument("--gap-eps", type=float, default=GAP_EPS)
    parser.add_argument("--fragile-quantile", type=float, default=FRAGILE_QUANTILE)
    parser.add_argument("--force-recompute-metrics", action="store_true")
    return parser.parse_args()


def clone_metric_cache(source: dict[int, LayerMetricBundle]) -> dict[int, LayerMetricBundle]:
    return {
        int(layer_idx): LayerMetricBundle(
            routing_counts=bundle.routing_counts.detach().cpu().clone(),
            w1_pair_scores=bundle.w1_pair_scores.detach().cpu().clone(),
            w2_channel_scores=bundle.w2_channel_scores.detach().cpu().clone(),
        )
        for layer_idx, bundle in source.items()
    }


def serialize_metric_cache_bundles(source: dict[int, LayerMetricBundle]) -> dict[int, dict[str, torch.Tensor]]:
    payload: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx, bundle in source.items():
        payload[int(layer_idx)] = {
            "routing_counts": bundle.routing_counts.detach().cpu().clone().to(torch.int64),
            "w1_pair_scores": bundle.w1_pair_scores.detach().cpu().clone().to(torch.float32),
            "w2_channel_scores": bundle.w2_channel_scores.detach().cpu().clone().to(torch.float32),
        }
    return payload


def restore_metric_cache_bundles(raw_cache: Any) -> dict[int, LayerMetricBundle] | None:
    if not isinstance(raw_cache, dict):
        return None
    restored: dict[int, LayerMetricBundle] = {}
    for raw_layer_idx, payload in raw_cache.items():
        if not isinstance(payload, dict):
            return None
        restored[int(raw_layer_idx)] = LayerMetricBundle(
            routing_counts=torch.as_tensor(payload["routing_counts"]).detach().cpu().clone().to(torch.int64),
            w1_pair_scores=torch.as_tensor(payload["w1_pair_scores"]).detach().cpu().clone().to(torch.float32),
            w2_channel_scores=torch.as_tensor(payload["w2_channel_scores"]).detach().cpu().clone().to(torch.float32),
        )
    return restored


def serialize_float_tensor_map(source: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {int(layer_idx): tensor.detach().cpu().clone().to(torch.float32) for layer_idx, tensor in source.items()}


def restore_float_tensor_map(raw_cache: Any) -> dict[int, torch.Tensor] | None:
    if not isinstance(raw_cache, dict):
        return None
    return {int(layer_idx): torch.as_tensor(tensor).detach().cpu().clone().to(torch.float32) for layer_idx, tensor in raw_cache.items()}


def metric_cache_is_compatible(payload: dict[str, Any], model_id: str, gap_eps: float, fragile_quantile: float) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and abs(float(payload.get("gap_eps", -1.0)) - float(gap_eps)) <= 1e-12
        and abs(float(payload.get("fragile_quantile", -1.0)) - float(fragile_quantile)) <= 1e-12
    )


def load_metric_cache(
    path: Path,
    model_id: str,
    gap_eps: float,
    fragile_quantile: float,
) -> tuple[dict[int, LayerMetricBundle] | None, dict[int, torch.Tensor] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id, gap_eps, fragile_quantile):
        return None, None, None
    metric_cache = restore_metric_cache_bundles(payload.get("gap_metric_cache"))
    gap_weighted_counts = restore_float_tensor_map(payload.get("gap_weighted_counts"))
    metadata = payload.get("metadata")
    if metric_cache is None or gap_weighted_counts is None or not isinstance(metadata, dict):
        return None, None, None
    return metric_cache, gap_weighted_counts, dict(metadata)


def save_metric_cache(
    path: Path,
    model_id: str,
    gap_eps: float,
    fragile_quantile: float,
    gap_metric_cache: dict[int, LayerMetricBundle],
    gap_weighted_counts: dict[int, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "gap_eps": float(gap_eps),
            "fragile_quantile": float(fragile_quantile),
            "gap_metric_cache": serialize_metric_cache_bundles(gap_metric_cache),
            "gap_weighted_counts": serialize_float_tensor_map(gap_weighted_counts),
            "metadata": metadata,
        },
        path,
    )


def init_gap_state(config: Any, device: torch.device) -> GapAwareState:
    return GapAwareState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        gap_weighted_counts=torch.zeros(config.num_experts, dtype=torch.float32, device=device),
        router_input_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        router_inter_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
    )


def finalize_gap_metric_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: GapAwareState,
    config: Any,
) -> LayerMetricBundle:
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=gate_up_proj.device)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=gate_up_proj.device)
    for expert_idx in range(config.num_experts):
        if int(state.counts[expert_idx].item()) <= 0:
            continue
        gate_up_diff = gate_up_proj[expert_idx].float() - fp4_gate_up[expert_idx].float()
        gate_diff_sq = gate_up_diff[: config.moe_intermediate_size].square()
        up_diff_sq = gate_up_diff[config.moe_intermediate_size :].square()
        w1_scores[expert_idx] = torch.matmul(gate_diff_sq + up_diff_sq, state.router_input_sq_sum[expert_idx])

        down_diff_sq = (down_proj[expert_idx].float() - fp4_down[expert_idx].float()).square()
        w2_scores[expert_idx] = torch.matmul(down_diff_sq, state.router_inter_sq_sum[expert_idx])

    return LayerMetricBundle(
        routing_counts=state.counts.detach().cpu().clone().to(torch.int64),
        w1_pair_scores=w1_scores.detach().cpu().clone(),
        w2_channel_scores=w2_scores.detach().cpu().clone(),
    )


def collect_layer_gap_metrics(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: GapAwareState,
    gap_eps: float,
    layer_gaps: list[torch.Tensor],
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    topk_count = min(int(config.num_experts_per_tok) + 1, int(router_logits.shape[1]))
    top_values, top_indices = torch.topk(router_logits, k=topk_count, dim=-1)
    selected_experts = top_indices[:, : config.num_experts_per_tok]
    selected_probs = routing_probs.gather(1, selected_experts)
    routing_weights = selected_probs / selected_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    routing_weights = routing_weights.to(hidden_states.dtype)

    kth_logits = top_values[:, config.num_experts_per_tok - 1]
    next_logits = top_values[:, config.num_experts_per_tok] if topk_count > config.num_experts_per_tok else kth_logits
    gaps = torch.clamp(kth_logits - next_logits, min=0.0)
    gap_scale = torch.reciprocal(gaps + float(gap_eps)).to(torch.float32)
    layer_gaps.append(gaps.detach().cpu())

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = moe_tensors["experts.gate_up_proj"]
    down_proj = moe_tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    state.counts.add_(expert_counts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        route_prob = routing_probs[token_idx, expert_idx].unsqueeze(-1).to(torch.float32)
        route_weight = routing_weights[token_idx, route_pos].unsqueeze(-1)
        gap_weight = gap_scale[token_idx].unsqueeze(-1)
        state.gap_weighted_counts[expert_idx].add_(gap_weight.sum())

        current_sq = current_state.float().square()
        state.router_input_sq_sum[expert_idx].add_((route_prob * gap_weight * current_sq).sum(dim=0))

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden_sq = hidden.float().square()
        state.router_inter_sq_sum[expert_idx].add_((route_prob * gap_weight * hidden_sq).sum(dim=0))

        out = F.linear(hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (route_weight * out.float()).to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


@torch.inference_mode()
def compute_gap_aware_metric_artifacts(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    gap_eps: float,
    fragile_quantile: float,
) -> tuple[dict[int, LayerMetricBundle], dict[int, torch.Tensor], dict[str, Any]]:
    print("\n=== Iteration 20 calibration: router-gap preservation ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    gap_metric_cache: dict[int, LayerMetricBundle] = {}
    gap_weighted_counts: dict[int, torch.Tensor] = {}
    raw_gap_chunks: dict[int, list[torch.Tensor]] = {}
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[router-gap] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors
        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_gap_state(config, device)
        raw_gap_chunks[layer_idx] = []

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
            moe_out = collect_layer_gap_metrics(mlp_input, moe_tensors, config, state, gap_eps, raw_gap_chunks[layer_idx])
            outs[chunk_idx] = residual + moe_out
            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[router-gap] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        gap_metric_cache[layer_idx] = finalize_gap_metric_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        gap_weighted_counts[layer_idx] = state.gap_weighted_counts.detach().cpu().clone().to(torch.float32)

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})

    layer_gap_summary: dict[str, Any] = {}
    total_gap_tokens = 0
    total_fragile_tokens = 0
    for layer_idx, chunks in raw_gap_chunks.items():
        gaps = torch.cat(chunks, dim=0).to(torch.float32) if chunks else torch.zeros(0, dtype=torch.float32)
        fragile_threshold = float(torch.quantile(gaps, fragile_quantile).item()) if int(gaps.numel()) > 0 else 0.0
        fragile_mask = gaps <= fragile_threshold if int(gaps.numel()) > 0 else torch.zeros(0, dtype=torch.bool)
        raw_counts = gap_metric_cache[layer_idx].routing_counts.to(torch.float32)
        weighted_counts = gap_weighted_counts[layer_idx]
        top_weighted_values, top_weighted_indices = torch.topk(weighted_counts, k=min(5, int(weighted_counts.numel())))
        layer_gap_summary[str(layer_idx)] = {
            "tokens": int(gaps.numel()),
            "fragile_threshold": round(fragile_threshold, 8),
            "mean_gap": round(float(gaps.mean().item()) if int(gaps.numel()) > 0 else 0.0, 8),
            "median_gap": round(float(gaps.median().item()) if int(gaps.numel()) > 0 else 0.0, 8),
            "fragile_tokens": int(fragile_mask.sum().item()),
            "fragile_fraction": round(float(fragile_mask.to(torch.float32).mean().item()) if int(gaps.numel()) > 0 else 0.0, 6),
            "top_gap_weighted_experts": [
                {
                    "expert_idx": int(expert_idx),
                    "gap_weighted_count": round(float(weight), 4),
                    "raw_count": int(raw_counts[int(expert_idx)].item()),
                }
                for weight, expert_idx in zip(top_weighted_values.tolist(), top_weighted_indices.tolist())
            ],
        }
        total_gap_tokens += int(gaps.numel())
        total_fragile_tokens += int(fragile_mask.sum().item())

    metadata = {
        "metric_name": "router_gap_affinity_weighted_qerror",
        "gap_eps": float(gap_eps),
        "fragile_quantile": float(fragile_quantile),
        "total_gap_tokens": int(total_gap_tokens),
        "total_fragile_tokens": int(total_fragile_tokens),
        "total_fragile_fraction": round(float(total_fragile_tokens) / float(max(total_gap_tokens, 1)), 6),
        "layer_gap_summary": layer_gap_summary,
        "runtime_seconds": round(time.time() - start_time, 3),
    }
    return gap_metric_cache, gap_weighted_counts, metadata


def build_projection_topup_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_proj: dict[int, torch.Tensor],
    w2_proj: dict[int, torch.Tensor],
    w1_topup_fraction: float,
    w2_topup_fraction: float,
    budget_source: str,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    topup_w1 = int(round(w1_topup_fraction * config.moe_intermediate_size))
    topup_w2 = int(round(w2_topup_fraction * config.hidden_size))
    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if int(bundle.routing_counts[expert_idx].item()) <= 0:
                continue
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            elif topup_w1 > 0:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            elif topup_w2 > 0:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": budget_source,
        "w1_topup_fraction": float(w1_topup_fraction),
        "w2_topup_fraction": float(w2_topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def merge_projection_promotions(
    config: Any,
    first_w1: dict[int, torch.Tensor],
    first_w2: dict[int, torch.Tensor],
    second_w1: dict[int, torch.Tensor],
    second_w2: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    merged_w1 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    merged_w2 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    for layer_idx in range(config.num_hidden_layers):
        merged_w1[layer_idx] = torch.logical_or(first_w1[layer_idx], second_w1[layer_idx])
        merged_w2[layer_idx] = torch.logical_or(first_w2[layer_idx], second_w2[layer_idx])
    return merged_w1, merged_w2, {
        "merged_base_w1_projections": int(sum(int(mask.sum().item()) for mask in merged_w1.values())),
        "merged_base_w2_projections": int(sum(int(mask.sum().item()) for mask in merged_w2.values())),
    }


def build_gap_joint_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    gap_weighted_counts: dict[int, torch.Tensor],
    config: Any,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    active_scores: list[torch.Tensor] = []
    weighted_combined: dict[int, torch.Tensor] = {}
    for layer_idx in range(config.num_hidden_layers):
        combined = (calibration.mxmoe_w1_deltas[layer_idx].to(torch.float32) + calibration.mxmoe_w2_deltas[layer_idx].to(torch.float32))
        weighted = combined * gap_weighted_counts[layer_idx].to(torch.float32)
        weighted_combined[layer_idx] = weighted
        active_mask = calibration.routing_counts[layer_idx] > 0
        if bool(active_mask.any()):
            active_scores.append(weighted[active_mask])

    if active_scores:
        flat_active = torch.cat(active_scores, dim=0)
        p50 = float(torch.quantile(flat_active, 0.50).item())
        p75 = float(torch.quantile(flat_active, 0.75).item())
    else:
        p50 = 0.0
        p75 = 0.0

    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    medium_w1_pairs = int(round(topup_fraction * config.moe_intermediate_size))
    medium_w2_channels = int(round(topup_fraction * config.hidden_size))
    high_count = 0
    medium_count = 0
    low_count = 0
    for layer_idx in range(config.num_hidden_layers):
        scores = weighted_combined[layer_idx]
        bundle = metric_cache[layer_idx]
        active_mask = calibration.routing_counts[layer_idx] > 0
        for expert_idx in range(config.num_experts):
            if not bool(active_mask[expert_idx]):
                continue
            score = float(scores[expert_idx].item())
            if score >= p75:
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
                high_count += 1
            elif score >= p50:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], medium_w1_pairs)
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], medium_w2_channels)
                medium_count += 1
            else:
                low_count += 1
    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "joint_gap_weighted_percentile_buckets_with_gap_topup",
        "joint_gap_threshold_p50": float(p50),
        "joint_gap_threshold_p75": float(p75),
        "joint_gap_high_experts": int(high_count),
        "joint_gap_medium_experts": int(medium_count),
        "joint_gap_low_experts": int(low_count),
        "medium_tier_w1_fraction": float(topup_fraction),
        "medium_tier_w2_fraction": float(topup_fraction),
        "medium_tier_w1_pairs": int(medium_w1_pairs),
        "medium_tier_w2_channels": int(medium_w2_channels),
    }


def build_gap_union_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    total_expert_elems: int,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        MXMOE_BASE_FRACTION,
    )
    joint_w1_proj, joint_w2_proj, joint_meta = build_joint_precision_promotions(calibration, config)
    merged_w1_proj, merged_w2_proj, merged_meta = merge_projection_promotions(
        config,
        mxmoe_w1_proj,
        mxmoe_w2_proj,
        joint_w1_proj,
        joint_w2_proj,
    )
    w1_masks, w2_masks, topup_meta = build_projection_topup_masks(
        metric_cache,
        config,
        merged_w1_proj,
        merged_w2_proj,
        topup_fraction,
        topup_fraction,
        budget_source="mxmoe_plus_joint_base_plus_router_gap_topup",
    )
    return w1_masks, w2_masks, {
        **joint_meta,
        **merged_meta,
        **topup_meta,
        "mxmoe_base_fraction": float(MXMOE_BASE_FRACTION),
        "topup_fraction": float(topup_fraction),
    }


def build_gap_mxmoe_topup_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    total_expert_elems: int,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        MXMOE_BASE_FRACTION,
    )
    w1_masks, w2_masks, extras = build_projection_topup_masks(
        metric_cache,
        config,
        w1_proj,
        w2_proj,
        topup_fraction,
        topup_fraction,
        budget_source="mxmoe_block_plus_router_gap_topup",
    )
    return w1_masks, w2_masks, {
        **extras,
        "mxmoe_base_fraction": float(MXMOE_BASE_FRACTION),
        "topup_fraction": float(topup_fraction),
    }


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    gap_metric_cache: dict[int, LayerMetricBundle],
    gap_weighted_counts: dict[int, torch.Tensor],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    channel_metric_meta = {
        "channel_metric_requested": "router_gap_affinity_weighted_qerror",
        "channel_metric_effective": "router_gap_affinity_weighted_qerror",
        "channel_metric_fallback_used": False,
        "routing_count_metric": "inverse_router_gap_weighted_topk_count",
    }

    for name, topup_fraction in (
        ("routergap_joint_topup_3pct", 0.03),
        ("routergap_joint_topup_5pct", 0.05),
        ("routergap_joint_topup_8pct", 0.08),
    ):
        w1_masks, w2_masks, extras = build_gap_joint_topup_masks(
            calibration,
            gap_metric_cache,
            gap_weighted_counts,
            config,
            topup_fraction,
        )
        plans.append((
            build_plan_from_masks(
                name,
                f"Gap-aware joint three-tier plan: rebucket experts with gap-weighted routing counts and rank medium-tier W1/W2 channels by inverse-gap router-affinity at {int(round(topup_fraction * 100))}% topup.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {**channel_metric_meta, **extras, "source_base": "joint_w1w2_with_topup"},
        ))

    w1_masks, w2_masks, extras = build_gap_union_topup_masks(
        calibration,
        gap_metric_cache,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "routergap_union_topup_5pct",
            "Union the MxMoE block base and joint three-tier base, then replace router-affinity topups with inverse-gap router-affinity on every remaining FP4 W1/W2 path.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**channel_metric_meta, **extras, "source_base": "output_perturbation_mxmoe_topup_router_affinity_5pct"},
    ))

    w1_masks, w2_masks, extras = build_gap_mxmoe_topup_masks(
        calibration,
        gap_metric_cache,
        config,
        total_expert_elems,
        topup_fraction=0.08,
    )
    plans.append((
        build_plan_from_masks(
            "routergap_mxmoe_topup_8pct",
            "Keep the MxMoE block projection base and replace activation-kurtosis topups with inverse-gap router-affinity ranking at 8% on non-promoted W1/W2 paths.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**channel_metric_meta, **extras, "source_base": "mxmoe_block_plus_channel_topup"},
    ))

    w1_masks, w2_masks, extras = build_weighted_masks_from_fractions(
        gap_metric_cache,
        config,
        gap_weighted_counts,
        gap_weighted_counts,
        0.04,
        0.16,
    )
    plans.append((
        build_plan_from_masks(
            "routergap_perchannel_20pct",
            "Pure per-channel plan with a 1:4 W1:W2 split: allocate expert budgets by inverse-gap routing counts and rank channels by inverse-gap router-affinity.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **channel_metric_meta,
            **extras,
            "budget_source": "inverse_gap_weighted_expert_budget_plus_router_gap_affinity",
            "w1_fraction_target": 0.04,
            "w2_fraction_target": 0.16,
            "source_base": "router_affinity_weighted_w1_4_w2_16",
        },
    ))

    return plans


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    target_best = min(SUCCESS_TARGETS.values())
    print("\n" + "=" * 188, flush=True)
    print("proper_iter20_router_preserve | inverse-gap router-rank preservation | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 188, flush=True)
    print(
        f"{'Config':<34} {'PPL':>10} {'dTarget':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 188, flush=True)
    for name, row in ordered:
        print(
            f"{name:<34} {float(row['ppl']):>10.4f} {float(row['ppl']) - target_best:>+10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def format_markdown_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), item[0]))
    lines = [
        "| Config | PPL | Delta vs best prior | Memory GB | FP8 frac |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    best_prior = min((float(row["ppl"]) for row in references.values()), default=min(SUCCESS_TARGETS.values()))
    for name, row in ordered:
        lines.append(
            f"| `{name}` | {float(row['ppl']):.4f} | {float(row['ppl']) - best_prior:+.4f} | {float(row['memory_gb']):.3f} | {float(row['fp8_fraction']):.4f} |"
        )
    return "\n".join(lines)


def render_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("render_exploration_section requires non-empty results")
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    winner_name, winner = ordered[0]
    gap_meta = payload["metadata"]["router_gap_calibration"]
    return (
        f"{SECTION_MARKER}\n"
        f"**Approach**: Reused the standard 128x2048 GPTQ calibration chunks, collected full-precision router logits per layer/token, computed the top-k vs (k+1) router gap, marked the bottom {int(round(100 * float(gap_meta['fragile_quantile'])))}% as fragile, and reweighted routed tokens by `1 / (gap + {float(gap_meta['gap_eps']):.0e})` when accumulating expert counts and router-affinity channel scores. The six evals keep the existing strong family shapes (joint / union / MxMoE / pure per-channel) but replace their routing-side scoring with boundary-aware versions.\n"
        f"**Gap stats**: {int(gap_meta['total_gap_tokens'])} routed token decisions across calibration, fragile fraction {float(gap_meta['total_fragile_fraction']):.4f}, metric runtime {float(gap_meta['runtime_seconds']):.1f}s.\n"
        f"**Result**:\n{format_markdown_table(results, references)}\n"
        f"**Winner**: `{winner_name}` at PPL {float(winner['ppl']):.4f}, memory {float(winner['memory_gb']):.3f} GB, FP8 fraction {float(winner['fp8_fraction']):.4f}.\n"
        f"**Insight**: Inverse-gap weighting shifts protection toward experts and channels that sit near the router decision boundary, directly targeting top-k flips rather than only per-channel quantization magnitude.\n"
        f"**Next**: If the best router-gap plan is competitive, combine it with the strongest residual or OBS remainder path so fragile-route protection and quantization-error correction act together instead of separately."
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
    _tokenizer, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    references = load_reference_rows(args.output_json)
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP32 logits and loss",
            "protocol": {
                "reference": "/home/jerry/Documents/fork_new/MC-MoE/eval_ppl_utils.py",
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "metric_cache_path": str(args.metric_cache_path),
            "references": references,
            "success_targets": SUCCESS_TARGETS,
            "experiment": "Iteration 20 router-rank preservation via inverse-gap weighted routing counts and router-affinity channel scores",
        },
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
                payload.setdefault("results", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise FileNotFoundError(f"Calibration cache missing or incompatible: {args.cache_path}")

    gap_metric_cache, gap_weighted_counts, gap_meta = (None, None, None)
    if not args.force_recompute_metrics:
        gap_metric_cache, gap_weighted_counts, gap_meta = load_metric_cache(
            args.metric_cache_path,
            args.model_id,
            args.gap_eps,
            args.fragile_quantile,
        )
        if gap_metric_cache is not None and gap_weighted_counts is not None and gap_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if gap_metric_cache is None or gap_weighted_counts is None or gap_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        gap_metric_cache, gap_weighted_counts, gap_meta = compute_gap_aware_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
            args.gap_eps,
            args.fragile_quantile,
        )
        save_metric_cache(
            args.metric_cache_path,
            args.model_id,
            args.gap_eps,
            args.fragile_quantile,
            gap_metric_cache,
            gap_weighted_counts,
            gap_meta,
        )
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["router_gap_calibration"] = gap_meta
    payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(
        calibration,
        gap_metric_cache,
        gap_weighted_counts,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )
    if requested_plans is not None:
        requested_set = set(requested_plans)
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_set]
        missing = sorted(requested_set - {plan.name for plan, _extras in plans})
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
    print(f"[plans] evaluating {len(plans)} router-gap configs", flush=True)

    for plan, extras in plans:
        evaluate_and_record_plan(
            payload,
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

    payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)
    section = render_exploration_section(payload, references)
    upsert_exploration_section(args.exploration_md, section)
    print_results_table(payload["results"])
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

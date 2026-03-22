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
import torch.nn as nn
import torch.nn.functional as F

from baselines_comparison import (
    LayerMetricBundle,
    quantize_down_proj_with_mask,
    quantize_gate_up_with_pair_mask,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
)
from proper_eval import (
    CALIBRATION_SAMPLES,
    LOGITS_CHUNK_TOKENS,
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
from proper_iter01 import (
    build_empty_masks,
    build_mxmoe_topup_masks,
    build_plan_from_masks,
    load_cache,
    total_channel_fraction,
    total_pair_fraction,
)
from proper_iter05 import build_joint_precision_promotions
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter10_novel_perchannel import (
    build_global_fraction_masks,
    compute_novel_metric_artifacts,
    load_metric_cache as load_router_metric_cache,
    save_metric_cache as save_router_metric_cache,
)
from proper_iter11_push_router_affinity import split_total_budget
from proper_iter14 import build_union_base_router_affinity_topup_masks
from proper_iter21_hybrid_residual import build_residual_masks_from_base_masks
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter22_wush_transform.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter22_wush_transform_metric_cache.pt"
DEFAULT_ROUTER_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
SECTION_MARKER = "## [25] Iteration 22 - WUSH Diagonal Transform"
METRIC_CACHE_VERSION = 1
DIAG_EPS = 1e-4
JOINT_TOPUP_5PCT = 0.05
MXMOE_TOPUP_5PCT = 0.05
UNION_TOPUP_5PCT = 0.05
ROUTER_AFFINITY_TOTAL_FRACTION = 0.20
JOINT_RESIDUAL_FRACTION = 0.02


@dataclass
class WushMomentState:
    counts: torch.Tensor
    input_sq_sum: torch.Tensor
    inter_sq_sum: torch.Tensor


@dataclass(frozen=True)
class WushDiagArtifacts:
    routing_counts: dict[int, torch.Tensor]
    w1_input_scales: dict[int, torch.Tensor]
    w2_input_scales: dict[int, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--router-metric-cache-path", type=Path, default=DEFAULT_ROUTER_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 22 WUSH-diagonal configs.",
    )
    parser.add_argument("--diag-eps", type=float, default=DIAG_EPS)
    parser.add_argument("--force-recompute-metrics", action="store_true")
    parser.add_argument("--force-recompute-router-metrics", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def clone_tensor_map(source: dict[int, torch.Tensor], dtype: torch.dtype) -> dict[int, torch.Tensor]:
    return {
        int(layer_idx): tensor.detach().cpu().clone().to(dtype)
        for layer_idx, tensor in source.items()
    }


def metric_cache_is_compatible(payload: dict[str, Any], model_id: str, diag_eps: float) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
        and abs(float(payload.get("diag_eps", -1.0)) - float(diag_eps)) <= 1e-12
    )


def load_metric_cache(path: Path, model_id: str, diag_eps: float) -> tuple[WushDiagArtifacts | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id, diag_eps):
        return None, None
    routing_counts_raw = payload.get("routing_counts")
    w1_raw = payload.get("w1_input_scales")
    w2_raw = payload.get("w2_input_scales")
    metadata = payload.get("metadata")
    if not isinstance(routing_counts_raw, dict) or not isinstance(w1_raw, dict) or not isinstance(w2_raw, dict):
        return None, None
    artifacts = WushDiagArtifacts(
        routing_counts=clone_tensor_map(routing_counts_raw, torch.int64),
        w1_input_scales=clone_tensor_map(w1_raw, torch.float32),
        w2_input_scales=clone_tensor_map(w2_raw, torch.float32),
    )
    return artifacts, dict(metadata) if isinstance(metadata, dict) else {}


def save_metric_cache(
    path: Path,
    model_id: str,
    diag_eps: float,
    artifacts: WushDiagArtifacts,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "calibration_samples": CALIBRATION_SAMPLES,
            "diag_eps": float(diag_eps),
            "routing_counts": clone_tensor_map(artifacts.routing_counts, torch.int64),
            "w1_input_scales": clone_tensor_map(artifacts.w1_input_scales, torch.float32),
            "w2_input_scales": clone_tensor_map(artifacts.w2_input_scales, torch.float32),
            "metadata": metadata,
        },
        path,
    )


def init_wush_state(config: Any, device: torch.device) -> WushMomentState:
    return WushMomentState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        input_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        inter_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
    )


def should_log_chunk(chunk_idx: int, total_chunks: int) -> bool:
    return chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == total_chunks


def collect_layer_wush_moments(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: WushMomentState,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    gate_up_proj = moe_tensors["experts.gate_up_proj"]
    down_proj = moe_tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    state.counts.add_(expert_counts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        state.input_sq_sum[expert_idx].add_(current_state.float().square().sum(dim=0))

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        state.inter_sq_sum[expert_idx].add_(hidden.float().square().sum(dim=0))

        out = F.linear(hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (routing_weights[token_idx, route_pos].unsqueeze(-1) * out.float()).to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def finalize_wush_layer(state: WushMomentState, diag_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    counts_f = state.counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    active_mask = (state.counts > 0).unsqueeze(-1)
    w1_scales = torch.sqrt(state.input_sq_sum / counts_f)
    w2_scales = torch.sqrt(state.inter_sq_sum / counts_f)
    w1_scales = torch.where(active_mask, torch.clamp(w1_scales, min=diag_eps), torch.ones_like(w1_scales))
    w2_scales = torch.where(active_mask, torch.clamp(w2_scales, min=diag_eps), torch.ones_like(w2_scales))
    return w1_scales.detach().cpu().to(torch.float32), w2_scales.detach().cpu().to(torch.float32)


def summarize_scale_layer(scales: torch.Tensor, counts: torch.Tensor) -> dict[str, float | int]:
    active_mask = counts > 0
    if not bool(active_mask.any()):
        return {
            "active_experts": 0,
            "min": 1.0,
            "median": 1.0,
            "mean": 1.0,
            "max": 1.0,
        }
    values = scales[active_mask].reshape(-1).to(torch.float32)
    return {
        "active_experts": int(active_mask.sum().item()),
        "min": round(float(values.min().item()), 8),
        "median": round(float(values.median().item()), 8),
        "mean": round(float(values.mean().item()), 8),
        "max": round(float(values.max().item()), 8),
    }


@torch.inference_mode()
def compute_wushdiag_artifacts(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    diag_eps: float,
) -> tuple[WushDiagArtifacts, dict[str, Any]]:
    print("\n=== Iteration 22 calibration: WUSH-diagonal second moments ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    w1_input_scales: dict[int, torch.Tensor] = {}
    w2_input_scales: dict[int, torch.Tensor] = {}
    layer_scale_summary: dict[str, Any] = {}
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[wushdiag] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors
        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        state = init_wush_state(config, device)

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
            moe_out = collect_layer_wush_moments(mlp_input, moe_tensors, config, state)
            outs[chunk_idx] = residual + moe_out
            if should_log_chunk(chunk_idx, inps.shape[0]):
                print(f"[wushdiag] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        layer_routing_counts = state.counts.detach().cpu().clone().to(torch.int64)
        layer_w1_scales, layer_w2_scales = finalize_wush_layer(state, diag_eps)
        routing_counts[layer_idx] = layer_routing_counts
        w1_input_scales[layer_idx] = layer_w1_scales
        w2_input_scales[layer_idx] = layer_w2_scales
        layer_scale_summary[str(layer_idx)] = {
            "w1_input_scales": summarize_scale_layer(layer_w1_scales, layer_routing_counts),
            "w2_input_scales": summarize_scale_layer(layer_w2_scales, layer_routing_counts),
        }

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_name": "wushdiag_second_moment_scales",
        "diag_eps": float(diag_eps),
        "calibration_chunks": int(calib_chunks.shape[0]),
        "formula": {
            "sigma": "sqrt(E[x^2])",
            "D": "diag(1 / (sigma + eps))",
            "weight_transform": "W_tilde = W @ D^{-1}",
            "activation_transform": "x_tilde = x @ D",
        },
        "layer_scale_summary": layer_scale_summary,
        "runtime_seconds": round(time.time() - start_time, 3),
    }
    return WushDiagArtifacts(
        routing_counts=routing_counts,
        w1_input_scales=w1_input_scales,
        w2_input_scales=w2_input_scales,
    ), metadata


def projection_promotions_to_masks(
    config: Any,
    w1_proj: dict[int, torch.Tensor],
    w2_proj: dict[int, torch.Tensor],
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
    return w1_pair_masks, w2_channel_masks


def merge_mask_sets(
    base_w1: dict[int, dict[int, torch.Tensor]],
    base_w2: dict[int, dict[int, torch.Tensor]],
    extra_w1: dict[int, dict[int, torch.Tensor]],
    extra_w2: dict[int, dict[int, torch.Tensor]],
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    merged_w1: dict[int, dict[int, torch.Tensor]] = {}
    merged_w2: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx in base_w1:
        merged_w1[layer_idx] = {}
        merged_w2[layer_idx] = {}
        for expert_idx in base_w1[layer_idx]:
            merged_w1[layer_idx][expert_idx] = torch.logical_or(base_w1[layer_idx][expert_idx], extra_w1[layer_idx][expert_idx])
            merged_w2[layer_idx][expert_idx] = torch.logical_or(base_w2[layer_idx][expert_idx], extra_w2[layer_idx][expert_idx])
    return merged_w1, merged_w2


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []

    joint_w1_proj, joint_w2_proj, joint_base_meta = build_joint_precision_promotions(calibration, config)
    joint_base_w1_masks, joint_base_w2_masks = projection_promotions_to_masks(config, joint_w1_proj, joint_w2_proj)
    plans.append((
        build_plan_from_masks(
            "wushdiag_joint_base",
            "Joint W1/W2 percentile bucket base only: high-tier experts keep full W1+W2 in FP8, medium-tier experts keep full W2 in FP8, and no within-expert topups are added.",
            config,
            non_expert_bytes,
            total_expert_elems,
            joint_base_w1_masks,
            joint_base_w2_masks,
        ),
        {
            **joint_base_meta,
            "base_assignment": "joint_precision_base",
            "channel_metric_requested": "none",
            "channel_metric_effective": "none",
            "channel_metric_fallback_used": False,
        },
    ))

    joint_topup_w1_masks, joint_topup_w2_masks, joint_topup_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_TOPUP_5PCT,
    )
    plans.append((
        build_plan_from_masks(
            "wushdiag_joint_topup_5pct",
            "Start from the joint W1/W2 base and add 5% activation-kurtosis topups inside the medium tier before WUSH-diagonal quantization.",
            config,
            non_expert_bytes,
            total_expert_elems,
            joint_topup_w1_masks,
            joint_topup_w2_masks,
        ),
        {
            **joint_topup_meta,
            "base_assignment": "joint_precision_base_plus_activation_topup",
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
        },
    ))

    mxmoe_w1_masks, mxmoe_w2_masks, mxmoe_meta = build_mxmoe_topup_masks(
        calibration,
        config,
        total_expert_elems,
        topup_fraction=MXMOE_TOPUP_5PCT,
    )
    plans.append((
        build_plan_from_masks(
            "wushdiag_mxmoe_topup_5pct",
            "Start from the MxMoE per-block base and add 5% activation-kurtosis within-expert topups before WUSH-diagonal quantization.",
            config,
            non_expert_bytes,
            total_expert_elems,
            mxmoe_w1_masks,
            mxmoe_w2_masks,
        ),
        {
            **mxmoe_meta,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
        },
    ))

    union_w1_masks, union_w2_masks, union_meta = build_union_base_router_affinity_topup_masks(
        calibration,
        router_affinity_cache,
        config,
        total_expert_elems,
        topup_fraction=UNION_TOPUP_5PCT,
    )
    plans.append((
        build_plan_from_masks(
            "wushdiag_union_topup_5pct",
            "Union the MxMoE and joint bases, then apply the existing 5% router-affinity topup before WUSH-diagonal quantization.",
            config,
            non_expert_bytes,
            total_expert_elems,
            union_w1_masks,
            union_w2_masks,
        ),
        {
            **union_meta,
            "channel_metric_requested": "router_affinity_weighted_qerror",
            "channel_metric_effective": "router_affinity_weighted_qerror",
            "channel_metric_fallback_used": False,
        },
    ))

    ra_w1_fraction, ra_w2_fraction = split_total_budget(ROUTER_AFFINITY_TOTAL_FRACTION)
    ra_w1_masks, ra_w2_masks, ra_meta = build_global_fraction_masks(
        router_affinity_cache,
        config,
        ra_w1_fraction,
        ra_w2_fraction,
        budget_source="router_affinity_weighted_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "wushdiag_perchannel_ra_20pct",
            "Pure router-affinity per-channel assignment at a 20% total budget with the usual 1:4 W1:W2 split, evaluated under WUSH-diagonal preconditioning.",
            config,
            non_expert_bytes,
            total_expert_elems,
            ra_w1_masks,
            ra_w2_masks,
        ),
        {
            **ra_meta,
            "channel_metric_requested": "router_affinity_weighted_qerror",
            "channel_metric_effective": "router_affinity_weighted_qerror",
            "channel_metric_fallback_used": False,
            "total_budget_fraction": float(ROUTER_AFFINITY_TOTAL_FRACTION),
            "per_projection_split": "1:4",
        },
    ))

    strong_joint_w1_masks, strong_joint_w2_masks, strong_joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    residual_w1_masks, residual_w2_masks, residual_meta = build_residual_masks_from_base_masks(
        router_affinity_cache,
        strong_joint_w1_masks,
        strong_joint_w2_masks,
        config,
        residual_fraction=JOINT_RESIDUAL_FRACTION,
        budget_source="joint_w1w2_with_topup_plus_residual",
    )
    joint_residual_w1_masks, joint_residual_w2_masks = merge_mask_sets(
        strong_joint_w1_masks,
        strong_joint_w2_masks,
        residual_w1_masks,
        residual_w2_masks,
    )
    plans.append((
        build_plan_from_masks(
            "wushdiag_joint_residual_2pct",
            "Start from the current `joint_w1w2_with_topup` base and add a 2% router-affinity residual branch on the remaining FP4 channels before WUSH-diagonal quantization.",
            config,
            non_expert_bytes,
            total_expert_elems,
            joint_residual_w1_masks,
            joint_residual_w2_masks,
        ),
        {
            **strong_joint_meta,
            **residual_meta,
            "base_assignment": "joint_w1w2_with_topup",
            "source_base": "joint_w1w2_with_topup",
            "channel_metric_requested": "activation_kurtosis_plus_router_affinity_residual",
            "channel_metric_effective": "activation_kurtosis_plus_router_affinity_residual",
            "channel_metric_fallback_used": False,
            "residual_metric_requested": "router_affinity_weighted_qerror",
        },
    ))

    return plans


def precondition_weight_columns(weight: torch.Tensor, input_scales: torch.Tensor) -> torch.Tensor:
    return weight * input_scales.to(device=weight.device, dtype=weight.dtype).view(1, -1)


def prepare_layer_tensors_for_wushdiag_plan(
    plan: EvalPlan,
    layer_idx: int,
    tensors: dict[str, torch.Tensor],
    artifacts: WushDiagArtifacts,
    config: Any,
) -> None:
    if plan.w1_pair_masks is None or plan.w2_channel_masks is None:
        raise ValueError(f"Plan {plan.name} requires per-channel masks for WUSH-diagonal evaluation")

    gate_up_proj = tensors["mlp.experts.gate_up_proj"]
    down_proj = tensors["mlp.experts.down_proj"]
    layer_w1_scales = artifacts.w1_input_scales[layer_idx]
    layer_w2_scales = artifacts.w2_input_scales[layer_idx]
    prepared_gate_up: list[torch.Tensor] = []
    prepared_down: list[torch.Tensor] = []

    for expert_idx in range(config.num_experts):
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]
        gate_up_mask = plan.w1_pair_masks[layer_idx][expert_idx].to(device=gate_up_weight.device, dtype=torch.bool)
        down_mask = plan.w2_channel_masks[layer_idx][expert_idx].to(device=down_weight.device, dtype=torch.bool)
        transformed_gate_up = precondition_weight_columns(gate_up_weight, layer_w1_scales[expert_idx])
        transformed_down = precondition_weight_columns(down_weight, layer_w2_scales[expert_idx])
        prepared_gate_up.append(quantize_gate_up_with_pair_mask(transformed_gate_up, gate_up_mask))
        prepared_down.append(quantize_down_proj_with_mask(transformed_down, down_mask))

    tensors["mlp.experts.gate_up_proj"] = torch.stack(prepared_gate_up, dim=0)
    tensors["mlp.experts.down_proj"] = torch.stack(prepared_down, dim=0)
    del gate_up_proj, down_proj, prepared_gate_up, prepared_down
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def moe_forward_wushdiag(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    layer_idx: int,
    artifacts: WushDiagArtifacts,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    layer_w1_scales = artifacts.w1_input_scales[layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
    layer_w2_scales = artifacts.w2_input_scales[layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        current_state_tilde = current_state * torch.reciprocal(layer_w1_scales[expert_idx]).view(1, -1)
        gate_up = F.linear(current_state_tilde, tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.silu(gate) * up
        current_hidden_tilde = current_hidden * torch.reciprocal(layer_w2_scales[expert_idx]).view(1, -1)
        current_out = F.linear(current_hidden_tilde, tensors["experts.down_proj"][expert_idx])
        final_hidden_states.index_add_(0, token_idx, (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_out).to(hidden_states.dtype))

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


@torch.inference_mode()
def evaluate_plan_wushdiag(
    plan: EvalPlan,
    artifacts: WushDiagArtifacts,
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
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

    root_tensors = store.load_tensors([norm_key, lm_head_key])
    final_norm = move_tensor(root_tensors[norm_key], device, dtype)
    lm_head = move_tensor(root_tensors[lm_head_key], device, dtype)
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
        prepare_layer_tensors_for_wushdiag_plan(plan, layer_idx, tensors, artifacts, config)

        for chunk_idx in range(nsamples):
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
            moe_out = moe_forward_wushdiag(mlp_input, moe_tensors, config, layer_idx, artifacts)
            outs[chunk_idx] = residual + moe_out
            if should_log_chunk(chunk_idx, nsamples):
                print(f"[{plan.name}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{nsamples}", flush=True)

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids_device = test_ids.to(device=device, non_blocking=True)
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
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
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels[:, start:end].reshape(-1))
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
    payload: dict[str, Any],
    plan: EvalPlan,
    extras: dict[str, Any],
    artifacts: WushDiagArtifacts,
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
    if plan.name in result_rows:
        print(f"[skip] {plan.name} already present", flush=True)
        return result_rows[plan.name]

    print(f"\n=== Eval: {plan.name} ===", flush=True)
    start_time = time.time()
    ppl, nll, nsamples = evaluate_plan_wushdiag(plan, artifacts, test_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - start_time
    row = {
        "description": plan.description,
        "mode": "wushdiag_per_channel",
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


def load_reference_rows(output_json: Path) -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in sorted(RESULTS_DIR.glob("proper*.json")):
        if path.resolve() == output_json.resolve() or not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        for name, row in payload.get("results", {}).items():
            if isinstance(row, dict) and "ppl" in row and "memory_gb" in row:
                references[str(name)] = {
                    "ppl": float(row["ppl"]),
                    "memory_gb": float(row["memory_gb"]),
                }
    return references


def format_markdown_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    lines = ["| Config | PPL | Delta vs ref | Memory (GB) |", "|--------|-----|--------------|-------------|"]
    best_reference = min((row["ppl"] for row in references.values()), default=None)
    for name, row in sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0])):
        delta = "-" if best_reference is None else f"{float(row['ppl']) - best_reference:+.4f}"
        lines.append(f"| {name} | {float(row['ppl']):.4f} | {delta} | {float(row['memory_gb']):.3f} |")
    return "\n".join(lines)


def render_exploration_section(results: dict[str, Any], references: dict[str, dict[str, float]], eval_info: dict[str, Any]) -> str:
    table = format_markdown_table(results, references)
    best_name = min(results, key=lambda key: float(results[key]["ppl"])) if results else "-"
    return (
        f"{SECTION_MARKER}\n"
        f"**Eval**: Full WikiText-2 test ({eval_info['total_tokens']} tokens, {eval_info['nsamples']} chunks of {SEQLEN}) using the standard GPTQ-style full-test path with WUSH-diagonal MoE transforms.\n"
        f"**Approach**: For each expert and projection, cache sigma = sqrt(E[x^2]), transform weights as W_tilde = W @ D^-1, quantize W_tilde under the existing mask families, and apply x_tilde = x @ D at inference.\n"
        f"**Result**:\n{table}\n"
        f"**Next**: Launch `{best_name}` first once the GPU is free, then compare it against the active SERQ residual branch."
    )


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_reference = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 176, flush=True)
    print("proper_iter22_wush_transform | WUSH-diagonal expert preconditioning | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 176, flush=True)
    print(
        f"{'Config':<38} {'PPL':>10} {'dBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 176, flush=True)
    for name, row in ordered:
        d_best = "-" if best_reference is None else f"{float(row['ppl']) - best_reference:+.4f}"
        print(
            f"{name:<38} {float(row['ppl']):>10.4f} {d_best:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
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
            "cache_path": str(args.cache_path),
            "metric_cache_path": str(args.metric_cache_path),
            "router_metric_cache_path": str(args.router_metric_cache_path),
            "quantization": "WUSH-diagonal approximation: per-expert second-moment preconditioning, transformed-weight quantization, matched activation rescaling, BF16 F.linear, FP32 logits/loss",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "CrossEntropyLoss(reduction='sum') chunked over logits, normalized per chunk, then accumulated as loss * seqlen",
            },
            "wushdiag": {
                "diag_eps": float(args.diag_eps),
                "weight_transform": "W_tilde = W @ D^{-1}",
                "activation_transform": "x_tilde = x @ D",
                "mask_family": "existing proper-iteration mask builders",
            },
        },
        "results": {},
    }

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise FileNotFoundError(f"Calibration cache missing or incompatible: {args.cache_path}")
    print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)

    router_affinity_cache = None
    if not args.force_recompute_router_metrics:
        router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, router_metric_meta = load_router_metric_cache(
            args.router_metric_cache_path,
            args.model_id,
        )
        if router_affinity_cache is not None:
            print(f"[router-cache] loaded {args.router_metric_cache_path}", flush=True)
            payload["metadata"]["router_metric_cache"] = router_metric_meta

    if router_affinity_cache is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        router_affinity_cache, hessian_cache, micromix_mean_abs, micromix_thresholds, router_metric_meta = compute_novel_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
        )
        save_router_metric_cache(
            args.router_metric_cache_path,
            args.model_id,
            router_affinity_cache,
            hessian_cache,
            micromix_mean_abs,
            micromix_thresholds,
            router_metric_meta,
        )
        print(f"[router-cache] saved {args.router_metric_cache_path}", flush=True)
        payload["metadata"]["router_metric_cache"] = router_metric_meta

    wush_artifacts = None
    if not args.force_recompute_metrics:
        wush_artifacts, wush_metric_meta = load_metric_cache(args.metric_cache_path, args.model_id, args.diag_eps)
        if wush_artifacts is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)
            payload["metadata"]["wush_metric_cache"] = wush_metric_meta

    if wush_artifacts is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        wush_artifacts, wush_metric_meta = compute_wushdiag_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
            args.diag_eps,
        )
        save_metric_cache(args.metric_cache_path, args.model_id, args.diag_eps, wush_artifacts, wush_metric_meta)
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)
        payload["metadata"]["wush_metric_cache"] = wush_metric_meta

    payload["metadata"]["wushdiag_active_experts_per_layer"] = {
        str(layer_idx): int((counts > 0).sum().item())
        for layer_idx, counts in wush_artifacts.routing_counts.items()
    }

    plans = build_iteration_plans(calibration, router_affinity_cache, text_config, non_expert_bytes, total_expert_elems)
    available_names = [plan.name for plan, _extras in plans]
    if requested_plans is not None:
        missing = sorted(requested_plans.difference(available_names))
        if missing:
            raise ValueError(f"Unknown plan names: {', '.join(missing)}")
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        payload["metadata"]["requested_plan_order"] = [plan.name for plan, _extras in plans]

    for plan, extras in plans:
        evaluate_and_record_plan(
            payload,
            plan,
            {
                **extras,
                "wushdiag_enabled": True,
                "wushdiag_diag_eps": float(args.diag_eps),
                "wushdiag_scale_cache_path": str(args.metric_cache_path),
            },
            wush_artifacts,
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

    section = render_exploration_section(payload["results"], references, eval_info)
    upsert_exploration_section(args.exploration_md, section)
    print_results_table(payload["results"], references)
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

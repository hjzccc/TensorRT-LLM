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

from baselines_comparison import LayerMetricBundle, fp8_weights_from_projection_promotions, resolve_non_expert_bytes, resolve_terminal_keys, topk_mask_from_scores
from proper_eval import CalibrationArtifacts, SEQLEN, atomic_json_dump, build_position_context, dtype_from_name, embed_chunks, layer_type_at, load_gptq_standard_data, moe_forward_eval
from proper_iter01 import build_empty_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter05 import build_joint_precision_promotions, build_mxmoe_projection_promotions_fraction
from proper_iter10_novel_perchannel import compute_novel_metric_artifacts, load_metric_cache, save_metric_cache
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    quantize_to_fp8,
    quantize_to_nvfp4_columns,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter15_residual_channels.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
MXMOE_BASE_FRACTION = 0.25
SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "output_perturbation_mxmoe_topup_router_affinity_5pct": 6.5758,
    "mxmoe_block_plus_channel_topup": 6.5763,
}


@dataclass(frozen=True)
class ResidualEvalPlan:
    name: str
    description: str
    mode: str
    memory_gb: float
    fp8_weights: int
    residual_weights: int
    w1_projection_fp8: dict[int, torch.Tensor] | None = None
    w2_projection_fp8: dict[int, torch.Tensor] | None = None
    w1_fp8_pair_masks: dict[int, dict[int, torch.Tensor]] | None = None
    w2_fp8_channel_masks: dict[int, dict[int, torch.Tensor]] | None = None
    w1_residual_pair_masks: dict[int, dict[int, torch.Tensor]] | None = None
    w2_residual_channel_masks: dict[int, dict[int, torch.Tensor]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 15 residual-channel plans.",
    )
    parser.add_argument("--force-recompute-metrics", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def load_reference_rows(output_json: Path) -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in sorted(RESULTS_DIR.glob("proper*.json")):
        if path == output_json or not path.exists():
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


def clone_projection_map(config: Any) -> dict[int, torch.Tensor]:
    return {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}


def count_projection_promotions(promotions: dict[int, torch.Tensor] | None) -> int:
    if promotions is None:
        return 0
    return int(sum(int(mask.sum().item()) for mask in promotions.values()))


def masked_weights_from_masks(
    config: Any,
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] | None,
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] | None,
) -> int:
    if w1_pair_masks is None or w2_channel_masks is None:
        return 0
    w1_pairs = sum(int(mask.sum().item()) for layer_masks in w1_pair_masks.values() for mask in layer_masks.values())
    w2_channels = sum(int(mask.sum().item()) for layer_masks in w2_channel_masks.values() for mask in layer_masks.values())
    return int((w1_pairs * 2 * config.hidden_size) + (w2_channels * config.moe_intermediate_size))


def estimate_residual_memory_gb(non_expert_bytes: int, total_expert_elems: int, fp8_weights: int, residual_weights: int) -> float:
    expert_bytes = int((0.5 * total_expert_elems) + (0.5 * fp8_weights) + (0.5 * residual_weights))
    return round(float(non_expert_bytes + expert_bytes) / 1e9, 3)


def pair_mask_to_channel_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat([pair_mask.to(torch.bool), pair_mask.to(torch.bool)], dim=0)


def quantize_with_optional_residual(weight_row_or_matrix: torch.Tensor, residual_mask: torch.Tensor) -> torch.Tensor:
    if weight_row_or_matrix.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {tuple(weight_row_or_matrix.shape)}")
    if int(residual_mask.numel()) != int(weight_row_or_matrix.shape[0]):
        raise ValueError(f"Expected residual mask length {weight_row_or_matrix.shape[0]}, got {residual_mask.numel()}")
    weight_t = weight_row_or_matrix.transpose(0, 1).contiguous()
    base = quantize_to_nvfp4_columns(weight_t)
    if not bool(residual_mask.any()):
        return base.transpose(0, 1).contiguous()
    residual = weight_t - base
    residual_q = quantize_to_nvfp4_columns(residual)
    residual_scale = residual_mask.to(device=weight_t.device, dtype=weight_t.dtype).view(1, -1)
    return (base + (residual_q * residual_scale)).transpose(0, 1).contiguous()


def quantize_with_fp8_and_optional_residual(weight: torch.Tensor, fp8_mask: torch.Tensor, residual_mask: torch.Tensor) -> torch.Tensor:
    if int(fp8_mask.numel()) != int(weight.shape[0]) or int(residual_mask.numel()) != int(weight.shape[0]):
        raise ValueError("Mask lengths must match weight output dimension")
    if bool(torch.logical_and(fp8_mask, residual_mask).any()):
        raise ValueError("FP8 and residual masks must be disjoint")
    mixed = quantize_with_optional_residual(weight, residual_mask)
    if not bool(fp8_mask.any()):
        return mixed
    mixed_t = mixed.transpose(0, 1).contiguous()
    fp8_t = quantize_to_fp8(weight.transpose(0, 1).contiguous())
    mask = fp8_mask.to(device=weight.device, dtype=torch.bool).view(1, -1)
    return torch.where(mask, fp8_t, mixed_t).transpose(0, 1).contiguous()


def quantize_gate_up_with_masks(weight: torch.Tensor, fp8_pair_mask: torch.Tensor, residual_pair_mask: torch.Tensor) -> torch.Tensor:
    if int(fp8_pair_mask.numel() * 2) != int(weight.shape[0]):
        raise ValueError(f"Expected pair mask length {weight.shape[0] // 2}, got {fp8_pair_mask.numel()}")
    return quantize_with_fp8_and_optional_residual(weight, pair_mask_to_channel_mask(fp8_pair_mask), pair_mask_to_channel_mask(residual_pair_mask))


def select_topk_and_next_topk(scores: torch.Tensor, fp8_k: int, residual_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_mask = topk_mask_from_scores(scores, fp8_k)
    residual_mask = torch.zeros_like(fp8_mask)
    remaining = int(scores.numel()) - int(fp8_mask.sum().item())
    if residual_k <= 0 or remaining <= 0:
        return fp8_mask, residual_mask
    masked_scores = scores.detach().clone()
    masked_scores[fp8_mask] = torch.finfo(masked_scores.dtype).min
    residual_mask = topk_mask_from_scores(masked_scores, min(residual_k, remaining))
    residual_mask = torch.logical_and(residual_mask, ~fp8_mask)
    return fp8_mask, residual_mask


def should_log_chunk(chunk_idx: int, total_chunks: int) -> bool:
    return chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == total_chunks


def build_projection_residual_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_proj: dict[int, torch.Tensor],
    w2_proj: dict[int, torch.Tensor],
    residual_fraction: float,
    budget_source: str,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_residual_pair_masks, w2_residual_channel_masks = build_empty_masks(config)
    w1_residual_pairs = int(round(residual_fraction * config.moe_intermediate_size))
    w2_residual_channels = int(round(residual_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if int(bundle.routing_counts[expert_idx].item()) <= 0:
                continue
            if not bool(w1_proj[layer_idx][expert_idx]) and w1_residual_pairs > 0:
                w1_residual_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], w1_residual_pairs)
            if not bool(w2_proj[layer_idx][expert_idx]) and w2_residual_channels > 0:
                w2_residual_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], w2_residual_channels)

    return w1_residual_pair_masks, w2_residual_channel_masks, {
        "budget_source": budget_source,
        "residual_fraction": float(residual_fraction),
        "w1_residual_fraction": float(residual_fraction),
        "w2_residual_fraction": float(residual_fraction),
        "mxmoe_promoted_w1_projections": count_projection_promotions(w1_proj),
        "mxmoe_promoted_w2_projections": count_projection_promotions(w2_proj),
    }


def build_projection_fp8_and_residual_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_proj: dict[int, torch.Tensor],
    w2_proj: dict[int, torch.Tensor],
    fp8_fraction: float,
    residual_fraction: float,
    budget_source: str,
) -> tuple[
    dict[int, dict[int, torch.Tensor]],
    dict[int, dict[int, torch.Tensor]],
    dict[int, dict[int, torch.Tensor]],
    dict[int, dict[int, torch.Tensor]],
    dict[str, Any],
]:
    w1_fp8_pair_masks, w2_fp8_channel_masks = build_empty_masks(config)
    w1_residual_pair_masks, w2_residual_channel_masks = build_empty_masks(config)
    w1_fp8_pairs = int(round(fp8_fraction * config.moe_intermediate_size))
    w2_fp8_channels = int(round(fp8_fraction * config.hidden_size))
    w1_residual_pairs = int(round(residual_fraction * config.moe_intermediate_size))
    w2_residual_channels = int(round(residual_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if int(bundle.routing_counts[expert_idx].item()) <= 0:
                continue
            if not bool(w1_proj[layer_idx][expert_idx]):
                w1_fp8_pair_masks[layer_idx][expert_idx], w1_residual_pair_masks[layer_idx][expert_idx] = select_topk_and_next_topk(
                    bundle.w1_pair_scores[expert_idx],
                    w1_fp8_pairs,
                    w1_residual_pairs,
                )
            if not bool(w2_proj[layer_idx][expert_idx]):
                w2_fp8_channel_masks[layer_idx][expert_idx], w2_residual_channel_masks[layer_idx][expert_idx] = select_topk_and_next_topk(
                    bundle.w2_channel_scores[expert_idx],
                    w2_fp8_channels,
                    w2_residual_channels,
                )

    return w1_fp8_pair_masks, w2_fp8_channel_masks, w1_residual_pair_masks, w2_residual_channel_masks, {
        "budget_source": budget_source,
        "topup_fp8_fraction": float(fp8_fraction),
        "residual_fraction": float(residual_fraction),
        "w1_topup_fp8_fraction": float(fp8_fraction),
        "w2_topup_fp8_fraction": float(fp8_fraction),
        "w1_residual_fraction": float(residual_fraction),
        "w2_residual_fraction": float(residual_fraction),
        "mxmoe_promoted_w1_projections": count_projection_promotions(w1_proj),
        "mxmoe_promoted_w2_projections": count_projection_promotions(w2_proj),
    }


def build_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_projection_fp8: dict[int, torch.Tensor],
    w2_projection_fp8: dict[int, torch.Tensor],
    w1_fp8_pair_masks: dict[int, dict[int, torch.Tensor]] | None,
    w2_fp8_channel_masks: dict[int, dict[int, torch.Tensor]] | None,
    w1_residual_pair_masks: dict[int, dict[int, torch.Tensor]] | None,
    w2_residual_channel_masks: dict[int, dict[int, torch.Tensor]] | None,
) -> ResidualEvalPlan:
    projection_fp8_weights = fp8_weights_from_projection_promotions(config, w1_projection_fp8, w2_projection_fp8)
    topup_fp8_weights = masked_weights_from_masks(config, w1_fp8_pair_masks, w2_fp8_channel_masks)
    residual_weights = masked_weights_from_masks(config, w1_residual_pair_masks, w2_residual_channel_masks)
    total_fp8_weights = projection_fp8_weights + topup_fp8_weights
    return ResidualEvalPlan(
        name=name,
        description=description,
        mode="residual_per_channel",
        memory_gb=estimate_residual_memory_gb(non_expert_bytes, total_expert_elems, total_fp8_weights, residual_weights),
        fp8_weights=total_fp8_weights,
        residual_weights=residual_weights,
        w1_projection_fp8=w1_projection_fp8,
        w2_projection_fp8=w2_projection_fp8,
        w1_fp8_pair_masks=w1_fp8_pair_masks,
        w2_fp8_channel_masks=w2_fp8_channel_masks,
        w1_residual_pair_masks=w1_residual_pair_masks,
        w2_residual_channel_masks=w2_residual_channel_masks,
    )


def build_iteration_plans(
    router_affinity_cache: dict[int, LayerMetricBundle],
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[ResidualEvalPlan, dict[str, Any]]]:
    plans: list[tuple[ResidualEvalPlan, dict[str, Any]]] = []
    channel_metric_meta = {
        "channel_metric_requested": "router_affinity_weighted_qerror",
        "channel_metric_effective": "router_affinity_weighted_qerror",
        "channel_metric_fallback_used": False,
    }

    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        MXMOE_BASE_FRACTION,
    )
    joint_w1_proj, joint_w2_proj, joint_meta = build_joint_precision_promotions(calibration, config)

    for fraction in (0.01, 0.03, 0.05, 0.08):
        w1_residual_masks, w2_residual_masks, extras = build_projection_residual_masks(
            router_affinity_cache,
            config,
            mxmoe_w1_proj,
            mxmoe_w2_proj,
            residual_fraction=fraction,
            budget_source="mxmoe_block_plus_router_affinity_residual",
        )
        plans.append((
            build_plan(
                f"mxmoe_block_plus_residual_{int(round(fraction * 100))}pct",
                f"Start from the 25% MxMoE per-block FP8 assignment, then add an FP4 residual top-up to the top {int(round(fraction * 100))}% router-affinity W1 pairs and W2 channels on every remaining FP4 projection.",
                config,
                non_expert_bytes,
                total_expert_elems,
                mxmoe_w1_proj,
                mxmoe_w2_proj,
                None,
                None,
                w1_residual_masks,
                w2_residual_masks,
            ),
            {
                **channel_metric_meta,
                **extras,
                "base_assignment": "mxmoe_per_block_25pct",
            },
        ))

    for fraction in (0.05, 0.03):
        w1_residual_masks, w2_residual_masks, extras = build_projection_residual_masks(
            router_affinity_cache,
            config,
            joint_w1_proj,
            joint_w2_proj,
            residual_fraction=fraction,
            budget_source="joint_percentile_buckets_plus_router_affinity_residual",
        )
        plans.append((
            build_plan(
                f"joint_topup_residual_{int(round(fraction * 100))}pct",
                f"Start from the joint three-tier FP8 base plan, then replace same-shape per-channel FP8 topups with an FP4 residual top-up on the top {int(round(fraction * 100))}% router-affinity W1 pairs and W2 channels for the remaining FP4 projections.",
                config,
                non_expert_bytes,
                total_expert_elems,
                joint_w1_proj,
                joint_w2_proj,
                None,
                None,
                w1_residual_masks,
                w2_residual_masks,
            ),
            {
                **channel_metric_meta,
                **joint_meta,
                **extras,
                "base_assignment": "joint_percentile_buckets",
                "same_channel_set_fp8_reference": f"ra_joint_w1w2_topup_{int(round(fraction * 100))}pct",
            },
        ))

    mx_w1_fp8_masks, mx_w2_fp8_masks, mx_w1_residual_masks, mx_w2_residual_masks, mx_hybrid_meta = build_projection_fp8_and_residual_masks(
        router_affinity_cache,
        config,
        mxmoe_w1_proj,
        mxmoe_w2_proj,
        fp8_fraction=0.03,
        residual_fraction=0.03,
        budget_source="mxmoe_block_plus_router_affinity_fp8_then_residual",
    )
    plans.append((
        build_plan(
            "mxmoe_block_plus_fp8_3pct_and_residual_3pct",
            "Start from the 25% MxMoE per-block FP8 assignment, then on the remaining FP4 projections promote the top 3% router-affinity channels to FP8 and give the next 3% an FP4 residual top-up.",
            config,
            non_expert_bytes,
            total_expert_elems,
            mxmoe_w1_proj,
            mxmoe_w2_proj,
            mx_w1_fp8_masks,
            mx_w2_fp8_masks,
            mx_w1_residual_masks,
            mx_w2_residual_masks,
        ),
        {
            **channel_metric_meta,
            **mx_hybrid_meta,
            "base_assignment": "mxmoe_per_block_25pct",
        },
    ))

    joint_w1_fp8_masks, joint_w2_fp8_masks, joint_w1_residual_masks, joint_w2_residual_masks, joint_hybrid_meta = build_projection_fp8_and_residual_masks(
        router_affinity_cache,
        config,
        joint_w1_proj,
        joint_w2_proj,
        fp8_fraction=0.03,
        residual_fraction=0.03,
        budget_source="joint_percentile_buckets_plus_router_affinity_fp8_then_residual",
    )
    plans.append((
        build_plan(
            "joint_fp8_3pct_residual_3pct",
            "Start from the joint three-tier FP8 base plan, then on the remaining FP4 projections promote the top 3% router-affinity channels to FP8 and give the next 3% an FP4 residual top-up.",
            config,
            non_expert_bytes,
            total_expert_elems,
            joint_w1_proj,
            joint_w2_proj,
            joint_w1_fp8_masks,
            joint_w2_fp8_masks,
            joint_w1_residual_masks,
            joint_w2_residual_masks,
        ),
        {
            **channel_metric_meta,
            **joint_meta,
            **joint_hybrid_meta,
            "base_assignment": "joint_percentile_buckets",
        },
    ))

    return plans


def prepare_layer_tensors_for_residual_plan(plan: ResidualEvalPlan, layer_idx: int, tensors: dict[str, torch.Tensor], config: Any) -> None:
    gate_up_proj = tensors["mlp.experts.gate_up_proj"]
    down_proj = tensors["mlp.experts.down_proj"]
    prepared_gate_up: list[torch.Tensor] = []
    prepared_down: list[torch.Tensor] = []

    empty_w1_pairs = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
    empty_w2_channels = torch.zeros(config.hidden_size, dtype=torch.bool)
    layer_w1_projection_fp8 = clone_projection_map(config)[layer_idx] if plan.w1_projection_fp8 is None else plan.w1_projection_fp8[layer_idx]
    layer_w2_projection_fp8 = clone_projection_map(config)[layer_idx] if plan.w2_projection_fp8 is None else plan.w2_projection_fp8[layer_idx]

    for expert_idx in range(config.num_experts):
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]

        if bool(layer_w1_projection_fp8[expert_idx]):
            quant_gate_up = quantize_to_fp8(gate_up_weight.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
        else:
            w1_fp8_pair_mask = empty_w1_pairs if plan.w1_fp8_pair_masks is None else plan.w1_fp8_pair_masks[layer_idx][expert_idx].to(torch.bool)
            w1_residual_pair_mask = empty_w1_pairs if plan.w1_residual_pair_masks is None else plan.w1_residual_pair_masks[layer_idx][expert_idx].to(torch.bool)
            quant_gate_up = quantize_gate_up_with_masks(
                gate_up_weight,
                w1_fp8_pair_mask.to(device=gate_up_weight.device),
                w1_residual_pair_mask.to(device=gate_up_weight.device),
            )

        if bool(layer_w2_projection_fp8[expert_idx]):
            quant_down = quantize_to_fp8(down_weight.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
        else:
            w2_fp8_mask = empty_w2_channels if plan.w2_fp8_channel_masks is None else plan.w2_fp8_channel_masks[layer_idx][expert_idx].to(torch.bool)
            w2_residual_mask = empty_w2_channels if plan.w2_residual_channel_masks is None else plan.w2_residual_channel_masks[layer_idx][expert_idx].to(torch.bool)
            quant_down = quantize_with_fp8_and_optional_residual(
                down_weight,
                w2_fp8_mask.to(device=down_weight.device),
                w2_residual_mask.to(device=down_weight.device),
            )

        prepared_gate_up.append(quant_gate_up)
        prepared_down.append(quant_down)

    tensors["mlp.experts.gate_up_proj"] = torch.stack(prepared_gate_up, dim=0)
    tensors["mlp.experts.down_proj"] = torch.stack(prepared_down, dim=0)
    del gate_up_proj, down_proj, prepared_gate_up, prepared_down
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate_plan_residual(
    plan: ResidualEvalPlan,
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
    final_norm = root_tensors[norm_key].to(device=device, dtype=dtype)
    lm_head = root_tensors[lm_head_key].to(device=device, dtype=dtype)
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
        prepare_layer_tensors_for_residual_plan(plan, layer_idx, tensors, config)

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
            moe_out = moe_forward_eval(mlp_input, moe_tensors, config)
            outs[chunk_idx] = residual + moe_out
            if should_log_chunk(chunk_idx, nsamples):
                print(f"[{plan.name}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{nsamples}", flush=True)

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids_device = test_ids.to(device=device, non_blocking=True)
    loss_fct = nn.CrossEntropyLoss()
    nlls: list[torch.Tensor] = []
    for chunk_idx in range(nsamples):
        hidden_states = inps[chunk_idx].unsqueeze(0)
        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
        logits = F.linear(hidden_states.float(), lm_head.float())
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = test_ids_device[:, chunk_idx * SEQLEN : (chunk_idx + 1) * SEQLEN][:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(loss.float() * SEQLEN)
        if should_log_chunk(chunk_idx, nsamples):
            print(f"[{plan.name}] logits chunk {chunk_idx + 1}/{nsamples}", flush=True)

    mean_nll = torch.stack(nlls).sum() / (nsamples * SEQLEN)
    ppl = torch.exp(mean_nll)

    release_tensors({
        "inps": inps,
        "outs": outs,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "causal_mask": causal_mask,
        "test_ids_device": test_ids_device,
    })
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(ppl.item()), float(mean_nll.item()), nsamples


def evaluate_and_record_plan(
    payload: dict[str, Any],
    plan: ResidualEvalPlan,
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
    if plan.name in result_rows:
        print(f"[skip] {plan.name} already present", flush=True)
        return result_rows[plan.name]

    print(f"\n=== Eval: {plan.name} ===", flush=True)
    start_time = time.time()
    ppl, nll, nsamples = evaluate_plan_residual(plan, test_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - start_time
    row = {
        "description": plan.description,
        "mode": plan.mode,
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "memory_gb": round(plan.memory_gb, 3),
        "fp8_weights": int(plan.fp8_weights),
        "fp8_fraction": round(float(plan.fp8_weights) / float(max(total_expert_elems, 1)), 6),
        "residual_weights": int(plan.residual_weights),
        "residual_fraction": round(float(plan.residual_weights) / float(max(total_expert_elems, 1)), 6),
        "w1_fp8_topup_pair_fraction": round(total_pair_fraction(config, plan.w1_fp8_pair_masks or {}), 6),
        "w2_fp8_topup_channel_fraction": round(total_channel_fraction(config, plan.w2_fp8_channel_masks or {}), 6),
        "w1_residual_pair_fraction": round(total_pair_fraction(config, plan.w1_residual_pair_masks or {}), 6),
        "w2_residual_channel_fraction": round(total_channel_fraction(config, plan.w2_residual_channel_masks or {}), 6),
        "base_w1_fp8_projections": count_projection_promotions(plan.w1_projection_fp8),
        "base_w2_fp8_projections": count_projection_promotions(plan.w2_projection_fp8),
        "eval_chunks": int(nsamples),
        "seqlen": SEQLEN,
        "time_s": round(elapsed, 1),
        **extras,
    }
    result_rows[plan.name] = row
    atomic_json_dump(output_json, payload)
    print(
        f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | residual={row['residual_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    target_best = min(SUCCESS_TARGETS.values())
    print("\n" + "=" * 196, flush=True)
    print("proper_iter15_residual_channels | residual-channel FP4 topups | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 196, flush=True)
    print(
        f"{'Config':<44} {'PPL':>10} {'dTarget':>10} {'Memory GB':>12} {'FP8 frac':>10} {'Residual':>10} {'W1 resid':>10} {'W2 resid':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 196, flush=True)
    for name, row in ordered:
        print(
            f"{name:<44} {float(row['ppl']):>10.4f} {float(row['ppl']) - target_best:>+10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['residual_fraction']):>10.4f} {float(row['w1_residual_pair_fraction']):>10.4f} {float(row['w2_residual_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
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
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP4 residual channels add a second NVFP4 term before BF16 matmul; FP32 logits and loss",
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
            "experiment": "Iteration 15 residual-channel ARCQuant-style FP4 topups on router-affinity-ranked channels",
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

    router_affinity_cache, _hessian_normalized_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = (None, None, None, None, None)
    if not args.force_recompute_metrics:
        router_affinity_cache, _hessian_normalized_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = load_metric_cache(
            args.metric_cache_path,
            args.model_id,
        )
        if router_affinity_cache is not None and metric_cache_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if router_affinity_cache is None or metric_cache_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = compute_novel_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
        )
        save_metric_cache(
            args.metric_cache_path,
            args.model_id,
            router_affinity_cache,
            hessian_normalized_cache,
            micromix_mean_abs,
            micromix_thresholds,
            metric_cache_meta,
        )
        del store
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache"] = metric_cache_meta
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(router_affinity_cache, calibration, text_config, non_expert_bytes, total_expert_elems)
    if requested_plans is not None:
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        missing = sorted(requested_plans.difference({plan.name for plan, _extras in plans}))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")

    for idx, (plan, extras) in enumerate(plans, start=1):
        print(f"\n=== Eval {idx}/{len(plans)}: {plan.name} ===", flush=True)
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
    print_results_table(payload["results"])
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

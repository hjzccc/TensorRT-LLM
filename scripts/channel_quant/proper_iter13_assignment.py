#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys, topk_mask_from_scores
from proper_eval import (
    SEQLEN,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    layer_type_at,
    load_gptq_standard_data,
)
from proper_iter01 import build_empty_masks, build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter10_novel_perchannel import build_global_fraction_masks, load_metric_cache as load_iter10_metric_cache
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    quantize_to_nvfp4_columns,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter13_assignment.json"
DEFAULT_ITER10_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
DEFAULT_ITER01_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
OBS_DAMP_PERCENT = 0.01
HOT_TIER_FRACTION = 0.10
MEDIUM_TIER_CUTOFF = 0.50
EPS = 1e-10


@dataclass(frozen=True)
class AssignmentPlan:
    plan: EvalPlan
    extras: dict[str, Any]
    quantization_impl: str


@dataclass
class CompensationState:
    w1_xtx_sum: dict[int, torch.Tensor]
    w2_xtx_sum: dict[int, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--iter10-cache-path", type=Path, default=DEFAULT_ITER10_CACHE_PATH)
    parser.add_argument("--iter01-cache-path", type=Path, default=DEFAULT_ITER01_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--obs-damp-percent", type=float, default=OBS_DAMP_PERCENT)
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 13 assignment plans.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def clone_metric_cache(source: dict[int, LayerMetricBundle]) -> dict[int, LayerMetricBundle]:
    cloned: dict[int, LayerMetricBundle] = {}
    for layer_idx, bundle in source.items():
        cloned[int(layer_idx)] = LayerMetricBundle(
            routing_counts=bundle.routing_counts.detach().cpu().clone(),
            w1_pair_scores=bundle.w1_pair_scores.detach().cpu().clone(),
            w2_channel_scores=bundle.w2_channel_scores.detach().cpu().clone(),
        )
    return cloned


def split_total_budget(total_fraction: float) -> tuple[float, float]:
    return total_fraction / 5.0, total_fraction * 4.0 / 5.0


def build_hybrid_router_gt_cache(
    router_affinity_cache: dict[int, LayerMetricBundle],
    hot_w2_scores: dict[int, dict[int, torch.Tensor]],
) -> dict[int, LayerMetricBundle]:
    hybrid = clone_metric_cache(router_affinity_cache)
    for layer_idx, expert_map in hot_w2_scores.items():
        if layer_idx not in hybrid:
            continue
        for expert_idx, scores in expert_map.items():
            hybrid[layer_idx].w2_channel_scores[expert_idx] = scores.detach().cpu().clone().to(torch.float32)
    return hybrid


def build_equal_loss_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    multiplier: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    per_layer_thresholds: dict[str, dict[str, float]] = {}

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        active = bundle.routing_counts > 0
        layer_meta = {"w1_threshold": 0.0, "w2_threshold": 0.0}
        if bool(active.any()):
            active_w1_scores = bundle.w1_pair_scores[active].detach().cpu().to(torch.float32).reshape(-1)
            active_w2_scores = bundle.w2_channel_scores[active].detach().cpu().to(torch.float32).reshape(-1)
            w1_threshold = float(active_w1_scores.mean().item()) * multiplier if int(active_w1_scores.numel()) > 0 else math.inf
            w2_threshold = float(active_w2_scores.mean().item()) * multiplier if int(active_w2_scores.numel()) > 0 else math.inf
            layer_meta = {
                "w1_threshold": round(w1_threshold, 8),
                "w2_threshold": round(w2_threshold, 8),
            }
            for expert_idx in range(config.num_experts):
                if not bool(active[expert_idx]):
                    continue
                w1_pair_masks[layer_idx][expert_idx] = (bundle.w1_pair_scores[expert_idx] > w1_threshold).to(torch.bool).cpu()
                w2_channel_masks[layer_idx][expert_idx] = (bundle.w2_channel_scores[expert_idx] > w2_threshold).to(torch.bool).cpu()
        per_layer_thresholds[str(layer_idx)] = layer_meta

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "baq_equal_loss_threshold",
        "loss_proxy": "router_affinity_weighted_qerror",
        "fp8_loss_proxy_ratio": 0.0625,
        "equal_loss_multiplier": float(multiplier),
        "threshold_scope": "per_layer_global_mean_per_projection",
        "per_layer_thresholds": per_layer_thresholds,
        "realized_w1_pair_fraction": round(total_pair_fraction(config, w1_pair_masks), 6),
        "realized_w2_channel_fraction": round(total_channel_fraction(config, w2_channel_masks), 6),
    }


def build_aggressive_hot_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    hot_channel_fraction: float,
    medium_channel_fraction: float,
    cold_channel_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    tier_summary: dict[str, dict[str, int]] = {}

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        active_experts = [
            expert_idx
            for expert_idx in range(config.num_experts)
            if int(bundle.routing_counts[expert_idx].item()) > 0
        ]
        ranked = sorted(
            active_experts,
            key=lambda expert_idx: (int(bundle.routing_counts[expert_idx].item()), -expert_idx),
            reverse=True,
        )
        hot_cut = max(1, int(math.ceil(len(ranked) * HOT_TIER_FRACTION))) if ranked else 0
        medium_cut = max(hot_cut, int(math.ceil(len(ranked) * MEDIUM_TIER_CUTOFF))) if ranked else 0
        hot_set = set(ranked[:hot_cut])
        medium_set = set(ranked[hot_cut:medium_cut])
        cold_set = set(ranked[medium_cut:])
        tier_summary[str(layer_idx)] = {
            "active": int(len(ranked)),
            "hot": int(len(hot_set)),
            "medium": int(len(medium_set)),
            "cold": int(len(cold_set)),
        }

        for expert_idx in ranked:
            if expert_idx in hot_set:
                fraction = hot_channel_fraction
            elif expert_idx in medium_set:
                fraction = medium_channel_fraction
            else:
                fraction = cold_channel_fraction
            w1_keep = int(round(fraction * config.moe_intermediate_size))
            w2_keep = int(round(fraction * config.hidden_size))
            w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], w1_keep)
            w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], w2_keep)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "routing_tiered_per_expert_channel_fraction",
        "hot_expert_fraction": float(HOT_TIER_FRACTION),
        "medium_expert_cutoff": float(MEDIUM_TIER_CUTOFF),
        "hot_channel_fraction": float(hot_channel_fraction),
        "medium_channel_fraction": float(medium_channel_fraction),
        "cold_channel_fraction": float(cold_channel_fraction),
        "tier_summary": tier_summary,
        "realized_w1_pair_fraction": round(total_pair_fraction(config, w1_pair_masks), 6),
        "realized_w2_channel_fraction": round(total_channel_fraction(config, w2_channel_masks), 6),
    }


def build_assignment_plans(
    router_affinity_cache: dict[int, LayerMetricBundle],
    hybrid_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[AssignmentPlan]:
    plans: list[AssignmentPlan] = []

    w1_masks, w2_masks, extras = build_equal_loss_masks(router_affinity_cache, config, multiplier=2.0)
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "baq_equalloss_2x",
                "BAQ-style equal-loss thresholding using router-affinity as the FP4 loss proxy, with per-layer global thresholds at 2x the mean channel loss.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras=extras,
            quantization_impl="standard",
        )
    )

    w1_masks, w2_masks, extras = build_equal_loss_masks(router_affinity_cache, config, multiplier=3.0)
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "baq_equalloss_3x",
                "BAQ-style equal-loss thresholding using router-affinity as the FP4 loss proxy, with per-layer global thresholds at 3x the mean channel loss.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras=extras,
            quantization_impl="standard",
        )
    )

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        router_affinity_cache,
        config,
        0.20,
        0.20,
        budget_source="router_affinity_global_topk_obs_compensated",
    )
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "ra_perchannel_compensated_20pct",
                "Router-affinity global top-k with 20% FP8 channels in both W1 and W2, while remaining FP4 rows use first-order OBS compensation.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={
                **extras,
                "channel_metric_requested": "router_affinity_weighted_qerror",
                "channel_metric_effective": "router_affinity_weighted_qerror",
                "per_projection_split": "20%/20%",
                "fp4_quantization": "first_order_obs_compensated",
            },
            quantization_impl="compensated",
        )
    )

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        router_affinity_cache,
        config,
        0.04,
        0.16,
        budget_source="router_affinity_global_topk_obs_compensated",
    )
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "ra_perchannel_compensated_w1_4_w2_16",
                "Router-affinity global top-k with W1=4% and W2=16% FP8, while remaining FP4 rows use first-order OBS compensation.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={
                **extras,
                "channel_metric_requested": "router_affinity_weighted_qerror",
                "channel_metric_effective": "router_affinity_weighted_qerror",
                "per_projection_split": "1:4",
                "fp4_quantization": "first_order_obs_compensated",
            },
            quantization_impl="compensated",
        )
    )

    for total_fraction, name, description in (
        (
            0.20,
            "ra_hybrid_gt_20pct",
            "Hybrid per-channel ranking: router-affinity everywhere, but hot experts with iter01 ground-truth W2 perturbation override router-affinity under a 20% total budget split 1:4.",
        ),
        (
            0.25,
            "ra_hybrid_gt_25pct",
            "Hybrid per-channel ranking: router-affinity everywhere, but hot experts with iter01 ground-truth W2 perturbation override router-affinity under a 25% total budget split 1:4.",
        ),
    ):
        w1_fraction, w2_fraction = split_total_budget(total_fraction)
        w1_masks, w2_masks, extras = build_global_fraction_masks(
            hybrid_cache,
            config,
            w1_fraction,
            w2_fraction,
            budget_source="router_affinity_plus_output_perturbation_global_topk",
        )
        plans.append(
            AssignmentPlan(
                plan=build_plan_from_masks(
                    name,
                    description,
                    config,
                    non_expert_bytes,
                    total_expert_elems,
                    w1_masks,
                    w2_masks,
                ),
                extras={
                    **extras,
                    "w1_channel_metric_requested": "router_affinity_weighted_qerror",
                    "w1_channel_metric_effective": "router_affinity_weighted_qerror",
                    "w2_channel_metric_requested": "output_perturbation_if_available_else_router_affinity",
                    "w2_channel_metric_effective": "iter01_hot_w2_ground_truth_with_router_affinity_fallback",
                    "per_projection_split": "1:4",
                    "total_budget_fraction": float(total_fraction),
                },
                quantization_impl="standard",
            )
        )

    w1_masks, w2_masks, extras = build_aggressive_hot_masks(
        router_affinity_cache,
        config,
        hot_channel_fraction=0.50,
        medium_channel_fraction=0.15,
        cold_channel_fraction=0.05,
    )
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "ra_aggressive_hot_50",
                "Tiered router-affinity assignment: top 10% routed experts get 50% FP8 channels, the next 40% get 15%, and the remaining active experts get 5%.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={
                **extras,
                "channel_metric_requested": "router_affinity_weighted_qerror",
                "channel_metric_effective": "router_affinity_weighted_qerror",
            },
            quantization_impl="standard",
        )
    )

    w1_masks, w2_masks, extras = build_aggressive_hot_masks(
        router_affinity_cache,
        config,
        hot_channel_fraction=0.75,
        medium_channel_fraction=0.10,
        cold_channel_fraction=0.02,
    )
    plans.append(
        AssignmentPlan(
            plan=build_plan_from_masks(
                "ra_aggressive_hot_75",
                "Tiered router-affinity assignment: top 10% routed experts get 75% FP8 channels, the next 40% get 10%, and the remaining active experts get 2%.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            extras={
                **extras,
                "channel_metric_requested": "router_affinity_weighted_qerror",
                "channel_metric_effective": "router_affinity_weighted_qerror",
            },
            quantization_impl="standard",
        )
    )

    return plans


def load_reference_rows() -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in (
        RESULTS_DIR / "proper_eval.json",
        RESULTS_DIR / "proper_iter01.json",
        RESULTS_DIR / "proper_iter02.json",
        RESULTS_DIR / "proper_iter03.json",
        RESULTS_DIR / "proper_iter04.json",
        RESULTS_DIR / "proper_iter05.json",
        RESULTS_DIR / "proper_iter06.json",
        RESULTS_DIR / "proper_iter07.json",
        RESULTS_DIR / "proper_iter08.json",
        RESULTS_DIR / "proper_iter09.json",
        RESULTS_DIR / "proper_iter09_perchannel.json",
        RESULTS_DIR / "proper_iter10_novel_perchannel.json",
        RESULTS_DIR / "proper_iter11_push_router_affinity.json",
        RESULTS_DIR / "proper_iter12_gptq_residual.json",
    ):
        if not path.exists():
            continue
        payload = load_json(path)
        for name, row in payload.get("results", {}).items():
            if isinstance(row, dict) and "ppl" in row and "memory_gb" in row:
                references[str(name)] = {
                    "ppl": float(row["ppl"]),
                    "memory_gb": float(row["memory_gb"]),
                }
    return references


def evaluate_and_record_plan(
    payload: dict[str, Any],
    assignment: AssignmentPlan,
    output_json: Path,
    model_id: str,
    config: Any,
    total_expert_elems: int,
    calib_chunks: torch.Tensor,
    test_ids: torch.Tensor,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
    obs_damp_percent: float,
) -> dict[str, Any]:
    plan = assignment.plan
    result_rows = payload.setdefault("results", {})
    if plan.name in result_rows:
        print(f"[skip] {plan.name} already present", flush=True)
        return result_rows[plan.name]

    print(f"\n=== Eval: {plan.name} ===", flush=True)
    start_time = time.time()
    if assignment.quantization_impl == "compensated":
        ppl, nll, nsamples = evaluate_compensated_plan(
            model_id,
            plan,
            calib_chunks,
            test_ids,
            config,
            weight_map,
            snapshot_dir,
            device,
            dtype,
            obs_damp_percent,
        )
    else:
        ppl, nll, nsamples = evaluate_plan(plan, test_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - start_time
    row = {
        "description": plan.description,
        "mode": plan.mode,
        "quantization_impl": assignment.quantization_impl,
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
        **assignment.extras,
    }
    result_rows[plan.name] = row
    atomic_json_dump(output_json, payload)
    print(
        f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    best_ref = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 190, flush=True)
    print("proper_iter13_assignment | novel assignment strategies | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 190, flush=True)
    print(
        f"{'Config':<40} {'Impl':<12} {'PPL':>10} {'dMxMoE':>10} {'dBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 190, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_best = "-" if best_ref is None else f"{float(row['ppl']) - best_ref:+.4f}"
        print(
            f"{name:<40} {str(row['quantization_impl']):<12} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_best:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def init_compensation_state() -> CompensationState:
    return CompensationState(w1_xtx_sum={}, w2_xtx_sum={})


def accumulate_xtx_cpu(storage: dict[int, torch.Tensor], expert_idx: int, values: torch.Tensor) -> None:
    values_f = values.float()
    xtx = torch.matmul(values_f.transpose(0, 1), values_f).detach().cpu()
    if expert_idx in storage:
        storage[expert_idx].add_(xtx)
    else:
        storage[expert_idx] = xtx


def collect_layer_compensation_stats(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: CompensationState,
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

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        accumulate_xtx_cpu(state.w1_xtx_sum, expert_idx, current_state)

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        accumulate_xtx_cpu(state.w2_xtx_sum, expert_idx, full_hidden)

        full_out = F.linear(full_hidden, down_proj[expert_idx])
        weighted_out = routing_weights[token_idx, route_pos].unsqueeze(-1) * full_out.float()
        final_hidden_states.index_add_(0, token_idx, weighted_out.to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def quantize_fp4_column(column: torch.Tensor) -> torch.Tensor:
    work = column.to(torch.float32)
    squeeze = False
    if work.ndim == 1:
        work = work.unsqueeze(-1)
        squeeze = True
    rows = work.shape[0]
    pad = (-rows) % 16
    if pad:
        work = torch.nn.functional.pad(work, (0, 0, 0, pad))
    q = quantize_to_nvfp4_columns(work).to(torch.float32)
    if pad:
        q = q[:rows]
    return q.squeeze(-1) if squeeze else q


def obs_compensated_quantize_rows(
    weight_rows: torch.Tensor,
    xtx_sum: torch.Tensor | None,
    damp_percent: float,
    device: torch.device,
) -> torch.Tensor:
    if weight_rows.numel() == 0:
        return weight_rows
    if xtx_sum is None:
        return quantize_linear_weight(weight_rows, "fp4").to(dtype=weight_rows.dtype)

    work = weight_rows.to(device=device, dtype=torch.float32).clone()
    hessian = xtx_sum.to(device=device, dtype=torch.float32).clone()
    diag = torch.diagonal(hessian)
    dead = diag <= 0
    if bool(dead.any()):
        dead_idx = torch.nonzero(dead, as_tuple=False).flatten()
        hessian[dead_idx, dead_idx] = 1.0
    mean_diag = float(torch.diagonal(hessian).mean().item()) if int(hessian.shape[0]) > 0 else 0.0
    damp = max(float(damp_percent) * mean_diag, 1e-6)
    diag_idx = torch.arange(hessian.shape[0], device=device)
    hessian[diag_idx, diag_idx] += damp

    for column_idx in range(work.shape[1]):
        hkk = float(hessian[column_idx, column_idx].item())
        quantized = quantize_fp4_column(work[:, column_idx])
        err = work[:, column_idx] - quantized
        work[:, column_idx] = quantized
        if column_idx + 1 >= work.shape[1] or hkk <= 0.0 or not math.isfinite(hkk):
            continue
        ratios = hessian[column_idx, column_idx + 1 :] / hkk
        work[:, column_idx + 1 :] -= err.unsqueeze(1) * ratios.unsqueeze(0)

    return work.to(dtype=weight_rows.dtype)


def compensated_quantize_rows(
    weight: torch.Tensor,
    fp8_row_mask: torch.Tensor,
    xtx_sum: torch.Tensor | None,
    damp_percent: float,
    device: torch.device,
) -> torch.Tensor:
    row_mask = fp8_row_mask.to(device=weight.device, dtype=torch.bool)
    if not bool(row_mask.any()):
        return obs_compensated_quantize_rows(weight, xtx_sum, damp_percent, device)
    if bool(row_mask.all()):
        return quantize_linear_weight(weight, "fp8").to(dtype=weight.dtype)

    prepared = weight.detach().clone().to(dtype=weight.dtype)
    prepared[row_mask] = quantize_linear_weight(weight[row_mask], "fp8").to(dtype=weight.dtype)
    prepared[~row_mask] = obs_compensated_quantize_rows(weight[~row_mask], xtx_sum, damp_percent, device)
    return prepared


def compensated_quantize_gate_up_with_pair_mask(
    weight: torch.Tensor,
    fp8_pair_mask: torch.Tensor,
    xtx_sum: torch.Tensor | None,
    damp_percent: float,
    device: torch.device,
) -> torch.Tensor:
    if int(fp8_pair_mask.numel() * 2) != int(weight.shape[0]):
        raise ValueError(f"Expected pair mask length {weight.shape[0] // 2}, got {fp8_pair_mask.numel()}")
    row_mask = torch.cat([fp8_pair_mask, fp8_pair_mask], dim=0).to(device=weight.device, dtype=torch.bool)
    return compensated_quantize_rows(weight, row_mask, xtx_sum, damp_percent, device)


def compensated_quantize_down_proj_with_mask(
    weight: torch.Tensor,
    fp8_mask: torch.Tensor,
    xtx_sum: torch.Tensor | None,
    damp_percent: float,
    device: torch.device,
) -> torch.Tensor:
    return compensated_quantize_rows(weight, fp8_mask.to(device=weight.device, dtype=torch.bool), xtx_sum, damp_percent, device)


def prepare_compensated_layer_tensors(
    plan: EvalPlan,
    layer_idx: int,
    tensors: dict[str, torch.Tensor],
    state: CompensationState,
    config: Any,
    damp_percent: float,
    device: torch.device,
) -> None:
    if plan.mode != "per_channel":
        raise ValueError(f"Compensated evaluation only supports per-channel plans, got {plan.mode}")
    if plan.w1_pair_masks is None or plan.w2_channel_masks is None:
        raise ValueError(f"Plan {plan.name} missing per-channel masks")

    gate_up_proj = tensors["mlp.experts.gate_up_proj"]
    down_proj = tensors["mlp.experts.down_proj"]
    prepared_gate_up: list[torch.Tensor] = []
    prepared_down: list[torch.Tensor] = []

    for expert_idx in range(config.num_experts):
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]
        prepared_gate_up.append(
            compensated_quantize_gate_up_with_pair_mask(
                gate_up_weight,
                plan.w1_pair_masks[layer_idx][expert_idx].to(device=gate_up_weight.device, dtype=torch.bool),
                state.w1_xtx_sum.get(expert_idx),
                damp_percent,
                device,
            )
        )
        prepared_down.append(
            compensated_quantize_down_proj_with_mask(
                down_weight,
                plan.w2_channel_masks[layer_idx][expert_idx].to(device=down_weight.device, dtype=torch.bool),
                state.w2_xtx_sum.get(expert_idx),
                damp_percent,
                device,
            )
        )

    tensors["mlp.experts.gate_up_proj"] = torch.stack(prepared_gate_up, dim=0)
    tensors["mlp.experts.down_proj"] = torch.stack(prepared_down, dim=0)
    del gate_up_proj, down_proj, prepared_gate_up, prepared_down
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def moe_forward_eval(hidden_states: torch.Tensor, tensors: dict[str, torch.Tensor], config: Any) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, tensors["experts.down_proj"][expert_idx])
        final_hidden_states.index_add_(0, token_idx, (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden).to(hidden_states.dtype))

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


@torch.inference_mode()
def evaluate_compensated_plan(
    model_id: str,
    plan: EvalPlan,
    calib_chunks: torch.Tensor,
    test_ids: torch.Tensor,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    damp_percent: float,
) -> tuple[float, float, int]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    nsamples = int(test_ids.numel() // SEQLEN)
    eval_chunks = test_ids[:, : nsamples * SEQLEN].view(nsamples, SEQLEN)
    store = WeightStore(model_id, snapshot_dir, weight_map)

    root_tensors = store.load_tensors([norm_key, lm_head_key])
    final_norm = root_tensors[norm_key].to(device=device, dtype=dtype)
    lm_head = root_tensors[lm_head_key].to(device=device, dtype=dtype)
    del root_tensors

    calib_inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    calib_outs = torch.zeros_like(calib_inps)
    eval_inps = embed_chunks(store, embed_key, eval_chunks, device, dtype)
    eval_outs = torch.zeros_like(eval_inps)
    causal_mask, position_embeddings = build_position_context(config, eval_inps[0:1], device)

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[{plan.name}] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        state = init_compensation_state()

        for chunk_idx in range(calib_inps.shape[0]):
            hidden_states = calib_inps[chunk_idx].unsqueeze(0)
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
            moe_out = collect_layer_compensation_stats(mlp_input, moe_tensors, config, state)
            calib_outs[chunk_idx] = residual + moe_out
            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == calib_inps.shape[0]:
                print(
                    f"[{plan.name}] calib layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{calib_inps.shape[0]}",
                    flush=True,
                )

        prepare_compensated_layer_tensors(plan, layer_idx, tensors, state, config, damp_percent, device)

        for chunk_idx in range(nsamples):
            hidden_states = eval_inps[chunk_idx].unsqueeze(0)
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
            prepared_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = moe_forward_eval(mlp_input, prepared_moe_tensors, config)
            eval_outs[chunk_idx] = residual + moe_out
            print(f"[{plan.name}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{nsamples}", flush=True)

        release_tensors(tensors)
        state.w1_xtx_sum.clear()
        state.w2_xtx_sum.clear()
        calib_inps, calib_outs = calib_outs, calib_inps
        eval_inps, eval_outs = eval_outs, eval_inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids_device = test_ids.to(device=device, non_blocking=True)
    loss_fct = nn.CrossEntropyLoss()
    nlls: list[torch.Tensor] = []
    for chunk_idx in range(nsamples):
        hidden_states = eval_inps[chunk_idx].unsqueeze(0)
        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
        logits = F.linear(hidden_states.float(), lm_head.float())
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = test_ids_device[:, chunk_idx * SEQLEN : (chunk_idx + 1) * SEQLEN][:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(loss.float() * SEQLEN)
        print(f"[{plan.name}] logits chunk {chunk_idx + 1}/{nsamples}", flush=True)

    mean_nll = torch.stack(nlls).sum() / (nsamples * SEQLEN)
    ppl = torch.exp(mean_nll)

    release_tensors(
        {
            "calib_inps": calib_inps,
            "calib_outs": calib_outs,
            "eval_inps": eval_inps,
            "eval_outs": eval_outs,
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


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.obs_damp_percent < 0.0:
        raise ValueError("--obs-damp-percent must be non-negative")

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

    references = load_reference_rows()
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "standard plans use simulated quantization (quantize -> dequantize -> BF16 -> F.linear); compensated plans replace FP4 rows with first-order OBS-compensated FP4 weights while FP8 rows remain round-to-nearest",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "iter10_cache_path": str(args.iter10_cache_path),
            "iter01_cache_path": str(args.iter01_cache_path),
            "references": references,
            "experiment": "Iteration 13 novel assignment strategies: BAQ equal-loss thresholding, OBS-compensated FP4, router-affinity plus GT hybridization, and routing-tiered budgets",
            "obs_compensation": {
                "damp_percent": float(args.obs_damp_percent),
                "formula": "after FP4-quantizing column k in the FP4 row subset, subtract err_k * (H[k, j] / H[k, k]) from later columns j using routed X^T X statistics",
            },
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

    router_affinity_cache, _hessian_normalized_cache, _micromix_mean_abs, _micromix_thresholds, iter10_meta = load_iter10_metric_cache(
        args.iter10_cache_path,
        args.model_id,
    )
    if router_affinity_cache is None or iter10_meta is None:
        raise FileNotFoundError(f"Router-affinity cache missing or incompatible: {args.iter10_cache_path}")

    calibration, hot_experts, hot_w2_scores = load_cache(args.iter01_cache_path, args.model_id)
    if calibration is None or hot_w2_scores is None:
        raise FileNotFoundError(f"Iter01 calibration/output-perturbation cache missing or incompatible: {args.iter01_cache_path}")

    hybrid_cache = build_hybrid_router_gt_cache(router_affinity_cache, hot_w2_scores)
    plans = build_assignment_plans(router_affinity_cache, hybrid_cache, text_config, non_expert_bytes, total_expert_elems)
    available_names = [assignment.plan.name for assignment in plans]
    if requested_plans is not None:
        missing = sorted(requested_plans.difference(available_names))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
        plans = [assignment for assignment in plans if assignment.plan.name in requested_plans]

    payload["metadata"]["iter10_metric_cache_metadata"] = iter10_meta
    payload["metadata"]["iter01_hot_experts"] = {} if hot_experts is None else {str(layer_idx): experts for layer_idx, experts in hot_experts.items()}
    payload["metadata"]["iter01_hot_gt_expert_counts"] = {str(layer_idx): int(len(expert_map)) for layer_idx, expert_map in hot_w2_scores.items()}
    payload["metadata"]["requested_plan_order"] = [assignment.plan.name for assignment in plans]
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    for assignment in plans:
        evaluate_and_record_plan(
            payload,
            assignment,
            args.output_json,
            args.model_id,
            text_config,
            total_expert_elems,
            calib_chunks,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
            args.obs_damp_percent,
        )

    payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload.get("results", {}), references)
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

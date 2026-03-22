#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import LayerMetricBundle, estimate_mixed_memory_gb, fp8_weights_from_projection_promotions, resolve_non_expert_bytes, topk_mask_from_scores
from proper_eval import CALIBRATION_SAMPLES, SEQLEN, CalibrationArtifacts, EvalPlan, atomic_json_dump, build_two_level_masks, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_empty_masks, build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter04.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all requested Iteration 4 plans.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def build_mxmoe_projection_promotions_fraction(
    config: Any,
    total_expert_elems: int,
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
    fraction: float,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    target_fp8_weights = int(round(fraction * total_expert_elems))
    w1_cost = 2 * config.moe_intermediate_size * config.hidden_size
    w2_cost = config.hidden_size * config.moe_intermediate_size
    items: list[tuple[float, int, int, int, str]] = []
    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            items.append((float(w1_deltas[layer_idx][expert_idx].item()) / float(w1_cost + EPS), w1_cost, layer_idx, expert_idx, "w1"))
            items.append((float(w2_deltas[layer_idx][expert_idx].item()) / float(w2_cost + EPS), w2_cost, layer_idx, expert_idx, "w2"))
    items.sort(key=lambda item: (item[0], -item[1], -item[2], -item[3], item[4]), reverse=True)

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    fp8_weights = 0
    for ratio, cost, layer_idx, expert_idx, projection in items:
        if ratio <= 0.0:
            continue
        if fp8_weights + cost > target_fp8_weights:
            continue
        if projection == "w1":
            w1_projection_fp8[layer_idx][expert_idx] = True
        else:
            w2_projection_fp8[layer_idx][expert_idx] = True
        fp8_weights += cost
        if fp8_weights >= target_fp8_weights:
            break
    return w1_projection_fp8, w2_projection_fp8


def build_mxmoe_layer_adaptive_promotions(
    config: Any,
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    w1_cost = 2 * config.moe_intermediate_size * config.hidden_size
    w2_cost = config.hidden_size * config.moe_intermediate_size
    projection_cost_per_expert = w1_cost + w2_cost
    group_specs = [
        ("layers_0_9", 0, min(10, config.num_hidden_layers), 0.15),
        ("layers_10_29", min(10, config.num_hidden_layers), min(30, config.num_hidden_layers), 0.25),
        ("layers_30_39", min(30, config.num_hidden_layers), config.num_hidden_layers, 0.40),
    ]

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    groups_meta: dict[str, dict[str, Any]] = {}

    for group_name, start_layer, end_layer, fraction in group_specs:
        if start_layer >= end_layer:
            continue
        target_fp8_weights = int(round((end_layer - start_layer) * config.num_experts * projection_cost_per_expert * fraction))
        items: list[tuple[float, int, int, int, str]] = []
        for layer_idx in range(start_layer, end_layer):
            for expert_idx in range(config.num_experts):
                items.append((float(w1_deltas[layer_idx][expert_idx].item()) / float(w1_cost + EPS), w1_cost, layer_idx, expert_idx, "w1"))
                items.append((float(w2_deltas[layer_idx][expert_idx].item()) / float(w2_cost + EPS), w2_cost, layer_idx, expert_idx, "w2"))
        items.sort(key=lambda item: (item[0], -item[1], -item[2], -item[3], item[4]), reverse=True)

        realized_fp8_weights = 0
        for ratio, cost, layer_idx, expert_idx, projection in items:
            if ratio <= 0.0:
                continue
            if realized_fp8_weights + cost > target_fp8_weights:
                continue
            if projection == "w1":
                w1_projection_fp8[layer_idx][expert_idx] = True
            else:
                w2_projection_fp8[layer_idx][expert_idx] = True
            realized_fp8_weights += cost
            if realized_fp8_weights >= target_fp8_weights:
                break

        groups_meta[group_name] = {
            "start_layer": int(start_layer),
            "end_layer_exclusive": int(end_layer),
            "fraction_target": float(fraction),
            "target_fp8_weights": int(target_fp8_weights),
            "realized_fp8_weights": int(realized_fp8_weights),
        }

    return w1_projection_fp8, w2_projection_fp8, {
        "layer_group_budgets": groups_meta,
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_projection_fp8.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_projection_fp8.values())),
    }


def build_global_channel_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    w1_flat_chunks = [metric_cache[layer_idx].w1_pair_scores.detach().cpu().to(torch.float32).reshape(-1) for layer_idx in range(config.num_hidden_layers)]
    w2_flat_chunks = [metric_cache[layer_idx].w2_channel_scores.detach().cpu().to(torch.float32).reshape(-1) for layer_idx in range(config.num_hidden_layers)]
    w1_flat = torch.cat(w1_flat_chunks, dim=0)
    w2_flat = torch.cat(w2_flat_chunks, dim=0)
    flat_scores = torch.cat((w1_flat, w2_flat), dim=0)
    total_channels = int(flat_scores.numel())
    target_channels = int(round(fraction * total_channels))
    flat_mask = torch.zeros(total_channels, dtype=torch.bool)

    if target_channels >= total_channels:
        flat_mask[:] = True
    elif target_channels > 0:
        kth = total_channels - target_channels + 1
        threshold = torch.kthvalue(flat_scores, kth).values
        flat_mask = flat_scores > threshold
        remaining = target_channels - int(flat_mask.sum().item())
        if remaining > 0:
            tie_indices = torch.nonzero(flat_scores == threshold, as_tuple=False).flatten()
            flat_mask[tie_indices[:remaining]] = True

    w1_total = int(w1_flat.numel())
    w1_mask_flat = flat_mask[:w1_total]
    w2_mask_flat = flat_mask[w1_total:]

    w1_offset = 0
    w2_offset = 0
    for layer_idx in range(config.num_hidden_layers):
        layer_w1_count = config.num_experts * config.moe_intermediate_size
        layer_w2_count = config.num_experts * config.hidden_size
        layer_w1_mask = w1_mask_flat[w1_offset : w1_offset + layer_w1_count].view(config.num_experts, config.moe_intermediate_size)
        layer_w2_mask = w2_mask_flat[w2_offset : w2_offset + layer_w2_count].view(config.num_experts, config.hidden_size)
        for expert_idx in range(config.num_experts):
            w1_pair_masks[layer_idx][expert_idx] = layer_w1_mask[expert_idx].clone()
            w2_channel_masks[layer_idx][expert_idx] = layer_w2_mask[expert_idx].clone()
        w1_offset += layer_w1_count
        w2_offset += layer_w2_count

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "global_flat_channel_ranking",
        "global_channel_fraction_target": float(fraction),
        "global_channels_total": int(total_channels),
        "global_channels_selected": int(flat_mask.sum().item()),
    }


def build_mxmoe_topup_w2_only_masks(
    calibration: CalibrationArtifacts,
    config: Any,
    total_expert_elems: int,
    base_fraction: float,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        base_fraction,
    )
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    topup_w2 = int(round(topup_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        bundle = calibration.activation_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "mxmoe_per_block_plus_w2_only_topup",
        "mxmoe_base_fraction": float(base_fraction),
        "topup_fraction": float(topup_fraction),
        "w1_topup_fraction": 0.0,
        "w2_topup_fraction": float(topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_per_block_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_proj: dict[int, torch.Tensor],
    w2_proj: dict[int, torch.Tensor],
) -> EvalPlan:
    fp8_weights = fp8_weights_from_projection_promotions(config, w1_proj, w2_proj)
    return EvalPlan(
        name=name,
        description=description,
        mode="per_block",
        memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_projection_fp8=w1_proj,
        w2_projection_fp8=w2_proj,
    )


def load_reference_rows() -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in (
        RESULTS_DIR / "proper_eval.json",
        RESULTS_DIR / "proper_iter01.json",
        RESULTS_DIR / "proper_iter02.json",
        RESULTS_DIR / "proper_iter03.json",
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
    plan: EvalPlan,
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
    ppl, nll, nsamples = evaluate_plan(plan, test_ids, config, weight_map, snapshot_dir, device, dtype)
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


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    topup_ref = references.get("topup_05pct", {}).get("ppl")
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    print("\n" + "=" * 160, flush=True)
    print("proper_iter04 | fundamentally different budget/ranking configs | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 160, flush=True)
    print(
        f"{'Config':<34} {'PPL':>10} {'dTopup05':>10} {'dMxMoE25':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 160, flush=True)
    for name, row in ordered:
        d_topup = "-" if topup_ref is None else f"{float(row['ppl']) - topup_ref:+.4f}"
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        print(
            f"{name:<34} {float(row['ppl']):>10.4f} {d_topup:>10} {d_mxmoe:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []

    w1_proj, w2_proj, adaptive_meta = build_mxmoe_layer_adaptive_promotions(config, calibration.mxmoe_w1_deltas, calibration.mxmoe_w2_deltas)
    plans.append((
        build_per_block_plan(
            "mxmoe_layer_adaptive",
            "MxMoE per-block knapsack with layer-group budgets: 15% for layers 0-9, 25% for 10-29, and 40% for 30+.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_proj,
            w2_proj,
        ),
        {
            "budget_source": "mxmoe_layer_group_knapsack",
            "channel_metric_requested": "output_perturbation",
            "channel_metric_effective": "output_perturbation",
            "channel_metric_fallback_used": False,
            **adaptive_meta,
        },
    ))

    w1_masks, w2_masks, global_meta = build_global_channel_masks(calibration.activation_cache, config, 0.25)
    plans.append((
        build_plan_from_masks(
            "global_channel_ranking",
            "Global activation_kurtosis ranking across all W1 pairs and W2 channels; top 25% channel units promoted regardless of layer or expert.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            **global_meta,
        },
    ))

    for fraction, name in ((0.15, "mxmoe_block_15pct"), (0.20, "mxmoe_block_20pct"), (0.10, "mxmoe_block_10pct")):
        w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
            config,
            total_expert_elems,
            calibration.mxmoe_w1_deltas,
            calibration.mxmoe_w2_deltas,
            fraction,
        )
        plans.append((
            build_per_block_plan(
                name,
                f"MxMoE greedy knapsack over per-projection output perturbation at a {fraction:.0%} FP8 budget.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_proj,
                w2_proj,
            ),
            {
                "budget_source": "mxmoe_projection_perturbation",
                "channel_metric_requested": "output_perturbation",
                "channel_metric_effective": "output_perturbation",
                "channel_metric_fallback_used": False,
                "mxmoe_fraction_target": float(fraction),
                "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
                "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
            },
        ))

    for w2_fraction, name in ((0.25, "perchannel_akurt_w1_0_w2_25"), (0.40, "perchannel_akurt_w1_0_w2_40")):
        w1_masks, w2_masks = build_two_level_masks(calibration.activation_cache, config, 0.0, w2_fraction)
        plans.append((
            build_plan_from_masks(
                name,
                f"Routing-aware activation_kurtosis masks with W1=0% and W2={w2_fraction:.0%} FP8.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                "budget_source": "routing_aware_two_level",
                "channel_metric_requested": "activation_kurtosis",
                "channel_metric_effective": "activation_kurtosis",
                "channel_metric_fallback_used": False,
                "w1_fraction_target": 0.0,
                "w2_fraction_target": float(w2_fraction),
            },
        ))

    w1_masks, w2_masks, topup_meta = build_mxmoe_topup_w2_only_masks(
        calibration,
        config,
        total_expert_elems,
        base_fraction=0.25,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "mxmoe_topup_w2_only_5pct",
            "Start from the 25% MxMoE per-block assignment, keep W1 fixed at block precision, and top up only W2 channels by 5% activation_kurtosis.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            **topup_meta,
        },
    ))

    return plans


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
    _tokenizer, _calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
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
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP32 logits and loss",
            "protocol": {
                "reference": "/home/jerry/Documents/fork_new/MC-MoE/eval_ppl_utils.py",
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "references": references,
            "experiment": "Iteration 4 fundamentally different budget/ranking variants using cached proper_eval calibration",
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
        raise RuntimeError(f"Calibration cache missing or incompatible: {args.cache_path}")
    print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)

    plans = build_iteration_plans(calibration, text_config, non_expert_bytes, total_expert_elems)
    available_names = [plan.name for plan, _extras in plans]
    if requested_plans is not None:
        missing = sorted(requested_plans.difference(available_names))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
    payload["metadata"]["requested_plan_order"] = [plan.name for plan, _extras in plans]
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

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
    payload["metadata"]["cache_version_expectation"] = {
        "seqlen": SEQLEN,
        "calibration_samples": CALIBRATION_SAMPLES,
    }
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload["results"], references)
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

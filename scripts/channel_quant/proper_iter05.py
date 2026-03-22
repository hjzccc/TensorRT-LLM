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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter05.json"
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
        help="Comma-separated subset of plan names to evaluate; default runs all requested Iteration 5 plans.",
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


def build_w2_metric_cache(
    activation_cache: dict[int, LayerMetricBundle],
    hot_w2_scores: dict[int, dict[int, torch.Tensor]] | None,
    mode: str,
) -> tuple[dict[int, LayerMetricBundle], dict[str, Any]]:
    if mode not in {"perturbation", "combined"}:
        raise ValueError(f"Unsupported W2 metric cache mode: {mode}")

    metric_cache = clone_metric_cache(activation_cache)
    total_active_experts = 0
    perturbation_backed_active_experts = 0
    fallback_active_experts = 0

    for layer_idx, bundle in metric_cache.items():
        available = {} if hot_w2_scores is None else hot_w2_scores.get(layer_idx, {})
        active_mask = activation_cache[layer_idx].routing_counts > 0
        total_active_experts += int(active_mask.sum().item())
        for expert_idx in range(int(bundle.routing_counts.numel())):
            activation_scores = activation_cache[layer_idx].w2_channel_scores[expert_idx].detach().cpu().to(torch.float32)
            perturb_scores = available.get(expert_idx)
            if perturb_scores is None:
                metric_cache[layer_idx].w2_channel_scores[expert_idx] = activation_scores
                if bool(active_mask[expert_idx]):
                    fallback_active_experts += 1
                continue

            perturbation_backed_active_experts += int(bool(active_mask[expert_idx]))
            perturb_scores_f = perturb_scores.detach().cpu().to(torch.float32)
            if mode == "perturbation":
                combined_scores = perturb_scores_f
            else:
                combined_scores = perturb_scores_f * activation_scores
            metric_cache[layer_idx].w2_channel_scores[expert_idx] = combined_scores

    requested_name = "output_perturbation_per_channel" if mode == "perturbation" else "output_perturbation_per_channel_times_activation_kurtosis"
    effective_name = requested_name if fallback_active_experts == 0 else f"{requested_name}_with_activation_kurtosis_fallback"
    return metric_cache, {
        "channel_metric_requested": requested_name,
        "channel_metric_effective": effective_name,
        "channel_metric_fallback_used": bool(fallback_active_experts > 0),
        "perturbation_backed_active_experts": int(perturbation_backed_active_experts),
        "fallback_active_experts": int(fallback_active_experts),
        "total_active_experts": int(total_active_experts),
    }


def build_mxmoe_projection_promotions_fraction(
    config: Any,
    total_expert_elems: int,
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
    fraction: float,
    w2_density_cost_scale: float = 1.0,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    target_fp8_weights = int(round(fraction * total_expert_elems))
    w1_cost = 2 * config.moe_intermediate_size * config.hidden_size
    w2_cost = config.hidden_size * config.moe_intermediate_size
    effective_w2_cost = max(float(w2_cost) * float(w2_density_cost_scale), EPS)
    items: list[tuple[float, int, int, int, str]] = []

    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            items.append((float(w1_deltas[layer_idx][expert_idx].item()) / float(w1_cost + EPS), w1_cost, layer_idx, expert_idx, "w1"))
            items.append((float(w2_deltas[layer_idx][expert_idx].item()) / effective_w2_cost, w2_cost, layer_idx, expert_idx, "w2"))
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


def build_mxmoe_topup_masks_for_w2_metric(
    metric_cache: dict[int, LayerMetricBundle],
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
    topup_w1 = int(round(topup_fraction * config.moe_intermediate_size))
    topup_w2 = int(round(topup_fraction * config.hidden_size))
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)

    for layer_idx in range(config.num_hidden_layers):
        activation_bundle = calibration.activation_cache[layer_idx]
        metric_bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            else:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(activation_bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(metric_bundle.w2_channel_scores[expert_idx], topup_w2)

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "mxmoe_per_block_plus_channel_topup",
        "mxmoe_base_fraction": float(base_fraction),
        "topup_fraction": float(topup_fraction),
        "w1_topup_fraction": float(topup_fraction),
        "w2_topup_fraction": float(topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_joint_precision_promotions(
    calibration: CalibrationArtifacts,
    config: Any,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    active_scores: list[torch.Tensor] = []

    for layer_idx in range(config.num_hidden_layers):
        combined_scores = (calibration.mxmoe_w1_deltas[layer_idx] + calibration.mxmoe_w2_deltas[layer_idx]).detach().cpu().to(torch.float32)
        active_mask = calibration.routing_counts[layer_idx] > 0
        if bool(active_mask.any()):
            active_scores.append(combined_scores[active_mask])

    if active_scores:
        flat_active = torch.cat(active_scores, dim=0)
        p50 = float(torch.quantile(flat_active, 0.50).item())
        p75 = float(torch.quantile(flat_active, 0.75).item())
    else:
        p50 = 0.0
        p75 = 0.0

    high_count = 0
    medium_count = 0
    low_count = 0
    for layer_idx in range(config.num_hidden_layers):
        combined_scores = (calibration.mxmoe_w1_deltas[layer_idx] + calibration.mxmoe_w2_deltas[layer_idx]).detach().cpu().to(torch.float32)
        active_mask = calibration.routing_counts[layer_idx] > 0
        for expert_idx in range(config.num_experts):
            if not bool(active_mask[expert_idx]):
                low_count += 1
                continue
            score = float(combined_scores[expert_idx].item())
            if score >= p75:
                w1_projection_fp8[layer_idx][expert_idx] = True
                w2_projection_fp8[layer_idx][expert_idx] = True
                high_count += 1
            elif score >= p50:
                w2_projection_fp8[layer_idx][expert_idx] = True
                medium_count += 1
            else:
                low_count += 1

    return w1_projection_fp8, w2_projection_fp8, {
        "budget_source": "global_perturbation_percentile_buckets",
        "joint_precision_threshold_p50": float(p50),
        "joint_precision_threshold_p75": float(p75),
        "joint_precision_high_experts": int(high_count),
        "joint_precision_medium_experts": int(medium_count),
        "joint_precision_low_experts": int(low_count),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_projection_fp8.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_projection_fp8.values())),
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
        RESULTS_DIR / "proper_iter04.json",
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
    print("\n" + "=" * 172, flush=True)
    print("proper_iter05 | per-channel output perturbation variants | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 172, flush=True)
    print(
        f"{'Config':<38} {'PPL':>10} {'dTopup05':>10} {'dMxMoE25':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 172, flush=True)
    for name, row in ordered:
        d_topup = "-" if topup_ref is None else f"{float(row['ppl']) - topup_ref:+.4f}"
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        print(
            f"{name:<38} {float(row['ppl']):>10.4f} {d_topup:>10} {d_mxmoe:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    hot_w2_scores: dict[int, dict[int, torch.Tensor]] | None,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    perturb_cache, perturb_meta = build_w2_metric_cache(calibration.activation_cache, hot_w2_scores, mode="perturbation")
    combined_cache, combined_meta = build_w2_metric_cache(calibration.activation_cache, hot_w2_scores, mode="combined")

    for name, topup_fraction in (
        ("mxmoe_topup_perturbation_channel", 0.05),
        ("mxmoe_topup_perturbation_10pct", 0.10),
        ("mxmoe_topup_perturbation_15pct", 0.15),
    ):
        w1_masks, w2_masks, topup_meta = build_mxmoe_topup_masks_for_w2_metric(
            perturb_cache,
            calibration,
            config,
            total_expert_elems,
            base_fraction=0.25,
            topup_fraction=topup_fraction,
        )
        plans.append((
            build_plan_from_masks(
                name,
                f"Start from the 25% MxMoE per-block assignment, then top up FP4 projections with {topup_fraction:.0%} W1 activation_kurtosis and W2 per-channel output perturbation (activation_kurtosis fallback where cache is missing).",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **perturb_meta,
                **topup_meta,
            },
        ))

    w1_masks, w2_masks = build_two_level_masks(perturb_cache, config, 0.25, 0.25)
    plans.append((
        build_plan_from_masks(
            "pure_perturbation_perchannel_25pct",
            "Pure routing-aware per-channel plan with W1=25% activation_kurtosis and W2=25% per-channel output perturbation, falling back to activation_kurtosis where perturbation cache is missing.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **perturb_meta,
            "budget_source": "routing_aware_two_level",
            "w1_fraction_target": 0.25,
            "w2_fraction_target": 0.25,
        },
    ))

    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        fraction=0.25,
        w2_density_cost_scale=0.25,
    )
    plans.append((
        build_per_block_plan(
            "mxmoe_block_w2_heavy",
            "MxMoE greedy knapsack at a 25% actual FP8 budget, but rank W2 projections as if they cost one quarter as much to bias the selection toward W2.",
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
            "mxmoe_fraction_target": 0.25,
            "w2_density_cost_scale": 0.25,
            "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
            "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
        },
    ))

    w1_proj, w2_proj, joint_meta = build_joint_precision_promotions(calibration, config)
    plans.append((
        build_per_block_plan(
            "joint_w1_w2_optimization",
            "Global three-tier expert bucketing from perturbation percentiles: p75 experts get both W1 and W2 in FP8, p50-p75 experts get W2 only, and the rest stay FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_proj,
            w2_proj,
        ),
        {
            "channel_metric_requested": "output_perturbation",
            "channel_metric_effective": "output_perturbation",
            "channel_metric_fallback_used": False,
            **joint_meta,
        },
    ))

    w1_masks, w2_masks, topup_meta = build_mxmoe_topup_masks_for_w2_metric(
        combined_cache,
        calibration,
        config,
        total_expert_elems,
        base_fraction=0.25,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "mxmoe_topup_combined_5pct",
            "Start from the 25% MxMoE per-block assignment, then top up FP4 projections with 5% W1 activation_kurtosis and a combined W2 score of per-channel output perturbation multiplied by activation_kurtosis.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **combined_meta,
            **topup_meta,
        },
    ))

    w1_masks, w2_masks = build_two_level_masks(perturb_cache, config, 0.04, 0.16)
    plans.append((
        build_plan_from_masks(
            "perturbation_per_channel_w1_4_w2_16",
            "Routing-aware per-channel plan with W1=4% activation_kurtosis and W2=16% per-channel output perturbation, using activation_kurtosis as the W2 fallback when perturbation cache is missing.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **perturb_meta,
            "budget_source": "routing_aware_two_level",
            "w1_fraction_target": 0.04,
            "w2_fraction_target": 0.16,
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
            "experiment": "Iteration 5 per-channel output perturbation variants using cached proper_eval calibration",
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

    calibration, hot_experts, hot_w2_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        raise RuntimeError(f"Calibration cache missing or incompatible: {args.cache_path}")
    print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)

    hot_layer_count = 0 if hot_w2_scores is None else len(hot_w2_scores)
    hot_expert_count = 0 if hot_w2_scores is None else sum(len(expert_map) for expert_map in hot_w2_scores.values())
    payload["metadata"]["hot_w2_channel_perturbation"] = {
        "present": hot_w2_scores is not None,
        "layers_with_cached_scores": int(hot_layer_count),
        "experts_with_cached_scores": int(hot_expert_count),
        "hot_experts_per_layer": {} if hot_experts is None else {str(layer_idx): len(experts) for layer_idx, experts in hot_experts.items()},
    }

    plans = build_iteration_plans(calibration, hot_w2_scores, text_config, non_expert_bytes, total_expert_elems)
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

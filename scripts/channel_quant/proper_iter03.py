#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import LayerMetricBundle, estimate_mixed_memory_gb, fp8_weights_from_projection_promotions, resolve_non_expert_bytes
from proper_eval import CALIBRATION_SAMPLES, SEQLEN, CalibrationArtifacts, EvalPlan, atomic_json_dump, build_two_level_masks, dtype_from_name, load_gptq_standard_data, run_calibration
from proper_iter01 import build_combined_perturbation_scores, build_plan_from_masks, build_weighted_masks_from_fractions, load_cache, maybe_eval_plan, save_cache
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter03.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"

HESSIAN_CACHE_KEYS = (
    "hessian_diag_cache",
    "metric_cache_hessian_diag",
    "perchannel_hessian_diag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clone_metric_cache(source: dict[int, LayerMetricBundle]) -> dict[int, LayerMetricBundle]:
    cloned: dict[int, LayerMetricBundle] = {}
    for layer_idx, bundle in source.items():
        cloned[int(layer_idx)] = LayerMetricBundle(
            routing_counts=bundle.routing_counts.detach().cpu().clone(),
            w1_pair_scores=bundle.w1_pair_scores.detach().cpu().clone(),
            w2_channel_scores=bundle.w2_channel_scores.detach().cpu().clone(),
        )
    return cloned


def restore_metric_cache(raw_cache: Any) -> dict[int, LayerMetricBundle] | None:
    if not isinstance(raw_cache, dict):
        return None
    restored: dict[int, LayerMetricBundle] = {}
    for raw_layer_idx, layer_payload in raw_cache.items():
        if not isinstance(layer_payload, dict):
            return None
        required = {"routing_counts", "w1_pair_scores", "w2_channel_scores"}
        if not required.issubset(layer_payload):
            return None
        layer_idx = int(raw_layer_idx)
        restored[layer_idx] = LayerMetricBundle(
            routing_counts=torch.as_tensor(layer_payload["routing_counts"]).detach().cpu().clone().to(torch.int64),
            w1_pair_scores=torch.as_tensor(layer_payload["w1_pair_scores"]).detach().cpu().clone().to(torch.float32),
            w2_channel_scores=torch.as_tensor(layer_payload["w2_channel_scores"]).detach().cpu().clone().to(torch.float32),
        )
    return restored


def resolve_hessian_metric_cache(cache_path: Path, calibration: CalibrationArtifacts) -> tuple[dict[int, LayerMetricBundle], dict[str, Any]]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        candidate_caches: list[Any] = [payload.get(key) for key in HESSIAN_CACHE_KEYS if isinstance(payload, dict)]
        if isinstance(payload, dict):
            metric_caches = payload.get("metric_caches")
            if isinstance(metric_caches, dict):
                candidate_caches.append(metric_caches.get("hessian_diag"))
            base = payload.get("base_calibration")
            if isinstance(base, dict):
                candidate_caches.extend(base.get(key) for key in HESSIAN_CACHE_KEYS)
        for candidate in candidate_caches:
            restored = restore_metric_cache(candidate)
            if restored is not None:
                return restored, {
                    "requested_metric": "hessian_diag",
                    "effective_metric": "hessian_diag",
                    "fallback_used": False,
                    "source": str(cache_path),
                }
    return clone_metric_cache(calibration.activation_cache), {
        "requested_metric": "hessian_diag",
        "effective_metric": "activation_kurtosis",
        "fallback_used": True,
        "source": str(cache_path),
        "fallback_reason": "proper_iter01 cache only stores routing counts, activation_kurtosis scores, and projection perturbation summaries; it does not retain E[x^2] or raw moments needed to reconstruct hessian_diag without recalibration",
    }


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
            items.append((float(w1_deltas[layer_idx][expert_idx].item()) / float(w1_cost + 1e-10), w1_cost, layer_idx, expert_idx, "w1"))
            items.append((float(w2_deltas[layer_idx][expert_idx].item()) / float(w2_cost + 1e-10), w2_cost, layer_idx, expert_idx, "w2"))
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


def build_mxmoe_topup_masks_for_metric(
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
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        w1_pair_masks[layer_idx] = {}
        w2_channel_masks[layer_idx] = {}
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            else:
                scores = bundle.w1_pair_scores[expert_idx]
                count = min(topup_w1, int(scores.numel()))
                if count <= 0:
                    w1_pair_masks[layer_idx][expert_idx] = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
                elif count >= int(scores.numel()):
                    w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                else:
                    indices = torch.topk(scores, k=count, largest=True, sorted=False).indices
                    mask = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
                    mask[indices] = True
                    w1_pair_masks[layer_idx][expert_idx] = mask

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                scores = bundle.w2_channel_scores[expert_idx]
                count = min(topup_w2, int(scores.numel()))
                if count <= 0:
                    w2_channel_masks[layer_idx][expert_idx] = torch.zeros(config.hidden_size, dtype=torch.bool)
                elif count >= int(scores.numel()):
                    w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
                else:
                    indices = torch.topk(scores, k=count, largest=True, sorted=False).indices
                    mask = torch.zeros(config.hidden_size, dtype=torch.bool)
                    mask[indices] = True
                    w2_channel_masks[layer_idx][expert_idx] = mask

    return w1_pair_masks, w2_channel_masks, {
        "mxmoe_base_fraction": float(base_fraction),
        "topup_fraction": float(topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def load_reference_rows() -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in (RESULTS_DIR / "proper_eval.json", RESULTS_DIR / "proper_iter01.json", RESULTS_DIR / "proper_iter02.json"):
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


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    topup_ref = references.get("topup_05pct", {}).get("ppl")
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    print("\n" + "=" * 164, flush=True)
    print("proper_iter03 | broader exploration | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 164, flush=True)
    print(
        f"{'Config':<36} {'PPL':>10} {'dTopup05':>10} {'dMxMoE25':>10} {'Memory GB':>12} {'FP8 frac':>10} {'Metric':<20} {'Time s':>10}",
        flush=True,
    )
    print("-" * 164, flush=True)
    for name, row in ordered:
        d_topup = "-" if topup_ref is None else f"{float(row['ppl']) - topup_ref:+.4f}"
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        metric = str(row.get("channel_metric_effective") or row.get("channel_metric") or row.get("budget_source") or row["mode"])
        print(
            f"{name:<36} {float(row['ppl']):>10.4f} {d_topup:>10} {d_mxmoe:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {metric:<20.20} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
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
            "experiment": "Iteration 3 broader exploration with cached proper_eval calibration",
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

    cache_calibration, _cache_hot_experts, _cache_hot_scores = load_cache(args.cache_path, args.model_id)
    if cache_calibration is not None:
        calibration = cache_calibration
        print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)
    else:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        calibration = run_calibration(store, text_config, calib_chunks, device, dtype)
        save_cache(args.cache_path, args.model_id, calibration)
        print(f"[cache] saved base calibration to {args.cache_path}", flush=True)

    hessian_metric_cache, hessian_meta = resolve_hessian_metric_cache(args.cache_path, calibration)
    payload["metadata"]["hessian_metric"] = hessian_meta
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    payload["metadata"]["calibration_cache_compatible"] = cache_calibration is not None
    atomic_json_dump(args.output_json, payload)

    combined_scores = build_combined_perturbation_scores(calibration)
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []

    w1_masks, w2_masks = build_two_level_masks(hessian_metric_cache, text_config, 0.25, 0.25)
    plans.append((
        build_plan_from_masks(
            "perchannel_hessian_25pct",
            "Routing-aware per-channel plan at 25% W1 and 25% W2, requesting hessian_diag channel ranking.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "routing_aware_two_level",
            "w1_fraction_target": 0.25,
            "w2_fraction_target": 0.25,
            "channel_metric_requested": "hessian_diag",
            "channel_metric_effective": hessian_meta["effective_metric"],
            "channel_metric_fallback_used": bool(hessian_meta["fallback_used"]),
        },
    ))

    w1_masks, w2_masks = build_two_level_masks(hessian_metric_cache, text_config, 0.04, 0.16)
    plans.append((
        build_plan_from_masks(
            "perchannel_hessian_w1_4_w2_16",
            "Routing-aware per-channel plan with W1=4% and W2=16%, requesting hessian_diag channel ranking.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "routing_aware_two_level",
            "w1_fraction_target": 0.04,
            "w2_fraction_target": 0.16,
            "channel_metric_requested": "hessian_diag",
            "channel_metric_effective": hessian_meta["effective_metric"],
            "channel_metric_fallback_used": bool(hessian_meta["fallback_used"]),
        },
    ))

    w1_masks, w2_masks = build_two_level_masks(calibration.activation_cache, text_config, 0.30, 0.30)
    plans.append((
        build_plan_from_masks(
            "perchannel_akurt_30pct",
            "Routing-aware per-channel activation_kurtosis plan at 30% W1 and 30% W2.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "routing_aware_two_level",
            "w1_fraction_target": 0.30,
            "w2_fraction_target": 0.30,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
        },
    ))

    w1_masks, w2_masks = build_two_level_masks(calibration.activation_cache, text_config, 0.40, 0.40)
    plans.append((
        build_plan_from_masks(
            "perchannel_akurt_40pct",
            "Routing-aware per-channel activation_kurtosis plan at 40% W1 and 40% W2.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "routing_aware_two_level",
            "w1_fraction_target": 0.40,
            "w2_fraction_target": 0.40,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
        },
    ))

    w1_masks, w2_masks, topup_meta = build_mxmoe_topup_masks_for_metric(
        hessian_metric_cache,
        calibration,
        text_config,
        total_expert_elems,
        base_fraction=0.25,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "mxmoe_topup_hessian_5pct",
            "Start from the 25% MxMoE per-block assignment, then top up remaining FP4 channels with a 5% requested hessian_diag ranking.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "mxmoe_per_block_plus_topup",
            "channel_metric_requested": "hessian_diag",
            "channel_metric_effective": hessian_meta["effective_metric"],
            "channel_metric_fallback_used": bool(hessian_meta["fallback_used"]),
            **topup_meta,
        },
    ))

    w1_masks, w2_masks, budget_meta = build_weighted_masks_from_fractions(
        calibration.activation_cache,
        text_config,
        combined_scores,
        combined_scores,
        0.30,
        0.30,
    )
    plans.append((
        build_plan_from_masks(
            "perturbation_proportional_30pct",
            "Allocate a 30% W1 and 30% W2 channel budget across experts in proportion to combined output perturbation, then rank channels by activation_kurtosis.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "combined_perturbation_global",
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "w1_fraction_target": 0.30,
            "w2_fraction_target": 0.30,
            **budget_meta,
        },
    ))

    w1_masks, w2_masks, budget_meta = build_weighted_masks_from_fractions(
        calibration.activation_cache,
        text_config,
        combined_scores,
        combined_scores,
        0.40,
        0.40,
    )
    plans.append((
        build_plan_from_masks(
            "perturbation_proportional_40pct",
            "Allocate a 40% W1 and 40% W2 channel budget across experts in proportion to combined output perturbation, then rank channels by activation_kurtosis.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "combined_perturbation_global",
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "w1_fraction_target": 0.40,
            "w2_fraction_target": 0.40,
            **budget_meta,
        },
    ))

    w1_proj, w2_proj = build_mxmoe_projection_promotions_fraction(
        text_config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        0.35,
    )
    mxmoe_fp8_weights = fp8_weights_from_projection_promotions(text_config, w1_proj, w2_proj)
    plans.append((
        EvalPlan(
            name="mxmoe_block_35pct",
            description="MxMoE greedy knapsack over per-projection output perturbation at a 35% FP8 budget.",
            mode="per_block",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, mxmoe_fp8_weights),
            fp8_weights=mxmoe_fp8_weights,
            w1_projection_fp8=w1_proj,
            w2_projection_fp8=w2_proj,
        ),
        {
            "budget_source": "mxmoe_projection_perturbation",
            "channel_metric_requested": "output_perturbation",
            "channel_metric_effective": "output_perturbation",
            "channel_metric_fallback_used": False,
            "mxmoe_fraction_target": 0.35,
            "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
            "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
        },
    ))

    for plan, extras in plans:
        maybe_eval_plan(
            payload,
            plan,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
            args.output_json,
            extras,
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

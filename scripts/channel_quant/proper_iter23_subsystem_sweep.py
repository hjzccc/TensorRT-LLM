#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import expert_full_weight_count, fp8_weights_from_masks, resolve_non_expert_bytes
from proper_eval import SEQLEN, CalibrationArtifacts, EvalPlan, atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter21_serq_salient import load_reference_rows
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter23_subsystem_sweep.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"


@dataclass(frozen=True)
class RestoreSpec:
    name: str
    description: str
    restore_bf16: frozenset[str]
    restore_moe_layers: frozenset[int]
    effective_restore: bool
    reason: str


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
        help="Comma-separated subset of plan names to evaluate; default runs the base plan plus all subsystem restorations.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def clone_masks(masks: dict[int, dict[int, torch.Tensor]]) -> dict[int, dict[int, torch.Tensor]]:
    return {
        int(layer_idx): {
            int(expert_idx): mask.detach().cpu().clone()
            for expert_idx, mask in layer_masks.items()
        }
        for layer_idx, layer_masks in masks.items()
    }


def zero_moe_layers(
    masks: dict[int, dict[int, torch.Tensor]],
    layers: frozenset[int],
) -> dict[int, dict[int, torch.Tensor]]:
    cloned = clone_masks(masks)
    for layer_idx in layers:
        for expert_idx, mask in cloned[layer_idx].items():
            cloned[layer_idx][expert_idx] = torch.zeros_like(mask, dtype=torch.bool)
    return cloned


def count_layer_fp8_weights(
    config: Any,
    w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_channel_masks: dict[int, dict[int, torch.Tensor]],
    layer_idx: int,
) -> int:
    w1_pairs = sum(int(mask.sum().item()) for mask in w1_pair_masks[layer_idx].values())
    w2_channels = sum(int(mask.sum().item()) for mask in w2_channel_masks[layer_idx].values())
    return int((w1_pairs * 2 * config.hidden_size) + (w2_channels * config.moe_intermediate_size))


def estimate_restored_memory_gb(
    non_expert_bytes: int,
    total_expert_elems: int,
    fp8_weights: int,
    restored_bf16_weights: int,
) -> float:
    expert_bytes = int((0.5 * total_expert_elems) + (0.5 * fp8_weights) + (1.5 * restored_bf16_weights))
    return round(float(non_expert_bytes + expert_bytes) / 1e9, 3)


def build_base_plan(
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[EvalPlan, dict[str, Any]]:
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    plan = build_plan_from_masks(
        "joint_w1w2_with_topup",
        (
            "Reuse the Iteration 5 global three-tier perturbation buckets, keep the high tier fully FP8, "
            "and replace the medium tier's coarse promotion with 8% within-expert W1/W2 topups by activation_kurtosis."
        ),
        config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )
    return plan, {
        **joint_meta,
        "channel_metric_requested": "activation_kurtosis",
        "channel_metric_effective": "activation_kurtosis",
        "channel_metric_fallback_used": False,
        "restored_components": [],
        "restored_moe_layers": [],
        "effective_restore": False,
        "restore_note": "Base joint_w1w2_with_topup plan with no BF16 restoration overrides.",
        "restored_bf16_weights": 0,
        "restored_bf16_fraction": 0.0,
    }


def build_layer_output_perturbation_scores(calibration: CalibrationArtifacts, config: Any) -> dict[int, float]:
    scores: dict[int, float] = {}
    for layer_idx in range(config.num_hidden_layers):
        combined = (calibration.mxmoe_w1_deltas[layer_idx].to(torch.float32) + calibration.mxmoe_w2_deltas[layer_idx].to(torch.float32))
        active_mask = calibration.routing_counts[layer_idx] > 0
        scores[layer_idx] = float(combined[active_mask].sum().item()) if bool(active_mask.any()) else 0.0
    return scores


def build_restore_specs(config: Any, top3_layers: list[int]) -> list[RestoreSpec]:
    if config.num_hidden_layers <= 39:
        raise RuntimeError(f"Expected at least 40 MoE layers for Iteration 23 sweep, found {config.num_hidden_layers}")
    return [
        RestoreSpec(
            name="restore_lm_head_bf16",
            description="Explicitly keep lm_head in BF16 while leaving the joint_w1w2_with_topup expert mask unchanged.",
            restore_bf16=frozenset({"lm_head"}),
            restore_moe_layers=frozenset(),
            effective_restore=False,
            reason="`lm_head` already loads as BF16 in proper_eval.evaluate_plan().",
        ),
        RestoreSpec(
            name="restore_embeddings_bf16",
            description="Explicitly keep token embeddings in BF16 while leaving the joint_w1w2_with_topup expert mask unchanged.",
            restore_bf16=frozenset({"embeddings"}),
            restore_moe_layers=frozenset(),
            effective_restore=False,
            reason="`embed_tokens.weight` already stays BF16 in proper_eval.embed_chunks().",
        ),
        RestoreSpec(
            name="restore_attention_bf16",
            description="Explicitly keep all self_attn and linear_attn weights in BF16; only MoE experts remain quantized by the joint mask.",
            restore_bf16=frozenset({"attention"}),
            restore_moe_layers=frozenset(),
            effective_restore=False,
            reason="Attention weights are already BF16 because prepare_layer_tensors_for_plan() only quantizes mlp.experts.* tensors.",
        ),
        RestoreSpec(
            name="restore_shared_expert_bf16",
            description="Explicitly keep the shared expert in BF16 while leaving routed-expert quantization unchanged.",
            restore_bf16=frozenset({"shared_expert"}),
            restore_moe_layers=frozenset(),
            effective_restore=False,
            reason="The shared expert already stays BF16 inside moe_forward_eval().",
        ),
        RestoreSpec(
            name="restore_layer0_moe_bf16",
            description="Restore all routed MoE experts in layer 0 to BF16 and keep the remaining layers on the joint_w1w2_with_topup mask.",
            restore_bf16=frozenset(),
            restore_moe_layers=frozenset({0}),
            effective_restore=True,
            reason="Layer 0 routed MoE experts skip quantization in prepare_layer_tensors_for_plan().",
        ),
        RestoreSpec(
            name="restore_layer19_moe_bf16",
            description="Restore all routed MoE experts in layer 19 to BF16 and keep the remaining layers on the joint_w1w2_with_topup mask.",
            restore_bf16=frozenset(),
            restore_moe_layers=frozenset({19}),
            effective_restore=True,
            reason="Layer 19 routed MoE experts skip quantization in prepare_layer_tensors_for_plan().",
        ),
        RestoreSpec(
            name="restore_layer39_moe_bf16",
            description="Restore all routed MoE experts in layer 39 to BF16 and keep the remaining layers on the joint_w1w2_with_topup mask.",
            restore_bf16=frozenset(),
            restore_moe_layers=frozenset({39}),
            effective_restore=True,
            reason="Layer 39 routed MoE experts skip quantization in prepare_layer_tensors_for_plan().",
        ),
        RestoreSpec(
            name="restore_top3_layers_moe_bf16",
            description="Restore the three MoE layers with the largest summed output perturbation to BF16 and keep the remaining layers on the joint_w1w2_with_topup mask.",
            restore_bf16=frozenset(),
            restore_moe_layers=frozenset(top3_layers),
            effective_restore=True,
            reason="The top-3 MoE layers are ranked by summed active-expert output perturbation from calibration.mxmoe_w1_deltas + calibration.mxmoe_w2_deltas.",
        ),
    ]


def build_restore_plan(
    base_plan: EvalPlan,
    spec: RestoreSpec,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[EvalPlan, dict[str, Any]]:
    w1_masks = base_plan.w1_pair_masks
    w2_masks = base_plan.w2_channel_masks
    if w1_masks is None or w2_masks is None:
        raise RuntimeError("joint_w1w2_with_topup base plan is missing per-channel masks")

    restored_bf16_weights = 0
    plan_w1_masks = w1_masks
    plan_w2_masks = w2_masks
    if spec.restore_moe_layers:
        restored_bf16_weights = len(spec.restore_moe_layers) * config.num_experts * expert_full_weight_count(config)
        plan_w1_masks = zero_moe_layers(w1_masks, spec.restore_moe_layers)
        plan_w2_masks = zero_moe_layers(w2_masks, spec.restore_moe_layers)

    fp8_weights = fp8_weights_from_masks(config, plan_w1_masks, plan_w2_masks)
    memory_gb = estimate_restored_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights, restored_bf16_weights)
    plan = replace(
        base_plan,
        name=spec.name,
        description=spec.description,
        memory_gb=memory_gb,
        fp8_weights=fp8_weights,
        w1_pair_masks=plan_w1_masks,
        w2_channel_masks=plan_w2_masks,
        restore_bf16=spec.restore_bf16,
        restore_moe_layers=spec.restore_moe_layers,
    )
    return plan, {
        "restored_components": sorted(spec.restore_bf16),
        "restored_moe_layers": sorted(int(layer_idx) for layer_idx in spec.restore_moe_layers),
        "effective_restore": bool(spec.effective_restore),
        "restore_note": spec.reason,
        "restored_bf16_weights": int(restored_bf16_weights),
        "restored_bf16_fraction": round(float(restored_bf16_weights) / float(max(total_expert_elems, 1)), 6),
    }


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
        f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | restored={row['restored_bf16_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def update_summary(
    payload: dict[str, Any],
    base_plan_name: str,
    layer_scores: dict[int, float],
) -> None:
    results = payload.get("results", {})
    if base_plan_name not in results:
        return

    base_ppl = float(results[base_plan_name]["ppl"])
    gain_rows: list[dict[str, Any]] = []
    for name, row in results.items():
        ppl_gain = round(base_ppl - float(row["ppl"]), 6)
        row["ppl_gain_vs_base"] = ppl_gain
        row["ppl_delta_vs_base"] = round(float(row["ppl"]) - base_ppl, 6)
        if name == base_plan_name:
            continue
        gain_rows.append({
            "name": name,
            "ppl": float(row["ppl"]),
            "memory_gb": float(row["memory_gb"]),
            "ppl_gain_vs_base": ppl_gain,
            "effective_restore": bool(row.get("effective_restore", False)),
            "restored_components": list(row.get("restored_components", [])),
            "restored_moe_layers": list(row.get("restored_moe_layers", [])),
        })

    gain_rows.sort(
        key=lambda row: (
            float(row["ppl_gain_vs_base"]),
            -float(row["memory_gb"]),
            row["name"],
        ),
        reverse=True,
    )
    best_restore = None if not gain_rows else gain_rows[0]
    payload["summary"] = {
        "base_plan": base_plan_name,
        "base_ppl": base_ppl,
        "top3_layers_by_output_perturbation": [
            {"layer": int(layer_idx), "score": round(float(score), 6)}
            for layer_idx, score in sorted(layer_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:3]
        ],
        "ranked_ppl_gains": gain_rows,
        "best_restore": best_restore,
    }


def print_results_table(payload: dict[str, Any], base_plan_name: str) -> None:
    results = payload.get("results", {})
    if base_plan_name not in results:
        return
    base_row = results[base_plan_name]
    ordered_names = [base_plan_name] + [
        row["name"]
        for row in payload.get("summary", {}).get("ranked_ppl_gains", [])
        if row["name"] in results
    ]
    seen: set[str] = set()

    print("\n" + "=" * 164, flush=True)
    print("proper_iter23_subsystem_sweep | BF16 restoration sweep on joint_w1w2_with_topup | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 164, flush=True)
    print(
        f"{'Config':<32} {'PPL':>10} {'Gain':>10} {'Delta':>10} {'Memory GB':>12} {'FP8 frac':>10} {'BF16 restore':>13} {'Effective':>10}",
        flush=True,
    )
    print("-" * 164, flush=True)
    for name in ordered_names:
        if name in seen:
            continue
        seen.add(name)
        row = results[name]
        print(
            f"{name:<32} {float(row['ppl']):>10.4f} {float(row.get('ppl_gain_vs_base', 0.0)):>+10.4f} {float(row.get('ppl_delta_vs_base', 0.0)):>+10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row.get('restored_bf16_fraction', 0.0)):>13.4f} {str(bool(row.get('effective_restore', False))):>10}",
            flush=True,
        )

    print("-" * 164, flush=True)
    print(f"Base reference: {base_plan_name} -> PPL {float(base_row['ppl']):.6f}", flush=True)
    best_restore = payload.get("summary", {}).get("best_restore")
    if best_restore is not None:
        print(
            f"Best restore by gain: {best_restore['name']} -> gain {float(best_restore['ppl_gain_vs_base']):+.6f}, PPL {float(best_restore['ppl']):.6f}",
            flush=True,
        )


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
    _tokenizer, _calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
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

    layer_scores = build_layer_output_perturbation_scores(calibration, text_config)
    top3_layers = [
        int(layer_idx)
        for layer_idx, _score in sorted(layer_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:3]
    ]
    base_plan, base_extras = build_base_plan(calibration, text_config, non_expert_bytes, total_expert_elems)
    restore_specs = build_restore_specs(text_config, top3_layers)

    references = load_reference_rows(args.output_json)
    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: MoE expert weights quantize -> dequantize -> BF16 -> F.linear; embeddings, attention, shared expert, and lm_head remain BF16 unless future infra changes.",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "cache_path": str(args.cache_path),
            "references": references,
            "experiment": "Iteration 23 subsystem BF16 restoration sweep on top of joint_w1w2_with_topup",
            "base_plan": base_plan.name,
            "base_plan_reference": references.get(base_plan.name),
            "bf16_reference": references.get("bf16"),
            "joint_topup_fraction": float(JOINT_MEDIUM_TOPUP_FRACTION),
            "bf16_component_confirmation": {
                "lm_head": "already BF16 in proper_eval.evaluate_plan()",
                "embeddings": "already BF16 in proper_eval.embed_chunks()",
                "attention": "already BF16 because prepare_layer_tensors_for_plan() only quantizes mlp.experts.* tensors",
                "shared_expert": "already BF16 in proper_eval.moe_forward_eval()",
            },
            "top3_layers_by_output_perturbation": [
                {"layer": int(layer_idx), "score": round(float(score), 6)}
                for layer_idx, score in sorted(layer_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:3]
            ],
        },
        "results": {},
    }
    if args.output_json.exists():
        payload = load_json(args.output_json)
        payload.setdefault("metadata", {}).update({
            "top3_layers_by_output_perturbation": [
                {"layer": int(layer_idx), "score": round(float(score), 6)}
                for layer_idx, score in sorted(layer_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:3]
            ],
            "base_plan": base_plan.name,
        })
        payload.setdefault("results", {})

    plan_builders: list[tuple[EvalPlan, dict[str, Any]]] = [(base_plan, base_extras)]
    for spec in restore_specs:
        plan_builders.append(build_restore_plan(base_plan, spec, text_config, non_expert_bytes, total_expert_elems))

    for plan, extras in plan_builders:
        if requested_plans is not None and plan.name not in requested_plans:
            continue
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

    update_summary(payload, base_plan.name, layer_scores)
    payload.setdefault("metadata", {})["elapsed_s"] = round(time.time() - overall_start, 1)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload, base_plan.name)


if __name__ == "__main__":
    main()

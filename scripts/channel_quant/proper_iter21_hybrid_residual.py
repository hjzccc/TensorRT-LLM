#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 21: hybrid residual channels on the strongest joint/union bases.

Build the strongest existing base masks first, then add FP4 residual channels only
on the channels that still remain FP4 after those bases. This keeps the existing
FP8 selections intact and measures whether residual correction works better when
stacked on top of stronger scaffolds than the earlier MxMoE-only and plain joint
residual runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines_comparison import LayerMetricBundle, resolve_non_expert_bytes, resolve_terminal_keys, topk_mask_from_scores
from proper_eval import (
    LOGITS_CHUNK_TOKENS,
    SEQLEN,
    CalibrationArtifacts,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    layer_type_at,
    load_gptq_standard_data,
    moe_forward_eval,
    upsert_exploration_section,
)
from proper_iter01 import build_empty_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter10_novel_perchannel import (
    compute_novel_metric_artifacts,
    load_metric_cache as load_router_affinity_metric_cache,
    save_metric_cache as save_router_affinity_metric_cache,
)
from proper_iter14 import build_union_base_router_affinity_topup_masks
from proper_iter15_residual_channels import (
    ResidualEvalPlan,
    estimate_residual_memory_gb,
    masked_weights_from_masks,
    prepare_layer_tensors_for_residual_plan,
    should_log_chunk,
)
from proper_iter20_router_preserve import (
    FRAGILE_QUANTILE,
    GAP_EPS,
    compute_gap_aware_metric_artifacts,
    load_metric_cache as load_gap_metric_cache,
    save_metric_cache as save_gap_metric_cache,
)
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter21_hybrid_residual.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
DEFAULT_GAP_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter20_router_preserve_metric_cache.pt"
SECTION_MARKER = "## [24] Iteration 21 - Hybrid Residual on Strongest Bases"
UNION_BASE_TOPUP_FRACTION = 0.05
ROUTERGAP_PLAN_NAME = "routergap_union_residual_3pct"
SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "joint_topup_residual_3pct": 6.5729,
    "routergap_union_topup_5pct": 6.5734,
    "output_perturbation_mxmoe_topup_router_affinity_5pct": 6.5758,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--gap-metric-cache-path", type=Path, default=DEFAULT_GAP_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 21 configs.",
    )
    parser.add_argument("--gap-eps", type=float, default=GAP_EPS)
    parser.add_argument("--fragile-quantile", type=float, default=FRAGILE_QUANTILE)
    parser.add_argument("--force-recompute-metrics", action="store_true")
    parser.add_argument("--force-recompute-gap-metrics", action="store_true")
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


def should_build_routergap_plan(requested_plans: set[str] | None) -> bool:
    return requested_plans is None or ROUTERGAP_PLAN_NAME in requested_plans


def select_complement_topk(scores: torch.Tensor, blocked_mask: torch.Tensor, k: int) -> torch.Tensor:
    blocked = blocked_mask.detach().cpu().to(torch.bool)
    result = torch.zeros_like(blocked, dtype=torch.bool)
    remaining = int((~blocked).sum().item())
    if k <= 0 or remaining <= 0:
        return result
    masked_scores = scores.detach().cpu().clone().to(torch.float32)
    masked_scores[blocked] = torch.finfo(masked_scores.dtype).min
    result = topk_mask_from_scores(masked_scores, min(k, remaining))
    return torch.logical_and(result, ~blocked)


def build_residual_masks_from_base_masks(
    metric_cache: dict[int, LayerMetricBundle],
    base_w1_masks: dict[int, dict[int, torch.Tensor]],
    base_w2_masks: dict[int, dict[int, torch.Tensor]],
    config: Any,
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
            if w1_residual_pairs > 0:
                w1_residual_pair_masks[layer_idx][expert_idx] = select_complement_topk(
                    bundle.w1_pair_scores[expert_idx],
                    base_w1_masks[layer_idx][expert_idx],
                    w1_residual_pairs,
                )
            if w2_residual_channels > 0:
                w2_residual_channel_masks[layer_idx][expert_idx] = select_complement_topk(
                    bundle.w2_channel_scores[expert_idx],
                    base_w2_masks[layer_idx][expert_idx],
                    w2_residual_channels,
                )

    return w1_residual_pair_masks, w2_residual_channel_masks, {
        "budget_source": budget_source,
        "residual_selection_scope": "remaining_fp4_only",
        "residual_fraction": float(residual_fraction),
        "w1_residual_fraction": float(residual_fraction),
        "w2_residual_fraction": float(residual_fraction),
    }


def build_masked_base_residual_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    base_w1_masks: dict[int, dict[int, torch.Tensor]],
    base_w2_masks: dict[int, dict[int, torch.Tensor]],
    w1_residual_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_residual_channel_masks: dict[int, dict[int, torch.Tensor]],
) -> ResidualEvalPlan:
    fp8_weights = masked_weights_from_masks(config, base_w1_masks, base_w2_masks)
    residual_weights = masked_weights_from_masks(config, w1_residual_pair_masks, w2_residual_channel_masks)
    return ResidualEvalPlan(
        name=name,
        description=description,
        mode="residual_per_channel",
        memory_gb=estimate_residual_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights, residual_weights),
        fp8_weights=fp8_weights,
        residual_weights=residual_weights,
        w1_projection_fp8=None,
        w2_projection_fp8=None,
        w1_fp8_pair_masks=base_w1_masks,
        w2_fp8_channel_masks=base_w2_masks,
        w1_residual_pair_masks=w1_residual_pair_masks,
        w2_residual_channel_masks=w2_residual_channel_masks,
    )


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    residual_metric_cache: dict[int, LayerMetricBundle],
    residual_metric_meta: dict[str, Any],
    union_base_metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    gap_metric_cache: dict[int, LayerMetricBundle] | None,
) -> list[tuple[ResidualEvalPlan, dict[str, Any]]]:
    plans: list[tuple[ResidualEvalPlan, dict[str, Any]]] = []

    joint_w1_masks, joint_w2_masks, joint_base_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    joint_base_extras = {
        **joint_base_meta,
        "base_assignment": "joint_w1w2_with_topup",
        "source_base": "joint_w1w2_with_topup",
        "base_channel_metric_requested": "activation_kurtosis",
        "base_channel_metric_effective": "activation_kurtosis",
        "base_channel_metric_fallback_used": False,
    }
    for fraction in (0.02, 0.04):
        pct = int(round(fraction * 100))
        w1_residual_masks, w2_residual_masks, residual_meta = build_residual_masks_from_base_masks(
            residual_metric_cache,
            joint_w1_masks,
            joint_w2_masks,
            config,
            residual_fraction=fraction,
            budget_source="joint_w1w2_with_topup_plus_residual",
        )
        plans.append((
            build_masked_base_residual_plan(
                f"joint_topup_residual_{pct}pct",
                f"Start from the Iteration 7 `joint_w1w2_with_topup` base, then add FP4 residual top-up to the top {pct}% residual-ranked W1 pairs and W2 channels only on the channels that still remain FP4.",
                config,
                non_expert_bytes,
                total_expert_elems,
                joint_w1_masks,
                joint_w2_masks,
                w1_residual_masks,
                w2_residual_masks,
            ),
            {
                **residual_metric_meta,
                **joint_base_extras,
                **residual_meta,
            },
        ))

    union_w1_masks, union_w2_masks, union_base_meta = build_union_base_router_affinity_topup_masks(
        calibration,
        union_base_metric_cache,
        config,
        total_expert_elems,
        topup_fraction=UNION_BASE_TOPUP_FRACTION,
    )
    union_base_extras = {
        **union_base_meta,
        "base_assignment": "router_affinity_union_topup_5pct",
        "source_base": "output_perturbation_mxmoe_topup_router_affinity_5pct",
        "base_channel_metric_requested": "router_affinity_weighted_qerror",
        "base_channel_metric_effective": "router_affinity_weighted_qerror",
        "base_channel_metric_fallback_used": False,
    }
    for fraction in (0.02, 0.03, 0.05):
        pct = int(round(fraction * 100))
        w1_residual_masks, w2_residual_masks, residual_meta = build_residual_masks_from_base_masks(
            residual_metric_cache,
            union_w1_masks,
            union_w2_masks,
            config,
            residual_fraction=fraction,
            budget_source="router_affinity_union_topup_5pct_plus_residual",
        )
        plans.append((
            build_masked_base_residual_plan(
                f"union_topup_residual_{pct}pct",
                f"Start from the Iteration 14 union base with 5% router-affinity topups, then add FP4 residual top-up to the top {pct}% residual-ranked W1 pairs and W2 channels only on the channels that still remain FP4.",
                config,
                non_expert_bytes,
                total_expert_elems,
                union_w1_masks,
                union_w2_masks,
                w1_residual_masks,
                w2_residual_masks,
            ),
            {
                **residual_metric_meta,
                **union_base_extras,
                **residual_meta,
            },
        ))

    if gap_metric_cache is not None:
        w1_residual_masks, w2_residual_masks, residual_meta = build_residual_masks_from_base_masks(
            gap_metric_cache,
            union_w1_masks,
            union_w2_masks,
            config,
            residual_fraction=0.03,
            budget_source="routergap_union_topup_5pct_plus_residual",
        )
        plans.append((
            build_masked_base_residual_plan(
                ROUTERGAP_PLAN_NAME,
                "Start from the same 5% union base, but rank the residual channels with inverse-gap router-affinity so the residual budget only patches channels that are both still FP4 and fragile under router-boundary decisions.",
                config,
                non_expert_bytes,
                total_expert_elems,
                union_w1_masks,
                union_w2_masks,
                w1_residual_masks,
                w2_residual_masks,
            ),
            {
                "channel_metric_requested": "router_gap_affinity_weighted_qerror",
                "channel_metric_effective": "router_gap_affinity_weighted_qerror",
                "channel_metric_fallback_used": False,
                "routing_count_metric": "inverse_router_gap_weighted_topk_count",
                **union_base_extras,
                **residual_meta,
                "source_base": "routergap_union_topup_5pct",
            },
        ))
    else:
        w1_residual_masks, w2_residual_masks, residual_meta = build_residual_masks_from_base_masks(
            residual_metric_cache,
            joint_w1_masks,
            joint_w2_masks,
            config,
            residual_fraction=0.01,
            budget_source="joint_w1w2_with_topup_plus_residual_fallback",
        )
        plans.append((
            build_masked_base_residual_plan(
                "joint_topup_residual_1pct",
                "Fallback config when router-gap residual ranking is unavailable: add a 1% residual top-up on the remaining FP4 channels of `joint_w1w2_with_topup`.",
                config,
                non_expert_bytes,
                total_expert_elems,
                joint_w1_masks,
                joint_w2_masks,
                w1_residual_masks,
                w2_residual_masks,
            ),
            {
                **residual_metric_meta,
                **joint_base_extras,
                **residual_meta,
            },
        ))

    return plans


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
        "w1_fp8_pair_fraction": round(total_pair_fraction(config, plan.w1_fp8_pair_masks or {}), 6),
        "w2_fp8_channel_fraction": round(total_channel_fraction(config, plan.w2_fp8_channel_masks or {}), 6),
        "w1_residual_pair_fraction": round(total_pair_fraction(config, plan.w1_residual_pair_masks or {}), 6),
        "w2_residual_channel_fraction": round(total_channel_fraction(config, plan.w2_residual_channel_masks or {}), 6),
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


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    target_best = min(SUCCESS_TARGETS.values())
    print("\n" + "=" * 210, flush=True)
    print("proper_iter21_hybrid_residual | strongest-base residual topups | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 210, flush=True)
    print(
        f"{'Config':<36} {'PPL':>10} {'dTarget':>10} {'dBase':>10} {'Memory GB':>12} {'FP8 frac':>10} {'Residual':>10} {'W1 resid':>10} {'W2 resid':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 210, flush=True)
    for name, row in ordered:
        source_base = str(row.get("source_base", ""))
        source_ref = references.get(source_base, {}).get("ppl")
        d_base = "-" if source_ref is None else f"{float(row['ppl']) - source_ref:+.4f}"
        print(
            f"{name:<36} {float(row['ppl']):>10.4f} {float(row['ppl']) - target_best:>+10.4f} {d_base:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['residual_fraction']):>10.4f} {float(row['w1_residual_pair_fraction']):>10.4f} {float(row['w2_residual_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def format_markdown_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_prior = min((float(row["ppl"]) for row in references.values()), default=min(SUCCESS_TARGETS.values()))
    lines = [
        "| Config | PPL | Delta vs source base | Delta vs best prior | Memory GB | Residual frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in ordered:
        source_base = str(row.get("source_base", ""))
        source_ref = references.get(source_base, {}).get("ppl")
        source_delta = "-" if source_ref is None else f"{float(row['ppl']) - source_ref:+.4f}"
        lines.append(
            f"| `{name}` | {float(row['ppl']):.4f} | {source_delta} | {float(row['ppl']) - best_prior:+.4f} | {float(row['memory_gb']):.3f} | {float(row['residual_fraction']):.4f} |"
        )
    return "\n".join(lines)


def build_insight_text(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    winner_name, winner = ordered[0]
    target_best = min(SUCCESS_TARGETS.values())
    if float(winner["ppl"]) < target_best:
        return f"`{winner_name}` sets a new best quantized result at {float(winner['ppl']):.4f}, beating the standing 6.5725 target by {target_best - float(winner['ppl']):.4f} PPL."

    improvements: list[tuple[float, str, str]] = []
    for name, row in ordered:
        source_base = str(row.get("source_base", ""))
        source_ref = references.get(source_base, {}).get("ppl")
        if source_ref is None:
            continue
        delta = float(row["ppl"]) - source_ref
        improvements.append((delta, name, source_base))
    if improvements:
        best_delta, best_name, base_name = min(improvements, key=lambda item: item[0])
        if best_delta < 0.0:
            return f"Residual-on-base helps most for `{best_name}`, which beats its non-residual parent `{base_name}` by {-best_delta:.4f} PPL while preserving the base FP8 mask and only patching the remaining FP4 channels."
        return f"None of the hybrid residual runs beats its immediate non-residual parent, but `{winner_name}` is the closest overall at {float(winner['ppl']):.4f}, only {float(winner['ppl']) - target_best:+.4f} from the 6.5725 target."
    return f"`{winner_name}` is the best run at {float(winner['ppl']):.4f}; compare it against the existing 6.5725 target to decide whether stronger residual ranking or base selection is the next lever."


def render_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("render_exploration_section requires non-empty results")
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    winner_name, winner = ordered[0]
    insight = build_insight_text(results, references)
    gap_meta = payload.get("metadata", {}).get("router_gap_calibration")
    gap_text = ""
    if isinstance(gap_meta, dict):
        gap_text = (
            f" **Router-gap cache**: fragile fraction {float(gap_meta.get('total_fragile_fraction', 0.0)):.4f}, "
            f"runtime {float(gap_meta.get('runtime_seconds', 0.0)):.1f}s."
        )
    return (
        f"{SECTION_MARKER}\n"
        f"**Approach**: Reused the Iteration 15 residual-channel quantization path but switched the eval loop to the current GPTQ-standard `proper_eval.py` loss/logit protocol. The joint base now means the exact Iteration 7 `joint_w1w2_with_topup` mask, the union base means the Iteration 14 router-affinity union with 5% topup, and every residual plan selects channels only from the FP4 complement of those base masks so no existing FP8 selections are overwritten.{gap_text}\n"
        f"**Result**:\n{format_markdown_table(results, references)}\n"
        f"**Winner**: `{winner_name}` at PPL {float(winner['ppl']):.4f}, memory {float(winner['memory_gb']):.3f} GB, FP8 fraction {float(winner['fp8_fraction']):.4f}, residual fraction {float(winner['residual_fraction']):.4f}.\n"
        f"**Insight**: {insight}\n"
        f"**Next**: If the best hybrid residual run still misses 6.5725, the next clean ablation is to vary only the residual ranking cache on the same strongest base so the gain from base choice and the gain from residual channel ordering are no longer entangled."
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    requested_plans = resolve_requested_plans(args.plans)
    need_routergap = should_build_routergap_plan(requested_plans)
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
                "loss_accumulation": "CrossEntropyLoss(reduction='sum') chunked over logits, normalized per chunk, then accumulated as loss * seqlen",
            },
            "cache_path": str(args.cache_path),
            "metric_cache_path": str(args.metric_cache_path),
            "gap_metric_cache_path": str(args.gap_metric_cache_path),
            "references": references,
            "success_targets": SUCCESS_TARGETS,
            "experiment": "Iteration 21 hybrid residual channels on top of the strongest joint/union bases without overwriting existing FP8 selections",
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

    router_affinity_cache, _hessian_normalized_cache, _micromix_mean_abs, _micromix_thresholds, router_metric_meta = (None, None, None, None, None)
    if not args.force_recompute_metrics:
        router_affinity_cache, _hessian_normalized_cache, _micromix_mean_abs, _micromix_thresholds, router_metric_meta = load_router_affinity_metric_cache(
            args.metric_cache_path,
            args.model_id,
        )
        if router_affinity_cache is not None and router_metric_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if router_affinity_cache is None or router_metric_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, router_metric_meta = compute_novel_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
        )
        save_router_affinity_metric_cache(
            args.metric_cache_path,
            args.model_id,
            router_affinity_cache,
            hessian_normalized_cache,
            micromix_mean_abs,
            micromix_thresholds,
            router_metric_meta,
        )
        del store
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    residual_metric_cache = router_affinity_cache if router_affinity_cache is not None else calibration.activation_cache
    residual_metric_meta = {
        "channel_metric_requested": "router_affinity_weighted_qerror" if router_affinity_cache is not None else "activation_kurtosis",
        "channel_metric_effective": "router_affinity_weighted_qerror" if router_affinity_cache is not None else "activation_kurtosis",
        "channel_metric_fallback_used": bool(router_affinity_cache is None),
    }
    payload["metadata"]["metric_cache"] = router_metric_meta
    payload["metadata"]["residual_ranking_metric"] = residual_metric_meta

    gap_metric_cache: dict[int, LayerMetricBundle] | None = None
    gap_meta: dict[str, Any] | None = None
    if need_routergap:
        gap_weighted_counts = None
        if not args.force_recompute_gap_metrics:
            gap_metric_cache, gap_weighted_counts, gap_meta = load_gap_metric_cache(
                args.gap_metric_cache_path,
                args.model_id,
                args.gap_eps,
                args.fragile_quantile,
            )
            if gap_metric_cache is not None and gap_weighted_counts is not None and gap_meta is not None:
                print(f"[gap-metric-cache] loaded {args.gap_metric_cache_path}", flush=True)

        if gap_metric_cache is None or gap_meta is None:
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
            save_gap_metric_cache(
                args.gap_metric_cache_path,
                args.model_id,
                args.gap_eps,
                args.fragile_quantile,
                gap_metric_cache,
                gap_weighted_counts,
                gap_meta,
            )
            del store
            print(f"[gap-metric-cache] saved {args.gap_metric_cache_path}", flush=True)
        payload["metadata"]["router_gap_calibration"] = gap_meta

    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(
        calibration,
        residual_metric_cache,
        residual_metric_meta,
        router_affinity_cache if router_affinity_cache is not None else calibration.activation_cache,
        text_config,
        non_expert_bytes,
        total_expert_elems,
        gap_metric_cache,
    )
    if requested_plans is not None:
        requested_set = set(requested_plans)
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_set]
        missing = sorted(requested_set - {plan.name for plan, _extras in plans})
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")

    print(f"[plans] evaluating {len(plans)} hybrid residual configs", flush=True)
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
    section = render_exploration_section(payload, references)
    upsert_exploration_section(args.exploration_md, section)
    print_results_table(payload["results"], references)
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

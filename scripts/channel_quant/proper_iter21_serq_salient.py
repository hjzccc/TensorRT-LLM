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

from baselines_comparison import LayerMetricBundle, fp8_weights_from_masks, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
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
from proper_iter01 import TOPUP_FRACTION as MXMOE_TOPUP_FRACTION
from proper_iter01 import build_mxmoe_topup_masks, load_cache, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter10_novel_perchannel import (
    compute_novel_metric_artifacts,
    load_metric_cache as load_novel_metric_cache,
    save_metric_cache as save_novel_metric_cache,
)
from proper_iter14 import build_union_base_router_affinity_topup_masks
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter21_serq_salient.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
SECTION_MARKER = "## [24] Iteration 21 - SERQ Salient Correction"

SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "routergap_union_topup_5pct": 6.5734,
    "output_perturbation_mxmoe_topup_router_affinity_5pct": 6.5758,
    "mxmoe_block_plus_channel_topup": 6.5763,
}


@dataclass(frozen=True)
class SerqEvalPlan:
    name: str
    description: str
    mode: str
    memory_gb: float
    fp8_weights: int
    exact_correction_weights: int
    base_w1_pair_masks: dict[int, dict[int, torch.Tensor]]
    base_w2_channel_masks: dict[int, dict[int, torch.Tensor]]
    correction_w1_pair_masks: dict[int, dict[int, torch.Tensor]]
    correction_w2_channel_masks: dict[int, dict[int, torch.Tensor]]


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 21 SERQ configs.",
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


def masked_weights_from_masks(
    config: Any,
    w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_channel_masks: dict[int, dict[int, torch.Tensor]],
) -> int:
    w1_pairs = sum(int(mask.sum().item()) for layer_masks in w1_pair_masks.values() for mask in layer_masks.values())
    w2_channels = sum(int(mask.sum().item()) for layer_masks in w2_channel_masks.values() for mask in layer_masks.values())
    return int((w1_pairs * 2 * config.hidden_size) + (w2_channels * config.moe_intermediate_size))


def estimate_serq_memory_gb(non_expert_bytes: int, total_expert_elems: int, fp8_weights: int, correction_weights: int) -> float:
    expert_bytes = int((0.5 * total_expert_elems) + (0.5 * fp8_weights) + (2.0 * correction_weights))
    return round(float(non_expert_bytes + expert_bytes) / 1e9, 3)


def pair_mask_to_channel_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    pair_mask_bool = pair_mask.to(torch.bool)
    return torch.cat([pair_mask_bool, pair_mask_bool], dim=0)


def should_log_chunk(chunk_idx: int, total_chunks: int) -> bool:
    return chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == total_chunks


def build_flat_saliency_scores(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    projection: str,
    base_masks: dict[int, dict[int, torch.Tensor]],
) -> tuple[torch.Tensor, int]:
    flat_chunks: list[torch.Tensor] = []
    units_per_expert = int(config.moe_intermediate_size if projection == "w1" else config.hidden_size)
    total_eligible = 0
    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        layer_scores = (
            bundle.w1_pair_scores if projection == "w1" else bundle.w2_channel_scores
        ).detach().cpu().clone().to(torch.float32)
        active_mask = (bundle.routing_counts > 0).view(config.num_experts, 1)
        layer_scores[~active_mask.expand(-1, units_per_expert)] = -torch.inf
        base_layer_mask = torch.stack(
            [base_masks[layer_idx][expert_idx].detach().cpu().to(torch.bool) for expert_idx in range(config.num_experts)],
            dim=0,
        )
        layer_scores[base_layer_mask] = -torch.inf
        total_eligible += int(torch.isfinite(layer_scores).sum().item())
        flat_chunks.append(layer_scores.reshape(-1))
    return torch.cat(flat_chunks, dim=0), total_eligible


def unravel_flat_mask(
    flat_mask: torch.Tensor,
    config: Any,
    units_per_expert: int,
) -> dict[int, dict[int, torch.Tensor]]:
    masks: dict[int, dict[int, torch.Tensor]] = {}
    units_per_layer = int(config.num_experts * units_per_expert)
    offset = 0
    for layer_idx in range(config.num_hidden_layers):
        layer_flat = flat_mask[offset : offset + units_per_layer].view(config.num_experts, units_per_expert)
        masks[layer_idx] = {
            expert_idx: layer_flat[expert_idx].clone()
            for expert_idx in range(config.num_experts)
        }
        offset += units_per_layer
    return masks


def build_global_salient_correction_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    base_w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    base_w2_channel_masks: dict[int, dict[int, torch.Tensor]],
    correction_fraction: float,
    budget_source: str,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    total_w1_pairs = int(round(correction_fraction * config.num_hidden_layers * config.num_experts * config.moe_intermediate_size))
    total_w2_channels = int(round(correction_fraction * config.num_hidden_layers * config.num_experts * config.hidden_size))

    flat_w1_scores, eligible_w1 = build_flat_saliency_scores(metric_cache, config, "w1", base_w1_pair_masks)
    flat_w2_scores, eligible_w2 = build_flat_saliency_scores(metric_cache, config, "w2", base_w2_channel_masks)

    realized_w1 = min(total_w1_pairs, eligible_w1)
    realized_w2 = min(total_w2_channels, eligible_w2)

    flat_w1_mask = torch.zeros_like(flat_w1_scores, dtype=torch.bool)
    flat_w2_mask = torch.zeros_like(flat_w2_scores, dtype=torch.bool)
    if realized_w1 > 0:
        flat_w1_mask[torch.topk(flat_w1_scores, k=realized_w1).indices] = True
    if realized_w2 > 0:
        flat_w2_mask[torch.topk(flat_w2_scores, k=realized_w2).indices] = True

    w1_masks = unravel_flat_mask(flat_w1_mask, config, int(config.moe_intermediate_size))
    w2_masks = unravel_flat_mask(flat_w2_mask, config, int(config.hidden_size))
    return w1_masks, w2_masks, {
        "budget_source": budget_source,
        "saliency_metric_requested": "router_affinity_weighted_qerror",
        "saliency_metric_effective": "router_affinity_weighted_qerror",
        "serq_correction_fraction": float(correction_fraction),
        "w1_serq_correction_fraction": float(correction_fraction),
        "w2_serq_correction_fraction": float(correction_fraction),
        "requested_w1_correction_pairs": int(total_w1_pairs),
        "requested_w2_correction_channels": int(total_w2_channels),
        "eligible_w1_pairs": int(eligible_w1),
        "eligible_w2_channels": int(eligible_w2),
        "realized_w1_correction_pairs": int(realized_w1),
        "realized_w2_correction_channels": int(realized_w2),
    }


def build_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    base_w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    base_w2_channel_masks: dict[int, dict[int, torch.Tensor]],
    correction_w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    correction_w2_channel_masks: dict[int, dict[int, torch.Tensor]],
) -> SerqEvalPlan:
    fp8_weights = fp8_weights_from_masks(config, base_w1_pair_masks, base_w2_channel_masks)
    exact_correction_weights = masked_weights_from_masks(config, correction_w1_pair_masks, correction_w2_channel_masks)
    return SerqEvalPlan(
        name=name,
        description=description,
        mode="serq_salient_exact_residual",
        memory_gb=estimate_serq_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights, exact_correction_weights),
        fp8_weights=fp8_weights,
        exact_correction_weights=exact_correction_weights,
        base_w1_pair_masks=base_w1_pair_masks,
        base_w2_channel_masks=base_w2_channel_masks,
        correction_w1_pair_masks=correction_w1_pair_masks,
        correction_w2_channel_masks=correction_w2_channel_masks,
    )


def quantize_with_fp8_and_exact_correction(weight: torch.Tensor, fp8_mask: torch.Tensor, correction_mask: torch.Tensor) -> torch.Tensor:
    if int(fp8_mask.numel()) != int(weight.shape[0]) or int(correction_mask.numel()) != int(weight.shape[0]):
        raise ValueError("Mask lengths must match weight output dimension")
    if bool(torch.logical_and(fp8_mask, correction_mask).any()):
        raise ValueError("FP8 and exact-correction masks must be disjoint")
    weight_t = weight.transpose(0, 1).contiguous()
    fp4_t = quantize_to_nvfp4_columns(weight_t)
    if bool(fp8_mask.any()):
        fp8_t = quantize_to_fp8(weight_t)
        base_t = torch.where(fp8_mask.to(device=weight.device, dtype=torch.bool).view(1, -1), fp8_t, fp4_t)
    else:
        base_t = fp4_t
    if bool(correction_mask.any()):
        correction_scale = correction_mask.to(device=weight.device, dtype=weight_t.dtype).view(1, -1)
        base_t = base_t + ((weight_t - base_t) * correction_scale)
    return base_t.transpose(0, 1).contiguous()


def quantize_gate_up_with_exact_correction(weight: torch.Tensor, fp8_pair_mask: torch.Tensor, correction_pair_mask: torch.Tensor) -> torch.Tensor:
    if int(fp8_pair_mask.numel() * 2) != int(weight.shape[0]):
        raise ValueError(f"Expected pair mask length {weight.shape[0] // 2}, got {fp8_pair_mask.numel()}")
    return quantize_with_fp8_and_exact_correction(
        weight,
        pair_mask_to_channel_mask(fp8_pair_mask),
        pair_mask_to_channel_mask(correction_pair_mask),
    )


def prepare_layer_tensors_for_serq_plan(plan: SerqEvalPlan, layer_idx: int, tensors: dict[str, torch.Tensor], config: Any) -> None:
    gate_up_proj = tensors["mlp.experts.gate_up_proj"]
    down_proj = tensors["mlp.experts.down_proj"]
    prepared_gate_up: list[torch.Tensor] = []
    prepared_down: list[torch.Tensor] = []

    for expert_idx in range(config.num_experts):
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]
        base_w1_pair_mask = plan.base_w1_pair_masks[layer_idx][expert_idx].to(device=gate_up_weight.device, dtype=torch.bool)
        base_w2_channel_mask = plan.base_w2_channel_masks[layer_idx][expert_idx].to(device=down_weight.device, dtype=torch.bool)
        correction_w1_pair_mask = plan.correction_w1_pair_masks[layer_idx][expert_idx].to(device=gate_up_weight.device, dtype=torch.bool)
        correction_w2_channel_mask = plan.correction_w2_channel_masks[layer_idx][expert_idx].to(device=down_weight.device, dtype=torch.bool)
        prepared_gate_up.append(
            quantize_gate_up_with_exact_correction(gate_up_weight, base_w1_pair_mask, correction_w1_pair_mask)
        )
        prepared_down.append(
            quantize_with_fp8_and_exact_correction(down_weight, base_w2_channel_mask, correction_w2_channel_mask)
        )

    tensors["mlp.experts.gate_up_proj"] = torch.stack(prepared_gate_up, dim=0)
    tensors["mlp.experts.down_proj"] = torch.stack(prepared_down, dim=0)
    del gate_up_proj, down_proj, prepared_gate_up, prepared_down
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate_plan_serq(
    plan: SerqEvalPlan,
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
        prepare_layer_tensors_for_serq_plan(plan, layer_idx, tensors, config)

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
    plan: SerqEvalPlan,
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
    ppl, nll, nsamples = evaluate_plan_serq(plan, test_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - start_time
    row = {
        "description": plan.description,
        "mode": plan.mode,
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "memory_gb": round(plan.memory_gb, 3),
        "fp8_weights": int(plan.fp8_weights),
        "fp8_fraction": round(float(plan.fp8_weights) / float(max(total_expert_elems, 1)), 6),
        "exact_correction_weights": int(plan.exact_correction_weights),
        "exact_correction_fraction": round(float(plan.exact_correction_weights) / float(max(total_expert_elems, 1)), 6),
        "base_w1_pair_fraction": round(total_pair_fraction(config, plan.base_w1_pair_masks), 6),
        "base_w2_channel_fraction": round(total_channel_fraction(config, plan.base_w2_channel_masks), 6),
        "w1_exact_correction_pair_fraction": round(total_pair_fraction(config, plan.correction_w1_pair_masks), 6),
        "w2_exact_correction_channel_fraction": round(total_channel_fraction(config, plan.correction_w2_channel_masks), 6),
        "eval_chunks": int(nsamples),
        "seqlen": SEQLEN,
        "time_s": round(elapsed, 1),
        **extras,
    }
    result_rows[plan.name] = row
    atomic_json_dump(output_json, payload)
    print(
        f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | exact_corr={row['exact_correction_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def build_iteration_plans(
    router_affinity_cache: dict[int, LayerMetricBundle],
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[SerqEvalPlan, dict[str, Any]]]:
    plans: list[tuple[SerqEvalPlan, dict[str, Any]]] = []

    joint_w1, joint_w2, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    union_w1, union_w2, union_meta = build_union_base_router_affinity_topup_masks(
        calibration,
        router_affinity_cache,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    mxmoe_w1, mxmoe_w2, mxmoe_meta = build_mxmoe_topup_masks(
        calibration,
        config,
        total_expert_elems,
        MXMOE_TOPUP_FRACTION,
    )

    base_specs = [
        (
            "joint",
            "joint_w1w2_with_topup",
            joint_w1,
            joint_w2,
            joint_meta,
            "Keep the Iteration 7 joint three-tier + 8% activation-topup base unchanged, then add exact residual back only on the globally most salient non-FP8 W1 pairs and W2 channels.",
        ),
        (
            "union",
            "output_perturbation_mxmoe_topup_router_affinity_5pct",
            union_w1,
            union_w2,
            union_meta,
            "Keep the Iteration 14 union output-perturbation/router-affinity base unchanged, then add exact residual back only on the globally most salient non-FP8 W1 pairs and W2 channels.",
        ),
        (
            "mxmoe",
            "mxmoe_block_plus_channel_topup",
            mxmoe_w1,
            mxmoe_w2,
            mxmoe_meta,
            "Keep the Iteration 1 MxMoE block-plus-channel-topup base unchanged, then add exact residual back only on the globally most salient non-FP8 W1 pairs and W2 channels.",
        ),
    ]

    for family_name, source_base, base_w1, base_w2, base_meta, base_desc in base_specs:
        for correction_fraction in (0.01, 0.03):
            pct = int(round(correction_fraction * 100))
            correction_w1, correction_w2, correction_meta = build_global_salient_correction_masks(
                router_affinity_cache,
                config,
                base_w1,
                base_w2,
                correction_fraction=correction_fraction,
                budget_source=f"serq_exact_residual_on_{family_name}_base",
            )
            plan_name = f"serq_{family_name}_salient_{pct}pct"
            plans.append((
                build_plan(
                    plan_name,
                    f"{base_desc} The SERQ correction budget is {pct}% of global W1 pairs and {pct}% of global W2 channels, ranked by Iteration 10 router-affinity scores.",
                    config,
                    non_expert_bytes,
                    total_expert_elems,
                    base_w1,
                    base_w2,
                    correction_w1,
                    correction_w2,
                ),
                {
                    **base_meta,
                    **correction_meta,
                    "source_base": source_base,
                },
            ))
    return plans


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    target_best = min(SUCCESS_TARGETS.values())
    print("\n" + "=" * 196, flush=True)
    print("proper_iter21_serq_salient | exact salient residual add-back | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 196, flush=True)
    print(
        f"{'Config':<30} {'PPL':>10} {'dTarget':>10} {'Memory GB':>12} {'FP8 frac':>10} {'Corr frac':>10} {'W1 corr':>10} {'W2 corr':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 196, flush=True)
    for name, row in ordered:
        print(
            f"{name:<30} {float(row['ppl']):>10.4f} {float(row['ppl']) - target_best:>+10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['exact_correction_fraction']):>10.4f} {float(row['w1_exact_correction_pair_fraction']):>10.4f} {float(row['w2_exact_correction_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def format_markdown_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), item[0]))
    best_prior = min((float(row["ppl"]) for row in references.values()), default=min(SUCCESS_TARGETS.values()))
    lines = [
        "| Config | PPL | Delta vs best prior | Memory GB | FP8 frac | Exact corr frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in ordered:
        lines.append(
            f"| `{name}` | {float(row['ppl']):.4f} | {float(row['ppl']) - best_prior:+.4f} | {float(row['memory_gb']):.3f} | {float(row['fp8_fraction']):.4f} | {float(row['exact_correction_fraction']):.4f} |"
        )
    return "\n".join(lines)


def render_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("render_exploration_section requires non-empty results")
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    winner_name, winner = ordered[0]
    return (
        f"{SECTION_MARKER}\n"
        f"**Approach**: Reused the strongest existing bases as the quantized scaffold, formed the quantization residual `R = W - Q(W)` after the base FP8/FP4 assignment, and then kept exact residual add-back only on a sparse global top-k of non-FP8 W1 pairs and W2 channels. This is a SERQ-style simulation in the standard BF16 eval path: base weights still use quantize-dequantize FP4/FP8, while the selected salient rows/channels receive an exact add-back term before the MoE `F.linear` calls. All salient masks are ranked by the Iteration 10 router-affinity per-channel cache so the correction budget targets high-impact routed channels rather than broad residual coverage.\n"
        f"**Eval**: Full WikiText-2 test ({payload['metadata']['evaluation']['total_tokens']} tokens, {payload['metadata']['evaluation']['nsamples']} chunks of {SEQLEN}), BF16 `F.linear`, FP32 loss.\n"
        f"**Result**:\n{format_markdown_table(results, references)}\n"
        f"**Winner**: `{winner_name}` at PPL {float(winner['ppl']):.4f}, memory {float(winner['memory_gb']):.3f} GB, FP8 fraction {float(winner['fp8_fraction']):.4f}, exact correction fraction {float(winner['exact_correction_fraction']):.4f}.\n"
        f"**Insight**: This isolates the algorithmic value of a very sparse exact residual path. If it helps, the gain comes from spending high-precision budget only where router-weighted quantization error is most consequential, rather than promoting whole projections or carrying a low-precision residual everywhere.\n"
        f"**Next**: If one of the 1% or 3% SERQ runs is competitive, the next obvious sweep is to keep the same exact-correction mechanism but compare router-affinity saliency against router-gap saliency on the same union base so the correction ranking and the base routing heuristic can be separated cleanly."
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
            "quantization": "simulated base quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; SERQ path adds exact residual back only on selected salient non-FP8 rows/channels; FP32 logits and loss",
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
            "experiment": "Iteration 21 SERQ-style exact residual correction on the most salient non-FP8 W1 pairs and W2 channels",
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

    router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = (None, None, None, None, None)
    if not args.force_recompute_metrics:
        router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = load_novel_metric_cache(
            args.metric_cache_path,
            args.model_id,
        )
        if router_affinity_cache is not None and metric_cache_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if router_affinity_cache is None or metric_cache_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        router_affinity_cache, hessian_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = compute_novel_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
        )
        save_novel_metric_cache(
            args.metric_cache_path,
            args.model_id,
            router_affinity_cache,
            hessian_cache,
            micromix_mean_abs,
            micromix_thresholds,
            metric_cache_meta,
        )
        del store
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache"] = metric_cache_meta
    payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(router_affinity_cache, calibration, text_config, non_expert_bytes, total_expert_elems)
    if requested_plans is not None:
        requested_set = set(requested_plans)
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_set]
        missing = sorted(requested_set - {plan.name for plan, _extras in plans})
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
    print(f"[plans] evaluating {len(plans)} SERQ configs", flush=True)

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

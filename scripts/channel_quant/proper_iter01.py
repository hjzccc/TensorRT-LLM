#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines_comparison import (
    LayerMetricBundle,
    allocate_weighted_counts,
    estimate_mixed_memory_gb,
    fp8_weights_from_masks,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
    topk_mask_from_scores,
)
from proper_eval import (
    CALIBRATION_SAMPLES,
    SEQLEN,
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    build_mxmoe_projection_promotions,
    build_position_context,
    build_two_level_masks,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    load_gptq_standard_data,
    run_calibration,
)
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter01.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
SECTION_MARKER = "## [19] Hybrid MxMoE-Perturbation + Per-Channel"

HOT_EXPERTS_PER_LAYER = 40
MIN_HOT_ROUTING_COUNT = 128
TOPUP_FRACTION = 0.05
PERTURB_SPLIT_W1 = 0.04
PERTURB_SPLIT_W2 = 0.16
GLOBAL_25_W1 = 0.25
GLOBAL_25_W2 = 0.25
CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--hot-experts-per-layer", type=int, default=HOT_EXPERTS_PER_LAYER)
    parser.add_argument("--min-hot-routing-count", type=int, default=MIN_HOT_ROUTING_COUNT)
    parser.add_argument("--force-recompute-channel-perturb", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quantize_linear_weight(weight: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bf16":
        return weight
    weight_t = weight.transpose(0, 1).contiguous()
    if mode == "fp4":
        quantized_t = quantize_to_nvfp4_columns(weight_t)
    elif mode == "fp8":
        return weight
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    return quantized_t.transpose(0, 1).contiguous()


def clone_tensor_map(source: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {int(layer_idx): tensor.detach().cpu().clone() for layer_idx, tensor in source.items()}


def clone_metric_cache(source: dict[int, LayerMetricBundle]) -> dict[int, LayerMetricBundle]:
    cloned: dict[int, LayerMetricBundle] = {}
    for layer_idx, bundle in source.items():
        cloned[int(layer_idx)] = LayerMetricBundle(
            routing_counts=bundle.routing_counts.detach().cpu().clone(),
            w1_pair_scores=bundle.w1_pair_scores.detach().cpu().clone(),
            w2_channel_scores=bundle.w2_channel_scores.detach().cpu().clone(),
        )
    return cloned


def cache_is_compatible(payload: dict[str, Any], model_id: str) -> bool:
    return (
        int(payload.get("cache_version", -1)) == CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
    )


def serialize_cache(
    model_id: str,
    calibration: CalibrationArtifacts,
    hot_experts: dict[int, list[int]] | None = None,
    hot_w2_scores: dict[int, dict[int, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "model_id": model_id,
        "seqlen": SEQLEN,
        "calibration_samples": CALIBRATION_SAMPLES,
        "base_calibration": {
            "routing_counts": clone_tensor_map(calibration.routing_counts),
            "activation_cache": {
                int(layer_idx): {
                    "routing_counts": bundle.routing_counts.detach().cpu().clone(),
                    "w1_pair_scores": bundle.w1_pair_scores.detach().cpu().clone(),
                    "w2_channel_scores": bundle.w2_channel_scores.detach().cpu().clone(),
                }
                for layer_idx, bundle in calibration.activation_cache.items()
            },
            "mxmoe_w1_deltas": clone_tensor_map(calibration.mxmoe_w1_deltas),
            "mxmoe_w2_deltas": clone_tensor_map(calibration.mxmoe_w2_deltas),
            "mc_moe_scores": clone_tensor_map(calibration.mc_moe_scores),
        },
        "hot_experts_per_layer": {} if hot_experts is None else {int(layer_idx): [int(expert_idx) for expert_idx in experts] for layer_idx, experts in hot_experts.items()},
        "hot_w2_channel_perturbation": {} if hot_w2_scores is None else {
            int(layer_idx): {int(expert_idx): tensor.detach().cpu().clone() for expert_idx, tensor in expert_map.items()}
            for layer_idx, expert_map in hot_w2_scores.items()
        },
    }


def restore_calibration(payload: dict[str, Any]) -> CalibrationArtifacts:
    base = payload["base_calibration"]
    activation_cache: dict[int, LayerMetricBundle] = {}
    for raw_layer_idx, layer_payload in base["activation_cache"].items():
        layer_idx = int(raw_layer_idx)
        activation_cache[layer_idx] = LayerMetricBundle(
            routing_counts=layer_payload["routing_counts"].detach().cpu().clone(),
            w1_pair_scores=layer_payload["w1_pair_scores"].detach().cpu().clone(),
            w2_channel_scores=layer_payload["w2_channel_scores"].detach().cpu().clone(),
        )
    return CalibrationArtifacts(
        routing_counts={int(layer_idx): tensor.detach().cpu().clone() for layer_idx, tensor in base["routing_counts"].items()},
        activation_cache=activation_cache,
        mxmoe_w1_deltas={int(layer_idx): tensor.detach().cpu().clone() for layer_idx, tensor in base["mxmoe_w1_deltas"].items()},
        mxmoe_w2_deltas={int(layer_idx): tensor.detach().cpu().clone() for layer_idx, tensor in base["mxmoe_w2_deltas"].items()},
        mc_moe_scores={int(layer_idx): tensor.detach().cpu().clone() for layer_idx, tensor in base["mc_moe_scores"].items()},
    )


def load_cache(path: Path, model_id: str) -> tuple[CalibrationArtifacts | None, dict[int, list[int]] | None, dict[int, dict[int, torch.Tensor]] | None]:
    if not path.exists():
        return None, None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not cache_is_compatible(payload, model_id):
        return None, None, None
    calibration = restore_calibration(payload)
    raw_hot_experts = payload.get("hot_experts_per_layer")
    hot_experts = None
    if isinstance(raw_hot_experts, dict):
        hot_experts = {int(layer_idx): [int(expert_idx) for expert_idx in experts] for layer_idx, experts in raw_hot_experts.items()}
    raw_hot_scores = payload.get("hot_w2_channel_perturbation")
    hot_scores = None
    if isinstance(raw_hot_scores, dict):
        hot_scores = {
            int(layer_idx): {int(expert_idx): tensor.detach().cpu().clone() for expert_idx, tensor in expert_map.items()}
            for layer_idx, expert_map in raw_hot_scores.items()
        }
    return calibration, hot_experts, hot_scores


def save_cache(
    path: Path,
    model_id: str,
    calibration: CalibrationArtifacts,
    hot_experts: dict[int, list[int]] | None = None,
    hot_w2_scores: dict[int, dict[int, torch.Tensor]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serialize_cache(model_id, calibration, hot_experts, hot_w2_scores), path)


def build_empty_masks(config: Any) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx in range(config.num_hidden_layers):
        w1_pair_masks[layer_idx] = {
            expert_idx: torch.zeros(config.moe_intermediate_size, dtype=torch.bool) for expert_idx in range(config.num_experts)
        }
        w2_channel_masks[layer_idx] = {
            expert_idx: torch.zeros(config.hidden_size, dtype=torch.bool) for expert_idx in range(config.num_experts)
        }
    return w1_pair_masks, w2_channel_masks


def total_pair_fraction(config: Any, w1_pair_masks: dict[int, dict[int, torch.Tensor]]) -> float:
    total = config.num_hidden_layers * config.num_experts * config.moe_intermediate_size
    active = sum(int(mask.sum().item()) for layer in w1_pair_masks.values() for mask in layer.values())
    return float(active) / float(max(total, 1))


def total_channel_fraction(config: Any, w2_channel_masks: dict[int, dict[int, torch.Tensor]]) -> float:
    total = config.num_hidden_layers * config.num_experts * config.hidden_size
    active = sum(int(mask.sum().item()) for layer in w2_channel_masks.values() for mask in layer.values())
    return float(active) / float(max(total, 1))


def build_combined_perturbation_scores(calibration: CalibrationArtifacts) -> dict[int, torch.Tensor]:
    return {
        layer_idx: (calibration.mxmoe_w1_deltas[layer_idx].to(torch.float32) + calibration.mxmoe_w2_deltas[layer_idx].to(torch.float32))
        for layer_idx in range(len(calibration.mxmoe_w1_deltas))
    }


def build_weighted_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_score_map: dict[int, torch.Tensor],
    w2_score_map: dict[int, torch.Tensor],
    total_w1_pairs: int,
    total_w2_channels: int,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    items: list[tuple[int, int]] = []
    w1_capacities: list[int] = []
    w2_capacities: list[int] = []
    w1_weights: list[float] = []
    w2_weights: list[float] = []

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            active = int(bundle.routing_counts[expert_idx].item()) > 0
            items.append((layer_idx, expert_idx))
            w1_capacities.append(config.moe_intermediate_size if active else 0)
            w2_capacities.append(config.hidden_size if active else 0)
            w1_weights.append(max(0.0, float(w1_score_map[layer_idx][expert_idx].item())) if active else 0.0)
            w2_weights.append(max(0.0, float(w2_score_map[layer_idx][expert_idx].item())) if active else 0.0)

    w1_alloc = allocate_weighted_counts(total_w1_pairs, w1_capacities, w1_weights)
    w2_alloc = allocate_weighted_counts(total_w2_channels, w2_capacities, w2_weights)

    for index, (layer_idx, expert_idx) in enumerate(items):
        bundle = metric_cache[layer_idx]
        w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], int(w1_alloc[index]))
        w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], int(w2_alloc[index]))

    return w1_pair_masks, w2_channel_masks, {
        "target_w1_pairs": int(total_w1_pairs),
        "target_w2_channels": int(total_w2_channels),
        "realized_w1_pairs": int(sum(w1_alloc)),
        "realized_w2_channels": int(sum(w2_alloc)),
    }


def build_weighted_masks_from_fractions(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_score_map: dict[int, torch.Tensor],
    w2_score_map: dict[int, torch.Tensor],
    w1_fraction: float,
    w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    total_w1_pairs = int(round(w1_fraction * config.num_hidden_layers * config.num_experts * config.moe_intermediate_size))
    total_w2_channels = int(round(w2_fraction * config.num_hidden_layers * config.num_experts * config.hidden_size))
    return build_weighted_masks(metric_cache, config, w1_score_map, w2_score_map, total_w1_pairs, total_w2_channels)


def build_plan_from_masks(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_pair_masks: dict[int, dict[int, torch.Tensor]],
    w2_channel_masks: dict[int, dict[int, torch.Tensor]],
) -> EvalPlan:
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvalPlan(
        name=name,
        description=description,
        mode="per_channel",
        memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def select_hot_experts(
    routing_counts: dict[int, torch.Tensor],
    per_layer: int,
    min_count: int,
) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    for layer_idx, counts in routing_counts.items():
        candidates = [
            (int(counts[expert_idx].item()), expert_idx)
            for expert_idx in range(int(counts.numel()))
            if int(counts[expert_idx].item()) >= min_count
        ]
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected[layer_idx] = [int(expert_idx) for _count, expert_idx in candidates[:per_layer]]
    return selected


def build_hybrid_metric_cache(
    activation_cache: dict[int, LayerMetricBundle],
    hot_w2_scores: dict[int, dict[int, torch.Tensor]],
) -> dict[int, LayerMetricBundle]:
    hybrid = clone_metric_cache(activation_cache)
    for layer_idx, expert_map in hot_w2_scores.items():
        if layer_idx not in hybrid:
            continue
        for expert_idx, scores in expert_map.items():
            hybrid[layer_idx].w2_channel_scores[expert_idx] = scores.detach().cpu().clone().to(torch.float32)
    return hybrid


def build_mxmoe_topup_masks(
    calibration: CalibrationArtifacts,
    config: Any,
    total_expert_elems: int,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj = build_mxmoe_projection_promotions(config, total_expert_elems, calibration.mxmoe_w1_deltas, calibration.mxmoe_w2_deltas)
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    topup_w1 = int(round(topup_fraction * config.moe_intermediate_size))
    topup_w2 = int(round(topup_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        bundle = calibration.activation_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            else:
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], topup_w1)

            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            else:
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], topup_w2)

    promoted_w1 = sum(int(mask.sum().item()) for mask in w1_proj.values())
    promoted_w2 = sum(int(mask.sum().item()) for mask in w2_proj.values())
    return w1_pair_masks, w2_channel_masks, {
        "mxmoe_promoted_w1_projections": int(promoted_w1),
        "mxmoe_promoted_w2_projections": int(promoted_w2),
        "topup_fraction": float(topup_fraction),
    }


@torch.inference_mode()
def compute_hot_w2_channel_perturbation(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    hot_experts: dict[int, list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[int, torch.Tensor]]:
    print("\n=== Per-channel W2 output perturbation for hot experts ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    sq_errors: dict[int, dict[int, torch.Tensor]] = {
        layer_idx: {expert_idx: torch.zeros(config.hidden_size, dtype=torch.float64) for expert_idx in experts}
        for layer_idx, experts in hot_experts.items()
    }

    for layer_idx in range(config.num_hidden_layers):
        layer_hot = set(hot_experts.get(layer_idx, []))
        layer_type = str(config.layer_types[layer_idx])
        print(f"[w2-perturb] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type}) | hot={len(layer_hot)}", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_down = {
            expert_idx: quantize_linear_weight(down_proj[expert_idx], "fp4")
            for expert_idx in layer_hot
        }

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
            batch_size, sequence_length, hidden_dim = mlp_input.shape
            flat = mlp_input.view(-1, hidden_dim)
            router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
            routing_probs = torch.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(mlp_input.dtype)
            final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=mlp_input.dtype, device=mlp_input.device)

            expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
            for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
                token_idx, route_pos = torch.where(selected_experts == expert_idx)
                current_state = flat[token_idx]
                weights = routing_weights[token_idx, route_pos].unsqueeze(-1)
                gate_up = F.linear(current_state, gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                full_hidden = F.silu(gate) * up
                full_out = F.linear(full_hidden, down_proj[expert_idx])
                final_hidden_states.index_add_(0, token_idx, (weights * full_out.float()).to(mlp_input.dtype))

                if expert_idx in layer_hot:
                    diff_weight = (fp4_down[expert_idx] - down_proj[expert_idx]).to(device=full_hidden.device, dtype=full_hidden.dtype)
                    per_channel_delta = F.linear(full_hidden, diff_weight).float() * weights.float()
                    sq_errors[layer_idx][expert_idx].add_(per_channel_delta.square().sum(dim=0).cpu().to(torch.float64))

            shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
            shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
            shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
            shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
            outs[chunk_idx] = residual + (final_hidden_states + shared_out * shared_gate_value).view(batch_size, sequence_length, hidden_dim)

            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[w2-perturb] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        release_tensors({f"fp4_down_{expert_idx}": tensor for expert_idx, tensor in fp4_down.items()})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return {
        layer_idx: {expert_idx: torch.sqrt(scores).to(torch.float32) for expert_idx, scores in expert_map.items()}
        for layer_idx, expert_map in sq_errors.items()
    }


def load_proper_eval_references() -> dict[str, dict[str, float]]:
    path = RESULTS_DIR / "proper_eval.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    results = payload.get("results", {})
    references: dict[str, dict[str, float]] = {}
    for name, row in results.items():
        if isinstance(row, dict) and "ppl" in row and "memory_gb" in row:
            references[str(name)] = {
                "ppl": float(row["ppl"]),
                "memory_gb": float(row["memory_gb"]),
            }
    return references


def summarize_hot_experts(hot_experts: dict[int, list[int]], routing_counts: dict[int, torch.Tensor]) -> dict[str, Any]:
    return {
        str(layer_idx): [
            {
                "expert_idx": int(expert_idx),
                "routing_count": int(routing_counts[layer_idx][expert_idx].item()),
            }
            for expert_idx in experts
        ]
        for layer_idx, experts in hot_experts.items()
    }


def maybe_eval_plan(
    payload: dict[str, Any],
    plan: EvalPlan,
    config: Any,
    total_expert_elems: int,
    test_ids: torch.Tensor,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
    output_json: Path,
    extras: dict[str, Any],
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
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    akurt_ref = references.get("perchannel_akurt_w1_4_w2_16", {}).get("ppl")
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    print("\n" + "=" * 138, flush=True)
    print("proper_iter01 | hybrid perturbation/channel configs | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 138, flush=True)
    print(
        f"{'Config':<42} {'PPL':>10} {'dMxMoE':>10} {'dAKurt':>10} {'Memory GB':>12} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 138, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_akurt = "-" if akurt_ref is None else f"{float(row['ppl']) - akurt_ref:+.4f}"
        print(
            f"{name:<42} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_akurt:>10} {float(row['memory_gb']):>12.3f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_exploration_section(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    rows = payload["results"]
    ordered = sorted(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    best_name, best_row = ordered[0]
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    akurt_ref = references.get("perchannel_akurt_w1_4_w2_16", {}).get("ppl")
    table_lines = [
        "| Config | PPL | Delta vs MxMoE | Delta vs perchannel_akurt_w1_4_w2_16 | Memory (GB) |",
        "|--------|-----|----------------|-------------------------------------|-------------|",
    ]
    for name, row in ordered:
        delta_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        delta_akurt = "-" if akurt_ref is None else f"{float(row['ppl']) - akurt_ref:+.4f}"
        table_lines.append(
            f"| {name} | {float(row['ppl']):.4f} | {delta_mxmoe} | {delta_akurt} | {float(row['memory_gb']):.3f} |"
        )

    insights: list[str] = []
    if akurt_ref is not None:
        insights.append(
            f"Best hybrid config is `{best_name}` at PPL {float(best_row['ppl']):.4f}, which shifts {float(best_row['ppl']) - akurt_ref:+.4f} relative to `perchannel_akurt_w1_4_w2_16` (PPL {akurt_ref:.4f})."
        )
    else:
        insights.append(f"Best hybrid config is `{best_name}` at PPL {float(best_row['ppl']):.4f}.")
    if mxmoe_ref is not None:
        relation = "beats" if float(best_row["ppl"]) < mxmoe_ref else "still trails"
        insights.append(
            f"Against `mxmoe_per_block` (PPL {mxmoe_ref:.4f}), the winner {relation} by {abs(float(best_row['ppl']) - mxmoe_ref):.4f} PPL."
        )

    return "\n".join([
        SECTION_MARKER,
        f"**Approach**: Reused `proper_eval.py` calibration and full GPTQ-standard WikiText-2 evaluation, then tested five hybrid strategies that use MxMoE output perturbation for expert/projection budgeting and per-channel masks for within-expert refinement. Config 5 also recomputes actual W2 per-channel output perturbation for the top {HOT_EXPERTS_PER_LAYER} routed experts per layer and falls back to activation_kurtosis elsewhere.",
        f"**Eval**: Full WikiText-2 test ({payload['metadata']['evaluation']['total_tokens']} tokens, {payload['metadata']['evaluation']['nsamples']} chunks of {SEQLEN}), BF16 `F.linear`, FP32 loss, `loss.float() * seqlen`.",
        "**Result**:",
        "\n".join(table_lines),
        f"**Insight**: {' '.join(insights)}",
        "**Next**: If the best row still trails MxMoE, try the same hybrid budget logic with a tighter per-projection split around the winner and test whether top-up should target W2 only instead of both projections.",
    ])


def upsert_exploration_section(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SECTION_MARKER in existing:
        prefix = existing.split(SECTION_MARKER, 1)[0].rstrip()
        updated = prefix + "\n\n" + content.strip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + content.strip() + "\n" if existing.strip() else content.strip() + "\n"
    path.write_text(updated, encoding="utf-8")


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

    references = load_proper_eval_references()
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
            "hot_experts_per_layer": int(args.hot_experts_per_layer),
            "min_hot_routing_count": int(args.min_hot_routing_count),
            "references": references,
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

    cache_calibration, cache_hot_experts, cache_hot_scores = load_cache(args.cache_path, args.model_id)
    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    if cache_calibration is not None:
        calibration = cache_calibration
        print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)
    else:
        calibration = run_calibration(store, text_config, calib_chunks, device, dtype)
        save_cache(args.cache_path, args.model_id, calibration)
        print(f"[cache] saved base calibration to {args.cache_path}", flush=True)

    hot_experts = select_hot_experts(calibration.routing_counts, args.hot_experts_per_layer, args.min_hot_routing_count)
    payload["metadata"]["hot_expert_summary"] = summarize_hot_experts(hot_experts, calibration.routing_counts)

    hot_w2_scores: dict[int, dict[int, torch.Tensor]]
    if (
        cache_hot_scores is not None
        and cache_hot_experts == hot_experts
        and not args.force_recompute_channel_perturb
    ):
        hot_w2_scores = cache_hot_scores
        print(f"[cache] loaded hot-expert W2 channel perturbation from {args.cache_path}", flush=True)
    else:
        hot_w2_scores = compute_hot_w2_channel_perturbation(store, text_config, calib_chunks, hot_experts, device, dtype)
        save_cache(args.cache_path, args.model_id, calibration, hot_experts, hot_w2_scores)
        print(f"[cache] saved hot-expert W2 channel perturbation to {args.cache_path}", flush=True)

    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    combined_scores = build_combined_perturbation_scores(calibration)
    hybrid_metric_cache = build_hybrid_metric_cache(calibration.activation_cache, hot_w2_scores)

    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions(
        text_config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
    )
    mxmoe_w1_pairs = int(sum(int(mask.sum().item()) for mask in mxmoe_w1_proj.values()) * text_config.moe_intermediate_size)
    mxmoe_w2_channels = int(sum(int(mask.sum().item()) for mask in mxmoe_w2_proj.values()) * text_config.hidden_size)

    plans: list[tuple[EvalPlan, dict[str, Any]]] = []

    w1_masks, w2_masks, budget_meta = build_weighted_masks(
        calibration.activation_cache,
        text_config,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        mxmoe_w1_pairs,
        mxmoe_w2_channels,
    )
    plans.append((
        build_plan_from_masks(
            "mxmoe_perturbation_perchannel",
            "Match MxMoE's realized W1/W2 projection budget, but spread it across experts by per-projection perturbation and rank channels by activation_kurtosis.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "mxmoe_projection_perturbation",
            "channel_metric": "activation_kurtosis",
            **budget_meta,
        },
    ))

    w1_masks, w2_masks, budget_meta = build_weighted_masks_from_fractions(
        calibration.activation_cache,
        text_config,
        combined_scores,
        combined_scores,
        GLOBAL_25_W1,
        GLOBAL_25_W2,
    )
    plans.append((
        build_plan_from_masks(
            "perturbation_proportional_25pct",
            "Allocate a 25% W1 and 25% W2 channel budget across experts in proportion to combined output perturbation, then rank channels by activation_kurtosis.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "combined_perturbation_global_25pct",
            "channel_metric": "activation_kurtosis",
            **budget_meta,
        },
    ))

    w1_masks, w2_masks, budget_meta = build_weighted_masks_from_fractions(
        calibration.activation_cache,
        text_config,
        combined_scores,
        combined_scores,
        PERTURB_SPLIT_W1,
        PERTURB_SPLIT_W2,
    )
    plans.append((
        build_plan_from_masks(
            "perturbation_proportional_w1_4_w2_16",
            "Allocate W1=4% and W2=16% channel budgets across experts in proportion to combined output perturbation, then rank channels by activation_kurtosis.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "combined_perturbation_w1_4_w2_16",
            "channel_metric": "activation_kurtosis",
            **budget_meta,
        },
    ))

    w1_masks, w2_masks, budget_meta = build_mxmoe_topup_masks(calibration, text_config, total_expert_elems, TOPUP_FRACTION)
    plans.append((
        build_plan_from_masks(
            "mxmoe_block_plus_channel_topup",
            "Start from MxMoE per-block assignment, then top up every FP4 projection with the top 5% activation_kurtosis channels promoted to FP8.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "mxmoe_per_block_plus_top5pct",
            "channel_metric": "activation_kurtosis",
            **budget_meta,
        },
    ))

    w1_masks, w2_masks = build_two_level_masks(hybrid_metric_cache, text_config, PERTURB_SPLIT_W1, PERTURB_SPLIT_W2)
    plans.append((
        build_plan_from_masks(
            "output_perturbation_per_channel_9experts",
            "Keep the routing-aware W1=4% / W2=16% plan, but replace activation_kurtosis with actual W2 per-channel output perturbation for the hottest experts and fall back elsewhere.",
            text_config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            "budget_source": "routing_aware_w1_4_w2_16",
            "channel_metric": "w2_output_perturbation_hot_experts",
            "hot_expert_count_total": int(sum(len(experts) for experts in hot_experts.values())),
            "w2_metric_fallback": "activation_kurtosis",
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
    payload["metadata"]["cache_path"] = str(args.cache_path)
    payload["metadata"]["hot_expert_summary"] = summarize_hot_experts(hot_experts, calibration.routing_counts)
    atomic_json_dump(args.output_json, payload)

    print_results_table(payload["results"], references)
    upsert_exploration_section(args.exploration_md, build_exploration_section(payload, references))
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

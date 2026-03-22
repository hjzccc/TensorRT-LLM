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
import torch.nn.functional as F

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
    CALIBRATION_SAMPLES,
    SEQLEN,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    build_two_level_masks,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
)
from proper_iter01 import (
    build_plan_from_masks,
    load_cache,
    save_cache,
    select_hot_experts,
    total_channel_fraction,
    total_pair_fraction,
)
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter09_perchannel.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter09_perchannel_metric_cache.pt"
HOT_EXPERTS_PER_LAYER = 20
MIN_HOT_ROUTING_COUNT = 1
METRIC_CACHE_VERSION = 1
EPS = 1e-10


@dataclass
class HessianMomentState:
    counts: torch.Tensor
    input_sq_sum: torch.Tensor
    inter_sq_sum: torch.Tensor


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 9 per-channel plans.",
    )
    parser.add_argument("--hot-experts-per-layer", type=int, default=HOT_EXPERTS_PER_LAYER)
    parser.add_argument("--min-hot-routing-count", type=int, default=MIN_HOT_ROUTING_COUNT)
    parser.add_argument("--force-recompute-metrics", action="store_true")
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


def serialize_metric_cache_bundles(source: dict[int, LayerMetricBundle]) -> dict[int, dict[str, torch.Tensor]]:
    return {
        int(layer_idx): {
            "routing_counts": bundle.routing_counts.detach().cpu().clone(),
            "w1_pair_scores": bundle.w1_pair_scores.detach().cpu().clone(),
            "w2_channel_scores": bundle.w2_channel_scores.detach().cpu().clone(),
        }
        for layer_idx, bundle in source.items()
    }


def restore_metric_cache_bundles(raw_cache: Any) -> dict[int, LayerMetricBundle] | None:
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


def restore_hot_w2_scores(raw_scores: Any) -> dict[int, dict[int, torch.Tensor]] | None:
    if not isinstance(raw_scores, dict):
        return None
    restored: dict[int, dict[int, torch.Tensor]] = {}
    for raw_layer_idx, expert_map in raw_scores.items():
        if not isinstance(expert_map, dict):
            return None
        layer_idx = int(raw_layer_idx)
        restored[layer_idx] = {
            int(expert_idx): torch.as_tensor(scores).detach().cpu().clone().to(torch.float32)
            for expert_idx, scores in expert_map.items()
        }
    return restored


def metric_cache_is_compatible(payload: dict[str, Any], model_id: str, hot_experts_per_layer: int, min_hot_routing_count: int) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
        and int(payload.get("hot_experts_per_layer", -1)) == int(hot_experts_per_layer)
        and int(payload.get("min_hot_routing_count", -1)) == int(min_hot_routing_count)
    )


def load_metric_cache(
    path: Path,
    model_id: str,
    hot_experts_per_layer: int,
    min_hot_routing_count: int,
) -> tuple[dict[int, LayerMetricBundle] | None, dict[int, dict[int, torch.Tensor]] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id, hot_experts_per_layer, min_hot_routing_count):
        return None, None, None
    metric_cache = restore_metric_cache_bundles(payload.get("real_hessian_metric_cache"))
    hot_scores = restore_hot_w2_scores(payload.get("hot_w2_output_perturbation"))
    metadata = payload.get("metadata")
    if metric_cache is None or hot_scores is None or not isinstance(metadata, dict):
        return None, None, None
    return metric_cache, hot_scores, dict(metadata)


def save_metric_cache(
    path: Path,
    model_id: str,
    hot_experts_per_layer: int,
    min_hot_routing_count: int,
    metric_cache: dict[int, LayerMetricBundle],
    hot_w2_scores: dict[int, dict[int, torch.Tensor]],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "calibration_samples": CALIBRATION_SAMPLES,
            "hot_experts_per_layer": int(hot_experts_per_layer),
            "min_hot_routing_count": int(min_hot_routing_count),
            "real_hessian_metric_cache": serialize_metric_cache_bundles(metric_cache),
            "hot_w2_output_perturbation": {
                int(layer_idx): {
                    int(expert_idx): scores.detach().cpu().clone() for expert_idx, scores in expert_map.items()
                }
                for layer_idx, expert_map in hot_w2_scores.items()
            },
            "metadata": metadata,
        },
        path,
    )


def init_hessian_state(config: Any, device: torch.device) -> HessianMomentState:
    return HessianMomentState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        input_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        inter_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
    )


def tensor_gini(values: torch.Tensor) -> float:
    flat = values.detach().cpu().to(torch.float64).flatten()
    if flat.numel() == 0:
        return 0.0
    flat = torch.clamp(flat, min=0.0)
    total = float(flat.sum().item())
    if total <= 0.0:
        return 0.0
    sorted_values = torch.sort(flat).values
    count = int(sorted_values.numel())
    indices = torch.arange(1, count + 1, dtype=torch.float64)
    gini = (2.0 * torch.sum(indices * sorted_values) / (count * total)) - ((count + 1) / count)
    return float(gini.item())


def safe_spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    lhs = left.detach().cpu().to(torch.float64).flatten()
    rhs = right.detach().cpu().to(torch.float64).flatten()
    if int(lhs.numel()) != int(rhs.numel()):
        raise ValueError(f"Mismatched vector lengths: {lhs.numel()} vs {rhs.numel()}")
    if int(lhs.numel()) == 0:
        return 0.0
    lhs_rank = torch.argsort(torch.argsort(lhs, stable=True), stable=True).to(torch.float64)
    rhs_rank = torch.argsort(torch.argsort(rhs, stable=True), stable=True).to(torch.float64)
    lhs_centered = lhs_rank - lhs_rank.mean()
    rhs_centered = rhs_rank - rhs_rank.mean()
    denom = torch.sqrt(torch.sum(lhs_centered.square()) * torch.sum(rhs_centered.square()))
    if float(denom.item()) <= 0.0:
        return 0.0
    value = torch.sum(lhs_centered * rhs_centered) / denom
    value_f = float(value.item())
    if not bool(torch.isfinite(value)):
        return 0.0
    return value_f


def summarize_w2_scores(w2_scores: dict[int, torch.Tensor], routing_counts: dict[int, torch.Tensor]) -> dict[str, Any]:
    ginis: list[float] = []
    layer_means: dict[str, float] = {}
    total_active = 0
    for layer_idx, scores in w2_scores.items():
        layer_counts = routing_counts[layer_idx]
        layer_ginis: list[float] = []
        for expert_idx in range(int(layer_counts.numel())):
            if int(layer_counts[expert_idx].item()) <= 0:
                continue
            gini_value = tensor_gini(scores[expert_idx])
            ginis.append(gini_value)
            layer_ginis.append(gini_value)
            total_active += 1
        if layer_ginis:
            layer_means[str(layer_idx)] = round(sum(layer_ginis) / len(layer_ginis), 6)
    if not ginis:
        return {
            "active_experts": 0,
            "mean_gini": 0.0,
            "median_gini": 0.0,
            "min_gini": 0.0,
            "max_gini": 0.0,
            "layer_mean_gini": layer_means,
        }
    ordered = sorted(ginis)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "active_experts": int(total_active),
        "mean_gini": round(sum(ordered) / len(ordered), 6),
        "median_gini": round(float(median), 6),
        "min_gini": round(min(ordered), 6),
        "max_gini": round(max(ordered), 6),
        "layer_mean_gini": layer_means,
    }


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


def build_output_perturbation_metric_cache(
    real_hessian_cache: dict[int, LayerMetricBundle],
    hot_w2_scores: dict[int, dict[int, torch.Tensor]],
) -> dict[int, LayerMetricBundle]:
    metric_cache = clone_metric_cache(real_hessian_cache)
    for layer_idx, expert_map in hot_w2_scores.items():
        if layer_idx not in metric_cache:
            continue
        for expert_idx, scores in expert_map.items():
            metric_cache[layer_idx].w2_channel_scores[expert_idx] = scores.detach().cpu().clone().to(torch.float32)
    return metric_cache


def metric_cache_w2_scores(metric_cache: dict[int, LayerMetricBundle]) -> dict[int, torch.Tensor]:
    return {
        int(layer_idx): bundle.w2_channel_scores.detach().cpu().clone().to(torch.float32)
        for layer_idx, bundle in metric_cache.items()
    }


def summarize_metric_against_gt(
    metric_name: str,
    metric_cache: dict[int, LayerMetricBundle],
    hot_w2_scores: dict[int, dict[int, torch.Tensor]],
    routing_counts: dict[int, torch.Tensor],
) -> dict[str, Any]:
    gini_summary = summarize_w2_scores(metric_cache_w2_scores(metric_cache), routing_counts)
    spearman_values: list[float] = []
    per_expert: dict[str, float] = {}
    per_layer_mean: dict[str, float] = {}
    compared = 0
    for layer_idx, expert_map in hot_w2_scores.items():
        layer_values: list[float] = []
        for expert_idx, target_scores in expert_map.items():
            label = f"L{layer_idx:02d}_E{expert_idx:03d}"
            value = 1.0 if metric_name == "output_perturbation_hot_experts_with_real_hessian_fallback" else safe_spearman(
                metric_cache[layer_idx].w2_channel_scores[expert_idx],
                target_scores,
            )
            per_expert[label] = round(float(value), 6)
            spearman_values.append(float(value))
            layer_values.append(float(value))
            compared += 1
        if layer_values:
            per_layer_mean[str(layer_idx)] = round(sum(layer_values) / len(layer_values), 6)
    return {
        "metric": metric_name,
        "gini_summary": gini_summary,
        "compared_hot_experts": int(compared),
        "mean_spearman": round(sum(spearman_values) / len(spearman_values), 6) if spearman_values else 0.0,
        "layer_mean_spearman": per_layer_mean,
        "spearman_per_hot_expert": per_expert,
    }


def collect_layer_hessian_and_perturbation(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    fp4_down: torch.Tensor,
    config: Any,
    state: HessianMomentState,
    hot_experts: set[int],
    hot_sq_errors: dict[int, torch.Tensor],
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
        weights = routing_weights[token_idx, route_pos].unsqueeze(-1)
        state.input_sq_sum[expert_idx].add_(current_state.float().square().sum(dim=0))

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        state.inter_sq_sum[expert_idx].add_(full_hidden.float().square().sum(dim=0))

        full_out = F.linear(full_hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (weights * full_out.float()).to(hidden_states.dtype))

        if expert_idx in hot_experts:
            diff_weight = (fp4_down[expert_idx] - down_proj[expert_idx]).to(device=full_hidden.device, dtype=full_hidden.dtype)
            per_channel_delta = F.linear(full_hidden, diff_weight).float() * weights.float()
            hot_sq_errors[expert_idx].add_(per_channel_delta.square().sum(dim=0))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def finalize_real_hessian_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: HessianMomentState,
    config: Any,
) -> LayerMetricBundle:
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=gate_up_proj.device)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=gate_up_proj.device)

    for expert_idx in range(config.num_experts):
        count = int(state.counts[expert_idx].item())
        if count <= 0:
            continue

        input_h_diag = 2.0 * state.input_sq_sum[expert_idx] / float(count)
        inter_h_diag = 2.0 * state.inter_sq_sum[expert_idx] / float(count)

        gate_up_diff = gate_up_proj[expert_idx].float() - fp4_gate_up[expert_idx].float()
        gate_diff_sq = gate_up_diff[: config.moe_intermediate_size].square()
        up_diff_sq = gate_up_diff[config.moe_intermediate_size :].square()
        pair_diff_sq = gate_diff_sq + up_diff_sq
        w1_scores[expert_idx] = torch.matmul(pair_diff_sq, input_h_diag)

        down_diff_sq = (down_proj[expert_idx].float() - fp4_down[expert_idx].float()).square()
        w2_scores[expert_idx] = torch.matmul(down_diff_sq, inter_h_diag)

    return LayerMetricBundle(
        routing_counts=state.counts.detach().cpu().to(torch.int64),
        w1_pair_scores=w1_scores.detach().cpu(),
        w2_channel_scores=w2_scores.detach().cpu(),
    )


@torch.inference_mode()
def compute_real_hessian_and_hot_perturbation(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    hot_experts: dict[int, list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, LayerMetricBundle], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    print("\n=== Real per-channel Hessian + hot-expert W2 output perturbation ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    real_hessian_cache: dict[int, LayerMetricBundle] = {}
    hot_w2_scores: dict[int, dict[int, torch.Tensor]] = {}
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        layer_hot = set(hot_experts.get(layer_idx, []))
        layer_type = layer_type_at(config, layer_idx)
        print(f"[metric] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type}) | hot={len(layer_hot)}", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_hessian_state(config, device)
        hot_sq_errors = {
            expert_idx: torch.zeros(config.hidden_size, dtype=torch.float64, device=device)
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
            moe_out = collect_layer_hessian_and_perturbation(
                mlp_input,
                moe_tensors,
                fp4_down,
                config,
                state,
                layer_hot,
                hot_sq_errors,
            )
            outs[chunk_idx] = residual + moe_out

            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[metric] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        real_hessian_cache[layer_idx] = finalize_real_hessian_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        hot_w2_scores[layer_idx] = {
            expert_idx: torch.sqrt(scores).detach().cpu().to(torch.float32)
            for expert_idx, scores in hot_sq_errors.items()
        }

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_name": "real_hessian_and_hot_w2_output_perturbation",
        "runtime_seconds": round(time.time() - start_time, 3),
        "calibration_chunks": int(calib_chunks.shape[0]),
        "hot_experts_per_layer": {str(layer_idx): len(experts) for layer_idx, experts in hot_experts.items()},
    }
    return real_hessian_cache, hot_w2_scores, metadata


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
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    akurt_ref = references.get("perchannel_akurt_w1_4_w2_16", {}).get("ppl")
    print("\n" + "=" * 178, flush=True)
    print("proper_iter09_perchannel | real Hessian + hot-expert per-channel perturbation | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 178, flush=True)
    print(
        f"{'Config':<44} {'PPL':>10} {'dMxMoE':>10} {'dAKurt':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 178, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_akurt = "-" if akurt_ref is None else f"{float(row['ppl']) - akurt_ref:+.4f}"
        print(
            f"{name:<44} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_akurt:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_iteration_plans(
    real_hessian_cache: dict[int, LayerMetricBundle],
    output_perturb_cache: dict[int, LayerMetricBundle],
    hot_w2_scores: dict[int, dict[int, torch.Tensor]],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    total_hot_experts = sum(len(expert_map) for expert_map in hot_w2_scores.values())
    total_active_experts = sum(int((bundle.routing_counts > 0).sum().item()) for bundle in real_hessian_cache.values())
    w2_fallback_active_experts = max(total_active_experts - total_hot_experts, 0)

    for w1_fraction, w2_fraction, name, description in (
        (
            0.25,
            0.25,
            "perchannel_real_hessian_25pct",
            "Routing-aware per-channel plan with real routed Hessian scores for both W1 pairs and W2 channels at 25% / 25%.",
        ),
        (
            0.04,
            0.16,
            "perchannel_real_hessian_w1_4_w2_16",
            "Routing-aware per-channel plan with real routed Hessian scores for W1=4% and W2=16%.",
        ),
        (
            0.00,
            0.25,
            "perchannel_real_hessian_w1_0_w2_25",
            "Routing-aware per-channel plan that spends the full channel budget on W2 using real routed Hessian scores.",
        ),
    ):
        w1_masks, w2_masks = build_two_level_masks(real_hessian_cache, config, w1_fraction, w2_fraction)
        plans.append((
            build_plan_from_masks(name, description, config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks),
            {
                "budget_source": "routing_aware_two_level",
                "w1_fraction_target": float(w1_fraction),
                "w2_fraction_target": float(w2_fraction),
                "w1_channel_metric_requested": "real_hessian",
                "w1_channel_metric_effective": "real_hessian",
                "w2_channel_metric_requested": "real_hessian",
                "w2_channel_metric_effective": "real_hessian",
                "w2_channel_metric_fallback_used": False,
            },
        ))

    for w1_fraction, w2_fraction, name, description in (
        (
            0.25,
            0.25,
            "perchannel_output_perturbation_25pct",
            "Routing-aware per-channel plan with real Hessian W1 scores and W2 scores from exact hot-expert output perturbation plus real-Hessian fallback elsewhere at 25% / 25%.",
        ),
        (
            0.04,
            0.16,
            "perchannel_output_perturbation_w1_4_w2_16",
            "Routing-aware per-channel plan with real Hessian W1 scores and W2 scores from exact hot-expert output perturbation plus real-Hessian fallback elsewhere at W1=4% and W2=16%.",
        ),
        (
            0.00,
            0.25,
            "perchannel_output_perturbation_w1_0_w2_25",
            "Routing-aware W2-only channel plan with exact hot-expert output perturbation for W2 and real-Hessian fallback for the remaining experts.",
        ),
    ):
        w1_masks, w2_masks = build_two_level_masks(output_perturb_cache, config, w1_fraction, w2_fraction)
        plans.append((
            build_plan_from_masks(name, description, config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks),
            {
                "budget_source": "routing_aware_two_level",
                "w1_fraction_target": float(w1_fraction),
                "w2_fraction_target": float(w2_fraction),
                "w1_channel_metric_requested": "real_hessian",
                "w1_channel_metric_effective": "real_hessian",
                "w2_channel_metric_requested": "output_perturbation_hot_experts",
                "w2_channel_metric_effective": "output_perturbation_hot_experts_with_real_hessian_fallback",
                "w2_channel_metric_fallback_used": bool(w2_fallback_active_experts > 0),
                "w2_hot_experts": int(total_hot_experts),
                "w2_fallback_active_experts": int(w2_fallback_active_experts),
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
            "metric_cache_path": str(args.metric_cache_path),
            "references": references,
            "experiment": "Iteration 9 per-channel rerun with routed real Hessian scores and hot-expert W2 output perturbation fallback",
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

    calibration, _cache_hot_experts, _cache_hot_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        print(f"[cache] base calibration missing at {args.cache_path}; recomputing with proper_eval.run_calibration", flush=True)
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        calibration = run_calibration(store, text_config, calib_chunks, device, dtype)
        save_cache(args.cache_path, args.model_id, calibration)
        print(f"[cache] saved base calibration to {args.cache_path}", flush=True)
    else:
        print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)

    hot_experts = select_hot_experts(calibration.routing_counts, args.hot_experts_per_layer, args.min_hot_routing_count)
    payload["metadata"]["hot_expert_selection"] = {
        "hot_experts_per_layer": int(args.hot_experts_per_layer),
        "min_hot_routing_count": int(args.min_hot_routing_count),
        "selection": summarize_hot_experts(hot_experts, calibration.routing_counts),
    }

    real_hessian_cache, hot_w2_scores, metric_cache_meta = (None, None, None)
    if not args.force_recompute_metrics:
        real_hessian_cache, hot_w2_scores, metric_cache_meta = load_metric_cache(
            args.metric_cache_path,
            args.model_id,
            args.hot_experts_per_layer,
            args.min_hot_routing_count,
        )
        if real_hessian_cache is not None and hot_w2_scores is not None and metric_cache_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if real_hessian_cache is None or hot_w2_scores is None or metric_cache_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        real_hessian_cache, hot_w2_scores, metric_cache_meta = compute_real_hessian_and_hot_perturbation(
            store,
            text_config,
            calib_chunks,
            hot_experts,
            device,
            dtype,
        )
        save_metric_cache(
            args.metric_cache_path,
            args.model_id,
            args.hot_experts_per_layer,
            args.min_hot_routing_count,
            real_hessian_cache,
            hot_w2_scores,
            metric_cache_meta,
        )
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    output_perturb_cache = build_output_perturbation_metric_cache(real_hessian_cache, hot_w2_scores)
    metric_analysis = {
        "activation_kurtosis": summarize_metric_against_gt(
            "activation_kurtosis",
            calibration.activation_cache,
            hot_w2_scores,
            calibration.routing_counts,
        ),
        "real_hessian": summarize_metric_against_gt(
            "real_hessian",
            real_hessian_cache,
            hot_w2_scores,
            calibration.routing_counts,
        ),
        "output_perturbation_hot_experts_with_real_hessian_fallback": summarize_metric_against_gt(
            "output_perturbation_hot_experts_with_real_hessian_fallback",
            output_perturb_cache,
            hot_w2_scores,
            calibration.routing_counts,
        ),
    }
    ranking_rows = [
        {
            "metric": name,
            "mean_spearman": float(summary["mean_spearman"]),
            "mean_gini": float(summary["gini_summary"]["mean_gini"]),
        }
        for name, summary in metric_analysis.items()
    ]
    ranking_rows.sort(key=lambda row: (-float(row["mean_spearman"]), -float(row["mean_gini"]), row["metric"]))
    payload["metadata"]["metric_cache"] = metric_cache_meta
    payload["metadata"]["metric_analysis"] = metric_analysis
    payload["metadata"]["metric_ranking"] = ranking_rows
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(real_hessian_cache, output_perturb_cache, hot_w2_scores, text_config, non_expert_bytes, total_expert_elems)
    available_names = [plan.name for plan, _extras in plans]
    if requested_plans is not None:
        missing = sorted(requested_plans.difference(available_names))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
    payload["metadata"]["requested_plan_order"] = [plan.name for plan, _extras in plans]
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
        "metric_cache_version": METRIC_CACHE_VERSION,
    }
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload["results"], references)
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

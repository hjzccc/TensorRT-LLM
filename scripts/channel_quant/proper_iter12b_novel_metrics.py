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

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys, topk_mask_from_scores
from proper_eval import (
    CALIBRATION_SAMPLES,
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
from proper_iter01 import build_empty_masks, build_plan_from_masks, total_channel_fraction, total_pair_fraction
from proper_iter10_novel_perchannel import build_global_fraction_masks
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter12b_novel_metrics.json"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter12b_novel_metrics_metric_cache.pt"
METRIC_CACHE_VERSION = 1
EPS = 1e-10
ENTROPY_BINS = 32
ENTROPY_ROW_CHUNK = 512
INITIAL_W1_FRACTION = 0.04
INITIAL_W2_FRACTION = 0.16
UNIFORM_FRACTION = 0.25
TOP_K_FOLLOWUP = 3

METRIC_ORDER = (
    "wanda",
    "rqe",
    "awre",
    "col_condition",
    "baq_score",
    "owq_exact",
    "ra_wanda",
    "entropy",
)

METRIC_DESCRIPTIONS = {
    "wanda": "WANDA importance = sum_k |w_jk| * E[|x_k|] per output channel.",
    "rqe": "Relative quantization error = ||w_j - q(w_j)||_2 / ||w_j||_2 per output channel.",
    "awre": "Activation-weighted relative error = sum_k E[x_k^2] * delta_jk^2 / (||w_j||_2^2 + eps).",
    "col_condition": "Approximate per-channel condition proxy = ||w_j||_2^2 / ||W||_F^2.",
    "baq_score": "BAQ-style score = range(w_j)^2 * mean(E[x^2]) / 12 per output channel.",
    "owq_exact": "OWQ-style exact channel score = sum_k E[x_k^2] * delta_jk^2.",
    "ra_wanda": "Router-affinity times WANDA = sum_t p_{t,e} * sum_k |w_jk| * |x_tk|.",
    "entropy": "Row entropy over 32-bin weight histogram; higher entropy rows are treated as harder to quantize.",
}


@dataclass
class MetricState:
    counts: torch.Tensor
    input_abs_sum: torch.Tensor
    input_sq_sum: torch.Tensor
    inter_abs_sum: torch.Tensor
    inter_sq_sum: torch.Tensor
    router_input_abs_sum: torch.Tensor
    router_inter_abs_sum: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help=(
            "Comma-separated subset of plan names to evaluate. "
            "Default runs all 20%% sort-and-split plans, then the top-3 metrics again at 25%% uniform per-expert budget."
        ),
    )
    parser.add_argument("--force-recompute-metrics", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


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


def serialize_metric_cache_collection(source: dict[str, dict[int, LayerMetricBundle]]) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    return {metric_name: serialize_metric_cache_bundles(metric_cache) for metric_name, metric_cache in source.items()}


def restore_metric_cache_collection(raw_cache: Any) -> dict[str, dict[int, LayerMetricBundle]] | None:
    if not isinstance(raw_cache, dict):
        return None
    restored: dict[str, dict[int, LayerMetricBundle]] = {}
    for metric_name in METRIC_ORDER:
        metric_cache = restore_metric_cache_bundles(raw_cache.get(metric_name))
        if metric_cache is None:
            return None
        restored[metric_name] = metric_cache
    return restored


def metric_cache_is_compatible(payload: dict[str, Any], model_id: str) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
        and int(payload.get("entropy_bins", -1)) == ENTROPY_BINS
    )


def load_metric_cache(
    path: Path,
    model_id: str,
) -> tuple[dict[str, dict[int, LayerMetricBundle]] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id):
        return None, None
    metric_caches = restore_metric_cache_collection(payload.get("metric_caches"))
    metadata = payload.get("metadata")
    if metric_caches is None or not isinstance(metadata, dict):
        return None, None
    return metric_caches, dict(metadata)


def save_metric_cache(
    path: Path,
    model_id: str,
    metric_caches: dict[str, dict[int, LayerMetricBundle]],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "calibration_samples": CALIBRATION_SAMPLES,
            "entropy_bins": ENTROPY_BINS,
            "metric_caches": serialize_metric_cache_collection(metric_caches),
            "metadata": metadata,
        },
        path,
    )


def init_metric_state(config: Any, device: torch.device) -> MetricState:
    return MetricState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        input_abs_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        input_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        inter_abs_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        inter_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        router_input_abs_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        router_inter_abs_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
    )


def collect_layer_metric_state(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: MetricState,
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
        route_prob = routing_probs[token_idx, expert_idx].unsqueeze(-1).to(torch.float32)
        route_weight = routing_weights[token_idx, route_pos].unsqueeze(-1)

        current_state_f = current_state.float()
        current_abs = current_state_f.abs()
        current_sq = current_state_f.square()
        state.input_abs_sum[expert_idx].add_(current_abs.sum(dim=0))
        state.input_sq_sum[expert_idx].add_(current_sq.sum(dim=0))
        state.router_input_abs_sum[expert_idx].add_((route_prob * current_abs).sum(dim=0))

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        full_hidden_f = full_hidden.float()
        full_hidden_abs = full_hidden_f.abs()
        full_hidden_sq = full_hidden_f.square()
        state.inter_abs_sum[expert_idx].add_(full_hidden_abs.sum(dim=0))
        state.inter_sq_sum[expert_idx].add_(full_hidden_sq.sum(dim=0))
        state.router_inter_abs_sum[expert_idx].add_((route_prob * full_hidden_abs).sum(dim=0))

        full_out = F.linear(full_hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (route_weight * full_out.float()).to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def row_entropy_scores(weight: torch.Tensor, bins: int = ENTROPY_BINS, row_chunk: int = ENTROPY_ROW_CHUNK) -> torch.Tensor:
    rows = int(weight.shape[0])
    scores = torch.zeros(rows, dtype=torch.float32, device=weight.device)
    if rows == 0:
        return scores
    cols = int(weight.shape[1])
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        chunk = weight[start:stop].to(torch.float32)
        row_min = chunk.min(dim=1, keepdim=True).values
        row_max = chunk.max(dim=1, keepdim=True).values
        span = row_max - row_min
        valid = span.squeeze(1) > EPS
        if not bool(valid.any()):
            continue
        valid_chunk = chunk[valid]
        valid_min = row_min[valid]
        valid_span = span[valid]
        scaled = (valid_chunk - valid_min) / valid_span
        scaled = torch.clamp(scaled, min=0.0, max=1.0 - torch.finfo(torch.float32).eps)
        bin_idx = torch.floor(scaled * float(bins)).to(torch.int64)
        hist = torch.zeros((int(valid_chunk.shape[0]), bins), dtype=torch.float32, device=weight.device)
        hist.scatter_add_(1, bin_idx, torch.ones_like(valid_chunk, dtype=torch.float32))
        probs = hist / float(max(cols, 1))
        entropy = -(probs * torch.log(torch.clamp(probs, min=EPS))).sum(dim=1)
        score_chunk = scores[start:stop]
        score_chunk[valid] = entropy
    return scores


def compute_row_metric_views(
    weight: torch.Tensor,
    qweight: torch.Tensor,
    abs_mean: torch.Tensor,
    sq_mean: torch.Tensor,
    router_abs_sum: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weight_f = weight.float()
    qweight_f = qweight.float()
    delta = weight_f - qweight_f
    delta_sq = delta.square()
    weight_sq = weight_f.square()
    row_norm_sq = weight_sq.sum(dim=1)
    row_norm = torch.sqrt(torch.clamp(row_norm_sq, min=EPS))
    matrix_norm_sq = float(weight_sq.sum().item())
    h_mean = float(sq_mean.mean().item())
    row_range = weight_f.max(dim=1).values - weight_f.min(dim=1).values

    return {
        "wanda": weight_f.abs().matmul(abs_mean),
        "rqe": torch.sqrt(delta_sq.sum(dim=1)) / row_norm,
        "awre": delta_sq.matmul(sq_mean) / torch.clamp(row_norm_sq, min=EPS),
        "col_condition": row_norm_sq / max(matrix_norm_sq, EPS),
        "baq_score": row_range.square() * (h_mean / 12.0),
        "owq_exact": delta_sq.matmul(sq_mean),
        "ra_wanda": weight_f.abs().matmul(router_abs_sum),
        "entropy": row_entropy_scores(weight_f),
    }


def finalize_metric_caches_for_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: MetricState,
    config: Any,
) -> dict[str, LayerMetricBundle]:
    counts_cpu = state.counts.detach().cpu().to(torch.int64)
    layer_w1_scores = {
        metric_name: torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
        for metric_name in METRIC_ORDER
    }
    layer_w2_scores = {
        metric_name: torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
        for metric_name in METRIC_ORDER
    }

    for expert_idx in range(config.num_experts):
        count = int(state.counts[expert_idx].item())
        if count <= 0:
            continue
        input_abs_mean = state.input_abs_sum[expert_idx] / float(count)
        input_sq_mean = state.input_sq_sum[expert_idx] / float(count)
        inter_abs_mean = state.inter_abs_sum[expert_idx] / float(count)
        inter_sq_mean = state.inter_sq_sum[expert_idx] / float(count)

        gate_up_rows = compute_row_metric_views(
            gate_up_proj[expert_idx],
            fp4_gate_up[expert_idx],
            input_abs_mean,
            input_sq_mean,
            state.router_input_abs_sum[expert_idx],
        )
        down_rows = compute_row_metric_views(
            down_proj[expert_idx],
            fp4_down[expert_idx],
            inter_abs_mean,
            inter_sq_mean,
            state.router_inter_abs_sum[expert_idx],
        )
        split = config.moe_intermediate_size
        for metric_name in METRIC_ORDER:
            layer_w1_scores[metric_name][expert_idx] = (
                gate_up_rows[metric_name][:split] + gate_up_rows[metric_name][split:]
            ).detach().cpu()
            layer_w2_scores[metric_name][expert_idx] = down_rows[metric_name].detach().cpu()

    return {
        metric_name: LayerMetricBundle(
            routing_counts=counts_cpu.clone(),
            w1_pair_scores=layer_w1_scores[metric_name],
            w2_channel_scores=layer_w2_scores[metric_name],
        )
        for metric_name in METRIC_ORDER
    }


@torch.inference_mode()
def compute_metric_artifacts(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, dict[int, LayerMetricBundle]], dict[str, Any]]:
    print("\n=== Iteration 12b calibration: novel per-channel metrics in one forward pass ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    metric_caches = {metric_name: {} for metric_name in METRIC_ORDER}
    per_layer_stats: dict[str, Any] = {}
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[metric] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_metric_state(config, device)

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
            moe_out = collect_layer_metric_state(mlp_input, moe_tensors, config, state)
            outs[chunk_idx] = residual + moe_out

            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[metric] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        layer_metric_bundles = finalize_metric_caches_for_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        for metric_name, bundle in layer_metric_bundles.items():
            metric_caches[metric_name][layer_idx] = bundle

        per_layer_stats[str(layer_idx)] = {
            "active_experts": int((state.counts > 0).sum().item()),
            "max_routing_count": int(state.counts.max().item()),
            "min_active_routing_count": int(state.counts[state.counts > 0].min().item()) if bool((state.counts > 0).any()) else 0,
        }

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_names": list(METRIC_ORDER),
        "runtime_seconds": round(time.time() - start_time, 3),
        "calibration_chunks": int(calib_chunks.shape[0]),
        "entropy_bins": ENTROPY_BINS,
        "w1_pair_reduction": "sum(gate_row_score, up_row_score)",
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "per_layer_stats": per_layer_stats,
    }
    return metric_caches, metadata


def build_uniform_fraction_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_fraction: float,
    w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    k_w1 = int(round(w1_fraction * config.moe_intermediate_size))
    k_w2 = int(round(w2_fraction * config.hidden_size))
    for layer_idx, bundle in metric_cache.items():
        for expert_idx in range(config.num_experts):
            w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], k_w1)
            w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], k_w2)
    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "uniform_per_expert_topk",
        "w1_fraction_target": float(w1_fraction),
        "w2_fraction_target": float(w2_fraction),
        "uniform_budget_per_expert": True,
    }


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


def build_initial_plans(
    metric_caches: dict[str, dict[int, LayerMetricBundle]],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    for metric_name in METRIC_ORDER:
        w1_masks, w2_masks, extras = build_global_fraction_masks(
            metric_caches[metric_name],
            config,
            INITIAL_W1_FRACTION,
            INITIAL_W2_FRACTION,
            budget_source=f"{metric_name}_global_sort_split",
        )
        plans.append((
            build_plan_from_masks(
                f"{metric_name}_w1_4_w2_16",
                (
                    f"Global sort-and-split plan using the {metric_name} metric with W1={int(round(INITIAL_W1_FRACTION * 100))}% "
                    f"and W2={int(round(INITIAL_W2_FRACTION * 100))}% FP8 budget."
                ),
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **extras,
                "channel_metric_requested": metric_name,
                "channel_metric_effective": metric_name,
                "channel_metric_fallback_used": False,
                "metric_description": METRIC_DESCRIPTIONS[metric_name],
                "plan_stage": "initial_20pct_sort_split",
            },
        ))
    return plans


def build_uniform_followup_plans(
    metric_caches: dict[str, dict[int, LayerMetricBundle]],
    selected_metric_names: list[str],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    for rank, metric_name in enumerate(selected_metric_names, start=1):
        w1_masks, w2_masks, extras = build_uniform_fraction_masks(
            metric_caches[metric_name],
            config,
            UNIFORM_FRACTION,
            UNIFORM_FRACTION,
        )
        plans.append((
            build_plan_from_masks(
                f"{metric_name}_uniform_25pct",
                f"Uniform per-expert 25% W1/W2 channel plan using the {metric_name} ranking within each expert.",
                config,
                non_expert_bytes,
                total_expert_elems,
                w1_masks,
                w2_masks,
            ),
            {
                **extras,
                "channel_metric_requested": metric_name,
                "channel_metric_effective": metric_name,
                "channel_metric_fallback_used": False,
                "metric_description": METRIC_DESCRIPTIONS[metric_name],
                "plan_stage": "top3_uniform_25pct_followup",
                "top3_rank_from_20pct": int(rank),
            },
        ))
    return plans


def select_top_metric_names(results: dict[str, Any], top_k: int) -> list[str]:
    scored: list[tuple[float, float, str]] = []
    for metric_name in METRIC_ORDER:
        plan_name = f"{metric_name}_w1_4_w2_16"
        row = results.get(plan_name)
        if not isinstance(row, dict) or "ppl" not in row or "memory_gb" not in row:
            continue
        scored.append((float(row["ppl"]), float(row["memory_gb"]), metric_name))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [metric_name for _ppl, _memory_gb, metric_name in scored[:top_k]]


def print_results_table(results: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    best_ref = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 184, flush=True)
    print("proper_iter12b_novel_metrics | fast novel per-channel metrics | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 184, flush=True)
    print(
        f"{'Config':<44} {'PPL':>10} {'dMxMoE':>10} {'dBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10} {'Stage':>16}",
        flush=True,
    )
    print("-" * 184, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_best = "-" if best_ref is None else f"{float(row['ppl']) - best_ref:+.4f}"
        print(
            f"{name:<44} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_best:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f} {str(row.get('plan_stage', '-')):>16}",
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
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "metric_cache_path": str(args.metric_cache_path),
            "references": references,
            "experiment": "Iteration 12b fast novel per-channel metrics from literature plus a router-affinity x WANDA combination",
            "novel_metric_protocol": {
                "initial_plans": [f"{metric_name}_w1_4_w2_16" for metric_name in METRIC_ORDER],
                "initial_budget": {"w1_fraction": INITIAL_W1_FRACTION, "w2_fraction": INITIAL_W2_FRACTION},
                "followup_policy": f"top {TOP_K_FOLLOWUP} metrics by PPL from the initial sweep are re-evaluated at a 25% uniform per-expert W1/W2 budget",
                "uniform_followup_fraction": UNIFORM_FRACTION,
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

    metric_caches, metric_cache_meta = (None, None)
    if not args.force_recompute_metrics:
        metric_caches, metric_cache_meta = load_metric_cache(args.metric_cache_path, args.model_id)
        if metric_caches is not None and metric_cache_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if metric_caches is None or metric_cache_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        metric_caches, metric_cache_meta = compute_metric_artifacts(store, text_config, calib_chunks, device, dtype)
        save_metric_cache(args.metric_cache_path, args.model_id, metric_caches, metric_cache_meta)
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache_metadata"] = metric_cache_meta
    atomic_json_dump(args.output_json, payload)

    initial_plans = build_initial_plans(metric_caches, text_config, non_expert_bytes, total_expert_elems)
    all_named_plans = {plan.name: (plan, extras) for plan, extras in initial_plans}
    for plan, extras in build_uniform_followup_plans(metric_caches, list(METRIC_ORDER), text_config, non_expert_bytes, total_expert_elems):
        all_named_plans[plan.name] = (plan, extras)

    if requested_plans is not None:
        missing = sorted(requested_plans.difference(set(all_named_plans)))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")
        plans_to_run = [all_named_plans[name] for name in all_named_plans if name in requested_plans]
        for plan, extras in plans_to_run:
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
    else:
        for plan, extras in initial_plans:
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

        top_metric_names = select_top_metric_names(payload.get("results", {}), TOP_K_FOLLOWUP)
        payload["metadata"]["top3_from_20pct_ppl"] = top_metric_names
        atomic_json_dump(args.output_json, payload)
        print(f"[top3] selected by PPL from 20% sweep: {', '.join(top_metric_names)}", flush=True)

        followup_plans = build_uniform_followup_plans(metric_caches, top_metric_names, text_config, non_expert_bytes, total_expert_elems)
        for plan, extras in followup_plans:
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

    payload["metadata"]["completed_at_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload.get("results", {}), references)


if __name__ == "__main__":
    main()

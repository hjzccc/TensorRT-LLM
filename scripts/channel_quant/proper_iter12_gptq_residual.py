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
import torch.nn.functional as F

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
    CALIBRATION_SAMPLES,
    SEQLEN,
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
)
from proper_iter01 import build_empty_masks, build_plan_from_masks, load_cache, save_cache, total_channel_fraction, total_pair_fraction
from proper_iter05 import build_joint_precision_promotions
from proper_iter07 import build_mxmoe_projection_promotions_fraction
from proper_iter10_novel_perchannel import build_global_fraction_masks
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter12_gptq_residual.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter12_gptq_residual_metric_cache.pt"
METRIC_CACHE_VERSION = 1
GPTQ_BLOCKSIZE = 128
GPTQ_DAMP_PERCENT = 0.01
MEDIUM_TIER_W2_FRACTION = 0.50
EPS = 1e-10


@dataclass
class GPTQResidualState:
    counts: torch.Tensor
    router_input_sq_sum: torch.Tensor
    router_inter_sq_sum: torch.Tensor
    w1_xtx_sum: dict[int, torch.Tensor]
    w2_xtx_sum: dict[int, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--metric-cache-path", type=Path, default=DEFAULT_METRIC_CACHE_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--blocksize", type=int, default=GPTQ_BLOCKSIZE)
    parser.add_argument("--damp-percent", type=float, default=GPTQ_DAMP_PERCENT)
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 12 GPTQ-residual plans.",
    )
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


def metric_cache_is_compatible(payload: dict[str, Any], model_id: str, blocksize: int, damp_percent: float) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
        and int(payload.get("blocksize", -1)) == int(blocksize)
        and math.isclose(float(payload.get("damp_percent", -1.0)), float(damp_percent), rel_tol=0.0, abs_tol=1e-12)
    )


def load_metric_cache(
    path: Path,
    model_id: str,
    blocksize: int,
    damp_percent: float,
) -> tuple[dict[int, LayerMetricBundle] | None, dict[int, LayerMetricBundle] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id, blocksize, damp_percent):
        return None, None, None
    gptq_residual_cache = restore_metric_cache_bundles(payload.get("gptq_residual_cache"))
    router_affinity_cache = restore_metric_cache_bundles(payload.get("router_affinity_cache"))
    metadata = payload.get("metadata")
    if gptq_residual_cache is None or router_affinity_cache is None or not isinstance(metadata, dict):
        return None, None, None
    return gptq_residual_cache, router_affinity_cache, dict(metadata)


def save_metric_cache(
    path: Path,
    model_id: str,
    blocksize: int,
    damp_percent: float,
    gptq_residual_cache: dict[int, LayerMetricBundle],
    router_affinity_cache: dict[int, LayerMetricBundle],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "calibration_samples": CALIBRATION_SAMPLES,
            "blocksize": int(blocksize),
            "damp_percent": float(damp_percent),
            "gptq_residual_cache": serialize_metric_cache_bundles(gptq_residual_cache),
            "router_affinity_cache": serialize_metric_cache_bundles(router_affinity_cache),
            "metadata": metadata,
        },
        path,
    )


def init_gptq_state(config: Any, device: torch.device) -> GPTQResidualState:
    return GPTQResidualState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        router_input_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        router_inter_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        w1_xtx_sum={},
        w2_xtx_sum={},
    )


def accumulate_xtx(storage: dict[int, torch.Tensor], expert_idx: int, values: torch.Tensor) -> None:
    values_f = values.float()
    xtx = torch.matmul(values_f.transpose(0, 1), values_f)
    if expert_idx in storage:
        storage[expert_idx].add_(xtx)
    else:
        storage[expert_idx] = xtx.clone()


def collect_layer_gptq_metrics(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: GPTQResidualState,
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
        state.router_input_sq_sum[expert_idx].add_((route_prob * current_state_f.square()).sum(dim=0))
        accumulate_xtx(state.w1_xtx_sum, expert_idx, current_state_f)

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        full_hidden_f = full_hidden.float()
        state.router_inter_sq_sum[expert_idx].add_((route_prob * full_hidden_f.square()).sum(dim=0))
        accumulate_xtx(state.w2_xtx_sum, expert_idx, full_hidden_f)

        full_out = F.linear(full_hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (route_weight * full_out.float()).to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def quantize_fp4_column(column: torch.Tensor) -> torch.Tensor:
    return quantize_to_nvfp4_columns(column).to(torch.float32)


def prepare_hinv_factor(hessian: torch.Tensor, damp_percent: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    hessian_f = hessian.to(device=device, dtype=torch.float32).clone()
    diag = torch.diag(hessian_f)
    dead = diag == 0
    if bool(dead.any()):
        indices = torch.nonzero(dead, as_tuple=False).flatten()
        hessian_f[indices, indices] = 1.0

    mean_diag = float(torch.diag(hessian_f).mean().item())
    damp = float(damp_percent * mean_diag) if mean_diag > 0.0 else float(max(damp_percent, 1e-6))
    eye_index = torch.arange(hessian_f.shape[0], device=device)
    attempt_damp = max(damp, 1e-6)
    for _attempt in range(8):
        try:
            work = hessian_f.clone()
            work[eye_index, eye_index] += attempt_damp
            hinv = torch.linalg.cholesky(work)
            hinv = torch.cholesky_inverse(hinv)
            hinv = torch.linalg.cholesky(hinv, upper=True)
            return hinv, dead, float(attempt_damp)
        except RuntimeError:
            attempt_damp *= 10.0
    raise RuntimeError("Failed to factor damped Hessian for GPTQ residual scoring")


def quantize_fp4_column_gpu(column: torch.Tensor) -> torch.Tensor:
    return quantize_to_nvfp4_columns(column.float()).to(torch.float32)


def gptq_residual_score_from_hessian(
    weight: torch.Tensor,
    xtx_sum: torch.Tensor | None,
    nsamples: int,
    blocksize: int,
    damp_percent: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, int]:
    if xtx_sum is None or nsamples <= 0:
        return torch.zeros(weight.shape[0], dtype=torch.float32), 0.0, 0

    work_weight = weight.to(device=device, dtype=torch.float32).clone()
    hessian = xtx_sum.to(device=device, dtype=torch.float32) * (2.0 / float(nsamples))
    hinv_factor, dead, used_damp = prepare_hinv_factor(hessian, damp_percent, device)
    dead_count = int(dead.sum().item())
    if dead_count > 0:
        work_weight[:, dead] = 0

    losses = torch.zeros_like(work_weight, dtype=torch.float32)
    columns = int(work_weight.shape[1])
    for k1 in range(0, columns, blocksize):
        k2 = min(k1 + blocksize, columns)
        count = k2 - k1
        block = work_weight[:, k1:k2].clone()
        block_losses = torch.zeros_like(block, dtype=torch.float32)
        block_err = torch.zeros_like(block, dtype=torch.float32)
        block_hinv = hinv_factor[k1:k2, k1:k2]
        for i in range(count):
            w = block[:, i]
            d = float(block_hinv[i, i].item())
            if d <= 0.0 or not math.isfinite(d):
                continue
            q = quantize_fp4_column_gpu(w)
            err = (w - q) / d
            block[:, i] = q
            block_losses[:, i] = err.square()
            block_err[:, i] = err
            if i + 1 < count:
                block[:, i + 1 :] -= err.unsqueeze(1) * block_hinv[i, i + 1 :].unsqueeze(0)
        losses[:, k1:k2] = block_losses
        work_weight[:, k1:k2] = block
        if k2 < columns:
            work_weight[:, k2:] -= block_err.matmul(hinv_factor[k1:k2, k2:])
    return losses.sum(dim=1).detach().cpu().to(torch.float32), used_damp, dead_count


def finalize_router_affinity_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: GPTQResidualState,
    config: Any,
) -> LayerMetricBundle:
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)

    for expert_idx in range(config.num_experts):
        if int(state.counts[expert_idx].item()) <= 0:
            continue
        gate_up_diff = gate_up_proj[expert_idx].float() - fp4_gate_up[expert_idx].float()
        gate_diff_sq = gate_up_diff[: config.moe_intermediate_size].square()
        up_diff_sq = gate_up_diff[config.moe_intermediate_size :].square()
        pair_diff_sq = gate_diff_sq + up_diff_sq
        w1_scores[expert_idx] = torch.matmul(pair_diff_sq.cpu(), state.router_input_sq_sum[expert_idx].detach().cpu())

        down_diff_sq = (down_proj[expert_idx].float() - fp4_down[expert_idx].float()).square()
        w2_scores[expert_idx] = torch.matmul(down_diff_sq.cpu(), state.router_inter_sq_sum[expert_idx].detach().cpu())

    return LayerMetricBundle(
        routing_counts=state.counts.detach().cpu().to(torch.int64),
        w1_pair_scores=w1_scores,
        w2_channel_scores=w2_scores,
    )


def finalize_gptq_residual_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    state: GPTQResidualState,
    config: Any,
    blocksize: int,
    damp_percent: float,
    layer_idx: int,
    device: torch.device,
) -> tuple[LayerMetricBundle, dict[str, Any]]:
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
    layer_stats: dict[str, Any] = {
        "active_experts": 0,
        "w1_dead_columns": 0,
        "w2_dead_columns": 0,
        "w1_mean_damp": 0.0,
        "w2_mean_damp": 0.0,
    }
    w1_damps: list[float] = []
    w2_damps: list[float] = []
    active_count = int((state.counts > 0).sum().item())
    scored = 0
    finalize_start = time.time()

    for expert_idx in range(config.num_experts):
        count = int(state.counts[expert_idx].item())
        if count <= 0:
            continue
        layer_stats["active_experts"] += 1
        scored += 1

        gate_up_score, w1_damp, w1_dead = gptq_residual_score_from_hessian(
            gate_up_proj[expert_idx],
            state.w1_xtx_sum.get(expert_idx),
            count,
            blocksize,
            damp_percent,
            device,
        )
        w1_scores[expert_idx] = gate_up_score[: config.moe_intermediate_size] + gate_up_score[config.moe_intermediate_size :]
        layer_stats["w1_dead_columns"] += int(w1_dead)
        w1_damps.append(float(w1_damp))

        down_score, w2_damp, w2_dead = gptq_residual_score_from_hessian(
            down_proj[expert_idx],
            state.w2_xtx_sum.get(expert_idx),
            count,
            blocksize,
            damp_percent,
            device,
        )
        w2_scores[expert_idx] = down_score
        layer_stats["w2_dead_columns"] += int(w2_dead)
        w2_damps.append(float(w2_damp))

        if scored == 1 or scored % 32 == 0 or scored == active_count:
            elapsed = time.time() - finalize_start
            print(f"[gptq] layer {layer_idx + 1} scored expert {scored}/{active_count} ({elapsed:.1f}s)", flush=True)

    if w1_damps:
        layer_stats["w1_mean_damp"] = float(sum(w1_damps) / len(w1_damps))
    if w2_damps:
        layer_stats["w2_mean_damp"] = float(sum(w2_damps) / len(w2_damps))

    return LayerMetricBundle(
        routing_counts=state.counts.detach().cpu().to(torch.int64),
        w1_pair_scores=w1_scores,
        w2_channel_scores=w2_scores,
    ), layer_stats


@torch.inference_mode()
def compute_gptq_residual_metric_artifacts(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    blocksize: int,
    damp_percent: float,
) -> tuple[dict[int, LayerMetricBundle], dict[int, LayerMetricBundle], dict[str, Any]]:
    print("\n=== Iteration 12 calibration: GPTQ residual per-channel + router affinity ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    gptq_residual_cache: dict[int, LayerMetricBundle] = {}
    router_affinity_cache: dict[int, LayerMetricBundle] = {}
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
        state = init_gptq_state(config, device)

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
            moe_out = collect_layer_gptq_metrics(mlp_input, moe_tensors, config, state)
            outs[chunk_idx] = residual + moe_out

            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[metric] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        router_affinity_cache[layer_idx] = finalize_router_affinity_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        gptq_residual_cache[layer_idx], per_layer_stats[str(layer_idx)] = finalize_gptq_residual_layer(
            gate_up_proj,
            down_proj,
            state,
            config,
            blocksize,
            damp_percent,
            layer_idx,
            device,
        )

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        state.w1_xtx_sum.clear()
        state.w2_xtx_sum.clear()
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_name": "iter12_gptq_residual_perchannel_artifacts",
        "runtime_seconds": round(time.time() - start_time, 3),
        "calibration_chunks": int(calib_chunks.shape[0]),
        "blocksize": int(blocksize),
        "damp_percent": float(damp_percent),
        "gptq_quant_mode": "fp4",
        "per_layer_stats": per_layer_stats,
    }
    return gptq_residual_cache, router_affinity_cache, metadata


def multiply_metric_caches(
    left: dict[int, LayerMetricBundle],
    right: dict[int, LayerMetricBundle],
) -> dict[int, LayerMetricBundle]:
    combined = clone_metric_cache(left)
    for layer_idx, bundle in combined.items():
        rhs = right[layer_idx]
        bundle.w1_pair_scores = bundle.w1_pair_scores * rhs.w1_pair_scores.to(torch.float32)
        bundle.w2_channel_scores = bundle.w2_channel_scores * rhs.w2_channel_scores.to(torch.float32)
    return combined


def split_total_budget(total_fraction: float) -> tuple[float, float]:
    return total_fraction / 5.0, total_fraction * 4.0 / 5.0


def build_mxmoe_topup_masks(
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
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
            if bool(w2_proj[layer_idx][expert_idx]):
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if not bool(w1_proj[layer_idx][expert_idx]) and topup_w1 > 0:
                topk = torch.topk(bundle.w1_pair_scores[expert_idx], k=min(topup_w1, config.moe_intermediate_size)).indices
                w1_pair_masks[layer_idx][expert_idx][topk] = True
            if not bool(w2_proj[layer_idx][expert_idx]) and topup_w2 > 0:
                topk = torch.topk(bundle.w2_channel_scores[expert_idx], k=min(topup_w2, config.hidden_size)).indices
                w2_channel_masks[layer_idx][expert_idx][topk] = True

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "mxmoe_per_block_plus_gptq_residual_topup",
        "mxmoe_base_fraction": float(base_fraction),
        "topup_fraction": float(topup_fraction),
        "w1_topup_fraction": float(topup_fraction),
        "w2_topup_fraction": float(topup_fraction),
        "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in w1_proj.values())),
        "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in w2_proj.values())),
    }


def build_joint_gptq_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    medium_w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj, tier_meta = build_joint_precision_promotions(calibration, config)
    medium_channels = int(round(medium_w2_fraction * config.hidden_size))
    w1_pair_masks = {
        layer_idx: {expert_idx: torch.zeros(config.moe_intermediate_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }
    w2_channel_masks = {
        layer_idx: {expert_idx: torch.zeros(config.hidden_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }

    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            elif bool(w2_proj[layer_idx][expert_idx]) and medium_channels > 0:
                topk = torch.topk(metric_cache[layer_idx].w2_channel_scores[expert_idx], k=min(medium_channels, config.hidden_size)).indices
                w2_channel_masks[layer_idx][expert_idx][topk] = True

    return w1_pair_masks, w2_channel_masks, {
        **tier_meta,
        "budget_source": "global_perturbation_percentile_buckets_with_gptq_residual_medium_channels",
        "medium_tier_w2_fraction": float(medium_w2_fraction),
        "medium_tier_w2_channels": int(medium_channels),
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
    best_ref = min((row["ppl"] for row in references.values()), default=None)
    print("\n" + "=" * 182, flush=True)
    print("proper_iter12_gptq_residual | GPTQ-residual per-channel selection | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 182, flush=True)
    print(
        f"{'Config':<48} {'PPL':>10} {'dMxMoE':>10} {'dBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 182, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_best = "-" if best_ref is None else f"{float(row['ppl']) - best_ref:+.4f}"
        print(
            f"{name:<48} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_best:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_iteration_plans(
    calibration: CalibrationArtifacts,
    gptq_residual_cache: dict[int, LayerMetricBundle],
    router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []

    for total_fraction, name, description in (
        (
            0.20,
            "gptq_residual_perchannel_20pct",
            "Global GPTQ-residual per-channel split at 20% total FP8 budget with W1:W2 = 1:4.",
        ),
        (
            0.25,
            "gptq_residual_perchannel_25pct",
            "Global GPTQ-residual per-channel split at 25% total FP8 budget with W1:W2 = 1:4.",
        ),
    ):
        w1_fraction, w2_fraction = split_total_budget(total_fraction)
        w1_masks, w2_masks, extras = build_global_fraction_masks(
            gptq_residual_cache,
            config,
            w1_fraction,
            w2_fraction,
            budget_source="gptq_residual_global_topk",
        )
        plans.append((
            build_plan_from_masks(name, description, config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks),
            {
                **extras,
                "channel_metric_requested": "gptq_residual",
                "channel_metric_effective": "gptq_residual",
                "channel_metric_fallback_used": False,
            },
        ))

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        gptq_residual_cache,
        config,
        0.04,
        0.16,
        budget_source="gptq_residual_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "gptq_residual_perchannel_w1_4_w2_16",
            "Global GPTQ-residual per-channel plan with explicit W1=4% and W2=16% budgets.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **extras,
            "channel_metric_requested": "gptq_residual",
            "channel_metric_effective": "gptq_residual",
            "channel_metric_fallback_used": False,
        },
    ))

    w1_masks, w2_masks, extras = build_mxmoe_topup_masks(
        gptq_residual_cache,
        calibration,
        config,
        total_expert_elems,
        base_fraction=0.25,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "gptq_residual_plus_mxmoe_topup_5pct",
            "MxMoE per-block base plan with a 5% GPTQ-residual per-channel topup on both projections.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **extras,
            "channel_metric_requested": "gptq_residual",
            "channel_metric_effective": "gptq_residual",
            "channel_metric_fallback_used": False,
        },
    ))

    router_weighted_cache = multiply_metric_caches(gptq_residual_cache, router_affinity_cache)
    w1_fraction, w2_fraction = split_total_budget(0.25)
    w1_masks, w2_masks, extras = build_global_fraction_masks(
        router_weighted_cache,
        config,
        w1_fraction,
        w2_fraction,
        budget_source="gptq_residual_times_router_affinity_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "gptq_residual_plus_router_affinity_25pct",
            "Global per-channel plan with GPTQ residual scores reweighted by router affinity under a 25% total budget and 1:4 W1:W2 split.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **extras,
            "channel_metric_requested": "gptq_residual_times_router_affinity",
            "channel_metric_effective": "gptq_residual_times_router_affinity",
            "channel_metric_fallback_used": False,
        },
    ))

    w1_masks, w2_masks, extras = build_joint_gptq_masks(
        calibration,
        gptq_residual_cache,
        config,
        MEDIUM_TIER_W2_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "gptq_residual_joint_w1w2",
            "Joint 3-tier precision plan with MxMoE high-tier projection promotions and GPTQ-residual W2 channel ranking for medium-tier experts.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **extras,
            "channel_metric_requested": "gptq_residual",
            "channel_metric_effective": "gptq_residual",
            "channel_metric_fallback_used": False,
        },
    ))

    return plans


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.blocksize <= 0:
        raise ValueError("--blocksize must be positive")

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
            "cache_path": str(args.cache_path),
            "metric_cache_path": str(args.metric_cache_path),
            "references": references,
            "experiment": "Iteration 12 GPTQ-residual per-channel ranking with exact routed Hessians and GPTQ-style error compensation at FP4",
            "gptq_metric": {
                "blocksize": int(args.blocksize),
                "damp_percent": float(args.damp_percent),
                "score_definition": "per-output-channel sum of squared GPTQ residual errors after blockwise FP4 compensation",
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

    calibration, _cache_hot_experts, _cache_hot_scores = load_cache(args.cache_path, args.model_id)
    if calibration is None:
        print(f"[cache] base calibration missing at {args.cache_path}; recomputing with proper_eval.run_calibration", flush=True)
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        calibration = run_calibration(store, text_config, calib_chunks, device, dtype)
        save_cache(args.cache_path, args.model_id, calibration)
        print(f"[cache] saved base calibration to {args.cache_path}", flush=True)
    else:
        print(f"[cache] loaded base calibration from {args.cache_path}", flush=True)

    gptq_residual_cache, router_affinity_cache, metric_cache_meta = (None, None, None)
    if not args.force_recompute_metrics:
        gptq_residual_cache, router_affinity_cache, metric_cache_meta = load_metric_cache(
            args.metric_cache_path,
            args.model_id,
            args.blocksize,
            args.damp_percent,
        )
        if gptq_residual_cache is not None and router_affinity_cache is not None and metric_cache_meta is not None:
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if gptq_residual_cache is None or router_affinity_cache is None or metric_cache_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        gptq_residual_cache, router_affinity_cache, metric_cache_meta = compute_gptq_residual_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
            args.blocksize,
            args.damp_percent,
        )
        save_metric_cache(
            args.metric_cache_path,
            args.model_id,
            args.blocksize,
            args.damp_percent,
            gptq_residual_cache,
            router_affinity_cache,
            metric_cache_meta,
        )
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache_metadata"] = metric_cache_meta
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(
        calibration,
        gptq_residual_cache,
        router_affinity_cache,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )
    if requested_plans is not None:
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        missing = sorted(requested_plans.difference({plan.name for plan, _extras in plans}))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")

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

    payload["metadata"]["completed_at_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload.get("results", {}), references)


if __name__ == "__main__":
    main()

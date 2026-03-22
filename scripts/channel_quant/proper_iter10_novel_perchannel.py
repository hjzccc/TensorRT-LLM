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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter10_novel_perchannel.json"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter10_novel_perchannel_metric_cache.pt"
METRIC_CACHE_VERSION = 1
EPS = 1e-10
NVFP4_E2M1_MAX = 6.0
OUTLIER_TOP_FRACTION = 0.05


@dataclass
class NovelMomentState:
    counts: torch.Tensor
    input_sq_sum: torch.Tensor
    inter_sq_sum: torch.Tensor
    router_input_sq_sum: torch.Tensor
    router_inter_sq_sum: torch.Tensor
    w2_output_abs_sum: torch.Tensor


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 10 novel per-channel plans.",
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


def serialize_tensor_cache(source: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {int(layer_idx): tensor.detach().cpu().clone() for layer_idx, tensor in source.items()}


def restore_tensor_cache(raw_cache: Any) -> dict[int, torch.Tensor] | None:
    if not isinstance(raw_cache, dict):
        return None
    restored: dict[int, torch.Tensor] = {}
    for raw_layer_idx, tensor in raw_cache.items():
        restored[int(raw_layer_idx)] = torch.as_tensor(tensor).detach().cpu().clone().to(torch.float32)
    return restored


def metric_cache_is_compatible(payload: dict[str, Any], model_id: str) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
    )


def load_metric_cache(
    path: Path,
    model_id: str,
) -> tuple[
    dict[int, LayerMetricBundle] | None,
    dict[int, LayerMetricBundle] | None,
    dict[int, torch.Tensor] | None,
    dict[int, torch.Tensor] | None,
    dict[str, Any] | None,
]:
    if not path.exists():
        return None, None, None, None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id):
        return None, None, None, None, None
    router_affinity_cache = restore_metric_cache_bundles(payload.get("router_affinity_cache"))
    hessian_normalized_cache = restore_metric_cache_bundles(payload.get("hessian_normalized_cache"))
    micromix_mean_abs = restore_tensor_cache(payload.get("micromix_mean_abs"))
    micromix_thresholds = restore_tensor_cache(payload.get("micromix_thresholds"))
    metadata = payload.get("metadata")
    if (
        router_affinity_cache is None
        or hessian_normalized_cache is None
        or micromix_mean_abs is None
        or micromix_thresholds is None
        or not isinstance(metadata, dict)
    ):
        return None, None, None, None, None
    return router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, dict(metadata)


def save_metric_cache(
    path: Path,
    model_id: str,
    router_affinity_cache: dict[int, LayerMetricBundle],
    hessian_normalized_cache: dict[int, LayerMetricBundle],
    micromix_mean_abs: dict[int, torch.Tensor],
    micromix_thresholds: dict[int, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "calibration_samples": CALIBRATION_SAMPLES,
            "router_affinity_cache": serialize_metric_cache_bundles(router_affinity_cache),
            "hessian_normalized_cache": serialize_metric_cache_bundles(hessian_normalized_cache),
            "micromix_mean_abs": serialize_tensor_cache(micromix_mean_abs),
            "micromix_thresholds": serialize_tensor_cache(micromix_thresholds),
            "metadata": metadata,
        },
        path,
    )


def init_novel_state(config: Any, device: torch.device) -> NovelMomentState:
    return NovelMomentState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        input_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        inter_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        router_input_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        router_inter_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        w2_output_abs_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
    )


def collect_layer_novel_metrics(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: NovelMomentState,
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

        current_sq = current_state.float().square()
        state.input_sq_sum[expert_idx].add_(current_sq.sum(dim=0))
        state.router_input_sq_sum[expert_idx].add_((route_prob * current_sq).sum(dim=0))

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        full_hidden_sq = full_hidden.float().square()
        state.inter_sq_sum[expert_idx].add_(full_hidden_sq.sum(dim=0))
        state.router_inter_sq_sum[expert_idx].add_((route_prob * full_hidden_sq).sum(dim=0))

        full_out = F.linear(full_hidden, down_proj[expert_idx])
        state.w2_output_abs_sum[expert_idx].add_(full_out.float().abs().sum(dim=0))
        final_hidden_states.index_add_(0, token_idx, (route_weight * full_out.float()).to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def finalize_router_affinity_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: NovelMomentState,
    config: Any,
) -> LayerMetricBundle:
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=gate_up_proj.device)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=gate_up_proj.device)

    for expert_idx in range(config.num_experts):
        if int(state.counts[expert_idx].item()) <= 0:
            continue
        gate_up_diff = gate_up_proj[expert_idx].float() - fp4_gate_up[expert_idx].float()
        gate_diff_sq = gate_up_diff[: config.moe_intermediate_size].square()
        up_diff_sq = gate_up_diff[config.moe_intermediate_size :].square()
        pair_diff_sq = gate_diff_sq + up_diff_sq
        w1_scores[expert_idx] = torch.matmul(pair_diff_sq, state.router_input_sq_sum[expert_idx])

        down_diff_sq = (down_proj[expert_idx].float() - fp4_down[expert_idx].float()).square()
        w2_scores[expert_idx] = torch.matmul(down_diff_sq, state.router_inter_sq_sum[expert_idx])

    return LayerMetricBundle(
        routing_counts=state.counts.detach().cpu().to(torch.int64),
        w1_pair_scores=w1_scores.detach().cpu(),
        w2_channel_scores=w2_scores.detach().cpu(),
    )


def finalize_hessian_normalized_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: NovelMomentState,
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
        raw_w1 = torch.matmul(gate_diff_sq + up_diff_sq, input_h_diag)

        down_diff_sq = (down_proj[expert_idx].float() - fp4_down[expert_idx].float()).square()
        raw_w2 = torch.matmul(down_diff_sq, inter_h_diag)

        w1_total = float(raw_w1.sum().item())
        w2_total = float(raw_w2.sum().item())
        w1_scores[expert_idx] = raw_w1 / max(w1_total, EPS)
        w2_scores[expert_idx] = raw_w2 / max(w2_total, EPS)

    return LayerMetricBundle(
        routing_counts=state.counts.detach().cpu().to(torch.int64),
        w1_pair_scores=w1_scores.detach().cpu(),
        w2_channel_scores=w2_scores.detach().cpu(),
    )


@torch.inference_mode()
def compute_novel_metric_artifacts(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    dict[int, LayerMetricBundle],
    dict[int, LayerMetricBundle],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[str, Any],
]:
    print("\n=== Iteration 10 calibration: router-affinity, MicroMix, cross-expert agreement, normalized Hessian ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    router_affinity_cache: dict[int, LayerMetricBundle] = {}
    hessian_normalized_cache: dict[int, LayerMetricBundle] = {}
    micromix_mean_abs: dict[int, torch.Tensor] = {}
    micromix_thresholds: dict[int, torch.Tensor] = {}
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
        state = init_novel_state(config, device)

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
            moe_out = collect_layer_novel_metrics(mlp_input, moe_tensors, config, state)
            outs[chunk_idx] = residual + moe_out

            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[metric] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        router_affinity_cache[layer_idx] = finalize_router_affinity_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        hessian_normalized_cache[layer_idx] = finalize_hessian_normalized_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)

        counts_f = state.counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        mean_abs = state.w2_output_abs_sum / counts_f
        active_mask = (state.counts > 0).unsqueeze(-1)
        micromix_mean_abs[layer_idx] = torch.where(active_mask, mean_abs, torch.zeros_like(mean_abs)).detach().cpu().to(torch.float32)
        micromix_thresholds[layer_idx] = (down_proj.float().abs().amax(dim=(1,2)).unsqueeze(-1) / (2.0 * NVFP4_E2M1_MAX)).detach().cpu().to(torch.float32)

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_name": "iter10_novel_perchannel_artifacts",
        "runtime_seconds": round(time.time() - start_time, 3),
        "calibration_chunks": int(calib_chunks.shape[0]),
        "micromix_threshold_formula": "weight_absmax / (2 * 6.0)",
        "cross_expert_outlier_fraction": OUTLIER_TOP_FRACTION,
    }
    return router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metadata


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
    print("\n" + "=" * 180, flush=True)
    print("proper_iter10_novel_perchannel | latest-research per-channel ideas | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 180, flush=True)
    print(
        f"{'Config':<46} {'PPL':>10} {'dMxMoE':>10} {'dAKurt':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 180, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_akurt = "-" if akurt_ref is None else f"{float(row['ppl']) - akurt_ref:+.4f}"
        print(
            f"{name:<46} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_akurt:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_routing_scale_map(metric_cache: dict[int, LayerMetricBundle]) -> dict[int, torch.Tensor]:
    return {
        int(layer_idx): bundle.routing_counts.detach().cpu().to(torch.float32)
        for layer_idx, bundle in metric_cache.items()
    }


def select_global_topk_mask(flat_scores: torch.Tensor, flat_valid: torch.Tensor, target: int) -> torch.Tensor:
    mask = torch.zeros(int(flat_scores.numel()), dtype=torch.bool)
    if target <= 0:
        return mask
    valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
    if int(valid_indices.numel()) == 0:
        return mask
    if target >= int(valid_indices.numel()):
        mask[valid_indices] = True
        return mask

    valid_scores = flat_scores[valid_indices]
    kth = int(valid_scores.numel()) - target + 1
    threshold = torch.kthvalue(valid_scores, kth).values
    chosen_valid = valid_scores > threshold
    remaining = target - int(chosen_valid.sum().item())
    if remaining > 0:
        tie_indices = torch.nonzero(valid_scores == threshold, as_tuple=False).flatten()
        chosen_valid[tie_indices[:remaining]] = True
    mask[valid_indices[chosen_valid]] = True
    return mask


def build_global_fraction_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    w1_fraction: float,
    w2_fraction: float,
    w1_expert_scale: dict[int, torch.Tensor] | None = None,
    w2_expert_scale: dict[int, torch.Tensor] | None = None,
    budget_source: str = "global_fraction",
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    total_w1 = config.num_hidden_layers * config.num_experts * config.moe_intermediate_size
    total_w2 = config.num_hidden_layers * config.num_experts * config.hidden_size
    target_w1 = int(round(w1_fraction * total_w1))
    target_w2 = int(round(w2_fraction * total_w2))

    w1_chunks: list[torch.Tensor] = []
    w1_valid_chunks: list[torch.Tensor] = []
    w2_chunks: list[torch.Tensor] = []
    w2_valid_chunks: list[torch.Tensor] = []

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        active = (bundle.routing_counts > 0).view(-1, 1)

        layer_w1_scores = bundle.w1_pair_scores.detach().cpu().to(torch.float32)
        if w1_expert_scale is not None:
            layer_w1_scores = layer_w1_scores * w1_expert_scale[layer_idx].view(-1, 1).to(torch.float32)
        w1_chunks.append(layer_w1_scores.reshape(-1))
        w1_valid_chunks.append(active.expand(-1, config.moe_intermediate_size).reshape(-1))

        layer_w2_scores = bundle.w2_channel_scores.detach().cpu().to(torch.float32)
        if w2_expert_scale is not None:
            layer_w2_scores = layer_w2_scores * w2_expert_scale[layer_idx].view(-1, 1).to(torch.float32)
        w2_chunks.append(layer_w2_scores.reshape(-1))
        w2_valid_chunks.append(active.expand(-1, config.hidden_size).reshape(-1))

    w1_mask_flat = select_global_topk_mask(torch.cat(w1_chunks, dim=0), torch.cat(w1_valid_chunks, dim=0), target_w1)
    w2_mask_flat = select_global_topk_mask(torch.cat(w2_chunks, dim=0), torch.cat(w2_valid_chunks, dim=0), target_w2)

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
        "budget_source": budget_source,
        "w1_fraction_target": float(w1_fraction),
        "w2_fraction_target": float(w2_fraction),
        "global_w1_selected": int(w1_mask_flat.sum().item()),
        "global_w2_selected": int(w2_mask_flat.sum().item()),
        "global_w1_total": int(total_w1),
        "global_w2_total": int(total_w2),
    }


def select_top_expert_fraction(routing_counts: dict[int, torch.Tensor], fraction: float) -> dict[int, set[int]]:
    selected: dict[int, set[int]] = {}
    for layer_idx, counts in routing_counts.items():
        active = [expert_idx for expert_idx in range(int(counts.numel())) if int(counts[expert_idx].item()) > 0]
        if not active:
            selected[layer_idx] = set()
            continue
        ranked = sorted(active, key=lambda expert_idx: (int(counts[expert_idx].item()), -expert_idx), reverse=True)
        keep = max(1, int(math.ceil(float(len(ranked)) * fraction)))
        selected[layer_idx] = set(ranked[:keep])
    return selected


def build_micromix_threshold_masks(
    micromix_mean_abs: dict[int, torch.Tensor],
    micromix_thresholds: dict[int, torch.Tensor],
    routing_counts: dict[int, torch.Tensor],
    config: Any,
    routed_expert_fraction: float | None = None,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    selected_experts = None if routed_expert_fraction is None else select_top_expert_fraction(routing_counts, routed_expert_fraction)

    for layer_idx in range(config.num_hidden_layers):
        layer_selected = None if selected_experts is None else selected_experts.get(layer_idx, set())
        mean_abs = micromix_mean_abs[layer_idx]
        thresholds = micromix_thresholds[layer_idx]
        counts = routing_counts[layer_idx]
        for expert_idx in range(config.num_experts):
            if int(counts[expert_idx].item()) <= 0:
                continue
            if layer_selected is not None and expert_idx not in layer_selected:
                continue
            t = thresholds[expert_idx]
            if t.dim() > 0 and t.shape[0] != mean_abs[expert_idx].shape[0]:
                t = t.mean()
            w2_channel_masks[layer_idx][expert_idx] = (mean_abs[expert_idx] > t).to(torch.bool).cpu()

    extras: dict[str, Any] = {
        "budget_source": "micromix_threshold",
        "w1_fraction_target": 0.0,
        "w2_fraction_target": None,
        "micromix_threshold_formula": "weight_absmax / (2 * 6.0)",
        "expert_fraction_gate": None if routed_expert_fraction is None else float(routed_expert_fraction),
    }
    if selected_experts is not None:
        extras["selected_experts_per_layer"] = {str(layer_idx): sorted(experts) for layer_idx, experts in selected_experts.items()}
    return w1_pair_masks, w2_channel_masks, extras


def build_cross_expert_outlier_masks(
    micromix_mean_abs: dict[int, torch.Tensor],
    routing_counts: dict[int, torch.Tensor],
    config: Any,
    agreement_threshold: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_pair_masks, w2_channel_masks = build_empty_masks(config)
    selected_per_layer: dict[str, int] = {}
    active_experts_per_layer: dict[str, int] = {}

    topk = max(1, int(round(config.hidden_size * OUTLIER_TOP_FRACTION)))
    for layer_idx in range(config.num_hidden_layers):
        counts = routing_counts[layer_idx]
        active = [expert_idx for expert_idx in range(config.num_experts) if int(counts[expert_idx].item()) > 0]
        active_experts_per_layer[str(layer_idx)] = int(len(active))
        if not active:
            selected_per_layer[str(layer_idx)] = 0
            continue
        votes = torch.zeros(config.hidden_size, dtype=torch.int64)
        layer_mean_abs = micromix_mean_abs[layer_idx]
        for expert_idx in active:
            votes.add_(topk_mask_from_scores(layer_mean_abs[expert_idx], topk).to(torch.int64))
        agreement = votes.to(torch.float32) / float(max(len(active), 1))
        shared_mask = agreement > agreement_threshold
        selected_per_layer[str(layer_idx)] = int(shared_mask.sum().item())
        for expert_idx in range(config.num_experts):
            w2_channel_masks[layer_idx][expert_idx] = shared_mask.clone()

    return w1_pair_masks, w2_channel_masks, {
        "budget_source": "cross_expert_outlier_agreement",
        "w1_fraction_target": 0.0,
        "w2_fraction_target": None,
        "outlier_fraction_per_expert": float(OUTLIER_TOP_FRACTION),
        "agreement_threshold": float(agreement_threshold),
        "selected_channels_per_layer": selected_per_layer,
        "active_experts_per_layer": active_experts_per_layer,
    }


def build_iteration_plans(
    router_affinity_cache: dict[int, LayerMetricBundle],
    hessian_normalized_cache: dict[int, LayerMetricBundle],
    micromix_mean_abs: dict[int, torch.Tensor],
    micromix_thresholds: dict[int, torch.Tensor],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    routing_counts = {int(layer_idx): bundle.routing_counts.detach().cpu().clone() for layer_idx, bundle in router_affinity_cache.items()}
    routing_scale = build_routing_scale_map(hessian_normalized_cache)

    w1_masks, w2_masks, extras = build_micromix_threshold_masks(micromix_mean_abs, micromix_thresholds, routing_counts, config)
    plans.append((
        build_plan_from_masks(
            "micromix_threshold",
            "MicroMix-style explicit threshold on per-channel mean |output activation| for W2; channels above weight_absmax/(2*6) use FP8, W1 stays FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
    ))

    w1_masks, w2_masks, extras = build_micromix_threshold_masks(micromix_mean_abs, micromix_thresholds, routing_counts, config, routed_expert_fraction=0.5)
    plans.append((
        build_plan_from_masks(
            "micromix_threshold_routing",
            "MicroMix threshold for W2 only on the top 50% routed experts per layer; colder experts remain all-FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
    ))

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        router_affinity_cache,
        config,
        0.25,
        0.25,
        budget_source="router_affinity_weighted_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "router_affinity_weighted_25pct",
            "Router-probability-weighted channel sensitivity with global top-k split at 25% / 25% for W1 and W2.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
    ))

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        router_affinity_cache,
        config,
        0.04,
        0.16,
        budget_source="router_affinity_weighted_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "router_affinity_weighted_w1_4_w2_16",
            "Router-probability-weighted channel sensitivity with W1=4% and W2=16% global sort-and-split budgets.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
    ))

    w1_masks, w2_masks, extras = build_cross_expert_outlier_masks(micromix_mean_abs, routing_counts, config, agreement_threshold=0.50)
    plans.append((
        build_plan_from_masks(
            "cross_expert_outlier_agreement_5pct",
            "Protect W2 channels that land in the top 5% mean-|activation| set for more than half of active experts in a layer; W1 stays FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
    ))

    w1_masks, w2_masks, extras = build_cross_expert_outlier_masks(micromix_mean_abs, routing_counts, config, agreement_threshold=0.25)
    plans.append((
        build_plan_from_masks(
            "cross_expert_outlier_10pct",
            "Protect W2 channels that land in the top 5% mean-|activation| set for more than a quarter of active experts in a layer; W1 stays FP4.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
    ))

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        hessian_normalized_cache,
        config,
        0.25,
        0.25,
        w1_expert_scale=routing_scale,
        w2_expert_scale=routing_scale,
        budget_source="routing_scaled_normalized_hessian_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "hessian_normalized_25pct",
            "Per-expert normalized Hessian sensitivity with routing-scaled global top-k allocation at 25% / 25%.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
    ))

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        hessian_normalized_cache,
        config,
        0.04,
        0.16,
        w1_expert_scale=routing_scale,
        w2_expert_scale=routing_scale,
        budget_source="routing_scaled_normalized_hessian_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "hessian_normalized_w1_4_w2_16",
            "Per-expert normalized Hessian sensitivity with routing-scaled global top-k allocation at W1=4% and W2=16%.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        extras,
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
            "metric_cache_path": str(args.metric_cache_path),
            "references": references,
            "experiment": "Iteration 10 novel per-channel techniques from 2025-2026 mixed-FP4/FP8 MoE ideas",
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

    router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = (None, None, None, None, None)
    if not args.force_recompute_metrics:
        router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = load_metric_cache(
            args.metric_cache_path,
            args.model_id,
        )
        if (
            router_affinity_cache is not None
            and hessian_normalized_cache is not None
            and micromix_mean_abs is not None
            and micromix_thresholds is not None
            and metric_cache_meta is not None
        ):
            print(f"[metric-cache] loaded {args.metric_cache_path}", flush=True)

    if (
        router_affinity_cache is None
        or hessian_normalized_cache is None
        or micromix_mean_abs is None
        or micromix_thresholds is None
        or metric_cache_meta is None
    ):
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        router_affinity_cache, hessian_normalized_cache, micromix_mean_abs, micromix_thresholds, metric_cache_meta = compute_novel_metric_artifacts(
            store,
            text_config,
            calib_chunks,
            device,
            dtype,
        )
        save_metric_cache(
            args.metric_cache_path,
            args.model_id,
            router_affinity_cache,
            hessian_normalized_cache,
            micromix_mean_abs,
            micromix_thresholds,
            metric_cache_meta,
        )
        print(f"[metric-cache] saved {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache"] = metric_cache_meta
    payload["metadata"]["runtime_seconds_pre_eval"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(
        router_affinity_cache,
        hessian_normalized_cache,
        micromix_mean_abs,
        micromix_thresholds,
        text_config,
        non_expert_bytes,
        total_expert_elems,
    )
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

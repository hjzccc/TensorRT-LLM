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
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    layer_type_at,
    load_gptq_standard_data,
)
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter05 import build_joint_precision_promotions
from proper_iter10_novel_perchannel import build_global_fraction_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan, resolve_requested_plans
from proper_iter14 import (
    build_joint_router_affinity_topup_masks,
    build_mxmoe_router_affinity_topup_masks,
    build_union_base_router_affinity_topup_masks,
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter17_agq.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"
DEFAULT_METRIC_CACHE_PATH = RESULTS_DIR / "proper_iter17_agq_metric_cache.pt"
METRIC_CACHE_VERSION = 1
EPS = 1e-10


@dataclass
class AgqMomentState:
    counts: torch.Tensor
    input_affinity_sq_sum: torch.Tensor
    input_affinity_weight_sum: torch.Tensor
    inter_affinity_sq_sum: torch.Tensor
    inter_affinity_weight_sum: torch.Tensor


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
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 17 AGQ plans.",
    )
    parser.add_argument("--force-recompute-metrics", action="store_true")
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
        restored[int(raw_layer_idx)] = LayerMetricBundle(
            routing_counts=torch.as_tensor(layer_payload["routing_counts"]).detach().cpu().clone().to(torch.int64),
            w1_pair_scores=torch.as_tensor(layer_payload["w1_pair_scores"]).detach().cpu().clone().to(torch.float32),
            w2_channel_scores=torch.as_tensor(layer_payload["w2_channel_scores"]).detach().cpu().clone().to(torch.float32),
        )
    return restored


def metric_cache_is_compatible(payload: dict[str, Any], model_id: str) -> bool:
    return (
        int(payload.get("cache_version", -1)) == METRIC_CACHE_VERSION
        and str(payload.get("model_id", "")) == model_id
        and int(payload.get("seqlen", -1)) == SEQLEN
        and int(payload.get("calibration_samples", -1)) == CALIBRATION_SAMPLES
    )


def load_metric_cache(path: Path, model_id: str) -> tuple[dict[int, LayerMetricBundle] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not metric_cache_is_compatible(payload, model_id):
        return None, None
    agq_cache = restore_metric_cache_bundles(payload.get("agq_cache"))
    metadata = payload.get("metadata")
    if agq_cache is None or not isinstance(metadata, dict):
        return None, None
    return agq_cache, dict(metadata)


def save_metric_cache(path: Path, model_id: str, agq_cache: dict[int, LayerMetricBundle], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": METRIC_CACHE_VERSION,
            "model_id": model_id,
            "seqlen": SEQLEN,
            "calibration_samples": CALIBRATION_SAMPLES,
            "agq_cache": serialize_metric_cache_bundles(agq_cache),
            "metadata": metadata,
        },
        path,
    )


def init_agq_state(config: Any, device: torch.device) -> AgqMomentState:
    return AgqMomentState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        input_affinity_sq_sum=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        input_affinity_weight_sum=torch.zeros(config.num_experts, dtype=torch.float32, device=device),
        inter_affinity_sq_sum=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        inter_affinity_weight_sum=torch.zeros(config.num_experts, dtype=torch.float32, device=device),
    )


def collect_layer_agq_metrics(
    hidden_states: torch.Tensor,
    moe_tensors: dict[str, torch.Tensor],
    config: Any,
    state: AgqMomentState,
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
        route_weight = routing_weights[token_idx, route_pos].unsqueeze(-1)
        affinity = route_weight.float().square()

        current_sq = current_state.float().square()
        state.input_affinity_sq_sum[expert_idx].add_((affinity * current_sq).sum(dim=0))
        state.input_affinity_weight_sum[expert_idx].add_(affinity.sum())

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        full_hidden_sq = full_hidden.float().square()
        state.inter_affinity_sq_sum[expert_idx].add_((affinity * full_hidden_sq).sum(dim=0))
        state.inter_affinity_weight_sum[expert_idx].add_(affinity.sum())

        full_out = F.linear(full_hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (route_weight * full_out.float()).to(hidden_states.dtype))

    shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def finalize_agq_layer(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: AgqMomentState,
    config: Any,
) -> LayerMetricBundle:
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=gate_up_proj.device)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=gate_up_proj.device)

    for expert_idx in range(config.num_experts):
        if int(state.counts[expert_idx].item()) <= 0:
            continue

        input_den = max(float(state.input_affinity_weight_sum[expert_idx].item()), EPS)
        inter_den = max(float(state.inter_affinity_weight_sum[expert_idx].item()), EPS)
        affinity_input_h_diag = state.input_affinity_sq_sum[expert_idx] / input_den
        affinity_inter_h_diag = state.inter_affinity_sq_sum[expert_idx] / inter_den

        gate_up_diff = gate_up_proj[expert_idx].float() - fp4_gate_up[expert_idx].float()
        gate_diff_sq = gate_up_diff[: config.moe_intermediate_size].square()
        up_diff_sq = gate_up_diff[config.moe_intermediate_size :].square()
        w1_scores[expert_idx] = torch.matmul(gate_diff_sq + up_diff_sq, affinity_input_h_diag)

        down_diff_sq = (down_proj[expert_idx].float() - fp4_down[expert_idx].float()).square()
        w2_scores[expert_idx] = torch.matmul(down_diff_sq, affinity_inter_h_diag)

    return LayerMetricBundle(
        routing_counts=state.counts.detach().cpu().to(torch.int64),
        w1_pair_scores=w1_scores.detach().cpu().to(torch.float32),
        w2_channel_scores=w2_scores.detach().cpu().to(torch.float32),
    )


@torch.inference_mode()
def compute_agq_metric_artifacts(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, LayerMetricBundle], dict[str, Any]]:
    print("\n=== Iteration 17 calibration: AGQ affinity-weighted Hessian ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    agq_cache: dict[int, LayerMetricBundle] = {}
    start_time = time.time()

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[agq] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_agq_state(config, device)

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
            moe_out = collect_layer_agq_metrics(mlp_input, moe_tensors, config, state)
            outs[chunk_idx] = residual + moe_out

            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[agq] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        agq_cache[layer_idx] = finalize_agq_layer(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    metadata = {
        "metric_name": "agq_affinity_weighted_hessian",
        "runtime_seconds": round(time.time() - start_time, 3),
        "calibration_chunks": int(calib_chunks.shape[0]),
        "affinity_formula": "g_sq = normalized_routing_weight^2 over routed top-k assignments",
        "affinity_hessian_diag": "sum(g_sq * x^2) / sum(g_sq)",
        "qerror_formula": "sum_k affinity_hessian_diag[k] * (W[j,k] - Q(W[j,k]))^2",
    }
    return agq_cache, metadata


def build_joint_agq_masks(
    calibration: CalibrationArtifacts,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    topup_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    w1_proj, w2_proj, tier_meta = build_joint_precision_promotions(calibration, config)
    w1_pair_masks = {
        layer_idx: {expert_idx: torch.zeros(config.moe_intermediate_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }
    w2_channel_masks = {
        layer_idx: {expert_idx: torch.zeros(config.hidden_size, dtype=torch.bool) for expert_idx in range(config.num_experts)}
        for layer_idx in range(config.num_hidden_layers)
    }
    medium_w1_pairs = int(round(topup_fraction * config.moe_intermediate_size))
    medium_w2_channels = int(round(topup_fraction * config.hidden_size))

    for layer_idx in range(config.num_hidden_layers):
        bundle = metric_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            if bool(w1_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = torch.ones(config.moe_intermediate_size, dtype=torch.bool)
                w2_channel_masks[layer_idx][expert_idx] = torch.ones(config.hidden_size, dtype=torch.bool)
            elif bool(w2_proj[layer_idx][expert_idx]):
                w1_pair_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], medium_w1_pairs)
                w2_channel_masks[layer_idx][expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], medium_w2_channels)

    return w1_pair_masks, w2_channel_masks, {
        **tier_meta,
        "medium_tier_w1_fraction": float(topup_fraction),
        "medium_tier_w2_fraction": float(topup_fraction),
        "medium_tier_w1_pairs": int(medium_w1_pairs),
        "medium_tier_w2_channels": int(medium_w2_channels),
        "budget_source": "joint_percentile_buckets_with_agq_medium_topup",
    }


def load_reference_rows() -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in sorted(RESULTS_DIR.glob("proper*.json")):
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
    best_ref = min((row["ppl"] for row in references.values()), default=None)
    mxmoe_ref = references.get("mxmoe_per_block", {}).get("ppl")
    print("\n" + "=" * 186, flush=True)
    print("proper_iter17_agq | affinity-weighted Hessian channel ranking | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 186, flush=True)
    print(
        f"{'Config':<44} {'PPL':>10} {'dMxMoE':>10} {'dBest':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 186, flush=True)
    for name, row in ordered:
        d_mxmoe = "-" if mxmoe_ref is None else f"{float(row['ppl']) - mxmoe_ref:+.4f}"
        d_best = "-" if best_ref is None else f"{float(row['ppl']) - best_ref:+.4f}"
        print(
            f"{name:<44} {float(row['ppl']):>10.4f} {d_mxmoe:>10} {d_best:>10} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_iteration_plans(
    agq_cache: dict[int, LayerMetricBundle],
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    metric_meta = {
        "channel_metric_requested": "agq_affinity_weighted_qerror",
        "channel_metric_effective": "agq_affinity_weighted_qerror",
        "channel_metric_fallback_used": False,
    }

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        agq_cache,
        config,
        0.10,
        0.10,
        budget_source="agq_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "agq_perchannel_20pct",
            "AGQ affinity-weighted Hessian per-channel plan with a 20% total budget split evenly across W1 and W2 global top-k masks.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**metric_meta, **extras, "total_budget_fraction": 0.20, "per_projection_split": "10:10"},
    ))

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        agq_cache,
        config,
        0.04,
        0.16,
        budget_source="agq_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "agq_perchannel_w1_4_w2_16",
            "AGQ affinity-weighted Hessian per-channel plan with W1=4% and W2=16% global sort-and-split budgets.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**metric_meta, **extras, "total_budget_fraction": 0.20, "per_projection_split": "4:16"},
    ))

    w1_masks, w2_masks, extras = build_joint_agq_masks(calibration, agq_cache, config, topup_fraction=0.08)
    plans.append((
        build_plan_from_masks(
            "agq_joint_w1w2",
            "Reuse the current joint three-tier perturbation buckets, but replace the medium-tier channel ranking with AGQ affinity-weighted Hessian scores at the original 8% topup size.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**metric_meta, **extras},
    ))

    w1_masks, w2_masks, extras = build_joint_router_affinity_topup_masks(calibration, agq_cache, config, topup_fraction=0.05)
    plans.append((
        build_plan_from_masks(
            "agq_joint_w1w2_topup_5pct",
            "Keep the joint perturbation base from later iterations and replace the 5% topup channel ranking with AGQ affinity-weighted Hessian scores.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**metric_meta, **extras},
    ))

    w1_masks, w2_masks, extras = build_mxmoe_router_affinity_topup_masks(
        calibration,
        agq_cache,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "agq_mxmoe_topup_5pct",
            "Keep the 25% MxMoE per-block base and replace the 5% per-channel topup ranking with AGQ affinity-weighted Hessian scores.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**metric_meta, **extras},
    ))

    w1_masks, w2_masks, extras = build_union_base_router_affinity_topup_masks(
        calibration,
        agq_cache,
        config,
        total_expert_elems,
        topup_fraction=0.05,
    )
    plans.append((
        build_plan_from_masks(
            "agq_union_router_affinity_topup_5pct",
            "Union the MxMoE block base with the joint base, then replace the 5% channel topup ranking with AGQ affinity-weighted Hessian scores.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {**metric_meta, **extras},
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
            "success_targets": {
                "best_overall": 6.5725,
                "best_union_style": 6.5758,
                "best_mxmoe_topup": 6.5763,
                "best_perchannel_vs_mxmoe": 6.5849,
                "best_pure_perchannel": 6.5861,
            },
            "experiment": "Iteration 17 AGQ affinity-weighted Hessian calibration for per-channel and hybrid mask ranking",
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

    agq_cache, metric_cache_meta = (None, None)
    if not args.force_recompute_metrics:
        agq_cache, metric_cache_meta = load_metric_cache(args.metric_cache_path, args.model_id)
        if agq_cache is not None and metric_cache_meta is not None:
            print(f"[cache] loaded AGQ metric cache from {args.metric_cache_path}", flush=True)

    if agq_cache is None or metric_cache_meta is None:
        store = WeightStore(args.model_id, snapshot_dir, weight_map)
        agq_cache, metric_cache_meta = compute_agq_metric_artifacts(store, text_config, calib_chunks, device, dtype)
        save_metric_cache(args.metric_cache_path, args.model_id, agq_cache, metric_cache_meta)
        print(f"[cache] wrote AGQ metric cache to {args.metric_cache_path}", flush=True)

    payload["metadata"]["metric_cache"] = metric_cache_meta
    atomic_json_dump(args.output_json, payload)

    plans = build_iteration_plans(agq_cache, calibration, text_config, non_expert_bytes, total_expert_elems)
    for plan, extras in plans:
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

    payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload.get("results", {}), references)


if __name__ == "__main__":
    main()

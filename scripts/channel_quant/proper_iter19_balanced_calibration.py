#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from baselines_comparison import LayerMetricBundle, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
    upsert_exploration_section,
)
from proper_iter01 import build_mxmoe_topup_masks, build_plan_from_masks
from proper_iter05 import build_mxmoe_projection_promotions_fraction, build_per_block_plan
from proper_iter07 import build_joint_with_topup_masks
from proper_iter10_novel_perchannel import build_global_fraction_masks, compute_novel_metric_artifacts
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter14 import build_union_base_router_affinity_topup_masks
from proper_iter19_union_residual import load_json, load_reference_rows, resolve_requested_plans
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter19_balanced_calibration.json"
DEFAULT_COVERAGE_JSON = RESULTS_DIR / "proper_iter19_calibration_coverage.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
SECTION_MARKER = "## [22] Iteration 19 - Balanced Calibration"

TOP_K = 8
NUM_EXPERTS = 256
BASE_MXMOE_FRACTION = 0.25
MAX_EXTRA_CHUNKS = 64
CANDIDATE_POOL_SIZE = 64
MXMOE_TOPUP_FRACTION = 0.08
JOINT_TOPUP_FRACTION = 0.08
UNION_TOPUP_FRACTION = 0.05
PERCHANNEL_W1_FRACTION = 0.04
PERCHANNEL_W2_FRACTION = 0.16

SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "output_perturbation_mxmoe_topup_router_affinity_5pct": 6.5758,
    "mxmoe_block_plus_channel_topup": 6.5763,
    "router_affinity_weighted_w1_4_w2_16": 6.5796,
    "mxmoe_per_block": 6.5849,
}


@dataclass(frozen=True)
class ChunkPool:
    chunks: torch.Tensor
    source: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--coverage-json", type=Path, default=DEFAULT_COVERAGE_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all balanced-calibration plans.",
    )
    parser.add_argument("--max-extra-chunks", type=int, default=MAX_EXTRA_CHUNKS)
    parser.add_argument("--candidate-pool-size", type=int, default=CANDIDATE_POOL_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_random_chunks(token_ids: torch.Tensor, sample_count: int, seed: int) -> torch.Tensor:
    if sample_count <= 0:
        return torch.empty((0, SEQLEN), dtype=torch.long)
    rng = random.Random(seed)
    max_start = int(token_ids.shape[1]) - SEQLEN - 1
    samples = []
    for _ in range(sample_count):
        start = rng.randint(0, max_start)
        samples.append(token_ids[:, start : start + SEQLEN])
    return torch.cat(samples, dim=0).contiguous()


def load_wikitext_candidate_pool(tokenizer: Any, sample_count: int, seed: int) -> ChunkPool:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_text = "\n\n".join(dataset["text"])
    token_ids = torch.as_tensor(tokenizer(train_text, return_tensors="pt").input_ids, dtype=torch.long)
    chunks = sample_random_chunks(token_ids, sample_count, seed)
    return ChunkPool(
        chunks=chunks,
        source="wikitext_train_fallback",
        metadata={
            "dataset": "wikitext/wikitext-2-raw-v1",
            "split": "train",
            "seed": int(seed),
            "sample_count": int(sample_count),
        },
    )


def load_c4_candidate_pool(tokenizer: Any, sample_count: int, seed: int) -> ChunkPool:
    rng = random.Random(seed)
    chunks: list[torch.Tensor] = []
    scanned = 0
    max_examples = max(512, sample_count * 32)
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    for example in dataset:
        scanned += 1
        text = str(example.get("text", "")).strip()
        if not text:
            if scanned >= max_examples:
                break
            continue
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.as_tensor(encoded.input_ids, dtype=torch.long)
        if int(input_ids.shape[1]) < SEQLEN:
            if scanned >= max_examples:
                break
            continue
        max_start = int(input_ids.shape[1]) - SEQLEN
        start = 0 if max_start <= 0 else rng.randint(0, max_start)
        chunks.append(input_ids[:, start : start + SEQLEN])
        if len(chunks) >= sample_count or scanned >= max_examples:
            break
    if len(chunks) < sample_count:
        raise RuntimeError(f"C4 yielded only {len(chunks)} chunks for target {sample_count}")
    return ChunkPool(
        chunks=torch.cat(chunks, dim=0).contiguous(),
        source="c4_train",
        metadata={
            "dataset": "allenai/c4",
            "config": "en",
            "split": "train",
            "streaming": True,
            "seed": int(seed),
            "sample_count": int(sample_count),
            "examples_scanned": int(scanned),
        },
    )


def load_candidate_pool(tokenizer: Any, sample_count: int, seed: int) -> ChunkPool:
    try:
        pool = load_c4_candidate_pool(tokenizer, sample_count, seed)
        print(f"[candidate-pool] loaded {pool.chunks.shape[0]} chunks from C4", flush=True)
        return pool
    except Exception as exc:
        print(f"[candidate-pool] C4 unavailable, falling back to WikiText-2 train: {exc}", flush=True)
        pool = load_wikitext_candidate_pool(tokenizer, sample_count, seed + 17)
        print(f"[candidate-pool] loaded {pool.chunks.shape[0]} fallback chunks from WikiText-2 train", flush=True)
        return pool


def moe_forward_profile(hidden_states: torch.Tensor, tensors: dict[str, torch.Tensor], config: Any) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, tensors["experts.down_proj"][expert_idx])
        weighted = routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden
        final_hidden_states.index_add_(0, token_idx, weighted.to(hidden_states.dtype))

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), expert_counts.detach().cpu().to(torch.int64)


@torch.inference_mode()
def profile_routing_chunks(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    print(f"\n=== Routing profile ({int(calib_chunks.shape[0])} chunks) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)
    per_chunk_counts = torch.zeros((int(inps.shape[0]), config.num_hidden_layers, config.num_experts), dtype=torch.int64)

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[profile] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

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
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out, expert_counts = moe_forward_profile(mlp_input, moe_tensors, config)
            per_chunk_counts[chunk_idx, layer_idx] = expert_counts
            outs[chunk_idx] = residual + moe_out
            if chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == inps.shape[0]:
                print(f"[profile] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return per_chunk_counts


def coverage_target(total_chunks: int) -> float:
    total_tokens = int(total_chunks) * SEQLEN
    return float(2.0 * TOP_K * total_tokens) / float(NUM_EXPERTS)


def flatten_top_undercovered(counts: torch.Tensor, target: float, limit: int = 20) -> list[dict[str, Any]]:
    deficits = torch.clamp(torch.full_like(counts, float(target), dtype=torch.float32) - counts.to(torch.float32), min=0.0)
    flat = deficits.view(-1)
    if int(flat.numel()) == 0:
        return []
    values, indices = torch.topk(flat, k=min(limit, int(flat.numel())))
    rows: list[dict[str, Any]] = []
    experts_per_layer = int(counts.shape[1])
    for deficit, flat_idx in zip(values.tolist(), indices.tolist()):
        if float(deficit) <= 0.0:
            continue
        layer_idx = int(flat_idx // experts_per_layer)
        expert_idx = int(flat_idx % experts_per_layer)
        count = int(counts[layer_idx, expert_idx].item())
        rows.append({
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "count": count,
            "deficit": round(float(deficit), 3),
        })
    return rows


def summarize_coverage(label: str, counts: torch.Tensor, total_chunks: int) -> dict[str, Any]:
    target = coverage_target(total_chunks)
    counts_f = counts.to(torch.float32)
    deficits = torch.clamp(target - counts_f, min=0.0)
    below = counts_f < target
    per_layer: dict[str, Any] = {}
    for layer_idx in range(int(counts.shape[0])):
        layer_counts = counts_f[layer_idx]
        layer_deficits = deficits[layer_idx]
        layer_below = below[layer_idx]
        per_layer[str(layer_idx)] = {
            "min": int(layer_counts.min().item()),
            "mean": round(float(layer_counts.mean().item()), 3),
            "median": round(float(layer_counts.median().item()), 3),
            "max": int(layer_counts.max().item()),
            "below_target": int(layer_below.sum().item()),
            "below_target_fraction": round(float(layer_below.to(torch.float32).mean().item()), 6),
            "worst_deficit": round(float(layer_deficits.max().item()), 3),
        }
    return {
        "label": label,
        "total_chunks": int(total_chunks),
        "total_tokens": int(total_chunks * SEQLEN),
        "top_k": TOP_K,
        "num_experts": NUM_EXPERTS,
        "expected_per_expert": round(float(target), 3),
        "min_count": int(counts_f.min().item()),
        "mean_count": round(float(counts_f.mean().item()), 3),
        "median_count": round(float(counts_f.median().item()), 3),
        "max_count": int(counts_f.max().item()),
        "experts_below_target": int(below.sum().item()),
        "experts_total": int(counts.numel()),
        "below_target_fraction": round(float(below.to(torch.float32).mean().item()), 6),
        "total_deficit": round(float(deficits.sum().item()), 3),
        "mean_deficit": round(float(deficits.mean().item()), 3),
        "worst_deficit": round(float(deficits.max().item()), 3),
        "top_undercovered": flatten_top_undercovered(counts, target),
        "per_layer": per_layer,
    }


def select_balanced_chunks(base_profile: torch.Tensor, candidate_profile: torch.Tensor, max_extra_chunks: int) -> tuple[list[int], list[dict[str, Any]], torch.Tensor]:
    selected: list[int] = []
    selection_trace: list[dict[str, Any]] = []
    current_counts = base_profile.sum(dim=0).to(torch.float32)
    available = set(range(int(candidate_profile.shape[0])))

    while available and len(selected) < max_extra_chunks:
        current_total_chunks = int(base_profile.shape[0]) + len(selected)
        target = coverage_target(current_total_chunks)
        deficits = torch.clamp(target - current_counts, min=0.0)
        if float(deficits.max().item()) <= 0.0:
            break
        best_idx = None
        best_gain = 0.0
        best_after_counts = None
        next_target = coverage_target(current_total_chunks + 1)
        current_total_deficit = float(deficits.sum().item())
        for idx in available:
            after_counts = current_counts + candidate_profile[idx].to(torch.float32)
            after_deficits = torch.clamp(next_target - after_counts, min=0.0)
            gain = current_total_deficit - float(after_deficits.sum().item())
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_after_counts = after_counts
        if best_idx is None or best_gain <= 0.0 or best_after_counts is None:
            break
        prev_worst = float(deficits.max().item())
        prev_total = float(deficits.sum().item())
        current_counts = best_after_counts
        new_deficits = torch.clamp(next_target - current_counts, min=0.0)
        selection_trace.append(
            {
                "candidate_index": int(best_idx),
                "deficit_gain": round(float(best_gain), 3),
                "worst_deficit_before": round(prev_worst, 3),
                "worst_deficit_after": round(float(new_deficits.max().item()), 3),
                "total_deficit_before": round(prev_total, 3),
                "total_deficit_after": round(float(new_deficits.sum().item()), 3),
            }
        )
        selected.append(int(best_idx))
        available.remove(best_idx)
    return selected, selection_trace, current_counts.to(torch.int64)


def build_balanced_plans(
    calibration: CalibrationArtifacts,
    router_affinity_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    plans: list[tuple[EvalPlan, dict[str, Any]]] = []
    channel_metric_meta = {
        "channel_metric_requested": "router_affinity_weighted_qerror",
        "channel_metric_effective": "router_affinity_weighted_qerror",
        "channel_metric_fallback_used": False,
        "balanced_calibration": True,
    }

    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions_fraction(
        config,
        total_expert_elems,
        calibration.mxmoe_w1_deltas,
        calibration.mxmoe_w2_deltas,
        BASE_MXMOE_FRACTION,
    )
    plans.append((
        build_per_block_plan(
            "balanced_mxmoe_block",
            "Rebuild the standard 25% MxMoE per-block output-perturbation promotion plan using the balanced calibration set.",
            config,
            non_expert_bytes,
            total_expert_elems,
            mxmoe_w1_proj,
            mxmoe_w2_proj,
        ),
        {
            "budget_source": "mxmoe_per_block",
            "mxmoe_base_fraction": float(BASE_MXMOE_FRACTION),
            "mxmoe_promoted_w1_projections": int(sum(int(mask.sum().item()) for mask in mxmoe_w1_proj.values())),
            "mxmoe_promoted_w2_projections": int(sum(int(mask.sum().item()) for mask in mxmoe_w2_proj.values())),
            "balanced_calibration": True,
        },
    ))

    w1_masks, w2_masks, extras = build_mxmoe_topup_masks(calibration, config, total_expert_elems, MXMOE_TOPUP_FRACTION)
    plans.append((
        build_plan_from_masks(
            "balanced_mxmoe_topup_8pct",
            "Rebuild the MxMoE block-plus-topup family on the balanced calibration set, using 8% within-expert activation-ranked W1 pairs and W2 channels on remaining FP4 projections.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **extras,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "source_base": "mxmoe_block_plus_channel_topup",
            "balanced_calibration": True,
        },
    ))

    w1_masks, w2_masks, extras = build_joint_with_topup_masks(calibration, calibration.activation_cache, config, JOINT_TOPUP_FRACTION)
    plans.append((
        build_plan_from_masks(
            "balanced_joint_w1w2_topup",
            "Rebuild the joint three-tier + 8% within-expert topup plan from the balanced calibration set.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **extras,
            "channel_metric_requested": "activation_kurtosis",
            "channel_metric_effective": "activation_kurtosis",
            "channel_metric_fallback_used": False,
            "source_base": "joint_w1w2_with_topup",
            "balanced_calibration": True,
        },
    ))

    w1_masks, w2_masks, extras = build_union_base_router_affinity_topup_masks(
        calibration,
        router_affinity_cache,
        config,
        total_expert_elems,
        topup_fraction=UNION_TOPUP_FRACTION,
    )
    plans.append((
        build_plan_from_masks(
            "balanced_union_router_affinity_topup_5pct",
            "Rebuild the union MxMoE-plus-joint base with 5% router-affinity topups from the balanced calibration set.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **channel_metric_meta,
            **extras,
            "source_base": "output_perturbation_mxmoe_topup_router_affinity_5pct",
        },
    ))

    w1_masks, w2_masks, extras = build_global_fraction_masks(
        router_affinity_cache,
        config,
        PERCHANNEL_W1_FRACTION,
        PERCHANNEL_W2_FRACTION,
        budget_source="router_affinity_weighted_global_topk",
    )
    plans.append((
        build_plan_from_masks(
            "balanced_router_affinity_w1_4_w2_16",
            "Rebuild the pure router-affinity global sort-and-split plan at W1=4% and W2=16% using the balanced calibration set.",
            config,
            non_expert_bytes,
            total_expert_elems,
            w1_masks,
            w2_masks,
        ),
        {
            **channel_metric_meta,
            **extras,
            "source_base": "router_affinity_weighted_w1_4_w2_16",
        },
    ))
    return plans


def format_result_table(results: dict[str, Any]) -> str:
    lines = [
        "| Config | PPL | Memory (GB) | FP8 frac | W1 frac | W2 frac |",
        "|--------|-----|-------------|----------|---------|---------|",
    ]
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    for name, row in ordered:
        lines.append(
            f"| {name} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} | {float(row['fp8_fraction']):.4f} | {float(row['w1_pair_fraction']):.4f} | {float(row['w2_channel_fraction']):.4f} |"
        )
    return "\n".join(lines)


def best_gain_text(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    improvements: list[tuple[str, str, float]] = []
    alias_map = {
        "balanced_mxmoe_block": "mxmoe_per_block",
        "balanced_mxmoe_topup_8pct": "mxmoe_block_plus_channel_topup",
        "balanced_joint_w1w2_topup": "joint_w1w2_with_topup",
        "balanced_union_router_affinity_topup_5pct": "output_perturbation_mxmoe_topup_router_affinity_5pct",
        "balanced_router_affinity_w1_4_w2_16": "router_affinity_weighted_w1_4_w2_16",
    }
    for new_name, ref_name in alias_map.items():
        if new_name not in results or ref_name not in references:
            continue
        delta = float(results[new_name]["ppl"]) - float(references[ref_name]["ppl"])
        improvements.append((new_name, ref_name, delta))
    if not improvements:
        return "No comparable historical reference rows were available for the rebuilt balanced-calibration plans."
    best_name, ref_name, best_delta = min(improvements, key=lambda item: item[2])
    return f"Relative to its historical counterpart `{ref_name}`, the biggest balanced-calibration gain is `{best_name}` at {best_delta:+.4f} PPL."


def render_exploration_section(results: dict[str, Any], coverage_payload: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    before = coverage_payload["before"]["summary"]
    after = coverage_payload["after"]["summary"]
    table = format_result_table(results)
    coverage_text = (
        f"below-target expert slots {before['experts_below_target']} -> {after['experts_below_target']}, "
        f"worst deficit {before['worst_deficit']:.1f} -> {after['worst_deficit']:.1f}, "
        f"mean count {before['mean_count']:.1f} -> {after['mean_count']:.1f}"
    )
    return (
        f"{SECTION_MARKER}\n"
        f"**Approach**: Started from the standard 128 WikiText-2 train chunks, profiled per-layer expert routing coverage, greedily added up to {coverage_payload['augmentation']['max_extra_chunks']} extra chunks from {coverage_payload['augmentation']['source']} to reduce under-covered experts, then rebuilt the strongest MxMoE/joint/router-affinity mask families on the balanced calibration set while keeping the full `proper_eval.py` test path unchanged.\n"
        f"**Coverage**: {coverage_text}.\n"
        f"**Result**:\n{table}\n"
        f"**Insight**: {best_gain_text(results, references)}\n"
        f"**Next**: If the best balanced plan still plateaus, combine balanced calibration with a stronger channel-ranking signal instead of changing the eval path again."
    )


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    print("\n" + "=" * 178, flush=True)
    print("proper_iter19_balanced_calibration | balanced calibration rebuilds | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 178, flush=True)
    print(
        f"{'Config':<44} {'PPL':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 178, flush=True)
    for name, row in ordered:
        print(
            f"{name:<44} {float(row['ppl']):>10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
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
    tokenizer, base_calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    output_payload: dict[str, Any] = {
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
            "experiment": "Iteration 19 balanced calibration / rare-expert coverage rebuilds",
            "success_targets": SUCCESS_TARGETS,
        },
        "results": {},
    }
    if args.output_json.exists():
        try:
            existing = load_json(args.output_json)
            if isinstance(existing, dict):
                merged_metadata = dict(output_payload["metadata"])
                merged_metadata.update(existing.get("metadata", {}))
                output_payload.update(existing)
                output_payload["metadata"] = merged_metadata
                output_payload.setdefault("results", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, output_payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    base_profile = profile_routing_chunks(store, text_config, base_calib_chunks, device, dtype)
    candidate_pool = load_candidate_pool(tokenizer, args.candidate_pool_size, args.seed + 101)
    candidate_profile = profile_routing_chunks(store, text_config, candidate_pool.chunks, device, dtype)

    selected_indices, selection_trace, balanced_counts = select_balanced_chunks(base_profile, candidate_profile, args.max_extra_chunks)
    extra_chunks = candidate_pool.chunks[selected_indices] if selected_indices else torch.empty((0, SEQLEN), dtype=torch.long)
    balanced_calib_chunks = torch.cat([base_calib_chunks, extra_chunks], dim=0).contiguous()
    balanced_total_chunks = int(balanced_calib_chunks.shape[0])

    coverage_payload = {
        "metadata": {
            "model": args.model_id,
            "seed": int(args.seed),
            "seqlen": SEQLEN,
            "top_k": TOP_K,
            "num_experts": NUM_EXPERTS,
        },
        "before": {
            "summary": summarize_coverage("before", base_profile.sum(dim=0), int(base_calib_chunks.shape[0])),
        },
        "after": {
            "summary": summarize_coverage("after", balanced_counts, balanced_total_chunks),
        },
        "augmentation": {
            "source": candidate_pool.source,
            "source_metadata": candidate_pool.metadata,
            "candidate_pool_chunks": int(candidate_pool.chunks.shape[0]),
            "max_extra_chunks": int(args.max_extra_chunks),
            "selected_extra_chunks": int(len(selected_indices)),
            "selected_candidate_indices": [int(idx) for idx in selected_indices],
            "selection_trace": selection_trace,
        },
    }
    atomic_json_dump(args.coverage_json, coverage_payload)
    print(f"[coverage] saved {args.coverage_json}", flush=True)

    output_payload["metadata"]["balanced_calibration"] = {
        "base_chunks": int(base_calib_chunks.shape[0]),
        "extra_chunks": int(len(selected_indices)),
        "total_chunks": int(balanced_total_chunks),
        "candidate_pool_source": candidate_pool.source,
        "candidate_pool_chunks": int(candidate_pool.chunks.shape[0]),
        "coverage_json": str(args.coverage_json),
    }
    atomic_json_dump(args.output_json, output_payload)

    calibration = run_calibration(store, text_config, balanced_calib_chunks, device, dtype)
    router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, metric_cache_meta = compute_novel_metric_artifacts(
        store,
        text_config,
        balanced_calib_chunks,
        device,
        dtype,
    )
    output_payload["metadata"]["metric_cache"] = metric_cache_meta
    output_payload["metadata"]["calibration_summary"] = {
        "active_experts_per_layer": {str(layer_idx): int((counts > 0).sum().item()) for layer_idx, counts in calibration.routing_counts.items()},
        "routing_top5_per_layer": {
            str(layer_idx): [
                {"expert_idx": int(expert_idx), "count": int(count)}
                for count, expert_idx in zip(
                    torch.topk(counts, k=min(5, counts.numel())).values.tolist(),
                    torch.topk(counts, k=min(5, counts.numel())).indices.tolist(),
                )
            ]
            for layer_idx, counts in calibration.routing_counts.items()
        },
    }
    atomic_json_dump(args.output_json, output_payload)

    plans = build_balanced_plans(calibration, router_affinity_cache, text_config, non_expert_bytes, total_expert_elems)
    if requested_plans is not None:
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        missing = sorted(requested_plans.difference({plan.name for plan, _extras in plans}))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")

    for idx, (plan, extras) in enumerate(plans, start=1):
        print(f"\n=== Eval {idx}/{len(plans)}: {plan.name} ===", flush=True)
        evaluate_and_record_plan(
            output_payload,
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

    references = load_reference_rows(args.output_json)
    section = render_exploration_section(output_payload["results"], coverage_payload, references)
    upsert_exploration_section(args.exploration_md, section)
    output_payload["metadata"]["references"] = references
    output_payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, output_payload)
    print_results_table(output_payload["results"])
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Saved coverage -> {args.coverage_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

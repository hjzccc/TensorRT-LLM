#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 32 (Fast): Expert-Balanced Self-Sampling (EBSS) Calibration

Fast version using BF16-only routing pass (no FP4 allocation).

Key finding: Per-layer expert imbalance is extreme (up to 108,338× ratio).
Layers 34, 39 have experts with ZERO activations in standard calibration.
Cold experts get near-zero Hessian signal → wrong sensitivity scores.

Strategy:
1. BF16 routing pass over 256 chunks (4096 tokens) — no FP4 allocation
2. Greedy selection of 64 chunks maximizing cold expert coverage
3. Hybrid calibration: 64 cold-expert chunks + 64 standard 2048-token chunks
4. Build masks with joint_w1w2_with_topup
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    atomic_json_dump,
    build_position_context,
    collect_layer_calibration,
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    finalize_layer_metrics,
    init_expert_moment_state,
    layer_type_at,
    load_gptq_standard_data,
    moe_forward_eval,
    run_calibration,
)
from proper_iter01 import build_plan_from_masks, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter21_serq_salient import load_reference_rows
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter32_ebss_fast.json"

POOL_SIZE = 256
TARGET_COLD_CHUNKS = 64
TARGET_STANDARD_CHUNKS = 64
CHUNK_LENGTH = 4096
EBSS_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="")
    return parser.parse_args()


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def load_wikitext_train_ids(tokenizer: Any) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])  # type: ignore[index]
    return tokenizer(text, return_tensors="pt").input_ids


def build_chunks(tokenizer: Any, n: int, chunk_length: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    train_ids = load_wikitext_train_ids(tokenizer)
    max_start = int(train_ids.shape[1]) - chunk_length - 1
    chunks = [train_ids[:, rng.randint(0, max_start) : rng.randint(0, max_start) + chunk_length] for _ in range(n)]
    # Fix: use consistent start
    rng2 = random.Random(seed)
    chunks2 = []
    for _ in range(n):
        start = rng2.randint(0, max_start)
        chunks2.append(train_ids[:, start : start + chunk_length])
    return torch.cat(chunks2, dim=0).contiguous()


def bf16_routing_pass(
    store: WeightStore,
    config: Any,
    pool_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    BF16-only routing pass — no FP4 allocation, much lower memory.
    Returns: [pool_size, num_layers, num_experts]
    """
    pool_size = pool_chunks.shape[0]
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts

    expert_activations = torch.zeros(pool_size, num_layers, num_experts, dtype=torch.int32)

    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, pool_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    for layer_idx in range(num_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[routing] layer {layer_idx + 1}/{num_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        for chunk_idx in range(pool_size):
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

            # BF16 routing — record expert activations
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            flat = mlp_input.view(-1, mlp_input.shape[-1])
            router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
            _, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), config.num_experts_per_tok, dim=-1)
            expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
            expert_activations[chunk_idx, layer_idx] = expert_counts.cpu().to(torch.int32)

            # BF16 MoE forward (no FP4 allocation)
            moe_out = moe_forward_eval(mlp_input, moe_tensors, config)
            outs[chunk_idx] = residual + moe_out

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return expert_activations


def select_cold_expert_chunks(
    expert_activations: torch.Tensor,
    target_chunks: int,
    num_experts: int,
) -> list[int]:
    pool_size = expert_activations.shape[0]
    chunk_expert_coverage = expert_activations.sum(dim=1).float()
    global_activation = chunk_expert_coverage.sum(dim=0)
    median_activation = global_activation.median()
    cold_experts = (global_activation < median_activation).nonzero(as_tuple=True)[0]

    print(f"[EBSS] Cold experts: {len(cold_experts)}/{num_experts}, median={median_activation:.1f}", flush=True)
    print(f"[EBSS] Min activation: {global_activation.min():.1f}, max: {global_activation.max():.1f}", flush=True)

    selected: list[int] = []
    remaining_cold = set(cold_experts.tolist())
    available = set(range(pool_size))

    while len(selected) < target_chunks and remaining_cold:
        cold_list = list(remaining_cold)
        best_chunk = max(available, key=lambda c: int((chunk_expert_coverage[c, cold_list] > 0).sum().item()))
        score = int((chunk_expert_coverage[best_chunk, cold_list] > 0).sum().item())
        if score == 0:
            break
        selected.append(best_chunk)
        available.discard(best_chunk)
        remaining_cold -= {e for e in remaining_cold if chunk_expert_coverage[best_chunk, e] > 0}

    print(f"[EBSS] Phase 1: {len(selected)} chunks, {len(remaining_cold)} cold experts uncovered", flush=True)

    while len(selected) < target_chunks and available:
        best_chunk = max(available, key=lambda c: float(chunk_expert_coverage[c].sum().item()))
        selected.append(best_chunk)
        available.discard(best_chunk)

    return selected


def run_hybrid_calibration(
    store: WeightStore,
    config: Any,
    hybrid_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    label: str = "hybrid",
) -> CalibrationArtifacts:
    n = hybrid_chunks.shape[0]
    print(f"\n=== {label} Calibration ({n} chunks, {hybrid_chunks.shape[1]}-token) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, hybrid_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[{label}] layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[e], "fp4") for e in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[e], "fp4") for e in range(config.num_experts)], dim=0)
        state = init_expert_moment_state(config, device)

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
            moe_out = collect_layer_calibration(mlp_input, moe_tensors, fp4_gate_up, fp4_down, config, state)
            outs[chunk_idx] = residual + moe_out
            print(f"[{label}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        bundle, layer_w1, layer_w2, layer_mc = finalize_layer_metrics(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        routing_counts[layer_idx] = bundle.routing_counts
        activation_cache[layer_idx] = bundle
        mxmoe_w1_deltas[layer_idx] = layer_w1
        mxmoe_w2_deltas[layer_idx] = layer_w2
        mc_moe_scores[layer_idx] = layer_mc

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return CalibrationArtifacts(
        routing_counts=routing_counts,
        activation_cache=activation_cache,
        mxmoe_w1_deltas=mxmoe_w1_deltas,
        mxmoe_w2_deltas=mxmoe_w2_deltas,
        mc_moe_scores=mc_moe_scores,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print("Loading model config...", flush=True)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)

    import json as _json
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = _json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    print("Loading tokenizer and test data...", flush=True)
    tokenizer, standard_calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)

    reference_rows = load_reference_rows(args.output_json)
    requested_plans = resolve_requested_plans(args.plans)
    results: dict[str, Any] = {}

    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    # =========================================================================
    # Phase 1: Build pool and run BF16 routing pass
    # =========================================================================
    print(f"\n=== Phase 1: Building pool of {POOL_SIZE} chunks ({CHUNK_LENGTH} tokens) ===", flush=True)
    pool_chunks = build_chunks(tokenizer, POOL_SIZE, CHUNK_LENGTH, EBSS_SEED)

    routing_cache = RESULTS_DIR / "iter32_fast_routing_matrix.pt"
    if routing_cache.exists():
        print(f"Loading cached routing matrix from {routing_cache}", flush=True)
        expert_activations = torch.load(routing_cache, weights_only=True)
    else:
        print(f"\n=== Phase 2: BF16 routing pass over {POOL_SIZE} chunks ===", flush=True)
        expert_activations = bf16_routing_pass(store, text_config, pool_chunks, device, dtype)
        torch.save(expert_activations, routing_cache)
        print(f"Saved routing matrix to {routing_cache}", flush=True)

    # =========================================================================
    # Phase 2: Select cold-expert chunks
    # =========================================================================
    print(f"\n=== Phase 3: Selecting {TARGET_COLD_CHUNKS} cold-expert chunks ===", flush=True)
    cold_indices = select_cold_expert_chunks(expert_activations, TARGET_COLD_CHUNKS, text_config.num_experts)
    cold_chunks = pool_chunks[cold_indices]

    # Standard 2048-token chunks padded to 4096
    std_chunks_2048 = standard_calib_chunks[:TARGET_STANDARD_CHUNKS]
    std_chunks_padded = torch.zeros(TARGET_STANDARD_CHUNKS, CHUNK_LENGTH, dtype=torch.long)
    std_chunks_padded[:, :SEQLEN] = std_chunks_2048

    # Hybrid set
    hybrid_chunks = torch.cat([cold_chunks, std_chunks_padded], dim=0)
    print(f"Hybrid: {hybrid_chunks.shape} ({TARGET_COLD_CHUNKS} cold + {TARGET_STANDARD_CHUNKS} standard)", flush=True)

    # Analyze balance
    sel_act = expert_activations[cold_indices].sum(dim=(0, 1)).float()
    rnd_act = expert_activations[:TARGET_COLD_CHUNKS].sum(dim=(0, 1)).float()
    print(f"EBSS cold: min={sel_act.min():.1f} max={sel_act.max():.1f} std={sel_act.std():.1f}", flush=True)
    print(f"Random:    min={rnd_act.min():.1f} max={rnd_act.max():.1f} std={rnd_act.std():.1f}", flush=True)

    # =========================================================================
    # Phase 3: Run calibrations
    # =========================================================================
    hybrid_calibration = run_hybrid_calibration(store, text_config, hybrid_chunks, device, dtype, "ebss-hybrid")

    maca_chunks = build_chunks(tokenizer, 128, CHUNK_LENGTH, seed=0)
    maca_calibration = run_hybrid_calibration(store, text_config, maca_chunks, device, dtype, "maca-4k")

    standard_calibration = run_calibration(store, text_config, standard_calib_chunks, device, dtype)

    # =========================================================================
    # Phase 4: Build masks and evaluate
    # =========================================================================
    configs_to_eval = [
        ("ebss_hybrid_joint_topup", hybrid_calibration, "EBSS hybrid (64 cold + 64 std) + joint W1/W2 topup"),
        ("maca_4k_joint_topup", maca_calibration, "MaCa uniform-4k (128 × 4096) + joint W1/W2 topup"),
        ("standard_joint_topup", standard_calibration, "Standard (128 × 2048) + joint W1/W2 topup"),
    ]

    for plan_name, calibration, description in configs_to_eval:
        if requested_plans and plan_name not in requested_plans:
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {plan_name}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()
        w1_masks, w2_masks = build_joint_with_topup_masks(calibration, text_config, JOINT_MEDIUM_TOPUP_FRACTION)
        plan = build_plan_from_masks(plan_name, description, text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
        ppl, _nll, _nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time

        results[plan_name] = {
            "ppl": round(ppl, 4),
            "memory_gb": round(plan.memory_gb, 3),
            "fp8_weights": int(plan.fp8_weights),
            "w1_fraction": round(total_pair_fraction(w1_masks, text_config), 4),
            "w2_fraction": round(total_channel_fraction(w2_masks, text_config), 4),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"PPL: {ppl:.4f} | Memory: {plan.memory_gb:.3f} GB | Time: {elapsed:.1f}s", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": 296960,
            "seqlen": SEQLEN,
            "nsamples": 145,
            "dtype": args.dtype,
            "approach": "Fast EBSS: Expert-Balanced Self-Sampling (BF16 routing pass)",
            "hypothesis": "Per-layer expert imbalance (up to 108,338×) causes cold experts to get near-zero Hessian signal",
            "pool_size": POOL_SIZE,
            "cold_chunks": TARGET_COLD_CHUNKS,
            "standard_chunks": TARGET_STANDARD_CHUNKS,
            "chunk_length": CHUNK_LENGTH,
            "seed": EBSS_SEED,
            "reference_rows": reference_rows,
        },
        "results": results,
    }
    atomic_json_dump(args.output_json, payload)
    print(f"\nResults saved to {args.output_json}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    for i, (name, result) in enumerate(sorted(results.items(), key=lambda x: x[1].get("ppl", float("inf")))[:5], 1):
        print(f"{i}. {name:50s} PPL={result.get('ppl','N/A')} Mem={result.get('memory_gb','N/A')} GB")


if __name__ == "__main__":
    main()

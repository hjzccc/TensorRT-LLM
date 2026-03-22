#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 32: Expert-Balanced Self-Sampling (EBSS) Calibration

Hypothesis: Standard WikiText-2 calibration under-samples cold experts (256 experts,
Qwen3.5-35B-A3B). Cold experts get insufficient calibration data → biased sensitivity
metrics → wrong FP4/FP8 assignments.

Approach (inspired by MoEQuant ICML 2025 EBSS):
1. Build a pool of 512 WikiText-2 chunks (4096 tokens each, MaCa-style)
2. Run a fast routing-only pass to get expert activation matrix [pool_size, num_layers, num_experts]
3. Greedy set cover: select 128 chunks that maximize cold expert coverage
4. Run full calibration on the balanced 128-chunk set
5. Build masks with joint_w1w2_with_topup strategy

Key difference from standard: chunk selection is routing-aware, not random.
Key difference from AGQ: we balance the CALIBRATION SET, not the quantization loss weighting.

Expected gain: 0.001-0.003 PPL (cold experts currently get ~0 calibration signal)
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
    run_calibration,
)
from proper_iter01 import build_plan_from_masks, load_cache, total_channel_fraction, total_pair_fraction
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter32_ebss_calibration.json"
DEFAULT_CACHE_PATH = RESULTS_DIR / "proper_iter01_calibration_cache.pt"

# EBSS parameters
POOL_SIZE = 512          # Number of chunks to sample routing from
TARGET_CHUNKS = 128      # Final calibration set size
CHUNK_LENGTH = 4096      # MaCa-style 4096-token chunks (best from iter29)
EBSS_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--plans", default="", help="Comma-separated subset of plan names to evaluate.")
    parser.add_argument("--skip-routing-pass", action="store_true", help="Skip routing pass if cache exists.")
    return parser.parse_args()


def load_wikitext_train_ids(tokenizer: Any) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])  # type: ignore[index]
    return tokenizer(text, return_tensors="pt").input_ids


def build_pool_chunks(tokenizer: Any, pool_size: int, chunk_length: int, seed: int) -> torch.Tensor:
    """Build a large pool of chunks for routing analysis."""
    rng = random.Random(seed)
    train_ids = load_wikitext_train_ids(tokenizer)
    max_start = int(train_ids.shape[1]) - chunk_length - 1
    chunks = []
    for _ in range(pool_size):
        start = rng.randint(0, max_start)
        chunks.append(train_ids[:, start : start + chunk_length])
    return torch.cat(chunks, dim=0).contiguous()


def fast_routing_pass(
    store: WeightStore,
    config: Any,
    pool_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Fast pass: compute routing decisions for all chunks.
    Returns: expert_activation_matrix [pool_size, num_layers, num_experts]
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
        print(f"[routing-pass] layer {layer_idx + 1}/{num_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[e], "fp4") for e in range(num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[e], "fp4") for e in range(num_experts)], dim=0)
        state = init_expert_moment_state(config, device)

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

            # Record routing decisions
            router_logits = F.linear(mlp_input, tensors["mlp.gate.weight"])
            routing_weights = F.softmax(router_logits.float(), dim=-1)
            _, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
            expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
            expert_activations[chunk_idx, layer_idx] = expert_counts.cpu().to(torch.int32)

            # Full MoE forward for hidden state propagation
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = collect_layer_calibration(mlp_input, moe_tensors, fp4_gate_up, fp4_down, config, state)
            outs[chunk_idx] = residual + moe_out

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return expert_activations


def select_balanced_chunks(
    expert_activations: torch.Tensor,
    target_chunks: int,
    num_experts: int,
) -> list[int]:
    """
    Greedy set cover: select chunks that maximize cold expert coverage.
    """
    pool_size = expert_activations.shape[0]

    # Sum across layers: [pool_size, num_experts]
    chunk_expert_coverage = expert_activations.sum(dim=1).float()

    # Global expert activation across all chunks: [num_experts]
    global_expert_activation = chunk_expert_coverage.sum(dim=0)

    # Cold experts: below 50th percentile
    median_activation = global_expert_activation.median()
    cold_experts = (global_expert_activation < median_activation).nonzero(as_tuple=True)[0]
    print(f"[EBSS] Cold experts (below median): {len(cold_experts)}/{num_experts}", flush=True)
    print(f"[EBSS] Median activation: {median_activation:.1f}, min: {global_expert_activation.min():.1f}, max: {global_expert_activation.max():.1f}", flush=True)

    selected: list[int] = []
    remaining_cold = set(cold_experts.tolist())
    available = set(range(pool_size))

    # Phase 1: Cover cold experts greedily
    while len(selected) < target_chunks and remaining_cold:
        best_chunk = -1
        best_score = -1
        cold_list = list(remaining_cold)
        for chunk_idx in available:
            activated = int((chunk_expert_coverage[chunk_idx, cold_list] > 0).sum().item())
            if activated > best_score:
                best_score = activated
                best_chunk = chunk_idx
        if best_chunk == -1 or best_score == 0:
            break
        selected.append(best_chunk)
        available.discard(best_chunk)
        newly_covered = [e for e in remaining_cold if chunk_expert_coverage[best_chunk, e] > 0]
        remaining_cold -= set(newly_covered)

    print(f"[EBSS] Phase 1: selected {len(selected)} chunks, {len(remaining_cold)} cold experts still uncovered", flush=True)

    # Phase 2: Fill remaining slots with diverse chunks
    while len(selected) < target_chunks and available:
        best_chunk = max(available, key=lambda c: float(chunk_expert_coverage[c].sum().item()))
        selected.append(best_chunk)
        available.discard(best_chunk)

    print(f"[EBSS] Final selection: {len(selected)} chunks", flush=True)
    return selected


def run_ebss_calibration(
    store: WeightStore,
    config: Any,
    selected_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> CalibrationArtifacts:
    """Run full calibration on the EBSS-selected chunks."""
    print(f"\n=== EBSS Calibration ({selected_chunks.shape[0]} balanced chunks, {selected_chunks.shape[1]}-token) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, selected_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[ebss-calib] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
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
            print(f"[ebss-calib] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

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
    # Phase 1: Build pool of chunks for routing analysis
    # =========================================================================
    print(f"\n=== Phase 1: Building pool of {POOL_SIZE} chunks ({CHUNK_LENGTH} tokens each) ===", flush=True)
    pool_chunks = build_pool_chunks(tokenizer, POOL_SIZE, CHUNK_LENGTH, EBSS_SEED)
    print(f"Pool shape: {pool_chunks.shape}", flush=True)

    # =========================================================================
    # Phase 2: Fast routing pass
    # =========================================================================
    routing_cache_path = RESULTS_DIR / "iter32_routing_matrix.pt"
    if routing_cache_path.exists() and args.skip_routing_pass:
        print(f"Loading cached routing matrix from {routing_cache_path}", flush=True)
        expert_activations = torch.load(routing_cache_path, weights_only=True)
    else:
        print(f"\n=== Phase 2: Fast routing pass over {POOL_SIZE} chunks ===", flush=True)
        expert_activations = fast_routing_pass(store, text_config, pool_chunks, device, dtype)
        torch.save(expert_activations, routing_cache_path)
        print(f"Saved routing matrix to {routing_cache_path}", flush=True)

    # =========================================================================
    # Phase 3: Select balanced chunks
    # =========================================================================
    print(f"\n=== Phase 3: Selecting {TARGET_CHUNKS} balanced chunks ===", flush=True)
    selected_indices = select_balanced_chunks(expert_activations, TARGET_CHUNKS, text_config.num_experts)
    selected_chunks = pool_chunks[selected_indices]
    print(f"Selected chunks shape: {selected_chunks.shape}", flush=True)

    # Analyze balance
    selected_activations = expert_activations[selected_indices].sum(dim=(0, 1)).float()
    random_activations = expert_activations[:TARGET_CHUNKS].sum(dim=(0, 1)).float()
    print(f"\nExpert activation balance:", flush=True)
    print(f"  EBSS selected: min={selected_activations.min():.1f}, max={selected_activations.max():.1f}, std={selected_activations.std():.1f}", flush=True)
    print(f"  Random first {TARGET_CHUNKS}: min={random_activations.min():.1f}, max={random_activations.max():.1f}, std={random_activations.std():.1f}", flush=True)
    zero_selected = int((selected_activations == 0).sum().item())
    zero_random = int((random_activations == 0).sum().item())
    print(f"  Zero-activation experts: EBSS={zero_selected}, random={zero_random}", flush=True)

    # =========================================================================
    # Phase 4: Run calibrations
    # =========================================================================
    print(f"\n=== Phase 4a: EBSS calibration ({TARGET_CHUNKS} balanced {CHUNK_LENGTH}-token chunks) ===", flush=True)
    ebss_calibration = run_ebss_calibration(store, text_config, selected_chunks, device, dtype)

    print(f"\n=== Phase 4b: Standard calibration (128 random 2048-token chunks, control) ===", flush=True)
    standard_calibration = run_calibration(store, text_config, standard_calib_chunks, device, dtype)

    # =========================================================================
    # Phase 5: Build masks and evaluate
    # =========================================================================
    configs_to_eval = [
        ("ebss_joint_topup", ebss_calibration, "EBSS-balanced calibration + joint W1/W2 topup"),
        ("standard_joint_topup", standard_calibration, "Standard calibration + joint W1/W2 topup (control)"),
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

    # Save results
    payload = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": 296960,
            "seqlen": SEQLEN,
            "nsamples": 145,
            "dtype": args.dtype,
            "approach": "Expert-Balanced Self-Sampling (EBSS) Calibration",
            "hypothesis": "Cold experts under-calibrated with standard WikiText-2; balanced sampling improves sensitivity metrics",
            "pool_size": POOL_SIZE,
            "target_chunks": TARGET_CHUNKS,
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
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("ppl", float("inf")))
    for i, (name, result) in enumerate(sorted_results[:5], 1):
        ppl = result.get("ppl", "N/A")
        mem = result.get("memory_gb", "N/A")
        print(f"{i}. {name:50s} PPL={ppl} Mem={mem} GB")


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


if __name__ == "__main__":
    main()

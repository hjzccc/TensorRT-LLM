#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""
Iteration 49: MaCa Uniform 8K

Foundation: MaCa Uniform 4K (Iter29 best: 6.567582 PPL)
Innovation: Use 8192-token chunks instead of 4096 for calibration statistics

Hypothesis: Longer context captures better long-range dependency statistics,
improving the quality of the MaCa calibration and thus the precision assignment.

Expected improvement: 0.000-0.001 PPL over Iter29 (6.567582 → 6.566-6.568)
Risk: LOW — minimal code change, same mask builder
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

from proper_eval import (
    SEQLEN,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    finalize_layer_metrics,
    init_expert_moment_state,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
    upsert_exploration_section,
)
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter26_maca_calibration import (
    run_maca_calibration,
    CalibrationArtifacts,
)
from baselines_comparison import (
    resolve_non_expert_bytes,
    resolve_terminal_keys,
)
from spike1_ground_truth import load_root_config, build_text_config

SCRIPT_DIR = Path(__file__).parent
SECTION_MARKER = "## Iteration 49: MaCa Uniform 8K"

# Configuration
CALIB_CHUNKS = 32
CALIB_LENGTH = 8192  # KEY CHANGE: 8K instead of 4K
TOPUP_FRACTION = JOINT_MEDIUM_TOPUP_FRACTION  # 0.05 (same as Iter29)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "results" / "proper_iter49_maca_uniform_8k.json",
    )
    parser.add_argument(
        "--exploration-md",
        type=Path,
        default=SCRIPT_DIR / "results" / "exploration.md",
    )
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Cap on WikiText-2 eval chunks for smoke tests; 0 = full 145-chunk test.",
    )
    return parser.parse_args()


def load_wikitext_train_ids(tokenizer: Any) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)


def maybe_slice_eval_chunks(
    test_ids: torch.Tensor, max_chunks: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_chunks <= 0:
        return test_ids, {}
    n = max_chunks * SEQLEN
    return test_ids[:n], {"eval_max_chunks": max_chunks, "truncated": True}


def build_8k_calibration_set(
    tokenizer: Any,
    train_ids: torch.Tensor,
    seed: int,
    num_chunks: int,
    chunk_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build calibration set with 8K-token chunks, padded to chunk_length."""
    rng = random.Random(f"iter49:{seed}:uniform_{chunk_length}")
    padded_chunks: list[torch.Tensor] = []
    actual_lengths: list[int] = []

    # train_ids is 1D tensor
    total_tokens = int(train_ids.shape[0])
    max_start = total_tokens - chunk_length - 1
    if max_start < 0:
        raise RuntimeError(f"WikiText-2 train is too short for {chunk_length}-token chunks")

    for _ in range(num_chunks):
        start = rng.randint(0, max_start)
        sample = train_ids[start : start + chunk_length]
        padded = torch.zeros(chunk_length, dtype=torch.long)
        padded[:chunk_length] = sample[:chunk_length]
        padded_chunks.append(padded.unsqueeze(0))
        actual_lengths.append(chunk_length)

    chunks = torch.cat(padded_chunks, dim=0).contiguous()  # (N, chunk_length)
    actual_lengths_t = torch.tensor(actual_lengths, dtype=torch.long)
    return chunks, actual_lengths_t


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"\n{'='*80}")
    print(f"Iteration 49: MaCa Uniform 8K")
    print(f"  Foundation: Iter29 best (maca_uniform_4k = 6.567582 PPL)")
    print(f"  Innovation: {CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens (8K context)")
    print(f"  Topup fraction: {TOPUP_FRACTION}")
    print(f"  Expected: 6.566-6.568 PPL")
    print(f"{'='*80}\n")

    overall_start = time.time()
    tokenizer, _standard_calib_chunks, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}
    train_ids = load_wikitext_train_ids(tokenizer)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "seed": int(args.seed),
            "evaluation": eval_info,
            "standard_calibration_reference": calib_info,
            "experiment": "Iteration 49: MaCa Uniform 8K",
            "foundation": "Iter29 maca_uniform_4k = 6.567582 PPL",
            "innovation": f"{CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens (8K context)",
            "expected_improvement": "0.000-0.001 PPL (target: 6.566-6.568)",
            "hephaestus_approved": True,
            "hephaestus_rationale": "Quick parameter sweep: longer context may capture better statistics for precision assignment.",
        },
        "results": {},
    }

    # Build 8K calibration set
    print(f"[iter49] Building 8K calibration set ({CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens)...")
    calib_chunks, actual_lengths = build_8k_calibration_set(
        tokenizer,
        train_ids,
        seed=args.seed,
        num_chunks=CALIB_CHUNKS,
        chunk_length=CALIB_LENGTH,
    )
    print(f"[iter49] Built {calib_chunks.shape[0]} calibration chunks of length {calib_chunks.shape[1]}")

    # Run MaCa calibration with 8K chunks
    # Note: run_maca_calibration expects (chunks, actual_lengths) where chunks is (N, L)
    print(f"[iter49] Running MaCa calibration with 8K chunks...")
    from proper_iter26_maca_calibration import WeightStore, layer_keys, shorten_layer_tensors, release_tensors, collect_layer_calibration_valid_tokens, should_log_chunk
    from spike1_ground_truth import rms_norm_qwen3_next, full_attention_forward, linear_attention_forward
    from baselines_comparison import quantize_linear_weight, LayerMetricBundle

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    embed_key, _, _ = resolve_terminal_keys(weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(text_config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, Any] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(text_config.num_hidden_layers):
        layer_type = layer_type_at(text_config, layer_idx)
        print(f"[maca-calib-8k] load layer {layer_idx + 1}/{text_config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(text_config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(text_config.num_experts)], dim=0)
        state = init_expert_moment_state(text_config, device)

        for chunk_idx in range(inps.shape[0]):
            actual_length = int(actual_lengths[chunk_idx].item())
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], text_config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", "", 1): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, text_config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", "", 1): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, text_config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], text_config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = collect_layer_calibration_valid_tokens(mlp_input, moe_tensors, fp4_gate_up, fp4_down, text_config, state, actual_length)
            outs[chunk_idx] = residual + moe_out
            if should_log_chunk(chunk_idx, int(inps.shape[0])):
                print(
                    f"[maca-calib-8k] layer {layer_idx + 1}/{text_config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]} | actual_len={actual_length}",
                    flush=True,
                )

        bundle, layer_w1, layer_w2, layer_mc = finalize_layer_metrics(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, text_config)
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
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    calib_artifacts = CalibrationArtifacts(
        routing_counts=routing_counts,
        activation_cache=activation_cache,
        mxmoe_w1_deltas=mxmoe_w1_deltas,
        mxmoe_w2_deltas=mxmoe_w2_deltas,
        mc_moe_scores=mc_moe_scores,
    )

    # Build masks using same joint_w1w2_with_topup as Iter29
    print(f"[iter49] Building joint_w1w2_with_topup masks (topup={TOPUP_FRACTION})...")
    w1_masks, w2_masks = build_joint_with_topup_masks(
        calib_artifacts,
        text_config,
        total_expert_elems,
        TOPUP_FRACTION,
    )

    plan = build_plan_from_masks(
        "maca_uniform_8k",
        f"MaCa Uniform 8K: {CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens, joint_w1w2_with_topup masks",
        text_config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )

    # Evaluate
    print(f"[iter49] Evaluating plan on WikiText-2 test set...")
    evaluate_and_record_plan(
        payload,
        plan,
        {},
        args.output_json,
        text_config,
        total_expert_elems,
        test_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
    )

    atomic_json_dump(payload, args.output_json)

    # Print summary
    ppl = payload.get("results", {}).get("ppl", "N/A")
    elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"Iteration 49 COMPLETE")
    print(f"  Config: maca_uniform_8k ({CALIB_CHUNKS}×{CALIB_LENGTH})")
    print(f"  PPL: {ppl}")
    print(f"  vs Iter29 (maca_uniform_4k): 6.567582")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Results: {args.output_json}")
    print(f"{'='*80}\n")

    if args.exploration_md.exists():
        upsert_exploration_section(
            args.exploration_md,
            SECTION_MARKER,
            f"MaCa Uniform 8K: {ppl} PPL\n"
            f"- Calibration: {CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens\n"
            f"- Topup: {TOPUP_FRACTION}\n"
            f"- vs Iter29: 6.567582 PPL",
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""Iteration 36: Sample-Normalized Calibration (MaCa-style)

Key insight from MaCa paper (arXiv:2602.07465):
  "Token-level averaging biases the Hessian toward longer sequences."
  "Per-sequence normalization: each sample contributes equally regardless of length."

Our current calibration accumulates token-level moments:
  sum1 += x.sum(dim=0)  # all tokens from all chunks
  kurtosis = f(sum1/total_tokens, sum2/total_tokens, ...)

MaCa-style sample-level normalization:
  For each chunk: kurtosis_chunk = f(x_chunk.mean(0), x_chunk.var(0), ...)
  Final kurtosis = mean(kurtosis_chunk) across chunks

For cold experts (few tokens per chunk), this gives more stable estimates
because each calibration chunk contributes equally, regardless of how many
tokens were routed to that expert in that chunk.

Also tests: routing-frequency-weighted kurtosis
  kurtosis_weighted = kurtosis * routing_freq / routing_freq.mean()
  This amplifies hot experts' sensitivity and dampens cold experts' noisy estimates.

Expected gain: 0.001-0.003 PPL (more stable sensitivity estimates for cold experts)
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass, field
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
    dtype_from_name,
    embed_chunks,
    evaluate_plan,
    layer_type_at,
    load_gptq_standard_data,
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter36_sample_normalized_calib.json"

CHUNK_LENGTH = 4096
NUM_CHUNKS = 128
SEED = 0
EPS = 1e-8


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


def build_maca_chunks(tokenizer: Any, n: int, chunk_length: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    train_ids = load_wikitext_train_ids(tokenizer)
    max_start = int(train_ids.shape[1]) - chunk_length - 1
    chunks = []
    for _ in range(n):
        start = rng.randint(0, max_start)
        chunks.append(train_ids[:, start : start + chunk_length])
    return torch.cat(chunks, dim=0).contiguous()


@dataclass
class SampleNormState:
    """Accumulates per-chunk kurtosis estimates for sample-level averaging."""
    # Per-expert: list of per-chunk kurtosis tensors
    input_kurtosis_chunks: list[dict[int, torch.Tensor]] = field(default_factory=list)
    inter_kurtosis_chunks: list[dict[int, torch.Tensor]] = field(default_factory=list)
    routing_counts: torch.Tensor = field(default_factory=lambda: torch.zeros(256, dtype=torch.int64))
    mxmoe_w1_sq: torch.Tensor = field(default_factory=lambda: torch.zeros(256, dtype=torch.float32))
    mxmoe_w2_sq: torch.Tensor = field(default_factory=lambda: torch.zeros(256, dtype=torch.float32))
    chunk_count: int = 0


def kurtosis_from_values(values: torch.Tensor) -> torch.Tensor:
    """Compute kurtosis from raw values (not accumulated moments)."""
    if values.shape[0] <= 1:
        return torch.zeros(values.shape[1], dtype=torch.float32, device=values.device)
    values_f = values.float()
    mean = values_f.mean(dim=0)
    centered = values_f - mean
    var = (centered ** 2).mean(dim=0)
    central4 = (centered ** 4).mean(dim=0)
    kurt = central4 / (var ** 2 + EPS)
    return torch.where(torch.isfinite(kurt), kurt, torch.zeros_like(kurt))


def run_sample_normalized_calibration(
    store: WeightStore,
    config: Any,
    chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> CalibrationArtifacts:
    """Run calibration with sample-level normalization (MaCa-style)."""
    n_chunks = chunks.shape[0]
    num_experts = config.num_experts
    print(f"\n=== Sample-Normalized Calibration ({n_chunks}×{chunks.shape[1]}-token) ===", flush=True)

    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[sample-norm] layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[e], "fp4") for e in range(num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[e], "fp4") for e in range(num_experts)], dim=0)

        # Per-expert accumulators for sample-level averaging
        layer_routing_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
        # For sample-level kurtosis: accumulate per-chunk kurtosis
        # We'll use a running mean approach
        input_kurt_sum = torch.zeros((num_experts, config.hidden_size), dtype=torch.float32, device=device)
        inter_kurt_sum = torch.zeros((num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device)
        expert_chunk_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)  # chunks where expert was active
        mxmoe_w1_sq = torch.zeros(num_experts, dtype=torch.float32, device=device)
        mxmoe_w2_sq = torch.zeros(num_experts, dtype=torch.float32, device=device)

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
            flat = mlp_input.view(-1, mlp_input.shape[-1])

            # Routing
            router_logits = F.linear(flat, tensors["mlp.gate.weight"]).float()
            routing_probs = torch.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(mlp_input.dtype)

            expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
            layer_routing_counts.add_(expert_counts)

            final_hidden_states = torch.zeros_like(flat)
            active_experts = torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist()

            for expert_idx in active_experts:
                token_idx, route_pos = torch.where(selected_experts == expert_idx)
                current_state = flat[token_idx]  # [n_tokens, hidden_size]
                weights = routing_weights[token_idx, route_pos].unsqueeze(-1).to(torch.float32)

                gate_up = F.linear(current_state, gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                full_hidden = F.silu(gate) * up  # [n_tokens, inter_size]
                full_out = F.linear(full_hidden, down_proj[expert_idx])
                final_hidden_states.index_add_(0, token_idx, (weights * full_out.float()).to(mlp_input.dtype))

                # Sample-level kurtosis: compute kurtosis for THIS CHUNK's tokens
                if current_state.shape[0] > 1:
                    chunk_input_kurt = kurtosis_from_values(current_state)
                    chunk_inter_kurt = kurtosis_from_values(full_hidden)
                    # Running mean: weight each chunk equally
                    n = int(expert_chunk_counts[expert_idx].item()) + 1
                    input_kurt_sum[expert_idx] += (chunk_input_kurt - input_kurt_sum[expert_idx] / max(n-1, 1)) / n if n > 1 else chunk_input_kurt
                    inter_kurt_sum[expert_idx] += (chunk_inter_kurt - inter_kurt_sum[expert_idx] / max(n-1, 1)) / n if n > 1 else chunk_inter_kurt
                    expert_chunk_counts[expert_idx] += 1

                # MxMoE deltas
                q_gate_up = F.linear(current_state, fp4_gate_up[expert_idx])
                q_gate, q_up = q_gate_up.chunk(2, dim=-1)
                q_hidden_w1 = F.silu(q_gate) * q_up
                q_out_w1 = F.linear(q_hidden_w1, down_proj[expert_idx]).float()
                q_out_w2 = F.linear(full_hidden, fp4_down[expert_idx]).float()
                full_out_f = full_out.float()
                mxmoe_w1_sq[expert_idx] += torch.sum((weights * (q_out_w1 - full_out_f)).square())
                mxmoe_w2_sq[expert_idx] += torch.sum((weights * (q_out_w2 - full_out_f)).square())

            # Shared expert
            shared_gate = F.linear(flat, tensors["mlp.shared_expert.gate_proj.weight"])
            shared_up = F.linear(flat, tensors["mlp.shared_expert.up_proj.weight"])
            shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["mlp.shared_expert.down_proj.weight"])
            shared_gate_value = torch.sigmoid(F.linear(flat, tensors["mlp.shared_expert_gate.weight"]))
            final_hidden_states = final_hidden_states + shared_out * shared_gate_value
            outs[chunk_idx] = residual + final_hidden_states.view_as(mlp_input)

            print(f"[sample-norm] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        # Build sensitivity scores using sample-normalized kurtosis
        w1_scores = torch.zeros((num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device)
        w2_scores = torch.zeros((num_experts, config.hidden_size), dtype=torch.float32, device=device)
        mc_scores_layer = torch.zeros(num_experts, dtype=torch.float32, device=device)

        for expert_idx in range(num_experts):
            n_chunks_active = int(expert_chunk_counts[expert_idx].item())
            if n_chunks_active == 0:
                continue

            # Use sample-normalized kurtosis
            input_kurt = input_kurt_sum[expert_idx]
            inter_kurt = inter_kurt_sum[expert_idx]

            gate_up_weight = gate_up_proj[expert_idx].float()
            gate_up_q = fp4_gate_up[expert_idx].float()
            gate_diff = gate_up_weight - gate_up_q
            pair_diff_abs = gate_diff[: config.moe_intermediate_size].abs() + gate_diff[config.moe_intermediate_size :].abs()
            w1_scores[expert_idx] = torch.matmul(pair_diff_abs, input_kurt)

            down_weight = down_proj[expert_idx].float()
            down_q = fp4_down[expert_idx].float()
            down_diff = down_weight - down_q
            w2_scores[expert_idx] = torch.matmul(down_diff.abs(), inter_kurt)

            count = int(layer_routing_counts[expert_idx].item())
            if count > 0:
                weight_norm = torch.sqrt(gate_up_weight.square().sum() + down_weight.square().sum())
                quant_loss = torch.sqrt((gate_up_weight - gate_up_q).square().sum() + (down_weight - down_q).square().sum())
                mc_scores_layer[expert_idx] = float(count ** 1.0 * float(weight_norm.item()) ** 1.5 * float(quant_loss.item()) ** 2.0)

        routing_counts[layer_idx] = layer_routing_counts.cpu().to(torch.int64)
        activation_cache[layer_idx] = {
            'routing_counts': routing_counts[layer_idx],
            'w1_pair_scores': w1_scores.cpu(),
            'w2_channel_scores': w2_scores.cpu(),
        }
        mxmoe_w1_deltas[layer_idx] = torch.sqrt(mxmoe_w1_sq).cpu()
        mxmoe_w2_deltas[layer_idx] = torch.sqrt(mxmoe_w2_sq).cpu()
        mc_moe_scores[layer_idx] = mc_scores_layer.cpu()

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

    # Build MaCa chunks (same as iter29 best)
    maca_chunks = build_maca_chunks(tokenizer, NUM_CHUNKS, CHUNK_LENGTH, SEED)

    # Run sample-normalized calibration
    cache_path = RESULTS_DIR / "iter36_sample_norm_cache.pt"
    if cache_path.exists():
        print(f"Loading cached calibration from {cache_path}", flush=True)
        calibration = torch.load(cache_path, weights_only=False, map_location='cpu')
    else:
        calibration = run_sample_normalized_calibration(store, text_config, maca_chunks, device, dtype)
        torch.save(calibration, cache_path)
        print(f"Saved calibration to {cache_path}", flush=True)

    # Evaluate
    configs_to_eval = [
        ("sample_norm_joint_topup", calibration, "Sample-normalized kurtosis + joint W1/W2 topup"),
    ]

    for plan_name, calib, description in configs_to_eval:
        if requested_plans and plan_name not in requested_plans:
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"Evaluating: {plan_name}", flush=True)
        print(f"{'='*80}", flush=True)

        start_time = time.time()
        w1_masks, w2_masks = build_joint_with_topup_masks(calib, text_config, JOINT_MEDIUM_TOPUP_FRACTION)
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
            "approach": "Sample-Normalized Calibration (MaCa-style)",
            "hypothesis": "Per-chunk kurtosis averaging gives more stable estimates for cold experts",
            "chunk_length": CHUNK_LENGTH,
            "num_chunks": NUM_CHUNKS,
            "seed": SEED,
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

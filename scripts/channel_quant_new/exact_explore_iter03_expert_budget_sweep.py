#!/usr/bin/env python3
"""Exact-path Exploration Iteration 3: Whole-Expert FP8 Budget Sweep.

Assigns entire experts to FP8 or NVFP4 based on routing frequency.
Hot experts (most frequently routed) get FP8; cold experts get NVFP4.
Sweeps FP8 budget from 0% to 100% to find the Pareto frontier.

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter03_expert_budget_sweep.py --nsamples 4
"""
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
sys.path.insert(0, str(SCRIPT_DIR.parent / "channel_quant"))

import exact_docker_eval as exact_eval
from spike1_ground_truth import (
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter03_expert_budget_sweep.json"
FP8_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def compute_routing_frequencies(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Run a forward pass collecting per-expert token counts per layer."""
    store = exact_eval.WeightStore(exact_eval.MODEL_ID, snapshot_dir, weight_map)
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    root_t = store.load_tensors([embed_key, norm_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    del root_t

    eval_chunks = eval_ids[:, : nsamples * seqlen].view(nsamples, seqlen).contiguous()
    hidden_bank = torch.empty(
        (nsamples, seqlen, config.hidden_size), dtype=dtype, device="cpu"
    )
    for i in range(nsamples):
        hidden_bank[i].copy_(F.embedding(eval_chunks[i : i + 1].to(device), embed_w).cpu())

    causal_mask = exact_eval.build_causal_mask(seqlen, device)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    rotary_input = torch.empty((1, seqlen, config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(rotary_input, position_ids)
    del rotary_input

    freq: dict[int, torch.Tensor] = {}

    with torch.inference_mode():
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[layer_idx]
            raw = store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_tensors = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            layer_counts = torch.zeros(config.num_experts, dtype=torch.float32)

            for sample_idx in range(nsamples):
                hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["input_layernorm.weight"], config.rms_norm_eps
                )
                if layer_type == "full_attention":
                    attn_t = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = exact_eval.full_attention_forward_exact(
                        hidden_states, attn_t, config, position_embeddings,
                        exact_eval.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_t = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_t, config, "bf16", "moe_only"
                    )
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["post_attention_layernorm.weight"], config.rms_norm_eps
                )

                flat = hidden_states.view(-1, config.hidden_size)
                gate_weight = layer_tensors.get("mlp.gate.weight")
                if gate_weight is not None:
                    router_logits = exact_eval.bf16_linear(flat, gate_weight).float()
                    routing_weights = torch.softmax(router_logits, dim=1)
                    _, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
                    counts = torch.bincount(selected_experts.reshape(-1).cpu(), minlength=config.num_experts).float()
                    layer_counts += counts

                moe_t = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
                moe_out = exact_eval.moe_forward_exact(hidden_states, moe_t, config, "bf16")
                hidden_states = residual + moe_out
                hidden_bank[sample_idx : sample_idx + 1].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            freq[layer_idx] = layer_counts
            release_tensors(layer_tensors)
            if (layer_idx + 1) % 10 == 0:
                print(f"  [routing] layer {layer_idx + 1}/{config.num_hidden_layers}", flush=True)

    del embed_w, store
    torch.cuda.empty_cache()
    return freq


def build_expert_fp8_mask(
    freq: dict[int, torch.Tensor],
    fp8_fraction: float,
    num_layers: int,
    num_experts: int,
) -> dict[int, set[int]]:
    """Return set of expert indices that should be FP8 for each layer."""
    fp8_experts: dict[int, set[int]] = {}
    n_fp8 = max(0, min(num_experts, int(round(fp8_fraction * num_experts))))

    for layer_idx in range(num_layers):
        counts = freq.get(layer_idx, torch.zeros(num_experts))
        if n_fp8 == 0:
            fp8_experts[layer_idx] = set()
        elif n_fp8 >= num_experts:
            fp8_experts[layer_idx] = set(range(num_experts))
        else:
            _, top_indices = torch.topk(counts, n_fp8)
            fp8_experts[layer_idx] = set(top_indices.tolist())

    return fp8_experts


def expert_budget_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    fp8_expert_set: set[int],
) -> torch.Tensor:
    """MoE forward with whole-expert FP8 vs NVFP4 assignment."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = exact_eval.bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(
        routing_weights, config.num_experts_per_tok, dim=-1
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        mode = "fp8" if expert_idx in fp8_expert_set else "nvfp4"

        if mode == "fp8":
            gate_up = exact_eval.fp8_linear(current_state, gate_up_proj[expert_idx])
        else:
            gate_up = exact_eval.nvfp4_linear(current_state, gate_up_proj[expert_idx])

        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up

        if mode == "fp8":
            current_hidden = exact_eval.fp8_linear(hidden, down_proj[expert_idx])
        else:
            current_hidden = exact_eval.nvfp4_linear(hidden, down_proj[expert_idx])

        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = exact_eval.bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * exact_eval.bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = exact_eval.bf16_linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(exact_eval.bf16_linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_expert_budget_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    fp8_experts: dict[int, set[int]],
    layer_batch_size: int,
    label: str,
) -> float:
    store = exact_eval.WeightStore(exact_eval.MODEL_ID, snapshot_dir, weight_map)
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"
    root_t = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    final_norm_w = move_tensor(root_t[norm_key], device, dtype)
    lm_head_w = move_tensor(root_t[lm_head_key], device, dtype)
    del root_t

    eval_chunks = eval_ids[:, : nsamples * seqlen].view(nsamples, seqlen).contiguous()
    hidden_bank = torch.empty(
        (nsamples, seqlen, config.hidden_size), dtype=dtype, device="cpu"
    )
    for i in range(nsamples):
        hidden_bank[i].copy_(F.embedding(eval_chunks[i : i + 1].to(device), embed_w).cpu())

    causal_mask = exact_eval.build_causal_mask(seqlen, device)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    rotary_input = torch.empty((1, seqlen, config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(rotary_input, position_ids)
    del rotary_input

    nlls: list[torch.Tensor] = []

    with torch.inference_mode():
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[layer_idx]
            raw = store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_tensors = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            for sample_idx in range(nsamples):
                hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["input_layernorm.weight"], config.rms_norm_eps
                )
                if layer_type == "full_attention":
                    attn_t = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = exact_eval.full_attention_forward_exact(
                        hidden_states, attn_t, config, position_embeddings,
                        exact_eval.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_t = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_t, config, "bf16", "moe_only"
                    )
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["post_attention_layernorm.weight"], config.rms_norm_eps
                )
                moe_t = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
                moe_out = expert_budget_moe_forward(
                    hidden_states, moe_t, config, fp8_experts.get(layer_idx, set())
                )
                hidden_states = residual + moe_out
                hidden_bank[sample_idx : sample_idx + 1].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            if (layer_idx + 1) % 10 == 0:
                print(f"  [{label}] layer {layer_idx + 1}/{config.num_hidden_layers}", flush=True)

        for sample_idx in range(nsamples):
            torch.cuda.empty_cache()
            chunk = eval_chunks[sample_idx : sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, config.rms_norm_eps)
            logits = F.linear(hidden_states, lm_head_w)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)
            del logits, shift_logits, hidden_states

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


def main() -> None:
    args = parse_args()
    exact_eval.ensure_runtime_available()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, full_nsamples, seqlen = exact_eval.load_eval_data(tokenizer, args.seqlen)
    nsamples = min(args.nsamples, full_nsamples)
    print(f"Eval: {nsamples} chunks of {seqlen} tokens", flush=True)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)

    print("\n=== Phase 1: Computing routing frequencies ===", flush=True)
    freq = compute_routing_frequencies(
        eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir, device, dtype
    )
    total_tokens = sum(f.sum().item() for f in freq.values())
    print(f"Total routed tokens: {total_tokens:.0f}", flush=True)

    results: dict[str, dict[str, Any]] = {}
    print("\n=== Phase 2: FP8 budget sweep ===", flush=True)

    for fp8_frac in FP8_FRACTIONS:
        label = f"fp8_{fp8_frac:.0%}".replace("%", "pct")
        fp8_experts = build_expert_fp8_mask(
            freq, fp8_frac, config.num_hidden_layers, config.num_experts
        )
        n_fp8_total = sum(len(s) for s in fp8_experts.values())
        n_total = config.num_hidden_layers * config.num_experts
        actual_frac = n_fp8_total / n_total

        print(f"\n--- {label}: {n_fp8_total}/{n_total} experts FP8 ({actual_frac:.1%}) ---", flush=True)
        start_time = time.time()
        ppl = evaluate_expert_budget_ppl(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, fp8_experts, args.layer_batch_size, label,
        )
        elapsed = time.time() - start_time
        results[label] = {
            "fp8_fraction_target": fp8_frac,
            "fp8_fraction_actual": round(actual_frac, 4),
            "n_fp8_experts": n_fp8_total,
            "n_total_experts": n_total,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "eval_tokens": nsamples * seqlen,
            "dtype": args.dtype,
            "experiment": "whole_expert_fp8_budget_sweep",
            "allocation": "routing_frequency (hot experts → FP8, cold → NVFP4)",
            "scope": "moe_only (attention/shared/DeltaNet stay BF16)",
            "fp8_fractions_tested": FP8_FRACTIONS,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved -> {args.output}", flush=True)
    print(f"\n{'Label':<20s} {'FP8%':>6s} {'PPL':>10s} {'Delta':>10s}")
    print("-" * 50)
    bf16_ppl = results.get("fp8_0pct", {}).get("ppl") or results.get(list(results.keys())[0], {}).get("ppl", 0)
    for label, row in sorted(results.items(), key=lambda item: item[1]["fp8_fraction_target"]):
        delta = row["ppl"] - bf16_ppl if bf16_ppl else 0
        print(f"{label:<20s} {row['fp8_fraction_actual']:>5.1%} {row['ppl']:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()

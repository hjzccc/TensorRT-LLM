#!/usr/bin/env python3
"""Exact-path Exploration Iteration 1: Projection Isolation Ablations.

Tests which projection (W1 gate_up vs W2 down) and which format (FP4 vs FP8)
contributes most to quantization error under exact TRT-LLM kernels.

Configurations (MoE experts only, attention/shared/DeltaNet stay BF16):
  1. w1_fp4_w2_fp8  — all W1 NVFP4, all W2 FP8
  2. w1_fp8_w2_fp4  — all W1 FP8, all W2 NVFP4
  3. w1_only_fp4    — only W1 NVFP4, W2 BF16
  4. w2_only_fp4    — only W2 NVFP4, W1 BF16
  5. w1_only_fp8    — only W1 FP8, W2 BF16
  6. w2_only_fp8    — only W2 FP8, W1 BF16

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter01_projection_isolation.py --nsamples 4
"""
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter01_projection_isolation.json"


@dataclass(frozen=True)
class ProjectionConfig:
    """Per-projection quantization mode for MoE experts."""
    label: str
    w1_mode: str  # "bf16", "nvfp4", or "fp8"
    w2_mode: str  # "bf16", "nvfp4", or "fp8"

    @property
    def description(self) -> str:
        return f"W1={self.w1_mode}, W2={self.w2_mode}"


ALL_CONFIGS = [
    ProjectionConfig("w1_fp4_w2_fp8", "nvfp4", "fp8"),
    ProjectionConfig("w1_fp8_w2_fp4", "fp8", "nvfp4"),
    ProjectionConfig("w1_only_fp4", "nvfp4", "bf16"),
    ProjectionConfig("w2_only_fp4", "bf16", "nvfp4"),
    ProjectionConfig("w1_only_fp8", "fp8", "bf16"),
    ProjectionConfig("w2_only_fp8", "bf16", "fp8"),
]


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


def projection_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Call the exact TRT-LLM wrapper for the given mode."""
    if mode == "bf16":
        return exact_eval.bf16_linear(input_tensor, weight, bias)
    elif mode == "nvfp4":
        return exact_eval.nvfp4_linear(input_tensor, weight, bias)
    elif mode == "fp8":
        return exact_eval.fp8_linear(input_tensor, weight, bias)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def projection_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    proj_config: ProjectionConfig,
) -> torch.Tensor:
    """MoE forward with per-projection precision control."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)

    # Router always BF16
    router_logits = exact_eval.bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(
        routing_weights, config.num_experts_per_tok, dim=-1
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(
        selected_experts.reshape(-1), minlength=config.num_experts
    )

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]

        # W1 (gate_up_proj) with configured precision
        gate_up = projection_linear(
            current_state, gate_up_proj[expert_idx], proj_config.w1_mode
        )
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up

        # W2 (down_proj) with configured precision
        current_hidden = projection_linear(
            hidden, down_proj[expert_idx], proj_config.w2_mode
        )
        current_hidden = (
            current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden.to(hidden_states.dtype)
        )

    # Shared expert always BF16
    shared = exact_eval.bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * exact_eval.bf16_linear(
        flat, tensors["shared_expert.up_proj.weight"]
    )
    shared = exact_eval.bf16_linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(
        exact_eval.bf16_linear(flat, tensors["shared_expert_gate.weight"])
    )
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_projection_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    proj_config: ProjectionConfig,
    layer_batch_size: int,
) -> float:
    """Evaluate PPL with per-projection precision control on MoE experts."""
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
    for batch_start in range(0, nsamples, layer_batch_size):
        batch_end = min(batch_start + layer_batch_size, nsamples)
        chunk_batch = eval_chunks[batch_start:batch_end].to(device)
        hidden_batch = F.embedding(chunk_batch, embed_w)
        hidden_bank[batch_start:batch_end].copy_(hidden_batch.cpu())
        del chunk_batch, hidden_batch

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

            for batch_start in range(0, nsamples, layer_batch_size):
                batch_end = min(batch_start + layer_batch_size, nsamples)
                hidden_states = hidden_bank[batch_start:batch_end].to(device)

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states,
                    layer_tensors["input_layernorm.weight"],
                    config.rms_norm_eps,
                )

                # Attention always BF16 (moe_only scope)
                if layer_type == "full_attention":
                    attn_tensors = {
                        k.replace("self_attn.", ""): v
                        for k, v in layer_tensors.items()
                        if k.startswith("self_attn.")
                    }
                    hidden_states = exact_eval.full_attention_forward_exact(
                        hidden_states,
                        attn_tensors,
                        config,
                        position_embeddings,
                        exact_eval.build_causal_mask(seqlen, device),
                        mode="bf16",
                        quantized=False,
                    )
                else:
                    attn_tensors = {
                        k.replace("linear_attn.", ""): v
                        for k, v in layer_tensors.items()
                        if k.startswith("linear_attn.")
                    }
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_tensors, config, "bf16", "moe_only"
                    )
                hidden_states = residual + hidden_states

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states,
                    layer_tensors["post_attention_layernorm.weight"],
                    config.rms_norm_eps,
                )
                moe_tensors = {
                    k.replace("mlp.", "", 1): v
                    for k, v in layer_tensors.items()
                    if k.startswith("mlp.")
                }
                moe_out = projection_moe_forward(
                    hidden_states, moe_tensors, config, proj_config
                )
                hidden_states = residual + moe_out

                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            print(
                f"  [{proj_config.label}] layer {layer_idx + 1}/{config.num_hidden_layers}",
                flush=True,
            )

        for sample_idx in range(nsamples):
            torch.cuda.empty_cache()
            chunk = eval_chunks[sample_idx : sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(
                hidden_states, final_norm_w, config.rms_norm_eps
            )
            logits = F.linear(hidden_states, lm_head_w)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1)
            )
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
    print(
        f"Eval: {nsamples} chunks of {seqlen} tokens ({nsamples * seqlen} total)",
        flush=True,
    )

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)

    # Run uniform baselines first for reference
    # Note: baselines quantize only MoE experts (moe_only scope).
    # This means attention/shared/DeltaNet stay BF16 for all configs.
    baselines = [
        exact_eval.EvalConfig("uniform_bf16", "bf16", "moe_only"),
        exact_eval.EvalConfig("uniform_fp8", "fp8", "moe_only"),
        exact_eval.EvalConfig("uniform_nvfp4", "nvfp4", "moe_only"),
    ]

    results: dict[str, dict[str, Any]] = {}
    for run_config in baselines:
        print(f"\n=== {run_config.label} (moe_only) ===", flush=True)
        start_time = time.time()
        ppl = exact_eval.evaluate_ppl(
            eval_ids,
            nsamples,
            seqlen,
            config,
            weight_map,
            snapshot_dir,
            device,
            dtype,
            run_config,
            args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[run_config.label] = {
            "w1_mode": run_config.mode,
            "w2_mode": run_config.mode,
            "quant_scope": "moe_only",
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    # Run projection isolation configs
    for proj_config in ALL_CONFIGS:
        print(
            f"\n=== {proj_config.label} ({proj_config.description}) ===",
            flush=True,
        )
        start_time = time.time()
        ppl = evaluate_projection_ppl(
            eval_ids,
            nsamples,
            seqlen,
            config,
            weight_map,
            snapshot_dir,
            device,
            dtype,
            proj_config,
            args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[proj_config.label] = {
            "w1_mode": proj_config.w1_mode,
            "w2_mode": proj_config.w2_mode,
            "quant_scope": "moe_only",
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
            "layer_batch_size": args.layer_batch_size,
            "runtime": "docker-only TRT-LLM fused wrappers",
            "experiment": "projection_isolation_ablations",
            "purpose": "Isolate which projection (W1/W2) and which format (FP4/FP8/BF16) contributes most to quantization error under exact TRT-LLM kernels",
            "nvfp4_linear": "torch.ops.auto_deploy.torch_quant_nvfp4_linear",
            "fp8_linear": "torch.ops.auto_deploy.torch_quant_fp8_linear",
            "quant_scope_note": "moe_only for ALL configs — attention, shared expert, DeltaNet stay BF16",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved -> {args.output}", flush=True)
    print(f"\n{'Method':<24s} {'W1':<8s} {'W2':<8s} {'PPL':>10s}")
    print("-" * 54)
    for label, row in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(
            f"{label:<24s} {row['w1_mode']:<8s} {row['w2_mode']:<8s} {row['ppl']:>10.4f}"
        )


if __name__ == "__main__":
    main()

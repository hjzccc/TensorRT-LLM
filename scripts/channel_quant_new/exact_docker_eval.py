#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false, reportMissingTypeArgument=false
"""Exact docker-native TRT-LLM eval using fused linear wrappers.

This first-pass baseline keeps the existing layer-by-layer Qwen traversal from
`real_eval_pipeline.py`, but swaps quantized `F.linear` calls over to TRT-LLM's
fused Python-callable wrappers inside the `trtllm-dual-tile` container.

Supported baselines:
  - uniform_bf16
  - uniform_nvfp4
  - uniform_fp8

Supported quant scopes:
  - moe_only
  - reference_nvfp4
  - reference_fp8
  - reference_fp8_moe_only (fair comparison: attention+shared+DeltaNet quantized, MoE experts in moe_only scope)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding, apply_rotary_pos_emb, repeat_kv, torch_chunk_gated_delta_rule

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/workspace/channel_quant_new')
sys.path.insert(0, '/workspace/channel_quant')
sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant"))

from real_eval_pipeline import load_eval_data
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_causal_mask,
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_gated,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCALING_VECTOR_SIZE = 16
FP8_MAX = 448.0


@dataclass(frozen=True)
class EvalConfig:
    label: str
    mode: str
    quant_scope: str


def ensure_runtime_available() -> None:
    assert Path("/.dockerenv").exists(), "Run this script inside the trtllm-dual-tile docker container"
    assert torch.cuda.is_available(), "CUDA is required"

    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    assert hasattr(torch.ops, "auto_deploy"), "torch.ops.auto_deploy is unavailable"
    assert hasattr(torch.ops, "trtllm"), "torch.ops.trtllm is unavailable"


def flatten_for_linear(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    return x.reshape(-1, x.shape[-1]), x.shape[:-1]


def restore_linear_shape(x: torch.Tensor, prefix_shape: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(*prefix_shape, x.shape[-1])


def fp8_weight_scale(weight: torch.Tensor) -> torch.Tensor:
    return (weight.float().abs().max() / FP8_MAX).to(device=weight.device, dtype=torch.float32)


def fp8_input_scale(input_2d: torch.Tensor) -> torch.Tensor:
    return torch.tensor(1.0, device=input_2d.device, dtype=torch.float32)


def bf16_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, mode: Optional[str] = None) -> torch.Tensor:
    del mode
    return F.linear(input, weight, bias)


def nvfp4_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    input_2d, prefix_shape = flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
    alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d,
        weight_fp4,
        bias=bias,
        input_scale=s_in2,
        weight_scale=weight_scale,
        alpha=alpha,
    )
    return restore_linear_shape(out, prefix_shape)


def fp8_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    input_2d, prefix_shape = flatten_for_linear(input)
    weight_scale = fp8_weight_scale(weight)
    weight_fp8 = (weight.float() / weight_scale).to(torch.float8_e4m3fn)
    out = torch.ops.auto_deploy.torch_quant_fp8_linear(
        input_2d,
        weight_fp8,
        bias=bias,
        input_scale=fp8_input_scale(input_2d),
        weight_scale=weight_scale,
    )
    return restore_linear_shape(out, prefix_shape)


def exact_linear(input: torch.Tensor, weight: torch.Tensor, mode: str, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mode == "bf16":
        return bf16_linear(input, weight, bias)
    if mode == "nvfp4":
        return nvfp4_linear(input, weight, bias)
    if mode == "fp8":
        return fp8_linear(input, weight, bias)
    raise ValueError(f"Unknown mode: {mode}")


def quantize_shared_expert(scope: str) -> bool:
    return scope in {"reference_nvfp4", "reference_fp8", "reference_fp8_moe_only"}


def quantize_full_attention(scope: str) -> bool:
    return scope in {"reference_nvfp4", "reference_fp8", "reference_fp8_moe_only"}


def quantize_linear_attention_projection(scope: str, name: str) -> bool:
    return scope in {"reference_fp8", "reference_fp8_moe_only"} and name in {"in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight"}


def full_attention_forward_exact(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    mode: str,
    quantized: bool,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    head_dim = config.head_dim
    hidden_shape = (batch_size, seq_len, -1, head_dim)
    def linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if quantized:
            return exact_linear(input, weight, mode=mode)
        return bf16_linear(input, weight)

    q_proj = linear(hidden_states, tensors["q_proj.weight"])
    query_states, gate = torch.chunk(q_proj.view(batch_size, seq_len, -1, head_dim * 2), 2, dim=-1)
    gate = gate.reshape(batch_size, seq_len, -1)

    q_norm_weight = tensors["q_norm.weight"]
    k_norm_weight = tensors["k_norm.weight"]
    query_states = rms_norm_qwen3_next(query_states.view(hidden_shape), q_norm_weight, config.rms_norm_eps).transpose(1, 2)
    key_states = rms_norm_qwen3_next(
        linear(hidden_states, tensors["k_proj.weight"]).view(hidden_shape),
        k_norm_weight,
        config.rms_norm_eps,
    ).transpose(1, 2)
    value_states = linear(hidden_states, tensors["v_proj.weight"]).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, config.num_attention_heads // config.num_key_value_heads)
    value_states = repeat_kv(value_states, config.num_attention_heads // config.num_key_value_heads)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (head_dim ** -0.5)
    attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(hidden_states.dtype)
    attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, -1)
    attn_output = attn_output * torch.sigmoid(gate)
    return linear(attn_output, tensors["o_proj.weight"])


def linear_attention_forward_exact(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config,
    mode: str,
    scope: str,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    key_dim = config.linear_key_head_dim * config.linear_num_key_heads
    value_dim = config.linear_value_head_dim * config.linear_num_value_heads
    num_k_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    head_k_dim = config.linear_key_head_dim
    head_v_dim = config.linear_value_head_dim
    heads_ratio = num_v_heads // num_k_heads

    def maybe_linear(name: str, input: torch.Tensor) -> torch.Tensor:
        if mode != "bf16" and quantize_linear_attention_projection(scope, name):
            return exact_linear(input, tensors[name], mode=mode)
        return bf16_linear(input, tensors[name])

    projected_qkv = maybe_linear("in_proj_qkv.weight", hidden_states)
    projected_z = maybe_linear("in_proj_z.weight", hidden_states)
    projected_b = bf16_linear(hidden_states, tensors["in_proj_b.weight"])
    projected_a = bf16_linear(hidden_states, tensors["in_proj_a.weight"])

    query = projected_qkv[..., :key_dim].reshape(batch_size, seq_len, num_k_heads, head_k_dim)
    key = projected_qkv[..., key_dim : key_dim * 2].reshape(batch_size, seq_len, num_k_heads, head_k_dim)
    value = projected_qkv[..., key_dim * 2 :].reshape(batch_size, seq_len, num_v_heads, head_v_dim)
    z = projected_z.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
    b = projected_b.reshape(batch_size, seq_len, num_v_heads)
    a = projected_a.reshape(batch_size, seq_len, num_v_heads)

    mixed_qkv = torch.cat(
        [query.reshape(batch_size, seq_len, -1), key.reshape(batch_size, seq_len, -1), value.reshape(batch_size, seq_len, -1)],
        dim=-1,
    ).transpose(1, 2)
    conv_out = F.conv1d(
        mixed_qkv,
        tensors["conv1d.weight"],
        bias=None,
        padding=config.linear_conv_kernel_dim - 1,
        groups=mixed_qkv.shape[1],
    )[:, :, :seq_len]
    mixed_qkv = F.silu(conv_out).transpose(1, 2)

    query = mixed_qkv[..., :key_dim].reshape(batch_size, seq_len, num_k_heads, head_k_dim)
    key = mixed_qkv[..., key_dim : key_dim * 2].reshape(batch_size, seq_len, num_k_heads, head_k_dim)
    value = mixed_qkv[..., key_dim * 2 :].reshape(batch_size, seq_len, num_v_heads, head_v_dim)

    beta = b.sigmoid()
    g = -tensors["A_log"].float().exp().view(1, 1, -1) * F.softplus(a.float() + tensors["dt_bias"].float().view(1, 1, -1))
    if heads_ratio > 1:
        query = query.repeat_interleave(heads_ratio, dim=2)
        key = key.repeat_interleave(heads_ratio, dim=2)

    core_attn_out, _ = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out = core_attn_out.reshape(-1, head_v_dim)
    gated = rms_norm_gated(core_attn_out, tensors["norm.weight"], z.reshape(-1, head_v_dim), config.rms_norm_eps)
    gated = gated.reshape(batch_size, seq_len, num_v_heads, head_v_dim).reshape(batch_size, seq_len, -1)
    return maybe_linear("out_proj.weight", gated)


def moe_forward_exact(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config,
    mode: str,
    scope: str,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = exact_linear(current_state, gate_up_proj[expert_idx], mode=mode)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = exact_linear(F.silu(gate) * up, down_proj[expert_idx], mode=mode)
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    if quantize_shared_expert(scope) and mode != "bf16":
        shared_gate_proj = exact_linear(flat, tensors["shared_expert.gate_proj.weight"], mode=mode)
        shared_up_proj = exact_linear(flat, tensors["shared_expert.up_proj.weight"], mode=mode)
        shared = exact_linear(F.silu(shared_gate_proj) * shared_up_proj, tensors["shared_expert.down_proj.weight"], mode=mode)
    else:
        shared = bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
        shared = F.silu(shared) * bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
        shared = bf16_linear(shared, tensors["shared_expert.down_proj.weight"])

    shared_gate = torch.sigmoid(bf16_linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config,
    weight_map: dict,
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    run_config: EvalConfig,
    layer_batch_size: int,
) -> float:
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"

    root_t = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    final_norm_w = move_tensor(root_t[norm_key], device, dtype)
    lm_head_w = move_tensor(root_t[lm_head_key], device, dtype)
    del root_t

    eval_chunks = eval_ids[:, : nsamples * seqlen].view(nsamples, seqlen).contiguous()
    hidden_bank = torch.empty((nsamples, seqlen, config.hidden_size), dtype=dtype, device="cpu")
    for batch_start in range(0, nsamples, layer_batch_size):
        batch_end = min(batch_start + layer_batch_size, nsamples)
        chunk_batch = eval_chunks[batch_start:batch_end].to(device)
        hidden_batch = F.embedding(chunk_batch, embed_w)
        hidden_bank[batch_start:batch_end].copy_(hidden_batch.cpu())
        del chunk_batch, hidden_batch

    causal_mask = build_causal_mask(seqlen, device)
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
                hidden_states = rms_norm_qwen3_next(hidden_states, layer_tensors["input_layernorm.weight"], config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_tensors = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = full_attention_forward_exact(
                        hidden_states,
                        attn_tensors,
                        config,
                        position_embeddings,
                        causal_mask,
                        run_config.mode,
                        quantized=(run_config.mode != "bf16" and quantize_full_attention(run_config.quant_scope)),
                    )
                else:
                    attn_tensors = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = linear_attention_forward_exact(hidden_states, attn_tensors, config, run_config.mode, run_config.quant_scope)
                hidden_states = residual + hidden_states

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(hidden_states, layer_tensors["post_attention_layernorm.weight"], config.rms_norm_eps)

                moe_tensors = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
                moe_out = moe_forward_exact(hidden_states, moe_tensors, config, run_config.mode, run_config.quant_scope)
                hidden_states = residual + moe_out

                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            print(f"  [{run_config.label}] layer {layer_idx + 1}/{config.num_hidden_layers}", flush=True)

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

            if (sample_idx + 1) % 10 == 0:
                print(f"  [{run_config.label}] logits {sample_idx + 1}/{nsamples}", flush=True)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()

    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


def default_run_configs() -> list[EvalConfig]:
    return [
        EvalConfig("uniform_bf16", "bf16", "moe_only"),
        EvalConfig("uniform_nvfp4", "nvfp4", "reference_nvfp4"),
        EvalConfig("uniform_fp8", "fp8", "reference_fp8"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=2)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["uniform_bf16", "uniform_nvfp4", "uniform_fp8"],
        choices=["uniform_bf16", "uniform_nvfp4", "uniform_fp8"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "exact_docker_eval.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_available()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, nsamples, seqlen = load_eval_data(tokenizer, args.seqlen)
    print(f"Eval: {nsamples} chunks of {seqlen} tokens ({nsamples * seqlen} total)", flush=True)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    config_map = {cfg.label: cfg for cfg in default_run_configs()}
    selected_configs = [config_map[label] for label in args.configs]

    results: dict[str, dict[str, float | str]] = {}
    for run_config in selected_configs:
        print(f"\n=== {run_config.label} ({run_config.quant_scope}) ===", flush=True)
        start_time = time.time()
        ppl = evaluate_ppl(
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
            "mode": run_config.mode,
            "quant_scope": run_config.quant_scope,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    output = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": nsamples * seqlen,
            "seqlen": seqlen,
            "nsamples": nsamples,
            "dtype": args.dtype,
            "layer_batch_size": args.layer_batch_size,
            "runtime": "docker-only TRT-LLM fused wrappers",
            "nvfp4_linear": "torch.ops.auto_deploy.torch_quant_nvfp4_linear",
            "fp8_linear": "torch.ops.auto_deploy.torch_quant_fp8_linear",
            "nvfp4_weight_prep": "fp4_global_scale + torch.ops.trtllm.fp4_quantize + alpha=1/(s_in2*s_w2)",
            "fp8_weight_prep": "weight_scale=max(abs(W))/448, weight_fp8=(W/scale).to(float8_e4m3fn)",
            "fp8_input_scale": "test_quant_linear_fp8_matches_fused_op recipe (scalar 1.0)",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {args.output}", flush=True)

    print(f"\n{'Method':<20s} {'Scope':<18s} {'PPL':>10s}")
    print("-" * 52)
    for label, result in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(f"{label:<20s} {result['quant_scope']:<18s} {result['ppl']:>10.4f}")


if __name__ == "__main__":
    main()

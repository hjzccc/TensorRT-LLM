#!/usr/bin/env python3
"""Evaluate an NVFP4 checkpoint using layer-by-layer eager inference.

Loads pre-quantized weights from safetensors, runs NVFP4 kernels directly.
No model tracing or engine build required.

Usage:
    docker exec trtllm-dual-tile bash -c \
        "cd /code/tensorrt_llm && python3 -u scripts/nvfp4_compress/eval_nvfp4_checkpoint.py"
"""
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

from spike1_ground_truth import (
    build_text_config, load_root_config, move_tensor, rms_norm_qwen3_next,
)
from real_eval_pipeline import load_eval_data
import exact_docker_eval as ee

from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

BF16_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
CKPT_DIR = "/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint"
BLOCK_SIZE = 16


def prequant_nvfp4_linear(input_tensor, weight_fp4, weight_scale, weight_scale_2):
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    s_in = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in * weight_scale_2.float())).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, weight_fp4, bias=None,
        input_scale=s_in, weight_scale=weight_scale, alpha=alpha,
    )
    return out.reshape(*input_tensor.shape[:-1], weight_fp4.shape[0])


class NVFPCheckpointStore:
    def __init__(self, ckpt_dir):
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self._weight_map = json.load(f)["weight_map"]
        self._ckpt_dir = ckpt_dir
        self._open_handles = {}

    def load_tensors(self, keys, device="cpu"):
        grouped = defaultdict(list)
        for k in keys:
            if k not in self._weight_map:
                continue
            grouped[self._weight_map[k]].append(k)

        tensors = {}
        for shard_file, shard_keys in grouped.items():
            shard_path = os.path.join(self._ckpt_dir, shard_file)
            with safe_open(shard_path, framework="pt", device=device) as sf:
                for k in shard_keys:
                    tensors[k] = sf.get_tensor(k)
        return tensors

    def has_key(self, key):
        return key in self._weight_map


def load_expert_nvfp4(store, layer_idx, expert_idx, proj_name, device):
    prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}"
    keys = [f"{prefix}.weight", f"{prefix}.weight_scale", f"{prefix}.weight_scale_2"]
    tensors = store.load_tensors(keys, device=device)
    return (
        tensors[f"{prefix}.weight"],
        tensors[f"{prefix}.weight_scale"],
        tensors[f"{prefix}.weight_scale_2"],
    )


def load_layer_bf16_tensors(store, layer_idx, layer_type, device, dtype):
    prefix = f"model.layers.{layer_idx}"
    bf16_keys = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate.weight",
        f"{prefix}.mlp.shared_expert_gate.weight",
    ]

    shared_proj_names = ["gate_proj", "up_proj", "down_proj"]
    for pn in shared_proj_names:
        base = f"{prefix}.mlp.shared_expert.{pn}"
        if store.has_key(f"{base}.weight_scale"):
            bf16_keys.extend([f"{base}.weight", f"{base}.weight_scale", f"{base}.weight_scale_2"])
        else:
            bf16_keys.append(f"{base}.weight")

    if layer_type == "full_attention":
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = f"{prefix}.self_attn.{proj}"
            if store.has_key(f"{base}.weight_scale"):
                bf16_keys.extend([f"{base}.weight", f"{base}.weight_scale", f"{base}.weight_scale_2"])
            else:
                bf16_keys.append(f"{base}.weight")
        for norm_key in ["q_norm.weight", "k_norm.weight"]:
            bf16_keys.append(f"{prefix}.self_attn.{norm_key}")
    else:
        for attn_key in ["in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight",
                         "conv1d.weight", "in_proj_a.weight", "in_proj_b.weight",
                         "norm.weight", "A_log", "dt_bias"]:
            full_key = f"{prefix}.linear_attn.{attn_key}"
            if store.has_key(full_key):
                if store.has_key(f"{prefix}.linear_attn.{attn_key.replace('.weight', '')}.weight_scale"):
                    bf16_keys.extend([full_key,
                                      f"{prefix}.linear_attn.{attn_key.replace('.weight', '')}.weight_scale",
                                      f"{prefix}.linear_attn.{attn_key.replace('.weight', '')}.weight_scale_2"])
                else:
                    bf16_keys.append(full_key)

    tensors_raw = store.load_tensors(bf16_keys, device="cpu")

    result = {}
    for key, tensor in tensors_raw.items():
        short = key.removeprefix(f"{prefix}.")
        if "A_log" in short or "dt_bias" in short:
            result[short] = move_tensor(tensor, device, torch.float32)
        elif tensor.is_floating_point():
            result[short] = move_tensor(tensor, device, dtype)
        else:
            result[short] = tensor.to(device)
    return result


def nvfp4_or_bf16_linear(input_tensor, tensors, key_base):
    if f"{key_base}.weight_scale" in tensors:
        return prequant_nvfp4_linear(
            input_tensor,
            tensors[f"{key_base}.weight"],
            tensors[f"{key_base}.weight_scale"],
            tensors[f"{key_base}.weight_scale_2"],
        )
    else:
        return ee.bf16_linear(input_tensor, tensors[f"{key_base}.weight"])


def evaluate_ppl(ckpt_dir=CKPT_DIR, device="cuda"):
    store = NVFPCheckpointStore(ckpt_dir)

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        ckpt_config = json.load(f)

    snapshot_dir, root_config, _ = load_root_config(BF16_MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(BF16_MODEL_ID)
    eval_ids, num_samples, seqlen = load_eval_data(tokenizer)
    eval_ids = eval_ids.to(device)

    embed_raw = store.load_tensors(["model.embed_tokens.weight"], device="cpu")
    embed_weight = move_tensor(embed_raw["model.embed_tokens.weight"], device, dtype)

    norm_raw = store.load_tensors(["model.norm.weight"], device="cpu")
    final_norm = move_tensor(norm_raw["model.norm.weight"], device, dtype)

    lm_head_raw = store.load_tensors(["lm_head.weight"], device="cpu")
    lm_head_weight = move_tensor(lm_head_raw["lm_head.weight"], device, dtype)

    eval_chunks = eval_ids[:, :num_samples * seqlen].view(num_samples, seqlen).contiguous()
    hidden_bank = torch.empty((num_samples, seqlen, model_config.hidden_size), dtype=dtype, device="cpu")
    for i in range(num_samples):
        hidden_bank[i].copy_(F.embedding(eval_chunks[i:i+1].to(device), embed_weight).squeeze(0).cpu())
    del embed_weight, embed_raw
    gc.collect()

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=model_config, device=device)
    dummy = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(dummy, position_ids)
    del dummy

    nlls = []
    t0 = time.time()

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            layer_tensors = load_layer_bf16_tensors(store, layer_idx, layer_type, device, dtype)

            for sample_idx in range(num_samples):
                hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
                residual = hidden

                hidden = rms_norm_qwen3_next(hidden, layer_tensors["input_layernorm.weight"], model_config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_kv = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden = ee.full_attention_forward_exact(
                        hidden, attn_kv, model_config, position_embeddings,
                        ee.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_kv = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden = ee.linear_attention_forward_exact(hidden, attn_kv, model_config, "bf16", "moe_only")

                hidden = residual + hidden
                residual = hidden
                hidden = rms_norm_qwen3_next(hidden, layer_tensors["post_attention_layernorm.weight"], model_config.rms_norm_eps)

                flat = hidden.view(-1, model_config.hidden_size)
                router_logits = ee.bf16_linear(flat, layer_tensors["mlp.gate.weight"]).float()
                routing_probs = torch.softmax(router_logits, dim=1)
                topk_weights, topk_ids = torch.topk(routing_probs, model_config.num_experts_per_tok, dim=-1)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(dtype)

                expert_output = torch.zeros_like(flat)
                active_counts = torch.bincount(topk_ids.reshape(-1), minlength=model_config.num_experts)

                for expert_idx in torch.nonzero(active_counts > 0, as_tuple=False).flatten().tolist():
                    token_indices, route_indices = torch.where(topk_ids == expert_idx)
                    expert_input = flat[token_indices]

                    gate_w, gate_s, gate_s2 = load_expert_nvfp4(store, layer_idx, expert_idx, "gate_proj", device)
                    up_w, up_s, up_s2 = load_expert_nvfp4(store, layer_idx, expert_idx, "up_proj", device)
                    down_w, down_s, down_s2 = load_expert_nvfp4(store, layer_idx, expert_idx, "down_proj", device)

                    gate = prequant_nvfp4_linear(expert_input, gate_w, gate_s, gate_s2)
                    up = prequant_nvfp4_linear(expert_input, up_w, up_s, up_s2)
                    intermediate = F.silu(gate) * up
                    expert_out = prequant_nvfp4_linear(intermediate, down_w, down_s, down_s2)

                    del gate_w, gate_s, gate_s2, up_w, up_s, up_s2, down_w, down_s, down_s2

                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))

                shared_gate_out = nvfp4_or_bf16_linear(flat, layer_tensors, "mlp.shared_expert.gate_proj")
                shared_up_out = nvfp4_or_bf16_linear(flat, layer_tensors, "mlp.shared_expert.up_proj")
                shared = F.silu(shared_gate_out) * shared_up_out
                shared = nvfp4_or_bf16_linear(shared, layer_tensors, "mlp.shared_expert.down_proj")
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, layer_tensors["mlp.shared_expert_gate.weight"]))

                hidden = residual + (expert_output + shared_gate * shared).view_as(residual)
                hidden_bank[sample_idx].copy_(hidden.squeeze(0).cpu())

            del layer_tensors
            gc.collect()
            torch.cuda.empty_cache()
            elapsed = time.time() - t0
            print(f"  Layer {layer_idx+1}/{model_config.num_hidden_layers} ({elapsed:.0f}s)", flush=True)

        for sample_idx in range(num_samples):
            hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
            hidden = rms_norm_qwen3_next(hidden, final_norm, model_config.rms_norm_eps)
            logits = F.linear(hidden, lm_head_weight)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = eval_chunks[sample_idx:sample_idx+1, 1:].to(device).contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.item())

    final_ppl = math.exp(sum(nlls) / len(nlls))
    elapsed = time.time() - t0
    print(f"\nPPL = {final_ppl:.4f} ({elapsed:.0f}s)", flush=True)
    return final_ppl


if __name__ == "__main__":
    evaluate_ppl()

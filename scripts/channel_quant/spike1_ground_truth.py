#!/usr/bin/env python3
# pyright: reportUnusedImport=false, reportDeprecated=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportUnusedVariable=false, reportCallIssue=false, reportArgumentType=false, reportOptionalSubscript=false, reportUnknownLambdaType=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    torch_chunk_gated_delta_rule,
)


MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_JSON_PATH = RESULTS_DIR / "spike1_sensitivity.json"
DEFAULT_MD_PATH = SCRIPT_DIR / "exploration.md"
DEFAULT_LAYER_INDICES = [5, 20, 35]
DEFAULT_TOKENS = 128
DEFAULT_TOP_EXPERTS = 3
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
EPS = 1e-10


@dataclass
class LayerCapture:
    layer_idx: int
    layer_type: str
    mlp_input: torch.Tensor
    selected_experts: torch.Tensor
    expert_counts: torch.Tensor


class WeightStore:
    def __init__(self, model_id: str, snapshot_dir: Path, weight_map: dict[str, str]):
        self.model_id = model_id
        self.snapshot_dir = snapshot_dir
        self.weight_map = weight_map
        self._downloaded: dict[str, Path] = {}

    def _ensure_file(self, filename: str) -> Path:
        if filename not in self._downloaded:
            path = hf_hub_download(repo_id=self.model_id, filename=filename, repo_type="model")
            self._downloaded[filename] = Path(path)
        return self._downloaded[filename]

    def load_tensors(self, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            grouped[self.weight_map[key]].append(key)

        tensors: dict[str, torch.Tensor] = {}
        for filename, shard_keys in grouped.items():
            shard_path = self._ensure_file(filename)
            print(f"Loading shard {filename} for {len(shard_keys)} tensor(s)", flush=True)
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    tensors[key] = handle.get_tensor(key)
        return tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spike 1 ground-truth channel sensitivity for Qwen3.5-35B-A3B MoE experts.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--layer-indices", nargs="+", type=int, default=DEFAULT_LAYER_INDICES)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--experts-per-layer", type=int, default=DEFAULT_TOP_EXPERTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-target-layer", type=int, default=None)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def load_root_config(model_id: str) -> tuple[Path, dict[str, Any], dict[str, str]]:
    config_path = Path(hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model"))
    index_path = Path(hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", repo_type="model"))
    with config_path.open() as f:
        root_config = json.load(f)
    with index_path.open() as f:
        index = json.load(f)
    return config_path.parent, root_config, index["weight_map"]


def build_text_config(root_config: dict[str, Any]) -> Qwen3NextConfig:
    text_cfg = root_config["text_config"]
    rope_params = text_cfg.get("rope_parameters", {})
    return Qwen3NextConfig(
        vocab_size=text_cfg["vocab_size"],
        hidden_size=text_cfg["hidden_size"],
        intermediate_size=text_cfg.get("intermediate_size", text_cfg["moe_intermediate_size"]),
        num_hidden_layers=text_cfg["num_hidden_layers"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        hidden_act=text_cfg["hidden_act"],
        max_position_embeddings=text_cfg["max_position_embeddings"],
        rms_norm_eps=text_cfg["rms_norm_eps"],
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=rope_params.get("rope_theta", 10000.0),
        rope_scaling=None,
        partial_rotary_factor=rope_params.get("partial_rotary_factor", 0.25),
        attention_bias=text_cfg.get("attention_bias", False),
        attention_dropout=text_cfg.get("attention_dropout", 0.0),
        head_dim=text_cfg["head_dim"],
        linear_conv_kernel_dim=text_cfg["linear_conv_kernel_dim"],
        linear_key_head_dim=text_cfg["linear_key_head_dim"],
        linear_value_head_dim=text_cfg["linear_value_head_dim"],
        linear_num_key_heads=text_cfg["linear_num_key_heads"],
        linear_num_value_heads=text_cfg["linear_num_value_heads"],
        decoder_sparse_step=1,
        moe_intermediate_size=text_cfg["moe_intermediate_size"],
        shared_expert_intermediate_size=text_cfg["shared_expert_intermediate_size"],
        num_experts_per_tok=text_cfg["num_experts_per_tok"],
        num_experts=text_cfg["num_experts"],
        norm_topk_prob=True,
        output_router_logits=True,
        router_aux_loss_coef=text_cfg.get("router_aux_loss_coef", 0.001),
        mlp_only_layers=text_cfg.get("mlp_only_layers", []),
        layer_types=text_cfg["layer_types"],
    )


def rms_norm_qwen3_next(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(dtype=x.dtype)


def rms_norm_gated(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = weight.float() * out
    out = out * F.silu(gate.float())
    return out.to(dtype=x.dtype)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
    return torch.triu(mask, diagonal=1)


def load_calibration_input_ids(model_id: str, token_count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    pieces: list[str] = []
    current = 0
    for item in dataset:
        text = item["text"].strip()
        if not text:
            continue
        pieces.append(text)
        encoded = tokenizer("\n\n".join(pieces), return_tensors="pt", add_special_tokens=False)
        current = encoded["input_ids"].shape[1]
        if current >= token_count:
            input_ids = encoded["input_ids"][:, :token_count]
            sample_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            return input_ids, {
                "tokenizer": tokenizer.name_or_path,
                "requested_tokens": token_count,
                "actual_tokens": int(input_ids.shape[1]),
                "source_dataset": "wikitext/wikitext-2-raw-v1",
                "sample_preview": sample_text[:400],
            }
    raise RuntimeError("Unable to collect enough WikiText2 tokens for calibration.")


def move_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is None:
        return tensor.to(device=device, non_blocking=True)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def full_attention_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Qwen3NextConfig,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    head_dim = config.head_dim
    hidden_shape = (batch_size, seq_len, -1, head_dim)

    q_proj = F.linear(hidden_states, tensors["q_proj.weight"])
    query_states, gate = torch.chunk(q_proj.view(batch_size, seq_len, -1, head_dim * 2), 2, dim=-1)
    gate = gate.reshape(batch_size, seq_len, -1)

    q_norm_weight = tensors["q_norm.weight"]
    k_norm_weight = tensors["k_norm.weight"]
    query_states = rms_norm_qwen3_next(query_states.view(hidden_shape), q_norm_weight, config.rms_norm_eps).transpose(1, 2)
    key_states = rms_norm_qwen3_next(
        F.linear(hidden_states, tensors["k_proj.weight"]).view(hidden_shape),
        k_norm_weight,
        config.rms_norm_eps,
    ).transpose(1, 2)
    value_states = F.linear(hidden_states, tensors["v_proj.weight"]).view(hidden_shape).transpose(1, 2)

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
    return F.linear(attn_output, tensors["o_proj.weight"])


def linear_attention_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Qwen3NextConfig,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    key_dim = config.linear_key_head_dim * config.linear_num_key_heads
    value_dim = config.linear_value_head_dim * config.linear_num_value_heads
    num_k_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    head_k_dim = config.linear_key_head_dim
    head_v_dim = config.linear_value_head_dim
    heads_ratio = num_v_heads // num_k_heads

    projected_qkv = F.linear(hidden_states, tensors["in_proj_qkv.weight"])
    projected_z = F.linear(hidden_states, tensors["in_proj_z.weight"])
    projected_b = F.linear(hidden_states, tensors["in_proj_b.weight"])
    projected_a = F.linear(hidden_states, tensors["in_proj_a.weight"])

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
    return F.linear(gated, tensors["out_proj.weight"])


def moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Qwen3NextConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
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
        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, down_proj[expert_idx])
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = F.linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits, selected_experts, expert_counts


def tensor_key_prefix(layer_idx: int) -> str:
    return f"model.language_model.layers.{layer_idx}."


def layer_keys(layer_idx: int, layer_type: str) -> list[str]:
    prefix = tensor_key_prefix(layer_idx)
    keys = [
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.experts.gate_up_proj",
        prefix + "mlp.experts.down_proj",
        prefix + "mlp.gate.weight",
        prefix + "mlp.shared_expert.gate_proj.weight",
        prefix + "mlp.shared_expert.up_proj.weight",
        prefix + "mlp.shared_expert.down_proj.weight",
        prefix + "mlp.shared_expert_gate.weight",
    ]
    if layer_type == "linear_attention":
        keys.extend(
            [
                prefix + "linear_attn.A_log",
                prefix + "linear_attn.conv1d.weight",
                prefix + "linear_attn.dt_bias",
                prefix + "linear_attn.in_proj_a.weight",
                prefix + "linear_attn.in_proj_b.weight",
                prefix + "linear_attn.in_proj_qkv.weight",
                prefix + "linear_attn.in_proj_z.weight",
                prefix + "linear_attn.norm.weight",
                prefix + "linear_attn.out_proj.weight",
            ]
        )
    else:
        keys.extend(
            [
                prefix + "self_attn.q_norm.weight",
                prefix + "self_attn.k_norm.weight",
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.o_proj.weight",
            ]
        )
    return keys


def shorten_layer_tensors(layer_idx: int, raw: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    prefix = tensor_key_prefix(layer_idx)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        short = key.removeprefix(prefix)
        if tensor.is_floating_point():
            target_dtype = torch.float32 if short in {"linear_attn.A_log", "linear_attn.dt_bias"} else dtype
            out[short] = move_tensor(tensor, device, target_dtype)
        else:
            out[short] = move_tensor(tensor, device)
    return out


def release_tensors(tensors: dict[str, torch.Tensor]) -> None:
    tensors.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sequential_forward(
    store: WeightStore,
    config: Qwen3NextConfig,
    input_ids: torch.Tensor,
    target_layers: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, LayerCapture]:
    max_layer = max(target_layers)
    seq_len = input_ids.shape[1]

    embed_key = "model.language_model.embed_tokens.weight"
    embed_weight = move_tensor(store.load_tensors([embed_key])[embed_key], device, dtype)
    hidden_states = F.embedding(input_ids.to(device), embed_weight)
    del embed_weight
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = build_causal_mask(seq_len, device)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    position_embeddings = rotary(hidden_states, position_ids)

    captures: dict[int, LayerCapture] = {}
    for layer_idx in range(max_layer + 1):
        layer_type = config.layer_types[layer_idx]
        print(f"Forwarding layer {layer_idx} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        residual = hidden_states
        hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
        if layer_type == "linear_attention":
            mixed = linear_attention_forward(hidden_norm, {
                "A_log": tensors["linear_attn.A_log"],
                "conv1d.weight": tensors["linear_attn.conv1d.weight"],
                "dt_bias": tensors["linear_attn.dt_bias"],
                "in_proj_a.weight": tensors["linear_attn.in_proj_a.weight"],
                "in_proj_b.weight": tensors["linear_attn.in_proj_b.weight"],
                "in_proj_qkv.weight": tensors["linear_attn.in_proj_qkv.weight"],
                "in_proj_z.weight": tensors["linear_attn.in_proj_z.weight"],
                "norm.weight": tensors["linear_attn.norm.weight"],
                "out_proj.weight": tensors["linear_attn.out_proj.weight"],
            }, config)
        else:
            mixed = full_attention_forward(hidden_norm, {
                "q_norm.weight": tensors["self_attn.q_norm.weight"],
                "k_norm.weight": tensors["self_attn.k_norm.weight"],
                "q_proj.weight": tensors["self_attn.q_proj.weight"],
                "k_proj.weight": tensors["self_attn.k_proj.weight"],
                "v_proj.weight": tensors["self_attn.v_proj.weight"],
                "o_proj.weight": tensors["self_attn.o_proj.weight"],
            }, config, position_embeddings, causal_mask)

        hidden_states = residual + mixed
        mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
        mlp_out, _, selected_experts, expert_counts = moe_forward(
            mlp_input,
            {
                "experts.gate_up_proj": tensors["mlp.experts.gate_up_proj"],
                "experts.down_proj": tensors["mlp.experts.down_proj"],
                "gate.weight": tensors["mlp.gate.weight"],
                "shared_expert.gate_proj.weight": tensors["mlp.shared_expert.gate_proj.weight"],
                "shared_expert.up_proj.weight": tensors["mlp.shared_expert.up_proj.weight"],
                "shared_expert.down_proj.weight": tensors["mlp.shared_expert.down_proj.weight"],
                "shared_expert_gate.weight": tensors["mlp.shared_expert_gate.weight"],
            },
            config,
        )

        if layer_idx in target_layers:
            captures[layer_idx] = LayerCapture(
                layer_idx=layer_idx,
                layer_type=layer_type,
                mlp_input=mlp_input.detach().cpu(),
                selected_experts=selected_experts.detach().cpu(),
                expert_counts=expert_counts.detach().cpu(),
            )
            print(f"Captured layer {layer_idx}: {int((expert_counts > 0).sum())} active experts", flush=True)

        hidden_states = hidden_states + mlp_out
        release_tensors(tensors)

    return captures


def round_to_e2m1_grid(values: torch.Tensor) -> torch.Tensor:
    grid = E2M1_GRID.to(device=values.device, dtype=torch.float32)
    sign = torch.sign(values)
    abs_values = values.abs().clamp(max=6.0).float()
    distances = (abs_values.unsqueeze(-1) - grid).abs()
    nearest = grid[distances.argmin(dim=-1)]
    return (nearest * sign.float()).to(dtype=values.dtype)


def quantize_to_nvfp4_columns(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 1:
        weight = weight.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    if weight.shape[0] % 16 != 0:
        raise ValueError(f"Expected K dimension multiple of 16, got {tuple(weight.shape)}")
    blocks = weight.reshape(weight.shape[0] // 16, 16, weight.shape[1])
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    scale = absmax / 6.0
    normalized = blocks / (scale + EPS)
    quantized = round_to_e2m1_grid(normalized)
    dequantized = quantized * scale
    dequantized = dequantized.reshape_as(weight)
    return dequantized.squeeze(-1) if squeeze else dequantized


def quantize_to_fp8(weight: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is unavailable in this environment")
    amax = weight.abs().amax().clamp(min=1e-12)
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    return (weight.float() / scale).to(torch.float8_e4m3fn).to(torch.bfloat16) * scale


def pick_experts(capture: LayerCapture, requested: int) -> list[dict[str, Any]]:
    active = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()
    if not active:
        return []
    ranked = sorted(active, key=lambda idx: (int(capture.expert_counts[idx]), idx))
    positions = [0, len(ranked) // 2, len(ranked) - 1]
    role_names = ["cold", "medium", "hot"]
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    for role, pos in zip(role_names, positions):
        expert_id = ranked[pos]
        if expert_id in seen:
            continue
        seen.add(expert_id)
        chosen.append({
            "role": role,
            "expert_id": int(expert_id),
            "activation_count": int(capture.expert_counts[expert_id]),
        })
        if len(chosen) >= requested:
            break
    if len(chosen) < min(requested, len(ranked)):
        for expert_id in reversed(ranked):
            if expert_id in seen:
                continue
            seen.add(expert_id)
            chosen.append({
                "role": f"extra_{len(chosen)}",
                "expert_id": int(expert_id),
                "activation_count": int(capture.expert_counts[expert_id]),
            })
            if len(chosen) >= requested:
                break
    return chosen


def analyze_expert(
    store: WeightStore,
    config: Qwen3NextConfig,
    layer_capture: LayerCapture,
    expert_choice: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    layer_idx = layer_capture.layer_idx
    expert_id = expert_choice["expert_id"]
    role = expert_choice["role"]
    token_mask = (layer_capture.selected_experts == expert_id).any(dim=1)
    token_positions = torch.nonzero(token_mask, as_tuple=False).flatten()
    x = layer_capture.mlp_input.squeeze(0)[token_positions].to(device=device, dtype=dtype)
    if x.numel() == 0:
        return {
            "role": role,
            "expert_id": expert_id,
            "activation_count": expert_choice["activation_count"],
            "token_count": 0,
            "w1_pair_sensitivity": [],
            "w2_channel_sensitivity": [],
            "top_w1_pairs": [],
            "top_w2_channels": [],
        }

    raw = store.load_tensors(
        [
            f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
            f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj",
        ]
    )
    gate_up = move_tensor(raw[f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"], device, dtype)[expert_id].transpose(0, 1).contiguous()
    down = move_tensor(raw[f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"], device, dtype)[expert_id].transpose(0, 1).contiguous()
    del raw

    print(f"Analyzing layer {layer_idx} expert {expert_id} ({role}) with {x.shape[0]} routed token(s)", flush=True)

    w1_fp8 = quantize_to_fp8(gate_up)
    w1_nvfp4 = quantize_to_nvfp4_columns(gate_up)
    gate_fp8 = torch.matmul(x, w1_fp8[:, : config.moe_intermediate_size])
    up_fp8 = torch.matmul(x, w1_fp8[:, config.moe_intermediate_size :])
    gate_delta = torch.matmul(x, w1_nvfp4[:, : config.moe_intermediate_size] - w1_fp8[:, : config.moe_intermediate_size])
    up_delta = torch.matmul(x, w1_nvfp4[:, config.moe_intermediate_size :] - w1_fp8[:, config.moe_intermediate_size :])

    hidden_fp8 = F.silu(gate_fp8) * up_fp8
    hidden_nvfp4 = F.silu(gate_fp8 + gate_delta) * (up_fp8 + up_delta)
    delta_hidden = hidden_nvfp4 - hidden_fp8

    w2_fp8 = quantize_to_fp8(down)
    w2_nvfp4 = quantize_to_nvfp4_columns(down)
    w2_row_norms = torch.linalg.vector_norm(w2_fp8.float(), dim=1)
    w1_sensitivity = (torch.linalg.vector_norm(delta_hidden.float(), dim=0) * w2_row_norms).cpu()

    w2_diff = torch.matmul(hidden_fp8, w2_nvfp4 - w2_fp8)
    w2_sensitivity = torch.linalg.vector_norm(w2_diff.float(), dim=0).cpu()

    top_w1 = torch.topk(w1_sensitivity, k=min(10, w1_sensitivity.numel()))
    top_w2 = torch.topk(w2_sensitivity, k=min(10, w2_sensitivity.numel()))

    release_tensors({"gate_up": gate_up, "down": down, "w1_fp8": w1_fp8, "w1_nvfp4": w1_nvfp4, "w2_fp8": w2_fp8, "w2_nvfp4": w2_nvfp4})
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "role": role,
        "expert_id": expert_id,
        "activation_count": expert_choice["activation_count"],
        "token_count": int(x.shape[0]),
        "token_positions": [int(idx) for idx in token_positions.tolist()],
        "w1_pair_sensitivity": [float(v) for v in w1_sensitivity.tolist()],
        "w2_channel_sensitivity": [float(v) for v in w2_sensitivity.tolist()],
        "top_w1_pairs": [{"index": int(idx), "score": float(score)} for score, idx in zip(top_w1.values.tolist(), top_w1.indices.tolist())],
        "top_w2_channels": [{"index": int(idx), "score": float(score)} for score, idx in zip(top_w2.values.tolist(), top_w2.indices.tolist())],
        "w1_stats": summarize_array(w1_sensitivity),
        "w2_stats": summarize_array(w2_sensitivity),
    }


def summarize_array(values: torch.Tensor) -> dict[str, float]:
    flat = values.float()
    return {
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "p90": float(torch.quantile(flat, 0.9).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
    }


def maybe_save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def plot_histograms(results: dict[str, Any], output_dir: Path) -> None:
    experts = [(layer["layer_idx"], expert) for layer in results["layers"] for expert in layer["experts"]]
    if not experts:
        return
    fig, axes = plt.subplots(len(experts), 2, figsize=(12, max(4, 3 * len(experts))), squeeze=False)
    for row, (layer_idx, expert) in enumerate(experts):
        axes[row, 0].hist(expert["w1_pair_sensitivity"], bins=40, color="#1f77b4", alpha=0.85)
        axes[row, 0].set_title(f"L{layer_idx} E{expert['expert_id']} {expert['role']} W1")
        axes[row, 0].set_xlabel("Sensitivity")
        axes[row, 0].set_ylabel("Count")
        axes[row, 1].hist(expert["w2_channel_sensitivity"], bins=40, color="#ff7f0e", alpha=0.85)
        axes[row, 1].set_title(f"L{layer_idx} E{expert['expert_id']} {expert['role']} W2")
        axes[row, 1].set_xlabel("Sensitivity")
        axes[row, 1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "spike1_histograms.png", dpi=180)
    plt.close(fig)


def plot_sorted_curves(results: dict[str, Any], output_dir: Path) -> None:
    experts = [(layer["layer_idx"], expert) for layer in results["layers"] for expert in layer["experts"]]
    if not experts:
        return
    fig, axes = plt.subplots(len(experts), 2, figsize=(12, max(4, 3 * len(experts))), squeeze=False)
    for row, (layer_idx, expert) in enumerate(experts):
        axes[row, 0].plot(sorted(expert["w1_pair_sensitivity"]), color="#1f77b4")
        axes[row, 0].set_title(f"L{layer_idx} E{expert['expert_id']} {expert['role']} W1 sorted")
        axes[row, 0].set_xlabel("Sorted channel pair")
        axes[row, 0].set_ylabel("Sensitivity")
        axes[row, 1].plot(sorted(expert["w2_channel_sensitivity"]), color="#ff7f0e")
        axes[row, 1].set_title(f"L{layer_idx} E{expert['expert_id']} {expert['role']} W2 sorted")
        axes[row, 1].set_xlabel("Sorted channel")
        axes[row, 1].set_ylabel("Sensitivity")
    fig.tight_layout()
    fig.savefig(output_dir / "spike1_sorted_curves.png", dpi=180)
    plt.close(fig)


def aggregate_by(results: dict[str, Any], key: str) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"w1": [], "w2": []})
    for layer in results["layers"]:
        for expert in layer["experts"]:
            name = str(expert[key])
            grouped[name]["w1"].append(expert["w1_stats"]["mean"])
            grouped[name]["w2"].append(expert["w2_stats"]["mean"])
    return grouped


def plot_cross_expert(results: dict[str, Any], output_dir: Path) -> None:
    grouped = aggregate_by(results, "role")
    if not grouped:
        return
    labels = list(grouped.keys())
    w1 = [sum(grouped[label]["w1"]) / len(grouped[label]["w1"]) for label in labels]
    w2 = [sum(grouped[label]["w2"]) / len(grouped[label]["w2"]) for label in labels]
    x = torch.arange(len(labels)).numpy()
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, w1, width=width, label="W1 pair mean", color="#1f77b4")
    ax.bar(x + width / 2, w2, width=width, label="W2 channel mean", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean sensitivity")
    ax.set_title("Cross-expert comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "spike1_cross_expert.png", dpi=180)
    plt.close(fig)


def plot_cross_layer(results: dict[str, Any], output_dir: Path) -> None:
    grouped = aggregate_by(results, "layer_idx")
    if not grouped:
        return
    labels = list(grouped.keys())
    w1 = [sum(grouped[label]["w1"]) / len(grouped[label]["w1"]) for label in labels]
    w2 = [sum(grouped[label]["w2"]) / len(grouped[label]["w2"]) for label in labels]
    x = torch.arange(len(labels)).numpy()
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, w1, width=width, label="W1 pair mean", color="#2ca02c")
    ax.bar(x + width / 2, w2, width=width, label="W2 channel mean", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean sensitivity")
    ax.set_title("Cross-layer comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "spike1_cross_layer.png", dpi=180)
    plt.close(fig)


def write_findings(results: dict[str, Any], path: Path) -> None:
    lines = ["# Spike 1: Ground-truth per-channel sensitivity", ""]
    lines.append(f"Calibration tokens: {results['calibration']['actual_tokens']} from WikiText-2.")
    lines.append(f"Analyzed layers: {', '.join(str(layer['layer_idx']) for layer in results['layers'])}.")
    lines.append("")

    role_group = aggregate_by(results, "role")
    if role_group:
        lines.append("## Cross-expert summary")
        lines.append("")
        for role, stats in role_group.items():
            w1_mean = sum(stats["w1"]) / len(stats["w1"])
            w2_mean = sum(stats["w2"]) / len(stats["w2"])
            lines.append(f"- {role}: mean W1 sensitivity {w1_mean:.6f}, mean W2 sensitivity {w2_mean:.6f}.")
        lines.append("")

    layer_group = aggregate_by(results, "layer_idx")
    if layer_group:
        lines.append("## Cross-layer summary")
        lines.append("")
        for layer_idx, stats in layer_group.items():
            w1_mean = sum(stats["w1"]) / len(stats["w1"])
            w2_mean = sum(stats["w2"]) / len(stats["w2"])
            lines.append(f"- Layer {layer_idx}: mean W1 sensitivity {w1_mean:.6f}, mean W2 sensitivity {w2_mean:.6f}.")
        lines.append("")

    lines.append("## Per-expert notes")
    lines.append("")
    for layer in results["layers"]:
        lines.append(f"### Layer {layer['layer_idx']} ({layer['layer_type']})")
        lines.append("")
        for expert in layer["experts"]:
            top_w1 = expert["top_w1_pairs"][:3]
            top_w2 = expert["top_w2_channels"][:3]
            top_w1_text = ", ".join(f"{item['index']} ({item['score']:.6f})" for item in top_w1)
            top_w2_text = ", ".join(f"{item['index']} ({item['score']:.6f})" for item in top_w2)
            lines.append(
                f"- Expert {expert['expert_id']} ({expert['role']}, routed {expert['token_count']} tokens): "
                f"W1 top pairs [{top_w1_text}] ; W2 top channels [{top_w2_text}]."
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def generate_plots(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_histograms(results, output_dir)
    plot_sorted_curves(results, output_dir)
    plot_cross_expert(results, output_dir)
    plot_cross_layer(results, output_dir)


def main() -> None:
    args = parse_args()
    start = time.time()
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.exploration_md.parent.mkdir(parents=True, exist_ok=True)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    input_ids, calibration_info = load_calibration_input_ids(args.model_id, args.tokens)

    requested_layers = sorted(set(args.layer_indices))
    if args.max_target_layer is not None:
        requested_layers = [layer for layer in requested_layers if layer <= args.max_target_layer]
    if not requested_layers:
        raise ValueError("No target layers remain after applying --max-target-layer")

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "dtype": args.dtype,
        "calibration": calibration_info,
        "target_layers": requested_layers,
        "layers": [],
        "runtime_seconds": None,
    }

    captures = run_sequential_forward(store, config, input_ids, requested_layers, device, dtype)

    for layer_idx in requested_layers:
        capture = captures[layer_idx]
        layer_result = {
            "layer_idx": layer_idx,
            "layer_type": capture.layer_type,
            "num_active_experts": int((capture.expert_counts > 0).sum().item()),
            "expert_choices": pick_experts(capture, args.experts_per_layer),
            "experts": [],
        }
        results["layers"].append(layer_result)
        for expert_choice in layer_result["expert_choices"]:
            expert_result = analyze_expert(store, config, capture, expert_choice, device, dtype)
            expert_result["layer_idx"] = layer_idx
            layer_result["experts"].append(expert_result)
            maybe_save_results(args.output_json, results)
        maybe_save_results(args.output_json, results)

    results["runtime_seconds"] = round(time.time() - start, 3)
    maybe_save_results(args.output_json, results)
    generate_plots(results, args.output_json.parent)
    write_findings(results, args.exploration_md)

    print(f"Saved JSON to {args.output_json}", flush=True)
    print(f"Saved plots to {args.output_json.parent}", flush=True)
    print(f"Saved findings to {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

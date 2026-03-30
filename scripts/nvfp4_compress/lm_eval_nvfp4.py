#!/usr/bin/env python3
"""Standalone lm-eval wrapper for the pre-quantized NVFP4 Qwen3Next checkpoint.

This script streams one transformer layer at a time to a single GPU, implements
the lm-evaluation-harness model interface in eager PyTorch, and can be run
directly for zero-shot MMLU / GSM8K evaluation.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lm_eval import evaluator as lm_eval_evaluator
from lm_eval import utils as lm_eval_utils
from lm_eval.api.model import TemplateLM
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    torch_chunk_gated_delta_rule,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (
    REPO_ROOT / "scripts" / "channel_quant_new",
    REPO_ROOT / "scripts" / "channel_quant",
):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)


import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401,E402
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (  # noqa: E402
    fp4_global_scale,
)


BF16_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MAX_GEN_TOKS = 256
DEFAULT_MAX_BATCH_TOTAL_TOKENS = 17760
LIBC = ctypes.CDLL("libc.so.6")


def resolve_default_ckpt_dir() -> Path:
    candidates = [
        Path("/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint"),
        REPO_ROOT / "scripts" / "nvfp4_compress" / "nvfp4_checkpoint",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def move_tensor(
    tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dtype is None:
        return tensor.to(device=device, non_blocking=True)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def build_text_config(root_config: dict[str, Any]) -> Qwen3NextConfig:
    text_cfg = root_config.get("text_config", root_config)
    rope_params = text_cfg.get("rope_parameters", {})
    return Qwen3NextConfig(
        vocab_size=text_cfg["vocab_size"],
        hidden_size=text_cfg["hidden_size"],
        intermediate_size=text_cfg.get(
            "intermediate_size", text_cfg["moe_intermediate_size"]
        ),
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


def rms_norm_qwen3_next(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(dtype=x.dtype)


def rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = weight.float() * out
    out = out * F.silu(gate.float())
    return out.to(dtype=x.dtype)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full(
        (1, 1, seq_len, seq_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    return torch.triu(mask, diagonal=1)


def truncate_tokens(
    tokens: list[int],
    max_length: int,
    side: str = "left",
) -> list[int]:
    if max_length <= 0:
        return []
    if side == "left":
        return tokens[-max_length:]
    if side == "right":
        return tokens[:max_length]
    raise ValueError(f"Unsupported truncation side: {side}")


def fit_tokens_to_context_window(
    tokens: list[int],
    max_gen_toks: int,
    max_model_len: int,
) -> tuple[list[int], int]:
    if len(tokens) + max_gen_toks <= max_model_len:
        return tokens, max_gen_toks
    return truncate_tokens(tokens, max_model_len - max_gen_toks, side="left"), max_gen_toks


def handle_stop_sequences(until: str | list[str] | None, eos: str | None) -> list[str]:
    if isinstance(until, str):
        until = [until]
    elif until is None:
        until = []
    elif not isinstance(until, list):
        raise ValueError(f"Expected stop sequences as str/list, got {until!r}")
    if eos is not None and eos not in until:
        until.append(eos)
    return until


def normalize_gen_kwargs(
    gen_kwargs: dict[str, Any],
    default_max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
) -> dict[str, Any]:
    kwargs = dict(gen_kwargs)
    until = kwargs.get("until", [])
    if not isinstance(until, list):
        until = [until]

    max_token_aliases = (
        kwargs.pop("max_gen_toks", None),
        kwargs.pop("max_new_tokens", None),
        kwargs.pop("max_tokens", None),
        kwargs.pop("max_completion_tokens", None),
    )
    max_gen_toks = next(
        (int(value) for value in max_token_aliases if value is not None),
        default_max_gen_toks,
    )

    do_sample = kwargs.get("do_sample")
    temperature = float(kwargs.get("temperature", 0.0) or 0.0)
    if do_sample is None:
        kwargs["do_sample"] = temperature > 0.0
    elif do_sample is False:
        kwargs["temperature"] = 0.0

    kwargs["until"] = until
    kwargs["max_gen_toks"] = max_gen_toks
    return kwargs


def postprocess_generated_text(
    generation: str,
    stop: list[str] | str | None,
    think_end_token: str | None,
) -> str:
    if stop:
        stop_terms = [stop] if isinstance(stop, str) else stop
        for term in stop_terms:
            if term:
                generation = generation.split(term)[0]
    if think_end_token:
        generation = generation.split(think_end_token)[-1].lstrip()
    return generation


def bf16_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.linear(input_tensor, weight, bias)


def prequant_nvfp4_linear(
    input_tensor: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    s_in = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in * weight_scale_2.float())).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d,
        weight_fp4,
        bias=None,
        input_scale=s_in,
        weight_scale=weight_scale,
        alpha=alpha,
    )
    return out.reshape(*input_tensor.shape[:-1], weight_fp4.shape[0])


def trim_host_allocator() -> None:
    LIBC.malloc_trim(0)


class NVFPCheckpointStore:
    def __init__(self, ckpt_dir: str | Path):
        ckpt_dir = Path(ckpt_dir)
        with (ckpt_dir / "model.safetensors.index.json").open() as f:
            self._weight_map = json.load(f)["weight_map"]
        self._ckpt_dir = ckpt_dir

    def load_tensors(
        self,
        keys: list[str],
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            if key in self._weight_map:
                grouped[self._weight_map[key]].append(key)

        # safe_open(device='cuda') crashes after TRT-LLM CUDA init — load CPU first.
        # posix_fadvise DONTNEED evicts pages after read to prevent host OOM.
        tensors: dict[str, torch.Tensor] = {}
        for shard_file, shard_keys in grouped.items():
            shard_path = str(self._ckpt_dir / shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    t = handle.get_tensor(key)
                    if device != "cpu":
                        t = t.to(device, non_blocking=True)
                    tensors[key] = t
            fd = os.open(shard_path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        return tensors

    def has_key(self, key: str) -> bool:
        return key in self._weight_map


class NVFP4LM(TemplateLM):
    backend = "causal"
    logits_cache = True

    def __init__(
        self,
        ckpt_dir: str | Path = resolve_default_ckpt_dir(),
        tokenizer_id: str = BF16_MODEL_ID,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = DEFAULT_MAX_LENGTH,
        max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
        batch_size: int = 128,
        max_batch_total_tokens: int = DEFAULT_MAX_BATCH_TOTAL_TOKENS,
    ) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for NVFP4 eager evaluation")

        self.ckpt_dir = Path(ckpt_dir)
        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.ckpt_dir}")

        self._torch_device = torch.device(device)
        if self._torch_device.type != "cuda":
            raise ValueError(f"NVFP4LM requires a CUDA device, got {device!r}")

        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)
        self._max_batch_total_tokens = int(max_batch_total_tokens)
        self.softmax_dtype = torch.float32

        with (self.ckpt_dir / "config.json").open() as f:
            ckpt_config = json.load(f)
        self.model_config = build_text_config(ckpt_config)
        if self.model_config.layer_types is None:
            raise ValueError("Checkpoint config is missing layer_types")
        self.layer_types = list(self.model_config.layer_types)
        self._max_length = min(int(max_length), self.model_config.max_position_embeddings)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True

        self.store = NVFPCheckpointStore(self.ckpt_dir)
        self.rotary = Qwen3NextRotaryEmbedding(
            config=self.model_config,
            device=self._torch_device,
        )
        self._causal_mask_cache: dict[int, torch.Tensor] = {}
        self._position_embedding_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        t0 = time.time()
        print(f"Loading NVFP4 checkpoint from {self.ckpt_dir} to {self.device}...", flush=True)
        self.embed_weight, self.final_norm, self.lm_head = self._load_root_tensors()
        self.layers = self._load_all_layers()
        elapsed = time.time() - t0
        print(
            f"Loaded full NVFP4 model to GPU in {elapsed:.1f}s",
            flush=True,
        )

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return str(self._torch_device)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": "eager_nvfp4",
            "checkpoint_dir": str(self.ckpt_dir),
            "tokenizer": self.tokenizer.name_or_path,
            "dtype": str(self.dtype),
            "max_length": self.max_length,
            "max_gen_toks": self.max_gen_toks,
            "max_batch_total_tokens": self._max_batch_total_tokens,
        }

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        encode_kwargs = dict(kwargs)
        if add_special_tokens is not None:
            encode_kwargs["add_special_tokens"] = add_special_tokens
        tokens = self.tokenizer.encode(string, **encode_kwargs)
        if left_truncate_len is not None:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(
        self,
        tokens: int | list[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens, **kwargs)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        return super().loglikelihood(requests, disable_tqdm=disable_tqdm)

    def _load_root_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_keys = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        tensors = self.store.load_tensors(root_keys, device=self.device)
        embed_weight = move_tensor(
            tensors["model.embed_tokens.weight"],
            self._torch_device,
            self.dtype,
        )
        final_norm = move_tensor(
            tensors["model.norm.weight"],
            self._torch_device,
            self.dtype,
        )
        lm_head = move_tensor(
            tensors["lm_head.weight"],
            self._torch_device,
            self.dtype,
        )
        return embed_weight, final_norm, lm_head

    def _append_linear_keys(self, keys: list[str], base: str) -> None:
        weight_key = f"{base}.weight"
        if self.store.has_key(f"{base}.weight_scale"):
            keys.extend(
                [
                    weight_key,
                    f"{base}.weight_scale",
                    f"{base}.weight_scale_2",
                ]
            )
        elif self.store.has_key(weight_key):
            keys.append(weight_key)

    def _load_layer(self, layer_idx: int) -> dict[str, Any]:
        prefix = f"model.layers.{layer_idx}"
        layer_type = self.layer_types[layer_idx]
        keys = [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        ]

        for base in (
            f"{prefix}.mlp.gate",
            f"{prefix}.mlp.shared_expert_gate",
            f"{prefix}.mlp.shared_expert.gate_proj",
            f"{prefix}.mlp.shared_expert.up_proj",
            f"{prefix}.mlp.shared_expert.down_proj",
        ):
            self._append_linear_keys(keys, base)

        if layer_type == "full_attention":
            keys.extend(
                [
                    f"{prefix}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight",
                ]
            )
            for base in (
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
            ):
                self._append_linear_keys(keys, base)
        else:
            keys.extend(
                [
                    f"{prefix}.linear_attn.conv1d.weight",
                    f"{prefix}.linear_attn.norm.weight",
                    f"{prefix}.linear_attn.A_log",
                    f"{prefix}.linear_attn.dt_bias",
                ]
            )
            for base in (
                f"{prefix}.linear_attn.in_proj_qkv",
                f"{prefix}.linear_attn.in_proj_z",
                f"{prefix}.linear_attn.in_proj_a",
                f"{prefix}.linear_attn.in_proj_b",
                f"{prefix}.linear_attn.out_proj",
            ):
                self._append_linear_keys(keys, base)

        proj_names = ("gate_proj", "up_proj", "down_proj")
        for expert_idx in range(self.model_config.num_experts):
            for proj_name in proj_names:
                expert_prefix = f"{prefix}.mlp.experts.{expert_idx}.{proj_name}"
                keys.extend(
                    [
                        f"{expert_prefix}.weight",
                        f"{expert_prefix}.weight_scale",
                        f"{expert_prefix}.weight_scale_2",
                    ]
                )

        tensors_raw = self.store.load_tensors(keys, device=self.device)
        layer: dict[str, Any] = {"layer_idx": layer_idx, "layer_type": layer_type}
        for key, tensor in tensors_raw.items():
            short = key.removeprefix(f"{prefix}.")
            if short in {"linear_attn.A_log", "linear_attn.dt_bias"}:
                layer[short] = move_tensor(tensor, self._torch_device, torch.float32)
            elif tensor.is_floating_point():
                layer[short] = move_tensor(tensor, self._torch_device, self.dtype)
            else:
                layer[short] = tensor.to(self._torch_device)

        experts: dict[str, dict[str, torch.Tensor]] = {}
        for proj_name in proj_names:
            weights = []
            weight_scales = []
            weight_scale_2 = []
            for expert_idx in range(self.model_config.num_experts):
                expert_prefix = f"mlp.experts.{expert_idx}.{proj_name}"
                weights.append(layer.pop(f"{expert_prefix}.weight"))
                weight_scales.append(layer.pop(f"{expert_prefix}.weight_scale"))
                weight_scale_2.append(layer.pop(f"{expert_prefix}.weight_scale_2"))
            experts[proj_name] = {
                "weight": torch.stack(weights, dim=0).contiguous(),
                "weight_scale": torch.stack(weight_scales, dim=0).contiguous(),
                "weight_scale_2": torch.stack(weight_scale_2, dim=0).contiguous(),
            }
        layer["experts"] = experts
        return layer

    def _load_all_layers(self) -> list[dict[str, Any]]:
        layers: list[dict[str, Any]] = []
        for layer_idx in range(self.model_config.num_hidden_layers):
            layer = self._load_layer(layer_idx)
            layers.append(layer)
            print(
                f"  loaded layer {layer_idx + 1}/{self.model_config.num_hidden_layers} "
                f"({layer['layer_type']}) GPU_MB={torch.cuda.memory_allocated()/1024/1024:.1f}",
                flush=True,
            )
        return layers

    def _unload_layer(self, layer: dict[str, Any]) -> None:
        experts = layer.pop("experts", None)
        if isinstance(experts, dict):
            for proj_group in experts.values():
                if isinstance(proj_group, dict):
                    proj_group.clear()
            experts.clear()
        layer.clear()
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        trim_host_allocator()

    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len not in self._causal_mask_cache:
            if len(self._causal_mask_cache) >= 8:
                self._causal_mask_cache.clear()
            self._causal_mask_cache[seq_len] = build_causal_mask(
                seq_len,
                self._torch_device,
            )
        return self._causal_mask_cache[seq_len]

    def _get_position_embeddings(
        self,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len not in self._position_embedding_cache:
            if len(self._position_embedding_cache) >= 4:
                self._position_embedding_cache.clear()
            position_ids = torch.arange(seq_len, device=self._torch_device).unsqueeze(0)
            dummy = torch.empty(
                (1, seq_len, self.model_config.hidden_size),
                device=self._torch_device,
                dtype=self.dtype,
            )
            self._position_embedding_cache[seq_len] = self.rotary(dummy, position_ids)
        return self._position_embedding_cache[seq_len]

    def _layer_linear(
        self,
        layer: dict[str, Any],
        key_base: str,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if f"{key_base}.weight_scale" in layer:
            return prequant_nvfp4_linear(
                input_tensor,
                layer[f"{key_base}.weight"],
                layer[f"{key_base}.weight_scale"],
                layer[f"{key_base}.weight_scale_2"],
            )
        return bf16_linear(input_tensor, layer[f"{key_base}.weight"])

    def _expert_linear(
        self,
        experts: dict[str, dict[str, torch.Tensor]],
        proj_name: str,
        expert_idx: int,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        expert_group = experts[proj_name]
        return prequant_nvfp4_linear(
            input_tensor,
            expert_group["weight"][expert_idx],
            expert_group["weight_scale"][expert_idx],
            expert_group["weight_scale_2"][expert_idx],
        )

    def _full_attention_forward(
        self,
        hidden_states: torch.Tensor,
        layer: dict[str, Any],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.model_config.head_dim
        hidden_shape = (batch_size, seq_len, -1, head_dim)

        q_proj = self._layer_linear(layer, "self_attn.q_proj", hidden_states)
        query_states, gate = torch.chunk(
            q_proj.view(batch_size, seq_len, -1, head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        query_states = rms_norm_qwen3_next(
            query_states.view(hidden_shape),
            layer["self_attn.q_norm.weight"],
            self.model_config.rms_norm_eps,
        ).transpose(1, 2)
        key_states = rms_norm_qwen3_next(
            self._layer_linear(layer, "self_attn.k_proj", hidden_states).view(hidden_shape),
            layer["self_attn.k_norm.weight"],
            self.model_config.rms_norm_eps,
        ).transpose(1, 2)
        value_states = self._layer_linear(
            layer,
            "self_attn.v_proj",
            hidden_states,
        ).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        key_states = repeat_kv(
            key_states,
            self.model_config.num_attention_heads
            // self.model_config.num_key_value_heads,
        )
        value_states = repeat_kv(
            value_states,
            self.model_config.num_attention_heads
            // self.model_config.num_key_value_heads,
        )

        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(2, 3),
        ) * (head_dim**-0.5)
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(hidden_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self._layer_linear(layer, "self_attn.o_proj", attn_output)

    def _linear_attention_forward(
        self,
        hidden_states: torch.Tensor,
        layer: dict[str, Any],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        key_dim = self.model_config.linear_key_head_dim * self.model_config.linear_num_key_heads
        value_dim = (
            self.model_config.linear_value_head_dim
            * self.model_config.linear_num_value_heads
        )
        num_k_heads = self.model_config.linear_num_key_heads
        num_v_heads = self.model_config.linear_num_value_heads
        head_k_dim = self.model_config.linear_key_head_dim
        head_v_dim = self.model_config.linear_value_head_dim
        heads_ratio = num_v_heads // num_k_heads

        projected_qkv = self._layer_linear(layer, "linear_attn.in_proj_qkv", hidden_states)
        projected_z = self._layer_linear(layer, "linear_attn.in_proj_z", hidden_states)
        projected_b = self._layer_linear(layer, "linear_attn.in_proj_b", hidden_states)
        projected_a = self._layer_linear(layer, "linear_attn.in_proj_a", hidden_states)

        query = projected_qkv[..., :key_dim].reshape(
            batch_size,
            seq_len,
            num_k_heads,
            head_k_dim,
        )
        key = projected_qkv[..., key_dim : key_dim * 2].reshape(
            batch_size,
            seq_len,
            num_k_heads,
            head_k_dim,
        )
        value = projected_qkv[..., key_dim * 2 :].reshape(
            batch_size,
            seq_len,
            num_v_heads,
            head_v_dim,
        )
        z = projected_z.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
        b = projected_b.reshape(batch_size, seq_len, num_v_heads)
        a = projected_a.reshape(batch_size, seq_len, num_v_heads)

        mixed_qkv = torch.cat(
            [
                query.reshape(batch_size, seq_len, -1),
                key.reshape(batch_size, seq_len, -1),
                value.reshape(batch_size, seq_len, -1),
            ],
            dim=-1,
        ).transpose(1, 2)
        conv_out = F.conv1d(
            mixed_qkv,
            layer["linear_attn.conv1d.weight"],
            bias=None,
            padding=self.model_config.linear_conv_kernel_dim - 1,
            groups=mixed_qkv.shape[1],
        )[:, :, :seq_len]
        mixed_qkv = F.silu(conv_out).transpose(1, 2)

        query = mixed_qkv[..., :key_dim].reshape(
            batch_size,
            seq_len,
            num_k_heads,
            head_k_dim,
        )
        key = mixed_qkv[..., key_dim : key_dim * 2].reshape(
            batch_size,
            seq_len,
            num_k_heads,
            head_k_dim,
        )
        value = mixed_qkv[..., key_dim * 2 :].reshape(
            batch_size,
            seq_len,
            num_v_heads,
            head_v_dim,
        )

        beta = b.sigmoid()
        g = -layer["linear_attn.A_log"].float().exp().view(1, 1, -1) * F.softplus(
            a.float() + layer["linear_attn.dt_bias"].float().view(1, 1, -1)
        )
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
        gated = rms_norm_gated(
            core_attn_out,
            layer["linear_attn.norm.weight"],
            z.reshape(-1, head_v_dim),
            self.model_config.rms_norm_eps,
        )
        gated = gated.reshape(batch_size, seq_len, num_v_heads, head_v_dim).reshape(
            batch_size,
            seq_len,
            -1,
        )
        return self._layer_linear(layer, "linear_attn.out_proj", gated)

    def _moe_forward(
        self,
        hidden_states: torch.Tensor,
        layer: dict[str, Any],
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)

        router_logits = self._layer_linear(layer, "mlp.gate", flat).float()
        routing_weights = torch.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.model_config.num_experts_per_tok,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_counts = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.model_config.num_experts,
        )
        experts = layer["experts"]

        for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
            token_idx, route_pos = torch.where(selected_experts == expert_idx)
            expert_input = flat[token_idx]
            gate = self._expert_linear(experts, "gate_proj", expert_idx, expert_input)
            up = self._expert_linear(experts, "up_proj", expert_idx, expert_input)
            intermediate = F.silu(gate) * up
            current_hidden = self._expert_linear(
                experts,
                "down_proj",
                expert_idx,
                intermediate,
            )
            current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

        shared_gate_proj = self._layer_linear(layer, "mlp.shared_expert.gate_proj", flat)
        shared_up_proj = self._layer_linear(layer, "mlp.shared_expert.up_proj", flat)
        shared = F.silu(shared_gate_proj) * shared_up_proj
        shared = self._layer_linear(layer, "mlp.shared_expert.down_proj", shared)
        shared_gate = torch.sigmoid(
            self._layer_linear(layer, "mlp.shared_expert_gate", flat)
        )
        final_hidden_states = final_hidden_states + shared_gate * shared
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

    @torch.inference_mode()
    def _forward_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self._torch_device)

        hidden_states = F.embedding(input_ids, self.embed_weight)
        seq_len = hidden_states.shape[1]
        print(
            f"  forward pass: batch={hidden_states.shape[0]} seq_len={seq_len}",
            flush=True,
        )
        attention_mask = self._get_causal_mask(seq_len)
        position_embeddings = self._get_position_embeddings(seq_len)

        for layer in self.layers:
            residual = hidden_states
            hidden_states = rms_norm_qwen3_next(
                hidden_states,
                layer["input_layernorm.weight"],
                self.model_config.rms_norm_eps,
            )
            if layer["layer_type"] == "full_attention":
                hidden_states = self._full_attention_forward(
                    hidden_states,
                    layer,
                    position_embeddings,
                    attention_mask,
                )
            else:
                hidden_states = self._linear_attention_forward(hidden_states, layer)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = rms_norm_qwen3_next(
                hidden_states,
                layer["post_attention_layernorm.weight"],
                self.model_config.rms_norm_eps,
            )
            hidden_states = residual + self._moe_forward(hidden_states, layer)

        hidden_states = rms_norm_qwen3_next(
            hidden_states,
            self.final_norm,
            self.model_config.rms_norm_eps,
        )
        return hidden_states

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self._forward_hidden_states(input_ids)
        return F.linear(hidden_states, self.lm_head)

    def _model_call(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.forward(input_ids)

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        contlen: int,
        inplen: int,
    ) -> torch.Tensor:
        return logits[inplen - contlen : inplen]

    def _score_context_continuation(
        self,
        request_key: tuple[str, str] | None,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        if not continuation_enc:
            return 0.0, True

        total_tokens = context_enc + continuation_enc
        input_tokens = total_tokens[-(self.max_length + 1) :][:-1]
        input_ids = torch.tensor(
            input_tokens,
            dtype=torch.long,
            device=self._torch_device,
        ).unsqueeze(0)
        hidden_states = self._forward_hidden_states(input_ids)[0]
        continuation_hidden = self._select_cont_toks(
            hidden_states,
            contlen=len(continuation_enc),
            inplen=input_ids.shape[1],
        )
        continuation_logits = F.log_softmax(
            F.linear(continuation_hidden, self.lm_head),
            dim=-1,
            dtype=self.softmax_dtype,
        )
        continuation_tokens = torch.tensor(
            continuation_enc,
            dtype=torch.long,
            device=self._torch_device,
        )
        greedy_tokens = continuation_logits.argmax(dim=-1)
        token_logprobs = torch.gather(
            continuation_logits,
            1,
            continuation_tokens.unsqueeze(-1),
        ).squeeze(-1)
        answer = (
            float(token_logprobs.sum().item()),
            bool(torch.equal(greedy_tokens, continuation_tokens)),
        )
        if request_key is not None:
            self.cache_hook.add_partial("loglikelihood", request_key, answer)
        return answer

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
        **_: Any,
    ) -> list[tuple[float, bool]]:
        batch_size = max(1, int(override_bs) if override_bs is not None else self.batch_size)
        results: list[tuple[float, bool]] = [(0.0, True)] * len(requests)
        prepared_requests: list[
            tuple[int, tuple[str, str] | None, list[int], list[int]]
        ] = []

        for idx, (request_key, context_enc, continuation_enc) in enumerate(requests):
            if not continuation_enc:
                answer = (0.0, True)
                results[idx] = answer
                if request_key is not None:
                    self.cache_hook.add_partial("loglikelihood", request_key, answer)
                continue

            total_tokens = context_enc + continuation_enc
            input_tokens = total_tokens[-(self.max_length + 1) :][:-1]
            prepared_requests.append((idx, request_key, input_tokens, continuation_enc))

        prepared_requests.sort(key=lambda item: len(item[2]), reverse=True)
        iterator = tqdm(
            total=len(prepared_requests),
            desc="Running loglikelihood requests",
            disable=(disable_tqdm or self.rank != 0),
        )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.eot_token_id

        start = 0
        while start < len(prepared_requests):
            max_input_len = len(prepared_requests[start][2])
            target_batch_size = max(
                1,
                min(batch_size, self._max_batch_total_tokens // max_input_len),
            )
            current_batch_size = min(target_batch_size, len(prepared_requests) - start)

            while True:
                chunk = prepared_requests[start : start + current_batch_size]
                max_input_len = max(len(input_tokens) for _, _, input_tokens, _ in chunk)
                padded_inputs = [
                    input_tokens + [pad_token_id] * (max_input_len - len(input_tokens))
                    for _, _, input_tokens, _ in chunk
                ]
                input_ids = torch.tensor(
                    padded_inputs,
                    dtype=torch.long,
                    device=self._torch_device,
                )
                try:
                    hidden_states = self._forward_hidden_states(input_ids)
                    break
                except torch.OutOfMemoryError:
                    if current_batch_size == 1:
                        raise
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    print(
                        f"  batch OOM, retrying with batch={current_batch_size}",
                        flush=True,
                    )

            for row_idx, (orig_idx, request_key, input_tokens, continuation_enc) in enumerate(chunk):
                continuation_hidden = self._select_cont_toks(
                    hidden_states[row_idx],
                    contlen=len(continuation_enc),
                    inplen=len(input_tokens),
                )
                continuation_logits = self._select_cont_toks(
                    F.log_softmax(
                        F.linear(continuation_hidden, self.lm_head),
                        dim=-1,
                        dtype=self.softmax_dtype,
                    ),
                    contlen=len(continuation_enc),
                    inplen=len(continuation_enc),
                )
                continuation_tokens = torch.tensor(
                    continuation_enc,
                    dtype=torch.long,
                    device=self._torch_device,
                )
                greedy_tokens = continuation_logits.argmax(dim=-1)
                token_logprobs = torch.gather(
                    continuation_logits,
                    1,
                    continuation_tokens.unsqueeze(-1),
                ).squeeze(-1)
                answer = (
                    float(token_logprobs.sum().item()),
                    bool(torch.equal(greedy_tokens, continuation_tokens)),
                )
                results[orig_idx] = answer
                if request_key is not None:
                    self.cache_hook.add_partial("loglikelihood", request_key, answer)

            iterator.update(len(chunk))
            start += len(chunk)
        iterator.close()
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        results: list[float] = []
        iterator = tqdm(
            requests,
            total=len(requests),
            desc="Running rolling loglikelihood requests",
            disable=(disable_tqdm or self.rank != 0),
        )
        for request in iterator:
            (string,) = request.args
            token_windows = list(
                map(
                    lm_eval_utils.make_disjoint_window,
                    lm_eval_utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            total_logprob = 0.0
            for context_tokens, continuation_tokens in token_windows:
                total_logprob += self._score_context_continuation(
                    None,
                    context_tokens,
                    continuation_tokens,
                )[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling",
                (string,),
                total_logprob,
            )
            results.append(total_logprob)
        return results

    def _sample_next_token(
        self,
        next_token_logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        if not do_sample or temperature <= 0.0:
            return next_token_logits.argmax(dim=-1)

        logits = next_token_logits / temperature
        if top_k is not None and top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            cutoff = top_values[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        generated = context.to(self._torch_device)
        do_sample = bool(generation_kwargs.get("do_sample", False))
        temperature = float(generation_kwargs.get("temperature", 0.0) or 0.0)
        top_k = generation_kwargs.get("top_k")
        top_p = generation_kwargs.get("top_p")
        context_len = generated.shape[1]

        while generated.shape[1] < max_length:
            hidden_states = self._forward_hidden_states(generated)
            next_token_logits = F.linear(hidden_states[:, -1:, :], self.lm_head)[:, 0, :].float()
            next_token = self._sample_next_token(
                next_token_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if int(next_token.item()) == self.eot_token_id:
                break

            continuation_tokens = generated[0, context_len:].tolist()
            continuation_text = self.tok_decode(
                continuation_tokens,
                skip_special_tokens=False,
            )
            if any(term and term in continuation_text for term in stop):
                break

        return generated

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        results: list[str] = []
        iterator = tqdm(
            requests,
            total=len(requests),
            desc="Running generate_until requests",
            disable=(disable_tqdm or self.rank != 0),
        )
        eos_text = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        for request in iterator:
            context, gen_kwargs = request.args
            gen_kwargs = dict(gen_kwargs)
            kwargs = normalize_gen_kwargs(gen_kwargs, self.max_gen_toks)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos_text)
            max_gen_toks = int(kwargs.pop("max_gen_toks"))

            context_tokens = self.tok_encode(context)
            context_tokens, max_gen_toks = fit_tokens_to_context_window(
                context_tokens,
                max_gen_toks=max_gen_toks,
                max_model_len=self.max_length,
            )
            context_ids = torch.tensor(
                context_tokens,
                dtype=torch.long,
                device=self._torch_device,
            ).unsqueeze(0)
            generated = self._model_generate(
                context=context_ids,
                max_length=context_ids.shape[1] + max_gen_toks,
                stop=until,
                **kwargs,
            )
            continuation_tokens = generated[0, context_ids.shape[1] :].tolist()
            continuation = self.tok_decode(
                continuation_tokens,
                skip_special_tokens=False,
            )
            continuation = postprocess_generated_text(
                continuation,
                stop=until,
                think_end_token=None,
            )
            self.cache_hook.add_partial(
                "generate_until",
                (context, gen_kwargs),
                continuation,
            )
            results.append(continuation)

        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=str(resolve_default_ckpt_dir()))
    parser.add_argument("--tokenizer-id", default=BF16_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--tasks", default="mmlu,gsm8k")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-batch-total-tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_TOTAL_TOKENS,
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKS)
    parser.add_argument("--limit", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    model = NVFP4LM(
        ckpt_dir=args.ckpt_dir,
        tokenizer_id=args.tokenizer_id,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        max_gen_toks=args.max_gen_toks,
        batch_size=args.batch_size,
        max_batch_total_tokens=args.max_batch_total_tokens,
    )
    results = lm_eval_evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results")
    print(json.dumps(results.get("results", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

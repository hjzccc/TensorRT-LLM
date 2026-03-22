#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_causal_mask,
    build_text_config,
    dtype_from_name,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    moe_forward,
    move_tensor,
    quantize_to_fp8,
    quantize_to_nvfp4_columns,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_JSON_PATH = RESULTS_DIR / "spike5_perplexity.json"
DEFAULT_MD_PATH = SCRIPT_DIR / "exploration.md"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
DEFAULT_LOGIT_CHUNK = 32
SECTION_MARKER = "## [5] End-to-End Perplexity Evaluation"


@dataclass(frozen=True)
class QuantConfig:
    name: str
    description: str
    mode: str
    fp8_fraction: float = 0.0


QUANT_CONFIGS = (
    QuantConfig("baseline_fp16", "All weights left in BF16.", "bf16"),
    QuantConfig("uniform_fp4", "All MoE expert weights simulated in NVFP4.", "fp4"),
    QuantConfig("uniform_fp8", "All MoE expert weights simulated in FP8.", "fp8"),
    QuantConfig("mixed_25pct", "Top 25% routed experts promoted to FP8, rest FP4.", "mixed", fp8_fraction=0.25),
    QuantConfig("mixed_50pct", "Top 50% routed experts promoted to FP8, rest FP4.", "mixed", fp8_fraction=0.50),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--logit-chunk-size", type=int, default=DEFAULT_LOGIT_CHUNK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def load_dataset_tokens(
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    token_count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected HuggingFace Dataset for split {split}, got {type(dataset)!r}")

    pieces: list[str] = []
    for item in dataset:
        text = str(item["text"]).strip()
        if not text:
            continue
        pieces.append(text)
        encoded = tokenizer("\n\n".join(pieces), return_tensors="pt", add_special_tokens=False)
        if int(encoded["input_ids"].shape[1]) >= token_count:
            input_ids = encoded["input_ids"][:, :token_count]
            return input_ids, {
                "split": split,
                "source_dataset": "wikitext/wikitext-2-raw-v1",
                "tokenizer": tokenizer.name_or_path,
                "requested_tokens": token_count,
                "actual_tokens": int(input_ids.shape[1]),
                "preview": tokenizer.decode(input_ids[0, : min(128, input_ids.shape[1])], skip_special_tokens=False),
            }

    raise RuntimeError(f"Unable to collect {token_count} tokens from WikiText-2 {split} split")


def expert_storage_bytes_per_weight(config: QuantConfig) -> float:
    if config.mode == "bf16":
        return 2.0
    if config.mode == "fp8":
        return 1.0
    if config.mode == "fp4":
        return 0.5
    if config.mode == "mixed":
        return 0.5 + (0.5 * config.fp8_fraction)
    raise ValueError(f"Unsupported mode: {config.mode}")


def estimate_memory_gb(total_bf16_bytes: int, model_config: Qwen3NextConfig, quant_config: QuantConfig) -> float:
    per_layer_expert_elems = model_config.num_experts * (
        (2 * model_config.moe_intermediate_size * model_config.hidden_size)
        + (model_config.hidden_size * model_config.moe_intermediate_size)
    )
    total_expert_elems = model_config.num_hidden_layers * per_layer_expert_elems
    total_expert_bf16_bytes = total_expert_elems * 2
    non_expert_bytes = total_bf16_bytes - total_expert_bf16_bytes
    effective_bytes = non_expert_bytes + (total_expert_elems * expert_storage_bytes_per_weight(quant_config))
    return round(float(effective_bytes) / 1e9, 3)


def print_routing_summary(layer_counts: dict[int, torch.Tensor], topk: int = 5) -> None:
    print("Calibration routing summary:", flush=True)
    for layer_idx in range(len(layer_counts)):
        counts = layer_counts[layer_idx]
        top = torch.topk(counts, k=min(topk, counts.numel()))
        summary = ", ".join(f"E{int(idx)}={int(val)}" for val, idx in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  Layer {layer_idx:02d}: {summary}", flush=True)


def resolve_promoted_experts(layer_counts: dict[int, torch.Tensor], fp8_fraction: float) -> dict[int, set[int]]:
    promoted: dict[int, set[int]] = {}
    for layer_idx, counts in layer_counts.items():
        promote_count = int(round(counts.numel() * fp8_fraction))
        ranked = sorted(range(counts.numel()), key=lambda expert_idx: (-int(counts[expert_idx]), expert_idx))
        promoted[layer_idx] = set(ranked[: max(0, promote_count)])
    return promoted


def quantize_linear_weight(weight: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bf16":
        return weight
    weight_t = weight.transpose(0, 1).contiguous()
    if mode == "fp8":
        quantized = quantize_to_fp8(weight_t)
    elif mode == "fp4":
        quantized = quantize_to_nvfp4_columns(weight_t)
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    return quantized.transpose(0, 1).contiguous()


def moe_forward_quantized(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Qwen3NextConfig,
    quant_config: QuantConfig,
    promoted_experts: set[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        precision = quant_config.mode
        if quant_config.mode == "mixed":
            precision = "fp8" if promoted_experts is not None and expert_idx in promoted_experts else "fp4"
        gate_up_weight = quantize_linear_weight(gate_up_proj[expert_idx], precision)
        down_weight = quantize_linear_weight(down_proj[expert_idx], precision)
        gate_up = F.linear(current_state, gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, down_weight)
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = F.linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), expert_counts

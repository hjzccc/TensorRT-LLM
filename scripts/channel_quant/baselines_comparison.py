#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

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
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_causal_mask,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    move_tensor,
    quantize_to_fp8,
    quantize_to_nvfp4_columns,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "baselines_comparison.json"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
DEFAULT_MINI_BATCH_TOKENS = 32
TARGET_FP8_FRACTION = 0.25
EMA_ALPHA = 0.9
EPS = 1e-10
RANDOM_SEED = 0
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


@dataclass
class LayerCapture:
    layer_idx: int
    layer_type: str
    mlp_input: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    expert_counts: torch.Tensor
    batch_counts: list[torch.Tensor]
    output_activation_sum: torch.Tensor


@dataclass
class LayerMetricBundle:
    routing_counts: torch.Tensor
    w1_pair_scores: torch.Tensor
    w2_channel_scores: torch.Tensor


@dataclass
class EvaluationPlan:
    name: str
    description: str
    mode: str
    memory_gb: float
    fp8_weights: int
    promoted_experts: dict[int, set[int]] | None = None
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] | None = None
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] | None = None
    w1_projection_fp8: dict[int, torch.Tensor] | None = None
    w2_projection_fp8: dict[int, torch.Tensor] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--mini-batch-tokens", type=int, default=DEFAULT_MINI_BATCH_TOKENS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def load_prefix_dataset_tokens(
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    token_count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset for split {split}, got {type(dataset)!r}")
    pieces: list[str] = []
    for raw_item in dataset:
        item = raw_item if isinstance(raw_item, dict) else dict(raw_item)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        pieces.append(text)
        encoded = tokenizer("\n\n".join(pieces), return_tensors="pt", add_special_tokens=False)
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        if int(input_ids.shape[1]) >= token_count:
            clipped = input_ids[:, :token_count]
            return clipped, {
                "split": split,
                "source_dataset": "wikitext/wikitext-2-raw-v1",
                "tokenizer": tokenizer.name_or_path,
                "actual_tokens": int(clipped.shape[1]),
                "preview": tokenizer.decode(clipped[0, : min(128, clipped.shape[1])], skip_special_tokens=False),
            }
    raise RuntimeError(f"Unable to collect {token_count} tokens from WikiText-2 {split}")


def resolve_terminal_keys(weight_map: dict[str, str]) -> tuple[str, str, str]:
    embed_key = next(key for key in weight_map if key.endswith("embed_tokens.weight") and ".layers." not in key)
    lm_head_key = next(key for key in weight_map if key.endswith("lm_head.weight"))
    norm_candidates = [
        key
        for key in weight_map
        if key.endswith("norm.weight")
        and ".layers." not in key
        and "q_norm" not in key
        and "k_norm" not in key
        and "linear_attn" not in key
    ]
    if not norm_candidates:
        raise KeyError("Unable to find final norm weight")
    return embed_key, norm_candidates[0], lm_head_key


def topk_mask_from_scores(scores: torch.Tensor, k: int) -> torch.Tensor:
    count = int(scores.numel())
    if k <= 0:
        return torch.zeros(count, dtype=torch.bool)
    if k >= count:
        return torch.ones(count, dtype=torch.bool)
    indices = torch.topk(scores, k=k, largest=True, sorted=False).indices
    mask = torch.zeros(count, dtype=torch.bool)
    mask[indices] = True
    return mask


def allocate_weighted_counts(total: int, capacities: list[int], weights: list[float]) -> list[int]:
    if len(capacities) != len(weights):
        raise ValueError("capacities and weights must have the same length")
    allocations = [0] * len(capacities)
    remaining_capacity = list(capacities)
    remaining_total = min(total, sum(capacities))
    active = {idx for idx, capacity in enumerate(capacities) if capacity > 0}
    while active and remaining_total > 0:
        weight_sum = sum(weights[idx] for idx in active)
        if weight_sum <= 0.0:
            ordered = sorted(active)
            for idx in ordered:
                if remaining_total <= 0:
                    break
                give = min(int(math.ceil(remaining_total / len(ordered))), remaining_capacity[idx])
                allocations[idx] += give
                remaining_capacity[idx] -= give
                remaining_total -= give
            active = {idx for idx in active if remaining_capacity[idx] > 0}
            continue
        quotas = {idx: remaining_total * weights[idx] / weight_sum for idx in active}
        floors = {idx: min(int(math.floor(quotas[idx])), remaining_capacity[idx]) for idx in active}
        floor_total = sum(floors.values())
        if floor_total == 0:
            ordered = sorted(active, key=lambda idx: (quotas[idx], weights[idx], -idx), reverse=True)
            for idx in ordered:
                if remaining_total <= 0:
                    break
                if remaining_capacity[idx] <= 0:
                    continue
                allocations[idx] += 1
                remaining_capacity[idx] -= 1
                remaining_total -= 1
            active = {idx for idx in active if remaining_capacity[idx] > 0}
            continue
        for idx, give in floors.items():
            allocations[idx] += give
            remaining_capacity[idx] -= give
        remaining_total -= floor_total
        if remaining_total <= 0:
            break
        ordered = sorted(active, key=lambda idx: (quotas[idx] - floors[idx], weights[idx], -idx), reverse=True)
        for idx in ordered:
            if remaining_total <= 0:
                break
            if remaining_capacity[idx] <= 0:
                continue
            allocations[idx] += 1
            remaining_capacity[idx] -= 1
            remaining_total -= 1
        active = {idx for idx in active if remaining_capacity[idx] > 0}
    return allocations


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


def quantize_gate_up_with_pair_mask(weight: torch.Tensor, fp8_pair_mask: torch.Tensor) -> torch.Tensor:
    if int(fp8_pair_mask.numel() * 2) != int(weight.shape[0]):
        raise ValueError(f"Expected pair mask length {weight.shape[0] // 2}, got {fp8_pair_mask.numel()}")
    if not bool(fp8_pair_mask.any()):
        return quantize_linear_weight(weight, "fp4")
    if bool(fp8_pair_mask.all()):
        return quantize_linear_weight(weight, "fp8")
    weight_t = weight.transpose(0, 1).contiguous()
    fp4_t = quantize_to_nvfp4_columns(weight_t)
    fp8_t = quantize_to_fp8(weight_t)
    row_mask = torch.cat([fp8_pair_mask, fp8_pair_mask], dim=0).to(device=weight_t.device, dtype=torch.bool)
    mixed_t = torch.where(row_mask.view(1, -1), fp8_t, fp4_t)
    return mixed_t.transpose(0, 1).contiguous()


def quantize_down_proj_with_mask(weight: torch.Tensor, fp8_mask: torch.Tensor) -> torch.Tensor:
    if not bool(fp8_mask.any()):
        return quantize_linear_weight(weight, "fp4")
    if bool(fp8_mask.all()):
        return quantize_linear_weight(weight, "fp8")
    weight_t = weight.transpose(0, 1).contiguous()
    fp4_t = quantize_to_nvfp4_columns(weight_t)
    fp8_t = quantize_to_fp8(weight_t)
    column_mask = fp8_mask.to(device=weight_t.device, dtype=torch.bool).view(1, -1)
    mixed_t = torch.where(column_mask, fp8_t, fp4_t)
    return mixed_t.transpose(0, 1).contiguous()


def expert_full_weight_count(config: Any) -> int:
    return int((2 * config.moe_intermediate_size * config.hidden_size) + (config.hidden_size * config.moe_intermediate_size))


def resolve_non_expert_bytes(total_bf16_bytes: int, config: Any) -> tuple[int, int]:
    per_layer_expert_elems = config.num_experts * expert_full_weight_count(config)
    total_expert_elems = int(config.num_hidden_layers * per_layer_expert_elems)
    total_expert_bf16_bytes = total_expert_elems * 2
    non_expert_bytes = total_bf16_bytes - total_expert_bf16_bytes
    return non_expert_bytes, total_expert_elems


def estimate_mixed_memory_gb(non_expert_bytes: int, total_expert_elems: int, fp8_weights: int) -> float:
    expert_bytes = int((0.5 * total_expert_elems) + (0.5 * fp8_weights))
    return round(float(non_expert_bytes + expert_bytes) / 1e9, 3)


def estimate_baseline_memory_gb(total_bf16_bytes: int, non_expert_bytes: int, total_expert_elems: int, mode: str) -> float:
    if mode == "bf16":
        return round(float(total_bf16_bytes) / 1e9, 3)
    if mode == "fp4":
        return round(float(non_expert_bytes + int(0.5 * total_expert_elems)) / 1e9, 3)
    if mode == "fp8":
        return round(float(non_expert_bytes + total_expert_elems) / 1e9, 3)
    raise ValueError(f"Unsupported memory mode: {mode}")


def fp8_weights_from_masks(config: Any, w1_pair_masks: dict[int, dict[int, torch.Tensor]], w2_channel_masks: dict[int, dict[int, torch.Tensor]]) -> int:
    w1_pairs = sum(int(mask.sum().item()) for layer_masks in w1_pair_masks.values() for mask in layer_masks.values())
    w2_channels = sum(int(mask.sum().item()) for layer_masks in w2_channel_masks.values() for mask in layer_masks.values())
    return int((w1_pairs * 2 * config.hidden_size) + (w2_channels * config.moe_intermediate_size))


def fp8_weights_from_promoted_experts(config: Any, promoted_experts: dict[int, set[int]]) -> int:
    promoted_count = sum(len(experts) for experts in promoted_experts.values())
    return promoted_count * expert_full_weight_count(config)


def fp8_weights_from_projection_promotions(config: Any, w1_projection_fp8: dict[int, torch.Tensor], w2_projection_fp8: dict[int, torch.Tensor]) -> int:
    w1_count = sum(int(mask.sum().item()) for mask in w1_projection_fp8.values())
    w2_count = sum(int(mask.sum().item()) for mask in w2_projection_fp8.values())
    return int((w1_count * 2 * config.moe_intermediate_size * config.hidden_size) + (w2_count * config.hidden_size * config.moe_intermediate_size))


def fp8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    w2d = w.contiguous()
    sx = (x2d.abs().amax().clamp(min=1e-12) / E4M3_MAX).float()
    sw = (w2d.abs().amax().clamp(min=1e-12) / E4M3_MAX).float()
    x8 = (x2d.float() / sx).to(torch.float8_e4m3fn)
    w8 = (w2d.float() / sw).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    return out.view(*x.shape[:-1], w.shape[0])


def apply_linear(x: torch.Tensor, w: torch.Tensor, quantized: bool) -> torch.Tensor:
    if quantized:
        return fp8_linear(x, w)
    return F.linear(x, w)


def moe_forward_capture(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    output_activation_sum = torch.zeros(config.num_experts, dtype=torch.float32, device=hidden_states.device)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, down_proj[expert_idx])
        output_activation_sum[expert_idx] = current_hidden.abs().sum().float()
        final_hidden_states.index_add_(0, token_idx, (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden).to(hidden_states.dtype))

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), selected_experts, routing_weights, expert_counts, output_activation_sum


def run_calibration_pass(
    store: WeightStore,
    config: Any,
    input_ids: torch.Tensor,
    mini_batch_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, LayerCapture]:
    print("\n=== Calibration (BF16, shared across baselines) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    embed_weight = move_tensor(store.load_tensors([embed_key])[embed_key], device, dtype)
    layer_buffers: dict[int, dict[str, Any]] = {
        layer_idx: {
            "mlp_input": [],
            "selected_experts": [],
            "routing_weights": [],
            "expert_counts": torch.zeros(config.num_experts, dtype=torch.int64),
            "batch_counts": [],
            "output_activation_sum": torch.zeros(config.num_experts, dtype=torch.float32),
        }
        for layer_idx in range(config.num_hidden_layers)
    }

    chunk_starts = list(range(0, int(input_ids.shape[1]), mini_batch_tokens))
    with torch.inference_mode():
        for chunk_idx, begin in enumerate(chunk_starts, start=1):
            end = min(begin + mini_batch_tokens, int(input_ids.shape[1]))
            chunk_ids = input_ids[:, begin:end]
            hidden_states = F.embedding(chunk_ids.to(device), embed_weight)
            seq_len = int(chunk_ids.shape[1])
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            causal_mask = build_causal_mask(seq_len, device)
            rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
            position_embeddings = rotary(hidden_states, position_ids)
            print(f"[calib] mini-batch {chunk_idx}/{len(chunk_starts)} tokens={begin}:{end}", flush=True)

            for layer_idx in range(config.num_hidden_layers):
                layer_type = config.layer_types[layer_idx]
                raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
                tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
                del raw_tensors

                residual = hidden_states
                hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
                if layer_type == "linear_attention":
                    attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = linear_attention_forward(hidden_norm, attn_tensors, config)
                else:
                    attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                    hidden_states = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
                hidden_states = residual + hidden_states

                residual = hidden_states
                mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
                moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
                moe_out, selected_experts, routing_weights, expert_counts, output_activation_sum = moe_forward_capture(mlp_input, moe_tensors, config)
                hidden_states = residual + moe_out

                buffer = layer_buffers[layer_idx]
                buffer["mlp_input"].append(mlp_input.detach().cpu().squeeze(0).to(torch.float32))
                buffer["selected_experts"].append(selected_experts.detach().cpu())
                buffer["routing_weights"].append(routing_weights.detach().cpu().to(torch.float32))
                counts_cpu = expert_counts.detach().cpu().to(torch.int64)
                buffer["expert_counts"] += counts_cpu
                buffer["batch_counts"].append(counts_cpu.clone())
                buffer["output_activation_sum"] += output_activation_sum.detach().cpu()
                release_tensors(tensors)

            del chunk_ids, hidden_states, position_ids, causal_mask, rotary, position_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    captures: dict[int, LayerCapture] = {}
    for layer_idx in range(config.num_hidden_layers):
        buffer = layer_buffers[layer_idx]
        captures[layer_idx] = LayerCapture(
            layer_idx=layer_idx,
            layer_type=config.layer_types[layer_idx],
            mlp_input=torch.cat(buffer["mlp_input"], dim=0),
            selected_experts=torch.cat(buffer["selected_experts"], dim=0),
            routing_weights=torch.cat(buffer["routing_weights"], dim=0),
            expert_counts=buffer["expert_counts"],
            batch_counts=buffer["batch_counts"],
            output_activation_sum=buffer["output_activation_sum"],
        )
        active = int((captures[layer_idx].expert_counts > 0).sum().item())
        print(f"[calib] layer {layer_idx:02d}: active experts={active}", flush=True)

    release_tensors({"embed_weight": embed_weight})
    return captures


def compute_gate_aware_projection_metrics(
    store: WeightStore,
    config: Any,
    captures: dict[int, LayerCapture],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, LayerMetricBundle]:
    print("\n=== Gate-Aware Channel Metrics ===", flush=True)
    metric_cache: dict[int, LayerMetricBundle] = {}
    for layer_idx in range(config.num_hidden_layers):
        capture = captures[layer_idx]
        print(f"[metric] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        raw = store.load_tensors([gate_key, down_key])
        gate_up_proj = move_tensor(raw[gate_key], device, dtype)
        down_proj = move_tensor(raw[down_key], device, dtype)
        del raw

        w1_pair_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
        w2_channel_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()

        for expert_idx in active_experts:
            token_idx, route_pos = torch.where(capture.selected_experts == expert_idx)
            if int(token_idx.numel()) == 0:
                continue
            x = capture.mlp_input[token_idx].to(device=device, dtype=dtype)
            route_weights = capture.routing_weights[token_idx, route_pos].to(device=device, dtype=torch.float32)
            denom = route_weights.sum().clamp_min(EPS)

            gate_up_weight = gate_up_proj[expert_idx]
            down_weight = down_proj[expert_idx]
            gate_up_fp8 = quantize_linear_weight(gate_up_weight, "fp8")
            gate_up_fp4 = quantize_linear_weight(gate_up_weight, "fp4")

            gate_up_out_fp8 = F.linear(x, gate_up_fp8)
            gate_up_out_fp4 = F.linear(x, gate_up_fp4)
            gate_fp8, up_fp8 = gate_up_out_fp8.chunk(2, dim=-1)
            gate_fp4, up_fp4 = gate_up_out_fp4.chunk(2, dim=-1)
            intermediate_fp8 = (F.silu(gate_fp8.float()) * up_fp8.float()).to(torch.float32)
            intermediate_fp4 = (F.silu(gate_fp4.float()) * up_fp4.float()).to(torch.float32)
            delta_hidden_abs = ((intermediate_fp4 - intermediate_fp8).abs() * route_weights.unsqueeze(-1)).sum(dim=0) / denom
            gate_weighted_abs = (intermediate_fp8.abs() * route_weights.unsqueeze(-1)).sum(dim=0) / denom

            down_fp4 = quantize_linear_weight(down_weight, "fp4")
            down_delta_abs = (down_weight.float() - down_fp4.float()).abs()
            down_col_norm = torch.linalg.vector_norm(down_weight.float(), dim=0)

            w1_pair_scores[expert_idx] = (delta_hidden_abs * down_col_norm).cpu()
            w2_channel_scores[expert_idx] = torch.einsum("k,nk->n", gate_weighted_abs, down_delta_abs).cpu()

        release_tensors({"gate_up_proj": gate_up_proj, "down_proj": down_proj})
        metric_cache[layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_pair_scores,
            w2_channel_scores=w2_channel_scores,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metric_cache


def compute_mxmoe_projection_deltas(
    store: WeightStore,
    config: Any,
    captures: dict[int, LayerCapture],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    print("\n=== MxMoE Output Perturbation ===", flush=True)
    w1_deltas: dict[int, torch.Tensor] = {}
    w2_deltas: dict[int, torch.Tensor] = {}
    for layer_idx in range(config.num_hidden_layers):
        capture = captures[layer_idx]
        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()
        print(f"[mxmoe] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d} active={len(active_experts)}", flush=True)
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        raw = store.load_tensors([gate_key, down_key])
        gate_up_proj = move_tensor(raw[gate_key], device, dtype)
        down_proj = move_tensor(raw[down_key], device, dtype)
        del raw

        layer_w1 = torch.zeros(config.num_experts, dtype=torch.float32)
        layer_w2 = torch.zeros(config.num_experts, dtype=torch.float32)
        for expert_pos, expert_idx in enumerate(active_experts, start=1):
            if expert_pos == 1 or expert_pos % 8 == 0 or expert_pos == len(active_experts):
                print(f"  [mxmoe] expert {expert_pos}/{len(active_experts)} (E{expert_idx})", flush=True)
            token_idx, route_pos = torch.where(capture.selected_experts == expert_idx)
            if int(token_idx.numel()) == 0:
                continue
            x = capture.mlp_input[token_idx].to(device=device, dtype=dtype)
            route_weights = capture.routing_weights[token_idx, route_pos].to(device=device, dtype=torch.float32)
            gate_up_weight = gate_up_proj[expert_idx]
            down_weight = down_proj[expert_idx]

            full_gate_up = F.linear(x, gate_up_weight)
            full_gate, full_up = full_gate_up.chunk(2, dim=-1)
            full_hidden = F.silu(full_gate) * full_up
            full_out = F.linear(full_hidden, down_weight).float()

            quant_w1 = quantize_linear_weight(gate_up_weight, "fp4")
            q_gate_up = F.linear(x, quant_w1)
            q_gate, q_up = q_gate_up.chunk(2, dim=-1)
            q_hidden = F.silu(q_gate) * q_up
            q_out_w1 = F.linear(q_hidden, down_weight).float()
            delta_w1 = (route_weights.unsqueeze(-1) * (q_out_w1 - full_out)).float()
            layer_w1[expert_idx] = torch.linalg.vector_norm(delta_w1).cpu()

            quant_w2 = quantize_linear_weight(down_weight, "fp4")
            q_out_w2 = F.linear(full_hidden, quant_w2).float()
            delta_w2 = (route_weights.unsqueeze(-1) * (q_out_w2 - full_out)).float()
            layer_w2[expert_idx] = torch.linalg.vector_norm(delta_w2).cpu()

        w1_deltas[layer_idx] = layer_w1
        w2_deltas[layer_idx] = layer_w2
        release_tensors({"gate_up_proj": gate_up_proj, "down_proj": down_proj})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return w1_deltas, w2_deltas


def compute_mc_moe_importance(
    store: WeightStore,
    config: Any,
    captures: dict[int, LayerCapture],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    print("\n=== MC-MoE Importance ===", flush=True)
    importance: dict[int, torch.Tensor] = {}
    alpha = 1.0
    beta = 1.5
    gamma = 2.0
    for layer_idx in range(config.num_hidden_layers):
        print(f"[mc-moe] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        raw = store.load_tensors([gate_key, down_key])
        gate_up_proj = move_tensor(raw[gate_key], device, dtype)
        down_proj = move_tensor(raw[down_key], device, dtype)
        del raw

        scores = torch.zeros(config.num_experts, dtype=torch.float64)
        act_counts = captures[layer_idx].expert_counts.to(torch.float64)
        for expert_idx in range(config.num_experts):
            gate_up_weight = gate_up_proj[expert_idx]
            down_weight = down_proj[expert_idx]
            gate_up_fp4 = quantize_linear_weight(gate_up_weight, "fp4")
            down_fp4 = quantize_linear_weight(down_weight, "fp4")

            weight_norm_sq = gate_up_weight.float().pow(2).sum() + down_weight.float().pow(2).sum()
            quant_loss_sq = (gate_up_weight.float() - gate_up_fp4.float()).pow(2).sum() + (down_weight.float() - down_fp4.float()).pow(2).sum()
            weight_norm = float(torch.sqrt(weight_norm_sq).item())
            quant_loss = float(torch.sqrt(quant_loss_sq).item())
            actnum = float(act_counts[expert_idx].item())
            scores[expert_idx] = (actnum ** alpha) * (weight_norm ** beta) * (quant_loss ** gamma)

        importance[layer_idx] = scores.to(torch.float32)
        release_tensors({"gate_up_proj": gate_up_proj, "down_proj": down_proj})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return importance


def build_random_channel_masks(config: Any, seed: int) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    k_w1 = int(round(TARGET_FP8_FRACTION * config.moe_intermediate_size))
    k_w2 = int(round(TARGET_FP8_FRACTION * config.hidden_size))
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx in range(config.num_hidden_layers):
        layer_w1: dict[int, torch.Tensor] = {}
        layer_w2: dict[int, torch.Tensor] = {}
        for expert_idx in range(config.num_experts):
            w1_mask = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
            w2_mask = torch.zeros(config.hidden_size, dtype=torch.bool)
            if k_w1 > 0:
                w1_indices = torch.randperm(config.moe_intermediate_size, generator=generator)[:k_w1]
                w1_mask[w1_indices] = True
            if k_w2 > 0:
                w2_indices = torch.randperm(config.hidden_size, generator=generator)[:k_w2]
                w2_mask[w2_indices] = True
            layer_w1[expert_idx] = w1_mask
            layer_w2[expert_idx] = w2_mask
        w1_pair_masks[layer_idx] = layer_w1
        w2_channel_masks[layer_idx] = layer_w2
    return w1_pair_masks, w2_channel_masks


def build_uniform_channel_masks(metric_cache: dict[int, LayerMetricBundle], config: Any) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    k_w1 = int(round(TARGET_FP8_FRACTION * config.moe_intermediate_size))
    k_w2 = int(round(TARGET_FP8_FRACTION * config.hidden_size))
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        w1_pair_masks[layer_idx] = {
            expert_idx: topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], k_w1)
            for expert_idx in range(config.num_experts)
        }
        w2_channel_masks[layer_idx] = {
            expert_idx: topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], k_w2)
            for expert_idx in range(config.num_experts)
        }
    return w1_pair_masks, w2_channel_masks


def build_two_level_masks(metric_cache: dict[int, LayerMetricBundle], config: Any) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        total_w1 = int(round(TARGET_FP8_FRACTION * config.moe_intermediate_size * config.num_experts))
        total_w2 = int(round(TARGET_FP8_FRACTION * config.hidden_size * config.num_experts))
        weights = [float(value) for value in bundle.routing_counts.tolist()]
        w1_alloc = allocate_weighted_counts(total_w1, [config.moe_intermediate_size] * config.num_experts, weights)
        w2_alloc = allocate_weighted_counts(total_w2, [config.hidden_size] * config.num_experts, weights)
        w1_pair_masks[layer_idx] = {
            expert_idx: topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], int(w1_alloc[expert_idx]))
            for expert_idx in range(config.num_experts)
        }
        w2_channel_masks[layer_idx] = {
            expert_idx: topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], int(w2_alloc[expert_idx]))
            for expert_idx in range(config.num_experts)
        }
    return w1_pair_masks, w2_channel_masks


def build_topk_expert_promotions(score_map: dict[int, torch.Tensor], config: Any, fraction: float) -> dict[int, set[int]]:
    target = int(round(config.num_experts * fraction))
    promoted: dict[int, set[int]] = {}
    for layer_idx, scores in score_map.items():
        ranked = sorted(range(config.num_experts), key=lambda expert_idx: (-float(scores[expert_idx].item()), expert_idx))
        promoted[layer_idx] = set(ranked[:target])
    return promoted


def kmeans_membership(scores: torch.Tensor, max_iters: int = 32) -> torch.Tensor:
    values = scores.to(torch.float32)
    if int(values.numel()) == 0:
        return torch.zeros(0, dtype=torch.float32)
    low = float(values.min().item())
    high = float(values.max().item())
    if not math.isfinite(low) or not math.isfinite(high) or abs(high - low) <= EPS:
        return torch.zeros_like(values)
    center_low = low
    center_high = high
    for _ in range(max_iters):
        dist_low = (values - center_low).abs()
        dist_high = (values - center_high).abs()
        assign_high = dist_high <= dist_low
        if not bool(assign_high.any()) or bool(assign_high.all()):
            break
        new_low = float(values[~assign_high].mean().item())
        new_high = float(values[assign_high].mean().item())
        if abs(new_low - center_low) <= 1e-6 and abs(new_high - center_high) <= 1e-6:
            center_low, center_high = new_low, new_high
            break
        center_low, center_high = new_low, new_high
    if center_high < center_low:
        center_low, center_high = center_high, center_low
    dist_low = (values - center_low).abs()
    dist_high = (values - center_high).abs()
    membership = dist_low / (dist_low + dist_high + EPS)
    return membership.cpu()


def build_dynamo_promotions(captures: dict[int, LayerCapture], config: Any, fraction: float) -> dict[int, set[int]]:
    target = int(round(config.num_experts * fraction))
    promoted: dict[int, set[int]] = {}
    for layer_idx, capture in captures.items():
        significance = capture.output_activation_sum.to(torch.float32)
        total = float(significance.sum().item())
        if total > 0.0:
            significance = significance / total
        membership = kmeans_membership(significance)
        ranked = sorted(
            range(config.num_experts),
            key=lambda expert_idx: (-float(membership[expert_idx].item()), -float(significance[expert_idx].item()), expert_idx),
        )
        promoted[layer_idx] = set(ranked[:target])
    return promoted


def build_dynaexq_promotions(captures: dict[int, LayerCapture], config: Any, fraction: float) -> tuple[dict[int, set[int]], dict[int, torch.Tensor]]:
    target = int(round(config.num_experts * fraction))
    promoted: dict[int, set[int]] = {}
    final_scores: dict[int, torch.Tensor] = {}
    for layer_idx, capture in captures.items():
        scores = torch.zeros(config.num_experts, dtype=torch.float64)
        current: set[int] = set()
        for step_idx, batch_counts in enumerate(capture.batch_counts, start=1):
            scores = (EMA_ALPHA * scores) + ((1.0 - EMA_ALPHA) * batch_counts.to(torch.float64))
            ranked = sorted(range(config.num_experts), key=lambda expert_idx: (-float(scores[expert_idx].item()), expert_idx))
            margin = 0.1 * float(scores.mean().item())
            if step_idx == 1:
                current = set(ranked[:target])
                continue
            if len(current) < target:
                for expert_idx in ranked:
                    if expert_idx in current:
                        continue
                    current.add(expert_idx)
                    if len(current) >= target:
                        break
            for expert_idx in ranked:
                if expert_idx in current:
                    continue
                weakest = min(current, key=lambda idx: (float(scores[idx].item()), idx))
                if float(scores[expert_idx].item()) > float(scores[weakest].item()) + margin:
                    current.remove(weakest)
                    current.add(expert_idx)
            if len(current) > target:
                keep = sorted(current, key=lambda idx: (-float(scores[idx].item()), idx))[:target]
                current = set(keep)
        if len(current) < target:
            ranked = sorted(range(config.num_experts), key=lambda expert_idx: (-float(scores[expert_idx].item()), expert_idx))
            for expert_idx in ranked:
                current.add(expert_idx)
                if len(current) >= target:
                    break
        promoted[layer_idx] = current
        final_scores[layer_idx] = scores.to(torch.float32)
    return promoted, final_scores


def build_mxmoe_projection_promotions(
    config: Any,
    total_expert_elems: int,
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    target_fp8_weights = int(round(TARGET_FP8_FRACTION * total_expert_elems))
    w1_cost = 2 * config.moe_intermediate_size * config.hidden_size
    w2_cost = config.hidden_size * config.moe_intermediate_size
    items: list[tuple[float, int, int, int, str]] = []
    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            delta_w1 = float(w1_deltas[layer_idx][expert_idx].item())
            delta_w2 = float(w2_deltas[layer_idx][expert_idx].item())
            items.append((delta_w1 / (w1_cost + EPS), w1_cost, layer_idx, expert_idx, "w1"))
            items.append((delta_w2 / (w2_cost + EPS), w2_cost, layer_idx, expert_idx, "w2"))
    items.sort(key=lambda item: (item[0], -item[1], -item[2], -item[3], item[4]), reverse=True)

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    fp8_weights = 0
    for ratio, cost, layer_idx, expert_idx, projection in items:
        if ratio <= 0.0:
            continue
        if fp8_weights + cost > target_fp8_weights:
            continue
        if projection == "w1":
            w1_projection_fp8[layer_idx][expert_idx] = True
        else:
            w2_projection_fp8[layer_idx][expert_idx] = True
        fp8_weights += cost
        if fp8_weights >= target_fp8_weights:
            break
    return w1_projection_fp8, w2_projection_fp8


def build_plans(
    config: Any,
    total_bf16_bytes: int,
    non_expert_bytes: int,
    total_expert_elems: int,
    metric_cache: dict[int, LayerMetricBundle],
    captures: dict[int, LayerCapture],
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
    mc_moe_scores: dict[int, torch.Tensor],
    dynamo_promoted: dict[int, set[int]],
    dynaexq_promoted: dict[int, set[int]],
) -> list[EvaluationPlan]:
    layer_counts = {layer_idx: capture.expert_counts.to(torch.float32) for layer_idx, capture in captures.items()}
    routing_promoted = build_topk_expert_promotions(layer_counts, config, TARGET_FP8_FRACTION)
    mc_moe_promoted = build_topk_expert_promotions(mc_moe_scores, config, TARGET_FP8_FRACTION)
    random_w1_masks, random_w2_masks = build_random_channel_masks(config, RANDOM_SEED)
    uniform_w1_masks, uniform_w2_masks = build_uniform_channel_masks(metric_cache, config)
    two_level_w1_masks, two_level_w2_masks = build_two_level_masks(metric_cache, config)
    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions(config, total_expert_elems, w1_deltas, w2_deltas)

    random_fp8_weights = fp8_weights_from_masks(config, random_w1_masks, random_w2_masks)
    uniform_fp8_weights = fp8_weights_from_masks(config, uniform_w1_masks, uniform_w2_masks)
    two_level_fp8_weights = fp8_weights_from_masks(config, two_level_w1_masks, two_level_w2_masks)
    routing_fp8_weights = fp8_weights_from_promoted_experts(config, routing_promoted)
    mxmoe_fp8_weights = fp8_weights_from_projection_promotions(config, mxmoe_w1_proj, mxmoe_w2_proj)
    mc_moe_fp8_weights = fp8_weights_from_promoted_experts(config, mc_moe_promoted)
    dynamo_fp8_weights = fp8_weights_from_promoted_experts(config, dynamo_promoted)
    dynaexq_fp8_weights = fp8_weights_from_promoted_experts(config, dynaexq_promoted)

    return [
        EvaluationPlan(
            name="bf16",
            description="No expert quantization.",
            mode="bf16",
            memory_gb=estimate_baseline_memory_gb(total_bf16_bytes, non_expert_bytes, total_expert_elems, "bf16"),
            fp8_weights=0,
        ),
        EvaluationPlan(
            name="uniform_fp4",
            description="All MoE expert weights in NVFP4, evaluated through real FP8 matmul.",
            mode="uniform_fp4",
            memory_gb=estimate_baseline_memory_gb(total_bf16_bytes, non_expert_bytes, total_expert_elems, "fp4"),
            fp8_weights=0,
        ),
        EvaluationPlan(
            name="uniform_fp8",
            description="All MoE expert weights in FP8.",
            mode="uniform_fp8",
            memory_gb=estimate_baseline_memory_gb(total_bf16_bytes, non_expert_bytes, total_expert_elems, "fp8"),
            fp8_weights=total_expert_elems,
        ),
        EvaluationPlan(
            name="random_25pct",
            description="Random 25% W1/W2 channel-pair promotion at fixed budget.",
            mode="mixed_channel",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, random_fp8_weights),
            fp8_weights=random_fp8_weights,
            w1_pair_masks=random_w1_masks,
            w2_channel_masks=random_w2_masks,
        ),
        EvaluationPlan(
            name="routing_per_expert",
            description="Top 25% routed experts fully promoted to FP8.",
            mode="expert_only",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, routing_fp8_weights),
            fp8_weights=routing_fp8_weights,
            promoted_experts=routing_promoted,
        ),
        EvaluationPlan(
            name="perchannel_uniform_budget",
            description="Uniform 25% per-expert W1/W2 channel promotion by gate-aware score.",
            mode="mixed_channel",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, uniform_fp8_weights),
            fp8_weights=uniform_fp8_weights,
            w1_pair_masks=uniform_w1_masks,
            w2_channel_masks=uniform_w2_masks,
        ),
        EvaluationPlan(
            name="two_level",
            description="Routing-aware expert budgeting plus per-channel gate-aware ranking.",
            mode="mixed_channel",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, two_level_fp8_weights),
            fp8_weights=two_level_fp8_weights,
            w1_pair_masks=two_level_w1_masks,
            w2_channel_masks=two_level_w2_masks,
        ),
        EvaluationPlan(
            name="mxmoe_per_block",
            description="MxMoE greedy knapsack over actual per-projection output perturbation.",
            mode="projection_block",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, mxmoe_fp8_weights),
            fp8_weights=mxmoe_fp8_weights,
            w1_projection_fp8=mxmoe_w1_proj,
            w2_projection_fp8=mxmoe_w2_proj,
        ),
        EvaluationPlan(
            name="mc_moe_per_expert",
            description="MC-MoE per-expert importance with alpha=1, beta=1.5, gamma=2.",
            mode="expert_only",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, mc_moe_fp8_weights),
            fp8_weights=mc_moe_fp8_weights,
            promoted_experts=mc_moe_promoted,
        ),
        EvaluationPlan(
            name="dynamo_fcm",
            description="DynaMo expert significance clustered with 2-means and budgeted via high-cluster membership.",
            mode="expert_only",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, dynamo_fp8_weights),
            fp8_weights=dynamo_fp8_weights,
            promoted_experts=dynamo_promoted,
        ),
        EvaluationPlan(
            name="dynaexq_ema",
            description="DynaExq EMA hotness with 4 mini-batch updates and hysteresis.",
            mode="expert_only",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, dynaexq_fp8_weights),
            fp8_weights=dynaexq_fp8_weights,
            promoted_experts=dynaexq_promoted,
        ),
    ]


def materialize_expert_weights(
    plan: EvaluationPlan,
    layer_idx: int,
    expert_idx: int,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if plan.mode == "bf16":
        return gate_up_weight, down_weight, False
    if plan.mode == "uniform_fp4":
        return quantize_linear_weight(gate_up_weight, "fp4"), quantize_linear_weight(down_weight, "fp4"), True
    if plan.mode == "uniform_fp8":
        return quantize_linear_weight(gate_up_weight, "fp8"), quantize_linear_weight(down_weight, "fp8"), True
    if plan.mode == "expert_only":
        layer_promoted = set() if plan.promoted_experts is None else (plan.promoted_experts.get(layer_idx) or set())
        precision = "fp8" if expert_idx in layer_promoted else "fp4"
        return quantize_linear_weight(gate_up_weight, precision), quantize_linear_weight(down_weight, precision), True
    if plan.mode == "projection_block":
        w1_mode = "fp8" if plan.w1_projection_fp8 is not None and bool(plan.w1_projection_fp8[layer_idx][expert_idx]) else "fp4"
        w2_mode = "fp8" if plan.w2_projection_fp8 is not None and bool(plan.w2_projection_fp8[layer_idx][expert_idx]) else "fp4"
        return quantize_linear_weight(gate_up_weight, w1_mode), quantize_linear_weight(down_weight, w2_mode), True
    if plan.mode == "mixed_channel":
        pair_mask = None if plan.w1_pair_masks is None else plan.w1_pair_masks[layer_idx][expert_idx]
        channel_mask = None if plan.w2_channel_masks is None else plan.w2_channel_masks[layer_idx][expert_idx]
        quant_gate_up = quantize_linear_weight(gate_up_weight, "fp4") if pair_mask is None else quantize_gate_up_with_pair_mask(gate_up_weight, pair_mask.to(device=gate_up_weight.device))
        quant_down = quantize_linear_weight(down_weight, "fp4") if channel_mask is None else quantize_down_proj_with_mask(down_weight, channel_mask.to(device=down_weight.device))
        return quant_gate_up, quant_down, True
    raise ValueError(f"Unsupported plan mode: {plan.mode}")


def moe_forward_real_plan(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    plan: EvaluationPlan,
    layer_idx: int,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        quant_gate_up, quant_down, use_fp8_matmul = materialize_expert_weights(plan, layer_idx, expert_idx, gate_up_proj[expert_idx], down_proj[expert_idx])
        gate_up = apply_linear(current_state, quant_gate_up, use_fp8_matmul)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = apply_linear(F.silu(gate) * up, quant_down, use_fp8_matmul)
        final_hidden_states.index_add_(0, token_idx, (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden).to(hidden_states.dtype))

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_plan(
    plan: EvaluationPlan,
    eval_ids: torch.Tensor,
    text_config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    root_tensors = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_weight = move_tensor(root_tensors[embed_key], device, dtype)
    final_norm = move_tensor(root_tensors[norm_key], device, dtype)
    lm_head = move_tensor(root_tensors[lm_head_key], device, dtype)
    del root_tensors

    with torch.inference_mode():
        hidden_states = F.embedding(eval_ids.to(device), embed_weight)
        seq_len = int(eval_ids.shape[1])
        causal_mask = build_causal_mask(seq_len, device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        rotary = Qwen3NextRotaryEmbedding(config=text_config, device=device)
        position_embeddings = rotary(hidden_states, position_ids)
        for layer_idx in range(text_config.num_hidden_layers):
            layer_type = text_config.layer_types[layer_idx]
            raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
            tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
            del raw_tensors

            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], text_config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                hidden_states = full_attention_forward(hidden_norm, attn_tensors, text_config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                hidden_states = linear_attention_forward(hidden_norm, attn_tensors, text_config)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], text_config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = moe_forward_real_plan(hidden_norm, moe_tensors, text_config, plan, layer_idx)
            hidden_states = residual + moe_out
            release_tensors(tensors)
            if (layer_idx + 1) % 10 == 0 or layer_idx == text_config.num_hidden_layers - 1:
                print(f"  [{plan.name}] layer {layer_idx + 1}/{text_config.num_hidden_layers}", flush=True)

        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, text_config.rms_norm_eps)
        logits = F.linear(hidden_states.float(), lm_head.float())
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            eval_ids[:, 1:].to(device).view(-1),
        )
        ppl = math.exp(loss.item())

    release_tensors({"embed_weight": embed_weight, "final_norm": final_norm, "lm_head": lm_head})
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl, loss.item()


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: float(item[1]["ppl"]))
    print("\n" + "=" * 88, flush=True)
    print("All 11 baselines (real FP8 matmul, 512 WikiText-2 test tokens)", flush=True)
    print("=" * 88, flush=True)
    print(f"{'Baseline':<28} {'PPL':>10} {'NLL':>10} {'Memory GB':>12} {'Time s':>10}", flush=True)
    print("-" * 88, flush=True)
    for name, row in ordered:
        print(
            f"{name:<28} {float(row['ppl']):>10.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>12.3f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    calib_ids, calib_info = load_prefix_dataset_tokens(tokenizer, "train", args.calibration_tokens)
    eval_ids, eval_info = load_prefix_dataset_tokens(tokenizer, "test", args.eval_tokens)
    if int(calib_ids.shape[1]) % int(args.mini_batch_tokens) != 0:
        raise ValueError("calibration_tokens must be divisible by mini_batch_tokens for DynaExq EMA simulation")

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    captures = run_calibration_pass(store, text_config, calib_ids, args.mini_batch_tokens, device, dtype)
    metric_cache = compute_gate_aware_projection_metrics(store, text_config, captures, device, dtype)
    mxmoe_w1_deltas, mxmoe_w2_deltas = compute_mxmoe_projection_deltas(store, text_config, captures, device, dtype)
    mc_moe_scores = compute_mc_moe_importance(store, text_config, captures, device, dtype)
    dynamo_promoted = build_dynamo_promotions(captures, text_config, TARGET_FP8_FRACTION)
    dynaexq_promoted, dynaexq_scores = build_dynaexq_promotions(captures, text_config, TARGET_FP8_FRACTION)
    del dynaexq_scores

    plans = build_plans(
        text_config,
        total_bf16_bytes,
        non_expert_bytes,
        total_expert_elems,
        metric_cache,
        captures,
        mxmoe_w1_deltas,
        mxmoe_w2_deltas,
        mc_moe_scores,
        dynamo_promoted,
        dynaexq_promoted,
    )

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "calibration": calib_info,
            "evaluation": eval_info,
            "calibration_tokens": int(calib_ids.shape[1]),
            "eval_tokens": int(eval_ids.shape[1]),
            "mini_batch_tokens": int(args.mini_batch_tokens),
            "target_fp8_fraction": TARGET_FP8_FRACTION,
            "quantization": "real FP8 matmul via torch._scaled_mm; FP4 weights dequantized from NVFP4 then executed through FP8 matmul",
            "notes": {
                "mxmoe": "Actual per-projection MoE output perturbation over active experts only.",
                "mc_moe": "Importance = actnum^1 * weight_norm^1.5 * quant_loss^2.",
                "dynamo": "2-means significance clustering converted to budgeted high-cluster membership.",
                "dynaexq": "EMA alpha=0.9 over 4x32-token mini-batches with 10% mean-score hysteresis.",
            },
        },
        "results": {},
    }

    for idx, plan in enumerate(plans, start=1):
        print(f"\n=== Eval {idx}/{len(plans)}: {plan.name} ===", flush=True)
        t0 = time.time()
        ppl, nll = evaluate_plan(plan, eval_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - t0
        payload["results"][plan.name] = {
            "description": plan.description,
            "ppl": round(ppl, 6),
            "nll": round(nll, 6),
            "memory_gb": plan.memory_gb,
            "time_s": round(elapsed, 1),
            "fp8_weights": int(plan.fp8_weights),
        }
        payload["metadata"]["runtime_seconds"] = round(time.time() - start_time, 3)
        atomic_json_dump(args.output_json, payload)
        print(f"  -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | time={elapsed:.1f}s", flush=True)

    print_results_table(payload["results"])
    print(f"\nSaved results -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

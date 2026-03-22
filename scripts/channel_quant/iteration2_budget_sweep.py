#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

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

from new_spike1_channel_metrics import (
    EPS,
    allocate_weighted_counts,
    quantize_down_proj_with_mask,
    quantize_linear_weight,
    resolve_promoted_experts,
    resolve_terminal_keys,
    run_calibration_forward,
)
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
    move_tensor,
    quantize_to_fp8,
    quantize_to_nvfp4_columns,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "iteration2_results.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_CHUNK_TOKENS = 1024
DEFAULT_LOGIT_CHUNK = 32
SECTION_MARKER = "## [8] Iteration 2 - Projection/Budget/Hierarchy Sweep"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS)
    parser.add_argument(
        "--eval-chunk-tokens",
        type=int,
        default=DEFAULT_EVAL_CHUNK_TOKENS,
        help="Tokens scored per WikiText-2 block; each block carries one token of left context.",
    )
    parser.add_argument("--logit-chunk-size", type=int, default=DEFAULT_LOGIT_CHUNK)
    parser.add_argument("--max-eval-tokens", type=int, default=None, help="Optional cap for debugging only.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_full_dataset_tokens(
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    max_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset for split {split}, got {type(dataset)!r}")
    pieces: list[str] = []
    for raw_item in dataset:
        item = raw_item if isinstance(raw_item, dict) else dict(raw_item)
        text = str(item.get("text", "")).strip()
        if text:
            pieces.append(text)
    encoded = tokenizer("\n\n".join(pieces), return_tensors="pt", add_special_tokens=False)
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    return input_ids, {
        "split": split,
        "source_dataset": "wikitext/wikitext-2-raw-v1",
        "tokenizer": tokenizer.name_or_path,
        "actual_tokens": int(input_ids.shape[1]),
        "max_eval_tokens": max_tokens,
        "preview": tokenizer.decode(input_ids[0, : min(128, input_ids.shape[1])], skip_special_tokens=False),
    }


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
                "max_eval_tokens": token_count,
                "preview": tokenizer.decode(clipped[0, : min(128, clipped.shape[1])], skip_special_tokens=False),
            }
    raise RuntimeError(f"Unable to collect {token_count} tokens from WikiText-2 {split}")


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


def quantize_gate_up_with_pair_mask(weight: torch.Tensor, fp8_pair_mask: torch.Tensor) -> torch.Tensor:
    if fp8_pair_mask.ndim != 1:
        raise ValueError(f"Expected 1D pair mask, got shape={tuple(fp8_pair_mask.shape)}")
    if int(fp8_pair_mask.numel() * 2) != int(weight.shape[0]):
        raise ValueError(
            f"Expected pair mask length {weight.shape[0] // 2}, got {fp8_pair_mask.numel()} for weight {tuple(weight.shape)}"
        )
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


def resolve_non_expert_bytes(total_bf16_bytes: int, config: Any) -> tuple[int, int]:
    per_layer_expert_elems = config.num_experts * (
        (2 * config.moe_intermediate_size * config.hidden_size)
        + (config.hidden_size * config.moe_intermediate_size)
    )
    total_expert_elems = int(config.num_hidden_layers * per_layer_expert_elems)
    total_expert_bf16_bytes = total_expert_elems * 2
    non_expert_bytes = total_bf16_bytes - total_expert_bf16_bytes
    return non_expert_bytes, total_expert_elems


def estimate_memory_gb(non_expert_bytes: int, total_expert_elems: int, fp8_weights: int) -> float:
    expert_bytes = int((0.5 * total_expert_elems) + (0.5 * fp8_weights))
    return round(float(non_expert_bytes + expert_bytes) / 1e9, 3)


def fp8_weights_from_masks(config: Any, w1_pair_masks: dict[int, dict[int, torch.Tensor]], w2_channel_masks: dict[int, dict[int, torch.Tensor]]) -> int:
    w1_pairs = sum(int(mask.sum().item()) for layer_masks in w1_pair_masks.values() for mask in layer_masks.values())
    w2_channels = sum(int(mask.sum().item()) for layer_masks in w2_channel_masks.values() for mask in layer_masks.values())
    return int((w1_pairs * 2 * config.hidden_size) + (w2_channels * config.moe_intermediate_size))


def fp8_weights_from_promoted_experts(config: Any, promoted_experts: dict[int, set[int]]) -> int:
    full_expert_weights = (
        (2 * config.moe_intermediate_size * config.hidden_size)
        + (config.hidden_size * config.moe_intermediate_size)
    )
    promoted_count = sum(len(experts) for experts in promoted_experts.values())
    return int(promoted_count * full_expert_weights)


def compute_gate_aware_projection_metrics(
    store: WeightStore,
    config: Any,
    captures: dict[int, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, LayerMetricBundle]:
    metric_cache: dict[int, LayerMetricBundle] = {}
    for layer_idx in range(config.num_hidden_layers):
        capture = captures[layer_idx]
        print(f"[metrics] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
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

            del x, route_weights, gate_up_fp8, gate_up_fp4, gate_up_out_fp8, gate_up_out_fp4
            del gate_fp8, up_fp8, gate_fp4, up_fp4, intermediate_fp8, intermediate_fp4
            del delta_hidden_abs, gate_weighted_abs, down_fp4, down_delta_abs, down_col_norm

        del gate_up_proj, down_proj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metric_cache[layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=w1_pair_scores,
            w2_channel_scores=w2_channel_scores,
        )
    return metric_cache


def build_uniform_masks(metric_cache: dict[int, LayerMetricBundle], config: Any, w1_fraction: float, w2_fraction: float) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    k_w1 = int(round(w1_fraction * config.moe_intermediate_size))
    k_w2 = int(round(w2_fraction * config.hidden_size))
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


def build_freq_weighted_masks(metric_cache: dict[int, LayerMetricBundle], config: Any, w1_fraction: float, w2_fraction: float) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        weights = [float(value) for value in bundle.routing_counts.tolist()]
        total_w1 = int(round(w1_fraction * config.moe_intermediate_size * config.num_experts))
        total_w2 = int(round(w2_fraction * config.hidden_size * config.num_experts))
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


def build_tiered_masks(metric_cache: dict[int, LayerMetricBundle], config: Any) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    hot_cutoff = math.ceil(config.num_experts / 3)
    medium_cutoff = math.ceil((2 * config.num_experts) / 3)
    for layer_idx, bundle in metric_cache.items():
        ranked = sorted(range(config.num_experts), key=lambda expert_idx: (-int(bundle.routing_counts[expert_idx]), expert_idx))
        hot = set(ranked[:hot_cutoff])
        medium = set(ranked[hot_cutoff:medium_cutoff])
        layer_w1: dict[int, torch.Tensor] = {}
        layer_w2: dict[int, torch.Tensor] = {}
        for expert_idx in range(config.num_experts):
            fraction = 0.0
            if expert_idx in hot:
                fraction = 0.5
            elif expert_idx in medium:
                fraction = 0.25
            layer_w1[expert_idx] = topk_mask_from_scores(
                bundle.w1_pair_scores[expert_idx],
                int(round(fraction * config.moe_intermediate_size)),
            )
            layer_w2[expert_idx] = topk_mask_from_scores(
                bundle.w2_channel_scores[expert_idx],
                int(round(fraction * config.hidden_size)),
            )
        w1_pair_masks[layer_idx] = layer_w1
        w2_channel_masks[layer_idx] = layer_w2
    return w1_pair_masks, w2_channel_masks


def build_projection_plan(
    name: str,
    description: str,
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    allocation: str,
    w1_fraction: float,
    w2_fraction: float,
) -> EvaluationPlan:
    if allocation == "uniform":
        w1_pair_masks, w2_channel_masks = build_uniform_masks(metric_cache, config, w1_fraction, w2_fraction)
    elif allocation == "freq_weighted":
        w1_pair_masks, w2_channel_masks = build_freq_weighted_masks(metric_cache, config, w1_fraction, w2_fraction)
    elif allocation == "tiered":
        w1_pair_masks, w2_channel_masks = build_tiered_masks(metric_cache, config)
    else:
        raise ValueError(f"Unsupported allocation={allocation}")
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvaluationPlan(
        name=name,
        description=description,
        mode="mixed_channel",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def build_expert_only_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    fp8_fraction: float,
) -> EvaluationPlan:
    layer_counts = {layer_idx: bundle.routing_counts for layer_idx, bundle in metric_cache.items()}
    promoted_experts = resolve_promoted_experts(layer_counts, fp8_fraction)
    fp8_weights = fp8_weights_from_promoted_experts(config, promoted_experts)
    return EvaluationPlan(
        name="expert_only",
        description="Top 25% routed experts fully FP8, rest fully FP4.",
        mode="expert_only",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        promoted_experts=promoted_experts,
    )


def moe_forward_with_plan(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    plan: EvaluationPlan,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    layer_promoted = plan.promoted_experts.get(layer_idx) if plan.promoted_experts is not None else None
    layer_w1_masks = plan.w1_pair_masks.get(layer_idx) if plan.w1_pair_masks is not None else None
    layer_w2_masks = plan.w2_channel_masks.get(layer_idx) if plan.w2_channel_masks is not None else None

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]

        if plan.mode == "expert_only":
            expert_mode = "fp8" if layer_promoted is not None and expert_idx in layer_promoted else "fp4"
            quant_gate_up = quantize_linear_weight(gate_up_weight, expert_mode)
            quant_down = quantize_linear_weight(down_weight, expert_mode)
        elif plan.mode == "mixed_channel":
            pair_mask = layer_w1_masks[expert_idx] if layer_w1_masks is not None else None
            channel_mask = layer_w2_masks[expert_idx] if layer_w2_masks is not None else None
            if pair_mask is None:
                quant_gate_up = quantize_linear_weight(gate_up_weight, "fp4")
            else:
                quant_gate_up = quantize_gate_up_with_pair_mask(gate_up_weight, pair_mask.to(device=gate_up_weight.device))
            if channel_mask is None:
                quant_down = quantize_linear_weight(down_weight, "fp4")
            elif not bool(channel_mask.any()):
                quant_down = quantize_linear_weight(down_weight, "fp4")
            elif bool(channel_mask.all()):
                quant_down = quantize_linear_weight(down_weight, "fp8")
            else:
                quant_down = quantize_down_proj_with_mask(down_weight, channel_mask.to(device=down_weight.device))
        else:
            raise ValueError(f"Unsupported plan mode={plan.mode}")

        gate_up = F.linear(current_state, quant_gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, quant_down)
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = F.linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), expert_counts


def evaluate_chunked_perplexity(
    store: WeightStore,
    config: Any,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logit_chunk_size: int,
    eval_chunk_tokens: int,
    plan: EvaluationPlan,
) -> tuple[float, float]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(store.weight_map)
    root_tensors = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_weight = move_tensor(root_tensors[embed_key], device, dtype)
    final_norm = move_tensor(root_tensors[norm_key], device, dtype)
    lm_head = move_tensor(root_tensors[lm_head_key], device, dtype)
    del root_tensors

    total_token_count = int(input_ids.shape[1])
    chunk_starts = list(range(0, total_token_count - 1, eval_chunk_tokens))
    total_nll = 0.0
    total_scored_tokens = 0

    with torch.inference_mode():
        for chunk_idx, begin in enumerate(chunk_starts, start=1):
            end = min(begin + eval_chunk_tokens + 1, total_token_count)
            if end - begin < 2:
                continue
            chunk_ids = input_ids[:, begin:end]
            print(
                f"[eval:{plan.name}] chunk {chunk_idx:03d}/{len(chunk_starts):03d} tokens={begin}:{end} mode={plan.mode}",
                flush=True,
            )
            hidden_states = F.embedding(chunk_ids.to(device), embed_weight)
            seq_len = int(chunk_ids.shape[1])
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            causal_mask = build_causal_mask(seq_len, device)
            rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
            position_embeddings = rotary(hidden_states, position_ids)

            for layer_idx in range(config.num_hidden_layers):
                layer_type = config.layer_types[layer_idx]
                print(
                    f"[eval:{plan.name}] chunk {chunk_idx:03d}/{len(chunk_starts):03d} layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d} ({layer_type})",
                    flush=True,
                )
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
                mlp_out, _ = moe_forward_with_plan(
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
                    plan,
                    layer_idx,
                )
                hidden_states = hidden_states + mlp_out
                release_tensors(tensors)

            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
            targets = chunk_ids[:, 1:].to(device)
            for start in range(0, seq_len - 1, logit_chunk_size):
                end = min(seq_len - 1, start + logit_chunk_size)
                logits = F.linear(hidden_states[:, start:end, :], lm_head)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets[:, start:end].reshape(-1),
                    reduction="sum",
                )
                total_nll += float(loss_sum.item())
                total_scored_tokens += int(targets[:, start:end].numel())
                del logits

            del chunk_ids, hidden_states, targets, position_ids, causal_mask, rotary, position_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    release_tensors({"embed_weight": embed_weight, "final_norm": final_norm, "lm_head": lm_head})
    avg_nll = total_nll / max(total_scored_tokens, 1)
    return avg_nll, float(math.exp(avg_nll))


def empty_payload(model_id: str, calibration: dict[str, Any], evaluation: dict[str, Any], eval_chunk_tokens: int) -> dict[str, Any]:
    return {
        "metadata": {
            "model": model_id,
            "calibration": calibration,
            "evaluation": evaluation,
            "eval_chunk_tokens": eval_chunk_tokens,
            "metric": "gate_aware_act_qerror",
            "note": "W1 uses pair-wise gate_up_proj assignment because SwiGLU gate/up rows are coupled.",
        },
        "per_projection": {},
        "budget_sweep": [],
        "hierarchy_comparison": {},
    }


def upsert_results(payload: dict[str, Any], category: str, key: str | int, value: dict[str, Any]) -> None:
    if category == "budget_sweep":
        bucket = payload[category]
        if not isinstance(bucket, list):
            raise TypeError("budget_sweep bucket must be a list")
        for idx, row in enumerate(bucket):
            if int(row["fp8_pct"]) == int(key):
                bucket[idx] = value
                return
        bucket.append(value)
        bucket.sort(key=lambda row: int(row["fp8_pct"]))
        return
    payload[category][str(key)] = value


def render_exploration_section(payload: dict[str, Any]) -> str:
    proj = payload["per_projection"]
    sweep = payload["budget_sweep"]
    hier = payload["hierarchy_comparison"]
    best_projection_key = min(proj, key=lambda name: proj[name]["ppl"]) if proj else None
    best_sweep = min(sweep, key=lambda row: row["ppl"]) if sweep else None
    best_hierarchy_key = min(hier, key=lambda name: hier[name]["ppl"]) if hier else None
    lines = [SECTION_MARKER, ""]
    lines.append(
        "**Approach**: Reused the streamed Spike 5 full-model forward together with Spike 1 quantizers and the new Spike 1 gate-aware activation metric, then ran three follow-up experiments: projection-specific FP8 splits, a 0-100% routing-aware budget sweep, and a hierarchy comparison between expert-only, channel-only, and two-level allocation. Calibration still uses a single 128-token WikiText-2 train pass and evaluation streams the full WikiText-2 test split in fixed-size blocks."
    )
    lines.append("")
    if best_projection_key is not None:
        best_projection = proj[best_projection_key]
        lines.append(
            f"**Per-projection**: The best split is `{best_projection_key}` at PPL {best_projection['ppl']:.4f} and approx {best_projection['memory_gb']:.3f} GB. This isolates whether W2 deserves more of the mixed-precision budget than W1 when channels are ranked by the gate-aware metric."
        )
        lines.append("")
        for name in ("uniform_25_25", "w2_heavy_50_0", "w2_heavy_40_10", "w1_heavy_0_50"):
            if name in proj:
                lines.append(
                    f"- `{name}`: PPL {proj[name]['ppl']:.4f}, memory {proj[name]['memory_gb']:.3f} GB."
                )
        lines.append("")
    if best_sweep is not None:
        lines.append(
            f"**Budget sweep**: The best routing-aware per-channel point is {best_sweep['fp8_pct']}% FP8 at PPL {best_sweep['ppl']:.4f} and {best_sweep['memory_gb']:.3f} GB. The sweep traces the Pareto curve from all-FP4 through all-FP8 under the same per-projection policy for both W1 and W2."
        )
        lines.append("")
    if best_hierarchy_key is not None:
        best_hierarchy = hier[best_hierarchy_key]
        lines.append(
            f"**Hierarchy comparison**: The best hierarchy variant is `{best_hierarchy_key}` at PPL {best_hierarchy['ppl']:.4f} and {best_hierarchy['memory_gb']:.3f} GB. Comparing this against `expert_only`, `channel_only`, and `aggressive_two_level` shows how much of the gain comes from routing-aware expert budgeting versus channel ranking alone."
        )
        lines.append("")
        for name in ("expert_only", "channel_only", "two_level", "aggressive_two_level"):
            if name in hier:
                lines.append(
                    f"- `{name}`: PPL {hier[name]['ppl']:.4f}, memory {hier[name]['memory_gb']:.3f} GB."
                )
        lines.append("")
    lines.append(
        "**Verdict**: This iteration tests whether the remaining gap is mostly a projection split problem, a global budget problem, or a hierarchy problem. The resulting JSON records the exact perplexity/memory trade-off for each setting, and the per-projection rows also expose that nominal W1/W2 percentage splits are not perfectly iso-memory once the true tensor sizes are accounted for."
    )
    return "\n".join(lines)


def upsert_exploration_section(path: Path, section_body: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SECTION_MARKER in existing:
        prefix = existing.split(SECTION_MARKER, maxsplit=1)[0].rstrip()
        updated = f"{prefix}\n\n{section_body}\n"
    else:
        updated = existing.rstrip()
        if updated:
            updated += "\n\n---\n\n"
        updated += f"{section_body}\n"
    path.write_text(updated, encoding="utf-8")


def maybe_get_saved_result(payload: dict[str, Any], category: str, key: str | int) -> dict[str, Any] | None:
    if category == "budget_sweep":
        for row in payload.get("budget_sweep", []):
            if int(row["fp8_pct"]) == int(key):
                return row
        return None
    return payload.get(category, {}).get(str(key))


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    calibration_ids, calibration_info = load_prefix_dataset_tokens(tokenizer, "train", args.calibration_tokens)
    eval_ids, eval_info = load_full_dataset_tokens(tokenizer, "test", args.max_eval_tokens)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    total_bf16_bytes = int(root_config.get("total_size", 0) or load_json(snapshot_dir / "model.safetensors.index.json")["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, config)

    print("[calibration] collecting routed activations", flush=True)
    captures = run_calibration_forward(store, config, calibration_ids, device, dtype)
    print("[metrics] computing W1/W2 gate-aware scores", flush=True)
    metric_cache = compute_gate_aware_projection_metrics(store, config, captures, device, dtype)

    payload = empty_payload(args.model_id, calibration_info, eval_info, args.eval_chunk_tokens)
    if args.output_json.exists():
        try:
            payload = load_json(args.output_json)
            print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing file at {args.output_json}", flush=True)

    evaluation_queue: list[tuple[str, str | int, EvaluationPlan]] = []

    per_projection_specs = [
        ("uniform_25_25", "Freq-weighted per-channel, 25% FP8 in W1 and 25% in W2.", "freq_weighted", 0.25, 0.25),
        ("w2_heavy_50_0", "Freq-weighted per-channel, 0% FP8 in W1 and 50% in W2.", "freq_weighted", 0.0, 0.50),
        ("w2_heavy_40_10", "Freq-weighted per-channel, 10% FP8 in W1 and 40% in W2.", "freq_weighted", 0.10, 0.40),
        ("w1_heavy_0_50", "Freq-weighted per-channel, 50% FP8 in W1 and 0% in W2.", "freq_weighted", 0.50, 0.0),
    ]
    projection_plans: dict[str, EvaluationPlan] = {}
    for name, description, allocation, w1_fraction, w2_fraction in per_projection_specs:
        projection_plans[name] = build_projection_plan(
            name,
            description,
            metric_cache,
            config,
            non_expert_bytes,
            total_expert_elems,
            allocation,
            w1_fraction,
            w2_fraction,
        )
        evaluation_queue.append(("per_projection", name, projection_plans[name]))

    budget_plans: dict[int, EvaluationPlan] = {}
    for fp8_pct in range(0, 101, 10):
        fraction = fp8_pct / 100.0
        budget_plans[fp8_pct] = build_projection_plan(
            f"budget_{fp8_pct:03d}",
            f"Freq-weighted per-channel, {fp8_pct}% FP8 for both W1 and W2.",
            metric_cache,
            config,
            non_expert_bytes,
            total_expert_elems,
            "freq_weighted",
            fraction,
            fraction,
        )
        evaluation_queue.append(("budget_sweep", fp8_pct, budget_plans[fp8_pct]))

    hierarchy_plans: dict[str, EvaluationPlan] = {
        "expert_only": build_expert_only_plan(metric_cache, config, non_expert_bytes, total_expert_elems, 0.25),
        "channel_only": build_projection_plan(
            "channel_only",
            "Uniform 25% FP8 per expert for both W1 and W2, channel-ranked only.",
            metric_cache,
            config,
            non_expert_bytes,
            total_expert_elems,
            "uniform",
            0.25,
            0.25,
        ),
        "two_level": projection_plans["uniform_25_25"],
        "aggressive_two_level": build_projection_plan(
            "aggressive_two_level",
            "Top third experts get 50% FP8, middle third 25%, bottom third 0%.",
            metric_cache,
            config,
            non_expert_bytes,
            total_expert_elems,
            "tiered",
            0.25,
            0.25,
        ),
    }
    for name, plan in hierarchy_plans.items():
        evaluation_queue.append(("hierarchy_comparison", name, plan))

    shared_results: dict[str, dict[str, Any]] = {}
    for category, key, plan in evaluation_queue:
        existing = maybe_get_saved_result(payload, category, key)
        if existing is not None:
            shared_results.setdefault(plan.name, existing)

    executed: set[tuple[str, str | int]] = set()
    for category, key, plan in evaluation_queue:
        if (category, key) in executed:
            continue
        executed.add((category, key))
        existing = maybe_get_saved_result(payload, category, key)
        if existing is not None:
            print(f"[skip] {category}:{key} already present", flush=True)
            continue
        if plan.name in shared_results:
            print(f"[reuse] {category}:{key} <- {plan.name}", flush=True)
            upsert_results(payload, category, key, shared_results[plan.name])
            atomic_json_dump(args.output_json, payload)
            continue
        print(
            f"\n=== Running {category}:{key} | {plan.description} | memory~{plan.memory_gb:.3f} GB ===",
            flush=True,
        )
        avg_nll, ppl = evaluate_chunked_perplexity(
            store,
            config,
            eval_ids,
            device,
            dtype,
            args.logit_chunk_size,
            args.eval_chunk_tokens,
            plan,
        )
        if category == "budget_sweep":
            result = {
                "fp8_pct": int(key),
                "avg_nll": avg_nll,
                "ppl": ppl,
                "memory_gb": plan.memory_gb,
            }
        else:
            result = {
                "avg_nll": avg_nll,
                "ppl": ppl,
                "memory_gb": plan.memory_gb,
            }
        shared_results[plan.name] = result
        upsert_results(payload, category, key, result)
        payload.setdefault("metadata", {})["runtime_seconds"] = round(time.time() - start_time, 3)
        atomic_json_dump(args.output_json, payload)
        print(
            f"[done] {category}:{key} -> ppl={ppl:.4f} memory={plan.memory_gb:.3f} GB", flush=True
        )

    section = render_exploration_section(payload)
    upsert_exploration_section(args.exploration_md, section)
    payload.setdefault("metadata", {})["runtime_seconds"] = round(time.time() - start_time, 3)
    atomic_json_dump(args.output_json, payload)
    print(f"Saved results -> {args.output_json}", flush=True)
    print(f"Updated exploration log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

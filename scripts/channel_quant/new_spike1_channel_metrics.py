#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, PreTrainedTokenizerBase
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
    move_tensor,
    quantize_to_fp8,
    quantize_to_nvfp4_columns,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

matplotlib.use("Agg")

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_METRICS_JSON = RESULTS_DIR / "new_spike1_metrics.json"
DEFAULT_PPL_JSON = RESULTS_DIR / "new_spike1_perplexity.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_SPIKE1_JSON = RESULTS_DIR / "spike1_sensitivity.json"
DEFAULT_SPIKE2_JSON = RESULTS_DIR / "spike2_metrics.json"
DEFAULT_SPIKE5_JSON = RESULTS_DIR / "spike5_perplexity.json"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 256
DEFAULT_LOGIT_CHUNK = 32
DEFAULT_CHANNEL_BUDGET_EQUIV = 0.75
NUM_EXPERTS = 256
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 512
TOPK = 8
EPS = 1e-10
NEW_SECTION_MARKER = "## [6] Better Per-Channel Sensitivity Metrics"
ROLE_ORDER = ("cold", "medium", "hot")
TARGET_EXPERT_SPECS = (
    (5, 2, "cold"),
    (5, 87, "medium"),
    (5, 242, "hot"),
    (20, 1, "cold"),
    (20, 104, "medium"),
    (20, 147, "hot"),
    (35, 0, "cold"),
    (35, 145, "medium"),
    (35, 19, "hot"),
)
EXISTING_METRICS = (
    "weight_l2",
    "weight_l1",
    "weight_kurtosis",
    "weight_max_abs",
    "weight_variance",
    "within_group_var",
    "sinq_column_scale",
    "awq_activation",
    "smoothquant_max",
    "owq_hessian_diag",
    "slim_llm_salience",
    "owq_sensitivity",
)
NEW_METRICS = (
    "act_weighted_qerror",
    "scalebits_qerror",
    "gate_aware_act_qerror",
    "input_variance_qerror",
)
ASSIGNMENT_BUDGETS = (10, 25, 50, 75)


@dataclass(frozen=True)
class TargetExpert:
    layer_idx: int
    expert_id: int
    role: str

    @property
    def label(self) -> str:
        role_suffix = {"cold": "cold", "medium": "med", "hot": "hot"}.get(self.role, self.role)
        return f"L{self.layer_idx}_E{self.expert_id}_{role_suffix}"


@dataclass
class LayerCapture:
    layer_idx: int
    layer_type: str
    mlp_input: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    expert_counts: torch.Tensor


@dataclass
class AssignmentError:
    rel_l2: float
    mse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--logit-chunk-size", type=int, default=DEFAULT_LOGIT_CHUNK)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--perplexity-json", type=Path, default=DEFAULT_PPL_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--spike1-json", type=Path, default=DEFAULT_SPIKE1_JSON)
    parser.add_argument("--spike2-json", type=Path, default=DEFAULT_SPIKE2_JSON)
    parser.add_argument("--spike5-json", type=Path, default=DEFAULT_SPIKE5_JSON)
    parser.add_argument(
        "--channel-budget-equivalent-fraction",
        type=float,
        default=DEFAULT_CHANNEL_BUDGET_EQUIV,
        help=(
            "Average FP8 fraction for W2 output channels during full-model evaluation. "
            "0.75 matches the extra FP8 bytes of spike5 mixed_25pct when W1 stays FP4."
        ),
    )
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


def load_dataset_tokens(
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    token_count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset for split {split}, got {type(dataset)!r}")
    pieces: list[str] = []
    for item in dataset:
        text = str(item["text"]).strip()
        if not text:
            continue
        pieces.append(text)
        encoded = tokenizer("\n\n".join(pieces), return_tensors="pt", add_special_tokens=False)
        encoded_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        if int(encoded_ids.shape[1]) >= token_count:
            input_ids = encoded_ids[:, :token_count]
            return input_ids, {
                "split": split,
                "source_dataset": "wikitext/wikitext-2-raw-v1",
                "tokenizer": tokenizer.name_or_path,
                "requested_tokens": token_count,
                "actual_tokens": int(input_ids.shape[1]),
                "preview": tokenizer.decode(input_ids[0, : min(128, input_ids.shape[1])], skip_special_tokens=False),
            }
    raise RuntimeError(f"Unable to collect {token_count} tokens from WikiText-2 {split}")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    if left.size == 0:
        return 0.0
    if spearmanr is not None:
        value = spearmanr(left, right)[0]
        if value is None or np.isnan(value):
            return 0.0
        return float(value)
    left_rank = left.argsort().argsort().astype(np.float64)
    right_rank = right.argsort().argsort().astype(np.float64)
    if left_rank.std() <= 0.0 or right_rank.std() <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if sorted_values.size == 0:
        return 0.0
    total = sorted_values.sum()
    if total <= 0.0:
        return 0.0
    index = np.arange(1, sorted_values.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (sorted_values.size * total)) - (sorted_values.size + 1.0) / sorted_values.size)


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    channel_count = int(scores.shape[0])
    if k <= 0:
        return np.zeros(channel_count, dtype=bool)
    if k >= channel_count:
        return np.ones(channel_count, dtype=bool)
    indices = np.argpartition(scores, channel_count - k)[channel_count - k :]
    mask = np.zeros(channel_count, dtype=bool)
    mask[indices] = True
    return mask


def allocate_weighted_counts(total: int, capacities: list[int], weights: list[float]) -> list[int]:
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


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().to(torch.float32).tolist()


def format_metric_table(rows: list[tuple[str, float, float, float]]) -> str:
    header = f"{'metric':<28} {'mean_rho':>9} {'mean_gini':>10} {'hot_rho':>9}"
    body = [header, "-" * len(header)]
    for metric_name, mean_rho, mean_gini, hot_rho in rows:
        body.append(f"{metric_name:<28} {mean_rho:>9.4f} {mean_gini:>10.4f} {hot_rho:>9.4f}")
    return "\n".join(body)


def quantize_linear_weight(weight: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bf16":
        return weight
    weight_t = weight.transpose(0, 1).contiguous()
    if mode == "fp8":
        quantized = quantize_to_fp8(weight_t)
    elif mode == "fp4":
        quantized = quantize_to_nvfp4_columns(weight_t)
    else:
        raise ValueError(f"Unsupported quant mode: {mode}")
    return quantized.transpose(0, 1).contiguous()


def quantize_down_proj_with_mask(weight: torch.Tensor, fp8_mask: torch.Tensor) -> torch.Tensor:
    weight_t = weight.transpose(0, 1).contiguous()
    fp4_t = quantize_to_nvfp4_columns(weight_t)
    fp8_t = quantize_to_fp8(weight_t)
    column_mask = fp8_mask.to(device=weight_t.device, dtype=torch.bool).view(1, -1)
    mixed_t = torch.where(column_mask, fp8_t, fp4_t)
    return mixed_t.transpose(0, 1).contiguous()


def resolve_terminal_keys(weight_map: dict[str, str]) -> tuple[str, str, str]:
    embed_key = next(key for key in weight_map if key.endswith("embed_tokens.weight") and ".layers." not in key)
    lm_head_key = next(key for key in weight_map if key.endswith("lm_head.weight"))
    norm_candidates = [
        key for key in weight_map
        if key.endswith("norm.weight")
        and ".layers." not in key
        and "q_norm" not in key
        and "k_norm" not in key
        and "linear_attn" not in key
    ]
    if not norm_candidates:
        raise KeyError("Unable to find final norm weight")
    return embed_key, norm_candidates[0], lm_head_key


def moe_forward_capture(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    num_experts_per_tok: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)

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
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), selected_experts, routing_weights, expert_counts


def moe_forward_quantized_mode(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    num_experts_per_tok: int,
    num_experts: int,
    mode: str,
    promoted_experts: set[int] | None = None,
    channel_masks: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]
        if mode == "fp4":
            gate_up_weight = quantize_linear_weight(gate_up_weight, "fp4")
            down_weight = quantize_linear_weight(down_weight, "fp4")
        elif mode == "fp8":
            gate_up_weight = quantize_linear_weight(gate_up_weight, "fp8")
            down_weight = quantize_linear_weight(down_weight, "fp8")
        elif mode == "mixed_expert":
            expert_mode = "fp8" if promoted_experts is not None and expert_idx in promoted_experts else "fp4"
            gate_up_weight = quantize_linear_weight(gate_up_weight, expert_mode)
            down_weight = quantize_linear_weight(down_weight, expert_mode)
        elif mode == "mixed_channel":
            gate_up_weight = quantize_linear_weight(gate_up_weight, "fp4")
            if channel_masks is None or expert_idx not in channel_masks:
                down_weight = quantize_linear_weight(down_weight, "fp4")
            else:
                down_weight = quantize_down_proj_with_mask(down_weight, channel_masks[expert_idx])
        elif mode != "bf16":
            raise ValueError(f"Unsupported MoE mode: {mode}")
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


def run_calibration_forward(
    store: WeightStore,
    config: Any,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, LayerCapture]:
    print(f"[calibration] streaming {config.num_hidden_layers} layers for activation/routing capture", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    embed_weight = move_tensor(store.load_tensors([embed_key])[embed_key], device, dtype)
    hidden_states = F.embedding(input_ids.to(device), embed_weight)
    del embed_weight
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seq_len = int(input_ids.shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = build_causal_mask(seq_len, device)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    position_embeddings = rotary(hidden_states, position_ids)

    captures: dict[int, LayerCapture] = {}
    for layer_idx in range(config.num_hidden_layers):
        layer_type = config.layer_types[layer_idx]
        print(f"[calibration] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d} ({layer_type})", flush=True)
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
        mlp_out, selected_experts, routing_weights, expert_counts = moe_forward_capture(
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
            config.num_experts_per_tok,
            config.num_experts,
        )
        captures[layer_idx] = LayerCapture(
            layer_idx=layer_idx,
            layer_type=layer_type,
            mlp_input=mlp_input.detach().cpu().squeeze(0).to(torch.float32),
            selected_experts=selected_experts.detach().cpu(),
            routing_weights=routing_weights.detach().cpu().to(torch.float32),
            expert_counts=expert_counts.detach().cpu(),
        )
        hidden_states = hidden_states + mlp_out
        release_tensors(tensors)
    return captures


def make_quantized_layer_forward(
    store: WeightStore,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    causal_mask: torch.Tensor,
    layer_idx: int,
    capture_mlp: bool,
):
    layer_type = config.layer_types[layer_idx]

    def _forward(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
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
        hidden_after_attn = residual + mixed
        mlp_input = rms_norm_qwen3_next(hidden_after_attn, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
        mlp_out, _ = moe_forward_quantized_mode(
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
            config.num_experts_per_tok,
            config.num_experts,
            mode="fp4",
        )
        hidden_out = hidden_after_attn + mlp_out
        release_tensors(tensors)
        if capture_mlp:
            return hidden_out, mlp_out
        return hidden_out

    return _forward


def run_quantized_backward_capture(
    store: WeightStore,
    config: Any,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logit_chunk_size: int,
) -> dict[int, torch.Tensor]:
    print("[gradient] running one draft-FP4 forward/backward with checkpointed layers", flush=True)
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(store.weight_map)
    root_tensors = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_weight = move_tensor(root_tensors[embed_key], device, dtype)
    final_norm = move_tensor(root_tensors[norm_key], device, dtype)
    lm_head = move_tensor(root_tensors[lm_head_key], device, dtype)
    del root_tensors

    hidden_states = F.embedding(input_ids.to(device), embed_weight)
    hidden_states = hidden_states.detach().requires_grad_(True)

    seq_len = int(input_ids.shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = build_causal_mask(seq_len, device)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    position_embeddings = rotary(hidden_states, position_ids)

    mlp_outputs: dict[int, torch.Tensor] = {}
    for layer_idx in range(config.num_hidden_layers):
        print(f"[gradient] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
        layer_fn = make_quantized_layer_forward(store, config, device, dtype, position_embeddings, causal_mask, layer_idx, True)
        result = checkpoint(layer_fn, hidden_states, use_reentrant=False)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(f"Expected (hidden_states, mlp_out) tuple from checkpoint for layer {layer_idx}")
        hidden_states, mlp_out = result
        mlp_out.retain_grad()
        mlp_outputs[layer_idx] = mlp_out

    hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
    targets = input_ids[:, 1:].to(device)
    total_nll = torch.zeros((), device=device, dtype=torch.float32)
    token_count = targets.numel()
    for start in range(0, seq_len - 1, logit_chunk_size):
        end = min(seq_len - 1, start + logit_chunk_size)
        logits = F.linear(hidden_states[:, start:end, :], lm_head)
        chunk_targets = targets[:, start:end]
        total_nll = total_nll + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            chunk_targets.reshape(-1),
            reduction="sum",
        )
        del logits
    loss = total_nll / max(token_count, 1)
    print(f"[gradient] loss={float(loss.item()):.6f}; backward()", flush=True)
    loss.backward()

    gradients: dict[int, torch.Tensor] = {}
    for layer_idx in range(config.num_hidden_layers):
        grad = mlp_outputs[layer_idx].grad
        if grad is None:
            raise RuntimeError(f"Missing retained gradient for layer {layer_idx}")
        gradients[layer_idx] = grad.detach().cpu().squeeze(0).to(torch.float32)

    del hidden_states, embed_weight, final_norm, lm_head, total_nll, loss
    mlp_outputs.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gradients


def target_lookup() -> dict[tuple[int, int], TargetExpert]:
    return {
        (layer_idx, expert_id): TargetExpert(layer_idx=layer_idx, expert_id=expert_id, role=role)
        for layer_idx, expert_id, role in TARGET_EXPERT_SPECS
    }


def routed_token_indices(capture: LayerCapture, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_idx, route_pos = torch.where(capture.selected_experts == expert_id)
    return token_idx, route_pos


def compute_new_metrics_for_all_layers(
    store: WeightStore,
    config: Any,
    captures: dict[int, LayerCapture],
    gradient_captures: dict[int, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[tuple[int, int], dict[str, Any]], dict[int, torch.Tensor]]:
    all_metrics: dict[str, dict[int, torch.Tensor]] = {metric: {} for metric in NEW_METRICS}
    target_cache: dict[tuple[int, int], dict[str, Any]] = {}
    routing_counts: dict[int, torch.Tensor] = {}
    targets = target_lookup()

    for layer_idx in range(config.num_hidden_layers):
        capture = captures[layer_idx]
        routing_counts[layer_idx] = capture.expert_counts.clone()
        print(f"[metrics] layer {layer_idx:02d}: computing intermediate stats + quant error", flush=True)
        mean_abs = torch.zeros((NUM_EXPERTS, MOE_INTERMEDIATE_SIZE), dtype=torch.float32)
        gate_weighted_abs = torch.zeros((NUM_EXPERTS, MOE_INTERMEDIATE_SIZE), dtype=torch.float32)
        input_variance = torch.zeros((NUM_EXPERTS, MOE_INTERMEDIATE_SIZE), dtype=torch.float32)
        output_grad_abs = torch.zeros((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32)

        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        gate_up_proj = move_tensor(store.load_tensors([gate_key])[gate_key], device, dtype)
        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()
        grad_tokens = gradient_captures[layer_idx]
        for expert_idx in active_experts:
            token_idx, route_pos = routed_token_indices(capture, expert_idx)
            if token_idx.numel() == 0:
                continue
            x = capture.mlp_input[token_idx].to(device=device, dtype=dtype)
            route_weights = capture.routing_weights[token_idx, route_pos].to(device=device, dtype=torch.float32)
            gate_up = F.linear(x, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = (F.silu(gate.float()) * up.float()).to(torch.float32)
            mean_abs[expert_idx] = intermediate.abs().mean(dim=0).cpu()
            denom = route_weights.sum().clamp_min(EPS)
            gate_weighted_abs[expert_idx] = ((intermediate.abs() * route_weights.unsqueeze(-1)).sum(dim=0) / denom).cpu()
            input_variance[expert_idx] = intermediate.var(dim=0, correction=0).cpu() if intermediate.shape[0] > 1 else torch.zeros(MOE_INTERMEDIATE_SIZE, dtype=torch.float32)
            output_grad_abs[expert_idx] = grad_tokens[token_idx].abs().mean(dim=0)
            target_key = (layer_idx, expert_idx)
            if target_key in targets:
                target_cache[target_key] = {
                    "target": targets[target_key],
                    "intermediate": intermediate.cpu(),
                    "token_idx": token_idx.clone(),
                    "route_weights": capture.routing_weights[token_idx, route_pos].clone(),
                }
        release_tensors({"gate_up_proj": gate_up_proj})

        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        down_weight = move_tensor(store.load_tensors([down_key])[down_key], device, dtype)
        down_fp4 = torch.empty_like(down_weight)
        for expert_idx in range(NUM_EXPERTS):
            down_fp4[expert_idx] = quantize_linear_weight(down_weight[expert_idx], "fp4")
        delta_abs = (down_weight.float() - down_fp4.float()).abs().cpu()

        act_metric = torch.einsum("ek,enk->en", mean_abs, delta_abs)
        gate_metric = torch.einsum("ek,enk->en", gate_weighted_abs, delta_abs)
        var_metric = torch.einsum("ek,enk->en", input_variance, delta_abs)
        scalebits_metric = act_metric * output_grad_abs

        all_metrics["act_weighted_qerror"][layer_idx] = act_metric
        all_metrics["gate_aware_act_qerror"][layer_idx] = gate_metric
        all_metrics["input_variance_qerror"][layer_idx] = var_metric
        all_metrics["scalebits_qerror"][layer_idx] = scalebits_metric

        del down_weight, down_fp4, delta_abs, mean_abs, gate_weighted_abs, input_variance, output_grad_abs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_metrics, target_cache, routing_counts


def load_ground_truth_and_existing_metrics(
    spike1_payload: dict[str, Any],
    spike2_payload: dict[str, Any],
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, dict[tuple[int, int], np.ndarray]], dict[tuple[int, int], str]]:
    targets = target_lookup()
    roles = {(target.layer_idx, target.expert_id): target.role for target in targets.values()}
    ground_truth: dict[tuple[int, int], np.ndarray] = {}
    for layer in spike1_payload["layers"]:
        layer_idx = int(layer["layer_idx"])
        for expert in layer["experts"]:
            key = (layer_idx, int(expert["expert_id"]))
            if key in targets:
                ground_truth[key] = np.asarray(expert["w2_channel_sensitivity"], dtype=np.float64)

    existing: dict[str, dict[tuple[int, int], np.ndarray]] = {metric: {} for metric in EXISTING_METRICS}
    for layer_idx, expert_id, _role in TARGET_EXPERT_SPECS:
        payload_key = f"layer_{layer_idx}_expert_{expert_id}_w2"
        expert_payload = spike2_payload[payload_key]
        for metric in EXISTING_METRICS:
            existing[metric][(layer_idx, expert_id)] = np.asarray(expert_payload[metric], dtype=np.float64)
    return ground_truth, existing, roles


def summarize_metric_family(
    metric_arrays: dict[str, dict[tuple[int, int], np.ndarray]],
    ground_truth: dict[tuple[int, int], np.ndarray],
    roles: dict[tuple[int, int], str],
) -> tuple[dict[str, Any], list[tuple[str, float, float, float]]]:
    summary: dict[str, Any] = {}
    ranking_rows: list[tuple[str, float, float, float]] = []
    for metric_name, expert_map in metric_arrays.items():
        gini_by_expert: dict[str, float] = {}
        spearman_by_expert: dict[str, float] = {}
        gini_by_role: dict[str, list[float]] = {role: [] for role in ROLE_ORDER}
        rho_by_role: dict[str, list[float]] = {role: [] for role in ROLE_ORDER}
        for key, values in expert_map.items():
            label = TargetExpert(key[0], key[1], roles[key]).label
            gini_value = gini(values)
            gini_by_expert[label] = gini_value
            gini_by_role[roles[key]].append(gini_value)
            if metric_name != "ground_truth":
                rho_value = safe_spearman(ground_truth[key], values)
                spearman_by_expert[label] = rho_value
                rho_by_role[roles[key]].append(rho_value)
        gini_mean_by_role = {
            role: float(np.mean(values, dtype=np.float64)) if values else 0.0
            for role, values in gini_by_role.items()
        }
        rho_mean_by_role = {
            role: float(np.mean(values, dtype=np.float64)) if values else 0.0
            for role, values in rho_by_role.items()
        }
        mean_rho = float(np.mean(list(spearman_by_expert.values()), dtype=np.float64)) if spearman_by_expert else 1.0
        mean_gini = float(np.mean(list(gini_by_expert.values()), dtype=np.float64)) if gini_by_expert else 0.0
        summary[metric_name] = {
            "gini_by_expert": gini_by_expert,
            "gini_mean_by_role": gini_mean_by_role,
            "spearman_by_expert": spearman_by_expert,
            "spearman_mean_by_role": rho_mean_by_role,
            "mean_spearman": mean_rho,
            "mean_gini": mean_gini,
        }
        ranking_rows.append((metric_name, mean_rho, mean_gini, rho_mean_by_role.get("hot", 0.0)))
    ranking_rows.sort(key=lambda item: item[1], reverse=True)
    return summary, ranking_rows


def build_assignment_errors(
    store: WeightStore,
    target_cache: dict[tuple[int, int], dict[str, Any]],
    new_metrics: dict[str, dict[int, torch.Tensor]],
    existing_metrics: dict[str, dict[tuple[int, int], np.ndarray]],
    best_new_metric: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    target_items = sorted(target_cache.items(), key=lambda item: (item[0][0], item[0][1]))
    for budget_pct in ASSIGNMENT_BUDGETS:
        k = int(round((budget_pct / 100.0) * HIDDEN_SIZE))
        bucket: dict[str, Any] = {"all_fp4": {}, "all_fp8": {}, "weight_l1": {}, best_new_metric: {}}
        aggregates: dict[str, list[float]] = {name: [] for name in bucket}
        for (layer_idx, expert_id), cache in target_items:
            target = cache["target"]
            intermediate = cache["intermediate"].to(device=device, dtype=dtype)
            down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
            down = move_tensor(store.load_tensors([down_key])[down_key], device, dtype)[expert_id]
            baseline = F.linear(intermediate, down).float()
            down_fp4 = quantize_linear_weight(down, "fp4")
            down_fp8 = quantize_linear_weight(down, "fp8")
            eval_weights: dict[str, torch.Tensor] = {
                "all_fp4": down_fp4,
                "all_fp8": down_fp8,
            }
            weight_mask = torch.from_numpy(topk_mask(existing_metrics["weight_l1"][(layer_idx, expert_id)], k)).to(device=device)
            new_mask = topk_mask(new_metrics[best_new_metric][layer_idx][expert_id].numpy(), k)
            eval_weights["weight_l1"] = quantize_down_proj_with_mask(down, weight_mask)
            eval_weights[best_new_metric] = quantize_down_proj_with_mask(down, torch.from_numpy(new_mask).to(device=device))
            for name, weight in eval_weights.items():
                quant_out = F.linear(intermediate, weight).float()
                delta = quant_out - baseline
                rel_l2 = float(torch.linalg.vector_norm(delta).item() / (torch.linalg.vector_norm(baseline).item() + EPS))
                mse = float(delta.pow(2).mean().item())
                bucket[name][target.label] = {"rel_l2": rel_l2, "mse": mse}
                aggregates[name].append(rel_l2)
            release_tensors({"down": down, "down_fp4": down_fp4, "down_fp8": down_fp8})
        results[str(budget_pct)] = {
            name: {
                "per_expert": bucket[name],
                "mean_rel_l2": float(np.mean(aggregates[name], dtype=np.float64)) if aggregates[name] else 0.0,
            }
            for name in bucket
        }
    return results


def estimate_memory_gb(total_bf16_bytes: int, config: Any, avg_down_fp8_fraction: float, mode: str) -> float:
    per_layer_gate_up = config.num_experts * (2 * config.moe_intermediate_size * config.hidden_size)
    per_layer_down = config.num_experts * (config.hidden_size * config.moe_intermediate_size)
    total_gate_up = config.num_hidden_layers * per_layer_gate_up
    total_down = config.num_hidden_layers * per_layer_down
    total_expert_bf16_bytes = 2 * (total_gate_up + total_down)
    non_expert_bytes = total_bf16_bytes - total_expert_bf16_bytes
    if mode == "fp4":
        expert_bytes = 0.5 * (total_gate_up + total_down)
    elif mode == "fp8":
        expert_bytes = 1.0 * (total_gate_up + total_down)
    elif mode == "mixed_expert_25":
        expert_bytes = (0.5 + (0.5 * 0.25)) * (total_gate_up + total_down)
    elif mode == "mixed_channel":
        expert_bytes = (0.5 * total_gate_up) + ((0.5 + (0.5 * avg_down_fp8_fraction)) * total_down)
    else:
        raise ValueError(f"Unsupported memory mode: {mode}")
    return round(float(non_expert_bytes + expert_bytes) / 1e9, 3)


def build_channel_masks_for_best_metric(
    metric_name: str,
    metric_scores: dict[str, dict[int, torch.Tensor]],
    routing_counts: dict[int, torch.Tensor],
    average_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[str, Any]]]:
    masks: dict[int, dict[int, torch.Tensor]] = {}
    metadata: dict[int, dict[str, Any]] = {}
    for layer_idx, counts in routing_counts.items():
        total_channels = int(round(average_fraction * HIDDEN_SIZE * NUM_EXPERTS))
        allocations = allocate_weighted_counts(
            total_channels,
            capacities=[HIDDEN_SIZE] * NUM_EXPERTS,
            weights=[float(value) for value in counts.tolist()],
        )
        layer_masks: dict[int, torch.Tensor] = {}
        for expert_idx in range(NUM_EXPERTS):
            k = allocations[expert_idx]
            if k <= 0:
                continue
            scores = metric_scores[metric_name][layer_idx][expert_idx].numpy()
            layer_masks[expert_idx] = torch.from_numpy(topk_mask(scores, k))
        masks[layer_idx] = layer_masks
        nonzero_alloc = [value for value in allocations if value > 0]
        top = torch.topk(counts, k=min(8, counts.numel()))
        metadata[layer_idx] = {
            "avg_channels_fp8": float(np.mean(nonzero_alloc, dtype=np.float64)) if nonzero_alloc else 0.0,
            "max_channels_fp8": int(max(allocations)) if allocations else 0,
            "min_channels_fp8_nonzero": int(min(nonzero_alloc)) if nonzero_alloc else 0,
            "total_channels_fp8": int(sum(allocations)),
            "top_routed_experts": [
                {"expert_id": int(idx), "token_count": int(value)}
                for value, idx in zip(top.values.tolist(), top.indices.tolist())
            ],
        }
    return masks, metadata


def resolve_promoted_experts(layer_counts: dict[int, torch.Tensor], fp8_fraction: float) -> dict[int, set[int]]:
    promoted: dict[int, set[int]] = {}
    for layer_idx, counts in layer_counts.items():
        promote_count = int(round(counts.numel() * fp8_fraction))
        ranked = sorted(range(counts.numel()), key=lambda expert_idx: (-int(counts[expert_idx]), expert_idx))
        promoted[layer_idx] = set(ranked[: max(promote_count, 0)])
    return promoted


def evaluate_perplexity(
    store: WeightStore,
    config: Any,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logit_chunk_size: int,
    moe_mode: str,
    promoted_experts: dict[int, set[int]] | None = None,
    channel_masks: dict[int, dict[int, torch.Tensor]] | None = None,
) -> tuple[float, float]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(store.weight_map)
    root = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_weight = move_tensor(root[embed_key], device, dtype)
    final_norm = move_tensor(root[norm_key], device, dtype)
    lm_head = move_tensor(root[lm_head_key], device, dtype)
    del root

    hidden_states = F.embedding(input_ids.to(device), embed_weight)
    seq_len = int(input_ids.shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = build_causal_mask(seq_len, device)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    position_embeddings = rotary(hidden_states, position_ids)

    with torch.inference_mode():
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[layer_idx]
            print(f"[perplexity] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d} mode={moe_mode}", flush=True)
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
            mlp_out, _ = moe_forward_quantized_mode(
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
                config.num_experts_per_tok,
                config.num_experts,
                mode=moe_mode,
                promoted_experts=promoted_experts.get(layer_idx) if promoted_experts is not None else None,
                channel_masks=channel_masks.get(layer_idx) if channel_masks is not None else None,
            )
            hidden_states = hidden_states + mlp_out
            release_tensors(tensors)

        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
        targets = input_ids[:, 1:].to(device)
        total_nll = 0.0
        token_count = targets.numel()
        for start in range(0, seq_len - 1, logit_chunk_size):
            end = min(seq_len - 1, start + logit_chunk_size)
            logits = F.linear(hidden_states[:, start:end, :], lm_head)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets[:, start:end].reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss_sum.item())
            del logits
    avg_nll = total_nll / max(token_count, 1)
    perplexity = float(math.exp(avg_nll))
    return avg_nll, perplexity


def print_gini_and_correlation_tables(summary_rows: list[tuple[str, float, float, float]], title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(format_metric_table(summary_rows), flush=True)


def render_exploration_section(
    best_new_metric: str,
    best_overall_metric: str,
    metric_rankings: list[tuple[str, float, float, float]],
    metric_summary: dict[str, Any],
    assignment_errors: dict[str, Any],
    perplexity_payload: dict[str, Any],
) -> str:
    lines = [NEW_SECTION_MARKER, ""]
    best_new = metric_summary[best_new_metric]
    best_overall = metric_summary[best_overall_metric]
    lines.append(
        f"**Approach**: Reused the Spike 1 streamed 40-layer forward to capture routed W2 inputs, then added four new per-output-channel metrics: activation-weighted FP4 quantization error, ScaleBITS-style gradient-weighted error from a draft FP4 backward pass, gate-aware activation weighting, and routed-token input variance weighting. I compared these against the 12 existing Spike 2 W2 metrics on the nine ground-truth experts, then used the best new metric for routing-aware per-channel FP8 assignment."
    )
    lines.append("")
    lines.append(
        f"**Result**: The best new metric is `{best_new_metric}` (mean Spearman {best_new['mean_spearman']:.4f}, mean Gini {best_new['mean_gini']:.4f}); the best overall metric in this run is `{best_overall_metric}` (mean Spearman {best_overall['mean_spearman']:.4f}). The top new metric's hot-expert correlation is {best_new['spearman_mean_by_role']['hot']:.4f}, and its tier-mean Ginis are cold={best_new['gini_mean_by_role']['cold']:.4f}, medium={best_new['gini_mean_by_role']['medium']:.4f}, hot={best_new['gini_mean_by_role']['hot']:.4f}."
    )
    lines.append("")
    lines.append("**Top metric ranking (mean Spearman / mean Gini / hot-expert Spearman)**:")
    lines.append("")
    for metric_name, mean_rho, mean_gini, hot_rho in metric_rankings[:6]:
        lines.append(f"- `{metric_name}`: rho={mean_rho:.4f}, gini={mean_gini:.4f}, hot-rho={hot_rho:.4f}.")
    lines.append("")
    lines.append("**Per-channel assignment on the 9 GT experts**:")
    lines.append("")
    for budget_pct in ASSIGNMENT_BUDGETS:
        bucket = assignment_errors[str(budget_pct)]
        lines.append(
            f"- K={budget_pct}% FP8 rows: all-FP4 mean rel-L2 {bucket['all_fp4']['mean_rel_l2']:.4f}, all-FP8 {bucket['all_fp8']['mean_rel_l2']:.4f}, weight_l1 sort {bucket['weight_l1']['mean_rel_l2']:.4f}, `{best_new_metric}` sort {bucket[best_new_metric]['mean_rel_l2']:.4f}."
        )
    lines.append("")
    new_cfg = perplexity_payload["configs"]["best_new_metric_freq_weighted"]
    lines.append(
        f"**Perplexity**: On the same streamed WikiText-2 evaluation span used for Spike 5, the new routing-aware per-channel config reaches PPL {new_cfg['perplexity']:.4f} at approx {new_cfg['memory_gb']:.3f} GB. For reference: uniform FP4 {perplexity_payload['configs']['uniform_fp4']['perplexity']:.4f}, uniform FP8 {perplexity_payload['configs']['uniform_fp8']['perplexity']:.4f}, spike5 routing-aware expert promotion {perplexity_payload['configs']['mixed_25pct']['perplexity']:.4f}."
    )
    lines.append("")
    lines.append(
        "**Verdict**: Routing-aware expert selection still matters, but these new activation-aware channel metrics test whether there is enough within-expert structure to exploit. The correlation table answers whether the metric is actually closer to ground truth; the assignment/error and perplexity results answer whether that extra discrimination survives contact with end-to-end quantization."
    )
    return "\n".join(lines)


def upsert_exploration_section(path: Path, section_body: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if NEW_SECTION_MARKER in existing:
        prefix = existing.split(NEW_SECTION_MARKER, maxsplit=1)[0].rstrip()
        updated = f"{prefix}\n\n{section_body}\n"
    else:
        updated = existing.rstrip()
        if updated:
            updated += "\n\n---\n\n"
        updated += f"{section_body}\n"
    path.write_text(updated, encoding="utf-8")


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
    calibration_ids, calibration_info = load_dataset_tokens(tokenizer, "train", args.calibration_tokens)
    eval_ids, eval_info = load_dataset_tokens(tokenizer, "test", args.eval_tokens)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    spike1_payload = load_json(args.spike1_json)
    spike2_payload = load_json(args.spike2_json)
    spike5_payload = load_json(args.spike5_json)

    captures = run_calibration_forward(store, config, calibration_ids, device, dtype)
    gradient_captures = run_quantized_backward_capture(store, config, calibration_ids, device, dtype, args.logit_chunk_size)
    new_metric_tensors, target_cache, routing_counts = compute_new_metrics_for_all_layers(
        store,
        config,
        captures,
        gradient_captures,
        device,
        dtype,
    )

    ground_truth, existing_metric_arrays, roles = load_ground_truth_and_existing_metrics(spike1_payload, spike2_payload)
    new_metric_arrays_np = {
        metric_name: {
            key: new_metric_tensors[metric_name][key[0]][key[1]].numpy()
            for key in ground_truth
        }
        for metric_name in NEW_METRICS
    }
    combined_metric_arrays = {"ground_truth": ground_truth} | existing_metric_arrays | new_metric_arrays_np
    metric_summary, ranking_rows = summarize_metric_family(combined_metric_arrays, ground_truth, roles)
    best_new_metric = max(NEW_METRICS, key=lambda name: metric_summary[name]["mean_spearman"])
    best_overall_metric = max(
        (metric_name for metric_name in combined_metric_arrays if metric_name != "ground_truth"),
        key=lambda name: metric_summary[name]["mean_spearman"],
    )

    print_gini_and_correlation_tables(ranking_rows, "Metric Ranking vs Ground Truth")

    assignment_errors = build_assignment_errors(
        store,
        target_cache,
        new_metric_tensors,
        existing_metric_arrays,
        best_new_metric,
        device,
        dtype,
    )
    print("\n=== Per-Expert Output Error (mean relative L2) ===", flush=True)
    for budget_pct in ASSIGNMENT_BUDGETS:
        bucket = assignment_errors[str(budget_pct)]
        print(
            f"K={budget_pct:>2d}% | all_fp4={bucket['all_fp4']['mean_rel_l2']:.4f} "
            f"all_fp8={bucket['all_fp8']['mean_rel_l2']:.4f} "
            f"weight_l1={bucket['weight_l1']['mean_rel_l2']:.4f} "
            f"{best_new_metric}={bucket[best_new_metric]['mean_rel_l2']:.4f}",
            flush=True,
        )

    channel_masks, channel_budget_meta = build_channel_masks_for_best_metric(
        best_new_metric,
        new_metric_tensors,
        routing_counts,
        args.channel_budget_equivalent_fraction,
    )
    total_bf16_bytes = int(root_config.get("total_size", 0) or load_json(snapshot_dir / "model.safetensors.index.json")["metadata"]["total_size"])

    avg_nll_new, ppl_new = evaluate_perplexity(
        store,
        config,
        eval_ids,
        device,
        dtype,
        args.logit_chunk_size,
        moe_mode="mixed_channel",
        channel_masks=channel_masks,
    )
    new_memory = estimate_memory_gb(total_bf16_bytes, config, args.channel_budget_equivalent_fraction, "mixed_channel")
    perplexity_payload = {
        "model": args.model_id,
        "calibration": calibration_info,
        "evaluation": eval_info,
        "best_new_metric": best_new_metric,
        "channel_budget_equivalent_fraction": args.channel_budget_equivalent_fraction,
        "configs": {
            "uniform_fp4": spike5_payload["configs"]["uniform_fp4"],
            "uniform_fp8": spike5_payload["configs"]["uniform_fp8"],
            "mixed_25pct": spike5_payload["configs"]["mixed_25pct"],
            "best_new_metric_freq_weighted": {
                "metric": best_new_metric,
                "avg_nll": avg_nll_new,
                "perplexity": ppl_new,
                "memory_gb": new_memory,
            },
        },
        "routing_budget": channel_budget_meta,
        "runtime_seconds": round(time.time() - start_time, 3),
    }
    atomic_json_dump(args.perplexity_json, perplexity_payload)

    print("\n=== Perplexity ===", flush=True)
    print(
        f"uniform_fp4={perplexity_payload['configs']['uniform_fp4']['perplexity']:.4f} "
        f"uniform_fp8={perplexity_payload['configs']['uniform_fp8']['perplexity']:.4f} "
        f"mixed_25pct={perplexity_payload['configs']['mixed_25pct']['perplexity']:.4f} "
        f"new_{best_new_metric}={ppl_new:.4f}",
        flush=True,
    )

    metrics_payload = {
        "model": args.model_id,
        "calibration": calibration_info,
        "gradient_source": {
            "description": "single draft-FP4 forward/backward on next-token cross-entropy",
            "tokens": calibration_info["actual_tokens"],
        },
        "target_experts": [
            {"layer_idx": layer_idx, "expert_id": expert_id, "role": role, "label": TargetExpert(layer_idx, expert_id, role).label}
            for layer_idx, expert_id, role in TARGET_EXPERT_SPECS
        ],
        "best_new_metric": best_new_metric,
        "best_overall_metric": best_overall_metric,
        "metric_summary": metric_summary,
        "metric_ranking": [
            {
                "metric": metric_name,
                "mean_spearman": mean_rho,
                "mean_gini": mean_gini,
                "hot_spearman": hot_rho,
            }
            for metric_name, mean_rho, mean_gini, hot_rho in ranking_rows
        ],
        "assignment_errors": assignment_errors,
        "runtime_seconds": round(time.time() - start_time, 3),
    }
    atomic_json_dump(args.metrics_json, metrics_payload)

    section = render_exploration_section(
        best_new_metric,
        best_overall_metric,
        ranking_rows,
        metric_summary,
        assignment_errors,
        perplexity_payload,
    )
    upsert_exploration_section(args.exploration_md, section)

    print(f"\nSaved metric summary -> {args.metrics_json}", flush=True)
    print(f"Saved perplexity summary -> {args.perplexity_json}", flush=True)
    print(f"Updated exploration log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

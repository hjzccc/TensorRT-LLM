#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from baselines_comparison import (
    LayerMetricBundle,
    allocate_weighted_counts,
    estimate_baseline_memory_gb,
    estimate_mixed_memory_gb,
    fp8_weights_from_masks,
    fp8_weights_from_projection_promotions,
    fp8_weights_from_promoted_experts,
    quantize_down_proj_with_mask,
    quantize_gate_up_with_pair_mask,
    quantize_linear_weight,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
    topk_mask_from_scores,
)
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
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_JSON_PATH = RESULTS_DIR / "proper_eval.json"
DEFAULT_MD_PATH = SCRIPT_DIR / "exploration.md"
SECTION_MARKER = "## [18] PROPER EVALUATION (GPTQ Standard)"
SEQLEN = 2048
CALIBRATION_SAMPLES = 128
PROMOTION_FRACTION = 0.25
EPS = 1e-10
LOGITS_CHUNK_TOKENS = 256


@dataclass(frozen=True)
class EvalPlan:
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
    restore_bf16: frozenset[str] = field(default_factory=frozenset)
    restore_moe_layers: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class CalibrationArtifacts:
    routing_counts: dict[int, torch.Tensor]
    activation_cache: dict[int, LayerMetricBundle]
    mxmoe_w1_deltas: dict[int, torch.Tensor]
    mxmoe_w2_deltas: dict[int, torch.Tensor]
    mc_moe_scores: dict[int, torch.Tensor]


@dataclass
class ExpertMomentState:
    counts: torch.Tensor
    input_sum1: torch.Tensor
    input_sum2: torch.Tensor
    input_sum3: torch.Tensor
    input_sum4: torch.Tensor
    inter_sum1: torch.Tensor
    inter_sum2: torch.Tensor
    inter_sum3: torch.Tensor
    inter_sum4: torch.Tensor
    mxmoe_w1_sq: torch.Tensor
    mxmoe_w2_sq: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_MD_PATH)
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


def build_prompt_text(dataset_split: str) -> str:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=dataset_split)
    return "\n\n".join(dataset["text"])


def load_gptq_standard_data(model_id: str) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    train_text = build_prompt_text("train")
    trainenc = tokenizer(train_text, return_tensors="pt")
    train_ids = torch.as_tensor(trainenc.input_ids, dtype=torch.long)

    random.seed(0)
    calib_samples: list[torch.Tensor] = []
    for _ in range(CALIBRATION_SAMPLES):
        start = random.randint(0, int(train_ids.shape[1]) - SEQLEN - 1)
        calib_samples.append(train_ids[:, start : start + SEQLEN])
    calib_chunks = torch.cat(calib_samples, dim=0).contiguous()

    test_text = build_prompt_text("test")
    testenc = tokenizer(test_text, return_tensors="pt")
    test_ids = torch.as_tensor(testenc.input_ids, dtype=torch.long)

    calib_info = {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "train",
        "tokenizer": tokenizer.name_or_path,
        "seqlen": SEQLEN,
        "samples": CALIBRATION_SAMPLES,
        "random_seed": 0,
        "train_tokens": int(train_ids.shape[1]),
        "sample_shape": [int(calib_chunks.shape[0]), int(calib_chunks.shape[1])],
    }
    eval_info = {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "test",
        "tokenizer": tokenizer.name_or_path,
        "seqlen": SEQLEN,
        "total_tokens": int(test_ids.numel()),
        "nsamples": int(test_ids.numel() // SEQLEN),
        "used_tokens": int((test_ids.numel() // SEQLEN) * SEQLEN),
        "non_overlapping": True,
    }
    return tokenizer, calib_chunks, test_ids, calib_info, eval_info


def embed_chunks(store: WeightStore, embed_key: str, chunks: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    embed_weight = move_tensor(store.load_tensors([embed_key])[embed_key], device, dtype)
    try:
        return F.embedding(chunks.to(device=device, non_blocking=True), embed_weight)
    finally:
        release_tensors({"embed_weight": embed_weight})


def build_position_context(config: Qwen3NextConfig, hidden_states: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    seq_len = int(hidden_states.shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = build_causal_mask(seq_len, device)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    position_embeddings = rotary(hidden_states, position_ids)
    return causal_mask, position_embeddings


def layer_type_at(config: Qwen3NextConfig, layer_idx: int) -> str:
    if config.layer_types is None:
        raise RuntimeError("Qwen3NextConfig.layer_types is required")
    return str(config.layer_types[layer_idx])


def plan_restores_bf16(plan: EvalPlan, component: str, layer_idx: int | None = None) -> bool:
    if component in plan.restore_bf16:
        return True
    return component == "moe" and layer_idx is not None and layer_idx in plan.restore_moe_layers


def init_expert_moment_state(config: Qwen3NextConfig, device: torch.device) -> ExpertMomentState:
    return ExpertMomentState(
        counts=torch.zeros(config.num_experts, dtype=torch.int64, device=device),
        input_sum1=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        input_sum2=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        input_sum3=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        input_sum4=torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=device),
        inter_sum1=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        inter_sum2=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        inter_sum3=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        inter_sum4=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=device),
        mxmoe_w1_sq=torch.zeros(config.num_experts, dtype=torch.float32, device=device),
        mxmoe_w2_sq=torch.zeros(config.num_experts, dtype=torch.float32, device=device),
    )


def accumulate_moments(sum1: torch.Tensor, sum2: torch.Tensor, sum3: torch.Tensor, sum4: torch.Tensor, values: torch.Tensor) -> None:
    values_f = values.float()
    sum1.add_(values_f.sum(dim=0))
    squared = values_f.square()
    sum2.add_(squared.sum(dim=0))
    cubed = squared * values_f
    sum3.add_(cubed.sum(dim=0))
    sum4.add_((cubed * values_f).sum(dim=0))


def kurtosis_from_moments(sum1: torch.Tensor, sum2: torch.Tensor, sum3: torch.Tensor, sum4: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 1:
        return torch.zeros_like(sum1, dtype=torch.float32)
    count_f = float(count)
    mean = sum1 / count_f
    mean_sq = mean.square()
    raw2 = sum2 / count_f
    raw3 = sum3 / count_f
    raw4 = sum4 / count_f
    var = torch.clamp(raw2 - mean_sq, min=0.0)
    central4 = raw4 - (4.0 * mean * raw3) + (6.0 * mean_sq * raw2) - (3.0 * mean_sq.square())
    kurt = central4 / (var.square() + EPS)
    return torch.where(torch.isfinite(kurt), kurt, torch.zeros_like(kurt)).to(torch.float32)


def collect_layer_calibration(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    config: Qwen3NextConfig,
    state: ExpertMomentState,
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
    state.counts.add_(expert_counts)

    active_experts = torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist()
    for expert_idx in active_experts:
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        weights = routing_weights[token_idx, route_pos].unsqueeze(-1).to(torch.float32)

        gate_up = F.linear(current_state, gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        full_hidden = F.silu(gate) * up
        full_out = F.linear(full_hidden, down_proj[expert_idx])
        final_hidden_states.index_add_(0, token_idx, (weights * full_out.float()).to(hidden_states.dtype))

        accumulate_moments(
            state.input_sum1[expert_idx],
            state.input_sum2[expert_idx],
            state.input_sum3[expert_idx],
            state.input_sum4[expert_idx],
            current_state,
        )
        accumulate_moments(
            state.inter_sum1[expert_idx],
            state.inter_sum2[expert_idx],
            state.inter_sum3[expert_idx],
            state.inter_sum4[expert_idx],
            full_hidden,
        )

        q_gate_up = F.linear(current_state, fp4_gate_up[expert_idx])
        q_gate, q_up = q_gate_up.chunk(2, dim=-1)
        q_hidden_w1 = F.silu(q_gate) * q_up
        q_out_w1 = F.linear(q_hidden_w1, down_proj[expert_idx]).float()
        q_out_w2 = F.linear(full_hidden, fp4_down[expert_idx]).float()
        full_out_f = full_out.float()
        state.mxmoe_w1_sq[expert_idx] += torch.sum((weights * (q_out_w1 - full_out_f)).square())
        state.mxmoe_w2_sq[expert_idx] += torch.sum((weights * (q_out_w2 - full_out_f)).square())

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def finalize_layer_metrics(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    state: ExpertMomentState,
    config: Qwen3NextConfig,
) -> tuple[LayerMetricBundle, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_scores = torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32, device=gate_up_proj.device)
    w2_scores = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32, device=gate_up_proj.device)
    mc_scores = torch.zeros(config.num_experts, dtype=torch.float32, device=gate_up_proj.device)

    for expert_idx in range(config.num_experts):
        count = int(state.counts[expert_idx].item())
        input_kurt = kurtosis_from_moments(
            state.input_sum1[expert_idx],
            state.input_sum2[expert_idx],
            state.input_sum3[expert_idx],
            state.input_sum4[expert_idx],
            count,
        )
        inter_kurt = kurtosis_from_moments(
            state.inter_sum1[expert_idx],
            state.inter_sum2[expert_idx],
            state.inter_sum3[expert_idx],
            state.inter_sum4[expert_idx],
            count,
        )

        gate_up_weight = gate_up_proj[expert_idx].float()
        gate_up_q = fp4_gate_up[expert_idx].float()
        gate_diff = gate_up_weight - gate_up_q
        pair_diff_abs = gate_diff[: config.moe_intermediate_size].abs() + gate_diff[config.moe_intermediate_size :].abs()
        w1_scores[expert_idx] = torch.matmul(pair_diff_abs, input_kurt)

        down_weight = down_proj[expert_idx].float()
        down_q = fp4_down[expert_idx].float()
        down_diff = down_weight - down_q
        w2_scores[expert_idx] = torch.matmul(down_diff.abs(), inter_kurt)

        weight_norm_sq = gate_up_weight.square().sum() + down_weight.square().sum()
        quant_loss_sq = gate_diff.square().sum() + down_diff.square().sum()
        actnum = float(count)
        if actnum > 0.0:
            weight_norm = torch.sqrt(weight_norm_sq)
            quant_loss = torch.sqrt(quant_loss_sq)
            mc_scores[expert_idx] = float((actnum ** 1.0) * (float(weight_norm.item()) ** 1.5) * (float(quant_loss.item()) ** 2.0))

    return (
        LayerMetricBundle(
            routing_counts=state.counts.detach().cpu().to(torch.int64),
            w1_pair_scores=w1_scores.detach().cpu(),
            w2_channel_scores=w2_scores.detach().cpu(),
        ),
        torch.sqrt(state.mxmoe_w1_sq).detach().cpu(),
        torch.sqrt(state.mxmoe_w2_sq).detach().cpu(),
        mc_scores.detach().cpu(),
    )


@torch.inference_mode()
def run_calibration(
    store: WeightStore,
    config: Qwen3NextConfig,
    calib_chunks: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> CalibrationArtifacts:
    print("\n=== Calibration (GPTQ standard, 128 random 2048-token chunks) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[calib] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_expert_moment_state(config, device)

        for chunk_idx in range(inps.shape[0]):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = collect_layer_calibration(mlp_input, moe_tensors, fp4_gate_up, fp4_down, config, state)
            outs[chunk_idx] = residual + moe_out
            print(f"[calib] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]}", flush=True)

        bundle, layer_w1, layer_w2, layer_mc = finalize_layer_metrics(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
        routing_counts[layer_idx] = bundle.routing_counts
        activation_cache[layer_idx] = bundle
        mxmoe_w1_deltas[layer_idx] = layer_w1
        mxmoe_w2_deltas[layer_idx] = layer_w2
        mc_moe_scores[layer_idx] = layer_mc

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return CalibrationArtifacts(
        routing_counts=routing_counts,
        activation_cache=activation_cache,
        mxmoe_w1_deltas=mxmoe_w1_deltas,
        mxmoe_w2_deltas=mxmoe_w2_deltas,
        mc_moe_scores=mc_moe_scores,
    )


def build_topk_expert_promotions(score_map: dict[int, torch.Tensor], config: Qwen3NextConfig, fraction: float) -> dict[int, set[int]]:
    target = int(round(config.num_experts * fraction))
    promoted: dict[int, set[int]] = {}
    for layer_idx, scores in score_map.items():
        ranked = sorted(range(config.num_experts), key=lambda expert_idx: (-float(scores[expert_idx].item()), expert_idx))
        promoted[layer_idx] = set(ranked[:target])
    return promoted


def build_two_level_masks(
    metric_cache: dict[int, LayerMetricBundle],
    config: Qwen3NextConfig,
    w1_fraction: float,
    w2_fraction: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
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


def build_mxmoe_projection_promotions(
    config: Qwen3NextConfig,
    total_expert_elems: int,
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    target_fp8_weights = int(round(PROMOTION_FRACTION * total_expert_elems))
    w1_cost = 2 * config.moe_intermediate_size * config.hidden_size
    w2_cost = config.hidden_size * config.moe_intermediate_size
    items: list[tuple[float, int, int, int, str]] = []
    for layer_idx in range(config.num_hidden_layers):
        for expert_idx in range(config.num_experts):
            items.append((float(w1_deltas[layer_idx][expert_idx].item()) / float(w1_cost + EPS), w1_cost, layer_idx, expert_idx, "w1"))
            items.append((float(w2_deltas[layer_idx][expert_idx].item()) / float(w2_cost + EPS), w2_cost, layer_idx, expert_idx, "w2"))
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


def build_eval_plans(
    calibration: CalibrationArtifacts,
    config: Qwen3NextConfig,
    total_bf16_bytes: int,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[EvalPlan]:
    routing_promoted = build_topk_expert_promotions(calibration.routing_counts, config, PROMOTION_FRACTION)
    mc_promoted = build_topk_expert_promotions(calibration.mc_moe_scores, config, PROMOTION_FRACTION)
    akurt_w1_4, akurt_w2_16 = build_two_level_masks(calibration.activation_cache, config, 0.04, 0.16)
    akurt_w1_10, akurt_w2_40 = build_two_level_masks(calibration.activation_cache, config, 0.10, 0.40)
    mxmoe_w1_proj, mxmoe_w2_proj = build_mxmoe_projection_promotions(config, total_expert_elems, calibration.mxmoe_w1_deltas, calibration.mxmoe_w2_deltas)

    routing_fp8_weights = fp8_weights_from_promoted_experts(config, routing_promoted)
    akurt_4_16_fp8_weights = fp8_weights_from_masks(config, akurt_w1_4, akurt_w2_16)
    akurt_10_40_fp8_weights = fp8_weights_from_masks(config, akurt_w1_10, akurt_w2_40)
    mxmoe_fp8_weights = fp8_weights_from_projection_promotions(config, mxmoe_w1_proj, mxmoe_w2_proj)
    mc_fp8_weights = fp8_weights_from_promoted_experts(config, mc_promoted)

    return [
        EvalPlan(
            name="bf16",
            description="No expert quantization.",
            mode="bf16",
            memory_gb=estimate_baseline_memory_gb(total_bf16_bytes, non_expert_bytes, total_expert_elems, "bf16"),
            fp8_weights=0,
        ),
        EvalPlan(
            name="uniform_fp4",
            description="All expert weights quantized to NVFP4, dequantized to BF16, evaluated with F.linear.",
            mode="uniform_fp4",
            memory_gb=estimate_baseline_memory_gb(total_bf16_bytes, non_expert_bytes, total_expert_elems, "fp4"),
            fp8_weights=0,
        ),
        EvalPlan(
            name="uniform_fp8",
            description="All expert weights quantized to FP8, dequantized to BF16, evaluated with F.linear.",
            mode="uniform_fp8",
            memory_gb=estimate_baseline_memory_gb(total_bf16_bytes, non_expert_bytes, total_expert_elems, "fp8"),
            fp8_weights=total_expert_elems,
        ),
        EvalPlan(
            name="routing_per_expert_25pct",
            description="Top 25% experts per layer by routing count promoted to FP8; remaining experts use FP4.",
            mode="per_expert",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, routing_fp8_weights),
            fp8_weights=routing_fp8_weights,
            promoted_experts=routing_promoted,
        ),
        EvalPlan(
            name="perchannel_akurt_w1_4_w2_16",
            description="Routing-aware activation_kurtosis masks with W1=4% and W2=16% FP8.",
            mode="per_channel",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, akurt_4_16_fp8_weights),
            fp8_weights=akurt_4_16_fp8_weights,
            w1_pair_masks=akurt_w1_4,
            w2_channel_masks=akurt_w2_16,
        ),
        EvalPlan(
            name="perchannel_akurt_w1_10_w2_40",
            description="Routing-aware activation_kurtosis masks with W1=10% and W2=40% FP8.",
            mode="per_channel",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, akurt_10_40_fp8_weights),
            fp8_weights=akurt_10_40_fp8_weights,
            w1_pair_masks=akurt_w1_10,
            w2_channel_masks=akurt_w2_40,
        ),
        EvalPlan(
            name="mxmoe_per_block",
            description="MxMoE greedy knapsack over per-projection output perturbation at 25% budget.",
            mode="per_block",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, mxmoe_fp8_weights),
            fp8_weights=mxmoe_fp8_weights,
            w1_projection_fp8=mxmoe_w1_proj,
            w2_projection_fp8=mxmoe_w2_proj,
        ),
        EvalPlan(
            name="mc_moe_per_expert",
            description="MC-MoE importance = freq^1 * weight^1.5 * quant_loss^2, greedy per-expert at 25%.",
            mode="per_expert",
            memory_gb=estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, mc_fp8_weights),
            fp8_weights=mc_fp8_weights,
            promoted_experts=mc_promoted,
        ),
    ]


def prepare_layer_tensors_for_plan(plan: EvalPlan, layer_idx: int, tensors: dict[str, torch.Tensor], config: Qwen3NextConfig) -> None:
    if plan.mode == "bf16" or plan_restores_bf16(plan, "moe", layer_idx):
        return

    gate_up_proj = tensors["mlp.experts.gate_up_proj"]
    down_proj = tensors["mlp.experts.down_proj"]
    prepared_gate_up: list[torch.Tensor] = []
    prepared_down: list[torch.Tensor] = []

    for expert_idx in range(config.num_experts):
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]
        if plan.mode == "uniform_fp4":
            quant_gate_up = quantize_linear_weight(gate_up_weight, "fp4")
            quant_down = quantize_linear_weight(down_weight, "fp4")
        elif plan.mode == "uniform_fp8":
            quant_gate_up = quantize_linear_weight(gate_up_weight, "fp8")
            quant_down = quantize_linear_weight(down_weight, "fp8")
        elif plan.mode == "per_expert":
            layer_promoted = set() if plan.promoted_experts is None else (plan.promoted_experts.get(layer_idx) or set())
            precision = "fp8" if expert_idx in layer_promoted else "fp4"
            quant_gate_up = quantize_linear_weight(gate_up_weight, precision)
            quant_down = quantize_linear_weight(down_weight, precision)
        elif plan.mode == "per_block":
            w1_mode = "fp8" if plan.w1_projection_fp8 is not None and bool(plan.w1_projection_fp8[layer_idx][expert_idx]) else "fp4"
            w2_mode = "fp8" if plan.w2_projection_fp8 is not None and bool(plan.w2_projection_fp8[layer_idx][expert_idx]) else "fp4"
            quant_gate_up = quantize_linear_weight(gate_up_weight, w1_mode)
            quant_down = quantize_linear_weight(down_weight, w2_mode)
        elif plan.mode == "per_channel":
            if plan.w1_pair_masks is None or plan.w2_channel_masks is None:
                raise ValueError(f"Plan {plan.name} missing per-channel masks")
            quant_gate_up = quantize_gate_up_with_pair_mask(gate_up_weight, plan.w1_pair_masks[layer_idx][expert_idx].to(device=gate_up_weight.device))
            quant_down = quantize_down_proj_with_mask(down_weight, plan.w2_channel_masks[layer_idx][expert_idx].to(device=down_weight.device))
        else:
            raise ValueError(f"Unsupported plan mode: {plan.mode}")
        prepared_gate_up.append(quant_gate_up)
        prepared_down.append(quant_down)

    tensors["mlp.experts.gate_up_proj"] = torch.stack(prepared_gate_up, dim=0)
    tensors["mlp.experts.down_proj"] = torch.stack(prepared_down, dim=0)
    del gate_up_proj, down_proj, prepared_gate_up, prepared_down
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def moe_forward_eval(hidden_states: torch.Tensor, tensors: dict[str, torch.Tensor], config: Qwen3NextConfig) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, tensors["experts.down_proj"][expert_idx])
        final_hidden_states.index_add_(0, token_idx, (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden).to(hidden_states.dtype))

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


@torch.inference_mode()
def evaluate_plan(
    plan: EvalPlan,
    test_ids: torch.Tensor,
    config: Qwen3NextConfig,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    nsamples = int(test_ids.numel() // SEQLEN)
    eval_chunks = test_ids[:, : nsamples * SEQLEN].view(nsamples, SEQLEN)
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

    root_tensors = store.load_tensors([norm_key, lm_head_key])
    final_norm = move_tensor(root_tensors[norm_key], device, dtype)
    lm_head = move_tensor(root_tensors[lm_head_key], device, dtype)
    del root_tensors

    inps = embed_chunks(store, embed_key, eval_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[{plan.name}] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors
        prepare_layer_tensors_for_plan(plan, layer_idx, tensors, config)

        for chunk_idx in range(nsamples):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = moe_forward_eval(mlp_input, moe_tensors, config)
            outs[chunk_idx] = residual + moe_out
            print(f"[{plan.name}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{nsamples}", flush=True)

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids_device = test_ids.to(device=device, non_blocking=True)
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nlls: list[torch.Tensor] = []
    for chunk_idx in range(nsamples):
        hidden_states = inps[chunk_idx].unsqueeze(0)
        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
        shift_labels = test_ids_device[:, chunk_idx * SEQLEN : (chunk_idx + 1) * SEQLEN][:, 1:]
        shift_tokens = int(shift_labels.numel())
        chunk_loss_sum = torch.zeros((), dtype=torch.float64, device=hidden_states.device)
        for start in range(0, shift_tokens, LOGITS_CHUNK_TOKENS):
            end = min(start + LOGITS_CHUNK_TOKENS, shift_tokens)
            logits = F.linear(hidden_states[:, start:end, :].float(), lm_head.float())
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels[:, start:end].reshape(-1))
            chunk_loss_sum = chunk_loss_sum + loss.to(torch.float64)
            del logits, loss
        nlls.append((chunk_loss_sum / float(max(shift_tokens, 1))).to(torch.float32) * SEQLEN)
        print(f"[{plan.name}] logits chunk {chunk_idx + 1}/{nsamples}", flush=True)

    mean_nll = torch.stack(nlls).sum() / (nsamples * SEQLEN)
    ppl = torch.exp(mean_nll)

    release_tensors({
        "inps": inps,
        "outs": outs,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "causal_mask": causal_mask,
        "test_ids_device": test_ids_device,
    })
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(ppl.item()), float(mean_nll.item()), nsamples


def load_legacy_reference_ppls() -> dict[str, float]:
    references: dict[str, float] = {}

    baseline_path = RESULTS_DIR / "baselines_comparison.json"
    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        mapping = {
            "bf16": "bf16",
            "uniform_fp4": "uniform_fp4",
            "uniform_fp8": "uniform_fp8",
            "routing_per_expert_25pct": "routing_per_expert",
            "mxmoe_per_block": "mxmoe_per_block",
            "mc_moe_per_expert": "mc_moe_per_expert",
        }
        for new_name, old_name in mapping.items():
            if old_name in payload.get("results", {}):
                references[new_name] = float(payload["results"][old_name]["ppl"])

    iter03_path = RESULTS_DIR / "exploration_iter03.json"
    if iter03_path.exists():
        with iter03_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        budget_sweep = payload.get("budget_sweep", {})
        split_ratio = payload.get("split_ratio_sweep", {})
        if "total_10" in budget_sweep:
            references["perchannel_akurt_w1_4_w2_16"] = float(budget_sweep["total_10"]["ppl"])
        if "w1_10_w2_40" in split_ratio:
            references["perchannel_akurt_w1_10_w2_40"] = float(split_ratio["w1_10_w2_40"]["ppl"])

    return references


def format_markdown_table(results: dict[str, Any], legacy_ppls: dict[str, float]) -> str:
    lines = ["| Config | PPL | Prev 512tok PPL | Delta | Memory (GB) |", "|--------|-----|-----------------|-------|-------------|"]
    for name, row in sorted(results.items(), key=lambda item: float(item[1]["ppl"])):
        prev = legacy_ppls.get(name)
        if prev is None:
            prev_text = "-"
            delta_text = "-"
        else:
            prev_text = f"{prev:.4f}"
            delta_text = f"{float(row['ppl']) - prev:+.4f}"
        lines.append(f"| {name} | {float(row['ppl']):.4f} | {prev_text} | {delta_text} | {float(row['memory_gb']):.3f} |")
    return "\n".join(lines)


def build_insight_text(results: dict[str, Any], legacy_ppls: dict[str, float]) -> str:
    deltas = {name: float(results[name]["ppl"]) - legacy_ppls[name] for name in results if name in legacy_ppls}
    if not deltas:
        return "The full 2048x145 evaluation changes the scale of the numbers enough that the earlier 512-token runs are no longer reliable for ranking methods."
    mean_delta = sum(deltas.values()) / len(deltas)
    biggest_name = max(deltas, key=lambda key: abs(deltas[key]))
    quantized_only = [name for name in results if name != "bf16"]
    best_name = min(quantized_only, key=lambda key: float(results[key]["ppl"])) if quantized_only else min(results, key=lambda key: float(results[key]["ppl"]))
    direction = "higher" if mean_delta > 0 else "lower"
    return (
        f"Relative to the earlier 512-token runs, the full 2048x145 protocol shifts PPL {direction} on average by {abs(mean_delta):.4f}. "
        f"The largest move is `{biggest_name}` at {deltas[biggest_name]:+.4f} PPL, and the new best quantized setting is `{best_name}`."
    )


def build_next_text(results: dict[str, Any]) -> str:
    ordered = [name for name, _row in sorted(results.items(), key=lambda item: float(item[1]["ppl"])) if name != "bf16"]
    if not ordered:
        return "Repeat with a second seed or a second dataset before drawing conclusions."
    winner = ordered[0]
    if winner.startswith("perchannel_"):
        return f"Sweep budgets tightly around `{winner}`, then ablate routing-aware allocation versus within-expert ranking under the same full-test protocol."
    if winner == "routing_per_expert_25pct":
        return "Test whether expert-only promotion stays ahead at nearby budgets (15%, 20%, 30%) before investing more time in channel masks."
    if winner == "mxmoe_per_block":
        return "Probe whether projection-level promotion remains stable at 20-30% budgets and whether a better perturbation cache improves the knapsack ranking."
    return f"Use `{winner}` as the new reference and run a local budget sweep around it under the same GPTQ-standard setup."


def upsert_exploration_section(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = ""
    if SECTION_MARKER in existing:
        before, _marker, _rest = existing.partition(SECTION_MARKER)
        updated = before.rstrip() + "\n\n" + content.strip() + "\n"
    else:
        prefix = existing.rstrip()
        if prefix:
            updated = prefix + "\n\n" + content.strip() + "\n"
        else:
            updated = content.strip() + "\n"
    path.write_text(updated, encoding="utf-8")


def render_exploration_section(results: dict[str, Any], eval_info: dict[str, Any], legacy_ppls: dict[str, float]) -> str:
    table = format_markdown_table(results, legacy_ppls)
    insight = build_insight_text(results, legacy_ppls)
    next_text = build_next_text(results)
    return (
        f"{SECTION_MARKER}\n"
        f"**Eval**: Full WikiText-2 test ({eval_info['total_tokens']} tokens, {eval_info['nsamples']} chunks of {SEQLEN}), simulated quant, BF16 F.linear, FP32 loss\n"
        f"**Approach**: Re-ran baselines + best approaches from prior work with correct GPTQ-standard evaluation\n"
        f"**Result**: \n{table}\n"
        f"**Insight**: {insight}\n"
        f"**Next**: {next_text}"
    )


def print_results_table(results: dict[str, Any], eval_info: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: float(item[1]["ppl"]))
    print("\n" + "=" * 116, flush=True)
    print(
        f"Proper evaluation results | full WikiText-2 | {eval_info['total_tokens']} tokens | {eval_info['nsamples']} x {SEQLEN} chunks | simulated quant",
        flush=True,
    )
    print("=" * 116, flush=True)
    print(f"{'Config':<34} {'Mode':<12} {'PPL':>10} {'NLL':>10} {'Memory GB':>12} {'FP8 frac':>10}", flush=True)
    print("-" * 116, flush=True)
    for name, row in ordered:
        print(
            f"{name:<34} {str(row['mode']):<12} {float(row['ppl']):>10.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    tokenizer, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    del tokenizer

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    if text_config.layer_types is None:
        raise RuntimeError("Qwen3NextConfig.layer_types is required for proper evaluation")
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: quantize -> dequantize -> BF16 -> F.linear for MoE expert weights; FP32 logits and loss",
            "protocol": {
                "reference": "/home/jerry/Documents/fork_new/MC-MoE/eval_ppl_utils.py",
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
        },
        "results": {},
    }

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    calibration = run_calibration(store, text_config, calib_chunks, device, dtype)
    plans = build_eval_plans(calibration, text_config, total_bf16_bytes, non_expert_bytes, total_expert_elems)

    payload["metadata"]["calibration_summary"] = {
        "active_experts_per_layer": {str(layer_idx): int((counts > 0).sum().item()) for layer_idx, counts in calibration.routing_counts.items()},
        "routing_top5_per_layer": {
            str(layer_idx): [
                {"expert_idx": int(expert_idx), "count": int(count)}
                for count, expert_idx in zip(
                    torch.topk(counts, k=min(5, counts.numel())).values.tolist(),
                    torch.topk(counts, k=min(5, counts.numel())).indices.tolist(),
                )
            ]
            for layer_idx, counts in calibration.routing_counts.items()
        },
    }

    for plan_idx, plan in enumerate(plans, start=1):
        print(f"\n=== Eval {plan_idx}/{len(plans)}: {plan.name} ===", flush=True)
        start_time = time.time()
        ppl, nll, nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - start_time
        payload["results"][plan.name] = {
            "description": plan.description,
            "mode": plan.mode,
            "ppl": round(ppl, 6),
            "nll": round(nll, 6),
            "memory_gb": round(plan.memory_gb, 3),
            "fp8_weights": int(plan.fp8_weights),
            "fp8_fraction": round(float(plan.fp8_weights) / float(max(total_expert_elems, 1)), 6),
            "eval_chunks": nsamples,
            "seqlen": SEQLEN,
            "time_s": round(elapsed, 1),
        }
        payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
        atomic_json_dump(args.output_json, payload)
        print(
            f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | time={elapsed:.1f}s",
            flush=True,
        )

    legacy_ppls = load_legacy_reference_ppls()
    section = render_exploration_section(payload["results"], eval_info, legacy_ppls)
    upsert_exploration_section(args.exploration_md, section)
    print_results_table(payload["results"], eval_info)
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

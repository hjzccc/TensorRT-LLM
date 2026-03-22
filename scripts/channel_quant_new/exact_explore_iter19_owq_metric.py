#!/usr/bin/env python3
"""Exact-path Exploration Iteration 19: OWQ-Inspired Metric (Weight-Magnitude Weighted).

IMPROVEMENT OVER ITER02:
  iter02 uses router-affinity metric (routing frequency weighted by quantization error).
  This variant implements a weight-magnitude weighted variant that prioritizes channels
  with larger weight magnitudes, which tend to be more sensitive to quantization.
  
  Hypothesis: Channels with larger weight magnitudes are more sensitive to quantization.
  Weight-magnitude weighting should improve PPL by 0.01-0.03 by prioritizing these channels.

APPROACH:
  1. Load calibration cache and router-affinity metric (same as iter02)
  2. For each layer, compute per-channel weight magnitude statistics
  3. Compute hybrid OWQ scores: router_affinity_score * weight_magnitude_factor
  4. Use global allocation with hybrid scores (same as iter02 infrastructure)
  5. Evaluate on full WikiText-2 (145 samples)

EXPECTED RESULTS:
  - OWQ should beat router-affinity (iter02: 6.6212) by 0.01-0.03 PPL
  - Target: PPL < 6.61 (0.0112 improvement over baseline)

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter19_owq_metric.py --nsamples 4
"""
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from baselines_comparison import LayerMetricBundle, topk_mask_from_scores
from proper_iter01 import load_cache
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter10_novel_perchannel import build_global_fraction_masks, load_metric_cache
from spike1_ground_truth import (
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
    WeightStore,
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter19_owq_metric.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter10_novel_perchannel_metric_cache.pt")
JOINT_BASE_BONUS = 1_000_000.0
W1_PAIR_GRANULARITY = 16
W2_CHANNEL_GRANULARITY = 32
EPS = 1e-10


@dataclass(frozen=True)
class BudgetConfig:
    """FP8 budget fractions for W1 and W2."""
    label: str
    total_fp8_fraction: float  # Total FP8 budget (W1 + W2 combined)
    w1_fraction: float  # Fixed W1 fraction (1/5 of total)
    w2_fraction: float  # Fixed W2 fraction (4/5 of total)

    @property
    def description(self) -> str:
        return f"Total FP8={self.total_fp8_fraction:.0%} (OWQ metric, fixed 1:4 W1:W2)"


# Fixed 1:4 W1:W2 ratio (from iter20 finding that adaptive ratios cause NaN)
ALL_BUDGET_CONFIGS = [
    BudgetConfig("budget_10pct",  0.10, 0.02, 0.08),
    BudgetConfig("budget_20pct",  0.20, 0.04, 0.16),
    BudgetConfig("budget_30pct",  0.30, 0.06, 0.24),
    BudgetConfig("budget_40pct",  0.40, 0.08, 0.32),
    BudgetConfig("budget_50pct",  0.50, 0.10, 0.40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-batch-size", type=int, default=1)
    return parser.parse_args()


def compute_weight_magnitude_factors(
    model_id: str,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
) -> dict[int, LayerMetricBundle]:
    """Compute weight-magnitude factors for OWQ metric.
    
    Returns:
        Dictionary mapping layer_idx -> LayerMetricBundle with weight-magnitude factors
    """
    store = WeightStore(model_id, snapshot_dir, weight_map)
    owq_cache: dict[int, LayerMetricBundle] = {}
    
    for layer_idx in range(config.num_hidden_layers):
        layer_type = config.layer_types[layer_idx]
        raw = store.load_tensors(layer_keys(layer_idx, layer_type))
        
        # Extract W1 and W2 weights
        gate_up_proj = raw.get("experts.gate_up_proj")
        down_proj = raw.get("experts.down_proj")
        
        if gate_up_proj is None or down_proj is None:
            print(f"Warning: Layer {layer_idx} missing expert weights")
            routing_counts = torch.ones(config.num_experts, dtype=torch.int64)
            w1_pair_scores = torch.ones(config.num_experts, config.hidden_size // 2, dtype=torch.float32)
            w2_channel_scores = torch.ones(config.num_experts, config.hidden_size, dtype=torch.float32)
            owq_cache[layer_idx] = LayerMetricBundle(
                routing_counts=routing_counts,
                w1_pair_scores=w1_pair_scores,
                w2_channel_scores=w2_channel_scores,
            )
            continue
        
        # Compute per-channel weight magnitude
        # W1: shape (num_experts, 2*hidden_size, hidden_size) -> score per pair
        # W2: shape (num_experts, hidden_size, 2*hidden_size) -> score per channel
        
        w1_magnitude = torch.abs(gate_up_proj).mean(dim=(0, 2))  # (2*hidden_size,)
        w2_magnitude = torch.abs(down_proj).mean(dim=(0, 2))     # (hidden_size,)
        
        # Normalize to [0, 1]
        w1_mag_norm = (w1_magnitude - w1_magnitude.min()) / (w1_magnitude.max() - w1_magnitude.min() + EPS)
        w2_mag_norm = (w2_magnitude - w2_magnitude.min()) / (w2_magnitude.max() - w2_magnitude.min() + EPS)
        
        # W1 is paired (gate_up has 2*hidden_size outputs, but we score hidden_size pairs)
        w1_pair_scores = w1_mag_norm.view(2, -1).mean(dim=0).unsqueeze(0).expand(config.num_experts, -1).clone()
        w2_channel_scores = w2_mag_norm.unsqueeze(0).expand(config.num_experts, -1).clone()
        
        routing_counts = torch.ones(config.num_experts, dtype=torch.int64)
        owq_cache[layer_idx] = LayerMetricBundle(
            routing_counts=routing_counts,
            w1_pair_scores=w1_pair_scores.to(torch.float32),
            w2_channel_scores=w2_channel_scores.to(torch.float32),
        )
    
    return owq_cache


def stack_mask_dict(mask_dict: dict[int, torch.Tensor], num_experts: int) -> torch.Tensor:
    """Stack per-expert masks into a single tensor."""
    stacked = []
    for expert_idx in range(num_experts):
        if expert_idx in mask_dict:
            stacked.append(mask_dict[expert_idx].to(torch.float32))
        else:
            stacked.append(torch.zeros_like(list(mask_dict.values())[0], dtype=torch.float32))
    return torch.stack(stacked, dim=0)


def count_selected(masks: dict[int, dict[int, torch.Tensor]]) -> int:
    """Count total selected channels across all layers and experts."""
    total = 0
    for layer_masks in masks.values():
        for expert_mask in layer_masks.values():
            total += int(expert_mask.sum().item())
    return total


def snap_count_to_granularity(count: int, total: int, granularity: int) -> int:
    """Snap count to nearest multiple of granularity."""
    if count == 0:
        return 0
    snapped = (count + granularity // 2) // granularity * granularity
    return min(snapped, total)


def align_masks_to_kernel_constraints(
    w1_masks: dict[int, dict[int, torch.Tensor]],
    w2_masks: dict[int, dict[int, torch.Tensor]],
    score_cache: dict[int, LayerMetricBundle],
    config: Any,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    """Align masks to TRT-LLM kernel granularity constraints."""
    aligned_w1 = {}
    aligned_w2 = {}
    changed_w1 = 0
    changed_w2 = 0

    for layer_idx in range(config.num_hidden_layers):
        aligned_w1[layer_idx] = {}
        aligned_w2[layer_idx] = {}
        bundle = score_cache[layer_idx]
        for expert_idx in range(config.num_experts):
            w1_mask = w1_masks[layer_idx][expert_idx].to(torch.bool).cpu()
            w2_mask = w2_masks[layer_idx][expert_idx].to(torch.bool).cpu()

            target_w1 = snap_count_to_granularity(
                int(w1_mask.sum().item()), w1_mask.numel(), W1_PAIR_GRANULARITY
            )
            target_w2 = snap_count_to_granularity(
                int(w2_mask.sum().item()), w2_mask.numel(), W2_CHANNEL_GRANULARITY
            )

            aligned_w1[layer_idx][expert_idx] = topk_mask_from_scores(
                bundle.w1_pair_scores[expert_idx], target_w1
            ).cpu()
            aligned_w2[layer_idx][expert_idx] = topk_mask_from_scores(
                bundle.w2_channel_scores[expert_idx], target_w2
            ).cpu()
            if target_w1 != int(w1_mask.sum().item()):
                changed_w1 += 1
            if target_w2 != int(w2_mask.sum().item()):
                changed_w2 += 1

    return aligned_w1, aligned_w2, {
        "w1_pair_granularity": W1_PAIR_GRANULARITY,
        "w2_channel_granularity": W2_CHANNEL_GRANULARITY,
        "w1_masks_resnapped": changed_w1,
        "w2_masks_resnapped": changed_w2,
    }


def build_budget_masks(
    model_id: str,
    config: Any,
    budget_config: BudgetConfig,
    weight_map: dict[str, str],
    snapshot_dir: Path,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    """Build FP8 channel masks with OWQ metric."""
    calibration, _, _ = load_cache(CALIBRATION_CACHE_PATH, model_id)
    if calibration is None:
        raise FileNotFoundError(f"Calibration cache missing: {CALIBRATION_CACHE_PATH}")

    # Load router-affinity metric
    router_affinity_cache, _, _, _, metric_meta = load_metric_cache(METRIC_CACHE_PATH, model_id)
    if router_affinity_cache is None:
        raise FileNotFoundError(f"Metric cache missing: {METRIC_CACHE_PATH}")
    
    # Compute weight-magnitude factors
    weight_mag_cache = compute_weight_magnitude_factors(model_id, config, weight_map, snapshot_dir)
    
    # Combine: OWQ = router_affinity * weight_magnitude
    combined_cache: dict[int, LayerMetricBundle] = {}
    for layer_idx in range(config.num_hidden_layers):
        ra_bundle = router_affinity_cache[layer_idx]
        wm_bundle = weight_mag_cache[layer_idx]
        
        # Multiply scores element-wise
        combined_w1 = ra_bundle.w1_pair_scores * (1.0 + wm_bundle.w1_pair_scores)
        combined_w2 = ra_bundle.w2_channel_scores * (1.0 + wm_bundle.w2_channel_scores)
        
        combined_cache[layer_idx] = LayerMetricBundle(
            routing_counts=ra_bundle.routing_counts,
            w1_pair_scores=combined_w1.to(torch.float32),
            w2_channel_scores=combined_w2.to(torch.float32),
        )
    
    score_cache = combined_cache

    # Build joint W1/W2 base masks
    joint_w1_masks, joint_w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )

    # Boost joint mask members
    prioritized_cache: dict[int, LayerMetricBundle] = {}
    for layer_idx in range(config.num_hidden_layers):
        bundle = score_cache[layer_idx]
        w1_bonus = (
            stack_mask_dict(joint_w1_masks[layer_idx], config.num_experts).to(torch.float32)
            * JOINT_BASE_BONUS
        )
        w2_bonus = (
            stack_mask_dict(joint_w2_masks[layer_idx], config.num_experts).to(torch.float32)
            * JOINT_BASE_BONUS
        )
        prioritized_cache[layer_idx] = LayerMetricBundle(
            routing_counts=bundle.routing_counts.detach().cpu().clone(),
            w1_pair_scores=bundle.w1_pair_scores.detach().cpu().to(torch.float32) + w1_bonus,
            w2_channel_scores=bundle.w2_channel_scores.detach().cpu().to(torch.float32) + w2_bonus,
        )

    # Apply global top-k with fixed 1:4 W1:W2 ratio
    mixed_w1_masks, mixed_w2_masks, mixed_meta = build_global_fraction_masks(
        prioritized_cache,
        config,
        budget_config.w1_fraction,
        budget_config.w2_fraction,
        budget_source=f"owq_metric_{budget_config.label}",
    )
    mixed_w1_masks, mixed_w2_masks, alignment_meta = align_masks_to_kernel_constraints(
        mixed_w1_masks, mixed_w2_masks, prioritized_cache, config
    )

    metadata = {
        **joint_meta,
        **mixed_meta,
        **alignment_meta,
        "budget_label": budget_config.label,
        "total_fp8_fraction_target": budget_config.total_fp8_fraction,
        "w1_fp8_fraction_fixed": budget_config.w1_fraction,
        "w2_fp8_fraction_fixed": budget_config.w2_fraction,
        "w1_selected": count_selected(mixed_w1_masks),
        "w2_selected": count_selected(mixed_w2_masks),
        "score_source": "owq_weight_magnitude_weighted_metric",
        "calibration_cache_path": str(CALIBRATION_CACHE_PATH),
        "metric_cache_path": str(METRIC_CACHE_PATH),
    }

    return mixed_w1_masks, mixed_w2_masks, metadata


def expand_w1_pair_mask(pair_mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pair_mask = pair_mask.to(device=weight.device, dtype=torch.bool)
    if weight.shape[0] == pair_mask.numel():
        return pair_mask
    if weight.shape[0] == pair_mask.numel() * 2:
        return torch.cat([pair_mask, pair_mask], dim=0)
    raise ValueError(
        f"Unexpected W1 weight shape {tuple(weight.shape)} for pair mask of length {pair_mask.numel()}"
    )


def mixed_exact_linear(
    input_tensor: torch.Tensor, weight: torch.Tensor, fp8_row_mask: torch.Tensor
) -> torch.Tensor:
    fp8_row_mask = fp8_row_mask.to(device=weight.device, dtype=torch.bool)
    if fp8_row_mask.numel() != weight.shape[0]:
        raise ValueError(
            f"Mask rows {fp8_row_mask.numel()} do not match weight rows {weight.shape[0]}"
        )
    if bool(fp8_row_mask.all()):
        return exact_eval.fp8_linear(input_tensor, weight)
    if not bool(fp8_row_mask.any()):
        return exact_eval.nvfp4_linear(input_tensor, weight)

    input_2d, prefix_shape = exact_eval.flatten_for_linear(input_tensor)
    out = torch.empty(
        (input_2d.shape[0], weight.shape[0]), dtype=input_2d.dtype, device=input_2d.device
    )
    fp8_rows = torch.nonzero(fp8_row_mask, as_tuple=False).flatten()
    fp4_rows = torch.nonzero(~fp8_row_mask, as_tuple=False).flatten()

    if fp4_rows.numel() % 32 != 0:
        deficit = fp4_rows.numel() % 32
        move = min(deficit, fp4_rows.numel())
        if move > 0:
            moved = fp4_rows[-move:]
            fp8_rows = torch.cat([fp8_rows, moved], dim=0)
            fp4_rows = fp4_rows[:-move]

    if fp8_rows.numel() > 0 and fp8_rows.numel() % 16 != 0:
        pad = 16 - (fp8_rows.numel() % 16)
        if pad < fp8_rows.numel():
            moved_back = fp8_rows[-pad:]
            fp4_rows = torch.cat([fp4_rows, moved_back], dim=0)
            fp8_rows = fp8_rows[:-pad]

    if fp4_rows.numel() > 0:
        w_fp4 = weight[fp4_rows]
        out_fp4 = exact_eval.nvfp4_linear(input_2d, w_fp4)
        out[:, fp4_rows] = out_fp4

    if fp8_rows.numel() > 0:
        w_fp8 = weight[fp8_rows]
        out_fp8 = exact_eval.fp8_linear(input_2d, w_fp8)
        out[:, fp8_rows] = out_fp8

    return exact_eval.restore_linear_shape(out, prefix_shape)


def mixed_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    w1_masks: dict[int, torch.Tensor],
    w2_masks: dict[int, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)

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
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]

        w1_fp8_mask = expand_w1_pair_mask(w1_masks[expert_idx], gate_up_proj[expert_idx])
        gate_up = mixed_exact_linear(current_state, gate_up_proj[expert_idx], w1_fp8_mask)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up

        w2_fp8_mask = w2_masks[expert_idx].to(device=down_proj.device, dtype=torch.bool)
        current_hidden = mixed_exact_linear(hidden, down_proj[expert_idx], w2_fp8_mask)
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = exact_eval.bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * exact_eval.bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = exact_eval.bf16_linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(
        exact_eval.bf16_linear(flat, tensors["shared_expert_gate.weight"])
    )
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_budget_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    w1_masks_all: dict[int, dict[int, torch.Tensor]],
    w2_masks_all: dict[int, dict[int, torch.Tensor]],
    layer_batch_size: int,
    label: str,
) -> float:
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
    hidden_bank = torch.empty((nsamples, seqlen, config.hidden_size), dtype=dtype, device="cpu")
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
                    hidden_states, layer_tensors["input_layernorm.weight"], config.rms_norm_eps
                )

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
                        causal_mask,
                    )
                elif layer_type == "linear_attention":
                    attn_tensors = {
                        k.replace("self_attn.", ""): v
                        for k, v in layer_tensors.items()
                        if k.startswith("self_attn.")
                    }
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_tensors, config, position_embeddings
                    )
                else:
                    raise ValueError(f"Unknown layer type: {layer_type}")

                hidden_states = hidden_states + residual

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["post_attention_layernorm.weight"], config.rms_norm_eps
                )

                hidden_states = mixed_moe_forward(
                    hidden_states,
                    layer_tensors,
                    config,
                    w1_masks_all[layer_idx],
                    w2_masks_all[layer_idx],
                    layer_idx,
                )
                hidden_states = hidden_states + residual
                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states

            release_tensors(layer_tensors)

        hidden_states = hidden_bank.to(device)
        hidden_states = rms_norm_qwen3_next(
            hidden_states, final_norm_w, config.rms_norm_eps
        )
        logits = F.linear(hidden_states, lm_head_w)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = eval_chunks[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1),
        )
        nlls = losses.view(nsamples, seqlen).sum(dim=1)

    ppl = torch.exp(nlls.mean()).item()
    return ppl


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"[Iter19 OWQ Metric] Starting evaluation")
    print(f"  Model: {args.model_id}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Samples: {args.nsamples}")
    print(f"  Seqlen: {args.seqlen}")

    # Load model config
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)

    # Load tokenizer and evaluation data
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, _, _ = exact_eval.load_eval_data(tokenizer, args.seqlen)

    results = {}
    start_time = time.time()

    for budget_config in ALL_BUDGET_CONFIGS:
        print(f"\n[{budget_config.label}] Building masks with OWQ metric...")
        try:
            w1_masks, w2_masks, metadata = build_budget_masks(
                args.model_id, config, budget_config, weight_map, snapshot_dir
            )
            print(f"  W1 selected: {metadata['w1_selected']}")
            print(f"  W2 selected: {metadata['w2_selected']}")
        except Exception as e:
            print(f"  ERROR building masks: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Evaluating PPL...")
        try:
            ppl = evaluate_budget_ppl(
                eval_ids,
                args.nsamples,
                args.seqlen,
                config,
                weight_map,
                snapshot_dir,
                device,
                dtype,
                w1_masks,
                w2_masks,
                args.layer_batch_size,
                budget_config.label,
            )
            print(f"  PPL: {ppl:.4f}")
            results[budget_config.label] = {
                "ppl": ppl,
                "metadata": metadata,
            }
        except Exception as e:
            print(f"  ERROR evaluating PPL: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n[Iter19 OWQ Metric] Completed in {elapsed:.1f}s")

    # Save results
    output_data = {
        "experiment": "exact_explore_iter19_owq_metric",
        "model_id": args.model_id,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

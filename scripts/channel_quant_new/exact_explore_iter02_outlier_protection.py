#!/usr/bin/env python3
"""Exact-path Exploration: Outlier Protection for iter02.

IMPROVEMENT OVER ITER02:
  iter02 allocates FP8 based purely on sensitivity scores.
  This variant PROTECTS OUTLIER CHANNELS with FP8 regardless of sensitivity.
  
  Hypothesis: Some channels have extreme weight values that are critical for accuracy.
  These outliers should be protected with FP8 even if sensitivity score is low.
  
  Approach:
  1. Identify outlier channels: channels with |w| > p99(|w|) per expert
  2. Force outliers to FP8 (protected)
  3. Allocate remaining budget to top-k non-outlier channels
  4. Expected gain: 0.001-0.003 PPL

RATIONALE:
  - Outlier values have disproportionate impact on output
  - Quantizing outliers to FP4 can cause large errors
  - Protecting outliers is a simple, principled approach
  - No new data needed, uses existing weights

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter02_outlier_protection.py --nsamples 4
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
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter02_outlier_protection.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter10_novel_perchannel_metric_cache.pt")
JOINT_BASE_BONUS = 1_000_000.0
W1_PAIR_GRANULARITY = 16
W2_CHANNEL_GRANULARITY = 32
OUTLIER_PERCENTILE = 99  # Protect channels with |w| > p99(|w|)


@dataclass(frozen=True)
class BudgetConfig:
    """FP8 budget fractions for W1 and W2."""
    label: str
    w1_fp8_fraction: float
    w2_fp8_fraction: float

    @property
    def description(self) -> str:
        return f"W1={self.w1_fp8_fraction:.0%} FP8, W2={self.w2_fp8_fraction:.0%} FP8 (with outlier protection)"


ALL_BUDGET_CONFIGS = [
    BudgetConfig("budget_10pct",       0.02, 0.08),
    BudgetConfig("budget_20pct",       0.04, 0.16),
    BudgetConfig("budget_30pct",       0.06, 0.24),
    BudgetConfig("budget_40pct",       0.08, 0.32),
    BudgetConfig("budget_50pct",       0.10, 0.40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def identify_outlier_channels(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    config: Any,
    percentile: float = 99.0,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Identify outlier channels per expert.
    
    Returns:
      w1_outlier_masks: dict[expert_idx] -> bool mask of outlier pairs
      w2_outlier_masks: dict[expert_idx] -> bool mask of outlier channels
    """
    w1_outlier_masks: dict[int, torch.Tensor] = {}
    w2_outlier_masks: dict[int, torch.Tensor] = {}
    
    for expert_idx in range(config.num_experts):
        # W1 (gate_up_proj): [2*inter_size, hidden_size]
        w1 = gate_up_proj[expert_idx].float().abs()
        w1_threshold = torch.quantile(w1.flatten(), percentile / 100.0)
        w1_outliers = (w1.max(dim=0).values > w1_threshold).to(torch.bool)
        w1_outlier_masks[expert_idx] = w1_outliers
        
        # W2 (down_proj): [hidden_size, inter_size]
        w2 = down_proj[expert_idx].float().abs()
        w2_threshold = torch.quantile(w2.flatten(), percentile / 100.0)
        w2_outliers = (w2.max(dim=0).values > w2_threshold).to(torch.bool)
        w2_outlier_masks[expert_idx] = w2_outliers
    
    return w1_outlier_masks, w2_outlier_masks


def stack_mask_dict(mask_dict: dict[int, torch.Tensor], num_experts: int) -> torch.Tensor:
    return torch.stack(
        [mask_dict[expert_idx].to(torch.bool).cpu() for expert_idx in range(num_experts)], dim=0
    )


def count_selected(mask_dicts: dict[int, dict[int, torch.Tensor]]) -> int:
    return sum(
        int(mask.sum().item()) for layer in mask_dicts.values() for mask in layer.values()
    )


def snap_count_to_granularity(selected: int, total: int, granularity: int) -> int:
    if selected <= 0:
        return 0
    if selected >= total:
        return total
    lower = (selected // granularity) * granularity
    upper = min(total, ((selected + granularity - 1) // granularity) * granularity)
    if lower == 0:
        return upper
    if upper == total:
        return lower
    if (selected - lower) <= (upper - selected):
        return lower
    return upper


def align_masks_to_kernel_constraints(
    w1_masks: dict[int, dict[int, torch.Tensor]],
    w2_masks: dict[int, dict[int, torch.Tensor]],
    score_cache: dict[int, LayerMetricBundle],
    config: Any,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    aligned_w1: dict[int, dict[int, torch.Tensor]] = {}
    aligned_w2: dict[int, dict[int, torch.Tensor]] = {}
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


def build_budget_masks_with_outlier_protection(
    model_id: str,
    config: Any,
    budget_config: BudgetConfig,
    store: Any,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    """Build FP8 channel masks with outlier protection."""
    calibration, _, _ = load_cache(CALIBRATION_CACHE_PATH, model_id)
    if calibration is None:
        raise FileNotFoundError(f"Calibration cache missing: {CALIBRATION_CACHE_PATH}")

    router_affinity_cache, _, _, _, metric_meta = load_metric_cache(METRIC_CACHE_PATH, model_id)
    score_cache = router_affinity_cache
    if score_cache is None:
        score_cache = calibration.activation_cache
        metric_meta = {"fallback": "activation_kurtosis"}

    # Build joint W1/W2 base masks
    joint_w1_masks, joint_w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )

    # Identify outlier channels
    outlier_w1_masks: dict[int, dict[int, torch.Tensor]] = {}
    outlier_w2_masks: dict[int, dict[int, torch.Tensor]] = {}
    total_w1_outliers = 0
    total_w2_outliers = 0
    
    for layer_idx in range(config.num_hidden_layers):
        layer_type = config.layer_types[layer_idx]
        raw = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw, config.device, config.dtype)
        del raw
        
        moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        gate_up_proj = moe_tensors["experts.gate_up_proj"]
        down_proj = moe_tensors["experts.down_proj"]
        
        w1_outliers, w2_outliers = identify_outlier_channels(gate_up_proj, down_proj, config, OUTLIER_PERCENTILE)
        outlier_w1_masks[layer_idx] = w1_outliers
        outlier_w2_masks[layer_idx] = w2_outliers
        
        total_w1_outliers += sum(int(m.sum().item()) for m in w1_outliers.values())
        total_w2_outliers += sum(int(m.sum().item()) for m in w2_outliers.values())
        
        release_tensors(tensors)

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
        
        # Also boost outliers
        w1_outlier_bonus = (
            stack_mask_dict(outlier_w1_masks[layer_idx], config.num_experts).to(torch.float32)
            * JOINT_BASE_BONUS
        )
        w2_outlier_bonus = (
            stack_mask_dict(outlier_w2_masks[layer_idx], config.num_experts).to(torch.float32)
            * JOINT_BASE_BONUS
        )
        
        prioritized_cache[layer_idx] = LayerMetricBundle(
            routing_counts=bundle.routing_counts.detach().cpu().clone(),
            w1_pair_scores=bundle.w1_pair_scores.detach().cpu().to(torch.float32) + w1_bonus + w1_outlier_bonus,
            w2_channel_scores=bundle.w2_channel_scores.detach().cpu().to(torch.float32) + w2_bonus + w2_outlier_bonus,
        )

    # Apply global top-k
    mixed_w1_masks, mixed_w2_masks, mixed_meta = build_global_fraction_masks(
        prioritized_cache,
        config,
        budget_config.w1_fp8_fraction,
        budget_config.w2_fp8_fraction,
        budget_source=f"outlier_protection_{budget_config.label}",
    )
    mixed_w1_masks, mixed_w2_masks, alignment_meta = align_masks_to_kernel_constraints(
        mixed_w1_masks, mixed_w2_masks, prioritized_cache, config
    )

    metadata = {
        **joint_meta,
        **mixed_meta,
        **alignment_meta,
        "budget_label": budget_config.label,
        "w1_fp8_fraction": budget_config.w1_fp8_fraction,
        "w2_fp8_fraction": budget_config.w2_fp8_fraction,
        "w1_selected": count_selected(mixed_w1_masks),
        "w2_selected": count_selected(mixed_w2_masks),
        "outlier_protection_percentile": OUTLIER_PERCENTILE,
        "total_w1_outliers_protected": total_w1_outliers,
        "total_w2_outliers_protected": total_w2_outliers,
        "score_source": "router_affinity_weighted_qerror_with_outlier_protection",
        "metric_cache_path": str(METRIC_CACHE_PATH),
        "calibration_cache_path": str(CALIBRATION_CACHE_PATH),
    }
    if isinstance(metric_meta, dict):
        metadata["metric_cache_metadata"] = metric_meta

    return mixed_w1_masks, mixed_w2_masks, metadata


# [Rest of the code is identical to iter02: expand_w1_pair_mask, mixed_exact_linear, mixed_moe_forward, evaluate_budget_ppl, main]

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
                        exact_eval.build_causal_mask(seqlen, device),
                        mode="bf16",
                        quantized=False,
                    )
                else:
                    attn_tensors = {
                        k.replace("linear_attn.", ""): v
                        for k, v in layer_tensors.items()
                        if k.startswith("linear_attn.")
                    }
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_tensors, config, "bf16", "moe_only"
                    )
                hidden_states = residual + hidden_states

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states,
                    layer_tensors["post_attention_layernorm.weight"],
                    config.rms_norm_eps,
                )
                moe_tensors = {
                    k.replace("mlp.", "", 1): v
                    for k, v in layer_tensors.items()
                    if k.startswith("mlp.")
                }
                moe_out = mixed_moe_forward(
                    hidden_states,
                    moe_tensors,
                    config,
                    w1_masks_all[layer_idx],
                    w2_masks_all[layer_idx],
                    layer_idx,
                )
                hidden_states = residual + moe_out
                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            print(f"  [{label}] layer {layer_idx + 1}/{config.num_hidden_layers}", flush=True)

        for sample_idx in range(nsamples):
            chunk = eval_chunks[sample_idx : sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, config.rms_norm_eps)
            logits = F.linear(hidden_states.float(), lm_head_w.float())
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1)
            )
            nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


def main() -> None:
    args = parse_args()
    exact_eval.ensure_runtime_available()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, full_nsamples, seqlen = exact_eval.load_eval_data(tokenizer, args.seqlen)
    nsamples = min(args.nsamples, full_nsamples)
    print(f"Eval: {nsamples} chunks of {seqlen} tokens ({nsamples * seqlen} total)", flush=True)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    config.device = device
    config.dtype = dtype
    
    store = exact_eval.WeightStore(exact_eval.MODEL_ID, snapshot_dir, weight_map)

    results: dict[str, dict[str, Any]] = {}

    for budget_config in ALL_BUDGET_CONFIGS:
        print(f"\n=== {budget_config.label} ({budget_config.description}) ===", flush=True)
        start_time = time.time()

        w1_masks, w2_masks, mask_meta = build_budget_masks_with_outlier_protection(
            args.model_id, config, budget_config, store
        )

        ppl = evaluate_budget_ppl(
            eval_ids,
            nsamples,
            seqlen,
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
        elapsed = time.time() - start_time
        results[budget_config.label] = {
            "w1_fp8_fraction": budget_config.w1_fp8_fraction,
            "w2_fp8_fraction": budget_config.w2_fp8_fraction,
            "quant_scope": "moe_only",
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
            **mask_meta,
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "eval_tokens": nsamples * seqlen,
            "dtype": args.dtype,
            "layer_batch_size": args.layer_batch_size,
            "runtime": "docker-only TRT-LLM fused wrappers",
            "experiment": "outlier_protection_budget_sweep",
            "purpose": "Protect outlier channels with FP8 regardless of sensitivity score",
            "sensitivity_metric": "router_affinity_weighted_qerror",
            "assignment_strategy": "joint_w1w2_3tier_then_global_topk_with_outlier_protection",
            "outlier_percentile": OUTLIER_PERCENTILE,
            "nvfp4_linear": "torch.ops.auto_deploy.torch_quant_nvfp4_linear",
            "fp8_linear": "torch.ops.auto_deploy.torch_quant_fp8_linear",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved -> {args.output}", flush=True)
    print(f"\n{'Method':<24s} {'W1%':>6s} {'W2%':>6s} {'PPL':>10s}")
    print("-" * 50)
    for label, row in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(
            f"{label:<24s} {row['w1_fp8_fraction']:>6.0%} {row['w2_fp8_fraction']:>6.0%} {row['ppl']:>10.4f}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact-path Exploration: Three-Tier Precision (BF16 + FP8 + NVFP4).

Tests channel-level assignment across three precision tiers:
  - BF16:  no quantization (most sensitive channels)
  - FP8:   moderate quantization (medium channels)
  - NVFP4: aggressive quantization (cold channels)

Also tests two-tier BF16+NVFP4 (skip FP8 entirely).

Configurations:
  1. bf16_5_nvfp4_95   — 5% BF16, 95% NVFP4 (two-tier, no FP8)
  2. bf16_10_nvfp4_90  — 10% BF16, 90% NVFP4
  3. bf16_20_nvfp4_80  — 20% BF16, 80% NVFP4
  4. bf16_5_fp8_45_nvfp4_50  — 5% BF16, 45% FP8, 50% NVFP4 (three-tier)
  5. bf16_10_fp8_40_nvfp4_50 — 10% BF16, 40% FP8, 50% NVFP4
  6. bf16_10_fp8_20_nvfp4_70 — 10% BF16, 20% FP8, 70% NVFP4
  7. bf16_5_fp8_15_nvfp4_80  — 5% BF16, 15% FP8, 80% NVFP4
  + baselines: uniform_bf16, uniform_fp8, uniform_nvfp4

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter12_three_tier.py --nsamples 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter12_three_tier.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter10_novel_perchannel_metric_cache.pt")

W1_CHANNELS = 1024
W2_CHANNELS = 2048
W1_GRANULARITY = 16
W2_GRANULARITY = 32


@dataclass(frozen=True)
class ThreeTierConfig:
    label: str
    bf16_fraction: float
    fp8_fraction: float
    nvfp4_fraction: float

    @property
    def description(self) -> str:
        parts = []
        if self.bf16_fraction > 0:
            parts.append(f"BF16={self.bf16_fraction:.0%}")
        if self.fp8_fraction > 0:
            parts.append(f"FP8={self.fp8_fraction:.0%}")
        if self.nvfp4_fraction > 0:
            parts.append(f"NVFP4={self.nvfp4_fraction:.0%}")
        return ", ".join(parts)


ALL_CONFIGS = [
    ThreeTierConfig("bf16_2_nvfp4_98",             0.02, 0.00, 0.98),
    ThreeTierConfig("bf16_5_nvfp4_95",             0.05, 0.00, 0.95),
    ThreeTierConfig("bf16_10_nvfp4_90",            0.10, 0.00, 0.90),
    ThreeTierConfig("bf16_15_nvfp4_85",            0.15, 0.00, 0.85),
    ThreeTierConfig("bf16_20_nvfp4_80",            0.20, 0.00, 0.80),
    ThreeTierConfig("bf16_30_nvfp4_70",            0.30, 0.00, 0.70),
    ThreeTierConfig("bf16_50_nvfp4_50",            0.50, 0.00, 0.50),
    ThreeTierConfig("bf16_5_fp8_45_nvfp4_50",      0.05, 0.45, 0.50),
    ThreeTierConfig("bf16_10_fp8_40_nvfp4_50",     0.10, 0.40, 0.50),
    ThreeTierConfig("bf16_10_fp8_20_nvfp4_70",     0.10, 0.20, 0.70),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def compute_channel_sensitivity(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute per-output-channel sensitivity score.
    
    Uses weight magnitude variance as a proxy — channels with high magnitude
    variance are more sensitive to quantization (outlier-prone).
    Falls back to absmax if metric cache unavailable.
    """
    return weight.float().abs().mean(dim=-1)


def build_three_tier_masks(
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    config: ThreeTierConfig,
    num_experts: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build per-expert channel tier assignments.
    
    Returns:
        w1_tiers: dict[expert_idx] -> Tensor of shape [W1_CHANNELS] with values 0=nvfp4, 1=fp8, 2=bf16
        w2_tiers: dict[expert_idx] -> Tensor of shape [W2_CHANNELS] with values 0=nvfp4, 1=fp8, 2=bf16
    """
    w1_tiers: dict[int, torch.Tensor] = {}
    w2_tiers: dict[int, torch.Tensor] = {}

    for expert_idx in range(num_experts):
        for proj, weights, n_channels, granularity, tiers_out in [
            ("w1", gate_up_weights[expert_idx], W1_CHANNELS, W1_GRANULARITY, w1_tiers),
            ("w2", down_weights[expert_idx], W2_CHANNELS, W2_GRANULARITY, w2_tiers),
        ]:
            sensitivity = compute_channel_sensitivity(weights)
            n_bf16_raw = int(round(config.bf16_fraction * n_channels))
            n_fp8_raw = int(round(config.fp8_fraction * n_channels))
            n_bf16 = _snap(n_bf16_raw, n_channels, granularity) if n_bf16_raw > 0 else 0
            n_fp8 = _snap(n_fp8_raw, n_channels - n_bf16, granularity) if n_fp8_raw > 0 else 0
            n_nvfp4 = n_channels - n_bf16 - n_fp8
            if n_nvfp4 < 0:
                n_fp8 = n_channels - n_bf16
                n_nvfp4 = 0

            tiers = torch.zeros(n_channels, dtype=torch.long)
            _, sorted_idx = sensitivity.sort(descending=True)

            if n_bf16 > 0:
                tiers[sorted_idx[:n_bf16]] = 2
            if n_fp8 > 0:
                tiers[sorted_idx[n_bf16 : n_bf16 + n_fp8]] = 1

            tiers_out[expert_idx] = tiers

    return w1_tiers, w2_tiers


def _snap(n: int, total: int, granularity: int) -> int:
    if n <= 0:
        return 0
    if n >= total:
        return total
    return max(granularity, ((n + granularity // 2) // granularity) * granularity)


def three_tier_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    tiers: torch.Tensor,
) -> torch.Tensor:
    n_out = weight.shape[0]
    device = input_tensor.device
    bf16_mask = tiers == 2
    fp8_mask = tiers == 1
    nvfp4_mask = tiers == 0

    output = torch.zeros(
        input_tensor.shape[0], n_out, dtype=input_tensor.dtype, device=device
    )

    if bf16_mask.any():
        idx = bf16_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        out_sub = exact_eval.bf16_linear(input_tensor, w_sub)
        output.index_copy_(1, idx, out_sub)

    if fp8_mask.any():
        idx = fp8_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        n_rows = w_sub.shape[0]
        pad_n = (16 - n_rows % 16) % 16
        if pad_n:
            w_sub = F.pad(w_sub, (0, 0, 0, pad_n))
        out_sub = exact_eval.fp8_linear(input_tensor, w_sub)
        if pad_n:
            out_sub = out_sub[:, :n_rows]
        output.index_copy_(1, idx, out_sub)

    if nvfp4_mask.any():
        idx = nvfp4_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        n_rows = w_sub.shape[0]
        pad_n = (32 - n_rows % 32) % 32
        if pad_n:
            w_sub = F.pad(w_sub, (0, 0, 0, pad_n))
        out_sub = exact_eval.nvfp4_linear(input_tensor, w_sub)
        if pad_n:
            out_sub = out_sub[:, :n_rows]
        output.index_copy_(1, idx, out_sub)

    return output


def three_tier_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    w1_tiers: dict[int, torch.Tensor],
    w2_tiers: dict[int, torch.Tensor],
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
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(
        selected_experts.reshape(-1), minlength=config.num_experts
    )

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]

        w1_tier = w1_tiers.get(expert_idx)
        w2_tier = w2_tiers.get(expert_idx)

        if w1_tier is not None and (w1_tier > 0).any():
            gate_up = three_tier_linear(current_state, gate_up_proj[expert_idx], w1_tier)
        else:
            gate_up = exact_eval.nvfp4_linear(current_state, gate_up_proj[expert_idx])

        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up

        if w2_tier is not None and (w2_tier > 0).any():
            current_hidden = three_tier_linear(hidden, down_proj[expert_idx], w2_tier)
        else:
            current_hidden = exact_eval.nvfp4_linear(hidden, down_proj[expert_idx])

        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = exact_eval.bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * exact_eval.bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = exact_eval.bf16_linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(exact_eval.bf16_linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_three_tier_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    model_config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    tier_config: ThreeTierConfig,
    layer_batch_size: int,
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
    hidden_bank = torch.empty(
        (nsamples, seqlen, model_config.hidden_size), dtype=dtype, device="cpu"
    )
    for i in range(nsamples):
        hidden_bank[i].copy_(F.embedding(eval_chunks[i : i + 1].to(device), embed_w).squeeze(0).cpu())

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=model_config, device=device)
    rotary_input = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(rotary_input, position_ids)
    del rotary_input

    nlls: list[torch.Tensor] = []

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_tensors = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            moe_t = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
            w1_tiers, w2_tiers = build_three_tier_masks(
                moe_t.get("experts.gate_up_proj", torch.zeros(1)),
                moe_t.get("experts.down_proj", torch.zeros(1)),
                tier_config,
                model_config.num_experts,
            )

            for sample_idx in range(nsamples):
                hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["input_layernorm.weight"], model_config.rms_norm_eps
                )
                if layer_type == "full_attention":
                    attn_t = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = exact_eval.full_attention_forward_exact(
                        hidden_states, attn_t, model_config, position_embeddings,
                        exact_eval.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_t = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_t, model_config, "bf16", "moe_only"
                    )
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["post_attention_layernorm.weight"], model_config.rms_norm_eps
                )
                moe_out = three_tier_moe_forward(
                    hidden_states, moe_t, model_config, w1_tiers, w2_tiers,
                )
                hidden_states = residual + moe_out
                hidden_bank[sample_idx : sample_idx + 1].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            if (layer_idx + 1) % 10 == 0:
                print(f"  [{tier_config.label}] layer {layer_idx + 1}/{model_config.num_hidden_layers}", flush=True)

        for sample_idx in range(nsamples):
            torch.cuda.empty_cache()
            chunk = eval_chunks[sample_idx : sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, model_config.rms_norm_eps)
            logits = F.linear(hidden_states, lm_head_w)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)
            del logits, shift_logits, hidden_states

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
    print(f"Eval: {nsamples} chunks of {seqlen} tokens", flush=True)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    model_config = build_text_config(root_config)

    baselines = [
        exact_eval.EvalConfig("uniform_bf16", "bf16", "moe_only"),
        exact_eval.EvalConfig("uniform_fp8", "fp8", "moe_only"),
        exact_eval.EvalConfig("uniform_nvfp4", "nvfp4", "moe_only"),
    ]

    results: dict[str, dict[str, Any]] = {}

    for run_config in baselines:
        print(f"\n=== {run_config.label} ===", flush=True)
        start_time = time.time()
        ppl = exact_eval.evaluate_ppl(
            eval_ids, nsamples, seqlen, model_config, weight_map, snapshot_dir,
            device, dtype, run_config, args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[run_config.label] = {
            "tiers": run_config.mode,
            "bf16_pct": 1.0 if run_config.mode == "bf16" else 0.0,
            "fp8_pct": 1.0 if run_config.mode == "fp8" else 0.0,
            "nvfp4_pct": 1.0 if run_config.mode == "nvfp4" else 0.0,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    for tier_config in ALL_CONFIGS:
        print(f"\n=== {tier_config.label} ({tier_config.description}) ===", flush=True)
        start_time = time.time()
        ppl = evaluate_three_tier_ppl(
            eval_ids, nsamples, seqlen, model_config, weight_map, snapshot_dir,
            device, dtype, tier_config, args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[tier_config.label] = {
            "tiers": tier_config.description,
            "bf16_pct": tier_config.bf16_fraction,
            "fp8_pct": tier_config.fp8_fraction,
            "nvfp4_pct": tier_config.nvfp4_fraction,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "eval_tokens": nsamples * seqlen,
            "experiment": "three_tier_precision",
            "scope": "moe_only",
            "sensitivity_metric": "per-channel weight absmax (simple proxy)",
            "tiers": "BF16 (tier 2) > FP8 (tier 1) > NVFP4 (tier 0)",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved -> {args.output}", flush=True)
    print(f"\n{'Label':<35s} {'BF16%':>6s} {'FP8%':>6s} {'FP4%':>6s} {'PPL':>10s}")
    print("-" * 70)
    for label, row in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(
            f"{label:<35s} {row['bf16_pct']:>5.0%} {row['fp8_pct']:>5.0%} "
            f"{row['nvfp4_pct']:>5.0%} {row['ppl']:>10.4f}"
        )


if __name__ == "__main__":
    main()

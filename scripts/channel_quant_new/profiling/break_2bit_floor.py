#!/usr/bin/env python3
"""Four approaches to break the 2-bit +0.6 PPL floor.

All approaches use the same 2-bit codebook {-4, 0, 3, 6} as base.
1) AdaRound: optimize rounding direction per-element using calibration data
2) GPTQ-style: column-wise error compensation after 2-bit quantization
3) Low-rank correction: add rank-R matrix to correct 2-bit error
4) Sparse corrections: store 1-bit fix for the K% worst errors
"""
import sys
import time
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new/profiling")

import tensorrt_llm._torch.auto_deploy.custom_ops
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)
from real_eval_pipeline import load_eval_data
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
BLOCK_SIZE = 16
CB_2BIT = torch.tensor([-4, 0, 3, 6], dtype=torch.float32)
CB_2BIT_SORTED = torch.tensor([-4, 0, 3, 6], dtype=torch.float32)


def nvfp4_block_decompose(weight_bf16, device='cuda'):
    """Decompose weight into NVFP4 block structure: returns (normalized, block_scales, global_scale, orig_shape, pad)."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape
    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale
    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))
    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=2)
    block_scales = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
    safe_scales = block_scales.clone()
    safe_scales[safe_scales == 0] = 1.0
    normalized = blocks / safe_scales.unsqueeze(2)
    return normalized, safe_scales, global_scale, (rows, cols), pad


def reconstruct_weight(snapped_normalized, safe_scales, global_scale, orig_shape, pad, dtype):
    result = snapped_normalized * safe_scales.unsqueeze(2) / global_scale
    rows, cols = orig_shape
    result = result.reshape(rows, -1)[:, :cols]
    return result.to(dtype)


def snap_nearest_2bit(normalized, device='cuda'):
    cb = CB_2BIT_SORTED.to(device)
    return cb[(normalized.unsqueeze(-1) - cb).abs().argmin(dim=-1)]


# ── Approach 1: AdaRound ──

def adaround_2bit(weight_bf16, calibration_input, device='cuda'):
    """Optimize rounding direction per-element: for ambiguous codes, pick up/down to minimize output MSE."""
    normalized, safe_scales, global_scale, orig_shape, pad = nvfp4_block_decompose(weight_bf16, device)
    cb = CB_2BIT_SORTED.to(device)

    diff = normalized.unsqueeze(-1) - cb
    abs_diff = diff.abs()
    sorted_dist, sorted_idx = abs_diff.sort(dim=-1)

    nearest = cb[sorted_idx[..., 0]]
    second = cb[sorted_idx[..., 1]]

    ambiguity = sorted_dist[..., 1] - sorted_dist[..., 0]
    ambiguous_mask = ambiguity < 1.5

    if calibration_input is not None and ambiguous_mask.any():
        weight_nearest = reconstruct_weight(nearest, safe_scales, global_scale, orig_shape, pad, weight_bf16.dtype)
        out_nearest = F.linear(calibration_input, weight_nearest)

        trial = nearest.clone()
        trial[ambiguous_mask] = second[ambiguous_mask]
        weight_trial = reconstruct_weight(trial, safe_scales, global_scale, orig_shape, pad, weight_bf16.dtype)
        out_trial = F.linear(calibration_input, weight_trial)

        weight_orig = weight_bf16.to(device)
        out_orig = F.linear(calibration_input, weight_orig)

        mse_nearest = ((out_nearest - out_orig) ** 2).mean()
        mse_trial = ((out_trial - out_orig) ** 2).mean()

        if mse_trial < mse_nearest:
            return weight_trial
        return weight_nearest

    return reconstruct_weight(nearest, safe_scales, global_scale, orig_shape, pad, weight_bf16.dtype)


# ── Approach 2: GPTQ-style compensation ──

def gptq_2bit(weight_bf16, calibration_input, device='cuda', damp=0.01):
    """Column-wise GPTQ: quantize columns left-to-right, compensating remaining columns for each error."""
    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    if calibration_input is not None:
        X = calibration_input.float().to(device)
        if X.dim() == 3:
            X = X.reshape(-1, X.shape[-1])
        H = X.T @ X
        H /= X.shape[0]
        diag_mean = H.diag().mean()
        H += damp * diag_mean * torch.eye(cols, device=device)
        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
        except:
            H_inv = torch.linalg.inv(H + 0.1 * diag_mean * torch.eye(cols, device=device))
    else:
        H_inv = torch.eye(cols, device=device)

    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
    global_scale = fp4_global_scale(weight).to(torch.float32)

    W = weight.clone()
    Q = torch.zeros_like(W)

    for col in range(cols):
        w_col = W[:, col]
        scaled_col = w_col * global_scale

        block_idx = col // BLOCK_SIZE
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, cols)
        block_vals = (W[:, block_start:block_end] * global_scale)
        block_amax = block_vals.abs().max(dim=1).values
        b_scale = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
        b_scale[b_scale == 0] = 1.0

        norm_col = scaled_col / b_scale
        cb = CB_2BIT_SORTED.to(device)
        q_norm = cb[(norm_col.unsqueeze(-1) - cb).abs().argmin(dim=-1)]
        q_col = q_norm * b_scale / global_scale

        Q[:, col] = q_col
        err = w_col - q_col

        if col + 1 < cols:
            W[:, col+1:] += err.unsqueeze(1) * H_inv[col, col+1:].unsqueeze(0) / H_inv[col, col]

    return Q.to(weight_bf16.dtype)


# ── Approach 3: Low-rank correction ──

def lowrank_2bit(weight_bf16, rank=4, device='cuda'):
    """2-bit quantization + rank-R SVD correction of the residual."""
    normalized, safe_scales, global_scale, orig_shape, pad = nvfp4_block_decompose(weight_bf16, device)
    snapped = snap_nearest_2bit(normalized, device)
    weight_2bit = reconstruct_weight(snapped, safe_scales, global_scale, orig_shape, pad, torch.float32)

    residual = weight_bf16.float().to(device) - weight_2bit
    U, S, V = torch.svd_lowrank(residual, q=rank)
    correction = U @ torch.diag(S) @ V.T

    corrected = weight_2bit + correction
    return corrected.to(weight_bf16.dtype)


# ── Approach 4: Sparse 1-bit corrections ──

def sparse_correction_2bit(weight_bf16, correction_fraction=0.25, device='cuda'):
    """2-bit base + 1-bit correction for the worst-error elements."""
    normalized, safe_scales, global_scale, orig_shape, pad = nvfp4_block_decompose(weight_bf16, device)

    cb = CB_2BIT_SORTED.to(device)
    abs_diff = (normalized.unsqueeze(-1) - cb).abs()
    sorted_dist, sorted_idx = abs_diff.sort(dim=-1)

    nearest = cb[sorted_idx[..., 0]]
    second = cb[sorted_idx[..., 1]]

    error_mag = sorted_dist[..., 0]
    flat_err = error_mag.flatten()
    n_correct = int(correction_fraction * flat_err.numel())
    threshold = flat_err.topk(n_correct).values[-1]

    needs_correction = error_mag >= threshold

    corrected = nearest.clone()
    corrected[needs_correction] = second[needs_correction]

    return reconstruct_weight(corrected, safe_scales, global_scale, orig_shape, pad, weight_bf16.dtype)


# ── Evaluation ──

def evaluate_approach(approach_name, snap_fn, device='cuda'):
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    print(f"Evaluating: {approach_name}")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eval_ids, num_samples, seqlen = load_eval_data(tokenizer)
    eval_ids = eval_ids.to(device)

    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    root_tensors = weight_store.load_tensors([
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ])
    embed_weight = move_tensor(root_tensors["model.language_model.embed_tokens.weight"], device, dtype)
    final_norm = move_tensor(root_tensors["model.language_model.norm.weight"], device, dtype)
    lm_head_weight = move_tensor(root_tensors["lm_head.weight"], device, dtype)
    del root_tensors

    eval_chunks = eval_ids[:, :num_samples * seqlen].view(num_samples, seqlen).contiguous()
    hidden_bank = torch.empty((num_samples, seqlen, model_config.hidden_size), dtype=dtype, device="cpu")
    for i in range(num_samples):
        hidden_bank[i].copy_(
            F.embedding(eval_chunks[i:i+1].to(device), embed_weight).squeeze(0).cpu()
        )

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=model_config, device=device)
    dummy = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(dummy, position_ids)
    del dummy

    negative_log_likelihoods = []
    t0 = time.time()

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            moe = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}

            for sample_idx in range(num_samples):
                hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
                residual = hidden

                hidden = rms_norm_qwen3_next(hidden, layer_weights["input_layernorm.weight"], model_config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_kv = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if k.startswith("self_attn.")}
                    hidden = ee.full_attention_forward_exact(
                        hidden, attn_kv, model_config, position_embeddings,
                        ee.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_kv = {k.replace("linear_attn.", ""): v for k, v in layer_weights.items() if k.startswith("linear_attn.")}
                    hidden = ee.linear_attention_forward_exact(hidden, attn_kv, model_config, "bf16", "moe_only")

                hidden = residual + hidden
                residual = hidden
                hidden = rms_norm_qwen3_next(hidden, layer_weights["post_attention_layernorm.weight"], model_config.rms_norm_eps)

                flat = hidden.view(-1, model_config.hidden_size)
                router_logits = ee.bf16_linear(flat, moe["gate.weight"]).float()
                routing_probs = torch.softmax(router_logits, dim=1)
                topk_weights, topk_ids = torch.topk(routing_probs, model_config.num_experts_per_tok, dim=-1)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(dtype)

                expert_output = torch.zeros_like(flat)
                active_counts = torch.bincount(topk_ids.reshape(-1), minlength=model_config.num_experts)

                for expert_idx in torch.nonzero(active_counts > 0, as_tuple=False).flatten().tolist():
                    token_indices, route_indices = torch.where(topk_ids == expert_idx)
                    expert_input = flat[token_indices]

                    w1 = moe["experts.gate_up_proj"][expert_idx]
                    w2 = moe["experts.down_proj"][expert_idx]

                    w1_q = snap_fn(w1, expert_input, device)
                    w2_q = snap_fn(w2, None, device)

                    gate_up = ee.nvfp4_linear(expert_input, w1_q)
                    gate, up = gate_up.chunk(2, dim=-1)
                    intermediate = F.silu(gate) * up
                    expert_out = ee.nvfp4_linear(intermediate, w2_q)

                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))

                shared = ee.bf16_linear(flat, moe["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, moe["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, moe["shared_expert.down_proj.weight"])
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, moe["shared_expert_gate.weight"]))

                hidden = residual + (expert_output + shared_gate * shared).view_as(residual)
                hidden_bank[sample_idx].copy_(hidden.squeeze(0).cpu())

            release_tensors(layer_weights)
            elapsed = time.time() - t0
            print(f"  Layer {layer_idx+1}/{model_config.num_hidden_layers} ({elapsed:.0f}s)", flush=True)

        for sample_idx in range(num_samples):
            hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
            hidden = rms_norm_qwen3_next(hidden, final_norm, model_config.rms_norm_eps)
            logits = F.linear(hidden, lm_head_weight)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = eval_chunks[sample_idx:sample_idx+1, 1:].to(device).contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            negative_log_likelihoods.append(loss.item())

    final_ppl = math.exp(sum(negative_log_likelihoods) / len(negative_log_likelihoods))
    elapsed = time.time() - t0

    BF16_PPL = 6.5896
    NVFP4_PPL = 6.8431

    print(f"\n{'='*60}")
    print(f"RESULT: {approach_name}")
    print(f"  PPL: {final_ppl:.4f}")
    print(f"  BF16={BF16_PPL:.4f} | NVFP4={NVFP4_PPL:.4f}")
    if final_ppl <= NVFP4_PPL:
        print(f"  Recovery: {(NVFP4_PPL - final_ppl) / (NVFP4_PPL - BF16_PPL) * 100:.1f}%")
    else:
        print(f"  WORSE than NVFP4 by +{final_ppl - NVFP4_PPL:.4f} PPL")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    return {'approach': approach_name, 'ppl': final_ppl, 'time': elapsed}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', required=True,
                        choices=['adaround', 'gptq', 'lowrank', 'sparse25', 'sparse50', 'baseline'])
    args = parser.parse_args()

    approaches = {
        'baseline': ('2bit_baseline (nearest)', lambda w, inp, d: reconstruct_weight(
            snap_nearest_2bit(*nvfp4_block_decompose(w, d)[:1], d),
            *nvfp4_block_decompose(w, d)[1:4],
            nvfp4_block_decompose(w, d)[4], w.dtype)),
        'adaround': ('2bit_adaround', lambda w, inp, d: adaround_2bit(w, inp, d)),
        'gptq': ('2bit_gptq', lambda w, inp, d: gptq_2bit(w, inp, d)),
        'lowrank': ('2bit_lowrank_r4', lambda w, inp, d: lowrank_2bit(w, rank=4, device=d)),
        'sparse25': ('2bit_sparse25pct', lambda w, inp, d: sparse_correction_2bit(w, 0.25, d)),
        'sparse50': ('2bit_sparse50pct', lambda w, inp, d: sparse_correction_2bit(w, 0.50, d)),
    }

    name, fn = approaches[args.approach]
    result = evaluate_approach(name, fn)

    out_path = Path(f"/code/tensorrt_llm/scripts/channel_quant_new/profiling/break2bit_{args.approach}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

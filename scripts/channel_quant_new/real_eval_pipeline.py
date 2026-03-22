#!/usr/bin/env python3
"""
Real-precision evaluation pipeline for mixed FP4/FP8 MoE quantization.

Quantizes BOTH weights AND activations to match SM120 tensor core execution:
  - FP4 channels: FP4 weight x FP4 activation -> FP32 accumulate
  - FP8 channels: FP8 weight x FP8 activation -> FP32 accumulate

Quantization functions are copied VERBATIM from TRT-LLM source:
  tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py
  tensorrt_llm/quantization/utils/fp8_utils.py

This ensures bit-exact matching with TRT-LLM's CUTLASS kernels.
"""

import math
import random
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant"))

from spike1_ground_truth import (
    MODEL_ID, WeightStore, build_causal_mask, build_text_config,
    full_attention_forward, linear_attention_forward,
    load_root_config, move_tensor,
    release_tensors, rms_norm_qwen3_next,
    layer_keys, shorten_layer_tensors,
)
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
from datasets import load_dataset

HIDDEN = 2048
MOE_INTER = 512
NUM_EXPERTS = 256
TOP_K = 8

# ── NVFP4 quantization — VERBATIM from TRT-LLM torch_quant.py ──────────
# Source: tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py
# These are the EXACT functions the CUTLASS kernels use for reference validation.

e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _nvfp4_get_weights_scaling_factor(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
):
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = input.abs().amax().float() / (6.0 * 448.0)

    [n, k] = input.shape[-2:]
    assert k % block_size == 0

    input = input.reshape((*tuple(input.shape[:-2]), n, k // block_size, block_size))
    per_block_amax = input.abs().amax(dim=-1).float()
    per_block_scale = per_block_amax / 6.0
    q_per_block_scale = per_block_scale / weights_scaling_factor_2
    q_per_block_scale[per_block_scale == 0] = 1.0
    if not keep_high_precision:
        q_per_block_scale = q_per_block_scale.to(torch.float8_e4m3fn)
    return q_per_block_scale, weights_scaling_factor_2


def _cast_fp4(weight: torch.Tensor):
    device = weight.device
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
    mask_shape = list(weight.shape)
    mask = mask.expand([*mask_shape, 7])

    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ord_val = torch.searchsorted(e2m1_bounds.to(device), weight_abs, out_int32=True).to(torch.uint8)
    round_up = torch.any((weight_abs.unsqueeze(-1) == e2m1_bounds.to(device)) * mask, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord_val + round_up).to(torch.uint8)
    return fp4_val


def _quantize_nvfp4(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
):
    weights_scaling_factor, weights_scaling_factor_2 = _nvfp4_get_weights_scaling_factor(
        input, block_size, weights_scaling_factor_2
    )
    input = input.view((*tuple(input.shape[:-1]), -1, block_size))
    scaled_weight = input / (
        (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    )
    scaled_weight = scaled_weight.view((*tuple(scaled_weight.shape[:-2]), -1))
    q_weight = _cast_fp4(scaled_weight)
    packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
    return packed_weight, weights_scaling_factor, weights_scaling_factor_2


def _dequantize_nvfp4(
    quantized_t: torch.Tensor,
    scale_1: torch.Tensor,
    scale_2: torch.Tensor,
    orig_shape: tuple,
    orig_dtype: torch.dtype,
) -> torch.Tensor:
    device = quantized_t.device
    N, K = orig_shape
    num_blocks = N * (K // 16)
    s1 = scale_1.reshape(-1)[:num_blocks]

    high = (quantized_t >> 4) & 0x0F
    low = quantized_t & 0x0F
    idx = torch.empty(N, (K // 2) * 2, dtype=torch.long, device=device)
    idx[..., 0::2] = low.long()
    idx[..., 1::2] = high.long()

    vals = e2m1_values.to(device)[idx]

    scale_real = (s1.to(torch.float32) * scale_2.to(torch.float32)).view(N, K // 16, 1)
    vals = vals.view(N, K // 16, 16) * scale_real
    return vals.view(N, K).to(orig_dtype)


def nvfp4_fake_quantize(w: torch.Tensor) -> torch.Tensor:
    """NVFP4 quantize then dequantize. Produces BF16 output with NVFP4 quantization error.
    Uses TRT-LLM's exact quantization math."""
    orig_shape = w.shape
    orig_dtype = w.dtype
    w2d = w.reshape(-1, w.shape[-1])
    N, K = w2d.shape
    if K % 16 != 0:
        pad = 16 - K % 16
        w2d = F.pad(w2d, (0, pad))
        K = w2d.shape[-1]

    packed, scale_1, scale_2 = _quantize_nvfp4(w2d, block_size=16)
    result = _dequantize_nvfp4(packed, scale_1, scale_2, (N, K), orig_dtype)

    if result.shape[-1] != orig_shape[-1]:
        result = result[..., :orig_shape[-1]]
    return result.reshape(orig_shape)


# ── FP8 quantization — VERBATIM from TRT-LLM torch_quant.py ────────────

def fp8_fake_quantize(w: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 quantize then dequantize. Uses TRT-LLM's exact math:
    scale = amax / 448, quantize = (w / scale).to(float8_e4m3fn), dequantize = q * scale"""
    amax = w.abs().amax().clamp(min=1e-12)
    scale = amax / 448.0
    return (w.float() / scale).to(torch.float8_e4m3fn).to(w.dtype) * scale


# ── FP8 activation quantization — from TRT-LLM fp8_utils.py ────────────
# Source: tensorrt_llm/quantization/utils/fp8_utils.py per_token_cast_to_fp8_e8m0

def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def fp8_activation_fake_quantize(x: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """FP8 E4M3 activation quantization with per-group UE8M0 scaling.
    Verbatim from TRT-LLM per_token_cast_to_fp8_e8m0."""
    orig_shape = x.shape
    K = orig_shape[-1]
    if K % group_size != 0:
        pad = group_size - K % group_size
        x = F.pad(x, (0, pad))
        K = x.shape[-1]
    flat = x.reshape(-1, K)
    M = flat.shape[0]
    x_view = flat.view(M, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).view(M, -1).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    quantized = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
    dequantized = quantized.to(x.dtype) * sf.to(x.dtype).unsqueeze(2)
    result = dequantized.reshape(flat.shape)
    if result.shape[-1] != orig_shape[-1]:
        result = result[..., :orig_shape[-1]]
    return result.reshape(orig_shape)


# ── NVFP4 activation quantization — from TRT-LLM fused_moe_cutlass.py ──
# Uses torch.ops.trtllm.fp4_quantize logic: same E2M1 grid, per-16-block scaling

def nvfp4_activation_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 activation quantization matching torch.ops.trtllm.fp4_quantize.
    Per-16-block FP8 scaling, same E2M1 grid as weight quantization."""
    orig_shape = x.shape
    orig_dtype = x.dtype
    x2d = x.reshape(-1, x.shape[-1])
    N, K = x2d.shape
    if K % 16 != 0:
        pad = 16 - K % 16
        x2d = F.pad(x2d, (0, pad))
        K = x2d.shape[-1]

    packed, scale_1, scale_2 = _quantize_nvfp4(x2d, block_size=16)
    result = _dequantize_nvfp4(packed, scale_1, scale_2, (N, K), orig_dtype)

    if result.shape[-1] != orig_shape[-1]:
        result = result[..., :orig_shape[-1]]
    return result.reshape(orig_shape)


# ── Mixed-precision linear ──────────────────────────────────────────────

def real_precision_linear(x: torch.Tensor, w: torch.Tensor,
                          fp8_mask: Optional[torch.Tensor] = None,
                          mode: str = "fp4") -> torch.Tensor:
    if mode == "bf16":
        return F.linear(x, w)

    if mode == "fp4":
        w_q = nvfp4_fake_quantize(w)
        x_q = nvfp4_activation_fake_quantize(x)
        return F.linear(x_q, w_q)

    if mode == "fp8":
        w_q = fp8_fake_quantize(w)
        x_q = fp8_activation_fake_quantize(x)
        return F.linear(x_q, w_q)

    if mode == "mixed":
        assert fp8_mask is not None
        if fp8_mask.all():
            return real_precision_linear(x, w, mode="fp8")
        if not fp8_mask.any():
            return real_precision_linear(x, w, mode="fp4")

        fp8_idx = fp8_mask.nonzero(as_tuple=True)[0]
        fp4_idx = (~fp8_mask).nonzero(as_tuple=True)[0]

        w_fp4 = nvfp4_fake_quantize(w[fp4_idx])
        x_fp4 = nvfp4_activation_fake_quantize(x)
        out_fp4 = F.linear(x_fp4, w_fp4)

        w_fp8 = fp8_fake_quantize(w[fp8_idx])
        x_fp8 = fp8_activation_fake_quantize(x)
        out_fp8 = F.linear(x_fp8, w_fp8)

        out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
        out[:, fp4_idx] = out_fp4
        out[:, fp8_idx] = out_fp8
        return out

    raise ValueError(f"Unknown mode: {mode}")


# ── MoE forward with real precision ─────────────────────────────────────

def moe_forward_real_precision(
    hidden_states: torch.Tensor,
    tensors: dict,
    config,
    mode: str = "fp4",
    w1_masks: Optional[dict] = None,
    w2_masks: Optional[dict] = None,
) -> torch.Tensor:
    B, S, H = hidden_states.shape
    flat = hidden_states.view(-1, H)

    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final = torch.zeros(B * S, H, dtype=hidden_states.dtype, device=hidden_states.device)

    gate_up = tensors["experts.gate_up_proj"]
    down = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    if "shared_expert.gate_proj.weight" in tensors:
        shared_g = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
        shared_u = F.linear(flat, tensors["shared_expert.up_proj.weight"])
        shared_out = F.linear(F.silu(shared_g) * shared_u, tensors["shared_expert.down_proj.weight"])
        shared_gate_val = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
        final = final + shared_out * shared_gate_val

    for eidx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        tok_idx, route_pos = torch.where(selected_experts == eidx)
        cur = flat[tok_idx]

        expert_mode = mode
        w1_mask = w1_masks.get(eidx) if w1_masks else None
        w2_mask = w2_masks.get(eidx) if w2_masks else None
        if mode == "mixed" and w1_mask is not None:
            if w1_mask.shape[0] == gate_up[eidx].shape[0] // 2:
                w1_mask_full = w1_mask.repeat_interleave(2)
            else:
                w1_mask_full = w1_mask
        else:
            w1_mask_full = None

        gu = real_precision_linear(cur, gate_up[eidx], fp8_mask=w1_mask_full, mode=expert_mode)
        g, u = gu.chunk(2, dim=-1)
        intermediate = F.silu(g) * u
        out = real_precision_linear(intermediate, down[eidx], fp8_mask=w2_mask, mode=expert_mode)

        final.index_add_(0, tok_idx, (routing_weights[tok_idx, route_pos].unsqueeze(-1) * out).to(hidden_states.dtype))

    return final.view(B, S, H)


# ── GPTQ-standard data loading ──────────────────────────────────────────

def load_calibration_data(tokenizer, n_samples=128, seqlen=2048, seed=0):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    random.seed(seed)
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        samples.append(trainenc.input_ids[:, i:i+seqlen])
    return samples


def load_eval_data(tokenizer, seqlen=2048):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    nsamples = testenc.input_ids.numel() // seqlen
    return testenc.input_ids, nsamples, seqlen


# ── Full model forward pass (layer-by-layer) ───────────────────────────

def evaluate_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config,
    weight_map: dict,
    snapshot_dir,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "fp4",
    w1_masks_per_layer: Optional[dict] = None,
    w2_masks_per_layer: Optional[dict] = None,
    label: str = "",
) -> float:
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"

    root_t = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    final_norm_w = move_tensor(root_t[norm_key], device, dtype)
    lm_head_w = move_tensor(root_t[lm_head_key], device, dtype)
    del root_t

    nlls = []
    with torch.inference_mode():
        for sample_idx in range(nsamples):
            chunk = eval_ids[:, sample_idx * seqlen : (sample_idx + 1) * seqlen].to(device)
            h = F.embedding(chunk, embed_w)
            mask = build_causal_mask(seqlen, device)
            pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
            rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
            pos_emb = rotary(h, pos_ids)

            for lidx in range(config.num_hidden_layers):
                lt = config.layer_types[lidx]
                raw = store.load_tensors(layer_keys(lidx, lt))
                s = shorten_layer_tensors(lidx, raw, device, dtype)
                del raw

                res = h
                h = rms_norm_qwen3_next(h, s["input_layernorm.weight"], config.rms_norm_eps)

                if lt == "full_attention":
                    attn_t = {k.replace("self_attn.", ""): v for k, v in s.items() if k.startswith("self_attn.")}
                    h = full_attention_forward(h, attn_t, config, pos_emb, mask)
                else:
                    attn_t = {k.replace("linear_attn.", ""): v for k, v in s.items() if k.startswith("linear_attn.")}
                    h = linear_attention_forward(h, attn_t, config)
                h = res + h

                res = h
                h = rms_norm_qwen3_next(h, s["post_attention_layernorm.weight"], config.rms_norm_eps)

                moe = {k.replace("mlp.", "", 1): v for k, v in s.items() if k.startswith("mlp.")}
                w1m = w1_masks_per_layer.get(lidx, {}) if w1_masks_per_layer else {}
                w2m = w2_masks_per_layer.get(lidx, {}) if w2_masks_per_layer else {}

                moe_out = moe_forward_real_precision(h, moe, config, mode=mode, w1_masks=w1m, w2_masks=w2m)
                h = res + moe_out

                release_tensors(s)

            h = rms_norm_qwen3_next(h, final_norm_w, config.rms_norm_eps)
            logits = F.linear(h.float(), lm_head_w.float())

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)

            if (sample_idx + 1) % 10 == 0:
                print(f"  [{label}] chunk {sample_idx+1}/{nsamples}", flush=True)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()

    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "real_eval_pipeline.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, nsamples, seqlen = load_eval_data(tokenizer, args.seqlen)
    print(f"Eval: {nsamples} chunks of {seqlen} tokens ({nsamples * seqlen} total)")

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)

    configs = [
        ("uniform_bf16", "bf16"),
        ("uniform_nvfp4", "fp4"),
        ("uniform_fp8", "fp8"),
    ]

    results = {}
    for label, mode in configs:
        print(f"\n=== {label} ===", flush=True)
        t0 = time.time()
        ppl = evaluate_ppl(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, mode=mode, label=label,
        )
        elapsed = time.time() - t0
        results[label] = {"ppl": round(ppl, 4), "time_s": round(elapsed, 1)}
        print(f"  → PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    output = {
        "metadata": {
            "model": args.model_id,
            "eval_tokens": nsamples * seqlen,
            "seqlen": seqlen,
            "nsamples": nsamples,
            "quantization_source": "TRT-LLM torch_quant.py (verbatim copy)",
            "nvfp4_weight": "E2M1 per-16-block FP8 scaling, two-level (scale_1 * scale_2)",
            "nvfp4_activation": "E2M1 per-16-block FP8 scaling, same as weight",
            "fp8_weight": "E4M3 per-tensor scaling (amax/448)",
            "fp8_activation": "E4M3 per-128-group UE8M0 scaling (2^ceil(log2(amax/448)))",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {args.output}")

    print(f"\n{'Method':<25s} {'PPL':>8s}")
    print("-" * 35)
    for label, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{label:<25s} {r['ppl']:>8.4f}")


if __name__ == "__main__":
    main()

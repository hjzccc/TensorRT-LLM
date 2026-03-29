#!/usr/bin/env python3
"""NVFP4 sub-format compression with correct code-space pipeline.

Operates directly on FP4 codes — no BF16 roundtrip.
Pipeline: fp4_quantize(orig) → unpack codes → map via LUT → repack → kernel(orig_scales)
"""
import sys
import time
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new/profiling")

import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
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
BF16_PPL = 6.5896
NVFP4_PPL = 6.8431

# E2M1 code → float value (16 entries, codes 0-15)
# Bit layout: [sign(1) | exponent(2) | mantissa(1)]
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # codes 0-7  (positive)
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # codes 8-15 (negative)
], dtype=torch.float32)


# ── FP4 pack/unpack ──────────────────────────────────────────────────────

def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] uint8 → [M, K] uint8 with values 0-15.

    Byte layout: low nibble = even element, high nibble = odd element.
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def repack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """[M, K] uint8 (values 0-15) → [M, K/2] uint8 packed."""
    M, K = codes.shape
    codes = codes.view(M, K // 2, 2)
    return (codes[:, :, 0] | (codes[:, :, 1] << 4)).to(torch.uint8)


# ── Code-space mapping ───────────────────────────────────────────────────

def build_code_lut(sub_values):
    """Build 16-entry LUT: lut[old_code] = new_code.

    sub_values: list of FP4 values in the sub-codebook, e.g. [-6,-4,-2,0,2,4,6]
    """
    cb = torch.tensor(sub_values, dtype=torch.float32)
    lut = torch.zeros(16, dtype=torch.uint8)

    for src_code in range(16):
        src_val = E2M1_TABLE[src_code]
        dists = (cb - src_val).abs()
        nearest_val = cb[dists.argmin()].item()

        best_code = None
        best_dist = float('inf')
        for dst_code in range(16):
            if abs(E2M1_TABLE[dst_code].item() - nearest_val) < 1e-6:
                d = abs(E2M1_TABLE[dst_code].item() - src_val)
                if d < best_dist or (d == best_dist and best_code is not None and dst_code < best_code):
                    best_dist = d
                    best_code = dst_code
        lut[src_code] = best_code

    return lut


def apply_code_lut(codes: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply lookup table to remap FP4 codes. Both on same device."""
    return lut[codes.long()]


CODEBOOKS = {
    'nvfp4_full': [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
    '3bit_uniform': [-6, -4, -2, 0, 2, 4, 6],
    '3bit_dense': [-6, -2, -1, 0, 1, 2, 6],
    '3bit_truncate': [-4, -2, -1, 0, 1, 2, 4],
    '2bit_opt1': [-4, 0, 3, 6],
    '2bit_opt3': [-4, 0, 2, 6],
    '2bit_sym_zero': [-4, 0, 4],
    '2bit_uniform': [-6, -2, 2, 6],
}


# ── Code-space NVFP4 linear ─────────────────────────────────────────────

def nvfp4_linear_codespace(input_tensor, weight_bf16, lut, device='cuda'):
    """NVFP4 linear with code-space sub-codebook mapping.

    1. Quantize original weight → (packed_fp4, block_scales, global_scale)
    2. Unpack → remap codes via LUT → repack
    3. Feed modified codes + ORIGINAL scales to kernel
    """
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])

    s_w = fp4_global_scale(weight_bf16).to(torch.float32)
    packed_orig, block_scales = torch.ops.trtllm.fp4_quantize(weight_bf16, s_w, BLOCK_SIZE, False)

    if lut is not None:
        codes = unpack_fp4_codes(packed_orig)
        codes_mapped = apply_code_lut(codes, lut.to(device))
        packed_new = repack_fp4_codes(codes_mapped)
    else:
        packed_new = packed_orig

    s_in = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in * s_w)).to(torch.float32)

    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, packed_new, bias=None,
        input_scale=s_in, weight_scale=block_scales, alpha=alpha,
    )
    return out.reshape(*input_tensor.shape[:-1], weight_bf16.shape[0])


# ── Per-block optimal codebook (exhaustive search) ──────────────────────

def find_per_block_optimal_lut(weight_bf16, k_codes, device='cuda'):
    """For each block of 16, find the best K-code subset by MSE.

    Returns per-block LUTs as [n_blocks, 16] uint8 tensor.
    """
    from itertools import combinations

    s_w = fp4_global_scale(weight_bf16).to(torch.float32)
    packed_orig, block_scales = torch.ops.trtllm.fp4_quantize(weight_bf16, s_w, BLOCK_SIZE, False)

    codes = unpack_fp4_codes(packed_orig)
    M, K = codes.shape
    n_blocks = K // BLOCK_SIZE

    code_blocks = codes.reshape(M, n_blocks, BLOCK_SIZE)
    values = E2M1_TABLE.to(device)

    all_luts = torch.zeros(M, n_blocks, 16, dtype=torch.uint8, device=device)

    positive_codes = list(range(8))
    combos = list(combinations(positive_codes, k_codes // 2))

    for row in range(M):
        for blk in range(n_blocks):
            block = code_blocks[row, blk]
            block_vals = values[block.long()]

            best_mse = float('inf')
            best_lut = None

            for pos_combo in combos:
                neg_combo = tuple(c + 8 for c in pos_combo)
                allowed = set(pos_combo) | set(neg_combo)

                lut = torch.zeros(16, dtype=torch.uint8, device=device)
                for src in range(16):
                    src_val = values[src]
                    best_code = min(allowed, key=lambda c: abs(values[c] - src_val))
                    lut[src] = best_code

                mapped = lut[block.long()]
                mapped_vals = values[mapped.long()]
                mse = ((block_vals - mapped_vals) ** 2).mean().item()

                if mse < best_mse:
                    best_mse = mse
                    best_lut = lut.clone()

            all_luts[row, blk] = best_lut

    return all_luts, packed_orig, block_scales, s_w


def nvfp4_linear_perblock(input_tensor, weight_bf16, k_codes, device='cuda'):
    """NVFP4 linear with per-block optimal K-code codebook."""
    all_luts, packed_orig, block_scales, s_w = find_per_block_optimal_lut(
        weight_bf16, k_codes, device
    )

    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    codes = unpack_fp4_codes(packed_orig)
    M, K = codes.shape
    n_blocks = K // BLOCK_SIZE

    code_blocks = codes.reshape(M, n_blocks, BLOCK_SIZE)
    for row in range(M):
        for blk in range(n_blocks):
            lut = all_luts[row, blk]
            code_blocks[row, blk] = lut[code_blocks[row, blk].long()]

    codes_mapped = code_blocks.reshape(M, K)
    packed_new = repack_fp4_codes(codes_mapped)

    s_in = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in * s_w)).to(torch.float32)

    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, packed_new, bias=None,
        input_scale=s_in, weight_scale=block_scales, alpha=alpha,
    )
    return out.reshape(*input_tensor.shape[:-1], weight_bf16.shape[0])


# ── Full model evaluation ────────────────────────────────────────────────

def evaluate(approach_name, codebook_name, lut, device='cuda', linear_fn=None):
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16

    if codebook_name in CODEBOOKS:
        cb_vals = CODEBOOKS[codebook_name]
        n_unique = len(set([abs(v) for v in cb_vals if v != 0])) * 2 + (1 if 0 in cb_vals else 0)
        n_unique = len(set(cb_vals))
        bits = math.log2(max(n_unique, 2))
    else:
        bits = 4.0
    effective_bpe = bits + 0.5

    print(f"\n{'='*60}")
    print(f"Approach: {approach_name}")
    if codebook_name in CODEBOOKS:
        print(f"Codebook: {sorted(CODEBOOKS[codebook_name])}")
    print(f"Bits/code: {bits:.2f}, Effective: {effective_bpe:.2f} bits/elem")
    print(f"Pipeline: code-space (no BF16 roundtrip)")
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

                    if linear_fn is not None:
                        gate_up = linear_fn(expert_input, w1, device)
                        intermediate = F.silu(gate_up[:, :gate_up.shape[1]//2]) * gate_up[:, gate_up.shape[1]//2:]
                        expert_out = linear_fn(intermediate, w2, device)
                    else:
                        gate_up = nvfp4_linear_codespace(expert_input, w1, lut, device)
                        intermediate = F.silu(gate_up[:, :gate_up.shape[1]//2]) * gate_up[:, gate_up.shape[1]//2:]
                        expert_out = nvfp4_linear_codespace(intermediate, w2, lut, device)

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

    print(f"\n{'='*60}")
    print(f"RESULT: {approach_name}")
    print(f"  PPL: {final_ppl:.4f}")
    print(f"  BF16={BF16_PPL:.4f} | NVFP4={NVFP4_PPL:.4f}")
    if final_ppl <= NVFP4_PPL:
        recovery = (NVFP4_PPL - final_ppl) / (NVFP4_PPL - BF16_PPL) * 100
        print(f"  Recovery: {recovery:.1f}%")
    else:
        print(f"  WORSE than NVFP4 by +{final_ppl - NVFP4_PPL:.4f} PPL")
    print(f"  Bits/elem: {effective_bpe:.2f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    return {
        'approach': approach_name, 'ppl': final_ppl, 'bits': bits,
        'effective_bpe': effective_bpe, 'time': elapsed,
        'codebook': codebook_name,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--codebook', type=str, default='nvfp4_full',
                        choices=list(CODEBOOKS.keys()))
    parser.add_argument('--verify-baseline', action='store_true',
                        help='Run identity mapping to verify pipeline matches NVFP4 baseline')
    args = parser.parse_args()

    if args.verify_baseline:
        print("VERIFICATION: identity mapping (should match NVFP4 baseline exactly)")
        result = evaluate('verify_identity', 'nvfp4_full', lut=None)
    else:
        lut = build_code_lut(CODEBOOKS[args.codebook])
        print(f"Code LUT for {args.codebook}:")
        for i in range(16):
            src_val = E2M1_TABLE[i].item()
            dst_val = E2M1_TABLE[lut[i].item()].item()
            changed = " *" if i != lut[i].item() else ""
            print(f"  code {i:2d} ({src_val:+5.1f}) → code {lut[i].item():2d} ({dst_val:+5.1f}){changed}")
        result = evaluate(f'codespace_{args.codebook}', args.codebook, lut)

    out_path = Path(f"/code/tensorrt_llm/scripts/nvfp4_compress/result_{args.codebook}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

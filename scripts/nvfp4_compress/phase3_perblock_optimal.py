#!/usr/bin/env python3
"""Phase 3: Per-block optimal codebook selection with library approach.

For each block of 16 elements, find the best K-code subset by MSE.
Use a shared library of codebooks to minimize overhead.
"""
import sys
import time
import json
import math
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

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

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] uint8 → [M, K] uint8 with values 0-15."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def repack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """[M, K] uint8 (values 0-15) → [M, K/2] uint8 packed."""
    M, K = codes.shape
    codes = codes.view(M, K // 2, 2)
    return (codes[:, :, 0] | (codes[:, :, 1] << 4)).to(torch.uint8)


def build_codebook_library(k_codes=8, max_codebooks=256):
    """Build a library of optimal K-code codebooks.
    
    Returns:
        library: list of codebooks (each is a list of FP4 values)
        codebook_to_id: dict mapping frozenset(codebook) -> id
    """
    library = []
    codebook_to_id = {}
    
    positive_codes = list(range(8))
    combos = list(combinations(positive_codes, k_codes // 2))
    
    for pos_combo in combos[:max_codebooks]:
        neg_combo = tuple(c + 8 for c in pos_combo)
        allowed = set(pos_combo) | set(neg_combo)
        
        codebook = [E2M1_TABLE[c].item() for c in sorted(allowed)]
        cb_key = frozenset(codebook)
        
        if cb_key not in codebook_to_id:
            codebook_to_id[cb_key] = len(library)
            library.append(codebook)
    
    return library, codebook_to_id


def find_best_codebook(block_codes, library, values):
    """Find best codebook from library for a block of codes.
    
    Args:
        block_codes: [BLOCK_SIZE] tensor of FP4 codes
        library: list of codebooks
        values: E2M1_TABLE
    
    Returns:
        best_cb_id: index into library
        best_lut: [16] LUT for this block
    """
    block_vals = values[block_codes.long()]
    
    best_mse = float('inf')
    best_cb_id = 0
    best_lut = None
    
    for cb_id, codebook in enumerate(library):
        cb_vals = torch.tensor(codebook, dtype=torch.float32, device=values.device)
        
        # Build LUT for this codebook
        lut = torch.zeros(16, dtype=torch.uint8, device=values.device)
        for src in range(16):
            src_val = values[src]
            dists = (cb_vals - src_val).abs()
            best_code = dists.argmin().item()
            
            # Find the actual code in E2M1_TABLE
            for code in range(16):
                if abs(values[code].item() - cb_vals[best_code].item()) < 1e-6:
                    lut[src] = code
                    break
        
        # Compute MSE for this block
        mapped = lut[block_codes.long()]
        mapped_vals = values[mapped.long()]
        mse = ((block_vals - mapped_vals) ** 2).mean().item()
        
        if mse < best_mse:
            best_mse = mse
            best_cb_id = cb_id
            best_lut = lut.clone()
    
    return best_cb_id, best_lut


def compress_weight_perblock(weight_bf16, k_codes=8, device='cuda'):
    """Compress a weight matrix using per-block optimal codebooks.
    
    Returns:
        packed_new: compressed FP4 codes
        block_scales: original block scales
        s_w: original global scale
        codebook_ids: [n_blocks] tensor of codebook IDs
        library: list of codebooks
    """
    s_w = fp4_global_scale(weight_bf16).to(torch.float32)
    packed_orig, block_scales = torch.ops.trtllm.fp4_quantize(weight_bf16, s_w, BLOCK_SIZE, False)
    
    codes = unpack_fp4_codes(packed_orig)
    M, K = codes.shape
    n_blocks = K // BLOCK_SIZE
    
    # Build library
    library, _ = build_codebook_library(k_codes)
    values = E2M1_TABLE.to(device)
    
    # Find best codebook for each block
    codebook_ids = torch.zeros(M, n_blocks, dtype=torch.uint8, device=device)
    all_luts = torch.zeros(M, n_blocks, 16, dtype=torch.uint8, device=device)
    
    code_blocks = codes.reshape(M, n_blocks, BLOCK_SIZE)
    
    for row in range(M):
        for blk in range(n_blocks):
            block = code_blocks[row, blk].to(device)
            cb_id, lut = find_best_codebook(block, library, values)
            codebook_ids[row, blk] = cb_id
            all_luts[row, blk] = lut
    
    # Apply LUTs to remap codes
    code_blocks_device = code_blocks.to(device)
    for row in range(M):
        for blk in range(n_blocks):
            lut = all_luts[row, blk]
            code_blocks_device[row, blk] = lut[code_blocks_device[row, blk].long()]
    
    codes_mapped = code_blocks_device.reshape(M, K).cpu()
    packed_new = repack_fp4_codes(codes_mapped)
    
    return packed_new, block_scales, s_w, codebook_ids, library


def nvfp4_linear_perblock(input_tensor, weight_bf16, packed_new, block_scales, s_w, device='cuda'):
    """NVFP4 linear with pre-compressed weights."""
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    s_in = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in * s_w)).to(torch.float32)
    
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, packed_new, bias=None,
        input_scale=s_in, weight_scale=block_scales, alpha=alpha,
    )
    return out.reshape(*input_tensor.shape[:-1], weight_bf16.shape[0])


def evaluate_perblock(k_codes=8, device='cuda'):
    """Evaluate per-block optimal codebook compression."""
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16
    
    bits = math.log2(k_codes)
    overhead_bits = math.log2(256) / BLOCK_SIZE  # 256 codebooks, 1 per block
    effective_bpe = bits + overhead_bits
    
    print(f"\n{'='*60}")
    print(f"Phase 3: Per-block optimal {k_codes}-code codebook")
    print(f"Bits/code: {bits:.2f}, Overhead: {overhead_bits:.3f}, Effective: {effective_bpe:.2f} bits/elem")
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
    
    # Pre-compress all expert weights
    print("Pre-compressing expert weights...", flush=True)
    compressed_experts = {}
    
    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw
            
            moe = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}
            
            # Compress expert weights
            gate_up_w1 = moe["experts.gate_up_proj"]
            down_w2 = moe["experts.down_proj"]
            
            for expert_idx in range(model_config.num_experts):
                packed_w1, scales_w1, s_w1, _, _ = compress_weight_perblock(
                    gate_up_w1[expert_idx], k_codes, device
                )
                packed_w2, scales_w2, s_w2, _, _ = compress_weight_perblock(
                    down_w2[expert_idx], k_codes, device
                )
                
                compressed_experts[(layer_idx, expert_idx, 'w1')] = (packed_w1, scales_w1, s_w1)
                compressed_experts[(layer_idx, expert_idx, 'w2')] = (packed_w2, scales_w2, s_w2)
            
            if (layer_idx + 1) % 5 == 0:
                print(f"  Compressed layer {layer_idx+1}/{model_config.num_hidden_layers}", flush=True)
            
            release_tensors(layer_weights)
    
    print("Running inference with compressed weights...", flush=True)
    
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
                    
                    # Use compressed weights
                    packed_w1, scales_w1, s_w1 = compressed_experts[(layer_idx, expert_idx, 'w1')]
                    packed_w2, scales_w2, s_w2 = compressed_experts[(layer_idx, expert_idx, 'w2')]
                    
                    gate_up = nvfp4_linear_perblock(expert_input, moe["experts.gate_up_proj"][expert_idx], packed_w1, scales_w1, s_w1, device)
                    intermediate = F.silu(gate_up[:, :gate_up.shape[1]//2]) * gate_up[:, gate_up.shape[1]//2:]
                    expert_out = nvfp4_linear_perblock(intermediate, moe["experts.down_proj"][expert_idx], packed_w2, scales_w2, s_w2, device)
                    
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
    print(f"RESULT: Per-block {k_codes}-code")
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
        'approach': f'perblock_{k_codes}code',
        'ppl': final_ppl,
        'bits': bits,
        'effective_bpe': effective_bpe,
        'time': elapsed,
        'k_codes': k_codes,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-codes', type=int, default=8, choices=[4, 8])
    args = parser.parse_args()
    
    result = evaluate_perblock(args.k_codes)
    
    out_path = Path(f"/code/tensorrt_llm/scripts/nvfp4_compress/result_perblock_{args.k_codes}code.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

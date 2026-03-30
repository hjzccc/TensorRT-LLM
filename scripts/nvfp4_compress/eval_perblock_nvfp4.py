#!/usr/bin/env python3
"""
Per-Block Sub-Codebook PPL Evaluation (NVFP4 Tensor Core Version)

Uses NVFP4 tensor cores for memory-efficient evaluation.
Applies per-block K-code subset remapping to packed FP4 codes.

This is more memory-efficient than the BF16 approach because:
- No need to store remapped BF16 weights in GPU memory
- Uses the same NVFP4 linear operation as the baseline
- Per-block remapping applied to packed FP4 codes

Usage:
  python eval_perblock_nvfp4.py --K 4
  python eval_perblock_nvfp4.py --K 6
  python eval_perblock_nvfp4.py --K 8
"""

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys, math, time, json, argparse
from pathlib import Path
from itertools import combinations

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant_new')
sys.path.insert(0, '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant')

import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)
from real_eval_pipeline import load_eval_data
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
BLOCK_SIZE = 16
BF16_PPL = 6.5896
NVFP4_PPL = 6.8431

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

UNIQUE_VALS = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

_subset_cache = {}
def get_subsets(K, device):
    key = (K, str(device))
    if key not in _subset_cache:
        subsets = list(combinations(range(15), K))
        subset_vals = torch.stack([UNIQUE_VALS[list(s)] for s in subsets]).to(device)
        _subset_cache[key] = (subsets, subset_vals)
    return _subset_cache[key]


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


def perblock_remap_packed(packed: torch.Tensor, K: int, device: str, chunk_size: int = 128) -> torch.Tensor:
    """
    Apply per-block K-code subset remapping to packed FP4 codes.
    
    Args:
        packed: [M, N/2] uint8 packed FP4 codes
        K: number of codes in subset
        device: computation device
        chunk_size: number of blocks to process at once
    
    Returns:
        [M, N/2] uint8 packed FP4 codes with per-block remapping applied
    """
    fp4_table = E2M1_TABLE.to(device)
    _, subset_vals = get_subsets(K, device)  # [N_subsets, K]
    
    M, N_half = packed.shape
    N = N_half * 2
    
    # Unpack to [M, N] codes
    codes = unpack_fp4_codes(packed)  # [M, N] uint8
    fp4_vals = fp4_table[codes.long()]  # [M, N] float
    
    # Process in blocks of BLOCK_SIZE
    n_blocks_per_row = N // BLOCK_SIZE
    fp4_flat = fp4_vals.view(M * n_blocks_per_row, BLOCK_SIZE)
    new_codes_flat = codes.view(M * n_blocks_per_row, BLOCK_SIZE).clone()
    
    for chunk_start in range(0, M * n_blocks_per_row, chunk_size):
        chunk_end = min(chunk_start + chunk_size, M * n_blocks_per_row)
        chunk = fp4_flat[chunk_start:chunk_end]  # [C, BLOCK_SIZE]
        C = chunk.shape[0]
        
        # Find best K-code subset for each block in chunk
        chunk_exp = chunk.unsqueeze(-1).unsqueeze(-1)  # [C, BLOCK_SIZE, 1, 1]
        sv_exp = subset_vals.unsqueeze(0).unsqueeze(0)  # [1, 1, N_subsets, K]
        dists_sub = torch.abs(chunk_exp - sv_exp)  # [C, BLOCK_SIZE, N_subsets, K]
        min_dists_sub = dists_sub.min(dim=-1).values  # [C, BLOCK_SIZE, N_subsets]
        mse_sub = min_dists_sub.pow(2).mean(dim=1)  # [C, N_subsets]
        best_idx = mse_sub.argmin(dim=-1)  # [C]
        
        best_cbs = subset_vals[best_idx]  # [C, K]
        block_dists = torch.abs(chunk.unsqueeze(-1) - best_cbs.unsqueeze(1))  # [C, BLOCK_SIZE, K]
        nearest = best_cbs[torch.arange(C, device=device).unsqueeze(1), block_dists.argmin(dim=-1)]
        
        # Find FP4 code indices for remapped values
        code_dists = torch.abs(nearest.unsqueeze(-1) - fp4_table.unsqueeze(0).unsqueeze(0))
        new_codes_flat[chunk_start:chunk_end] = code_dists.argmin(dim=-1).to(torch.uint8)
    
    # Repack
    new_codes = new_codes_flat.view(M, N).to(torch.uint8)
    return repack_fp4_codes(new_codes)


def nvfp4_linear_perblock(input_tensor, weight_bf16, K, device='cuda'):
    """NVFP4 linear with per-block sub-codebook remapping."""
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    
    s_w = fp4_global_scale(weight_bf16).to(torch.float32)
    packed_orig, block_scales = torch.ops.trtllm.fp4_quantize(weight_bf16, s_w, BLOCK_SIZE, False)
    
    # Apply per-block remapping to packed codes
    packed_remapped = perblock_remap_packed(packed_orig, K, device)
    
    s_in = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in * s_w)).to(torch.float32)
    
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, packed_remapped, bias=None,
        input_scale=s_in, weight_scale=block_scales, alpha=alpha,
    )
    return out.reshape(*input_tensor.shape[:-1], weight_bf16.shape[0])


def evaluate_perblock_nvfp4(K: int, device: str = 'cuda'):
    """Evaluate PPL with per-block K-code subset selection using NVFP4 tensor cores."""
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16
    
    bits = math.log2(K)
    effective_bpe = bits + 0.5
    
    print(f"\n{'='*60}")
    print(f"Per-Block Sub-Codebook PPL Evaluation (NVFP4 Tensor Cores)")
    print(f"K={K} codes per block, {bits:.2f} bits/code")
    print(f"Effective: {effective_bpe:.2f} bits/elem")
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
    remap_total_time = 0
    
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
                
                t_remap = time.time()
                for expert_idx in torch.nonzero(active_counts > 0, as_tuple=False).flatten().tolist():
                    token_indices, route_indices = torch.where(topk_ids == expert_idx)
                    expert_input = flat[token_indices]
                    w1 = moe["experts.gate_up_proj"][expert_idx]
                    w2 = moe["experts.down_proj"][expert_idx]
                    
                    gate_up = nvfp4_linear_perblock(expert_input, w1, K, device)
                    intermediate = F.silu(gate_up[:, :gate_up.shape[1]//2]) * gate_up[:, gate_up.shape[1]//2:]
                    expert_out = nvfp4_linear_perblock(intermediate, w2, K, device)
                    
                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))
                remap_total_time += time.time() - t_remap
                
                shared = ee.bf16_linear(flat, moe["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, moe["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, moe["shared_expert.down_proj.weight"])
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, moe["shared_expert_gate.weight"]))
                
                hidden = residual + (expert_output + shared_gate * shared).view_as(residual)
                hidden_bank[sample_idx].copy_(hidden.squeeze(0).cpu())
            
            release_tensors(layer_weights)
            elapsed = time.time() - t0
            print(f"  Layer {layer_idx+1}/{model_config.num_hidden_layers} ({elapsed:.0f}s, remap={remap_total_time:.0f}s)", flush=True)
        
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
    print(f"RESULT: Per-Block K={K} Sub-Codebook (NVFP4 Tensor Cores)")
    print(f"  PPL: {final_ppl:.4f}")
    print(f"  BF16={BF16_PPL:.4f} | NVFP4={NVFP4_PPL:.4f}")
    if final_ppl <= NVFP4_PPL:
        recovery = (NVFP4_PPL - final_ppl) / (NVFP4_PPL - BF16_PPL) * 100
        print(f"  Recovery: {recovery:.1f}%")
    else:
        print(f"  WORSE than NVFP4 by +{final_ppl - NVFP4_PPL:.4f} PPL")
    print(f"  Bits/elem: {effective_bpe:.2f}")
    print(f"  Remap time: {remap_total_time:.0f}s")
    print(f"  Total time: {elapsed:.0f}s")
    print(f"{'='*60}")
    
    result = {
        'approach': f'perblock_nvfp4_K{K}',
        'K': K,
        'ppl': final_ppl,
        'bits': bits,
        'effective_bpe': effective_bpe,
        'ppl_delta': final_ppl - NVFP4_PPL,
        'remap_time': remap_total_time,
        'total_time': elapsed,
    }
    
    out_path = Path(f'result_perblock_nvfp4_K{K}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=4, choices=[2, 3, 4, 6, 8])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    evaluate_perblock_nvfp4(K=args.K, device=args.device)

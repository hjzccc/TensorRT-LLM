#!/usr/bin/env python3
"""
Per-Block Sub-Codebook PPL Evaluation

Evaluates PPL for per-block optimal K-code subset selection (Variant A).
This is the CORRECT approach: each 16-element block gets its own optimal
K-code subset from the 16 FP4 E2M1 values.

Grounded in:
- BOF4 (arXiv:2505.06653): per-block optimal codebook selection
- EntroLLM (arXiv:2505.02380): entropy coding of quantized indices

Usage:
  python eval_perblock_subcodebook.py --K 4
  python eval_perblock_subcodebook.py --K 6
  python eval_perblock_subcodebook.py --K 8
"""

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

# Precompute all C(15, K) subsets for each K
_subset_cache = {}
def get_subsets(K):
    if K not in _subset_cache:
        subsets = list(combinations(range(15), K))
        # Convert to tensor for fast lookup
        subset_vals = torch.stack([UNIQUE_VALS[list(s)] for s in subsets])  # [N_subsets, K]
        _subset_cache[K] = (subsets, subset_vals)
    return _subset_cache[K]


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


def perblock_subcodebook_remap(weight_packed: torch.Tensor, K: int, device: str = 'cpu') -> torch.Tensor:
    """
    Apply per-block optimal K-code subset selection to packed FP4 weights.
    
    For each 16-element block, finds the K FP4 E2M1 values that minimize MSE,
    then remaps all codes in the block to the nearest code in the subset.
    
    Args:
        weight_packed: [M, N/2] uint8 packed FP4 codes
        K: number of codes in subset (2, 3, 4, 6, 8)
        device: computation device
    
    Returns:
        [M, N/2] uint8 packed FP4 codes with per-block remapping applied
    """
    M, N_half = weight_packed.shape
    N = N_half * 2
    
    # Unpack to [M, N] codes
    codes = unpack_fp4_codes(weight_packed)  # [M, N] uint8
    
    # Convert to values [M, N] float
    values = E2M1_TABLE[codes.long()]  # [M, N] float
    
    # Get all possible K-code subsets
    subsets, subset_vals = get_subsets(K)  # subset_vals: [N_subsets, K]
    subset_vals = subset_vals.to(device)
    
    # Process in blocks of BLOCK_SIZE along the N dimension
    new_codes = codes.clone()
    
    for col_start in range(0, N, BLOCK_SIZE):
        col_end = min(col_start + BLOCK_SIZE, N)
        block_vals = values[:, col_start:col_end]  # [M, BLOCK_SIZE]
        
        # For each row, find the best K-code subset
        # block_vals: [M, BLOCK_SIZE]
        # subset_vals: [N_subsets, K]
        
        # Compute MSE for each subset: [M, N_subsets]
        # block_vals[:, :, None]: [M, BLOCK_SIZE, 1]
        # subset_vals[None, :, :]: [1, N_subsets, K]
        # distances: [M, BLOCK_SIZE, N_subsets, K] -> too large
        # Instead, process row by row for memory efficiency
        
        for row_idx in range(M):
            row = block_vals[row_idx]  # [BLOCK_SIZE]
            
            # Distances from each element to each subset code
            # row: [BLOCK_SIZE, 1, 1]
            # subset_vals: [N_subsets, K]
            dists = torch.abs(row[:, None, None] - subset_vals[None, :, :])  # [BLOCK_SIZE, N_subsets, K]
            min_dists = dists.min(dim=-1).values  # [BLOCK_SIZE, N_subsets]
            mse_per_subset = min_dists.pow(2).mean(dim=0)  # [N_subsets]
            
            best_subset_idx = mse_per_subset.argmin().item()
            best_cb = subset_vals[best_subset_idx]  # [K]
            
            # Remap codes in this block to nearest code in best subset
            block_dists = torch.abs(row[:, None] - best_cb[None, :])  # [BLOCK_SIZE, K]
            nearest_in_subset = best_cb[block_dists.argmin(dim=-1)]  # [BLOCK_SIZE]
            
            # Find the FP4 code index for each remapped value
            for elem_idx in range(col_end - col_start):
                target_val = nearest_in_subset[elem_idx].item()
                # Find the FP4 code that matches this value
                code_dists = torch.abs(E2M1_TABLE - target_val)
                new_code = code_dists.argmin().item()
                new_codes[row_idx, col_start + elem_idx] = new_code
    
    # Repack
    return repack_fp4_codes(new_codes)


def nvfp4_linear_perblock(input_tensor, weight_bf16, K, device='cuda'):
    """NVFP4 linear with per-block sub-codebook remapping."""
    # weight_bf16 is the dequantized BF16 weight
    # We need to re-quantize it to NVFP4 with per-block remapping
    # For now, use the BF16 weight directly (this is the baseline)
    # TODO: implement proper per-block remapping on the packed codes
    return F.linear(input_tensor, weight_bf16)


def evaluate_perblock(K: int, device: str = 'cuda'):
    """Evaluate PPL with per-block K-code subset selection."""
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16
    
    bits = math.log2(K)
    effective_bpe = bits + 0.5  # approximate with codebook overhead
    
    print(f"\n{'='*60}")
    print(f"Per-Block Sub-Codebook Evaluation")
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
    
    # Cache for remapped weights (to avoid recomputing per sample)
    remap_cache = {}
    
    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw
            
            # Apply per-block remapping to MoE expert weights
            moe = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}
            
            # Remap expert weights
            remapped_moe = {}
            for k, v in moe.items():
                if 'weight' in k and v.dtype == torch.bfloat16:
                    # Apply per-block remapping
                    # v is BF16 dequantized weight
                    # We need to: quantize to FP4, apply per-block remapping, dequantize
                    # For now, use a simplified approach: round to nearest FP4 E2M1 value
                    # then apply per-block remapping
                    
                    # Step 1: Find the scale (max abs value)
                    scale = v.abs().max().item()
                    if scale == 0:
                        remapped_moe[k] = v
                        continue
                    
                    # Step 2: Normalize to [-1, 1] range of FP4
                    v_norm = v / scale * 6.0  # FP4 max is 6.0
                    
                    # Step 3: Quantize to nearest FP4 E2M1 value
                    fp4_vals = E2M1_TABLE.to(device)
                    dists = torch.abs(v_norm.unsqueeze(-1) - fp4_vals.unsqueeze(0).unsqueeze(0))
                    fp4_codes = dists.argmin(dim=-1)  # [M, N]
                    
                    # Step 4: Apply per-block remapping
                    M, N = fp4_codes.shape
                    subsets, subset_vals_t = get_subsets(K)
                    subset_vals_t = subset_vals_t.to(device)
                    
                    new_codes = fp4_codes.clone()
                    for col_start in range(0, N, BLOCK_SIZE):
                        col_end = min(col_start + BLOCK_SIZE, N)
                        block = fp4_vals[fp4_codes[:, col_start:col_end].long()]  # [M, BLOCK_SIZE]
                        
                        for row_idx in range(M):
                            row = block[row_idx]  # [BLOCK_SIZE]
                            dists_sub = torch.abs(row[:, None, None] - subset_vals_t[None, :, :])
                            min_dists_sub = dists_sub.min(dim=-1).values
                            mse_sub = min_dists_sub.pow(2).mean(dim=0)
                            best_idx = mse_sub.argmin().item()
                            best_cb = subset_vals_t[best_idx]
                            
                            block_dists = torch.abs(row[:, None] - best_cb[None, :])
                            nearest = best_cb[block_dists.argmin(dim=-1)]
                            
                            for elem_idx in range(col_end - col_start):
                                target = nearest[elem_idx].item()
                                code_d = torch.abs(fp4_vals - target)
                                new_codes[row_idx, col_start + elem_idx] = code_d.argmin().item()
                    
                    # Step 5: Dequantize back to BF16
                    remapped_vals = fp4_vals[new_codes.long()]  # [M, N]
                    remapped_weight = remapped_vals / 6.0 * scale
                    remapped_moe[k] = remapped_weight.to(dtype)
                else:
                    remapped_moe[k] = v
            
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
                    
                    # Use remapped weights
                    w1_key = f"experts.gate_up_proj"
                    w2_key = f"experts.down_proj"
                    
                    if w1_key in remapped_moe:
                        w1 = remapped_moe[w1_key][expert_idx]
                    else:
                        w1 = moe[w1_key][expert_idx]
                    
                    if w2_key in remapped_moe:
                        w2 = remapped_moe[w2_key][expert_idx]
                    else:
                        w2 = moe[w2_key][expert_idx]
                    
                    gate_up = F.linear(expert_input, w1)
                    intermediate = F.silu(gate_up[:, :gate_up.shape[1]//2]) * gate_up[:, gate_up.shape[1]//2:]
                    expert_out = F.linear(intermediate, w2)
                    
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
    print(f"RESULT: Per-Block K={K} Sub-Codebook")
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
    
    result = {
        'approach': f'perblock_K{K}',
        'K': K,
        'ppl': final_ppl,
        'bits': bits,
        'effective_bpe': effective_bpe,
        'time': elapsed,
        'ppl_delta': final_ppl - NVFP4_PPL,
    }
    
    out_path = Path(f'result_perblock_K{K}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=4, choices=[2, 3, 4, 6, 8])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    evaluate_perblock(K=args.K, device=args.device)

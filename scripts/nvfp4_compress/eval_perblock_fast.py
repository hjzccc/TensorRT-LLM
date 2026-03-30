#!/usr/bin/env python3
"""
Fast Per-Block Sub-Codebook PPL Evaluation

GPU-accelerated per-block optimal K-code subset selection.
Applies remapping once per layer, then evaluates PPL.

Grounded in:
- BOF4 (arXiv:2505.06653): per-block optimal codebook selection
- EntroLLM (arXiv:2505.02380): entropy coding of quantized indices

Usage:
  python eval_perblock_fast.py --K 4
  python eval_perblock_fast.py --K 6
  python eval_perblock_fast.py --K 8
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

_subset_cache = {}
def get_subsets(K, device):
    key = (K, device)
    if key not in _subset_cache:
        subsets = list(combinations(range(15), K))
        subset_vals = torch.stack([UNIQUE_VALS[list(s)] for s in subsets]).to(device)
        _subset_cache[key] = (subsets, subset_vals)
    return _subset_cache[key]


def perblock_remap_weight(weight_bf16: torch.Tensor, K: int, device: str, chunk_size: int = 128) -> torch.Tensor:
    """
    Apply per-block K-code subset remapping to a BF16 weight tensor.
    
    The weight is assumed to be dequantized from NVFP4 (values are FP4 E2M1 values
    scaled by the block scale). We re-quantize to FP4, apply per-block remapping,
    then dequantize back.
    
    Args:
        weight_bf16: [M, N] BF16 weight tensor (dequantized NVFP4)
        K: number of codes in subset
        device: computation device
    
    Returns:
        [M, N] BF16 weight tensor with per-block remapping applied
    """
    M, N = weight_bf16.shape
    fp4_table = E2M1_TABLE.to(device)
    unique_vals = UNIQUE_VALS.to(device)
    _, subset_vals = get_subsets(K, device)  # [N_subsets, K]
    
    # Step 1: Find the per-block scale (max abs value in each block)
    # Reshape to [M * N/BLOCK_SIZE, BLOCK_SIZE]
    n_blocks_per_row = N // BLOCK_SIZE
    w_blocks = weight_bf16.view(M, n_blocks_per_row, BLOCK_SIZE)  # [M, n_blocks, BLOCK_SIZE]
    
    # Per-block scale
    block_scales = w_blocks.abs().amax(dim=-1, keepdim=True)  # [M, n_blocks, 1]
    block_scales = block_scales.clamp(min=1e-8)
    
    # Step 2: Normalize to FP4 range [-6, 6]
    w_norm = w_blocks / block_scales * 6.0  # [M, n_blocks, BLOCK_SIZE]
    
    # Step 3: Quantize to nearest FP4 E2M1 value
    # w_norm: [M, n_blocks, BLOCK_SIZE, 1] vs fp4_table: [16]
    dists = torch.abs(w_norm.unsqueeze(-1) - fp4_table.view(1, 1, 1, -1))  # [M, n_blocks, BLOCK_SIZE, 16]
    fp4_codes = dists.argmin(dim=-1)  # [M, n_blocks, BLOCK_SIZE]
    fp4_vals = fp4_table[fp4_codes]  # [M, n_blocks, BLOCK_SIZE]
    
    # Step 4: Per-block optimal K-code subset selection
    # fp4_vals: [M, n_blocks, BLOCK_SIZE]
    # subset_vals: [N_subsets, K]
    
    # Process in chunks to avoid OOM
    fp4_flat = fp4_vals.view(M * n_blocks_per_row, BLOCK_SIZE)
    nearest_in_subset = torch.zeros_like(fp4_flat)
    
    for chunk_start in range(0, M * n_blocks_per_row, chunk_size):
        chunk_end = min(chunk_start + chunk_size, M * n_blocks_per_row)
        chunk = fp4_flat[chunk_start:chunk_end]  # [C, BLOCK_SIZE]
        C = chunk.shape[0]
        
        # [C, BLOCK_SIZE, 1, 1] vs [1, 1, N_subsets, K]
        chunk_exp = chunk.unsqueeze(-1).unsqueeze(-1)
        sv_exp = subset_vals.unsqueeze(0).unsqueeze(0)
        dists_sub = torch.abs(chunk_exp - sv_exp)  # [C, BLOCK_SIZE, N_subsets, K]
        min_dists_sub = dists_sub.min(dim=-1).values  # [C, BLOCK_SIZE, N_subsets]
        mse_sub = min_dists_sub.pow(2).mean(dim=1)  # [C, N_subsets]
        best_idx = mse_sub.argmin(dim=-1)  # [C]
        
        best_cbs = subset_vals[best_idx]  # [C, K]
        block_dists = torch.abs(chunk.unsqueeze(-1) - best_cbs.unsqueeze(1))  # [C, BLOCK_SIZE, K]
        nearest = best_cbs[torch.arange(C, device=device).unsqueeze(1), block_dists.argmin(dim=-1)]
        nearest_in_subset[chunk_start:chunk_end] = nearest
    
    # Step 5: Dequantize back to BF16
    # nearest_in_subset: [M*n_blocks, BLOCK_SIZE] (FP4 values in [-6, 6])
    # block_scales: [M, n_blocks, 1]
    remapped_norm = nearest_in_subset.view(M, n_blocks_per_row, BLOCK_SIZE)  # [M, n_blocks, BLOCK_SIZE]
    remapped_weight = remapped_norm / 6.0 * block_scales  # [M, n_blocks, BLOCK_SIZE]
    
    return remapped_weight.view(M, N).to(weight_bf16.dtype)


def evaluate_perblock_fast(K: int, device: str = 'cuda'):
    """Evaluate PPL with GPU-accelerated per-block K-code subset selection."""
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16
    
    bits = math.log2(K)
    effective_bpe = bits + 0.5
    
    print(f"\n{'='*60}")
    print(f"Per-Block Sub-Codebook PPL Evaluation (GPU-accelerated)")
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
            
            # Apply per-block remapping to expert weights
            t_remap = time.time()
            remapped_moe = {}
            for k, v in moe.items():
                if ('gate_up_proj' in k or 'down_proj' in k) and v.dtype == dtype:
                    if v.dim() == 3:
                        # [n_experts, M, N] - process each expert
                        remapped = torch.stack([
                            perblock_remap_weight(v[i], K, device)
                            for i in range(v.shape[0])
                        ])
                        remapped_moe[k] = remapped
                    elif v.dim() == 2:
                        remapped_moe[k] = perblock_remap_weight(v, K, device)
                    else:
                        remapped_moe[k] = v
                else:
                    remapped_moe[k] = v
            remap_total_time += time.time() - t_remap
            
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
                router_logits = ee.bf16_linear(flat, remapped_moe["gate.weight"]).float()
                routing_probs = torch.softmax(router_logits, dim=1)
                topk_weights, topk_ids = torch.topk(routing_probs, model_config.num_experts_per_tok, dim=-1)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(dtype)
                
                expert_output = torch.zeros_like(flat)
                active_counts = torch.bincount(topk_ids.reshape(-1), minlength=model_config.num_experts)
                
                for expert_idx in torch.nonzero(active_counts > 0, as_tuple=False).flatten().tolist():
                    token_indices, route_indices = torch.where(topk_ids == expert_idx)
                    expert_input = flat[token_indices]
                    w1 = remapped_moe["experts.gate_up_proj"][expert_idx]
                    w2 = remapped_moe["experts.down_proj"][expert_idx]
                    
                    gate_up = F.linear(expert_input, w1)
                    intermediate = F.silu(gate_up[:, :gate_up.shape[1]//2]) * gate_up[:, gate_up.shape[1]//2:]
                    expert_out = F.linear(intermediate, w2)
                    
                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))
                
                shared = ee.bf16_linear(flat, remapped_moe["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, remapped_moe["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, remapped_moe["shared_expert.down_proj.weight"])
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, remapped_moe["shared_expert_gate.weight"]))
                
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
    print(f"RESULT: Per-Block K={K} Sub-Codebook")
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
        'approach': f'perblock_K{K}',
        'K': K,
        'ppl': final_ppl,
        'bits': bits,
        'effective_bpe': effective_bpe,
        'ppl_delta': final_ppl - NVFP4_PPL,
        'remap_time': remap_total_time,
        'total_time': elapsed,
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
    
    evaluate_perblock_fast(K=args.K, device=args.device)

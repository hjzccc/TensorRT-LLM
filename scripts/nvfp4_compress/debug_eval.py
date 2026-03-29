#!/usr/bin/env python3
"""Debug version of evaluation with detailed logging."""
import sys
import time
import math
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

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

def unpack_fp4_codes(packed):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)

def repack_fp4_codes(codes):
    M, K = codes.shape
    codes = codes.view(M, K // 2, 2)
    return (codes[:, :, 0] | (codes[:, :, 1] << 4)).to(torch.uint8)

def build_code_lut(sub_values):
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

def nvfp4_linear_codespace(input_tensor, weight_bf16, lut, device='cuda'):
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    
    s_w = fp4_global_scale(weight_bf16).to(torch.float32)
    packed_orig, block_scales = torch.ops.trtllm.fp4_quantize(weight_bf16, s_w, BLOCK_SIZE, False)
    
    if lut is not None:
        codes = unpack_fp4_codes(packed_orig)
        codes_mapped = lut[codes.long()]
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

def debug_eval(codebook_name, codebook_vals, device='cuda', num_samples=2):
    """Debug evaluation with detailed logging."""
    print(f"\nDEBUG EVAL: {codebook_name}")
    print(f"Codebook: {codebook_vals}")
    print(f"Num samples: {num_samples}")
    
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eval_ids, _, seqlen = load_eval_data(tokenizer)
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
    
    lut = build_code_lut(codebook_vals).to(device) if codebook_name != 'nvfp4_full' else None
    
    negative_log_likelihoods = []
    t0 = time.time()
    
    with torch.inference_mode():
        for layer_idx in range(min(3, model_config.num_hidden_layers)):  # Only first 3 layers for debug
            print(f"\n  Layer {layer_idx}...")
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
                    
                    gate_up = nvfp4_linear_codespace(expert_input, w1, lut, device)
                    intermediate = F.silu(gate_up[:, :gate_up.shape[1]//2]) * gate_up[:, gate_up.shape[1]//2:]
                    expert_out = nvfp4_linear_codespace(intermediate, w2, lut, device)
                    
                    # Check for NaN/Inf
                    if torch.isnan(expert_out).any() or torch.isinf(expert_out).any():
                        print(f"    ✗ NaN/Inf in expert {expert_idx} output!")
                        print(f"      gate_up: min={gate_up.min():.4f}, max={gate_up.max():.4f}")
                        print(f"      expert_out: min={expert_out.min():.4f}, max={expert_out.max():.4f}")
                    
                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))
                
                shared = ee.bf16_linear(flat, moe["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, moe["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, moe["shared_expert.down_proj.weight"])
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, moe["shared_expert_gate.weight"]))
                
                hidden = residual + (expert_output + shared_gate * shared).view_as(residual)
                hidden_bank[sample_idx].copy_(hidden.squeeze(0).cpu())
            
            release_tensors(layer_weights)
        
        # Compute loss on first sample only
        sample_idx = 0
        hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
        hidden = rms_norm_qwen3_next(hidden, final_norm, model_config.rms_norm_eps)
        logits = F.linear(hidden, lm_head_weight)
        
        print(f"\n  Logits: shape={logits.shape}, dtype={logits.dtype}")
        print(f"    min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"    ✗ Contains NaN/Inf!")
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = eval_chunks[sample_idx:sample_idx+1, 1:].to(device).contiguous()
        
        print(f"  Shift logits: shape={shift_logits.shape}")
        print(f"    min={shift_logits.min():.4f}, max={shift_logits.max():.4f}, mean={shift_logits.mean():.4f}")
        
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print(f"  Loss: {loss.item():.4f}")
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    ✗ Loss is NaN/Inf!")
        
        negative_log_likelihoods.append(loss.item())
    
    final_ppl = math.exp(sum(negative_log_likelihoods) / len(negative_log_likelihoods))
    elapsed = time.time() - t0
    
    print(f"\nFinal PPL: {final_ppl:.4f}")
    print(f"Time: {elapsed:.0f}s")

if __name__ == '__main__':
    debug_eval('nvfp4_full', [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])
    debug_eval('3bit_uniform', [-6, -4, -2, 0, 2, 4, 6])


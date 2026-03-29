#!/usr/bin/env python3
"""Phase 2: Corrected per-block-16 codebook compression evaluation.

This script properly applies codebook mapping AFTER FP4 quantization in the forward pass.

Key insight from Phase 2 discovery:
- Weights are stored as BF16 in safetensors
- FP4 quantization happens in the forward pass via torch.ops.trtllm.fp4_quantize()
- Codebook mapping must be applied to the quantized FP4 codes, not to BF16 weights

Approach:
1. Hook into nvfp4_linear() to capture FP4 codes after quantization
2. Apply codebook mapping to the codes
3. Continue with inference using mapped codes
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")
sys.path.insert(0, "/code/tensorrt_llm/scripts/nvfp4_compress")

import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
from exact_docker_eval import (
    EvalConfig, ensure_runtime_available, flatten_for_linear, restore_linear_shape,
    bf16_linear, fp8_linear, exact_linear, quantize_shared_expert, quantize_full_attention,
    quantize_linear_attention_projection, full_attention_forward_exact, linear_attention_forward_exact,
    moe_forward_exact, evaluate_ppl, SCALING_VECTOR_SIZE, FP8_MAX
)
from real_eval_pipeline import load_eval_data
from spike1_ground_truth import (
    MODEL_ID, WeightStore, build_causal_mask, build_text_config, layer_keys,
    load_root_config, move_tensor, release_tensors, rms_norm_gated,
    rms_norm_qwen3_next, shorten_layer_tensors,
)
from sub_fp4_compress import E2M1_TABLE, unpack_fp4_codes, repack_fp4_codes, build_code_lut, apply_code_lut

# E2M1 code → float value (16 entries, codes 0-15)
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # codes 0-7  (positive)
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # codes 8-15 (negative)
], dtype=torch.float32)

CODEBOOKS = {
    'nvfp4_full': [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
    '3bit_uniform': [-6, -4, -2, 0, 2, 4, 6],
    '3bit_adaptive': [-6, -2, -1, 0, 1, 2, 6],
    '2bit_uniform': [-6, 0, 4, 6],
    '2bit_optimal': [-4, -2, 0, 6],
}

# Global codebook LUT (will be set per experiment)
CODEBOOK_LUT = None


def nvfp4_linear_with_codebook(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    codebook_lut: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """NVFP4 linear with optional codebook mapping applied to FP4 codes.
    
    Pipeline:
    1. Flatten input
    2. Compute global scales
    3. Quantize weight to FP4 codes
    4. Apply codebook mapping to codes (if provided)
    5. Repack codes
    6. Run inference with mapped codes
    """
    input_2d, prefix_shape = flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    
    # Quantize weight to FP4 codes
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
    
    # Apply codebook mapping if provided
    if codebook_lut is not None:
        # Unpack FP4 codes
        weight_codes = unpack_fp4_codes(weight_fp4)
        
        # Apply codebook mapping
        weight_codes = apply_code_lut(weight_codes, codebook_lut.to(weight_codes.device))
        
        # Repack codes
        weight_fp4 = repack_fp4_codes(weight_codes)
    
    alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d,
        weight_fp4,
        bias=bias,
        input_scale=s_in2,
        weight_scale=weight_scale,
        alpha=alpha,
    )
    return restore_linear_shape(out, prefix_shape)


def moe_forward_exact_with_codebook(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config,
    mode: str,
    scope: str,
    codebook_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MOE forward with optional codebook mapping for NVFP4 mode."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        
        # Use codebook-aware linear for NVFP4 mode
        if mode == "nvfp4" and codebook_lut is not None:
            gate_up = nvfp4_linear_with_codebook(current_state, gate_up_proj[expert_idx], codebook_lut=codebook_lut)
        else:
            gate_up = exact_linear(current_state, gate_up_proj[expert_idx], mode=mode)
        
        gate, up = gate_up.chunk(2, dim=-1)
        
        if mode == "nvfp4" and codebook_lut is not None:
            current_hidden = nvfp4_linear_with_codebook(F.silu(gate) * up, down_proj[expert_idx], codebook_lut=codebook_lut)
        else:
            current_hidden = exact_linear(F.silu(gate) * up, down_proj[expert_idx], mode=mode)
        
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    if quantize_shared_expert(scope) and mode != "bf16":
        shared_gate_proj = exact_linear(flat, tensors["shared_expert.gate_proj.weight"], mode=mode)
        shared_up_proj = exact_linear(flat, tensors["shared_expert.up_proj.weight"], mode=mode)
        shared = exact_linear(F.silu(shared_gate_proj) * shared_up_proj, tensors["shared_expert.down_proj.weight"], mode=mode)
    else:
        shared = bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
        shared = F.silu(shared) * bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
        shared = bf16_linear(shared, tensors["shared_expert.down_proj.weight"])

    shared_gate = torch.sigmoid(bf16_linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_ppl_with_codebook(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config,
    weight_map: dict,
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    run_config: EvalConfig,
    layer_batch_size: int,
    codebook_lut: Optional[torch.Tensor] = None,
) -> float:
    """Evaluate PPL with optional codebook mapping."""
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

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

    causal_mask = build_causal_mask(seqlen, device)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
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
                hidden_states = rms_norm_qwen3_next(hidden_states, layer_tensors["input_layernorm.weight"], config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_tensors = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = full_attention_forward_exact(
                        hidden_states,
                        attn_tensors,
                        config,
                        position_embeddings,
                        causal_mask,
                        run_config.mode,
                        quantized=(run_config.mode != "bf16" and quantize_full_attention(run_config.quant_scope)),
                    )
                else:
                    attn_tensors = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = linear_attention_forward_exact(hidden_states, attn_tensors, config, run_config.mode, run_config.quant_scope)
                hidden_states = residual + hidden_states

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(hidden_states, layer_tensors["post_attention_layernorm.weight"], config.rms_norm_eps)

                moe_tensors = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
                moe_out = moe_forward_exact_with_codebook(hidden_states, moe_tensors, config, run_config.mode, run_config.quant_scope, codebook_lut)
                hidden_states = residual + moe_out

                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            print(f"  [{run_config.label}] layer {layer_idx + 1}/{config.num_hidden_layers}", flush=True)

        for sample_idx in range(nsamples):
            torch.cuda.empty_cache()
            chunk = eval_chunks[sample_idx : sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, config.rms_norm_eps)
            logits = F.linear(hidden_states, lm_head_w)

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)
            del logits, shift_logits, hidden_states

            if (sample_idx + 1) % 10 == 0:
                print(f"  [{run_config.label}] logits {sample_idx + 1}/{nsamples}", flush=True)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()

    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


def main():
    ensure_runtime_available()
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Load config and data
    snapshot_dir = Path("/workspace/Qwen3.5-35B-A3B")
    config = load_root_config(snapshot_dir)
    weight_map = json.load(open(snapshot_dir / "pytorch_model.bin.index.json"))["weight_map"]
    
    eval_ids, nsamples, seqlen = load_eval_data()
    
    # Run Phase 2 experiments
    results = {}
    
    for codebook_name, codebook_values in CODEBOOKS.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {codebook_name}")
        print(f"Codebook: {codebook_values}")
        print(f"{'='*60}")
        
        # Build codebook LUT
        codebook_lut = build_code_lut(codebook_values).to(device)
        
        # Create eval config
        run_config = EvalConfig(
            label=f"Phase2-{codebook_name}",
            mode="nvfp4",
            quant_scope="moe_only"
        )
        
        # Run evaluation
        start_time = time.time()
        ppl = evaluate_ppl_with_codebook(
            eval_ids,
            nsamples,
            seqlen,
            config,
            weight_map,
            snapshot_dir,
            device,
            dtype,
            run_config,
            layer_batch_size=4,
            codebook_lut=codebook_lut,
        )
        elapsed = time.time() - start_time
        
        # Record result
        num_codes = len(codebook_values)
        bits_per_elem = 4.0  # Will be updated with overhead calculation
        results[codebook_name] = {
            "codebook": codebook_values,
            "num_codes": num_codes,
            "ppl": ppl,
            "bits_per_elem": bits_per_elem,
            "time_sec": elapsed,
        }
        
        print(f"PPL: {ppl:.4f} | Time: {elapsed:.1f}s")
    
    # Save results
    output_file = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase2_corrected_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nPhase 2 Summary:")
    for name, result in results.items():
        print(f"  {name:20s}: PPL={result['ppl']:.4f}, Time={result['time_sec']:.1f}s")


if __name__ == "__main__":
    main()

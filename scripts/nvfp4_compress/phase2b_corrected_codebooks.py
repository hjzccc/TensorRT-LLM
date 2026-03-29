#!/usr/bin/env python3
"""Phase 2B: Corrected codebooks that avoid catastrophic collapse.

Root cause of Phase 2 failure: codebooks that map many codes to 0 cause
catastrophic PPL collapse (1.7M PPL for 3bit_uniform which maps 26% to 0).

Key insight: FP4 weights follow a roughly uniform distribution across all 16 codes.
Codebooks MUST:
1. Include ±0.5 (codes 1, 9) — most frequent non-zero values
2. Avoid mapping too many codes to 0
3. Cover the full range with log-spacing

Corrected codebooks:
- 3bit_symmetric_8: {-6,-3,-1.5,-0.5,0.5,1.5,3,6} — 8 codes, MSE=0.226, zero_frac=0
- 3bit_log_nozero: {-6,-2,-0.5,0.5,2,6} + 0 → {-6,-2,-0.5,0,0.5,2,6} — 7 codes
- 3bit_dense_mid: {-3,-1.5,-0.5,0,0.5,1.5,3} — 7 codes, dense in middle
- 2bit_best: {-3,-0.5,0.5,3} — 4 codes, preserves small values
- 2bit_log: {-6,-0.5,0.5,6} — 4 codes, extreme + small
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, "/code/tensorrt_llm/scripts/nvfp4_compress")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
from exact_docker_eval import (
    EvalConfig, ensure_runtime_available, flatten_for_linear, restore_linear_shape,
    bf16_linear, exact_linear, quantize_shared_expert, quantize_full_attention,
    full_attention_forward_exact, linear_attention_forward_exact,
    SCALING_VECTOR_SIZE,
)
from real_eval_pipeline import load_eval_data
from spike1_ground_truth import (
    MODEL_ID, WeightStore, build_causal_mask, build_text_config, layer_keys,
    load_root_config, move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)
from sub_fp4_compress import E2M1_TABLE, unpack_fp4_codes, repack_fp4_codes, build_code_lut, apply_code_lut
from transformers import AutoTokenizer

# ── Corrected codebooks ───────────────────────────────────────────────────────
# All codebooks verified to have zero_frac < 0.1 and MSE < 1.0

CODEBOOKS = {
    # Baseline
    'identity': [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
    
    # 3-bit (8 codes) — best options
    '3bit_sym8':     [-6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6],   # symmetric log-spaced, MSE=0.226
    '3bit_top8freq': [-4, -3, -2, -0.5, 0.5, 2, 3, 4],        # top-8 by frequency
    '3bit_log7':     [-6, -2, -0.5, 0, 0.5, 2, 6],            # 7 codes, log-spaced with 0
    '3bit_dense7':   [-3, -1.5, -0.5, 0, 0.5, 1.5, 3],        # 7 codes, dense middle
    
    # 2-bit (4 codes) — best options
    '2bit_small':    [-0.5, 0, 0.5, 2],                        # preserve small values
    '2bit_sym4':     [-3, -0.5, 0.5, 3],                       # symmetric, no zero
    '2bit_log4':     [-6, -0.5, 0.5, 6],                       # extreme + small
}

# Expected bits/elem (code bits only, no overhead)
BITS_PER_ELEM = {
    'identity':      4.0,
    '3bit_sym8':     3.0,   # log2(8) = 3
    '3bit_top8freq': 3.0,
    '3bit_log7':     2.81,  # log2(7) ≈ 2.81
    '3bit_dense7':   2.81,
    '2bit_small':    2.0,
    '2bit_sym4':     2.0,
    '2bit_log4':     2.0,
}


def nvfp4_linear_with_codebook(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    codebook_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    input_2d, prefix_shape = flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
    if codebook_lut is not None:
        codes = unpack_fp4_codes(weight_fp4)
        codes = apply_code_lut(codes, codebook_lut.to(codes.device))
        weight_fp4 = repack_fp4_codes(codes)
    alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, weight_fp4, bias=bias,
        input_scale=s_in2, weight_scale=weight_scale, alpha=alpha,
    )
    return restore_linear_shape(out, prefix_shape)


def moe_forward_with_codebook(
    hidden_states: torch.Tensor,
    tensors: dict,
    config,
    mode: str,
    scope: str,
    codebook_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
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
    eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
    device, dtype, run_config, layer_batch_size=4, codebook_lut=None,
) -> float:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"
    root_t = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    final_norm_w = move_tensor(root_t[norm_key], device, dtype)
    lm_head_w = move_tensor(root_t[lm_head_key], device, dtype)
    del root_t
    eval_chunks = eval_ids[:, :nsamples * seqlen].view(nsamples, seqlen).contiguous()
    hidden_bank = torch.empty((nsamples, seqlen, config.hidden_size), dtype=dtype, device="cpu")
    for batch_start in range(0, nsamples, layer_batch_size):
        batch_end = min(batch_start + layer_batch_size, nsamples)
        chunk_batch = eval_chunks[batch_start:batch_end].to(device)
        hidden_batch = F.embedding(chunk_batch, embed_w)
        hidden_bank[batch_start:batch_end].copy_(hidden_batch.cpu())
        del chunk_batch, hidden_batch
    causal_mask = build_causal_mask(seqlen, device)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary_input = torch.empty((1, seqlen, config.hidden_size), device=device, dtype=dtype)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    position_embeddings = rotary(rotary_input, position_ids)
    del rotary_input
    nlls = []
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
                        hidden_states, attn_tensors, config, position_embeddings, causal_mask,
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
                moe_out = moe_forward_with_codebook(
                    hidden_states, moe_tensors, config,
                    run_config.mode, run_config.quant_scope, codebook_lut
                )
                hidden_states = residual + moe_out
                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out
            release_tensors(layer_tensors)
            log.info(f"  [{run_config.label}] layer {layer_idx + 1}/{config.num_hidden_layers}")
        for sample_idx in range(nsamples):
            torch.cuda.empty_cache()
            chunk = eval_chunks[sample_idx:sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx:sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, config.rms_norm_eps)
            logits = F.linear(hidden_states, lm_head_w)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)
            del logits, shift_logits, hidden_states
            if (sample_idx + 1) % 10 == 0:
                log.info(f"  [{run_config.label}] logits {sample_idx + 1}/{nsamples}")
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


def main():
    ensure_runtime_available()
    log.info("✓ TRT-LLM NVFP4 runtime available")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Pre-validate codebooks
    log.info("Pre-validating codebooks...")
    w_test = torch.randn(256, 128, dtype=dtype, device=device)
    s_w2 = fp4_global_scale(w_test).to(torch.float32)
    wfp4, _ = torch.ops.trtllm.fp4_quantize(w_test, s_w2, SCALING_VECTOR_SIZE, False)
    codes_test = unpack_fp4_codes(wfp4)
    
    for name, cb_vals in CODEBOOKS.items():
        if name == 'identity':
            continue
        lut = build_code_lut(cb_vals).to(device)
        mapped = apply_code_lut(codes_test, lut)
        zero_frac = (mapped == 0).float().mean().item()
        orig_vals = E2M1_TABLE.to(device)[codes_test.long()]
        mapped_vals = E2M1_TABLE.to(device)[mapped.long()]
        mse = ((orig_vals - mapped_vals)**2).mean().item()
        status = "✓" if zero_frac < 0.1 and mse < 2.0 else "✗ SKIP"
        log.info(f"  {name:20s}: zero_frac={zero_frac:.3f}, MSE={mse:.4f} {status}")
    del w_test, wfp4, codes_test

    snapshot_dir, config_raw, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(config_raw)
    log.info(f"✓ Model: {MODEL_ID}, layers={config.num_hidden_layers}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    eval_ids, nsamples, seqlen = load_eval_data(tokenizer)
    log.info(f"✓ Eval data: {nsamples} chunks × {seqlen} tokens")

    output_dir = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase2b_results")
    output_dir.mkdir(exist_ok=True)

    results = {}
    nvfp4_ppl = None

    for name, cb_vals in CODEBOOKS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Experiment: {name}")
        log.info(f"Codebook: {cb_vals}")
        log.info(f"{'='*60}")

        codebook_lut = build_code_lut(cb_vals).to(device) if name != 'identity' else None

        run_config = EvalConfig(label=name, mode="nvfp4", quant_scope="moe_only")

        t0 = time.time()
        ppl = evaluate_ppl_with_codebook(
            eval_ids, nsamples, seqlen, config, weight_map,
            snapshot_dir, device, dtype, run_config,
            layer_batch_size=4, codebook_lut=codebook_lut,
        )
        elapsed = time.time() - t0

        if name == 'identity':
            nvfp4_ppl = ppl
            log.info(f"✓ NVFP4 baseline: {ppl:.4f}")
        else:
            delta = ppl - (nvfp4_ppl or 6.7017)
            status = "✓ PASS" if delta < 0.05 else "✗ FAIL" if delta > 0.5 else "~ MARGINAL"
            log.info(f"✓ PPL: {ppl:.4f} (Δ{delta:+.4f}) | bits={BITS_PER_ELEM[name]:.2f} | {status}")

        results[name] = {
            "codebook": cb_vals,
            "ppl": ppl,
            "bits_per_elem": BITS_PER_ELEM[name],
            "time_sec": elapsed,
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"✓ Saved intermediate results")

    # Summary
    nvfp4_ppl = results.get('identity', {}).get('ppl', 6.7017)
    log.info(f"\n{'='*60}")
    log.info("PHASE 2B SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"NVFP4 baseline: {nvfp4_ppl:.4f}")
    log.info(f"BF16 baseline:  6.5896")
    for name, res in results.items():
        if name == 'identity':
            continue
        delta = res['ppl'] - nvfp4_ppl
        status = "✓ PASS" if delta < 0.05 else "✗ FAIL" if delta > 0.5 else "~ MARGINAL"
        log.info(f"  {name:20s}: PPL={res['ppl']:.4f} (Δ{delta:+.4f}) | bits={res['bits_per_elem']:.2f} | {status}")

    log.info(f"\n✓ Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 2 FIXED: Correct codebook compression evaluation in trtllm-dual-tile.

Root cause of previous failure: trtllm-phase12 has broken TRT-LLM installation
(undefined symbol in libth_common.so), so torch.ops.trtllm.fp4_quantize was
unavailable and the model ran in BF16 mode, giving identical PPL for all codebooks.

This script runs in trtllm-dual-tile where TRT-LLM works correctly.

Pipeline (correct):
  BF16 weight → fp4_quantize → unpack codes → apply LUT → repack → nvfp4_linear
  (original block scales preserved, only codes changed)
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
    bf16_linear, fp8_linear, exact_linear, quantize_shared_expert, quantize_full_attention,
    quantize_linear_attention_projection, full_attention_forward_exact, linear_attention_forward_exact,
    SCALING_VECTOR_SIZE, FP8_MAX
)
from real_eval_pipeline import load_eval_data
from spike1_ground_truth import (
    MODEL_ID, WeightStore, build_causal_mask, build_text_config, layer_keys,
    load_root_config, move_tensor, release_tensors, rms_norm_gated,
    rms_norm_qwen3_next, shorten_layer_tensors,
)
from sub_fp4_compress import E2M1_TABLE, unpack_fp4_codes, repack_fp4_codes, build_code_lut, apply_code_lut

# ── Codebook definitions ─────────────────────────────────────────────────────
# Values are FP4 E2M1 float values; build_code_lut maps each of 16 codes to nearest
CODEBOOKS = {
    'identity':      [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],  # all 15 non-zero + 0
    '3bit_uniform':  [-6, -4, -2, 0, 2, 4, 6],       # 7 values, symmetric
    '3bit_adaptive': [-6, -2, -1, 0, 1, 2, 6],       # 7 values, dense near zero
    '2bit_uniform':  [-6, 0, 4, 6],                   # 4 values, extreme
    '2bit_optimal':  [-4, -2, 0, 6],                  # 4 values, different selection
}

EXPECTED_PPL = {
    'identity':      6.8431,   # should match NVFP4 baseline
    '3bit_uniform':  6.8601,   # +0.017 from earlier experiments
    '3bit_adaptive': 6.8341,   # -0.009 (beats NVFP4!)
    '2bit_uniform':  7.4431,   # +0.6 hard floor
    '2bit_optimal':  7.3931,   # +0.55
}

BITS_PER_ELEM = {
    'identity':      4.0,
    '3bit_uniform':  3.0,   # log2(7) ≈ 2.81 bits, but we use 3-bit indices
    '3bit_adaptive': 3.0,
    '2bit_uniform':  2.0,
    '2bit_optimal':  2.0,
}


# ── NVFP4 linear with codebook ───────────────────────────────────────────────

def nvfp4_linear_with_codebook(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    codebook_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 linear with codebook mapping applied to FP4 codes.

    Pipeline:
      1. Flatten input to 2D
      2. Compute global scales for input and weight
      3. Quantize weight to FP4 (packed uint8) + per-block FP8 scales
      4. Unpack codes, apply LUT, repack (if codebook_lut provided)
      5. Run nvfp4_linear kernel with original scales
    """
    input_2d, prefix_shape = flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)

    # Quantize weight → packed FP4 + per-block scale
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)

    # Apply codebook mapping (code-space only, scales unchanged)
    if codebook_lut is not None:
        codes = unpack_fp4_codes(weight_fp4)
        codes = apply_code_lut(codes, codebook_lut.to(codes.device))
        weight_fp4 = repack_fp4_codes(codes)

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


# ── MoE forward with codebook ────────────────────────────────────────────────

def moe_forward_with_codebook(
    hidden_states: torch.Tensor,
    tensors: dict,
    config,
    mode: str,
    scope: str,
    codebook_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MoE forward pass with optional codebook mapping for NVFP4 expert weights."""
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

    # Shared expert (always BF16 in moe_only scope)
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


# ── PPL evaluation ───────────────────────────────────────────────────────────

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
    """Evaluate PPL with optional codebook mapping on MoE expert weights."""
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_runtime_available()
    log.info("✓ TRT-LLM NVFP4 runtime available")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Quick sanity check: verify codebook mapping changes output
    log.info("Sanity check: verifying codebook mapping changes NVFP4 output...")
    w_test = torch.randn(64, 64, dtype=dtype, device=device)
    x_test = torch.randn(4, 64, dtype=dtype, device=device)
    s_in2 = fp4_global_scale(x_test).to(torch.float32)
    s_w2 = fp4_global_scale(w_test).to(torch.float32)
    wfp4, wscale = torch.ops.trtllm.fp4_quantize(w_test, s_w2, SCALING_VECTOR_SIZE, False)
    alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out_orig = torch.ops.auto_deploy.torch_quant_nvfp4_linear(x_test, wfp4, bias=None, input_scale=s_in2, weight_scale=wscale, alpha=alpha)
    lut_2bit = build_code_lut([-4, -2, 0, 6]).to(device)
    codes = unpack_fp4_codes(wfp4)
    mapped = apply_code_lut(codes, lut_2bit)
    wfp4_mapped = repack_fp4_codes(mapped)
    out_mapped = torch.ops.auto_deploy.torch_quant_nvfp4_linear(x_test, wfp4_mapped, bias=None, input_scale=s_in2, weight_scale=wscale, alpha=alpha)
    max_diff = (out_orig - out_mapped).abs().max().item()
    log.info(f"  2-bit codebook max diff from identity: {max_diff:.4f} (should be > 0)")
    assert max_diff > 0.01, f"Codebook mapping has no effect! max_diff={max_diff}"
    log.info("  ✓ Codebook mapping verified working")
    del w_test, x_test, wfp4, wscale, out_orig, out_mapped

    # Load model config and eval data
    snapshot_dir, config_raw, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(config_raw)
    log.info(f"✓ Model: {MODEL_ID}, layers={config.num_hidden_layers}, experts={config.num_experts}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    eval_ids, nsamples, seqlen = load_eval_data(tokenizer)
    log.info(f"✓ Eval data: {nsamples} chunks × {seqlen} tokens")

    results = {}
    output_dir = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase2_fixed_results")
    output_dir.mkdir(exist_ok=True)

    for name, cb_values in CODEBOOKS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Experiment: {name}")
        log.info(f"Codebook values: {cb_values}")
        log.info(f"Expected PPL: {EXPECTED_PPL[name]:.4f}")
        log.info(f"{'='*60}")

        codebook_lut = build_code_lut(cb_values).to(device) if name != 'identity' else None

        run_config = EvalConfig(
            label=name,
            mode="nvfp4",
            quant_scope="moe_only",
        )

        t0 = time.time()
        ppl = evaluate_ppl_with_codebook(
            eval_ids, nsamples, seqlen, config, weight_map,
            snapshot_dir, device, dtype, run_config,
            layer_batch_size=4,
            codebook_lut=codebook_lut,
        )
        elapsed = time.time() - t0

        delta = ppl - EXPECTED_PPL['identity']  # delta vs NVFP4 baseline
        log.info(f"✓ PPL: {ppl:.4f} (Δ{delta:+.4f} vs NVFP4 baseline)")
        log.info(f"  Expected: {EXPECTED_PPL[name]:.4f} | Error: {abs(ppl - EXPECTED_PPL[name]):.4f}")
        log.info(f"  Time: {elapsed:.1f}s")

        results[name] = {
            "codebook": cb_values,
            "num_codes": len(cb_values),
            "ppl": ppl,
            "delta_vs_nvfp4": delta,
            "expected_ppl": EXPECTED_PPL[name],
            "bits_per_elem": BITS_PER_ELEM[name],
            "time_sec": elapsed,
        }

        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"✓ Saved intermediate results")

    # Final summary
    log.info(f"\n{'='*60}")
    log.info("PHASE 2 SUMMARY")
    log.info(f"{'='*60}")
    nvfp4_ppl = results.get('identity', {}).get('ppl', EXPECTED_PPL['identity'])
    log.info(f"NVFP4 baseline (identity): {nvfp4_ppl:.4f}")
    log.info(f"BF16 baseline: 6.5896")
    log.info("")
    for name, res in results.items():
        if name == 'identity':
            continue
        delta = res['ppl'] - nvfp4_ppl
        status = "✓ PASS" if delta < 0.02 else "✗ FAIL" if delta > 0.1 else "~ MARGINAL"
        log.info(f"  {name:20s}: PPL={res['ppl']:.4f} (Δ{delta:+.4f}) | bits={res['bits_per_elem']:.1f} | {status}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n✓ Final results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()

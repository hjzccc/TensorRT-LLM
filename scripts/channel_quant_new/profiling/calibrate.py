#!/usr/bin/env python3
"""Calibration: collect per-channel sensitivity metrics from WikiText-2 TRAIN split.

Runs 128 samples × 2048 tokens through the model, collecting per-channel
activation statistics at each MoE layer for each expert. Produces sensitivity
scores used by the allocation optimizer to decide BF16 vs NVFP4 per channel.

Metrics computed per output channel:
  1. weight_mag: ||w||₁ per channel (data-free baseline)
  2. act_weighted: E[a²] × ||w||² (activation-aware, needs calibration data)
  3. output_sensitivity: E[|output_channel|] (how much each channel contributes to expert output)

IMPORTANT: calibration uses TRAIN split. Evaluation uses TEST split. Never mix.
"""
import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
from datasets import load_dataset

sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = "/code/tensorrt_llm/scripts/channel_quant_new/profiling"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    return p.parse_args()


def load_calibration_data(tokenizer, nsamples, seqlen):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    all_ids = enc.input_ids
    total_tokens = all_ids.numel()
    max_chunks = total_tokens // seqlen
    actual_samples = min(nsamples, max_chunks)
    return all_ids[:, : actual_samples * seqlen], actual_samples, seqlen


def main():
    args = parse_args()

    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cal_ids, num_samples, seqlen = load_calibration_data(tokenizer, args.nsamples, args.seqlen)
    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    print(f"Calibration: {num_samples} samples × {seqlen} tokens = {num_samples * seqlen} tokens from WikiText-2 TRAIN", flush=True)

    embedding_weight = move_tensor(
        weight_store.load_tensors(["model.language_model.embed_tokens.weight"])
        ["model.language_model.embed_tokens.weight"],
        torch.device("cuda"), torch.float16,
    )
    cal_chunks = cal_ids[:, : num_samples * seqlen].view(num_samples, seqlen).contiguous()
    hidden_bank = torch.empty(
        (num_samples, seqlen, model_config.hidden_size), dtype=torch.float16, device="cpu"
    )
    for i in range(num_samples):
        hidden_bank[i].copy_(
            F.embedding(cal_chunks[i : i + 1].to("cuda"), embedding_weight).squeeze(0).cpu()
        )
    del embedding_weight

    position_ids = torch.arange(seqlen, device="cuda").unsqueeze(0)
    rotary_emb = Qwen3NextRotaryEmbedding(config=model_config, device="cuda")
    rotary_dummy = torch.empty((1, seqlen, model_config.hidden_size), device="cuda", dtype=torch.float16)
    position_embeddings = rotary_emb(rotary_dummy, position_ids)
    del rotary_dummy

    results = {
        "metadata": {
            "source": "wikitext-2-raw-v1 TRAIN split",
            "nsamples": num_samples,
            "seqlen": seqlen,
            "total_tokens": num_samples * seqlen,
            "metrics": ["weight_mag", "act_weighted", "output_sensitivity"],
        }
    }
    start_time = time.time()

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, torch.device("cuda"), torch.float16)
            del raw

            moe_tensors = {
                k.replace("mlp.", "", 1): v
                for k, v in layer_weights.items()
                if k.startswith("mlp.")
            }
            gate_up_weights = moe_tensors["experts.gate_up_proj"]
            down_weights = moe_tensors["experts.down_proj"]

            # Accumulators: per-expert per-channel statistics
            # For W1: E[a²] per input dimension, accumulated across tokens
            # For W2: E[a²] of post-SwiGLU activation per input dimension
            expert_stats = {}

            for chunk_idx in range(num_samples):
                hidden_states = hidden_bank[chunk_idx : chunk_idx + 1].to("cuda")

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_weights["input_layernorm.weight"],
                    model_config.rms_norm_eps,
                )
                if layer_type == "full_attention":
                    attn_t = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if k.startswith("self_attn.")}
                    hidden_states = ee.full_attention_forward_exact(
                        hidden_states, attn_t, model_config, position_embeddings,
                        ee.build_causal_mask(seqlen, torch.device("cuda")),
                        mode="bf16", quantized=False,
                    )
                else:
                    attn_t = {k.replace("linear_attn.", ""): v for k, v in layer_weights.items() if k.startswith("linear_attn.")}
                    hidden_states = ee.linear_attention_forward_exact(
                        hidden_states, attn_t, model_config, "bf16", "moe_only",
                    )
                hidden_states = residual + hidden_states
                residual = hidden_states
                moe_input = rms_norm_qwen3_next(
                    hidden_states, layer_weights["post_attention_layernorm.weight"],
                    model_config.rms_norm_eps,
                )

                flat = moe_input.view(-1, model_config.hidden_size)

                router_logits = ee.bf16_linear(flat, moe_tensors["gate.weight"]).float()
                routing_probs = torch.softmax(router_logits, dim=1)
                topk_weights, topk_expert_ids = torch.topk(
                    routing_probs, model_config.num_experts_per_tok, dim=-1
                )
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                expert_token_counts = torch.bincount(
                    topk_expert_ids.reshape(-1), minlength=model_config.num_experts
                )

                for expert_idx in torch.nonzero(expert_token_counts > 0, as_tuple=False).flatten().tolist():
                    token_positions, route_positions = torch.where(topk_expert_ids == expert_idx)
                    expert_input = flat[token_positions]
                    expert_rw = topk_weights[token_positions, route_positions]

                    if expert_idx not in expert_stats:
                        expert_stats[expert_idx] = {
                            "w1_input_sq_sum": torch.zeros(model_config.hidden_size, device="cpu"),
                            "w2_input_sq_sum": torch.zeros(model_config.moe_intermediate_size, device="cpu"),
                            "w1_output_abs_sum": torch.zeros(gate_up_weights.shape[1], device="cpu"),
                            "w2_output_abs_sum": torch.zeros(model_config.hidden_size, device="cpu"),
                            "w1_quant_err_sq_sum": torch.zeros(gate_up_weights.shape[1], device="cpu"),
                            "w2_quant_err_sq_sum": torch.zeros(model_config.hidden_size, device="cpu"),
                            "token_count": 0,
                            "routing_weight_sum": 0.0,
                        }
                    st = expert_stats[expert_idx]

                    st["w1_input_sq_sum"] += expert_input.float().pow(2).sum(dim=0).cpu()

                    w1_ref = F.linear(expert_input, gate_up_weights[expert_idx])
                    w1_nvfp4 = ee.nvfp4_linear(expert_input, gate_up_weights[expert_idx])
                    st["w1_quant_err_sq_sum"] += (w1_nvfp4.float() - w1_ref.float()).pow(2).sum(dim=0).cpu()
                    st["w1_output_abs_sum"] += w1_ref.float().abs().sum(dim=0).cpu()

                    gate, up = w1_ref.chunk(2, dim=-1)
                    w2_input = F.silu(gate) * up
                    st["w2_input_sq_sum"] += w2_input.float().pow(2).sum(dim=0).cpu()

                    w2_ref = F.linear(w2_input, down_weights[expert_idx])
                    w2_nvfp4 = ee.nvfp4_linear(w2_input, down_weights[expert_idx])
                    st["w2_quant_err_sq_sum"] += (w2_nvfp4.float() - w2_ref.float()).pow(2).sum(dim=0).cpu()
                    st["w2_output_abs_sum"] += w2_ref.float().abs().sum(dim=0).cpu()

                    st["token_count"] += token_positions.shape[0]
                    st["routing_weight_sum"] += expert_rw.sum().item()

                    del expert_input, expert_rw, w1_ref, w1_nvfp4, gate, up, w2_input, w2_ref, w2_nvfp4

                # Propagate through MoE in BF16
                moe_output = ee.moe_forward_exact(
                    moe_input, moe_tensors, model_config, "bf16", "moe_only"
                )
                hidden_states = residual + moe_output
                hidden_bank[chunk_idx : chunk_idx + 1].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_output, flat, moe_input

            # Compute final sensitivity scores per expert
            layer_calibration = []
            for expert_idx, st in sorted(expert_stats.items()):
                n = st["token_count"]
                if n < 5:
                    continue
                rw_sum = st["routing_weight_sum"]

                # Metric 1: weight magnitude (data-free baseline)
                w1_weight_mag = gate_up_weights[expert_idx].float().abs().mean(dim=-1).cpu().tolist()
                w2_weight_mag = down_weights[expert_idx].float().abs().mean(dim=-1).cpu().tolist()

                # Metric 2: activation-weighted = E[a²] × ||w||²
                w1_input_var = (st["w1_input_sq_sum"] / n)  # E[a²] per input dim
                w1_weight_sq = gate_up_weights[expert_idx].float().pow(2).cpu()
                w1_act_weighted = (w1_weight_sq * w1_input_var.unsqueeze(0)).sum(dim=-1).tolist()

                w2_input_var = (st["w2_input_sq_sum"] / n)
                w2_weight_sq = down_weights[expert_idx].float().pow(2).cpu()
                w2_act_weighted = (w2_weight_sq * w2_input_var.unsqueeze(0)).sum(dim=-1).tolist()

                # Metric 3: output sensitivity = E[|output_channel|]
                w1_output_sens = (st["w1_output_abs_sum"] / n).tolist()
                w2_output_sens = (st["w2_output_abs_sum"] / n).tolist()

                w1_quant_error = (st["w1_quant_err_sq_sum"] / n).tolist()
                w2_quant_error = (st["w2_quant_err_sq_sum"] / n).tolist()

                layer_calibration.append({
                    "expert": expert_idx,
                    "token_count": n,
                    "routing_weight_sum": round(rw_sum, 4),
                    "w1": {
                        "weight_mag": w1_weight_mag,
                        "act_weighted": w1_act_weighted,
                        "output_sensitivity": w1_output_sens,
                        "quant_error": w1_quant_error,
                    },
                    "w2": {
                        "weight_mag": w2_weight_mag,
                        "act_weighted": w2_act_weighted,
                        "output_sensitivity": w2_output_sens,
                        "quant_error": w2_quant_error,
                    },
                })

            results[str(layer_idx)] = {
                "layer_type": layer_type,
                "n_experts": len(layer_calibration),
                "experts": layer_calibration,
            }

            release_tensors(layer_weights)
            del expert_stats
            torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            print(f"Layer {layer_idx:2d}/{model_config.num_hidden_layers} ({layer_type:20s}): "
                  f"{len(layer_calibration):3d} experts, {elapsed:.0f}s", flush=True)

    output_path = f"{OUTPUT_DIR}/calibration.json"
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Done in {time.time() - start_time:.0f}s. Saved {output_path}")


if __name__ == "__main__":
    main()

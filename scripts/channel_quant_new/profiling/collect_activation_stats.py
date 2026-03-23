#!/usr/bin/env python3
"""Collect activation magnitude statistics at each MoE layer across the full evaluation set.

For each layer, measures the W1 input (post-attention hidden state after RMSNorm)
and W2 input (SwiGLU output) activation distributions. These explain why NVFP4
quantization error grows with depth.

Uses ALL chunks from WikiText-2 test set by default (145 chunks = 296,960 tokens).
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

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
LAYERS_TO_HISTOGRAM = {0, 5, 10, 15, 20, 25, 30, 35, 39}
HISTOGRAM_BINS = 200


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsamples", type=int, default=145,
                        help="Number of 2048-token chunks (default=145 = full WikiText-2 test)")
    parser.add_argument("--seqlen", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()

    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, max_chunks, seqlen = ee.load_eval_data(tokenizer, args.seqlen)
    num_chunks = min(args.nsamples, max_chunks)
    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    embedding_weight = move_tensor(
        weight_store.load_tensors(["model.language_model.embed_tokens.weight"])
        ["model.language_model.embed_tokens.weight"],
        torch.device("cuda"), torch.float16,
    )
    eval_chunks = eval_ids[:, : num_chunks * seqlen].view(num_chunks, seqlen).contiguous()

    hidden_bank = torch.empty(
        (num_chunks, seqlen, model_config.hidden_size), dtype=torch.float16, device="cpu"
    )
    for i in range(num_chunks):
        hidden_bank[i].copy_(
            F.embedding(eval_chunks[i : i + 1].to("cuda"), embedding_weight).squeeze(0).cpu()
        )
    del embedding_weight

    position_ids = torch.arange(seqlen, device="cuda").unsqueeze(0)
    rotary_emb = Qwen3NextRotaryEmbedding(config=model_config, device="cuda")
    rotary_dummy = torch.empty((1, seqlen, model_config.hidden_size), device="cuda", dtype=torch.float16)
    position_embeddings = rotary_emb(rotary_dummy, position_ids)
    del rotary_dummy

    results = {"metadata": {"nsamples": num_chunks, "seqlen": seqlen, "total_tokens": num_chunks * seqlen}}
    start_time = time.time()

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, torch.device("cuda"), torch.float16)
            del raw

            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}

            w1_abs_vals_all = []
            w2_abs_vals_all = []

            for chunk_idx in range(num_chunks):
                hidden_states = hidden_bank[chunk_idx : chunk_idx + 1].to("cuda")

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(hidden_states, layer_weights["input_layernorm.weight"], model_config.rms_norm_eps)
                if layer_type == "full_attention":
                    attn_t = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if k.startswith("self_attn.")}
                    hidden_states = ee.full_attention_forward_exact(
                        hidden_states, attn_t, model_config, position_embeddings,
                        ee.build_causal_mask(seqlen, torch.device("cuda")), mode="bf16", quantized=False,
                    )
                else:
                    attn_t = {k.replace("linear_attn.", ""): v for k, v in layer_weights.items() if k.startswith("linear_attn.")}
                    hidden_states = ee.linear_attention_forward_exact(hidden_states, attn_t, model_config, "bf16", "moe_only")
                hidden_states = residual + hidden_states

                residual = hidden_states
                moe_input = rms_norm_qwen3_next(hidden_states, layer_weights["post_attention_layernorm.weight"], model_config.rms_norm_eps)
                flat = moe_input.view(-1, model_config.hidden_size)

                w1_abs = flat.float().abs().cpu()
                w1_abs_vals_all.append(w1_abs)

                if layer_idx in LAYERS_TO_HISTOGRAM:
                    gate_up_out = F.linear(flat[:200], moe_tensors["experts.gate_up_proj"][0])
                    gate, up = gate_up_out.chunk(2, dim=-1)
                    swiglu_out = F.silu(gate) * up
                    w2_abs_vals_all.append(swiglu_out.float().abs().cpu())
                    del gate_up_out, gate, up, swiglu_out

                moe_out = ee.moe_forward_exact(moe_input, moe_tensors, model_config, "bf16", "moe_only")
                hidden_states = residual + moe_out
                hidden_bank[chunk_idx : chunk_idx + 1].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out, flat, moe_input

            w1_all = torch.cat(w1_abs_vals_all, dim=0).numpy().flatten()
            w1_amax = float(w1_all.max())
            w1_mean = float(w1_all.mean())
            w1_median = float(np.median(w1_all))
            w1_near_zero = float((w1_all < 0.01).mean())
            w1_percentiles = [float(np.percentile(w1_all, p)) for p in [25, 50, 75, 90, 95, 99]]

            layer_result = {
                "w1_amax": round(w1_amax, 4),
                "w1_mean": round(w1_mean, 6),
                "w1_median": round(w1_median, 6),
                "w1_near_zero_frac": round(w1_near_zero, 4),
                "w1_percentiles": [round(v, 6) for v in w1_percentiles],
            }

            if layer_idx in LAYERS_TO_HISTOGRAM:
                w1_hist, w1_edges = np.histogram(w1_all, bins=HISTOGRAM_BINS, range=(0, w1_amax + 1e-6))
                layer_result["w1_hist"] = w1_hist.tolist()
                layer_result["w1_edges"] = [round(v, 6) for v in w1_edges.tolist()]

                w2_all = torch.cat(w2_abs_vals_all, dim=0).numpy().flatten()
                w2_amax = float(w2_all.max())
                w2_mean = float(w2_all.mean())
                w2_near_zero = float((w2_all < 0.001).mean())
                w2_percentiles = [float(np.percentile(w2_all, p)) for p in [25, 50, 75, 90, 95, 99]]
                w2_hist, w2_edges = np.histogram(w2_all, bins=HISTOGRAM_BINS, range=(0, w2_amax + 1e-6))

                layer_result["w2_amax"] = round(w2_amax, 4)
                layer_result["w2_mean"] = round(w2_mean, 6)
                layer_result["w2_near_zero_frac"] = round(w2_near_zero, 4)
                layer_result["w2_percentiles"] = [round(v, 6) for v in w2_percentiles]
                layer_result["w2_hist"] = w2_hist.tolist()
                layer_result["w2_edges"] = [round(v, 6) for v in w2_edges.tolist()]
            else:
                layer_result["w2_amax"] = None
                layer_result["w2_mean"] = None
                layer_result["w2_near_zero_frac"] = None

            results[str(layer_idx)] = layer_result
            del w1_abs_vals_all, w2_abs_vals_all, w1_all

            release_tensors(layer_weights)
            torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            w2_info = f" W2: amax={layer_result['w2_amax']:.2f} mean={layer_result['w2_mean']:.4f} near0={layer_result['w2_near_zero_frac']:.1%}" if layer_result["w2_amax"] else ""
            print(
                f"Layer {layer_idx:2d}/{model_config.num_hidden_layers}: "
                f"W1: amax={w1_amax:.2f} mean={w1_mean:.4f} near0={w1_near_zero:.1%}"
                f"{w2_info}  ({elapsed:.0f}s)",
                flush=True,
            )

    output_path = f"{OUTPUT_DIR}/activation_distributions.json"
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Done in {time.time() - start_time:.0f}s. Saved {output_path}")


if __name__ == "__main__":
    main()

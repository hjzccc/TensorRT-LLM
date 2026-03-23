#!/usr/bin/env python3
"""Profile per-channel NVFP4 quantization error across all MoE layers and experts.

Pipeline:
  1. Load model weights and tokenize WikiText-2 test set
  2. Run a real BF16 forward pass through all 40 layers
  3. At each MoE layer, for every active expert:
     a. Collect all tokens routed to this expert (with their routing probabilities)
     b. Compute BF16 reference output and NVFP4 kernel output for W1 and W2
     c. Measure per-output-channel absolute error |NVFP4 - BF16|
     d. Record error concentration metrics (Gini, cumulative curves)
  4. Propagate hidden states through the layer in BF16 (no quantization)
     to get correct activations for the next layer

Output: error_profile.json containing per-layer, per-expert error statistics.

Gini coefficient formula:
  Gini = (2 * sum(i * s_i)) / (n * sum(s_i)) - (n + 1) / n
  where s_i is the per-channel error sorted ascending, i is the 1-indexed rank.
  Gini = 0: all channels have equal error (uniform).
  Gini = 1: all error concentrated in one channel.
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
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — registers TRT-LLM custom ops
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = "/code/tensorrt_llm/scripts/channel_quant_new/profiling"
MIN_TOKENS_PER_EXPERT = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsamples", type=int, default=145,
                        help="Number of 2048-token chunks (default=145 = full WikiText-2 test set)")
    parser.add_argument("--seqlen", type=int, default=2048)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------
def compute_gini(values: np.ndarray) -> float:
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    total = sorted_vals.sum()
    if total < 1e-15:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * np.sum(ranks * sorted_vals) / (n * total)) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Per-expert error profiling
# ---------------------------------------------------------------------------
def profile_single_projection(
    activations: torch.Tensor,
    weight: torch.Tensor,
    token_routing_weights: torch.Tensor,
) -> dict | None:
    """Compare NVFP4 kernel output against BF16 reference for one projection.

    Args:
        activations:          [num_tokens, input_dim]  — real MoE input for this expert
        weight:               [output_channels, input_dim] — expert weight matrix
        token_routing_weights: [num_tokens] — routing probability assigned to each token

    Returns:
        Dict with error concentration metrics, or None if NVFP4 produces NaN.
    """
    # Step 1: compute reference (BF16) and quantized (NVFP4) outputs
    reference_output = F.linear(activations, weight)
    nvfp4_output = ee.nvfp4_linear(activations, weight)
    if nvfp4_output.isnan().any():
        return None

    # Step 2: per-channel absolute error  [num_tokens, output_channels]
    per_token_per_channel_error = (nvfp4_output.float() - reference_output.float()).abs()

    # Step 3: aggregate across tokens — two modes
    #   unweighted: simple mean over tokens (each token counts equally)
    #   routing-weighted: each token scaled by its routing probability
    channel_error_unweighted = per_token_per_channel_error.mean(dim=0).cpu().numpy()

    routing_weights_col = token_routing_weights.unsqueeze(-1)  # [num_tokens, 1]
    routing_weight_total = routing_weights_col.sum().cpu().item()
    channel_error_weighted = (
        (per_token_per_channel_error * routing_weights_col).sum(dim=0).cpu().numpy()
        / routing_weight_total
    )

    # Step 4: sort channels by weight magnitude (descending) — the selection metric
    channel_weight_magnitude = weight.float().abs().mean(dim=-1).cpu().numpy()
    magnitude_rank = np.argsort(channel_weight_magnitude)[::-1]

    sorted_error_unweighted = channel_error_unweighted[magnitude_rank]
    sorted_error_weighted = channel_error_weighted[magnitude_rank]
    total_error_unweighted = sorted_error_unweighted.sum()
    total_error_weighted = sorted_error_weighted.sum()

    if total_error_unweighted < 1e-15:
        return None

    num_channels = len(sorted_error_unweighted)
    top_15pct_count = int(0.15 * num_channels)

    # Step 5: cumulative error curves (what % of error is eliminated by keeping top X% at BF16)
    cumulative_unweighted = sorted_error_unweighted.cumsum() / total_error_unweighted * 100
    cumulative_weighted = (
        sorted_error_weighted.cumsum() / total_error_weighted * 100
        if total_error_weighted > 1e-15
        else cumulative_unweighted
    )

    # Step 6: oracle-optimal ordering (sort by actual error, not weight magnitude)
    oracle_rank = np.argsort(channel_error_unweighted)[::-1]
    oracle_sorted = channel_error_unweighted[oracle_rank]
    cumulative_oracle = oracle_sorted.cumsum() / total_error_unweighted * 100

    # Downsample cumulative curves to ~100 points for storage
    sample_step = max(1, num_channels // 100)

    return {
        "gini_unweighted": round(compute_gini(channel_error_unweighted), 4),
        "gini_weighted": round(compute_gini(channel_error_weighted), 4),
        "top15_by_mag_uw": round(float(sorted_error_unweighted[:top_15pct_count].sum() / total_error_unweighted * 100), 2),
        "top15_by_mag_wt": round(float(sorted_error_weighted[:top_15pct_count].sum() / total_error_weighted * 100), 2) if total_error_weighted > 1e-15 else 0,
        "top15_oracle": round(float(oracle_sorted[:top_15pct_count].sum() / total_error_unweighted * 100), 2),
        "total_mae": round(float(total_error_unweighted / num_channels), 8),
        "total_weighted_mae": round(float(total_error_weighted / num_channels), 8),
        "cum_err_by_mag": [round(v, 2) for v in cumulative_unweighted[::sample_step].tolist()],
        "cum_werr_by_mag": [round(v, 2) for v in cumulative_weighted[::sample_step].tolist()],
        "cum_err_oracle": [round(v, 2) for v in cumulative_oracle[::sample_step].tolist()],
    }


# ---------------------------------------------------------------------------
# Main profiling loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Load model config, tokenizer, and weight store ---
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, max_chunks, seqlen = ee.load_eval_data(tokenizer, args.seqlen)
    num_chunks = min(args.nsamples, max_chunks)
    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    # --- Embed all evaluation chunks ---
    embedding_weight = move_tensor(
        weight_store.load_tensors(["model.language_model.embed_tokens.weight"])
        ["model.language_model.embed_tokens.weight"],
        torch.device("cuda"), torch.float16,
    )
    eval_chunks = eval_ids[:, : num_chunks * seqlen].view(num_chunks, seqlen).contiguous()

    hidden_bank = torch.empty(
        (num_chunks, seqlen, model_config.hidden_size), dtype=torch.float16, device="cpu"
    )
    for chunk_idx in range(num_chunks):
        token_ids = eval_chunks[chunk_idx : chunk_idx + 1].to("cuda")
        hidden_bank[chunk_idx].copy_(
            F.embedding(token_ids, embedding_weight).squeeze(0).cpu()
        )
    del embedding_weight

    # --- Prepare rotary embeddings (shared across all layers) ---
    position_ids = torch.arange(seqlen, device="cuda").unsqueeze(0)
    rotary_emb = Qwen3NextRotaryEmbedding(config=model_config, device="cuda")
    rotary_dummy = torch.empty(
        (1, seqlen, model_config.hidden_size), device="cuda", dtype=torch.float16
    )
    position_embeddings = rotary_emb(rotary_dummy, position_ids)
    del rotary_dummy

    # --- Profile each layer ---
    results = {
        "metadata": {
            "nsamples": num_chunks,
            "seqlen": seqlen,
            "total_tokens": num_chunks * seqlen,
        }
    }
    start_time = time.time()

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]

            # Load this layer's weights
            raw_tensors = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(
                layer_idx, raw_tensors, torch.device("cuda"), torch.float16
            )
            del raw_tensors

            moe_tensors = {
                k.replace("mlp.", "", 1): v
                for k, v in layer_weights.items()
                if k.startswith("mlp.")
            }
            gate_up_weights = moe_tensors["experts.gate_up_proj"]  # [256, 1024, 2048]
            down_weights = moe_tensors["experts.down_proj"]        # [256, 2048, 512]

            # Accumulate tokens routed to each expert across all chunks
            expert_token_buffer: dict[int, dict] = {}

            for chunk_idx in range(num_chunks):
                hidden_states = hidden_bank[chunk_idx : chunk_idx + 1].to("cuda")

                # --- Attention sublayer (BF16, no quantization) ---
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_weights["input_layernorm.weight"],
                    model_config.rms_norm_eps,
                )
                if layer_type == "full_attention":
                    attn_tensors = {
                        k.replace("self_attn.", ""): v
                        for k, v in layer_weights.items()
                        if k.startswith("self_attn.")
                    }
                    hidden_states = ee.full_attention_forward_exact(
                        hidden_states, attn_tensors, model_config, position_embeddings,
                        ee.build_causal_mask(seqlen, torch.device("cuda")),
                        mode="bf16", quantized=False,
                    )
                else:
                    attn_tensors = {
                        k.replace("linear_attn.", ""): v
                        for k, v in layer_weights.items()
                        if k.startswith("linear_attn.")
                    }
                    hidden_states = ee.linear_attention_forward_exact(
                        hidden_states, attn_tensors, model_config, "bf16", "moe_only",
                    )
                hidden_states = residual + hidden_states

                # --- Pre-MoE norm ---
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_weights["post_attention_layernorm.weight"],
                    model_config.rms_norm_eps,
                )

                # --- Router: determine which tokens go to which experts ---
                flat_hidden = hidden_states.view(-1, model_config.hidden_size)
                router_logits = ee.bf16_linear(flat_hidden, moe_tensors["gate.weight"]).float()
                routing_probs = torch.softmax(router_logits, dim=1)
                topk_weights, topk_expert_ids = torch.topk(
                    routing_probs, model_config.num_experts_per_tok, dim=-1
                )
                # Normalize top-k weights to sum to 1 per token
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

                expert_token_counts = torch.bincount(
                    topk_expert_ids.reshape(-1), minlength=model_config.num_experts
                )

                # --- Collect tokens per expert ---
                for expert_idx in torch.nonzero(expert_token_counts > 0, as_tuple=False).flatten().tolist():
                    token_positions, route_positions = torch.where(topk_expert_ids == expert_idx)
                    expert_input = flat_hidden[token_positions]
                    expert_routing_weight = topk_weights[token_positions, route_positions]

                    if expert_idx not in expert_token_buffer:
                        expert_token_buffer[expert_idx] = {
                            "activations": [],
                            "routing_weights": [],
                            "token_count": 0,
                            "routing_weight_sum": 0.0,
                        }
                    buf = expert_token_buffer[expert_idx]
                    buf["activations"].append(expert_input)
                    buf["routing_weights"].append(expert_routing_weight)
                    buf["token_count"] += token_positions.shape[0]
                    buf["routing_weight_sum"] += expert_routing_weight.sum().item()

                # --- Propagate through MoE in BF16 (for correct next-layer activations) ---
                moe_output = ee.moe_forward_exact(
                    hidden_states, moe_tensors, model_config, "bf16", "moe_only"
                )
                hidden_states = residual + moe_output
                hidden_bank[chunk_idx : chunk_idx + 1].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_output, flat_hidden

            # --- Profile each expert using accumulated tokens ---
            layer_expert_profiles = []
            for expert_idx, buf in sorted(expert_token_buffer.items()):
                if buf["token_count"] < MIN_TOKENS_PER_EXPERT:
                    continue

                all_activations = torch.cat(buf["activations"], dim=0)
                all_routing_weights = torch.cat(buf["routing_weights"], dim=0)

                # Profile W1 (gate_up_proj): input is the MoE hidden state
                w1_profile = profile_single_projection(
                    all_activations, gate_up_weights[expert_idx], all_routing_weights
                )
                if w1_profile is None:
                    continue

                # Profile W2 (down_proj): input is SwiGLU(W1(x)), computed in BF16
                gate_up_output = F.linear(all_activations, gate_up_weights[expert_idx])
                gate_output, up_output = gate_up_output.chunk(2, dim=-1)
                w2_input = F.silu(gate_output) * up_output
                w2_profile = profile_single_projection(
                    w2_input, down_weights[expert_idx], all_routing_weights
                )
                del gate_up_output, gate_output, up_output, w2_input

                layer_expert_profiles.append({
                    "expert": expert_idx,
                    "token_count": buf["token_count"],
                    "routing_weight_sum": round(buf["routing_weight_sum"], 4),
                    "w1": w1_profile,
                    "w2": w2_profile,
                })
                del all_activations, all_routing_weights

            results[str(layer_idx)] = {
                "layer_type": layer_type,
                "n_profiled_experts": len(layer_expert_profiles),
                "experts": layer_expert_profiles,
            }

            release_tensors(layer_weights)
            torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            print(
                f"Layer {layer_idx:2d}/{model_config.num_hidden_layers} "
                f"({layer_type:20s}): {len(layer_expert_profiles):3d} experts, "
                f"{elapsed:.0f}s",
                flush=True,
            )

    # --- Save results ---
    output_path = f"{OUTPUT_DIR}/error_profile.json"
    with open(output_path, "w") as f:
        json.dump(results, f)
    total_time = time.time() - start_time
    print(f"Done in {total_time:.0f}s. Saved {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Optimization-based BF16/NVFP4 allocation via global greedy knapsack.

Given a BF16 budget (fraction of total channels), finds the globally optimal
channel assignment by ranking ALL channels across ALL experts and layers by
their activation-weighted sensitivity score, then assigning the top-ranked
channels to BF16 and the rest to NVFP4.

Calibration: WikiText-2 TRAIN (per-layer JSON files in calibration_layers/).
Evaluation: WikiText-2 TEST (full 145 chunks, exact TRT-LLM kernels).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops
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
CALIBRATION_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/calibration_layers")
OUTPUT_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling")

W1_CHANNELS = 1024
W2_CHANNELS = 2048
NVFP4_ALIGNMENT = 32

NUM_LAYERS = 40
NUM_EXPERTS = 256

BF16_BASELINE_PPL = 6.5896
NVFP4_BASELINE_PPL = 6.8431
PPL_GAP = NVFP4_BASELINE_PPL - BF16_BASELINE_PPL

BUDGETS_TO_TEST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.65]
USE_HADAMARD = False
METRIC = "act_weighted"
DEPTH_BOOST = 0
USE_ROUTING_WEIGHT = True


def snap_to_alignment(n, total, alignment):
    if n <= 0:
        return 0
    if n >= total:
        return total
    return max(alignment, ((n + alignment // 2) // alignment) * alignment)


def load_layer_calibration(layer_idx):
    path = CALIBRATION_DIR / f"layer_{layer_idx}.json"
    with open(path) as f:
        return json.load(f)


# ── Step 1: Build global channel ranking from calibration scores ──────────

def rank_all_channels():
    """Rank all channels globally by sensitivity using compact numpy arrays.

    Returns (scores, layers, experts, projs, ch_indices) as numpy arrays
    sorted descending by score. projs: 0=w1, 1=w2.
    ~240MB for 31M channels (vs ~4.5GB with Python tuples).
    """
    score_chunks = []
    meta_chunks = []

    for layer_idx in range(NUM_LAYERS):
        layer_cal = load_layer_calibration(layer_idx)
        for expert_entry in layer_cal.get("experts", []):
            expert_idx = expert_entry["expert"]
            routing_weight = expert_entry.get("routing_weight_sum", 1.0) if USE_ROUTING_WEIGHT else 1.0
            for proj_code, proj_name, n_ch in [(0, "w1", W1_CHANNELS), (1, "w2", W2_CHANNELS)]:
                raw_scores = expert_entry[proj_name].get(METRIC)
                if not raw_scores:
                    continue
                arr = np.array(raw_scores, dtype=np.float32) * routing_weight
                n = len(arr)
                score_chunks.append(arr)
                meta = np.zeros((n, 4), dtype=np.int32)
                meta[:, 0] = layer_idx
                meta[:, 1] = expert_idx
                meta[:, 2] = proj_code
                meta[:, 3] = np.arange(n)
                meta_chunks.append(meta)
        del layer_cal

    all_scores = np.concatenate(score_chunks)
    all_meta = np.concatenate(meta_chunks)

    if DEPTH_BOOST > 0:
        depth_weights = (all_meta[:, 0].astype(np.float32) / (NUM_LAYERS - 1)) ** DEPTH_BOOST
        all_scores = all_scores * depth_weights
    del score_chunks, meta_chunks

    order = np.argsort(all_scores)[::-1]
    return all_scores[order], all_meta[order]


def assign_tiers_from_ranking(ranked_scores, ranked_meta, budget_fraction):
    total = len(ranked_scores)
    n_bf16 = int(round(budget_fraction * total))
    tier_masks = {}
    for i in range(n_bf16):
        layer_idx, expert_idx, proj_code, ch_idx = ranked_meta[i]
        proj_name = "w1" if proj_code == 0 else "w2"
        key = (int(layer_idx), int(expert_idx), proj_name)
        if key not in tier_masks:
            n_ch = W1_CHANNELS if proj_name == "w1" else W2_CHANNELS
            tier_masks[key] = torch.zeros(n_ch, dtype=torch.long)
        tier_masks[key][ch_idx] = 2
    return tier_masks


# ── Step 2: Assign BF16/NVFP4 tiers from the global ranking ──────────────

    return tier_masks


# ── Step 3: Sanitize masks to prevent NVFP4 kernel NaN ───────────────────

def sanitize_masks_for_layer(tier_masks, gate_up_weights, down_weights, layer_idx):
    """Promote zero-weight NVFP4 channels to BF16 to avoid division-by-zero.

    Sparse experts may have channels with all-zero weights. If these end up
    in the NVFP4 tier, the kernel computes scale = 0/2688 = 0, then
    divides by it → NaN. Moving them to BF16 is safe since zero × input = 0
    regardless of precision.
    """
    for expert_idx in range(NUM_EXPERTS):
        for proj_name, weights in [("w1", gate_up_weights[expert_idx]),
                                    ("w2", down_weights[expert_idx])]:
            key = (layer_idx, expert_idx, proj_name)
            if key not in tier_masks:
                n_ch = W1_CHANNELS if proj_name == "w1" else W2_CHANNELS
                tier_masks[key] = torch.zeros(n_ch, dtype=torch.long)

            tiers = tier_masks[key]
            nvfp4_channels = (tiers == 0).nonzero(as_tuple=True)[0]
            if nvfp4_channels.numel() == 0:
                continue

            channel_magnitudes = weights.float().abs().mean(dim=-1).cpu()
            zero_weight_mask = channel_magnitudes[nvfp4_channels] == 0
            if zero_weight_mask.any():
                tiers[nvfp4_channels[zero_weight_mask]] = 2


# ── Step 4: Mixed-precision linear (BF16 + NVFP4) ────────────────────────

def nvfp4_linear_with_fixed_scale(input_tensor, weight_subset, full_weight_scale):
    """NVFP4 linear using a pre-computed global weight scale from the FULL weight matrix."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight_subset, full_weight_scale, 16, False)
    alpha = (1.0 / (s_in2 * full_weight_scale)).to(torch.float32)
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, weight_fp4, bias=None, input_scale=s_in2, weight_scale=weight_scale, alpha=alpha,
    ).reshape(*input_tensor.shape[:-1], weight_subset.shape[0])


def mixed_bf16_nvfp4_linear(input_tensor, weight, channel_tiers):
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    n_output = weight.shape[0]
    device = input_tensor.device
    output = torch.zeros(input_tensor.shape[0], n_output, dtype=input_tensor.dtype, device=device)

    # Compute global weight scale from FULL weight matrix BEFORE subsetting.
    # This ensures removing channels to BF16 doesn't change the quantization
    # grid for the remaining NVFP4 channels.
    full_weight_scale = fp4_global_scale(weight).to(torch.float32)

    bf16_mask = (channel_tiers == 2)
    nvfp4_mask = (channel_tiers == 0)

    nvfp4_indices = nvfp4_mask.nonzero(as_tuple=True)[0]
    n_nvfp4 = nvfp4_indices.numel()
    misalignment = n_nvfp4 % NVFP4_ALIGNMENT
    if misalignment > 0 and n_nvfp4 > 0:
        demoted = nvfp4_indices[-misalignment:]
        bf16_mask[demoted] = True
        nvfp4_mask[demoted] = False

    bf16_indices = bf16_mask.nonzero(as_tuple=True)[0].to(device)
    nvfp4_indices = nvfp4_mask.nonzero(as_tuple=True)[0].to(device)

    if bf16_indices.numel() > 0:
        output.index_copy_(1, bf16_indices, ee.bf16_linear(input_tensor, weight[bf16_indices]))

    if nvfp4_indices.numel() > 0:
        nvfp4_output = nvfp4_linear_with_fixed_scale(input_tensor, weight[nvfp4_indices], full_weight_scale)
        output.index_copy_(1, nvfp4_indices, nvfp4_output)

    return output


# ── Step 5: Full model evaluation with mixed-precision MoE ────────────────

def evaluate_ppl(eval_ids, num_samples, seqlen, model_config, weight_map,
                 snapshot_dir, device, dtype, tier_masks, run_label):
    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    root_tensors = weight_store.load_tensors([
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ])
    embed_weight = move_tensor(root_tensors["model.language_model.embed_tokens.weight"], device, dtype)
    final_norm_weight = move_tensor(root_tensors["model.language_model.norm.weight"], device, dtype)
    lm_head_weight = move_tensor(root_tensors["lm_head.weight"], device, dtype)
    del root_tensors

    eval_chunks = eval_ids[:, :num_samples * seqlen].view(num_samples, seqlen).contiguous()
    hidden_bank = torch.empty((num_samples, seqlen, model_config.hidden_size), dtype=dtype, device="cpu")
    for i in range(num_samples):
        hidden_bank[i].copy_(
            F.embedding(eval_chunks[i : i + 1].to(device), embed_weight).squeeze(0).cpu()
        )

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=model_config, device=device)
    dummy = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(dummy, position_ids)
    del dummy

    negative_log_likelihoods = []

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            moe = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}

            if USE_HADAMARD:
                from hadamard_utils import rotate_weight_offline
                gup = moe["experts.gate_up_proj"]
                dp = moe["experts.down_proj"]
                for ei in range(model_config.num_experts):
                    gup[ei] = rotate_weight_offline(gup[ei])
                    dp[ei] = rotate_weight_offline(dp[ei])

            sanitize_masks_for_layer(
                tier_masks, moe["experts.gate_up_proj"], moe["experts.down_proj"], layer_idx
            )

            for sample_idx in range(num_samples):
                hidden = hidden_bank[sample_idx : sample_idx + 1].to(device)
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

                    if USE_HADAMARD:
                        from hadamard_utils import rotate_input_online
                        expert_input = rotate_input_online(expert_input)

                    w1_key = (layer_idx, expert_idx, "w1")
                    w2_key = (layer_idx, expert_idx, "w2")
                    w1_tiers = tier_masks.get(w1_key, torch.zeros(W1_CHANNELS, dtype=torch.long))
                    w2_tiers = tier_masks.get(w2_key, torch.zeros(W2_CHANNELS, dtype=torch.long))

                    gate_up = mixed_bf16_nvfp4_linear(expert_input, moe["experts.gate_up_proj"][expert_idx], w1_tiers)
                    gate, up = gate_up.chunk(2, dim=-1)
                    intermediate = F.silu(gate) * up
                    if USE_HADAMARD:
                        intermediate = rotate_input_online(intermediate)
                    expert_out = mixed_bf16_nvfp4_linear(intermediate, moe["experts.down_proj"][expert_idx], w2_tiers)

                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))

                shared = ee.bf16_linear(flat, moe["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, moe["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, moe["shared_expert.down_proj.weight"])
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, moe["shared_expert_gate.weight"]))

                hidden = residual + (expert_output + shared_gate * shared).view_as(residual)
                hidden_bank[sample_idx : sample_idx + 1].copy_(hidden.cpu())
                del hidden, residual, expert_output

            release_tensors(layer_weights)
            if (layer_idx + 1) % 10 == 0:
                print(f"  [{run_label}] layer {layer_idx + 1}/{model_config.num_hidden_layers}", flush=True)

        for sample_idx in range(num_samples):
            torch.cuda.empty_cache()
            tokens = eval_chunks[sample_idx : sample_idx + 1].to(device)
            hidden = hidden_bank[sample_idx : sample_idx + 1].to(device)
            hidden = rms_norm_qwen3_next(hidden, final_norm_weight, model_config.rms_norm_eps)
            logits = F.linear(hidden, lm_head_weight)
            shifted_logits = logits[:, :-1, :].contiguous().float()
            shifted_labels = tokens[:, 1:]
            loss = F.cross_entropy(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
            negative_log_likelihoods.append(loss.float() * seqlen)
            del logits, shifted_logits, hidden

    ppl = torch.exp(torch.stack(negative_log_likelihoods).sum() / (num_samples * seqlen)).item()
    del embed_weight, final_norm_weight, lm_head_weight, weight_store
    torch.cuda.empty_cache()
    return ppl


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, max_samples, seqlen = ee.load_eval_data(tokenizer, 2048)
    num_samples = min(145, max_samples)
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)

    print("Ranking all channels globally...", flush=True)
    ranked_scores, ranked_meta = rank_all_channels()
    print(f"Total channels: {len(ranked_scores):,} ({ranked_scores.nbytes / 1e6:.0f}MB scores + {ranked_meta.nbytes / 1e6:.0f}MB meta)", flush=True)
    print(f"Eval: {num_samples} chunks from WikiText-2 TEST\n", flush=True)

    results = {}
    for budget in BUDGETS_TO_TEST:
        label = f"optimal_{int(budget * 100)}pct"
        print(f"=== {label} ({budget:.0%} BF16, global greedy) ===", flush=True)

        tier_masks = assign_tiers_from_ranking(ranked_scores, ranked_meta, budget)

        start = time.time()
        ppl = evaluate_ppl(
            eval_ids, num_samples, seqlen, model_config, weight_map,
            snapshot_dir, device, dtype, tier_masks, label,
        )
        elapsed = time.time() - start
        recovery = (NVFP4_BASELINE_PPL - ppl) / PPL_GAP * 100

        results[label] = {
            "budget": budget,
            "ppl": round(ppl, 4),
            "recovery_pct": round(recovery, 1),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f}  recovery={recovery:.1f}%  ({elapsed:.0f}s)\n", flush=True)

    output_path = OUTPUT_DIR / "optimal_allocation.json"
    with output_path.open("w") as f:
        json.dump({
            "metadata": {
                "method": "global_greedy_knapsack",
                "metric": "act_weighted",
                "calibration": "wikitext2_train_128x2048",
                "evaluation": "wikitext2_test_145x2048",
            },
            "results": results,
        }, f, indent=2)

    print(f"Saved -> {output_path}")
    for _, r in sorted(results.items(), key=lambda x: x[1]["budget"]):
        print(f"  {r['budget']:>5.0%}  PPL={r['ppl']:.4f}  recovery={r['recovery_pct']:.1f}%")


if __name__ == "__main__":
    main()

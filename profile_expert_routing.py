#!/usr/bin/env python3
"""
Profile expert batch sizes for Qwen3-30B-A3B-NVFP4 MoE model.

Loads gate weights + embedding from safetensors, runs routing on wikitext data,
and computes the expert batch size distribution for dual-tile configuration.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

# ─── Config ───
MODEL_ID = "nvidia/Qwen3-30B-A3B-NVFP4"
DEVICE = "cuda"
# Simulate different batch sizes (number of tokens per decoding step)
BATCH_SIZES_TO_PROFILE = [1, 8, 16, 32, 64, 128, 256]
NUM_WIKITEXT_SAMPLES = 200   # Number of wikitext passages to use
MAX_SEQ_LEN = 512            # Max tokens per sample


def load_gate_weights_and_embedding(model_id):
    """Load only the gate/router weights and embedding from safetensors."""
    from huggingface_hub import hf_hub_download

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    hidden_size = config.hidden_size
    rms_norm_eps = config.rms_norm_eps

    # Get the weight map
    idx_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Identify needed weights
    needed_keys = {}
    # Embedding
    needed_keys["model.embed_tokens.weight"] = weight_map["model.embed_tokens.weight"]
    # Final norm (for completeness)
    if "model.norm.weight" in weight_map:
        needed_keys["model.norm.weight"] = weight_map["model.norm.weight"]
    # Per-layer: gate weight + post_attention_layernorm (input to MoE)
    for i in range(num_layers):
        gate_key = f"model.layers.{i}.mlp.gate.weight"
        norm_key = f"model.layers.{i}.post_attention_layernorm.weight"
        if gate_key in weight_map:
            needed_keys[gate_key] = weight_map[gate_key]
        if norm_key in weight_map:
            needed_keys[norm_key] = weight_map[norm_key]

    # Group by shard
    shards_needed = defaultdict(list)
    for key, shard in needed_keys.items():
        shards_needed[shard].append(key)

    # Load tensors
    tensors = {}
    for shard_name, keys in sorted(shards_needed.items()):
        shard_path = hf_hub_download(model_id, shard_name)
        print(f"  Loading {len(keys)} tensors from {shard_name}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)

    # Organize into model components
    embedding_weight = tensors["model.embed_tokens.weight"].to(DEVICE)
    gate_weights = []
    norm_weights = []
    for i in range(num_layers):
        gw = tensors.get(f"model.layers.{i}.mlp.gate.weight")
        nw = tensors.get(f"model.layers.{i}.post_attention_layernorm.weight")
        gate_weights.append(gw.to(DEVICE) if gw is not None else None)
        norm_weights.append(nw.to(DEVICE) if nw is not None else None)

    print(f"  Loaded: embedding {embedding_weight.shape}, "
          f"{sum(1 for g in gate_weights if g is not None)} gates, "
          f"{sum(1 for n in norm_weights if n is not None)} norms")

    return config, embedding_weight, gate_weights, norm_weights


def rms_norm(x, weight, eps=1e-6):
    """Apply RMS normalization."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(x.dtype)


def route_topk(logits, top_k):
    """Apply softmax routing and return top-k expert indices."""
    # Softmax over experts
    probs = torch.softmax(logits.float(), dim=-1)
    # Top-k selection
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
    return top_k_indices  # (num_tokens, top_k)


def profile_routing(config, embedding_weight, gate_weights, norm_weights,
                    tokenizer, dataset_texts, batch_sizes):
    """Profile expert routing patterns across different batch sizes."""
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    top_k = config.num_experts_per_tok
    rms_eps = config.rms_norm_eps

    # Tokenize all texts
    print(f"\nTokenizing {len(dataset_texts)} texts...")
    all_token_ids = []
    for text in dataset_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN, add_special_tokens=False)
        if enc.input_ids.shape[1] > 10:  # Skip very short texts
            all_token_ids.append(enc.input_ids.squeeze(0))
    print(f"  Got {len(all_token_ids)} sequences, "
          f"total tokens: {sum(t.shape[0] for t in all_token_ids)}")

    # Flatten all tokens for sampling batches
    all_tokens = torch.cat(all_token_ids, dim=0).to(DEVICE)
    total_tokens = all_tokens.shape[0]
    print(f"  Total tokens available: {total_tokens}")

    # Results: per_layer_per_expert_counts[layer_idx][batch_size] = list of per-expert counts
    results = defaultdict(lambda: defaultdict(list))

    for batch_size in batch_sizes:
        if batch_size > total_tokens:
            print(f"\nSkipping batch_size={batch_size} (not enough tokens)")
            continue

        num_batches = min(200, total_tokens // batch_size)
        print(f"\nProfiling batch_size={batch_size}, {num_batches} batches...")

        for b in range(num_batches):
            # Sample a batch of tokens (sequential for locality)
            start = (b * batch_size) % (total_tokens - batch_size)
            batch_ids = all_tokens[start:start + batch_size]  # (batch_size,)

            # Get embeddings
            with torch.no_grad():
                hidden = embedding_weight[batch_ids]  # (batch_size, hidden_size)

                # Run through each layer's gate
                for layer_idx in range(num_layers):
                    if gate_weights[layer_idx] is None:
                        continue

                    # Apply post-attention layer norm (approximate: in reality,
                    # hidden states are transformed by attention first)
                    if norm_weights[layer_idx] is not None:
                        normed = rms_norm(hidden, norm_weights[layer_idx], rms_eps)
                    else:
                        normed = hidden

                    # Compute gate logits and route
                    normed_bf16 = normed.to(gate_weights[layer_idx].dtype)
                    logits = torch.nn.functional.linear(normed_bf16, gate_weights[layer_idx])
                    expert_indices = route_topk(logits, top_k)  # (batch_size, top_k)

                    # Count tokens per expert
                    flat_indices = expert_indices.flatten()
                    counts = torch.bincount(flat_indices, minlength=num_experts)
                    results[layer_idx][batch_size].append(counts.cpu())

        sys.stdout.flush()

    return results


def analyze_results(results, config, batch_sizes):
    """Analyze routing results and determine dual-tile threshold."""
    num_experts = config.num_experts
    top_k = config.num_experts_per_tok

    print("\n" + "=" * 80)
    print("EXPERT BATCH SIZE ANALYSIS")
    print("=" * 80)

    # Aggregate across all layers
    all_layer_stats = defaultdict(list)

    for batch_size in batch_sizes:
        all_counts = []
        for layer_idx in sorted(results.keys()):
            if batch_size in results[layer_idx]:
                layer_counts = torch.stack(results[layer_idx][batch_size])  # (num_batches, num_experts)
                all_counts.append(layer_counts)

        if not all_counts:
            continue

        # Shape: (total_observations, num_experts)
        counts = torch.cat(all_counts, dim=0).float()

        # Per-expert statistics (across all layers and batches)
        mean_per_expert = counts.mean(dim=0)
        median_all = counts.median().item()
        p30 = torch.quantile(counts.flatten().float(), 0.30).item()
        p50 = torch.quantile(counts.flatten().float(), 0.50).item()
        p70 = torch.quantile(counts.flatten().float(), 0.70).item()
        p90 = torch.quantile(counts.flatten().float(), 0.90).item()

        # Theoretical expected
        expected = batch_size * top_k / num_experts

        # Count experts with 0 tokens (idle)
        zero_frac = (counts == 0).float().mean().item()

        print(f"\n--- Batch Size = {batch_size} tokens ---")
        print(f"  Expected (uniform): {expected:.1f} tokens/expert")
        print(f"  Actual mean:        {counts.mean().item():.1f}")
        print(f"  Percentiles:  P30={p30:.0f}  P50={p50:.0f}  P70={p70:.0f}  P90={p90:.0f}")
        print(f"  Min={counts.min().item():.0f}  Max={counts.max().item():.0f}")
        print(f"  Idle experts (0 tokens): {zero_frac*100:.1f}%")

        # Distribution histogram
        hist_bins = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        hist_bins = [b for b in hist_bins if b <= batch_size * top_k]
        if hist_bins[-1] < batch_size * top_k:
            hist_bins.append(batch_size * top_k + 1)
        flat = counts.flatten().numpy()
        hist, _ = np.histogram(flat, bins=hist_bins)
        print(f"  Distribution:")
        for i in range(len(hist)):
            lo = hist_bins[i]
            hi = hist_bins[i + 1] if i + 1 < len(hist_bins) else float('inf')
            pct = hist[i] / len(flat) * 100
            bar = "█" * int(pct / 2)
            print(f"    [{lo:3d}-{hi:3d}): {pct:5.1f}% {bar}")

        # 70/30 split
        threshold_70 = torch.quantile(counts.flatten().float(), 0.70).item()
        print(f"\n  70/30 SPLIT THRESHOLD: {threshold_70:.0f} tokens")
        print(f"    70% of expert batches have <= {threshold_70:.0f} tokens (use SMALL tile)")
        print(f"    30% of expert batches have >  {threshold_70:.0f} tokens (use LARGE tile)")

        all_layer_stats[batch_size] = {
            'p30': p30, 'p50': p50, 'p70': p70, 'p90': p90,
            'mean': counts.mean().item(),
            'threshold_70': threshold_70,
            'zero_frac': zero_frac,
        }

    return all_layer_stats


def main():
    print("=" * 80)
    print("MoE Expert Batch Size Profiler")
    print(f"Model: {MODEL_ID}")
    print("=" * 80)

    # 1. Load gate weights and embedding
    print("\n[1/4] Loading gate weights and embedding...")
    config, embedding_weight, gate_weights, norm_weights = \
        load_gate_weights_and_embedding(MODEL_ID)

    print(f"\nModel config:")
    print(f"  num_experts={config.num_experts}, top_k={config.num_experts_per_tok}")
    print(f"  hidden_size={config.hidden_size}, moe_intermediate_size={config.moe_intermediate_size}")
    print(f"  num_layers={config.num_hidden_layers}")

    # 2. Load tokenizer and dataset
    print("\n[2/4] Loading tokenizer and wikitext dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:NUM_WIKITEXT_SAMPLES]
    print(f"  Using {len(texts)} wikitext passages")

    # 3. Profile routing
    print("\n[3/4] Profiling expert routing...")
    results = profile_routing(
        config, embedding_weight, gate_weights, norm_weights,
        tokenizer, texts, BATCH_SIZES_TO_PROFILE)

    # 4. Analyze results
    print("\n[4/4] Analyzing results...")
    stats = analyze_results(results, config, BATCH_SIZES_TO_PROFILE)

    # Summary
    print("\n" + "=" * 80)
    print("DUAL-TILE CONFIGURATION RECOMMENDATIONS")
    print("=" * 80)
    for bs, s in sorted(stats.items()):
        print(f"\n  Batch {bs:3d} tokens → threshold={s['threshold_70']:.0f} "
              f"(P70={s['p70']:.0f}, P50={s['p50']:.0f}, idle={s['zero_frac']*100:.0f}%)")

    print("\n  Recommendation: For typical decode batch sizes,")
    print("  set dual_tile_threshold to the P70 value for your target batch size.")
    print("  Experts with <= threshold tokens use SMALL tile (faster for small M)")
    print("  Experts with >  threshold tokens use LARGE tile (better utilization)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Profile per-expert token counts for Qwen3-30B-A3B-NVFP4 MoE routing.

Uses gate weights + embedding to simulate routing decisions on wikitext data.
Shows per-expert M distribution at different global batch sizes to understand
which CTA tile size (M32, M64, M128) is optimal.

NOTE: This uses embedding-only hidden states (no attention). Routing patterns
are approximate but representative of the distribution shape.
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = '/root/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3'
NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48
HIDDEN = 2048
RMS_EPS = 1e-6


def load_weights(model_path):
    with open(f'{model_path}/model.safetensors.index.json') as f:
        weight_map = json.load(f)['weight_map']

    needed = {}
    needed['model.embed_tokens.weight'] = weight_map['model.embed_tokens.weight']
    for i in range(NUM_LAYERS):
        for suffix in ['mlp.gate.weight', 'post_attention_layernorm.weight']:
            key = f'model.layers.{i}.{suffix}'
            if key in weight_map:
                needed[key] = weight_map[key]

    shards = defaultdict(list)
    for key, shard in needed.items():
        shards[shard].append(key)

    tensors = {}
    for shard_name, keys in sorted(shards.items()):
        with safe_open(f'{model_path}/{shard_name}', framework='pt', device='cuda') as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)

    embedding = tensors['model.embed_tokens.weight']
    gates, norms = [], []
    for i in range(NUM_LAYERS):
        gates.append(tensors.get(f'model.layers.{i}.mlp.gate.weight'))
        norms.append(tensors.get(f'model.layers.{i}.post_attention_layernorm.weight'))

    return embedding, gates, norms


def route_batch(token_ids, embedding, gates, norms, num_layers):
    """Route a batch of tokens through all layers. Returns per-layer per-expert counts."""
    hidden = embedding[token_ids]
    layer_counts = []

    for li in range(num_layers):
        if gates[li] is None:
            continue
        if norms[li] is not None:
            normed = F.rms_norm(hidden.float(), (HIDDEN,), norms[li].float(), RMS_EPS).to(torch.bfloat16)
        else:
            normed = hidden
        logits = normed.float() @ gates[li].float().T
        probs = F.softmax(logits, dim=-1)
        _, topk_idx = torch.topk(probs, TOP_K, dim=-1)
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device='cuda')
        counts.scatter_add_(0, topk_idx.reshape(-1).long(),
                            torch.ones(token_ids.shape[0] * TOP_K, dtype=torch.int32, device='cuda'))
        layer_counts.append(counts.cpu().numpy())

    return np.stack(layer_counts)


def optimal_tile(m):
    if m <= 32:
        return 'M32'
    elif m <= 64:
        return 'M64'
    else:
        return 'M128'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-sizes', type=str, default='1,4,8,16,32,64,128,256,512',
                        help='Comma-separated global batch sizes')
    parser.add_argument('--num-layers', type=int, default=48)
    parser.add_argument('--max-batches', type=int, default=0,
                        help='Max batches per size (0=use all data)')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    num_layers = min(args.num_layers, NUM_LAYERS)

    print(f"{'=' * 80}")
    print(f"Expert Routing Profiler — Qwen3-30B-A3B-NVFP4")
    print(f"  {NUM_EXPERTS} experts, top-{TOP_K}, {num_layers} layers")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"{'=' * 80}")

    print("\nLoading weights...")
    embedding, gates, norms = load_weights(MODEL_PATH)
    print(f"  Embedding: {embedding.shape}")

    print("Loading tokenizer + wikitext-103 test set...")
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    full_text = " ".join([t for t in dataset["text"] if t.strip()])
    all_tokens = tokenizer.encode(full_text)
    total_tokens = len(all_tokens)
    print(f"  Total tokens: {total_tokens}")

    all_tokens_t = torch.tensor(all_tokens, device='cuda')
    summary_rows = []  # collect (bs, uniform, p50, p90, max, idle%, m32%, m64%, m128%)

    for bs in batch_sizes:
        if bs > total_tokens:
            print(f"\nSkipping batch_size={bs} (not enough tokens)")
            continue

        num_batches = total_tokens // bs
        if args.max_batches > 0:
            num_batches = min(num_batches, args.max_batches)

        print(f"\n{'=' * 80}")
        print(f"BATCH SIZE: {bs} tokens  ({num_batches} batches, {num_batches * bs} total tokens)")
        print(f"  Expected uniform per-expert M: {bs * TOP_K / NUM_EXPERTS:.1f}")
        print(f"{'=' * 80}")

        all_counts = []
        t0 = time.time()

        with torch.no_grad():
            for bi in range(num_batches):
                start = bi * bs
                token_ids = all_tokens_t[start:start + bs]
                counts = route_batch(token_ids, embedding, gates, norms, num_layers)
                all_counts.append(counts)

                if (bi + 1) % max(1, num_batches // 5) == 0:
                    elapsed = time.time() - t0
                    print(f"  {bi + 1}/{num_batches} batches ({elapsed:.1f}s)", flush=True)

        elapsed = time.time() - t0
        print(f"  Done: {num_batches} batches in {elapsed:.1f}s")

        counts_all = np.concatenate(all_counts, axis=0)
        flat = counts_all.flatten()
        active = flat[flat > 0]

        if len(active) == 0:
            print("  No active experts!")
            continue

        p10 = np.percentile(active, 10)
        p25 = np.percentile(active, 25)
        p50 = np.percentile(active, 50)
        p75 = np.percentile(active, 75)
        p90 = np.percentile(active, 90)
        p99 = np.percentile(active, 99)
        idle_pct = np.mean(flat == 0) * 100

        print(f"\n  Per-Expert Token Count Distribution (active experts only):")
        print(f"    P10={p10:.0f}  P25={p25:.0f}  P50={p50:.0f}  P75={p75:.0f}  P90={p90:.0f}  P99={p99:.0f}")
        print(f"    Mean={active.mean():.1f}  Max={flat.max():.0f}  Idle={idle_pct:.1f}%")

        bins = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        bins = [b for b in bins if b <= flat.max() + 1]
        if bins[-1] <= flat.max():
            bins.append(int(flat.max()) + 1)

        print(f"\n  Histogram (per-expert token count):")
        print(f"  {'Range':>12s}  {'Count':>8s}  {'Pct':>6s}  {'Optimal':>8s}  Bar")
        print(f"  {'-'*60}")

        hist, _ = np.histogram(active, bins=bins)
        for i in range(len(hist)):
            lo, hi = bins[i], bins[i + 1]
            pct = hist[i] / len(active) * 100
            bar = '█' * int(pct / 2)
            tile = optimal_tile((lo + hi) / 2)
            label = f"[{lo:>3d}-{hi:>3d})"
            print(f"  {label:>12s}  {hist[i]:>8d}  {pct:>5.1f}%  {tile:>8s}  {bar}")

        m32_pct = np.mean(active <= 32) * 100
        m64_pct = np.mean((active > 32) & (active <= 64)) * 100
        m128_pct = np.mean(active > 64) * 100

        print(f"\n  Optimal Tile Split:")
        print(f"    M32  (≤32 tokens):  {m32_pct:5.1f}%")
        print(f"    M64  (33-64):       {m64_pct:5.1f}%")
        print(f"    M128 (>64):         {m128_pct:5.1f}%")

        summary_rows.append((bs, bs * TOP_K / NUM_EXPERTS, p50, p90, float(flat.max()), idle_pct, m32_pct, m64_pct, m128_pct))

        per_batch_means = []
        per_batch_maxs = []
        for chunk in all_counts:
            for layer_counts in chunk:
                act = layer_counts[layer_counts > 0]
                if len(act) > 0:
                    per_batch_means.append(act.mean())
                    per_batch_maxs.append(act.max())

        print(f"\n  Per-Layer Stats (across all batches):")
        print(f"    Avg per-expert M:  mean={np.mean(per_batch_means):.1f}, "
              f"std={np.std(per_batch_means):.1f}")
        print(f"    Max per-expert M:  mean={np.mean(per_batch_maxs):.1f}, "
              f"P90={np.percentile(per_batch_maxs, 90):.0f}, "
              f"max={np.max(per_batch_maxs):.0f}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  {'Batch':>6s}  {'Uniform':>8s}  {'P50':>5s}  {'P90':>5s}  {'Max':>5s}  "
          f"{'Idle%':>6s}  {'M32%':>6s}  {'M64%':>6s}  {'M128%':>6s}")
    print(f"  {'-' * 70}")

    for row in summary_rows:
        bs, uniform, p50, p90, mx, idle, m32, m64, m128 = row
        print(f"  {bs:>6d}  {uniform:>8.1f}  {p50:>5.0f}  {p90:>5.0f}  {mx:>5.0f}  "
              f"{idle:>5.1f}%  {m32:>5.1f}%  {m64:>5.1f}%  {m128:>5.1f}%")

    print()


if __name__ == '__main__':
    main()

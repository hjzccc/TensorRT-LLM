#!/usr/bin/env python3
"""Chain rule entropy: H(x_k | x_1,...,x_{k-1}) for increasing k within NVFP4 blocks.
Measures how predictable each element is given ALL previous elements in the block.
Uses full dataset, 16 parallel workers, vectorized numpy.
"""
import sys
import time
import json
import math
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from safetensors import safe_open

N_CODES = 15
BOUNDARIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"
MAX_DEPTH = 8


def quantize_block_vectorized(weight_2d, block_size=16):
    rows, cols = weight_2d.shape
    pad = (block_size - cols % block_size) % block_size
    if pad > 0:
        weight_2d = np.pad(weight_2d, ((0, 0), (0, pad)))
    blocks = weight_2d.reshape(-1, block_size)
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    amax[amax == 0] = 1.0
    scales = np.float16(amax / 6.0).astype(np.float32)
    scales[scales == 0] = 1.0
    normalized = blocks / scales
    codes = np.searchsorted(BOUNDARIES, normalized.ravel()).reshape(blocks.shape)
    return np.clip(codes, 0, N_CODES - 1).astype(np.int8)


def tuple_hash(codes_prefix):
    """Convert a short array of codes to a single int64 key for fast counting."""
    h = np.int64(0)
    for c in codes_prefix:
        h = h * N_CODES + np.int64(c)
    return h


def process_expert_chain(args):
    """Count n-gram patterns up to MAX_DEPTH for chain rule entropy."""
    fpath, key, expert_idx = args

    with safe_open(fpath, framework='pt', device='cpu') as f:
        weight = f.get_tensor(key)[expert_idx].float().numpy()

    codes = quantize_block_vectorized(weight, block_size=16)
    n_blocks = codes.shape[0]

    ngram_counts = {}
    for depth in range(1, MAX_DEPTH + 1):
        ngram_counts[depth] = defaultdict(int)

    for bi in range(n_blocks):
        block = codes[bi]
        for depth in range(1, min(MAX_DEPTH + 1, 17)):
            for start in range(16 - depth + 1):
                key_tuple = tuple(block[start:start + depth].tolist())
                ngram_counts[depth][key_tuple] += 1

    return ngram_counts, n_blocks


def process_expert_fast(args):
    """Faster version: only count patterns starting from position 0 (prefix conditioning)."""
    fpath, key, expert_idx = args

    with safe_open(fpath, framework='pt', device='cpu') as f:
        weight = f.get_tensor(key)[expert_idx].float().numpy()

    codes = quantize_block_vectorized(weight, block_size=16)
    n_blocks = codes.shape[0]

    prefix_counts = {}
    for depth in range(1, MAX_DEPTH + 1):
        prefix_counts[depth] = defaultdict(int)
        prefixes = codes[:, :depth]
        for bi in range(n_blocks):
            k = tuple(prefixes[bi].tolist())
            prefix_counts[depth][k] += 1

    return prefix_counts, n_blocks


def shannon_from_dict(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            H -= p * math.log2(p)
    return H


def main():
    model_dir = Path(MODEL_DIR)
    snapshot = list(model_dir.iterdir())[0]
    files = sorted(snapshot.glob("*.safetensors"))

    tasks = []
    for fpath in files:
        with safe_open(str(fpath), framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'experts' not in key:
                    continue
                if 'gate_up_proj' not in key and 'down_proj' not in key:
                    continue
                if 'mtp.' in key or 'shared_expert' in key:
                    continue
                n_exp = f.get_tensor(key).shape[0]
                for ei in range(n_exp):
                    tasks.append((str(fpath), key, ei))

    print(f"Total: {len(tasks)} experts, measuring prefix chain entropy up to depth {MAX_DEPTH}", flush=True)
    t0 = time.time()

    agg_prefix = {}
    for d in range(1, MAX_DEPTH + 1):
        agg_prefix[d] = defaultdict(int)
    total_blocks = 0
    done = 0

    N_WORKERS = 16
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_expert_fast, t): t for t in tasks}
        for fut in as_completed(futures):
            prefix_counts, nb = fut.result()
            total_blocks += nb
            for d in range(1, MAX_DEPTH + 1):
                for k, v in prefix_counts[d].items():
                    agg_prefix[d][k] += v
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(tasks)} experts ({total_blocks/1e6:.1f}M blocks, {elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s, {total_blocks:,} blocks total", flush=True)

    print(f"\n{'='*60}")
    print(f"CHAIN RULE ENTROPY: H(x_k | x_1,...,x_{{k-1}})")
    print(f"{'='*60}")
    print(f"{'Depth':>6s} {'H(prefix)':>12s} {'H(x_k|prev)':>14s} {'Unique patterns':>18s} {'Possible':>18s} {'Coverage':>10s}")

    prev_H = 0.0
    results = []
    for d in range(1, MAX_DEPTH + 1):
        H_joint = shannon_from_dict(agg_prefix[d])
        H_conditional = H_joint - prev_H
        n_unique = len(agg_prefix[d])
        n_possible = N_CODES ** d
        coverage = n_unique / n_possible

        print(f"{d:6d} {H_joint:12.4f} {H_conditional:14.4f} {n_unique:18,} {n_possible:18,} {coverage:10.6f}")

        results.append({
            'depth': d,
            'joint_entropy': H_joint,
            'conditional_entropy': H_conditional,
            'unique_patterns': n_unique,
            'possible_patterns': n_possible,
            'coverage': coverage,
        })
        prev_H = H_joint

    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"If elements are independent: H(x_k|prev) = H(x_1) = {results[0]['conditional_entropy']:.4f} for all k")
    print(f"If elements are perfectly predictable: H(x_k|prev) → 0 as k grows")
    print()
    for r in results[1:]:
        drop = (results[0]['conditional_entropy'] - r['conditional_entropy']) / results[0]['conditional_entropy'] * 100
        print(f"  Depth {r['depth']}: H={r['conditional_entropy']:.4f} ({drop:+.2f}% vs independent)")

    reliable_limit = None
    for r in results:
        if r['coverage'] > 0.5:
            reliable_limit = r['depth'] - 1
            break
    if reliable_limit:
        print(f"\n  WARNING: Results at depth >{reliable_limit} may be unreliable (>50% patterns are unique)")

    out_path = "/code/tensorrt_llm/scripts/nvfp4_compress/chain_entropy.json"
    with open(out_path, 'w') as f:
        json.dump({'total_blocks': total_blocks, 'chain': results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

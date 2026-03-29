#!/usr/bin/env python3
"""Measure per-block joint entropy at block sizes 4, 8, 16, 32.

If 16 elements within a block are independent: joint entropy = 16 × 3.977 = 63.6 bits
If correlated: joint entropy < 63.6 bits → compressible structure exists

Also measures: conditional entropy, mutual information, and unique block pattern count.
"""
import sys
import time
import json
import math
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

FULL_BLOCK = 16
FP4_UNIQUE = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
BOUNDARIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)

MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"


def quantize_to_fp4_codes(block):
    amax = np.abs(block).max()
    if amax == 0:
        return np.zeros(len(block), dtype=np.int8)
    scale = np.float16(amax / 6.0).astype(np.float32)
    if scale == 0:
        scale = 1.0
    norm = block / scale
    idx = np.searchsorted(BOUNDARIES, norm)
    return np.clip(idx, 0, len(FP4_UNIQUE) - 1).astype(np.int8)


def block_to_tuple(codes):
    return tuple(codes.tolist())


def shannon_entropy(counter, total):
    H = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            H -= p * math.log2(p)
    return H


def analyze_block_size(all_codes_flat, block_size, total_elements):
    n_blocks = total_elements // block_size
    trimmed = all_codes_flat[:n_blocks * block_size]
    blocks = trimmed.reshape(-1, block_size)

    pattern_counts = Counter()
    for i in range(len(blocks)):
        pattern_counts[block_to_tuple(blocks[i])] += 1

    joint_entropy = shannon_entropy(pattern_counts, n_blocks)
    per_element = joint_entropy / block_size

    marginal_entropy = 3.977
    independent_joint = marginal_entropy * block_size
    mutual_info = independent_joint - joint_entropy
    mi_per_element = mutual_info / block_size

    unique_patterns = len(pattern_counts)
    max_possible = 15 ** block_size

    top_patterns = pattern_counts.most_common(10)

    return {
        'block_size': block_size,
        'n_blocks': n_blocks,
        'unique_patterns': unique_patterns,
        'max_possible_patterns': max_possible,
        'pattern_coverage': unique_patterns / max_possible if max_possible > 0 else 0,
        'joint_entropy': joint_entropy,
        'per_element_entropy': per_element,
        'marginal_entropy': marginal_entropy,
        'independent_joint': independent_joint,
        'mutual_information': mutual_info,
        'mi_per_element': mi_per_element,
        'compression_gain_pct': (1 - per_element / marginal_entropy) * 100,
    }


def main():
    model_dir = Path(MODEL_DIR)
    snapshot = list(model_dir.iterdir())[0]
    files = sorted(snapshot.glob("*.safetensors"))

    print("Collecting FP4 codes from expert weights...", flush=True)
    t0 = time.time()

    all_codes = []
    total_elements = 0
    np.random.seed(42)

    for fi, fpath in enumerate(files):
        print(f"  [{fi+1}/{len(files)}] {fpath.name}", flush=True)
        with safe_open(str(fpath), framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'experts' not in key:
                    continue
                if 'gate_up_proj' not in key and 'down_proj' not in key:
                    continue
                if 'mtp.' in key or 'shared_expert' in key:
                    continue

                packed = f.get_tensor(key).float().numpy()
                n_exp = packed.shape[0]
                sample_exp = min(30, n_exp)
                indices = np.random.choice(n_exp, sample_exp, replace=False)

                for ei in indices:
                    weight = packed[ei]
                    rows, cols = weight.shape
                    pad = (FULL_BLOCK * 2 - cols % (FULL_BLOCK * 2)) % (FULL_BLOCK * 2)
                    if pad > 0:
                        weight = np.pad(weight, ((0, 0), (0, pad)))

                    blocks_16 = weight.reshape(-1, FULL_BLOCK)
                    for bi in range(blocks_16.shape[0]):
                        codes = quantize_to_fp4_codes(blocks_16[bi])
                        all_codes.append(codes)
                        total_elements += FULL_BLOCK

                if total_elements > 50_000_000:
                    break
        if total_elements > 50_000_000:
            break

    all_codes = np.concatenate(all_codes)
    elapsed = time.time() - t0
    print(f"Collected {total_elements:,} elements ({total_elements//FULL_BLOCK:,} blocks) in {elapsed:.0f}s\n", flush=True)

    print(f"{'='*70}")
    print(f"BLOCK ENTROPY ANALYSIS")
    print(f"{'='*70}\n")

    results = {}
    for bs in [1, 2, 4, 8, 16]:
        print(f"Analyzing block_size={bs}...", flush=True)
        t1 = time.time()
        r = analyze_block_size(all_codes, bs, total_elements)
        dt = time.time() - t1

        print(f"  Block size:          {bs}")
        print(f"  Blocks analyzed:     {r['n_blocks']:,}")
        print(f"  Unique patterns:     {r['unique_patterns']:,}")
        print(f"  Joint entropy:       {r['joint_entropy']:.4f} bits per block")
        print(f"  Per-element entropy:  {r['per_element_entropy']:.4f} bits")
        print(f"  If independent:      {r['independent_joint']:.4f} bits per block")
        print(f"  Mutual information:  {r['mutual_information']:.4f} bits per block ({r['mi_per_element']:.4f}/elem)")
        print(f"  Compression gain:    {r['compression_gain_pct']:.2f}%")
        print(f"  Time: {dt:.0f}s\n", flush=True)

        results[f'block_{bs}'] = r

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Block':>6s} {'Joint H':>10s} {'Per-elem H':>12s} {'MI/elem':>10s} {'Gain':>8s} {'Unique':>12s}")
    for bs in [1, 2, 4, 8, 16]:
        r = results[f'block_{bs}']
        print(f"{bs:6d} {r['joint_entropy']:10.4f} {r['per_element_entropy']:12.4f} {r['mi_per_element']:10.4f} {r['compression_gain_pct']:7.2f}% {r['unique_patterns']:12,}")

    print(f"\nInterpretation:")
    gain_16 = results['block_16']['compression_gain_pct']
    per_elem_16 = results['block_16']['per_element_entropy']
    if gain_16 > 5:
        print(f"  Block-16 per-element entropy = {per_elem_16:.3f} bits (vs 3.977 marginal)")
        print(f"  {gain_16:.1f}% compression gain from intra-block correlation → EXPLOITABLE")
    elif gain_16 > 1:
        print(f"  Block-16 per-element entropy = {per_elem_16:.3f} bits (vs 3.977 marginal)")
        print(f"  {gain_16:.1f}% compression gain — modest but real correlation exists")
    else:
        print(f"  Block-16 per-element entropy = {per_elem_16:.3f} bits (vs 3.977 marginal)")
        print(f"  {gain_16:.1f}% gain — elements within blocks are nearly independent")

    out_path = "/code/tensorrt_llm/scripts/channel_quant_new/profiling/block_entropy.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
